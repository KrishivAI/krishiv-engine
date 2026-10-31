// SPDX-License-Identifier: Apache-2.0
//! The collect and vector accumulators: `collect_list`, `collect_set`
//! (`array_agg`, which the analyzer resolves as `collect_list`) and the
//! element-wise `vector_sum` / `vector_avg` over `ARRAY<FLOAT>`.
//!
//! A collect accumulator keeps, per group, the appended value chunks in
//! fold order (slices of raw inputs or of merged state lists); merging
//! concatenates the chunks, so partial states merge associatively and a
//! partial phase, an exchange and a final phase compose. `collect_set`
//! additionally keeps the encoded distinct rows per group in first-seen
//! order, through the shared grouping key encoding
//! ([`crate::hash::KeyEncoder`], as grouping and joining use it): NaN
//! equals NaN and `-0.0` equals `0.0`, whatever the input carrier (raw
//! `Utf8` chunks are normalized to views on append, as the planner types
//! arguments). NULL elements are skipped by both; an empty group is `[]`,
//! never NULL.
//!
//! A vector accumulator keeps, per group, the running sums as `double`s
//! and the count of non-NULL vectors (`None` until a non-NULL vector
//! arrives, so all-NULL is NULL). Empty vectors count as non-NULL vectors
//! of dimension zero; any other dimension change across rows or states is
//! `VECTOR_DIMENSION_MISMATCH`. `vector_avg` divides by the count.
//! NULL elements inside a vector contribute nothing (they are skipped, as
//! NULL vectors are). Sums fold in input order within a partition and
//! merge in partition-index order, so results are deterministic. The
//! partial state is the sums with the count, so merging adds them
//! element-wise.
//!
//! Memory: chunks share their inputs' buffers and the distinct rows live
//! in key tables; [`kept_bytes`] bounds what an append keeps, and the
//! accumulator's reservation (made by the table from
//! [`CollectAcc::projected`] / [`VectorAcc::projected`] before folding)
//! covers it. Appends beyond the bound only over-reserve, never leak: a
//! failed fold drops its temporaries with the call.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, Float32Array, Float64Array, Int64Array, ListArray,
    StringViewArray, UInt32Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use arrow_select::concat::concat;
use arrow_select::take::take;

use crate::error::{Result, internal};
use crate::hash::{HashSeed, KeyEncoder, KeyTable};
use krishiv_common::error::VectorDimensionMismatch;

use super::accumulator::Running;

/// Whether `collect_set` dedups values of type `t`: the shared grouping
/// key encoding must order them. Floats nested inside arrays, structs,
/// maps or dictionaries have per-element normalization in Spark that the
/// encoder does not do; it refuses them rather than split groups.
pub(crate) fn collect_set_supported(t: &DataType) -> bool {
    KeyEncoder::new(std::slice::from_ref(t)).is_ok()
}

/// An upper bound of the bytes a kept copy of `a`'s values holds (validity
/// included). Slices count only the span they keep of variable-width
/// buffers; shared fixed-width buffers count in full (an overestimate, but
/// never an underestimate: reservations are made against this).
pub(crate) fn kept_bytes(a: &ArrayRef) -> usize {
    use arrow_schema::DataType as T;
    let n = a.len();
    let add = |a: usize, b: usize| a.saturating_add(b);
    let mul = |a: usize, b: usize| a.saturating_mul(b);
    let base = n.div_ceil(8);
    add(
        base,
        match a.data_type() {
            T::Null => 0,
            T::Boolean => n.div_ceil(8),
            T::Int8 | T::UInt8 => n,
            T::Int16 | T::UInt16 | T::Float16 => mul(n, 2),
            T::Int32 | T::UInt32 | T::Float32 | T::Date32 => mul(n, 4),
            T::Int64
            | T::UInt64
            | T::Float64
            | T::Date64
            | T::Timestamp(..)
            | T::Time32(_)
            | T::Time64(_)
            | T::Duration(_)
            | T::Decimal64(..) => mul(n, 8),
            T::Decimal128(..) | T::Interval(_) | T::Decimal32(..) | T::Decimal256(..) => mul(n, 16),
            T::Utf8 | T::Binary => {
                let o = match a.data_type() {
                    T::Utf8 => a.as_string::<i32>().value_offsets(),
                    _ => a.as_binary::<i32>().value_offsets(),
                };
                span32(o) + data_span(a, 0)
            }
            T::LargeUtf8 | T::LargeBinary => {
                let o = match a.data_type() {
                    T::LargeUtf8 => a.as_string::<i64>().value_offsets(),
                    _ => a.as_binary::<i64>().value_offsets(),
                };
                span64(o) + data_span(a, 1)
            }
            T::Utf8View | T::BinaryView => {
                // Views hold short values inline (16 bytes each) and the
                // lengths of long ones: both, to stay an upper bound.
                add(mul(n, 16), views_len(a))
            }
            T::FixedSizeBinary(size) => mul(n, usize::try_from(*size).unwrap_or(0)),
            T::List(_) | T::LargeList(_) => {
                let l = a.as_any().downcast_ref::<ListArray>();
                match l {
                    Some(l) => {
                        let offsets = l.value_offsets();
                        let (s, e) = match (offsets.first(), offsets.last()) {
                            (Some(&s), Some(&e)) => (s.max(0), e.max(0)),
                            _ => (0, 0),
                        };
                        let (s, len) = (
                            usize::try_from(s).unwrap_or(0),
                            usize::try_from(e.saturating_sub(s)).unwrap_or(0),
                        );
                        add(mul(offsets.len(), 4), kept_bytes(&l.values().slice(s, len)))
                    }
                    None => a.get_buffer_memory_size(),
                }
            }
            _ => a.get_buffer_memory_size(),
        },
    )
}

/// The byte span of `i32` offsets (empty when degenerate).
fn span32(o: &[i32]) -> usize {
    match (o.first(), o.last()) {
        (Some(&s), Some(&e)) => usize::try_from(e.saturating_sub(s).max(0)).unwrap_or(0),
        _ => 0,
    }
}

/// The byte span of `i64` offsets (empty when degenerate).
fn span64(o: &[i64]) -> usize {
    match (o.first(), o.last()) {
        (Some(&s), Some(&e)) => usize::try_from(e.saturating_sub(s).max(0)).unwrap_or(0),
        _ => 0,
    }
}

/// The data-buffer span of a variable-width array: its last buffer's
/// length (an overestimate for slices, never an underestimate).
fn data_span(a: &ArrayRef, skip: usize) -> usize {
    a.to_data()
        .buffers()
        .iter()
        .skip(skip)
        .map(arrow_buffer::Buffer::len)
        .sum()
}

/// The summed lengths of a view array's values.
fn views_len(a: &ArrayRef) -> usize {
    match a.data_type() {
        DataType::Utf8View => a
            .as_string_view()
            .views()
            .iter()
            .map(|v| usize::try_from(v & 0xFFFF_FFFF).unwrap_or(0))
            .sum(),
        DataType::BinaryView => a
            .as_binary_view()
            .views()
            .iter()
            .map(|v| usize::try_from(v & 0xFFFF_FFFF).unwrap_or(0))
            .sum(),
        _ => 0,
    }
}

/// `chunk` on the accumulator's element type `elem`: raw string and binary
/// columns keep whatever carrier the input had, while states are views
/// (the planner types arguments as views). Anything else passes through;
/// a carrier that still disagrees fails when the chunks concatenate, never
/// silently.
pub(crate) fn normalize(chunk: ArrayRef, elem: &DataType) -> Result<ArrayRef> {
    if chunk.data_type() == elem {
        return Ok(chunk);
    }
    Ok(match (elem, chunk.data_type()) {
        (DataType::Utf8View, DataType::Utf8) => {
            Arc::new(StringViewArray::from(chunk.as_string::<i32>()))
        }
        (DataType::Utf8View, DataType::LargeUtf8) => {
            Arc::new(StringViewArray::from(chunk.as_string::<i64>()))
        }
        (DataType::BinaryView, DataType::Binary) => {
            Arc::new(BinaryViewArray::from(chunk.as_binary::<i32>()))
        }
        (DataType::BinaryView, DataType::LargeBinary) => {
            Arc::new(BinaryViewArray::from(chunk.as_binary::<i64>()))
        }
        _ => {
            return Err(internal(format!(
                "a collect value of {} for {elem}",
                chunk.data_type()
            )));
        }
    })
}

/// The rows `idx` of `col`, in order.
fn take_rows(col: &ArrayRef, idx: &[u32]) -> Result<ArrayRef> {
    let idx = UInt32Array::from(idx.to_vec());
    take(col.as_ref(), &idx, None).map_err(|e| internal(e.to_string()))
}

/// The per-group state of one collect aggregate: appended chunks in fold
/// order, with the distinct rows for the set.
#[derive(Debug)]
pub(crate) struct CollectAcc {
    /// One chunk per appended batch slice, per group, in fold order.
    values: Vec<Vec<ArrayRef>>,
    /// Encoded distinct rows per group, in first-seen order (the set only).
    seen: Vec<KeyTable>,
    /// The distinct-row encoding (the set only).
    encoder: Option<KeyEncoder>,
    /// The hash seed of the distinct tables (fixed: the hash is internal,
    /// never observable).
    seed: HashSeed,
    /// The accumulator's element type (states and results).
    elem: DataType,
    /// Heap bytes: chunk buffers plus distinct rows.
    heap: usize,
}

impl CollectAcc {
    /// An accumulator of element type `element`, deduping when `set`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an element type the shared key
    /// encoding refuses.
    pub(crate) fn new(element: DataType, set: bool) -> Result<Self> {
        let encoder = set
            .then(|| KeyEncoder::new(std::slice::from_ref(&element)))
            .transpose()?;
        Ok(Self {
            values: Vec::new(),
            seen: Vec::new(),
            encoder,
            seed: HashSeed::new(Some(0)),
            elem: element,
            heap: 0,
        })
    }

    /// Every per-group vector as (length, capacity, element size).
    pub(crate) fn vecs(&self) -> Vec<(usize, usize, usize)> {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        vec![v(&self.values), v(&self.seen)]
    }

    /// Bytes held.
    pub(crate) fn size(&self) -> usize {
        self.heap + self.vecs().iter().map(|(_, c, e)| c * e).sum::<usize>()
    }

    /// [`size`](Self::size) after [`resize`](Self::resize) to `new` more
    /// groups and an update with `arg` (twice its kept bytes: the append
    /// plus the transient take and encode buffers).
    pub(crate) fn projected(&self, new: usize, arg: Option<&ArrayRef>) -> usize {
        let vecs: usize = self
            .vecs()
            .iter()
            .map(|&(len, cap, e)| {
                let need = len.saturating_add(new);
                let c = if need <= cap {
                    cap
                } else {
                    need.max(cap * 2).max(if e == 1 { 8 } else { 4 })
                };
                c * e
            })
            .sum();
        let keep = arg.map(kept_bytes).unwrap_or(0).saturating_mul(2);
        vecs + self.heap + keep
    }

    /// Adds groups up to `n` with empty states.
    pub(crate) fn resize(&mut self, n: usize) {
        self.values.resize_with(n, Vec::new);
        self.seen.resize_with(n, KeyTable::default);
    }

    /// Reserves room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        self.values.reserve(new);
        self.seen.reserve(new);
    }

    /// Folds the valid rows of `col` into `groups`: one take per touched
    /// group (in row order), deduped for the set. With `run`, folds
    /// row-by-row so each snapshot sees the state after its row.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        col: &ArrayRef,
        valid: &dyn Fn(usize) -> bool,
        run: Option<&Running<'_>>,
    ) -> Result<()> {
        if col.len() != groups.len() {
            return Err(internal(
                "a collect column of another length than the group ids".to_owned(),
            ));
        }
        if matches!(col.data_type(), DataType::Null) {
            // An untyped NULL column holds no values; snapshots still copy
            // the (empty) states.
            if let Some(r) = run {
                for (i, &g) in groups.iter().enumerate() {
                    let _ = (i, g);
                    let to = r.snapshots[i];
                    if to != Running::NONE {
                        let (to, from) = (to as usize, r.parts[i] as usize);
                        self.values[to] = self.values[from].clone();
                        self.seen[to] = self.seen[from].clone();
                    }
                }
            }
            return Ok(());
        }
        if run.is_some() {
            for (i, &g) in groups.iter().enumerate() {
                if valid(i) {
                    self.append(g as usize, col.slice(i, 1))?;
                }
                self.snapshot(run, i);
            }
            return Ok(());
        }
        let mut order: Vec<Vec<u32>> = vec![Vec::new(); self.values.len()];
        for (i, &g) in groups.iter().enumerate() {
            if valid(i) {
                order[g as usize].push(
                    u32::try_from(i)
                        .map_err(|_| internal("a collect batch past 2^32 rows".to_owned()))?,
                );
            }
        }
        for (g, idx) in order.iter().enumerate() {
            if !idx.is_empty() {
                self.append(g, take_rows(col, idx)?)?;
            }
        }
        Ok(())
    }

    /// Merges one state column (lists of collected values) into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], state: &ArrayRef) -> Result<()> {
        let Some(list) = state.as_any().downcast_ref::<ListArray>() else {
            return Err(internal(format!(
                "a collect state of another type than a list: {}",
                state.data_type()
            )));
        };
        if list.len() != groups.len() {
            return Err(internal(
                "a collect state of another length than the group ids".to_owned(),
            ));
        }
        let nulls = state.logical_nulls();
        let values = list.values();
        let offsets = list.value_offsets();
        for (i, &g) in groups.iter().enumerate() {
            if nulls.as_ref().is_some_and(|n| !n.is_valid(i)) {
                continue;
            }
            let len = usize::try_from(list.value_length(i).max(0)).unwrap_or(0);
            if len == 0 {
                continue;
            }
            let start = usize::try_from(offsets[i].max(0)).unwrap_or(0);
            self.append(g as usize, values.slice(start, len))?;
        }
        Ok(())
    }

    /// Appends `chunk` to group `g` (deduped for the set).
    fn append(&mut self, g: usize, chunk: ArrayRef) -> Result<()> {
        let chunk = normalize(chunk, &self.elem)?;
        let chunk = match &self.encoder {
            None => chunk,
            Some(_) => match self.insert_new(g, &chunk)? {
                Some(kept) => kept,
                None => return Ok(()),
            },
        };
        if chunk.is_empty() {
            return Ok(());
        }
        self.heap = self.heap.saturating_add(kept_bytes(&chunk));
        self.values[g].push(chunk);
        Ok(())
    }

    /// `chunk`'s unseen rows, in order (`None` when every row repeats).
    fn insert_new(&mut self, g: usize, chunk: &ArrayRef) -> Result<Option<ArrayRef>> {
        let Some(enc) = &self.encoder else {
            return Ok(Some(Arc::clone(chunk)));
        };
        let rows = enc.encode(std::slice::from_ref(chunk))?;
        let before = self.seen[g].memory_size();
        let mut ids = Vec::with_capacity(chunk.len());
        let new = self.seen[g].find_or_insert(&rows, &self.seed, &mut ids)?;
        self.heap = self
            .heap
            .saturating_add(self.seen[g].memory_size().saturating_sub(before));
        if new == 0 {
            return Ok(None);
        }
        if new == chunk.len() {
            return Ok(Some(Arc::clone(chunk)));
        }
        // New ids are assigned sequentially from the previous length, in
        // row order: an id is new exactly when it is the next one.
        let mut next = self.seen[g].len().saturating_sub(new);
        let mut keep = Vec::with_capacity(new);
        for (j, &id) in ids.iter().enumerate() {
            if id as usize == next {
                next += 1;
                keep.push(
                    u32::try_from(j)
                        .map_err(|_| internal("a collect batch past 2^32 rows".to_owned()))?,
                );
            }
        }
        Ok(Some(take_rows(chunk, &keep)?))
    }

    /// Copies group `from`'s state to snapshot group `to` after row `i`
    /// (a running frame's snapshot; `None` when not running).
    fn snapshot(&mut self, run: Option<&Running<'_>>, i: usize) {
        if let Some(r) = run {
            let to = r.snapshots[i];
            if to != Running::NONE {
                let (to, from) = (to as usize, r.parts[i] as usize);
                self.values[to] = self.values[from].clone();
                self.seen[to] = self.seen[from].clone();
            }
        }
    }

    /// The state column of groups `ids`: one list of the collected values
    /// per group (never NULL). The final values are the same lists.
    pub(crate) fn state(&self, ids: &[u32], result: &DataType) -> Result<ArrayRef> {
        let DataType::List(field) = result else {
            return Err(internal(format!(
                "a collect result of another type than an array: {result}"
            )));
        };
        let elem = field.data_type();
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        offsets.push(0_i32);
        let mut children = Vec::new();
        let mut total = 0_i32;
        for i in ids {
            for chunk in &self.values[*i as usize] {
                if chunk.data_type() != elem {
                    return Err(internal(format!(
                        "a collect chunk of {} for {elem}",
                        chunk.data_type()
                    )));
                }
                children.push(Arc::clone(chunk));
                total = total
                    .checked_add(
                        i32::try_from(chunk.len())
                            .map_err(|_| internal("a collect group past 2^31 values".to_owned()))?,
                    )
                    .ok_or_else(|| internal("a collect group past 2^31 values".to_owned()))?;
            }
            offsets.push(total);
        }
        let child = if children.is_empty() {
            arrow_array::new_empty_array(elem)
        } else {
            let refs: Vec<&dyn Array> = children.iter().map(AsRef::as_ref).collect();
            concat(&refs).map_err(|e| internal(e.to_string()))?
        };
        ListArray::try_new(
            Arc::clone(field),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            child,
            None,
        )
        .map_err(|e| internal(e.to_string()))
        .map(|a| Arc::new(a) as ArrayRef)
    }
}

/// The per-group state of one vector aggregate: running sums and the
/// count of non-NULL vectors (`None` until one arrives).
#[derive(Debug)]
pub(crate) struct VectorAcc {
    /// Running sums per group (`None`: no non-NULL vector yet).
    sums: Vec<Option<Vec<f64>>>,
    /// Non-NULL vectors per group (empty ones included).
    count: Vec<i64>,
    /// `vector_avg` (divide by the count) rather than `vector_sum`.
    avg: bool,
}

impl VectorAcc {
    /// An accumulator for `vector_avg` when `avg`, else `vector_sum`.
    pub(crate) fn new(avg: bool) -> Self {
        Self {
            sums: Vec::new(),
            count: Vec::new(),
            avg,
        }
    }

    /// The SQL name (for `VECTOR_DIMENSION_MISMATCH`).
    fn name(&self) -> &'static str {
        if self.avg { "vector_avg" } else { "vector_sum" }
    }

    /// Every per-group vector as (length, capacity, element size).
    pub(crate) fn vecs(&self) -> Vec<(usize, usize, usize)> {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        vec![v(&self.sums), v(&self.count)]
    }

    /// Bytes held (the sums live on the heap, past their `Vec` shells).
    pub(crate) fn size(&self) -> usize {
        let vecs: usize = self.vecs().iter().map(|(_, c, e)| c * e).sum();
        let sums: usize = self
            .sums
            .iter()
            .flatten()
            .map(|s| s.capacity() * size_of::<f64>())
            .sum();
        vecs + sums
    }

    /// [`size`](Self::size) after [`resize`](Self::resize) to `new` more
    /// groups and an update with `arg` (twice its kept bytes in `f64`
    /// sums, plus the transient buffers).
    pub(crate) fn projected(&self, new: usize, arg: Option<&ArrayRef>) -> usize {
        let vecs: usize = self
            .vecs()
            .iter()
            .map(|&(len, cap, e)| {
                let need = len.saturating_add(new);
                let c = if need <= cap {
                    cap
                } else {
                    need.max(cap * 2).max(if e == 1 { 8 } else { 4 })
                };
                c * e
            })
            .sum();
        let sums: usize = self
            .sums
            .iter()
            .flatten()
            .map(|s| s.capacity() * size_of::<f64>())
            .sum();
        let keep = arg.map(kept_bytes).unwrap_or(0).saturating_mul(2);
        vecs + sums + keep
    }

    /// Adds groups up to `n` with empty states.
    pub(crate) fn resize(&mut self, n: usize) {
        self.sums.resize_with(n, || None);
        self.count.resize(n, 0);
    }

    /// Reserves room for `new` more groups.
    pub(crate) fn reserve(&mut self, new: usize) {
        self.sums.reserve(new);
        self.count.reserve(new);
    }

    /// Folds the non-NULL vectors of `col` into `groups`, in order.
    pub(crate) fn update(
        &mut self,
        groups: &[u32],
        col: &ArrayRef,
        valid: &dyn Fn(usize) -> bool,
        run: Option<&Running<'_>>,
    ) -> Result<()> {
        if col.len() != groups.len() {
            return Err(internal(
                "a vector column of another length than the group ids".to_owned(),
            ));
        }
        if matches!(col.data_type(), DataType::Null) {
            if run.is_some() {
                // An untyped NULL column holds no values; snapshots still
                // copy the (empty) states.
                for (i, &g) in groups.iter().enumerate() {
                    let _ = g;
                    self.snapshot(run, i);
                }
            }
            return Ok(());
        }
        let Some(list) = col.as_any().downcast_ref::<ListArray>() else {
            return Err(internal(format!(
                "a vector of another type than a list: {}",
                col.data_type()
            )));
        };
        let Some(values) = list.values().as_primitive_opt::<Float32Type>() else {
            return Err(internal(format!(
                "a vector of another element type than float: {}",
                list.values().data_type()
            )));
        };
        let offsets = list.value_offsets();
        for (i, &g) in groups.iter().enumerate() {
            if valid(i) {
                let len = usize::try_from(list.value_length(i).max(0)).unwrap_or(0);
                let start = usize::try_from(offsets[i].max(0)).unwrap_or(0);
                // The child's null buffer is absolute (the child is never
                // sliced); `start` is absolute too.
                self.add(g as usize, start, len, values.values(), values.nulls())?;
            }
            self.snapshot(run, i);
        }
        Ok(())
    }

    /// Merges one state (sums with counts) into `groups`.
    pub(crate) fn merge(&mut self, groups: &[u32], states: &[&ArrayRef]) -> Result<()> {
        let [sums_col, count_col] = states else {
            return Err(internal(
                "a vector state of other columns than (sums, count)".to_owned(),
            ));
        };
        let Some(list) = sums_col.as_any().downcast_ref::<ListArray>() else {
            return Err(internal(format!(
                "a vector sums state of another type than a list: {}",
                sums_col.data_type()
            )));
        };
        let Some(values) = list.values().as_primitive_opt::<Float64Type>() else {
            return Err(internal(format!(
                "a vector sums state of another element type than double: {}",
                list.values().data_type()
            )));
        };
        if list.len() != groups.len() {
            return Err(internal(
                "a vector state of another length than the group ids".to_owned(),
            ));
        }
        let counts = counts_of(count_col)?;
        if counts.len() != groups.len() {
            return Err(internal(
                "a vector count state of another length than the group ids".to_owned(),
            ));
        }
        let nulls = sums_col.logical_nulls();
        let offsets = list.value_offsets();
        for (i, &g) in groups.iter().enumerate() {
            if nulls.as_ref().is_some_and(|n| !n.is_valid(i)) {
                continue;
            }
            let len = usize::try_from(list.value_length(i).max(0)).unwrap_or(0);
            let start = usize::try_from(offsets[i].max(0)).unwrap_or(0);
            self.add_state(
                g as usize,
                &values.values()[start..start.saturating_add(len).min(values.len())],
                counts[i],
            )?;
        }
        Ok(())
    }

    /// Adds one raw vector's values (NULL elements contribute nothing).
    /// `values` is the absolute child array, `start` the vector's first
    /// element (both absolute, as the null buffer is). A dimension-zero
    /// state (only empty vectors so far) adopts the first non-empty
    /// dimension; empty vectors are otherwise neutral but counted.
    fn add(
        &mut self,
        g: usize,
        start: usize,
        len: usize,
        values: &[f32],
        nulls: Option<&arrow_buffer::NullBuffer>,
    ) -> Result<()> {
        let name = self.name();
        match &mut self.sums[g] {
            None => self.sums[g] = Some(vec![0.0; len]),
            // Neutral: only empty vectors so far.
            Some(s) if s.is_empty() && len != 0 => self.sums[g] = Some(vec![0.0; len]),
            Some(s) if !s.is_empty() && len != 0 && s.len() != len => {
                let held = s.len();
                return Err(dimension_mismatch(name, held, len));
            }
            _ => {}
        }
        let s = self.sums[g]
            .as_mut()
            .ok_or_else(|| internal("a vector group without sums".to_owned()))?;
        let end = start.saturating_add(len).min(values.len());
        match nulls {
            None => {
                for (s, &v) in s.iter_mut().zip(&values[start..end]) {
                    *s += f64::from(v);
                }
            }
            Some(n) => {
                for (k, (s, &v)) in s.iter_mut().zip(&values[start..end]).enumerate() {
                    if n.is_valid(start + k) {
                        *s += f64::from(v);
                    }
                }
            }
        }
        self.count[g] = self.count[g].wrapping_add(1);
        Ok(())
    }

    /// Adds one state (sums with its count). A dimension-zero state is
    /// neutral, as for [`add`](Self::add).
    fn add_state(&mut self, g: usize, values: &[f64], count: i64) -> Result<()> {
        let len = values.len();
        let name = self.name();
        match &mut self.sums[g] {
            None => self.sums[g] = Some(vec![0.0; len]),
            Some(s) if s.is_empty() && len != 0 => self.sums[g] = Some(vec![0.0; len]),
            Some(s) if !s.is_empty() && len != 0 && s.len() != len => {
                return Err(dimension_mismatch(name, s.len(), len));
            }
            _ => {}
        }
        let s = self.sums[g]
            .as_mut()
            .ok_or_else(|| internal("a vector group without sums".to_owned()))?;
        for (s, &v) in s.iter_mut().zip(values) {
            *s += v;
        }
        self.count[g] = self.count[g].wrapping_add(count);
        Ok(())
    }

    /// Copies group `from`'s state to snapshot group `to` after row `i`
    /// (a running frame's snapshot; `None` when not running).
    fn snapshot(&mut self, run: Option<&Running<'_>>, i: usize) {
        if let Some(r) = run {
            let to = r.snapshots[i];
            if to != Running::NONE {
                let (to, from) = (to as usize, r.parts[i] as usize);
                self.sums[to] = self.sums[from].clone();
                self.count[to] = self.count[from];
            }
        }
    }

    /// The sums state column of groups `ids` (NULL: no non-NULL vector).
    /// The element field is `element` of `double`, non-nullable, as
    /// [`super::vector_sums_type`] lays the state out.
    pub(crate) fn state_sums(&self, ids: &[u32]) -> Result<ArrayRef> {
        let field = Arc::new(Field::new("element", DataType::Float64, false));
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        offsets.push(0_i32);
        let mut values = Vec::new();
        let mut validity = Vec::with_capacity(ids.len());
        let mut total = 0_i32;
        for i in ids {
            match &self.sums[*i as usize] {
                None => validity.push(false),
                Some(s) => {
                    validity.push(true);
                    values.extend_from_slice(s);
                    total =
                        total
                            .checked_add(i32::try_from(s.len()).map_err(|_| {
                                internal("a vector past 2^31 dimensions".to_owned())
                            })?)
                            .ok_or_else(|| internal("a vector past 2^31 dimensions".to_owned()))?;
                }
            }
            offsets.push(total);
        }
        let nulls = arrow_buffer::NullBuffer::from(validity);
        ListArray::try_new(
            field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(Float64Array::from(values)),
            Some(nulls),
        )
        .map_err(|e| internal(e.to_string()))
        .map(|a| Arc::new(a) as ArrayRef)
    }

    /// The count state column of groups `ids`.
    pub(crate) fn state_count(&self, ids: &[u32]) -> ArrayRef {
        Arc::new(Int64Array::from_iter_values(
            ids.iter().map(|i| self.count[*i as usize]),
        ))
    }

    /// The final values of groups `ids`: element-wise sums, or means.
    pub(crate) fn evaluate(&self, ids: &[u32], result: &DataType) -> Result<ArrayRef> {
        let DataType::List(field) = result else {
            return Err(internal(format!(
                "a vector result of another type than an array: {result}"
            )));
        };
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        offsets.push(0_i32);
        let mut values = Vec::new();
        let mut validity = Vec::with_capacity(ids.len());
        let mut total = 0_i32;
        for i in ids {
            match &self.sums[*i as usize] {
                None => validity.push(false),
                Some(s) => {
                    validity.push(true);
                    if self.avg {
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "Spark divides by the count as double"
                        )]
                        let n = self.count[*i as usize] as f64;
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "Spark's vectors are float, the sums double"
                        )]
                        values.extend(s.iter().map(|&v| (v / n) as f32));
                    } else {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "the state holds the running sums as doubles"
                        )]
                        values.extend(s.iter().map(|&v| v as f32));
                    }
                    total =
                        total
                            .checked_add(i32::try_from(s.len()).map_err(|_| {
                                internal("a vector past 2^31 dimensions".to_owned())
                            })?)
                            .ok_or_else(|| internal("a vector past 2^31 dimensions".to_owned()))?;
                }
            }
            offsets.push(total);
        }
        let nulls = arrow_buffer::NullBuffer::from(validity);
        ListArray::try_new(
            Arc::clone(field),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(Float32Array::from(values)),
            Some(nulls),
        )
        .map_err(|e| internal(e.to_string()))
        .map(|a| Arc::new(a) as ArrayRef)
    }
}

/// A `VECTOR_DIMENSION_MISMATCH` of `name` (`left` held, `right` incoming).
fn dimension_mismatch(
    name: &str,
    left: usize,
    right: usize,
) -> krishiv_common::error::KrishivError {
    VectorDimensionMismatch {
        function_name: format!("`{name}`"),
        left_dim: left.to_string(),
        right_dim: right.to_string(),
    }
    .into()
}

/// A `bigint` state column (the vector counts).
fn counts_of(a: &ArrayRef) -> Result<&[i64]> {
    a.as_primitive_opt::<Int64Type>()
        .map(|c| c.values().as_ref())
        .ok_or_else(|| internal(format!("count state of type {}", a.data_type())))
}

#[cfg(test)]
mod tests {
    use arrow_array::{BinaryArray, LargeBinaryArray, StringArray};
    use arrow_buffer::NullBuffer;

    use super::*;

    #[test]
    fn kept_bytes_counts_every_variable_width_carrier() {
        let binary: ArrayRef = Arc::new(BinaryArray::from_iter_values([b"ab".as_ref(), b"c"]));
        let large: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values([b"ab".as_ref(), b"c"]));
        let utf8: ArrayRef = Arc::new(StringArray::from(vec!["ab", "c"]));
        // `Binary` used to panic (read as `Utf8`); each counts the data.
        for (name, a) in [("binary", &binary), ("large", &large), ("utf8", &utf8)] {
            assert!(kept_bytes(a) >= 3, "{name}");
        }
    }

    /// An empty vector is neutral even when the child array has NULLs: it
    /// reads none of the next vectors' slots (and none past the array).
    #[test]
    fn an_empty_vector_adds_nothing_with_a_null_buffer() {
        // Child array: [1, NULL, | (empty) | NULL, 5].
        let values = [1.0_f32, 0.0, 0.0, 5.0];
        let nulls = NullBuffer::from(vec![true, false, false, true]);
        let mut v = VectorAcc::new(false);
        v.resize(1);
        v.add(0, 0, 2, &values, Some(&nulls)).unwrap();
        v.add(0, 2, 0, &values, Some(&nulls)).unwrap();
        v.add(0, 2, 2, &values, Some(&nulls)).unwrap();
        assert_eq!(v.sums[0], Some(vec![1.0, 5.0]));
        assert_eq!(v.count[0], 3);
        // An empty vector at the very end of the child array.
        let nulls = NullBuffer::from(vec![true, false]);
        let mut v = VectorAcc::new(false);
        v.resize(1);
        v.add(0, 0, 2, &values[..2], Some(&nulls)).unwrap();
        v.add(0, 2, 0, &values[..2], Some(&nulls)).unwrap();
        assert_eq!(v.sums[0], Some(vec![1.0, 0.0]));
    }

    /// Every carrier of variable-width and nested values counts the bytes
    /// it keeps, never less than its values.
    #[test]
    fn kept_bytes_covers_views_fixed_binary_and_lists() {
        use arrow_array::{FixedSizeBinaryArray, LargeStringArray};
        let views: ArrayRef = Arc::new(StringViewArray::from(vec![
            "a",
            "a value longer than twelve bytes",
        ]));
        assert!(kept_bytes(&views) >= 32 + 33);
        let bviews: ArrayRef = Arc::new(BinaryViewArray::from(vec![b"ab".as_ref(), b"c"]));
        assert!(kept_bytes(&bviews) >= 32 + 3);
        let fixed: ArrayRef =
            Arc::new(FixedSizeBinaryArray::try_from_iter([[1_u8, 2], [3, 4]].into_iter()).unwrap());
        assert!(kept_bytes(&fixed) >= 4);
        let large: ArrayRef = Arc::new(LargeStringArray::from(vec!["ab", "c"]));
        assert!(kept_bytes(&large) >= 3);
        let child: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4]));
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("e", DataType::Int64, true)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4])),
            child,
            None,
        ));
        assert!(kept_bytes(&list) >= 32);
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(kept_bytes(&ints) >= 16);
    }

    /// Raw string and binary chunks convert to the view element type; any
    /// other pairing is refused rather than concatenated wrongly.
    #[test]
    fn normalize_converts_carriers_and_refuses_others() {
        use arrow_array::LargeStringArray;
        let utf8: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        let large: ArrayRef = Arc::new(LargeStringArray::from(vec!["a"]));
        let binary: ArrayRef = Arc::new(BinaryArray::from_iter_values([b"a".as_ref()]));
        let lbinary: ArrayRef = Arc::new(LargeBinaryArray::from_iter_values([b"a".as_ref()]));
        for c in [&utf8, &large] {
            let out = normalize(Arc::clone(c), &DataType::Utf8View).unwrap();
            assert_eq!(out.data_type(), &DataType::Utf8View);
        }
        for c in [&binary, &lbinary] {
            let out = normalize(Arc::clone(c), &DataType::BinaryView).unwrap();
            assert_eq!(out.data_type(), &DataType::BinaryView);
        }
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert!(normalize(ints, &DataType::Utf8View).is_err());
    }

    fn list_of(values: Vec<i64>, offsets: Vec<i32>, nulls: Option<NullBuffer>) -> ArrayRef {
        Arc::new(ListArray::new(
            Arc::new(Field::new("e", DataType::Int64, true)),
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(Int64Array::from(values)),
            nulls,
        ))
    }

    /// Merging partial states appends the lists of the non-NULL rows to
    /// their groups, and refuses a state of another type or length.
    #[test]
    fn merge_appends_lists_and_refuses_misshapen_states() {
        let mut acc = CollectAcc::new(DataType::Int64, false).unwrap();
        acc.resize(2);
        // Rows: [1, 2], NULL, [], [3].
        let state = list_of(
            vec![1, 2, 3],
            vec![0, 2, 2, 2, 3],
            Some(NullBuffer::from(vec![true, false, true, true])),
        );
        acc.merge(&[0, 0, 1, 1], &state).unwrap();
        let out = acc
            .state(
                &[0, 1],
                &DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            )
            .unwrap();
        let out = out.as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(out.value_length(0), 2);
        assert_eq!(out.value_length(1), 1);
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(acc.merge(&[0, 0], &ints).is_err());
        assert!(acc.merge(&[0], &state).is_err());
        assert!(acc.state(&[0], &DataType::Int64).is_err());
    }

    /// A column of another length than the group ids is refused; an untyped
    /// NULL column adds nothing.
    #[test]
    fn update_refuses_a_column_of_another_length() {
        let mut acc = CollectAcc::new(DataType::Int64, false).unwrap();
        acc.resize(1);
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        assert!(acc.update(&[0], &col, &|_| true, None).is_err());
        let nulls: ArrayRef = arrow_array::new_null_array(&DataType::Null, 2);
        acc.update(&[0, 0], &nulls, &|_| true, None).unwrap();
        assert!(acc.values[0].is_empty());
    }
}
