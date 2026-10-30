// SPDX-License-Identifier: Apache-2.0
//! The `listagg` accumulator: the non-NULL values of a group joined by a
//! delimiter, in input order or sorted by ordering keys.
//!
//! A group keeps the non-NULL value chunks in fold order with the
//! ordering-key chunks they were folded with (parallel takes of the same
//! rows, as the collect accumulator folds them). Merging concatenates the
//! chunks, so partial states merge associatively and a partial phase, an
//! exchange and a final phase compose. Unordered groups merge in
//! partition-index order (see [`is_order_sensitive`](super::is_order_sensitive));
//! ordered groups sort once, at evaluation, so the merge order never
//! matters.
//!
//! The partial state is one list per struct field (the values, then one
//! list per ordering key), all Arrow arrays. NULL values never enter the
//! state; NULL ordering keys do (they order first or last, as written).
//!
//! # Memory
//! Chunks share their inputs' buffers; [`kept_bytes`] (as the collect
//! accumulator's reservation) bounds what an append keeps, and the
//! accumulator's reservation (made by the table from
//! [`ListAggAcc::projected`] before folding) covers it.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, ListArray, UInt32Array};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, TimeUnit};
use arrow_select::concat::concat;
use arrow_select::take::take;

use super::collect::{kept_bytes, normalize};
use crate::error::{Result, internal};

/// One group's values with the ordering keys they were folded with.
#[derive(Debug, Clone, Default)]
struct Group {
    /// Non-NULL value chunks, in fold order.
    values: Vec<ArrayRef>,
    /// Ordering-key chunks, parallel to [`values`](Group::values): one
    /// vector per key.
    keys: Vec<Vec<ArrayRef>>,
}

/// The per-group state of one `listagg` aggregate.
#[derive(Debug)]
pub(super) struct ListAggAcc {
    /// The delimiter bytes (empty when none or NULL was written).
    delimiter: Vec<u8>,
    /// Whether the values (and the delimiter) are binary.
    binary: bool,
    /// One per ordering key, in order: descending, nulls first.
    orders: Vec<(bool, bool)>,
    /// The value type (`STRING` or `BINARY` as Arrow views).
    value_type: DataType,
    /// The ordering-key types.
    key_types: Vec<DataType>,
    groups: Vec<Group>,
    /// Heap bytes: value and key chunks.
    heap: usize,
}

impl ListAggAcc {
    /// An accumulator joining `value_type` values (of `delimiter`, with
    /// `orders` directions) over ordering keys of `key_types`.
    pub(super) fn new(
        delimiter: Vec<u8>,
        orders: Vec<(bool, bool)>,
        value_type: DataType,
        key_types: Vec<DataType>,
    ) -> Result<Self> {
        // Only keys with a value ordering here sort: the others would
        // compare equal and leave the values unsorted without an error.
        if let Some(t) = key_types.iter().find(|t| !orderable(t)) {
            return Err(super::unsupported(format!(
                "listagg ordered by a key of type {t}"
            )));
        }
        let binary = matches!(
            value_type,
            DataType::Binary | DataType::BinaryView | DataType::LargeBinary
        );
        Ok(Self {
            delimiter,
            binary,
            orders,
            value_type,
            key_types,
            groups: Vec::new(),
            heap: 0,
        })
    }

    /// Every per-group vector as (length, capacity, element size).
    pub(super) fn vecs(&self) -> Vec<(usize, usize, usize)> {
        fn v<T>(x: &Vec<T>) -> (usize, usize, usize) {
            (x.len(), x.capacity(), size_of::<T>())
        }
        vec![v(&self.groups)]
    }

    /// Bytes held.
    pub(super) fn size(&self) -> usize {
        self.heap + self.vecs().iter().map(|(_, c, e)| c * e).sum::<usize>()
    }

    /// [`size`](Self::size) after [`resize`](Self::resize) to `new` more
    /// groups and an update with `arg` (twice its kept bytes: the transient
    /// take buffers plus the kept chunks).
    pub(super) fn projected(&self, new: usize, arg: Option<&ArrayRef>) -> usize {
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
    pub(super) fn resize(&mut self, n: usize) {
        self.groups.resize_with(n, || Group {
            values: Vec::new(),
            keys: vec![Vec::new(); self.key_types.len()],
        });
    }

    /// Reserves room for `new` more groups.
    pub(super) fn reserve(&mut self, new: usize) {
        self.groups.reserve(new);
    }

    /// Folds the valid rows of `cols` (`values`, then one ordering-key
    /// column per key) into `groups`.
    pub(super) fn fold(
        &mut self,
        groups: &[u32],
        cols: &[&ArrayRef],
        valid: &dyn Fn(usize) -> bool,
    ) -> Result<()> {
        let [values, keys @ ..] = cols else {
            return Err(internal(
                "a listagg fold of another shape than its value and keys".to_owned(),
            ));
        };
        if keys.len() != self.key_types.len()
            || values.len() != groups.len()
            || keys.iter().any(|k| k.len() != groups.len())
        {
            return Err(internal(
                "a listagg column of another length than the group ids".to_owned(),
            ));
        }
        if matches!(values.data_type(), DataType::Null) {
            return Ok(());
        }
        // One take per touched group, in row order (as the collect
        // accumulator folds them).
        let mut order: Vec<Vec<u32>> = vec![Vec::new(); self.groups.len()];
        for (i, &g) in groups.iter().enumerate() {
            if valid(i) {
                order[g as usize].push(
                    u32::try_from(i)
                        .map_err(|_| internal("a listagg batch past 2^32 rows".to_owned()))?,
                );
            }
        }
        for (g, idx) in order.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let idx = UInt32Array::from(idx.clone());
            let chunk = take(values.as_ref(), &idx, None).map_err(|e| internal(e.to_string()))?;
            let mut key_chunks = Vec::with_capacity(keys.len());
            for key in keys {
                key_chunks
                    .push(take(key.as_ref(), &idx, None).map_err(|e| internal(e.to_string()))?);
            }
            self.append(g, &chunk, &key_chunks)?;
        }
        Ok(())
    }

    /// Appends `chunk` (no NULLs) with its `key_chunks` to group `g`.
    fn append(&mut self, g: usize, chunk: &ArrayRef, key_chunks: &[ArrayRef]) -> Result<()> {
        if key_chunks.len() != self.key_types.len() {
            return Err(internal(
                "a listagg append of another shape than its keys".to_owned(),
            ));
        }
        let chunk = normalize(Arc::clone(chunk), &self.value_type)?;
        let mut keys = Vec::with_capacity(key_chunks.len());
        for (key, t) in key_chunks.iter().zip(&self.key_types) {
            keys.push(normalize(Arc::clone(key), t)?);
        }
        let group = &mut self.groups[g];
        self.heap = self.heap.saturating_add(kept_bytes(&chunk));
        group.values.push(chunk);
        for (stored, key) in group.keys.iter_mut().zip(keys) {
            self.heap = self.heap.saturating_add(kept_bytes(&key));
            stored.push(key);
        }
        Ok(())
    }

    /// Merges partial states (values lists, then one key list per key)
    /// into `groups`.
    pub(super) fn merge(&mut self, groups: &[u32], states: &[&ArrayRef]) -> Result<()> {
        let lists: Vec<&ListArray> = states
            .iter()
            .map(|s| {
                s.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                    internal(format!(
                        "a listagg state of another type than a list: {}",
                        s.data_type()
                    ))
                })
            })
            .collect::<Result<_>>()?;
        if lists.len() != self.key_types.len() + 1 {
            return Err(internal(
                "a listagg state of another shape than its values and keys".to_owned(),
            ));
        }
        if lists.iter().any(|l| l.len() != groups.len()) {
            return Err(internal(
                "a listagg state of another length than the group ids".to_owned(),
            ));
        }
        for (i, &g) in groups.iter().enumerate() {
            if lists.iter().any(|l| l.is_null(i)) {
                continue;
            }
            let mut chunks: Vec<ArrayRef> = Vec::with_capacity(lists.len());
            let mut len = None;
            for l in &lists {
                let start = usize::try_from(l.value_offsets()[i].max(0)).unwrap_or(0);
                let n = usize::try_from(l.value_length(i).max(0)).unwrap_or(0);
                if len.is_some_and(|m| m != n) {
                    return Err(internal(
                        "a listagg state whose values and keys differ".to_owned(),
                    ));
                }
                len = Some(n);
                chunks.push(l.values().slice(start, n));
            }
            if len == Some(0) {
                continue;
            }
            let (chunk, key_chunks) = chunks
                .split_first()
                .ok_or_else(|| internal("a listagg state without values".to_owned()))?;
            // The merged values hold no NULLs by construction.
            self.append(g as usize, chunk, key_chunks)?;
        }
        Ok(())
    }

    /// Copies group `from`'s state to snapshot group `to` (a running
    /// frame's snapshot; shares the value and key chunks).
    pub(super) fn snapshot(&mut self, from: usize, to: usize) {
        self.groups[to] = self.groups[from].clone();
    }

    /// The partial states of groups `ids`: the values, then one list per
    /// ordering key (never NULL; empty groups hold empty lists).
    pub(super) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let mut lists: Vec<ArrayRef> = Vec::new();
        let types: Vec<DataType> = std::iter::once(self.value_type.clone())
            .chain(self.key_types.iter().cloned())
            .collect();
        for (f, t) in types.iter().enumerate() {
            let mut children: Vec<ArrayRef> = Vec::new();
            let mut offsets = Vec::with_capacity(ids.len() + 1);
            offsets.push(0_i32);
            let mut total = 0_i32;
            for &g in ids {
                let group = &self.groups[g as usize];
                let chunks = if f == 0 {
                    &group.values
                } else {
                    &group.keys[f - 1]
                };
                for chunk in chunks {
                    children.push(Arc::clone(chunk));
                    total = total
                        .checked_add(
                            i32::try_from(chunk.len()).map_err(|_| {
                                internal("a listagg group past 2^31 values".to_owned())
                            })?,
                        )
                        .ok_or_else(|| internal("a listagg group past 2^31 values".to_owned()))?;
                }
                offsets.push(total);
            }
            let child = if children.is_empty() {
                arrow_array::new_empty_array(t)
            } else {
                let refs: Vec<&dyn Array> = children.iter().map(AsRef::as_ref).collect();
                concat(&refs).map_err(|e| internal(e.to_string()))?
            };
            let field = Arc::new(Field::new("element", t.clone(), true));
            lists.push(
                ListArray::try_new(
                    field,
                    OffsetBuffer::new(ScalarBuffer::from(offsets)),
                    child,
                    None,
                )
                .map_err(|e| internal(e.to_string()))
                .map(|a| Arc::new(a) as ArrayRef)?,
            );
        }
        Ok(lists)
    }

    /// The final values of groups `ids` (NULL for no value).
    pub(super) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(ids.len());
        for &g in ids {
            out.push(self.joined(g)?);
        }
        if self.binary {
            let array: arrow_array::BinaryViewArray = out.iter().map(|v| v.as_deref()).collect();
            Ok(Arc::new(array))
        } else {
            use arrow_array::builder::StringViewBuilder;
            let mut array = StringViewBuilder::new();
            for v in &out {
                match v {
                    Some(bytes) => array.append_value(
                        std::str::from_utf8(bytes).map_err(|e| internal(e.to_string()))?,
                    ),
                    None => array.append_null(),
                }
            }
            Ok(Arc::new(array.finish()))
        }
    }

    /// The joined non-NULL values of group `g` (none for no value).
    fn joined(&self, g: u32) -> Result<Option<Vec<u8>>> {
        let group = &self.groups[g as usize];
        if group.values.is_empty() {
            return Ok(None);
        }
        // Row (chunk, row) pairs in output order: fold order, or sorted
        // by the ordering keys once.
        let mut rows: Vec<(usize, usize)> = group
            .values
            .iter()
            .enumerate()
            .flat_map(|(c, chunk)| (0..chunk.len()).map(move |r| (c, r)))
            .collect();
        if !self.orders.is_empty() {
            rows.sort_by(|&a, &b| self.row_cmp(group, a, b));
        }
        let mut out = Vec::new();
        for (i, &(c, r)) in rows.iter().enumerate() {
            if i > 0 {
                out.extend_from_slice(&self.delimiter);
            }
            out.extend_from_slice(value_bytes(&group.values[c], r)?);
        }
        Ok(Some(out))
    }

    /// Spark's ordering of rows `a` and `b` by the ordering keys.
    fn row_cmp(&self, group: &Group, a: (usize, usize), b: (usize, usize)) -> Ordering {
        for (k, &(desc, nulls_first)) in self.orders.iter().enumerate() {
            let (x, xr) = (&group.keys[k][a.0], a.1);
            let (y, yr) = (&group.keys[k][b.0], b.1);
            let ord = key_cmp(x, xr, y, yr, &self.key_types[k], nulls_first);
            let ord = if desc { ord.reverse() } else { ord };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    }
}

/// The non-NULL value bytes of `col` at `row`.
fn value_bytes(col: &ArrayRef, row: usize) -> Result<&[u8]> {
    debug_assert!(col.is_valid(row));
    Ok(match col.data_type() {
        DataType::Utf8View => col.as_string_view().value(row).as_bytes(),
        DataType::Utf8 => col.as_string::<i32>().value(row).as_bytes(),
        DataType::BinaryView => col.as_binary_view().value(row),
        DataType::Binary => col.as_binary::<i32>().value(row),
        t => return Err(internal(format!("a listagg value of type {t}"))),
    })
}

/// Spark's ordering of two ordering-key values: NULLs first or last as
/// written, NaN the largest, `-0.0` equal to `0.0`.
fn key_cmp(
    x: &ArrayRef,
    xr: usize,
    y: &ArrayRef,
    yr: usize,
    t: &DataType,
    nulls_first: bool,
) -> Ordering {
    let (xn, yn) = (!x.is_valid(xr), !y.is_valid(yr));
    match (xn, yn) {
        (true, true) => return Ordering::Equal,
        (true, false) => {
            return if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
        (false, true) => {
            return if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
        (false, false) => {}
    }
    match t {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Date32
        | DataType::Int64
        | DataType::Timestamp(..)
        | DataType::Boolean
        | DataType::Decimal64(..)
        | DataType::Decimal128(..) => int_key(x, xr).cmp(&int_key(y, yr)),
        DataType::Float32 | DataType::Float64 => {
            let (u, v) = (float_key(x, xr), float_key(y, yr));
            match (u.is_nan(), v.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => u.partial_cmp(&v).unwrap_or(Ordering::Equal),
            }
        }
        DataType::Utf8 | DataType::Utf8View | DataType::Binary | DataType::BinaryView => {
            byte_key(x, xr).cmp(byte_key(y, yr))
        }
        // `ListAggAcc::new` refuses every other key type.
        _ => Ordering::Equal,
    }
}

/// Whether `key_cmp` orders keys of type `t`.
fn orderable(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Date32
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Boolean
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::BinaryView
    )
}

/// `a[i]` as an `i128` (integral, boolean, decimal, date, timestamp).
fn int_key(a: &ArrayRef, i: usize) -> i128 {
    match a.data_type() {
        DataType::Int8 => i128::from(a.as_primitive::<Int8Type>().value(i)),
        DataType::Int16 => i128::from(a.as_primitive::<Int16Type>().value(i)),
        DataType::Int32 => i128::from(a.as_primitive::<Int32Type>().value(i)),
        DataType::Date32 => i128::from(a.as_primitive::<Date32Type>().value(i)),
        DataType::Int64 => i128::from(a.as_primitive::<Int64Type>().value(i)),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            i128::from(a.as_primitive::<TimestampMicrosecondType>().value(i))
        }
        DataType::Boolean => i128::from(a.as_boolean().value(i)),
        DataType::Decimal64(..) => i128::from(a.as_primitive::<Decimal64Type>().value(i)),
        DataType::Decimal128(..) => a.as_primitive::<Decimal128Type>().value(i),
        _ => 0,
    }
}

/// `a[i]` as an `f64` (floats only; anything else is zero).
fn float_key(a: &ArrayRef, i: usize) -> f64 {
    if *a.data_type() == DataType::Float32 {
        return f64::from(a.as_primitive::<Float32Type>().value(i));
    }
    if *a.data_type() == DataType::Float64 {
        return a.as_primitive::<Float64Type>().value(i);
    }
    0.0
}

/// `a[i]` as bytes (strings and binaries; anything else is empty).
fn byte_key(a: &ArrayRef, i: usize) -> &[u8] {
    match a.data_type() {
        DataType::Utf8View => a.as_string_view().value(i).as_bytes(),
        DataType::Utf8 => a.as_string::<i32>().value(i).as_bytes(),
        DataType::BinaryView => a.as_binary_view().value(i),
        DataType::Binary => a.as_binary::<i32>().value(i),
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{Field, Fields, TimeUnit};
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn new(key: DataType) -> Result<ListAggAcc> {
        ListAggAcc::new(
            Vec::new(),
            vec![(false, false)],
            DataType::Utf8View,
            vec![key],
        )
    }

    /// Keys without a value ordering are refused rather than left unsorted.
    #[test]
    fn unorderable_keys_are_refused() {
        for key in [
            DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, true)])),
            DataType::List(Arc::new(Field::new("e", DataType::Int32, true))),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            DataType::Duration(TimeUnit::Microsecond),
        ] {
            let e = new(key.clone()).err().map(|e| e.error_class());
            assert_eq!(e, Some(ErrorClass::KrishivUnsupportedFeature), "{key}");
        }
    }

    #[test]
    fn orderable_keys_are_accepted() {
        for key in [
            DataType::Int8,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Float64,
            DataType::Utf8View,
            DataType::Decimal128(10, 2),
        ] {
            assert!(new(key.clone()).is_ok(), "{key}");
        }
    }
}
