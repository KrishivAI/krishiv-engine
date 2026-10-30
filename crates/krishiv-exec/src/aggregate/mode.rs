// SPDX-License-Identifier: Apache-2.0
//! The `mode` accumulator: the most frequent non-NULL value of a group,
//! with Spark 4.2's tie-breaking.
//!
//! A group's state is its distinct values in first-seen order, with their
//! counts and first-seen samples (the input chunks shared, as the collect
//! accumulator shares its). Merging adds the counts, so partial states
//! merge associatively and a partial phase, an exchange and a final phase
//! compose. Partitions merge in partition-index order where the merge
//! knows them (see [`is_order_sensitive`](super::is_order_sensitive)): the
//! in-memory driver tables and the runs spilled from them. The row-table
//! and repartitioned runs carry no partition label and merge after the
//! labelled ones, so a tie-break that depends on first-seen order can
//! still vary with spill timing on those paths.
//!
//! # Tie-breaking
//! Where several values share the greatest count, Spark's `Mode` keeps
//! the first in its hash map's slot order (`maxBy(_._2)` over an
//! `OpenHashMap`): the slot is `murmur3_32_fixed` (Guava, seed 0) of the
//! key's `hashCode`, in a table of capacity 64 grown past 70% full, with
//! linear probing. The key `hashCode` is the value's for integers (a
//! `Long` folds its halves), Spark's `Murmur3_x86_32` (seed 42) of the
//! bytes for strings, `Arrays.hashCode` for binaries, and the normalized
//! bits for floating point (`-0.0` reads as `0.0`, every NaN as the
//! canonical one, as `NormalizeFloatingNumbers` normalizes them).
//!
//! The accumulator counts by the shared grouping-key encoding (so `NaN`
//! equals `NaN` and `-0.0` equals `0.0`, as for grouping keys) and, for an
//! arbitrary tie-break, replays Spark's table over the tied keys in
//! first-seen order at evaluation: for key sets without collisions the
//! slot order (and the pick) does not depend on insertion order at all,
//! and the golden models never rehash. Only the tied keys are replayed,
//! so a tie in an input with more than 44 distinct values (where Spark's
//! table has grown) may pick another key than Spark; decimal keys hash
//! by their unscaled value, not by `Decimal.hashCode`. No golden covers
//! either case yet. The ordered forms (`ModeMin`,
//! `ModeMax`) pick the smallest or largest among the most frequent by
//! Spark's value ordering instead, and never touch the hash order.
//!
//! # Memory
//! First-seen samples share their input buffers; [`kept_bytes`] (as the
//! collect accumulator's reservation) bounds what an append keeps, and the
//! accumulator's reservation (made by the table from
//! [`ModeAcc::projected`] before folding) covers it.

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
use super::{AggregateFunction, unsupported};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, KeyEncoder, KeyTable};

/// How ties break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tiebreak {
    /// `mode(col)`: Spark's hash-map order.
    Arbitrary,
    /// `mode(col, true)` and `mode() WITHIN GROUP (ORDER BY col)` (ASC):
    /// the smallest among the most frequent.
    Smallest,
    /// `mode() WITHIN GROUP (ORDER BY col DESC)`: the largest.
    Largest,
}

impl Tiebreak {
    fn of(function: &AggregateFunction) -> Result<Self> {
        match function {
            AggregateFunction::Mode => Ok(Self::Arbitrary),
            AggregateFunction::ModeMin => Ok(Self::Smallest),
            AggregateFunction::ModeMax => Ok(Self::Largest),
            f => Err(unsupported(format!("{f:?} as a mode"))),
        }
    }
}

/// Java's `Long.hashCode`: the low 32 bits of `v ^ (v >>> 32)`.
fn long_hash(v: i64) -> i32 {
    u32::try_from((v ^ (v >> 32)) & 0xffff_ffff)
        .unwrap_or_default()
        .cast_signed()
}

/// Java's `Double.hashCode` over the normalized bits: the low 32 bits of
/// `bits ^ (bits >>> 32)`.
fn double_hash(bits: u64) -> i32 {
    u32::try_from((bits ^ (bits >> 32)) & 0xffff_ffff)
        .unwrap_or_default()
        .cast_signed()
}

/// Guava's `murmur3_32_fixed` hash of one `int` (seed 0): what Spark's
/// `OpenHashSet` hashes every key's `hashCode` with.
fn murmur_fixed(h: i32) -> i32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;
    let mut k = h.cast_unsigned();
    k = k.wrapping_mul(C1);
    k = k.rotate_left(15);
    k = k.wrapping_mul(C2);
    let mut h1 = k;
    h1 = h1.rotate_left(13);
    h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    // `fmix(h1, 4)`.
    h1 ^= 4;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85ebca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2ae35);
    h1 ^= h1 >> 16;
    h1.cast_signed()
}

/// Spark's `Murmur3_x86_32.hashUnsafeBytes` (seed 42): `UTF8String`'s
/// `hashCode`.
fn murmur_bytes(data: &[u8]) -> i32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;
    let mut h1: u32 = 42;
    let (words, tail) = data.as_chunks::<4>();
    for w in words {
        let mut k = u32::from_le_bytes(*w);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(C2);
        h1 ^= k;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe6546b64);
    }
    if !tail.is_empty() {
        let mut k: u32 = 0;
        for (i, &b) in tail.iter().enumerate() {
            // `Platform.getByte` is signed and widened to int.
            k ^= i32::from(b.cast_signed()).cast_unsigned() << (8 * i);
        }
        k = k.wrapping_mul(C1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(C2);
        h1 ^= k;
    }
    h1 ^= u32::try_from(data.len()).unwrap_or(u32::MAX);
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85ebca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2ae35);
    h1 ^= h1 >> 16;
    h1.cast_signed()
}

/// The winning key index of `hashes` with `counts` in Spark's slot order:
/// the first maximum count in ascending slot order, growing the table
/// past 70% full exactly as Spark's `OpenHashSet` grows it (a rehash
/// reinserts in slot order).
fn spark_pick(hashes: &[i32], counts: &[u64]) -> Option<usize> {
    if hashes.is_empty() {
        return None;
    }
    fn slot(slots: &[Option<usize>], hash: i32) -> usize {
        let mut pos = (murmur_fixed(hash).cast_unsigned() as usize) & (slots.len() - 1);
        while slots[pos].is_some() {
            pos = (pos + 1) & (slots.len() - 1);
        }
        pos
    }
    let mut slots: Vec<Option<usize>> = vec![None; 64];
    for key in 0..hashes.len() {
        if (key + 1) as f64 > slots.len() as f64 * 0.7 {
            let cap = slots.len() * 2;
            let old = std::mem::replace(&mut slots, vec![None; cap]);
            for s in old.into_iter().flatten() {
                let pos = slot(&slots, hashes[s]);
                slots[pos] = Some(s);
            }
        }
        let pos = slot(&slots, hashes[key]);
        slots[pos] = Some(key);
    }
    let mut best: Option<usize> = None;
    for s in slots.into_iter().flatten() {
        if best.is_none_or(|b| counts[s] > counts[b]) {
            best = Some(s);
        }
    }
    best
}

/// One group's distinct values in first-seen order.
#[derive(Debug, Clone, Default)]
struct Group {
    /// The distinct keys (first-seen order).
    seen: KeyTable,
    /// One count per key.
    counts: Vec<u64>,
    /// First-seen sample chunks (shared input slices).
    chunks: Vec<ArrayRef>,
    /// `(chunk, row)` of each key's first sample.
    loc: Vec<(usize, usize)>,
}

/// The per-group state of one `mode` aggregate.
#[derive(Debug)]
pub(super) struct ModeAcc {
    tiebreak: Tiebreak,
    arg_type: DataType,
    encoder: KeyEncoder,
    seed: HashSeed,
    groups: Vec<Group>,
    /// Heap bytes: key tables plus sample chunks.
    heap: usize,
}

impl ModeAcc {
    /// An accumulator of value type `arg` for `function` (`mode` and its
    /// ordered forms).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a value type the shared key
    /// encoding refuses.
    pub(super) fn new(arg: DataType, function: &AggregateFunction) -> Result<Self> {
        Ok(Self {
            tiebreak: Tiebreak::of(function)?,
            encoder: KeyEncoder::new(std::slice::from_ref(&arg))?,
            arg_type: arg,
            seed: HashSeed::new(Some(0)),
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
    /// groups and an update with `arg` (twice its kept bytes: the encode
    /// buffer plus the kept samples).
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
        self.groups.resize_with(n, Group::default);
    }

    /// Reserves room for `new` more groups.
    pub(super) fn reserve(&mut self, new: usize) {
        self.groups.reserve(new);
    }

    /// Folds the valid rows of `col` into `groups`.
    pub(super) fn fold(
        &mut self,
        groups: &[u32],
        col: &ArrayRef,
        valid: &dyn Fn(usize) -> bool,
    ) -> Result<()> {
        if col.len() != groups.len() {
            return Err(internal(
                "a mode column of another length than the group ids".to_owned(),
            ));
        }
        if matches!(col.data_type(), DataType::Null) {
            return Ok(());
        }
        // One take per touched group, in row order (as the collect
        // accumulator folds them).
        let mut order: Vec<Vec<u32>> = vec![Vec::new(); self.groups.len()];
        for (i, &g) in groups.iter().enumerate() {
            if valid(i) {
                order[g as usize].push(
                    u32::try_from(i)
                        .map_err(|_| internal("a mode batch past 2^32 rows".to_owned()))?,
                );
            }
        }
        for (g, idx) in order.iter().enumerate() {
            if idx.is_empty() {
                continue;
            }
            let chunk = take(col.as_ref(), &UInt32Array::from(idx.clone()), None)
                .map_err(|e| internal(e.to_string()))?;
            let counts = vec![1_u64; chunk.len()];
            self.fold_chunk(g, &chunk, &counts)?;
        }
        Ok(())
    }

    /// Folds `chunk` (no NULLs) with per-row `counts` into group `g`.
    fn fold_chunk(&mut self, g: usize, chunk: &ArrayRef, counts: &[u64]) -> Result<()> {
        if chunk.len() != counts.len() {
            return Err(internal(
                "a mode chunk of another length than its counts".to_owned(),
            ));
        }
        let chunk = normalize(Arc::clone(chunk), &self.arg_type)?;
        let rows = self.encoder.encode(std::slice::from_ref(&chunk))?;
        let group = &mut self.groups[g];
        let before = group.seen.memory_size();
        let mut ids = Vec::with_capacity(chunk.len());
        group.seen.find_or_insert(&rows, &self.seed, &mut ids)?;
        // New ids are sequential from the previous length, in row order.
        let mut next = group.counts.len();
        let chunk_at = group.chunks.len();
        let mut kept = false;
        for (j, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id == next {
                next += 1;
                if !kept {
                    group.chunks.push(Arc::clone(&chunk));
                    kept = true;
                }
                group.loc.push((chunk_at, j));
                group.counts.push(0);
            }
            group.counts[id] = group.counts[id].saturating_add(counts[j]);
        }
        self.heap = self
            .heap
            .saturating_add(group.seen.memory_size().saturating_sub(before));
        if kept {
            self.heap = self.heap.saturating_add(kept_bytes(&chunk));
        }
        Ok(())
    }

    /// Merges partial states (keys lists with counts lists) into `groups`.
    pub(super) fn merge(&mut self, groups: &[u32], states: &[&ArrayRef]) -> Result<()> {
        let [keys, counts] = states else {
            return Err(internal(
                "a mode state of another shape than its keys and counts".to_owned(),
            ));
        };
        let Some(keys) = keys.as_any().downcast_ref::<ListArray>() else {
            return Err(internal(format!(
                "a mode keys state of another type than a list: {}",
                keys.data_type()
            )));
        };
        let Some(counts) = counts.as_any().downcast_ref::<ListArray>() else {
            return Err(internal(format!(
                "a mode counts state of another type than a list: {}",
                counts.data_type()
            )));
        };
        if keys.len() != groups.len() || counts.len() != groups.len() {
            return Err(internal(
                "a mode state of another length than the group ids".to_owned(),
            ));
        }
        let key_values = keys.values();
        let key_offsets = keys.value_offsets();
        let count_values = counts
            .values()
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .ok_or_else(|| {
                internal(format!(
                    "a mode counts state of another type than bigint: {}",
                    counts.values().data_type()
                ))
            })?;
        let count_offsets = counts.value_offsets();
        for (i, &g) in groups.iter().enumerate() {
            if keys.is_null(i) || counts.is_null(i) {
                continue;
            }
            let klen = usize::try_from(keys.value_length(i).max(0)).unwrap_or(0);
            if klen == 0 {
                continue;
            }
            if usize::try_from(counts.value_length(i).max(0)).unwrap_or(0) != klen {
                return Err(internal(
                    "a mode state whose keys and counts differ".to_owned(),
                ));
            }
            let kstart = usize::try_from(key_offsets[i].max(0)).unwrap_or(0);
            let cstart = usize::try_from(count_offsets[i].max(0)).unwrap_or(0);
            let chunk = key_values.slice(kstart, klen);
            let mut add = Vec::with_capacity(klen);
            for k in 0..klen {
                add.push(u64::try_from(count_values.value(cstart + k).max(0)).unwrap_or(0));
            }
            self.fold_chunk(g as usize, &chunk, &add)?;
        }
        Ok(())
    }

    /// Copies group `from`'s state to snapshot group `to` (a running
    /// frame's snapshot; shares the sample chunks).
    pub(super) fn snapshot(&mut self, from: usize, to: usize) {
        self.groups[to] = self.groups[from].clone();
    }

    /// The partial states of groups `ids`: the distinct keys with their
    /// counts (never NULL; empty groups hold empty lists).
    pub(super) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let mut children: Vec<ArrayRef> = Vec::new();
        let mut counts: Vec<ArrayRef> = Vec::new();
        let mut key_total = 0_i32;
        let mut key_offsets = Vec::with_capacity(ids.len() + 1);
        let mut count_offsets = Vec::with_capacity(ids.len() + 1);
        key_offsets.push(0);
        count_offsets.push(0);
        for &g in ids {
            let group = &self.groups[g as usize];
            let mut taking: Vec<ArrayRef> = Vec::with_capacity(group.loc.len());
            for &(c, r) in &group.loc {
                taking.push(group.chunks[c].slice(r, 1));
            }
            if !taking.is_empty() {
                let refs: Vec<&dyn Array> = taking.iter().map(AsRef::as_ref).collect();
                children.push(concat(&refs).map_err(|e| internal(e.to_string()))?);
                key_total = key_total
                    .checked_add(
                        i32::try_from(taking.len())
                            .map_err(|_| internal("a mode group past 2^31 values".to_owned()))?,
                    )
                    .ok_or_else(|| internal("a mode group past 2^31 values".to_owned()))?;
            }
            key_offsets.push(key_total);
            let array: ArrayRef = Arc::new(
                group
                    .counts
                    .iter()
                    .map(|&c| i64::try_from(c).unwrap_or(i64::MAX))
                    .collect::<arrow_array::Int64Array>(),
            );
            if !group.counts.is_empty() {
                counts.push(array);
            }
            count_offsets.push(
                count_offsets.last().copied().unwrap_or(0)
                    + i32::try_from(group.counts.len()).unwrap_or(i32::MAX),
            );
        }
        let key_values = if children.is_empty() {
            arrow_array::new_empty_array(&self.arg_type)
        } else {
            let refs: Vec<&dyn Array> = children.iter().map(AsRef::as_ref).collect();
            concat(&refs).map_err(|e| internal(e.to_string()))?
        };
        let count_values = if counts.is_empty() {
            Arc::new(arrow_array::Int64Array::new_null(0)) as ArrayRef
        } else {
            let refs: Vec<&dyn Array> = counts.iter().map(AsRef::as_ref).collect();
            concat(&refs).map_err(|e| internal(e.to_string()))?
        };
        let keys_field = Arc::new(Field::new("key", self.arg_type.clone(), true));
        let counts_field = Arc::new(Field::new("count", DataType::Int64, false));
        let keys_list = ListArray::try_new(
            keys_field,
            OffsetBuffer::new(ScalarBuffer::from(key_offsets)),
            key_values,
            None,
        )
        .map_err(|e| internal(e.to_string()))
        .map(|a| Arc::new(a) as ArrayRef)?;
        let counts_list = ListArray::try_new(
            counts_field,
            OffsetBuffer::new(ScalarBuffer::from(count_offsets)),
            count_values,
            None,
        )
        .map_err(|e| internal(e.to_string()))
        .map(|a| Arc::new(a) as ArrayRef)?;
        Ok(vec![keys_list, counts_list])
    }

    /// The final values of groups `ids` (NULL for no value).
    pub(super) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let mut out: Vec<ArrayRef> = Vec::with_capacity(ids.len());
        for &g in ids {
            let group = &self.groups[g as usize];
            out.push(match self.pick(group)? {
                Some((c, r)) => group.chunks[c].slice(r, 1),
                None => arrow_array::new_null_array(&self.arg_type, 1),
            });
        }
        let refs: Vec<&dyn Array> = out.iter().map(AsRef::as_ref).collect();
        concat(&refs).map_err(|e| internal(e.to_string()))
    }

    /// The winning sample `(chunk, row)` of `group`, if it has a value.
    fn pick(&self, group: &Group) -> Result<Option<(usize, usize)>> {
        if group.counts.is_empty() {
            return Ok(None);
        }
        let top = group.counts.iter().max().copied().unwrap_or(0);
        let tied: Vec<usize> = group
            .counts
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == top)
            .map(|(i, _)| i)
            .collect();
        let win = match self.tiebreak {
            Tiebreak::Arbitrary => {
                let mut hashes = Vec::with_capacity(tied.len());
                for &i in &tied {
                    let (c, r) = group.loc[i];
                    hashes.push(spark_hash(&group.chunks[c], r, &self.arg_type)?);
                }
                let counts: Vec<u64> = tied.iter().map(|&i| group.counts[i]).collect();
                spark_pick(&hashes, &counts).map(|k| tied[k])
            }
            Tiebreak::Smallest | Tiebreak::Largest => {
                let mut win = tied[0];
                for &i in tied.iter().skip(1) {
                    let ord = self.cmp(group, win, i)?;
                    let better = if self.tiebreak == Tiebreak::Smallest {
                        ord.is_gt()
                    } else {
                        ord.is_lt()
                    };
                    if better {
                        win = i;
                    }
                }
                Some(win)
            }
        };
        Ok(win.map(|i| group.loc[i]))
    }

    /// Spark's value ordering of keys `a` and `b` (by first-seen index).
    fn cmp(&self, group: &Group, a: usize, b: usize) -> Result<std::cmp::Ordering> {
        use std::cmp::Ordering;
        let (ac, ar) = group.loc[a];
        let (bc, br) = group.loc[b];
        let (x, y) = (&group.chunks[ac], &group.chunks[bc]);
        Ok(match &self.arg_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Date32
            | DataType::Int64
            | DataType::Timestamp(TimeUnit::Microsecond, _)
            | DataType::Boolean
            | DataType::Decimal64(..)
            | DataType::Decimal128(..) => int_key(x, ar)?.cmp(&int_key(y, br)?),
            DataType::Float32 | DataType::Float64 => {
                let (u, v) = (float_key(x, ar)?, float_key(y, br)?);
                match (u.is_nan(), v.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => u.partial_cmp(&v).unwrap_or(Ordering::Equal),
                }
            }
            DataType::Utf8 | DataType::Utf8View | DataType::Binary | DataType::BinaryView => {
                byte_key(x, ar)?.cmp(byte_key(y, br)?)
            }
            // Nested types have no value ordering here; their encoded
            // keys compare deterministically (no golden pins it down).
            _ => group.seen.row(a).cmp(group.seen.row(b)),
        })
    }
}

/// Spark's `hashCode` of one value for an arbitrary mode tie-break (see
/// the module docs); nested types hash to zero (their ties keep
/// first-seen order, which no golden pins down).
fn spark_hash(col: &ArrayRef, row: usize, t: &DataType) -> Result<i32> {
    debug_assert!(col.is_valid(row));
    Ok(match t {
        DataType::Int8 => i32::from(col.as_primitive::<Int8Type>().value(row)),
        DataType::Int16 => i32::from(col.as_primitive::<Int16Type>().value(row)),
        DataType::Int32 => col.as_primitive::<Int32Type>().value(row),
        DataType::Date32 => col.as_primitive::<Date32Type>().value(row),
        DataType::Int64 => long_hash(col.as_primitive::<Int64Type>().value(row)),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            long_hash(col.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        DataType::Boolean => {
            if col.as_boolean().value(row) {
                1231
            } else {
                1237
            }
        }
        DataType::Float32 => {
            let mut v = col.as_primitive::<Float32Type>().value(row);
            if v == 0.0 {
                v = 0.0;
            } else if v.is_nan() {
                v = f32::from_bits(0x7FC0_0000);
            }
            v.to_bits().cast_signed()
        }
        DataType::Float64 => {
            let mut v = col.as_primitive::<Float64Type>().value(row);
            if v == 0.0 {
                v = 0.0;
            } else if v.is_nan() {
                v = f64::from_bits(0x7FF8_0000_0000_0000);
            }
            double_hash(v.to_bits())
        }
        DataType::Decimal64(..) => long_hash(col.as_primitive::<Decimal64Type>().value(row)),
        DataType::Decimal128(..) => {
            let v = col.as_primitive::<Decimal128Type>().value(row);
            let lo = i64::try_from(v & 0xFFFF_FFFF).unwrap_or_default();
            let hi = i64::try_from((v >> 64) & 0xFFFF_FFFF).unwrap_or_default();
            long_hash(lo ^ hi)
        }
        DataType::Utf8 => murmur_bytes(col.as_string::<i32>().value(row).as_bytes()),
        DataType::Utf8View => murmur_bytes(col.as_string_view().value(row).as_bytes()),
        DataType::Binary => {
            let b = col.as_binary::<i32>().value(row);
            b.iter().fold(1_i32, |h, &x| {
                31_i32.wrapping_mul(h).wrapping_add(i32::from(x))
            })
        }
        DataType::BinaryView => {
            let b = col.as_binary_view().value(row);
            b.iter().fold(1_i32, |h, &x| {
                31_i32.wrapping_mul(h).wrapping_add(i32::from(x))
            })
        }
        _ => 0,
    })
}

/// `a[i]` as an `i128` (integral, boolean, decimal, date, timestamp).
fn int_key(a: &ArrayRef, i: usize) -> Result<i128> {
    let t = a.data_type();
    Ok(match t {
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
        _ => return Err(internal(format!("a mode key of type {t} as an integer"))),
    })
}

/// `a[i]` as an `f64`.
fn float_key(a: &ArrayRef, i: usize) -> Result<f64> {
    if *a.data_type() == DataType::Float32 {
        return Ok(f64::from(a.as_primitive::<Float32Type>().value(i)));
    }
    if *a.data_type() == DataType::Float64 {
        return Ok(a.as_primitive::<Float64Type>().value(i));
    }
    Err(internal(format!(
        "a mode key of type {} as a float",
        a.data_type()
    )))
}

/// `a[i]` as bytes (strings and binaries).
fn byte_key(a: &ArrayRef, i: usize) -> Result<&[u8]> {
    Ok(match a.data_type() {
        DataType::Utf8View => a.as_string_view().value(i).as_bytes(),
        DataType::Utf8 => a.as_string::<i32>().value(i).as_bytes(),
        DataType::BinaryView => a.as_binary_view().value(i),
        DataType::Binary => a.as_binary::<i32>().value(i),
        t => {
            return Err(internal(format!("a mode key of type {t} as bytes")));
        }
    })
}
