// SPDX-License-Identifier: Apache-2.0
//! [`RowLayout`]: where each key column lives in a row, how a batch of key
//! columns becomes rows ([`EncodedKeys`]), and how two rows compare.

use std::hash::BuildHasher;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use krishiv_common::error::KrishivUnsupportedFeature;

use crate::error::{Result, internal};

/// Bytes of a string/binary view in a row.
pub(super) const VIEW_BYTES: usize = 16;
/// Longest string kept inline in its view.
pub(super) const INLINE_MAX: usize = 12;
/// Where the key region starts (after the hash).
const KEYS_START: usize = 8;

/// The value written for, and hashed as, a NULL.
const NULL_HASH: u64 = 0xA5A5_5A5A_C3C3_3C3C;
/// Odd constants for the hash mix (from the fractional digits of pi and e).
const MIX: u64 = 0x243F_6A88_85A3_08D3;
const FINISH: u64 = 0xB7E1_5162_8AED_2A6B;
/// Odd constant of [`finalize`] (the fractional digits of sqrt(2)).
const FIN: u64 = 0x6A09_E667_F3BC_C909;

/// How a column is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Little-endian fixed-width value of this many bytes.
    Fixed(usize),
    /// Normalized `f32` (4 bytes).
    F32,
    /// Normalized `f64` (8 bytes).
    F64,
    /// One byte, 0 or 1.
    Bool,
    /// A 16-byte view of a string or binary value.
    View,
}

impl Kind {
    fn width(self) -> usize {
        match self {
            Self::Fixed(w) => w,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Bool => 1,
            Self::View => VIEW_BYTES,
        }
    }

    fn of(t: &DataType) -> Option<Self> {
        Some(match t {
            DataType::Boolean => Self::Bool,
            DataType::Int8 | DataType::UInt8 => Self::Fixed(1),
            DataType::Int16 | DataType::UInt16 => Self::Fixed(2),
            DataType::Int32 | DataType::UInt32 | DataType::Date32 | DataType::Decimal32(..) => {
                Self::Fixed(4)
            }
            DataType::Int64
            | DataType::UInt64
            | DataType::Date64
            | DataType::Timestamp(..)
            | DataType::Decimal64(..) => Self::Fixed(8),
            DataType::Decimal128(..) => Self::Fixed(16),
            DataType::Float32 => Self::F32,
            DataType::Float64 => Self::F64,
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView => Self::View,
            _ => return None,
        })
    }
}

/// One key column's place in the row.
#[derive(Clone, Debug)]
struct Column {
    kind: Kind,
    /// The key type; fixed-width columns must have exactly this type (a
    /// string or binary column may use any string/binary carrier).
    data_type: DataType,
    /// Byte offset of the value in the row.
    offset: usize,
    /// Bit of the validity bytes that is set when the value is NULL.
    null_bit: Option<usize>,
}

/// The fastest equality test this layout allows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compare {
    /// The key region is one word.
    One,
    /// The key region is two words.
    Two,
    /// Word by word, then the heap bytes of long views.
    Words,
}

/// The row format of one list of key types (P2-T157 §4.2, keys only).
#[derive(Clone, Debug)]
pub struct RowLayout {
    columns: Vec<Column>,
    stride: usize,
    /// Offsets of the words compared as plain words.
    words: Vec<usize>,
    /// Offsets of the views (their second word is compared separately).
    views: Vec<usize>,
    compare: Compare,
    /// A NULL in a key without a validity bit is written as zero bytes
    /// instead of refused: the caller drops such rows (a join key compared
    /// with `=` never matches when NULL).
    lenient: bool,
}

impl RowLayout {
    /// The layout for keys of `types`; `nullable[i]` says whether key `i`
    /// may be NULL (non-nullable keys need no validity bit).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a key type without a row encoding
    /// (nested and dictionary types: they keep the arrow-row path), and
    /// `INTERNAL_ERROR` if `nullable` does not match `types`.
    pub fn new(types: &[DataType], nullable: &[bool]) -> Result<Self> {
        if types.len() != nullable.len() {
            return Err(internal(format!(
                "{} key types but {} nullability flags",
                types.len(),
                nullable.len()
            )));
        }
        let kinds = types
            .iter()
            .map(|t| {
                Kind::of(t).ok_or_else(|| {
                    let feature = format!("row-format key of type {t}");
                    KrishivUnsupportedFeature { feature }.into()
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut next_bit = 0;
        let null_bits: Vec<Option<usize>> = nullable
            .iter()
            .map(|&n| {
                n.then(|| {
                    next_bit += 1;
                    next_bit - 1
                })
            })
            .collect();
        let validity_bytes = next_bit.div_ceil(8);
        // Views first (8-aligned), then fixed values widest first, so every
        // value sits at a multiple of its width.
        let mut at = round8(KEYS_START + validity_bytes);
        let mut offsets = vec![0; kinds.len()];
        let mut order: Vec<usize> = (0..kinds.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(kinds[i].width()));
        for &i in &order {
            offsets[i] = at;
            at += kinds[i].width();
        }
        let stride = round8(at).max(KEYS_START + 8);
        let columns: Vec<Column> = kinds
            .iter()
            .zip(types)
            .zip(offsets)
            .zip(null_bits)
            .map(|(((&kind, data_type), offset), null_bit)| Column {
                kind,
                data_type: data_type.clone(),
                offset,
                null_bit,
            })
            .collect();
        let views: Vec<usize> = columns
            .iter()
            .filter(|c| c.kind == Kind::View)
            .map(|c| c.offset)
            .collect();
        let words: Vec<usize> = (KEYS_START..stride)
            .step_by(8)
            .filter(|w| !views.iter().any(|v| *w == v + 8))
            .collect();
        let compare = match (views.is_empty(), stride - KEYS_START) {
            (true, 8) => Compare::One,
            (true, 16) => Compare::Two,
            _ => Compare::Words,
        };
        Ok(Self {
            columns,
            stride,
            words,
            views,
            compare,
            lenient: false,
        })
    }

    /// The layout of a hash join's keys: only keys compared with `<=>`
    /// (`null_safe`) get a validity bit. A NULL in a key compared with `=`
    /// is written as zero bytes, so the caller must drop every row with
    /// such a NULL before it stores or probes the row (the join does: such
    /// a row can never match).
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn for_join(types: &[DataType], null_safe: &[bool]) -> Result<Self> {
        let mut layout = Self::new(types, null_safe)?;
        layout.lenient = true;
        Ok(layout)
    }

    /// Bytes per row (a multiple of 8).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Offsets of the string/binary views in a row.
    pub(super) fn views(&self) -> &[usize] {
        &self.views
    }

    /// Encodes and hashes the key `columns` of one batch (one per key type,
    /// in order) with `seed`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a column does not have its key type, or a value
    /// is longer than 4 GiB.
    pub fn encode(&self, columns: &[ArrayRef], seed: u64) -> Result<EncodedKeys> {
        if columns.len() != self.columns.len() {
            return Err(internal(format!(
                "{} key columns for a {}-key layout",
                columns.len(),
                self.columns.len()
            )));
        }
        let n = columns.first().map_or(0, Array::len);
        let mut keys = EncodedKeys {
            stride: self.stride,
            len: n,
            rows: vec![0; n * self.stride],
            heap: Vec::new(),
            hashes: vec![fold(seed, MIX); n],
        };
        let strings = foldhash::fast::FixedState::with_seed(seed);
        for (column, array) in self.columns.iter().zip(columns) {
            if array.len() != n {
                return Err(internal("key columns of different lengths".to_owned()));
            }
            let nulls = array.logical_nulls();
            if let Some(bit) = column.null_bit {
                keys.write_nulls(bit, nulls.as_ref());
            } else if !self.lenient && nulls.as_ref().is_some_and(|v| v.null_count() > 0) {
                return Err(internal("NULL in a key declared non-nullable".to_owned()));
            }
            match column.kind {
                Kind::View => keys.write_views(column.offset, array, nulls.as_ref(), &strings)?,
                kind => {
                    // Raw buffer bytes are copied for fixed widths, so the
                    // exact type matters: an `Int64` buffer under an `f64`
                    // key would skip float normalization, and a decimal of
                    // another scale would compare by unscaled value.
                    if array.data_type() != &column.data_type {
                        return Err(internal(format!(
                            "key column of type {} for a {} key",
                            array.data_type(),
                            column.data_type
                        )));
                    }
                    keys.write_fixed(column.offset, kind, array, nulls.as_ref())?;
                }
            }
        }
        for (i, h) in keys.hashes.iter_mut().enumerate() {
            *h = fold(*h, FINISH);
            keys.rows[i * self.stride..i * self.stride + 8].copy_from_slice(&h.to_le_bytes());
        }
        Ok(keys)
    }

    /// Whether the rows have string or binary views (and so may point into
    /// a heap).
    pub fn has_views(&self) -> bool {
        !self.views.is_empty()
    }

    /// Whether row `a` (with heap `a_heap`) has the same key as row `b`
    /// (with heap `b_heap`). Rows must be of this layout.
    #[inline]
    pub fn eq(&self, a: &[u8], a_heap: &[u8], b: &[u8], b_heap: &[u8]) -> bool {
        match self.compare {
            Compare::One => word(a, 8) == word(b, 8),
            Compare::Two => word(a, 8) == word(b, 8) && word(a, 16) == word(b, 16),
            Compare::Words => {
                self.words.iter().all(|&w| word(a, w) == word(b, w))
                    && self.views.iter().all(|&v| {
                        let len = view_len(a, v);
                        if len <= INLINE_MAX {
                            word(a, v + 8) == word(b, v + 8)
                        } else {
                            heap_bytes(a, a_heap, v, len) == heap_bytes(b, b_heap, v, len)
                        }
                    })
            }
        }
    }
}

/// One batch of keys as rows, with their hashes; long strings are in
/// `heap`. Produced by [`RowLayout::encode`].
#[derive(Debug)]
pub struct EncodedKeys {
    stride: usize,
    len: usize,
    rows: Vec<u8>,
    heap: Vec<u8>,
    hashes: Vec<u64>,
}

impl EncodedKeys {
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The hash of each row.
    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    /// Row `i`'s bytes (empty if out of range).
    #[inline]
    pub fn row(&self, i: usize) -> &[u8] {
        self.rows
            .get(i * self.stride..(i + 1) * self.stride)
            .unwrap_or_default()
    }

    /// The heap long strings of these rows point into.
    pub fn heap(&self) -> &[u8] {
        &self.heap
    }

    /// Rows `idx` with self-contained heaps: the row bytes back to back in
    /// `fixed`, and, if the layout has views, each row's long values in
    /// `heap` between `offsets[j]` and `offsets[j + 1]`, with the views'
    /// offsets rewritten to be relative to the row's own heap value. Such
    /// rows survive `take`, concatenation and spilling as plain Arrow
    /// `FixedSizeBinary` and `LargeBinary` columns.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for an index out of range or a view outside the
    /// heap.
    pub fn gather(&self, layout: &RowLayout, idx: &[usize]) -> Result<GatheredRows> {
        let stride = self.stride;
        let mut fixed = Vec::new();
        fixed
            .try_reserve_exact(idx.len().saturating_mul(stride))
            .map_err(|e| internal(e.to_string()))?;
        let mut heap = layout.has_views().then(|| (vec![0i64], Vec::new()));
        for &i in idx {
            let row = self
                .rows
                .get(i * stride..(i + 1) * stride)
                .ok_or_else(|| internal(format!("key row {i} out of range")))?;
            let start = fixed.len();
            fixed.extend_from_slice(row);
            let Some((offsets, bytes)) = heap.as_mut() else {
                continue;
            };
            let base = bytes.len();
            for &v in layout.views() {
                let len = view_len(row, v);
                if len <= INLINE_MAX {
                    continue;
                }
                let from = usize::try_from(word(row, v + 8)).unwrap_or(usize::MAX);
                let value = self
                    .heap
                    .get(from..from.saturating_add(len))
                    .ok_or_else(|| internal("view outside its heap".to_owned()))?;
                let at = (bytes.len() - base) as u64;
                bytes.extend_from_slice(value);
                fixed[start + v + 8..start + v + 16].copy_from_slice(&at.to_le_bytes());
            }
            let end = i64::try_from(bytes.len())
                .map_err(|_| internal("key heap over 2^63 bytes".to_owned()))?;
            offsets.push(end);
        }
        Ok(GatheredRows { fixed, heap })
    }

    fn row_mut(&mut self, i: usize) -> &mut [u8] {
        let s = self.stride;
        &mut self.rows[i * s..(i + 1) * s]
    }

    fn write_nulls(&mut self, bit: usize, nulls: Option<&NullBuffer>) {
        let Some(nulls) = nulls else { return };
        for i in nulls.iter().enumerate().filter(|(_, v)| !v).map(|(i, _)| i) {
            self.row_mut(i)[KEYS_START + bit / 8] |= 1 << (bit % 8);
            self.hashes[i] = mix(self.hashes[i], NULL_HASH);
        }
    }

    fn write_fixed(
        &mut self,
        offset: usize,
        kind: Kind,
        array: &ArrayRef,
        nulls: Option<&NullBuffer>,
    ) -> Result<()> {
        let valid = |i: usize| nulls.is_none_or(|v| v.is_valid(i));
        match kind {
            Kind::Bool => {
                let a = array
                    .as_boolean_opt()
                    .ok_or_else(|| internal("boolean key expected".to_owned()))?;
                for i in (0..self.len).filter(|&i| valid(i)) {
                    let v = u8::from(a.value(i));
                    self.row_mut(i)[offset] = v;
                    self.hashes[i] = mix(self.hashes[i], u64::from(v));
                }
            }
            Kind::F32 => {
                let a = array
                    .as_primitive_opt::<Float32Type>()
                    .ok_or_else(|| internal("float key expected".to_owned()))?;
                for i in (0..self.len).filter(|&i| valid(i)) {
                    let bits = normalize32(a.value(i)).to_bits();
                    self.row_mut(i)[offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
                    self.hashes[i] = mix(self.hashes[i], u64::from(bits));
                }
            }
            Kind::F64 => {
                let a = array
                    .as_primitive_opt::<Float64Type>()
                    .ok_or_else(|| internal("double key expected".to_owned()))?;
                for i in (0..self.len).filter(|&i| valid(i)) {
                    let bits = normalize64(a.value(i)).to_bits();
                    self.row_mut(i)[offset..offset + 8].copy_from_slice(&bits.to_le_bytes());
                    self.hashes[i] = mix(self.hashes[i], bits);
                }
            }
            Kind::Fixed(w) => {
                let data = array.to_data();
                let bytes = data
                    .buffers()
                    .first()
                    .and_then(|b| {
                        b.as_slice()
                            .get(data.offset() * w..(data.offset() + self.len) * w)
                    })
                    .ok_or_else(|| internal("fixed-width key buffer too short".to_owned()))?;
                match w {
                    1 => self.put_fixed::<1>(offset, bytes, nulls),
                    2 => self.put_fixed::<2>(offset, bytes, nulls),
                    4 => self.put_fixed::<4>(offset, bytes, nulls),
                    8 => self.put_fixed::<8>(offset, bytes, nulls),
                    16 => self.put_fixed::<16>(offset, bytes, nulls),
                    _ => return Err(internal(format!("no {w}-byte key encoding"))),
                }
            }
            Kind::View => return Err(internal("view written as fixed".to_owned())),
        }
        Ok(())
    }

    /// Copies `W`-byte values into the rows and mixes them into the hashes
    /// (one loop per width, no per-row bounds arithmetic).
    fn put_fixed<const W: usize>(
        &mut self,
        offset: usize,
        bytes: &[u8],
        nulls: Option<&NullBuffer>,
    ) {
        let rows = self.rows.chunks_exact_mut(self.stride);
        let (values, _) = bytes.as_chunks::<W>();
        for (i, ((row, h), v)) in rows.zip(self.hashes.iter_mut()).zip(values).enumerate() {
            if nulls.is_some_and(|n| n.is_null(i)) {
                continue;
            }
            if let Some(dst) = row.get_mut(offset..offset + W) {
                dst.copy_from_slice(v);
            }
            let mut acc = *h;
            for chunk in v.chunks(8) {
                let mut word = [0u8; 8];
                word[..chunk.len()].copy_from_slice(chunk);
                acc = mix(acc, u64::from_le_bytes(word));
            }
            *h = acc;
        }
    }

    fn write_views(
        &mut self,
        offset: usize,
        array: &ArrayRef,
        nulls: Option<&NullBuffer>,
        strings: &foldhash::fast::FixedState,
    ) -> Result<()> {
        match array.data_type() {
            DataType::Utf8 => {
                let a = array.as_string::<i32>();
                self.put_views(offset, nulls, strings, |i| a.value(i).as_bytes())
            }
            DataType::LargeUtf8 => {
                let a = array.as_string::<i64>();
                self.put_views(offset, nulls, strings, |i| a.value(i).as_bytes())
            }
            DataType::Utf8View => {
                let a = array.as_string_view();
                self.put_views(offset, nulls, strings, |i| a.value(i).as_bytes())
            }
            DataType::Binary => {
                let a = array.as_binary::<i32>();
                self.put_views(offset, nulls, strings, |i| a.value(i))
            }
            DataType::LargeBinary => {
                let a = array.as_binary::<i64>();
                self.put_views(offset, nulls, strings, |i| a.value(i))
            }
            DataType::BinaryView => {
                let a = array.as_binary_view();
                self.put_views(offset, nulls, strings, |i| a.value(i))
            }
            t => Err(internal(format!("string key expected, got {t}"))),
        }
    }

    fn put_views<'a>(
        &mut self,
        offset: usize,
        nulls: Option<&NullBuffer>,
        strings: &foldhash::fast::FixedState,
        value: impl Fn(usize) -> &'a [u8],
    ) -> Result<()> {
        for i in (0..self.len).filter(|&i| nulls.is_none_or(|v| v.is_valid(i))) {
            let bytes = value(i);
            let len = u32::try_from(bytes.len())
                .map_err(|_| internal("a key value longer than 4 GiB".to_owned()))?;
            self.hashes[i] = mix(self.hashes[i], strings.hash_one(bytes));
            let at = self.heap.len();
            let stride = self.stride;
            let row = &mut self.rows[i * stride..(i + 1) * stride];
            let view = &mut row[offset..offset + VIEW_BYTES];
            view[..4].copy_from_slice(&len.to_le_bytes());
            if bytes.len() <= INLINE_MAX {
                view[4..4 + bytes.len()].copy_from_slice(bytes);
            } else {
                view[4..8].copy_from_slice(&bytes[..4]);
                view[8..].copy_from_slice(&(at as u64).to_le_bytes());
                self.heap.extend_from_slice(bytes);
            }
        }
        Ok(())
    }
}

/// Rows taken out of [`EncodedKeys`] by [`EncodedKeys::gather`].
#[derive(Debug)]
pub struct GatheredRows {
    /// The row bytes, back to back.
    pub fixed: Vec<u8>,
    /// Per-row heap values: offsets (one more than rows) and bytes; `None`
    /// when the layout has no views.
    pub heap: Option<(Vec<i64>, Vec<u8>)>,
}

/// The table-local hash `g = fin(h ^ seed)` of a stored row hash `h`
/// (P2-T157 Appendix C): one folded multiply. Salts, directory and bucket
/// indexes and grace sub-partitions come from `g`. Keys with equal `h`
/// have equal `g` under every seed, so `h` itself must not be collidable
/// by whoever chooses the keys (the join seeds it); `g` never leaves the
/// operator instance that drew `seed`.
#[inline]
pub fn finalize(h: u64, seed: u64) -> u64 {
    fold(h ^ seed, FIN)
}

/// The little-endian word at `at` of `row` (0 if out of range).
#[inline]
pub(super) fn word(row: &[u8], at: usize) -> u64 {
    row.get(at..at + 8)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map_or(0, u64::from_le_bytes)
}

/// The length stored in the view at `at` of `row`.
#[inline]
pub(super) fn view_len(row: &[u8], at: usize) -> usize {
    u32::try_from(word(row, at) & u64::from(u32::MAX)).map_or(0, |l| l as usize)
}

/// The heap bytes of the long view at `at` of `row`.
#[inline]
fn heap_bytes<'a>(row: &[u8], heap: &'a [u8], at: usize, len: usize) -> &'a [u8] {
    let start = usize::try_from(word(row, at + 8)).unwrap_or(usize::MAX);
    heap.get(start..start.saturating_add(len))
        .unwrap_or_default()
}

fn round8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// The folded multiply (foldhash's mixing step).
#[inline]
fn fold(a: u64, b: u64) -> u64 {
    let p = u128::from(a) * u128::from(b);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the low half of the product is intended"
    )]
    let low = p as u64;
    low ^ ((p >> 64) as u64)
}

#[inline]
fn mix(h: u64, v: u64) -> u64 {
    fold(h ^ v, MIX)
}

/// Spark's key normalization: every NaN is one NaN, `-0.0` is `0.0`.
fn normalize64(v: f64) -> f64 {
    if v.is_nan() { f64::NAN } else { v + 0.0 }
}

fn normalize32(v: f32) -> f32 {
    if v.is_nan() { f32::NAN } else { v + 0.0 }
}

#[cfg(test)]
mod flooding {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::DataType;

    use super::*;

    /// Two-word keys built to collide under a known starting state (the
    /// mix's second input cancels the first word) share `h` with that seed
    /// and spread under another one.
    #[test]
    fn colliding_keys_need_the_seed() {
        let layout = RowLayout::for_join(&[DataType::Int64, DataType::Int64], &[false, false])
            .unwrap_or_else(|e| panic!("{e}"));
        let known = 0u64;
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for v1 in 0..1000u64 {
            let s1 = mix(fold(known, MIX), v1);
            a.push(v1.cast_signed());
            b.push((s1 ^ 0x1234).cast_signed());
        }
        let cols: Vec<ArrayRef> =
            vec![Arc::new(Int64Array::from(a)), Arc::new(Int64Array::from(b))];
        let distinct = |seed: u64| {
            let k = layout.encode(&cols, seed).unwrap_or_else(|e| panic!("{e}"));
            let mut h = k.hashes().to_vec();
            h.sort_unstable();
            h.dedup();
            h.len()
        };
        assert_eq!(distinct(known), 1, "the construction collides");
        assert_eq!(distinct(0x5EED_1234_ABCD), 1000);
    }
}
