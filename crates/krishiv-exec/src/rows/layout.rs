// SPDX-License-Identifier: Apache-2.0
//! [`RowLayout`]: where each key column lives in a row, how a batch of key
//! columns becomes rows ([`EncodedKeys`]), and how two rows compare.
//!
//! # The stored row hash `h`
//! A key row starts with its hash `h`, a fast column-wise mix (one folded
//! multiply per 8-byte word, foldhash for values longer than 12 bytes)
//! computed while the values are written. It has no seed, so every
//! operator instance and worker computes the same `h` for a key, and
//! sketches over it (the aggregate's HLLs) merge across tables. Tables
//! place rows by the seeded `g = fin(h ^ seed)` ([`finalize`]).
//!
//! `h` is a table-internal hash and never routes a row between processes:
//! exchanges, key groups and anything written to a plan or to state use the
//! routing hash `H_kg` (`krishiv_kernels::hash::routing`, P2-T175), which is
//! versioned and costs more per row (XXH64 per column). Keeping the two
//! apart is the P2-T171 §5.1 fallback: `H_kg` as the tables' hash slowed
//! short, hash-bound aggregates measurably, and a hash that never leaves
//! the process need not be the routing function.

use std::hash::BuildHasher;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, make_array};
use arrow_buffer::{BooleanBuffer, MutableBuffer, NullBuffer};
use arrow_data::ArrayData;
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
/// The seed of the stored hash `h`: none (see the module docs).
const SEEDLESS: u64 = 0;
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
    /// Where the validity bytes start.
    validity_at: usize,
    /// The row starts with its hash (a key row); a payload row does not.
    hashed: bool,
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
        Self::with_shape(types, nullable, true)
    }

    /// [`new`](Self::new) for key rows (`hashed`: hash word, validity, values)
    /// or payload rows (values, then validity).
    fn with_shape(types: &[DataType], nullable: &[bool], hashed: bool) -> Result<Self> {
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
        let mut at = if hashed {
            round8(KEYS_START + validity_bytes)
        } else {
            0
        };
        let mut offsets = vec![0; kinds.len()];
        let mut order: Vec<usize> = (0..kinds.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(kinds[i].width()));
        for &i in &order {
            offsets[i] = at;
            at += kinds[i].width();
        }
        let validity_at = if hashed { KEYS_START } else { at };
        let stride = if hashed {
            round8(at).max(KEYS_START + 8)
        } else {
            round8(at + validity_bytes).max(8)
        };
        let start = if hashed { KEYS_START } else { 0 };
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
        let words: Vec<usize> = (start..stride)
            .step_by(8)
            .filter(|w| !views.iter().any(|v| *w == v + 8))
            .collect();
        // `One`/`Two` read the words after the hash: key rows only.
        let compare = match (hashed, views.is_empty(), stride - start) {
            (true, true, 8) => Compare::One,
            (true, true, 16) => Compare::Two,
            _ => Compare::Words,
        };
        Ok(Self {
            columns,
            stride,
            words,
            views,
            compare,
            lenient: false,
            validity_at,
            hashed,
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

    /// The layout of a hash join's build columns carried as a payload: every
    /// column nullable, floats stored bit for bit (a payload is returned as
    /// it came, so `-0.0` and NaN payloads are not normalized), strings and
    /// binaries as views. [`decode_rows`](Self::decode_rows) restores the
    /// columns. Payload rows are not hashed ([`EncodedKeys::hashes`] is empty).
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn for_payload(types: &[DataType]) -> Result<Self> {
        let mut layout = Self::with_shape(types, &vec![true; types.len()], false)?;
        for c in &mut layout.columns {
            c.kind = match c.kind {
                Kind::F32 => Kind::Fixed(4),
                Kind::F64 => Kind::Fixed(8),
                k => k,
            };
        }
        Ok(layout)
    }

    /// Bytes per row (a multiple of 8).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Offsets of the string/binary views in a row.
    pub fn views(&self) -> &[usize] {
        &self.views
    }

    /// Whether every key is fixed-width (no string or binary views): such
    /// rows hold the whole key and [`decode`](Self::decode) restores it.
    pub fn is_fixed_width(&self) -> bool {
        self.views.is_empty()
    }

    /// The key columns of `rows` (rows of this layout, in output order), in
    /// key order: the inverse of [`encode`](Self::encode). Floats come back
    /// normalized, as they were stored; NULL keys come back NULL.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for string or binary keys (their bytes
    /// may live in a heap); `INTERNAL_ERROR` for a row shorter than the
    /// layout's stride.
    pub fn decode(&self, rows: &[&[u8]]) -> Result<Vec<ArrayRef>> {
        if let Some(r) = rows.iter().find(|r| r.len() < self.stride) {
            return Err(internal(format!(
                "a {}-byte key row for a {}-byte layout",
                r.len(),
                self.stride
            )));
        }
        (0..self.columns.len())
            .map(|i| self.decode_column(i, rows))
            .collect()
    }

    /// The columns of `rows`, each a row of this layout with the heap its
    /// long views point into (in output order), in column order: the
    /// inverse of [`encode`](Self::encode) for every column type, strings
    /// and binaries included. NULLs come back NULL; floats come back as
    /// stored (normalized in a key layout, bit for bit in a
    /// [`for_payload`](Self::for_payload) layout).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a row shorter than the stride, a view outside
    /// its heap or a string that is not UTF-8 (a corrupt row).
    pub fn decode_rows(&self, rows: &[(&[u8], &[u8])]) -> Result<Vec<ArrayRef>> {
        if let Some((r, _)) = rows.iter().find(|(r, _)| r.len() < self.stride) {
            return Err(internal(format!(
                "a {}-byte row for a {}-byte layout",
                r.len(),
                self.stride
            )));
        }
        let fixed: Vec<&[u8]> = rows.iter().map(|&(r, _)| r).collect();
        self.columns
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if c.kind != Kind::View {
                    return self.decode_column(i, &fixed);
                }
                let valid = |r: &[u8]| {
                    c.null_bit
                        .is_none_or(|bit| r[self.validity_at + bit / 8] & (1 << (bit % 8)) == 0)
                };
                let values = rows.iter().map(|&(r, heap)| {
                    if !valid(r) {
                        return Ok(None);
                    }
                    let len = view_len(r, c.offset);
                    if len <= INLINE_MAX {
                        Ok(Some(&r[c.offset + 4..c.offset + 4 + len]))
                    } else {
                        let start = usize::try_from(word(r, c.offset + 8)).unwrap_or(usize::MAX);
                        heap.get(start..start.saturating_add(len))
                            .map(Some)
                            .ok_or_else(|| internal("a row view outside its heap".to_owned()))
                    }
                });
                let values = values.collect::<Result<Vec<Option<&[u8]>>>>()?;
                strings_array(&c.data_type, &values)
            })
            .collect()
    }

    /// Fixed-width column `i` of `rows` (see [`decode`](Self::decode)).
    fn decode_column(&self, i: usize, rows: &[&[u8]]) -> Result<ArrayRef> {
        let c = &self.columns[i];
        let nulls = c.null_bit.map(|bit| {
            let (byte, mask) = (self.validity_at + bit / 8, 1u8 << (bit % 8));
            NullBuffer::new(rows.iter().map(|r| r[byte] & mask == 0).collect())
        });
        let at = c.offset;
        if c.kind == Kind::Bool {
            let values: BooleanBuffer = rows.iter().map(|r| r[at] != 0).collect();
            return Ok(Arc::new(BooleanArray::new(values, nulls)) as ArrayRef);
        }
        if c.kind == Kind::View {
            let feature = format!("decoding a row-format key of type {}", c.data_type);
            return Err(KrishivUnsupportedFeature { feature }.into());
        }
        let w = c.kind.width();
        let mut values = MutableBuffer::from_len_zeroed(rows.len() * w);
        match w {
            4 => gather::<4>(values.as_slice_mut(), rows, at),
            8 => gather::<8>(values.as_slice_mut(), rows, at),
            16 => gather::<16>(values.as_slice_mut(), rows, at),
            _ => {
                for (out, r) in values.as_slice_mut().chunks_exact_mut(w.max(1)).zip(rows) {
                    out.copy_from_slice(&r[at..at + w]);
                }
            }
        }
        let data = ArrayData::builder(c.data_type.clone())
            .len(rows.len())
            .add_buffer(values.into())
            .nulls(nulls)
            .build()
            .map_err(|e| internal(e.to_string()))?;
        Ok(make_array(data))
    }

    /// Encodes the key `columns` of one batch (one per key type, in order)
    /// and, for a key layout, computes their stored hash `h` (see the
    /// module docs).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a column does not have its key type, or a value
    /// is longer than 4 GiB.
    pub fn encode(&self, columns: &[ArrayRef]) -> Result<EncodedKeys> {
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
            // A payload layout's rows are never hashed.
            hashes: if self.hashed {
                vec![fold(SEEDLESS, MIX); n]
            } else {
                Vec::new()
            },
            hashed: self.hashed,
        };
        let strings = foldhash::fast::FixedState::with_seed(SEEDLESS);
        for (column, array) in self.columns.iter().zip(columns) {
            if array.len() != n {
                return Err(internal("key columns of different lengths".to_owned()));
            }
            let nulls = array.logical_nulls();
            if let Some(bit) = column.null_bit {
                keys.write_nulls(self.validity_at + bit / 8, 1 << (bit % 8), nulls.as_ref());
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
            if self.hashed {
                keys.rows[i * self.stride..i * self.stride + 8].copy_from_slice(&h.to_le_bytes());
            }
        }
        Ok(keys)
    }

    /// Whether the rows have string or binary views (and so may point into
    /// a heap).
    pub fn has_views(&self) -> bool {
        !self.views.is_empty()
    }

    /// Bytes of `row`'s values kept in a heap (views longer than inline).
    pub fn long_bytes(&self, row: &[u8]) -> usize {
        self.views
            .iter()
            .map(|&v| view_len(row, v))
            .filter(|&len| len > INLINE_MAX)
            .sum()
    }

    /// Appends `row` (a row of this layout whose long values are in `heap`)
    /// to `rows`, and its long values to `dst_heap`, with the appended
    /// row's views pointing into `dst_heap`: the row moves to another
    /// table with its own heap.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a row shorter than the stride or a view outside
    /// `heap` (a corrupt row); nothing is appended then.
    pub fn push_row(
        &self,
        row: &[u8],
        heap: &[u8],
        rows: &mut Vec<u8>,
        dst_heap: &mut Vec<u8>,
    ) -> Result<()> {
        let row = row
            .get(..self.stride)
            .ok_or_else(|| internal("a key row shorter than the stride".to_owned()))?;
        let (start, heap_start) = (rows.len(), dst_heap.len());
        rows.extend_from_slice(row);
        for &v in &self.views {
            let len = view_len(row, v);
            if len <= INLINE_MAX {
                continue;
            }
            let from = usize::try_from(word(row, v + 8)).unwrap_or(usize::MAX);
            let Some(bytes) = heap.get(from..from.saturating_add(len)) else {
                rows.truncate(start);
                dst_heap.truncate(heap_start);
                return Err(internal("a row view outside its heap".to_owned()));
            };
            let at = dst_heap.len() as u64;
            dst_heap.extend_from_slice(bytes);
            rows[start + v + 8..start + v + 16].copy_from_slice(&at.to_le_bytes());
        }
        Ok(())
    }

    /// Whether row `a` (with heap `a_heap`) has the same key as row `b`
    /// (with heap `b_heap`). Rows must be of this layout.
    ///
    /// Always inlined: it runs once per probe of a hash table, and a call
    /// there cost as much as the compare (P2-T159 profile of H2O q01).
    #[inline(always)]
    pub fn eq(&self, a: &[u8], a_heap: &[u8], b: &[u8], b_heap: &[u8]) -> bool {
        match self.compare {
            Compare::One => word(a, 8) == word(b, 8),
            Compare::Two => word(a, 8) == word(b, 8) && word(a, 16) == word(b, 16),
            Compare::Words => {
                let (Some(a), Some(b)) = (a.get(..self.stride), b.get(..self.stride)) else {
                    return false;
                };
                let ((wa, _), (wb, _)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
                let at = |w: &[[u8; 8]], i: usize| w.get(i).map_or(0, |x| u64::from_le_bytes(*x));
                // Every compared word first (one branch), then the views.
                let diff = self
                    .words
                    .iter()
                    .fold(0, |d, &w| d | (at(wa, w / 8) ^ at(wb, w / 8)));
                diff == 0
                    && self.views.iter().all(|&v| {
                        let len = view_len(a, v);
                        if len <= INLINE_MAX {
                            at(wa, v / 8 + 1) == at(wb, v / 8 + 1)
                        } else {
                            // A view outside its heap (a corrupt row)
                            // equals nothing, not even another corrupt row.
                            match (heap_bytes(a, a_heap, v, len), heap_bytes(b, b_heap, v, len)) {
                                (Some(x), Some(y)) => x == y,
                                _ => false,
                            }
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
    /// Empty when the layout is not hashed.
    hashes: Vec<u64>,
    hashed: bool,
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

    /// The hash of each row, writable: tests forge `h` collisions.
    #[cfg(test)]
    pub(super) fn hashes_mut(&mut self) -> &mut [u64] {
        &mut self.hashes
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

    /// Bytes these keys hold (rows, heap and hashes).
    pub fn bytes(&self) -> usize {
        self.rows.capacity() + self.heap.capacity() + self.hashes.capacity() * size_of::<u64>()
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
        // Exact capacities: the arena accounts these buffers by capacity.
        let mut heap = None;
        if layout.has_views() {
            let long: usize = idx
                .iter()
                .filter_map(|&i| self.rows.get(i * stride..(i + 1) * stride))
                .flat_map(|row| layout.views().iter().map(move |&v| view_len(row, v)))
                .filter(|&len| len > INLINE_MAX)
                .fold(0usize, usize::saturating_add);
            let mut offsets = Vec::new();
            let mut bytes = Vec::new();
            offsets
                .try_reserve_exact(idx.len() + 1)
                .map_err(|e| internal(e.to_string()))?;
            bytes
                .try_reserve_exact(long)
                .map_err(|e| internal(e.to_string()))?;
            offsets.push(0i64);
            heap = Some((offsets, bytes));
        }
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

    /// Sets bit `mask` of byte `at` in each row whose value is NULL.
    fn write_nulls(&mut self, at: usize, mask: u8, nulls: Option<&NullBuffer>) {
        let Some(nulls) = nulls else { return };
        for i in nulls.iter().enumerate().filter(|(_, v)| !v).map(|(i, _)| i) {
            self.row_mut(i)[at] |= mask;
            if self.hashed {
                self.hashes[i] = mix(self.hashes[i], NULL_HASH);
            }
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
                    if self.hashed {
                        self.hashes[i] = mix(self.hashes[i], u64::from(v));
                    }
                }
            }
            Kind::F32 => {
                let a = array
                    .as_primitive_opt::<Float32Type>()
                    .ok_or_else(|| internal("float key expected".to_owned()))?;
                for i in (0..self.len).filter(|&i| valid(i)) {
                    let bits = normalize32(a.value(i)).to_bits();
                    self.row_mut(i)[offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
                    if self.hashed {
                        self.hashes[i] = mix(self.hashes[i], u64::from(bits));
                    }
                }
            }
            Kind::F64 => {
                let a = array
                    .as_primitive_opt::<Float64Type>()
                    .ok_or_else(|| internal("double key expected".to_owned()))?;
                for i in (0..self.len).filter(|&i| valid(i)) {
                    let bits = normalize64(a.value(i)).to_bits();
                    self.row_mut(i)[offset..offset + 8].copy_from_slice(&bits.to_le_bytes());
                    if self.hashed {
                        self.hashes[i] = mix(self.hashes[i], bits);
                    }
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
        if !self.hashed {
            for (i, (row, v)) in rows.zip(values).enumerate() {
                if nulls.is_none_or(|n| n.is_valid(i))
                    && let Some(dst) = row.get_mut(offset..offset + W)
                {
                    dst.copy_from_slice(v);
                }
            }
            return;
        }
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
                self.put_arrow_views(offset, nulls, strings, a.views(), |i| a.value(i).as_bytes())
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
                self.put_arrow_views(offset, nulls, strings, a.views(), |i| a.value(i))
            }
            t => Err(internal(format!("string key expected, got {t}"))),
        }
    }

    /// [`put_views`](Self::put_views) for Arrow view arrays: a value of at
    /// most 12 bytes is already a row view (its length, then the bytes
    /// inline, zero-padded: Arrow validates the padding, and the copy clears
    /// it again so equal keys always have equal bytes), so it is copied
    /// without touching the data buffers; longer values go to the heap as
    /// `put_views` does.
    fn put_arrow_views<'a>(
        &mut self,
        offset: usize,
        nulls: Option<&NullBuffer>,
        strings: &foldhash::fast::FixedState,
        views: &[u128],
        value: impl Fn(usize) -> &'a [u8],
    ) -> Result<()> {
        let stride = self.stride;
        for (i, &v) in views
            .iter()
            .enumerate()
            .take(self.len)
            .filter(|&(i, _)| nulls.is_none_or(|n| n.is_valid(i)))
        {
            let len = usize::try_from(v & u128::from(u32::MAX)).unwrap_or(usize::MAX);
            if len > INLINE_MAX {
                self.put_view(i, offset, strings, value(i))?;
                continue;
            }
            // Clear the padding with a mask (no per-row `memset`).
            let keep = 32 + 8 * len;
            let v = if keep >= 128 {
                v
            } else {
                v & ((1u128 << keep) - 1)
            };
            if self.hashed {
                self.hashes[i] = mix(self.hashes[i], inline_hash(v));
            }
            self.rows[i * stride + offset..i * stride + offset + VIEW_BYTES]
                .copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }

    fn put_views<'a>(
        &mut self,
        offset: usize,
        nulls: Option<&NullBuffer>,
        strings: &foldhash::fast::FixedState,
        value: impl Fn(usize) -> &'a [u8],
    ) -> Result<()> {
        for i in (0..self.len).filter(|&i| nulls.is_none_or(|v| v.is_valid(i))) {
            self.put_view(i, offset, strings, value(i))?;
        }
        Ok(())
    }

    /// Writes value `bytes` as row `i`'s view at `offset`, hashing it.
    fn put_view(
        &mut self,
        i: usize,
        offset: usize,
        strings: &foldhash::fast::FixedState,
        bytes: &[u8],
    ) -> Result<()> {
        let len = u32::try_from(bytes.len())
            .map_err(|_| internal("a key value longer than 4 GiB".to_owned()))?;
        let at = self.heap.len();
        let stride = self.stride;
        let row = &mut self.rows[i * stride..(i + 1) * stride];
        let view = &mut row[offset..offset + VIEW_BYTES];
        view[..4].copy_from_slice(&len.to_le_bytes());
        if bytes.len() <= INLINE_MAX {
            view[4..4 + bytes.len()].copy_from_slice(bytes);
            if self.hashed {
                let mut v = [0; VIEW_BYTES];
                v.copy_from_slice(view);
                self.hashes[i] = mix(self.hashes[i], inline_hash(u128::from_le_bytes(v)));
            }
        } else {
            if self.hashed {
                self.hashes[i] = mix(self.hashes[i], strings.hash_one(bytes));
            }
            view[4..8].copy_from_slice(&bytes[..4]);
            view[8..].copy_from_slice(&(at as u64).to_le_bytes());
            self.heap.extend_from_slice(bytes);
        }
        Ok(())
    }
}

/// Copies the `N` bytes at `at` of each row into `out`, one row after another:
/// a fixed-size copy, where a width known only at run time calls `memcpy`.
fn gather<const N: usize>(out: &mut [u8], rows: &[&[u8]], at: usize) {
    for (out, r) in out.as_chunks_mut::<N>().0.iter_mut().zip(rows) {
        out.copy_from_slice(&r[at..at + N]);
    }
}

/// An array of string or binary `data_type` holding `values`.
fn strings_array(data_type: &DataType, values: &[Option<&[u8]>]) -> Result<ArrayRef> {
    use arrow_array::{
        BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
        StringViewArray,
    };
    let text = || {
        values
            .iter()
            .map(|v| v.map(std::str::from_utf8).transpose())
            .collect::<std::result::Result<Vec<Option<&str>>, _>>()
            .map_err(|e| internal(format!("a row string is not UTF-8: {e}")))
    };
    Ok(match data_type {
        DataType::Utf8 => Arc::new(StringArray::from(text()?)),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(text()?)),
        DataType::Utf8View => Arc::new(StringViewArray::from(text()?)),
        DataType::Binary => Arc::new(BinaryArray::from(values.to_vec())),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from(values.to_vec())),
        DataType::BinaryView => Arc::new(BinaryViewArray::from(values.to_vec())),
        t => return Err(internal(format!("no row view decoding for {t}"))),
    })
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
/// indexes and grace sub-partitions come from `g`, so keys with different
/// `h` cannot be aimed at one slot run without the seed (keys with equal
/// `h` share `g` under every seed); `g` never leaves the operator instance
/// that drew `seed`.
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

/// The heap bytes of the long view at `at` of `row` (`None` outside the
/// heap: a corrupt row).
#[inline]
fn heap_bytes<'a>(row: &[u8], heap: &'a [u8], at: usize, len: usize) -> Option<&'a [u8]> {
    let start = usize::try_from(word(row, at + 8)).ok()?;
    heap.get(start..start.checked_add(len)?)
}

/// The hash of an inline view (a value of at most 12 bytes, zero-padded):
/// its two words folded, so a short key hashes without a byte loop. Every
/// writer hashes short values this way and long ones with foldhash at the
/// constant seed, so equal values hash equally whatever their array type.
///
/// No seed is folded in here: `h` is seedless (P2-T157 Appendix C), so a
/// seed would not change which values collide, and folding one cost 1.5-3.5% on the
/// short-key encode benchmarks. Collision resistance is the table-local
/// `g` ([`finalize`]) and the flooding defence's concern, not `h`'s.
#[inline]
fn inline_hash(view: u128) -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the view's two words are intended"
    )]
    let (lo, hi) = (view as u64, (view >> 64) as u64);
    fold(fold(lo ^ MIX, FINISH) ^ hi, MIX)
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
mod tests {
    use super::*;

    /// The zero-padded inline view of `value` (at most 12 bytes).
    fn view(value: &[u8]) -> u128 {
        let mut b = [0u8; VIEW_BYTES];
        b[..4].copy_from_slice(&u32::try_from(value.len()).unwrap().to_le_bytes());
        b[4..4 + value.len()].copy_from_slice(value);
        u128::from_le_bytes(b)
    }

    /// Inline Arrow views are copied with their padding cleared: views
    /// whose bytes past the value are not zero (Arrow's validation rejects
    /// them, but unchecked arrays need not have been validated) give the
    /// rows and hashes of zero-padded views, for values of 0 to 12 bytes.
    #[test]
    fn inline_views_clear_their_padding() {
        let values: [&[u8]; 5] = [b"", b"x", b"abcdef", b"abcdefghijk", b"abcdefghijkl"];
        let clean: Vec<u128> = values.iter().map(|v| view(v)).collect();
        let dirty: Vec<u128> = clean
            .iter()
            .zip(values)
            .map(|(&v, b)| match 32 + 8 * b.len() {
                128 => v,
                keep => v | (u128::MAX << keep),
            })
            .collect();
        assert_ne!(dirty, clean);
        let layout = RowLayout::new(&[DataType::BinaryView], &[true]).unwrap();
        let offset = layout.columns[0].offset;
        let strings = foldhash::fast::FixedState::with_seed(3);
        let put = |views: &[u128]| {
            let n = views.len();
            let mut keys = EncodedKeys {
                stride: layout.stride,
                len: n,
                rows: vec![0; n * layout.stride],
                heap: Vec::new(),
                hashes: vec![0; n],
                hashed: true,
            };
            keys.put_arrow_views(offset, None, &strings, views, |i| values[i])
                .unwrap();
            (keys.rows, keys.hashes)
        };
        assert_eq!(put(&dirty), put(&clean));
    }
}
