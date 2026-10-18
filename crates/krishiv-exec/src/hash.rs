// SPDX-License-Identifier: Apache-2.0
//! Hash-table building blocks shared by the hash operators (ADR 0011, E02
//! decision: `hashbrown::HashTable` over a row-format key arena).
//!
//! - [`KeyEncoder`] turns key columns into binary-comparable rows (arrow-row)
//!   after Spark's key normalization: `-0.0` becomes `0.0` and every NaN
//!   becomes one NaN, so they group (and join) together. NULL keys encode
//!   equal to each other: `GROUP BY` puts them in one group; a join skips
//!   them before probing.
//! - [`HashSeed`] is the seeded, non-observable key hash (foldhash). The seed
//!   comes from the query's `ExecConfig::seed`, so deterministic mode
//!   replays the same table layouts and spill partitions; otherwise it is
//!   random per operator, so key data cannot be crafted to collide (hash
//!   flooding the table or piling every key into one spill partition).
//! - [`KeyTable`] maps each distinct key row to a dense `u32` id: the rows
//!   live in one arena (arrow-row [`Rows`]), the index is a
//!   `HashTable<u32>` and each key's hash is cached, so growing the index
//!   never re-hashes keys. Aggregation keeps accumulators per id; a join
//!   build keeps payload rows per id.
//! - [`spill_partition`] picks a grace / spill partition from hash bits
//!   that the table itself does not use.

use std::hash::BuildHasher;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, PrimitiveArray};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::DataType;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry;

use krishiv_common::error::KrishivUnsupportedFeature;

use crate::error::{Result, internal};

/// The key hash: foldhash with a per-query seed. Never observable (Spark's
/// observable hash is Murmur3, in the kernels).
#[derive(Clone, Debug)]
pub struct HashSeed(foldhash::fast::FixedState);

impl HashSeed {
    /// A hasher seeded with `seed` (deterministic mode) or, with `None`, a
    /// fresh random seed (from std's OS-seeded `RandomState`).
    pub fn new(seed: Option<u64>) -> Self {
        let seed =
            seed.unwrap_or_else(|| std::collections::hash_map::RandomState::new().hash_one(0u8));
        Self(foldhash::fast::FixedState::with_seed(seed))
    }

    /// Hashes one encoded key row.
    pub fn hash(&self, row: &[u8]) -> u64 {
        self.0.hash_one(row)
    }
}

/// Bits of the hash per spill partitioning level (8 partitions).
pub const PARTITION_BITS: u32 = 3;
/// Partitions per spill level.
pub const PARTITIONS: usize = 1 << PARTITION_BITS;
/// Deepest re-partitioning level: the levels use the hash bits below the top
/// seven (hashbrown's tag) from the top down, and the table's bucket index
/// uses the low bits, so neither correlates with a partition.
pub const MAX_LEVEL: u32 = (64 - 7) / PARTITION_BITS - 1;

/// The partition of `hash` at re-partitioning `level` (`0..=MAX_LEVEL`).
pub fn spill_partition(hash: u64, level: u32) -> usize {
    let shift = 64 - 7 - PARTITION_BITS * (level.min(MAX_LEVEL) + 1);
    usize::try_from((hash >> shift) & (PARTITIONS as u64 - 1)).unwrap_or(0)
}

/// Encodes key columns as binary-comparable rows with Spark's key
/// normalization; decodes rows back to columns of the original types.
#[derive(Debug)]
pub struct KeyEncoder {
    converter: RowConverter,
    types: Vec<DataType>,
}

impl KeyEncoder {
    /// An encoder for keys of `types`.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a key type arrow-row cannot encode.
    pub fn new(types: &[DataType]) -> Result<Self> {
        // Normalization covers top-level float keys only (Spark normalizes
        // nested ones too); refuse the rest rather than split groups.
        if let Some(t) = types.iter().find(|t| needs_nested_normalization(t)) {
            let feature = format!("grouping or join key of type {t}");
            return Err(KrishivUnsupportedFeature { feature }.into());
        }
        let fields = types.iter().map(|t| SortField::new(t.clone())).collect();
        let converter = RowConverter::new(fields).map_err(|e| internal(e.to_string()))?;
        Ok(Self {
            converter,
            types: types.to_vec(),
        })
    }

    /// The key types.
    pub fn types(&self) -> &[DataType] {
        &self.types
    }

    /// Encodes `columns` (one per key type) into rows.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a column does not have its key type.
    pub fn encode(&self, columns: &[ArrayRef]) -> Result<Rows> {
        let normalized: Vec<ArrayRef> = columns.iter().map(normalize_key).collect();
        self.converter
            .convert_columns(&normalized)
            .map_err(|e| internal(e.to_string()))
    }

    /// Decodes keys `ids` of `table` (encoded by this encoder) into columns.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if an id is out of range.
    pub fn decode(
        &self,
        table: &KeyTable,
        ids: impl Iterator<Item = usize>,
    ) -> Result<Vec<ArrayRef>> {
        let parser = self.converter.parser();
        let n = table.len();
        let picked: Vec<_> = ids
            .map(|i| {
                (i < n)
                    .then(|| parser.parse(table.row(i)))
                    .ok_or_else(|| internal(format!("key row {i} of {n}")))
            })
            .collect::<Result<_>>()?;
        self.converter
            .convert_rows(picked)
            .map_err(|e| internal(e.to_string()))
    }
}

/// Whether `t` holds floats this encoder does not normalize: anything but
/// a plain `Float32`/`Float64` column that contains floats (`Float16`,
/// floats inside nested, dictionary or run-end encoded types).
fn needs_nested_normalization(t: &DataType) -> bool {
    fn has_float(t: &DataType) -> bool {
        match t {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => true,
            DataType::Struct(f) => f.iter().any(|f| has_float(f.data_type())),
            DataType::List(f)
            | DataType::LargeList(f)
            | DataType::FixedSizeList(f, _)
            | DataType::ListView(f)
            | DataType::LargeListView(f)
            | DataType::Map(f, _) => has_float(f.data_type()),
            DataType::Dictionary(_, v) => has_float(v),
            DataType::RunEndEncoded(_, v) => has_float(v.data_type()),
            _ => false,
        }
    }
    !matches!(t, DataType::Float32 | DataType::Float64) && has_float(t)
}

/// Spark's `NormalizeFloatingNumbers` for keys: `-0.0` → `0.0`, any NaN →
/// the canonical NaN. Other types pass through.
fn normalize_key(column: &ArrayRef) -> ArrayRef {
    match column.data_type() {
        DataType::Float64 => {
            let a = column.as_primitive::<Float64Type>();
            Arc::new(a.unary::<_, Float64Type>(|v| if v.is_nan() { f64::NAN } else { v + 0.0 }))
        }
        DataType::Float32 => {
            let a: &PrimitiveArray<Float32Type> = column.as_primitive();
            Arc::new(a.unary::<_, Float32Type>(|v| if v.is_nan() { f32::NAN } else { v + 0.0 }))
        }
        _ => Arc::clone(column),
    }
}

/// Distinct key rows with dense ids `0..len` in insertion order, in one
/// byte arena.
#[derive(Debug, Default)]
pub struct KeyTable {
    data: Vec<u8>,
    /// `offsets[id]..offsets[id + 1]` is key `id`'s row; starts with 0.
    offsets: Vec<usize>,
    hashes: Vec<u64>,
    index: HashTable<u32>,
}

/// Bytes per index bucket: a `u32` id and hashbrown's control byte.
const BUCKET_BYTES: usize = size_of::<u32>() + 1;

impl KeyTable {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct keys.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// The encoded row of key `id` (empty if out of range).
    pub fn row(&self, id: usize) -> &[u8] {
        match (self.offsets.get(id), self.offsets.get(id + 1)) {
            (Some(&s), Some(&e)) => self.data.get(s..e).unwrap_or_default(),
            _ => &[],
        }
    }

    /// The cached hash of each key, by id.
    pub fn hashes(&self) -> &[u64] {
        &self.hashes
    }

    /// Bytes held.
    pub fn memory_size(&self) -> usize {
        self.data.capacity()
            + self.offsets.capacity() * size_of::<usize>()
            + self.hashes.capacity() * size_of::<u64>()
            + self.index.num_buckets() * BUCKET_BYTES
    }

    /// [`memory_size`](Self::memory_size) after [`reserve`](Self::reserve)
    /// for `new_rows` keys of `new_bytes` bytes in total.
    pub fn projected_size(&self, new_rows: usize, new_bytes: usize) -> usize {
        let len = self.len().saturating_add(new_rows);
        let buckets = if len > self.index.capacity() {
            buckets_for(len.max(self.index.capacity() + 1))
        } else {
            self.index.num_buckets()
        };
        grown(&self.data, new_bytes)
            + grown(
                &self.offsets,
                new_rows + usize::from(self.offsets.is_empty()),
            ) * size_of::<usize>()
            + grown(&self.hashes, new_rows) * size_of::<u64>()
            + buckets * BUCKET_BYTES
    }

    /// Grows every buffer for `new_rows` more keys of `new_bytes` bytes, as
    /// [`projected_size`](Self::projected_size) assumed.
    pub fn reserve(&mut self, new_rows: usize, new_bytes: usize) {
        self.data.reserve(new_bytes);
        self.offsets
            .reserve(new_rows + usize::from(self.offsets.is_empty()));
        self.hashes.reserve(new_rows);
        let hashes = &self.hashes;
        self.index.reserve(new_rows, |&id| {
            hashes.get(id as usize).copied().unwrap_or(0)
        });
    }

    /// The id of each row of `keys` (hashed with `seed`), inserting new keys;
    /// writes the ids to `ids` and returns how many keys were new.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` past `u32::MAX` keys.
    pub fn find_or_insert(
        &mut self,
        keys: &Rows,
        seed: &HashSeed,
        ids: &mut Vec<u32>,
    ) -> Result<usize> {
        ids.clear();
        let before = self.len();
        for key in keys {
            let bytes = key.data();
            ids.push(self.insert_hashed(bytes, seed.hash(bytes))?.0);
        }
        Ok(self.len() - before)
    }

    /// The id of key row `bytes` whose hash is `hash`, inserting it if new;
    /// `true` if it was new. For callers that keep the hashes (a join's
    /// build rows carry theirs).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` past `u32::MAX` keys.
    pub fn insert_hashed(&mut self, bytes: &[u8], hash: u64) -> Result<(u32, bool)> {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        let Self {
            data,
            offsets,
            hashes,
            index,
        } = self;
        let eq = |&id: &u32| key_eq(data, offsets, hashes, id, bytes, hash);
        match index.entry(hash, eq, |&id| hashes[id as usize]) {
            Entry::Occupied(o) => Ok((*o.get(), false)),
            Entry::Vacant(v) => {
                let id = u32::try_from(hashes.len())
                    .map_err(|_| internal("more than 2^32 keys in one table".to_owned()))?;
                data.extend_from_slice(bytes);
                offsets.push(data.len());
                hashes.push(hash);
                v.insert(id);
                Ok((id, true))
            }
        }
    }

    /// The id of key row `bytes` whose hash is `hash`, if present (a join
    /// probe).
    pub fn find(&self, bytes: &[u8], hash: u64) -> Option<u32> {
        let eq = |&id: &u32| key_eq(&self.data, &self.offsets, &self.hashes, id, bytes, hash);
        self.index.find(hash, eq).copied()
    }
}

/// Whether key `id` is `bytes` with hash `hash`.
#[inline]
fn key_eq(
    data: &[u8],
    offsets: &[usize],
    hashes: &[u64],
    id: u32,
    bytes: &[u8],
    hash: u64,
) -> bool {
    let id = id as usize;
    hashes.get(id) == Some(&hash)
        && match (offsets.get(id), offsets.get(id + 1)) {
            (Some(&s), Some(&e)) => data.get(s..e).is_some_and(|k| eq_bytes(k, bytes)),
            _ => false,
        }
}

/// Byte-slice equality without a `memcmp` call for short keys (most
/// fixed-width keys encode to at most 16 bytes).
#[inline]
fn eq_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a.len() <= 16 {
        let mut x = [0u8; 16];
        let mut y = [0u8; 16];
        x[..a.len()].copy_from_slice(a);
        y[..b.len()].copy_from_slice(b);
        return u128::from_ne_bytes(x) == u128::from_ne_bytes(y);
    }
    a == b
}

/// `v`'s capacity after `v.reserve(extra)` (std's amortized growth: at
/// least double, at least 8 elements for small types).
pub fn grown<T>(v: &Vec<T>, extra: usize) -> usize {
    let need = v.len().saturating_add(extra);
    if need <= v.capacity() {
        v.capacity()
    } else {
        let min = if size_of::<T>() == 1 { 8 } else { 4 };
        need.max(v.capacity().saturating_mul(2)).max(min)
    }
}

/// hashbrown's bucket count for `cap` entries (load factor 7/8).
fn buckets_for(cap: usize) -> usize {
    if cap < 4 {
        4
    } else if cap < 8 {
        8
    } else {
        (cap.saturating_mul(8) / 7).next_power_of_two()
    }
}

#[cfg(test)]
#[cfg(not(loom))]
mod tests {
    use arrow_array::{Float64Array, Int32Array, StringViewArray};

    use super::*;

    #[test]
    fn normalizes_and_groups_null_nan_and_signed_zero() {
        let enc = KeyEncoder::new(&[DataType::Float64, DataType::Utf8View]).unwrap();
        let f: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.0),
            Some(-0.0),
            Some(f64::NAN),
            Some(-f64::NAN),
            None,
            None,
        ]));
        let s: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("a"),
            Some("a"),
            Some("a"),
            Some("a"),
            None,
            None,
        ]));
        let rows = enc.encode(&[f, s]).unwrap();
        let mut t = KeyTable::new();
        let projected = t.projected_size(6, rows.size());
        t.reserve(6, rows.size());
        assert!(t.memory_size() <= projected);
        let mut ids = Vec::new();
        let seed = HashSeed::new(Some(1));
        assert_eq!(t.find_or_insert(&rows, &seed, &mut ids).unwrap(), 3);
        assert_eq!(ids, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!((t.len(), t.is_empty(), t.hashes().len()), (3, false, 3));
        let back = enc.decode(&t, 0..3).unwrap();
        let f = back[0].as_primitive::<Float64Type>();
        assert!(f.value(0).to_bits() == 0.0f64.to_bits() && f.value(1).is_nan() && f.is_null(2));
        assert!(enc.decode(&t, 5..6).is_err());
        assert!(t.row(9).is_empty());
        assert_eq!(enc.types().len(), 2);
    }

    #[test]
    fn projection_bounds_growth() {
        let enc = KeyEncoder::new(&[DataType::Int32]).unwrap();
        let mut t = KeyTable::new();
        let seed = HashSeed::new(None);
        let mut ids = Vec::new();
        for round in 0..20 {
            let keys: ArrayRef = Arc::new(Int32Array::from_iter_values(
                (0..1000).map(|i| i + round * 700),
            ));
            let f32s: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![-0.0f32]));
            assert_eq!(
                normalize_key(&f32s)
                    .as_primitive::<Float32Type>()
                    .value(0)
                    .to_bits(),
                0
            );
            let rows = enc.encode(&[keys]).unwrap();
            let bytes = rows.size();
            let projected = t.projected_size(1000, bytes);
            t.reserve(1000, bytes);
            t.find_or_insert(&rows, &seed, &mut ids).unwrap();
            assert!(t.memory_size() <= projected, "round {round}");
        }
        assert_eq!(t.len(), 19 * 700 + 1000);
        assert!(buckets_for(2) == 4 && buckets_for(5) == 8);
        assert!(spill_partition(u64::MAX, 0) < PARTITIONS);
        assert!(spill_partition(u64::MAX, MAX_LEVEL + 5) < PARTITIONS);
        assert!(KeyEncoder::new(&[DataType::Null]).is_ok());
        let floats = arrow_schema::Field::new("f", DataType::Float64, true);
        let nested = DataType::List(Arc::new(floats));
        assert!(KeyEncoder::new(&[nested]).is_err());
        assert!(KeyEncoder::new(&[DataType::Float16]).is_err());
        // Dictionary-encoded floats would skip normalization (-0.0 and 0.0
        // in two groups): refused.
        let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Float64));
        assert!(KeyEncoder::new(&[dict]).is_err());
        let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert!(KeyEncoder::new(&[dict]).is_ok());
        // Unseeded hashers are randomly seeded (not a public constant).
        let (a, b) = (HashSeed::new(None), HashSeed::new(None));
        assert_ne!(a.hash(b"key"), b.hash(b"key"));
        assert_eq!(
            HashSeed::new(Some(3)).hash(b"k"),
            HashSeed::new(Some(3)).hash(b"k")
        );
        let ints = arrow_schema::Field::new("i", DataType::Int32, true);
        assert!(KeyEncoder::new(&[DataType::List(Arc::new(ints))]).is_ok());
    }
}
