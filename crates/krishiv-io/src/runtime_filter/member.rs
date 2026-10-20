// SPDX-License-Identifier: Apache-2.0
//! The row-level form of a [`KeyFilter`](super::KeyFilter), chosen at
//! finish from the build's distinct key count (design P2-T163 part 2):
//!
//! | distinct build keys | form | cost per probe row |
//! |---|---|---|
//! | dense integers (`count = max - min + 1`) | the range alone: exact | two compares |
//! | ≤ [`SMALL_SET_MAX`] | [`Small`]: compared against every member, no hash | a few compares |
//! | ≤ [`EXACT_SET_MAX`] (`krishiv.runtimeFilter.exactSet.maxValues`) | [`Exact`]: a hash set probed with `h` | one hash, one probe |
//! | more | a split-block Bloom filter probed with `h` | one hash, one block |
//!
//! DuckDB and Velox push exact IN-lists for small builds and ranges
//! otherwise; Spark builds a Bloom filter for every eligible join. Exact
//! forms pass no false positive, and below the cutoffs they cost no more
//! than a Bloom probe (the cutoffs come from the `runtime_filter`
//! microbenchmark).
//!
//! Evaluation is batch-at-a-time: the range is one compare loop over the
//! batch; the hashed forms hash the rows still passing **once**, into one
//! vector ([`super::hash::row_hashes`]), and probe that vector in a loop.

use arrow_array::Array;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder, Buffer};
use arrow_schema::DataType;
use hashbrown::HashTable;

use super::bloom::{RuntimeBloom, xxh64};
use super::hash::{decimal_hash, int_hash, row_hashes, xxh64_fixed};
use super::key::{KeyRange, KeyValue, Norm, NormalizedKeys, float_bits, for_each_bytes};

/// Most distinct keys compared without hashing.
pub const SMALL_SET_MAX: usize = 8;

/// Default for the most distinct keys an exact hash set holds
/// (`krishiv.runtimeFilter.exactSet.maxValues`); a larger build gets a
/// Bloom filter.
pub const EXACT_SET_MAX: usize = 4096;

/// How a key type's values are held by the exact forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// Integers, dates, timestamps and booleans in `width` bytes.
    Int { width: usize },
    /// Float bit patterns (`single`: a `float` key).
    Float { single: bool },
    /// Unscaled decimals.
    Decimal,
    /// String and binary bytes.
    Bytes,
}

impl Class {
    pub(crate) fn of(key_type: &DataType) -> Option<Self> {
        use DataType as D;
        Some(match key_type {
            D::Boolean | D::Int8 => Self::Int { width: 1 },
            D::Int16 => Self::Int { width: 2 },
            D::Int32 | D::Date32 => Self::Int { width: 4 },
            D::Int64 | D::Timestamp(..) => Self::Int { width: 8 },
            D::Float32 => Self::Float { single: true },
            D::Float64 => Self::Float { single: false },
            D::Decimal128(..) => Self::Decimal,
            D::Utf8 | D::LargeUtf8 | D::Utf8View | D::Binary | D::LargeBinary | D::BinaryView => {
                Self::Bytes
            }
            _ => return None,
        })
    }

    /// The row hash of an integer or float member held as
    /// [`fixed64`] (as the probe side hashes it).
    pub(crate) fn hash64(self, v: i64) -> u64 {
        #[expect(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "bit patterns"
        )]
        match self {
            Self::Int { width } => int_hash(v, width),
            Self::Float { single: true } => xxh64_fixed::<4>(u128::from(v as u32)),
            Self::Float { single: false } | Self::Decimal | Self::Bytes => {
                xxh64_fixed::<8>(u128::from(v as u64))
            }
        }
    }
}

/// An integer or float member as the 64-bit value the exact forms compare
/// (integers as they are; float bits, which fit 64 bits, reinterpreted).
#[expect(
    clippy::cast_possible_truncation,
    reason = "integer keys fit i64; float bits fit u64 and are reinterpreted"
)]
fn fixed64(n: i128) -> i64 {
    n as i64
}

/// A small exact set, compared member by member (no hash).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Small {
    /// Integers and float bits.
    Int(Box<[i64]>),
    /// Decimals.
    Wide(Box<[i128]>),
    /// Strings and binaries.
    Bytes(Box<[Box<[u8]>]>),
}

/// An exact hash set, placed by the row hash `h`.
#[derive(Clone)]
pub(crate) enum Exact {
    /// Integers and float bits.
    Int(HashTable<i64>),
    /// Decimals.
    Wide(HashTable<i128>),
    /// Strings and binaries.
    Bytes(HashTable<Box<[u8]>>),
}

impl Exact {
    fn len(&self) -> usize {
        match self {
            Self::Int(t) => t.len(),
            Self::Wide(t) => t.len(),
            Self::Bytes(t) => t.len(),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Self::Int(t) => t.capacity() * 9,
            Self::Wide(t) => t.capacity() * 17,
            Self::Bytes(t) => t.capacity() * 17 + t.iter().map(|b| b.len()).sum::<usize>(),
        }
    }
}

impl std::fmt::Debug for Exact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exact").field("len", &self.len()).finish()
    }
}

impl PartialEq for Exact {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => {
                a.len() == b.len() && a.iter().all(|&v| b.iter().any(|&w| w == v))
            }
            (Self::Wide(a), Self::Wide(b)) => {
                a.len() == b.len() && a.iter().all(|&v| b.iter().any(|&w| w == v))
            }
            (Self::Bytes(a), Self::Bytes(b)) => {
                a.len() == b.len() && a.iter().all(|v| b.iter().any(|w| w == v))
            }
            _ => false,
        }
    }
}

/// The row-level form (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Member {
    /// No membership test: the range decides (exact for dense integer
    /// keys), or no other form was built.
    Range,
    /// A small exact set.
    Small(Small),
    /// An exact hash set.
    Exact(Exact),
    /// A Bloom filter.
    Bloom(RuntimeBloom),
}

/// What [`Member`] a filter holds, for `EXPLAIN` and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterForm {
    /// Only the min/max range (exact for dense integer keys).
    Range,
    /// A small exact set of this many keys, compared without hashing.
    SmallSet(usize),
    /// An exact hash set of this many keys.
    HashSet(usize),
    /// A Bloom filter of this many bytes.
    Bloom(usize),
}

impl Member {
    /// The exact form for `members` (distinct, with their row hashes) of
    /// a key of `class`: `None` when a member does not fit the class.
    pub(crate) fn exact(class: Class, members: Vec<(u64, KeyValue)>) -> Option<Self> {
        if members.len() <= SMALL_SET_MAX {
            return Some(Self::Small(match class {
                Class::Int { .. } | Class::Float { .. } => Small::Int(
                    members
                        .iter()
                        .map(|(_, v)| match v {
                            KeyValue::Fixed(n) => Some(fixed64(*n)),
                            KeyValue::Bytes(_) => None,
                        })
                        .collect::<Option<_>>()?,
                ),
                Class::Decimal => Small::Wide(
                    members
                        .iter()
                        .map(|(_, v)| match v {
                            KeyValue::Fixed(n) => Some(*n),
                            KeyValue::Bytes(_) => None,
                        })
                        .collect::<Option<_>>()?,
                ),
                Class::Bytes => Small::Bytes(
                    members
                        .into_iter()
                        .map(|(_, v)| match v {
                            KeyValue::Bytes(b) => Some(b),
                            KeyValue::Fixed(_) => None,
                        })
                        .collect::<Option<_>>()?,
                ),
            }));
        }
        let n = members.len();
        // Sized for every member up front: the re-hash closures never run,
        // but stay correct (they recompute the member's row hash).
        Some(Self::Exact(match class {
            Class::Int { .. } | Class::Float { .. } => {
                let mut t = HashTable::with_capacity(n);
                for (h, v) in members {
                    let KeyValue::Fixed(x) = v else { return None };
                    t.insert_unique(h, fixed64(x), |&y| class.hash64(y));
                }
                Exact::Int(t)
            }
            Class::Decimal => {
                let mut t = HashTable::with_capacity(n);
                for (h, v) in members {
                    let KeyValue::Fixed(x) = v else { return None };
                    t.insert_unique(h, x, |&y| decimal_hash(y));
                }
                Exact::Wide(t)
            }
            Class::Bytes => {
                let mut t = HashTable::with_capacity(n);
                for (h, v) in members {
                    let KeyValue::Bytes(b) = v else { return None };
                    t.insert_unique(h, b, |y| xxh64(y));
                }
                Exact::Bytes(t)
            }
        }))
    }

    pub(crate) fn form(&self) -> FilterForm {
        match self {
            Self::Range => FilterForm::Range,
            Self::Small(s) => FilterForm::SmallSet(match s {
                Small::Int(v) => v.len(),
                Small::Wide(v) => v.len(),
                Small::Bytes(v) => v.len(),
            }),
            Self::Exact(e) => FilterForm::HashSet(e.len()),
            Self::Bloom(b) => FilterForm::Bloom(b.num_bytes()),
        }
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Range => 0,
            Self::Small(Small::Int(v)) => v.len() * 8,
            Self::Small(Small::Wide(v)) => v.len() * 16,
            Self::Small(Small::Bytes(v)) => v.iter().map(|b| 16 + b.len()).sum(),
            Self::Exact(e) => e.heap_bytes(),
            Self::Bloom(b) => b.num_bytes(),
        }
    }

    /// Estimated cost of testing one row, in nanoseconds: what the row
    /// stage's adaptive switch-off weighs against the rows a filter drops.
    /// The `runtime_filter` microbenchmark's per-row times at x86-64-v3
    /// (the shipped build), rounded up: range 0.4, small set 3.0, hash set
    /// 5.0, Bloom 4.3 ns (the baseline x86-64 build: 1.2, 5.4, 6.8, 10.5).
    pub(crate) fn row_cost_ns(&self) -> f64 {
        match self {
            Self::Range => 0.5,
            Self::Small(_) => 3.0,
            Self::Bloom(_) => 4.5,
            Self::Exact(_) => 5.0,
        }
    }
}

/// Rows of `keys` that may match: in `range`, valid (`valid`: `None`
/// means no NULL), and in `member`. NULL rows are `false`.
pub(crate) fn evaluate(
    range: Option<&KeyRange>,
    member: &Member,
    keys: &NormalizedKeys,
    valid: Option<&BooleanBuffer>,
) -> BooleanBuffer {
    let mut pass = range_pass(range, keys);
    if let Some(v) = valid {
        pass = &pass & v;
    }
    member_pass(member, keys, pass)
}

/// Whether each row is in `range` (every row without one): one compare
/// loop, packed 64 rows a word. NULL rows get an arbitrary answer.
fn range_pass(range: Option<&KeyRange>, keys: &NormalizedKeys) -> BooleanBuffer {
    let n = keys.len();
    match (&keys.norm, range) {
        (Norm::Int { values, .. }, Some(&KeyRange::Int(lo, hi))) => {
            pack(values, |v| (v >= lo) & (v <= hi))
        }
        (Norm::Decimal(values), Some(&KeyRange::Decimal(lo, hi))) => {
            pack(values, |v| (v >= lo) & (v <= hi))
        }
        // NaN fails both compares; a range exists only when the build had
        // no NaN key.
        (Norm::Float { values, .. }, Some(&KeyRange::Float(lo, hi))) => {
            pack(values, |v| (v >= lo) & (v <= hi))
        }
        (Norm::Bytes(a), Some(KeyRange::Bytes(lo, hi))) => {
            let mut out = BooleanBufferBuilder::new(n);
            for_each_bytes(a.as_ref(), |_, b| {
                out.append(b >= lo.as_ref() && b <= hi.as_ref());
            });
            out.finish()
        }
        _ => BooleanBuffer::new_set(n),
    }
}

/// `pass` restricted to the rows `member` may hold.
fn member_pass(member: &Member, keys: &NormalizedKeys, pass: BooleanBuffer) -> BooleanBuffer {
    match (member, &keys.norm) {
        (Member::Small(Small::Int(set)), Norm::Int { values, .. }) => {
            // No early exit: the compares vectorize.
            let hit = pack(values, |v| set.iter().fold(false, |a, &x| a | (x == v)));
            &pass & &hit
        }
        (Member::Small(Small::Int(set)), Norm::Float { values, single }) => {
            let hit = pack(values, |v| {
                let bits = fixed64(float_bits(v, *single));
                set.iter().fold(false, |a, &x| a | (x == bits))
            });
            &pass & &hit
        }
        (Member::Small(Small::Wide(set)), Norm::Decimal(values)) => {
            let hit = pack(values, |v| set.iter().fold(false, |a, &x| a | (x == v)));
            &pass & &hit
        }
        (Member::Small(Small::Bytes(set)), Norm::Bytes(a)) => {
            bytes_pass(a.as_ref(), &pass, |b| set.iter().any(|x| x.as_ref() == b))
        }
        (Member::Exact(Exact::Bytes(t)), Norm::Bytes(a)) => bytes_pass(a.as_ref(), &pass, |b| {
            t.find(xxh64(b), |x| x.as_ref() == b).is_some()
        }),
        (Member::Bloom(bloom), Norm::Bytes(a)) => {
            bytes_pass(a.as_ref(), &pass, |b| bloom.check_hash(xxh64(b)))
        }
        (Member::Exact(Exact::Int(t)), Norm::Int { values, .. }) => {
            hashed(keys, pass, |rows, hashes| match rows {
                None => probe_all(hashes, |i, h| t.find(h, |&x| x == values[i]).is_some()),
                Some(r) => probe_rows(r, hashes, |i, h| t.find(h, |&x| x == values[i]).is_some()),
            })
        }
        (Member::Exact(Exact::Int(t)), Norm::Float { values, single }) => {
            let probe = |i: usize, h: u64| {
                let bits = fixed64(float_bits(values[i], *single));
                t.find(h, |&x| x == bits).is_some()
            };
            hashed(keys, pass, |rows, hashes| match rows {
                None => probe_all(hashes, probe),
                Some(r) => probe_rows(r, hashes, probe),
            })
        }
        (Member::Exact(Exact::Wide(t)), Norm::Decimal(values)) => {
            hashed(keys, pass, |rows, hashes| match rows {
                None => probe_all(hashes, |i, h| t.find(h, |&x| x == values[i]).is_some()),
                Some(r) => probe_rows(r, hashes, |i, h| t.find(h, |&x| x == values[i]).is_some()),
            })
        }
        (Member::Bloom(bloom), Norm::Int { .. } | Norm::Float { .. } | Norm::Decimal(_)) => {
            hashed(keys, pass, |_, hashes| bloom.check_batch(hashes))
        }
        // The range decided; a form that does not fit the keys' class
        // keeps every row (a superset).
        _ => pass,
    }
}

/// `pass` restricted to the rows whose bytes satisfy `hit` (tested only on
/// rows still passing).
fn bytes_pass(
    a: &dyn Array,
    pass: &BooleanBuffer,
    mut hit: impl FnMut(&[u8]) -> bool,
) -> BooleanBuffer {
    let mut out = BooleanBufferBuilder::new(pass.len());
    for_each_bytes(a, |i, b| out.append(pass.value(i) && hit(b)));
    out.finish()
}

/// `probe(i, h)` of every row `i` with hash `hashes[i]`, packed.
fn probe_all(hashes: &[u64], probe: impl Fn(usize, u64) -> bool) -> BooleanBuffer {
    let mut words = Vec::with_capacity(hashes.len().div_ceil(64));
    for (c, chunk) in hashes.chunks(64).enumerate() {
        let mut w = 0u64;
        for (j, &h) in chunk.iter().enumerate() {
            w |= u64::from(probe(c * 64 + j, h)) << j;
        }
        words.push(w);
    }
    BooleanBuffer::new(Buffer::from_vec(words), 0, hashes.len())
}

/// `f(v)` of every value, packed 64 answers a word: a fixed 64-step inner
/// loop over a slice (no bounds checks, bounds of `f` held in registers),
/// which LLVM unrolls and vectorizes where the target allows.
pub(crate) fn pack<T: Copy>(values: &[T], f: impl Fn(T) -> bool) -> BooleanBuffer {
    let mut words = Vec::with_capacity(values.len().div_ceil(64));
    for chunk in values.chunks(64) {
        let mut w = 0u64;
        for (j, &v) in chunk.iter().enumerate() {
            w |= u64::from(f(v)) << j;
        }
        words.push(w);
    }
    BooleanBuffer::new(Buffer::from_vec(words), 0, values.len())
}

/// `probe(rows[k], hashes[k])` of each selected row, as a list of answers.
fn probe_rows(rows: &[u32], hashes: &[u64], probe: impl Fn(usize, u64) -> bool) -> BooleanBuffer {
    BooleanBuffer::collect_bool(rows.len(), |k| probe(rows[k] as usize, hashes[k]))
}

/// Below this fraction of rows still passing, only those rows are hashed
/// and probed (a selection vector); above it, every row is (branch-free).
const SELECT_BELOW: usize = 4;

/// Hashes the rows of `pass` once — every row when most pass, else the
/// passing rows only — and restricts `pass` to the rows `probe` accepts.
/// `probe(rows, hashes)` answers each hashed row: `rows` is `None` when
/// every row was hashed (answer `i` is row `i`), else the hashed rows.
fn hashed(
    keys: &NormalizedKeys,
    pass: BooleanBuffer,
    probe: impl FnOnce(Option<&[u32]>, &[u64]) -> BooleanBuffer,
) -> BooleanBuffer {
    let n = pass.len();
    let count = pass.count_set_bits();
    if count == 0 {
        return pass;
    }
    let mut hashes = Vec::with_capacity(n);
    if count * SELECT_BELOW >= n {
        row_hashes(keys, None, &mut hashes);
        let hit = probe(None, &hashes);
        return &pass & &hit;
    }
    let rows: Vec<u32> = pass
        .set_indices()
        .filter_map(|i| u32::try_from(i).ok())
        .collect();
    row_hashes(keys, Some(&rows), &mut hashes);
    let hit = probe(Some(&rows), &hashes);
    let mut out = BooleanBufferBuilder::new(n);
    out.append_n(n, false);
    for (k, &i) in rows.iter().enumerate() {
        if hit.value(k) {
            out.set_bit(i as usize, true);
        }
    }
    out.finish()
}
