// SPDX-License-Identifier: Apache-2.0
//! A filter's test of listed rows only ([`KeyFilter::retain_keys_at`](super::KeyFilter::retain_keys_at)):
//! the row stage's later filters, once most rows of a batch are gone, test
//! the survivors' row ids over the whole normalized column instead of a
//! filtered copy of it, and drop the failing ids from the list in place
//! (design P3 0.3.1 O1; DuckDB's selection vectors). A row is kept exactly
//! when [`evaluate`](super::member::evaluate) passes it: the same range,
//! validity and row form, the hashed forms hashing and probing only the
//! listed rows still passing the range.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;

use super::bloom::xxh64;
use super::hash::row_hashes;
use super::key::{KeyRange, Norm, NormalizedKeys, float_bits};
use super::member::{Exact, Member, Small, fixed64};

/// `rows` (ids into `keys`, any order) restricted to the rows
/// [`super::member::evaluate`] passes, order kept. An id past the keys'
/// end is dropped.
pub(crate) fn retain_at(
    range: Option<&KeyRange>,
    member: &Member,
    keys: &NormalizedKeys,
    valid: Option<&BooleanBuffer>,
    rows: &mut Vec<u32>,
) {
    let len = keys.len();
    let ok = |i: usize| i < len && valid.is_none_or(|v| v.value(i));
    if let Norm::Bytes(a) = &keys.norm {
        bytes_at(range, member, a.as_ref(), rows, ok);
        return;
    }
    // Range and validity first (two compares a row), so the row form sees
    // only the rows still passing.
    match (&keys.norm, range) {
        (Norm::Int { values, .. }, Some(&KeyRange::Int(lo, hi))) => rows.retain(|&r| {
            let i = r as usize;
            ok(i) && values.get(i).is_some_and(|&v| v >= lo && v <= hi)
        }),
        (Norm::Decimal(values), Some(&KeyRange::Decimal(lo, hi))) => rows.retain(|&r| {
            let i = r as usize;
            ok(i) && values.get(i).is_some_and(|&v| v >= lo && v <= hi)
        }),
        // NaN fails both compares.
        (Norm::Float { values, .. }, Some(&KeyRange::Float(lo, hi))) => rows.retain(|&r| {
            let i = r as usize;
            ok(i) && values.get(i).is_some_and(|&v| v >= lo && v <= hi)
        }),
        _ => rows.retain(|&r| ok(r as usize)),
    }
    // Every id left is inside the keys.
    match (member, &keys.norm) {
        (Member::Small(Small::Int(set)), Norm::Int { values, .. }) => {
            rows.retain(|&r| values.get(r as usize).is_some_and(|v| set.contains(v)));
        }
        (Member::Small(Small::Int(set)), Norm::Float { values, single }) => rows.retain(|&r| {
            values
                .get(r as usize)
                .is_some_and(|&v| set.contains(&fixed64(float_bits(v, *single))))
        }),
        (Member::Small(Small::Wide(set)), Norm::Decimal(values)) => {
            rows.retain(|&r| values.get(r as usize).is_some_and(|v| set.contains(v)));
        }
        (Member::Exact(Exact::Dense(d)), Norm::Int { values, .. }) => {
            rows.retain(|&r| values.get(r as usize).is_some_and(|&v| d.contains(v)));
        }
        (Member::Exact(Exact::Int(t)), Norm::Int { values, .. }) => {
            hashed_at(keys, rows, |i, h| {
                values
                    .get(i)
                    .is_some_and(|&v| t.find(h, |&x| x == v).is_some())
            });
        }
        (Member::Exact(Exact::Int(t)), Norm::Float { values, single }) => {
            hashed_at(keys, rows, |i, h| {
                values.get(i).is_some_and(|&v| {
                    let bits = fixed64(float_bits(v, *single));
                    t.find(h, |&x| x == bits).is_some()
                })
            });
        }
        (Member::Exact(Exact::Wide(t)), Norm::Decimal(values)) => {
            hashed_at(keys, rows, |i, h| {
                values
                    .get(i)
                    .is_some_and(|&v| t.find(h, |&x| x == v).is_some())
            });
        }
        (Member::Bloom(bloom), Norm::Int { .. } | Norm::Float { .. } | Norm::Decimal(_)) => {
            let mut hashes = Vec::with_capacity(rows.len());
            row_hashes(keys, Some(rows), &mut hashes);
            let hit = bloom.check_batch(&hashes);
            retain_by(rows, |k| k < hit.len() && hit.value(k));
        }
        // The range decided; a form that does not fit the keys' class keeps
        // every row (a superset), as `member_pass` does.
        _ => {}
    }
}

/// `rows` restricted to the positions `keep(k)` accepts, order kept.
fn retain_by(rows: &mut Vec<u32>, keep: impl Fn(usize) -> bool) {
    let mut k = 0;
    rows.retain(|_| {
        let kept = keep(k);
        k += 1;
        kept
    });
}

/// The hashed exact forms: the listed rows hashed once, then kept where
/// `probe(row, hash)` holds.
fn hashed_at(keys: &NormalizedKeys, rows: &mut Vec<u32>, probe: impl Fn(usize, u64) -> bool) {
    let mut hashes = Vec::with_capacity(rows.len());
    row_hashes(keys, Some(rows), &mut hashes);
    let mut k = 0;
    rows.retain(|&r| {
        let kept = hashes.get(k).is_some_and(|&h| probe(r as usize, h));
        k += 1;
        kept
    });
}

/// [`retain_at`] for string and binary keys: each listed row's bytes read
/// in place (no copy), range then the row form; a Bloom probes the rows
/// passing the range as one batch (`check_batch` prefetches).
fn bytes_at(
    range: Option<&KeyRange>,
    member: &Member,
    a: &dyn Array,
    rows: &mut Vec<u32>,
    ok: impl Fn(usize) -> bool,
) {
    let bounds = match range {
        Some(KeyRange::Bytes(lo, hi)) => Some((lo.as_ref(), hi.as_ref())),
        _ => None,
    };
    let in_range = |i: usize, b: &[u8]| ok(i) && bounds.is_none_or(|(lo, hi)| b >= lo && b <= hi);
    if let Member::Bloom(bloom) = member {
        let mut hashes = Vec::with_capacity(rows.len());
        retain_bytes(a, rows, |i, b| {
            let kept = in_range(i, b);
            if kept {
                hashes.push(xxh64(b));
            }
            kept
        });
        let hit = bloom.check_batch(&hashes);
        retain_by(rows, |k| k < hit.len() && hit.value(k));
        return;
    }
    retain_bytes(a, rows, |i, b| {
        in_range(i, b)
            && match member {
                Member::Small(Small::Bytes(set)) => set.iter().any(|x| x.as_ref() == b),
                Member::Exact(Exact::Bytes(t)) => t.find(xxh64(b), |x| x.as_ref() == b).is_some(),
                // The range decided; a form of another class keeps the row.
                _ => true,
            }
    });
}

/// `rows` restricted to the rows inside the array for which `keep(row,
/// bytes)` holds, with each row's string or binary bytes (a NULL slot's
/// placeholder bytes too: the caller checks validity). Order kept.
fn retain_bytes(a: &dyn Array, rows: &mut Vec<u32>, mut keep: impl FnMut(usize, &[u8]) -> bool) {
    let len = a.len();
    macro_rules! each {
        ($get:expr) => {{
            let get = $get;
            rows.retain(|&r| {
                let i = r as usize;
                i < len && keep(i, get(i))
            });
        }};
    }
    match a.data_type() {
        DataType::Utf8 => {
            let s = a.as_string::<i32>();
            each!(|i| s.value(i).as_bytes());
        }
        DataType::LargeUtf8 => {
            let s = a.as_string::<i64>();
            each!(|i| s.value(i).as_bytes());
        }
        DataType::Utf8View => {
            let s = a.as_string_view();
            each!(|i| s.value(i).as_bytes());
        }
        DataType::Binary => {
            let s = a.as_binary::<i32>();
            each!(|i| s.value(i));
        }
        DataType::LargeBinary => {
            let s = a.as_binary::<i64>();
            each!(|i| s.value(i));
        }
        DataType::BinaryView => {
            let s = a.as_binary_view();
            each!(|i| s.value(i));
        }
        // Not a string or binary array: no row is a key of this class.
        _ => rows.clear(),
    }
}
