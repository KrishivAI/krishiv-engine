// SPDX-License-Identifier: Apache-2.0
//! A filter's test of listed rows only ([`KeyFilter::evaluate_keys_at`](super::KeyFilter::evaluate_keys_at)):
//! the row stage's later filters, once most rows of a batch are gone, test
//! the survivors' row ids over the whole normalized column instead of a
//! filtered copy of it (design P3 0.3.1 O1; DuckDB's selection vectors).
//! Each answer is the one [`evaluate`](super::member::evaluate) gives that
//! row: the same range, validity and row form, the hashed forms hashing and
//! probing the listed rows still passing the range only.

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_schema::DataType;

use super::bloom::xxh64;
use super::hash::row_hashes;
use super::key::{KeyRange, Norm, NormalizedKeys, float_bits};
use super::member::{Exact, Member, Small, fixed64};

/// [`super::member::evaluate`] of the rows `rows` (ids into `keys`): answer
/// `k` is row `rows[k]`'s. A row id past the keys' end is `false`.
pub(crate) fn evaluate_at(
    range: Option<&KeyRange>,
    member: &Member,
    keys: &NormalizedKeys,
    valid: Option<&BooleanBuffer>,
    rows: &[u32],
) -> BooleanBuffer {
    let len = keys.len();
    let ok = |i: usize| i < len && valid.is_none_or(|v| v.value(i));
    if let Norm::Bytes(a) = &keys.norm {
        return bytes_at(range, member, a.as_ref(), rows, ok);
    }
    // Range and validity, row by row (the range is two compares).
    let pre = BooleanBuffer::collect_bool(rows.len(), |k| {
        let i = rows[k] as usize;
        ok(i) && in_range(range, &keys.norm, i)
    });
    match (member, &keys.norm) {
        (Member::Small(Small::Int(set)), Norm::Int { values, .. }) => {
            select(&pre, rows, |i| set.contains(&values[i]))
        }
        (Member::Small(Small::Int(set)), Norm::Float { values, single }) => {
            select(&pre, rows, |i| {
                set.contains(&fixed64(float_bits(values[i], *single)))
            })
        }
        (Member::Small(Small::Wide(set)), Norm::Decimal(values)) => {
            select(&pre, rows, |i| set.contains(&values[i]))
        }
        (Member::Exact(Exact::Dense(d)), Norm::Int { values, .. }) => {
            select(&pre, rows, |i| d.contains(values[i]))
        }
        (Member::Exact(Exact::Int(t)), Norm::Int { values, .. }) => {
            hashed_at(keys, &pre, rows, |i, h| {
                t.find(h, |&x| x == values[i]).is_some()
            })
        }
        (Member::Exact(Exact::Int(t)), Norm::Float { values, single }) => {
            hashed_at(keys, &pre, rows, |i, h| {
                let bits = fixed64(float_bits(values[i], *single));
                t.find(h, |&x| x == bits).is_some()
            })
        }
        (Member::Exact(Exact::Wide(t)), Norm::Decimal(values)) => {
            hashed_at(keys, &pre, rows, |i, h| {
                t.find(h, |&x| x == values[i]).is_some()
            })
        }
        (Member::Bloom(bloom), Norm::Int { .. } | Norm::Float { .. } | Norm::Decimal(_)) => {
            let (at, ids) = candidates(&pre, rows);
            let mut hashes = Vec::with_capacity(ids.len());
            row_hashes(keys, Some(&ids), &mut hashes);
            let hit = bloom.check_batch(&hashes);
            spread(rows.len(), &at, |c| hit.value(c))
        }
        // The range decided; a form that does not fit the keys' class keeps
        // every row (a superset), as `member_pass` does.
        _ => pre,
    }
}

/// Whether row `i` is in `range` (every row without one; NaN fails).
fn in_range(range: Option<&KeyRange>, norm: &Norm, i: usize) -> bool {
    match (norm, range) {
        (Norm::Int { values, .. }, Some(&KeyRange::Int(lo, hi))) => {
            values.get(i).is_some_and(|&v| v >= lo && v <= hi)
        }
        (Norm::Decimal(values), Some(&KeyRange::Decimal(lo, hi))) => {
            values.get(i).is_some_and(|&v| v >= lo && v <= hi)
        }
        (Norm::Float { values, .. }, Some(&KeyRange::Float(lo, hi))) => {
            values.get(i).is_some_and(|&v| v >= lo && v <= hi)
        }
        _ => true,
    }
}

/// `pre` restricted to the listed rows `hit` accepts (called only on rows
/// `pre` passes, so on ids inside the keys).
fn select(pre: &BooleanBuffer, rows: &[u32], hit: impl Fn(usize) -> bool) -> BooleanBuffer {
    BooleanBuffer::collect_bool(rows.len(), |k| pre.value(k) && hit(rows[k] as usize))
}

/// The positions in `rows` that `pre` passes, and their row ids.
fn candidates(pre: &BooleanBuffer, rows: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let at: Vec<usize> = pre.set_indices().collect();
    let ids = at.iter().map(|&k| rows[k]).collect();
    (at, ids)
}

/// Answers of `n` listed rows: position `at[c]` takes `hit(c)`, the rest
/// `false`.
fn spread(n: usize, at: &[usize], hit: impl Fn(usize) -> bool) -> BooleanBuffer {
    let mut out = BooleanBufferBuilder::new(n);
    out.append_n(n, false);
    for (c, &k) in at.iter().enumerate() {
        if hit(c) {
            out.set_bit(k, true);
        }
    }
    out.finish()
}

/// The hashed exact forms: the listed rows `pre` passes hashed once, then
/// `probe(row, hash)` of each.
fn hashed_at(
    keys: &NormalizedKeys,
    pre: &BooleanBuffer,
    rows: &[u32],
    probe: impl Fn(usize, u64) -> bool,
) -> BooleanBuffer {
    let (at, ids) = candidates(pre, rows);
    let mut hashes = Vec::with_capacity(ids.len());
    row_hashes(keys, Some(&ids), &mut hashes);
    spread(rows.len(), &at, |c| probe(ids[c] as usize, hashes[c]))
}

/// [`evaluate_at`] for string and binary keys: each listed row's bytes read
/// in place (no copy), range then the row form; a Bloom probes the rows
/// passing the range as one batch (`check_batch` prefetches).
fn bytes_at(
    range: Option<&KeyRange>,
    member: &Member,
    a: &dyn Array,
    rows: &[u32],
    ok: impl Fn(usize) -> bool,
) -> BooleanBuffer {
    let bounds = match range {
        Some(KeyRange::Bytes(lo, hi)) => Some((lo.as_ref(), hi.as_ref())),
        _ => None,
    };
    let n = rows.len();
    let mut pass = vec![false; n];
    let mut at = Vec::new();
    let mut hashes = Vec::new();
    let bloom = match member {
        Member::Bloom(b) => Some(b),
        _ => None,
    };
    rows_bytes(a, rows, |k, i, b| {
        if !ok(i) || bounds.is_some_and(|(lo, hi)| b < lo || b > hi) {
            return;
        }
        if bloom.is_some() {
            at.push(k);
            hashes.push(xxh64(b));
            return;
        }
        let hit = match member {
            Member::Small(Small::Bytes(set)) => set.iter().any(|x| x.as_ref() == b),
            Member::Exact(Exact::Bytes(t)) => t.find(xxh64(b), |x| x.as_ref() == b).is_some(),
            // The range decided; a form of another class keeps the row.
            _ => true,
        };
        if let Some(p) = pass.get_mut(k) {
            *p = hit;
        }
    });
    if let Some(bloom) = bloom {
        let hit = bloom.check_batch(&hashes);
        return spread(n, &at, |c| hit.value(c));
    }
    BooleanBuffer::collect_bool(n, |k| pass[k])
}

/// Calls `f(k, rows[k], bytes)` for each listed row inside the array, with
/// its string or binary bytes (a NULL slot's placeholder bytes too: the
/// caller checks validity).
fn rows_bytes(a: &dyn Array, rows: &[u32], mut f: impl FnMut(usize, usize, &[u8])) {
    let len = a.len();
    macro_rules! each {
        ($get:expr) => {{
            let get = $get;
            for (k, &r) in rows.iter().enumerate() {
                let i = r as usize;
                if i < len {
                    f(k, i, get(i));
                }
            }
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
        _ => {}
    }
}
