// SPDX-License-Identifier: Apache-2.0
//! Map builders for the collection constructor and the map higher-order
//! functions: `map` packs its key and value arguments row by row (Spark's
//! `CreateMap`), and the higher-order functions rebuild maps from
//! flattened entries (see `krishiv-exec`'s `hof`).
//!
//! Every builder works on Arrow arrays directly, never one Rust value per
//! row: keys are interleaved once in row-major order, then deduplicated
//! per row over their canonical encodings (see [`super::order`]). A NULL
//! key is a `NULL_MAP_KEY` error, and a duplicate key is a
//! `DUPLICATED_MAP_KEY` error under the `EXCEPTION` policy or keeps the
//! last value under `LAST_WIN` (Spark's `ArrayBasedMapBuilder`, first-seen
//! order either way).
//!
//! Offsets and lengths cross the `i32`/`u32`/`usize` boundary with checked
//! conversions: a batch that does not fit is an engine error, never a
//! wrap.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, MapArray, StructArray};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use krishiv_common::error::{DuplicatedMapKey, KrishivError, NullMapKey};

use super::error::KernelError;
use super::order::{normalize_key, render_value};

/// Spark's `toSQLConf("spark.sql.mapKeyDedupPolicy")`.
const DEDUP_POLICY_CONF: &str = "\"spark.sql.mapKeyDedupPolicy\"";

/// What does not fit a 32-bit offset or index.
fn too_big(n: usize) -> KrishivError {
    KernelError::internal(format!("a map of {n} rows")).into()
}

/// `NULL_MAP_KEY`: a NULL key is never a value.
fn null_key() -> KrishivError {
    NullMapKey {}.into()
}

/// `DUPLICATED_MAP_KEY`: the duplicate `key`, quoting the policy.
fn duplicate_key(keys: &dyn Array, idx: usize) -> KrishivError {
    DuplicatedMapKey {
        key: render_value(keys, idx),
        map_key_dedup_policy: DEDUP_POLICY_CONF.to_owned(),
    }
    .into()
}

/// The deduplicated entries of one map's `len` keys: the kept key indices
/// and the value indices (they differ under `LAST_WIN`, where the last
/// value wins but the key keeps its first-seen position).
///
/// `keys` holds the map's keys in order (a slice of the flattened keys).
/// A NULL key is a `NULL_MAP_KEY` error; under the `EXCEPTION` policy the
/// first duplicate key is a `DUPLICATED_MAP_KEY` error naming it.
///
/// # Errors
/// `NULL_MAP_KEY`, `DUPLICATED_MAP_KEY` (under `EXCEPTION`), or an engine
/// error for a key type the analyzer never lets through.
pub fn dedup_entries(
    keys: &dyn Array,
    len: usize,
    last_win: bool,
) -> Result<(Vec<u32>, Vec<u32>), KrishivError> {
    if keys.len() < len {
        return Err(
            KernelError::internal(format!("dedup over {len} keys of {}", keys.len())).into(),
        );
    }
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::with_capacity(len);
    let mut kept: Vec<u32> = Vec::with_capacity(len);
    let mut values: Vec<u32> = Vec::with_capacity(len);
    for e in 0..len {
        let Some(code) = normalize_key(keys, e)? else {
            return Err(null_key());
        };
        match seen.get(&code) {
            None => {
                let at = u32::try_from(e).map_err(|_| too_big(e))?;
                seen.insert(code, kept.len());
                kept.push(at);
                values.push(at);
            }
            Some(&pos) => {
                if last_win {
                    values[pos] = u32::try_from(e).map_err(|_| too_big(e))?;
                } else {
                    return Err(duplicate_key(keys, e));
                }
            }
        }
    }
    Ok((kept, values))
}

/// One key of the union of two maps' key sets: the key's index into the
/// concatenated keys, and each map's value index (`None` for a key the map
/// does not hold). A map holding a key twice contributes its first entry
/// (Spark's `MapZipWith`).
pub type UnionEntry = (u32, Option<u32>, Option<u32>);

/// The union of two maps' key sets (see [`UnionEntry`]).
///
/// `keys1` and `keys2` hold each map's keys in order. A NULL key is a
/// `NULL_MAP_KEY` error.
///
/// # Errors
/// `NULL_MAP_KEY`, or an engine error for a key type the analyzer never
/// lets through.
pub fn union_keys(
    keys1: &dyn Array,
    len1: usize,
    keys2: &dyn Array,
    len2: usize,
) -> Result<Vec<UnionEntry>, KrishivError> {
    if keys1.len() < len1 || keys2.len() < len2 {
        return Err(KernelError::internal(format!(
            "zip over {len1} and {len2} keys of {} and {}",
            keys1.len(),
            keys2.len()
        ))
        .into());
    }
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut out: Vec<UnionEntry> = Vec::new();
    let push = |code: Vec<u8>,
                key: u32,
                value: Option<u32>,
                first: bool,
                seen: &mut HashMap<Vec<u8>, usize>,
                out: &mut Vec<UnionEntry>| {
        match seen.get(&code) {
            None => {
                seen.insert(code, out.len());
                out.push(if first {
                    (key, value, None)
                } else {
                    (key, None, value)
                });
            }
            Some(&pos) => {
                let slot = if first {
                    &mut out[pos].1
                } else {
                    &mut out[pos].2
                };
                if slot.is_none() {
                    *slot = value;
                }
            }
        }
    };
    for e in 0..len1 {
        let Some(code) = normalize_key(keys1, e)? else {
            return Err(null_key());
        };
        let at = u32::try_from(e).map_err(|_| too_big(e))?;
        push(code, at, Some(at), true, &mut seen, &mut out);
    }
    for e in 0..len2 {
        let Some(code) = normalize_key(keys2, e)? else {
            return Err(null_key());
        };
        // The key index counts into the concatenated keys, but the value
        // index counts into the second values alone.
        let key = u32::try_from(len1.checked_add(e).ok_or_else(|| too_big(e))?)
            .map_err(|_| too_big(e))?;
        let at = u32::try_from(e).map_err(|_| too_big(e))?;
        push(code, key, Some(at), false, &mut seen, &mut out);
    }
    Ok(out)
}

/// `map(k0, v0, ...)`: one map per row over the interleaved keys and
/// values (`[k0[r], k1[r], ...]`), deduplicated per row under `last_win`
/// (Spark's `CreateMap`). `keys[i]` and `values[i]` hold `rows` values
/// each; an empty pair list is an empty map per row.
///
/// # Errors
/// `NULL_MAP_KEY`, `DUPLICATED_MAP_KEY` (unless `last_win`), or an engine
/// error when the arguments' lengths disagree with `rows`.
pub fn create(
    keys: &[ArrayRef],
    values: &[ArrayRef],
    rows: usize,
    value_nullable: bool,
    last_win: bool,
) -> Result<MapArray, KrishivError> {
    for a in keys.iter().chain(values.iter()) {
        if a.len() != rows {
            return Err(KernelError::internal(format!(
                "map of {rows} rows over an argument of {} rows",
                a.len()
            ))
            .into());
        }
    }
    let pairs = keys.len();
    if pairs == 0 {
        let keys = arrow_array::new_empty_array(&DataType::Null);
        let values = arrow_array::new_empty_array(&DataType::Null);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![
            0_i32;
            rows.checked_add(1).ok_or_else(
                || too_big(rows)
            )?
        ]));
        return build_map(&keys, &values, offsets, None, value_nullable);
    }
    // Row `r` takes element `r` of each key and value argument.
    let width = rows.checked_mul(pairs).ok_or_else(|| too_big(rows))?;
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(width);
    for r in 0..rows {
        for c in 0..pairs {
            indices.push((c, r));
        }
    }
    let key_refs: Vec<&dyn Array> = keys.iter().map(AsRef::as_ref).collect();
    let value_refs: Vec<&dyn Array> = values.iter().map(AsRef::as_ref).collect();
    let flat_keys = arrow_select::interleave::interleave(&key_refs, &indices)
        .map_err(|e| KernelError::internal(e.to_string()))?;
    let flat_values = arrow_select::interleave::interleave(&value_refs, &indices)
        .map_err(|e| KernelError::internal(e.to_string()))?;
    let mut kept: Vec<u32> = Vec::with_capacity(width);
    let mut taken_values: Vec<u32> = Vec::with_capacity(width);
    let mut offsets = Vec::with_capacity(rows.checked_add(1).ok_or_else(|| too_big(rows))?);
    offsets.push(0_i32);
    for r in 0..rows {
        let start = r.checked_mul(pairs).ok_or_else(|| too_big(r))?;
        let slice = flat_keys.slice(start, pairs);
        let (mut k, mut v) = dedup_entries(slice.as_ref(), pairs, last_win)?;
        let base = u32::try_from(start).map_err(|_| too_big(start))?;
        for i in &mut k {
            *i = i.checked_add(base).ok_or_else(|| too_big(start))?;
        }
        for i in &mut v {
            *i = i.checked_add(base).ok_or_else(|| too_big(start))?;
        }
        kept.extend_from_slice(&k);
        taken_values.extend_from_slice(&v);
        offsets.push(i32::try_from(kept.len()).map_err(|_| too_big(kept.len()))?);
    }
    let keys = arrow_select::take::take(
        flat_keys.as_ref(),
        &arrow_array::UInt32Array::from(kept),
        None,
    )
    .map_err(|e| KernelError::internal(e.to_string()))?;
    let values = arrow_select::take::take(
        flat_values.as_ref(),
        &arrow_array::UInt32Array::from(taken_values),
        None,
    )
    .map_err(|e| KernelError::internal(e.to_string()))?;
    build_map(
        &keys,
        &values,
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        None,
        value_nullable,
    )
}

/// The `entries` field of a map of `key_type` to `value_type`.
pub fn entries_field(key_type: &DataType, value_type: &DataType, value_nullable: bool) -> Field {
    Field::new(
        "entries",
        DataType::Struct(
            vec![
                Field::new("key", key_type.clone(), false),
                Field::new("value", value_type.clone(), value_nullable),
            ]
            .into(),
        ),
        false,
    )
}

/// A map over `keys` and `values` with `offsets` (one entry per element)
/// and the row `nulls`: the entries struct carries non-null keys and
/// `value_nullable` values, as the Arrow mapping reads maps.
///
/// # Errors
/// [`KernelError`] when the lengths disagree.
pub fn build_map(
    keys: &ArrayRef,
    values: &ArrayRef,
    offsets: OffsetBuffer<i32>,
    nulls: Option<NullBuffer>,
    value_nullable: bool,
) -> Result<MapArray, KrishivError> {
    let entries = StructArray::new(
        vec![
            Arc::new(Field::new("key", keys.data_type().clone(), false)),
            Arc::new(Field::new(
                "value",
                values.data_type().clone(),
                value_nullable,
            )),
        ]
        .into(),
        vec![Arc::clone(keys), Arc::clone(values)],
        None,
    );
    if keys.len() != values.len() {
        return Err(KernelError::internal(format!(
            "a map over {} keys and {} values",
            keys.len(),
            values.len()
        ))
        .into());
    }
    MapArray::try_new(
        Arc::new(entries_field(
            keys.data_type(),
            values.data_type(),
            value_nullable,
        )),
        offsets,
        entries,
        nulls,
        false,
    )
    .map_err(|e| KernelError::internal(e.to_string()).into())
}

#[cfg(test)]
mod tests;
