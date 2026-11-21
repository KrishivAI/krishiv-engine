// SPDX-License-Identifier: Apache-2.0
//! `theta_sketch_agg`, `theta_union_agg`, `theta_intersection_agg` and the `theta_*` scalars, after Spark's
//! `thetasketchesAggregates.scala`, `thetasketchesExpressions.scala` and `ThetaSketchUtils.scala` (v4.3.0, since 4.1.0).
//!
//! Update encodings follow `ThetaSketchAgg.update`: `INT` is widened to `long`, `FLOAT` promoted to `double`
//! (signed zeros and NaNs canonical), `STRING` and `BINARY` hash their raw bytes with the empty value skipped,
//! and an array of `INT` or `BIGINT` hashes its elements' little-endian bytes (an empty array is skipped).
//! Sketches are written compressed, as Spark's `toByteArrayCompressed`.
//!
//! Known difference: the crate's `CompactThetaSketch::deserialize` refuses a non-empty sketch built with another
//! seed; Spark's `CompactSketch.wrap(memory)` does not check the seed, so `theta_sketch_estimate` of such a sketch
//! works there.
//!
//! `lgNomEntries` 4 (legal in Spark, `4..=26`): the crate builds nothing below 5, so such a sketch or union is built
//! at 5 and every result is cut by [`finish`] to the 16 smallest hashes with theta the 17th.
//!
//! Profiles: the empty-state rule of `theta_intersection_agg` (a group with no sketch) is new in Spark 4.3.0; this
//! module has one behaviour for every profile until the 4.2.1 and 4.1.4 oracles are run (see the PR notes).

use datasketches::hash::value::{canonical_float, raw_bytes};
pub use datasketches::theta::{CompactThetaSketch, ThetaIntersection, ThetaSketch, ThetaUnion};
use datasketches::theta::{ThetaSketchBuilder, ThetaUnionBuilder};
use krishiv_common::error::{
    InternalError, KrishivError, SketchInvalidLgNomEntries, ThetaInvalidInputSketchBuffer,
    ThetaInvalidInputSketchBufferFamily,
};

/// Smallest `lgNomEntries` Spark accepts (`ThetaSketchUtils.MIN_LG_NOM_LONGS`).
pub const MIN_LG_NOM: i32 = 4;
/// Largest `lgNomEntries` Spark accepts.
pub const MAX_LG_NOM: i32 = 26;
/// `lgNomEntries` when the argument is omitted.
pub const DEFAULT_LG_NOM: i32 = 12;

/// The `COMPACT` family id in a DataSketches preamble (byte 2).
const FAMILY_COMPACT: u8 = 3;

/// Check `lg` is in `4..=26`, else `SKETCH_INVALID_LG_NOM_ENTRIES` naming `function`.
///
/// # Errors
/// `SKETCH_INVALID_LG_NOM_ENTRIES` for a value outside the range.
pub fn check_lg_nom(lg: i32, function: &str) -> Result<u8, KrishivError> {
    u8::try_from(lg)
        .ok()
        .filter(|k| (MIN_LG_NOM..=MAX_LG_NOM).contains(&i32::from(*k)))
        .ok_or_else(|| {
            SketchInvalidLgNomEntries {
                function: format!("`{function}`"),
                min: MIN_LG_NOM.to_string(),
                max: MAX_LG_NOM.to_string(),
                value: lg.to_string(),
            }
            .into()
        })
}

fn invalid_buffer(function: &str) -> KrishivError {
    ThetaInvalidInputSketchBuffer {
        function: format!("`{function}`"),
    }
    .into()
}

/// Smallest `lg_k` the `datasketches` crate builds. Spark accepts 4: such a sketch is built at 5 and cut back to
/// Spark's size by [`finish`] (which holds the details of the cut).
const CRATE_MIN_LG_K: u8 = 5;

/// The `lg_k` the crate is asked for: `lg_nom`, but at least the crate's minimum.
fn build_lg(k: u8) -> u8 {
    k.max(CRATE_MIN_LG_K)
}

/// Cut a compact sketch to what a sketch of `lg_nom` keeps. Only `lg_nom` 4 differs from what the crate built
/// (at 5): the sketch keeps the 16 smallest hashes and its theta is the 17th smallest, which is the state Java's
/// quick-select rebuild leaves at `lgNomEntries` 4. A sketch of 16 hashes or fewer is returned as it is.
///
/// # Errors
/// An internal error if the cut sketch's bytes do not read back (an invariant of this function, not user input).
pub fn finish(sketch: CompactThetaSketch, lg_nom: u8) -> Result<CompactThetaSketch, KrishivError> {
    const K: usize = 16;
    if lg_nom >= CRATE_MIN_LG_K || sketch.num_retained() <= K {
        return Ok(sketch);
    }
    let mut hashes: Vec<u64> = sketch.iter().map(|e| e.hash()).collect();
    hashes.sort_unstable();
    let theta = hashes[K];
    let original = sketch.serialize();
    // Preamble of 8 bytes: preLongs, version, family, 2 unused, flags, seed hash.
    let mut out = original[..8].to_vec();
    out[0] = 3;
    out[5] |= 0x10; // ordered
    out.extend_from_slice(&16u32.to_le_bytes()); // retained entries
    out.extend_from_slice(&1.0f32.to_le_bytes()); // the float p of preamble long 1, as Java writes it
    out.extend_from_slice(&theta.to_le_bytes());
    for h in &hashes[..K] {
        out.extend_from_slice(&h.to_le_bytes());
    }
    CompactThetaSketch::deserialize(&out).map_err(|e| {
        InternalError {
            message: format!("a Theta sketch cut to lgNomEntries 4 does not read back: {e}"),
        }
        .into()
    })
}

/// A sketch builder refused parameters this module already checked: an engine fault, not a user buffer.
fn builder_failure(function: &str, e: &dyn std::fmt::Display) -> KrishivError {
    InternalError {
        message: format!("`{function}`: the Theta sketch builder failed: {e}"),
    }
    .into()
}

/// Spark's `Family.toString` for a preamble family id (the names of `org.apache.datasketches.common.Family`).
fn family_name(id: u8) -> Option<&'static str> {
    Some(match id {
        1 => "ALPHA",
        2 => "QUICKSELECT",
        3 => "COMPACT",
        4 => "UNION",
        5 => "INTERSECTION",
        6 => "ANOTB",
        7 => "HLL",
        8 => "QUANTILES",
        9 => "TUPLE",
        10 => "FREQUENCY",
        11 => "RESERVOIR",
        12 => "RESERVOIR_UNION",
        13 => "VAROPT",
        14 => "VAROPT_UNION",
        15 => "KLL",
        16 => "CPC",
        17 => "REQ",
        18 => "COUNTMIN",
        19 => "EBPPS",
        20 => "TDIGEST",
        21 => "BLOOMFILTER",
        _ => return None,
    })
}

/// A new updatable sketch of `lg_nom` (already checked).
///
/// # Errors
/// `SKETCH_INVALID_LG_NOM_ENTRIES` when `lg_nom` is out of range. A 4 is built at 5 and cut by [`finish`].
pub fn new_sketch(lg_nom: i32, function: &str) -> Result<ThetaSketch, KrishivError> {
    let k = check_lg_nom(lg_nom, function)?;
    ThetaSketchBuilder::default()
        .lg_k(build_lg(k))
        .build()
        .map_err(|e| builder_failure(function, &e))
}

/// A new union of `lg_nom`.
///
/// # Errors
/// `SKETCH_INVALID_LG_NOM_ENTRIES` when `lg_nom` is out of range. A 4 is built at 5 and cut by [`finish`].
pub fn new_union(lg_nom: i32, function: &str) -> Result<ThetaUnion, KrishivError> {
    let k = check_lg_nom(lg_nom, function)?;
    ThetaUnionBuilder::default()
        .lg_k(build_lg(k))
        .build()
        .map_err(|e| builder_failure(function, &e))
}

/// A new intersection (Spark builds it with the default seed and no size).
///
#[must_use]
pub fn new_intersection() -> ThetaIntersection {
    ThetaIntersection::default()
}

/// `INT` and `BIGINT` inputs: both hash as a `long`.
pub fn update_long(sketch: &mut ThetaSketch, value: i64) {
    sketch.update(value);
}

/// `DOUBLE` input (`FLOAT` is promoted first).
pub fn update_double(sketch: &mut ThetaSketch, value: f64) {
    sketch.update(canonical_float::from_f64(value));
}

/// `STRING` input: the UTF-8 bytes; the empty string is ignored.
pub fn update_str(sketch: &mut ThetaSketch, value: &str) {
    if !value.is_empty() {
        sketch.update(raw_bytes::from_str(value));
    }
}

/// `BINARY` input: the raw bytes; the empty array is ignored.
pub fn update_bytes(sketch: &mut ThetaSketch, value: &[u8]) {
    if !value.is_empty() {
        sketch.update(raw_bytes::from_slice(value));
    }
}

/// `ARRAY<INT>` input: the elements as little-endian 4-byte words; an empty array is ignored.
pub fn update_ints(sketch: &mut ThetaSketch, values: &[i32]) {
    if !values.is_empty() {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        sketch.update(raw_bytes::from_vec(bytes));
    }
}

/// `ARRAY<BIGINT>` input: the elements as little-endian 8-byte words; an empty array is ignored.
pub fn update_longs(sketch: &mut ThetaSketch, values: &[i64]) {
    if !values.is_empty() {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        sketch.update(raw_bytes::from_vec(bytes));
    }
}

/// The bytes Krishiv stores in a `BINARY` column: the compact, compressed form (Spark's `toByteArrayCompressed`).
#[must_use]
pub fn to_bytes(sketch: &CompactThetaSketch) -> Vec<u8> {
    sketch.serialize_compressed()
}

/// The compact sketch of an updatable one, as `sketch.rebuild.compact` (Spark trims to the nominal size first).
#[must_use]
pub fn compact(sketch: &mut ThetaSketch) -> CompactThetaSketch {
    sketch.trim();
    sketch.compact(true)
}

/// Read a compact sketch buffer, as `ThetaSketchUtils.wrapCompactSketch` does for `function`.
///
/// # Errors
/// `THETA_INVALID_INPUT_SKETCH_BUFFER` for bytes that are no sketch, `THETA_INVALID_INPUT_SKETCH_BUFFER_FAMILY`
/// for a sketch of another family.
pub fn from_bytes(bytes: &[u8], function: &str) -> Result<CompactThetaSketch, KrishivError> {
    let Some(&family) = bytes.get(2) else {
        return Err(invalid_buffer(function));
    };
    let Some(name) = family_name(family) else {
        return Err(invalid_buffer(function));
    };
    if family != FAMILY_COMPACT {
        return Err(ThetaInvalidInputSketchBufferFamily {
            function: format!("`{function}`"),
            expected_family: "COMPACT".to_owned(),
            actual_family: name.to_owned(),
        }
        .into());
    }
    CompactThetaSketch::deserialize(bytes).map_err(|_| invalid_buffer(function))
}

/// `theta_sketch_estimate`: `Math.round(getEstimate)`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "an estimate is far below i64::MAX"
)]
pub fn estimate(sketch: &CompactThetaSketch) -> i64 {
    sketch.estimate().round() as i64
}

/// `theta_union(a, b, lgNomEntries)`.
///
/// # Errors
/// `SKETCH_INVALID_LG_NOM_ENTRIES` for an out-of-range size.
pub fn union(
    a: &CompactThetaSketch,
    b: &CompactThetaSketch,
    lg_nom: i32,
    function: &str,
) -> Result<CompactThetaSketch, KrishivError> {
    let mut u = new_union(lg_nom, function)?;
    u.update(a).map_err(|_| invalid_buffer(function))?;
    u.update(b).map_err(|_| invalid_buffer(function))?;
    finish(
        u.to_sketch(true),
        u8::try_from(lg_nom).unwrap_or(CRATE_MIN_LG_K),
    )
}

/// `theta_intersection(a, b)`.
///
/// # Errors
/// An invalid-buffer error for sketches the library cannot combine (a different seed).
pub fn intersection(
    a: &CompactThetaSketch,
    b: &CompactThetaSketch,
    function: &str,
) -> Result<CompactThetaSketch, KrishivError> {
    let mut i = new_intersection();
    i.update(a).map_err(|_| invalid_buffer(function))?;
    i.update(b).map_err(|_| invalid_buffer(function))?;
    i.to_sketch(true).ok_or_else(|| invalid_buffer(function))
}

/// `theta_difference(a, b)`: the entries of `a` not in `b`.
///
/// # Errors
/// An invalid-buffer error for sketches the library cannot combine (a different seed).
pub fn difference(
    a: &CompactThetaSketch,
    b: &CompactThetaSketch,
    function: &str,
) -> Result<CompactThetaSketch, KrishivError> {
    datasketches::theta::ThetaANotB::default()
        .compute(a, b, true)
        .map_err(|_| invalid_buffer(function))
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: &str = "theta_sketch_estimate";

    fn of(values: &[i64]) -> CompactThetaSketch {
        let mut s = new_sketch(DEFAULT_LG_NOM, "theta_sketch_agg").unwrap();
        for v in values {
            update_long(&mut s, *v);
        }
        compact(&mut s)
    }

    #[test]
    fn finish_cuts_lg_4_to_sixteen_hashes_with_theta_the_next() {
        let at5 = |n: i64| {
            let mut s = new_sketch(5, "f").unwrap();
            for v in 0..n {
                update_long(&mut s, v);
            }
            compact(&mut s)
        };
        // 16 or fewer hashes: untouched.
        for n in [1, 16] {
            let c = at5(n);
            let kept = c.num_retained();
            assert_eq!(finish(c, 4).unwrap().num_retained(), kept);
        }
        // More than 16: the 16 smallest, theta the 17th smallest.
        for n in [17, 1000] {
            let c = at5(n);
            let mut all: Vec<u64> = c.iter().map(|e| e.hash()).collect();
            all.sort_unstable();
            let cut = finish(c, 4).unwrap();
            assert_eq!(cut.num_retained(), 16);
            let mut kept: Vec<u64> = cut.iter().map(|e| e.hash()).collect();
            kept.sort_unstable();
            assert_eq!(kept, all[..16]);
            assert_eq!(cut.theta64(), all[16], "theta is the 17th smallest hash");
        }
        // A union of two estimation-mode inputs with different thetas.
        let u = union(&at5(5000), &at5(300), 4, "theta_union").unwrap();
        assert_eq!(u.num_retained(), 16);
    }

    #[test]
    fn lg_nom_bounds() {
        assert!(check_lg_nom(3, "f").is_err());
        assert_eq!(check_lg_nom(4, "f").unwrap(), 4);
        assert_eq!(check_lg_nom(26, "f").unwrap(), 26);
        assert!(check_lg_nom(27, "f").is_err());
        assert!(check_lg_nom(-1, "f").is_err());
    }

    #[test]
    fn small_inputs_count_exactly_and_empties_are_skipped() {
        assert_eq!(estimate(&of(&[1, 2, 2, 3])), 3);
        let mut s = new_sketch(12, "f").unwrap();
        update_str(&mut s, "");
        update_bytes(&mut s, b"");
        update_ints(&mut s, &[]);
        update_longs(&mut s, &[]);
        assert_eq!(estimate(&compact(&mut s)), 0);
        update_str(&mut s, "a");
        update_bytes(&mut s, b"a");
        assert_eq!(
            estimate(&compact(&mut s)),
            1,
            "a string and the same bytes hash alike"
        );
        update_double(&mut s, 0.0);
        update_double(&mut s, -0.0);
        assert_eq!(estimate(&compact(&mut s)), 2, "signed zeros are one value");
    }

    #[test]
    fn bytes_round_trip_and_other_families_are_named() {
        let back = from_bytes(&to_bytes(&of(&[7])), F).unwrap();
        assert_eq!(estimate(&back), 1);
        assert!(from_bytes(b"not a sketch", F).is_err());
        assert!(from_bytes(&[], F).is_err());
        // An HLL preamble (family 7) is a sketch of another family.
        let hll = [2_u8, 1, 7, 12, 0, 0, 0, 0];
        let e = from_bytes(&hll, F).unwrap_err().to_string();
        assert!(
            e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER_FAMILY"),
            "{e}"
        );
    }

    #[test]
    fn family_names_are_javas() {
        for (id, name) in [
            (6_u8, "ANOTB"),
            (9, "TUPLE"),
            (15, "KLL"),
            (21, "BLOOMFILTER"),
        ] {
            let e = from_bytes(&[2_u8, 1, id, 12, 0, 0, 0, 0], F)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER_FAMILY"),
                "{e}"
            );
            assert!(e.contains(name), "{id}: {e}");
        }
        // 22 is no family: the plain buffer error.
        let e = from_bytes(&[2_u8, 1, 22, 12, 0, 0, 0, 0], F)
            .unwrap_err()
            .to_string();
        assert!(!e.contains("_FAMILY"), "{e}");
        assert!(e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER"), "{e}");
    }

    #[test]
    fn set_operations() {
        let a = of(&[1, 2, 3, 4]);
        let b = of(&[3, 4, 5]);
        assert_eq!(estimate(&union(&a, &b, 12, "theta_union").unwrap()), 5);
        assert_eq!(
            estimate(&intersection(&a, &b, "theta_intersection").unwrap()),
            2
        );
        assert_eq!(
            estimate(&difference(&a, &b, "theta_difference").unwrap()),
            2
        );
        assert!(union(&a, &b, 3, "theta_union").is_err());
    }

    #[test]
    fn mutated_and_truncated_buffers_never_panic() {
        let bytes = to_bytes(&of(&(0..5_000).collect::<Vec<_>>()));
        for len in 0..bytes.len().min(200) {
            let _ = from_bytes(&bytes[..len], F);
        }
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..300 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut m = bytes.clone();
            let i = usize::try_from(state % m.len() as u64).unwrap();
            m[i] ^= 1 << (state % 8);
            if let Ok(s) = from_bytes(&m, F) {
                let _ = estimate(&s);
            }
        }
    }
}
