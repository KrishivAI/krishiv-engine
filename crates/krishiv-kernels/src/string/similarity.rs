// SPDX-License-Identifier: Apache-2.0
//! `jaro_winkler_similarity` (Spark 4.3): Spark's own `ExpressionImplUtils.jaroWinklerSimilarity` over the strings'
//! UTF-16 code units, as the JVM compares them (an emoji is two units).
//!
//! The match window is `max(longer / 2 - 1, 0)`, the prefix bonus counts at most four equal leading units with
//! the scaling factor `0.1`, and the bonus always applies (`jaro + prefix * 0.1 * (1 - jaro)`; Spark has no
//! minimum Jaro score, unlike Commons Text). Equal strings
//! (also two empty ones) are `1.0`; no match is `0.0`. The comparison is binary: case and collation do not matter.

use arrow_array::{Float64Array, OffsetSizeTrait};

use super::{Slot, StrOperand, all_valid, row_count};
use crate::error::KernelError;

/// The prefix length the bonus counts (Spark's prefix limit).
const PREFIX_LIMIT: usize = 4;

/// The similarity of `a` and `b` in `[0, 1]`.
fn jaro_winkler(a: &[u16], b: &[u16]) -> f64 {
    if a == b {
        return 1.0;
    }
    let (max, min) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let range = (max.len() / 2).saturating_sub(1);
    let mut match_index: Vec<Option<usize>> = vec![None; min.len()];
    let mut taken = vec![false; max.len()];
    let mut matches = 0_usize;
    for (mi, &c) in min.iter().enumerate() {
        let from = mi.saturating_sub(range);
        let to = mi.saturating_add(range).saturating_add(1).min(max.len());
        for xi in from..to {
            if !taken[xi] && max[xi] == c {
                match_index[mi] = Some(xi);
                taken[xi] = true;
                matches = matches.saturating_add(1);
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let from_min = min
        .iter()
        .zip(&match_index)
        .filter_map(|(c, m)| m.map(|_| *c));
    let from_max = max.iter().zip(&taken).filter_map(|(c, t)| t.then_some(*c));
    let transpositions = from_min.zip(from_max).filter(|(x, y)| x != y).count();
    let prefix = a
        .iter()
        .zip(b)
        .take(min.len().min(PREFIX_LIMIT))
        .take_while(|(x, y)| x == y)
        .count();
    let m = matches as f64;
    let jaro =
        (m / a.len() as f64 + m / b.len() as f64 + (m - transpositions as f64 / 2.0) / m) / 3.0;
    jaro + 0.1 * prefix as f64 * (1.0 - jaro)
}

/// `jaro_winkler_similarity(a, b)`: NULL when either operand is NULL.
///
/// # Errors
/// `INTERNAL_ERROR` for operands of different lengths.
pub fn jaro_winkler_similarity<O: OffsetSizeTrait>(
    left: StrOperand<'_, O>,
    right: StrOperand<'_, O>,
) -> Result<Float64Array, KernelError> {
    let slots: [&dyn Slot; 2] = [&left, &right];
    let rows = row_count(&slots)?;
    let units = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
    Ok((0..rows)
        .map(|i| {
            all_valid(&slots, i).then(|| jaro_winkler(&units(left.at(i)), &units(right.at(i))))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jw(a: &str, b: &str) -> f64 {
        let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        jaro_winkler(&u(a), &u(b))
    }

    /// Values from a Spark 4.3.0 run.
    #[test]
    fn matches_spark_4_3_0() {
        for (a, b, want) in [
            ("martha", "marhta", 0.9611111111111111),
            ("dixon", "dicksonx", 0.8133333333333332),
            ("abc", "xyz", 0.0),
            ("", "a", 0.0),
            ("", "", 1.0),
            ("ABC", "abc", 0.0),
            ("prefixabcdefg", "prefixabcdxfg", 0.9692307692307692),
            ("jellyfish", "smellyfish", 0.8962962962962964),
            ("😀a", "😀b", 0.8222222222222222),
            ("aaaaaaaa", "aaaaaaab", 0.95),
            (
                "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzzz",
                "zzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
                0.824132308409038,
            ),
            ("ab", "ba", 0.0),
            ("abcd", "abdc", 0.9333333333333333),
        ] {
            assert!((jw(a, b) - want).abs() < 1e-12, "{a:?} {b:?}: {}", jw(a, b));
        }
    }
}
