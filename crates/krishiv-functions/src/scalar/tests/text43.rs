// SPDX-License-Identifier: Apache-2.0
//! `to_base32`, `from_base32` and `jaro_winkler_similarity` (Spark 4.3.0); the values are rows of a 4.3.0 run.

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::Float64Type;
use arrow_array::{ArrayRef, BinaryArray, StringArray};

use super::{owned, run, text};

#[test]
fn base32_round_trips_and_keeps_nulls() {
    let bin: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(&b"abc"[..]),
        Some(&b""[..]),
        None,
        Some(&b"foobar"[..]),
    ]));
    let encoded = run("to_base32", &[&bin]);
    assert_eq!(
        text(&encoded),
        owned(&[Some("MFRGG==="), Some(""), None, Some("MZXW6YTBOI======")])
    );
    let decoded = run("from_base32", &[&encoded]);
    let decoded: Vec<Option<&[u8]>> = decoded.as_binary::<i32>().iter().collect();
    assert_eq!(
        decoded,
        vec![
            Some(&b"abc"[..]),
            Some(&b""[..]),
            None,
            Some(&b"foobar"[..])
        ]
    );
}

#[test]
fn from_base32_is_lenient() {
    let s: ArrayRef = Arc::new(StringArray::from(vec![
        "!!!",
        "mzxw6",
        "MZ XW6===",
        "MY======MZXW6===",
    ]));
    let out = run("from_base32", &[&s]);
    let out: Vec<_> = out
        .as_binary::<i32>()
        .iter()
        .map(|b| b.map(<[u8]>::to_vec))
        .collect();
    assert_eq!(
        out,
        vec![
            Some(vec![]),
            Some(b"foo".to_vec()),
            Some(b"foo".to_vec()),
            Some(b"f".to_vec())
        ]
    );
}

#[test]
fn jaro_winkler_matches_spark() {
    let a: ArrayRef = Arc::new(StringArray::from(vec![
        Some("martha"),
        Some("dixon"),
        Some("abc"),
        None,
        Some(""),
    ]));
    let b: ArrayRef = Arc::new(StringArray::from(vec![
        Some("marhta"),
        Some("dicksonx"),
        Some("xyz"),
        Some("a"),
        Some(""),
    ]));
    let out = run("jaro_winkler_similarity", &[&a, &b]);
    let out: Vec<_> = out.as_primitive::<Float64Type>().iter().collect();
    assert_eq!(
        out,
        vec![
            Some(0.961_111_111_111_111_1),
            Some(0.813_333_333_333_333_2),
            Some(0.0),
            None,
            Some(1.0)
        ]
    );
}

/// Spark's own `jaroWinklerSimilarity` adds the prefix bonus at every Jaro score (no 0.7 threshold, unlike
/// Commons Text); the values are from a Spark 4.3.0 run.
#[test]
fn jaro_winkler_has_no_minimum_score_for_the_prefix_bonus() {
    let a: ArrayRef = Arc::new(StringArray::from(vec![Some("abcd"), Some("aa")]));
    let b: ArrayRef = Arc::new(StringArray::from(vec![Some("axyz"), Some("abca")]));
    let out = run("jaro_winkler_similarity", &[&a, &b]);
    let out: Vec<_> = out.as_primitive::<Float64Type>().iter().collect();
    assert_eq!(out, vec![Some(0.55), Some(0.625)]);
}
