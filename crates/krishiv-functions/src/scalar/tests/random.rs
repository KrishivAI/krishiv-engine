// SPDX-License-Identifier: Apache-2.0
//! The seeded random bindings: Spark's streams, golden values and the
//! per-partition contract.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type};
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array};
use arrow_schema::DataType;

use super::{Arg, BindArg, BindOptions, ints, lit, strs};

fn seed_i32(v: i32) -> ArrayRef {
    Arc::new(Int32Array::from(vec![v]))
}

fn seed_i64(v: i64) -> ArrayRef {
    Arc::new(Int64Array::from(vec![v]))
}

fn null_i32() -> ArrayRef {
    Arc::new(Int32Array::from(vec![None as Option<i32>]))
}

/// The `f64` values of `a`.
fn f64s(a: &ArrayRef) -> Vec<Option<f64>> {
    a.as_primitive::<Float64Type>().iter().collect()
}

/// `f` over `rows` rows in `partition`, with a literal seed `s`.
fn seeded(name: &str, s: &ArrayRef, rows: usize, partition: u32) -> ArrayRef {
    let bound = vec![BindArg::Literal(Arc::clone(s))];
    let f = super::bind_scalar(name, &bound, &BindOptions::new()).unwrap();
    assert_eq!(f.return_type(), &DataType::Float64);
    f.invoke_in(&[Arg::Scalar(Arc::clone(s))], rows, partition)
        .unwrap()
}

#[test]
fn rand_zero_matches_spark() {
    // `SELECT rand(0)` in `random.sql`.
    let s = seed_i32(0);
    let got = seeded("rand", &s, 1, 0);
    assert_eq!(f64s(&got), vec![Some(0.7604953758285915)]);
    // The alias binds the same stream.
    let got = seeded("random", &s, 1, 0);
    assert_eq!(f64s(&got), vec![Some(0.7604953758285915)]);
}

#[test]
fn rand_advances_one_draw_per_row() {
    // `SELECT uniform(10, 20, 0) FROM VALUES (0), (1), (2)` is 15, 16, 17:
    // the rows draw consecutively from one stream.
    let s = seed_i32(0);
    let got = seeded("rand", &s, 3, 0);
    let rows = f64s(&got);
    assert_eq!(rows.len(), 3);
    assert!(
        (rows[0].unwrap() - 0.7604953758285915).abs() < 1e-15,
        "{rows:?}"
    );
    assert!(rows.windows(2).all(|w| w[0] != w[1]), "{rows:?}");
}

#[test]
fn randn_zero_matches_spark() {
    // `SELECT randn(0)` in `random.sql`.
    let s = seed_i64(0);
    let got = seeded("randn", &s, 1, 0);
    assert_eq!(f64s(&got), vec![Some(1.6034991609278433)]);
}

#[test]
fn null_seeds_read_as_zero() {
    // `rand(NULL)` and `rand(cast(NULL AS INT))` are `rand(0)`.
    let s = null_i32();
    let got = seeded("rand", &s, 1, 0);
    assert_eq!(f64s(&got), vec![Some(0.7604953758285915)]);
    let got = seeded("randn", &s, 1, 0);
    assert_eq!(f64s(&got), vec![Some(1.6034991609278433)]);
}

#[test]
fn long_seeds_match_int_seeds() {
    let i = seed_i32(0);
    let l = seed_i64(0);
    assert_eq!(
        f64s(&seeded("rand", &i, 2, 0)),
        f64s(&seeded("rand", &l, 2, 0))
    );
    assert_eq!(
        f64s(&seeded("randn", &i, 2, 0)),
        f64s(&seeded("randn", &l, 2, 0))
    );
}

#[test]
fn partitions_own_their_streams() {
    // The same query in another partition draws another stream; the
    // partition seeds the generator (`seed + partitionIndex`, Spark's
    // `initializeInternal`).
    let s = seed_i32(0);
    let p0 = f64s(&seeded("rand", &s, 3, 0));
    let p1 = f64s(&seeded("rand", &s, 3, 1));
    assert_ne!(p0, p1);
    // `monotonically_increasing_id` and `rand` of two partitions differ,
    // and one partition is stable across batches.
    let f = super::bind_scalar(
        "rand",
        &[BindArg::Literal(Arc::clone(&s))],
        &BindOptions::new(),
    )
    .unwrap();
    let lit = Arg::Scalar(Arc::clone(&s));
    let a = f.invoke_in(std::slice::from_ref(&lit), 2, 0).unwrap();
    let b = f.invoke_in(std::slice::from_ref(&lit), 2, 0).unwrap();
    assert_ne!(f64s(&a), f64s(&b));
}

#[test]
fn uniform_matches_the_golden_values() {
    // `SELECT uniform(0, 10, 0)` is 7, `(10, 20, 0)` is 17.
    let bound = |lo: &ArrayRef, hi: &ArrayRef, sd: &ArrayRef| {
        let bound = vec![
            BindArg::Literal(Arc::clone(lo)),
            BindArg::Literal(Arc::clone(hi)),
            BindArg::Literal(Arc::clone(sd)),
        ];
        super::bind_scalar("uniform", &bound, &BindOptions::new()).unwrap()
    };
    let (zero, ten, twenty) = (seed_i32(0), seed_i32(10), seed_i32(20));
    let f = bound(&zero, &ten, &zero);
    assert_eq!(f.return_type(), &DataType::Int32);
    let out = f
        .invoke_in(&[lit(&zero), lit(&ten), lit(&zero)], 1, 0)
        .unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().values(), &[7]);
    let f = bound(&ten, &twenty, &zero);
    let out = f
        .invoke_in(&[lit(&ten), lit(&twenty), lit(&zero)], 1, 0)
        .unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().values(), &[17]);
    // Partition 1 draws its own stream (`16`, then `15`).
    let out = f
        .invoke_in(&[lit(&ten), lit(&twenty), lit(&zero)], 2, 1)
        .unwrap();
    assert_eq!(out.as_primitive::<Int32Type>().values(), &[16, 15]);
}

#[test]
fn randstr_matches_the_golden_strings() {
    // `randstr(10, 0)` is `ceV0PXaR2I`, `randstr(3, 0)` is `ceV`.
    let len = seed_i32(10);
    let sd = seed_i32(0);
    let bound = vec![
        BindArg::Literal(Arc::clone(&len)),
        BindArg::Literal(Arc::clone(&sd)),
    ];
    let f = super::bind_scalar("randstr", &bound, &BindOptions::new()).unwrap();
    assert_eq!(f.return_type(), &DataType::Utf8);
    let out = f.invoke_in(&[lit(&len), lit(&sd)], 1, 0).unwrap();
    let texts: Vec<Option<String>> = out
        .as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    assert_eq!(texts, vec![Some("ceV0PXaR2I".to_owned())]);
    // A fresh bind draws the same stream again.
    let len3 = seed_i32(3);
    let bound3 = vec![
        BindArg::Literal(Arc::clone(&len3)),
        BindArg::Literal(Arc::clone(&sd)),
    ];
    let g = super::bind_scalar("randstr", &bound3, &BindOptions::new()).unwrap();
    let out = g.invoke_in(&[lit(&len3), lit(&sd)], 1, 0).unwrap();
    let texts: Vec<Option<String>> = out
        .as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    assert_eq!(texts, vec![Some("ceV".to_owned())]);
}

#[test]
fn randstr_rows_draw_consecutively() {
    // One partition draws one stream: rows are consecutive slices of it
    // (as a real Spark run with one partition shows).
    let len = seed_i32(10);
    let sd = seed_i32(0);
    let bound = vec![
        BindArg::Literal(Arc::clone(&len)),
        BindArg::Literal(Arc::clone(&sd)),
    ];
    let f = super::bind_scalar("randstr", &bound, &BindOptions::new()).unwrap();
    let out = f.invoke_in(&[lit(&len), lit(&sd)], 3, 0).unwrap();
    let texts: Vec<Option<String>> = out
        .as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    assert_eq!(
        texts,
        vec![
            Some("ceV0PXaR2I".to_owned()),
            Some("lBhHi56d0u".to_owned()),
            Some("CtuwSOTLiL".to_owned())
        ]
    );
    // The golden's second partition draws its own stream: with partitions
    // `[0]`, `[1, 2]` the rows are `ceV0PXaR2I`, `fYxVfArnv7`,
    // `iSIv0VT2XL`.
    let out = f.invoke_in(&[lit(&len), lit(&sd)], 2, 1).unwrap();
    let texts: Vec<Option<String>> = out
        .as_string::<i32>()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect();
    assert_eq!(
        texts,
        vec![Some("fYxVfArnv7".to_owned()), Some("iSIv0VT2XL".to_owned())]
    );
}

#[test]
fn randstr_null_length_is_empty_and_negative_fails() {
    use krishiv_common::error::ErrorClass;
    let sd = seed_i32(0);
    // A NULL length is the empty string.
    let bound = vec![
        BindArg::Literal(null_i32()),
        BindArg::Literal(Arc::clone(&sd)),
    ];
    let f = super::bind_scalar("randstr", &bound, &BindOptions::new()).unwrap();
    let out = f.invoke_in(&[lit(&null_i32()), lit(&sd)], 2, 0).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.as_string::<i32>().iter().all(|s| s == Some("")),);
    // A negative length is `INVALID_PARAMETER_VALUE.LENGTH`.
    let neg = seed_i32(-1);
    let bound = vec![
        BindArg::Literal(Arc::clone(&neg)),
        BindArg::Literal(Arc::clone(&sd)),
    ];
    let f = super::bind_scalar("randstr", &bound, &BindOptions::new()).unwrap();
    let err = f.invoke_in(&[lit(&neg), lit(&sd)], 1, 0).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValueLength);
    assert!(err.to_string().contains("`randstr`"), "{err}");
    assert!(err.to_string().contains("-1"), "{err}");
}

#[test]
fn uuids_have_shape_and_partitioned_streams() {
    let sd = seed_i32(42);
    let bound = vec![BindArg::Literal(Arc::clone(&sd))];
    let f = super::bind_scalar("uuid", &bound, &BindOptions::new()).unwrap();
    assert_eq!(f.return_type(), &DataType::Utf8);
    let out = f.invoke_in(&[lit(&sd)], 2, 0).unwrap();
    let texts: Vec<String> = out
        .as_string::<i32>()
        .iter()
        .map(|s| s.unwrap().to_owned())
        .collect();
    assert_eq!(texts.len(), 2);
    assert_ne!(texts[0], texts[1]);
    for u in &texts {
        assert_eq!(u.len(), 36, "{u}");
        assert_eq!(&u[8..9], "-");
        assert_eq!(&u[14..15], "4");
        assert!(matches!(&u[19..20], "8" | "9" | "a" | "b"), "{u}");
    }
    // Another partition draws another stream.
    let p1 = f.invoke_in(&[lit(&sd)], 2, 1).unwrap();
    let other: Vec<String> = p1
        .as_string::<i32>()
        .iter()
        .map(|s| s.unwrap().to_owned())
        .collect();
    assert_ne!(texts, other);
    // The unseeded form binds with no arguments.
    let g = super::bind_scalar("uuid", &[], &BindOptions::new()).unwrap();
    let out = g.invoke_in(&[], 1, 0).unwrap();
    assert_eq!(out.len(), 1);
}

#[test]
fn unseeded_draws_come_from_the_query_seed() {
    // Two binds of `rand()` under one query seed agree; another query
    // seed draws another stream. (Separate `BindOptions` hold separate
    // salts; one query's binds draw different streams, below.)
    let f =
        super::bind_scalar("rand", &[], &BindOptions::new().with_query_start(1_000_000)).unwrap();
    let g =
        super::bind_scalar("rand", &[], &BindOptions::new().with_query_start(1_000_000)).unwrap();
    assert_eq!(
        f64s(&f.invoke_in(&[], 2, 0).unwrap()),
        f64s(&g.invoke_in(&[], 2, 0).unwrap())
    );
    let other = BindOptions::new().with_query_start(2_000_000);
    let h = super::bind_scalar("rand", &[], &other).unwrap();
    assert_ne!(
        f64s(&f.invoke_in(&[], 2, 0).unwrap()),
        f64s(&h.invoke_in(&[], 2, 0).unwrap())
    );
}

#[test]
fn unseeded_calls_in_one_query_draw_different_streams() {
    // Two unseeded binds sharing one query's salt draw different streams
    // (Spark's `ResolveRandomSeed` assigns each its own seed), and the
    // same holds for `uuid` and `uniform` without a seed.
    let opts = BindOptions::new().with_query_start(1_000_000);
    let f = super::bind_scalar("rand", &[], &opts).unwrap();
    let g = super::bind_scalar("rand", &[], &opts).unwrap();
    assert_ne!(
        f64s(&f.invoke_in(&[], 2, 0).unwrap()),
        f64s(&g.invoke_in(&[], 2, 0).unwrap())
    );
    let u = super::bind_scalar("uuid", &[], &opts).unwrap();
    let v = super::bind_scalar("uuid", &[], &opts).unwrap();
    let texts = |f: &Arc<dyn super::ScalarFunction>| {
        f.invoke_in(&[], 1, 0)
            .unwrap()
            .as_string::<i32>()
            .iter()
            .map(|s| s.unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_ne!(texts(&u), texts(&v));
}

#[test]
fn wrong_argument_types_do_not_bind() {
    use krishiv_common::error::ErrorClass;
    // A seed that is not an `int`/`bigint` is not bound (the analyzer
    // reports `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` first): the error
    // names the bad signature, not a missing function.
    for t in [DataType::Float64, DataType::Utf8] {
        let a = arrow_array::new_empty_array(&t);
        let err =
            super::bind_scalar("rand", &[BindArg::Column(t)], &BindOptions::new()).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(err.to_string().contains("rand("), "{err}");
        let _ = a;
    }
    // `uniform` takes two or three arguments.
    let err = super::bind_scalar("uniform", &[], &BindOptions::new()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // `randstr` takes one or two.
    let ty = || ints(&[Some(1)]).data_type().clone();
    let err = super::bind_scalar(
        "randstr",
        &[ty().into(), ty().into(), ty().into()],
        &BindOptions::new(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A string length casts like Spark's implicit cast; a bad one fails
    // the cast at runtime.
    let len = || {
        super::bind_scalar(
            "randstr",
            &[BindArg::Literal(strs(&[Some("10")]))],
            &BindOptions::new(),
        )
        .unwrap()
    };
    let out = len()
        .invoke_in(&[Arg::Scalar(strs(&[Some("10")]))], 1, 0)
        .unwrap();
    assert_eq!(out.as_string::<i32>().value(0).len(), 10);
    let err = len()
        .invoke_in(&[Arg::Scalar(strs(&[Some("abc")]))], 1, 0)
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::CastInvalidInput);
}
