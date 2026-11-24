// SPDX-License-Identifier: Apache-2.0
//! The vector bindings, checked against Spark 4.2's `vector-norm.sql` and
//! `vector-distance.sql` goldens (the session golden run pins the rest).

use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, Float32Array};
use arrow_schema::{DataType, Field};
use krishiv_common::error::ErrorClass;

use super::{BindArg, BindOptions, bind, bind_scalar, col};

/// Vectors with a NULL row (`None`) and NULL elements (`Some(None)`).
fn vectors(rows: &[Option<Vec<Option<f32>>>]) -> ArrayRef {
    let field = Arc::new(Field::new("element", DataType::Float32, true));
    let mut b = ListBuilder::new(Float32Builder::new()).with_field(field);
    for row in rows {
        match row {
            Some(values) => {
                for v in values {
                    b.values().append_option(*v);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn plain(v: &[f32]) -> ArrayRef {
    vectors(&[Some(v.iter().copied().map(Some).collect())])
}

fn floats(v: &[Option<f32>]) -> ArrayRef {
    Arc::new(Float32Array::from(v.to_vec()))
}

fn one(name: &str, args: &[&ArrayRef]) -> Option<f32> {
    let f = bind(name, args);
    let rows = args[0].len();
    let out = f
        .invoke(&args.iter().map(|a| col(a)).collect::<Vec<_>>(), rows)
        .unwrap();
    out.as_primitive::<Float32Type>().iter().next().flatten()
}

fn error(name: &str, args: &[&ArrayRef]) -> (ErrorClass, String) {
    let f = bind(name, args);
    let e = f
        .invoke(
            &args.iter().map(|a| col(a)).collect::<Vec<_>>(),
            args[0].len(),
        )
        .unwrap_err();
    (e.error_class(), e.to_string())
}

#[test]
fn norms_follow_the_degree() {
    let v = plain(&[3.0, -4.0]);
    assert_eq!(one("vector_norm", &[&v, &floats(&[Some(2.0)])]), Some(5.0));
    assert_eq!(one("vector_norm", &[&v, &floats(&[Some(1.0)])]), Some(7.0));
    assert_eq!(
        one("vector_norm", &[&v, &floats(&[Some(f32::INFINITY)])]),
        Some(4.0)
    );
    // Without a degree it is the L2 norm; empty vectors have norm 0.
    assert_eq!(one("vector_norm", &[&v]), Some(5.0));
    let empty = plain(&[]);
    assert_eq!(
        one("vector_norm", &[&empty, &floats(&[Some(2.0)])]),
        Some(0.0)
    );
}

/// `vector_norm` / `vector_normalize` of `[NaN, 5]` with degree `inf` on `version`.
fn nan_inf(name: &str, version: krishiv_kernels::SparkVersion) -> Vec<f32> {
    let v = plain(&[f32::NAN, 5.0]);
    let d = floats(&[Some(f32::INFINITY)]);
    let opts = BindOptions::new()
        .with_kernel(krishiv_kernels::KernelCtx::new().with_spark_version(version));
    let args = [&v, &d];
    let f = super::bind_with(name, &args, &opts);
    let out = f
        .invoke(&args.iter().map(|a| col(a)).collect::<Vec<_>>(), 1)
        .unwrap();
    if name == "vector_norm" {
        out.as_primitive::<Float32Type>().values().to_vec()
    } else {
        let list = out.as_list::<i32>().value(0);
        list.as_primitive::<Float32Type>().values().to_vec()
    }
}

#[test]
fn a_nan_element_is_skipped_by_the_infinity_norm_in_42_and_propagated_in_43() {
    use krishiv_kernels::SparkVersion::{V4_2, V4_3};
    assert_eq!(nan_inf("vector_norm", V4_2), vec![5.0]);
    assert!(nan_inf("vector_norm", V4_3)[0].is_nan());
    // normalize divides by the norm: 4.2 by 5.0, 4.3 by NaN.
    assert!(nan_inf("vector_normalize", V4_2)[0].is_nan());
    assert_eq!(nan_inf("vector_normalize", V4_2)[1], 1.0);
    assert!(nan_inf("vector_normalize", V4_3).iter().all(|x| x.is_nan()));
}

#[test]
fn null_inputs_are_null_and_a_null_vector_skips_the_degree_check() {
    let two = floats(&[Some(2.0)]);
    let bad = floats(&[Some(3.0)]);
    let null_vec = vectors(&[None]);
    let null_elem = vectors(&[Some(vec![Some(1.0), None])]);
    assert_eq!(one("vector_norm", &[&null_vec, &bad]), None);
    assert_eq!(one("vector_norm", &[&null_elem, &two]), None);
    assert_eq!(
        one("vector_norm", &[&plain(&[1.0]), &floats(&[None])]),
        None
    );
    assert_eq!(
        one("vector_cosine_similarity", &[&null_vec, &plain(&[1.0])]),
        None
    );
    assert_eq!(one("vector_l2_distance", &[&null_elem, &null_elem]), None);
}

#[test]
fn a_degree_other_than_one_two_or_infinity_is_refused() {
    let v = plain(&[3.0, 4.0]);
    for (degree, shown) in [
        (0.0, "0.0"),
        (3.0, "3.0"),
        (-1.0, "-1.0"),
        (f32::NEG_INFINITY, "-Infinity"),
        (f32::NAN, "NaN"),
        // Java's `Float.toString` leaves plain notation outside [1e-3, 1e7).
        (1e7, "1.0E7"),
        (1.5e-4, "1.5E-4"),
        (-2.5e10, "-2.5E10"),
        (0.001, "0.001"),
        (9_999_999.0, "9999999.0"),
    ] {
        for name in ["vector_norm", "vector_normalize"] {
            let (class, message) = error(name, &[&v, &floats(&[Some(degree)])]);
            assert_eq!(
                class.name(),
                "INVALID_VECTOR_NORM_DEGREE",
                "{name} {degree}"
            );
            assert_eq!(
                message,
                format!(
                    "[INVALID_VECTOR_NORM_DEGREE] For the `{name}` function, degree must be \
                     1.0, 2.0, or float('inf'), but got {shown}. SQLSTATE: 22023"
                )
            );
        }
    }
    // Positive infinity is the infinity norm, not an error.
    assert!(
        bind_scalar(
            "vector_norm",
            &[BindArg::from(v.data_type().clone())],
            &BindOptions::new()
        )
        .is_ok()
    );
}

#[test]
fn normalize_divides_by_the_norm() {
    let f = bind(
        "vector_normalize",
        &[&plain(&[3.0, 4.0]), &floats(&[Some(1.0)])],
    );
    let out = f
        .invoke(&[col(&plain(&[3.0, 4.0])), col(&floats(&[Some(1.0)]))], 1)
        .unwrap();
    let list = out.as_list::<i32>();
    let values = list.value(0);
    let values = values.as_primitive::<Float32Type>();
    assert_eq!(values.values().to_vec(), vec![3.0 / 7.0, 4.0 / 7.0]);
    // The result's field is the analyzer's: non-null `element`.
    assert_eq!(
        f.return_type(),
        &DataType::List(Arc::new(Field::new("element", DataType::Float32, false)))
    );
    // A zero vector, a NULL vector and a NULL degree are NULL; empty stays empty.
    for input in [plain(&[0.0, 0.0]), vectors(&[None])] {
        let f = bind("vector_normalize", &[&input, &floats(&[Some(2.0)])]);
        let out = f
            .invoke(&[col(&input), col(&floats(&[Some(2.0)]))], 1)
            .unwrap();
        assert!(out.is_null(0));
    }
    let f = bind("vector_normalize", &[&plain(&[1.0]), &floats(&[None])]);
    assert!(
        f.invoke(&[col(&plain(&[1.0])), col(&floats(&[None]))], 1)
            .unwrap()
            .is_null(0)
    );
    let f = bind("vector_normalize", &[&plain(&[]), &floats(&[Some(2.0)])]);
    let out = f
        .invoke(&[col(&plain(&[])), col(&floats(&[Some(2.0)]))], 1)
        .unwrap();
    assert_eq!(out.as_list::<i32>().value_length(0), 0);
}

#[test]
fn pairs_combine_equal_dimensions() {
    let (a, b) = (plain(&[1.0, 2.0, 3.0]), plain(&[4.0, 5.0, 6.0]));
    assert_eq!(one("vector_inner_product", &[&a, &b]), Some(32.0));
    assert_eq!(one("vector_l2_distance", &[&a, &b]), Some(5.196_152));
    assert_eq!(
        one("vector_cosine_similarity", &[&a, &b]),
        // Spark 4.3 sums in double: the float of 32 / sqrt(14 * 77). 4.2's float sums give 0.9746319 (pinned by
        // the 4.2.0 goldens in krishiv-session).
        Some(0.974_631_85)
    );
    // A zero vector has no direction; empty vectors have no distance.
    assert_eq!(
        one("vector_cosine_similarity", &[&plain(&[0.0, 0.0, 0.0]), &b]),
        None
    );
    assert_eq!(
        one("vector_inner_product", &[&plain(&[]), &plain(&[])]),
        Some(0.0)
    );
    let (class, message) = error("vector_l2_distance", &[&a, &plain(&[1.0, 2.0])]);
    assert_eq!(class.name(), "VECTOR_DIMENSION_MISMATCH");
    assert_eq!(
        message,
        "[VECTOR_DIMENSION_MISMATCH] Vectors passed to `vector_l2_distance` must have the same \
         dimension, but got 3 and 2. SQLSTATE: 22000"
    );
}

/// The error of binding `name` to `args`: the class and the message.
fn refused(name: &str, args: &[DataType]) -> (ErrorClass, String) {
    let args: Vec<BindArg> = args.iter().cloned().map(BindArg::from).collect();
    let e = bind_scalar(name, &args, &BindOptions::new())
        .err()
        .unwrap_or_else(|| panic!("{name} bound"));
    (e.error_class(), e.to_string())
}

#[test]
fn only_float_vectors_bind() {
    let doubles = DataType::List(Arc::new(Field::new("element", DataType::Float64, true)));
    let floats = DataType::List(Arc::new(Field::new("element", DataType::Float32, true)));
    let shown = |t: &DataType| t.to_string();
    for name in [
        "vector_inner_product",
        "vector_cosine_similarity",
        "vector_l2_distance",
    ] {
        for args in [vec![doubles.clone(), floats.clone()], vec![floats.clone()]] {
            let (class, message) = refused(name, &args);
            let types: Vec<String> = args.iter().map(shown).collect();
            assert_eq!(class.name(), "KRISHIV_UNSUPPORTED_FEATURE", "{name}");
            assert!(
                message.contains(&format!("{name}({})", types.join(", "))),
                "{message}"
            );
        }
    }
    for name in ["vector_norm", "vector_normalize"] {
        for args in [
            vec![doubles.clone()],
            vec![floats.clone(), DataType::Float64],
        ] {
            let (class, message) = refused(name, &args);
            let types: Vec<String> = args.iter().map(shown).collect();
            assert_eq!(class.name(), "KRISHIV_UNSUPPORTED_FEATURE", "{name}");
            assert!(
                message.contains(&format!("{name}({})", types.join(", "))),
                "{message}"
            );
        }
    }
}

/// Spark checks the degree before it looks for a NULL element (`vectorNorm`), so a bad degree raises even over a
/// vector with a NULL element; a NULL vector row is NULL first.
#[test]
fn a_bad_degree_raises_before_a_null_element_is_looked_for() {
    let null_elem = vectors(&[Some(vec![Some(1.0), None])]);
    let bad = floats(&[Some(3.0)]);
    for name in ["vector_norm", "vector_normalize"] {
        let (class, _) = error(name, &[&null_elem, &bad]);
        assert_eq!(class, ErrorClass::InvalidVectorNormDegree, "{name}");
    }
}

/// A literal query vector is read at its one row: every row of the column gives what a column of `rows` copies of
/// the literal gives (the top-k search never builds those copies).
#[test]
fn a_literal_query_vector_gives_the_same_as_its_broadcast() {
    use super::lit;
    let rows: Vec<Option<Vec<Option<f32>>>> = (0..40_u8)
        .map(|r| match r % 7 {
            0 => None,
            3 => Some(vec![Some(1.0), None, Some(2.0)]),
            r => Some(vec![Some(f32::from(r)), Some(0.5), Some(-1.5)]),
        })
        .collect();
    let col_a = vectors(&rows);
    let q = plain(&[0.25, 4.0, -2.0]);
    let copies = vectors(&vec![
        Some(vec![Some(0.25), Some(4.0), Some(-2.0)]);
        rows.len()
    ]);
    for name in [
        "vector_l2_distance",
        "vector_cosine_similarity",
        "vector_inner_product",
    ] {
        let f = bind(name, &[&col_a, &q]);
        let n = rows.len();
        let with_literal = f.invoke(&[col(&col_a), lit(&q)], n).unwrap();
        let with_copies = f.invoke(&[col(&col_a), col(&copies)], n).unwrap();
        assert_eq!(
            with_literal.as_primitive::<Float32Type>(),
            with_copies.as_primitive::<Float32Type>(),
            "{name}"
        );
        let flipped = f.invoke(&[lit(&q), col(&col_a)], n).unwrap();
        let flipped_copies = f.invoke(&[col(&copies), col(&col_a)], n).unwrap();
        assert_eq!(
            flipped.as_primitive::<Float32Type>(),
            flipped_copies.as_primitive::<Float32Type>(),
            "{name}"
        );
    }
}

/// `vector_norm` (`norm`) or `vector_normalize` of `values` with degree `degree` on `version`.
fn run_on(
    name: &str,
    values: &[f32],
    degree: f32,
    version: krishiv_kernels::SparkVersion,
) -> Option<Vec<f32>> {
    let v = plain(values);
    let d = floats(&[Some(degree)]);
    let opts = BindOptions::new()
        .with_kernel(krishiv_kernels::KernelCtx::new().with_spark_version(version));
    let args = [&v, &d];
    let f = super::bind_with(name, &args, &opts);
    let out = f
        .invoke(&args.iter().map(|a| col(a)).collect::<Vec<_>>(), 1)
        .unwrap();
    if name == "vector_norm" {
        out.as_primitive::<Float32Type>()
            .iter()
            .next()
            .flatten()
            .map(|x| vec![x])
    } else {
        let list = out.as_list::<i32>();
        (!list.is_null(0)).then(|| {
            list.value(0)
                .as_primitive::<Float32Type>()
                .values()
                .to_vec()
        })
    }
}

/// Spark 4.3 accumulates the L1 and L2 norms in `double` and divides by the double norm: a float square that
/// underflows or a float sum that overflows is still a normal double. 4.2 keeps the float result.
#[test]
fn the_l1_and_l2_norms_are_double_precision_in_43_and_float_in_42() {
    use krishiv_kernels::SparkVersion::{V4_2, V4_3};
    // 1e-30 squared is 1e-60: zero as a float, normal as a double.
    assert_eq!(
        run_on("vector_normalize", &[1e-30], 2.0, V4_3),
        Some(vec![1.0])
    );
    assert_eq!(run_on("vector_normalize", &[1e-30], 2.0, V4_2), None);
    // f32::MAX + f32::MAX overflows a float sum, not a double one.
    assert_eq!(
        run_on("vector_normalize", &[f32::MAX, f32::MAX], 1.0, V4_3),
        Some(vec![0.5, 0.5])
    );
    assert_eq!(
        run_on("vector_normalize", &[f32::MAX, f32::MAX], 1.0, V4_2),
        Some(vec![0.0, 0.0])
    );
    // The norm itself, cast to float at the end.
    assert_eq!(
        run_on("vector_norm", &[f32::MAX, f32::MAX], 1.0, V4_3),
        Some(vec![f32::INFINITY])
    );
    assert_eq!(
        run_on("vector_norm", &[3.0, -4.0], 2.0, V4_3),
        Some(vec![5.0])
    );
}

/// Spark 4.3 adds the double terms in blocks of 8 and then the accumulator (`VectorFunctionImplUtils`): the
/// second block of `[1, tiny x 15]` sums to `2^-51` before it meets the `1`, where a left-to-right fold
/// rounds every `tiny` away and stays at `1.0`.
#[test]
fn the_43_double_norms_sum_in_blocks_of_eight() {
    use crate::scalar::vector::{Degree, norm_of};
    let mut l1 = vec![2.0_f32.powi(-54); 16];
    l1[0] = 1.0;
    assert_eq!(norm_of(&l1, Degree::One, true), 1.0 + 2.0_f64.powi(-51));
    let mut l2 = vec![2.0_f32.powi(-27); 16];
    l2[0] = 1.0;
    assert_eq!(
        norm_of(&l2, Degree::Two, true),
        (1.0 + 2.0_f64.powi(-51)).sqrt()
    );
    // Under eight elements the block loop is empty: the tail is the plain fold.
    assert_eq!(norm_of(&[3.0, 4.0], Degree::Two, true), 5.0);
}
