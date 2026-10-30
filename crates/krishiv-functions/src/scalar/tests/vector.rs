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
            assert!(
                message.contains(shown) && message.contains(name),
                "{message}"
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
        Some(0.974_631_9)
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
    assert!(message.contains('3') && message.contains('2'), "{message}");
}

#[test]
fn only_float_vectors_bind() {
    let doubles = DataType::List(Arc::new(Field::new("element", DataType::Float64, true)));
    let floats = DataType::List(Arc::new(Field::new("element", DataType::Float32, true)));
    for name in [
        "vector_inner_product",
        "vector_cosine_similarity",
        "vector_l2_distance",
    ] {
        assert!(
            bind_scalar(
                name,
                &[doubles.clone().into(), floats.clone().into()],
                &BindOptions::new()
            )
            .is_err(),
            "{name}"
        );
        assert!(
            bind_scalar(name, &[floats.clone().into()], &BindOptions::new()).is_err(),
            "{name}"
        );
    }
    for name in ["vector_norm", "vector_normalize"] {
        assert!(bind_scalar(name, &[doubles.clone().into()], &BindOptions::new()).is_err());
        assert!(
            bind_scalar(
                name,
                &[floats.clone().into(), DataType::Float64.into()],
                &BindOptions::new()
            )
            .is_err()
        );
    }
}
