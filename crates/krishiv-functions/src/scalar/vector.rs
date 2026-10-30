// SPDX-License-Identifier: Apache-2.0
//! The vector functions: `vector_norm`, `vector_normalize`,
//! `vector_cosine_similarity`, `vector_inner_product` and
//! `vector_l2_distance` over `ARRAY<FLOAT>` (Spark 4.2's `VectorFunctions`).
//!
//! A NULL vector, element or degree is NULL, and a NULL vector is NULL
//! before its degree is checked. The degree must be `1.0`, `2.0` or
//! `float('inf')` (`INVALID_VECTOR_NORM_DEGREE` otherwise), and vectors of
//! two dimensions do not combine (`VECTOR_DIMENSION_MISMATCH`). The sums run
//! in `f32`, element by element, as Spark's do.

use std::sync::Arc;

use arrow_array::builder::{Float32Builder, ListBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, Float32Array};
use arrow_schema::{DataType, Field};
use krishiv_common::error::{InvalidVectorNormDegree, KrishivError, VectorDimensionMismatch};

use super::args::Inputs;
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "vector_cosine_similarity",
    "vector_inner_product",
    "vector_l2_distance",
    "vector_norm",
    "vector_normalize",
];

/// Whether `t` is a list of `FLOAT`.
fn is_vector(t: &DataType) -> bool {
    matches!(t, DataType::List(f) if f.data_type() == &DataType::Float32)
}

/// Binds the vector function `name` to `args`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let norm = matches!(listed, "vector_norm" | "vector_normalize");
    let shape_ok = if norm {
        matches!(types.len(), 1 | 2)
            && is_vector(&types[0])
            && types.get(1).is_none_or(|t| *t == DataType::Float32)
    } else {
        types.len() == 2 && types.iter().all(is_vector)
    };
    if !shape_ok {
        return Some((listed, Err(bad_types(listed, args))));
    }
    let with_degree = types.len() == 2;
    let binding = match listed {
        "vector_norm" => Binding::new(DataType::Float32, move |x: &Inputs<'_>| {
            let (v, d) = norm_inputs(x, with_degree)?;
            let mut out = Float32Builder::with_capacity(x.rows());
            for row in 0..x.rows() {
                out.append_option(row_norm(&v, &d, row, "vector_norm")?);
            }
            Ok(Arc::new(out.finish()) as ArrayRef)
        }),
        "vector_normalize" => {
            let item = Arc::new(Field::new("element", DataType::Float32, false));
            let ty = DataType::List(Arc::clone(&item));
            Binding::new(ty, move |x: &Inputs<'_>| {
                let (v, d) = norm_inputs(x, with_degree)?;
                let mut out = ListBuilder::with_capacity(Float32Builder::new(), x.rows())
                    .with_field(Arc::clone(&item));
                for row in 0..x.rows() {
                    match row_normalized(&v, &d, row)? {
                        Some(values) => {
                            out.values().append_slice(&values);
                            out.append(true);
                        }
                        None => out.append(false),
                    }
                }
                Ok(Arc::new(out.finish()) as ArrayRef)
            })
        }
        _ => {
            let kind = match listed {
                "vector_cosine_similarity" => Pair::Cosine,
                "vector_inner_product" => Pair::Inner,
                _ => Pair::L2,
            };
            Binding::new(DataType::Float32, move |x: &Inputs<'_>| {
                let (a, b) = (x.column(0)?, x.column(1)?);
                let (a, b) = (a.as_list::<i32>(), b.as_list::<i32>());
                let mut out = Float32Builder::with_capacity(x.rows());
                for row in 0..x.rows() {
                    out.append_option(pair(kind, listed, a, b, row)?);
                }
                Ok(Arc::new(out.finish()) as ArrayRef)
            })
        }
    };
    Some((listed, Ok(binding)))
}

/// The vector column and the degree column (`2.0` when written without one).
fn norm_inputs(
    x: &Inputs<'_>,
    with_degree: bool,
) -> Result<(ArrayRef, Float32Array), KrishivError> {
    let v = x.column(0)?;
    let d = if with_degree {
        x.column(1)?.as_primitive::<Float32Type>().clone()
    } else {
        Float32Array::from(vec![2.0_f32; x.rows()])
    };
    Ok((v, d))
}

/// The elements of `list` row `row`, or `None` for a NULL vector or one with
/// a NULL element.
fn elements(list: &arrow_array::ListArray, row: usize) -> Option<Vec<f32>> {
    if list.is_null(row) {
        return None;
    }
    let values = list.value(row);
    let values = values.as_primitive::<Float32Type>();
    if values.null_count() > 0 {
        return None;
    }
    Some(values.values().to_vec())
}

#[derive(Clone, Copy)]
enum Degree {
    One,
    Two,
    Infinity,
}

/// Java's `Float.toString` for the values a bad degree can take.
fn java_float(v: f32) -> String {
    if v.is_nan() {
        "NaN".to_owned()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        format!("{v:?}")
    }
}

fn degree_of(name: &str, d: f32) -> Result<Degree, KrishivError> {
    #[expect(clippy::float_cmp, reason = "the degree is one of three exact values")]
    let degree = if d == 1.0 {
        Degree::One
    } else if d == 2.0 {
        Degree::Two
    } else if d == f32::INFINITY {
        Degree::Infinity
    } else {
        return Err(InvalidVectorNormDegree {
            function_name: format!("`{name}`"),
            degree: java_float(d),
        }
        .into());
    };
    Ok(degree)
}

fn norm_of(v: &[f32], degree: Degree) -> f32 {
    match degree {
        Degree::One => v.iter().fold(0.0_f32, |s, x| s + x.abs()),
        Degree::Two => v.iter().fold(0.0_f32, |s, x| s + x * x).sqrt(),
        Degree::Infinity => v.iter().fold(0.0_f32, |m, x| {
            if x.abs() > m || x.is_nan() {
                x.abs()
            } else {
                m
            }
        }),
    }
}

/// The norm of row `row`: NULL for a NULL vector, element or degree.
fn row_norm(
    v: &ArrayRef,
    d: &Float32Array,
    row: usize,
    name: &str,
) -> Result<Option<f32>, KrishivError> {
    let Some(values) = elements(v.as_list::<i32>(), row) else {
        return Ok(None);
    };
    if d.is_null(row) {
        return Ok(None);
    }
    let degree = degree_of(name, d.value(row))?;
    Ok(Some(norm_of(&values, degree)))
}

/// The row's vector divided by its norm: NULL for a zero norm.
fn row_normalized(
    v: &ArrayRef,
    d: &Float32Array,
    row: usize,
) -> Result<Option<Vec<f32>>, KrishivError> {
    let Some(values) = elements(v.as_list::<i32>(), row) else {
        return Ok(None);
    };
    if d.is_null(row) {
        return Ok(None);
    }
    let degree = degree_of("vector_normalize", d.value(row))?;
    if values.is_empty() {
        return Ok(Some(values));
    }
    let norm = norm_of(&values, degree);
    if norm == 0.0 {
        return Ok(None);
    }
    Ok(Some(values.iter().map(|x| x / norm).collect()))
}

#[derive(Clone, Copy)]
enum Pair {
    Cosine,
    Inner,
    L2,
}

fn pair(
    kind: Pair,
    name: &str,
    a: &arrow_array::ListArray,
    b: &arrow_array::ListArray,
    row: usize,
) -> Result<Option<f32>, KrishivError> {
    let (Some(x), Some(y)) = (elements(a, row), elements(b, row)) else {
        return Ok(None);
    };
    if x.len() != y.len() {
        return Err(VectorDimensionMismatch {
            function_name: format!("`{name}`"),
            left_dim: x.len().to_string(),
            right_dim: y.len().to_string(),
        }
        .into());
    }
    let dot = x.iter().zip(&y).fold(0.0_f32, |s, (p, q)| s + p * q);
    Ok(match kind {
        Pair::Inner => Some(dot),
        Pair::L2 => Some(
            x.iter()
                .zip(&y)
                .fold(0.0_f32, |s, (p, q)| s + (p - q) * (p - q))
                .sqrt(),
        ),
        Pair::Cosine => {
            // `dot / sqrt(|x|^2 * |y|^2)`, the order Spark's float result
            // follows (not the product of the two roots).
            let sq = |v: &[f32]| v.iter().fold(0.0_f32, |s, p| s + p * p);
            let (nx, ny) = (sq(&x), sq(&y));
            if nx == 0.0 || ny == 0.0 {
                None
            } else {
                Some(dot / (nx * ny).sqrt())
            }
        }
    })
}
