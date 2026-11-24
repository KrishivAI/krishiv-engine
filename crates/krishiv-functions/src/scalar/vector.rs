// SPDX-License-Identifier: Apache-2.0
//! The vector functions: `vector_norm`, `vector_normalize`,
//! `vector_cosine_similarity`, `vector_inner_product` and
//! `vector_l2_distance` over `ARRAY<FLOAT>` (Spark 4.2's `VectorFunctions`).
//!
//! A NULL vector, element or degree is NULL, and a NULL vector is NULL
//! before its degree is checked. The degree must be `1.0`, `2.0` or
//! `float('inf')` (`INVALID_VECTOR_NORM_DEGREE` otherwise), and vectors of
//! two dimensions do not combine (`VECTOR_DIMENSION_MISMATCH`). The sums are
//! `krishiv_kernels::vector`'s: blocks of 8 in `f32`, in Spark's order.
//!
//! Spark 4.2 and 4.3 differ here (the infinity norm's NaN, the norm's
//! normalizability, `double` accumulation); `bind` reads the version from
//! `BindOptions::kernel`. The executor builds its kernel context with the
//! default (4.3) version: the session's profile does not reach `EvalEnv` yet
//! (it would travel as an `ExprEnv` field and a plan codec feature), so a 4.2
//! session runs these functions with the 4.3 rules. Only `BindOptions` built
//! with an explicit version (the unit tests, the planner's analysis) see 4.2.

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
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    // Spark 4.3.0's infinity norm propagates a NaN element; 4.2.0's skips it.
    let nan = opts.kernel.spark_version() >= krishiv_kernels::SparkVersion::V4_3;
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
                out.append_option(row_norm(&v, &d, row, "vector_norm", nan)?);
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
                    match row_normalized(&v, &d, row, nan)? {
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
                // A literal operand (the query vector of a top-k search) is read at its one row, not
                // broadcast to `rows` copies of itself.
                let ((a, a_lit), (b, b_lit)) = (x.operand(0)?, x.operand(1)?);
                let (a, b) = (a.as_list::<i32>(), b.as_list::<i32>());
                let mut out = Float32Builder::with_capacity(x.rows());
                for row in 0..x.rows() {
                    let rows = (if a_lit { 0 } else { row }, if b_lit { 0 } else { row });
                    out.append_option(pair(kind, listed, a, b, rows)?);
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

/// Row `row` of `list` as floats, with whether any element is NULL (the
/// values under a NULL element are not read); `None` for a NULL vector. The
/// values are a slice of the list's child, not a copy.
fn row_values(list: &arrow_array::ListArray, row: usize) -> Option<(ArrayRef, bool)> {
    if list.is_null(row) {
        return None;
    }
    let values = list.value(row);
    let has_null = values.null_count() > 0;
    Some((values, has_null))
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
    } else if v == 0.0 || (1e-3..1e7).contains(&v.abs()) {
        format!("{v:?}")
    } else {
        // Java switches to computerized scientific notation outside
        // [1e-3, 1e7): `1.0E7`, `1.5E-4`.
        let text = format!("{v:e}");
        let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
        let point = if mantissa.contains('.') { "" } else { ".0" };
        format!("{mantissa}{point}E{exponent}")
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

/// The norm in double precision. Spark 4.3 (`nan_propagates`) accumulates the L1 and L2 norms in `double` and
/// divides by the double norm; 4.2 and earlier accumulate in `float` (the kernels), so their norm widens exactly.
fn norm_of(v: &[f32], degree: Degree, nan_propagates: bool) -> f64 {
    match degree {
        Degree::One if nan_propagates => v.iter().map(|&x| f64::from(x).abs()).sum(),
        Degree::Two if nan_propagates => v
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt(),
        Degree::One => f64::from(krishiv_kernels::vector::l1_norm(v)),
        Degree::Two => f64::from(krishiv_kernels::vector::l2_norm(v)),
        Degree::Infinity if nan_propagates => f64::from(krishiv_kernels::vector::inf_norm_nan(v)),
        Degree::Infinity => f64::from(krishiv_kernels::vector::inf_norm(v)),
    }
}

/// The norm of row `row`: NULL for a NULL vector, element or degree. The
/// degree is checked before a NULL element is looked for (Spark's order).
fn row_norm(
    v: &ArrayRef,
    d: &Float32Array,
    row: usize,
    name: &str,
    nan: bool,
) -> Result<Option<f32>, KrishivError> {
    let Some((values, has_null)) = row_values(v.as_list::<i32>(), row) else {
        return Ok(None);
    };
    if d.is_null(row) {
        return Ok(None);
    }
    let degree = degree_of(name, d.value(row))?;
    if has_null {
        return Ok(None);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the result column is FLOAT"
    )]
    Ok(Some(
        norm_of(values.as_primitive::<Float32Type>().values(), degree, nan) as f32,
    ))
}

/// The row's vector divided by its norm: NULL for a norm below `Float.MIN_NORMAL` (4.2 and earlier) or
/// zero (4.3, `nan`).
fn row_normalized(
    v: &ArrayRef,
    d: &Float32Array,
    row: usize,
    nan: bool,
) -> Result<Option<Vec<f32>>, KrishivError> {
    let Some((values, has_null)) = row_values(v.as_list::<i32>(), row) else {
        return Ok(None);
    };
    if d.is_null(row) {
        return Ok(None);
    }
    let degree = degree_of("vector_normalize", d.value(row))?;
    if has_null {
        return Ok(None);
    }
    let values = values.as_primitive::<Float32Type>().values();
    if values.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let norm = norm_of(values, degree, nan);
    // Spark 4.3 rejects only a zero norm (a subnormal one divides) and divides in double; 4.2 and earlier
    // reject below `Float.MIN_NORMAL` (`VectorFunctionImplUtils`) and divide in float.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a 4.2 norm is a widened float"
    )]
    let normalizable = if nan {
        norm != 0.0
    } else {
        krishiv_kernels::vector::is_normalizable(norm as f32)
    };
    if !normalizable {
        return Ok(None);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the result elements are FLOAT"
    )]
    Ok(Some(if nan {
        values
            .iter()
            .map(|&x| (f64::from(x) / norm) as f32)
            .collect()
    } else {
        let norm = norm as f32;
        values.iter().map(|x| x / norm).collect()
    }))
}

#[derive(Clone, Copy)]
enum Pair {
    Cosine,
    Inner,
    L2,
}

/// One row of a two-vector function. Spark's order: a NULL vector is NULL,
/// then the dimensions are compared (a mismatch raises even when an element
/// is NULL), then an empty vector answers, then a NULL element is NULL.
fn pair(
    kind: Pair,
    name: &str,
    a: &arrow_array::ListArray,
    b: &arrow_array::ListArray,
    (a_row, b_row): (usize, usize),
) -> Result<Option<f32>, KrishivError> {
    let (Some((x, x_null)), Some((y, y_null))) = (row_values(a, a_row), row_values(b, b_row))
    else {
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
    if x.is_empty() {
        return Ok(match kind {
            Pair::Cosine => None,
            Pair::Inner | Pair::L2 => Some(0.0),
        });
    }
    if x_null || y_null {
        return Ok(None);
    }
    let (x, y) = (
        x.as_primitive::<Float32Type>().values(),
        y.as_primitive::<Float32Type>().values(),
    );
    Ok(match kind {
        Pair::Inner => Some(krishiv_kernels::vector::inner_product(x, y)),
        Pair::L2 => Some(krishiv_kernels::vector::l2_distance(x, y)),
        Pair::Cosine => krishiv_kernels::vector::cosine_similarity(x, y),
    })
}
