// SPDX-License-Identifier: Apache-2.0
//! The Tuple sketch functions (Spark 4.2+, `tupleSketch*.scala`): signatures and analysis checks. The analyzer fills the
//! omitted `lgNomEntries` (12) and mode (`'sum'`) as Spark's constructors do. `lgNomEntries` and the mode are constants
//! checked at analysis (`NON_FOLDABLE_INPUT`, `TYPE_CHECK_FAILURE_WITH_HINT` carrying `SKETCH_INVALID_LG_NOM_ENTRIES` or
//! `TUPLE_INVALID_SKETCH_MODE`); the aggregates are run by `krishiv-exec`, the scalars by `scalar/tuple.rs`.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchTypeCheckFailureWithHint, KrishivError,
};
use krishiv_types::{ArrayType, DataType};

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Constant, InputBuilder, InputType, Inputs};

fn call_sql(name: &str, c: &CallTypes<'_>) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

fn non_foldable(name: &str, c: &CallTypes<'_>, i: usize, input: &str, ty: &str) -> KrishivError {
    DatatypeMismatchNonFoldableInput {
        sql_expr: call_sql(name, c),
        input_name: input.to_owned(),
        input_type: ty.to_owned(),
        input_expr: c.sql(i).unwrap_or_default(),
    }
    .into()
}

fn hint(name: &str, c: &CallTypes<'_>, msg: String) -> KrishivError {
    DatatypeMismatchTypeCheckFailureWithHint {
        sql_expr: call_sql(name, c),
        msg,
        hint: String::new(),
    }
    .into()
}

/// Spark's `checkLgNomEntriesParameter`: foldable, in `4..=26`.
fn check_lg(name: &str, c: &CallTypes<'_>, i: usize) -> Result<(), KrishivError> {
    match c.constant(i) {
        None => Err(non_foldable(name, c, i, "lgNomEntries", "int")),
        Some(Constant::Integral(k)) if !(4..=26).contains(k) => Err(hint(
            name,
            c,
            format!(
                "[SKETCH_INVALID_LG_NOM_ENTRIES] Invalid call to `{name}`; the `lgNomEntries` value must be between 4 and 26, inclusive: {k}. SQLSTATE: 22003"
            ),
        )),
        Some(_) => Ok(()),
    }
}

/// Spark's `checkModeParameter`: foldable, one of the four modes.
fn check_mode(name: &str, c: &CallTypes<'_>, i: usize) -> Result<(), KrishivError> {
    match c.constant(i) {
        None => Err(non_foldable(name, c, i, "mode", "string")),
        Some(Constant::String(m))
            if !matches!(
                m.to_lowercase().as_str(),
                "sum" | "min" | "max" | "alwaysone"
            ) =>
        {
            Err(hint(
                name,
                c,
                format!(
                    "[TUPLE_INVALID_SKETCH_MODE] Invalid call to `{name}`: mode '{m}' is not supported. Valid modes are: sum, min, max, alwaysone. SQLSTATE: 22023"
                ),
            ))
        }
        Some(_) => Ok(()),
    }
}

fn binary(_c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, false))
}

/// The key of `tuple_sketch_agg_*`: Spark's `TypeCollection`, as `theta_sketch_agg`'s, taken as it is.
fn key_input(c: &CallTypes<'_>) -> InputType {
    match c.data_type(0) {
        Some(t @ DataType::Array(a))
            if matches!(a.element_type(), DataType::Integer | DataType::Long) =>
        {
            InputType::Strict(t.clone())
        }
        _ => InputType::StrictOneOf(vec![
            DataType::Array(ArrayType::new(DataType::Integer, true).into()),
            DataType::Array(ArrayType::new(DataType::Long, true).into()),
            DataType::Binary,
            DataType::Double,
            DataType::Float,
            DataType::Integer,
            DataType::Long,
            DataType::String,
        ]),
    }
}

fn agg_inputs_double(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        key_input(c),
        InputType::Exact(DataType::Double),
        InputType::Exact(DataType::Integer),
        InputType::Exact(DataType::String),
    ]
}

fn agg_inputs_integer(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        key_input(c),
        InputType::Exact(DataType::Integer),
        InputType::Exact(DataType::Integer),
        InputType::Exact(DataType::String),
    ]
}

macro_rules! agg_rules {
    ($( $f:ident, $name:literal, $lg:expr, $mode:expr; )*) => {$(
        /// The result rule of the aggregate: a `binary` once its constants check out.
        pub(super) fn $f(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            if c.args().len() == $mode + 1 {
                if let Some(lg) = $lg { check_lg($name, c, lg)?; }
                check_mode($name, c, $mode)?;
            }
            binary(c)
        }
    )*};
}
agg_rules! {
    sketch_agg_double, "tuple_sketch_agg_double", Some(2), 3;
    sketch_agg_integer, "tuple_sketch_agg_integer", Some(2), 3;
    union_agg_double, "tuple_union_agg_double", Some(1), 2;
    union_agg_integer, "tuple_union_agg_integer", Some(1), 2;
    intersection_agg_double, "tuple_intersection_agg_double", None::<usize>, 1;
    intersection_agg_integer, "tuple_intersection_agg_integer", None::<usize>, 1;
}

/// The signatures of the aggregates' inputs, for `defs/aggregates.rs`.
pub(super) fn sketch_agg_inputs(integer: bool) -> Inputs {
    Inputs::Built(InputBuilder(if integer {
        agg_inputs_integer
    } else {
        agg_inputs_double
    }))
}

/// `tuple_union_agg_*(sketch, lgNomEntries, mode)`.
pub(super) fn union_agg_inputs() -> Inputs {
    Inputs::Expects(vec![
        InputType::Strict(DataType::Binary),
        InputType::Exact(DataType::Integer),
        InputType::Exact(DataType::String),
    ])
}

/// `tuple_intersection_agg_*(sketch, mode)`.
pub(super) fn intersection_agg_inputs() -> Inputs {
    Inputs::Expects(vec![
        InputType::Strict(DataType::Binary),
        InputType::Exact(DataType::String),
    ])
}

macro_rules! scalar_rule {
    ($f:ident, $name:literal, $ty:expr, $lg:expr, $mode:expr) => {
        fn $f(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            if let Some(lg) = $lg {
                if c.args().len() > lg {
                    check_lg($name, c, lg)?;
                }
            }
            if let Some(m) = $mode {
                if c.args().len() > m {
                    check_mode($name, c, m)?;
                }
            }
            Ok(ArgType::new($ty, c.any_nullable()))
        }
    };
}

macro_rules! family {
    ($( $f:ident, $name:literal, $ty:expr, $lg:expr, $mode:expr; )*) => {$(
        scalar_rule!($f, $name, $ty, $lg, $mode);
    )*};
}
family! {
    est_d, "tuple_sketch_estimate_double", DataType::Double, None::<usize>, None::<usize>;
    est_i, "tuple_sketch_estimate_integer", DataType::Double, None::<usize>, None::<usize>;
    th_d, "tuple_sketch_theta_double", DataType::Double, None::<usize>, None::<usize>;
    th_i, "tuple_sketch_theta_integer", DataType::Double, None::<usize>, None::<usize>;
    sum_d, "tuple_sketch_summary_double", DataType::Double, None::<usize>, Some(1usize);
    sum_i, "tuple_sketch_summary_integer", DataType::Long, None::<usize>, Some(1usize);
    un_d, "tuple_union_double", DataType::Binary, Some(2usize), Some(3usize);
    un_i, "tuple_union_integer", DataType::Binary, Some(2usize), Some(3usize);
    unt_d, "tuple_union_theta_double", DataType::Binary, Some(2usize), Some(3usize);
    unt_i, "tuple_union_theta_integer", DataType::Binary, Some(2usize), Some(3usize);
    in_d, "tuple_intersection_double", DataType::Binary, None::<usize>, Some(2usize);
    in_i, "tuple_intersection_integer", DataType::Binary, None::<usize>, Some(2usize);
    int_d, "tuple_intersection_theta_double", DataType::Binary, None::<usize>, Some(2usize);
    int_i, "tuple_intersection_theta_integer", DataType::Binary, None::<usize>, Some(2usize);
    df_d, "tuple_difference_double", DataType::Binary, None::<usize>, None::<usize>;
    df_i, "tuple_difference_integer", DataType::Binary, None::<usize>, None::<usize>;
    dft_d, "tuple_difference_theta_double", DataType::Binary, None::<usize>, None::<usize>;
    dft_i, "tuple_difference_theta_integer", DataType::Binary, None::<usize>, None::<usize>;
}

/// The signatures of the scalar functions.
pub(super) fn specs() -> Vec<Spec> {
    let bin = || InputType::Strict(DataType::Binary);
    let string = || InputType::Exact(DataType::String);
    let one = Inputs::Expects(vec![bin()]);
    let summary = Inputs::Expects(vec![bin(), string()]);
    let two = Inputs::Expects(vec![bin(), bin()]);
    let union = Inputs::Expects(vec![
        bin(),
        bin(),
        InputType::Exact(DataType::Integer),
        string(),
    ]);
    let inter = Inputs::Expects(vec![bin(), bin(), string()]);
    let mut out = Vec::new();
    out.extend(scalar(&["tuple_sketch_estimate_double"], &one, est_d));
    out.extend(scalar(&["tuple_sketch_estimate_integer"], &one, est_i));
    out.extend(scalar(&["tuple_sketch_theta_double"], &one, th_d));
    out.extend(scalar(&["tuple_sketch_theta_integer"], &one, th_i));
    out.extend(scalar(&["tuple_sketch_summary_double"], &summary, sum_d));
    out.extend(scalar(&["tuple_sketch_summary_integer"], &summary, sum_i));
    out.extend(scalar(&["tuple_union_double"], &union, un_d));
    out.extend(scalar(&["tuple_union_integer"], &union, un_i));
    out.extend(scalar(&["tuple_union_theta_double"], &union, unt_d));
    out.extend(scalar(&["tuple_union_theta_integer"], &union, unt_i));
    out.extend(scalar(&["tuple_intersection_double"], &inter, in_d));
    out.extend(scalar(&["tuple_intersection_integer"], &inter, in_i));
    out.extend(scalar(&["tuple_intersection_theta_double"], &inter, int_d));
    out.extend(scalar(&["tuple_intersection_theta_integer"], &inter, int_i));
    out.extend(scalar(&["tuple_difference_double"], &two, df_d));
    out.extend(scalar(&["tuple_difference_integer"], &two, df_i));
    out.extend(scalar(&["tuple_difference_theta_double"], &two, dft_d));
    out.extend(scalar(&["tuple_difference_theta_integer"], &two, dft_i));
    out
}
