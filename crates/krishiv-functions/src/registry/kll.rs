// SPDX-License-Identifier: Apache-2.0
//! The KLL scalar functions (Spark 4.1+, `kllExpressions.scala`): `kll_sketch_to_string_*` (`string`),
//! `kll_sketch_get_n_*` (`bigint`), `kll_sketch_merge_*` (`binary`), `kll_sketch_get_quantile_*` and
//! `kll_sketch_get_rank_*` for `bigint`, `float` and `double` sketches. The quantile's rank and the rank's quantile
//! are a constant, a single value or an array (`NON_FOLDABLE_INPUT` otherwise); the result is an array of the
//! same length for an array. NULL in, NULL out.

use krishiv_common::error::{DatatypeMismatchNonFoldableInput, KrishivError};
use krishiv_types::{ArrayType, DataType};

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, InputBuilder, InputType, Inputs};

fn call_sql(name: &str, c: &CallTypes<'_>) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

fn array_of(t: DataType) -> DataType {
    DataType::Array(ArrayType::new(t, false).into())
}

/// The second argument of `get_quantile` (`double`) and `get_rank` (the sketch's element type): the type or an
/// array of it, as Spark's `TypeCollection` takes it.
fn second(c: &CallTypes<'_>, elem: DataType) -> InputType {
    match c.data_type(1) {
        Some(t @ DataType::Array(a)) if *a.element_type() == elem => InputType::Strict(t.clone()),
        _ => InputType::StrictOneOf(vec![elem.clone(), array_of(elem)]),
    }
}

fn quantile_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    vec![
        InputType::Strict(DataType::Binary),
        second(c, DataType::Double),
    ]
}

macro_rules! rank_inputs {
    ($f:ident, $t:expr) => {
        fn $f(c: &CallTypes<'_>) -> Vec<InputType> {
            vec![InputType::Strict(DataType::Binary), second(c, $t)]
        }
    };
}
rank_inputs!(rank_bigint, DataType::Long);
rank_inputs!(rank_float, DataType::Float);
rank_inputs!(rank_double, DataType::Double);

fn check_foldable(name: &str, c: &CallTypes<'_>) -> Result<(), KrishivError> {
    if c.constant(1).is_none() && c.args().len() == 2 {
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call_sql(name, c),
            input_name: "`quantile`".to_owned(),
            input_type: c
                .data_type(1)
                .map(|t| format!("\"{}\"", t.sql()))
                .unwrap_or_default(),
            input_expr: format!("\"{}\"", c.sql(1).unwrap_or_default()),
        }
        .into());
    }
    Ok(())
}

macro_rules! quantile_rule {
    ($f:ident, $name:literal, $elem:expr) => {
        fn $f(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            check_foldable($name, c)?;
            let t = if matches!(c.data_type(1), Some(DataType::Array(_))) {
                array_of($elem)
            } else {
                $elem
            };
            Ok(ArgType::new(t, c.any_nullable()))
        }
    };
}
quantile_rule!(
    quantile_bigint,
    "kll_sketch_get_quantile_bigint",
    DataType::Long
);
quantile_rule!(
    quantile_float,
    "kll_sketch_get_quantile_float",
    DataType::Float
);
quantile_rule!(
    quantile_double,
    "kll_sketch_get_quantile_double",
    DataType::Double
);

macro_rules! rank_rule {
    ($f:ident, $name:literal) => {
        fn $f(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
            check_foldable($name, c)?;
            let t = if matches!(c.data_type(1), Some(DataType::Array(_))) {
                array_of(DataType::Double)
            } else {
                DataType::Double
            };
            Ok(ArgType::new(t, c.any_nullable()))
        }
    };
}
rank_rule!(rank_rule_bigint, "kll_sketch_get_rank_bigint");
rank_rule!(rank_rule_float, "kll_sketch_get_rank_float");
rank_rule!(rank_rule_double, "kll_sketch_get_rank_double");

fn string(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, c.any_nullable()))
}
fn long(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, c.any_nullable()))
}
fn binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, c.any_nullable()))
}

/// The signatures of this family.
pub(super) fn specs() -> Vec<Spec> {
    let bin = || InputType::Strict(DataType::Binary);
    let one = Inputs::Expects(vec![bin()]);
    let two = Inputs::Expects(vec![bin(), bin()]);
    let mut out = scalar(
        &[
            "kll_sketch_to_string_bigint",
            "kll_sketch_to_string_float",
            "kll_sketch_to_string_double",
        ],
        &one,
        string,
    );
    out.extend(scalar(
        &[
            "kll_sketch_get_n_bigint",
            "kll_sketch_get_n_float",
            "kll_sketch_get_n_double",
        ],
        &one,
        long,
    ));
    out.extend(scalar(
        &[
            "kll_sketch_merge_bigint",
            "kll_sketch_merge_float",
            "kll_sketch_merge_double",
        ],
        &two,
        binary,
    ));
    let q = Inputs::Built(InputBuilder(quantile_inputs));
    out.extend(scalar(
        &["kll_sketch_get_quantile_bigint"],
        &q,
        quantile_bigint,
    ));
    out.extend(scalar(
        &["kll_sketch_get_quantile_float"],
        &q,
        quantile_float,
    ));
    out.extend(scalar(
        &["kll_sketch_get_quantile_double"],
        &q,
        quantile_double,
    ));
    out.extend(scalar(
        &["kll_sketch_get_rank_bigint"],
        &Inputs::Built(InputBuilder(rank_bigint)),
        rank_rule_bigint,
    ));
    out.extend(scalar(
        &["kll_sketch_get_rank_float"],
        &Inputs::Built(InputBuilder(rank_float)),
        rank_rule_float,
    ));
    out.extend(scalar(
        &["kll_sketch_get_rank_double"],
        &Inputs::Built(InputBuilder(rank_double)),
        rank_rule_double,
    ));
    out
}
