// SPDX-License-Identifier: Apache-2.0
//! The misc scalar signatures: `version`, `current_database`
//! (`current_schema`), `current_catalog`, the `input_file_*` placeholders,
//! `monotonically_increasing_id`, `spark_partition_id`, `typeof` and the
//! seeded random functions (`rand` / `random`, `randn`, `uuid`, `uniform`,
//! `randstr`).
//!
//! Ground truth: Spark 4.2's `misc.scala` (`SparkVersion`,
//! `CurrentDatabase`, `CurrentCatalog`, `InputFileName`, `Uuid`),
//! `randomExpressions.scala` (`Rand`, `Randn`, `Uniform`, `RandStr`,
//! `NondeterministicUnaryRDG`), `MonotonicallyIncreasingID.scala`,
//! `SparkPartitionID.scala`, and the `random.sql`,
//! `current_database_catalog.sql` and `misc-functions.sql` goldens (checked
//! against a real Spark 4.2.0 run where the goldens were ambiguous).
//!
//! Arity, determinism and availability come from Spark's catalog (the
//! `FunctionDef` assembly reads them off the TSV row); only the implicit
//! casts (`Inputs`) and the result types live here. Every rule mirrors an
//! error the goldens pin: `rand`/`randn` and `randstr`/`uniform` seeds must
//! be foldable, `uuid`'s seed a plain int/long literal (Spark's
//! `expressionToSeed`), and `uniform`/`randstr` bounds the types Spark's
//! implicit casts accept.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    InvalidParameterValueLong, KrishivError, SeedExpressionIsUnfoldable,
};
use krishiv_types::DataType;

use super::super::uniform::{UniformKind, uniform_result};
use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Constant, Inputs};

/// `version`, `current_database` (`current_schema`), `current_catalog`,
/// `input_file_name`: a non-nullable string; `input_file_block_start` /
/// `input_file_block_length`: a non-nullable `bigint`;
/// `monotonically_increasing_id`: a non-nullable `bigint`;
/// `spark_partition_id`: a non-nullable `int`.
fn string(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, c.any_nullable()))
}

fn long(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Long, c.any_nullable()))
}

fn int(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Integer, c.any_nullable()))
}

/// `typeof(x)`: a non-nullable string. The name itself is resolved by the
/// analyzer (a literal rewrite); the rule only types the call.
fn typeof_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    string(c)
}

/// Whether `t` is an `int` or a `bigint` (what every non-`uuid` seed takes).
fn is_seed(t: Option<&DataType>) -> bool {
    matches!(t, Some(DataType::Integer | DataType::Long) | None)
}

/// The whole call as Spark quotes it in errors (`rand(id)`).
fn call_sql(name: &str, c: &CallTypes<'_>) -> String {
    let args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", args.join(", "))
}

/// `rand(seed?)` / `random(seed?)` / `randn(seed?)`: a non-nullable
/// `double`. The seed must be an `int`/`bigint`
/// (`DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`, as Spark's `Rand` reports
/// it) and foldable (`SEED_EXPRESSION_IS_UNFOLDABLE`); an untyped `NULL`
/// reads as 0.
fn rand_result(name: &'static str, c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !c.args().is_empty() {
        if !is_seed(c.data_type(0)) {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql(name, c),
                param_index: "first".to_owned(),
                required_type: "(\"INT\" or \"BIGINT\")".to_owned(),
                input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
                input_type: c
                    .data_type(0)
                    .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
            }
            .into());
        }
        if c.constant(0).is_none() {
            return Err(SeedExpressionIsUnfoldable {
                seed_expr: format!("\"{}\"", c.sql(0).unwrap_or_default()),
                expr_with_seed: call_sql(name, c),
            }
            .into());
        }
    }
    Ok(ArgType::new(DataType::Double, false))
}

fn rand(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    rand_result("rand", c)
}

fn randn(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    rand_result("randn", c)
}

/// `uuid(seed?)`: a non-nullable string. The seed must be a plain `int` /
/// `bigint` literal (Spark's `expressionToSeed`); a NULL literal draws from
/// the query seed.
fn uuid(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !c.args().is_empty() {
        let literal = c.is_literal(0)
            && matches!(
                (c.data_type(0), c.constant(0)),
                (None, Some(Constant::Null)) | (Some(DataType::Integer | DataType::Long), Some(_))
            );
        if !literal {
            return Err(InvalidParameterValueLong {
                function_name: "`UUID`".to_owned(),
                parameter: "`seed`".to_owned(),
                invalid_value: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            }
            .into());
        }
    }
    Ok(ArgType::new(DataType::String, false))
}

/// Whether `t` is numeric or an untyped `NULL` (what `uniform`'s bounds
/// take; Spark casts the `NULL` to `DOUBLE`).
fn is_uniform_bound(t: Option<&DataType>) -> bool {
    t.is_none_or(|t| {
        matches!(
            t,
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
        )
    })
}

/// The [`UniformKind`] of a `uniform` bound (an untyped `NULL` reads as
/// `DOUBLE`, as Spark casts it).
fn uniform_kind(t: Option<&DataType>) -> UniformKind {
    match t {
        None | Some(DataType::Double) => UniformKind::Double,
        Some(DataType::Byte) => UniformKind::Byte,
        Some(DataType::Short) => UniformKind::Short,
        Some(DataType::Integer) => UniformKind::Int,
        Some(DataType::Long) => UniformKind::Long,
        Some(DataType::Float) => UniformKind::Float,
        Some(DataType::Decimal(d)) => UniformKind::Decimal {
            precision: d.precision(),
            // A decimal scale is `0..=38`: the widening never fires.
            scale: i8::try_from(d.scale()).unwrap_or(i8::MAX),
        },
        Some(_) => UniformKind::Other,
    }
}

/// `uniform(min, max[, seed])`: Spark's `Uniform.dataType` (an untyped
/// `NULL` reads as `DOUBLE`), nullable when an argument may be. A NULL
/// bound is a NULL result; a NULL seed reads as 0.
fn uniform(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    for i in 0..3 {
        if i >= c.args().len() {
            continue;
        }
        let bound = i < 2;
        if bound && !is_uniform_bound(c.data_type(i)) || !bound && !is_seed(c.data_type(i)) {
            return Err(DatatypeMismatchUnexpectedInputType {
                sql_expr: call_sql("uniform", c),
                param_index: ["first", "second", "third"][i].to_owned(),
                required_type: if bound {
                    "\"NUMERIC\"".to_owned()
                } else {
                    "(\"INT\" or \"BIGINT\")".to_owned()
                },
                input_sql: format!("\"{}\"", c.sql(i).unwrap_or_default()),
                input_type: c
                    .data_type(i)
                    .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
            }
            .into());
        }
    }
    for (i, name) in [(0, "min"), (1, "max"), (2, "seed")] {
        if i >= c.args().len() {
            continue;
        }
        match c.constant(i) {
            Some(Constant::Error(e)) => return Err(e.clone()),
            None => {
                return Err(DatatypeMismatchNonFoldableInput {
                    sql_expr: call_sql("uniform", c),
                    input_name: format!("`{name}`"),
                    input_type: "integer or floating-point".to_owned(),
                    input_expr: format!("\"{}\"", c.sql(i).unwrap_or_default()),
                }
                .into());
            }
            Some(_) => {}
        }
    }
    let data_type = match uniform_result(uniform_kind(c.data_type(0)), uniform_kind(c.data_type(1)))
    {
        UniformKind::Byte => DataType::Byte,
        UniformKind::Short => DataType::Short,
        UniformKind::Int => DataType::Integer,
        UniformKind::Long => DataType::Long,
        UniformKind::Float => DataType::Float,
        UniformKind::Double | UniformKind::Other => DataType::Double,
        UniformKind::Decimal { precision, scale } => {
            use krishiv_types::DecimalType;
            // The kind came from a valid decimal bound, so the
            // reconstruction holds; `try_new` guards the impossible fall.
            let scale = u8::try_from(scale).unwrap_or(u8::MAX);
            DataType::Decimal(DecimalType::try_new(
                u32::from(precision),
                u32::from(scale),
            )?)
        }
    };
    Ok(ArgType::new(data_type, c.any_nullable()))
}

/// Whether `t` is a type Spark implicitly casts to `INT` (the numerics and
/// strings, or an untyped `NULL`): what `randstr`'s length takes.
fn is_length(t: Option<&DataType>) -> bool {
    t.is_none_or(|t| {
        matches!(
            t,
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal(_)
                | DataType::String
        )
    })
}

/// `randstr(length[, seed])`: a non-nullable string. The length takes what
/// Spark casts to `INT`; the seed an `int`/`bigint`. Both must be foldable.
fn randstr(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if !is_length(c.data_type(0)) {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql("randstr", c),
            param_index: "first".to_owned(),
            required_type: "\"INT\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: c
                .data_type(0)
                .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
        }
        .into());
    }
    if c.args().len() > 1 && !is_seed(c.data_type(1)) {
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql("randstr", c),
            param_index: "second".to_owned(),
            required_type: "(\"INT\" or \"BIGINT\")".to_owned(),
            input_sql: format!("\"{}\"", c.sql(1).unwrap_or_default()),
            input_type: c
                .data_type(1)
                .map_or_else(|| "\"VOID\"".to_owned(), |t| format!("\"{}\"", t.sql())),
        }
        .into());
    }
    for (i, name) in [(0, "length"), (1, "seed")] {
        if i >= c.args().len() {
            continue;
        }
        match c.constant(i) {
            Some(Constant::Error(e)) => return Err(e.clone()),
            None => {
                return Err(DatatypeMismatchNonFoldableInput {
                    sql_expr: call_sql("randstr", c),
                    input_name: format!("`{name}`"),
                    input_type: "integer".to_owned(),
                    input_expr: format!("\"{}\"", c.sql(i).unwrap_or_default()),
                }
                .into());
            }
            Some(_) => {}
        }
    }
    Ok(ArgType::new(DataType::String, false))
}

/// Every implemented misc function.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(
        &[
            "version",
            "current_database",
            "current_schema",
            "current_catalog",
        ],
        &Inputs::Any,
        string,
    );
    out.extend(scalar(&["input_file_name"], &Inputs::Any, string));
    out.extend(scalar(
        &["input_file_block_start", "input_file_block_length"],
        &Inputs::Any,
        long,
    ));
    out.extend(scalar(&["monotonically_increasing_id"], &Inputs::Any, long));
    out.extend(scalar(&["spark_partition_id"], &Inputs::Any, int));
    out.extend(scalar(&["typeof"], &Inputs::Any, typeof_result));
    out.extend(scalar(&["rand", "random"], &Inputs::Any, rand));
    out.extend(scalar(&["randn"], &Inputs::Any, randn));
    out.extend(scalar(&["uuid"], &Inputs::Any, uuid));
    out.extend(scalar(&["uniform"], &Inputs::Any, uniform));
    out.extend(scalar(&["randstr"], &Inputs::Any, randstr));
    out
}
