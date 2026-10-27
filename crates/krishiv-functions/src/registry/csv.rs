// SPDX-License-Identifier: Apache-2.0
//! `schema_of_csv`'s analysis signature: Spark's `SchemaOfCsv` takes any
//! first argument and checks it by hand (whether it folds, whether it is
//! null, whether it is exactly a `STRING`), and any second argument is
//! refused (options are a `MAP<STRING, STRING>`, which Krishiv cannot type
//! yet, so only `map(...)` could pass it and `map` is unimplemented). The
//! analyzer replaces a well-typed call with the DDL literal, so the call
//! never executes: its execution is `Execution::Evaluator`, which needs no
//! scalar binding.

use krishiv_common::error::{
    DatatypeMismatchNonFoldableInput, DatatypeMismatchUnexpectedInputType,
    DatatypeMismatchUnexpectedNull, InvalidOptionsNonMapFunction, KrishivError,
};
use krishiv_types::DataType;

use super::Execution;
use super::defs::Spec;
use crate::signature::{ArgType, CallTypes, Constant, FunctionKind, Inputs};

/// `schema_of_csv(csv[, options])`: Spark's checks in Spark's order (the
/// options map, then whether the CSV folds, whether it is null, whether it
/// is a string). The DDL itself is resolved by the analyzer.
fn schema_of_csv(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() > 1 {
        return Err(KrishivError::from(InvalidOptionsNonMapFunction {}));
    }
    let sql = |i: usize| c.sql(i).unwrap_or_default();
    let call = format!("\"schema_of_csv({})\"", sql(0));
    let input_type = || {
        format!(
            "\"{}\"",
            c.data_type(0)
                .map_or_else(|| "VOID".to_owned(), DataType::sql)
        )
    };
    // A plain literal the folder has no constant for (a timestamp or a
    // binary) still folds in Spark: it is non-null, so only the type
    // check below can refuse it.
    let folded = c.constant(0);
    if folded.is_none() && !c.is_plain_literal(0) {
        return Err(DatatypeMismatchNonFoldableInput {
            sql_expr: call,
            input_name: "`csv`".to_owned(),
            input_type: input_type(),
            input_expr: format!("\"{}\"", sql(0)),
        }
        .into());
    }
    match folded {
        Some(Constant::Error(e)) => Err(e.clone()),
        Some(Constant::Null) => Err(DatatypeMismatchUnexpectedNull {
            sql_expr: call,
            expr_name: "csv".to_owned(),
        }
        .into()),
        _ => {
            if c.data_type(0) != Some(&DataType::String) {
                return Err(DatatypeMismatchUnexpectedInputType {
                    sql_expr: call,
                    param_index: "first".to_owned(),
                    required_type: "\"STRING\"".to_owned(),
                    input_sql: format!("\"{}\"", sql(0)),
                    input_type: input_type(),
                }
                .into());
            }
            Ok(ArgType::new(DataType::String, false))
        }
    }
}

pub(super) fn specs() -> Vec<Spec> {
    vec![Spec {
        name: "schema_of_csv",
        kind: FunctionKind::Scalar,
        inputs: Inputs::Any,
        result: schema_of_csv,
        execution: Execution::Evaluator,
    }]
}
