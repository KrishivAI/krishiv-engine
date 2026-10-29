// SPDX-License-Identifier: Apache-2.0
//! The struct functions' analysis signatures: `named_struct` and `struct`
//! (Spark 4.2's `CreateNamedStruct` / `CreateStruct`).
//!
//! Ground truth: the function-matrix fixture
//! `tests/fixtures/spark-4.2.0/function-matrix/cases/struct.jsonl` (real
//! Spark 4.2 runs: every name/value type, the two analysis error classes,
//! nulls and nesting) and the golden `struct.sql` statements.
//!
//! - `named_struct(name1, val1, ...)`: an even number of arguments; every odd
//!   (name) argument must be a foldable `STRING`
//!   (`DATATYPE_MISMATCH.CREATE_NAMED_STRUCT_WITHOUT_FOLDABLE_STRING`),
//!   and a null name is
//!   `DATATYPE_MISMATCH.UNEXPECTED_NULL`. Field names are the folded names;
//!   field types are the value types as they are (no coercion:
//!   `Inputs::Any`).
//! - `struct(v1, ...)`: any number of arguments of any type; field names are
//!   `col1`, `col2`, ... (a sibling syntax names children after themselves,
//!   resolved by the analyzer).
//! - Both results are non-nullable structs (a struct value is never null:
//!   nulls live in its fields).

use krishiv_common::error::{
    DatatypeMismatchCreateNamedStructWithoutFoldableString, DatatypeMismatchUnexpectedNull,
    KrishivError, WrongNumArgsWithoutSuggestion,
};
use krishiv_types::{DataType, StructField, StructType};

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Inputs};

/// Spark's documentation root, as the arity error cites it.
const SPARK_DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

/// `named_struct(name, value, ...)`: the struct of the value arguments with
/// the folded name arguments as field names.
fn named_struct(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let n = c.args().len();
    if !n.is_multiple_of(2) {
        return Err(WrongNumArgsWithoutSuggestion {
            function_name: "`named_struct`".to_owned(),
            expected_num: "2n (n > 0)".to_owned(),
            actual_num: n.to_string(),
            docroot: SPARK_DOC_ROOT.to_owned(),
        }
        .into());
    }
    let mut fields = Vec::with_capacity(n / 2);
    let mut null_names = Vec::new();
    let mut bad_names = Vec::new();
    for i in (0..n).step_by(2) {
        // Only a foldable `STRING` names a field; an untyped `NULL` is not
        // one (a typed null name is `UNEXPECTED_NULL` below).
        let is_string = matches!(c.data_type(i), Some(DataType::String));
        let name = match (c.constant(i), is_string) {
            (Some(crate::signature::Constant::String(s)), _) => s.clone(),
            (Some(crate::signature::Constant::Null), true) => {
                null_names.push(c.sql(i).unwrap_or_default());
                continue;
            }
            _ => {
                bad_names.push(c.sql(i).unwrap_or_default());
                continue;
            }
        };
        let (data_type, nullable) = match c.data_type(i + 1) {
            Some(t) => (t.clone(), c.args()[i + 1].nullable),
            None => (DataType::Null, true),
        };
        fields.push(StructField::new(name, data_type, nullable));
    }
    if !bad_names.is_empty() {
        let quoted: Vec<String> = bad_names.iter().map(|s| format!("\"{s}\"")).collect();
        return Err(DatatypeMismatchCreateNamedStructWithoutFoldableString {
            sql_expr: format!("\"named_struct({})\"", struct_sql(c)),
            input_exprs: format!("[{}]", quoted.join(", ")),
        }
        .into());
    }
    if !null_names.is_empty() {
        let quoted: Vec<String> = null_names.iter().map(|s| format!("\"{s}\"")).collect();
        return Err(DatatypeMismatchUnexpectedNull {
            sql_expr: format!("\"named_struct({})\"", struct_sql(c)),
            expr_name: format!("[{}]", quoted.join(", ")),
        }
        .into());
    }
    Ok(ArgType::new(
        DataType::Struct(StructType::new(fields)),
        false,
    ))
}

/// The shown argument list of the call, for error quotes.
fn struct_sql(c: &CallTypes<'_>) -> String {
    (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect::<Vec<_>>()
        .join(", ")
}

/// `struct(v, ...)`: the struct of the arguments as `col1`, `col2`, ....
fn make_struct(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let fields: Vec<StructField> = c
        .args()
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let (data_type, nullable) = match &a.data_type {
                Some(t) => (t.clone(), a.nullable),
                None => (DataType::Null, true),
            };
            StructField::new(format!("col{}", i + 1), data_type, nullable)
        })
        .collect();
    Ok(ArgType::new(
        DataType::Struct(StructType::new(fields)),
        false,
    ))
}

pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["named_struct"], &Inputs::Any, named_struct);
    out.extend(scalar(&["struct"], &Inputs::Any, make_struct));
    out
}
