// SPDX-License-Identifier: Apache-2.0
//! Variable values of `ARRAY`, `STRUCT` and `MAP` type (T196): one row of an
//! Arrow column as the unresolved constructor that rebuilds it
//! (`array(..)`, `named_struct(..)`, `map(..)` over literals), cast to the
//! column's type, so a query reading the variable resolves it like any
//! expression. Scalar values stay literals ([`crate::command::arrow_value`]).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::VariableValue;
use krishiv_plan::{CastMode, Expr, ExprNode, Literal};
use krishiv_types::DataType;

use crate::command::arrow_value;

/// Whether `t` is a nested Arrow carrier this module rebuilds.
pub(crate) fn is_nested(t: &ArrowType) -> bool {
    matches!(
        t,
        ArrowType::List(_) | ArrowType::LargeList(_) | ArrowType::Struct(_) | ArrowType::Map(..)
    )
}

/// The value of `array` at `row` with its type: a literal for a scalar
/// type, the constructor of a nested one.
pub(crate) fn arrow_variable_value(
    array: &ArrayRef,
    row: usize,
) -> Result<(DataType, VariableValue), KrishivError> {
    if !is_nested(array.data_type()) {
        return arrow_value(array, row).map(|(t, l)| (t, l.into()));
    }
    let data_type = DataType::from_arrow(array.data_type())?;
    if array.is_null(row) {
        return Ok((data_type.clone(), Literal::Null(Some(data_type)).into()));
    }
    let value = Expr::new(ExprNode::Cast {
        expr: Arc::new(constructor(array, row)?),
        to: data_type.clone(),
        mode: CastMode::Default,
    });
    Ok((data_type, VariableValue::Nested(Arc::new(value))))
}

/// One value of `array` as an expression: a literal, or a constructor.
fn element(array: &ArrayRef, row: usize) -> Result<Expr, KrishivError> {
    if is_nested(array.data_type()) && !array.is_null(row) {
        return constructor(array, row);
    }
    if is_nested(array.data_type()) {
        let t = DataType::from_arrow(array.data_type())?;
        return Ok(Expr::new(ExprNode::Literal(Literal::Null(Some(t)))));
    }
    Ok(Expr::new(ExprNode::Literal(arrow_value(array, row)?.1)))
}

/// The constructor of the non-NULL nested value of `array` at `row`.
fn constructor(array: &ArrayRef, row: usize) -> Result<Expr, KrishivError> {
    let items = |values: &ArrayRef| -> Result<Vec<Expr>, KrishivError> {
        (0..values.len()).map(|i| element(values, i)).collect()
    };
    Ok(match array.data_type() {
        ArrowType::List(_) => Expr::function("array", items(&array.as_list::<i32>().value(row))?),
        ArrowType::LargeList(_) => {
            Expr::function("array", items(&array.as_list::<i64>().value(row))?)
        }
        ArrowType::Struct(fields) => {
            let s = array.as_struct();
            let mut args = Vec::with_capacity(fields.len() * 2);
            for (f, c) in fields.iter().zip(s.columns()) {
                args.push(Expr::new(ExprNode::Literal(Literal::String(
                    f.name().clone(),
                ))));
                args.push(element(c, row)?);
            }
            Expr::function("named_struct", args)
        }
        ArrowType::Map(..) => {
            let entries = array.as_map().value(row);
            let (keys, values) = (entries.column(0), entries.column(1));
            let mut args = Vec::with_capacity(entries.len() * 2);
            for i in 0..entries.len() {
                args.push(element(keys, i)?);
                args.push(element(values, i)?);
            }
            Expr::function("map", args)
        }
        other => {
            return Err(crate::session::internal(format!(
                "a nested variable value of type {other}"
            )));
        }
    })
}
