// SPDX-License-Identifier: Apache-2.0
//! Python table functions (UDTFs) over Connect (P3-T014, PR-26c-4), after Spark's `SparkConnectPlanner` at
//! v4.3.0: `createPythonUserDefinedTableFunction`, `handleRegisterUserDefinedTableFunction` (command 10,
//! `register_table_function`) and `transformCommonInlineUserDefinedTableFunction` (relation 38,
//! `common_inline_user_defined_table_function`, a DataFrame call of a UDTF object).
//!
//! A `CommonInlineUserDefinedTableFunction` holding a `PythonUDTF` is read into a [`PythonUdtf`]: the name, the
//! determinism, the eval type **as the client sent it** (300, 301 or 302: the client decides it from its
//! `spark.sql.execution.pythonUDTF.arrow.enabled` and the decorator; never re-derived here), the declared
//! result (a struct, an `unparsed` type string parsed as the planner's `parseDatatypeString` parses it, or none for a UDTF whose static `analyze` gives it per call), the client's `command`
//! bytes (the cloudpickled class, never unpickled here) and the client's Python `major.minor`. A return type
//! that is not a struct is Spark's `INVALID_SCHEMA_TYPE_NON_STRUCT`; an eval type that is no table kind is
//! `KRISHIV_UNSUPPORTED_FEATURE` (Spark passes any code to its worker, which then fails).
//!
//! An inline call asks the [`UdfHost`](super::UdfHost) for the name the session resolves it by (the server
//! registers it as a hidden, per-definition UDTF of the session) and becomes [`PlanNode::TableFunction`]
//! over that name, the node SQL's `FROM f(..)` lowers to, so analysis (including `analyze`) and planning are
//! SQL's. A `NamedArgumentExpression` argument is an [`ExprNode::Alias`], as SQL lowers `f(n => 3)`; a plain
//! alias is refused (the analyzer would read it as a parameter name). A `TABLE` argument (`df.asTable()`, a
//! `SubqueryExpression` of type `TABLE_ARG`, which also needs `with_relations`) is refused with
//! `KRISHIV_UNSUPPORTED_FEATURE`.

use std::fmt;
use std::sync::Arc;

use krishiv_common::error::InvalidSchemaTypeNonStruct;
use krishiv_plan::{Expr, ExprNode, PlanNode, PythonEvalType, PythonUdfKind};
use krishiv_types::DataType;

use super::errors::{not_set, unsupported, unsupported_field};
use super::types::data_type;
use super::{Result, Translator};
use crate::proto::v4_3 as pb;
use crate::proto::v4_3::common_inline_user_defined_table_function::Function;
use crate::proto::v4_3::expression::ExprType;
use crate::proto::v4_3::subquery_expression::SubqueryType;

/// A Python UDTF as `register_table_function` or an inline call carries it.
#[derive(Clone, PartialEq, Eq)]
pub struct PythonUdtf {
    /// `function_name`: the name it is registered or shown under.
    pub name: String,
    /// `deterministic`.
    pub deterministic: bool,
    /// Spark's eval type as sent (300, 301 or 302): what the worker is told.
    pub eval_type: i32,
    /// The same eval type as the plan carries it.
    pub plan_eval_type: PythonEvalType,
    /// `return_type`: a struct, or `None` when the UDTF's `analyze` gives it per call.
    pub return_type: Option<DataType>,
    /// `command`: the cloudpickled class, client code the engine never unpickles.
    pub command: Vec<u8>,
    /// `python_ver`: the client's Python `major.minor`.
    pub python_ver: String,
}

impl fmt::Debug for PythonUdtf {
    // No payload bytes: the command is client code, and its size is what matters in a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonUdtf")
            .field("name", &self.name)
            .field("deterministic", &self.deterministic)
            .field("eval_type", &self.eval_type)
            .field("return_type", &self.return_type)
            .field("command_len", &self.command.len())
            .field("python_ver", &self.python_ver)
            .finish()
    }
}

/// The Python UDTF `fun` defines (its arguments are not read: a registration has none, and a call
/// translates them itself).
///
/// # Errors
/// `INVALID_ONE_OF_FIELD_NOT_SET` for no function, `KRISHIV_UNSUPPORTED_FEATURE` for an eval type other than
/// 300, 301 and 302, `INVALID_SCHEMA_TYPE_NON_STRUCT` for a return type that is not a struct, and the errors
/// of the return type's translation.
pub fn python_udtf(fun: &pb::CommonInlineUserDefinedTableFunction) -> Result<PythonUdtf> {
    let Some(Function::PythonUdtf(udtf)) = &fun.function else {
        return Err(not_set(
            "spark.connect.CommonInlineUserDefinedTableFunction",
            "FUNCTION_NOT_SET",
        ));
    };
    let plan_eval_type = PythonEvalType::from_code(udtf.eval_type)
        .filter(|t| t.kind() == PythonUdfKind::Udtf)
        .ok_or_else(|| {
            unsupported(format!(
                "Python UDTFs of eval type {} over Spark Connect (spark.connect.PythonUDTF.eval_type)",
                udtf.eval_type
            ))
        })?;
    let return_type = match &udtf.return_type {
        None => None,
        Some(t) => match transform_data_type(t)? {
            s @ DataType::Struct(_) => Some(s),
            other => {
                return Err(InvalidSchemaTypeNonStruct {
                    data_type: format!("\"{}\"", other.sql()),
                }
                .into());
            }
        },
    };
    Ok(PythonUdtf {
        name: fun.function_name.clone(),
        deterministic: fun.deterministic,
        eval_type: udtf.eval_type,
        plan_eval_type,
        return_type,
        command: udtf.command.clone(),
        python_ver: udtf.python_ver.clone(),
    })
}

/// The planner's `transformDataType` (v4.3.0): an `unparsed` type (what a client sends for a string
/// `returnType`, `@udtf(returnType="a int")`) through `parseDatatypeString`, any other through the proto's
/// type conversion.
pub(super) fn transform_data_type(t: &pb::DataType) -> Result<DataType> {
    match &t.kind {
        Some(pb::data_type::Kind::Unparsed(u)) => {
            super::schema::parse_datatype_string(&u.data_type_string)
        }
        _ => data_type(t),
    }
}

impl Translator<'_> {
    /// `common_inline_user_defined_table_function`: the table function the host names, over the call's
    /// arguments.
    pub(super) fn inline_udtf(
        &mut self,
        fun: &pb::CommonInlineUserDefinedTableFunction,
    ) -> Result<PlanNode> {
        const MESSAGE: &str = "CommonInlineUserDefinedTableFunction";
        let udtf = python_udtf(fun)?;
        let mut args = Vec::with_capacity(fun.arguments.len());
        for a in &fun.arguments {
            match &a.expr_type {
                Some(ExprType::SubqueryExpression(s))
                    if s.subquery_type == SubqueryType::TableArg as i32 =>
                {
                    return Err(unsupported_field(
                        MESSAGE,
                        "arguments",
                        "TABLE arguments of Python UDTFs (DataFrame.asTable())",
                    ));
                }
                Some(ExprType::NamedArgumentExpression(n)) => {
                    let value = self.required_expression(n.value.as_deref())?;
                    args.push(Arc::new(Expr::new(ExprNode::Alias {
                        expr: value,
                        name: n.key.clone(),
                    })));
                }
                _ => {
                    let e = self.expression(a)?;
                    if matches!(e.node, ExprNode::Alias { .. }) {
                        return Err(unsupported_field(
                            MESSAGE,
                            "arguments",
                            "aliased arguments of Python UDTFs",
                        ));
                    }
                    args.push(e);
                }
            }
        }
        let name = match self.udfs.as_deref_mut() {
            Some(host) => host.inline_python_udtf(&udtf)?,
            None => return Err(super::udf::no_host()),
        };
        Ok(PlanNode::TableFunction { name, args })
    }
}

#[cfg(test)]
mod tests;
