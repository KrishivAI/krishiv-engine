// SPDX-License-Identifier: Apache-2.0
//! The objects of the session a permanent view may not reference (review r5 Z2): Spark v4.2.0
//! `views.scala` lines 628-650 (`verifyTemporaryObjectsNotExists`) refuses a permanent view whose
//! query reads a temporary view, then one calling a temporary function, then one reading a
//! temporary variable, each `INVALID_TEMP_OBJ_REFERENCE` with the parameters of
//! `QueryCompilationErrors.notAllowedToCreatePermanentViewByReferencingTemp{View,Func,Var}Error`
//! (lines 3447-3481). Session UDFs (`spark.udf.register`) are temporary functions there. A stored
//! SQL function's body may read no session variable either (review r8 item 2,
//! [`Session::refuse_session_variables`]).

use std::sync::Arc;

use krishiv_common::error::{InvalidTempObjReference, KrishivError};
use krishiv_plan::analyzer::VariableSnapshot;
use krishiv_plan::{Expr, ExprNode, LogicalPlan};

use crate::session::{QueryEnv, Session};

/// `parts` as Spark's `toSQLId` prints a name: each part in backticks (a backtick doubled).
pub(crate) fn sql_id(parts: &[String]) -> String {
    parts
        .iter()
        .map(|p| format!("`{}`", p.replace('`', "``")))
        .collect::<Vec<_>>()
        .join(".")
}

/// The refusal of the permanent view `view` (as Spark prints it) referencing the temporary object
/// `temp` (printed) of kind `kind` (`VIEW`, `FUNCTION`, `VARIABLE`).
pub(crate) fn refusal(view: &str, temp: String, kind: &str) -> KrishivError {
    InvalidTempObjReference {
        obj_name: view.to_owned(),
        obj: "VIEW".to_owned(),
        temp_obj_name: temp,
        temp_obj: kind.to_owned(),
    }
    .into()
}

impl Session {
    /// Refuses the permanent view `view` (printed as Spark does) whose query `plan`, already
    /// analyzed in `env`, calls a temporary function or session UDF, or reads a session variable
    /// (review r5 Z2): functions first, then variables, as Spark checks them.
    ///
    /// A name reads a variable only where no column has it, so a query naming one is analyzed
    /// again with no variable: if that fails, the name it could not resolve is the variable.
    ///
    /// # Errors
    /// `INVALID_TEMP_OBJ_REFERENCE`.
    pub(crate) async fn refuse_session_objects(
        &self,
        view: &str,
        plan: &Arc<LogicalPlan>,
        env: &QueryEnv,
    ) -> Result<(), KrishivError> {
        if let Some(f) = env.sql_functions.temporary_call(plan, self.state.profile) {
            return Err(refusal(view, sql_id(&[f]), "FUNCTION"));
        }
        let Some(first) = variable_name(plan, &env.analysis.variables) else {
            return Ok(());
        };
        let mut without = env.clone();
        without.set_quiet_denials(true, &self.state.policy);
        without.analysis.variables = Arc::new(VariableSnapshot {
            // Never a version a resolution with variables was cached under.
            version: u64::MAX,
            ..VariableSnapshot::default()
        });
        match Box::pin(self.analyze_in(plan, &without, None)).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let unresolved = e
                    .error_class()
                    .name()
                    .starts_with("UNRESOLVED_COLUMN")
                    .then(|| {
                        e.message_parameters()
                            .iter()
                            .find(|(k, _)| *k == "objectName")
                            .map(|(_, v)| v.clone())
                    })
                    .flatten();
                Err(refusal(
                    view,
                    unresolved.unwrap_or_else(|| sql_id(&first)),
                    "VARIABLE",
                ))
            }
        }
    }
}

impl Session {
    /// Refuses the stored function `object` (printed as Spark does) whose body `def`, already
    /// probed in `env`, reads a session variable (review r8 item 2; Spark v4.2.0
    /// `CreateSQLFunctionCommand.scala` lines 388-392, the last of its temporary-object checks).
    /// A name reads a variable only where nothing else has it (a parameter, a column), so a body
    /// naming one is probed again with no variable: if that fails, the name it could not resolve
    /// is the variable.
    ///
    /// # Errors
    /// `INVALID_TEMP_OBJ_REFERENCE`.
    pub(crate) async fn refuse_session_variables(
        &self,
        object: &str,
        def: &krishiv_plan::analyzer::SqlFunction,
        mut env: QueryEnv,
        redacted: &krishiv_common::redact::RedactedSql,
    ) -> Result<(), KrishivError> {
        let Some(first) = variable_name_in_expr(&def.body, &env.analysis.variables) else {
            return Ok(());
        };
        env.set_quiet_denials(true, &self.state.policy);
        env.analysis.variables = Arc::new(VariableSnapshot {
            // Never a version a resolution with variables was cached under.
            version: u64::MAX,
            ..VariableSnapshot::default()
        });
        let Err(e) = Box::pin(self.probe(def, &def.qualifier, env, redacted)).await else {
            return Ok(());
        };
        let unresolved = e
            .error_class()
            .name()
            .starts_with("UNRESOLVED_COLUMN")
            .then(|| {
                e.message_parameters()
                    .iter()
                    .find(|(k, _)| *k == "objectName")
                    .map(|(_, v)| v.clone())
            })
            .flatten();
        Err(krishiv_common::error::InvalidTempObjReference {
            obj_name: object.to_owned(),
            obj: "FUNCTION".to_owned(),
            temp_obj_name: unresolved.unwrap_or_else(|| sql_id(&first)),
            temp_obj: "VARIABLE".to_owned(),
        }
        .into())
    }
}

/// The first name in `plan` (operators top down, as Spark collects them) that a session variable
/// of `variables` could answer: a variable's name, or a struct variable's with fields after it.
fn variable_name(plan: &Arc<LogicalPlan>, variables: &VariableSnapshot) -> Option<Vec<String>> {
    if variables.variables.is_empty() && variables.locals.is_empty() {
        return None;
    }
    let mut plans = vec![plan];
    while let Some(p) = plans.pop() {
        let mut exprs: Vec<&Arc<Expr>> = Vec::new();
        p.for_each_expr(&mut |e| exprs.push(e));
        exprs.reverse();
        while let Some(e) = exprs.pop() {
            if let ExprNode::Column(c) = &e.node {
                if variables.get(&c.name).is_some() {
                    return Some(c.name.clone());
                }
                if let Some((_, used)) = variables.get_struct_prefix(&c.name) {
                    return Some(c.name[..used].to_vec());
                }
            }
            if let ExprNode::Subquery { plan, .. } = &e.node {
                plans.push(plan);
            }
            let mut children = Vec::new();
            e.for_each_child(&mut |c| children.push(c));
            exprs.extend(children.into_iter().rev());
        }
        let mut children = Vec::new();
        p.for_each_child(&mut |c| children.push(c));
        plans.extend(children.into_iter().rev());
    }
    None
}

/// As [`variable_name`], for the expression `expr` (a stored function's body, review r8 item 2):
/// the first name in it, its subqueries' included, that a session variable could answer.
pub(crate) fn variable_name_in_expr(
    expr: &Arc<Expr>,
    variables: &VariableSnapshot,
) -> Option<Vec<String>> {
    if variables.variables.is_empty() && variables.locals.is_empty() {
        return None;
    }
    let mut exprs = vec![expr];
    while let Some(e) = exprs.pop() {
        if let ExprNode::Column(c) = &e.node {
            if variables.get(&c.name).is_some() {
                return Some(c.name.clone());
            }
            if let Some((_, used)) = variables.get_struct_prefix(&c.name) {
                return Some(c.name[..used].to_vec());
            }
        }
        if let ExprNode::Subquery { plan, .. } = &e.node
            && let Some(name) = variable_name(plan, variables)
        {
            return Some(name);
        }
        let mut children = Vec::new();
        e.for_each_child(&mut |c| children.push(c));
        exprs.extend(children.into_iter().rev());
    }
    None
}
