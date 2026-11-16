// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relations over Connect (P3-T014, PR-26d part 2; design `P3-T014-connect-udfs-sandbox.md`
//! §2, §3.2 and `P3-T085-embedded-python-udf-kinds.md` §3.1), after Spark's `SparkConnectPlanner` at
//! v4.3.0 (`transformMapPartitions`, `transformGroupMap`, `transformCoGroupMap`):
//!
//! - `MapPartitions` (28): `mapInPandas` (eval type 205) and `mapInArrow` (207) into
//!   [`PlanNode::MapPartitions`]. `is_barrier = true` and `profile_id` (stage-level scheduling) are
//!   refused with `KRISHIV_UNSUPPORTED_FEATURE`.
//! - `GroupMap` (31): `applyInPandas` (201, iterator form 216) and `applyInArrow` (209, 215) into
//!   [`PlanNode::GroupMap`] over the grouping expressions. The `transformWithState` eval types (211-214) and
//!   every field of the Scala and stateful APIs (`sorting_expressions`, `initial_input`,
//!   `initial_grouping_expressions`, `is_map_groups_with_state`, `output_mode`, `timeout_conf`,
//!   `state_schema`, `transform_with_state_info`) are refused with `KRISHIV_UNSUPPORTED_FEATURE` (streaming
//!   state is P4); Spark ignores the Scala-only ones for a Python UDF, a stock PySpark client never sets them.
//! - `CoGroupMap` (32): `cogroup().applyInPandas` (206) and `applyInArrow` (210) into
//!   [`PlanNode::CoGroupMap`]; `input_sorting_expressions` / `other_sorting_expressions` (Scala) are refused,
//!   and so are sides with different key counts.
//!
//! Any other eval type is Spark's `CONNECT_INVALID_PLAN.FUNCTION_EVAL_TYPE_NOT_SUPPORTED`
//! (`InvalidInputErrors.functionEvalTypeNotSupported`); a Scala or Java function is
//! `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` (Krishiv runs no JVM code), an unset one `INVALID_ONE_OF_FIELD_NOT_SET`.
//! The UDF's `output_type` must be a struct: its fields are the node's columns (Spark's `toAttributes`).
//!
//! The function itself is never run or stored here: the translator asks the [`RelationUdfHost`] the caller
//! passes for the [`PythonUdfRef`] the plan names it by (the server registers the payload with the session's
//! workers under that name and version); without a host the relation is refused. The UDF's `arguments`
//! (the DataFrame's columns, as PySpark sends them) are translated and dropped: the node passes every column
//! of its input (`group_ops.py` L305), and Spark's cogroup ignores them too.
//!
//! The analyzer resolves these nodes (inputs, keys, output ids) and refuses a nondeterministic grouped or
//! cogrouped map there, as Spark's `CheckAnalysis` does.

use std::fmt;
use std::sync::Arc;

use krishiv_common::error::ConnectInvalidPlanFunctionEvalTypeNotSupported;
use krishiv_plan::{AttrId, BoundColumn, LogicalPlan, PlanNode, PythonEvalType, PythonUdfRef};
use krishiv_types::DataType;

use super::errors::{assertion, not_set, unsupported, unsupported_field, unsupported_variant};
use super::types::data_type;
use super::{Result, Translator};
use crate::proto::v4_3 as pb;
use crate::proto::v4_3::common_inline_user_defined_function::Function;

/// A Python UDF a relation runs (`MapPartitions`, `GroupMap`, `CoGroupMap`), as the plan carries it.
#[derive(Clone, PartialEq, Eq)]
pub struct RelationUdf {
    /// `function_name`.
    pub name: String,
    /// `deterministic`.
    pub deterministic: bool,
    /// Spark's eval type as sent: what the worker is told.
    pub eval_type: i32,
    /// The same eval type as the plan names it.
    pub plan_eval_type: PythonEvalType,
    /// `output_type`: a struct, the relation's columns.
    pub return_type: DataType,
    /// `command`: the cloudpickled `(func, return_type)`, client code the engine never unpickles.
    pub command: Vec<u8>,
    /// `python_ver`: the client's Python `major.minor`.
    pub python_ver: String,
}

impl fmt::Debug for RelationUdf {
    // No payload bytes: the command is client code, and its size is what matters in a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelationUdf")
            .field("name", &self.name)
            .field("deterministic", &self.deterministic)
            .field("eval_type", &self.eval_type)
            .field("return_type", &self.return_type)
            .field("command_len", &self.command.len())
            .field("python_ver", &self.python_ver)
            .finish()
    }
}

/// Where the translator sends the Python UDFs of a plan's relations: the server, which owns the session and
/// its UDF workers. Called once per relation, in plan order.
pub trait RelationUdfHost: Send {
    /// The reference the plan names `udf` by. A host that runs nothing (analysis) still answers, so a plan
    /// can be typed; one that will run the plan records `udf` so it can register it once the plan's
    /// privileges are checked.
    ///
    /// # Errors
    /// Whatever the host refuses (no UDF runtime, too many UDFs).
    fn relation_python_udf(&mut self, udf: &RelationUdf) -> Result<PythonUdfRef>;
}

/// A relation's UDF without a host.
fn no_host() -> krishiv_common::error::KrishivError {
    unsupported("Python UDF relations outside a Spark Connect server (spark.connect.PythonUDF)")
}

/// The `transformWithState` eval types (`SQL_TRANSFORM_WITH_STATE_*`, 211-214), which a `GroupMap` may carry.
const TRANSFORM_WITH_STATE: [i32; 4] = [211, 212, 213, 214];

/// What a relation's UDF translates to: the plan's reference and the declared columns.
struct Translated {
    udf: PythonUdfRef,
    output: Vec<BoundColumn>,
}

impl Translator<'_> {
    /// `MapPartitions`.
    pub(super) fn map_partitions(&mut self, m: &pb::MapPartitions) -> Result<PlanNode> {
        let input = self.udf_input("MapPartitions", m.input.as_deref())?;
        if m.is_barrier == Some(true) {
            return Err(unsupported_field(
                "MapPartitions",
                "is_barrier",
                "barrier execution mode of mapInPandas and mapInArrow",
            ));
        }
        if m.profile_id.is_some() {
            return Err(unsupported_field(
                "MapPartitions",
                "profile_id",
                "stage-level scheduling resource profiles",
            ));
        }
        let fun = required_udf("MapPartitions", m.func.as_ref())?;
        let t = self.relation_udf(fun, true, |code| {
            matches!(code, 205 | 207)
                .then(|| PythonEvalType::from_code(code))
                .flatten()
        })?;
        Ok(PlanNode::MapPartitions {
            input,
            udf: t.udf,
            output: t.output,
            barrier: false,
        })
    }

    /// `GroupMap` with a Python function.
    pub(super) fn group_map(&mut self, g: &pb::GroupMap) -> Result<PlanNode> {
        for (set, field, what) in [
            (
                !g.sorting_expressions.is_empty(),
                "sorting_expressions",
                "sorted grouped maps (the Scala API)",
            ),
            (
                g.initial_input.is_some(),
                "initial_input",
                "initial state of a stateful grouped map",
            ),
            (
                !g.initial_grouping_expressions.is_empty(),
                "initial_grouping_expressions",
                "initial state of a stateful grouped map",
            ),
            (
                g.is_map_groups_with_state.is_some(),
                "is_map_groups_with_state",
                "stateful grouped maps",
            ),
            (
                g.output_mode.is_some(),
                "output_mode",
                "stateful grouped maps",
            ),
            (
                g.timeout_conf.is_some(),
                "timeout_conf",
                "stateful grouped maps",
            ),
            (
                g.state_schema.is_some(),
                "state_schema",
                "stateful grouped maps",
            ),
            (
                g.transform_with_state_info.is_some(),
                "transform_with_state_info",
                "transformWithState",
            ),
        ] {
            if set {
                return Err(unsupported_field("GroupMap", field, what));
            }
        }
        let fun = required_udf("GroupMap", g.func.as_ref())?;
        let input = self.udf_input("GroupMap", g.input.as_deref())?;
        let keys = self.udf_keys(&g.grouping_expressions)?;
        let t = self.relation_udf(fun, true, |code| {
            if TRANSFORM_WITH_STATE.contains(&code) {
                return None;
            }
            matches!(code, 201 | 209 | 215 | 216)
                .then(|| PythonEvalType::from_code(code))
                .flatten()
        })?;
        Ok(PlanNode::GroupMap {
            input,
            keys,
            udf: t.udf,
            output: t.output,
        })
    }

    /// `CoGroupMap` with a Python function.
    pub(super) fn cogroup_map(&mut self, g: &pb::CoGroupMap) -> Result<PlanNode> {
        for (set, field) in [
            (
                !g.input_sorting_expressions.is_empty(),
                "input_sorting_expressions",
            ),
            (
                !g.other_sorting_expressions.is_empty(),
                "other_sorting_expressions",
            ),
        ] {
            if set {
                return Err(unsupported_field(
                    "CoGroupMap",
                    field,
                    "sorted cogrouped maps (the Scala API)",
                ));
            }
        }
        let fun = required_udf("CoGroupMap", g.func.as_ref())?;
        let left_keys = self.udf_keys(&g.input_grouping_expressions)?;
        let right_keys = self.udf_keys(&g.other_grouping_expressions)?;
        if left_keys.len() != right_keys.len() {
            return Err(unsupported(
                "cogrouped maps whose sides have different key counts \
                 (spark.connect.CoGroupMap.other_grouping_expressions)",
            ));
        }
        let left = self.udf_input("CoGroupMap", g.input.as_deref())?;
        let right = self.udf_input("CoGroupMap", g.other.as_deref())?;
        // Spark builds the cogroup's UDF over both sides' columns, not its `arguments`.
        let t = self.relation_udf(fun, false, |code| {
            matches!(code, 206 | 210)
                .then(|| PythonEvalType::from_code(code))
                .flatten()
        })?;
        Ok(PlanNode::CoGroupMap {
            left,
            right,
            left_keys,
            right_keys,
            udf: t.udf,
            output: t.output,
        })
    }

    fn udf_input(
        &mut self,
        message: &str,
        input: Option<&pb::Relation>,
    ) -> Result<Arc<LogicalPlan>> {
        match input {
            Some(rel) => self.relation(rel),
            None => Err(assertion(format!("{message} needs an input relation"))),
        }
    }

    fn udf_keys(&mut self, list: &[pb::Expression]) -> Result<Vec<Arc<krishiv_plan::Expr>>> {
        list.iter().map(|e| self.expression(e)).collect()
    }

    /// The Python UDF `fun` holds, with an eval type `accept` maps to the plan's, sent to the host; its
    /// `arguments` are translated (and dropped) when `read_args`.
    fn relation_udf(
        &mut self,
        fun: &pb::CommonInlineUserDefinedFunction,
        read_args: bool,
        accept: impl Fn(i32) -> Option<PythonEvalType>,
    ) -> Result<Translated> {
        let udf = relation_udf(fun, accept)?;
        if read_args {
            for a in &fun.arguments {
                self.expression(a)?;
            }
        }
        let DataType::Struct(st) = &udf.return_type else {
            return Err(assertion(format!(
                "The returnType of the udf must be a struct, got {}",
                udf.return_type.sql()
            )));
        };
        let output = st
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| BoundColumn {
                // Ids of the translator's choosing: the analyzer keeps them once per plan.
                id: AttrId(i as u64),
                name: f.name().to_owned(),
                data_type: f.data_type().clone(),
                nullable: f.nullable(),
            })
            .collect();
        let reference = match self.relation_udfs.as_deref_mut() {
            Some(host) => host.relation_python_udf(&udf)?,
            None => return Err(no_host()),
        };
        Ok(Translated {
            udf: reference,
            output,
        })
    }
}

/// The function of a relation, which the protocol requires.
fn required_udf<'a>(
    message: &str,
    func: Option<&'a pb::CommonInlineUserDefinedFunction>,
) -> Result<&'a pb::CommonInlineUserDefinedFunction> {
    func.ok_or_else(|| assertion(format!("{message} needs a function")))
}

/// The Python UDF of a relation, read from `fun`.
///
/// # Errors
/// `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` for a Scala or Java function, `INVALID_ONE_OF_FIELD_NOT_SET` for none,
/// `KRISHIV_UNSUPPORTED_FEATURE` for `additional_includes` and the `transformWithState` eval types,
/// `CONNECT_INVALID_PLAN.FUNCTION_EVAL_TYPE_NOT_SUPPORTED` for an eval type `accept` refuses, and the errors of
/// the return type's translation.
fn relation_udf(
    fun: &pb::CommonInlineUserDefinedFunction,
    accept: impl Fn(i32) -> Option<PythonEvalType>,
) -> Result<RelationUdf> {
    const PARENT: &str = "spark.connect.CommonInlineUserDefinedFunction";
    let udf = match &fun.function {
        Some(Function::PythonUdf(udf)) => udf,
        Some(other @ (Function::ScalarScalaUdf(_) | Function::JavaUdf(_))) => {
            return Err(unsupported_variant(PARENT, other.variant()));
        }
        None => return Err(not_set(PARENT, "FUNCTION_NOT_SET")),
    };
    if !udf.additional_includes.is_empty() {
        return Err(unsupported_field(
            "PythonUDF",
            "additional_includes",
            "additional Python includes",
        ));
    }
    let code = udf.eval_type;
    let Some(plan_eval_type) = accept(code) else {
        if TRANSFORM_WITH_STATE.contains(&code) {
            return Err(unsupported(format!(
                "transformWithState (Python eval type {code}, spark.connect.PythonUDF.eval_type)"
            )));
        }
        return Err(ConnectInvalidPlanFunctionEvalTypeNotSupported {
            eval_type: code.to_string(),
        }
        .into());
    };
    let return_type = data_type(udf.output_type.as_ref().unwrap_or(&pb::DataType::default()))?;
    Ok(RelationUdf {
        name: fun.function_name.clone(),
        deterministic: fun.deterministic,
        eval_type: code,
        plan_eval_type,
        return_type,
        command: udf.command.clone(),
        python_ver: udf.python_ver.clone(),
    })
}

#[cfg(test)]
mod tests;
