// SPDX-License-Identifier: Apache-2.0
//! Python UDFs over Connect (P3-T014, PR-26d; design `P3-T014-connect-udfs-sandbox.md` §2, §3.2), after
//! Spark's `SparkConnectPlanner` at v4.3.0: `transformCommonInlineUserDefinedFunction`,
//! `createUserDefinedPythonFunction` and `handleRegisterUserDefinedFunction`.
//!
//! A `CommonInlineUserDefinedFunction` holding a `PythonUDF` is read into a [`PythonUdf`]: the name, the
//! determinism, Spark's eval type, the declared return type, the client's `command` bytes (cloudpickled
//! `(func, return_type)`, never unpickled here) and the client's Python `major.minor`. The translator never
//! runs or stores one: an inline call asks the [`UdfHost`] the caller passes for the function the call
//! resolves to (the server registers the payload with the session's worker pool and answers with the
//! session's encoded name for it); without a host the call is refused. The Python UDF of a relation
//! (`MapPartitions`, `GroupMap`, `CoGroupMap`, `relation_udf`) is read into the same [`PythonUdf`] and asks the
//! same host for the reference the relation node runs. A `register_function` command is
//! read by [`translate_command`] for the server to run.
//!
//! Supported: the scalar eval types `SQL_BATCHED_UDF` (100, run on the Arrow-batched path as the
//! embedded runtime does), `SQL_ARROW_BATCHED_UDF` (101), `SQL_SCALAR_PANDAS_UDF` (200) and
//! `SQL_SCALAR_ARROW_UDF` (250). Refused with `KRISHIV_UNSUPPORTED_FEATURE`: other eval types (iterator and
//! aggregate UDFs) and `additional_includes` (the 4.3.0 client never sets it; archives are PR-26c). A Scala
//! or Java function answers Spark's `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` naming the case of
//! `CommonInlineUserDefinedFunction.function` (Krishiv runs no JVM code), an unset one
//! `INVALID_ONE_OF_FIELD_NOT_SET`. `is_distinct` is ignored, as Spark ignores it for every function that is
//! not an aggregate.

use std::fmt;

use krishiv_plan::PythonEvalType;
use krishiv_types::DataType;

use super::Result;
use super::errors::{not_set, unsupported, unsupported_field, unsupported_variant};
use super::types::data_type;
use crate::extension::{Extension, Slot, resolve};
use crate::proto::v4_3 as pb;
use crate::proto::v4_3::command::CommandType;
use crate::proto::v4_3::common_inline_user_defined_function::Function;

/// Spark's `SQL_BATCHED_UDF`: a plain `@udf` with Arrow off, and every `spark.udf.register` of a bare Python
/// function.
pub const SQL_BATCHED_UDF: i32 = 100;

/// A Python UDF as a plan or a `register_function` command carries it.
#[derive(Clone, PartialEq, Eq)]
pub struct PythonUdf {
    /// `function_name`: the name it is registered or shown under.
    pub name: String,
    /// `deterministic`.
    pub deterministic: bool,
    /// Spark's eval type as sent (100, 101, 200 or 250 for a call; a relation kind for a relation): what
    /// the worker is told.
    pub eval_type: i32,
    /// The eval type the plan carries: [`SQL_BATCHED_UDF`] runs on the Arrow-batched path.
    pub plan_eval_type: PythonEvalType,
    /// `output_type` (a struct for a relation: its columns).
    pub return_type: DataType,
    /// `command`: the cloudpickled `(func, return_type)`, client code the engine never unpickles.
    pub command: Vec<u8>,
    /// `python_ver`: the client's Python `major.minor`.
    pub python_ver: String,
}

impl fmt::Debug for PythonUdf {
    // No payload bytes: the command is client code, and its size is what matters in a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonUdf")
            .field("name", &self.name)
            .field("deterministic", &self.deterministic)
            .field("eval_type", &self.eval_type)
            .field("return_type", &self.return_type)
            .field("command_len", &self.command.len())
            .field("python_ver", &self.python_ver)
            .finish()
    }
}

/// The function an inline UDF's call resolves to: the qualifier and the name of the IR call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdfCallName {
    /// The qualifier (`["session"]` for the session's encoded names).
    pub qualifier: Vec<String>,
    /// The function name.
    pub name: String,
}

/// Where the translator sends the Python UDFs of a plan, inline calls and relations alike: the server, which
/// owns the session and its UDF workers. Called once per call or relation, in plan order.
pub trait UdfHost: Send {
    /// The function a call of `udf` resolves to. A host that runs nothing (analysis) still answers, so a
    /// plan can be typed; one that will run the plan records `udf` so it can register it once the plan's
    /// privileges are checked.
    ///
    /// # Errors
    /// Whatever the host refuses (no UDF runtime, too many UDFs).
    fn inline_python_udf(&mut self, udf: &PythonUdf) -> Result<UdfCallName>;

    /// The reference a relation (`mapInPandas`, `applyInPandas`, `cogroup().applyInPandas` and their Arrow
    /// forms) names `udf` by. As for [`Self::inline_python_udf`]: a host that runs nothing still answers, and
    /// one that will run the plan records `udf` for registration after the plan's privilege check.
    ///
    /// # Errors
    /// Whatever the host refuses.
    fn relation_python_udf(&mut self, udf: &PythonUdf) -> Result<krishiv_plan::PythonUdfRef>;
}

/// The host of a translation without one: every Python UDF, inline or in a relation, is refused.
pub(crate) fn no_host() -> krishiv_common::error::KrishivError {
    unsupported("Python UDFs outside a Spark Connect server (spark.connect.PythonUDF)")
}

/// A command the server runs, as translated. Exhaustive on purpose: a new arm does not compile until the
/// server runs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// `register_function` with a Python UDF: register `udf` under its name in the session.
    RegisterPythonUdf(PythonUdf),
    /// `checkpoint_command`: checkpoint the command's relation (translated by the server, which
    /// runs the command) and hold the result under a new id.
    Checkpoint(CheckpointSpec),
    /// `remove_cached_remote_relation_command`: forget the frame held under `relation_id` (an id
    /// the session does not hold is not an error, as in Spark).
    RemoveCachedRemoteRelation {
        /// The id to forget.
        relation_id: String,
    },
    /// `write_operation`: write the command's relation (translated by the server, which runs the
    /// command) with a `DataFrameWriter`.
    Write(super::write::WriteSpec),
    /// `write_operation_v2`: write the command's relation with a `DataFrameWriterV2`.
    WriteV2(super::write::WriteV2Spec),
}

/// What a `checkpoint_command` asks besides its relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointSpec {
    /// `local`: a local checkpoint (no checkpoint directory).
    pub local: bool,
    /// `eager`: take the rows now.
    pub eager: bool,
    /// `storage_level`, for a local checkpoint (Spark ignores it for a reliable one).
    pub storage_level: Option<pb::StorageLevel>,
}

/// Translates a command the server runs itself.
///
/// # Errors
/// [`python_udf`]'s errors for `register_function`, the writer-state errors of the write commands
/// ([`super::write::write_spec`], [`super::write::write_v2_spec`]); every other command is refused as
/// [`super::command_not_supported`] refuses it (`INVALID_ONE_OF_FIELD_NOT_SUPPORTED`, or
/// `INVALID_ONE_OF_FIELD_NOT_SET` for an empty one).
pub fn translate_command(command: &pb::Command) -> Result<Command> {
    match &command.command_type {
        Some(CommandType::RegisterFunction(fun)) => python_udf(fun).map(Command::RegisterPythonUdf),
        Some(CommandType::CheckpointCommand(c)) => Ok(Command::Checkpoint(CheckpointSpec {
            local: c.local,
            eager: c.eager,
            storage_level: c.storage_level,
        })),
        Some(CommandType::RemoveCachedRemoteRelationCommand(c)) => {
            Ok(Command::RemoveCachedRemoteRelation {
                relation_id: c
                    .relation
                    .as_ref()
                    .map(|r| r.relation_id.clone())
                    .unwrap_or_default(),
            })
        }
        Some(CommandType::WriteOperation(w)) => super::write::write_spec(w).map(Command::Write),
        Some(CommandType::WriteOperationV2(w)) => {
            super::write::write_v2_spec(w).map(Command::WriteV2)
        }
        // No command extension is registered yet: `resolve` names the capability it lacks, or the
        // slot a registered message does sit in.
        Some(CommandType::Extension(any)) => match resolve(any, Slot::Command)? {
            Extension::Echo(_) => Err(super::command_not_supported(command)),
        },
        _ => Err(super::command_not_supported(command)),
    }
}

/// The Python UDF `fun` defines (its arguments are not read: a registration has none, and a call
/// translates them itself).
///
/// # Errors
/// `INVALID_ONE_OF_FIELD_NOT_SUPPORTED` for a Scala or Java function, `INVALID_ONE_OF_FIELD_NOT_SET` for none,
/// `KRISHIV_UNSUPPORTED_FEATURE` for an eval type other than 100, 101, 200 and 250 or for
/// `additional_includes`, and the errors of the return type's translation.
pub fn python_udf(fun: &pb::CommonInlineUserDefinedFunction) -> Result<PythonUdf> {
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
    let plan_eval_type = scalar_eval_type(udf.eval_type)?;
    let return_type = data_type(udf.output_type.as_ref().unwrap_or(&pb::DataType::default()))?;
    Ok(PythonUdf {
        name: fun.function_name.clone(),
        deterministic: fun.deterministic,
        eval_type: udf.eval_type,
        plan_eval_type,
        return_type,
        command: udf.command.clone(),
        python_ver: udf.python_ver.clone(),
    })
}

/// The plan's eval type for a scalar eval type the pool runs.
fn scalar_eval_type(code: i32) -> Result<PythonEvalType> {
    if code == SQL_BATCHED_UDF {
        return Ok(PythonEvalType::ArrowBatched);
    }
    match PythonEvalType::from_code(code) {
        Some(t) if t.is_scalar() => Ok(t),
        _ => Err(unsupported(format!(
            "Python UDFs of eval type {code} over Spark Connect (spark.connect.PythonUDF.eval_type)"
        ))),
    }
}

#[cfg(test)]
mod tests;
