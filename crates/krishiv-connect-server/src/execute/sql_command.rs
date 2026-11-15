// SPDX-License-Identifier: Apache-2.0
//! `spark.sql()`: the `SqlCommand` of `Plan.command` (design P3-connect-mapping, command 5).
//!
//! The statement runs through [`Session::sql_command`]: a command (`SET`, `CREATE VIEW`, `SHOW`,
//! `INSERT`, ...) runs at once and answers a `LocalRelation` of its result rows; a query is only
//! analyzed (its errors surface here) and answers the `SQL` relation the client sends again as a
//! root relation. The stream is one `sql_command_result`, then `result_complete` for a
//! reattachable operation: no schema, batch or metrics response (Spark's `handleSqlCommand`).

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature, OperationCanceled};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::{
    ResponseType, ResultComplete, SqlCommandResult,
};
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::plan::compressed_operation::OpType as CompressedOpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::translate::{sql_command_relation, sql_has_parameters};
use krishiv_session::{QueryOptions, Session, SqlOutcome};

use super::{Sender, Stop, arrow, decode_command, decompress, settle};
use crate::operation::ExecuteLimits;

/// The `SqlCommand` of `plan`, when its command is one (a compressed command is decompressed
/// first, off the async workers, bounded by `maxPlanSize`).
pub(super) async fn of(
    plan: &pb::Plan,
    limits: &ExecuteLimits,
) -> Result<Option<pb::SqlCommand>, KrishivError> {
    let command = match &plan.op_type {
        Some(OpType::Command(command)) => command.command_type.clone(),
        Some(OpType::CompressedOperation(c))
            if CompressedOpType::try_from(c.op_type) == Ok(CompressedOpType::Command) =>
        {
            let c = (**c).clone();
            let max = limits.max_plan_size;
            let bytes = tokio::task::spawn_blocking(move || decompress(&c, max))
                .await
                .map_err(|e| -> KrishivError {
                    krishiv_common::error::InternalError {
                        message: format!("decompressing a plan: {e}"),
                    }
                    .into()
                })??;
            match &decode_command(&bytes)?.op_type {
                Some(OpType::Command(command)) => command.command_type.clone(),
                _ => None,
            }
        }
        _ => None,
    };
    Ok(match command {
        Some(pb::command::CommandType::SqlCommand(sql)) => Some(*sql),
        _ => None,
    })
}

/// Runs `command` on `session` under operation `id`: the relation to answer.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for `input` and for parameters (the analyzer refuses SQL
/// parameters on this base); the statement's parse, analysis and run errors.
pub(super) async fn run(
    session: &Session,
    command: &pb::SqlCommand,
    id: &str,
    tz: Option<&Arc<str>>,
) -> Result<pb::Relation, KrishivError> {
    let relation = sql_command_relation(command)?;
    let Some(RelType::Sql(sql)) = &relation.rel_type else {
        return Err(internal(
            "a SQL command did not translate to a SQL relation",
        ));
    };
    if sql_has_parameters(sql) {
        return Err(KrishivUnsupportedFeature {
            feature: "parameterised SQL (spark.connect.SqlCommand.args)".to_owned(),
        }
        .into());
    }
    match session
        .sql_command_with(&sql.query, QueryOptions::new().operation_id(id))
        .await?
    {
        SqlOutcome::Query(_) => Ok(relation),
        SqlOutcome::Command(df) => {
            let schema = df.schema().await?;
            let mut batches = Vec::new();
            for batch in df.collect().await? {
                batches.push(arrow::spark_batch(&batch, tz)?);
            }
            if batches.is_empty() {
                batches.push(arrow::empty_batch(&schema, tz)?);
            }
            let data = arrow::ipc_stream_of(&batches, None)?;
            Ok(pb::Relation {
                common: None,
                rel_type: Some(RelType::LocalRelation(pb::LocalRelation {
                    data: Some(data),
                    schema: None,
                })),
            })
        }
        _ => Err(internal("an unknown outcome of a SQL command")),
    }
}

fn internal(message: &str) -> KrishivError {
    krishiv_common::error::InternalError {
        message: message.to_owned(),
    }
    .into()
}

/// Streams the answer of a `sql_command` and ends the operation.
pub(super) async fn encode(sender: Sender, relation: pb::Relation) {
    let op = Arc::clone(&sender.op);
    let outcome = tokio::select! {
        biased;
        () = op.closed() => Err(Stop::Gone),
        () = op.cancelled() => Err(Stop::Failed(OperationCanceled {}.into())),
        r = produce(&sender, relation) => r,
    };
    settle(&op, outcome);
}

async fn produce(sender: &Sender, relation: pb::Relation) -> Result<(), Stop> {
    sender.admit().await?;
    sender
        .send(
            sender.response(Some(ResponseType::SqlCommandResult(SqlCommandResult {
                relation: Some(relation),
            }))),
        )
        .await?;
    // As `produce` of the batches ends: atomic with interrupts.
    if !sender.op.finish() {
        return Err(Stop::Failed(OperationCanceled {}.into()));
    }
    if sender.op.reattachable() {
        sender
            .send(sender.response(Some(ResponseType::ResultComplete(ResultComplete {}))))
            .await?;
    }
    sender.op.replay().finish(None);
    Ok(())
}
