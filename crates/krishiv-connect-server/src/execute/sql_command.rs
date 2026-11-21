// SPDX-License-Identifier: Apache-2.0
//! `spark.sql()`: the `SqlCommand` of `Plan.command` (design P3-connect-mapping, command 5).
//!
//! The statement runs through [`Session::sql_command`]: a command (`SET`, `CREATE VIEW`, `SHOW`,
//! `INSERT`, ...) runs at once and answers a `LocalRelation` of its result rows; a query is only
//! analyzed (its errors surface here) and answers the `SQL` relation the client sends again as a
//! root relation. The stream is one `sql_command_result`, then, for a command only, one
//! `metrics` response (empty until command metrics exist; Spark sends it when `isCommand` and
//! skips it for a query), then `result_complete` for a reattachable operation: no schema or
//! batch response (Spark's `handleSqlCommand`).
//!
//! The statement is `SqlCommand.input` when set (what PySpark 4.x sends for every `spark.sql()`),
//! else the deprecated `sql` field. A command's result is one IPC stream; it is capped at
//! `0.7 x maxBatchSize` as Spark's single batch is (Spark asserts; Krishiv refuses with a typed
//! error). Unlike Spark's single batch it may hold several record batches.

use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature, OperationCanceled};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::{
    Metrics, ResponseType, ResultComplete, SqlCommandResult,
};
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::plan::compressed_operation::OpType as CompressedOpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_connect::translate::{sql_command_relation, sql_has_parameters};
use krishiv_session::{QueryOptions, Session, SqlOutcome};

use super::{Sender, Stop, arrow, decode_command, decompress};
use crate::operation::ExecuteLimits;

/// The `SqlCommand` of `plan`, when its command is one (a compressed command is decompressed
/// first, off the async workers, bounded by `maxPlanSize`).
pub(super) async fn of(
    plan: &pb::Plan,
    limits: &ExecuteLimits,
) -> Result<Option<pb::SqlCommand>, KrishivError> {
    let command = match &plan.op_type {
        Some(OpType::Command(command)) => match &command.command_type {
            Some(pb::command::CommandType::SqlCommand(sql)) => {
                Some(pb::command::CommandType::SqlCommand(sql.clone()))
            }
            _ => None,
        },
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
                Some(OpType::Command(command)) => match &command.command_type {
                    Some(pb::command::CommandType::SqlCommand(sql)) => {
                        Some(pb::command::CommandType::SqlCommand(sql.clone()))
                    }
                    _ => None,
                },
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

/// What a `sql_command` answers.
pub(super) struct Answer {
    /// The relation of the `sql_command_result`.
    pub(super) relation: pb::Relation,
    /// Whether the statement was a command that ran (a `metrics` response follows).
    pub(super) is_command: bool,
}

/// Runs `command` on `session` under operation `id`: the relation to answer. The session time
/// zone of a command's result is read after the command ran (it may have been `SET`).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an `input` that is not a plain `SQL` relation, for parameters
/// (the analyzer refuses SQL parameters on this base) and for a result over the batch cap;
/// `CONNECT_INVALID_PLAN.SQL_COMMAND_EXPECTS_SQL_OR_WITH_RELATIONS` for another `input`; the
/// statement's parse, analysis and run errors.
pub(super) async fn run(
    session: &Session,
    command: &pb::SqlCommand,
    options: QueryOptions,
    max_batch_size: usize,
) -> Result<Answer, KrishivError> {
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
    match session.sql_command_with(&sql.query, options).await? {
        SqlOutcome::Query(_) => Ok(Answer {
            relation,
            is_command: false,
        }),
        SqlOutcome::Command(df) => {
            let tz: Option<Arc<str>> = session
                .conf()
                .get("spark.sql.session.timeZone")
                .map(Arc::from);
            let tz = tz.as_ref();
            let schema = df.schema_boxed().await?;
            let mut batches = Vec::new();
            for batch in df.collect_boxed().await? {
                batches.push(arrow::spark_batch(&batch, tz)?);
            }
            if batches.is_empty() {
                batches.push(arrow::empty_batch(&schema, tz)?);
            }
            let data = arrow::ipc_stream_of(&batches, None)?;
            drop(batches);
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "Spark's (maxBatchSize * 0.7).toLong on a positive size"
            )]
            let cap = (max_batch_size as f64 * 0.7) as usize;
            if data.len() > cap {
                return Err(KrishivUnsupportedFeature {
                    feature: format!(
                        "a SQL command result of {} bytes, over the {cap}-byte batch cap \
                         (0.7 x spark.connect.grpc.arrow.maxBatchSize)",
                        data.len()
                    ),
                }
                .into());
            }
            Ok(Answer {
                relation: pb::Relation {
                    common: None,
                    rel_type: Some(RelType::LocalRelation(pb::LocalRelation {
                        data: Some(data),
                        schema: None,
                    })),
                },
                is_command: true,
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

/// Streams the answer of a `sql_command` and ends the operation (`Work::Command`).
pub(super) async fn produce(sender: &Sender, answer: Answer) -> Result<(), Stop> {
    sender.admit().await?;
    sender
        .send(
            sender.response(Some(ResponseType::SqlCommandResult(SqlCommandResult {
                relation: Some(answer.relation),
            }))),
        )
        .await?;
    if answer.is_command {
        // The operator tree needs a session hook for a command's metrics: empty until then.
        let mut metrics = sender.response(None);
        metrics.metrics = Some(Metrics::default());
        sender.send(metrics).await?;
    }
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
