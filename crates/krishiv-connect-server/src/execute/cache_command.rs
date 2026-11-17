// SPDX-License-Identifier: Apache-2.0
//! The cache commands of `Plan.command` (design P3-T083 §3.4; Spark's `SparkConnectPlanner`
//! `handleCheckpointCommand` and `handleRemoveCachedRemoteRelationCommand`, identical at v4.0.0
//! and v4.2.0): a `checkpoint_command` checkpoints its relation (`DataFrame::checkpoint` or
//! `local_checkpoint`), holds the result under a fresh UUID in the session
//! ([`SessionEntry::cache_frame`], Spark's `dataFrameCache`) and answers one
//! `checkpoint_command_result` holding a `CachedRemoteRelation`; a
//! `remove_cached_remote_relation_command` lets go of the id (an id the session does not hold is
//! not an error). Neither answers a schema, rows or metrics: the stream is the result, then
//! `result_complete` for a reattachable operation.

use std::sync::Arc;

use krishiv_common::error::{KrishivError, OperationCanceled};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::execute_plan_response::{ResponseType, ResultComplete};
use krishiv_connect::translate::CheckpointSpec;
use krishiv_session::DataFrame;

use super::{Sender, Stop, settle};
use crate::cache::storage_level_of;
use crate::operation::{random_uuid, replay::uuid_text};

/// Checkpoints `source` as `spec` says and holds the result in `entry`: the id it is held under.
///
/// # Errors
/// `_LEGACY_ERROR_TEMP_3016` for a reliable checkpoint with no `spark.checkpoint.dir` (for a lazy
/// one too); the errors of running the relation.
pub(super) async fn checkpoint(
    entry: &crate::manager::SessionEntry,
    source: &DataFrame,
    spec: CheckpointSpec,
) -> Result<String, KrishivError> {
    let result = if spec.local {
        source
            .local_checkpoint(
                spec.eager,
                spec.storage_level.as_ref().map(storage_level_of),
            )
            .await?
    } else {
        source.checkpoint(spec.eager).await?
    };
    let id = uuid_text(random_uuid()?);
    entry.cache_frame(id.clone(), result);
    Ok(id)
}

/// Streams the answer of a `checkpoint_command` (the id the result is held under) and ends the
/// operation; `None` is a command that answers nothing but the end.
pub(super) async fn encode(sender: Sender, id: Option<String>) {
    let op = Arc::clone(&sender.op);
    let outcome = tokio::select! {
        biased;
        () = op.closed() => Err(Stop::Gone),
        () = op.cancelled() => Err(Stop::Failed(OperationCanceled {}.into())),
        r = produce(&sender, id) => r,
    };
    settle(&op, outcome);
}

async fn produce(sender: &Sender, id: Option<String>) -> Result<(), Stop> {
    sender.admit().await?;
    if let Some(relation_id) = id {
        sender
            .send(sender.response(Some(ResponseType::CheckpointCommandResult(
                pb::CheckpointCommandResult {
                    relation: Some(pb::CachedRemoteRelation { relation_id }),
                },
            ))))
            .await?;
    }
    // As the batches' `produce` ends: atomic with interrupts.
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
