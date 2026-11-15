// SPDX-License-Identifier: Apache-2.0
//! The commit loop (design note P3-T021 section 5.2; see the parent module).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, MapArray, RecordBatch, StructArray};
use arrow_schema::DataType as ArrowType;
use delta_kernel::committer::FileSystemCommitter;
use delta_kernel::expressions::{MapData, Scalar};
use delta_kernel::schema::{DataType, MapType, StructField, StructType};
use delta_kernel::transaction::CommitResult;
use delta_kernel::{FilteredEngineData, Snapshot, SnapshotRef};
use krishiv_common::error::{
    DeltaMaxCommitRetriesExceeded, InternalError, KrishivError, OperationCanceled,
};

use super::conflict::{self, CurrentTxn, PartitionColumn};
use super::winner::{self, FileAction};
use super::{Committed, RetryLimits, StagedCommit, StagedRemoves, WRITE_ID_TAG};
use crate::engine::eval::create_many;
use crate::engine::{DeltaEngine, KrishivEngineData};
use crate::error::from_kernel;
use crate::features::{WriteProtocol, check_write};

/// Winners read per log-pool call; more are read on the next round.
const WINNERS_PER_READ: usize = 1000;

/// What one attempt came to.
enum Outcome {
    Committed { version: u64, since_checkpoint: u64 },
    Conflicted(u64),
    Retryable(KrishivError),
    Failed(KrishivError),
}

/// What every attempt shares.
struct Prepared {
    staged: StagedCommit,
    engine_info: String,
    blind_append: bool,
    read_version: u64,
}

/// Commits `staged` on `snapshot`, the snapshot its statement read (see the parent module).
///
/// # Errors
/// The writer allow-list's refusals; a conflict's class; `DELTA_MAX_COMMIT_RETRIES_EXCEEDED`;
/// `OPERATION_CANCELED`; a store failure, also one of the landed check (the commit's outcome is
/// then unknown, and it is never retried blind).
pub async fn commit(
    engine: &DeltaEngine,
    snapshot: SnapshotRef,
    staged: StagedCommit,
    limits: RetryLimits,
) -> Result<Committed, KrishivError> {
    let started = Instant::now();
    let read_protocol = WriteProtocol::of(&snapshot);
    check_write(&staged.table, &read_protocol)?;
    let partition_columns = partition_columns(&snapshot, &staged.table)?;
    let removes = staged
        .removes
        .iter()
        .map(remove_actions)
        .collect::<Result<Vec<_>, _>>()?
        .concat();
    let read_version = snapshot.version();
    let blind_append = removes.is_empty() && !staged.read_set.reads_table();
    let prepared = Arc::new(Prepared {
        engine_info: format!("krishiv {}", env!("CARGO_PKG_VERSION")),
        blind_append,
        read_version,
        staged,
    });
    let txn = CurrentTxn {
        table: &prepared.staged.table,
        read_protocol: &read_protocol,
        partition_columns: &partition_columns,
        read_set: &prepared.staged.read_set,
        removes: &removes,
        changes_metadata: false,
        changes_protocol: false,
        domains: &[],
        blind_append,
    };
    let start_version = read_version + 1;
    let mut attempt_version = start_version;
    let mut snapshot = snapshot;
    let mut attempts: u64 = 0;
    let mut conflicts: u32 = 0;
    loop {
        if attempts > limits.max_attempts || (attempts > 0 && started.elapsed() > limits.timeout) {
            return Err(retries_exceeded(
                attempts,
                start_version,
                attempt_version,
                &prepared,
                &removes,
                started,
            ));
        }
        attempts += 1;
        let (p, s) = (Arc::clone(&prepared), Arc::clone(&snapshot));
        let outcome = engine.delta_blocking(move |e| attempt(e, s, &p)).await??;
        match outcome {
            Outcome::Committed {
                version,
                since_checkpoint,
            } => {
                return Ok(Committed {
                    version,
                    attempts,
                    found_by_landed_check: false,
                    commits_since_checkpoint: Some(since_checkpoint),
                });
            }
            Outcome::Conflicted(version) => {
                let table = prepared.staged.table.clone();
                let winners = engine
                    .delta_blocking(move |e| {
                        winner::read_from(e, version, WINNERS_PER_READ, &table)
                    })
                    .await??;
                let Some(last) = winners.last().map(|w| w.version) else {
                    return Err(internal(&format!(
                        "the commit at version {version} conflicted but no commit file is there"
                    )));
                };
                for w in &winners {
                    if w.tag(WRITE_ID_TAG) == Some(prepared.staged.write_id.as_str()) {
                        return Ok(landed(w.version, attempts));
                    }
                    conflict::check(&txn, w)?;
                }
                snapshot = snapshot_at(engine, last, &prepared.staged.table).await?;
                attempt_version = last + 1;
                conflicts = conflicts.saturating_add(1);
                backoff(engine, &limits, &prepared.staged.write_id, conflicts).await?;
            }
            Outcome::Retryable(error) => {
                if let Some(v) = landed_check(engine, &prepared).await? {
                    return Ok(landed(v, attempts));
                }
                // The commit file was not written: the same transaction again.
                drop(error);
                conflicts = conflicts.saturating_add(1);
                backoff(engine, &limits, &prepared.staged.write_id, conflicts).await?;
            }
            Outcome::Failed(error) => {
                return match landed_check(engine, &prepared).await? {
                    Some(v) => Ok(landed(v, attempts)),
                    None => Err(error),
                };
            }
        }
    }
}

fn landed(version: u64, attempts: u64) -> Committed {
    Committed {
        version,
        attempts,
        found_by_landed_check: true,
        commits_since_checkpoint: None,
    }
}

/// One attempt, on the log pool.
fn attempt(e: &DeltaEngine, snapshot: SnapshotRef, p: &Prepared) -> Result<Outcome, KrishivError> {
    let table = p.staged.table.as_str();
    let kernel = |err: &delta_kernel::KernelError| from_kernel(err, table);
    let mut builder = snapshot
        .update_table(p.staged.operation.kernel(), p.engine_info.clone(), true)
        .with_commit_info(
            Box::new(KrishivEngineData::new(commit_info_row(p)?)),
            commit_info_schema(),
        );
    if !p.staged.operation_parameters.is_empty() {
        builder = builder.with_operation_parameters(p.staged.operation_parameters.clone());
    }
    if !p.staged.operation_metrics.is_empty() {
        builder = builder.with_operation_metrics(p.staged.operation_metrics.clone());
    }
    if p.blind_append {
        builder = builder.with_blind_append();
    }
    let mut txn = builder
        .build(e, Box::new(FileSystemCommitter::new()))
        .map_err(|err| kernel(&err))?;
    for add in &p.staged.adds {
        txn.add_files(Box::new(KrishivEngineData::new(add.clone())));
    }
    for r in &p.staged.removes {
        let rows = FilteredEngineData::try_new(
            Box::new(KrishivEngineData::new(r.scan_rows.clone())),
            r.selection.clone(),
        )
        .map_err(|err| kernel(&err))?;
        txn.remove_files(rows);
    }
    Ok(match txn.commit(e) {
        Ok(CommitResult::Committed(c)) => Outcome::Committed {
            version: c.commit_version(),
            since_checkpoint: c.post_commit_stats().commits_since_checkpoint,
        },
        Ok(CommitResult::Conflicted(c)) => Outcome::Conflicted(c.conflict_version()),
        Ok(CommitResult::Retryable(r)) => Outcome::Retryable(kernel(&r.error)),
        Err(err) => Outcome::Failed(kernel(&err)),
    })
}

/// The commit's version if a commit from the read version on carries this write's id.
async fn landed_check(
    engine: &DeltaEngine,
    p: &Arc<Prepared>,
) -> Result<Option<u64>, KrishivError> {
    let mut from = p.read_version + 1;
    loop {
        let table = p.staged.table.clone();
        let commits = engine
            .delta_blocking(move |e| winner::read_from(e, from, WINNERS_PER_READ, &table))
            .await??;
        if let Some(w) = commits
            .iter()
            .find(|w| w.tag(WRITE_ID_TAG) == Some(p.staged.write_id.as_str()))
        {
            return Ok(Some(w.version));
        }
        match commits.last() {
            Some(last) if commits.len() == WINNERS_PER_READ => from = last.version + 1,
            _ => return Ok(None),
        }
    }
}

/// The table's snapshot at `version`, on the log pool.
async fn snapshot_at(
    engine: &DeltaEngine,
    version: u64,
    table: &str,
) -> Result<SnapshotRef, KrishivError> {
    let root = engine.table_root().as_str().to_owned();
    let table = table.to_owned();
    engine
        .delta_blocking(move |e| {
            Snapshot::builder_for(&root)
                .with_cancellation_token(e.kernel_cancellation())
                .at_version(version)
                .build(e)
                .map_err(|err| from_kernel(&err, &table))
        })
        .await?
}

/// Waits before the next attempt after the `n`th conflict: `backoff_min` doubled per conflict up
/// to `backoff_max`, less a jitter of up to half derived from the write id; ends early with
/// `OPERATION_CANCELED` when the query is cancelled.
async fn backoff(
    engine: &DeltaEngine,
    limits: &RetryLimits,
    write_id: &str,
    n: u32,
) -> Result<(), KrishivError> {
    let wait = delay(limits, write_id, n);
    tokio::select! {
        biased;
        () = engine.cancelled() => Err(OperationCanceled {}.into()),
        () = tokio::time::sleep(wait) => Ok(()),
    }
}

/// The wait after the `n`th conflict (see [`backoff`]).
pub(super) fn delay(limits: &RetryLimits, write_id: &str, n: u32) -> Duration {
    let doubled = limits
        .backoff_min
        .saturating_mul(1u32 << n.saturating_sub(1).min(20));
    let full = doubled.min(limits.backoff_max);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (write_id, n).hash(&mut h);
    // A fraction in [0, 1/2) of the wait, the same for one writer's nth wait.
    let jitter = full.mul_f64((h.finish() % 1000) as f64 / 2000.0);
    full.saturating_sub(jitter)
}

fn retries_exceeded(
    attempts: u64,
    start_version: u64,
    fail_version: u64,
    p: &Prepared,
    removes: &[FileAction],
    started: Instant,
) -> KrishivError {
    let adds: usize = p.staged.adds.iter().map(RecordBatch::num_rows).sum();
    // The adds, the removes and the commitInfo.
    let actions = adds + removes.len() + 1;
    DeltaMaxCommitRetriesExceeded {
        num_attempts: attempts.to_string(),
        start_version: start_version.to_string(),
        fail_version: fail_version.to_string(),
        num_actions: actions.to_string(),
        time_spent: started.elapsed().as_millis().to_string(),
    }
    .into()
}

/// The schema of Krishiv's commitInfo row.
fn commit_info_schema() -> Arc<StructType> {
    Arc::new(StructType::new_unchecked([
        StructField::nullable("isolationLevel", DataType::STRING),
        StructField::nullable("readVersion", DataType::LONG),
        StructField::nullable(
            "tags",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
    ]))
}

/// Krishiv's commitInfo row: `isolationLevel`, `readVersion` and the write id's tag.
fn commit_info_row(p: &Prepared) -> Result<RecordBatch, KrishivError> {
    let table = p.staged.table.as_str();
    let tags = MapData::try_new(
        MapType::new(DataType::STRING, DataType::STRING, true),
        [(
            Scalar::String(WRITE_ID_TAG.to_owned()),
            Scalar::String(p.staged.write_id.clone()),
        )],
    )
    .map_err(|e| from_kernel(&e, table))?;
    let read_version = i64::try_from(p.read_version)
        .map_err(|_| internal("the read version does not fit a long"))?;
    create_many(
        &commit_info_schema(),
        &[vec![
            Scalar::String("Serializable".to_owned()),
            Scalar::Long(read_version),
            Scalar::Map(tags),
        ]],
    )
    .map_err(|e| from_kernel(&e, table))
}

/// The partition columns of `snapshot`, in schema order.
fn partition_columns(
    snapshot: &Snapshot,
    table: &str,
) -> Result<Vec<PartitionColumn>, KrishivError> {
    let tc = snapshot.table_configuration();
    let mode = tc.column_mapping_mode();
    let schema = snapshot.schema();
    let names = tc.metadata().partition_columns();
    schema
        .fields()
        .filter(|f| names.iter().any(|n| n == f.name()))
        .map(|f| {
            Ok(PartitionColumn {
                name: f.name().clone(),
                physical: f.physical_name(mode).to_owned(),
                data_type: f.data_type().clone(),
            })
        })
        .collect::<Result<Vec<_>, KrishivError>>()
        .and_then(|columns| {
            if columns.len() == names.len() {
                Ok(columns)
            } else {
                Err(internal(&format!(
                    "a partition column of {table} is not in its schema"
                )))
            }
        })
}

/// The removed files of `r`: their paths and partition values.
pub(super) fn remove_actions(r: &StagedRemoves) -> Result<Vec<FileAction>, KrishivError> {
    let rows = &r.scan_rows;
    if r.selection.len() > rows.num_rows() {
        return Err(internal("a remove selection is longer than its scan rows"));
    }
    let paths = strings(
        rows.column_by_name("path")
            .ok_or_else(|| internal("scan rows without a path column"))?,
    )?;
    let values = rows
        .column_by_name("fileConstantValues")
        .and_then(|c| c.as_any().downcast_ref::<StructArray>())
        .and_then(|s| s.column_by_name("partitionValues"))
        .and_then(|c| c.as_any().downcast_ref::<MapArray>());
    let mut out = Vec::new();
    for (i, selected) in r.selection.iter().enumerate() {
        if !*selected {
            continue;
        }
        let path = paths
            .get(i)
            .cloned()
            .flatten()
            .ok_or_else(|| internal("a removed scan row has no path"))?;
        let partition_values = match values {
            Some(m) if m.is_valid(i) => map_row(m, i)?,
            _ => HashMap::new(),
        };
        out.push(FileAction {
            path,
            partition_values,
        });
    }
    Ok(out)
}

fn map_row(m: &MapArray, row: usize) -> Result<HashMap<String, Option<String>>, KrishivError> {
    let entry = m.value(row);
    let keys = strings(entry.column(0))?;
    let values = strings(entry.column(1))?;
    Ok(keys
        .into_iter()
        .zip(values)
        .filter_map(|(k, v)| Some((k?, v)))
        .collect())
}

fn strings(a: &ArrayRef) -> Result<Vec<Option<String>>, KrishivError> {
    let a = arrow_cast::cast(a, &ArrowType::Utf8)
        .map_err(|e| internal(&format!("a scan row string column: {e}")))?;
    let a = a.as_string::<i32>();
    Ok((0..a.len())
        .map(|i| a.is_valid(i).then(|| a.value(i).to_owned()))
        .collect())
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}
