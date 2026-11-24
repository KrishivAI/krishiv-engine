// SPDX-License-Identifier: Apache-2.0
//! The source of a `MERGE INTO` (`merge.rs`), read twice (find, then rewrite) and so the same
//! rows both times. delta-spark's `auto` rule (`MergeIntoMaterializeSource.scala:275-315`,
//! called from `MergeIntoCommandBase.scala:175-185`) copies the source unless the `MERGE` is
//! insert-only (read once) or the source reads only pinned snapshots of lake tables:
//!
//! - **insert-only**: read as it is, once;
//! - **one bare scan of a pinned table** ([`TableProvider::pinned`](krishiv_exec::planner::TableProvider::pinned): a Delta snapshot at one
//!   version; every column in table order, no pushed filter, no row id): the provider the
//!   statement opened, held as a hidden table, so both reads take that snapshot;
//! - **anything else** (files, a query, a non-deterministic expression, a UDF): streamed into a
//!   spill file of its own, a batch at a time, each batch reserved while it is written, then read
//!   back a batch at a time by both queries ([`SpilledTable`]). Its memory is the batch in
//!   flight; the file is on disk under `krishiv.spill.maxBytes` and goes with the statement.
//!   When the statement is cancelled while a batch is being written, that batch finishes
//!   writing on the blocking pool after the consumer is gone (one batch unaccounted).

use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::PhysNode;
use krishiv_exec::planner::{SpilledTable, TableProviderResolver as _};
use krishiv_memory::{ConsumerId, MemoryPool, Reservation, SpillConfig, SpillManager, SpillWriter};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::batch::Batch;

use crate::dataframe::DataFrame;
use crate::session::{Session, internal};

/// The source as the statement reads it.
pub(super) struct MergeSource {
    /// The rows, the same for every read.
    pub(super) frame: DataFrame,
    /// Its row count, when known without a read.
    pub(super) rows: Option<u64>,
    /// How the source is held, for `materializeSourceTimeMs` and the module docs.
    pub(super) kind: SourceKind,
}

/// How a `MERGE` holds its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SourceKind {
    /// Read as it is (an insert-only `MERGE`).
    AsIs,
    /// A pinned table's snapshot.
    Pinned,
    /// Copied into a spill file.
    Spilled,
}

/// Unregisters the copy's memory consumer on every path, a dropped future included.
struct Unregister {
    memory: Arc<dyn MemoryPool>,
    consumer: ConsumerId,
}

impl Drop for Unregister {
    fn drop(&mut self) {
        self.memory.unregister(self.consumer);
    }
}

impl Session {
    /// The source `df` of a `MERGE`, held as the module docs say.
    pub(super) async fn merge_source(
        &self,
        df: &DataFrame,
        insert_only: bool,
    ) -> Result<MergeSource, KrishivError> {
        if insert_only {
            return Ok(MergeSource {
                frame: df.clone(),
                rows: None,
                kind: SourceKind::AsIs,
            });
        }
        let analysis = df.action_analysis().await?;
        let columns: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
                char_varchar: None,
            })
            .collect();
        let (_, physical) = self.physical(&analysis.analyzed, &analysis.env).await?;
        if let PhysNode::Scan(s) = physical.root()
            && let Ok(provider) = analysis.env.catalog.provider(&s.scan)
            && provider.pinned()
            && bare_scan(s, &provider.scan()?.table_schema(), &columns)
        {
            // A table's statistics may be estimates: the count is read when needed.
            let rows = None;
            let (name, entry) = self
                .state
                .catalog
                .put_hidden_provider(columns.clone(), provider);
            return Ok(MergeSource {
                frame: self.hidden_frame(&name, entry, &columns),
                rows,
                kind: SourceKind::Pinned,
            });
        }
        let table = self.spill_rows(df).await?;
        let rows = Some(table.file().num_rows());
        let (name, entry) = self
            .state
            .catalog
            .put_hidden_provider(columns.clone(), Arc::new(table));
        Ok(MergeSource {
            frame: self.hidden_frame(&name, entry, &columns),
            rows,
            kind: SourceKind::Spilled,
        })
    }

    /// `df`'s rows streamed into a spill file of its own (in the runtime's spill directory,
    /// under its cap), each batch reserved while it is written.
    async fn spill_rows(&self, df: &DataFrame) -> Result<SpilledTable, KrishivError> {
        let state = &self.state;
        let consumer = state.memory.register("merge_source", None);
        let _registered = Unregister {
            memory: Arc::clone(&state.memory),
            consumer,
        };
        let executor = Arc::clone(&state.runtime.executor);
        let config = SpillConfig::from_config(&state.runtime.config);
        let spill = Arc::clone(&state.runtime.spill);
        let writer: Arc<Mutex<Option<SpillWriter>>> = Arc::default();
        let mut stream = df.execute_stream_here(true).await?;
        let mut canonical = None;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let schema = Arc::clone(
                canonical.get_or_insert_with(|| crate::carrier::canonical_schema(batch.schema())),
            );
            let batch = Batch::new(crate::carrier::recarried(&batch, &schema)?);
            let bytes = batch.data().get_array_memory_size();
            // The batch's reservation goes with the batch into the blocking write: dropping
            // this future (a cancel) while the write still runs keeps its bytes reserved until
            // the write ends (#962 review).
            let mut held = Reservation::new(Arc::clone(&state.memory), consumer);
            held.grow(bytes).await.map_err(KrishivError::from)?;
            let (w, s, c) = (Arc::clone(&writer), Arc::clone(&spill), config.clone());
            let scratch = Reservation::new(Arc::clone(&state.memory), consumer);
            let (tx, rx) = tokio::sync::oneshot::channel();
            executor
                .spawn_blocking(Box::new(move || {
                    let out = (|| {
                        let mut slot = w.lock().unwrap_or_else(PoisonError::into_inner);
                        if slot.is_none() {
                            *slot = Some(open_writer(&s, &c, schema, scratch)?);
                        }
                        match slot.as_mut() {
                            Some(writer) => writer.write(&batch).map_err(KrishivError::from),
                            None => Err(internal("the MERGE source's spill file".to_owned())),
                        }
                    })();
                    drop(held);
                    drop(tx.send(out));
                }))
                .await?;
            rx.await.map_err(|_| {
                internal("a MERGE source write ended without a result".to_owned())
            })??;
            #[cfg(test)]
            {
                let armed = AFTER_FIRST_WRITE
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                if let Some((reached, resume)) = armed {
                    let _sent = reached.send(());
                    drop(resume.await);
                }
            }
        }
        let empty = match canonical {
            Some(_) => None,
            None => Some(crate::carrier::canonical_schema(df.schema().await?)),
        };
        let scratch = Reservation::new(Arc::clone(&state.memory), consumer);
        let (tx, rx) = tokio::sync::oneshot::channel();
        executor
            .spawn_blocking(Box::new(move || {
                let out = (|| {
                    let mut slot = writer.lock().unwrap_or_else(PoisonError::into_inner);
                    let w = match (slot.take(), empty) {
                        (Some(w), _) => w,
                        (None, Some(schema)) => open_writer(&spill, &config, schema, scratch)?,
                        (None, None) => {
                            return Err(internal("the MERGE source's spill file".to_owned()));
                        }
                    };
                    w.finish().map_err(KrishivError::from)
                })();
                drop(tx.send(out));
            }))
            .await?;
        let file = rx
            .await
            .map_err(|_| internal("a MERGE source write ended without a result".to_owned()))??;
        Ok(SpilledTable::new(file))
    }
}

/// Test hook: told once a spilled source's first batch is written; the copy then waits for the
/// second channel (the test cancels in between) and goes on, so the cancel falls right after the
/// first spill write, not wherever a time limit falls (#962 review).
#[cfg(test)]
pub(crate) type AfterFirstWrite = (
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
);

/// The armed [`AfterFirstWrite`] hook, taken by the first spill write that finds it.
#[cfg(test)]
pub(crate) static AFTER_FIRST_WRITE: Mutex<Option<AfterFirstWrite>> = Mutex::new(None);

/// A writer in a new scope of the runtime's spill directory (opened on first use).
fn open_writer(
    spill: &Mutex<Option<SpillManager>>,
    config: &SpillConfig,
    schema: arrow_schema::SchemaRef,
    scratch: Reservation,
) -> Result<SpillWriter, KrishivError> {
    let scope = {
        let mut slot = spill.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(SpillManager::open(config)?);
        }
        match slot.as_ref() {
            Some(manager) => manager.scope()?,
            None => return Err(internal("the spill directory is not open".to_owned())),
        }
    };
    Ok(scope.writer(schema, scratch.for_spill_scratch())?)
}

/// Whether `s` reads its whole table as the source's columns: every column, in table order,
/// no pushed filter (an exact one has no `Filter` above it), no row id and no runtime filter.
/// Only such a scan can stand in for the source as its provider: anything else is copied.
fn bare_scan(
    s: &krishiv_exec::physical::ScanNode,
    table: &arrow_schema::SchemaRef,
    columns: &[TableColumn],
) -> bool {
    let names: Vec<&str> = table.fields().iter().map(|f| f.name().as_str()).collect();
    s.filter.is_none()
        && !s.row_id
        && s.runtime_filters.is_empty()
        && names.len() == columns.len()
        && names.iter().zip(columns).all(|(n, c)| *n == c.name)
        && s.projection.len() == names.len()
        && s.projection
            .iter()
            .zip(&names)
            .all(|(p, n)| *p == krishiv_io::ColumnPath::new(*n))
}
