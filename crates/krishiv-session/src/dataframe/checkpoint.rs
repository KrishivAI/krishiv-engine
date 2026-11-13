// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::checkpoint` and `local_checkpoint` (design P3-T083 §3.3): a snapshot of the rows that
//! cuts the lineage. The result reads a hidden table (the one a script result uses: held by the
//! frame and every frame derived from it), so a write to a source, `CLEAR CACHE`, `REFRESH` and
//! `unpersist` never reach it and `is_cached` is false. With `eager = false` the rows are taken
//! when the result is first analyzed (its first action, or the first call that needs its schema):
//! `LazyCheckpoint` stands in for the snapshot until then.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use krishiv_common::config::CHECKPOINT_DIR;
use krishiv_common::error::{KrishivError, LegacyErrorTemp3016};
use krishiv_memory::{Reservation, SpillConfig, SpillFile, SpillManager};
use krishiv_plan::{LogicalPlan, Origin, PlanId};
use krishiv_types::batch::Batch;

use super::{DataFrame, StorageLevel};
use crate::cache::{CacheDisk, batch_bytes};
use crate::session::Session;

/// A checkpoint whose rows are not taken yet: `marker` is the plan id of the node standing in for
/// the snapshot in the result's plan, `source` the frame that produces the rows.
#[derive(Debug)]
pub(super) struct LazyCheckpoint {
    source: DataFrame,
    dir: Option<PathBuf>,
    marker: PlanId,
    taken: futures::lock::Mutex<Option<DataFrame>>,
}

/// The file a reliable checkpoint wrote; it and its directory go when the snapshot drops.
#[derive(Debug)]
struct CheckpointFile {
    _file: SpillFile,
    _spill: Arc<Mutex<Option<SpillManager>>>,
}

impl LazyCheckpoint {
    /// The snapshot frame, taking the rows on the first call; later calls (and a concurrent one,
    /// which waits) read the same rows.
    async fn frame(&self) -> Result<DataFrame, KrishivError> {
        let mut taken = self.taken.lock().await;
        if let Some(frame) = &*taken {
            return Ok(frame.clone());
        }
        let frame = self.source.take_snapshot(self.dir.clone()).await?;
        *taken = Some(frame.clone());
        Ok(frame)
    }
}

impl DataFrame {
    /// A reliable checkpoint (Spark's `checkpoint`): the rows are written under the checkpoint
    /// directory (`spark.checkpoint.dir`) as well as held, and the result does not see later
    /// changes to this frame's sources. `eager = false` takes the rows at the result's first
    /// action. The files are deleted when the result and every frame derived from it are gone.
    ///
    /// # Errors
    /// `_LEGACY_ERROR_TEMP_3016` when no checkpoint directory is set (for `eager = false` too);
    /// analysis and execution errors, and an unwritable directory.
    pub async fn checkpoint(&self, eager: bool) -> Result<DataFrame, KrishivError> {
        let dir = self
            .session
            .state
            .conf
            .snapshot()
            .get(&CHECKPOINT_DIR)
            .and_then(|v| v.as_str())
            .map(PathBuf::from);
        let Some(dir) = dir else {
            return Err(LegacyErrorTemp3016 {}.into());
        };
        self.checkpoint_to(eager, Some(dir)).await
    }

    /// A local checkpoint (Spark's `localCheckpoint`): the rows are held by the session, no
    /// directory is needed. `level` is accepted as Spark's overload takes it; the rows are held in
    /// the session's relation store whatever it is.
    ///
    /// # Errors
    /// Analysis and execution errors.
    pub async fn local_checkpoint(
        &self,
        eager: bool,
        level: Option<StorageLevel>,
    ) -> Result<DataFrame, KrishivError> {
        let _ = level;
        self.checkpoint_to(eager, None).await
    }

    async fn checkpoint_to(
        &self,
        eager: bool,
        dir: Option<PathBuf>,
    ) -> Result<DataFrame, KrishivError> {
        if let Some(error) = &self.invalid {
            return Err(error.clone());
        }
        if eager {
            return self.take_snapshot(dir).await;
        }
        // Analyzed now, as Spark's `checkpoint` does: an invalid plan fails here.
        self.analysis().await?;
        let marker = self.session.next_plan_id();
        let plan = LogicalPlan::clone(&self.plan).with_origin(Origin::plan(marker));
        let mut out = Self {
            plan: Arc::new(plan),
            analysis: Arc::default(),
            trace: None,
            ..self.clone()
        };
        out.keep.push(Arc::new(LazyCheckpoint {
            source: self.clone(),
            dir,
            marker,
            taken: futures::lock::Mutex::new(None),
        }));
        Ok(out)
    }

    /// Runs this frame and holds its rows: the frame a checkpoint returns.
    async fn take_snapshot(&self, dir: Option<PathBuf>) -> Result<DataFrame, KrishivError> {
        let schema = self.schema().await?;
        let batches: Vec<RecordBatch> = self.collect().await?;
        // The analysed schema can name a type (`Utf8View`) the result batches carry as another
        // (`Utf8`, a table registered by a client); the snapshot's table takes the batches' types.
        let schema = match batches.first() {
            Some(first) => {
                let fields: Vec<_> = schema
                    .fields()
                    .iter()
                    .zip(first.schema().fields())
                    .map(|(f, b)| {
                        Arc::new(f.as_ref().clone().with_data_type(b.data_type().clone()))
                    })
                    .collect();
                Arc::new(arrow_schema::Schema::new_with_metadata(
                    fields,
                    schema.metadata().clone(),
                ))
            }
            None => schema,
        };
        let state = &self.session.state;
        let memory_bytes: usize = batches.iter().map(batch_bytes).sum();
        let file = match dir {
            Some(dir) => Some(write_file(&self.session, dir, &schema, &batches).await?),
            None => None,
        };
        let disk_bytes = file.as_ref().map_or(0, |(f, _)| f.bytes());
        let mut frame = self.session.script_result(schema, batches).await?;
        frame.keep.push(Arc::new(
            state
                .cache
                .checkpoints
                .add(u64::try_from(memory_bytes).unwrap_or(u64::MAX), disk_bytes),
        ));
        if let Some((file, spill)) = file {
            frame.keep.push(Arc::new(CheckpointFile {
                _file: file,
                _spill: spill,
            }));
        }
        Ok(frame)
    }

    /// This frame's plan with each checkpoint it stands on that was not taken yet taken: the node
    /// that marks it is replaced by a scan of the snapshot.
    pub(crate) fn plan_with_checkpoints(
        &self,
    ) -> futures::future::BoxFuture<'_, Result<Arc<LogicalPlan>, KrishivError>> {
        // Boxed: taking a snapshot analyzes (and runs) frames, which can be checkpoints themselves.
        Box::pin(async move {
            let lazy: Vec<Arc<LazyCheckpoint>> = self
                .keep
                .iter()
                .filter_map(|k| Arc::clone(k).downcast::<LazyCheckpoint>().ok())
                .collect();
            let mut plan = Arc::clone(&self.plan);
            for checkpoint in lazy {
                let snapshot = checkpoint.frame().await?;
                let next = Arc::clone(&plan).transform_down(&mut |node| {
                    Ok::<_, std::convert::Infallible>(
                        if node.origin.plan_id == Some(checkpoint.marker) {
                            Arc::new(LogicalPlan::clone(&snapshot.plan).with_origin(node.origin))
                        } else {
                            node
                        },
                    )
                });
                plan = next.unwrap_or(plan);
            }
            Ok(plan)
        })
    }
}

/// Writes `batches` to a spill-format file in a directory of its own under `dir`.
async fn write_file(
    session: &Session,
    dir: PathBuf,
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<(SpillFile, Arc<Mutex<Option<SpillManager>>>), KrishivError> {
    let state = &session.state;
    let canonical = crate::carrier::canonical_schema(Arc::clone(schema));
    let batches = batches
        .iter()
        .map(|b| {
            let b = RecordBatch::try_new(Arc::clone(schema), b.columns().to_vec())
                .map_err(|e| crate::session::internal(e.to_string()))?;
            crate::carrier::recarried(&b, &canonical).map(Batch::new)
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    let spill: Arc<Mutex<Option<SpillManager>>> = Arc::default();
    let disk = CacheDisk {
        executor: Arc::clone(&state.runtime.executor),
        spill: Arc::clone(&spill),
        config: SpillConfig {
            dir,
            ..SpillConfig::from_config(&state.runtime.config)
        },
    };
    let consumer = state.memory.register("checkpoint", None);
    let scratch = Reservation::new(Arc::clone(&state.memory), consumer);
    let file = disk.write(canonical, batches, scratch).await;
    state.memory.unregister(consumer);
    // Poisoned or not, the manager stays with the file.
    drop(spill.lock().unwrap_or_else(PoisonError::into_inner));
    Ok((file?, spill))
}
