// SPDX-License-Identifier: Apache-2.0
//! The deletes of one built scan, loaded for the data files that survive
//! pruning (PR-25d; reviews 25b S1 and 25c N6).
//!
//! [`IcebergScan::build`](super::provider) knows the pushed filters, so it
//! knows which data files the scan reads; it assigns their deletes (Java's
//! `DeleteFileIndex.forDataFile` runs per planned file too) and hands them to
//! a [`DeleteState`]. The delete files are read when the scan's first batch is
//! asked for ([`DeleteSource`]), not when the provider opens: a query whose
//! filter prunes a data file never reads its deletes, and the refusals that
//! depend on a data file (an unreadable partition tuple under an equality
//! delete, a mixed-spec identity column the file lacks) only fire for files
//! that are read.
//!
//! Loading on the first batch rather than in `build` is forced by the
//! interfaces: `build` is synchronous and has no query context, while the
//! source's first `get_data` is async and runs after `global_state` handed it
//! the query (its memory pool). The position selections are read by the file
//! scan when it opens a split, which only happens inside the inner source's
//! `get_data`, after [`DeleteState::ensure`] returned: the wrapper is the only
//! way to that inner source.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, OnceLock};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::{FileSelections, GlobalState, LocalState, Source, TaskContext};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    FieldMapping, FileRef, FilterId, FormatOptions, IoContext, RowSelection, WithoutFieldIds,
};
use krishiv_memory::{ConsumerHandle, Reservation};
use krishiv_types::batch::Batch;

use super::deletes::{self, DeleteSelections};
use super::eq_deletes::{self, EqProjection, EqualityDeletes};
use crate::plan::{DataFileEntry, SchemaField, TableSnapshot, TaskDeletes, Transform};

/// Locates a file of the table's storage by its Iceberg path.
pub(super) type Locate = Arc<dyn Fn(&str) -> Result<FileRef, KrishivError> + Send + Sync>;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// What every scan of a provider loads its deletes with.
pub(super) struct DeleteEnv {
    pub(super) io: IoContext,
    pub(super) locate: Locate,
    /// The table schema (the read schema of equality delete keys).
    pub(super) table: SchemaRef,
    pub(super) snapshot: Arc<TableSnapshot>,
    /// How many data files the provider has (equality sets are indexed by
    /// the file's ordinal).
    pub(super) files: usize,
    /// Top-level columns that are an identity partition source in some data
    /// files' specs and read from the files (see [`refuse_missing_mixed_identity`]).
    pub(super) mixed: Vec<SchemaField>,
    /// The data files' field mapping (the name mapping, if any).
    pub(super) mapping: FieldMapping,
    pub(super) format_options: FormatOptions,
}

impl DeleteEnv {
    /// Whether `e`'s spec has an identity field on the column `id`.
    fn identity_of(&self, e: &DataFileEntry, id: i32) -> bool {
        self.snapshot
            .partition_specs
            .get(&e.file.spec_id)
            .is_some_and(|spec| {
                spec.fields
                    .iter()
                    .any(|f| f.transform == Transform::Identity && f.source_id == id)
            })
    }

    /// Whether reading `e` needs the mixed-spec check.
    pub(super) fn needs_mixed_check(&self, e: &DataFileEntry) -> bool {
        self.mixed.iter().any(|c| self.identity_of(e, c.id))
    }
}

/// One data file a scan reads that has deletes or needs the mixed-spec check.
#[derive(Clone)]
pub(super) struct ScanTask {
    /// Its index in the provider's files (the [`eq_deletes::FILE_ORDINAL`]).
    pub(super) ordinal: usize,
    pub(super) entry: Arc<DataFileEntry>,
    pub(super) file: FileRef,
    pub(super) deletes: TaskDeletes,
}

/// What [`DeleteState::ensure`] loaded.
pub(super) struct Loaded {
    pub(super) selections: DeleteSelections,
    pub(super) equality: EqualityDeletes,
    /// Pays for `selections` (the key sets hold their own).
    _charges: Vec<Reservation>,
}

/// The deletes of the files one built scan reads, loaded once, on the
/// scan's first batch.
pub(super) struct DeleteState {
    env: Arc<DeleteEnv>,
    tasks: Vec<ScanTask>,
    loaded: tokio::sync::OnceCell<Loaded>,
    /// The query's pool, from the first `global_state`.
    pool: OnceLock<Arc<dyn krishiv_memory::MemoryPool>>,
}

impl fmt::Debug for DeleteState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeleteState")
            .field("tasks", &self.tasks.len())
            .field("loaded", &self.loaded.initialized())
            .finish_non_exhaustive()
    }
}

impl DeleteState {
    pub(super) fn new(env: Arc<DeleteEnv>, tasks: Vec<ScanTask>) -> Self {
        Self {
            env,
            tasks,
            loaded: tokio::sync::OnceCell::new(),
            pool: OnceLock::new(),
        }
    }

    /// Whether any task has position deletes or a DV.
    pub(super) fn has_positions(&self) -> bool {
        self.tasks
            .iter()
            .any(|t| t.deletes.deletion_vector.is_some() || !t.deletes.position_deletes.is_empty())
    }

    /// Remembers the query's memory pool (the first one given wins: one
    /// built source runs in one query).
    pub(super) fn set_pool(&self, pool: &Arc<dyn krishiv_memory::MemoryPool>) {
        let _ = self.pool.set(Arc::clone(pool));
    }

    /// The loaded deletes, loading them on the first call. Cancel-safe: a
    /// dropped call leaves the state unloaded, and the next call loads.
    ///
    /// # Errors
    /// The mixed-spec refusal; the errors of reading the delete files
    /// (`deletes::load`, `eq_deletes::load`).
    pub(super) async fn ensure(&self) -> Result<&Loaded, KrishivError> {
        self.loaded.get_or_try_init(|| self.load()).await
    }

    /// The loaded deletes, if [`Self::ensure`] finished.
    pub(super) fn loaded(&self) -> Option<&Loaded> {
        self.loaded.get()
    }

    async fn load(&self) -> Result<Loaded, KrishivError> {
        let env = &self.env;
        let pool = self.pool.get().ok_or_else(|| {
            internal("the Iceberg deletes were asked for before the scan's query".to_owned())
        })?;
        // A consumer of the query's pool that unregisters when the last
        // reservation made from it drops: when the load fails or its future
        // is dropped, the reads' reservations go and nothing stays charged.
        let mem = ConsumerHandle::register(Arc::clone(pool), "iceberg deletes", None);
        refuse_missing_mixed_identity(env, &self.tasks).await?;
        let locate = &*env.locate;
        let positions: Vec<_> = self
            .tasks
            .iter()
            .filter(|t| {
                t.deletes.deletion_vector.is_some() || !t.deletes.position_deletes.is_empty()
            })
            .map(|t| (t.entry.as_ref(), &t.file, &t.deletes))
            .collect();
        let (selections, charges) = if positions.is_empty() {
            (DeleteSelections::default(), Vec::new())
        } else {
            deletes::load(&env.io, locate, &mem, &positions).await?
        };
        let per_file: Vec<_> = self
            .tasks
            .iter()
            .filter(|t| !t.deletes.equality_deletes.is_empty())
            .map(|t| (t.ordinal, t.deletes.equality_deletes.as_slice()))
            .collect();
        let equality = eq_deletes::load(
            &env.io,
            locate,
            &mem,
            &env.table,
            &env.snapshot.fields,
            env.files,
            &per_file,
        )
        .await?;
        Ok(Loaded {
            selections,
            equality,
            _charges: charges,
        })
    }
}

/// The position selections of a [`DeleteState`], as the file scan reads
/// them when it opens a split (after the state loaded: see the module docs).
#[derive(Debug)]
pub(super) struct LazySelections(pub(super) Arc<DeleteState>);

impl FileSelections for LazySelections {
    fn selection(&self, file: &FileRef) -> Option<RowSelection> {
        self.0.loaded().and_then(|l| l.selections.selection(file))
    }
}

/// A file scan's source that loads the scan's deletes before its first
/// batch and, with equality deletes, drops the rows they delete.
pub(super) struct DeleteSource {
    inner: Arc<dyn Source>,
    state: Arc<DeleteState>,
    eq: Option<EqProjection>,
    schema: SchemaRef,
}

impl DeleteSource {
    /// The source over `inner`; with `eq`, its batches are filtered by the
    /// equality deletes and projected to `eq`'s output.
    pub(super) fn new(
        inner: Arc<dyn Source>,
        state: Arc<DeleteState>,
        eq: Option<EqProjection>,
    ) -> Self {
        let schema = eq
            .as_ref()
            .map_or_else(|| inner.schema(), EqProjection::schema);
        Self {
            inner,
            state,
            eq,
            schema,
        }
    }
}

impl Source for DeleteSource {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        self.state.set_pool(ctx.query().pool());
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.inner.max_parallelism(global)
    }

    fn deterministic_split(&self, global: &GlobalState) -> Option<usize> {
        self.inner.deterministic_split(global)
    }

    fn partitions(&self, global: &GlobalState) -> usize {
        self.inner.partitions(global)
    }

    fn local_state(
        &self,
        ctx: &TaskContext,
        global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        self.inner.local_state(ctx, global)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, krishiv_exec::Result<Option<Batch>>> {
        Box::pin(async move {
            let loaded = self.state.ensure().await?;
            let Some(eq) = &self.eq else {
                return self.inner.get_data(local, global).await;
            };
            loop {
                let Some(batch) = self.inner.get_data(local, global).await? else {
                    return Ok(None);
                };
                let out = eq.apply(&loaded.equality, batch.data())?;
                // A batch whose rows were all deleted is skipped, not sent.
                if out.num_rows() > 0 || batch.num_rows() == 0 {
                    return Ok(Some(Batch::new(out).with_split(batch.split())));
                }
            }
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        self.inner.morsels_claimed(local)
    }

    fn reports_positions(&self) -> bool {
        self.inner.reports_positions()
    }

    fn position(&self, local: &LocalState) -> Option<u64> {
        self.inner.position(local)
    }

    fn runtime_filters(&self) -> Vec<FilterId> {
        self.inner.runtime_filters()
    }

    fn extra_metrics(&self) -> krishiv_exec::ExtraMetrics {
        self.inner.extra_metrics()
    }
}

/// Mixed specs (review of PR-25a): a top-level column that is an identity
/// partition source in some data files' specs but not in others is read from
/// the files, not from the manifests. That is right while every file whose spec
/// partitions by it also stores it (conforming writers do); a file that lacks it
/// would read NULL where Java reads the manifest's value, so reading such a file
/// is refused instead. Reads the footers of the scan's files whose spec makes a
/// mixed column an identity partition, and only those.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the column and the file; the error of
/// a footer read.
async fn refuse_missing_mixed_identity(
    env: &DeleteEnv,
    tasks: &[ScanTask],
) -> Result<(), KrishivError> {
    if env.mixed.is_empty() {
        return Ok(());
    }
    // One footer read per such file, one after another: only tables whose
    // specs disagree on a column pay it (rare; spec evolution after an
    // import).
    let format = ParquetFormat;
    for t in tasks {
        let wanted: Vec<_> = env
            .mixed
            .iter()
            .filter(|c| env.identity_of(&t.entry, c.id))
            .collect();
        if wanted.is_empty() {
            continue;
        }
        let meta = env
            .io
            .read_metadata(
                &format,
                &t.file,
                krishiv_io::MetadataNeeds::default(),
                &env.format_options,
            )
            .await?;
        let mut schema = meta.schema();
        // Any id at any depth counts, as `SchemaMapper` decides (FC S4); a
        // file without ids takes the name mapping's, as the read does.
        let mut has_ids = krishiv_io::has_field_ids(schema.fields());
        if !has_ids && let FieldMapping::FieldIdOr(WithoutFieldIds::Names(m)) = &env.mapping {
            schema = m.apply(&schema);
            has_ids = true;
        }
        for c in wanted {
            let present = if has_ids {
                let id = c.id.to_string();
                schema
                    .fields()
                    .iter()
                    .any(|f| f.metadata().get(krishiv_io::FIELD_ID_KEY) == Some(&id))
            } else {
                // Positional fallback ids (FC S4).
                usize::try_from(c.id).is_ok_and(|k| k >= 1 && k <= schema.fields().len())
            };
            if !present {
                return Err(KrishivUnsupportedFeature {
                    feature: format!(
                        "reading the Iceberg identity partition column {} from the manifest for \
                         some data files but not others: the file {} lacks the column",
                        c.path.0.join("."),
                        t.entry.file.path
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// The mixed columns of a snapshot whose data files are `entries`: top-level,
/// not in `constant`, and an identity partition source in some file's spec.
pub(super) fn mixed_columns(
    snapshot: &TableSnapshot,
    constant: &BTreeSet<&str>,
    entries: &[Arc<DataFileEntry>],
) -> Vec<SchemaField> {
    let identity_of = |e: &DataFileEntry, id: i32| {
        snapshot
            .partition_specs
            .get(&e.file.spec_id)
            .is_some_and(|spec| {
                spec.fields
                    .iter()
                    .any(|f| f.transform == Transform::Identity && f.source_id == id)
            })
    };
    snapshot
        .fields
        .iter()
        .filter(|f| f.path.0.len() == 1 && !constant.contains(f.path.root()))
        .filter(|f| entries.iter().any(|e| identity_of(e, f.id)))
        .cloned()
        .collect()
}

/// `INTERNAL_ERROR` for a file of a built scan the provider does not have.
pub(super) fn unknown_file(f: &FileRef) -> KrishivError {
    internal(format!("the Iceberg scan has no data file {}", f.path))
}
