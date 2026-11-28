// SPDX-License-Identifier: Apache-2.0
//! [`DeltaReads`], the Delta tables one query reads (feature `delta`): each table root opened
//! once, before planning, through the runtime's [`DeltaCache`] (see the parent module).

use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_connectors_delta::engine::pool::LogPool;
use krishiv_connectors_delta::engine::{DeltaEngine, DeltaLimits};
use krishiv_connectors_delta::table::{
    CacheKey, DeltaTable, OpenOptions, SnapshotCache, hadoop_path, has_delta_log, root_holds_files,
};
use krishiv_exec::CancellationToken;
use krishiv_exec::files::PathLocator;
use krishiv_exec::planner::TableProvider;
use krishiv_io::{IoContext, StoreId};
use krishiv_plan::ReadSource;
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::DataType;
use url::Url;

use super::DeltaInputs;

/// The runtime's cross-query snapshot cache (ADR 0037).
pub(crate) type DeltaCache = SnapshotCache;

/// Delta's read options Krishiv does not take yet (time travel, PR-08b's `TableVersionSpec`; the
/// change data feed, PR-37e).
const REFUSED_OPTIONS: &[&str] = &[
    "versionasof",
    "timestampasof",
    "readchangefeed",
    "readchangedata",
    "startingversion",
    "startingtimestamp",
];

/// What `CREATE TABLE ... USING delta LOCATION` reads of the table at a location.
#[derive(Debug)]
pub(crate) struct DeltaShape {
    /// The log's columns, nullability kept.
    pub(crate) columns: Vec<TableColumn>,
    /// The log's partition columns.
    pub(crate) partition_columns: Vec<String>,
    /// Whether the location holds a Delta table (a commit or checkpoint under `_delta_log/`);
    /// when it does not, the other fields are empty.
    pub(crate) existing: bool,
    /// The location in the Hadoop form delta-spark's errors print.
    pub(crate) path: String,
}

/// The Delta tables of one query, by normalized root URL.
pub(crate) struct DeltaReads {
    io: IoContext,
    paths: Arc<dyn PathLocator>,
    runtime: tokio::runtime::Handle,
    config: Arc<Config>,
    cache: Arc<DeltaCache>,
    pub(super) credential: String,
    /// The planning-time kernel calls' cancellation (opening tables, `CREATE`'s shape): fired
    /// when the query's reads are dropped (a dropped planning future), so a call still waiting
    /// for a log-pool slot gives up and one already running ends at its next handler call.
    /// Scans do not use it: their engine is bound to the running query's cancellation
    /// (`DeltaEngine::for_query`). Reads built inside an action are also cancelled with it
    /// (`action::cancelled`), so an interrupt during analysis ends their kernel calls even while
    /// a frame still holds the reads. The action is subscribed when the reads are built (in its
    /// scope), but the watcher task starts only at the first planning kernel call
    /// ([`engine`](Self::engine)), so a query that opens no Delta table, or a caller that only
    /// reads the config, spawns nothing. Limits: it watches only the innermost action that
    /// built them, so reads a cached analysis hands to a later action still follow the first
    /// one, and cancelling only an outer action reaches them through the dropped work alone.
    cancel: Arc<CancellationToken>,
    /// The action's interrupt watcher, not started yet (`None` outside an action, or once
    /// started): it cancels [`cancel`](Self::cancel) when the action is interrupted and ends
    /// when the token fires, so it never outlives the reads (their drop cancels the token).
    watcher: Mutex<Option<futures::future::BoxFuture<'static, ()>>>,
    opened: Mutex<Vec<(String, Arc<DeltaTable>)>>,
}

impl std::fmt::Debug for DeltaReads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaReads")
            .field("opened", &self.lock().len())
            .finish_non_exhaustive()
    }
}

impl Drop for DeltaReads {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl DeltaReads {
    pub(crate) fn new(inputs: DeltaInputs) -> Self {
        let cancel = Arc::new(CancellationToken::new());
        // Subscribed here, in the action's scope; spawned by the first `engine` call.
        let watcher = crate::action::in_action().then(|| {
            let interrupted = crate::action::cancelled();
            let token = Arc::clone(&cancel);
            let watch: futures::future::BoxFuture<'static, ()> = Box::pin(async move {
                let own = token.cancelled();
                futures::future::select(std::pin::pin!(interrupted), std::pin::pin!(own)).await;
                token.cancel();
            });
            watch
        });
        Self {
            io: inputs.io,
            paths: inputs.paths,
            runtime: inputs.runtime,
            config: inputs.config,
            cache: inputs.cache,
            credential: inputs.credential,
            cancel,
            watcher: Mutex::new(watcher),
            opened: Mutex::new(Vec::new()),
        }
    }

    /// Starts the action's interrupt watcher, once (nothing outside an action). It is polled
    /// once here first, so an action interrupted before this call has cancelled the token before
    /// the caller's first kernel call, not a scheduling round-trip later.
    fn start_watcher(&self) {
        let watch = self
            .watcher
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(mut watch) = watch
            && futures::FutureExt::now_or_never(&mut watch).is_none()
        {
            drop(self.runtime.spawn(watch));
        }
    }

    /// Whether the planning kernel calls' cancellation has fired (tests).
    #[cfg(test)]
    pub(super) fn cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Whether the interrupt watcher is still waiting to start (tests).
    #[cfg(test)]
    pub(super) fn watcher_pending(&self) -> bool {
        self.watcher
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(String, Arc<DeltaTable>)>> {
        self.opened.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The store and root URL of the Delta read `source`.
    pub(super) fn root_of(&self, source: &ReadSource) -> Result<(StoreId, Url), KrishivError> {
        let ReadSource::Files {
            paths,
            options,
            schema,
            ..
        } = source
        else {
            return Err(internal("a Delta read that is not a read of files"));
        };
        // The table's schema comes from its log. What Spark does with a declared one
        // (`spark.read.schema(..).format("delta")`) is not pinned by the oracle yet, so it is
        // refused by name rather than ignored.
        if schema.is_some() {
            return Err(unsupported("a user-specified schema on a Delta read"));
        }
        if let Some(key) = options
            .keys()
            .find(|k| REFUSED_OPTIONS.contains(&k.to_ascii_lowercase().as_str()))
        {
            return Err(unsupported(&format!(
                "the Delta read option {key} (time travel and the change data feed)"
            )));
        }
        let [path] = paths.as_slice() else {
            return Err(unsupported("a Delta read of other than one path"));
        };
        let Some(text) = path.as_plain() else {
            return Err(unsupported("a Delta table path from a secret"));
        };
        let at = self.paths.locate_with_options(text, options)?;
        let url = if at.display.starts_with("file:") {
            let local = local_dir_path(at.path.as_ref());
            Url::from_directory_path(&local)
                .map_err(|()| internal(&format!("the Delta table path {local}")))?
        } else {
            let text = format!("{}/", at.display.trim_end_matches('/'));
            Url::parse(&text).map_err(|e| internal(&format!("the Delta table URL {text}: {e}")))?
        };
        Ok((at.store, url))
    }

    pub(super) fn engine(&self, store: StoreId, root: Url) -> Result<DeltaEngine, KrishivError> {
        self.start_watcher();
        DeltaEngine::new(
            self.io.clone(),
            store,
            root,
            self.runtime.clone(),
            DeltaLimits::from_config(&self.config),
            Arc::clone(&self.cancel),
            LogPool::process(&self.config),
        )
    }

    /// The table `source` reads, opened now (through the snapshot cache) unless this query
    /// already did: one snapshot per store and root per query. `name` is how errors name a catalog table;
    /// a path is named by its Hadoop form.
    pub(crate) async fn open(
        &self,
        source: &ReadSource,
        name: Option<String>,
    ) -> Result<Arc<DeltaTable>, KrishivError> {
        let (store, root) = self.root_of(source)?;
        let root_text = root.as_str().to_owned();
        let key = table_key(&store, &root);
        if let Some(t) = self.find(&key) {
            return Ok(t);
        }
        let engine = self.engine(store.clone(), root)?;
        // A location under a credential scope reads through a store id with a per-registry
        // nonce (`#scope:`): its entry could never be hit again and would only push reusable
        // ones out, so such a table is not cached. Other store ids carry the `fs.*` settings
        // hash and the principal, which with the credential identity keep entries apart.
        let cache = (!store.key().contains("#scope:")).then(|| {
            (
                Arc::clone(&self.cache),
                CacheKey {
                    root: root_text,
                    store,
                    credential: self.credential.clone(),
                },
            )
        });
        let options = OpenOptions {
            version: None,
            name,
            cache,
        };
        let config = Arc::clone(&self.config);
        let table = Arc::new(DeltaTable::open_with(engine, options, &config).await?);
        let mut opened = self.lock();
        // Another scan of the root may have opened it meanwhile: the first one stays.
        if let Some((_, t)) = opened.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(t));
        }
        opened.push((key, Arc::clone(&table)));
        Ok(table)
    }

    fn find(&self, root: &str) -> Option<Arc<DeltaTable>> {
        self.lock()
            .iter()
            .find(|(k, _)| k == root)
            .map(|(_, t)| Arc::clone(t))
    }

    /// Opens the table `source` reads for a scan whose analyzed columns are `scanned`.
    ///
    /// # Errors
    /// As [`Self::open`]; `KRISHIV_UNSUPPORTED_FEATURE` when the log's columns are no longer the
    /// ones a catalog table was registered with.
    pub(crate) async fn open_scanned(
        &self,
        source: &ReadSource,
        name: Option<String>,
        scanned: &arrow_schema::SchemaRef,
    ) -> Result<(), KrishivError> {
        let (_, root) = self.root_of(source)?;
        let label = name.clone().unwrap_or_else(|| hadoop_path(&root));
        let table = self.open(source, name).await?;
        let log = columns_of(&table)?;
        let same = log.len() == scanned.fields().len()
            && log.iter().zip(scanned.fields()).all(|(c, f)| {
                c.name == *f.name() && DataType::from_arrow_field(f).is_ok_and(|t| t == c.data_type)
            });
        if same {
            Ok(())
        } else {
            Err(unsupported(&format!(
                "reading the Delta table {label} after its schema changed in the log (the catalog \
                 keeps the schema the table was created with; create it again)"
            )))
        }
    }

    /// The columns of the table `source` reads (opening it, see [`Self::open`]).
    pub(crate) async fn columns(
        &self,
        source: &ReadSource,
        name: Option<String>,
    ) -> Result<Vec<TableColumn>, KrishivError> {
        columns_of(&*self.open(source, name).await?)
    }

    /// The table at the location `source` reads, for `CREATE TABLE ... USING delta`: its
    /// columns and partitioning, or that there is none.
    pub(crate) async fn shape(
        &self,
        source: &ReadSource,
        name: String,
    ) -> Result<DeltaShape, KrishivError> {
        let (store, root) = self.root_of(source)?;
        let path = hadoop_path(&root);
        let engine = self.engine(store, root)?;
        if !has_delta_log(&engine).await? {
            return Ok(DeltaShape {
                columns: Vec::new(),
                partition_columns: Vec::new(),
                existing: false,
                path,
            });
        }
        let table = self.open(source, Some(name)).await?;
        Ok(DeltaShape {
            columns: columns_of(&table)?,
            partition_columns: table.partition_columns(),
            existing: true,
            path,
        })
    }

    /// The table a `CREATE TABLE .. USING delta` or a first `save` creates at the location
    /// `source` names, with the columns of `schema` partitioned by `partition_columns`.
    ///
    /// # Errors
    /// As [`Self::root_of`]; a column type Delta has no type for.
    pub(crate) fn new_table(
        &self,
        source: &ReadSource,
        schema: &arrow_schema::SchemaRef,
        partition_columns: Vec<String>,
        properties: std::collections::BTreeMap<String, String>,
        name: Option<String>,
    ) -> Result<krishiv_connectors_delta::write::NewDeltaTable, KrishivError> {
        let (store, root) = self.root_of(source)?;
        let engine = self.engine(store, root)?;
        krishiv_connectors_delta::write::NewDeltaTable::new(
            engine,
            schema,
            partition_columns,
            properties,
            name,
        )
    }

    /// Whether the location `source` names holds a Delta table.
    ///
    /// # Errors
    /// As [`Self::root_of`]; storage errors listing the log.
    pub(crate) async fn exists(&self, source: &ReadSource) -> Result<bool, KrishivError> {
        let (store, root) = self.root_of(source)?;
        let engine = self.engine(store, root)?;
        has_delta_log(&engine).await
    }

    /// Whether the root `source` names holds any object (a Delta table or not).
    ///
    /// # Errors
    /// As [`Self::root_of`]; storage errors listing the root.
    pub(crate) async fn holds_files(&self, source: &ReadSource) -> Result<bool, KrishivError> {
        let (store, root) = self.root_of(source)?;
        let engine = self.engine(store, root)?;
        root_holds_files(&engine).await
    }

    /// The root of the table `source` reads, as delta-spark's errors name it (`file:/abs/dir`).
    ///
    /// # Errors
    /// As [`Self::root_of`].
    pub(crate) fn hadoop_root(&self, source: &ReadSource) -> Result<String, KrishivError> {
        let (_, root) = self.root_of(source)?;
        Ok(hadoop_path(&root))
    }

    /// The provider of the table `source` reads, opened before planning.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a table this query did not open.
    pub(crate) fn provider(
        &self,
        source: &ReadSource,
    ) -> Result<Arc<dyn TableProvider>, KrishivError> {
        let (store, root) = self.root_of(source)?;
        self.find(&table_key(&store, &root))
            .map(|t| t as Arc<dyn TableProvider>)
            .ok_or_else(|| internal("a Delta scan whose table was not opened before planning"))
    }
}

/// The columns of `table`'s schema, nullability kept.
fn columns_of(table: &DeltaTable) -> Result<Vec<TableColumn>, KrishivError> {
    table
        .schema()
        .fields()
        .iter()
        .map(|f| {
            Ok(TableColumn {
                name: f.name().clone(),
                data_type: DataType::from_arrow_field(f)?,
                nullable: f.is_nullable(),
                char_varchar: None,
            })
        })
        .collect()
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

/// The query-local key of the table at `root` read through `store`: two reads of one URL with
/// different `fs.*` options reach it through different stores (endpoint, credentials), so each
/// opens its own table.
pub(super) fn table_key(store: &StoreId, root: &Url) -> String {
    format!("{}\u{0}{}", store.key(), root.as_str())
}

/// The absolute local directory path for a store-relative `path`: a POSIX path gains its root `/`;
/// a Windows drive path (`C:/Users/t`) is already absolute and `Url::from_directory_path` takes it
/// as is (a leading `/` would make it `/C:/...`, which Windows refuses). A UNC location
/// (`//server/share/t`) is not handled: it still becomes `///server/share/t`.
fn local_dir_path(path: &str) -> String {
    local_dir_path_for(path, cfg!(windows))
}

/// [`local_dir_path`] with the platform named, so both shapes are testable on one host.
fn local_dir_path_for(path: &str, windows: bool) -> String {
    let drive = windows && matches!(path.as_bytes(), [d, b':', ..] if d.is_ascii_alphabetic());
    if drive {
        path.to_owned()
    } else {
        format!("/{path}")
    }
}

#[cfg(test)]
mod local_dir_path_tests {
    use super::local_dir_path_for;

    #[test]
    fn a_posix_path_gains_its_root() {
        assert_eq!(local_dir_path_for("tmp/t", false), "/tmp/t");
        assert_eq!(local_dir_path_for("tmp/t", true), "/tmp/t");
    }

    #[test]
    fn a_windows_drive_path_is_already_absolute_only_on_windows() {
        assert_eq!(local_dir_path_for("C:/Users/t", true), "C:/Users/t");
        assert_eq!(local_dir_path_for("C:/Users/t", false), "/C:/Users/t");
    }
}
