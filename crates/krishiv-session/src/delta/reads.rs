// SPDX-License-Identifier: Apache-2.0
//! [`DeltaReads`], the Delta tables one query reads (feature `delta`): each table root opened
//! once, before planning, through the runtime's [`DeltaCache`] (see the parent module).

use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_connectors_delta::engine::pool::LogPool;
use krishiv_connectors_delta::engine::{DeltaEngine, DeltaLimits};
use krishiv_connectors_delta::table::{
    CacheKey, DeltaTable, OpenOptions, SnapshotCache, hadoop_path, has_delta_log,
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
    credential: String,
    /// The planning-time kernel calls' cancellation (opening tables, `CREATE`'s shape): fired
    /// when the query's reads are dropped (a dropped planning future), so a call still waiting
    /// for a log-pool slot gives up and one already running ends at its next handler call.
    /// Scans do not use it: their engine is bound to the running query's cancellation
    /// (`DeltaEngine::for_query`). An interrupt during analysis does not reach it yet: the
    /// query is registered with the session's cancel registry only after planning (a recorded
    /// follow-up).
    cancel: Arc<CancellationToken>,
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
        Self {
            io: inputs.io,
            paths: inputs.paths,
            runtime: inputs.runtime,
            config: inputs.config,
            cache: inputs.cache,
            credential: inputs.credential,
            cancel: Arc::new(CancellationToken::new()),
            opened: Mutex::new(Vec::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(String, Arc<DeltaTable>)>> {
        self.opened.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The store and root URL of the Delta read `source`.
    pub(super) fn root_of(&self, source: &ReadSource) -> Result<(StoreId, Url), KrishivError> {
        let ReadSource::Files { paths, options, .. } = source else {
            return Err(internal("a Delta read that is not a read of files"));
        };
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
            let local = format!("/{}", at.path.as_ref());
            Url::from_directory_path(&local)
                .map_err(|()| internal(&format!("the Delta table path {local}")))?
        } else {
            let text = format!("{}/", at.display.trim_end_matches('/'));
            Url::parse(&text).map_err(|e| internal(&format!("the Delta table URL {text}: {e}")))?
        };
        Ok((at.store, url))
    }

    pub(super) fn engine(&self, store: StoreId, root: Url) -> Result<DeltaEngine, KrishivError> {
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
    /// already did: one snapshot per root per query. `name` is how errors name a catalog table;
    /// a path is named by its Hadoop form.
    pub(crate) async fn open(
        &self,
        source: &ReadSource,
        name: Option<String>,
    ) -> Result<Arc<DeltaTable>, KrishivError> {
        let (store, root) = self.root_of(source)?;
        let key = root.as_str().to_owned();
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
                    root: key.clone(),
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

    /// The provider of the table `source` reads, opened before planning.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a table this query did not open.
    pub(crate) fn provider(
        &self,
        source: &ReadSource,
    ) -> Result<Arc<dyn TableProvider>, KrishivError> {
        let (_, root) = self.root_of(source)?;
        self.find(root.as_str())
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
