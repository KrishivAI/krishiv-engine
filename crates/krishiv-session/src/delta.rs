// SPDX-License-Identifier: Apache-2.0
//! Delta tables in the session (P3-T021 PR-08b part 2; feature `delta`, ADR 0027).
//!
//! A Delta read is the third kind of scan beside in-memory and file tables: `spark.read
//! .format("delta").load(p)`, `` delta.`p` ``, a catalog table `USING delta LOCATION 'p'` and a
//! temporary view `USING delta OPTIONS (path 'p')`. Each table root a query reads is opened once,
//! before planning, with the query's `IoContext` ([`DeltaReads`], one per query), so every scan
//! of it in the query (a self join, a subquery) reads one snapshot; the table's directory is
//! never listed as files. Across queries the runtime's [`DeltaCache`] keeps each table's latest
//! snapshot under its root, store and the session's principal, and each query brings it up to
//! date (ADR 0037 `per_query`). Both stay in the session, never in a plan node.
//!
//! Errors name a catalog table by `namespace.table` and a path by its Hadoop form
//! (`file:/abs/dir`), as delta-spark does (`tests/fixtures/delta-4.4.1/SESSION.json`).
//!
//! Without the feature, `delta` is not a data source: every Delta read and DDL fails as before
//! (`DATA_SOURCE_NOT_FOUND`), and [`DeltaReads`] is an empty stand-in.

use std::collections::BTreeMap;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::FileReads;
use krishiv_plan::ReadSource;
use krishiv_plan::analyzer::{BoxFuture, FileSchemaResolver, SourceKind, TableColumn};

#[cfg(feature = "delta")]
mod reads;
#[cfg(feature = "delta")]
pub(crate) use reads::{DeltaCache, DeltaReads};

/// The data source name of Delta tables.
pub(crate) const DELTA: &str = "delta";

/// Whether this build reads Delta tables.
pub(crate) const ENABLED: bool = cfg!(feature = "delta");

/// Whether `format` names a Delta read this build handles.
pub(crate) fn handles(format: &str) -> bool {
    ENABLED && format.eq_ignore_ascii_case(DELTA)
}

/// Whether `source` is a Delta read this build handles.
pub(crate) fn is_delta(source: &ReadSource) -> bool {
    matches!(source, ReadSource::Files { format, .. } if handles(format))
}

/// What a query's [`DeltaReads`] is built from.
#[cfg_attr(
    not(feature = "delta"),
    expect(
        dead_code,
        reason = "read by the Delta reads of the `delta` feature only"
    )
)]
pub(crate) struct DeltaInputs {
    /// The query's I/O.
    pub(crate) io: krishiv_io::IoContext,
    /// Where paths live (local and cloud stores).
    pub(crate) paths: Arc<dyn krishiv_exec::files::PathLocator>,
    /// The executor's I/O runtime, which the kernel's handlers bridge to.
    pub(crate) runtime: tokio::runtime::Handle,
    /// The query's configuration.
    pub(crate) config: Arc<krishiv_common::config::Config>,
    /// The runtime's snapshot cache.
    pub(crate) cache: Arc<DeltaCache>,
    /// The credential identity snapshots are cached under (the session's principal).
    pub(crate) credential: String,
}

#[cfg(not(feature = "delta"))]
mod disabled {
    use std::sync::Arc;

    use krishiv_common::error::{InternalError, KrishivError};
    use krishiv_exec::planner::TableProvider;
    use krishiv_plan::ReadSource;
    use krishiv_plan::analyzer::TableColumn;

    /// The runtime's snapshot cache: nothing without the `delta` feature.
    #[derive(Debug, Default)]
    pub(crate) struct DeltaCache;

    /// A query's Delta tables: none without the `delta` feature ([`super::handles`] is false,
    /// so nothing reaches these).
    #[derive(Debug, Default)]
    pub(crate) struct DeltaReads;

    /// What `CREATE TABLE ... USING delta` reads of a table.
    #[derive(Debug)]
    pub(crate) struct DeltaShape {
        pub(crate) columns: Vec<TableColumn>,
        pub(crate) partition_columns: Vec<String>,
        pub(crate) existing: bool,
        pub(crate) path: String,
    }

    fn off() -> KrishivError {
        InternalError {
            message: "a Delta read in a build without the delta feature".to_owned(),
        }
        .into()
    }

    impl DeltaReads {
        pub(crate) fn new(_: super::DeltaInputs) -> Self {
            Self
        }

        pub(crate) async fn columns(
            &self,
            _: &ReadSource,
            _: Option<String>,
        ) -> Result<Vec<TableColumn>, KrishivError> {
            Err(off())
        }

        pub(crate) async fn open_scanned(
            &self,
            _: &ReadSource,
            _: Option<String>,
            _: &arrow_schema::SchemaRef,
        ) -> Result<(), KrishivError> {
            Err(off())
        }

        pub(crate) async fn shape(
            &self,
            _: &ReadSource,
            _: String,
        ) -> Result<DeltaShape, KrishivError> {
            Err(off())
        }

        pub(crate) fn provider(
            &self,
            _: &ReadSource,
        ) -> Result<Arc<dyn TableProvider>, KrishivError> {
            Err(off())
        }
    }
}
#[cfg(not(feature = "delta"))]
pub(crate) use disabled::{DeltaCache, DeltaReads};

/// The query's file reads with Delta read beside them: a `delta` source reads its table's
/// schema from the log; every other source goes to the file reads unchanged.
pub(crate) struct SessionFiles {
    pub(crate) files: Arc<FileReads>,
    pub(crate) delta: Arc<DeltaReads>,
}

impl FileSchemaResolver for SessionFiles {
    fn source_kind(&self, name: &str) -> SourceKind {
        if handles(name) {
            SourceKind::File
        } else {
            self.files.source_kind(name)
        }
    }

    fn qualify(&self, path: &OptionValue) -> Result<OptionValue, KrishivError> {
        self.files.qualify(path)
    }

    fn infer<'a>(
        &'a self,
        format: &'a str,
        paths: &'a [OptionValue],
        options: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, Result<Vec<TableColumn>, KrishivError>> {
        if !handles(format) {
            return self.files.infer(format, paths, options);
        }
        Box::pin(async move {
            let source = ReadSource::Files {
                format: DELTA.to_owned(),
                paths: paths.to_vec(),
                options: options.clone(),
            };
            self.delta.columns(&source, None).await
        })
    }
}

#[cfg(all(test, feature = "delta"))]
mod tests;
