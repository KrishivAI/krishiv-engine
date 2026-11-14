// SPDX-License-Identifier: Apache-2.0
//! Reads by path (P2-T154): user paths located in stores, listed as Spark
//! lists them, and their schema inferred by their format. [`FileReads`] is
//! the session's [`FileSchemaResolver`] for one query: it lists each read
//! once and keeps the listing, so the scan reads the files the analyzer saw.
//!
//! Listing follows Spark 4.2's `InMemoryFileIndex` (checked on a Spark run):
//!
//! | Path | Result |
//! |---|---|
//! | a file | that file, unless its name is hidden |
//! | a directory | the files in it whose names are not hidden, sorted by path |
//! | neither | `PATH_NOT_FOUND`, the path qualified (`file:/abs/path`) |
//! | no file left | `UNABLE_TO_INFER_SCHEMA` |
//!
//! `name=value` directories under a path are partition columns (Spark's
//! partition discovery, `krishiv_io::partition_columns`): typed by
//! inference, appended after the data columns, read from the path of each
//! file. A table with a known schema (a permanent catalog table) takes
//! them from its declared partition columns instead: the directories must
//! name exactly those columns, in order, and each value is read as the
//! column's declared type, never inferred.
//!
//! A name is hidden when it starts with `.`, or with `_` without a `=` in
//! it, or ends with `._COPYING_`; hidden directories are skipped whole.
//! The format infers the schema from the files by path (Spark without
//! `mergeSchema`): Parquet reads the first file's footer, CSV samples
//! all of them. Local paths and cloud URLs (`s3://`, `s3a://`, `gs://`,
//! `abfs://`, `abfss://`, `az://`, resolved by [`ObjectPaths`] through the
//! [`StoreRegistry`]) are supported. Not yet supported, failing with
//! `KRISHIV_UNSUPPORTED_FEATURE`: subdirectories that are
//! not `name=value` partition directories in a read by path (a catalog
//! table reads its registered partitions only and skips other directories,
//! as Spark's `CatalogFileIndex`), paths with any other scheme, file sources Spark
//! knows but the engine has no [`FileFormat`] for, and merging the files'
//! schemas (`mergeSchema`). A name Spark does not know at all fails with
//! Spark's `DATA_SOURCE_NOT_FOUND`.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_schema::{FieldRef, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::config::Config;
use krishiv_common::error::{
    CannotMergeIncompatibleDataType, FailedReadFileUnsupportedFileSystem, InvalidEmptyLocation,
    InvalidPartitionValue, KrishivError, KrishivUnsupportedFeature, LegacyErrorTemp1290,
    PathNotFound, UnableToInferSchema, UnsupportedDataTypeForDatasource,
};
use krishiv_common::secret::OptionValue;
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, FormatRegistry, IoContext, IoScheduler, Segment, StoreId,
    StoreRegistry, is_spark_file_source, parse_object_url, parse_segment, partition_columns,
    register, register_with_options,
};
use krishiv_plan::analyzer::{FileSchemaResolver, SourceKind, TableColumn};
use krishiv_types::{DataType, StructType};
use object_store::path::Path;

use crate::error::Result;

/// Spark 4.2's built-in data sources that read no files:
/// `UNSUPPORTED_DATASOURCE_FOR_DIRECT_QUERY`.
const SPARK_OTHER_SOURCES: &[&str] = &["jdbc", "console", "noop"];

/// A user path in a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocatedPath {
    /// The store.
    pub store: StoreId,
    /// The path in it.
    pub path: Path,
    /// The path as errors print it (Spark's qualified form, `file:/a/b`).
    pub display: String,
}

/// Maps user paths to stores. A path's [`LocatedPath::display`] must
/// locate the same path again: it is the form the access policy is asked
/// about and the scan reads.
pub trait PathLocator: Send + Sync + fmt::Debug {
    /// Where `path` lives.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a path it cannot place.
    fn locate(&self, path: &str) -> Result<LocatedPath>;

    /// Where `path` lives for a read with per-relation `options`: cloud
    /// stores resolve with the read's `fs.*` data-source options overriding
    /// the session configuration (Spark's per-relation Hadoop
    /// configuration). The default ignores the options.
    ///
    /// # Errors
    /// As [`Self::locate`], plus the store's configuration errors.
    fn locate_with_options(
        &self,
        path: &str,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<LocatedPath> {
        let _ = options;
        self.locate(path)
    }

    /// The display form of `path` without building any store (what
    /// [`Self::locate`] would print). The default locates and prints; cloud
    /// paths override it to re-parse the URL, so qualifying never needs the
    /// read's options and never fails on store configuration.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a path it cannot place.
    fn qualified_display(&self, path: &str) -> Result<String> {
        self.locate(path).map(|at| at.display)
    }
}

/// Local file-system paths, in a store rooted at `/` (an
/// `object_store::local::LocalFileSystem` without a prefix, registered
/// under the same [`StoreId`]): absolute paths, paths relative to a working
/// directory, and `file:` URLs. `.` and `..` segments are resolved in the
/// text, as Hadoop's `Path` normalizes them.
#[derive(Debug, Clone)]
pub struct LocalPaths {
    store: StoreId,
    cwd: String,
}

impl LocalPaths {
    /// Paths in `store`, relative ones under the absolute directory `cwd`.
    #[must_use]
    pub fn new(store: StoreId, cwd: impl Into<String>) -> Self {
        Self {
            store,
            cwd: cwd.into(),
        }
    }
}

impl PathLocator for LocalPaths {
    fn locate(&self, path: &str) -> Result<LocatedPath> {
        let local = if let Some(rest) = path.strip_prefix("file://") {
            // `file:///a` and `file://localhost/a`; another host is remote.
            let at = rest.find('/').unwrap_or(rest.len());
            let (host, rest) = rest.split_at(at);
            if !(host.is_empty() || host == "localhost") {
                return Err(unsupported(format!("file URLs with a host ({host})")));
            }
            rest.to_owned()
        } else if let Some(rest) = path.strip_prefix("file:") {
            rest.to_owned()
        } else if let Some(scheme) = scheme(path) {
            return Err(unsupported(format!("paths with the scheme {scheme}")));
        } else if path.starts_with('/') {
            path.to_owned()
        } else {
            format!("{}/{path}", self.cwd)
        };
        let mut parts: Vec<&str> = Vec::new();
        for part in local.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                p => parts.push(p),
            }
        }
        let display = format!("file:/{}", parts.join("/"));
        Ok(LocatedPath {
            store: self.store.clone(),
            path: Path::from_iter(parts),
            display,
        })
    }
}

/// User paths in the local file system and in cloud object stores
/// (P2-T084): `file:` URLs and bare paths resolve as [`LocalPaths`] does,
/// while `s3://`, `s3a://`, `gs://`, `abfs://`, `abfss://` and `az://` URLs
/// resolve through the session's [`StoreRegistry`],
/// which builds (or reuses, within the query) the bucket's store from the
/// session configuration with the read's per-relation `fs.*` options
/// overriding it (Spark's per-relation Hadoop configuration) and registers
/// it on the scheduler. Cloud URLs need the `cloud` feature: without it the
/// registry reports them as unsupported. Locating is synchronous and performs
/// no I/O: store construction only reads configuration and the environment.
/// A located cloud path re-parses to itself through its [`LocatedPath::display`].
/// Cloud URLs need the `cloud` feature: without it the
/// registry reports them as unsupported.
#[derive(Debug, Clone)]
pub struct ObjectPaths {
    local: LocalPaths,
    sched: Arc<IoScheduler>,
    stores: Arc<StoreRegistry>,
}

impl ObjectPaths {
    /// Paths in `store` under `cwd`, with cloud stores built by `stores`
    /// and registered on `sched`.
    #[must_use]
    pub fn new(
        store: StoreId,
        cwd: impl Into<String>,
        sched: Arc<IoScheduler>,
        stores: Arc<StoreRegistry>,
    ) -> Self {
        Self {
            local: LocalPaths::new(store, cwd),
            sched,
            stores,
        }
    }
}

impl PathLocator for ObjectPaths {
    fn locate(&self, path: &str) -> Result<LocatedPath> {
        let cloud = match scheme(path) {
            None | Some("file") => false,
            Some(_) => true,
        };
        if !cloud {
            return self.local.locate(path);
        }
        let (store, key, display) = register(&self.sched, &self.stores, path)?;
        Ok(LocatedPath {
            store,
            path: key,
            display,
        })
    }

    fn locate_with_options(
        &self,
        path: &str,
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<LocatedPath> {
        let cloud = match scheme(path) {
            None | Some("file") => false,
            Some(_) => true,
        };
        if !cloud {
            return self.local.locate(path);
        }
        let (store, key, display) =
            register_with_options(&self.sched, &self.stores, path, options)?;
        Ok(LocatedPath {
            store,
            path: key,
            display,
        })
    }

    fn qualified_display(&self, path: &str) -> Result<String> {
        // No store is built here, so qualifying never needs the read's
        // options and never fails on store configuration: the store is
        // instantiated at listing from the plan's option values.
        match scheme(path) {
            None | Some("file") => self.local.locate(path).map(|at| at.display),
            Some(_) => parse_object_url(path).map(|url| url.display),
        }
    }
}

/// The URL scheme of `path` (`s3a` in `s3a://b/k`), if it has one.
fn scheme(path: &str) -> Option<&str> {
    let (scheme, _) = path.split_once(':')?;
    let mut chars = scheme.chars();
    let valid = chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then_some(scheme)
}

/// One read by path, listed: what its scan reads.
#[derive(Clone)]
pub struct FileListing {
    /// The format.
    pub format: Arc<dyn FileFormat>,
    /// The files, sorted by path.
    pub files: Arc<[FileRef]>,
    /// The schema the format infers from the listed files, followed by the
    /// partition columns.
    pub schema: SchemaRef,
    /// The partition columns discovered from the `k=v` directories above
    /// the files (already at the end of `schema`); empty for a flat read.
    pub partition_columns: Arc<[FieldRef]>,
    /// The parsed read options.
    pub options: FormatOptions,
}

impl fmt::Debug for FileListing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileListing")
            .field("format", &self.format.name())
            .field("files", &self.files.len())
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

/// What identifies a read: format, paths, options and the user's schema.
type ReadKey = (
    String,
    Vec<OptionValue>,
    BTreeMap<String, OptionValue>,
    Option<StructType>,
);

/// The reads by path of one query: lists each once and keeps the listing.
/// Make one per query; the next query lists the files again.
pub struct FileReads {
    io: IoContext,
    formats: Arc<FormatRegistry>,
    paths: Arc<dyn PathLocator>,
    session: Arc<Config>,
    listed: Mutex<Vec<(ReadKey, Arc<FileListing>)>>,
}

impl fmt::Debug for FileReads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileReads")
            .field("formats", &self.formats.names())
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

impl FileReads {
    /// Reads through `io` with `formats`, placing paths with `paths`;
    /// options fall back to the `session` config.
    #[must_use]
    pub fn new(
        io: IoContext,
        formats: Arc<FormatRegistry>,
        paths: Arc<dyn PathLocator>,
        session: Arc<Config>,
    ) -> Self {
        Self {
            io,
            formats,
            paths,
            session,
            listed: Mutex::new(Vec::new()),
        }
    }

    /// The listing of reading `paths` as `format` with `options`: listed
    /// on the first call, the same listing after.
    ///
    /// # Errors
    /// `PATH_NOT_FOUND`, `UNABLE_TO_INFER_SCHEMA`, Spark's
    /// `DATA_SOURCE_NOT_FOUND` for a name Spark does not know, option
    /// errors, `KRISHIV_UNSUPPORTED_FEATURE` (see the module docs) and
    /// storage failures.
    pub async fn listing(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<Arc<FileListing>> {
        self.listing_with_schema(format, paths, options, None).await
    }

    /// [`Self::listing`] of a read with the user's `schema`
    /// (`spark.read.schema`), when it has one: the files are listed as for
    /// any read, nothing is inferred from them (and no files is no error),
    /// and the listing's schema is the user's data columns in their order,
    /// then the partition columns found under the paths, typed as the schema
    /// names them (inferred when it does not), as Spark's
    /// `PartitioningAwareFileIndex` combines them. Every column is nullable.
    ///
    /// # Errors
    /// As [`Self::listing`], plus `INVALID_PARTITION_VALUE` for a partition
    /// value that does not cast to the schema's type, and the text format's
    /// checks of the schema (`_LEGACY_ERROR_TEMP_1290`,
    /// `UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE`).
    pub async fn listing_with_schema(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
        schema: Option<&StructType>,
    ) -> Result<Arc<FileListing>> {
        let key = (
            format.to_ascii_lowercase(),
            paths.to_vec(),
            options.clone(),
            schema.cloned(),
        );
        if let Some(l) = self.cached(&key) {
            return Ok(l);
        }
        let listing =
            Arc::new(Box::pin(self.list_with(&key.0, paths, options, None, schema)).await?);
        let mut listed = self.listed.lock().unwrap_or_else(PoisonError::into_inner);
        // A concurrent call may have listed it first: keep one listing.
        if let Some((_, l)) = listed.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(l));
        }
        listed.push((key, Arc::clone(&listing)));
        Ok(listing)
    }

    /// The listing of a table whose columns are known: the files under
    /// `paths` (none is fine: the table is empty), with `schema` as their
    /// schema. `partitions` are the table's partition columns (the last
    /// columns of `schema`, in this order) and the partitions its catalog
    /// registered. Only the files of registered partitions are read, as
    /// Spark's `CatalogFileIndex` lists only their locations: a file under a
    /// directory the catalog does not know (before `MSCK REPAIR`), or under
    /// any directory that is not a partition of the declared columns, is not
    /// one of the table's files (`spark.sql.hive.manageFilesourcePartitions`
    /// is not modelled: Spark's default, true). Values are read as the
    /// columns' declared types. Nothing is inferred and nothing is cached,
    /// so every query lists the table afresh.
    ///
    /// # Errors
    /// The listing errors of [`Self::listing`], except that no files is not
    /// an error; `KRISHIV_UNSUPPORTED_FEATURE` when a partition column's type
    /// is not read from a directory yet.
    pub async fn listing_known(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
        schema: SchemaRef,
        partitions: &KnownPartitions,
    ) -> Result<Arc<FileListing>> {
        let listing = Box::pin(self.list_with(
            &format.to_ascii_lowercase(),
            paths,
            options,
            Some((schema, partitions)),
            None,
        ))
        .await?;
        Ok(Arc::new(listing))
    }

    fn cached(&self, key: &ReadKey) -> Option<Arc<FileListing>> {
        let listed = self.listed.lock().unwrap_or_else(PoisonError::into_inner);
        listed
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, l)| Arc::clone(l))
    }

    /// Lists a read; with `known` the schema and the partition columns
    /// are given, not inferred; with `user` the data columns are the user's
    /// ([`Self::listing_with_schema`]).
    async fn list_with(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
        known: Option<(SchemaRef, &KnownPartitions)>,
        user: Option<&StructType>,
    ) -> Result<FileListing> {
        let mut mode = self.list_mode(options)?;
        // A partitioned catalog table reads its registered partitions only
        // (Spark's `CatalogFileIndex`, which lists only their locations).
        let partitioned = known.as_ref().is_some_and(|(_, d)| !d.columns.is_empty());
        if partitioned && mode.recursive {
            return Err(unsupported(
                "recursiveFileLookup on a partitioned table".to_owned(),
            ));
        }
        // Any catalog table skips a directory that is not `name=value`
        // (Spark 4.2.0 reads an unpartitioned table's root files only, and a
        // partitioned one's registered locations only).
        // (`recursiveFileLookup` keeps reading every directory, as before.)
        mode.skip_foreign_dirs = known.is_some() && mode.base.is_none() && !mode.recursive;
        mode.skip_all_dirs = mode.skip_foreign_dirs && !partitioned;
        // Spark checks the paths before the format.
        let mut listed: Vec<(FileRef, Vec<Segment>)> = Vec::new();
        for p in paths {
            let Some(text) = p.as_plain() else {
                return Err(unsupported("paths with credentials".to_owned()));
            };
            if krishiv_io::partition::glob::has_glob(text) {
                // Boxed, as the listing below: these futures sit on the recursion of nested scripts, where a
                // larger one overflows the stack (`script_nesting_is_bounded_at_the_limit`).
                Box::pin(self.list_glob(text, options, &mode, &mut listed)).await?;
                continue;
            }
            // The store is instantiated from the plan's option values
            // (plans stay pure data): the read's `fs.*` options override
            // the session configuration.
            let at = self.paths.locate_with_options(text, options)?;
            Box::pin(self.list_path(&at, &mode, known.is_some(), &mut listed)).await?;
        }
        // By store, then path; of a file reached through overlapping roots the entry with the fewest partition
        // segments (the nearest root's) comes first and is the one kept.
        listed.sort_by(|a, b| {
            a.0.store
                .key()
                .cmp(b.0.store.key())
                .then_with(|| a.0.path.as_ref().cmp(b.0.path.as_ref()))
                .then_with(|| a.1.len().cmp(&b.1.len()))
        });
        // Overlapping roots (`/d/{a,a/b}`, or a glob and a plain path) reach a file twice: it is read once.
        listed.dedup_by(|a, b| a.0.path == b.0.path && a.0.store == b.0.store);
        // A partitioned catalog table reads its registered partitions only
        // (Spark's `CatalogFileIndex`): other directories under its root are
        // not its locations, so their files are not read (Spark 4.2.0: an
        // external table's `p=1` directory left on disk after `DROP PARTITION`
        // is not read).
        if let Some((_, declared)) = &known
            && partitioned
        {
            // With a `basePath` above the location, directories that are not
            // the table's partition columns sit above its files (`q=1/p=2`
            // for a table partitioned by `p`): refused, as before, rather
            // than read as an empty table.
            if mode.base.is_some()
                && let Some((f, segs)) = listed.iter().find(|(_, segs)| {
                    segs.len() != declared.columns.len()
                        || segs.iter().zip(&declared.columns).any(|((n, _), c)| n != c)
                })
            {
                let found: Vec<&str> = segs.iter().map(|(n, _)| n.as_str()).collect();
                return Err(unsupported(format!(
                    "reading {} with basePath: its partition directories ({}) are not the table's partition columns ({})",
                    f.display_path(),
                    found.join(", "),
                    declared.columns.join(", ")
                )));
            }
            let registered: HashSet<&[Option<String>]> =
                declared.registered.iter().map(Vec::as_slice).collect();
            listed.retain(|(_, segs)| is_registered(&declared.columns, &registered, segs));
        }
        let format = match self.formats.get(format) {
            Ok(format) => format,
            Err(e) => {
                if is_spark_file_source(format) {
                    return Err(unsupported(format!("the {format} file format")));
                }
                return Err(e);
            }
        };
        let merge = self.merges_schemas(format.name(), options);
        let (options, _) = format.options().parse(options, &self.session)?;
        let discovered: Vec<(String, Vec<Segment>)> = listed
            .iter()
            .map(|(f, segs)| (f.display_path(), segs.clone()))
            .collect();
        let files: Vec<FileRef> = listed.into_iter().map(|(f, _)| f).collect();
        // A known table's partition values take its declared types: they
        // are never inferred (nor refused for looking like timestamps).
        let infer = known.is_none() && self.infers_partition_types();
        if let Some(user) = user {
            check_user_schema(format.name(), user)?;
            let (schema, partition_columns) =
                user_schema_columns(user, &discovered, infer, self.case_sensitive())?;
            return Ok(FileListing {
                format,
                files: files.into(),
                schema,
                partition_columns: partition_columns.into(),
                options,
            });
        }
        let partition_columns = partition_columns(&discovered, infer)?;
        if let Some((schema, declared)) = known {
            let partition_columns = declared_partitions(&schema, declared, &partition_columns)?;
            return Ok(FileListing {
                format,
                files: files.into(),
                schema,
                partition_columns: partition_columns.into(),
                options,
            });
        }
        if files.is_empty() {
            return Err(UnableToInferSchema {
                format: display_name(format.name()),
            }
            .into());
        }
        let schema = if merge {
            // Every file's own schema, merged by column name in first-seen order; names match
            // under `spark.sql.caseSensitive` (default false), so `id` and `ID` are one column.
            let case_sensitive = self.case_sensitive();
            let mut merged: Option<SchemaRef> = None;
            for file in &files {
                let one = krishiv_io::infer_schema_cached(
                    format.as_ref(),
                    &self.io,
                    std::slice::from_ref(file),
                    &options,
                )
                .await
                .map_err(|e| e.into_error(&file.display_path()))?;
                merged = Some(match merged {
                    None => one,
                    Some(m) => merge_schemas(&m, &one, case_sensitive)?,
                });
            }
            merged.ok_or_else(|| {
                KrishivError::from(UnableToInferSchema {
                    format: display_name(format.name()),
                })
            })?
        } else {
            krishiv_io::infer_schema_cached(format.as_ref(), &self.io, &files, &options)
                .await
                .map_err(|e| e.into_error(&files[0].display_path()))?
        };
        let schema = with_partition_columns(schema, &partition_columns)?;
        Ok(FileListing {
            format,
            files: files.into(),
            schema,
            partition_columns: partition_columns.into(),
            options,
        })
    }

    /// `spark.sql.caseSensitive` (default false).
    fn case_sensitive(&self) -> bool {
        self.session
            .get_unregistered("spark.sql.caseSensitive")
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("true"))
    }

    /// Whether partition values are typed by inference:
    /// `spark.sql.sources.partitionColumnTypeInference.enabled` (default
    /// true).
    fn infers_partition_types(&self) -> bool {
        self.session
            .get_unregistered("spark.sql.sources.partitionColumnTypeInference.enabled")
            .is_none_or(|v| !v.trim().eq_ignore_ascii_case("false"))
    }

    /// Whether the read asks to merge the files' schemas: the
    /// `mergeSchema` option, else `spark.sql.<format>.mergeSchema`.
    fn merges_schemas(&self, format: &str, options: &BTreeMap<String, OptionValue>) -> bool {
        let on = |v: &str| v.trim().eq_ignore_ascii_case("true");
        let option = options
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("mergeSchema"))
            .map(|(_, v)| v.as_plain().is_some_and(on));
        option.unwrap_or_else(|| {
            self.session
                .get_unregistered(&format!("spark.sql.{format}.mergeSchema"))
                .is_some_and(on)
        })
    }

    /// Adds the files `at` names to `out` (none when it is absent and `missing_is_empty`), each with the `name=value`
    /// directories between `at` and the file.
    /// The files under every path `text` (a Hadoop glob) matches; each match
    /// is a root of its own, as in Spark, so partition discovery starts there.
    async fn list_glob(
        &self,
        text: &str,
        options: &BTreeMap<String, OptionValue>,
        mode: &ListMode,
        out: &mut Vec<(FileRef, Vec<Segment>)>,
    ) -> Result<()> {
        let first = text.find(['*', '?', '[', '{']).unwrap_or(text.len());
        let cut = text[..first].rfind('/').map_or(0, |i| i + 1);
        let (prefix, rest) = text.split_at(cut);
        let glob = krishiv_io::partition::glob::Glob::parse(rest)
            .map_err(|e| unsupported(format!("the glob pattern {text}: {e}")))?;
        let at = self.paths.locate_with_options(prefix, options)?;
        let base = at.path.parts().count();
        let mut matched = std::collections::BTreeMap::new();
        // A glob under a logged directory (or below a logged `basePath`) expands over its committed
        // files only: an abandoned append's files would otherwise come back as roots of their own.
        // `outer` is the log's root depth and its files, keyed relative to that root.
        let own = Box::pin(self.committed_files(&at)).await?;
        let outer = match (own, &mode.base) {
            (Some(files), _) => Some((base, Arc::new(files))),
            // The nearest logged ancestor of the glob's prefix (it may sit between the base and the
            // prefix), else the base's own log or its ancestors'.
            (None, Some(root)) if at.path.prefix_matches(&root.path) => {
                match self.logged_ancestor(&at).await? {
                    Some(found) => Some(found),
                    None => self.base_log(root).await?,
                }
            }
            // A glob whose prefix is a partition directory of a logged root.
            (None, None) => self.logged_ancestor(&at).await?,
            _ => None,
        };
        // The committed files this glob covers: each must still be there, or a concurrent overwrite
        // deleted some and the read would silently lose rows.
        let mut expected: std::collections::BTreeSet<Vec<String>> =
            std::collections::BTreeSet::new();
        if let Some((root_depth, files)) = &outer {
            let at_parts: Vec<String> = at.path.parts().map(|p| p.as_ref().to_owned()).collect();
            for rel in files.keys() {
                let mut all: Vec<String> = at_parts[..*root_depth].to_vec();
                all.extend(rel.split('/').map(str::to_owned));
                if all.len() <= base || all[..base] != at_parts[..] {
                    continue;
                }
                let names: Vec<&str> = all[base..].iter().map(String::as_str).collect();
                if names.iter().any(|n| hidden(n)) {
                    continue;
                }
                if (1..=names.len()).any(|n| glob.matches(&names[..n])) {
                    expected.insert(all);
                }
            }
        }
        let mut found = 0_usize;
        for meta in self.io.list(&at.store, &at.path).await? {
            let below: Vec<String> = meta
                .location
                .parts()
                .skip(base)
                .map(|p| p.as_ref().to_owned())
                .collect();
            if let Some((root_depth, files)) = &outer {
                let all: Vec<String> = meta
                    .location
                    .parts()
                    .map(|p| p.as_ref().to_owned())
                    .collect();
                let rel = all[(*root_depth).min(all.len())..].join("/");
                if !files.contains_key(&rel) {
                    continue;
                }
                if expected.contains(&all) {
                    found += 1;
                }
            }
            // The file and each directory above it are candidates; a hidden name (`_krishiv_log`,
            // `.staging`) is never a root of its own, and nothing below one is a candidate.
            for n in 1..=below.len() {
                if below[..n].iter().any(|p| hidden(p)) {
                    break;
                }
                let names: Vec<&str> = below[..n].iter().map(String::as_str).collect();
                if glob.matches(&names) {
                    matched
                        .entry(below[..n].to_vec())
                        .or_insert_with(|| Path::from_iter(meta.location.parts().take(base + n)));
                }
            }
        }
        // An unlogged listing is rechecked: a first logged write may have made its log since.
        if outer.is_none() && self.log_appeared(&at, mode).await? {
            return Err(krishiv_common::error::KrishivConcurrentCommit {
                root: at.display.clone(),
                version: "the version this read started at".to_owned(),
            }
            .into());
        }
        if found < expected.len() {
            return Err(krishiv_common::error::KrishivConcurrentCommit {
                root: at.display.clone(),
                version: "the version this read started at".to_owned(),
            }
            .into());
        }
        if matched.is_empty() {
            return Err(PathNotFound {
                path: format!("{}{rest}", at.display.trim_end_matches('/')),
            }
            .into());
        }
        let inner = ListMode {
            recursive: mode.recursive,
            base: mode.base.clone(),
            logged: outer,
            skip_foreign_dirs: mode.skip_foreign_dirs,
            skip_all_dirs: mode.skip_all_dirs,
        };
        for (names, path) in matched {
            let display = format!("{}/{}", at.display.trim_end_matches('/'), names.join("/"));
            let found = LocatedPath {
                store: at.store.clone(),
                path,
                display,
            };
            Box::pin(self.list_path(&found, &inner, false, out)).await?;
        }
        Ok(())
    }

    /// The `recursiveFileLookup` and `basePath` read options.
    fn list_mode(&self, options: &BTreeMap<String, OptionValue>) -> Result<ListMode> {
        let get = |name: &str| {
            options
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v)
        };
        let recursive = get("recursiveFileLookup")
            .and_then(OptionValue::as_plain)
            .is_some_and(|v| v.trim().eq_ignore_ascii_case("true"));
        let base = match get("basePath") {
            Some(v) => {
                let Some(text) = v.as_plain() else {
                    return Err(unsupported("paths with credentials".to_owned()));
                };
                Some(self.paths.locate_with_options(text, options)?)
            }
            None => None,
        };
        Ok(ListMode {
            recursive,
            base,
            logged: None,
            skip_foreign_dirs: false,
            skip_all_dirs: false,
        })
    }

    /// The `name=value` directories from `mode`'s base path down to the
    /// directories of `dirs` (those below `at`, already split).
    fn base_segments(at: &LocatedPath, mode: &ListMode, dirs: &[String]) -> Result<Vec<Segment>> {
        if mode.recursive {
            return Ok(Vec::new());
        }
        let mut names: Vec<String> = Vec::new();
        let mut display = at.display.clone();
        if let Some(base) = &mode.base {
            let base_parts: Vec<String> =
                base.path.parts().map(|p| p.as_ref().to_owned()).collect();
            let at_parts: Vec<String> = at.path.parts().map(|p| p.as_ref().to_owned()).collect();
            if at_parts.len() < base_parts.len() || at_parts[..base_parts.len()] != base_parts[..] {
                return Err(unsupported(format!(
                    "the basePath {} is not a parent of the path {}",
                    base.display, at.display
                )));
            }
            names.extend(at_parts[base_parts.len()..].iter().cloned());
            display = base.display.clone();
        }
        names.extend(dirs.iter().cloned());
        names
            .iter()
            .map(|dir| {
                parse_segment(dir).ok_or_else(|| {
                    unsupported(format!(
                        "reading the subdirectory {display}/{dir} (it is not a name=value partition directory)"
                    ))
                })
            })
            .collect()
    }

    /// The files the commit log of the directory `at` shows, `None` when it
    /// has no log.
    async fn committed_files(
        &self,
        at: &LocatedPath,
    ) -> Result<Option<BTreeMap<String, krishiv_io::commit_log::AddedFile>>> {
        let log = krishiv_io::commit_log::CommitLog::new(
            &self.io,
            at.store.clone(),
            at.path.clone(),
            at.display.clone(),
        );
        Ok(log
            .snapshot_if_exists()
            .await?
            .map(|snapshot| snapshot.files))
    }

    /// The log of the nearest ancestor of `at`, reached through `name=value` directories only (a
    /// partition directory read by its own path): the ancestor's depth and its committed files,
    /// `None` when no such ancestor has a log.
    async fn logged_ancestor(
        &self,
        at: &LocatedPath,
    ) -> Result<
        Option<(
            usize,
            Arc<BTreeMap<String, krishiv_io::commit_log::AddedFile>>,
        )>,
    > {
        let mut parts: Vec<String> = at.path.parts().map(|p| p.as_ref().to_owned()).collect();
        let mut display = at.display.clone();
        while parts
            .last()
            .is_some_and(|name| parse_segment(name).is_some())
        {
            parts.pop();
            display = display
                .rsplit_once('/')
                .map_or_else(String::new, |(head, _)| head.to_owned());
            let parent = LocatedPath {
                store: at.store.clone(),
                path: Path::from_iter(parts.iter().map(String::as_str)),
                display: display.clone(),
            };
            if let Some(files) = Box::pin(self.committed_files(&parent)).await? {
                return Ok(Some((parts.len(), Arc::new(files))));
            }
        }
        Ok(None)
    }

    /// The log that applies to a read with `basePath` `base`: the base's own, else the nearest
    /// logged ancestor of the base reached through `name=value` directories.
    async fn base_log(
        &self,
        base: &LocatedPath,
    ) -> Result<
        Option<(
            usize,
            Arc<BTreeMap<String, krishiv_io::commit_log::AddedFile>>,
        )>,
    > {
        if let Some(files) = Box::pin(self.committed_files(base)).await? {
            return Ok(Some((base.path.parts().count(), Arc::new(files))));
        }
        self.logged_ancestor(base).await
    }

    /// Whether a commit log exists now for `at` (or for the `basePath` root above it): checked after an
    /// unlogged listing, because a first logged write to an existing directory creates its log before
    /// its tasks write files, so a listing taken in between would accept uncommitted files.
    async fn log_appeared(&self, at: &LocatedPath, mode: &ListMode) -> Result<bool> {
        if Box::pin(self.committed_files(at)).await?.is_some() {
            return Ok(true);
        }
        match &mode.base {
            // The same order as the read itself: the nearest logged ancestor of the path (it may sit
            // between the base and the path), then the base's own log and its ancestors'.
            Some(root) if at.path.prefix_matches(&root.path) => {
                if self.logged_ancestor(at).await?.is_some() {
                    return Ok(true);
                }
                Ok(self.base_log(root).await?.is_some())
            }
            Some(_) => Ok(false),
            None => Ok(self.logged_ancestor(at).await?.is_some()),
        }
    }

    async fn list_path(
        &self,
        at: &LocatedPath,
        mode: &ListMode,
        missing_is_empty: bool,
        out: &mut Vec<(FileRef, Vec<Segment>)>,
    ) -> Result<()> {
        if let Some(meta) = self.io.head(&at.store, &at.path).await? {
            if !at.path.filename().is_some_and(hidden) {
                // The file's own directories above `at` count only with a basePath.
                let parent = at.path.parts().count().saturating_sub(1);
                let probe = LocatedPath {
                    store: at.store.clone(),
                    path: Path::from_iter(at.path.parts().take(parent)),
                    display: at.display.clone(),
                };
                let segs = if mode.base.is_some() {
                    Self::base_segments(&probe, mode, &[])?
                } else {
                    Vec::new()
                };
                out.push((FileRef::from_meta(at.store.clone(), &meta), segs));
            }
            return Ok(());
        }
        if !self.io.is_directory(&at.store, &at.path).await? {
            // A catalog table whose location is not made yet (an external
            // table's first write makes it) holds no rows.
            if missing_is_empty {
                return Ok(());
            }
            return Err(PathNotFound {
                path: at.display.clone(),
            }
            .into());
        }
        let base = at.path.parts().count();
        // A directory with a commit log shows only the files committed there.
        // Boxed: the listing future sits on deep recursion paths (nested
        // scripts), so it must stay small.
        let committed = Box::pin(self.committed_files(at)).await?;
        // A directory below a logged root (a partition subdirectory read with `basePath`, or a glob match)
        // shows only the files that root committed.
        let inherited = match (&committed, &mode.logged, &mode.base) {
            (None, Some(logged), _) => Some(logged.clone()),
            // The nearest logged ancestor of the directory itself (it may sit between the base and
            // the directory), else the base's own log or the base's ancestors'.
            (None, None, Some(base)) => match self.logged_ancestor(at).await? {
                Some(found) => Some(found),
                None => self.base_log(base).await?,
            },
            // A partition directory read by its own path: the log of the root above it.
            (None, None, None) => self.logged_ancestor(at).await?,
            _ => None,
        };
        let mut seen = 0;
        let mut seen_inherited = 0;
        for meta in self.io.list(&at.store, &at.path).await? {
            let below: Vec<_> = meta.location.parts().skip(base).collect();
            if below.iter().any(|p| hidden(p.as_ref())) {
                continue;
            }
            if let Some(files) = &committed {
                let rel = below
                    .iter()
                    .map(AsRef::as_ref)
                    .collect::<Vec<_>>()
                    .join("/");
                if !files.contains_key(&rel) {
                    continue;
                }
                seen += 1;
            } else if let Some((root_base, files)) = &inherited {
                // Below a logged directory a glob matched: only its committed files.
                let rel = meta
                    .location
                    .parts()
                    .skip(*root_base)
                    .map(|p| p.as_ref().to_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                if !files.contains_key(&rel) {
                    continue;
                }
                seen_inherited += 1;
            }
            let dirs = below.len().saturating_sub(1);
            let names: Vec<String> = below[..dirs]
                .iter()
                .map(|p| p.as_ref().to_owned())
                .collect();
            // A catalog table: a directory that is not a `name=value`
            // partition directory holds none of its files (Spark 4.2.0 reads
            // neither an unpartitioned table's subdirectories nor a partitioned
            // one's unregistered locations). Any other read refuses it.
            if (mode.skip_all_dirs && !names.is_empty())
                || (mode.skip_foreign_dirs && names.iter().any(|d| parse_segment(d).is_none()))
            {
                continue;
            }
            let segments = Self::base_segments(at, mode, &names)?;
            out.push((FileRef::from_meta(at.store.clone(), &meta), segments));
        }
        if committed.is_none() && inherited.is_none() && self.log_appeared(at, mode).await? {
            return Err(krishiv_common::error::KrishivConcurrentCommit {
                root: at.display.clone(),
                version: "the version this read started at".to_owned(),
            }
            .into());
        }
        let missing = match (&committed, &inherited) {
            (Some(files), _) => seen < files.len(),
            // The root's snapshot lists the files of this directory too: one that is gone (a
            // concurrent overwrite deleted it) must not shrink the read silently.
            (None, Some((root_base, files))) => {
                let mut prefix = at
                    .path
                    .parts()
                    .skip(*root_base)
                    .map(|p| p.as_ref().to_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                if !prefix.is_empty() {
                    prefix.push('/');
                }
                seen_inherited < files.keys().filter(|k| k.starts_with(&prefix)).count()
            }
            (None, None) => false,
        };
        if missing {
            // A committed file is gone (a concurrent overwrite deleted it).
            return Err(krishiv_common::error::KrishivConcurrentCommit {
                root: at.display.clone(),
                version: "the version this read started at".to_owned(),
            }
            .into());
        }
        Ok(())
    }
}

/// Spark's `StructType.merge`: the columns of `a`, then those only in `b`;
/// a column in both keeps its type (structs merge field by field) and is
/// nullable if either is.
///
/// # Errors
/// `CANNOT_MERGE_INCOMPATIBLE_DATA_TYPE` when a column has two types.
fn merge_schemas(a: &SchemaRef, b: &SchemaRef, case_sensitive: bool) -> Result<SchemaRef> {
    let fields = merge_fields(a.fields(), b.fields(), case_sensitive)?;
    Ok(Arc::new(arrow_schema::Schema::new_with_metadata(
        fields,
        a.metadata().clone(),
    )))
}

fn merge_fields(
    a: &arrow_schema::Fields,
    b: &arrow_schema::Fields,
    case_sensitive: bool,
) -> Result<Vec<FieldRef>> {
    let same = |x: &str, y: &str| {
        if case_sensitive {
            x == y
        } else {
            x.eq_ignore_ascii_case(y)
        }
    };
    let mut out: Vec<FieldRef> = Vec::with_capacity(a.len());
    for f in a {
        match b.iter().find(|g| same(g.name(), f.name())) {
            None => out.push(Arc::clone(f)),
            Some(g) => out.push(merge_field(f, g, case_sensitive)?),
        }
    }
    out.extend(
        b.iter()
            .filter(|g| !a.iter().any(|f| same(f.name(), g.name())))
            .cloned(),
    );
    Ok(out)
}

fn merge_field(a: &FieldRef, b: &FieldRef, case_sensitive: bool) -> Result<FieldRef> {
    use arrow_schema::DataType as Arrow;
    let nullable = a.is_nullable() || b.is_nullable();
    let data_type = match (a.data_type(), b.data_type()) {
        (x, y) if x == y => x.clone(),
        (Arrow::Struct(x), Arrow::Struct(y)) => {
            Arrow::Struct(merge_fields(x, y, case_sensitive)?.into())
        }
        // Collections merge their element (or key and value) types the same way.
        (Arrow::List(x), Arrow::List(y)) => Arrow::List(merge_field(x, y, case_sensitive)?),
        (Arrow::LargeList(x), Arrow::LargeList(y)) => {
            Arrow::LargeList(merge_field(x, y, case_sensitive)?)
        }
        (Arrow::Map(x, sorted), Arrow::Map(y, _)) => {
            Arrow::Map(merge_field(x, y, case_sensitive)?, *sorted)
        }
        (x, y) => {
            let sql = |f: &FieldRef, t: &Arrow| {
                DataType::from_arrow_field(&arrow_schema::Field::new(f.name(), t.clone(), true))
                    .map_or_else(|_| format!("{t}"), |d| d.sql())
            };
            return Err(CannotMergeIncompatibleDataType {
                left: format!("\"{}\"", sql(a, x)),
                right: format!("\"{}\"", sql(b, y)),
            }
            .into());
        }
    };
    Ok(Arc::new(
        a.as_ref()
            .clone()
            .with_data_type(data_type)
            .with_nullable(nullable),
    ))
}

/// How a read lists files: `recursiveFileLookup` (every file below the
/// paths, no partition discovery) and `basePath` (where partition discovery
/// starts).
struct ListMode {
    recursive: bool,
    base: Option<LocatedPath>,
    /// Set while a glob under a directory with a commit log is expanded: how many path parts that
    /// directory has, and the files committed in it. Matched roots below it list only those files.
    logged: Option<(
        usize,
        Arc<BTreeMap<String, krishiv_io::commit_log::AddedFile>>,
    )>,
    /// A catalog table's listing: files under a directory that is not a
    /// `name=value` partition directory are skipped, not refused.
    skip_foreign_dirs: bool,
    /// An unpartitioned catalog table's listing: every subdirectory is
    /// skipped, `name=value` ones too (its files are its root's).
    skip_all_dirs: bool,
}

/// The partition columns of a permanent table and the partitions its
/// catalog registered, which a read of it lists ([`FileReads::listing_known`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnownPartitions {
    /// The partition columns, in partition order (the table's last columns).
    pub columns: Vec<String>,
    /// Each registered partition's values in partition order, as the
    /// directory names hold them unescaped (`None` for the default
    /// partition, NULL).
    pub registered: Vec<Vec<Option<String>>>,
}

/// The partition columns of a known table: its `declared` columns, the
/// last of `schema`, typed as declared, when the partition columns
/// `discovered` from its files name exactly those (in order); none when there
/// are no partition directories (an empty or unpartitioned table). The
/// listing has kept the registered partitions' files only (`is_registered`,
/// as Spark's `CatalogFileIndex` reads only the registered locations), so a
/// mismatch is left only for an unpartitioned table with partition
/// directories. A partition column of a type whose directory
/// value the scan does not read yet is refused (only strings, integers and
/// dates are: a timestamp is local time in the session zone, which the scan
/// does not have).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` in those cases.
fn declared_partitions(
    schema: &SchemaRef,
    declared: &KnownPartitions,
    discovered: &[FieldRef],
) -> Result<Vec<FieldRef>> {
    use arrow_schema::DataType as T;
    if discovered.is_empty() {
        return Ok(Vec::new());
    }
    let mismatch =
        || unsupported("reading the subdirectory of a table (partition discovery)".to_owned());
    let names = &declared.columns;
    let fields = schema.fields();
    if names.len() != discovered.len() || names.len() > fields.len() {
        return Err(mismatch());
    }
    let tail = &fields[fields.len() - names.len()..];
    let mut out = Vec::with_capacity(tail.len());
    for ((field, name), found) in tail.iter().zip(names).zip(discovered) {
        if !field.name().eq_ignore_ascii_case(name) || !name.eq_ignore_ascii_case(found.name()) {
            return Err(mismatch());
        }
        if !matches!(
            field.data_type(),
            T::Utf8 | T::Utf8View | T::Int8 | T::Int16 | T::Int32 | T::Int64 | T::Date32
        ) {
            return Err(unsupported(format!(
                "reading the partition column {} of type {} of a table",
                field.name(),
                field.data_type()
            )));
        }
        out.push(Arc::clone(field));
    }
    Ok(out)
}

/// Whether a file's partition directories (`segs`) are one of the
/// `registered` partitions: exactly the declared `columns`, in order and as
/// written (Spark reads the exact location `<root>/p=1`, never `P=1`), with
/// registered values.
fn is_registered(
    columns: &[String],
    registered: &HashSet<&[Option<String>]>,
    segs: &[Segment],
) -> bool {
    if segs.len() != columns.len()
        || !segs
            .iter()
            .zip(columns)
            .all(|((name, _), column)| name == column)
    {
        return false;
    }
    let values: Vec<Option<String>> = segs.iter().map(|(_, v)| v.clone()).collect();
    registered.contains(values.as_slice())
}

/// `schema` followed by `partitions`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` when a partition column has the name of a
/// data column.
/// The text format's checks of a user schema (Spark's
/// `TextFileFormat.verifySchema`): exactly one column, a string.
fn check_user_schema(format: &str, schema: &StructType) -> Result<()> {
    if format != "text" {
        return Ok(());
    }
    let [field] = schema.fields() else {
        return Err(LegacyErrorTemp1290 {
            schema_size: schema.fields().len().to_string(),
        }
        .into());
    };
    if !field.data_type().is_string_type() {
        return Err(UnsupportedDataTypeForDatasource {
            column_name: format!("`{}`", field.name().replace('`', "``")),
            column_type: format!("\"{}\"", field.data_type().sql()),
            format: "Text".to_owned(),
        }
        .into());
    }
    Ok(())
}

/// The schema of a read with the user's `schema` and the partition columns
/// `discovered` under its paths, as Spark 4.2.0 combines them: the user's
/// columns that are not partition columns, in the user's order, then the
/// partition columns in directory order. A partition column the schema
/// names (by `case_sensitive` comparison) takes the schema's name and type,
/// and each of its values must cast to that type; one it does not name is
/// typed by inference (`infer`). Every column is nullable, whatever the
/// schema says.
///
/// # Errors
/// The partition discovery errors, `INVALID_PARTITION_VALUE` for a value
/// that does not cast, and `KRISHIV_UNSUPPORTED_FEATURE` for a partition
/// column of a timestamp type (its directory value is local time in the
/// session zone, which the scan does not have).
fn user_schema_columns(
    schema: &StructType,
    discovered: &[(String, Vec<Segment>)],
    infer: bool,
    case_sensitive: bool,
) -> Result<(SchemaRef, Vec<FieldRef>)> {
    let same = |a: &str, b: &str| {
        if case_sensitive {
            a == b
        } else {
            krishiv_types::equals_ignore_case(a, b)
        }
    };
    // Names and layout only: the types come from the schema or below.
    let found = partition_columns(discovered, false)?;
    let mut partitions: Vec<FieldRef> = Vec::with_capacity(found.len());
    for (i, column) in found.iter().enumerate() {
        let one: Vec<(String, Vec<Segment>)> = discovered
            .iter()
            .map(|(path, segs)| (path.clone(), vec![segs[i].clone()]))
            .collect();
        let Some(named) = schema
            .fields()
            .iter()
            .find(|f| same(f.name(), column.name()))
        else {
            let mut typed = partition_columns(&one, infer)?;
            partitions.push(
                typed.pop().ok_or_else(|| {
                    unsupported(format!("the partition column {}", column.name()))
                })?,
            );
            continue;
        };
        let data_type = named.data_type();
        if matches!(data_type, DataType::Timestamp | DataType::TimestampNtz) {
            return Err(unsupported(format!(
                "a {} partition column ({}): the session time zone does not reach partition discovery yet",
                data_type.sql(),
                named.name()
            )));
        }
        let field = data_type.to_arrow_field(named.name(), true);
        let mut checked: HashSet<&str> = HashSet::new();
        for (_, segs) in &one {
            let Some(value) = segs[0].1.as_deref() else {
                continue;
            };
            if checked.insert(value) && krishiv_io::constant_column(&field, Some(value), 1).is_err()
            {
                return Err(InvalidPartitionValue {
                    value: format!("'{value}'"),
                    data_type: format!("\"{}\"", data_type.sql()),
                    column_name: format!("`{}`", named.name().replace('`', "``")),
                }
                .into());
            }
        }
        partitions.push(Arc::new(field));
    }
    let data = schema
        .fields()
        .iter()
        .filter(|f| !partitions.iter().any(|p| same(f.name(), p.name())))
        .map(|f| Arc::new(f.data_type().to_arrow_field(f.name(), true)));
    let fields: Vec<FieldRef> = data.chain(partitions.iter().cloned()).collect();
    Ok((Arc::new(arrow_schema::Schema::new(fields)), partitions))
}

fn with_partition_columns(schema: SchemaRef, partitions: &[FieldRef]) -> Result<SchemaRef> {
    if partitions.is_empty() {
        return Ok(schema);
    }
    // A data column with a partition column's name gives way to it: the
    // value comes from the directory and the column comes last (Spark 4.2.0).
    let fields: Vec<FieldRef> = schema
        .fields()
        .iter()
        .filter(|f| {
            !partitions
                .iter()
                .any(|p| f.name().eq_ignore_ascii_case(p.name()))
        })
        .cloned()
        .chain(partitions.iter().cloned())
        .collect();
    Ok(Arc::new(arrow_schema::Schema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    )))
}

/// The columns of a listing's schema.
fn listing_columns(listing: &FileListing) -> std::result::Result<Vec<TableColumn>, KrishivError> {
    listing
        .schema
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

impl FileSchemaResolver for FileReads {
    /// The located path's qualified form (`file:/abs/path`); a path with
    /// credentials stays as written (its read is refused).
    /// The located path's qualified form; a path with credentials stays as
    /// written (its read is refused). Qualifying never builds a store, so it
    /// needs no options and never fails on store configuration: the store is
    /// instantiated at listing from the read's options.
    fn qualify(&self, path: &OptionValue) -> std::result::Result<OptionValue, KrishivError> {
        match path.as_plain() {
            // Not the working directory: Spark refuses the empty location of `` parquet.`` `` as
            // `INVALID_EMPTY_LOCATION`. A DataFrame or Connect read of `""` reaches this too; Spark 4.2 raises a
            // plain `IllegalArgumentException` (no error class) there, and reading the working directory was wrong.
            Some("") => Err(InvalidEmptyLocation {
                location: String::new(),
            }
            .into()),
            Some(text) => Ok(OptionValue::from_user(
                "path",
                self.paths.qualified_display(text)?,
            )),
            None => Ok(path.clone()),
        }
    }

    fn source_kind(&self, name: &str) -> SourceKind {
        let name = name.to_ascii_lowercase();
        if is_spark_file_source(&name) || self.formats.contains(&name) {
            SourceKind::File
        } else if SPARK_OTHER_SOURCES.contains(&name.as_str()) {
            SourceKind::Other
        } else {
            SourceKind::Unknown
        }
    }

    fn infer<'a>(
        &'a self,
        format: &'a str,
        paths: &'a [OptionValue],
        options: &'a BTreeMap<String, OptionValue>,
    ) -> BoxFuture<'a, std::result::Result<Vec<TableColumn>, KrishivError>> {
        Box::pin(async move {
            // Spark's `ResolveSQLOnFile` refuses these before it lists anything.
            for text in paths.iter().filter_map(OptionValue::as_plain) {
                check_sql_location(text)?;
            }
            let listing = self.listing(format, paths, options).await?;
            listing_columns(&listing)
        })
    }

    fn with_schema<'a>(
        &'a self,
        format: &'a str,
        paths: &'a [OptionValue],
        options: &'a BTreeMap<String, OptionValue>,
        schema: &'a StructType,
    ) -> BoxFuture<'a, std::result::Result<Vec<TableColumn>, KrishivError>> {
        Box::pin(async move {
            for text in paths.iter().filter_map(OptionValue::as_plain) {
                check_sql_location(text)?;
            }
            let listing = self
                .listing_with_schema(format, paths, options, Some(schema))
                .await?;
            listing_columns(&listing)
        })
    }
}

/// The error Spark raises for a `` format.`location` `` on `http` or `https`: `FAILED_READ_FILE.UNSUPPORTED_FILE_SYSTEM`
/// (Hadoop's HTTP file systems do not implement `listStatus`).
fn check_sql_location(text: &str) -> Result<()> {
    let class = match scheme(text) {
        Some("http") => "org.apache.hadoop.fs.http.HttpFileSystem",
        Some("https") => "org.apache.hadoop.fs.http.HttpsFileSystem",
        _ => return Ok(()),
    };
    Err(FailedReadFileUnsupportedFileSystem {
        path: text.to_owned(),
        file_system_class: class.to_owned(),
        method: "listStatus".to_owned(),
    }
    .into())
}

/// Whether Spark skips a file or directory named `name`.
fn hidden(name: &str) -> bool {
    name.starts_with('.')
        || name.starts_with('_') && !name.contains('=')
        || name.ends_with("._COPYING_")
}

/// A format as Spark names it in messages (`Parquet`, `JSON`).
fn display_name(format: &str) -> String {
    match format {
        "parquet" => "Parquet".to_owned(),
        "avro" => "Avro".to_owned(),
        "orc" => "ORC".to_owned(),
        "json" => "JSON".to_owned(),
        "csv" => "CSV".to_owned(),
        "text" => "Text".to_owned(),
        other => other.to_owned(),
    }
}

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn discovered(values: &[&str]) -> Vec<(String, Vec<Segment>)> {
        values
            .iter()
            .map(|v| {
                (
                    format!("/t/k={v}/f"),
                    vec![krishiv_io::parse_segment(&format!("k={v}")).unwrap()],
                )
            })
            .collect()
    }

    fn ddl(columns: &[(&str, DataType, bool)]) -> StructType {
        StructType::new(
            columns
                .iter()
                .map(|(n, t, nl)| krishiv_types::StructField::new(*n, t.clone(), *nl)),
        )
    }

    fn shape(schema: &SchemaRef) -> Vec<(String, String, bool)> {
        schema
            .fields()
            .iter()
            .map(|f| {
                let t = DataType::from_arrow_field(f).unwrap();
                (f.name().clone(), t.simple_string(), f.is_nullable())
            })
            .collect()
    }

    #[test]
    fn a_user_schema_puts_partition_columns_last_typed_as_it_names_them() {
        let named = ddl(&[("K", DataType::Long, false), ("v", DataType::String, false)]);
        let (schema, parts) =
            user_schema_columns(&named, &discovered(&["1", "2"]), true, false).unwrap();
        // Moved last, the schema's name and type, nullable.
        assert_eq!(
            shape(&schema),
            [
                ("v".to_owned(), "string".to_owned(), true),
                ("K".to_owned(), "bigint".to_owned(), true),
            ]
        );
        assert_eq!(parts.len(), 1);
        // Case-sensitive, `K` is no partition column: `k` is appended, inferred.
        let (schema, _) =
            user_schema_columns(&named, &discovered(&["1", "2"]), true, true).unwrap();
        assert_eq!(
            shape(&schema),
            [
                ("K".to_owned(), "bigint".to_owned(), true),
                ("v".to_owned(), "string".to_owned(), true),
                ("k".to_owned(), "int".to_owned(), true),
            ]
        );
        // A value that does not cast is Spark's error.
        let e = user_schema_columns(
            &ddl(&[("k", DataType::Integer, true)]),
            &discovered(&["1", "abc"]),
            true,
            false,
        )
        .unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_PARTITION_VALUE");
        assert_eq!(
            e.message_parameters()
                .iter()
                .map(|(k, v)| (*k, v.as_str()))
                .collect::<Vec<_>>(),
            [
                ("value", "'abc'"),
                ("dataType", "\"INT\""),
                ("columnName", "`k`")
            ]
        );
    }

    #[test]
    fn the_text_format_takes_one_string_column() {
        assert!(check_user_schema("text", &ddl(&[("line", DataType::String, false)])).is_ok());
        let two = ddl(&[("a", DataType::String, true), ("b", DataType::String, true)]);
        assert_eq!(
            check_user_schema("text", &two)
                .unwrap_err()
                .error_class()
                .name(),
            "_LEGACY_ERROR_TEMP_1290"
        );
        let int = ddl(&[("line", DataType::Integer, true)]);
        assert_eq!(
            check_user_schema("text", &int)
                .unwrap_err()
                .error_class()
                .name(),
            "UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE"
        );
        assert!(check_user_schema("csv", &two).is_ok());
    }

    fn locate(path: &str) -> Result<LocatedPath> {
        LocalPaths::new(StoreId::new("file"), "/work/dir").locate(path)
    }

    #[test]
    fn local_paths_are_qualified_like_hadoop() {
        for (input, path, display) in [
            ("/a/b.parquet", "a/b.parquet", "file:/a/b.parquet"),
            ("nope", "work/dir/nope", "file:/work/dir/nope"),
            ("../x/./y/", "work/x/y", "file:/work/x/y"),
            ("file:/a/b", "a/b", "file:/a/b"),
            ("file:///a/b", "a/b", "file:/a/b"),
            ("file://localhost/a", "a", "file:/a"),
            ("/", "", "file:/"),
        ] {
            let at = locate(input).unwrap();
            assert_eq!(at.path.as_ref(), path, "{input}");
            assert_eq!(at.display, display, "{input}");
            assert_eq!(at.store, StoreId::new("file"));
            assert_eq!(locate(&at.display).unwrap(), at, "{input}");
        }
        for input in ["s3a://bucket/k", "ftp://x/y", "file://other/a"] {
            let e = locate(input).unwrap_err();
            assert_eq!(
                e.error_class().name(),
                "KRISHIV_UNSUPPORTED_FEATURE",
                "{input}"
            );
        }
        // A colon later in a name is no scheme.
        assert_eq!(locate("/a:b").unwrap().display, "file:/a:b");
        assert_eq!(locate("1a:b").unwrap().display, "file:/work/dir/1a:b");
    }

    fn object_paths() -> ObjectPaths {
        use krishiv_common::config::Config;
        use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};

        let config = Config::new();
        ObjectPaths::new(
            StoreId::new("file"),
            "/work/dir",
            IoScheduler::new(IoConfig::from_config(&config)),
            Arc::new(StoreRegistry::new(Arc::new(config))),
        )
    }

    /// Qualifying never builds a store: cloud paths print their canonical
    /// form whatever the build, local paths print as `LocalPaths` does, and
    /// local locating is the same through the options form.
    #[test]
    fn object_paths_qualify_without_building_stores() {
        let paths = object_paths();
        for (input, display) in [
            ("s3a://bucket/k", "s3a://bucket/k"),
            ("gs://bucket/k", "gs://bucket/k"),
            ("/a/b.parquet", "file:/a/b.parquet"),
            ("rel", "file:/work/dir/rel"),
            ("file:/a/b", "file:/a/b"),
        ] {
            assert_eq!(paths.qualified_display(input).unwrap(), display, "{input}");
        }
        assert_eq!(
            paths
                .qualified_display("ftp://x/y")
                .unwrap_err()
                .error_class()
                .name(),
            "KRISHIV_UNSUPPORTED_FEATURE"
        );
        let options = BTreeMap::new();
        assert_eq!(
            paths.locate_with_options("/a/b", &options).unwrap(),
            paths.locate("/a/b").unwrap()
        );
    }

    /// Without the `cloud` feature a cloud path locates to the unsupported
    /// error rather than a store, with or without per-read options.
    #[cfg(not(feature = "cloud"))]
    #[test]
    fn cloud_paths_need_the_cloud_feature_to_locate() {
        let paths = object_paths();
        let options = BTreeMap::new();
        for e in [
            paths.locate("s3://bucket/k").unwrap_err(),
            paths
                .locate_with_options("s3://bucket/k", &options)
                .unwrap_err(),
        ] {
            assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
        }
    }

    #[cfg(feature = "cloud")]
    #[test]
    fn cloud_urls_locate_in_their_bucket_stores() {
        use krishiv_common::config::Config;
        use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};

        let (config, _) = Config::from_pairs([
            ("fs.s3a.endpoint", "http://127.0.0.1:9000"),
            ("fs.azure.account.name", "acct"),
            (
                "fs.azure.account.key.acct",
                "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==",
            ),
        ])
        .unwrap();
        let sched = IoScheduler::new(IoConfig::from_config(&config));
        let paths = ObjectPaths::new(
            StoreId::new("file"),
            "/work/dir",
            sched,
            Arc::new(StoreRegistry::new(Arc::new(config))),
        );
        for (input, store, path, display) in [
            ("s3a://bucket/k", "s3://bucket", "k", "s3a://bucket/k"),
            ("s3://bucket/", "s3://bucket", "", "s3://bucket"),
            ("gs://bucket/k", "gs://bucket", "k", "gs://bucket/k"),
            (
                "abfss://cont@acct.dfs.core.windows.net/k",
                "abfss://cont@acct",
                "k",
                "abfss://cont@acct.dfs.core.windows.net/k",
            ),
            ("az://cont/k", "abfss://cont", "k", "az://cont/k"),
            ("/a/b.parquet", "file", "a/b.parquet", "file:/a/b.parquet"),
        ] {
            let at = paths.locate(input).unwrap();
            // A store built from `fs.*` settings is registered as
            // `<store>#cfg:<hash>`; its display is the location.
            assert_eq!(
                at.store.to_string(),
                StoreId::new(store).to_string(),
                "{input}"
            );
            if store == "file" {
                assert_eq!(at.store, StoreId::new(store), "{input}");
            } else {
                assert!(
                    at.store.key().contains("#cfg:"),
                    "{input}: {}",
                    at.store.key()
                );
            }
            assert_eq!(at.path.as_ref(), path, "{input}");
            assert_eq!(at.display, display, "{input}");
            assert_eq!(paths.locate(&at.display).unwrap(), at, "{input}");
        }
        for input in ["ftp://x/y", "s3a://AKIA:secret@bucket/k"] {
            let e = paths.locate(input).unwrap_err();
            assert_eq!(
                e.error_class().name(),
                "KRISHIV_UNSUPPORTED_FEATURE",
                "{input}"
            );
        }
    }

    #[test]
    fn hidden_names_follow_spark() {
        for name in [".x", "_SUCCESS", "_temporary", "a._COPYING_"] {
            assert!(hidden(name), "{name}");
        }
        for name in ["a.parquet", "_k=1", "k=_1"] {
            assert!(!hidden(name), "{name}");
        }
        assert_eq!(display_name("parquet"), "Parquet");
        assert_eq!(display_name("avro"), "Avro");
        assert_eq!(display_name("json"), "JSON");
        assert_eq!(display_name("mock"), "mock");
    }

    #[cfg(feature = "cloud")]
    #[test]
    fn cloud_locating_honours_per_read_options() {
        use std::collections::BTreeMap;

        use krishiv_common::config::Config;
        use krishiv_io::{IoConfig, IoScheduler, StoreRegistry};

        fn options(pairs: &[(&str, &str)]) -> BTreeMap<String, OptionValue> {
            pairs
                .iter()
                .map(|&(k, v)| (k.to_owned(), OptionValue::from_user(k, v)))
                .collect()
        }

        // An invalid session flag fails a plain locate ...
        let (config, _) = Config::from_pairs([("fs.s3a.path.style.access", "banana")]).unwrap();
        let sched = IoScheduler::new(IoConfig::from_config(&config));
        let paths = ObjectPaths::new(
            StoreId::new("file"),
            "/work/dir",
            sched,
            Arc::new(StoreRegistry::new(Arc::new(config))),
        );
        let error = paths.locate("s3://bucket/k").unwrap_err();
        assert_eq!(error.error_class().name(), "KRISHIV_INVALID_CONFIG_VALUE");
        // ... but a valid per-read option overrides it, and an invalid
        // per-read option beats a valid session value the other way.
        let at = paths
            .locate_with_options(
                "s3://bucket/k",
                &options(&[("fs.s3a.path.style.access", "true")]),
            )
            .unwrap();
        assert_eq!(at.display, "s3://bucket/k");
        let (good, _) = Config::from_pairs([("fs.s3a.path.style.access", "true")]).unwrap();
        let sched = IoScheduler::new(IoConfig::from_config(&good));
        let paths = ObjectPaths::new(
            StoreId::new("file"),
            "/work/dir",
            sched,
            Arc::new(StoreRegistry::new(Arc::new(good))),
        );
        let error = paths
            .locate_with_options(
                "s3://bucket/k",
                &options(&[("fs.s3a.path.style.access", "banana")]),
            )
            .unwrap_err();
        assert_eq!(error.error_class().name(), "KRISHIV_INVALID_CONFIG_VALUE");
        assert!(!error.to_string().contains("banana"));
        // Qualifying never builds a store: it succeeds despite the invalid
        // session flag, and unknown schemes still fail as before.
        assert_eq!(
            paths.qualified_display("s3://bucket/k").unwrap(),
            "s3://bucket/k"
        );
        for input in ["ftp://x/y", "s3a://AKIA:secret@bucket/k"] {
            assert_eq!(
                paths
                    .locate_with_options(input, &options(&[]))
                    .unwrap_err()
                    .error_class()
                    .name(),
                "KRISHIV_UNSUPPORTED_FEATURE",
                "{input}"
            );
            assert_eq!(
                paths
                    .qualified_display(input)
                    .unwrap_err()
                    .error_class()
                    .name(),
                "KRISHIV_UNSUPPORTED_FEATURE",
                "{input}"
            );
        }
    }
}
