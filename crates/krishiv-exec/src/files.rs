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
//! file. A table with a known schema does not discover them yet.
//!
//! A name is hidden when it starts with `.`, or with `_` without a `=` in
//! it, or ends with `._COPYING_`; hidden directories are skipped whole.
//! The format infers the schema from the files by path (Spark without
//! `mergeSchema`): Parquet reads the first file's footer, CSV samples
//! all of them. Local paths and cloud URLs (`s3://`, `s3a://`, `gs://`,
//! `abfs://`, `abfss://`, `az://`, resolved by [`ObjectPaths`] through the
//! [`StoreRegistry`]) are supported. Not yet supported, failing with
//! `KRISHIV_UNSUPPORTED_FEATURE`: subdirectories that are
//! not `name=value` partition directories, paths with any other scheme, file sources Spark
//! knows but the engine has no [`FileFormat`] for, and merging the files'
//! schemas (`mergeSchema`). A name Spark does not know at all fails with
//! Spark's `DATA_SOURCE_NOT_FOUND`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_schema::{FieldRef, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::config::Config;
use krishiv_common::error::{
    CannotMergeIncompatibleDataType, KrishivError, KrishivUnsupportedFeature, PathNotFound,
    UnableToInferSchema,
};
use krishiv_common::secret::OptionValue;
use krishiv_io::{
    FileFormat, FileRef, FormatOptions, FormatRegistry, IoContext, IoScheduler, Segment, StoreId,
    StoreRegistry, is_spark_file_source, parse_object_url, parse_segment, partition_columns,
};
use krishiv_plan::analyzer::{FileSchemaResolver, SourceKind, TableColumn};
use krishiv_types::DataType;
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
        let (store, key, display) = self.stores.ensure(&self.sched, path)?;
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
        let (store, key, display) = self
            .stores
            .ensure_with_options(&self.sched, path, options)?;
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

/// What identifies a read: format, paths and options.
type ReadKey = (String, Vec<OptionValue>, BTreeMap<String, OptionValue>);

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
        let key = (format.to_ascii_lowercase(), paths.to_vec(), options.clone());
        if let Some(l) = self.cached(&key) {
            return Ok(l);
        }
        let listing = Arc::new(self.list(&key.0, paths, options).await?);
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
    /// schema. Nothing is inferred and nothing is cached, so every query
    /// lists the table afresh.
    ///
    /// # Errors
    /// The listing errors of [`Self::listing`], except that no files is not
    /// an error.
    pub async fn listing_known(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
        schema: SchemaRef,
    ) -> Result<Arc<FileListing>> {
        let listing = self
            .list_with(&format.to_ascii_lowercase(), paths, options, Some(schema))
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

    async fn list(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
    ) -> Result<FileListing> {
        self.list_with(format, paths, options, None).await
    }

    /// [`Self::list`]; with `known` the schema is given, not inferred.
    async fn list_with(
        &self,
        format: &str,
        paths: &[OptionValue],
        options: &BTreeMap<String, OptionValue>,
        known: Option<SchemaRef>,
    ) -> Result<FileListing> {
        let mode = self.list_mode(options)?;
        // Spark checks the paths before the format.
        let mut listed: Vec<(FileRef, Vec<Segment>)> = Vec::new();
        for p in paths {
            let Some(text) = p.as_plain() else {
                return Err(unsupported("paths with credentials".to_owned()));
            };
            if krishiv_io::partition::glob::has_glob(text) {
                self.list_glob(text, options, &mode, &mut listed).await?;
                continue;
            }
            // The store is instantiated from the plan's option values
            // (plans stay pure data): the read's `fs.*` options override
            // the session configuration.
            let at = self.paths.locate_with_options(text, options)?;
            self.list_path(&at, &mode, &mut listed).await?;
        }
        listed.sort_by(|a, b| a.0.path.as_ref().cmp(b.0.path.as_ref()));
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
        let partition_columns = partition_columns(&discovered, self.infers_partition_types())?;
        if let Some(schema) = known {
            if !partition_columns.is_empty() {
                return Err(unsupported(
                    "reading the subdirectory of a table (partition discovery)".to_owned(),
                ));
            }
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
            // Every file's own schema, merged by column name in first-seen order.
            let mut merged: Option<SchemaRef> = None;
            for file in &files {
                let one = format
                    .infer_schema(&self.io, std::slice::from_ref(file), &options)
                    .await
                    .map_err(|e| e.into_error(&file.display_path()))?;
                merged = Some(match merged {
                    None => one,
                    Some(m) => merge_schemas(&m, &one)?,
                });
            }
            merged.ok_or_else(|| {
                KrishivError::from(UnableToInferSchema {
                    format: display_name(format.name()),
                })
            })?
        } else {
            format
                .infer_schema(&self.io, &files, &options)
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

    /// Adds the files `at` names to `out`, each with the `name=value`
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
        for meta in self.io.list(&at.store, &at.path).await? {
            let below: Vec<String> = meta
                .location
                .parts()
                .skip(base)
                .map(|p| p.as_ref().to_owned())
                .collect();
            // The file and each directory above it are candidates.
            for n in 1..=below.len() {
                let names: Vec<&str> = below[..n].iter().map(String::as_str).collect();
                if glob.matches(&names) {
                    matched
                        .entry(below[..n].to_vec())
                        .or_insert_with(|| Path::from_iter(meta.location.parts().take(base + n)));
                }
            }
        }
        if matched.is_empty() {
            return Err(PathNotFound {
                path: format!("{}{rest}", at.display.trim_end_matches('/')),
            }
            .into());
        }
        for (names, path) in matched {
            let display = format!("{}/{}", at.display.trim_end_matches('/'), names.join("/"));
            let found = LocatedPath {
                store: at.store.clone(),
                path,
                display,
            };
            self.list_path(&found, mode, out).await?;
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
        Ok(ListMode { recursive, base })
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

    async fn list_path(
        &self,
        at: &LocatedPath,
        mode: &ListMode,
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
            return Err(PathNotFound {
                path: at.display.clone(),
            }
            .into());
        }
        let base = at.path.parts().count();
        for meta in self.io.list(&at.store, &at.path).await? {
            let below: Vec<_> = meta.location.parts().skip(base).collect();
            if below.iter().any(|p| hidden(p.as_ref())) {
                continue;
            }
            let dirs = below.len().saturating_sub(1);
            let names: Vec<String> = below[..dirs]
                .iter()
                .map(|p| p.as_ref().to_owned())
                .collect();
            let segments = Self::base_segments(at, mode, &names)?;
            out.push((FileRef::from_meta(at.store.clone(), &meta), segments));
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
fn merge_schemas(a: &SchemaRef, b: &SchemaRef) -> Result<SchemaRef> {
    let fields = merge_fields(a.fields(), b.fields())?;
    Ok(Arc::new(arrow_schema::Schema::new_with_metadata(
        fields,
        a.metadata().clone(),
    )))
}

fn merge_fields(a: &arrow_schema::Fields, b: &arrow_schema::Fields) -> Result<Vec<FieldRef>> {
    let mut out: Vec<FieldRef> = Vec::with_capacity(a.len());
    for f in a {
        match b.iter().find(|g| g.name() == f.name()) {
            None => out.push(Arc::clone(f)),
            Some(g) => out.push(merge_field(f, g)?),
        }
    }
    out.extend(
        b.iter()
            .filter(|g| !a.iter().any(|f| f.name() == g.name()))
            .cloned(),
    );
    Ok(out)
}

fn merge_field(a: &FieldRef, b: &FieldRef) -> Result<FieldRef> {
    let nullable = a.is_nullable() || b.is_nullable();
    let data_type = match (a.data_type(), b.data_type()) {
        (x, y) if x == y => x.clone(),
        (arrow_schema::DataType::Struct(x), arrow_schema::DataType::Struct(y)) => {
            arrow_schema::DataType::Struct(merge_fields(x, y)?.into())
        }
        (x, y) => {
            let sql = |f: &FieldRef, t: &arrow_schema::DataType| {
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
}

/// `schema` followed by `partitions`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` when a partition column has the name of a
/// data column.
fn with_partition_columns(schema: SchemaRef, partitions: &[FieldRef]) -> Result<SchemaRef> {
    if partitions.is_empty() {
        return Ok(schema);
    }
    if let Some(p) = partitions.iter().find(|p| {
        schema
            .fields()
            .iter()
            .any(|f| f.name().eq_ignore_ascii_case(p.name()))
    }) {
        return Err(unsupported(format!(
            "a partition column ({}) with the name of a data column",
            p.name()
        )));
    }
    let fields: Vec<FieldRef> = schema
        .fields()
        .iter()
        .cloned()
        .chain(partitions.iter().cloned())
        .collect();
    Ok(Arc::new(arrow_schema::Schema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    )))
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
            let listing = self.listing(format, paths, options).await?;
            listing
                .schema
                .fields()
                .iter()
                .map(|f| {
                    Ok(TableColumn {
                        name: f.name().clone(),
                        data_type: DataType::from_arrow_field(f)?,
                        nullable: f.is_nullable(),
                    })
                })
                .collect()
        })
    }
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
            assert_eq!(at.store, StoreId::new(store), "{input}");
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
