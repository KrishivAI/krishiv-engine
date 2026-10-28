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
//! A name is hidden when it starts with `.`, or with `_` without a `=` in
//! it, or ends with `._COPYING_`; hidden directories are skipped whole.
//! The format infers the schema from the files by path (Spark without
//! `mergeSchema`): Parquet reads the first file's footer, CSV samples
//! all of them. Not yet supported, failing with
//! `KRISHIV_UNSUPPORTED_FEATURE`: glob patterns, subdirectories (Spark's
//! partition discovery), paths with a scheme other than `file`, and file
//! sources without a [`FileFormat`] here, and merging the files' schemas
//! (`mergeSchema`).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::config::Config;
use krishiv_common::error::{
    KrishivError, KrishivUnsupportedFeature, PathNotFound, UnableToInferSchema,
};
use krishiv_common::secret::OptionValue;
use krishiv_io::{FileFormat, FileRef, FormatOptions, IoContext, StoreId};
use krishiv_plan::analyzer::{FileSchemaResolver, SourceKind, TableColumn};
use krishiv_types::DataType;
use object_store::path::Path;

use crate::error::Result;

/// Spark 4.2's built-in data sources that read files (`` name.`path` ``
/// reads them), as a Spark run classifies them.
const SPARK_FILE_SOURCES: &[&str] = &[
    "parquet",
    "orc",
    "json",
    "csv",
    "text",
    "binaryfile",
    "xml",
    "libsvm",
];

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
    /// The schema the format infers from the listed files.
    pub schema: SchemaRef,
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
    formats: Vec<Arc<dyn FileFormat>>,
    paths: Arc<dyn PathLocator>,
    session: Arc<Config>,
    listed: Mutex<Vec<(ReadKey, Arc<FileListing>)>>,
}

impl fmt::Debug for FileReads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileReads")
            .field(
                "formats",
                &self.formats.iter().map(|x| x.name()).collect::<Vec<_>>(),
            )
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
        formats: Vec<Arc<dyn FileFormat>>,
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
    /// `PATH_NOT_FOUND`, `UNABLE_TO_INFER_SCHEMA`, option errors,
    /// `KRISHIV_UNSUPPORTED_FEATURE` (see the module docs) and storage
    /// failures.
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
        // Spark checks the paths before the format.
        let mut files = Vec::new();
        for p in paths {
            let Some(text) = p.as_plain() else {
                return Err(unsupported("paths with credentials".to_owned()));
            };
            if text.contains(['*', '?', '[', '{']) {
                return Err(unsupported("glob patterns in paths".to_owned()));
            }
            let at = self.paths.locate(text)?;
            self.list_path(&at, &mut files).await?;
        }
        files.sort_by(|a, b| a.path.as_ref().cmp(b.path.as_ref()));
        let Some(f) = self
            .formats
            .iter()
            .find(|f| f.name().eq_ignore_ascii_case(format))
        else {
            return Err(unsupported(format!("the {format} file format")));
        };
        let format = Arc::clone(f);
        if self.merges_schemas(format.name(), options) {
            return Err(unsupported(
                "merging the schemas of files (mergeSchema)".to_owned(),
            ));
        }
        let (options, _) = format.options().parse(options, &self.session)?;
        if files.is_empty() {
            return Err(UnableToInferSchema {
                format: display_name(format.name()),
            }
            .into());
        }
        let schema = format
            .infer_schema(&self.io, &files, &options)
            .await
            .map_err(|e| e.into_error(&files[0].display_path()))?;
        Ok(FileListing {
            format,
            files: files.into(),
            schema,
            options,
        })
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

    /// Adds the files `at` names to `out`.
    async fn list_path(&self, at: &LocatedPath, out: &mut Vec<FileRef>) -> Result<()> {
        if let Some(meta) = self.io.head(&at.store, &at.path).await? {
            if !at.path.filename().is_some_and(hidden) {
                out.push(FileRef::from_meta(at.store.clone(), &meta));
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
            if let [dir, _, ..] = below.as_slice() {
                return Err(unsupported(format!(
                    "reading the subdirectory {}/{} (partition discovery)",
                    at.display,
                    dir.as_ref()
                )));
            }
            out.push(FileRef::from_meta(at.store.clone(), &meta));
        }
        Ok(())
    }
}

impl FileSchemaResolver for FileReads {
    /// The located path's qualified form (`file:/abs/path`); a path with
    /// credentials stays as written (its read is refused).
    fn qualify(&self, path: &OptionValue) -> std::result::Result<OptionValue, KrishivError> {
        match path.as_plain() {
            Some(text) => Ok(OptionValue::from_user(
                "path",
                self.paths.locate(text)?.display,
            )),
            None => Ok(path.clone()),
        }
    }

    fn source_kind(&self, name: &str) -> SourceKind {
        let name = name.to_ascii_lowercase();
        if SPARK_FILE_SOURCES.contains(&name.as_str())
            || self.formats.iter().any(|f| f.name() == name)
        {
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
}
