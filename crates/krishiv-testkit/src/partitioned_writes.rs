// SPDX-License-Identifier: Apache-2.0
//! Partitioned writes Spark reads back (P2-T082b acceptance). Each case of
//! `tests/fixtures/partitioned-writes/cases.json` is written by Krishiv (`partitionBy`), then read by Spark 4.2.0
//! (`tests/scripts/gen_partitioned_write_goldens.py`, which runs the `write-partition-cases` binary first). The
//! fixture `tests/fixtures/spark-4.2.0/partitioned-write-goldens.json` keeps what Spark read: the schema, the rows
//! and the partition directories. The test `partitioned_write_goldens` writes every case again and checks that the
//! directories are the ones Spark read and that Krishiv reads Spark's schema and rows back.
//!
//! Rows on both sides are each column cast to `STRING` (`NULL` spelled out) joined by `|`, sorted, so the
//! comparison is of Spark-exact text and needs no Arrow-to-JSON rendering.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

/// One partitioned write.
#[derive(Debug, Clone)]
pub struct Case {
    /// The name of the case and of its directory.
    pub id: String,
    /// The file format (`parquet`, `csv`, `json`).
    pub format: String,
    /// The partition columns, outermost first.
    pub partition_by: Vec<String>,
    /// Writer and reader options (`header` for CSV).
    pub options: BTreeMap<String, String>,
    /// The query whose rows are written.
    pub query: String,
}

/// What a read of a case's directory returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBack {
    /// `struct<...>` in Spark's simple-string form.
    pub schema: String,
    /// The rows as text, sorted.
    pub rows: Vec<String>,
}

/// The case file under `tests/`.
#[must_use]
pub fn cases_path() -> PathBuf {
    crate::tests_dir().join("fixtures/partitioned-writes/cases.json")
}

/// The Spark fixture under `tests/`.
#[must_use]
pub fn goldens_path() -> PathBuf {
    crate::tests_dir().join("fixtures/spark-4.2.0/partitioned-write-goldens.json")
}

/// The cases of `path`.
///
/// # Errors
/// An unreadable or malformed file, as text.
pub fn load(path: &Path) -> Result<Vec<Case>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = |what: &str| format!("{}: {what}", path.display());
    let text_of = |v: &Value, key: &str| -> Result<String, String> {
        v.get(key)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| bad(&format!("a case without `{key}`")))
    };
    doc.get("cases")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("no `cases` array"))?
        .iter()
        .map(|c| {
            let partition_by = c
                .get("partition_by")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("a case without `partition_by`"))?
                .iter()
                .map(|p| {
                    p.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| bad("a partition column that is not text"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let options = c
                .get("options")
                .and_then(Value::as_object)
                .ok_or_else(|| bad("a case without `options`"))?
                .iter()
                .map(|(k, v)| {
                    v.as_str()
                        .map(|v| (k.clone(), v.to_owned()))
                        .ok_or_else(|| bad("an option value that is not text"))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            Ok(Case {
                id: text_of(c, "id")?,
                format: text_of(c, "format")?,
                partition_by,
                options,
                query: text_of(c, "query")?,
            })
        })
        .collect()
}

/// Writes `case` with Krishiv into `dir` (created by the writer).
///
/// # Errors
/// The query's or the write's error.
pub fn write(session: &Session, case: &Case, dir: &Path) -> Result<(), KrishivError> {
    block_on(async {
        let df = session.sql(&case.query).await?;
        let mut writer = df
            .write()
            .format(&case.format)
            .partition_by(&case.partition_by);
        for (k, v) in &case.options {
            writer = writer.option(k, v);
        }
        writer.save(&dir.to_string_lossy()).await
    })
}

/// Reads `dir` back with Krishiv, as Spark's `spark.read.format(f).options(o).load(dir)` does.
///
/// # Errors
/// The read's error.
pub fn read_back(session: &Session, case: &Case, dir: &Path) -> Result<ReadBack, KrishivError> {
    block_on(async {
        let mut reader = session.read().format(&case.format);
        for (k, v) in &case.options {
            reader = reader.option(k, v);
        }
        let df = reader.load([dir.to_string_lossy().into_owned()]);
        let analyzed = df.analyzed().await?;
        let fields: Vec<String> = analyzed
            .output
            .iter()
            .map(|c| format!("{}:{}", c.name, c.data_type.simple_string()))
            .collect();
        let schema = format!("struct<{}>", fields.join(","));
        let parts: Vec<String> = analyzed
            .output
            .iter()
            .map(|c| {
                format!(
                    "coalesce(CAST(`{}` AS STRING), 'NULL')",
                    c.name.replace('`', "``")
                )
            })
            .collect();
        let text = df.select_expr([format!("concat_ws('|', {}) AS row", parts.join(", "))])?;
        let mut rows = Vec::new();
        for batch in text.collect().await? {
            let a = batch.column(0);
            for i in 0..batch.num_rows() {
                rows.push(a.as_string_opt::<i32>().map_or_else(
                    || a.as_string_view().value(i).to_owned(),
                    |s| s.value(i).to_owned(),
                ));
            }
        }
        rows.sort();
        Ok(ReadBack { schema, rows })
    })
}

/// The directories below `dir` that hold data files, relative and sorted (`p=1`, `a=x/b=2`). Names starting
/// with `_` or `.` (the commit log, `_SUCCESS`, checksums) are skipped, as Spark's listing skips them.
///
/// # Errors
/// A listing error, as text.
pub fn partition_dirs(dir: &Path) -> Result<Vec<String>, String> {
    fn walk(root: &Path, at: &Path, out: &mut Vec<String>) -> Result<(), String> {
        let mut has_file = false;
        for entry in std::fs::read_dir(at).map_err(|e| format!("{}: {e}", at.display()))? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(['_', '.']) {
                continue;
            }
            if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
                walk(root, &entry.path(), out)?;
            } else {
                has_file = true;
            }
        }
        if has_file {
            let rel = at.strip_prefix(root).map_err(|e| e.to_string())?;
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort();
    Ok(out)
}

/// The `write-partition-cases CASES.json OUT_DIR` command: writes every case under `OUT_DIR/<id>` and returns
/// the directories written.
///
/// # Errors
/// A usage error, an unreadable case file, or a case that fails to write, as text.
pub fn cli(args: &[String]) -> Result<Vec<String>, String> {
    let [cases, out] = args else {
        return Err("usage: write-partition-cases CASES.json OUT_DIR".to_owned());
    };
    let cases = load(Path::new(cases))?;
    let session = Session::new().map_err(|e| e.to_string())?;
    let mut dirs = Vec::new();
    for case in &cases {
        let dir = Path::new(out).join(&case.id);
        write(&session, case, &dir).map_err(|e| format!("{}: {e}", case.id))?;
        dirs.push(dir.display().to_string());
    }
    Ok(dirs)
}
