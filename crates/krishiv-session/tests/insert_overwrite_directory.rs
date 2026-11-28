// SPDX-License-Identifier: Apache-2.0
//! `INSERT OVERWRITE DIRECTORY` (acceptance RS-10) against the Spark 4.2.0 run of
//! `tests/spark-golden/write-semantics/insert_overwrite_directory.json`: each case runs its statement
//! into a fresh directory and compares the error class, or the files written (part files by kind) and
//! the rows read back (unordered). Cases that need a setup are covered by the named tests below. A case
//! that does not match is named in `KNOWN_GAPS` with its reason and must still differ.
#![cfg(test)]

use arrow_array::Array;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// A path as SQL text: `/` separators, since `\` starts an escape in a string literal (Windows).
fn sql_path(p: impl AsRef<std::path::Path>) -> String {
    p.as_ref().to_str().unwrap().replace('\\', "/")
}

/// Cases whose Krishiv result differs from Spark's, with the reason; each must still differ.
const KNOWN_GAPS: &[(&str, &str)] = &[(
    "IOD-05-orc",
    "the engine writes no ORC (KRISHIV_UNSUPPORTED_FEATURE, as df.write.orc); queue-sonnet.md item IOD-05",
)];

/// Cases not replayed by the loop: they need a setup, a configuration, or touch the working
/// directory or an unknown scheme (whose Spark error has no class).
const SKIPPED: &[&str] = &["IOD-35-relative-path", "IOD-37-unknown-scheme"];

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/spark-golden/write-semantics/insert_overwrite_directory.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A file name with its task number, UUID and file counter replaced, as the fixture writes them
/// (the writer starts a task's next file at `-c001`).
fn normalised(name: &str) -> String {
    let Some(rest) = name.strip_prefix("part-") else {
        return name.to_owned();
    };
    let Some(at) = rest.rfind("-c0") else {
        return name.to_owned();
    };
    let tail = rest[at + 2..].trim_start_matches(|c: char| c.is_ascii_digit());
    format!("part-<N>-<UUID>-c000{tail}")
}

/// The directory's entries, normalised, each kind once (Spark ran with one task; the part count
/// follows the query's partitions). The writer's commit log (`_krishiv_log/`, hidden from readers
/// as every `_` name is) is the engine's own and left out.
fn files(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            let n = normalised(&e.file_name().to_string_lossy());
            if e.file_type().unwrap().is_dir() {
                format!("{n}/")
            } else {
                n
            }
        })
        .filter(|n| !n.starts_with('.') && n != "_krishiv_log/")
        .collect();
    names.sort();
    names.dedup();
    names
}

fn cell(x: &Value) -> String {
    match x {
        Value::Null => "NULL".to_owned(),
        Value::String(s) => s.clone(),
        o => o.to_string(),
    }
}

fn expected_rows(rows: &Value) -> Vec<String> {
    let mut v: Vec<String> = rows
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(cell)
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    v.sort();
    v
}

/// The rows of `dir` read as `read_back` says, every column as text, sorted; and the read schema's
/// column names.
fn read_back(s: &Session, dir: &str, read: &Value) -> Result<(Vec<String>, Vec<String>), String> {
    let mut reader = s
        .read()
        .format(read["format"].as_str().unwrap_or("parquet"));
    if let Some(options) = read["options"].as_object() {
        for (k, v) in options {
            reader = reader.option(k, v.as_str().unwrap_or_default());
        }
    }
    let df = reader.load([dir]);
    let schema = block_on(df.schema()).map_err(|e| e.to_string())?;
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let cols: Vec<String> = names
        .iter()
        .map(|n| {
            format!(
                "coalesce(cast(`{}` AS string), 'NULL')",
                n.replace('`', "``")
            )
        })
        .collect();
    let line = format!("concat_ws('|', {}) AS line", cols.join(", "));
    let out = block_on(df.select_expr([line]).map_err(|e| e.to_string())?.collect())
        .map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for b in out {
        let a = b.column(0);
        for i in 0..a.len() {
            rows.push(arrow_cast::display::array_value_to_string(a, i).map_err(|e| e.to_string())?);
        }
    }
    rows.sort();
    Ok((rows, names))
}

/// Field names of a `struct<a:int,b:string>` text, split at the top level.
fn schema_names(schema: &str) -> Vec<String> {
    let inner = schema
        .strip_prefix("struct<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or_default();
    let (mut names, mut depth, mut field) = (Vec::new(), 0usize, String::new());
    for ch in inner.chars() {
        match ch {
            '<' | '(' => depth += 1,
            '>' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                names.push(std::mem::take(&mut field));
                continue;
            }
            _ => {}
        }
        field.push(ch);
    }
    if !field.is_empty() {
        names.push(field);
    }
    // The name is up to the last top-level `:` before the type.
    names
        .into_iter()
        .map(|f| {
            let mut depth = 0usize;
            let mut cut = f.len();
            for (i, ch) in f.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => depth = depth.saturating_sub(1),
                    ':' if depth == 0 => {
                        cut = i;
                        break;
                    }
                    _ => {}
                }
            }
            f[..cut].to_owned()
        })
        .collect()
}

/// A message's text up to its position and SQL context (` (line 1, pos 50)`, `== SQL ==`), which Spark
/// adds to parse errors and the engine places elsewhere.
fn headline(message: &str) -> String {
    let first = message.trim_start().lines().next().unwrap_or("");
    match first.rfind(" (line ") {
        Some(at) => first[..at].to_owned(),
        None => first.to_owned(),
    }
}

/// Runs one case; `Err` names the first difference.
#[expect(
    clippy::unwrap_in_result,
    reason = "a malformed fixture or an unusable temporary directory fails the test outright"
)]
fn run_case(case: &Value) -> Result<(), String> {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("out");
    let stmt = case["stmt"]
        .as_str()
        .unwrap()
        .replace("<DIR>", &sql_path(&dir))
        .replace("<TMP>", &sql_path(tmp.path()));
    let s = Session::new().unwrap();
    let expect = &case["expect"];
    let result = block_on(async { s.sql(&stmt).await?.collect().await });
    if expect["kind"] == "error" {
        let want = expect["class"].as_str().unwrap();
        return match result {
            Ok(_) => Err(format!("ran, Spark fails with {want}")),
            Err(e) if e.error_class().name() != want => {
                Err(format!("{}: {e}, Spark: {want}", e.error_class().name()))
            }
            Err(e) => {
                let want_state = expect["sqlstate"].as_str();
                let want_message = expect["message"]
                    .as_str()
                    .unwrap()
                    .replace("<DIR>", &sql_path(&dir))
                    .replace("<TMP>", &sql_path(tmp.path()));
                let state = e.sql_state();
                if state != want_state {
                    Err(format!("SQLSTATE {state:?}: {e}, Spark: {want_state:?}"))
                } else if headline(&e.to_string()) != headline(&want_message) {
                    Err(format!("message {e}, Spark: {want_message}"))
                } else {
                    Ok(())
                }
            }
        };
    }
    let batches = result.map_err(|e| format!("{}: {e}", e.error_class().name()))?;
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    let columns = batches
        .first()
        .map_or(0, arrow_array::RecordBatch::num_columns);
    if rows != 0 || columns != 0 {
        return Err(format!(
            "result has {rows} rows, {columns} columns; Spark: none"
        ));
    }
    let target = if stmt.contains("'file://") {
        tmp.path().join("u")
    } else if stmt.contains("/a/b/c'") {
        tmp.path().join("a/b/c")
    } else {
        dir
    };
    let mut want_files: Vec<String> = expect["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap().to_owned())
        .collect();
    want_files.sort();
    let got_files = files(&target);
    if got_files != want_files {
        return Err(format!("files {got_files:?}, Spark: {want_files:?}"));
    }
    let raw = expect
        .get("read_back")
        .filter(|r| r.get("rows").is_none())
        .and_then(|r| r.get("raw_lines"));
    if let Some(raw) = raw {
        // The text of the part files, line by line: what the writer put on disk.
        let mut got = Vec::new();
        for e in std::fs::read_dir(&target).map_err(|e| e.to_string())? {
            let path = e.map_err(|e| e.to_string())?.path();
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            if name.starts_with("part-") {
                let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                got.extend(text.lines().map(str::to_owned));
            }
        }
        got.sort();
        let mut want: Vec<String> = raw
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l.as_str().unwrap().to_owned())
            .collect();
        want.sort();
        if got != want {
            return Err(format!("lines {got:?}, Spark: {want:?}"));
        }
    } else if let Some(read) = expect.get("read_back") {
        let (got, names) = read_back(&s, &sql_path(&target), read)?;
        let want = expected_rows(&read["rows"]);
        if got != want {
            return Err(format!("rows {got:?}, Spark: {want:?}"));
        }
        if let Some(schema) = read["schema"].as_str() {
            let want = schema_names(schema);
            if names != want {
                return Err(format!("columns {names:?}, Spark: {want:?}"));
            }
        }
    }
    Ok(())
}

#[test]
fn insert_overwrite_directory_matches_spark() {
    let fixture = fixture();
    let mut failures = Vec::new();
    let mut ran = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let setup = case.get("setup").is_some()
            || case
                .get("conf")
                .and_then(Value::as_object)
                .is_some_and(|c| !c.contains_key("spark.sql.catalogImplementation"));
        // With the `delta` feature (workspace test runs unify it on) `USING delta` is a known source,
        // so the case of an unknown provider named delta does not apply.
        let delta_known = cfg!(feature = "delta") && id == "IOD-22-unknown-delta";
        if setup || delta_known || SKIPPED.contains(&id) {
            continue;
        }
        ran += 1;
        let outcome = run_case(case);
        let gap = KNOWN_GAPS.iter().find(|(g, _)| *g == id);
        match (outcome, gap) {
            (Ok(()), Some((_, why))) => {
                failures.push(format!("{id}: matches now, leave KNOWN_GAPS ({why})"));
            }
            (Err(e), None) => failures.push(format!("{id}: {e}")),
            _ => {}
        }
    }
    assert!(ran > 30, "only {ran} cases ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn sql(s: &Session, stmt: &str) {
    block_on(async { s.sql(stmt).await?.collect().await }).unwrap();
}

/// IOD-07, IOD-34: whatever was at the path is replaced, a regular file too.
#[test]
fn the_directory_is_replaced_wholesale() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("out");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    std::fs::write(dir.join("old.txt"), b"x").unwrap();
    std::fs::write(dir.join("sub/deep.txt"), b"x").unwrap();
    let s = Session::new().unwrap();
    let d = sql_path(&dir);
    sql(
        &s,
        &format!("INSERT OVERWRITE DIRECTORY '{d}' USING parquet SELECT 1 AS a"),
    );
    // Every old file goes. A difference from Spark: the writer (df.write too) removes files, not
    // the directories they leave empty, as on an object store, which has none.
    let sub: Vec<_> = std::fs::read_dir(dir.join("sub"))
        .map(|r| r.map(|e| e.unwrap().file_name()).collect())
        .unwrap_or_default();
    assert!(sub.is_empty(), "{sub:?}");
    assert_eq!(
        files(&dir),
        ["_SUCCESS", "part-<N>-<UUID>-c000.snappy.parquet", "sub/"]
    );
    let file = tmp.path().join("file");
    std::fs::write(&file, b"x").unwrap();
    let f = sql_path(&file);
    sql(
        &s,
        &format!("INSERT OVERWRITE DIRECTORY '{f}' USING parquet SELECT 1 AS a"),
    );
    assert_eq!(
        files(&file),
        ["_SUCCESS", "part-<N>-<UUID>-c000.snappy.parquet"]
    );
}

/// IOD-31: an empty result still replaces the directory and writes a part file with the schema.
#[test]
fn an_empty_result_writes_the_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("out");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("old.txt"), b"x").unwrap();
    let s = Session::new().unwrap();
    let d = sql_path(&dir);
    sql(
        &s,
        &format!("INSERT OVERWRITE DIRECTORY '{d}' USING parquet SELECT 1 AS a WHERE false"),
    );
    assert!(!dir.join("old.txt").exists());
    let (rows, names) = read_back(&s, &d, &serde_json::json!({"format": "parquet"})).unwrap();
    assert!(rows.is_empty());
    assert_eq!(names, ["a"]);
}

/// IOD-23..27, a difference from Spark (which deletes the directory, then fails reading it and
/// leaves it empty): the writer's logged overwrite reads the old files before it removes them, so
/// a query that reads the directory it overwrites succeeds with the rows it read.
#[test]
fn a_query_reading_its_target_writes_what_it_read() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("out");
    let s = Session::new().unwrap();
    let d = sql_path(&dir);
    sql(
        &s,
        &format!("INSERT OVERWRITE DIRECTORY '{d}' USING parquet SELECT id FROM range(3)"),
    );
    sql(
        &s,
        &format!(
            "INSERT OVERWRITE DIRECTORY '{d}' USING parquet SELECT id + 10 AS id FROM parquet.`{d}`"
        ),
    );
    let (rows, _) = read_back(&s, &d, &serde_json::json!({"format": "parquet"})).unwrap();
    assert_eq!(rows, ["10", "11", "12"]);
}

/// IOD-22: a source that writes no files is refused when the command runs, as `_LEGACY_ERROR_TEMP_2233`.
#[test]
fn a_source_that_writes_no_files_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path().join("out");
    let s = Session::new().unwrap();
    let stmt = format!(
        "INSERT OVERWRITE DIRECTORY '{}' USING noop SELECT 1 AS a",
        sql_path(&d)
    );
    let e = block_on(async { s.sql(&stmt).await?.collect().await }).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::LegacyErrorTemp2233,
        "{e}"
    );
}
