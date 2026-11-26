// SPDX-License-Identifier: Apache-2.0
//! `input_file_name()`, `input_file_block_start()` and `input_file_block_length()` against Spark 4.2.0: every case of
//! `tests/fixtures/spark-4.2.0/input-file-goldens.json` (generator `tests/scripts/gen_input_file_goldens.py`, a real
//! `local[1]` run; research notes `research/spark/input-file-name/notes.md`) runs through a session over files written
//! with the fixture's names and ids, and must give Spark's rows (sorted, as strings) or error class and SQLSTATE.
//!
//! The files differ in bytes from Spark's (another Parquet writer), so a block length that Spark printed as its file's
//! size is compared with the size of ours. The split of the big file depends on the engine's split plan (Spark's
//! depends on its default parallelism: `local[1]` and `local[2]` differ), so those cases assert the plan's
//! invariants instead: the blocks cover the file contiguously and every row is read once.
//! Cases that do not match yet are listed in [`KNOWN_GAPS`] with the reason, and must still mismatch.
//!
//! Recorded divergences outside the fixture (module docs of `krishiv-plan` `analyzer/input_file.rs`): an Iceberg
//! or Delta scan gives the no-file values (no `_metadata` column there; asserted by `krishiv-plan`
//! `tests/input_file.rs` `versioned_table_scans_count_but_give_the_constants`, as these tests have no Iceberg or
//! Delta catalog); a data column named `_metadata` hides the file ([`a_metadata_data_column_gives_the_no_file_value`]);
//! an inner join whose file side Spark broadcasts gives Spark `''`, Krishiv the file; EXPLAIN shows the rewrite.
#![cfg(test)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

/// Spark's data directory in the fixture's SQL text.
const SPARK_ROOT: &str = "/w/data";

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/input-file-goldens.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Writes the rows of `sql` as the one file `<root>/<dir>/<name>`.
fn one(s: &Session, root: &Path, sql: &str, format: &str, dir: &str, name: &str) {
    let tmp = root.join(format!("_tmp_{name}"));
    let df = block_on(s.sql(&format!("SELECT /*+ REPARTITION(1) */ * FROM ({sql})"))).unwrap();
    let w = df.write().mode("overwrite").option("header", "true");
    block_on(w.format(format).save(tmp.to_str().unwrap())).unwrap();
    let parts: Vec<PathBuf> = std::fs::read_dir(&tmp)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy();
            n.starts_with("part-") && !n.ends_with(".crc")
        })
        .collect();
    assert_eq!(parts.len(), 1, "{name}: {parts:?}");
    std::fs::create_dir_all(root.join(dir)).unwrap();
    std::fs::rename(&parts[0], root.join(dir).join(name)).unwrap();
    std::fs::remove_dir_all(&tmp).unwrap();
}

/// The fixture's files and views (`setup` in the fixture), under `root`.
fn setup(s: &Session, root: &Path) {
    let ints = |v: &[i32]| {
        let rows: Vec<String> = v.iter().map(|i| format!("({i})")).collect();
        format!("SELECT * FROM VALUES {} AS t(id)", rows.join(", "))
    };
    one(s, root, &ints(&[1, 2]), "parquet", "a", "a1.parquet");
    one(s, root, &ints(&[3]), "parquet", "a", "a2.parquet");
    one(s, root, &ints(&[1, 10]), "parquet", "b", "b1.parquet");
    one(s, root, &ints(&[1, 7]), "csv", "c", "c1.csv");
    one(s, root, &ints(&[2, 8]), "json", "d", "d1.json");
    // The big file in row groups of 800 rows (about 64 KB, as Spark wrote it) (the engine's writer rolls files over at a target size).
    let batches = block_on(async {
        let sql = "SELECT id, sha2(cast(id AS STRING), 256) AS h FROM range(0, 200000)";
        s.sql(sql).await?.collect().await
    })
    .unwrap();
    std::fs::create_dir_all(root.join("big")).unwrap();
    let file = std::fs::File::create(root.join("big/big.parquet")).unwrap();
    let props = parquet::file::properties::WriterProperties::builder()
        .set_max_row_group_row_count(Some(800))
        .build();
    let mut w =
        parquet::arrow::ArrowWriter::try_new(file, batches[0].schema(), Some(props)).unwrap();
    for b in &batches {
        w.write(b).unwrap();
    }
    w.close().unwrap();
    let path = |d: &str| format!("{}/{d}", root_text(root));
    block_on(async {
        s.read()
            .parquet([path("a")])
            .create_or_replace_temp_view("a")
            .await?;
        s.read()
            .parquet([path("b")])
            .create_or_replace_temp_view("b")
            .await?;
        s.read()
            .format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load([path("c")])
            .create_or_replace_temp_view("c")
            .await?;
        s.read()
            .format("json")
            .load([path("d")])
            .create_or_replace_temp_view("d")
            .await?;
        s.read()
            .parquet([path("big")])
            .create_or_replace_temp_view("big")
            .await?;
        s.sql(&format!(
            "CREATE TABLE t USING parquet LOCATION '{}'",
            path("a")
        ))
        .await?
        .collect()
        .await?;
        Ok::<_, KrishivError>(())
    })
    .unwrap();
}

fn cell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "null".to_owned();
    }
    match a.data_type() {
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::LargeUtf8 => a.as_string::<i64>().value(i).to_owned(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Boolean => a.as_boolean().value(i).to_string(),
        t => panic!("result type {t}"),
    }
}

fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |i| b.columns().iter().map(|c| cell(c, i)).collect())
        })
        .collect();
    out.sort();
    out
}

/// A DataFrame-API case (`DF:...`): the generator's `runDf`.
async fn run_df(s: &Session, root: &Path, name: &str) -> Result<Vec<RecordBatch>, KrishivError> {
    let a = s.read().parquet([root.join("a").to_str().unwrap()]);
    let df = match name {
        "DF:select" => a.select_expr(["input_file_name()", "id"])?,
        "DF:filter" => a
            .filter_expr("contains(input_file_name(), 'a1')")?
            .select_expr(["input_file_name()", "id"])?,
        "DF:blocks" => a.select_expr([
            "input_file_name()",
            "input_file_block_start()",
            "input_file_block_length()",
            "id",
        ])?,
        other => panic!("unknown DataFrame case {other}"),
    };
    df.collect().await
}

/// Spark's file sizes by file (`setup.files`), and ours.
struct Sizes {
    spark: BTreeMap<String, u64>,
    ours: BTreeMap<String, u64>,
}

impl Sizes {
    fn new(g: &Value, root: &Path) -> Self {
        let mut spark = BTreeMap::new();
        let mut ours = BTreeMap::new();
        for f in g["setup"]["files"].as_array().unwrap() {
            // `<root>/a/a1.parquet (452 bytes)`
            let f = f.as_str().unwrap();
            let (path, size) = f.split_once(" (").unwrap();
            let rel = path.strip_prefix("<root>/").unwrap().to_owned();
            let size: u64 = size.trim_end_matches(" bytes)").parse().unwrap();
            ours.insert(
                rel.clone(),
                std::fs::metadata(root.join(&rel)).unwrap().len(),
            );
            spark.insert(rel, size);
        }
        Self { spark, ours }
    }

    /// Spark's row with `<root>` read as `root` and each file size printed beside a file read as ours.
    fn row(&self, row: &[Value], root: &str) -> Vec<String> {
        let cells: Vec<String> = row
            .iter()
            .map(|v| {
                v.as_str()
                    .map_or_else(|| v.to_string(), |s| s.replace("<root>", root))
            })
            .collect();
        let file = row
            .iter()
            .filter_map(Value::as_str)
            .find_map(|c| self.spark.keys().find(|k| c.ends_with(&format!("/{k}"))));
        let Some(file) = file else { return cells };
        cells
            .into_iter()
            .map(|c| {
                if c == self.spark[file].to_string() {
                    self.ours[file].to_string()
                } else {
                    c
                }
            })
            .collect()
    }
}

/// The big file's split cases: the invariants of any split plan. With `split` (`maxPartitionBytes`) every block
/// but the last is that long. A split that holds no row group's midpoint reads no rows, so it has no group: the
/// last one or two (the last row group's second half, then the footer) may be missing.
fn check_splits(
    sql: &str,
    rows: &[Vec<String>],
    size: u64,
    split: Option<u64>,
) -> Result<(), String> {
    let n = |s: &str| s.parse::<u64>().map_err(|e| format!("{s}: {e}"));
    if sql.contains("GROUP BY 1, 2") {
        // (start, length, count) per block: contiguous from 0 to the file size.
        let mut blocks = rows
            .iter()
            .map(|r| Ok((n(&r[0])?, n(&r[1])?, n(&r[2])?)))
            .collect::<Result<Vec<_>, String>>()?;
        blocks.sort_unstable();
        let mut at = 0;
        for (start, len, _) in &blocks {
            if *start != at || *len == 0 || split.is_some_and(|s| *len > s) {
                return Err(format!("blocks not contiguous: {blocks:?}"));
            }
            at = start + len;
        }
        let total: u64 = blocks.iter().map(|b| b.2).sum();
        let tail = size.saturating_sub(at);
        if at > size || tail >= 2 * split.unwrap_or(1) || total != 200_000 {
            return Err(format!(
                "blocks end at {at} of {size}, {total} rows: {blocks:?}"
            ));
        }
        return Ok(());
    }
    // count(*), min(s), max(s), min(l), max(l)
    let [count, min_s, max_s, min_l, max_l] = rows
        .first()
        .ok_or("no row")?
        .iter()
        .map(|c| n(c))
        .collect::<Result<Vec<_>, String>>()?[..]
    else {
        return Err(format!("{rows:?}"));
    };
    let bad_len = split.is_some_and(|s| max_l > s);
    if count != 200_000
        || min_s != 0
        || max_s >= size
        || min_l == 0
        || max_l > size
        || max_s + min_l > size
        || bad_len
    {
        return Err(format!("not a split plan of {size} bytes: {rows:?}"));
    }
    Ok(())
}

/// `(case, other)`: the case's one row may be any row of case `other` (a `LIMIT` without an order).
const ANY_ROW_OF: &[(usize, usize)] = &[(12, 9)];

/// Cases Krishiv does not match yet, by fixture position, with the reason. Each must still mismatch: a fix
/// removes its line.
const KNOWN_GAPS: &[(usize, &str)] = &[
    (
        13,
        "the analyzer drops a REPARTITION hint, so no exchange breaks the pipeline (Spark: '')",
    ),
    (
        23,
        "a sort-merge join (autoBroadcastJoinThreshold=-1) gives Spark '' on its probe side; Krishiv's join keeps the file",
    ),
    (
        35,
        "CACHE TABLE ca AS SELECT * FROM a does not serve later reads of a from the cache (Spark substitutes it: '')",
    ),
    (
        54,
        "a DataFrame temporary view over files offers `_metadata` here; Spark's does not (UNRESOLVED_COLUMN)",
    ),
    (
        66,
        "a sort-merge semi join (autoBroadcastJoinThreshold=-1) gives Spark ''; Krishiv's join keeps the file",
    ),
];

fn run_case(s: &Session, root: &Path, case: &Value) -> Result<Vec<RecordBatch>, KrishivError> {
    let root_s = &root_text(root);
    let conf = s.conf();
    let settings = case["conf"].as_object().unwrap();
    for (k, v) in settings {
        conf.set(k, v.as_str().unwrap())?;
    }
    let sql = case["sql"].as_str().unwrap().replace(SPARK_ROOT, root_s);
    let result = block_on(async {
        for pre in case["pre"].as_array().into_iter().flatten() {
            let pre = pre.as_str().unwrap().replace(SPARK_ROOT, root_s);
            s.sql(&pre).await?.collect().await?;
        }
        if sql.starts_with("DF:") {
            run_df(s, root, &sql).await
        } else {
            s.sql(&sql).await?.collect().await
        }
    });
    for k in settings.keys() {
        conf.unset(k)?;
    }
    block_on(async {
        s.sql("CLEAR CACHE").await?.collect().await?;
        s.sql("DROP VIEW IF EXISTS ca").await?.collect().await?;
        s.sql("DROP VIEW IF EXISTS av2").await?.collect().await
    })?;
    result
}

/// The mismatch of one case, empty when it matches.
fn check(s: &Session, root: &Path, sizes: &Sizes, case: &Value, any_of: Option<&Value>) -> String {
    let sql = case["sql"].as_str().unwrap();
    let result = run_case(s, root, case);
    if let Some(err) = case.get("error") {
        let class = err["class"].as_str().unwrap();
        let state = err["sqlstate"].as_str().unwrap();
        return match result {
            Err(e) if e.error_class().name() == class && e.sql_state() == Some(state) => {
                String::new()
            }
            Err(e) => format!(
                "error {} {}: {e}; Spark {class} {state}",
                e.error_class().name(),
                e.sql_state().unwrap_or_default()
            ),
            Ok(b) => format!("rows {:?}; Spark {class}", rows_of(&b)),
        };
    }
    let got = match result {
        Ok(b) => rows_of(&b),
        Err(e) => return format!("failed: {e}"),
    };
    if sql.contains("FROM big") && sql.contains("input_file_block_start") {
        let split = case["conf"]["spark.sql.files.maxPartitionBytes"]
            .as_str()
            .map(|v| v.parse().unwrap());
        return check_splits(sql, &got, sizes.ours["big/big.parquet"], split)
            .err()
            .unwrap_or_default();
    }
    let root_s = &root_text(root);
    let mut want: Vec<Vec<String>> = case["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| sizes.row(r.as_array().unwrap(), root_s))
        .collect();
    want.sort();
    // `LIMIT 1` without `ORDER BY`: Spark's row is the first file's, ours that of whichever file's scan
    // delivers first; it must be one of the rows of the query without the limit.
    if let Some(all) = any_of {
        let all: Vec<Vec<String>> = all["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| sizes.row(r.as_array().unwrap(), root_s))
            .collect();
        return if got.len() == 1 && all.contains(&got[0]) {
            String::new()
        } else {
            format!("rows\n  got  {got:?}\n  want one of {all:?}")
        };
    }
    if got == want {
        String::new()
    } else {
        format!("rows\n  got  {got:?}\n  want {want:?}")
    }
}

#[test]
fn input_file_functions_match_spark() {
    let g = fixture();
    let dir = tempfile::tempdir().unwrap();
    // The fixture's paths print `<root>` for Spark's data directory: ours, canonical (no symlinked temp dir).
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let s = Session::new().unwrap();
    setup(&s, &root);
    let sizes = Sizes::new(&g, &root);
    let mut report = String::new();
    let mut gaps = 0;
    let cases = g["cases"].as_array().unwrap();
    for (n, case) in cases.iter().enumerate() {
        let sql = case["sql"].as_str().unwrap();
        let any_of = ANY_ROW_OF
            .iter()
            .find(|(c, _)| *c == n)
            .map(|(_, o)| &cases[*o]);
        let one = check(&s, &root, &sizes, case, any_of);
        match (KNOWN_GAPS.iter().find(|(g, _)| *g == n), one.is_empty()) {
            (Some(_), false) => gaps += 1,
            (Some((_, why)), true) => {
                let _ = writeln!(
                    report,
                    "{n} `{sql}`: now matches Spark; remove it from KNOWN_GAPS ({why})"
                );
            }
            (None, true) => {}
            (None, false) => {
                let _ = writeln!(report, "{n} `{sql}`: {one}");
            }
        }
    }
    assert!(report.is_empty(), "{report}");
    assert_eq!(gaps, KNOWN_GAPS.len());
    assert_eq!(cases.len(), 73);
}

/// A session with the views `a` (`a1.parquet`: ids 1, 2; `a2.parquet`: id 3) and `b` (`b1.parquet`: ids 1, 10),
/// for the Krishiv-authored cases below (no Spark run behind them; each states the rule it follows).
fn small() -> (tempfile::TempDir, PathBuf, Session) {
    let dir = tempfile::tempdir().unwrap();
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let s = Session::new().unwrap();
    let ints = |v: &[i32]| {
        let rows: Vec<String> = v.iter().map(|i| format!("({i})")).collect();
        format!("SELECT * FROM VALUES {} AS t(id)", rows.join(", "))
    };
    one(&s, &root, &ints(&[1, 2]), "parquet", "a", "a1.parquet");
    one(&s, &root, &ints(&[3]), "parquet", "a", "a2.parquet");
    one(&s, &root, &ints(&[1, 10]), "parquet", "b", "b1.parquet");
    for v in ["a", "b"] {
        let path = root.join(v).to_str().unwrap().to_owned();
        block_on(s.read().parquet([path]).create_or_replace_temp_view(v)).unwrap();
    }
    (dir, root, s)
}

/// The rows of `sql` (sorted) and whether each output column is nullable.
fn run_sql(s: &Session, sql: &str) -> Result<(Vec<Vec<String>>, Vec<bool>), KrishivError> {
    block_on(async {
        let df = s.sql(sql).await?;
        let schema = df.schema().await?;
        let rows = rows_of(&df.collect().await?);
        Ok((
            rows,
            schema.fields().iter().map(|f| f.is_nullable()).collect(),
        ))
    })
}

/// The root directory as SQL text and in the engine's `file://` URLs: the path of its `file:` URL
/// (`/tmp/x`; `/C:/x` on Windows, where a canonical `\\?\C:\x` would not survive a SQL literal).
fn root_text(root: &Path) -> String {
    let url = krishiv_exec::files::file_url(root);
    url.strip_prefix("file://").unwrap_or(&url).to_owned()
}

fn files(root: &Path, names: &[&str]) -> Vec<Vec<String>> {
    let mut v: Vec<Vec<String>> = names
        .iter()
        .map(|n| vec![format!("file://{}/{n}", root_text(root))])
        .collect();
    v.sort();
    v
}

fn empty(n: usize) -> Vec<Vec<String>> {
    vec![vec![String::new()]; n]
}

/// The three functions are never NULL: a join passes the file on only from a side it keeps whole. Spark pads the
/// broadcast build side (or plans a sort-merge join), so the padded side's rows get the no-file values there too.
#[test]
fn outer_joins_take_the_file_only_from_a_side_kept_whole() {
    let (_dir, root, s) = small();
    let a = ["a/a1.parquet", "a/a1.parquet", "a/a2.parquet"];
    let cases = [
        // The kept side has no file: `''` on every row, padded or not.
        (
            "SELECT input_file_name() FROM range(5) r LEFT JOIN a ON r.id = a.id",
            empty(5),
        ),
        (
            "SELECT input_file_name() FROM a RIGHT JOIN range(5) r ON a.id = r.id",
            empty(5),
        ),
        (
            "SELECT input_file_name() FROM a FULL JOIN range(5) r ON a.id = r.id",
            empty(5),
        ),
        // The kept side holds the file.
        (
            "SELECT input_file_name() FROM a LEFT JOIN range(5) r ON a.id = r.id",
            files(&root, &a),
        ),
        (
            "SELECT input_file_name() FROM range(5) r RIGHT JOIN a ON r.id = a.id",
            files(&root, &a),
        ),
        (
            "SELECT input_file_name() FROM range(5) r JOIN a ON r.id = a.id",
            files(&root, &a),
        ),
    ];
    for (sql, want) in cases {
        let (rows, nullable) = run_sql(&s, sql).unwrap();
        assert_eq!(rows, want, "{sql}");
        assert_eq!(nullable, [false], "{sql}");
    }
    let (rows, nullable) = run_sql(
        &s,
        "SELECT input_file_block_start(), input_file_block_length() FROM a FULL JOIN range(5) r ON a.id = r.id",
    )
    .unwrap();
    assert_eq!(rows, vec![vec!["-1".to_owned(), "-1".to_owned()]; 5]);
    assert_eq!(nullable, [false, false]);
}

/// Spark checks the plan after inlining: a CTE reference counts its definition's sources, and a SQL function
/// whose body calls `input_file_name()` holds that call where it is called.
#[test]
fn ctes_and_sql_functions_count_as_inlined() {
    let (_dir, root, s) = small();
    let class = |sql: &str| {
        run_sql(&s, sql)
            .err()
            .map(|e| e.error_class().name().to_owned())
    };
    let multi = Some("MULTI_SOURCES_UNSUPPORTED_FOR_EXPRESSION".to_owned());
    assert_eq!(
        class("WITH x AS (SELECT * FROM a) SELECT input_file_name() FROM x JOIN a USING (id)"),
        multi
    );
    assert_eq!(
        class(
            "WITH x AS (SELECT * FROM a), y AS (SELECT * FROM x) SELECT input_file_name() FROM y JOIN b USING (id)"
        ),
        multi
    );
    // A CTE over a range is no source.
    assert_eq!(
        class(
            "WITH x AS (SELECT * FROM range(3)) SELECT input_file_name() FROM x JOIN a USING (id)"
        ),
        None
    );
    block_on(async {
        s.sql("CREATE TEMPORARY FUNCTION f() RETURNS STRING RETURN input_file_name()")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(class("SELECT f() FROM a JOIN b USING (id)"), multi);
    let (rows, _) = run_sql(&s, "SELECT f() FROM a").unwrap();
    assert_eq!(
        rows,
        files(&root, &["a/a1.parquet", "a/a1.parquet", "a/a2.parquet"])
    );
    let (rows, _) = run_sql(
        &s,
        "WITH x AS (SELECT * FROM a) SELECT input_file_name() FROM x",
    )
    .unwrap();
    assert_eq!(
        rows,
        files(&root, &["a/a1.parquet", "a/a1.parquet", "a/a2.parquet"])
    );
    // A definition that calls the function itself: bound, it is deterministic and inlined (Spark inlines it).
    let sql =
        "WITH x AS (SELECT input_file_name() f, id FROM a) SELECT input_file_name(), f FROM x";
    let (rows, _) = run_sql(&s, sql).unwrap();
    let pairs: Vec<Vec<String>> = files(&root, &["a/a1.parquet", "a/a1.parquet", "a/a2.parquet"])
        .into_iter()
        .map(|r| vec![r[0].clone(), r[0].clone()])
        .collect();
    assert_eq!(rows, pairs);
    // Two references: each reads its own instance of the definition.
    let sql = "WITH x AS (SELECT * FROM a) SELECT input_file_name() FROM x UNION ALL SELECT input_file_name() FROM x";
    let (rows, _) = run_sql(&s, sql).unwrap();
    let a = ["a/a1.parquet", "a/a1.parquet", "a/a2.parquet"];
    assert_eq!(rows, files(&root, &[a, a].concat()));
}

/// Recorded divergence: a data column named `_metadata` hides the hidden column, so a call over the scan gives the
/// no-file value (Spark's value does not depend on `_metadata`: it is the file).
#[test]
fn a_metadata_data_column_gives_the_no_file_value() {
    let (_dir, root, s) = small();
    one(
        &s,
        &root,
        "SELECT 1 AS id, 2 AS _metadata",
        "parquet",
        "m",
        "m1.parquet",
    );
    let sql = format!(
        "SELECT input_file_name(), _metadata FROM parquet.`{}`",
        root.join("m").display()
    );
    let (rows, _) = run_sql(&s, &sql).unwrap();
    assert_eq!(rows, vec![vec![String::new(), "2".to_owned()]]);
}

/// The check counts a CTE reference's definition once per scope: 30 chained CTEs that each read the previous one
/// twice analyse in milliseconds (counted per reference, 2^30 counts). Analysis only: the physical planner's first
/// pass still plans each deterministic reference on its own (a separate follow-up).
#[test]
fn chained_ctes_are_counted_once_per_scope() {
    let (_dir, _root, s) = small();
    let mut sql = String::from("WITH c0 AS (SELECT id FROM a)");
    for i in 1..=30 {
        let p = i - 1;
        sql.push_str(&format!(
            ", c{i} AS (SELECT l.id FROM c{p} l JOIN c{p} r ON l.id = r.id)"
        ));
    }
    let start = std::time::Instant::now();
    let class = block_on(async {
        s.sql(&format!("{sql} SELECT input_file_name() FROM c30"))
            .await?
            .schema()
            .await
    })
    .err()
    .map(|e| e.error_class().name().to_owned());
    // c1 joins two reads of `a`: two sources under the call.
    assert_eq!(
        class.as_deref(),
        Some("MULTI_SOURCES_UNSUPPORTED_FOR_EXPRESSION")
    );
    let elapsed = start.elapsed();
    // Linear is milliseconds; the old count was hours. The bound leaves room for coverage builds.
    assert!(elapsed < std::time::Duration::from_secs(5), "{elapsed:?}");
}
