// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.files.ignoreCorruptFiles` over Parquet files whose footer
//! cannot be read (RS-26): a directory of one good file and one cut short,
//! in both name orders, and of a bad file alone, with and without
//! `spark.sql.parquet.mergeSchema`, and the per-read option against the
//! session conf. Expected answers: a Spark 4.2.0 run
//! (`spark-shell`, `local[1]`) over the same layout, a 1,000-row file of
//! `id BIGINT, s STRING` written by Spark, cut to half its length or by its
//! last 3 bytes (`tests/scripts/ignore_corrupt_parquet.scala`).
#![cfg(test)]
mod literal_path;
use literal_path::sql_path;

use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::Session;

/// The good file's bytes, written by Krishiv: 1,000 rows of `id`, `s`.
fn good_file(s: &Session, dir: &Path) -> Vec<u8> {
    let out = dir.join("src");
    let df =
        block_on(s.sql("SELECT id, CAST(id AS STRING) AS s FROM range(0, 1000, 1, 1)")).unwrap();
    block_on(df.write().parquet(out.to_str().unwrap())).unwrap();
    let file = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .unwrap();
    std::fs::read(file).unwrap()
}

/// Directory `name` holding `files` (name, bytes).
fn layout(root: &Path, name: &str, files: &[(&str, &[u8])]) -> PathBuf {
    let d = root.join(name);
    std::fs::create_dir_all(&d).unwrap();
    for (f, b) in files {
        std::fs::write(d.join(f), b).unwrap();
    }
    d
}

/// `count(*)` and `sum(id)` over `parquet.<dir>`, or the error class.
fn query(s: &Session, dir: &Path) -> Result<(i64, i64), String> {
    let sql = format!("SELECT count(*), sum(id) FROM parquet.`{}`", dir.display());
    match block_on(async { s.sql(&sql).await?.collect().await }) {
        Ok(b) => {
            let b = &b[0];
            Ok((
                b.column(0).as_primitive::<Int64Type>().value(0),
                b.column(1).as_primitive::<Int64Type>().value(0),
            ))
        }
        Err(e) => Err(e.error_class().name().to_owned()),
    }
}

#[test]
fn a_corrupt_footer_is_skipped_as_spark_skips_it() {
    let s = Session::new().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let good = good_file(&s, tmp.path());
    let half = &good[..good.len() / 2];
    let tail_cut = &good[..good.len() - 3];
    let dirs = [
        (
            "good_bad",
            layout(
                tmp.path(),
                "good_bad",
                &[("a.parquet", &good), ("b.parquet", half)],
            ),
        ),
        (
            "bad_good",
            layout(
                tmp.path(),
                "bad_good",
                &[("a.parquet", half), ("b.parquet", &good)],
            ),
        ),
        (
            "bad_only",
            layout(tmp.path(), "bad_only", &[("a.parquet", half)]),
        ),
        (
            "good_tailcut",
            layout(
                tmp.path(),
                "good_tailcut",
                &[("a.parquet", &good), ("b.parquet", tail_cut)],
            ),
        ),
        (
            "tailcut_good",
            layout(
                tmp.path(),
                "tailcut_good",
                &[("a.parquet", tail_cut), ("b.parquet", &good)],
            ),
        ),
    ];
    let ok = Ok((1000, 499_500));
    let infer = Err("UNABLE_TO_INFER_SCHEMA".to_owned());
    // Spark 4.2.0, `ignoreCorruptFiles=true`: without `mergeSchema` the
    // schema comes from the first file alone, so a corrupt first file is
    // skipped and leaves nothing to infer from; with it every footer is read
    // and the corrupt ones are skipped.
    let want = |dir: &str, merge: bool| match (dir, merge) {
        ("bad_only", _) | ("bad_good" | "tailcut_good", false) => infer.clone(),
        _ => ok.clone(),
    };
    let mut failures = vec![];
    for merge in [false, true] {
        block_on(s.sql("SET spark.sql.files.ignoreCorruptFiles = true")).unwrap();
        block_on(s.sql(&format!("SET spark.sql.parquet.mergeSchema = {merge}"))).unwrap();
        for (name, dir) in &dirs {
            let got = query(&s, dir);
            let want = want(name, merge);
            if got != want {
                failures.push(format!(
                    "{name} mergeSchema={merge}: want {want:?}, got {got:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `count(*)` and `sum(id)` of `dir` read by the DataFrame reader (Spark's
/// `spark.read.parquet`), or the error class.
fn read(s: &Session, dir: &Path) -> Result<(i64, i64), String> {
    let df = s.read().parquet([dir.to_str().unwrap().to_owned()]);
    match block_on(async {
        df.create_or_replace_temp_view("t").await?;
        s.sql("SELECT count(*), sum(id) FROM t")
            .await?
            .collect()
            .await
    }) {
        Ok(b) => Ok((
            b[0].column(0).as_primitive::<Int64Type>().value(0),
            b[0].column(1).as_primitive::<Int64Type>().value(0),
        )),
        // The class, then its cause's (Spark prints the cause too).
        Err(e) => Err(match e.cause() {
            Some(c) => format!("{} <- {}", e.error_class().name(), c.error_class().name()),
            None => e.error_class().name().to_owned(),
        }),
    }
}

/// With the key off (Spark 4.2.0, `spark.read.parquet`): a corrupt footer
/// that schema inference reads is `CANNOT_READ_FILE_FOOTER`; one only the
/// scan meets (the second file, without `mergeSchema`) is
/// `FAILED_READ_FILE.NO_HINT`. Spark's SQL `parquet.<dir>` wraps the
/// inference failures in `FAILED_TO_CREATE_PLAN_FOR_DIRECT_QUERY`, which
/// Krishiv does not (a known difference).
#[test]
fn without_the_key_a_corrupt_footer_fails_as_in_spark() {
    let s = Session::new().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let good = good_file(&s, tmp.path());
    let half = &good[..good.len() / 2];
    let good_bad = layout(
        tmp.path(),
        "good_bad",
        &[("a.parquet", &good), ("b.parquet", half)],
    );
    let bad_good = layout(
        tmp.path(),
        "bad_good",
        &[("a.parquet", half), ("b.parquet", &good)],
    );
    let footer = Err("FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER".to_owned());
    // The scan's error carries the footer error as its cause.
    let no_hint =
        Err("FAILED_READ_FILE.NO_HINT <- FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER".to_owned());
    for merge in [false, true] {
        block_on(s.sql(&format!("SET spark.sql.parquet.mergeSchema = {merge}"))).unwrap();
        let want = if merge {
            footer.clone()
        } else {
            no_hint.clone()
        };
        assert_eq!(read(&s, &good_bad), want, "good_bad mergeSchema={merge}");
        assert_eq!(read(&s, &bad_good), footer, "bad_good mergeSchema={merge}");
    }
}

/// `count(*)` and `sum(id)` of `dir` read with the per-read
/// `ignoreCorruptFiles` option `opt`, or the error class.
fn read_with(s: &Session, dir: &Path, opt: bool) -> Result<(i64, i64), String> {
    let df = s
        .read()
        .option("ignoreCorruptFiles", if opt { "true" } else { "false" })
        .parquet([dir.to_str().unwrap().to_owned()]);
    match block_on(async {
        df.create_or_replace_temp_view("t").await?;
        s.sql("SELECT count(*), sum(id) FROM t")
            .await?
            .collect()
            .await
    }) {
        Ok(b) => Ok((
            b[0].column(0).as_primitive::<Int64Type>().value(0),
            b[0].column(1).as_primitive::<Int64Type>().value(0),
        )),
        Err(e) => Err(e.error_class().name().to_owned()),
    }
}

/// The per-read `ignoreCorruptFiles` option takes precedence over
/// `spark.sql.files.ignoreCorruptFiles`, in schema inference and in the scan
/// (Spark 4.2.0 `FileSourceOptions`; the "option" rows of
/// `tests/scripts/ignore_corrupt_parquet.scala`, identical for the conf on
/// and off).
#[test]
fn the_read_option_takes_precedence_over_the_session_conf() {
    let s = Session::new().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let good = good_file(&s, tmp.path());
    let half = &good[..good.len() / 2];
    let good_bad = layout(
        tmp.path(),
        "good_bad",
        &[("a.parquet", &good), ("b.parquet", half)],
    );
    let bad_good = layout(
        tmp.path(),
        "bad_good",
        &[("a.parquet", half), ("b.parquet", &good)],
    );
    let ok = Ok((1000, 499_500));
    let footer = Err("FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER".to_owned());
    let no_hint = Err("FAILED_READ_FILE.NO_HINT".to_owned());
    let infer = Err("UNABLE_TO_INFER_SCHEMA".to_owned());
    let mut failures = vec![];
    for conf in [false, true] {
        block_on(s.sql(&format!("SET spark.sql.files.ignoreCorruptFiles = {conf}"))).unwrap();
        for merge in [false, true] {
            block_on(s.sql(&format!("SET spark.sql.parquet.mergeSchema = {merge}"))).unwrap();
            for opt in [false, true] {
                let cases = [
                    (
                        "good_bad",
                        &good_bad,
                        match (opt, merge) {
                            (true, _) => &ok,
                            (false, false) => &no_hint,
                            (false, true) => &footer,
                        },
                    ),
                    (
                        "bad_good",
                        &bad_good,
                        match (opt, merge) {
                            (true, true) => &ok,
                            (true, false) => &infer,
                            (false, _) => &footer,
                        },
                    ),
                ];
                for (name, dir, want) in cases {
                    let got = read_with(&s, dir, opt);
                    if &got != want {
                        failures.push(format!(
                            "{name} conf={conf} option={opt} mergeSchema={merge}: want {want:?}, got {got:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Only Parquet without `mergeSchema` infers from the first file alone, so
/// only there does a skipped inference error become `UNABLE_TO_INFER_SCHEMA`.
/// CSV infers from every file: Spark 4.2.0 keeps the lines its codec decoded
/// before the cut, in inference and in the scan (1,474 rows summing to
/// 611,601 over the fixtures `tests/fixtures/spark-4.2.0/ignore-corrupt-csv/`,
/// its own gzip CSV of 1,000 rows and that file cut to half its length), with
/// and without `inferSchema`, in either file order. Without the key the read
/// fails with `FAILED_READ_FILE.NO_HINT`.
#[test]
fn a_broken_gzip_csv_keeps_the_lines_decoded_before_the_cut() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/ignore-corrupt-csv");
    let good = std::fs::read(fixtures.join("good.csv.gz")).unwrap();
    let cut = std::fs::read(fixtures.join("truncated.csv.gz")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    for (name, files) in [
        ("good_bad", [("a.csv.gz", &good), ("b.csv.gz", &cut)]),
        ("bad_good", [("a.csv.gz", &cut), ("b.csv.gz", &good)]),
    ] {
        let files: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (*n, b.as_slice())).collect();
        let dir = layout(tmp.path(), name, &files);
        for ignore in [true, false] {
            block_on(s.sql(&format!(
                "SET spark.sql.files.ignoreCorruptFiles = {ignore}"
            )))
            .unwrap();
            for infer in ["true", "false"] {
                let df = s
                    .read()
                    .format("csv")
                    .option("header", "true")
                    .option("inferSchema", infer)
                    .load([dir.to_str().unwrap()]);
                let got = block_on(async {
                    df.create_or_replace_temp_view("c").await?;
                    s.sql("SELECT count(*), CAST(sum(id) AS BIGINT) FROM c")
                        .await?
                        .collect()
                        .await
                });
                let case = format!("{name} ignore={ignore} inferSchema={infer}");
                match got {
                    Ok(b) if ignore => {
                        let count = b[0].column(0).as_primitive::<Int64Type>().value(0);
                        let sum = b[0].column(1).as_primitive::<Int64Type>().value(0);
                        assert_eq!((count, sum), (1474, 611_601), "{case}");
                    }
                    Ok(_) => panic!("{case}: read without the key"),
                    Err(e) => {
                        assert!(!ignore, "{case}: {e}");
                        assert_eq!(e.error_class().name(), "FAILED_READ_FILE.NO_HINT", "{case}");
                    }
                }
            }
        }
    }
}

/// The read's own `ignoreCorruptFiles` option wins over the SQL conf (Spark's
/// `FileSourceOptions`), in inference and in the scan.
#[test]
fn the_read_option_ignore_corrupt_files_wins_over_the_conf() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/ignore-corrupt-csv");
    let good = std::fs::read(fixtures.join("good.csv.gz")).unwrap();
    let cut = std::fs::read(fixtures.join("truncated.csv.gz")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let dir = layout(
        tmp.path(),
        "opt",
        &[("a.csv.gz", good.as_slice()), ("b.csv.gz", cut.as_slice())],
    );
    let s = Session::new().unwrap();
    for (conf, option, rows) in [("false", "true", Some(1474)), ("true", "false", None)] {
        block_on(s.sql(&format!("SET spark.sql.files.ignoreCorruptFiles = {conf}"))).unwrap();
        let df = s
            .read()
            .format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .option("ignoreCorruptFiles", option)
            .load([dir.to_str().unwrap()]);
        let got = block_on(async {
            df.create_or_replace_temp_view("o").await?;
            s.sql("SELECT count(*) FROM o").await?.collect().await
        });
        let got = got
            .map(|b| b[0].column(0).as_primitive::<Int64Type>().value(0))
            .map_err(|e| e.error_class().name().to_owned());
        match rows {
            Some(n) => assert_eq!(got, Ok(n), "conf={conf} option={option}"),
            None => assert_eq!(
                got,
                Err("FAILED_READ_FILE.NO_HINT".to_owned()),
                "conf={conf} option={option}"
            ),
        }
    }
}

/// A partitioned directory and a table with a declared schema (no inference:
/// the table's footer reads are the first) skip the corrupt file the same
/// way, and without the key fail as Spark's scan does,
/// `FAILED_READ_FILE.NO_HINT` (Spark 4.2.0, the same layouts and table).
#[test]
fn partitions_and_declared_tables_skip_a_corrupt_footer() {
    let s = Session::new().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let good = good_file(&s, tmp.path());
    let half = &good[..good.len() / 2];
    let parts = tmp.path().join("parts");
    layout(&parts, "p=1", &[("a.parquet", &good)]);
    layout(&parts, "p=2", &[("b.parquet", half)]);
    let flat = layout(
        tmp.path(),
        "flat",
        &[("a.parquet", half), ("b.parquet", &good)],
    );
    block_on(s.sql(&format!(
        "CREATE TABLE declared (id BIGINT, s STRING) USING parquet LOCATION '{}'",
        sql_path(&flat)
    )))
    .unwrap();
    let run = |sql: &str| match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(b) => Ok((
            b[0].column(0).as_primitive::<Int64Type>().value(0),
            b[0].column(1).as_primitive::<Int64Type>().value(0),
        )),
        Err(e) => Err(e.error_class().name().to_owned()),
    };
    let partitioned = format!(
        "SELECT count(*), sum(id) FROM parquet.`{}` WHERE p IN (1, 2)",
        parts.display()
    );
    let declared = "SELECT count(*), sum(id) FROM declared";
    block_on(s.sql("SET spark.sql.files.ignoreCorruptFiles = true")).unwrap();
    assert_eq!(run(&partitioned), Ok((1000, 499_500)));
    assert_eq!(run(declared), Ok((1000, 499_500)));
    block_on(s.sql("SET spark.sql.files.ignoreCorruptFiles = false")).unwrap();
    let no_hint = Err("FAILED_READ_FILE.NO_HINT".to_owned());
    assert_eq!(run(&partitioned), no_hint);
    assert_eq!(run(declared), no_hint);
}
