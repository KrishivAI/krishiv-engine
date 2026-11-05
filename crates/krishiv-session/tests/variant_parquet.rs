// SPDX-License-Identifier: Apache-2.0
//! `VARIANT` columns in Parquet (P2-T001c): the files Spark 4.2.0 writes
//! (unshredded with either child order, unannotated with only the footer's
//! Spark schema saying `variant`, and shredded objects, nested objects and
//! arrays) read to the values Spark reads from them, and what Krishiv writes
//! reads back the same. Fixtures and Spark's answers:
//! `tests/fixtures/parquet/variant/`, written by
//! `tests/scripts/variant_parquet_fixture.scala`.
#![cfg(test)]

use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

const FILES: &[&str] = &[
    "unshredded",
    "unshredded-value-first",
    "unannotated",
    "shredded",
    "shredded-nested",
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/parquet/variant")
}

/// One row as Spark printed it: id, `to_json`, `$.a` as bigint, `$.b` as
/// string, `CAST(v AS STRING)`.
type Row = (
    i64,
    Option<String>,
    Option<i64>,
    Option<String>,
    Option<String>,
);

fn expected(name: &str) -> Vec<Row> {
    let text = std::fs::read_to_string(fixtures().join(format!("{name}.json"))).unwrap();
    let rows: Vec<Value> = serde_json::from_str(&text).unwrap();
    let s = |v: &Value| v.as_str().map(str::to_owned);
    rows.iter()
        .map(|r| {
            (
                r["id"].as_i64().unwrap(),
                s(&r["json"]),
                r["a"].as_i64(),
                s(&r["b"]),
                s(&r["s"]),
            )
        })
        .collect()
}

fn rows(batches: &[RecordBatch]) -> Vec<Row> {
    let mut out = vec![];
    for b in batches {
        let text = |c: usize, i: usize| {
            let a = b.column(c);
            (!a.is_null(i)).then(|| a.as_string_view().value(i).to_owned())
        };
        for i in 0..b.num_rows() {
            let a = b.column(2).as_primitive::<Int64Type>();
            out.push((
                b.column(0).as_primitive::<Int64Type>().value(i),
                text(1, i),
                (!a.is_null(i)).then(|| a.value(i)),
                text(3, i),
                text(4, i),
            ));
        }
    }
    out
}

const QUERY: &str = "SELECT id, to_json(v), try_variant_get(v, '$.a', 'bigint'), \
    try_variant_get(v, '$.b', 'string'), CAST(v AS STRING) FROM t ORDER BY id";

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    s
}

fn read(s: &Session, path: &Path) -> Vec<Row> {
    let df = s.read().parquet([path.to_str().unwrap().to_owned()]);
    block_on(df.create_or_replace_temp_view("t")).unwrap();
    let out = block_on(async { s.sql(QUERY).await?.collect().await }).unwrap();
    rows(&out)
}

#[test]
fn spark_written_files_read_as_spark_reads_them() {
    let s = session();
    for name in FILES {
        let got = read(&s, &fixtures().join(format!("{name}.parquet")));
        let want = expected(name);
        assert_eq!(got.len(), want.len(), "{name}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g, w, "{name}");
        }
    }
}

#[test]
fn written_variants_read_back_unchanged() {
    let s = session();
    let dir = tempfile::tempdir().unwrap();
    for name in ["unshredded", "shredded-nested"] {
        let df = s.read().parquet([fixtures()
            .join(format!("{name}.parquet"))
            .to_str()
            .unwrap()
            .to_owned()]);
        let out = dir.path().join(name);
        block_on(df.write().parquet(out.to_str().unwrap())).unwrap();
        assert_eq!(read(&s, &out), expected(name), "{name}");
    }
}

/// Spark 4.2.0 reads the file Krishiv writes from `shredded-nested.parquet`
/// (`krishiv-written.parquet`) to the values it reads from the original:
/// `krishiv-written.json` is Spark's answers for it
/// (`tests/scripts/variant_parquet_read.scala`). With
/// `KRISHIV_WRITE_FIXTURES=1` the file is written again first (then rerun
/// the Spark script).
#[test]
fn spark_reads_what_krishiv_writes() {
    let dir = fixtures();
    if std::env::var_os("KRISHIV_WRITE_FIXTURES").is_some() {
        let s = session();
        let df = s.read().parquet([dir
            .join("shredded-nested.parquet")
            .to_str()
            .unwrap()
            .to_owned()]);
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out");
        block_on(df.write().parquet(out.to_str().unwrap())).unwrap();
        let part = std::fs::read_dir(&out)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|x| x == "parquet"))
            .unwrap();
        std::fs::copy(part, dir.join("krishiv-written.parquet")).unwrap();
    }
    assert_eq!(expected("krishiv-written"), expected("shredded-nested"));
    // And Krishiv reads its own file the same way.
    assert_eq!(
        read(&session(), &dir.join("krishiv-written.parquet")),
        expected("shredded-nested")
    );
}
