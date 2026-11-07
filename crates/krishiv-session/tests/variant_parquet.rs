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
    "shredded-types",
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

/// `schema_of_variant` of every row equals Spark's.
#[test]
fn schemas_of_variants_are_spark_s() {
    let s = session();
    for name in FILES {
        let path = fixtures().join(format!("{name}.parquet"));
        let df = s.read().parquet([path.to_str().unwrap().to_owned()]);
        block_on(df.create_or_replace_temp_view("t")).unwrap();
        let out = block_on(async {
            s.sql("SELECT schema_of_variant(v) FROM t ORDER BY id")
                .await?
                .collect()
                .await
        })
        .unwrap();
        let got: Vec<Option<String>> = out
            .iter()
            .flat_map(|b| {
                let a = b.column(0).as_string_view();
                (0..b.num_rows())
                    .map(|i| (!a.is_null(i)).then(|| a.value(i).to_owned()))
                    .collect::<Vec<_>>()
            })
            .collect();
        let text = std::fs::read_to_string(fixtures().join(format!("{name}.json"))).unwrap();
        let rows: Vec<Value> = serde_json::from_str(&text).unwrap();
        let want: Vec<Option<String>> = rows
            .iter()
            .map(|r| r["schema"].as_str().map(str::to_owned))
            .collect();
        assert_eq!(got, want, "{name}");
    }
}

/// `to_variant_object` of a struct, map and array (Spark 4.2.0's examples).
#[test]
fn to_variant_object_builds_objects() {
    let s = session();
    let out = block_on(async {
        s.sql(
            "SELECT to_json(to_variant_object(named_struct('a', 1, 'b', 2))), \
             to_json(to_variant_object(array(1, 2, 3))), \
             to_json(to_variant_object(array(named_struct('a', 1)))), \
             to_json(to_variant_object(array(map('a', 2))))",
        )
        .await?
        .collect()
        .await
    })
    .unwrap();
    let got: Vec<&str> = (0..4)
        .map(|c| out[0].column(c).as_string_view().value(0))
        .collect();
    assert_eq!(
        got,
        [
            r#"{"a":1,"b":2}"#,
            "[1,2,3]",
            r#"[{"a":1}]"#,
            r#"[{"a":2}]"#
        ]
    );
    let e = block_on(async { s.sql("SELECT to_variant_object(1)").await?.collect().await })
        .unwrap_err();
    assert!(e.to_string().contains("CAST_WITHOUT_SUGGESTION"), "{e}");
}

/// Variants nested in a struct, an array and a map value (found through the
/// footer's Spark schema) read as Spark reads them.
#[test]
fn nested_variant_columns_read_as_spark_reads_them() {
    let s = session();
    let path = fixtures().join("nested-columns.parquet");
    let df = s.read().parquet([path.to_str().unwrap().to_owned()]);
    block_on(df.create_or_replace_temp_view("t")).unwrap();
    let out = block_on(async {
        s.sql("SELECT to_json(s.x), to_json(a[0]), to_json(m['k']) FROM t ORDER BY id")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let mut got = vec![];
    for b in &out {
        for i in 0..b.num_rows() {
            let text = |c: usize| {
                let a = b.column(c);
                (!a.is_null(i)).then(|| a.as_string_view().value(i).to_owned())
            };
            got.push((text(0), text(1), text(2)));
        }
    }
    let text = std::fs::read_to_string(fixtures().join("nested-columns.json")).unwrap();
    let rows: Vec<Value> = serde_json::from_str(&text).unwrap();
    let s = |v: &Value| v.as_str().map(str::to_owned);
    let want: Vec<_> = rows
        .iter()
        .map(|r| (s(&r["sx"]), s(&r["a0"]), s(&r["mk"])))
        .collect();
    assert_eq!(got, want);
}

/// `variant_explode(NULL)` gives no rows and the `_outer` form one NULL row
/// (Spark 4.2.0, oracle).
#[test]
fn variant_explode_of_null() {
    let s = session();
    let rows = |sql: &str| {
        let out = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
        out.iter().map(RecordBatch::num_rows).sum::<usize>()
    };
    assert_eq!(rows("SELECT * FROM variant_explode(NULL)"), 0);
    assert_eq!(rows("SELECT * FROM variant_explode_outer(NULL)"), 1);
    let out = block_on(async {
        s.sql("SELECT pos, key, value FROM variant_explode_outer(NULL)")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let b = out.iter().find(|b| b.num_rows() > 0).unwrap();
    assert!((0..3).all(|c| b.column(c).is_null(0)));
}
