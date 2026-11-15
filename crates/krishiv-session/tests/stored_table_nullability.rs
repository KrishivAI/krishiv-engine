// SPDX-License-Identifier: Apache-2.0
//! A permanent file-source table is nullable at every level, nested elements, map values and struct fields too,
//! whatever its DDL or the query that created it said; a permanent view keeps its query's types. The expected
//! schemas are Spark 4.2.0's (`spark.table(t).schema` after `CREATE TABLE`, CTAS, `saveAsTable`, `ALTER TABLE ADD
//! COLUMNS` and `CREATE VIEW`).
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_schema::{DataType as ArrowType, Field};
use futures::executor::block_on;
use krishiv_session::Session;

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn collect(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// A type as Spark's DDL writes it, `NOT NULL` where a level cannot hold a NULL (map keys never can).
fn ddl(dt: &ArrowType) -> String {
    let inner = |f: &Field| {
        let t = ddl(f.data_type());
        if f.is_nullable() {
            t
        } else {
            format!("{t} NOT NULL")
        }
    };
    match dt {
        ArrowType::Int32 => "int".into(),
        ArrowType::Utf8 | ArrowType::Utf8View | ArrowType::LargeUtf8 => "string".into(),
        ArrowType::List(e) | ArrowType::LargeList(e) | ArrowType::ListView(e) => {
            format!("array<{}>", inner(e))
        }
        ArrowType::Map(entries, _) => match entries.data_type() {
            ArrowType::Struct(kv) => format!("map<{}, {}>", ddl(kv[0].data_type()), inner(&kv[1])),
            other => panic!("map entries {other}"),
        },
        ArrowType::Struct(fields) => format!(
            "struct<{}>",
            fields
                .iter()
                .map(|f| format!("{}: {}", f.name(), inner(f)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => panic!("unexpected {other}"),
    }
}

/// The columns of `table` as `name: type`, `NOT NULL` where Spark's schema says `nullable = false`.
fn schema(s: &Session, table: &str) -> Vec<String> {
    let sc = block_on(async { s.table(table).schema().await })
        .unwrap_or_else(|e| panic!("{table}: {e}"));
    sc.fields()
        .iter()
        .map(|f| {
            let t = ddl(f.data_type());
            let t = if f.is_nullable() {
                t
            } else {
                format!("{t} NOT NULL")
            };
            format!("{}: {t}", f.name())
        })
        .collect()
}

#[test]
fn ctas_and_save_as_table_store_every_level_nullable() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    collect(
        &s,
        "CREATE TABLE t1 USING parquet AS SELECT 1 AS i, array(1) a, map(1, 'x') m, named_struct('x', 1) s",
    );
    assert_eq!(
        schema(&s, "t1"),
        [
            "i: int",
            "a: array<int>",
            "m: map<int, string>",
            "s: struct<x: int>"
        ]
    );
    block_on(async {
        s.sql("SELECT array(1) a")
            .await?
            .write()
            .save_as_table("t4")
            .await
    })
    .unwrap();
    assert_eq!(schema(&s, "t4"), ["a: array<int>"]);
    // Appending a query whose element is never NULL leaves the table's types alone.
    block_on(async {
        s.sql("SELECT array(2) a")
            .await?
            .write()
            .mode("append")
            .save_as_table("t4")
            .await
    })
    .unwrap();
    assert_eq!(schema(&s, "t4"), ["a: array<int>"]);
    assert_eq!(
        collect(&s, "SELECT * FROM t4")
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    // The table holds the NULLs its type now allows.
    collect(
        &s,
        "INSERT INTO t1 SELECT 2, array(CAST(NULL AS INT)), map(2, CAST(NULL AS STRING)), named_struct('x', CAST(NULL AS INT))",
    );
    let b = collect(
        &s,
        "SELECT count_if(a[0] IS NULL), count_if(m[2] IS NULL AND i = 2), count_if(s.x IS NULL) FROM t1",
    );
    for c in 0..3 {
        let v = arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
            b[0].column(c),
        );
        assert_eq!(v.value(0), 1, "column {c}");
    }
}

#[test]
fn ddl_not_null_is_dropped_at_every_level() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    collect(
        &s,
        "CREATE TABLE t2 (i INT NOT NULL, s STRUCT<x: INT NOT NULL>, a ARRAY<INT>) USING parquet",
    );
    collect(
        &s,
        "INSERT INTO t2 SELECT 1, named_struct('x', 2), array(3)",
    );
    assert_eq!(
        schema(&s, "t2"),
        ["i: int", "s: struct<x: int>", "a: array<int>"]
    );
    collect(&s, "ALTER TABLE t2 ADD COLUMNS (n STRUCT<y: INT NOT NULL>)");
    assert_eq!(
        schema(&s, "t2"),
        [
            "i: int",
            "s: struct<x: int>",
            "a: array<int>",
            "n: struct<y: int>"
        ]
    );
    collect(&s, "ALTER TABLE t2 RENAME TO t5");
    assert_eq!(
        schema(&s, "t5"),
        [
            "i: int",
            "s: struct<x: int>",
            "a: array<int>",
            "n: struct<y: int>"
        ]
    );
}

#[test]
fn a_permanent_view_keeps_its_query_types() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    collect(
        &s,
        "CREATE VIEW v AS SELECT array(1) a, named_struct('x', 1) s",
    );
    assert_eq!(
        schema(&s, "v"),
        [
            "a: array<int NOT NULL> NOT NULL",
            "s: struct<x: int NOT NULL> NOT NULL"
        ]
    );
}
