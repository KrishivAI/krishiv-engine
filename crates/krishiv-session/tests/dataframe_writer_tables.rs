// SPDX-License-Identifier: Apache-2.0
//! `DataFrameWriter.saveAsTable`, `insertInto`, `text`, `bucketBy` and `sortBy` (P2-T189),
//! against the behaviour of Spark 4.0's `DataFrameWriter` source.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use krishiv_session::functions::{col, lit};

fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int32, false),
        Field::new("name", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![3, 1, 2])),
            Arc::new(StringArray::from(vec![Some("c"), None, Some("b")])),
        ],
    )
    .unwrap();
    s.register_batches("people", schema, vec![batch]).unwrap();
    s
}

fn class(result: Result<(), KrishivError>) -> String {
    result.unwrap_err().error_class().name().to_owned()
}

fn count(s: &Session, table: &str) -> usize {
    block_on(s.table(table).count())
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn save_as_table_creates_a_table_the_session_reads() {
    let s = session();
    block_on(s.table("people").write().save_as_table("saved")).unwrap();
    assert_eq!(count(&s, "saved"), 3);
    let df = block_on(s.sql("SELECT id FROM saved ORDER BY id")).unwrap();
    let rows = block_on(df.collect()).unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
}

#[test]
fn save_as_table_modes() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    // errorifexists (the default) on an existing table.
    assert_eq!(
        class(block_on(s.table("people").write().save_as_table("t"))),
        "TABLE_OR_VIEW_ALREADY_EXISTS"
    );
    // ignore leaves it.
    block_on(s.table("people").write().mode("ignore").save_as_table("t")).unwrap();
    assert_eq!(count(&s, "t"), 3);
    // append adds, by column name: the DataFrame's columns in another order.
    let swapped = s.table("people").select(["name", "id"]);
    block_on(swapped.write().mode("append").save_as_table("t")).unwrap();
    assert_eq!(count(&s, "t"), 6);
    let df = block_on(s.sql("SELECT sum(id) FROM t")).unwrap();
    let sums = block_on(df.collect()).unwrap();
    assert_eq!(sums[0].column(0).as_primitive::<Int64Type>().value(0), 12);
    // overwrite drops the table and creates it again from the query.
    block_on(
        s.table("people")
            .filter(col("id").gt(lit(1)))
            .write()
            .mode("overwrite")
            .save_as_table("t"),
    )
    .unwrap();
    assert_eq!(count(&s, "t"), 2);
    // overwrite of a table that does not exist creates it.
    block_on(
        s.table("people")
            .write()
            .mode("overwrite")
            .save_as_table("made"),
    )
    .unwrap();
    assert_eq!(count(&s, "made"), 3);
    // append to a table that does not exist creates it.
    block_on(
        s.table("people")
            .write()
            .mode("append")
            .save_as_table("fresh"),
    )
    .unwrap();
    assert_eq!(count(&s, "fresh"), 3);
    // another mode name.
    assert_eq!(
        class(block_on(
            s.table("people").write().mode("nope").save_as_table("x")
        )),
        "INVALID_SAVE_MODE"
    );
}

#[test]
fn save_as_table_append_with_other_names_is_refused() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let renamed = s
        .table("people")
        .select_expr(["id AS a", "name AS b"])
        .unwrap();
    assert_eq!(
        class(block_on(renamed.write().mode("append").save_as_table("t"))),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn insert_into_appends_by_position_and_ignores_the_first_two_modes() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    block_on(s.table("people").write().insert_into("t")).unwrap();
    assert_eq!(count(&s, "t"), 6);
    block_on(s.table("people").write().mode("ignore").insert_into("t")).unwrap();
    block_on(s.table("people").write().mode("append").insert_into("t")).unwrap();
    assert_eq!(count(&s, "t"), 12);
    // overwrite is INSERT OVERWRITE, by position: the table holds just these rows.
    block_on(
        s.table("people")
            .filter(col("id").gt(lit(1)))
            .write()
            .mode("overwrite")
            .insert_into("t"),
    )
    .unwrap();
    assert_eq!(count(&s, "t"), 2);
    // reading the table it overwrites is refused.
    assert!(block_on(s.table("t").write().mode("overwrite").insert_into("t")).is_err());
    assert_eq!(count(&s, "t"), 2, "and leaves it alone");
    // a missing table.
    assert_eq!(
        class(block_on(s.table("people").write().insert_into("missing"))),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

#[test]
fn insert_into_arity_mismatch_and_partition_by() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let one = s.table("people").select(["id"]);
    assert!(block_on(one.write().insert_into("t")).is_err());
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .partition_by(["id"])
                .insert_into("t")
        )),
        "PARTITION_BY_NOT_ALLOWED_WITH_INSERT_INTO"
    );
}

#[test]
fn bucketing_is_refused_as_spark_refuses_it() {
    let s = session();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o").to_str().unwrap().to_owned();
    // `save` and `insertInto` do not bucket (Spark: _LEGACY_ERROR_TEMP_1312 / 1313).
    let w = s.table("people").write().bucket_by(2, ["id"]);
    assert_eq!(
        class(block_on(w.clone().save(&path))),
        "_LEGACY_ERROR_TEMP_1312"
    );
    assert_eq!(
        class(block_on(w.insert_into("t"))),
        "_LEGACY_ERROR_TEMP_1312"
    );
    let w = s
        .table("people")
        .write()
        .bucket_by(2, ["id"])
        .sort_by(["name"]);
    assert_eq!(class(block_on(w.save(&path))), "_LEGACY_ERROR_TEMP_1313");
    // sortBy alone.
    let w = s.table("people").write().sort_by(["name"]);
    assert_eq!(
        class(block_on(w.clone().save(&path))),
        "SORT_BY_WITHOUT_BUCKETING"
    );
    assert_eq!(
        class(block_on(w.save_as_table("t"))),
        "SORT_BY_WITHOUT_BUCKETING"
    );
    // saveAsTable would bucket: the engine has no bucketed tables.
    let w = s.table("people").write().bucket_by(2, ["id"]);
    assert_eq!(
        class(block_on(w.save_as_table("t"))),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn text_is_unsupported_until_the_engine_writes_it() {
    let s = session();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("txt").to_str().unwrap().to_owned();
    let one = s.table("people").select(["name"]);
    assert_eq!(
        class(block_on(one.write().text(&path))),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn append_by_name_is_exact_about_names_and_format() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    // a column name with a dot is one column, not a nested reference.
    let dotted = s
        .table("people")
        .select_expr(["id AS `a.b`", "name"])
        .unwrap();
    block_on(dotted.write().save_as_table("d")).unwrap();
    block_on(dotted.write().mode("append").save_as_table("d")).unwrap();
    assert_eq!(count(&s, "d"), 6);
    // a different format than the table's, and a partitioning, are refused.
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .format("csv")
                .mode("append")
                .save_as_table("t")
        )),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .partition_by(["id"])
                .mode("append")
                .save_as_table("t")
        )),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    // overwrite with a partitioning on a new table fails and creates nothing.
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .partition_by(["id"])
                .mode("overwrite")
                .save_as_table("p")
        )),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert!(block_on(s.table("p").count()).is_err());
}

#[test]
fn overwrite_checks_before_it_drops() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    // The query reads the table it replaces.
    assert_eq!(
        class(block_on(
            s.table("t").write().mode("overwrite").save_as_table("t")
        )),
        "UNSUPPORTED_OVERWRITE.TABLE"
    );
    // So does one over a view of it.
    block_on(s.sql("CREATE TEMPORARY VIEW tv AS SELECT * FROM t")).unwrap();
    assert_eq!(
        class(block_on(
            s.table("tv").write().mode("overwrite").save_as_table("t")
        )),
        "UNSUPPORTED_OVERWRITE.TABLE"
    );
    // A new table that cannot be created (a partitioning) leaves the old one in place.
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .partition_by(["id"])
                .mode("overwrite")
                .save_as_table("t")
        )),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    assert_eq!(count(&s, "t"), 3, "the table is still there");
    // A temporary view of the same name is not dropped in the table's place.
    block_on(s.sql("CREATE TEMPORARY VIEW t AS SELECT 1 AS id, 'x' AS name")).unwrap();
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .mode("overwrite")
                .save_as_table("t")
        )),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}
