// SPDX-License-Identifier: Apache-2.0
//! `DataFrameWriter.saveAsTable`, `insertInto`, `text`, `bucketBy` and `sortBy` (P2-T189),
//! against the behaviour of Spark 4.0's `DataFrameWriter` source.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
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
fn text_writes_one_string_column_and_orc_is_unsupported() {
    let s = session();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("txt").to_str().unwrap().to_owned();
    let one = s.table("people").select(["name"]);
    block_on(one.write().text(&path)).unwrap();
    let back = block_on(s.read().format("text").load([path.as_str()]).count()).unwrap();
    assert_eq!(back, 3);
    // The text source takes one column, as in Spark.
    let two = dir.path().join("two").to_str().unwrap().to_owned();
    assert!(block_on(s.table("people").write().text(&two)).is_err());
    let orc = dir.path().join("orc").to_str().unwrap().to_owned();
    assert_eq!(
        class(block_on(s.table("people").write().format("orc").save(&orc))),
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

#[test]
fn overwrite_never_drops_a_table_the_write_would_refuse_to_replace() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    // A format that cannot hold this table (the text source takes one column), duplicate output
    // columns and bad options fail before the drop.
    let refused = class(block_on(
        s.table("people")
            .write()
            .format("text")
            .mode("overwrite")
            .save_as_table("t"),
    ));
    assert!(
        ["KRISHIV_UNSUPPORTED_FEATURE", "_LEGACY_ERROR_TEMP_1290"].contains(&refused.as_str()),
        "{refused}"
    );
    let dup = s
        .table("people")
        .select_expr(["id AS a", "name AS A"])
        .unwrap();
    assert_eq!(
        class(block_on(dup.write().mode("overwrite").save_as_table("t"))),
        "COLUMN_ALREADY_EXISTS"
    );
    assert_eq!(count(&s, "t"), 3, "the table survived every refusal");
}

#[test]
fn a_table_of_another_namespace_with_the_same_name_is_not_the_target() {
    let s = session();
    block_on(s.sql("CREATE DATABASE other")).unwrap();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    block_on(s.table("people").write().save_as_table("other.t")).unwrap();
    // Reading default.t while replacing other.t is fine; reading the target is not.
    block_on(
        s.table("t")
            .filter(col("id").gt(lit(1)))
            .write()
            .mode("overwrite")
            .save_as_table("other.t"),
    )
    .unwrap();
    assert_eq!(count(&s, "other.t"), 2);
    assert_eq!(
        class(block_on(
            s.table("other.t")
                .write()
                .mode("overwrite")
                .save_as_table("other.t")
        )),
        "UNSUPPORTED_OVERWRITE.TABLE"
    );
}

fn describe_text(s: &Session, table: &str) -> String {
    let df = block_on(s.sql(&format!("DESCRIBE TABLE EXTENDED {table}"))).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let mut out = String::new();
    for b in &batches {
        for col in b.columns() {
            for i in 0..col.len() {
                if col.is_null(i) {
                    continue;
                }
                match col.data_type() {
                    ArrowType::Utf8View => out.push_str(col.as_string_view().value(i)),
                    ArrowType::LargeUtf8 => out.push_str(col.as_string::<i64>().value(i)),
                    ArrowType::Utf8 => out.push_str(col.as_string::<i32>().value(i)),
                    _ => continue,
                }
                out.push('\n');
            }
        }
    }
    out
}

#[test]
fn the_default_format_is_spark_sql_sources_default() {
    let s = session();
    block_on(s.table("people").write().save_as_table("p")).unwrap();
    assert!(describe_text(&s, "p").to_lowercase().contains("parquet"));
    s.conf().set("spark.sql.sources.default", "csv").unwrap();
    block_on(s.table("people").write().save_as_table("c")).unwrap();
    let text = describe_text(&s, "c").to_lowercase();
    assert!(text.contains("csv") && !text.contains("parquet"), "{text}");
    // An explicit format wins over the default, and appending with another one than the table's is refused.
    block_on(
        s.table("people")
            .write()
            .format("parquet")
            .save_as_table("e"),
    )
    .unwrap();
    assert!(describe_text(&s, "e").to_lowercase().contains("parquet"));
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .format("parquet")
                .mode("append")
                .save_as_table("c")
        )),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    block_on(s.table("people").write().mode("append").save_as_table("c")).unwrap();
}

#[test]
fn an_invalid_file_option_does_not_destroy_the_table_it_would_replace() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let w = s
        .table("people")
        .write()
        .option("maxRecordsPerFile", "many")
        .mode("overwrite");
    assert!(block_on(w.save_as_table("t")).is_err());
    assert_eq!(count(&s, "t"), 3, "the table survived");
}

#[test]
fn a_column_named_star_appends_as_a_plain_column() {
    let s = session();
    let starred = s
        .table("people")
        .select_expr(["id AS `*`", "name"])
        .unwrap();
    block_on(starred.write().save_as_table("st")).unwrap();
    block_on(starred.write().mode("append").save_as_table("st")).unwrap();
    assert_eq!(count(&s, "st"), 6);
}

#[test]
fn the_collected_call_the_python_bridge_makes_runs_the_same_writers() {
    let s = session();
    let df = s.table("people");
    let opts = || vec![("compression".to_owned(), "snappy".to_owned())];
    // saveAsTable, then insertInto, through the one-call forms.
    block_on(df.write_table_with(
        false,
        "bridge",
        Some("parquet"),
        "error",
        opts(),
        vec![],
        None,
        None,
    ))
    .unwrap();
    block_on(df.write_table_with(true, "bridge", None, "append", vec![], vec![], None, None))
        .unwrap();
    assert_eq!(count(&s, "bridge"), 6);
    // The same refusals as the builder: buckets and sort columns, partition columns on insertInto.
    let e = block_on(df.write_table_with(
        false,
        "b2",
        None,
        "error",
        vec![],
        vec![],
        Some((2, vec!["id".to_owned()])),
        None,
    ));
    assert_eq!(class(e), "KRISHIV_UNSUPPORTED_FEATURE");
    let e = block_on(df.write_table_with(
        true,
        "bridge",
        None,
        "append",
        vec![],
        vec!["id".to_owned()],
        None,
        None,
    ));
    assert_eq!(class(e), "PARTITION_BY_NOT_ALLOWED_WITH_INSERT_INTO");
    let e = block_on(df.write_table_with(
        false,
        "b3",
        None,
        "error",
        vec![],
        vec![],
        None,
        Some(vec!["name".to_owned()]),
    ));
    assert_eq!(class(e), "SORT_BY_WITHOUT_BUCKETING");
    // The file form: `save` refuses buckets as Spark does, and writes without them.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("o").to_str().unwrap().to_owned();
    let e = block_on(df.write_to_with(
        "parquet",
        "error",
        &path,
        vec![],
        vec![],
        Some((2, vec!["id".to_owned()])),
        Some(vec!["name".to_owned()]),
    ));
    assert_eq!(class(e), "_LEGACY_ERROR_TEMP_1313");
    block_on(df.write_to_with("parquet", "error", &path, opts(), vec![], None, None)).unwrap();
}

#[test]
fn an_invalid_codec_does_not_destroy_the_table_it_would_replace() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let w = s
        .table("people")
        .write()
        .option("compression", "no-such-codec")
        .mode("overwrite");
    assert!(block_on(w.save_as_table("t")).is_err());
    assert_eq!(count(&s, "t"), 3, "the table survived");
}
