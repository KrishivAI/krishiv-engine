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
        "_LEGACY_ERROR_TEMP_1162"
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
    // saveAsTable buckets: the table is written and reads back whole.
    let w = s.table("people").write().bucket_by(2, ["id"]);
    block_on(w.save_as_table("t")).unwrap();
    assert_eq!(count(&s, "t"), count(&s, "people"));
}

/// E34: an explicit idempotency key on the `saveAsTable` that creates a file table is not stored with
/// the table: a later append without a key adds its rows.
#[test]
fn a_creation_key_is_not_kept_in_the_files_tables_options() {
    let s = session();
    let n = count(&s, "people");
    block_on(
        s.table("people")
            .write()
            .option("krishiv.write.idempotencyKey", "job-1")
            .save_as_table("keyed_files"),
    )
    .unwrap();
    assert_eq!(count(&s, "keyed_files"), n);
    block_on(
        s.table("people")
            .write()
            .mode("append")
            .save_as_table("keyed_files"),
    )
    .unwrap();
    assert_eq!(count(&s, "keyed_files"), 2 * n);
}

/// E34: the key reaches the file table's own writes, the creating one and an append to the existing
/// table: repeating either with the same key adds no rows.
#[test]
fn an_explicit_key_deduplicates_file_table_saves() {
    let s = session();
    let n = count(&s, "people");
    let keyed = |mode: &str, key: &str| {
        block_on(
            s.table("people")
                .write()
                .mode(mode)
                .option("krishiv.write.idempotencyKey", key)
                .save_as_table("keyed_dedup"),
        )
    };
    keyed("append", "job-1").unwrap();
    assert_eq!(count(&s, "keyed_dedup"), n);
    keyed("append", "job-1").unwrap();
    assert_eq!(count(&s, "keyed_dedup"), n);
    keyed("append", "job-2").unwrap();
    assert_eq!(count(&s, "keyed_dedup"), 2 * n);
}

/// E34 (review of #966): an overwrite whose key the write would refuse leaves the old table in place.
#[test]
fn a_refused_key_on_an_overwrite_keeps_the_old_table() {
    let s = session();
    let n = count(&s, "people");
    block_on(s.table("people").write().save_as_table("kept")).unwrap();
    let empty = block_on(
        s.table("people")
            .write()
            .mode("overwrite")
            .option("krishiv.write.idempotencyKey", "")
            .save_as_table("kept"),
    );
    assert!(empty.is_err());
    assert_eq!(count(&s, "kept"), n);
    s.conf()
        .set("krishiv.sql.files.commitProtocol", "NONE")
        .unwrap();
    let none = block_on(
        s.table("people")
            .write()
            .mode("overwrite")
            .option("krishiv.write.idempotencyKey", "job-1")
            .save_as_table("kept"),
    );
    assert_eq!(class(none), "KRISHIV_UNSUPPORTED_FEATURE");
    assert_eq!(count(&s, "kept"), n);
}

/// E34: an overwrite with a usable key replaces the table, and repeating it with that key adds nothing.
#[test]
fn a_usable_key_on_an_overwrite_replaces_the_table() {
    let s = session();
    let n = count(&s, "people");
    let keyed = || {
        block_on(
            s.table("people")
                .write()
                .mode("overwrite")
                .option("krishiv.write.idempotencyKey", "job-9")
                .save_as_table("replaced"),
        )
    };
    keyed().unwrap();
    assert_eq!(count(&s, "replaced"), n);
    keyed().unwrap();
    assert_eq!(count(&s, "replaced"), n);
}

/// E34 (review of #966): a retried keyed overwrite or `errorifexists` create writes nothing, so it
/// cannot drop what a later write added.
#[test]
fn a_retried_keyed_save_as_table_does_not_undo_a_later_append() {
    let s = session();
    let n = count(&s, "people");
    let write = |mode: &str, key: &str| {
        block_on(
            s.table("people")
                .write()
                .mode(mode)
                .option("krishiv.write.idempotencyKey", key)
                .save_as_table("retried"),
        )
    };
    write("overwrite", "k1").unwrap();
    write("append", "k2").unwrap();
    assert_eq!(count(&s, "retried"), 2 * n);
    // The retry of the overwrite: nothing is dropped.
    write("overwrite", "k1").unwrap();
    assert_eq!(count(&s, "retried"), 2 * n);
    // The retry of a creating write does not fail on the table the first try made.
    write("errorifexists", "k1").unwrap();
    assert_eq!(count(&s, "retried"), 2 * n);
}

/// E34 (review of #966): strict option checking accepts the forwarded key.
#[test]
fn strict_options_accept_the_idempotency_key_on_a_file_table() {
    let s = session();
    s.conf().set("krishiv.io.strictOptions", "true").unwrap();
    block_on(
        s.table("people")
            .write()
            .option("krishiv.write.idempotencyKey", "job-1")
            .save_as_table("strict_keyed"),
    )
    .unwrap();
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
    // a different format than the table's, and a partitioning it does not have, are refused as Spark does.
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .format("csv")
                .mode("append")
                .save_as_table("t")
        )),
        "_LEGACY_ERROR_TEMP_1159"
    );
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .partition_by(["id"])
                .mode("append")
                .save_as_table("t")
        )),
        "_LEGACY_ERROR_TEMP_1163"
    );
    // overwrite with a partitioning on a new table creates it, partitioned.
    block_on(
        s.table("people")
            .write()
            .partition_by(["id"])
            .mode("overwrite")
            .save_as_table("p"),
    )
    .unwrap();
    assert_eq!(count(&s, "p"), 3);
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
    // An overwrite that replaces the table with a partitioned one.
    block_on(
        s.table("people")
            .write()
            .partition_by(["id"])
            .mode("overwrite")
            .save_as_table("t"),
    )
    .unwrap();
    assert_eq!(count(&s, "t"), 3);
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
        "_LEGACY_ERROR_TEMP_1159"
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
    // The same refusals as the builder: sort columns, partition columns on insertInto; buckets write.
    block_on(df.write_table_with(
        false,
        "b2",
        None,
        "error",
        vec![],
        vec![],
        Some((2, vec!["id".to_owned()])),
        None,
    ))
    .unwrap();
    assert_eq!(count(&s, "b2"), 3);
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

/// A `saveAsTable` overwrite whose new table a collated schema refuses (schema-level collations, row 14) is
/// refused before the old table is dropped: the table and its rows survive.
#[test]
fn an_overwrite_a_collated_schema_refuses_keeps_the_old_table() {
    let s = session();
    block_on(s.sql("CREATE SCHEMA c")).unwrap();
    block_on(s.table("people").write().save_as_table("c.t")).unwrap();
    assert_eq!(count(&s, "c.t"), 3);
    block_on(s.sql("SET spark.sql.collation.schemaLevel.enabled = true")).unwrap();
    block_on(s.sql("ALTER SCHEMA c DEFAULT COLLATION UTF8_LCASE")).unwrap();
    let refused = block_on(
        s.table("people")
            .filter(col("id").gt(lit(1)))
            .write()
            .mode("overwrite")
            .save_as_table("c.t"),
    );
    assert!(
        refused.is_err(),
        "the create in a collated schema is refused"
    );
    assert_eq!(count(&s, "c.t"), 3, "the old table survives");
}

/// A table that exists before its schema gains a default collation can still be renamed: the rename drops the
/// source entry and creates the new one, and the store must not refuse either (the new-table refusal lives in the
/// SQL create and the catalog adapter, before the store).
#[test]
fn a_table_in_a_schema_that_gained_a_collation_can_be_renamed() {
    let s = session();
    block_on(s.sql("CREATE SCHEMA c")).unwrap();
    block_on(s.table("people").write().save_as_table("c.t")).unwrap();
    block_on(s.sql("SET spark.sql.collation.schemaLevel.enabled = true")).unwrap();
    block_on(s.sql("ALTER SCHEMA c DEFAULT COLLATION UTF8_LCASE")).unwrap();
    block_on(s.sql("ALTER TABLE c.t RENAME TO c.u")).unwrap();
    assert_eq!(count(&s, "c.u"), 3, "the renamed table keeps its rows");
    assert!(
        block_on(s.sql("SELECT * FROM c.t")).is_err(),
        "the old name is gone"
    );
}

#[test]
fn an_overwrite_with_a_missing_layout_column_keeps_the_old_table() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let r = block_on(
        s.table("people")
            .write()
            .partition_by(["missing"])
            .mode("overwrite")
            .save_as_table("t"),
    );
    assert!(r.is_err());
    assert_eq!(count(&s, "t"), 3, "the old table must survive the refusal");
    let r = block_on(
        s.table("people")
            .write()
            .bucket_by(2, ["missing"])
            .mode("overwrite")
            .save_as_table("t"),
    );
    assert!(r.is_err());
    assert_eq!(count(&s, "t"), 3);
}

#[test]
fn an_empty_bucket_column_list_is_refused_before_the_old_table_is_dropped() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let r = block_on(
        s.table("people")
            .write()
            .bucket_by(4, Vec::<String>::new())
            .mode("overwrite")
            .save_as_table("t"),
    );
    assert_eq!(class(r), "KRISHIV_UNSUPPORTED_FEATURE");
    assert_eq!(count(&s, "t"), 3, "the old table must survive the refusal");
}

#[test]
fn a_bucketed_table_of_one_string_column_writes_as_text() {
    let s = session();
    let one = s.table("people").select(["name"]);
    block_on(
        one.write()
            .format("text")
            .bucket_by(2, ["name"])
            .save_as_table("tb"),
    )
    .unwrap();
    assert_eq!(count(&s, "tb"), 3);
}

/// A text table keeps one stored column: the partition column is not counted by the format check.
#[test]
fn a_partitioned_and_bucketed_text_table_checks_only_the_stored_column() {
    let s = session();
    let two = s
        .table("people")
        .select_expr(["name", "CAST(id AS STRING) AS p"])
        .unwrap();
    block_on(
        two.write()
            .format("text")
            .partition_by(["p"])
            .bucket_by(2, ["name"])
            .save_as_table("tp"),
    )
    .unwrap();
    assert_eq!(count(&s, "tp"), 3);
}

/// A bucket column the hash refuses (a MAP) is found before an overwrite drops the old table.
#[test]
fn an_unhashable_bucket_column_keeps_the_table_an_overwrite_would_replace() {
    let s = session();
    block_on(s.table("people").write().save_as_table("keep")).unwrap();
    let m = s
        .table("people")
        .select_expr(["id", "map('k', name) AS m"])
        .unwrap();
    let e = block_on(
        m.write()
            .mode("overwrite")
            .bucket_by(2, ["m"])
            .save_as_table("keep"),
    )
    .unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert_eq!(count(&s, "keep"), 3);
    assert_eq!(count(&s, "keep"), 3);
}

#[test]
fn an_append_names_the_tables_bucketing_or_none() {
    let s = session();
    block_on(
        s.table("people")
            .write()
            .bucket_by(4, ["id"])
            .save_as_table("b"),
    )
    .unwrap();
    for w in [
        s.table("people").write().bucket_by(8, ["id"]),
        s.table("people").write().bucket_by(4, ["name"]),
        s.table("people")
            .write()
            .bucket_by(4, ["id"])
            .sort_by(["name"]),
    ] {
        assert_eq!(
            class(block_on(w.mode("append").save_as_table("b"))),
            "_LEGACY_ERROR_TEMP_1164"
        );
    }
    block_on(
        s.table("people")
            .write()
            .bucket_by(4, ["ID"])
            .mode("append")
            .save_as_table("b"),
    )
    .unwrap();
    assert_eq!(count(&s, "b"), 6);
    // Spark compares the two specs as options: no `bucketBy` into a bucketed table is refused,
    // and so is `bucketBy` into a plain one.
    assert_eq!(
        class(block_on(
            s.table("people").write().mode("append").save_as_table("b")
        )),
        "_LEGACY_ERROR_TEMP_1164"
    );
    block_on(s.table("people").write().save_as_table("plain")).unwrap();
    assert_eq!(
        class(block_on(
            s.table("people")
                .write()
                .bucket_by(4, ["id"])
                .mode("append")
                .save_as_table("plain")
        )),
        "_LEGACY_ERROR_TEMP_1164"
    );
    assert_eq!(count(&s, "b"), 6);
}

#[test]
fn a_column_named_like_the_bucket_helper_does_not_collide() {
    let s = session();
    let df = s
        .table("people")
        .select_expr(["id", "name AS __krishiv_bucket_id"])
        .unwrap();
    block_on(df.write().bucket_by(2, ["id"]).save_as_table("h")).unwrap();
    assert_eq!(count(&s, "h"), 3);
    let df = s
        .table("people")
        .select_expr(["id", "name AS __KRISHIV_BUCKET_ID"])
        .unwrap();
    block_on(df.write().bucket_by(2, ["id"]).save_as_table("h2")).unwrap();
    assert_eq!(count(&s, "h2"), 3);
}

fn part_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            part_files(&p, out);
        } else if p
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("part-"))
        {
            out.push(p);
        }
    }
}

/// An SQL `INSERT` into a bucketed table, and into a bucketed and partitioned one, places each row in
/// Spark's bucket too (the files, not just the row count).
#[test]
fn sql_insert_into_bucketed_tables_places_rows_in_spark_buckets() {
    for ddl in [
        "CREATE TABLE sb (id INT, name STRING) USING parquet CLUSTERED BY (id) INTO 4 BUCKETS",
        "CREATE TABLE sb (id INT, name STRING, p INT) USING parquet PARTITIONED BY (p) CLUSTERED BY (id) INTO 4 BUCKETS",
    ] {
        let s = Session::new().unwrap();
        let warehouse = tempfile::tempdir().unwrap();
        s.conf()
            .set(
                "spark.sql.warehouse.dir",
                warehouse.path().to_str().unwrap(),
            )
            .unwrap();
        block_on(block_on(s.sql(ddl)).unwrap().collect()).unwrap();
        let cols = if ddl.contains("PARTITIONED") {
            "CAST(id AS INT), CAST(concat('n', id % 5) AS STRING), CAST(id % 3 AS INT)"
        } else {
            "CAST(id AS INT), CAST(concat('n', id % 5) AS STRING)"
        };
        block_on(
            block_on(s.sql(&format!("INSERT INTO sb SELECT {cols} FROM range(0, 40)")))
                .unwrap()
                .collect(),
        )
        .unwrap();
        let mut files = Vec::new();
        part_files(warehouse.path(), &mut files);
        assert!(files.len() >= 2, "{files:?}");
        let mut total = 0;
        for f in &files {
            let bucket =
                krishiv_io::bucket_id_of_file_name(f.file_name().unwrap().to_str().unwrap())
                    .expect("a bucketed file name carries its bucket id");
            let rows = s.read().format("parquet").load([f.to_str().unwrap()]);
            let wrong = block_on(
                rows.clone()
                    .filter_expr(&format!("pmod(hash(id), 4) != {bucket}"))
                    .unwrap()
                    .count(),
            )
            .unwrap();
            assert_eq!(wrong, 0, "{f:?}: rows outside bucket {bucket}");
            total += block_on(rows.count()).unwrap();
        }
        assert_eq!(total, 40, "{ddl}");
    }
}

/// Every row of every file of a bucketed table is in the bucket its file name says, and that bucket is
/// Spark's `pmod(hash(columns), n)` (Murmur3, seed 42) over the bucket columns in the order given.
#[test]
fn rows_land_in_the_bucket_spark_puts_them_in() {
    for (cols, expr) in [
        (vec!["id"], "pmod(hash(id), 4)"),
        (vec!["name", "id"], "pmod(hash(name, id), 4)"),
    ] {
        let s = Session::new().unwrap();
        let warehouse = tempfile::tempdir().unwrap();
        s.conf()
            .set(
                "spark.sql.warehouse.dir",
                warehouse.path().to_str().unwrap(),
            )
            .unwrap();
        let df = block_on(s.sql(
            "SELECT CAST(id AS INT) AS id, CAST(concat('n', id % 5) AS STRING) AS name FROM range(0, 40)",
        ))
        .unwrap();
        block_on(df.write().bucket_by(4, cols.clone()).save_as_table("b")).unwrap();
        let mut files = Vec::new();
        part_files(warehouse.path(), &mut files);
        assert!(files.len() >= 2, "{files:?}");
        let mut total = 0;
        let mut seen = std::collections::BTreeSet::new();
        for f in &files {
            let bucket =
                krishiv_io::bucket_id_of_file_name(f.file_name().unwrap().to_str().unwrap())
                    .expect("a bucketed file name carries its bucket id");
            seen.insert(bucket);
            let path = f.to_str().unwrap();
            let rows = s.read().format("parquet").load([path]);
            let n = block_on(rows.clone().count()).unwrap();
            let wrong = block_on(
                rows.filter_expr(&format!("{expr} != {bucket}"))
                    .unwrap()
                    .count(),
            )
            .unwrap();
            assert_eq!(wrong, 0, "{path}: rows outside bucket {bucket} ({expr})");
            total += n;
        }
        assert_eq!(total, 40);
        assert!(seen.iter().all(|b| *b < 4), "{seen:?}");
    }
    // Spark's `hash(1)` is -559580957, so `pmod(hash(1), 4)` is 3: the row `id = 1` is in bucket 3.
    let s = Session::new().unwrap();
    let warehouse = tempfile::tempdir().unwrap();
    s.conf()
        .set(
            "spark.sql.warehouse.dir",
            warehouse.path().to_str().unwrap(),
        )
        .unwrap();
    let one = block_on(s.sql("SELECT CAST(1 AS INT) AS id")).unwrap();
    block_on(one.write().bucket_by(4, ["id"]).save_as_table("one")).unwrap();
    let mut files = Vec::new();
    part_files(warehouse.path(), &mut files);
    assert_eq!(files.len(), 1);
    assert_eq!(
        krishiv_io::bucket_id_of_file_name(files[0].file_name().unwrap().to_str().unwrap()),
        Some(3)
    );
}

#[test]
fn an_empty_bucketed_write_leaves_no_file_with_the_helper_column() {
    let s = session();
    let none = s.table("people").filter(col("id").gt(lit(100)));
    block_on(none.write().bucket_by(2, ["id"]).save_as_table("eb")).unwrap();
    assert_eq!(count(&s, "eb"), 0);
    // An append of an empty frame to it is the same.
    let none = s.table("people").filter(col("id").gt(lit(100)));
    block_on(
        none.write()
            .bucket_by(2, ["id"])
            .mode("append")
            .save_as_table("eb"),
    )
    .unwrap();
    assert_eq!(count(&s, "eb"), 0);
    let schema = block_on(s.table("eb").schema()).unwrap();
    assert_eq!(schema.fields().len(), 2, "no bucket id column: {schema:?}");
}

#[test]
fn an_append_of_other_columns_is_refused_by_count() {
    let s = session();
    block_on(s.table("people").write().save_as_table("t")).unwrap();
    let one = s.table("people").select(["id"]);
    assert_eq!(
        class(block_on(one.write().mode("append").save_as_table("t"))),
        "_LEGACY_ERROR_TEMP_1161"
    );
    assert_eq!(count(&s, "t"), 3);
}

#[test]
fn saving_into_a_view_is_refused() {
    let s = session();
    block_on(s.table("people").write().save_as_table("base")).unwrap();
    block_on(s.sql("CREATE VIEW v AS SELECT * FROM base")).unwrap();
    assert_eq!(
        class(block_on(
            s.table("people").write().mode("append").save_as_table("v")
        )),
        "_LEGACY_ERROR_TEMP_1158"
    );
}

fn param(e: &KrishivError, name: &str) -> String {
    e.message_parameters()
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| v.clone())
        .unwrap_or_else(|| panic!("no parameter {name}"))
}

/// Spark compares the writer's source always: an append with no `format` is `spark.sql.sources.default`
/// (parquet) against the table's, with Spark's class names in the message.
#[test]
fn an_append_with_no_format_is_checked_against_the_tables_format() {
    let s = session();
    block_on(s.table("people").write().format("json").save_as_table("j")).unwrap();
    let e = block_on(s.table("people").write().mode("append").save_as_table("j")).unwrap_err();
    assert_eq!(e.error_class().name(), "_LEGACY_ERROR_TEMP_1159");
    assert_eq!(param(&e, "existingProvider"), "JsonDataSourceV2");
    assert_eq!(param(&e, "specifiedProvider"), "ParquetDataSourceV2");
}

/// Spark's check order is 1159, 1161, 1162, 1163, 1164: a partitioned table with an append of an extra
/// column and no `partitionBy` is 1161 (not 1163), and one with other names is 1162.
#[test]
fn append_refusals_come_in_spark_order_and_1164_prints_spark_text() {
    let s = session();
    block_on(
        s.table("people")
            .write()
            .partition_by(["name"])
            .save_as_table("pt"),
    )
    .unwrap();
    let extra = s
        .table("people")
        .select_expr(["id", "name", "1 AS x"])
        .unwrap();
    assert_eq!(
        class(block_on(extra.write().mode("append").save_as_table("pt"))),
        "_LEGACY_ERROR_TEMP_1161"
    );
    let renamed = s.table("people").select_expr(["id AS i", "name"]).unwrap();
    assert_eq!(
        class(block_on(renamed.write().mode("append").save_as_table("pt"))),
        "_LEGACY_ERROR_TEMP_1162"
    );
    block_on(
        s.table("people")
            .write()
            .bucket_by(4, ["id"])
            .sort_by(["name"])
            .save_as_table("bk"),
    )
    .unwrap();
    let e = block_on(s.table("people").write().mode("append").save_as_table("bk")).unwrap_err();
    assert_eq!(param(&e, "specifiedBucketString"), "not bucketed");
    assert_eq!(
        param(&e, "existingBucketString"),
        "4 buckets, bucket columns: [id], sort columns: [name]"
    );
    block_on(
        s.table("people")
            .write()
            .bucket_by(4, ["id"])
            .save_as_table("bn"),
    )
    .unwrap();
    let e = block_on(
        s.table("people")
            .write()
            .bucket_by(8, ["id"])
            .mode("append")
            .save_as_table("bn"),
    )
    .unwrap_err();
    assert_eq!(
        param(&e, "existingBucketString"),
        "4 buckets, bucket columns: [id]"
    );
    assert_eq!(
        param(&e, "specifiedBucketString"),
        "8 buckets, bucket columns: [id]"
    );
}
