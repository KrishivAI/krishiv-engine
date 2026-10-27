// SPDX-License-Identifier: Apache-2.0
//! `DataFrameWriterV2` (`df.writeTo`): table round trips through
//! `spark.table` for every provider, `DESCRIBE TABLE`, Spark's existence
//! error classes, and the engine errors for `INSERT` (append and
//! overwrites) and partitioned or clustered tables.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn people() -> (SchemaRef, RecordBatch) {
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
    (schema, batch)
}

fn session() -> Session {
    let s = Session::new().unwrap();
    let (schema, batch) = people();
    s.register_batches("people", schema, vec![batch]).unwrap();
    s
}

fn class(result: Result<(), KrishivError>) -> ErrorClass {
    result.unwrap_err().error_class()
}

/// One string column value: reads may decode strings as views, so match
/// the physical type.
fn text(column: &dyn Array, i: usize) -> Option<String> {
    match column.data_type() {
        ArrowType::Utf8View => {
            let strings = column.as_string_view();
            (!strings.is_null(i)).then(|| strings.value(i).to_owned())
        }
        ArrowType::LargeUtf8 => {
            let strings = column.as_string::<i64>();
            (!strings.is_null(i)).then(|| strings.value(i).to_owned())
        }
        _ => {
            let strings = column.as_any().downcast_ref::<StringArray>()?;
            (!strings.is_null(i)).then(|| strings.value(i).to_owned())
        }
    }
}

/// `(id, name)` pairs of the table `table`, ordered by `id`.
fn rows(s: &Session, table: &str) -> Vec<(i32, Option<String>)> {
    let batches = block_on(s.table(table).collect()).unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        let ids = batch.column(0).as_primitive::<Int32Type>();
        for i in 0..batch.num_rows() {
            rows.push((ids.value(i), text(batch.column(1), i)));
        }
    }
    rows.sort();
    rows
}

/// Every cell of `sql` as text (`NULL` for null), with the schema names.
fn described(s: &Session, sql: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let batches = block_on(df.collect()).unwrap();
    let mut out = Vec::new();
    for batch in &batches {
        for i in 0..batch.num_rows() {
            let mut row = Vec::new();
            for c in batch.columns() {
                if c.is_null(i) {
                    row.push("NULL".to_owned());
                } else {
                    row.push(text(c.as_ref(), i).unwrap_or_default());
                }
            }
            out.push(row);
        }
    }
    (names, out)
}

#[test]
fn create_parquet_round_trip() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    // The table's schema is the DataFrame's.
    let schema = block_on(s.table("t").schema()).unwrap();
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.is_nullable()))
            .collect::<Vec<_>>(),
        [("id".to_owned(), false), ("name".to_owned(), true)]
    );
    assert_eq!(
        rows(&s, "t"),
        vec![
            (1, None),
            (2, Some("b".to_owned())),
            (3, Some("c".to_owned())),
        ]
    );
}

#[test]
fn create_csv_round_trip_through_using() {
    let s = session();
    block_on(
        s.table("people")
            .write_to_table("t")
            .using("csv")
            .option("header", "true")
            .create(),
    )
    .unwrap();
    assert_eq!(
        rows(&s, "t"),
        vec![
            (1, None),
            (2, Some("b".to_owned())),
            (3, Some("c".to_owned())),
        ]
    );
}

#[test]
fn create_json_round_trip_through_using() {
    let s = session();
    block_on(s.table("people").write_to_table("t").using("json").create()).unwrap();
    assert_eq!(
        rows(&s, "t"),
        vec![
            (1, None),
            (2, Some("b".to_owned())),
            (3, Some("c".to_owned())),
        ]
    );
}

#[test]
fn create_text_round_trip_through_using() {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        ArrowType::Utf8,
        true,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![
            Some("b"),
            None,
            Some("a"),
        ]))],
    )
    .unwrap();
    s.register_batches("lines", schema, vec![batch]).unwrap();
    block_on(s.table("lines").write_to_table("t").using("text").create()).unwrap();
    let batches = block_on(s.table("t").collect()).unwrap();
    let mut values: Vec<Option<String>> = Vec::new();
    for batch in &batches {
        for i in 0..batch.num_rows() {
            values.push(text(batch.column(0), i));
        }
    }
    values.sort();
    // Text carries no nulls: a NULL writes as an empty line and reads back
    // as an empty string, as in Spark's text source.
    assert_eq!(
        values,
        vec![
            Some(String::new()),
            Some("a".to_owned()),
            Some("b".to_owned())
        ]
    );
}

#[test]
fn describe_table_shows_the_schema() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    let (names, found) = described(&s, "DESCRIBE TABLE t");
    assert_eq!(names, ["col_name", "data_type", "comment"]);
    assert_eq!(
        found,
        [
            vec!["id".to_owned(), "int".to_owned(), String::new()],
            vec!["name".to_owned(), "string".to_owned(), String::new()],
        ]
    );
    // A missing table fails as a read does.
    assert_eq!(
        block_on(s.sql("DESCRIBE TABLE missing"))
            .unwrap_err()
            .error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn create_replace_and_create_or_replace_report_existence() {
    let s = session();
    // `create` on an existing table fails, leaving its rows behind.
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    assert_eq!(
        class(block_on(s.table("people").write_to_table("t").create())),
        ErrorClass::TableOrViewAlreadyExists
    );
    assert_eq!(rows(&s, "t").len(), 3);
    // A temporary entry counts as existing too.
    assert_eq!(
        class(block_on(
            s.table("people").write_to_table("people").create()
        )),
        ErrorClass::TableOrViewAlreadyExists
    );
    // `replace` on a missing table fails.
    assert_eq!(
        class(block_on(
            s.table("people").write_to_table("missing").replace()
        )),
        ErrorClass::TableOrViewNotFound
    );
    // `replace` swaps the rows.
    let (schema, _) = people();
    s.register_batches(
        "one",
        schema,
        vec![
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("id", ArrowType::Int32, false),
                    Field::new("name", ArrowType::Utf8, true),
                ])),
                vec![
                    Arc::new(Int32Array::from(vec![9])),
                    Arc::new(StringArray::from(vec![Some("z")])),
                ],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    block_on(s.table("one").write_to_table("t").replace()).unwrap();
    assert_eq!(rows(&s, "t"), vec![(9, Some("z".to_owned()))]);
    // `createOrReplace` works both ways.
    block_on(s.table("people").write_to_table("u").create_or_replace()).unwrap();
    assert_eq!(rows(&s, "u").len(), 3);
    block_on(s.table("one").write_to_table("u").create_or_replace()).unwrap();
    assert_eq!(rows(&s, "u"), vec![(9, Some("z".to_owned()))]);
}

#[test]
fn create_in_a_namespace() {
    let s = session();
    block_on(s.sql("CREATE DATABASE mydb")).unwrap();
    block_on(s.table("people").write_to_table("mydb.t").create()).unwrap();
    assert_eq!(rows(&s, "mydb.t").len(), 3);
    // A missing namespace fails as `CREATE TABLE` does.
    assert_eq!(
        class(block_on(
            s.table("people").write_to_table("missing.t").create()
        )),
        ErrorClass::SchemaNotFound
    );
}

#[test]
fn failing_create_leaves_no_table() {
    let s = session();
    // An unsupported provider fails before anything is written.
    let err = block_on(
        s.table("people")
            .write_to_table("t")
            .using("delta")
            .create(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::DataSourceNotFound);
    assert_eq!(
        block_on(s.table("t").schema()).unwrap_err().error_class(),
        ErrorClass::TableOrViewNotFound
    );
    // A write failure (two columns as `text`) leaves neither metadata nor
    // files: the same all-or-nothing as CTAS. A secret option value never
    // reaches the error text.
    let err = block_on(
        s.table("people")
            .write_to_table("u")
            .using("text")
            .option("password", "hunter2")
            .create(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::LegacyErrorTemp1290);
    assert!(!err.to_string().contains("hunter2"), "{err}");
    assert_eq!(
        block_on(s.table("u").schema()).unwrap_err().error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn invalid_frame_reports_analysis_error() {
    let s = session();
    // The DataFrame's errors come first, as for `df.write`.
    assert_eq!(
        class(block_on(s.table("missing").write_to_table("t").create())),
        ErrorClass::TableOrViewNotFound
    );
    assert_eq!(
        block_on(s.table("t").schema()).unwrap_err().error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn append_reports_insert_as_unsupported() {
    // Written so it fails visibly when `INSERT` lands: then no error is
    // raised and the assertion on the class fails.
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    let err = block_on(s.table("people").write_to_table("t").append()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("INSERT"), "{err}");
    assert_eq!(rows(&s, "t").len(), 3);
}

#[test]
fn overwrite_reports_insert_as_unsupported() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    let err = block_on(s.table("people").write_to_table("t").overwrite("id > 1")).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("INSERT"), "{err}");
    assert_eq!(rows(&s, "t").len(), 3);
}

#[test]
fn overwrite_partitions_reports_insert_as_unsupported() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    let err = block_on(s.table("people").write_to_table("t").overwrite_partitions()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("INSERT"), "{err}");
    assert_eq!(rows(&s, "t").len(), 3);
}

#[test]
fn partitioned_tables_are_unsupported() {
    // Written so it fails visibly when partitioned tables land: then the
    // write succeeds and the assertion on the class fails.
    let s = session();
    let err = block_on(
        s.table("people")
            .write_to_table("t")
            .partitioned_by(["id"])
            .create(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("partitioned tables"), "{err}");
    assert_eq!(
        block_on(s.table("t").schema()).unwrap_err().error_class(),
        ErrorClass::TableOrViewNotFound
    );
    // An empty call clears the columns, as in Spark.
    block_on(
        s.table("people")
            .write_to_table("u")
            .partitioned_by(["id"])
            .partitioned_by(Vec::<&str>::new())
            .create(),
    )
    .unwrap();
    assert_eq!(rows(&s, "u").len(), 3);
}

#[test]
fn clustered_tables_are_unsupported() {
    let s = session();
    let err = block_on(
        s.table("people")
            .write_to_table("t")
            .cluster_by(["id"])
            .create(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("partitioned tables"), "{err}");
    assert_eq!(
        block_on(s.table("t").schema()).unwrap_err().error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn options_and_table_properties_reach_the_table() {
    let s = session();
    // The options write the files (a header names the columns with `|`) ...
    block_on(
        s.table("people")
            .write_to_table("t")
            .using("csv")
            .option("header", "true")
            .options([("sep", "|")])
            .table_property("k", "v")
            .create(),
    )
    .unwrap();
    assert_eq!(rows(&s, "t").len(), 3);
    // ... and `replace` keeps them working.
    block_on(
        s.table("people")
            .write_to_table("t")
            .using("csv")
            .option("header", "true")
            .options([("sep", "|")])
            .replace(),
    )
    .unwrap();
    assert_eq!(rows(&s, "t").len(), 3);
}
