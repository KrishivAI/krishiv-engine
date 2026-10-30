// SPDX-License-Identifier: Apache-2.0
//! `DataFrameWriterV2` (`df.writeTo`): table round trips through
//! `spark.table`, append, Spark's existence error classes, and the engine
//! errors for replacing, overwriting and partitioned or clustered tables.
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

#[test]
fn create_round_trips_and_a_second_create_fails() {
    let s = session();
    block_on(
        s.table("people")
            .write_to_table("t")
            .using("parquet")
            .create(),
    )
    .unwrap();
    assert_eq!(
        rows(&s, "t"),
        vec![
            (1, None),
            (2, Some("b".to_owned())),
            (3, Some("c".to_owned()))
        ]
    );
    // Spark's CTAS makes every column nullable.
    let schema = block_on(s.table("t").schema()).unwrap();
    assert!(schema.fields().iter().all(|f| f.is_nullable()));
    assert_eq!(
        class(block_on(s.table("people").write_to_table("t").create())),
        ErrorClass::TableOrViewAlreadyExists
    );
    // The failed second create left the first table's rows alone.
    assert_eq!(rows(&s, "t").len(), 3);
}

#[test]
fn create_csv_through_using_reads_back_with_a_header() {
    let s = session();
    block_on(s.table("people").write_to_table("c").using("csv").create()).unwrap();
    assert_eq!(rows(&s, "c").len(), 3);
}

#[test]
fn append_adds_the_rows_by_position() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    block_on(s.table("people").write_to_table("t").append()).unwrap();
    assert_eq!(rows(&s, "t").len(), 6);
    assert_eq!(
        class(block_on(s.table("people").write_to_table("nope").append())),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn replace_and_create_or_replace_fail_as_replace_table_as_select_does() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    for result in [
        block_on(s.table("people").write_to_table("t").replace()),
        block_on(s.table("people").write_to_table("t").create_or_replace()),
        block_on(
            s.table("people")
                .write_to_table("fresh")
                .create_or_replace(),
        ),
    ] {
        let err = result.unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::UnsupportedFeatureTableOperation
        );
        assert!(err.to_string().contains("REPLACE TABLE AS SELECT"), "{err}");
    }
    // Nothing was created or changed.
    assert_eq!(rows(&s, "t").len(), 3);
    assert_eq!(
        block_on(s.table("fresh").schema())
            .unwrap_err()
            .error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn failing_create_leaves_no_table() {
    let s = session();
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
    // A secret option is never stored in the table's metadata.
    let err = block_on(
        s.table("people")
            .write_to_table("u")
            .option("password", "hunter2")
            .create(),
    )
    .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(!err.to_string().contains("hunter2"), "{err}");
    assert_eq!(
        block_on(s.table("u").schema()).unwrap_err().error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn invalid_frame_reports_analysis_error() {
    let s = session();
    assert_eq!(
        class(block_on(s.table("missing").write_to_table("t").create())),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn overwrites_partitioned_and_clustered_tables_are_unsupported() {
    let s = session();
    block_on(s.table("people").write_to_table("t").create()).unwrap();
    assert_eq!(
        class(block_on(
            s.table("people").write_to_table("t").overwrite("true")
        )),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(
        class(block_on(
            s.table("people").write_to_table("t").overwrite_partitions()
        )),
        ErrorClass::KrishivUnsupportedFeature
    );
    for w in [
        s.table("people")
            .write_to_table("p")
            .partitioned_by(["name"]),
        s.table("people").write_to_table("p").cluster_by(["name"]),
    ] {
        let err = block_on(w.create()).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(err.to_string().contains("partitioned tables"), "{err}");
    }
}

#[test]
fn options_and_properties_reach_the_table_and_the_name_may_be_qualified() {
    let s = session();
    block_on(
        s.table("people")
            .write_to_table("spark_catalog.default.q")
            .using("csv")
            .option("sep", ";")
            .table_property("k", "v")
            .create(),
    )
    .unwrap();
    assert_eq!(rows(&s, "q").len(), 3);
    assert_eq!(rows(&s, "default.q").len(), 3);
}

#[test]
fn the_bridge_call_runs_every_mode() {
    let s = session();
    let df = s.table("people");
    let run = |table: &str, mode: &str, extra: bool| {
        block_on(df.write_to_table_with(
            table,
            extra.then(|| "parquet".to_owned()),
            if extra {
                vec![("compression".to_owned(), "snappy".to_owned())]
            } else {
                Vec::new()
            },
            if extra {
                vec![("k".to_owned(), "v".to_owned())]
            } else {
                Vec::new()
            },
            Vec::new(),
            Vec::new(),
            mode,
            Some("true".to_owned()),
        ))
    };
    run("b", "create", true).unwrap();
    run("b", "append", false).unwrap();
    assert_eq!(rows(&s, "b").len(), 6);
    for mode in [
        "replace",
        "createOrReplace",
        "overwrite",
        "overwritePartitions",
    ] {
        assert!(run("b", mode, false).is_err(), "{mode}");
    }
    assert_eq!(
        run("b", "bogus", false).unwrap_err().error_class(),
        ErrorClass::InternalError
    );
    // Partition and cluster columns reach the writer and are refused.
    let e = block_on(df.write_to_table_with(
        "c",
        None,
        Vec::new(),
        Vec::new(),
        vec!["name".to_owned()],
        vec!["id".to_owned()],
        "create",
        None,
    ))
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}
