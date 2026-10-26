// SPDX-License-Identifier: Apache-2.0
//! `DataFrameWriter` (`df.write`): Parquet/CSV round trips through the
//! existing reader, save modes, writer options, and the engine errors for
//! partitioning, unknown formats and unknown modes.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
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

fn out(dir: &tempfile::TempDir, name: &str) -> String {
    dir.path().join(name).to_str().unwrap().to_owned()
}

fn class(result: Result<(), KrishivError>) -> String {
    result.unwrap_err().error_class().name().to_owned()
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

/// `(id, name)` pairs of the Parquet table at `path`, ordered by `id`.
fn rows(s: &Session, path: &str) -> Vec<(i32, Option<String>)> {
    let batches = block_on(s.read().parquet([path]).collect()).unwrap();
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

/// `(id, name)` text pairs of the CSV table at `path`, ordered by `id`
/// (CSV carries no schema, so every column reads back as text).
fn csv_rows(s: &Session, path: &str, options: &[(&str, &str)]) -> Vec<(String, Option<String>)> {
    let mut reader = s.read().format("csv");
    for (key, value) in options {
        reader = reader.option(key, value);
    }
    let batches = block_on(reader.load([path]).collect()).unwrap();
    let mut rows = Vec::new();
    for batch in &batches {
        for i in 0..batch.num_rows() {
            rows.push((
                text(batch.column(0), i).unwrap_or_default(),
                text(batch.column(1), i),
            ));
        }
    }
    rows.sort();
    rows
}

#[test]
fn parquet_save_round_trips_and_marks_success() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    block_on(s.table("people").write().parquet(&path)).unwrap();
    assert!(dir.path().join("t").join("_SUCCESS").exists());
    assert_eq!(block_on(s.read().parquet([&path]).count()).unwrap(), 3);
    assert_eq!(
        rows(&s, &path),
        vec![
            (1, None),
            (2, Some("b".to_owned())),
            (3, Some("c".to_owned())),
        ]
    );
}

#[test]
fn parquet_save_modes_append_overwrite_error_ignore() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");

    // A first write, then an append: twice the rows.
    block_on(
        s.table("people")
            .write()
            .format("parquet")
            .mode("append")
            .save(&path),
    )
    .unwrap();
    block_on(
        s.table("people")
            .write()
            .format("parquet")
            .mode("append")
            .save(&path),
    )
    .unwrap();
    assert_eq!(block_on(s.read().parquet([&path]).count()).unwrap(), 6);

    // Overwrite replaces (mixed case, as Spark's modes are case-insensitive).
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
    block_on(
        s.table("one")
            .write()
            .format("parquet")
            .mode("Overwrite")
            .save(&path),
    )
    .unwrap();
    assert_eq!(rows(&s, &path), vec![(9, Some("z".to_owned()))]);

    // Every error spelling fails on an existing path and keeps its rows.
    for mode in ["error", "errorifexists", "default", "ERRORIFEXISTS"] {
        assert_eq!(
            class(block_on(
                s.table("people")
                    .write()
                    .format("parquet")
                    .mode(mode)
                    .save(&path)
            )),
            "PATH_ALREADY_EXISTS",
            "{mode}"
        );
    }
    assert_eq!(rows(&s, &path), vec![(9, Some("z".to_owned()))]);

    // Ignore leaves the existing rows and writes a missing path.
    block_on(
        s.table("people")
            .write()
            .format("parquet")
            .mode("ignore")
            .save(&path),
    )
    .unwrap();
    assert_eq!(rows(&s, &path), vec![(9, Some("z".to_owned()))]);
    let fresh = out(&dir, "fresh");
    block_on(
        s.table("people")
            .write()
            .format("parquet")
            .mode("ignore")
            .save(&fresh),
    )
    .unwrap();
    assert_eq!(block_on(s.read().parquet([&fresh]).count()).unwrap(), 3);
}

#[test]
fn csv_shortcut_round_trips_with_options() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    block_on(
        s.table("people")
            .write()
            .option("header", "true")
            .option("sep", "|")
            .csv(&path),
    )
    .unwrap();
    assert!(dir.path().join("t").join("_SUCCESS").exists());
    // The options reached the writer: the header names the columns with `|`.
    let mut header = false;
    for entry in std::fs::read_dir(dir.path().join("t")).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != "_SUCCESS" {
            let text = std::fs::read_to_string(entry.path()).unwrap();
            header |= text.lines().next().is_some_and(|line| line == "id|name");
        }
    }
    assert!(header);
    assert_eq!(
        csv_rows(&s, &path, &[("header", "true"), ("sep", "|")]),
        vec![
            ("1".to_owned(), None),
            ("2".to_owned(), Some("b".to_owned())),
            ("3".to_owned(), Some("c".to_owned())),
        ]
    );
}

#[test]
fn options_plural_and_format_csv_write_csv() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    block_on(
        s.table("people")
            .write()
            .format("csv")
            .options([("header", "true"), ("sep", ";")])
            .save(&path),
    )
    .unwrap();
    assert_eq!(
        csv_rows(&s, &path, &[("header", "true"), ("sep", ";")]),
        vec![
            ("1".to_owned(), None),
            ("2".to_owned(), Some("b".to_owned())),
            ("3".to_owned(), Some("c".to_owned())),
        ]
    );
}

#[test]
fn partition_by_fails_with_unsupported_feature() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    let err = block_on(s.table("people").write().partition_by(["name"]).save(&path)).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert!(
        err.to_string().contains("writing partitioned output"),
        "{err}"
    );
    // The failed write leaves nothing behind.
    assert!(!dir.path().join("t").exists());
}

#[test]
fn partition_by_empty_is_a_noop() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    block_on(
        s.table("people")
            .write()
            .partition_by(Vec::<&str>::new())
            .save(&path),
    )
    .unwrap();
    assert!(dir.path().join("t").join("_SUCCESS").exists());
    assert_eq!(block_on(s.read().parquet([&path]).count()).unwrap(), 3);
    assert_eq!(
        rows(&s, &path),
        vec![
            (1, None),
            (2, Some("b".to_owned())),
            (3, Some("c".to_owned())),
        ]
    );
}

#[test]
fn partition_by_replaces_the_earlier_columns() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    block_on(
        s.table("people")
            .write()
            .partition_by(["name"])
            .partition_by(Vec::<&str>::new())
            .save(&path),
    )
    .unwrap();
    assert_eq!(block_on(s.read().parquet([&path]).count()).unwrap(), 3);
}

#[test]
fn write_to_collects_format_mode_options_and_partitions() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    let df = s.table("people");
    block_on(df.write_to(
        "csv",
        "error",
        &path,
        vec![("header".to_owned(), "true".to_owned())],
        Vec::new(),
    ))
    .unwrap();
    assert_eq!(
        csv_rows(&s, &path, &[("header", "true")]).len(),
        3,
        "the options reach the writer"
    );
    let err = block_on(df.write_to(
        "parquet",
        "error",
        &out(&dir, "p"),
        Vec::new(),
        vec!["name".to_owned()],
    ))
    .unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn invalid_frame_reports_analysis_error_before_partitioning() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    let err = block_on(
        s.table("missing")
            .write()
            .partition_by(["name"])
            .save(&path),
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    assert!(!dir.path().join("t").exists());
}

#[test]
fn partition_by_with_bogus_mode_reports_invalid_save_mode() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    // The save-mode name is checked before partitioning.
    let err = block_on(
        s.table("people")
            .write()
            .partition_by(["name"])
            .mode("bogus")
            .save(&path),
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "INVALID_SAVE_MODE");
    assert!(err.to_string().contains("\"bogus\""), "{err}");
    assert!(!dir.path().join("t").exists());
}

#[test]
fn unknown_format_fails_with_unsupported_feature() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    // The JSON shortcut writes JSON files.
    block_on(s.table("people").write().json(&out(&dir, "j"))).unwrap();
    assert!(dir.path().join("j").join("_SUCCESS").exists());
    // `text` writes one string column; `people` has two, so it fails at
    // the write (not for an unknown format).
    let err = block_on(
        s.table("people")
            .write()
            .format("text")
            .save(&out(&dir, "text")),
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "_LEGACY_ERROR_TEMP_1290");
    // Any other unregistered format, through `format`.
    let path = out(&dir, "orc");
    let err = block_on(s.table("people").write().format("orc").save(&path)).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert!(err.to_string().contains("writing the orc format"), "{err}");
    // The failed write leaves nothing behind.
    assert!(!dir.path().join("orc").exists());
}

#[test]
fn csv_save_modes_append_overwrite_error_ignore() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");

    block_on(
        s.table("people")
            .write()
            .format("csv")
            .mode("append")
            .save(&path),
    )
    .unwrap();
    block_on(
        s.table("people")
            .write()
            .format("csv")
            .mode("append")
            .save(&path),
    )
    .unwrap();
    assert_eq!(csv_rows(&s, &path, &[]).len(), 6);

    // Overwrite replaces (mixed case, as Spark's modes are case-insensitive).
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
    block_on(
        s.table("one")
            .write()
            .format("csv")
            .mode("Overwrite")
            .save(&path),
    )
    .unwrap();
    assert_eq!(
        csv_rows(&s, &path, &[]),
        vec![("9".to_owned(), Some("z".to_owned()))]
    );

    // Every error spelling fails on an existing path and keeps its rows.
    for mode in ["error", "errorifexists", "default", "ERRORIFEXISTS"] {
        assert_eq!(
            class(block_on(
                s.table("people")
                    .write()
                    .format("csv")
                    .mode(mode)
                    .save(&path)
            )),
            "PATH_ALREADY_EXISTS",
            "{mode}"
        );
    }
    assert_eq!(
        csv_rows(&s, &path, &[]),
        vec![("9".to_owned(), Some("z".to_owned()))]
    );

    // Ignore leaves the existing rows and writes a missing path.
    block_on(
        s.table("people")
            .write()
            .format("csv")
            .mode("ignore")
            .save(&path),
    )
    .unwrap();
    assert_eq!(
        csv_rows(&s, &path, &[]),
        vec![("9".to_owned(), Some("z".to_owned()))]
    );
    let fresh = out(&dir, "fresh");
    block_on(
        s.table("people")
            .write()
            .format("csv")
            .mode("ignore")
            .save(&fresh),
    )
    .unwrap();
    assert_eq!(csv_rows(&s, &fresh, &[]).len(), 3);
}

#[test]
fn csv_compression_codec_fails_with_unsupported_feature() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let path = out(&dir, "t");
    let err = block_on(
        s.table("people")
            .write()
            .format("csv")
            .option("compression", "gzip")
            .save(&path),
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert!(err.to_string().contains("compression codec gzip"), "{err}");
    assert!(!dir.path().join("t").exists());
}

#[test]
fn unknown_mode_fails_with_invalid_save_mode() {
    let dir = tempfile::tempdir().unwrap();
    let s = session();
    let err = block_on(
        s.table("people")
            .write()
            .mode("bogus")
            .save(&out(&dir, "t")),
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "INVALID_SAVE_MODE");
    assert!(err.to_string().contains("\"bogus\""), "{err}");
    assert!(!dir.path().join("t").exists());
}
