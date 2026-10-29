// SPDX-License-Identifier: Apache-2.0
//! The cast-named functions (`float(x)`, `int(x)`, ...) end to end: each
//! call runs as its cast, with Spark's schema names, types, rows and
//! error classes.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

/// The rows of a one-row `SELECT`, rendered as Spark renders them.
fn cell(batch: &RecordBatch, col: usize) -> String {
    if batch.column(col).is_null(0) {
        return "NULL".to_owned();
    }
    match batch.column(col).data_type() {
        ArrowType::Int8 => batch
            .column(col)
            .as_primitive::<arrow_array::types::Int8Type>()
            .value(0)
            .to_string(),
        ArrowType::Int16 => batch
            .column(col)
            .as_primitive::<arrow_array::types::Int16Type>()
            .value(0)
            .to_string(),
        ArrowType::Int32 => batch
            .column(col)
            .as_primitive::<Int32Type>()
            .value(0)
            .to_string(),
        ArrowType::Int64 => batch
            .column(col)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0)
            .to_string(),
        ArrowType::Float32 => batch
            .column(col)
            .as_primitive::<Float32Type>()
            .value(0)
            .to_string(),
        ArrowType::Float64 => batch
            .column(col)
            .as_primitive::<Float64Type>()
            .value(0)
            .to_string(),
        ArrowType::Boolean => batch.column(col).as_boolean().value(0).to_string(),
        ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
            match batch.column(col).data_type() {
                ArrowType::Utf8View => batch.column(col).as_string_view().value(0).to_owned(),
                ArrowType::LargeUtf8 => batch.column(col).as_string::<i64>().value(0).to_owned(),
                _ => batch.column(col).as_string::<i32>().value(0).to_owned(),
            }
        }
        ArrowType::Decimal128(_, s) => {
            let v = batch
                .column(col)
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(0);
            let (neg, abs) = if v < 0 {
                (true, v.unsigned_abs())
            } else {
                (false, v.cast_unsigned())
            };
            let digits = abs.to_string();
            let s = usize::try_from(*s).unwrap_or(0);
            let text = if s == 0 {
                digits
            } else if digits.len() > s {
                let (a, b) = digits.split_at(digits.len() - s);
                format!("{a}.{b}")
            } else {
                format!("0.{:0>width$}", digits, width = s)
            };
            format!("{}{text}", if neg { "-" } else { "" })
        }
        ArrowType::Date32 => batch
            .column(col)
            .as_primitive::<arrow_array::types::Date32Type>()
            .value(0)
            .to_string(),
        ArrowType::Timestamp(_, _) => batch
            .column(col)
            .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0)
            .to_string(),
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView => {
            format!("{:?}", batch.column(col).as_binary::<i32>().value(0))
        }
        t => panic!("unrendered {t}"),
    }
}

fn one_row(s: &Session, sql: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let types: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.data_type().to_string())
        .collect();
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let batch = &batches[0];
    let row: Vec<String> = (0..batch.num_columns()).map(|c| cell(batch, c)).collect();
    (names, types, row)
}

fn error_of(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Ok(df) => match block_on(df.collect()) {
            Ok(b) => panic!("{sql} ran: {b:?}"),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

#[test]
fn cast_named_functions_match_casts() {
    let s = session();
    // The ground-truth cases: Spark's names, types and values.
    let (names, types, row) = one_row(&s, "SELECT float(15.0), double('1.5'), int('7')");
    assert_eq!(names, ["15.0", "1.5", "7"]);
    assert_eq!(types, ["Float32", "Float64", "Int32"]);
    assert_eq!(row, ["15", "1.5", "7"]);
    // `boolean` over a boolean (wider boolean casts are not in the cast
    // path yet, and fail exactly as `CAST` does).
    let (names, _, row) = one_row(&s, "SELECT boolean(true), string(1), date('2024-01-02')");
    assert_eq!(names, ["true", "1", "2024-01-02"]);
    assert_eq!(row[0], "true");
    assert_eq!(row[1], "1");
    // `decimal` is `decimal(10,0)`: halves round up and fractions
    // truncate, as the cast does.
    let (names, types, row) = one_row(&s, "SELECT decimal(1.5), decimal('1.234')");
    assert_eq!(names, ["1.5", "1.234"]);
    assert_eq!(types, ["Decimal128(10, 0)", "Decimal128(10, 0)"]);
    assert_eq!(row, ["2", "1"]);
}

#[test]
fn cast_named_functions_reject_bad_input_per_ansi() {
    let s = session();
    // ANSI on: malformed input is an error.
    let e = error_of(&s, "SELECT int('x')");
    assert_eq!(e.error_class(), ErrorClass::CastInvalidInput);
    // ANSI off: malformed input is NULL.
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    let (_, _, row) = one_row(&s, "SELECT int('x')");
    assert_eq!(row, ["NULL"]);
    // The wrong argument count quotes the call.
    for (sql, actual) in [("SELECT float(1, 2)", "2"), ("SELECT float()", "0")] {
        let e = error_of(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::WrongNumArgsWithoutSuggestion,
            "{sql}"
        );
        let params = e.message_parameters();
        let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
        assert_eq!(get("functionName"), "`float`", "{sql}");
        assert_eq!(get("expectedNum"), "1", "{sql}");
        assert_eq!(get("actualNum"), actual, "{sql}");
    }
}
