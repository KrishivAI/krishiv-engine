// SPDX-License-Identifier: Apache-2.0
//! `from_csv` and `to_csv` end to end: the literal statements of Spark 4.2's
//! `csv-functions.sql` that need no other unimplemented function, plus the
//! CSV statements of `parse-schema-string.sql`.

#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

/// The rows of a `SELECT`, rendered as Spark renders them for structs
/// (JSON-like) and strings.
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for col in b.columns() {
                if col.is_null(i) {
                    row.push("NULL".to_owned());
                    continue;
                }
                row.push(match col.data_type() {
                    ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
                        match col.data_type() {
                            ArrowType::Utf8View => col.as_string_view().value(i).to_owned(),
                            ArrowType::LargeUtf8 => col.as_string::<i64>().value(i).to_owned(),
                            _ => col.as_string::<i32>().value(i).to_owned(),
                        }
                    }
                    ArrowType::Struct(_) => {
                        let s = col.as_struct();
                        let mut parts = Vec::new();
                        for k in 0..s.num_columns() {
                            let c = s.column(k);
                            let name = s.column_names()[k];
                            if c.is_null(i) {
                                parts.push(format!("\"{name}\":null"));
                            } else {
                                match c.data_type() {
                                    ArrowType::Utf8View => parts.push(format!(
                                        "\"{name}\":\"{}\"",
                                        c.as_string_view().value(i)
                                    )),
                                    ArrowType::Utf8 => parts.push(format!(
                                        "\"{name}\":\"{}\"",
                                        c.as_string::<i32>().value(i)
                                    )),
                                    ArrowType::LargeUtf8 => parts.push(format!(
                                        "\"{name}\":\"{}\"",
                                        c.as_string::<i64>().value(i)
                                    )),
                                    ArrowType::Float32 => parts.push(format!(
                                        "\"{name}\":{}",
                                        c.as_primitive::<arrow_array::types::Float32Type>()
                                            .value(i)
                                    )),
                                    ArrowType::Float64 => parts.push(format!(
                                        "\"{name}\":{}",
                                        c.as_primitive::<arrow_array::types::Float64Type>()
                                            .value(i)
                                    )),
                                    ArrowType::Int32 => parts.push(format!(
                                        "\"{name}\":{}",
                                        c.as_primitive::<arrow_array::types::Int32Type>().value(i)
                                    )),
                                    ArrowType::Int64 => parts.push(format!(
                                        "\"{name}\":{}",
                                        c.as_primitive::<arrow_array::types::Int64Type>().value(i)
                                    )),
                                    _ => parts.push(format!("\"{name}\":?")),
                                }
                            }
                        }
                        format!("{{{}}}", parts.join(","))
                    }
                    t => panic!("unrendered {t}"),
                });
            }
            out.push(row.join("\t"));
        }
    }
    out
}

/// Runs `sql`, expecting rows.
fn run(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    rows_of(&block_on(df.collect()).unwrap())
}

/// The output names of `sql`.
fn names(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Runs `sql`, expecting an analysis or execution error class.
fn class_of(s: &Session, sql: &str) -> ErrorClass {
    error_of(s, sql).error_class()
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
fn from_csv_to_csv_goldens() {
    let s = session();
    // `from_csv('1, 3.14', 'a INT, f FLOAT')`.
    assert_eq!(
        run(&s, "select from_csv('1, 3.14', 'a INT, f FLOAT')"),
        ["{\"a\":1,\"f\":3.14}"]
    );
    assert_eq!(
        names(&s, "select from_csv('1, 3.14', 'a INT, f FLOAT')"),
        ["from_csv(1, 3.14)"]
    );
    // `from_csv('1,abc', schema_of_csv('1,abc'))`.
    assert_eq!(
        run(&s, "select from_csv('1,abc', schema_of_csv('1,abc'))"),
        ["{\"_c0\":1,\"_c1\":\"abc\"}"]
    );
    // `to_csv(named_struct('a', 1, 'b', 2))`.
    assert_eq!(
        run(&s, "select to_csv(named_struct('a', 1, 'b', 2))"),
        ["1,2"]
    );
    assert_eq!(
        names(&s, "select to_csv(named_struct('a', 1, 'b', 2))"),
        ["to_csv(named_struct(a, 1, b, 2))"]
    );
    // Error cases from the golden.
    assert_eq!(
        class_of(&s, "select from_csv('1', 1)"),
        ErrorClass::InvalidSchemaNonStringLiteral
    );
    assert_eq!(
        class_of(&s, "select from_csv('1', 'a InvalidType')"),
        ErrorClass::ParseSyntaxError
    );
    assert_eq!(
        class_of(&s, "select from_csv('1', 'Array<int>')"),
        ErrorClass::InvalidSchemaNonStructType
    );
    assert_eq!(
        class_of(
            &s,
            "select from_csv('1', 'a INT', named_struct('mode', 'PERMISSIVE'))"
        ),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        class_of(&s, "select from_csv()"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    assert_eq!(
        class_of(
            &s,
            "select to_csv(named_struct('a', 1, 'b', 2), named_struct('mode', 'PERMISSIVE'))"
        ),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    // `parse-schema-string.sql` CSV statements.
    assert_eq!(
        run(&s, "select from_csv('1', 'create INT')"),
        ["{\"create\":1}"]
    );
    assert_eq!(
        run(&s, "select from_csv('1', 'cube INT')"),
        ["{\"cube\":1}"]
    );
}
