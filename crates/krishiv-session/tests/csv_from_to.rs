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

/// The first column of the first row of `sql` as text, or the error class.
fn one(sql: &str) -> String {
    let s = session();
    // Spark's reference run used UTC.
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    match block_on(s.sql(sql)).and_then(|df| block_on(df.collect())) {
        Ok(batches) => rows_of(&batches).into_iter().next().unwrap_or_default(),
        Err(e) => format!("ERR {}", e.error_class().name()),
    }
}

/// Every scalar type through `from_csv`, malformed-record modes and empty
/// input; the expected values are what Spark 4.2.0 returns for the same
/// statements (checked against a live 4.2.0 server). Options travel as
/// `map(...)`, which the engine cannot evaluate over string views yet, so
/// they are covered by the function-level tests.
#[test]
fn from_csv_reads_every_scalar_type_and_option() {
    let json = |csv: &str, schema: &str, options: &str| {
        let opt = if options.is_empty() {
            String::new()
        } else {
            format!(", map({options})")
        };
        one(&format!(
            "SELECT to_json(from_csv('{csv}', '{schema}'{opt}))"
        ))
    };
    assert_eq!(
        json(
            "1,2,3,4,5.5,6.5,7.25,2020-01-02,2020-01-02 03:04:05,2020-01-02 03:04:05,abc,true",
            "b TINYINT, s SMALLINT, i INT, l BIGINT, f FLOAT, d DOUBLE, dec DECIMAL(5,2), \
             dt DATE, ts TIMESTAMP, ntz TIMESTAMP_NTZ, str STRING, bo BOOLEAN",
            "",
        ),
        r#"{"b":1,"s":2,"i":3,"l":4,"f":5.5,"d":6.5,"dec":7.25,"dt":"2020-01-02","ts":"2020-01-02T03:04:05.000Z","ntz":"2020-01-02T03:04:05.000","str":"abc","bo":true}"#
    );
    // Out-of-range integers, decimals, signed zero, empty input and short rows.
    assert_eq!(
        json("9999999999,300,70000", "i INT, b TINYINT, s SMALLINT", ""),
        "{}"
    );
    assert_eq!(
        json("1.5,-0,1e3", "a DECIMAL(4,1), b DOUBLE, c FLOAT", ""),
        r#"{"a":1.5,"b":-0.0,"c":1000.0}"#
    );
    assert_eq!(json("", "a STRING, b INT", ""), "{}");
    assert_eq!(json("x", "a STRING, b INT", ""), r#"{"a":"x"}"#);
}

/// Every scalar type through `to_csv` (values from Spark 4.2.0).
#[test]
fn to_csv_writes_every_scalar_type_and_option() {
    assert_eq!(
        one(
            "SELECT to_csv(named_struct('b', CAST(1 AS TINYINT), 's', CAST(2 AS SMALLINT), \
             'l', 4L, 'f', CAST(1.5 AS FLOAT), 'd', 2.5D, 'dec', CAST(7.25 AS DECIMAL(5,2)), \
             'dt', DATE'2020-01-02', 'ts', TIMESTAMP'2020-01-02 03:04:05', \
             'ntz', TIMESTAMP_NTZ'2020-01-02 03:04:05', 'bin', X'6162', 'bo', false))"
        ),
        "1,2,4,1.5,2.5,7.25,2020-01-02,2020-01-02T03:04:05.000Z,2020-01-02T03:04:05.000,[61 62],false"
    );
}

/// Schema arguments that are not a struct schema, against Spark 4.2.0: a
/// bare complex type is incomplete, a complete non-struct type and a
/// non-literal schema have their own `INVALID_SCHEMA` classes.
#[test]
fn from_csv_schema_errors_match_spark() {
    for (sql, class) in [
        (
            "SELECT from_csv('1', 'array')",
            "INCOMPLETE_TYPE_DEFINITION.ARRAY",
        ),
        (
            "SELECT from_csv('1', 'map')",
            "INCOMPLETE_TYPE_DEFINITION.MAP",
        ),
        (
            "SELECT from_csv('1', 'array<int>')",
            "INVALID_SCHEMA.NON_STRUCT_TYPE",
        ),
        (
            "SELECT from_csv('1', 'int')",
            "INVALID_SCHEMA.NON_STRUCT_TYPE",
        ),
        (
            "SELECT from_csv('1', 1)",
            "INVALID_SCHEMA.NON_STRING_LITERAL",
        ),
        (
            "SELECT from_csv('1', c) FROM (SELECT 'a INT' AS c)",
            "INVALID_SCHEMA.NON_STRING_LITERAL",
        ),
    ] {
        assert_eq!(one(sql), format!("ERR {class}"), "{sql}");
    }
}
