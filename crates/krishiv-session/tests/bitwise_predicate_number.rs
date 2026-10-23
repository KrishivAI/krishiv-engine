// SPDX-License-Identifier: Apache-2.0
//! Bitwise, predicate and number-formatting functions end to end: the
//! literal statements of Spark 4.2's `bitwise.sql`, `misc-functions.sql`
//! (minus the legacy-flag session), `null-handling.sql` (`equal_null`),
//! `charvarchar.sql` (`to_varchar`), `string-functions.sql`
//! (`to_number`) and `math.sql` (`conv`, `bin`) run with Spark's rows,
//! output names and error classes.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

/// The rows of a `SELECT`, rendered as Spark renders them (columns
/// joined by tabs).
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
                    ArrowType::Null => {
                        // A `void` column (e.g. `assert_true`) is all NULL.
                        "NULL".to_owned()
                    }
                    ArrowType::Int32 => col
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Int64 => col
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Int8 => col
                        .as_primitive::<arrow_array::types::Int8Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Boolean => col.as_boolean().value(i).to_string(),
                    ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
                        match col.data_type() {
                            ArrowType::Utf8View => col.as_string_view().value(i).to_owned(),
                            ArrowType::LargeUtf8 => col.as_string::<i64>().value(i).to_owned(),
                            _ => col.as_string::<i32>().value(i).to_owned(),
                        }
                    }
                    ArrowType::Decimal128(_, s) => {
                        let v = col
                            .as_primitive::<arrow_array::types::Decimal128Type>()
                            .value(i);
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
    match block_on(s.sql(sql)) {
        Ok(df) => match block_on(df.collect()) {
            Ok(b) => panic!("{sql} ran: {b:?}"),
            Err(e) => e.error_class(),
        },
        Err(e) => e.error_class(),
    }
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
fn bitwise_goldens() {
    let s = session();
    // `bit_count`: null, booleans, every width, negatives and edges.
    assert_eq!(run(&s, "SELECT bit_count(NULL)"), ["NULL"]);
    assert_eq!(
        run(&s, "SELECT bit_count(TRUE), bit_count(FALSE)"),
        ["1\t0"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT bit_count(CAST(1 AS TINYINT)), bit_count(2S), bit_count(3)"
        ),
        ["1\t1\t2"]
    );
    assert_eq!(run(&s, "SELECT bit_count(1L), bit_count(-1L)"), ["1\t64"]);
    assert_eq!(
        run(
            &s,
            "SELECT bit_count(9223372036854775807L), bit_count(-9223372036854775808L)"
        ),
        ["63\t1"]
    );
    assert_eq!(names(&s, "SELECT bit_count(1L)"), ["bit_count(1)"]);
    assert_eq!(
        class_of(&s, "SELECT bit_count('a')"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // `getbit`: right to left from zero, per the value's width.
    assert_eq!(
        run(
            &s,
            "SELECT getbit(11L, 3), getbit(11L, 2), getbit(11L, 1), getbit(11L, 0)"
        ),
        ["1\t0\t1\t1"]
    );
    assert_eq!(run(&s, "SELECT getbit(11L, 63)"), ["0"]);
    assert_eq!(
        class_of(&s, "SELECT getbit(11L, -1)"),
        ErrorClass::InvalidParameterValueBitPositionRange
    );
    assert_eq!(
        class_of(&s, "SELECT getbit(11L, 64)"),
        ErrorClass::InvalidParameterValueBitPositionRange
    );
    // Shifts and their precedence.
    assert_eq!(run(&s, "SELECT 20181117 >> 2"), ["5045279"]);
    assert_eq!(run(&s, "SELECT 20181117 << 2"), ["80724468"]);
    assert_eq!(run(&s, "SELECT 20181117 >>> 2"), ["5045279"]);
    assert_eq!(names(&s, "SELECT 20181117 >> 2"), ["(20181117 >> 2)"]);
    assert_eq!(run(&s, "SELECT 1 << 1 + 2 AS plus_over_shift"), ["8"]);
    assert_eq!(run(&s, "SELECT 2 >> 1 << 1 AS left_to_right"), ["2"]);
    assert_eq!(run(&s, "SELECT 1 & 2 >> 1 AS shift_over_ampersand"), ["1"]);
    // `&`, `|`, `^`, `~` as operators.
    assert_eq!(run(&s, "SELECT 3 & 5, 3 | 5, 3 ^ 5, ~0"), ["1\t7\t6\t-1"]);
    assert_eq!(names(&s, "SELECT 3 & 5"), ["(3 & 5)"]);
}

#[test]
fn predicate_goldens() {
    let s = session();
    // `misc-functions.sql`: `assert_true` passes void NULLs...
    assert_eq!(run(&s, "SELECT assert_true(TRUE)"), ["NULL"]);
    assert_eq!(
        names(&s, "SELECT assert_true(TRUE)"),
        ["assert_true(true, 'true' is not true!)"]
    );
    // ... and raises `USER_RAISED_EXCEPTION` otherwise.
    assert_eq!(
        class_of(&s, "SELECT assert_true(FALSE)"),
        ErrorClass::UserRaisedException
    );
    assert_eq!(
        error_of(&s, "SELECT assert_true(FALSE)").to_string(),
        "[USER_RAISED_EXCEPTION] 'false' is not true! SQLSTATE: P0001"
    );
    assert_eq!(
        class_of(&s, "SELECT assert_true(NULL)"),
        ErrorClass::UserRaisedException
    );
    assert_eq!(
        error_of(&s, "SELECT assert_true(FALSE, 'custom error message')").to_string(),
        "[USER_RAISED_EXCEPTION] custom error message SQLSTATE: P0001"
    );
    // `raise_error` always raises (even for NULL, as `"null"`).
    assert_eq!(
        error_of(&s, "SELECT raise_error('error message')").to_string(),
        "[USER_RAISED_EXCEPTION] error message SQLSTATE: P0001"
    );
    assert_eq!(
        error_of(&s, "SELECT raise_error(NULL)").to_string(),
        "[USER_RAISED_EXCEPTION] null SQLSTATE: P0001"
    );
    assert_eq!(
        error_of(&s, "SELECT raise_error(1)").to_string(),
        "[USER_RAISED_EXCEPTION] 1 SQLSTATE: P0001"
    );
    assert_eq!(
        class_of(&s, "SELECT raise_error()"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    // `null-handling.sql`: `equal_null` over a table with NULLs.
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, false),
        Field::new("b", ArrowType::Int32, true),
        Field::new("c", ArrowType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7])),
            Arc::new(Int32Array::from(vec![
                Some(0),
                Some(0),
                Some(1),
                Some(1),
                None,
                None,
                None,
            ])),
            Arc::new(Int32Array::from(vec![
                Some(0),
                Some(1),
                Some(0),
                Some(1),
                Some(0),
                Some(1),
                None,
            ])),
        ],
    )
    .unwrap();
    s.register_batches("t1", schema, vec![batch]).unwrap();
    let df = block_on(s.sql("SELECT equal_null(b, c), equal_null(c, b) FROM t1")).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let col = |n: usize| {
        batches
            .iter()
            .flat_map(|b| {
                b.column(n)
                    .as_boolean()
                    .iter()
                    .map(|v| v.unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        col(0),
        ["true", "false", "false", "true", "false", "false", "true"]
    );
    assert_eq!(
        col(1),
        ["true", "false", "false", "true", "false", "false", "true"]
    );
    // `isnan`: `NaN` is, NULL is not, never NULL itself.
    assert_eq!(
        run(
            &s,
            "SELECT isnan(CAST('NaN' AS DOUBLE)), isnan(1.0), isnan(NULL)"
        ),
        ["true\tfalse\tfalse"]
    );
}

#[test]
fn number_goldens() {
    let s = session();
    // `math.sql`: `conv` and `bin`.
    assert_eq!(run(&s, "SELECT conv('100', 2, 10)"), ["4"]);
    assert_eq!(run(&s, "SELECT conv(-10, 16, -10)"), ["-16"]);
    assert_eq!(
        run(&s, "SELECT conv('9223372036854775808', 10, 16)"),
        ["8000000000000000"]
    );
    assert_eq!(
        class_of(&s, "SELECT conv('92233720368547758070', 10, 16)"),
        ErrorClass::ArithmeticOverflow
    );
    assert_eq!(
        run(&s, "SELECT BIN(0), BIN(25), BIN(25L)"),
        ["0\t11001\t11001"]
    );
    // `charvarchar.sql`: `to_varchar` is `to_char`.
    assert_eq!(run(&s, "SELECT to_varchar(78.12, '$99.99')"), ["$78.12"]);
    assert_eq!(
        names(&s, "SELECT to_varchar(78.12, '$99.99')"),
        ["to_char(78.12, $99.99)"]
    );
    assert_eq!(run(&s, "SELECT to_varchar(111.11, '99.9')"), ["##.#"]);
    assert_eq!(
        run(&s, "SELECT to_varchar(12454.8, '99,999.9S')"),
        ["12,454.8+"]
    );
    // `string-functions.sql`: `to_number`.
    for (input, format, want) in [
        ("'454'", "'000'", "454"),
        ("'454.2'", "'000.0'", "454.2"),
        ("'12,454'", "'00,000'", "12454"),
        ("'$78.12'", "'$00.00'", "78.12"),
        ("'+454'", "'S000'", "454"),
        ("'-454'", "'S000'", "-454"),
        ("'12,454.8-'", "'00,000.9MI'", "-12454.8"),
        ("'00,454.8-'", "'00,000.9MI'", "-454.8"),
        ("'<00,454.8>'", "'00,000.9PR'", "-454.8"),
    ] {
        assert_eq!(
            run(&s, &format!("SELECT to_number({input}, {format})")),
            [want]
        );
    }
    // `hex`/`unhex` round-trip (inside `lpad`, as the goldens use them).
    assert_eq!(run(&s, "SELECT hex(unhex('aabb'))"), ["AABB"]);
    assert_eq!(run(&s, "SELECT hex('Spark SQL')"), ["537061726B2053514C"]);
    // `format_number` in both modes.
    assert_eq!(
        run(&s, "SELECT format_number(12332.123456, 4)"),
        ["12,332.1235"]
    );
    assert_eq!(
        run(
            &s,
            "SELECT format_number(12332.123456, '##################.###')"
        ),
        ["12332.123"]
    );
}
