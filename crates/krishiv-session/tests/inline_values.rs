// SPDX-License-Identifier: Apache-2.0
//! Inline `VALUES` tables end to end: the `inline-table.sql` statements of
//! the Spark 4.2.0 goldens through `Session::sql`, with Spark's schema and
//! rows (rows sorted, as Spark's suite sorts queries without `ORDER BY`) or
//! Spark's error class and message parameters, plus a temporary view over
//! `VALUES` read back by a join. Double-quoted golden strings are written
//! single-quoted, which parses to the same literals.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, TimestampMicrosecondType};
use arrow_array::{Date32Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

fn fails(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err(sql),
    }
}

/// A civil date from days since the epoch (Howard Hinnant's algorithm).
fn ymd(days: i32) -> (i32, u32, u32) {
    let z = i64::from(days) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).expect("day of month fits");
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).expect("month fits");
    let y = i32::try_from(if m <= 2 { y + 1 } else { y }).expect("year fits");
    (y, m, d)
}

/// Microseconds since the epoch as Spark prints a timestamp (UTC).
fn timestamp(micros: i64) -> String {
    let secs = micros.div_euclid(1_000_000);
    let frac = u32::try_from(micros.rem_euclid(1_000_000)).expect("microseconds fit");
    let (y, m, d) = ymd(i32::try_from(secs.div_euclid(86_400)).unwrap());
    let rest = u32::try_from(secs.rem_euclid(86_400)).expect("seconds of day fit");
    let mut out = format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        rest / 3_600,
        rest / 60 % 60,
        rest % 60
    );
    if frac != 0 {
        out.push('.');
        out.push_str(format!("{frac:06}").trim_end_matches('0'));
    }
    out
}

fn cell(batch: &RecordBatch, col: usize, row: usize) -> String {
    let c = batch.column(col);
    if c.is_null(row) {
        return "NULL".to_owned();
    }
    match c.data_type() {
        ArrowType::Null => "NULL".to_owned(),
        ArrowType::Int32 => c
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(row)
            .to_string(),
        ArrowType::Int64 => c
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(row)
            .to_string(),
        ArrowType::Float64 => {
            let v = c
                .as_primitive::<arrow_array::types::Float64Type>()
                .value(row);
            if v.is_finite() && v.fract() == 0.0 {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        ArrowType::Boolean => c.as_boolean().value(row).to_string(),
        ArrowType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        ArrowType::Utf8View => c.as_string_view().value(row).to_owned(),
        ArrowType::LargeUtf8 => c.as_string::<i64>().value(row).to_owned(),
        ArrowType::Date32 => {
            let days = c
                .as_primitive::<arrow_array::types::Date32Type>()
                .value(row);
            let _ = Date32Array::from(vec![days]);
            let (y, m, d) = ymd(days);
            format!("{y:04}-{m:02}-{d:02}")
        }
        ArrowType::Timestamp(_, _) => {
            timestamp(c.as_primitive::<TimestampMicrosecondType>().value(row))
        }
        ArrowType::Decimal128(..) => c.as_primitive::<Decimal128Type>().value_as_string(row),
        other => panic!("unexpected type {other}"),
    }
}

/// The arrow type as Spark names it.
fn spark_type(t: &ArrowType) -> String {
    match t {
        ArrowType::Null => "void".to_owned(),
        ArrowType::Boolean => "boolean".to_owned(),
        ArrowType::Int32 => "int".to_owned(),
        ArrowType::Int64 => "bigint".to_owned(),
        ArrowType::Float64 => "double".to_owned(),
        ArrowType::Utf8 | ArrowType::Utf8View | ArrowType::LargeUtf8 => "string".to_owned(),
        ArrowType::Date32 => "date".to_owned(),
        ArrowType::Timestamp(_, _) => "timestamp".to_owned(),
        ArrowType::Decimal128(p, s) => format!("decimal({p},{s})"),
        other => panic!("unexpected type {other}"),
    }
}

/// Schema names, Spark type names and rows as text, rows sorted.
fn frame(s: &Session, sql: &str) -> (Vec<String>, Vec<String>, Vec<Vec<String>>) {
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let types: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| spark_type(f.data_type()))
        .collect();
    let batches = block_on(df.collect()).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            rows.push((0..b.num_columns()).map(|c| cell(b, c, r)).collect());
        }
    }
    rows.sort();
    (names, types, rows)
}

fn check(s: &Session, sql: &str, names: &[&str], types: &[&str], rows: &[&[&str]]) {
    let (found_names, found_types, mut found_rows) = frame(s, sql);
    assert_eq!(found_names, names, "{sql}");
    assert_eq!(found_types, types, "{sql}");
    let mut expected: Vec<Vec<String>> = rows
        .iter()
        .map(|r| r.iter().map(|c| (*c).to_owned()).collect())
        .collect();
    expected.sort();
    found_rows.sort();
    assert_eq!(found_rows, expected, "{sql}");
}

#[test]
fn inline_tables_return_spark_rows_and_schemas() {
    let s = session();
    // The clock and the typed literals read the session zone: run in UTC,
    // as the goldens do.
    block_on(s.sql("SET TIME ZONE 'UTC'")).unwrap();
    // Single rows, with and without a table and column alias.
    check(
        &s,
        "select * from values ('one', 1)",
        &["col1", "col2"],
        &["string", "int"],
        &[&["one", "1"]],
    );
    check(
        &s,
        "select * from values ('one', 1) as data",
        &["col1", "col2"],
        &["string", "int"],
        &[&["one", "1"]],
    );
    check(
        &s,
        "select * from values ('one', 1) as data(a, b)",
        &["a", "b"],
        &["string", "int"],
        &[&["one", "1"]],
    );
    // A single-column row written without parentheses is one column.
    check(
        &s,
        "select * from values 1, 2, 3 as data(a)",
        &["a"],
        &["int"],
        &[&["1"], &["2"], &["3"]],
    );
    // Three rows: a NULL makes its column nullable.
    check(
        &s,
        "select * from values ('one', 1), ('two', 2), ('three', null) as data(a, b)",
        &["a", "b"],
        &["string", "int"],
        &[&["one", "1"], &["three", "NULL"], &["two", "2"]],
    );
    // An all-NULL column is VOID.
    check(
        &s,
        "select * from values ('one', null), ('two', null) as data(a, b)",
        &["a", "b"],
        &["string", "void"],
        &[&["one", "NULL"], &["two", "NULL"]],
    );
    // `INT` and `BIGINT` meet at `BIGINT`.
    check(
        &s,
        "select * from values ('one', 1), ('two', 2L) as data(a, b)",
        &["a", "b"],
        &["string", "bigint"],
        &[&["one", "1"], &["two", "2"]],
    );
    // Foldable expressions evaluate.
    check(
        &s,
        "select * from values ('one', 1 + 0), ('two', 1 + 3L) as data(a, b)",
        &["a", "b"],
        &["string", "bigint"],
        &[&["one", "1"], &["two", "4"]],
    );
    // Row aliases are ignored.
    check(
        &s,
        "select * from values ('one', 1 as one) as data(a, b)",
        &["a", "b"],
        &["string", "int"],
        &[&["one", "1"]],
    );
    check(
        &s,
        "select a from values ('one', current_timestamp) as data(a, b)",
        &["a"],
        &["string"],
        &[&["one"]],
    );
    // A decimal and a double meet at a double.
    check(
        &s,
        "select * from values ('one', 2.0), ('two', 3.0D) as data(a, b)",
        &["a", "b"],
        &["string", "double"],
        &[&["one", "2.0"], &["two", "3.0"]],
    );
    // Fallible expressions evaluate to NULL.
    check(
        &s,
        "select * from values (try_add(5, 0))",
        &["col1"],
        &["int"],
        &[&["5"]],
    );
    check(
        &s,
        "select * from values (try_divide(5, 0))",
        &["col1"],
        &["double"],
        &[&["NULL"]],
    );
    check(
        &s,
        "select * from values (10 + try_divide(5, 0))",
        &["col1"],
        &["double"],
        &[&["NULL"]],
    );
    // The clock reads once per query, so every row agrees.
    check(
        &s,
        "select count(distinct ct) from values now(), now(), now() as data(ct)",
        &["count(DISTINCT ct)"],
        &["bigint"],
        &[&["1"]],
    );
    check(
        &s,
        "select count(distinct ct) from values current_timestamp(), current_timestamp() as data(ct)",
        &["count(DISTINCT ct)"],
        &["bigint"],
        &[&["1"]],
    );
    check(
        &s,
        "select a from (values (1 as id, current_timestamp() as ts), (2 as id, current_timestamp() as ts)) as t(a, b)",
        &["a"],
        &["int"],
        &[&["1"], &["2"]],
    );
    check(
        &s,
        "select * from values (1 as id, 'a' as name), (2 as id, 'b' as name) as t(a, b)",
        &["a", "b"],
        &["int", "string"],
        &[&["1", "a"], &["2", "b"]],
    );
    check(
        &s,
        "select * from values (1 as a, 2 as b)",
        &["col1", "col2"],
        &["int", "int"],
        &[&["1", "2"]],
    );
}

#[test]
fn inline_tables_support_typed_literals() {
    let s = session();
    block_on(s.sql("SET TIME ZONE 'UTC'")).unwrap();
    check(
        &s,
        "select * from values (TIMESTAMP '1991-12-06 00:00:00') as data(a)",
        &["a"],
        &["timestamp"],
        &[&["1991-12-06 00:00:00"]],
    );
}

#[test]
fn inline_tables_report_spark_errors() {
    let s = session();
    // A nondeterministic call cannot be evaluated.
    let e = fails(
        &s,
        "select * from values ('one', rand(5)), ('two', 3.0D) as data(a, b)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidInlineTableCannotEvaluateExpressionInInlineTable
    );
    assert_eq!(e.sql_state(), Some("42000"));
    assert_eq!(e.message_parameters(), [("expr", "\"rand(5)\"".to_owned())]);
    // Rows of different widths fail while parsing.
    let e = fails(
        &s,
        "select * from values ('one', 2.0), ('two') as data(a, b)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidInlineTableNumColumnsMismatch
    );
    assert_eq!(e.sql_state(), Some("42000"));
    assert_eq!(
        e.message_parameters(),
        [
            ("expectedNumCols", "2".to_owned()),
            ("actualNumCols", "1".to_owned()),
            ("rowIndex", "1".to_owned()),
        ]
    );
    // Columns of incompatible types name the column, quoted.
    let e = fails(
        &s,
        "select * from values ('one', 1), ('two', true) as data(a, b)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidInlineTableIncompatibleTypesInInlineTable
    );
    assert_eq!(e.sql_state(), Some("42000"));
    assert_eq!(e.message_parameters(), [("colName", "`b`".to_owned())]);
    // An alias list longer than the rows fails while parsing.
    let e = fails(&s, "select * from values ('one'), ('two') as data(a, b)");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidInlineTableNumColumnsMismatch
    );
    assert_eq!(
        e.message_parameters(),
        [
            ("expectedNumCols", "2".to_owned()),
            ("actualNumCols", "1".to_owned()),
            ("rowIndex", "0".to_owned()),
        ]
    );
    // An unknown routine is `UNRESOLVED_ROUTINE`, as elsewhere.
    let e = fails(
        &s,
        "select * from values ('one', random_not_exist_func(1)), ('two', 2) as data(a, b)",
    );
    assert_eq!(e.error_class(), ErrorClass::UnresolvedRoutine);
    assert_eq!(e.sql_state(), Some("42883"));
    assert_eq!(
        e.message_parameters(),
        [
            ("routineName", "`random_not_exist_func`".to_owned()),
            (
                "searchPath",
                "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]".to_owned()
            ),
        ]
    );
    // An unknown name is an unresolved column (as Spark), not an
    // expression that cannot be evaluated.
    for sql in [
        "select * from values (x)",
        "select * from values (1), (x + 1)",
        "select * from values (t.x)",
    ] {
        let e = fails(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::UnresolvedColumnWithoutSuggestion,
            "{sql}"
        );
        assert_eq!(e.sql_state(), Some("42703"), "{sql}");
    }
    // An aggregate cannot be evaluated.
    let e = fails(
        &s,
        "select * from values ('one', count(1)), ('two', 2) as data(a, b)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidInlineTableCannotEvaluateExpressionInInlineTable
    );
    assert_eq!(
        e.message_parameters(),
        [("expr", "\"count(1)\"".to_owned())]
    );
}

#[test]
fn temporary_views_over_values_join() {
    let s = session();
    block_on(s.sql("CREATE TEMPORARY VIEW v AS VALUES ('one', 1), ('two', 2)")).unwrap();
    check(
        &s,
        "SELECT * FROM v ORDER BY col2",
        &["col1", "col2"],
        &["string", "int"],
        &[&["one", "1"], &["two", "2"]],
    );
    block_on(s.sql("CREATE TEMPORARY VIEW w AS SELECT * FROM VALUES ('one', 10) AS t(a, b)"))
        .unwrap();
    let (names, _, rows) = frame(
        &s,
        "SELECT v.col1, v.col2, w.b FROM v JOIN w ON v.col1 = w.a ORDER BY v.col2",
    );
    assert_eq!(names, ["col1", "col2", "b"]);
    assert_eq!(rows, [vec!["one", "1", "10"]]);
}

#[test]
fn values_plans_hold_no_inline_table() {
    let s = session();
    let plan =
        block_on(s.sql("EXPLAIN EXTENDED SELECT * FROM VALUES ('one', 1), ('two', 2L) AS t(a, b)"))
            .unwrap();
    let batches = block_on(plan.collect()).unwrap();
    let text: String = batches
        .iter()
        .flat_map(|b| (0..b.num_rows()).map(|r| b.column(0).as_string_view().value(r).to_owned()))
        .collect();
    assert!(text.contains("== Analyzed Logical Plan =="), "{text}");
    assert!(text.contains("== Optimized Logical Plan =="), "{text}");
    assert!(text.contains("== Physical Plan =="), "{text}");
    // The parsed plan holds the unresolved node; every later section
    // holds only resolved nodes.
    for section in [
        "== Analyzed Logical Plan ==",
        "== Optimized Logical Plan ==",
    ] {
        let body = text
            .split(section)
            .nth(1)
            .unwrap()
            .split("== ")
            .next()
            .unwrap();
        assert!(!body.contains("InlineTable"), "{body}");
    }
    // The analyzed plan is a union of row projects, and the physical plan
    // runs the existing operators.
    let analyzed = text
        .split("== Analyzed Logical Plan ==")
        .nth(1)
        .unwrap()
        .split("== Optimized Logical Plan ==")
        .next()
        .unwrap();
    assert!(analyzed.contains("Union"), "{analyzed}");
    assert!(analyzed.contains("Project"), "{analyzed}");
    assert!(analyzed.contains("OneRowRelation"), "{analyzed}");
    let physical = text.split("== Physical Plan ==").nth(1).unwrap();
    assert!(!physical.contains("InlineTable"), "{physical}");
}

/// The views and queries of the `using-join.sql` and `group-by-all.sql`
/// goldens that need nothing else this engine lacks.
#[test]
fn golden_views_over_values_read_back() {
    let s = session();
    block_on(s.sql(
        "create temporary view nt1 as select * from values \
         ('one', 1), ('two', 2), ('three', 3) as nt1(k, v1)",
    ))
    .unwrap();
    block_on(s.sql(
        "create temporary view nt2 as select * from values \
         ('one', 1), ('two', 22), ('one', 5), ('four', 4) as nt2(k, v2)",
    ))
    .unwrap();
    check(
        &s,
        "SELECT * FROM nt1 left outer join nt2 using (k)",
        &["k", "v1", "v2"],
        &["string", "int", "int"],
        &[
            &["one", "1", "1"],
            &["one", "1", "5"],
            &["three", "3", "NULL"],
            &["two", "2", "22"],
        ],
    );
    check(
        &s,
        "SELECT k FROM nt1 left outer join nt2 using (k)",
        &["k"],
        &["string"],
        &[&["one"], &["one"], &["three"], &["two"]],
    );
    // A column named `all` is a column, so the grouping has three groups.
    check(
        &s,
        "SELECT count(1) FROM VALUES(1), (2), (3) AS T(all) GROUP BY all",
        &["count(1)"],
        &["bigint"],
        &[&["1"], &["1"], &["1"]],
    );
}
