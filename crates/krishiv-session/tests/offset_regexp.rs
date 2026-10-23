// SPDX-License-Identifier: Apache-2.0
//! `OFFSET` (SQL and `DataFrame::offset`) and `regexp_replace` end to end.
//! Expected values come from the Spark 4.2.0 goldens `postgreSQL/limit.sql`,
//! `subquery/subquery-offset.sql` and `regexp-functions.sql`, and from
//! Spark's documented `LIMIT n OFFSET m` order: skip `m` rows, then keep `n`.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;
use krishiv_session::functions::col;

/// `t(id)`: 1 to 10, shuffled so only an `ORDER BY` fixes the order.
fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![
            4, 9, 1, 7, 10, 2, 8, 3, 6, 5,
        ]))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("s", ArrowType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![
            Some("https://www.example.com/a/b"),
            Some("http://foo.org/x"),
            None,
            Some("plain"),
        ]))],
    )
    .unwrap();
    s.register_batches("u", schema, vec![batch]).unwrap();
    s
}

fn ints(s: &Session, sql: &str) -> Vec<i32> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect())
        .unwrap()
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect()
}

fn strings(s: &Session, sql: &str) -> Vec<Option<String>> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect())
        .unwrap()
        .iter()
        .flat_map(|b| {
            let c = b.column(0);
            let owned = |v: Option<&str>| v.map(str::to_owned);
            match c.data_type() {
                ArrowType::Utf8View => c.as_string_view().iter().map(owned).collect::<Vec<_>>(),
                ArrowType::LargeUtf8 => c.as_string::<i64>().iter().map(owned).collect(),
                _ => c.as_string::<i32>().iter().map(owned).collect(),
            }
        })
        .collect()
}

fn fails(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err(sql),
    }
}

#[test]
fn offset_skips_then_limit_keeps() {
    let s = session();
    assert_eq!(
        ints(&s, "SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 2"),
        [3, 4, 5]
    );
    assert_eq!(ints(&s, "SELECT id FROM t ORDER BY id OFFSET 8"), [9, 10]);
    assert_eq!(
        ints(&s, "SELECT id FROM t ORDER BY id DESC LIMIT 20 OFFSET 7"),
        [3, 2, 1]
    );
    // Offset 0 is a no-op; a limit of 0 returns nothing whatever the offset.
    assert_eq!(ints(&s, "SELECT id FROM t ORDER BY id OFFSET 0").len(), 10);
    assert!(ints(&s, "SELECT id FROM t ORDER BY id LIMIT 0 OFFSET 3").is_empty());
    // At or past the row count: no rows, no error.
    assert!(ints(&s, "SELECT id FROM t ORDER BY id LIMIT 8 OFFSET 10").is_empty());
    assert!(ints(&s, "SELECT id FROM t ORDER BY id LIMIT 8 OFFSET 99").is_empty());
    assert!(ints(&s, "SELECT id FROM t OFFSET 99").is_empty());
    // Without an ORDER BY the rows are unspecified, the count is not.
    assert_eq!(ints(&s, "SELECT id FROM t LIMIT 4 OFFSET 3").len(), 4);
    assert_eq!(ints(&s, "SELECT id FROM t OFFSET 3").len(), 7);
    // `LIMIT ALL` keeps the rest.
    assert_eq!(
        ints(&s, "SELECT id FROM t ORDER BY id LIMIT ALL OFFSET 7"),
        [8, 9, 10]
    );
}

#[test]
fn offset_in_subqueries_and_set_operations() {
    let s = session();
    assert_eq!(
        ints(
            &s,
            "SELECT id FROM (SELECT id FROM t ORDER BY id LIMIT 4 OFFSET 1) WHERE id > 2 ORDER BY id"
        ),
        [3, 4, 5]
    );
    let mut ids = ints(
        &s,
        "(SELECT id FROM t ORDER BY id OFFSET 8) UNION ALL (SELECT id FROM t ORDER BY id LIMIT 1)",
    );
    ids.sort_unstable();
    assert_eq!(ids, [1, 9, 10]);
    // After the union: the offset applies to the union's rows.
    assert_eq!(
        ints(
            &s,
            "SELECT id FROM t UNION ALL SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 17"
        ),
        [9, 10, 10]
    );
    assert_eq!(
        ints(
            &s,
            "SELECT id FROM t WHERE id IN (SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 2) ORDER BY id"
        ),
        [3, 4, 5]
    );
    assert_eq!(
        ints(
            &s,
            "WITH c AS (SELECT id FROM t ORDER BY id OFFSET 5) SELECT id FROM c ORDER BY id"
        ),
        [6, 7, 8, 9, 10]
    );
}

#[test]
fn offset_in_the_dataframe_api() {
    let s = session();
    let df = s.table("t").order_by([col("id")]).offset(2).limit(3);
    let ids: Vec<i32> = block_on(df.collect())
        .unwrap()
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(ids, [3, 4, 5]);
    assert_eq!(block_on(s.table("t").offset(7).count()).unwrap(), 3);
    assert_eq!(block_on(s.table("t").offset(10).count()).unwrap(), 0);
}

#[test]
fn offset_must_be_a_non_negative_constant_integer() {
    let s = session();
    let e = fails(&s, "SELECT id FROM t OFFSET -1");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLimitLikeExpressionIsNegative
    );
    assert!(
        e.message()
            .contains("The offset expression must be equal to or greater than 0, but got -1."),
        "{}",
        e.message()
    );
    let e = fails(&s, "SELECT id FROM t LIMIT -1");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLimitLikeExpressionIsNegative
    );
    // A column resolves, and is then not a constant (Spark 4.2.0).
    let e = fails(&s, "SELECT id FROM t OFFSET id");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLimitLikeExpressionIsUnfoldable
    );
    // Not an INT.
    let e = fails(&s, "SELECT id FROM t OFFSET 'a'");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLimitLikeExpressionDataType
    );
    let e = fails(&s, "SELECT id FROM t OFFSET CAST(NULL AS INT)");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidLimitLikeExpressionIsNull
    );
}

#[test]
fn regexp_replace_in_sql() {
    let s = session();
    let one = |sql: &str| strings(&s, sql).remove(0);
    let healthy = "SELECT regexp_replace('healthy, wealthy, and wise', '\\\\w+thy', 'something'";
    assert_eq!(
        one(&format!("{healthy})")).as_deref(),
        Some("something, something, and wise")
    );
    assert_eq!(
        one(&format!("{healthy}, 2)")).as_deref(),
        Some("hsomething, something, and wise")
    );
    assert_eq!(one(&format!("{healthy}, NULL)")), None);
    assert_eq!(
        one("SELECT regexp_replace('2024-02-29', '(\\\\d+)-(\\\\d+)-(\\\\d+)', '$3/$2/$1')")
            .as_deref(),
        Some("29/02/2024")
    );
    // The ClickBench referrer domain.
    assert_eq!(
        strings(
            &s,
            "SELECT regexp_replace(s, '^https?://(?:www\\\\.)?([^/]+)/.*$', '$1') FROM u"
        ),
        [
            Some("example.com".to_owned()),
            Some("foo.org".to_owned()),
            None,
            Some("plain".to_owned())
        ]
    );
    let e = fails(&s, "SELECT regexp_replace('a', '(', 'x')");
    assert_eq!(e.error_class(), ErrorClass::InvalidParameterValuePattern);
    let e = fails(&s, "SELECT regexp_replace('abc', 'b', 'x', 0)");
    assert_eq!(e.error_class(), ErrorClass::DatatypeMismatchValueOutOfRange);
}

/// Spark 4.2.0 checks `position` while typing the call: it must be foldable
/// and positive whatever the rows hold (NULL is allowed).
#[test]
fn regexp_replace_position_is_checked_before_any_row() {
    let s = session();
    let out_of_range = ErrorClass::DatatypeMismatchValueOutOfRange;
    for sql in [
        "SELECT regexp_replace(CAST(NULL AS STRING), 'a', 'b', 0)",
        "SELECT regexp_replace('', 'a', 'b', 0)",
        "SELECT regexp_replace('a', 'a', 'b', -1)",
    ] {
        assert_eq!(fails(&s, sql).error_class(), out_of_range, "{sql}");
    }
    for sql in [
        "SELECT regexp_replace('a', 'a', 'b', id) FROM t",
        "SELECT regexp_replace('a', 'a', 'b', id + 1) FROM t",
    ] {
        assert_eq!(
            fails(&s, sql).error_class(),
            ErrorClass::DatatypeMismatchNonFoldableInput,
            "{sql}"
        );
    }
    let one = |sql: &str| strings(&s, sql).remove(0);
    assert_eq!(
        one("SELECT regexp_replace('a', 'a', 'b', CAST(NULL AS INT))"),
        None
    );
    assert_eq!(
        one("SELECT regexp_replace(CAST(NULL AS STRING), 'a', 'b', CAST(NULL AS INT))"),
        None
    );
    // Known divergence: Spark folds `1 + 1`; the analyzer only folds
    // literals and casts of them (as for the scale of `round`).
    assert_eq!(
        fails(&s, "SELECT regexp_replace('a', 'a', 'b', 1 + 1)").error_class(),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    assert_eq!(
        one("SELECT regexp_replace('abc', 'b', 'x', CAST(2 AS BIGINT))").as_deref(),
        Some("axc")
    );
}

/// Results and error classes of Spark 4.2.0 (local session, same data) for
/// the edge cases of `OFFSET`.
#[test]
fn offset_edge_cases_match_spark() {
    let s = session();
    let rows = [
        ("ORDER BY id LIMIT 2 OFFSET 1", vec![2, 3]),
        ("ORDER BY id OFFSET 9", vec![10]),
        ("ORDER BY id OFFSET 10", vec![]),
        ("ORDER BY id LIMIT 10 OFFSET 100", vec![]),
        ("ORDER BY id OFFSET 1 + 1", (3..=10).collect()),
        ("ORDER BY id OFFSET 2147483647", vec![]),
        ("ORDER BY id LIMIT ALL OFFSET 8", vec![9, 10]),
    ];
    for (tail, want) in rows {
        assert_eq!(
            ints(&s, &format!("SELECT id FROM t {tail}")),
            want,
            "{tail}"
        );
    }
    let whole = [
        (
            "SELECT * FROM (SELECT id FROM t ORDER BY id OFFSET 1) WHERE id > 8 ORDER BY id",
            vec![9, 10],
        ),
        (
            "SELECT * FROM (SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 1) ORDER BY id DESC LIMIT 1 OFFSET 1",
            vec![3],
        ),
        (
            "SELECT * FROM (SELECT id FROM t ORDER BY id OFFSET 1) ORDER BY id OFFSET 1",
            (3..=10).collect(),
        ),
        (
            "SELECT (SELECT id FROM t ORDER BY id LIMIT 1 OFFSET 2)",
            vec![3],
        ),
        (
            "SELECT id FROM t UNION ALL SELECT id FROM t ORDER BY id OFFSET 18",
            vec![10, 10],
        ),
        (
            "(SELECT id FROM t ORDER BY id LIMIT 3) UNION ALL (SELECT id FROM t ORDER BY id OFFSET 8) ORDER BY id",
            vec![1, 2, 3, 9, 10],
        ),
        (
            "SELECT id % 2 AS g FROM t GROUP BY id % 2 ORDER BY g OFFSET 1",
            vec![1],
        ),
        (
            "SELECT id FROM t WHERE id IN (SELECT id FROM t ORDER BY id LIMIT 3 OFFSET 2) ORDER BY id",
            vec![3, 4, 5],
        ),
    ];
    for (sql, want) in whole {
        assert_eq!(ints(&s, sql), want, "{sql}");
    }
    let unfoldable = ErrorClass::InvalidLimitLikeExpressionIsUnfoldable;
    let data_type = ErrorClass::InvalidLimitLikeExpressionDataType;
    let errors = [
        (
            "OFFSET -1",
            ErrorClass::InvalidLimitLikeExpressionIsNegative,
        ),
        (
            "LIMIT 2 OFFSET -1",
            ErrorClass::InvalidLimitLikeExpressionIsNegative,
        ),
        ("OFFSET id", unfoldable),
        ("OFFSET (SELECT 1)", unfoldable),
        ("OFFSET 1.5", data_type),
        ("OFFSET 'a'", data_type),
        ("OFFSET CAST(1 AS BIGINT)", data_type),
        ("OFFSET 2147483648", data_type),
        ("OFFSET NULL", data_type),
        (
            "OFFSET CAST(NULL AS INT)",
            ErrorClass::InvalidLimitLikeExpressionIsNull,
        ),
        (
            "LIMIT 2147483647 OFFSET 2147483647",
            ErrorClass::SumOfLimitAndOffsetExceedsMaxInt,
        ),
        ("OFFSET 1 LIMIT 2", ErrorClass::ParseSyntaxError),
        ("LIMIT 3 OFFSET 1 OFFSET 1", ErrorClass::ParseSyntaxError),
        ("OFFSET 1 ROWS", ErrorClass::ParseSyntaxError),
        ("FETCH FIRST 2 ROWS ONLY", ErrorClass::ParseSyntaxError),
    ];
    for (tail, class) in errors {
        let e = fails(&s, &format!("SELECT id FROM t ORDER BY id {tail}"));
        assert_eq!(e.error_class(), class, "{tail}: {}", e.message());
    }
}
