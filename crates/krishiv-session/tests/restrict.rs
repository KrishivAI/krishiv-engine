// SPDX-License-Identifier: Apache-2.0
//! Aggregates restricted to the keys their join can match (the optimizer's
//! `restrict` pass), end to end: decorrelated scalar subqueries with NULL
//! keys, keys without inner rows (the count bug), duplicate outer keys and
//! two correlation columns, and `IN` over a grouped subquery, each against
//! the rows Spark 4.2.0 returns on the same data.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

/// `t(a int, b string, c bigint)`, `u(a int, d string)` and `w(k int, v
/// bigint)`: NULL keys on both sides, outer keys repeated, outer keys with
/// no inner row, and in `w` a group (`k = 9`) whose `sum` overflows.
fn session() -> Session {
    let s = Session::new().unwrap();
    let t = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("b", ArrowType::Utf8View, true),
        Field::new("c", ArrowType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&t),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(2),
                Some(3),
                None,
                Some(4),
                Some(2),
                Some(5),
            ])),
            Arc::new(StringViewArray::from(vec![
                "x", "x", "x", "x", "x", "y", "y", "x",
            ])),
            Arc::new(Int64Array::from(vec![
                Some(0),
                Some(5),
                Some(1),
                Some(0),
                Some(0),
                Some(100),
                Some(-1),
                None,
            ])),
        ],
    )
    .unwrap();
    s.register_batches("t", t, vec![batch]).unwrap();
    let u = Arc::new(Schema::new(vec![
        Field::new("a", ArrowType::Int32, true),
        Field::new("d", ArrowType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&u),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                None,
                None,
                Some(4),
                Some(4),
                Some(6),
                Some(7),
            ])),
            Arc::new(StringViewArray::from(vec![
                Some("x"),
                Some("x"),
                Some("y"),
                Some("x"),
                Some("x"),
                None,
                Some("y"),
                Some("y"),
                Some("z"),
                Some("x"),
            ])),
        ],
    )
    .unwrap();
    s.register_batches("u", u, vec![batch]).unwrap();
    // Many groups, so the restriction pays off by the estimates.
    let w = Arc::new(Schema::new(vec![
        Field::new("k", ArrowType::Int32, false),
        Field::new("v", ArrowType::Int64, false),
    ]));
    let mut k = vec![1, 9, 9];
    let mut v = vec![5, i64::MAX, 1];
    k.extend(100..200);
    v.extend(std::iter::repeat_n(1, 100));
    let batch = RecordBatch::try_new(
        Arc::clone(&w),
        vec![Arc::new(Int32Array::from(k)), Arc::new(Int64Array::from(v))],
    )
    .unwrap();
    s.register_batches("w", w, vec![batch]).unwrap();
    s
}

fn value(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        ArrowType::Int32 => a.as_primitive::<Int32Type>().value(row).to_string(),
        ArrowType::Int64 => a.as_primitive::<Int64Type>().value(row).to_string(),
        ArrowType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(row)),
        other => panic!("unexpected column type {other}"),
    }
}

/// The rows of `query`, each as its values joined by `,`, sorted.
fn rows(s: &Session, query: &str) -> Vec<String> {
    let df = block_on(s.sql(query)).unwrap();
    let mut out = Vec::new();
    for b in block_on(df.collect()).unwrap() {
        for r in 0..b.num_rows() {
            let row: Vec<String> = b.columns().iter().map(|c| value(c, r)).collect();
            out.push(row.join(","));
        }
    }
    out.sort();
    out
}

/// Whether `query`'s optimized plan holds more semi joins than its
/// analyzed plan (the restriction's).
fn restricted(s: &Session, query: &str) -> bool {
    let df = block_on(s.sql(query)).unwrap();
    let text = block_on(df.explain(ExplainMode::Extended)).unwrap();
    let section = |name: &str| {
        let start = text
            .find(name)
            .unwrap_or_else(|| panic!("{name} in {text}"));
        let rest = &text[start + name.len()..];
        rest[..rest.find("\n== ").unwrap_or(rest.len())].to_owned()
    };
    let semis = |plan: &str| plan.matches("LeftSemi").count();
    semis(&section("== Optimized Logical Plan ==")) > semis(&section("== Analyzed Logical Plan =="))
}

#[test]
fn restricted_aggregates_return_sparks_rows() {
    let s = session();
    // (query, Spark 4.2.0's rows, sorted as strings).
    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT a FROM t WHERE b = 'x' AND c > (SELECT avg(u.a) FROM u WHERE u.a = t.a)",
            &["1"],
        ),
        // Keys without inner rows count 0 (the count bug), NULL keys too.
        (
            "SELECT a, (SELECT count(*) FROM u WHERE u.a = t.a) AS n FROM t WHERE b = 'x'",
            &["1,3", "1,3", "2,1", "3,0", "5,0", "NULL,0"],
        ),
        // ... and read NULL from other aggregates.
        (
            "SELECT a, (SELECT avg(u.a) FROM u WHERE u.a = t.a) AS v FROM t WHERE b = 'x'",
            &["1,1.0", "1,1.0", "2,2.0", "3,NULL", "5,NULL", "NULL,NULL"],
        ),
        // Outer keys repeated by a join.
        (
            "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE t.b = 'x' AND t.c > (SELECT max(u2.a) FROM u u2 WHERE u2.a = t.a)",
            &["1", "1", "1"],
        ),
        // Two correlation columns.
        (
            "SELECT a FROM t WHERE b = 'x' AND c >= (SELECT count(*) FROM u WHERE u.a = t.a AND u.d = t.b)",
            &["1", "2", "3", "NULL"],
        ),
        // `IN` over a grouped subquery with HAVING.
        (
            "SELECT a FROM t WHERE b = 'x' AND a IN (SELECT a FROM u GROUP BY a HAVING count(*) > 1)",
            &["1", "1"],
        ),
    ];
    for (query, want) in cases {
        assert!(restricted(&s, query), "not restricted: {query}");
        assert_eq!(rows(&s, query), *want, "{query}");
    }
}

/// A group no outer row asks for is never formed, so its overflow is
/// never raised: Spark 4.2.0 fails this query with `ARITHMETIC_OVERFLOW`
/// (ANSI), Krishiv returns the rows of the groups the query reads. A
/// documented divergence, like a runtime filter's.
#[test]
fn unread_groups_raise_no_errors() {
    let s = session();
    let query = "SELECT a FROM t WHERE b = 'x' AND c < (SELECT sum(w.v) FROM w WHERE w.k = t.a)";
    assert!(restricted(&s, query));
    assert_eq!(rows(&s, query), ["1"]);
}
