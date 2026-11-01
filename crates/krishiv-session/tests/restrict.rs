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
        ArrowType::Utf8View => a.as_string_view().value(row).to_owned(),
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

/// A correlated scalar subquery without an aggregate is a single-match
/// join: NULL without a match, the row with one, and
/// `SCALAR_SUBQUERY_TOO_MANY_ROWS` with several (Spark's `Single` join).
#[test]
fn a_scalar_subquery_without_an_aggregate_matches_at_most_one_row() {
    let s = session();
    let query = "SELECT a, (SELECT d FROM u WHERE u.a = t.a AND u.a > 1) FROM t WHERE b = 'x'";
    assert_eq!(
        rows(&s, query),
        ["1,NULL", "1,NULL", "2,x", "3,NULL", "5,NULL", "NULL,NULL"]
    );
    let many = "SELECT a, (SELECT d FROM u WHERE u.a = t.a) FROM t WHERE b = 'x'";
    let err = block_on(block_on(s.sql(many)).unwrap().collect()).unwrap_err();
    assert!(
        err.to_string().contains("SCALAR_SUBQUERY_TOO_MANY_ROWS"),
        "{err}"
    );
}

/// `IN (... WHERE g = o ORDER BY y LIMIT 1)` ranks by the subquery's order
/// even when the sort column is not selected, and partitions only by an
/// equality whose outer side reads no inner column.
#[test]
fn a_correlated_limit_keeps_its_order_and_partitions_by_clean_keys() {
    let s = correlated_session();
    let desc = "SELECT id FROM lt WHERE k IN (SELECT lc.k FROM lc WHERE lc.g = lt.id ORDER BY y DESC LIMIT 1)";
    assert_eq!(rows(&s, desc), ["4"]);
    let asc = "SELECT id FROM lt WHERE k IN (SELECT lc.k FROM lc WHERE lc.g = lt.id ORDER BY y ASC LIMIT 1)";
    assert_eq!(rows(&s, asc), ["1", "4"]);
    // An outer side reading inner columns is refused, never partitioned on it.
    let mixed = "SELECT id FROM lt WHERE k IN (SELECT lc.k FROM lc WHERE lc.g + lc.y = lt.id + lc.y ORDER BY lc.k DESC LIMIT 1)";
    assert!(failure(&s, mixed).contains("LIMIT of this form"));
    // A sort below a derived table still decides the ranks.
    let derived = "SELECT id FROM lt WHERE k IN (SELECT k FROM (SELECT lc.k FROM lc WHERE lc.g = lt.id ORDER BY y DESC) q LIMIT 1)";
    assert_eq!(rows(&s, derived), ["4"]);
    // A filter between them would have to rank after filtering: refused.
    let filtered = "SELECT id FROM lt WHERE k IN (SELECT k FROM (SELECT lc.k FROM lc WHERE lc.g = lt.id ORDER BY y DESC) q WHERE k < 9 LIMIT 1)";
    assert!(failure(&s, filtered).contains("LIMIT of this form"));
    // And a key the window and the join would evaluate apart.
    let random = "SELECT id FROM lt WHERE k IN (SELECT lc.k FROM lc WHERE CAST(rand(1) * 2 AS INT) = lt.id ORDER BY y LIMIT 1)";
    assert!(failure(&s, random).contains("LIMIT of this form"));
}

/// A correlated column that is a grouping key but not selected still reaches
/// the aggregate's output, so the join above can read it.
#[test]
fn a_correlated_grouping_key_need_not_be_selected() {
    let s = correlated_session();
    let value =
        "SELECT id FROM lt WHERE k IN (SELECT min(lc.k) FROM lc WHERE lc.g = lt.id GROUP BY lc.g)";
    assert_eq!(rows(&s, value), ["1", "4"]);
    let having = "SELECT id FROM lt WHERE EXISTS (SELECT 1 FROM lc WHERE lc.g = lt.id GROUP BY lc.g HAVING count(*) > 1)";
    assert_eq!(rows(&s, having), ["1", "2"]);
    let nested = "SELECT id, (SELECT max(c) FROM (SELECT count(*) c FROM lc WHERE lc.g = lt.id GROUP BY lc.g) q) FROM lt";
    assert_eq!(rows(&s, nested), ["1,2", "2,2", "3,NULL", "4,1"]);
}

/// Outer references Spark refuses, also where the rows would not matter.
#[test]
fn correlated_references_spark_refuses() {
    let s = correlated_session();
    for query in [
        "SELECT id FROM lt WHERE (SELECT count(*) FROM lc WHERE lc.g = lt.id GROUP BY lt.id) = 1",
        "SELECT id FROM lt WHERE k IN (SELECT count(*) FROM lc WHERE lc.g = lt.id GROUP BY lt.id)",
        "SELECT id FROM lt WHERE EXISTS (SELECT count(*) FROM (SELECT * FROM lc ORDER BY lt.id) q)",
    ] {
        let err = failure(&s, query);
        assert!(err.contains("CORRELATED_REFERENCE"), "{query}: {err}");
    }
}

/// Each outer row reads the subquery as if it ran for that row alone.
#[test]
fn correlated_rewrites_keep_per_row_results() {
    let s = correlated_session();
    for (query, want) in [
        // A global aggregate has one row even with no matches: NULL here.
        (
            "SELECT id FROM lt WHERE k IN (SELECT max(lc.k) FROM lc WHERE lc.g = lt.id)",
            vec!["4"],
        ),
        (
            "SELECT id FROM lt WHERE k NOT IN (SELECT max(lc.k) FROM lc WHERE lc.g = lt.id)",
            vec!["1", "2"],
        ),
        // A count over no matches is 0, so the value is 3 for id 3.
        (
            "SELECT id FROM lt WHERE id IN (SELECT count(*) + 3L FROM lc WHERE lc.g = lt.id)",
            vec!["3", "4"],
        ),
        // EXISTS: a grouped aggregate or a LIMIT has a row when its input has.
        (
            "SELECT id FROM lt WHERE EXISTS (SELECT max(y) FROM lc WHERE lc.g = lt.id GROUP BY lc.g)",
            vec!["1", "2", "4"],
        ),
        (
            "SELECT id FROM lt WHERE EXISTS (SELECT 1 FROM lc WHERE lc.g = lt.id LIMIT 1)",
            vec!["1", "2", "4"],
        ),
        // The correlated column is neither a key nor selected.
        (
            "SELECT id FROM lt WHERE k IN (SELECT min(lc.k) FROM lc WHERE lc.g = lt.id GROUP BY lc.y)",
            vec!["1", "4"],
        ),
        // The equality either way round; a LIMIT with no sort (id 4 has one row).
        (
            "SELECT id FROM lt WHERE k IN (SELECT lc.k FROM lc WHERE lt.id = lc.g ORDER BY y DESC LIMIT 1)",
            vec!["4"],
        ),
        (
            "SELECT id FROM lt WHERE id = 4 AND k IN (SELECT lc.k FROM lc WHERE lc.g = lt.id LIMIT 1)",
            vec!["4"],
        ),
        // Grouped by a constant: no row, so NULL and not the count 0.
        (
            "SELECT id, (SELECT count(*) FROM lc WHERE lc.g = lt.id GROUP BY 'a') FROM lt",
            vec!["1,2", "2,2", "3,NULL", "4,1"],
        ),
    ] {
        assert_eq!(rows(&s, query), want, "{query}");
    }
}

/// Shapes the rewrites cannot keep correct stay named refusals.
#[test]
fn correlated_shapes_without_a_rewrite_are_refused() {
    let s = correlated_session();
    for (query, want) in [
        (
            "SELECT id FROM lt WHERE k IN (SELECT max(lc.k) FROM lc WHERE lc.g > lt.id GROUP BY lc.y)",
            "not equalities",
        ),
        (
            "SELECT id FROM lt WHERE k IN (SELECT lc.k FROM lc WHERE lc.g > lt.id ORDER BY y LIMIT 1)",
            "LIMIT of this form",
        ),
        (
            "SELECT id, (SELECT count(*) FROM lc WHERE lc.g = lt.id GROUP BY lc.y + 1) FROM lt",
            "with GROUP BY",
        ),
        (
            "SELECT id, (SELECT sum(v) FROM (SELECT lc.k v FROM lc WHERE lc.g = lt.id UNION ALL SELECT k FROM lc) GROUP BY v) FROM lt",
            "with GROUP BY",
        ),
        (
            "SELECT id, (SELECT c FROM (SELECT count(*) c FROM lc WHERE lc.g = lt.id GROUP BY lc.g) q) FROM lt",
            "of this shape",
        ),
        (
            "SELECT id, (SELECT lc.y FROM lc WHERE lc.g + 1 = lt.id) FROM lt",
            "correlation key of this form",
        ),
    ] {
        let err = failure(&s, query);
        assert!(err.contains(want), "{query}: {err}");
    }
}

fn correlated_session() -> Session {
    let s = session();
    for ddl in [
        "CREATE TEMPORARY VIEW lt AS SELECT * FROM VALUES (1, 1), (2, 2), (3, CAST(NULL AS INT)), (4, 4) AS lt(id, k)",
        "CREATE TEMPORARY VIEW lc AS SELECT * FROM VALUES (1, 1, 10), (1, 5, 20), (2, 9, 30), (2, CAST(NULL AS INT), 5), (4, 4, 1) AS lc(g, k, y)",
    ] {
        block_on(s.sql(ddl)).unwrap();
    }
    s
}

/// The error of `query`, from analysis or from running it; empty if it works.
fn failure(s: &Session, query: &str) -> String {
    match block_on(s.sql(query)) {
        Err(e) => e.to_string(),
        Ok(df) => block_on(df.collect())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default(),
    }
}
