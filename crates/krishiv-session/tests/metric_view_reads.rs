// SPDX-License-Identifier: Apache-2.0
//! Reads of a metric view beyond the Spark goldens (`metric_views.rs`): the reader is asked about
//! the source, what a dimension or measure may not call yet, `measure(..)` of a column that is not
//! a measure, one view read twice, and the leftover refusals (design note `P3-metric-views.md` §4,
//! review 2026-11-20).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::BoxFuture;
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, Principal, PrincipalKind, Privilege, Securable, Session,
};

const VIEW: &str = "CREATE VIEW mv WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: src
dimensions:
  - name: region
    expr: region
measures:
  - name: total
    expr: sum(n)
$$";

fn session_with(policy: Arc<dyn AccessPolicy>) -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(policy)
        .build()
        .unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", &dir.path().display().to_string())
        .unwrap();
    for sql in [
        "CREATE TABLE src (region STRING, n INT) USING parquet",
        "INSERT INTO src VALUES ('a', 1), ('a', 2), ('b', 5)",
        VIEW,
    ] {
        run(&s, sql).unwrap();
    }
    (s, dir)
}

fn session() -> (Session, tempfile::TempDir) {
    session_with(Arc::new(AllowAll))
}

fn run(s: &Session, sql: &str) -> Result<Vec<RecordBatch>, KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await })
}

fn longs(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    let mut out: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(column)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect();
    out.sort_unstable();
    out
}

fn class(e: &KrishivError) -> String {
    e.error_class().name().to_owned()
}

/// Denies `SELECT` on the table `src` once `armed`; everything else is allowed.
struct DenySource;

impl AccessPolicy for DenySource {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let denied = privilege == Privilege::Select
            && matches!(securable, Securable::Table(t) if t.name == "src");
        Box::pin(std::future::ready(Ok(if denied {
            AccessDecision::Deny {
                reason: "private".to_owned(),
            }
        } else {
            AccessDecision::Allow
        })))
    }
}

#[test]
fn the_reader_is_asked_about_the_source() {
    // The view is created where the source may be read, then read by a reader who may not read it:
    // a metric view's source is the reader's read (no definer rights).
    let (s, _dir) = session();
    let reader = s.new_session_as(
        Principal::new("bo", PrincipalKind::User),
        Arc::new(DenySource),
    );
    let e = run(
        &reader,
        "SELECT region, measure(total) FROM mv GROUP BY region",
    )
    .unwrap_err();
    assert_eq!(class(&e), "KRISHIV_INSUFFICIENT_PERMISSIONS", "{e}");
    let e = run(&reader, "SELECT * FROM mv").unwrap_err();
    assert_eq!(class(&e), "KRISHIV_INSUFFICIENT_PERMISSIONS", "{e}");
}

#[test]
fn a_measure_with_a_subquery_or_a_sql_function_is_refused_at_read() {
    let (s, _dir) = session();
    run(&s, "CREATE TABLE other (m INT) USING parquet").unwrap();
    run(
        &s,
        "CREATE FUNCTION plus_one(x INT) RETURNS INT RETURN x + 1",
    )
    .unwrap();
    for (name, measure) in [
        ("with_subquery", "sum(n) + (SELECT max(m) FROM other)"),
        ("with_function", "sum(plus_one(n))"),
    ] {
        let create = format!(
            "CREATE VIEW {name} WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: src\n\
             dimensions:\n  - name: region\n    expr: region\nmeasures:\n  - name: m\n    expr: {measure}\n$$"
        );
        run(&s, &create).unwrap();
        let e = run(&s, &format!("SELECT measure(m) FROM {name}")).unwrap_err();
        assert_eq!(class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{name}: {e}");
    }
}

#[test]
fn measure_of_a_column_that_is_not_a_measure_is_sparks_internal_error() {
    let (s, _dir) = session();
    // A projection's alias, even one spelled like a measure, is not looked up by name.
    for sql in [
        "SELECT measure(x) FROM (SELECT region AS x FROM mv)",
        "SELECT measure(total) FROM (SELECT region AS total FROM mv)",
        "SELECT measure(region) FROM mv",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(class(&e), "INTERNAL_ERROR", "{sql}: {e}");
        assert!(e.to_string().contains("key not found"), "{sql}: {e}");
    }
}

#[test]
fn one_view_read_twice_gives_each_read_its_own_columns() {
    let (s, _dir) = session();
    let rows = run(
        &s,
        "SELECT a.t, b.t FROM (SELECT region, measure(total) AS t FROM mv GROUP BY region) a \
         JOIN (SELECT region, measure(total) AS t FROM mv GROUP BY region) b ON a.region = b.region",
    )
    .unwrap();
    assert_eq!(longs(&rows, 0), [3, 5]);
    assert_eq!(longs(&rows, 1), [3, 5]);
    let rows = run(
        &s,
        "SELECT measure(total) FROM mv UNION ALL SELECT measure(total) FROM mv",
    )
    .unwrap();
    assert_eq!(longs(&rows, 0), [8, 8]);
}

#[test]
fn a_read_left_without_an_aggregate_is_analyzable_but_not_plannable() {
    let (s, _dir) = session();
    let df = block_on(s.sql("SELECT * FROM mv")).unwrap();
    // Analysis succeeds (the schema is the stored one) ...
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["region", "total"]);
    // ... and planning refuses it, as Spark's planner does.
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(class(&e), "INTERNAL_ERROR");
    assert!(e.to_string().contains("phase planning failed"), "{e}");
    // A `measure(..)` left over is refused first, with its own text.
    let e = run(&s, "SELECT measure(total) FROM mv JOIN src ON true").unwrap_err();
    assert!(
        e.to_string()
            .contains("UnevaluableAggregateFunc.aggBufferAttributes"),
        "{e}"
    );
}

/// The create aggregate groups by the dimensions' ordinals whatever `spark.sql.groupByOrdinal`
/// says: with it off a create still groups, and a dimension that is an aggregate is still left
/// out of the grouping (Spark groups by the dimensions' attributes).
#[test]
fn a_create_groups_by_its_dimensions_with_ordinals_off() {
    let (s, _dir) = session();
    s.conf().set("spark.sql.groupByOrdinal", "false").unwrap();
    run(
        &s,
        "CREATE VIEW mv2 WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: src
dimensions:
  - name: region
    expr: upper(region)
  - name: everything
    expr: sum(n)
measures:
  - name: total
    expr: sum(n)
$$",
    )
    .unwrap();
    let err = run(
        &s,
        "SELECT region, measure(total) FROM mv2 GROUP BY region ORDER BY region",
    )
    .unwrap_err();
    // The aggregate dimension makes every read Spark's `MISSING_GROUP_BY`, as the golden
    // `yaml_dimension_is_aggregate` shows; the create itself succeeded above.
    assert_eq!(class(&err), "MISSING_GROUP_BY");
    let rows = run(
        &s,
        "SELECT region, measure(total) FROM mv GROUP BY region ORDER BY region",
    )
    .unwrap();
    assert_eq!(longs(&rows, 1), vec![3, 5]);
}

/// A metric view's operators keep the creator's ANSI mode, the functions under them the reader's
/// (a Spark 4.2.1 run, 2026-11-21: created under ANSI, read without it, `abs(i) + 0` of
/// `INT_MIN` is `INT_MIN`, `-i` overflows).
#[test]
fn an_operator_keeps_the_creators_mode_and_its_operand_the_readers() {
    let (s, _dir) = session();
    run(&s, "CREATE TABLE t (k STRING, i INT) USING parquet").unwrap();
    run(&s, "INSERT INTO t VALUES ('a', -2147483648), ('b', 1)").unwrap();
    s.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    run(
        &s,
        "CREATE VIEW mvo WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\nsource: t\ndimensions:\n  - name: a\n    expr: abs(i) + 0\n  - name: n\n    expr: -i\nmeasures:\n  - name: c\n    expr: count(1)\n$$",
    )
    .unwrap();
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    let got = run(
        &s,
        "SELECT CAST(a AS BIGINT), measure(c) FROM mvo GROUP BY a",
    )
    .unwrap();
    assert_eq!(longs(&got, 0), vec![-2_147_483_648, 1]);
    let e = run(&s, "SELECT n, measure(c) FROM mvo GROUP BY n").unwrap_err();
    assert_eq!(class(&e), "ARITHMETIC_OVERFLOW");
}

/// A dimension or measure ending in a `--` comment (each parses alone, as Spark parses it) ends
/// there: the create aggregate keeps its alias and its source (review b047d8901 finding 1).
#[test]
fn an_expression_ending_in_a_comment_keeps_its_name_and_source() {
    let (s, _dir) = session();
    let yaml = |source: &str| {
        format!(
            "CREATE VIEW mv3 WITH METRICS LANGUAGE YAML AS $$
version: 0.1
source: {source}
dimensions:
  - name: r
    expr: upper(region) -- the region
measures:
  - name: rows
    expr: count(1) -- every row
$$"
        )
    };
    let err = run(&s, &yaml("no_such_table")).unwrap_err();
    assert_eq!(class(&err), "TABLE_OR_VIEW_NOT_FOUND");
    run(&s, &yaml("src")).unwrap();
    let rows = run(&s, "SELECT r, measure(rows) FROM mv3 GROUP BY r ORDER BY r").unwrap();
    let schema = rows[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["r", "measure(rows)"]);
    assert_eq!(longs(&rows, 1), vec![1, 2]);
}
