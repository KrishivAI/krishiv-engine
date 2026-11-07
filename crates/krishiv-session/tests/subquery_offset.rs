// SPDX-License-Identifier: Apache-2.0
//! `OFFSET` (with or without `LIMIT`) and `LIMIT 0` in correlated subqueries
//! (Spark's `DecorrelateInnerQuery`: a `row_number()` per correlation value,
//! ranks `offset + 1` to `offset + limit`), a `LIMIT` / `OFFSET` over a
//! correlated global aggregate, and the `decorrelateLimit` /
//! `decorrelateOffset` switches (and Spark 3.5, which has neither). Expected
//! rows are worked out from the views below; Spark 4.2.0's own answers for
//! these shapes are the `subquery-offset`, `in-limit` and
//! `exists-orderby-limit` goldens.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        other => format!("<{other}>"),
    }
}

fn run(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| cell(c.as_ref(), r))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect();
        out.sort();
        Ok(out)
    })
}

fn rows(v: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = v.iter().map(|r| (*r).to_owned()).collect();
    out.sort();
    out
}

fn views(s: &Session) -> R {
    for v in [
        "create temporary view x(x1, x2) as values (1, 1), (2, 2), (3, 3)",
        "create temporary view y(y1, y2) as values (1, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6)",
    ] {
        run(s, v)?;
    }
    Ok(())
}

fn session() -> R<Session> {
    let s = Session::new()?;
    views(&s)?;
    Ok(s)
}

fn check(s: &Session, cases: &[(&str, &[&str])]) -> R {
    for (q, want) in cases {
        assert_eq!(run(s, q)?, rows(want), "{q}");
    }
    Ok(())
}

#[test]
fn an_offset_over_an_equality_correlation_skips_rows_per_outer_row() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1, (select y2 from y where y1 = x1 order by y2 limit 1 offset 1) from x",
                &["1|2", "2|5", "3|NULL"],
            ),
            (
                "select x1 from x where x2 + 1 in (select y2 from y where y1 = x1 order by y2 offset 1)",
                &["1"],
            ),
            (
                "select x1 from x where x2 + 2 in (select y2 from y where y1 = x1 order by y2 limit 1 offset 1)",
                &[],
            ),
            (
                "select x1 from x where exists (select 1 from y where y1 = x1 offset 2)",
                &["1"],
            ),
            (
                "select x1 from x where not exists (select 1 from y where y1 = x1 limit 5 offset 2)",
                &["2", "3"],
            ),
            (
                "select x1, y2 from x join lateral (select y2 from y where y1 = x1 order by y2 offset 1)",
                &["1|2", "1|3", "2|5"],
            ),
        ],
    )
}

#[test]
fn a_zero_limit_leaves_no_row_for_any_outer_row() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1 from x where exists (select 1 from y where y1 = x1 limit 0)",
                &[],
            ),
            (
                "select x1 from x where not exists (select 1 from y where y1 = x1 order by y2 limit 0)",
                &["1", "2", "3"],
            ),
            (
                "select x1 from x where x2 in (select y2 from y where y1 = x1 limit 0)",
                &[],
            ),
        ],
    )
}

#[test]
fn an_offset_over_a_non_equality_correlation_ranks_through_the_domain() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1 from x where exists (select 1 from y where y1 < x1 offset 2)",
                &["2", "3"],
            ),
            (
                "select x1 from x where 4 in (select y2 from y where y1 < x1 order by y2 limit 1 offset 3)",
                &["3"],
            ),
            (
                "select x1 from x where not exists (select 1 from y where y1 < x1 limit 1 offset 4)",
                &["1", "2"],
            ),
        ],
    )
}

/// A global aggregate is one row per outer row: a positive `LIMIT` keeps
/// it, an `OFFSET` drops it (the scalar is NULL, EXISTS false).
#[test]
fn a_limit_or_offset_over_a_correlated_global_aggregate() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1, (select sum(y2) from y where y1 = x1 limit 1) from x",
                &["1|6", "2|9", "3|6"],
            ),
            (
                "select x1, (select sum(y2) from y where y1 = x1 limit 1 offset 1) from x",
                &["1|NULL", "2|NULL", "3|NULL"],
            ),
            (
                "select x1, (select count(*) from y where y1 = x1 offset 1) from x",
                &["1|NULL", "2|NULL", "3|NULL"],
            ),
            (
                "select x1 from x where exists (select sum(y2) from y where y1 = x1 offset 1)",
                &[],
            ),
            (
                "select x1 from x where 6 in (select sum(y2) from y where y1 = x1 limit 10)",
                &["1", "3"],
            ),
        ],
    )
}

const OUTER: &str = "ACCESSING_OUTER_QUERY_COLUMN_IS_NOT_ALLOWED";

fn failure(s: &Session, sql: &str) -> String {
    match run(s, sql) {
        Ok(rows) => format!("ok: {rows:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn decorrelation_switches_and_spark_3_5_refuse_a_correlation_below() -> R {
    let limit = "select x1, (select y2 from y where y1 = x1 order by y2 limit 1) from x";
    let offset = "select x1 from x where exists (select 1 from y where y1 = x1 offset 2)";
    let lateral = "select x1, y2 from x join lateral (select y2 from y where y1 = x1 offset 1)";
    let s = session()?;
    run(
        &s,
        "set spark.sql.optimizer.decorrelateLimit.enabled = false",
    )?;
    assert!(failure(&s, limit).contains(OUTER), "{}", failure(&s, limit));
    // An OFFSET alone is a node of its own (Spark's `Offset`), with its own switch.
    assert_eq!(run(&s, offset)?, rows(&["1"]));
    run(
        &s,
        "set spark.sql.optimizer.decorrelateLimit.enabled = true",
    )?;
    run(
        &s,
        "set spark.sql.optimizer.decorrelateOffset.enabled = false",
    )?;
    assert!(
        failure(&s, offset).contains(OUTER),
        "{}",
        failure(&s, offset)
    );
    assert!(
        failure(&s, lateral).contains(OUTER),
        "{}",
        failure(&s, lateral)
    );
    assert_eq!(run(&s, limit)?, rows(&["1|1", "2|4", "3|6"]));

    let old = Session::builder().profile(SparkProfile::V3_5).build()?;
    views(&old)?;
    for q in [limit, offset] {
        assert!(
            failure(&old, q).contains(OUTER),
            "{q}: {}",
            failure(&old, q)
        );
    }
    Ok(())
}

/// A window on the correlation path runs per outer value (Spark's
/// `DecorrelateInnerQuery` adds the correlated inner columns, or the
/// domain's, to its partitioning): a running sum restarts for each `y1`.
#[test]
fn a_window_below_a_correlation_is_partitioned_per_outer_value() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1, (select max(s) from (select sum(y2) over (order by y2) s from y where y1 = x1) t) from x",
                &["1|6", "2|9", "3|6"],
            ),
            (
                "select x1 from x where x2 in (select rank() over (order by y2) from y where y1 = x1)",
                &["1", "2"],
            ),
            (
                "select x1 from x where exists (select rank() over (order by y2) from y where y1 = x1 + 1)",
                &["1", "2"],
            ),
            // Not an equality on a bare column: through the domain.
            (
                "select x1, (select max(s) from (select count(*) over (order by y2) s from y where y1 < x1) t) from x",
                &["1|NULL", "2|3", "3|5"],
            ),
            (
                "select x1, (select max(s) from (select sum(y2) over (order by y2) s from y where y1 + 0 = x1) t) from x",
                &["1|6", "2|9", "3|6"],
            ),
            // Ranked, then limited per outer value.
            (
                "select x1, (select s from (select sum(y2) over (order by y2) s from y where y1 = x1) t order by s desc limit 1) from x",
                &["1|6", "2|9", "3|6"],
            ),
        ],
    )
}

/// Spark 3.5's `CheckAnalysis.checkPlan` has no case for `Window`: outer
/// references below one are refused there, as below a `LIMIT` (v3.5.3 source);
/// 4.x decorrelates it, and so does the default profile here.
#[test]
fn spark_3_5_refuses_a_correlation_below_a_window() -> R {
    let q = "select x1, (select max(s) from (select sum(y2) over (order by y2) s \
             from y where y1 = x1) t) from x";
    let new = session()?;
    assert_eq!(run(&new, q)?, rows(&["1|6", "2|9", "3|6"]), "{q}");
    let old = Session::builder().profile(SparkProfile::V3_5).build()?;
    views(&old)?;
    assert!(
        failure(&old, q).contains(OUTER),
        "{q}: {}",
        failure(&old, q)
    );
    Ok(())
}

/// Spark checks a subquery with its CTEs inlined, so a window (or LIMIT) behind
/// a reference to a correlated `WITH` definition is refused on 3.5 too; the
/// default profile answers it as it does the written-out form.
#[test]
fn spark_3_5_checks_a_bound_behind_a_correlated_with_reference() -> R {
    let cte = "select x1, (with t as (select y2 from y where y1 = x1) \
               select max(s) from (select sum(y2) over (order by y2) s from t) q) from x";
    let written = "select x1, (select max(s) from (select sum(y2) over (order by y2) s \
                   from y where y1 = x1) q) from x";
    let new = session()?;
    assert_eq!(run(&new, cte)?, run(&new, written)?, "{cte}");
    assert_eq!(run(&new, cte)?, rows(&["1|6", "2|9", "3|6"]), "{cte}");
    let old = Session::builder().profile(SparkProfile::V3_5).build()?;
    views(&old)?;
    assert!(
        failure(&old, cte).contains(OUTER),
        "{cte}: {}",
        failure(&old, cte)
    );
    Ok(())
}
