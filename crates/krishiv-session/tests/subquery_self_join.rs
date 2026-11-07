// SPDX-License-Identifier: Apache-2.0
//! A correlated subquery that reads one view twice: `INTERSECT`, `EXCEPT`
//! and joins of two correlated filters over the same view (the analyzer's
//! correlated instance: the view shifted to fresh ids, the correlation
//! keeping the outer query's). Expected rows are worked out from the views
//! below; Spark 4.2.0's own answers for these shapes are the
//! `exists-joins-and-set-ops` goldens.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
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
fn set_operations_of_two_correlated_filters_over_one_view() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1 from x where exists (select y2 from y where y1 = x1 and y2 < 3 \
                 intersect select y2 from y where y1 = x1 and y2 > 1)",
                &["1"],
            ),
            (
                "select x1 from x where exists (select y2 from y where y1 = x1 \
                 except select y2 from y where y1 = x1 and y2 > 1)",
                &["1"],
            ),
            (
                "select x1 from x where not exists (select y2 from y where y1 = x1 \
                 except all select y2 from y where y1 = x1 and y2 > 1)",
                &["2", "3"],
            ),
            (
                "select x1 from x where exists (select * from y where y1 = x1 \
                 intersect all select * from y where y1 = x1 and y2 >= 5)",
                &["2", "3"],
            ),
        ],
    )
}

#[test]
fn a_join_of_two_correlated_filters_over_one_view() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1 from x where exists (select 1 from (select y2 from y where y1 = x1) a \
                 join (select y2 from y where y1 = x1 and y2 > 4) b on a.y2 = b.y2)",
                &["2", "3"],
            ),
            (
                "select x1, (select count(*) from (select y2 from y where y1 = x1) a \
                 join (select y2 v from y where y1 = x1) b on a.y2 <= b.v) from x",
                &["1|6", "2|3", "3|1"],
            ),
        ],
    )
}

#[test]
fn a_with_inside_a_correlated_subquery_is_inlined() -> R {
    let s = session()?;
    check(
        &s,
        &[
            (
                "select x1, (with t as (select 1 as a) select a + x1 from t) from x",
                &["1|2", "2|3", "3|4"],
            ),
            (
                "select x1, (with t as (select * from y where y1 = x1) select sum(y2) from t) from x",
                &["1|6", "2|9", "3|6"],
            ),
            (
                "select x1, (with t as (select y1 + 1 as k, y2 from y), \
                 u as (select * from t where k = x1 + 1) select count(*) from u) from x",
                &["1|3", "2|2", "3|1"],
            ),
            (
                "select x1, (with t as (select * from y) select sum(v) from \
                 (select y1, y2 from t union select y2, y1 from t) r(k, v) where k = x1) from x",
                &["1|6", "2|10", "3|7"],
            ),
            (
                "select x1 from x where exists (with t as (select y2 from y where y1 = x1) \
                 select 1 from t a join t b on a.y2 < b.y2)",
                &["1", "2"],
            ),
        ],
    )
}

#[test]
fn a_nondeterministic_with_read_twice_stays_refused() -> R {
    let s = session()?;
    let err = run(
        &s,
        "select x1, (with t as (select y2, rand(0) r from y where y1 = x1) \
         select count(*) from t a join t b on a.r = b.r) from x",
    )
    .err()
    .ok_or("a nondeterministic CTE read twice was inlined")?;
    let msg = err.to_string();
    assert!(msg.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{msg}");
    assert!(
        msg.contains("a CTE read twice in a correlated subquery"),
        "{msg}"
    );
    Ok(())
}

/// A closed definition holding an `IN` subquery, read more than once in a
/// correlated subquery, gets fresh ids for the subquery's plan too: it answers
/// like the same query with the definition written out at each read.
#[test]
fn a_closed_definition_with_an_in_subquery_read_again_matches_its_expansion() -> R {
    let s = session()?;
    let d = "select y2 from y where y2 in (select y1 from y)";
    let cases = [
        (
            format!(
                "select x1, (with t as ({d}) select count(*) from t a join t b \
                 on a.y2 = b.y2 where a.y2 = x1) from x"
            ),
            format!(
                "select x1, (select count(*) from ({d}) a join ({d}) b \
                 on a.y2 = b.y2 where a.y2 = x1) from x"
            ),
            &["1|1", "2|1", "3|1"][..],
        ),
        (
            format!(
                "select x1, (with t as ({d}) select count(*) from t a join t b \
                 on a.y2 <= b.y2 join t c on b.y2 <= c.y2 where a.y2 = x1) from x"
            ),
            format!(
                "select x1, (select count(*) from ({d}) a join ({d}) b on a.y2 <= b.y2 \
                 join ({d}) c on b.y2 <= c.y2 where a.y2 = x1) from x"
            ),
            &["1|6", "2|3", "3|1"][..],
        ),
        (
            format!(
                "select x1, (with t as ({d}), u as (select a.y2 from t a join t b \
                 on a.y2 = b.y2) select count(*) from u p join u q on p.y2 = q.y2 \
                 where p.y2 = x1) from x"
            ),
            format!(
                "select x1, (select count(*) from (select a.y2 from ({d}) a join ({d}) b \
                 on a.y2 = b.y2) p join (select a.y2 from ({d}) a join ({d}) b \
                 on a.y2 = b.y2) q on p.y2 = q.y2 where p.y2 = x1) from x"
            ),
            &["1|1", "2|1", "3|1"][..],
        ),
    ];
    // A definition whose only subquery sits in its select list, over a leaf
    // with no ids of its own below the subquery's.
    let d2 = "select (select max(y1) from y) m, y2 from y";
    let extra = [(
        format!(
            "select x1, (with t as ({d2}) select count(*) from t a join t b \
             on a.m = b.m and a.y2 = b.y2 where a.y2 = x1) from x"
        ),
        format!(
            "select x1, (select count(*) from ({d2}) a join ({d2}) b \
             on a.m = b.m and a.y2 = b.y2 where a.y2 = x1) from x"
        ),
    )];
    // The subquery's plan holds the lowest ids: the definition is only a
    // select list over it, with no relation of its own.
    let d3 = "select (select max(y1) from y) m";
    let low = [(
        format!(
            "select x1, (with t as ({d3}) select count(*) from t a join t b \
             on a.m = b.m where a.m = x1) from x"
        ),
        format!(
            "select x1, (select count(*) from ({d3}) a join ({d3}) b \
             on a.m = b.m where a.m = x1) from x"
        ),
    )];
    for (cte, expanded) in extra.iter().chain(&low) {
        assert_eq!(run(&s, cte)?, run(&s, expanded)?, "{cte}");
    }
    for (cte, expanded, want) in &cases {
        assert_eq!(run(&s, expanded)?, rows(want), "{expanded}");
        assert_eq!(run(&s, cte)?, rows(want), "{cte}");
    }
    Ok(())
}
