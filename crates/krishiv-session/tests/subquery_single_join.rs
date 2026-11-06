// SPDX-License-Identifier: Apache-2.0
//! Correlated scalar subqueries that may have more than one row per outer
//! row: Spark 4.x runs them as a single join, which raises
//! `SCALAR_SUBQUERY_TOO_MANY_ROWS` at run time on a second row; with
//! `spark.sql.optimizer.scalarSubqueryUseSingleJoin` off analysis refuses
//! them (`MUST_AGGREGATE_CORRELATED_SCALAR_SUBQUERY`, or
//! `NON_CORRELATED_COLUMNS_IN_GROUP_BY` for a grouping the correlation does
//! not fix), and
//! `spark.sql.legacy.scalarSubqueryAllowGroupByNonEqualityCorrelatedPredicate`
//! lets the grouped ones run as an outer join. Every expected answer and
//! error class is what a Spark 4.2.0 run gave.
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

/// The sorted rows of `sql`, or its error class.
fn run(s: &Session, sql: &str) -> Result<Vec<String>, String> {
    block_on(async {
        let batches = s
            .sql(sql)
            .await
            .map_err(|e| e.error_class().name().to_owned())?
            .collect()
            .await
            .map_err(|e| e.error_class().name().to_owned())?;
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

fn ok(v: &[&str]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = v.iter().map(|r| (*r).to_owned()).collect();
    out.sort();
    Ok(out)
}

fn err(class: &str) -> Result<Vec<String>, String> {
    Err(class.to_owned())
}

fn session() -> R<Session> {
    let s = Session::new()?;
    for v in [
        "create temporary view x(x1, x2) as values (1, 1), (2, 2)",
        "create temporary view y(y1, y2) as values (1, 1), (1, 2), (2, 4)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

const SINGLE: &str = "spark.sql.optimizer.scalarSubqueryUseSingleJoin";
const MUST: &str =
    "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.MUST_AGGREGATE_CORRELATED_SCALAR_SUBQUERY";
const LOOSE: &str = "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.NON_CORRELATED_COLUMNS_IN_GROUP_BY";

#[test]
fn single_join_off_refuses_what_may_have_several_rows() -> R {
    let s = session()?;
    run(&s, &format!("set {SINGLE} = false"))?;
    let q = "select x1, (select y2 from y where y1 = x1) from x";
    assert_eq!(run(&s, q), err(MUST), "{q}");
    let q = "select x1, (select y2 from y where y1 = x1 and y2 = 4) from x";
    assert_eq!(run(&s, q), err(MUST), "{q}");
    let q = "select x1, (select count(*) from y where y1 > x1 group by y1) from x";
    assert_eq!(run(&s, q), err(LOOSE), "{q}");
    Ok(())
}

#[test]
fn single_join_on_answers_or_fails_on_a_second_row() -> R {
    let s = session()?;
    let q = "select x1, (select y2 from y where y1 = x1) from x";
    assert_eq!(run(&s, q), err("SCALAR_SUBQUERY_TOO_MANY_ROWS"), "{q}");
    let q = "select x1, (select y2 from y where y1 = x1 and y2 = 4) from x";
    assert_eq!(run(&s, q), ok(&["1|NULL", "2|4"]), "{q}");
    let q = "select x1, (select count(*) from y where y1 > x1 group by y1) from x";
    assert_eq!(run(&s, q), ok(&["1|1", "2|NULL"]), "{q}");
    Ok(())
}

#[test]
fn the_legacy_flag_runs_a_loose_grouping_as_an_outer_join() -> R {
    let s = session()?;
    run(&s, &format!("set {SINGLE} = false"))?;
    run(
        &s,
        "set spark.sql.legacy.scalarSubqueryAllowGroupByNonEqualityCorrelatedPredicate = true",
    )?;
    let q = "select x1, (select count(*) from y where y1 >= x1 group by y1) from x";
    assert_eq!(run(&s, q), ok(&["1|1", "1|2", "2|1"]), "{q}");
    Ok(())
}
