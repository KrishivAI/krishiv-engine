// SPDX-License-Identifier: Apache-2.0
//! `IN` subqueries in a join condition with
//! `spark.sql.optimizer.optimizeUncorrelatedInSubqueriesInJoinCondition.enabled`
//! off: a conjunct Spark's optimizer leaves in the join (one reading both
//! sides, or only the side an outer or anti join keeps) evaluates the `IN`
//! as an expression, three-valued, and NULL for a NULL value under
//! `spark.sql.legacy.nullInEmptyListBehavior`. A conjunct pushed into one
//! side, and every conjunct with the flag on, is a join: two-valued. Every
//! expected answer and error class is what a Spark 4.2.0 run gave
//! (`in-null-semantics.sql` #27-#28 have the legacy shape).
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
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
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
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

/// The error class `sql` fails with.
fn error_class(s: &Session, sql: &str) -> String {
    match run(s, sql) {
        Ok(rows) => format!("answered {rows:?}"),
        Err(e) => e.to_string(),
    }
}

fn session() -> R<Session> {
    let s = Session::new()?;
    for q in [
        "create table t(c int) using parquet",
        "insert into t values (1), (null)",
        "create table t2(d int) using parquet",
        "insert into t2 values (2)",
        "create table t3(e int) using parquet",
        "insert into t3 values (5), (null)",
        "create table t_empty(e int) using parquet",
    ] {
        run(&s, q)?;
    }
    Ok(s)
}

fn set(s: &Session, key: &str, value: bool) -> R {
    run(s, &format!("set spark.sql.{key} = {value}"))?;
    Ok(())
}

const AS_JOIN: &str = "optimizer.optimizeUncorrelatedInSubqueriesInJoinCondition.enabled";
const LEGACY: &str = "legacy.nullInEmptyListBehavior";

#[test]
fn as_a_join_by_default_the_in_is_two_valued() -> R {
    let s = session()?;
    for q in [
        "select * from t left join t2 on (t.c in (select e from t3)) is null",
        "select * from t left join t2 on (t.c in (select e from t_empty)) is null",
    ] {
        assert_eq!(run(&s, q)?, rows(&["1|NULL", "NULL|NULL"]), "{q}");
    }
    // The legacy flag alone changes nothing in a join condition.
    set(&s, LEGACY, true)?;
    let q = "select * from t left join t2 on (t.c in (select e from t_empty)) is null";
    assert_eq!(run(&s, q)?, rows(&["1|NULL", "NULL|NULL"]), "{q}");
    Ok(())
}

#[test]
fn a_conjunct_left_in_the_join_is_three_valued() -> R {
    let s = session()?;
    set(&s, AS_JOIN, false)?;
    let cases: [(&str, &[&str]); 9] = [
        (
            "select * from t left join t2 on (t.c in (select e from t3)) is null",
            &["1|2", "NULL|2"],
        ),
        (
            "select * from t left join t2 on (t.c not in (select e from t3)) is null",
            &["1|2", "NULL|2"],
        ),
        (
            "select * from t left join t2 on (t.c in (select e from t_empty)) is null",
            &["1|NULL", "NULL|NULL"],
        ),
        // Pushed into one side: a filter there, rewritten to a join.
        (
            "select * from t join t2 on (t.c in (select e from t3)) is null",
            &[],
        ),
        (
            "select * from t right join t2 on (t.c in (select e from t3)) is null",
            &["NULL|2"],
        ),
        (
            "select * from t left semi join t2 on (t.c in (select e from t3)) is null",
            &[],
        ),
        // Kept: the conjunct reads both sides, or the join keeps its side.
        (
            "select * from t join t2 on (t.c in (select e from t3)) is null or t2.d = 3",
            &["1|2", "NULL|2"],
        ),
        (
            "select * from t full join t2 on (t.c in (select e from t3)) is null",
            &["1|2", "NULL|2"],
        ),
        // A multi-value IN compares a struct, never NULL.
        (
            "select * from t left join t2 on ((t.c, 1) in (select e, e from t3)) is null",
            &["1|NULL", "NULL|NULL"],
        ),
    ];
    for (q, want) in cases {
        assert_eq!(run(&s, q)?, rows(want), "{q}");
    }
    let q = "select * from t left anti join t2 on (t.c in (select e from t3)) is null";
    assert_eq!(run(&s, q)?, rows(&[]), "{q}");
    let q = "select * from t join t2 on (t.c + t2.d in (select e from t3)) is null";
    let e = error_class(&s, q);
    assert!(
        e.contains("UNSUPPORTED_CORRELATED_EXPRESSION_IN_JOIN_CONDITION")
            && e.contains("((c + d) IN (listquery()))")
            && !e.contains("true IN"),
        "{q}: {e}"
    );
    Ok(())
}

#[test]
fn legacy_null_in_an_empty_list_is_null_where_the_in_stays() -> R {
    let s = session()?;
    set(&s, AS_JOIN, false)?;
    set(&s, LEGACY, true)?;
    let cases: [(&str, &[&str]); 7] = [
        (
            "select * from t left join t2 on (t.c in (select e from t_empty)) is null",
            &["1|NULL", "NULL|2"],
        ),
        (
            "select * from t left join t2 on (t.c not in (select e from t_empty)) is null",
            &["1|NULL", "NULL|2"],
        ),
        (
            "select * from t join t2 on (t.c in (select e from t_empty)) is null",
            &[],
        ),
        (
            "select * from t join t2 on (t.c in (select e from t_empty)) is null or t2.d = 3",
            &["NULL|2"],
        ),
        (
            "select * from t right join t2 on (t.c in (select e from t_empty)) is null",
            &["NULL|2"],
        ),
        (
            "select * from t full join t2 on (t.c in (select e from t_empty)) is null",
            &["1|NULL", "NULL|2"],
        ),
        (
            "select * from t left semi join t2 on (t.c in (select e from t_empty)) is null",
            &[],
        ),
    ];
    for (q, want) in cases {
        assert_eq!(run(&s, q)?, rows(want), "{q}");
    }
    let q = "select * from t left anti join t2 on (t.c in (select e from t_empty)) is null";
    assert_eq!(run(&s, q)?, rows(&["1"]), "{q}");
    // Outside a join condition the IN is a join: false.
    let q = "select c, c in (select e from t_empty) from t";
    assert_eq!(run(&s, q)?, rows(&["1|false", "NULL|false"]), "{q}");
    // The flag off alone keeps the three-valued answer, not the legacy one.
    set(&s, LEGACY, false)?;
    let q = "select * from t left join t2 on (t.c in (select e from t_empty)) is null";
    assert_eq!(run(&s, q)?, rows(&["1|NULL", "NULL|NULL"]), "{q}");
    Ok(())
}
