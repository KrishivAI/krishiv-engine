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

/// A neither-side `IN` that is a conjunct of its own keeps a row only when it
/// is true, so NULL and false agree and it stays a plain join, whatever its
/// width or the legacy flag; nested under `NOT` or `IS NULL` the two differ
/// and the multi-value and legacy forms stay refused. The answers follow from
/// `5 IN (5, NULL)` being true (hand-worked; Spark runs are requested).
#[test]
fn a_neither_side_in_conjunct_stays_a_join_in_every_width_and_mode() -> R {
    let s = session()?;
    for legacy in [false, true] {
        set(&s, LEGACY, legacy)?;
        for q in [
            "select count(*) from t join t2 on 5 in (select e from t3)",
            "select count(*) from t join t2 on (5, 5) in (select e, e from t3)",
        ] {
            assert_eq!(run(&s, q)?, rows(&["2"]), "legacy {legacy}: {q}");
        }
        let q = "select count(*) from t join t2 on 3 in (select e from t3)";
        assert_eq!(run(&s, q)?, rows(&["0"]), "legacy {legacy}: {q}");
    }
    // A `NOT (multi-value IN)` conjunct is a null-aware anti join: (5, 5) matches a `t3` row, so
    // NOT IN is false; (5, 6) matches none but meets the all-NULL row, so IN is NULL and NOT IN
    // is not TRUE either (a plain anti join would keep the pairs and count 2).
    for q in [
        "select count(*) from t join t2 on not ((5, 5) in (select e, e from t3))",
        "select count(*) from t join t2 on not ((5, 6) in (select e, e from t3))",
    ] {
        assert_eq!(run(&s, q)?, rows(&["0"]), "{q}");
    }
    // Nested under `IS NULL` NULL and false differ, and the rewrite is still refused.
    let q = "select count(*) from t join t2 on (5 in (select e from t3)) is null";
    let e = error_class(&s, q);
    assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{q}: {e}");
    Ok(())
}

/// A correlation that stays inside a `LIMIT`'s own input is not an access to
/// the outer query, even with the limit's decorrelation off.
#[test]
fn a_nested_correlation_inside_the_limit_input_is_not_refused() -> R {
    let s = session()?;
    set(&s, "optimizer.decorrelateLimit.enabled", false)?;
    let q = "select exists (select 1 from (select * from t2 where exists \
             (select 1 from t3 where t3.e is null or t3.e = t2.d)) limit 1)";
    assert_eq!(run(&s, q)?, rows(&["true"]), "{q}");
    // A correlation that does escape the limit's input is still refused.
    let q = "select t.c, exists (select 1 from t2 where t2.d = t.c limit 1) from t";
    let e = error_class(&s, q);
    assert!(
        e.contains("ACCESSING_OUTER_QUERY_COLUMN_IS_NOT_ALLOWED"),
        "{q}: {e}"
    );
    Ok(())
}

/// A conjunct `NOT (v IN (subquery))` that reads neither side keeps a pair only
/// when it is TRUE: a null-aware anti join, which every join type plans (a FULL
/// OUTER join cannot hold an `exists` expression, which the three-valued rewrite
/// of a nested `IN` would make). `t` holds 1 and NULL, `t2` holds 2, `t3` holds 5
/// and NULL, and `t_empty` nothing. Rows are worked out from those: 5 is in `t3`
/// (NOT IN false), 7 is not but `t3` has a NULL (NULL), and nothing is in
/// `t_empty` (TRUE). Spark 4.3's `in-joins` statement 32 is the first shape.
#[test]
fn a_not_in_conjunct_that_reads_neither_side_plans_in_a_full_outer_join() -> R {
    let s = session()?;
    let full = |cond: &str| format!("select t.c, t2.d from t full outer join t2 on ({cond})");
    // FALSE and NULL: no pair matches, so each side's rows come back alone.
    for cond in ["5 not in (select e from t3)", "7 not in (select e from t3)"] {
        assert_eq!(
            run(&s, &full(cond))?,
            rows(&["1|NULL", "NULL|NULL", "NULL|2"]),
            "{cond}"
        );
    }
    // TRUE: every pair matches.
    let cond = "7 not in (select e from t_empty)";
    assert_eq!(run(&s, &full(cond))?, rows(&["1|2", "NULL|2"]), "{cond}");
    // The same conjunct beside an equality, and as a left join.
    let q = "select t.c, t2.d from t left join t2 on (7 not in (select e from t_empty))";
    assert_eq!(run(&s, q)?, rows(&["1|2", "NULL|2"]), "{q}");
    Ok(())
}
