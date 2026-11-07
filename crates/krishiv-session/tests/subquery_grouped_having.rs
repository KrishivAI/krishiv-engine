// SPDX-License-Identifier: Apache-2.0
//! A correlated scalar subquery with `GROUP BY` on the correlated column and a
//! `HAVING` (scalar-subquery-predicate.sql #18 and #19 hold Spark 4.2.0's
//! answers). A group the `HAVING` rejects, or no group, reads NULL, a `count`
//! value included. The positive checks compare the subquery with the explicit
//! left join of the grouped, filtered relation; the other shapes stay refused.
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
                (0..b.num_rows()).map(move |r| {
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

fn session() -> R<Session> {
    let s = Session::new()?;
    for q in [
        "create or replace temporary view t(k, a) as values (1, 10), (2, 20), (3, 30), (NULL, 40)",
        "create or replace temporary view u(k, d, y) as values \
         (1, 1, 5), (1, 2, 7), (2, 1, 9), (NULL, 1, 11)",
    ] {
        run(&s, q)?;
    }
    Ok(s)
}

#[test]
fn a_grouped_having_reads_null_for_no_group_and_for_a_rejected_group() -> R {
    let s = session()?;
    // t(k) is 1, 2, 3, NULL; u holds k = 1 twice (y 5, 7), k = 2 once (y 9) and a
    // NULL group (y 11) that no outer row matches. The expected rows are worked
    // out from that data by hand (the left join is a second formulation).
    for (having, value, want) in [
        (
            "count(*) >= 0",
            "max(y)",
            ["1|7", "2|9", "3|NULL", "NULL|NULL"],
        ),
        (
            "count(*) >= 2",
            "max(y)",
            ["1|7", "2|NULL", "3|NULL", "NULL|NULL"],
        ),
        (
            "max(y) > 8",
            "max(y) + 1",
            ["1|NULL", "2|10", "3|NULL", "NULL|NULL"],
        ),
        (
            "count(*) >= 1",
            "count(*)",
            ["1|2", "2|1", "3|NULL", "NULL|NULL"],
        ),
    ] {
        let sub = format!(
            "select k, (select {value} from u where u.k = t.k group by u.k having {having}) from t"
        );
        let join = format!(
            "select t.k, m.v from t left join \
             (select k, {value} as v from u group by k having {having}) m on t.k = m.k"
        );
        let (a, b) = (run(&s, &sub)?, run(&s, &join)?);
        assert_eq!(a, b, "{sub}");
        assert_eq!(a, rows(&want), "{sub}");
        // The unmatched outer rows (k = 3 and NULL) read NULL.
        assert!(a.contains(&"3|NULL".to_owned()), "{sub}: {a:?}");
        assert!(a.contains(&"NULL|NULL".to_owned()), "{sub}: {a:?}");
    }
    Ok(())
}

#[test]
fn a_key_the_correlation_does_not_fix_stays_refused() -> R {
    let s = session()?;
    let q = "select k, (select max(y) from u where u.k = t.k group by u.k, u.d having count(*) >= 1) from t";
    let e = run(&s, q).expect_err("loose key").to_string();
    assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{e}");
    Ok(())
}

/// Spark's legacy count handling reads a `count` value over no rows as 0
/// (with `HAVING count(*) >= 0`); the single join gives NULL, so it is refused.
#[test]
fn the_legacy_count_flag_refuses_a_count_value() -> R {
    let s = session()?;
    run(
        &s,
        "set spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled=true",
    )?;
    let q = "select k, (select count(*) from u where u.k = t.k group by u.k having count(*) >= 0) from t";
    let e = run(&s, q).expect_err("legacy count").to_string();
    assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{e}");
    Ok(())
}

/// The explicit left-join form of `select k, (select <value> from u where
/// u.k = t.k group by <keys> having <having>) from t`.
fn joined(value: &str, having: &str) -> String {
    format!(
        "select t.k, m.v from t left join \
         (select k, {value} as v from u group by k having {having}) m on t.k = m.k"
    )
}

#[test]
fn a_constant_group_key_is_fixed_like_the_correlated_one() -> R {
    let s = session()?;
    let sub = "select k, (select max(y) from u where u.k = t.k group by u.k, 'x' \
               having count(*) >= 1) from t";
    assert_eq!(
        run(&s, sub)?,
        run(&s, &joined("max(y)", "count(*) >= 1"))?,
        "{sub}"
    );
    Ok(())
}

/// Spark's legacy flag changes only a value that is not NULL over no rows, so a
/// `max` value answers as it does by default.
#[test]
fn the_legacy_count_flag_leaves_a_null_over_no_rows_value_alone() -> R {
    let s = session()?;
    let sub = "select k, (select max(y) from u where u.k = t.k group by u.k \
               having count(*) >= 1) from t";
    let default = run(&s, sub)?;
    run(
        &s,
        "set spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled=true",
    )?;
    assert_eq!(run(&s, sub)?, default, "{sub}");
    assert_eq!(
        default,
        run(&s, &joined("max(y)", "count(*) >= 1"))?,
        "{sub}"
    );
    Ok(())
}

/// A HAVING that reads the outer query or holds a subquery, and a correlation
/// that is not an equality, are not the fixed-key shape: refused as before.
#[test]
fn the_other_having_shapes_stay_refused() -> R {
    let s = session()?;
    for (q, why) in [
        (
            "select k, (select max(y) from u where u.k = t.k group by u.k having count(*) > t.a) from t",
            "a correlated scalar subquery of this shape",
        ),
        (
            "select k, (select max(y) from u where u.k = t.k group by u.k \
             having count(*) >= (select 1)) from t",
            "a correlated scalar subquery of this shape",
        ),
        // `pull` and the key check report their own refusal, not the generic one.
        (
            "select k, (select max(y) from u where u.k < t.k group by u.k having count(*) >= 1) from t",
            "correlated scalar subqueries whose correlation is not an equality",
        ),
    ] {
        let e = run(&s, q).expect_err(q).to_string();
        assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{q}: {e}");
        assert!(e.contains(why), "{q}: {e}");
    }
    Ok(())
}

/// An aggregate call that reads a column of its own query and an outer column,
/// in the `HAVING` of the subquery's own aggregate, is Spark's
/// `AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES` (invalid-correlation.sql
/// #5 and #6 hold Spark 4.2.0's class and the printed call).
#[test]
fn a_mixed_aggregate_in_a_subquerys_own_having_is_the_mixed_reference_error() -> R {
    let s = session()?;
    run(
        &s,
        "create or replace temporary view t3(k, z) as values (1, 1), (2, 2)",
    )?;
    for (q, call) in [
        (
            "select k from t group by 1 having exists \
             (select y from u group by 1 having y < min(a + y))",
            "min((outer(t.a) + u.y))",
        ),
        (
            "select k from t where k in (select k from u where exists \
             (select 1 from t3 group by 1 having min(d + z) > 1))",
            "min((outer(u.d) + t3.z))",
        ),
    ] {
        let e = run(&s, q).expect_err(q).to_string();
        assert!(
            e.contains(
                "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES"
            ),
            "{q}: {e}"
        );
        assert!(e.contains(call), "{q}: {e}");
    }
    Ok(())
}

/// `SELECT DISTINCT y` is a grouping by `y` with a bare column as its value
/// (scalar-subquery-predicate.sql #58 holds Spark 4.2.0's empty answer): one
/// distinct value per outer row is that value, none is NULL, and two raise
/// `SCALAR_SUBQUERY_TOO_MANY_ROWS` when the query runs.
#[test]
fn a_distinct_value_reads_the_one_value_or_fails_on_two() -> R {
    let s = session()?;
    let one = "select k, (select distinct y from u where u.k = t.k and y > 6) from t";
    let join = "select t.k, m.y from t left join \
                (select distinct k, y from u where y > 6) m on t.k = m.k";
    assert_eq!(run(&s, one)?, run(&s, join)?, "{one}");
    assert_eq!(
        run(&s, one)?,
        ["1|7", "2|9", "3|NULL", "NULL|NULL"]
            .map(String::from)
            .to_vec()
    );
    let two = "select k, (select distinct y from u where u.k = t.k) from t";
    let e = run(&s, two).expect_err(two).to_string();
    assert!(e.contains("SCALAR_SUBQUERY_TOO_MANY_ROWS"), "{two}: {e}");
    Ok(())
}

/// A grouping key that is an expression of an inner column the correlation
/// does not equate to an outer one is not fixed: refused. Under the legacy
/// count flag a value whose result over no rows is unknown (`collect_list`) is
/// refused with its own message.
#[test]
fn an_expression_key_and_an_unknown_legacy_value_are_refused() -> R {
    let s = session()?;
    let q = "select k, (select max(y) from u where u.k = t.k \
             group by u.k, y + 1 having count(*) >= 1) from t";
    let e = run(&s, q).expect_err(q).to_string();
    assert!(
        e.contains("a correlated scalar subquery of this shape"),
        "{q}: {e}"
    );
    let q = "select k, (select collect_list(y) from u where u.k = t.k \
             group by u.k having count(*) >= 1) from t";
    // Lists print as their type: k = 1 and 2 have a group, 3 and NULL do not.
    let shown: Vec<String> = run(&s, q)?
        .into_iter()
        .map(|r| {
            if r.contains("<List") {
                r.split('|').next().unwrap_or("").to_owned() + "|LIST"
            } else {
                r
            }
        })
        .collect();
    assert_eq!(
        shown,
        rows(&["1|LIST", "2|LIST", "3|NULL", "NULL|NULL"]),
        "{q}"
    );
    run(
        &s,
        "set spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled=true",
    )?;
    let e = run(&s, q).expect_err(q).to_string();
    assert!(
        e.contains("whose value over no rows is not NULL or a count"),
        "{q}: {e}"
    );
    Ok(())
}

/// `SELECT DISTINCT u.k` where `u.k = t.k`: the value is the correlated key
/// itself, so the grouping column is also the join key. Rows are worked out from
/// the data (t.k is 1, 2, 3, NULL; u.k is 1, 1, 2, NULL, and the NULL key never
/// matches an outer row) and match the explicit form. The column's type and
/// nullability are not checked here.
#[test]
fn a_distinct_of_the_correlated_key_is_the_key_or_null() -> R {
    let s = session()?;
    let sub = "select k, (select distinct u.k from u where u.k = t.k) from t";
    let join = "select t.k, m.k from t left join (select distinct k from u) m on t.k = m.k";
    assert_eq!(run(&s, sub)?, run(&s, join)?, "{sub}");
    assert_eq!(
        run(&s, sub)?,
        rows(&["1|1", "2|2", "3|NULL", "NULL|NULL"]),
        "{sub}"
    );
    Ok(())
}

/// KNOWN GAP, not Spark behaviour: under the legacy count flag Spark most likely
/// answers a bare grouping column (NULL over an empty group, as the other
/// non-count values), but the value over no rows of a bare column is not worked
/// out here, so Krishiv refuses it rather than guess. Without the flag it is
/// answered. A Spark run should decide whether the refusal can go.
#[test]
fn the_legacy_flag_refuses_a_bare_column_value() -> R {
    let s = session()?;
    let q = "select k, (select distinct y from u where u.k = t.k and y > 6) from t";
    assert_eq!(
        run(&s, q)?,
        rows(&["1|7", "2|9", "3|NULL", "NULL|NULL"]),
        "{q}"
    );
    run(
        &s,
        "set spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled=true",
    )?;
    let e = run(&s, q).expect_err(q).to_string();
    assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{q}: {e}");
    assert!(
        e.contains("whose value over no rows is not NULL or a count"),
        "{q}: {e}"
    );
    Ok(())
}
