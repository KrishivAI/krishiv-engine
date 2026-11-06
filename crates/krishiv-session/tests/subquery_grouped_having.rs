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
    for (having, value) in [
        ("count(*) >= 0", "max(y)"),
        ("count(*) >= 2", "max(y)"),
        ("max(y) > 8", "max(y) + 1"),
        ("count(*) >= 1", "count(*)"),
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
    for q in [
        "select k, (select max(y) from u where u.k = t.k group by u.k having count(*) > t.a) from t",
        "select k, (select max(y) from u where u.k = t.k group by u.k \
         having count(*) >= (select 1)) from t",
        "select k, (select max(y) from u where u.k < t.k group by u.k having count(*) >= 1) from t",
    ] {
        let e = run(&s, q).expect_err(q).to_string();
        assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{q}: {e}");
        assert!(
            e.contains("a correlated scalar subquery of this shape"),
            "{q}: {e}"
        );
    }
    Ok(())
}
