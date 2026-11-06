// SPDX-License-Identifier: Apache-2.0
//! Nested aggregates in a correlated scalar subquery (Spark's count bug): a
//! global aggregate below another operator has a row for an outer row without
//! matches. Every expected answer is what a Spark 4.2.0 run gave
//! (`nested-scalar-subquery-count-bug.sql`, with
//! `spark.sql.optimizer.decorrelateInnerQuery.enabled=true` and
//! `spark.sql.legacy.scalarSubqueryCountBugBehavior=false`).
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn run(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                let c = b.column(0).as_primitive::<Int64Type>();
                (0..b.num_rows()).map(move |r| {
                    if c.is_null(r) {
                        "NULL".to_owned()
                    } else {
                        c.value(r).to_string()
                    }
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
        "create or replace temporary view t1(a1, a2) as values (0, 1), (1, 2)",
        "create or replace temporary view t2(b1, b2) as values (0, 2), (0, 3)",
        "create or replace temporary view t3(c1, c2) as values (0, 2), (0, 3)",
        "set spark.sql.optimizer.decorrelateInnerQuery.enabled=true",
        "set spark.sql.legacy.scalarSubqueryCountBugBehavior=false",
    ] {
        block_on(async { s.sql(q).await?.collect().await })?;
    }
    Ok(s)
}

#[test]
fn a_sum_over_a_count_reads_the_count_of_an_unmatched_row() -> R {
    let q =
        "select (select sum(cnt) from (select count(*) cnt from t2 where t1.a1 = t2.b1)) a from t1";
    assert_eq!(run(&session()?, q)?, rows(&["2", "0"]), "{q}");
    Ok(())
}

#[test]
fn a_count_over_a_count_is_one_row_per_outer_row() -> R {
    let q =
        "select (select count(*) from (select count(*) cnt from t2 where t1.a1 = t2.b1)) a from t1";
    assert_eq!(run(&session()?, q)?, rows(&["1", "1"]), "{q}");
    Ok(())
}

#[test]
fn two_counts_joined_keep_the_unmatched_row() -> R {
    let s = session()?;
    // With HAVING only the unmatched outer row has cnt = 0.
    let q = "select (select sum(l.cnt + r.cnt) \
             from (select count(*) cnt from t2 where t1.a1 = t2.b1 having cnt = 0) l \
             join (select count(*) cnt from t3 where t1.a1 = t3.c1 having cnt = 0) r \
             on l.cnt = r.cnt) a from t1";
    assert_eq!(run(&s, q)?, rows(&["0", "NULL"]), "{q}");
    let q = "select (select sum(l.cnt + r.cnt) \
             from (select count(*) cnt from t2 where t1.a1 = t2.b1) l \
             join (select count(*) cnt from t3 where t1.a1 = t3.c1) r \
             on l.cnt = r.cnt) a from t1";
    assert_eq!(run(&s, q)?, rows(&["4", "0"]), "{q}");
    Ok(())
}
