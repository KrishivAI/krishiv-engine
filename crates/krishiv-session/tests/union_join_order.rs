// SPDX-License-Identifier: Apache-2.0
//! A `UNION ALL` reads its branches by position: a branch whose joins the
//! optimizer reorders must still give its columns in the order it wrote
//! them. The views and statements are Spark's `join-lateral.sql` ("lateral
//! join under union"); the expected rows are what Spark 4.2.0 returns.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn run(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| {
                            if c.is_null(r) {
                                "NULL".to_owned()
                            } else {
                                c.as_primitive::<Int32Type>().value(r).to_string()
                            }
                        })
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
    for v in [
        "CREATE TEMPORARY VIEW t1(c1, c2) AS VALUES (0, 1), (1, 2)",
        "CREATE TEMPORARY VIEW t2(c1, c2) AS VALUES (0, 2), (0, 3)",
        "CREATE TEMPORARY VIEW t4(c1, c2) AS VALUES (0, 1), (0, 2), (1, 1), (1, 3)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

/// `t1 JOIN t4`'s eight rows, `t1`'s columns first.
const T1_T4: [&str; 8] = [
    "0|1|0|1", "0|1|0|2", "0|1|1|1", "0|1|1|3", "1|2|0|1", "1|2|0|2", "1|2|1|1", "1|2|1|3",
];

fn sorted(rows: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = rows.iter().map(|r| (*r).to_owned()).collect();
    v.sort();
    v
}

#[test]
fn a_lateral_join_under_union_keeps_the_branch_columns_in_order() -> R {
    let s = session()?;
    let got = run(
        &s,
        "SELECT * FROM t1 JOIN LATERAL (SELECT * FROM t2 WHERE t2.c1 = t1.c1) \
         UNION ALL SELECT * FROM t1 JOIN t4",
    )?;
    let mut want = vec!["0|1|0|2", "0|1|0|3"];
    want.extend(T1_T4);
    assert_eq!(got, sorted(&want));
    Ok(())
}

#[test]
fn joins_in_both_branches_keep_their_columns_in_order() -> R {
    let s = session()?;
    let got = run(
        &s,
        "SELECT * FROM t1 JOIN t2 UNION ALL SELECT * FROM t1 JOIN t4",
    )?;
    let mut want = vec!["0|1|0|2", "0|1|0|3", "1|2|0|2", "1|2|0|3"];
    want.extend(T1_T4);
    assert_eq!(got, sorted(&want));
    // Each branch on its own gives the same rows.
    assert_eq!(run(&s, "SELECT * FROM t1 JOIN t4")?, sorted(&T1_T4));
    Ok(())
}

#[test]
fn set_operations_read_reordered_branches_by_position() -> R {
    let s = session()?;
    let intersect = run(
        &s,
        "SELECT * FROM t1 JOIN t4 INTERSECT ALL SELECT * FROM t1 JOIN t4 WHERE t4.c2 > 1",
    )?;
    assert_eq!(
        intersect,
        sorted(&["0|1|0|2", "0|1|1|3", "1|2|0|2", "1|2|1|3"])
    );
    let except = run(
        &s,
        "SELECT * FROM t1 JOIN t4 EXCEPT ALL SELECT * FROM t1 JOIN t4 WHERE t4.c2 > 1",
    )?;
    assert_eq!(
        except,
        sorted(&["0|1|0|1", "0|1|1|1", "1|2|0|1", "1|2|1|1"])
    );
    Ok(())
}
