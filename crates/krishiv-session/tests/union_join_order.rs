// SPDX-License-Identifier: Apache-2.0
//! A `UNION ALL` and a recursive CTE read their branches by position: a
//! branch whose joins the optimizer reorders, or whose filter common
//! subexpression elimination widens, must still give its columns in the
//! order it wrote them. The views and the first statement are Spark's
//! `join-lateral.sql` ("lateral join under union"); every expected answer
//! is what a Spark 4.2.0 run returned.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
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
                            } else if c.data_type() == &DataType::Int64 {
                                c.as_primitive::<Int64Type>().value(r).to_string()
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

#[test]
fn a_recursive_cte_reads_a_reordered_anchor_by_position() -> R {
    let s = session()?;
    let got = run(
        &s,
        "WITH RECURSIVE r(a, b, c, d) AS (SELECT * FROM t1 JOIN t4 \
         UNION ALL SELECT a, b, c, d + 10 FROM r WHERE d < 10) SELECT * FROM r",
    )?;
    let mut want: Vec<String> = T1_T4.iter().map(|r| (*r).to_owned()).collect();
    for r in T1_T4 {
        let (head, d) = r.rsplit_once('|').ok_or("row")?;
        want.push(format!("{head}|{}", d.parse::<i32>()? + 10));
    }
    want.sort();
    assert_eq!(got, want);
    Ok(())
}

/// The re-planned tree outputs more columns than the branch (a column
/// selection it looks through): the branch keeps its own.
#[test]
fn a_reordered_branch_with_more_columns_keeps_its_own() -> R {
    let s = session()?;
    let got = run(
        &s,
        "SELECT * FROM (SELECT c1 FROM t1) a JOIN t4 ON a.c1 = t4.c1 \
         UNION ALL SELECT c1, c1, c2 FROM t4",
    )?;
    let want = [
        "0|0|1", "0|0|1", "0|0|2", "0|0|2", "1|1|1", "1|1|1", "1|1|3", "1|1|3",
    ];
    assert_eq!(got, sorted(&want));
    Ok(())
}

/// A repeated subexpression in a recursive CTE's anchor filter: the
/// extracted column must not widen the anchor.
#[test]
fn a_recursive_cte_anchor_keeps_its_width_after_common_subexpressions() -> R {
    let s = session()?;
    let got = run(
        &s,
        "WITH RECURSIVE r(a) AS (SELECT id FROM range(4) \
         WHERE abs(id * 3) >= 0 AND abs(id * 3) < 10 \
         UNION ALL SELECT a + 1 FROM r WHERE a < 2) SELECT * FROM r",
    )?;
    assert_eq!(got, sorted(&["0", "1", "1", "2", "2", "2", "3"]));
    Ok(())
}

/// A branch that repeats a column (`SELECT x, x`) still has two columns:
/// the union's second column reads the other branches' second column
/// (Spark's `DeduplicateUnionChildOutput`, SPARK-37865). The name `x`
/// above the union is the first copy, not ambiguous. Answers from a Spark
/// 4.2.0 run.
/// A union whose first branch repeats its column.
const U: &str = "SELECT x, x FROM VALUES (1) t(x) UNION ALL SELECT 5, 6";
/// A relation that repeats its column.
const REPEAT: &str = "SELECT x, x FROM VALUES (1) t(x)";

#[test]
fn a_branch_that_repeats_a_column_keeps_both_positions() -> R {
    let s = Session::new()?;
    for (sql, want) in [
        (
            "SELECT * FROM (SELECT x, x FROM VALUES (1) t(x) UNION ALL SELECT 5, 6) u(a, b)"
                .to_owned(),
            vec!["1|1", "5|6"],
        ),
        (
            "SELECT * FROM (SELECT x, x AS y FROM VALUES (1) t(x) UNION ALL SELECT 5, 6)"
                .to_owned(),
            vec!["1|1", "5|6"],
        ),
        (
            "SELECT x FROM (SELECT x, x FROM VALUES (1) t(x) UNION ALL SELECT 5, 6)".to_owned(),
            vec!["1", "5"],
        ),
        (
            "SELECT * FROM (SELECT x, x FROM VALUES (1) t(x) UNION SELECT 5, 6)".to_owned(),
            vec!["1|1", "5|6"],
        ),
        (
            "SELECT * FROM (SELECT 5, 6 UNION ALL SELECT x, x FROM VALUES (1) t(x))".to_owned(),
            vec!["1|1", "5|6"],
        ),
        // The repeat stays a repeat through column aliases, CTE references,
        // `USING` keys and `* EXCEPT`, as Spark's attribute metadata does.
        (format!("SELECT a FROM ({U}) u(a, a)"), vec!["1", "5"]),
        (
            format!("WITH u AS ({U}) SELECT x FROM u WHERE x IN (SELECT x FROM u)"),
            vec!["1", "5"],
        ),
        (
            format!("WITH u AS ({U}) SELECT a.x, b.x FROM u a JOIN u b ON a.x = b.x"),
            vec!["1|1", "5|5"],
        ),
        (
            format!("SELECT * FROM ({U}) a JOIN (SELECT 1 AS x) b USING (x)"),
            vec!["1|1"],
        ),
        (format!("SELECT * EXCEPT (x) FROM ({U})"), vec!["1", "6"]),
        (format!("SELECT * FROM ({U}) WHERE x = 5"), vec!["5|6"]),
        (
            format!("SELECT x, count(*) FROM ({U}) GROUP BY x"),
            vec!["1|1", "5|1"],
        ),
        // Spark fails these two with an internal error; the rows are the
        // multiset difference and intersection.
        (
            format!("SELECT * FROM ({REPEAT} EXCEPT ALL SELECT 1, 2)"),
            vec!["1|1"],
        ),
        (
            format!("SELECT * FROM ({REPEAT} INTERSECT ALL SELECT 1, 2)"),
            vec![],
        ),
    ] {
        assert_eq!(run(&s, &sql)?, want, "{sql}");
    }
    Ok(())
}

/// `df["x"]` and a self-join of such a union resolve `x` to the first copy
/// (Spark 4.2.0: `[1, 5]`, `[1, 5, 7]` and `[1, 1, 5, 5]`).
#[test]
fn dataframe_references_to_a_repeated_column() -> R {
    let s = Session::new()?;
    let df = block_on(s.sql(U))?;
    let other = block_on(s.sql("SELECT 7 AS x, 8 AS y"))?;
    let rows = |d: &krishiv_session::DataFrame| -> R<Vec<String>> {
        let mut out: Vec<String> = block_on(d.collect())?
            .iter()
            .flat_map(|b| {
                let c = b.column(0).as_primitive::<Int32Type>();
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort();
        Ok(out)
    };
    assert_eq!(rows(&df.select([df.col("x")]))?, ["1", "5"]);
    assert_eq!(
        rows(&df.union(&other).select([df.col("x")]))?,
        ["1", "5", "7"]
    );
    let joined = df.alias("a").cross_join(&df.alias("b"));
    assert_eq!(
        rows(&joined.select([krishiv_session::functions::col("b.x")]))?,
        ["1", "1", "5", "5"]
    );
    assert_eq!(
        rows(&joined.select([krishiv_session::functions::col("a.x")]))?,
        ["1", "1", "5", "5"]
    );
    Ok(())
}
