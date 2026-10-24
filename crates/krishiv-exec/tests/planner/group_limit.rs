// SPDX-License-Identifier: Apache-2.0
//! Spark's window group limit (`InferWindowGroupLimit`): where a filter on
//! `row_number`, `rank` or `dense_rank` plans a `WindowGroupLimit` below the
//! window (conditions checked against Spark 4.2.0's plans), and that the
//! limited plans return Spark's rows, ties included.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_exec::planner::{DefaultPlanner, ExplainNode, PhysicalPlanner as _, PlanningContext};

use super::{Env, Runnable, env, run};

/// The plan of `sql` as `EXPLAIN` prints it, under `threshold`
/// (`spark.sql.optimizer.windowGroupLimitThreshold`; `None`: unset).
fn explain_with(env: &Env, sql: &str, threshold: Option<&str>) -> String {
    plan_with(env, sql, threshold).explain().to_string()
}

fn explain(env: &Env, sql: &str) -> String {
    explain_with(env, sql, None)
}

fn plan_with(env: &Env, sql: &str, threshold: Option<&str>) -> Runnable {
    let mut config = Config::new();
    if let Some(t) = threshold {
        config
            .set("spark.sql.optimizer.windowGroupLimitThreshold", t)
            .unwrap();
    }
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let plan = krishiv_optimizer::optimize(&env.analyze(sql)).unwrap();
    let plan = DefaultPlanner
        .plan(&plan, &cx)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    env.runnable(plan)
}

/// The names of the nodes of `n`, depth first.
fn names(n: &ExplainNode, out: &mut Vec<String>) {
    out.push(n.name.clone());
    for c in &n.children {
        names(c, out);
    }
}

/// The group limit's `EXPLAIN` line of `sql`, if one is planned.
fn group_limit(env: &Env, sql: &str) -> Option<String> {
    let plan = plan_with(env, sql, None);
    find(plan.explain(), "WindowGroupLimit")
}

fn find(n: &ExplainNode, name: &str) -> Option<String> {
    if n.name == name {
        return Some(n.detail.clone());
    }
    n.children.iter().find_map(|c| find(c, name))
}

/// `SELECT * FROM (SELECT a, b, <f> OVER (<w>) AS r, <more> FROM t) WHERE <cond>`.
fn query(f: &str, w: &str, more: &str, cond: &str) -> String {
    format!("SELECT * FROM (SELECT a, b, {f} OVER ({w}) AS r{more} FROM t) WHERE {cond}")
}

const W: &str = "PARTITION BY a ORDER BY b";

#[test]
fn a_rank_filter_limits_each_partition_below_the_window() {
    let env = env();
    insta::assert_snapshot!(explain(&env, &query("row_number()", W, "", "r <= 2")), @"
    Project [a#0, b#1, _we0#4 AS r#3]
    +- Filter (_we0#4 <= 2)
       +- Window [_we0], partition=[a#0], order=[b#1 ASC NULLS FIRST]
          +- WindowGroupLimit [a#0], [b#1 ASC NULLS FIRST], row_number(), 2
             +- Sort [a#0 ASC NULLS FIRST, b#1 ASC NULLS FIRST]
                +- Exchange hashpartitioning(a#0)
                   +- Scan spark_catalog.default.t [a#0, b#1]
    ");
    insta::assert_snapshot!(
        explain(&env, &query("rank()", "ORDER BY b DESC", "", "r < 3")),
        @"
    Project [a#0, b#1, _we0#4 AS r#3]
    +- Filter (_we0#4 < 3)
       +- Window [_we0], partition=[], order=[b#1 DESC NULLS LAST]
          +- WindowGroupLimit [], [b#1 DESC NULLS LAST], rank(), 2
             +- Sort [b#1 DESC NULLS LAST]
                +- Exchange SinglePartition
                   +- Scan spark_catalog.default.t [a#0, b#1]
    "
    );
    // `row_number` without partitioning: Spark's top-K path.
    insta::assert_snapshot!(
        explain(&env, &query("row_number()", "ORDER BY b", "", "r <= 2")),
        @"
    Project [a#0, b#1, _we0#4 AS r#3]
    +- Filter (_we0#4 <= 2)
       +- Window [_we0], partition=[], order=[b#1 ASC NULLS FIRST]
          +- TopN 2, [b#1 ASC NULLS FIRST]
             +- Exchange SinglePartition
                +- LocalTopN 2, [b#1 ASC NULLS FIRST]
                   +- Scan spark_catalog.default.t [a#0, b#1]
    "
    );
}

#[test]
fn every_limit_form_spark_reads() {
    let env = env();
    let limit = |f: &str, cond: &str| group_limit(&env, &query(f, W, "", cond));
    let detail = |f: &str, k: u64| Some(format!("[a#0], [b#1 ASC NULLS FIRST], {f}(), {k}"));
    for (cond, k) in [
        ("r <= 2", 2),
        ("r < 3", 2),
        ("r = 2", 2),
        ("2 >= r", 2),
        ("3 > r", 2),
        ("2 = r", 2),
        // The smallest limit of the conjuncts; other conjuncts stay.
        ("r <= 5 AND r <= 2 AND b > 3", 2),
        ("r BETWEEN 1 AND 2", 2),
        // Spark's `UnwrapCastInBinaryComparison` reads these over `INT`.
        ("r <= 2L", 2),
        ("r < 2.5", 2),
        ("r <= 2.5", 2),
        ("r < 3.0D", 2),
        ("r = 2L", 2),
        // The threshold, inclusive.
        ("r <= 1000", 1000),
    ] {
        assert_eq!(
            limit("row_number()", cond),
            detail("row_number", k),
            "{cond}"
        );
    }
    assert_eq!(limit("rank()", "r <= 2"), detail("rank", 2));
    assert_eq!(limit("dense_rank()", "r <= 2"), detail("dense_rank", 2));
    for cond in [
        "r >= 2",
        "r > 2",
        "r <= 2 OR b > 3",
        "r != 2",
        "r + 0 <= 2",
        "r = 2.5",
        "r <= 1001",
        "r <= 3000000000L",
        // Spark's `Int`: `MinValue - 1` wraps to a limit past the threshold.
        "r < -2147483648",
    ] {
        assert_eq!(limit("row_number()", cond), None, "{cond}");
    }
}

#[test]
fn row_number_wins_then_the_smallest_limit() {
    let env = env();
    let sql = |cond: &str| {
        format!(
            "SELECT * FROM (SELECT a, b, rank() OVER ({W}) AS r, \
             row_number() OVER ({W}) AS n, dense_rank() OVER ({W}) AS d FROM t) WHERE {cond}"
        )
    };
    let detail = |f: &str, k: u64| Some(format!("[a#0], [b#1 ASC NULLS FIRST], {f}(), {k}"));
    assert_eq!(
        group_limit(&env, &sql("r <= 1 AND n <= 5")),
        detail("row_number", 5)
    );
    assert_eq!(
        group_limit(&env, &sql("r <= 3 AND d <= 2")),
        detail("dense_rank", 2)
    );
    // The first of equal limits.
    assert_eq!(
        group_limit(&env, &sql("d <= 2 AND r <= 2")),
        detail("rank", 2)
    );
    // `row_number`'s limit past the threshold: no limit, even with a
    // smaller `rank` one.
    assert_eq!(group_limit(&env, &sql("r <= 1 AND n <= 1001")), None);
}

#[test]
fn only_running_rows_frames_without_size_or_look_around() {
    let env = env();
    let with = |more: &str| group_limit(&env, &query("row_number()", W, more, "r <= 2"));
    let running = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW";
    for more in [
        format!(", count(b) OVER ({W} {running}) AS c"),
        format!(", max(b) OVER ({W} {running}) AS c"),
        format!(", first(b) OVER ({W} {running}) AS c"),
        format!(", sum(b) OVER ({W} {running}) AS c"),
    ] {
        assert!(with(&more).is_some(), "{more}");
    }
    for more in [
        // The default frame of an ordered aggregate is `RANGE`.
        format!(", sum(b) OVER ({W}) AS c"),
        ", sum(b) OVER (PARTITION BY a) AS c".to_owned(),
        format!(
            ", sum(b) OVER ({W} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS c"
        ),
        format!(", ntile(2) OVER ({W}) AS c"),
        format!(", percent_rank() OVER ({W}) AS c"),
        format!(", cume_dist() OVER ({W}) AS c"),
    ] {
        assert_eq!(with(&more), None, "{more}");
    }
}

#[test]
fn the_threshold_and_bounded_inputs() {
    let env = env();
    let sql = query("rank()", W, "", "r <= 2");
    let has = |threshold: Option<&str>| {
        let mut out = Vec::new();
        names(plan_with(&env, &sql, threshold).explain(), &mut out);
        out.iter().any(|n| n == "WindowGroupLimit")
    };
    assert!(has(None));
    assert!(has(Some("2")));
    assert!(!has(Some("1")));
    assert!(!has(Some("-1")));
    // Spark refuses values below -1; they turn the rule off here.
    assert!(!has(Some("-5")));
    // An input of at most `k` rows: nothing to drop.
    let bounded = |n: u32| {
        group_limit(
            &env,
            &format!(
                "SELECT * FROM (SELECT a, b, rank() OVER ({W}) AS r \
                 FROM (SELECT * FROM t LIMIT {n})) WHERE r <= 2"
            ),
        )
    };
    assert_eq!(bounded(2), None);
    assert!(bounded(3).is_some());
    // `row_number` without partitioning past the top-N's rows: a group
    // limit of the one partition.
    let top = query("row_number()", "ORDER BY b", "", "r <= 70000");
    assert_eq!(
        find(
            plan_with(&env, &top, Some("100000")).explain(),
            "WindowGroupLimit"
        ),
        Some("[], [b#1 ASC NULLS FIRST], row_number(), 70000".to_owned())
    );
}

#[test]
fn a_limit_below_one_keeps_nothing() {
    let env = env();
    insta::assert_snapshot!(explain(&env, &query("rank()", W, "", "r <= 0")), @"
    Project [a#0, b#1, _we0#4 AS r#3]
    +- Filter (_we0#4 <= 0)
       +- Window [_we0], partition=[a#0], order=[b#1 ASC NULLS FIRST]
          +- Sort [a#0 ASC NULLS FIRST, b#1 ASC NULLS FIRST]
             +- Exchange hashpartitioning(a#0)
                +- LocalLimit 0
                   +- Scan spark_catalog.default.t [a#0, b#1]
    ");
    let (_, rows) = env.rows(&query("rank()", W, "", "r < 1"));
    assert!(rows.is_empty());
}

/// The rows of `sql`, sorted.
fn sorted_rows(env: &Env, sql: &str, threshold: Option<&str>) -> Vec<String> {
    let plan = plan_with(env, sql, threshold);
    let batches = run(&plan).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| super::cell(c.as_ref(), r))
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

/// Spark 4.2.0's rows for each query (over the same `t`), with the group
/// limit and without it.
#[test]
fn limited_plans_return_sparks_rows() {
    let env = env();
    let base = "(SELECT a % 2 AS p, b DIV 40 AS o, b FROM t)";
    let cases: [(String, &[&str]); 7] = [
        (
            format!(
                "SELECT p, o, b, r FROM (SELECT p, o, b, rank() OVER (PARTITION BY p ORDER BY o) \
                 AS r FROM {base}) WHERE r <= 2"
            ),
            &[
                "0|0|0|1",
                "0|0|20|1",
                "1|0|10|1",
                "1|1|50|2",
                "NULL|0|30|1",
                "NULL|1|70|2",
            ],
        ),
        (
            format!(
                "SELECT p, o, b, r FROM (SELECT p, o, b, dense_rank() OVER (PARTITION BY p \
                 ORDER BY o DESC) AS r FROM {base}) WHERE r < 2"
            ),
            &["0|2|80|1", "1|2|90|1", "NULL|1|70|1"],
        ),
        (
            format!(
                "SELECT p, o, r FROM (SELECT p, o, row_number() OVER (PARTITION BY p ORDER BY o) \
                 AS r FROM {base}) WHERE 2 >= r"
            ),
            &["0|0|1", "0|0|2", "1|0|1", "1|1|2", "NULL|0|1", "NULL|1|2"],
        ),
        (
            format!(
                "SELECT b, r FROM (SELECT b, rank() OVER (ORDER BY o DESC) AS r FROM {base}) \
                 WHERE r <= 3"
            ),
            &["40|3", "50|3", "60|3", "70|3", "80|1", "90|1"],
        ),
        (
            format!(
                "SELECT b, r FROM (SELECT b, row_number() OVER (ORDER BY o DESC, b) AS r \
                 FROM {base}) WHERE r <= 3"
            ),
            &["40|3", "80|1", "90|2"],
        ),
        (
            format!(
                "SELECT p, b, r, c FROM (SELECT p, b, rank() OVER (PARTITION BY p ORDER BY o, b) \
                 AS r, count(b) OVER (PARTITION BY p ORDER BY o, b ROWS BETWEEN UNBOUNDED \
                 PRECEDING AND CURRENT ROW) AS c FROM {base}) WHERE r <= 2"
            ),
            &[
                "0|0|1|1",
                "0|20|2|2",
                "1|10|1|1",
                "1|50|2|2",
                "NULL|30|1|1",
                "NULL|70|2|2",
            ],
        ),
        (
            format!(
                "SELECT p, b, r FROM (SELECT p, b, rank() OVER (PARTITION BY p ORDER BY o) AS r \
                 FROM {base}) WHERE r <= 0"
            ),
            &[],
        ),
    ];
    for (sql, want) in &cases {
        assert_eq!(sorted_rows(&env, sql, None), *want, "{sql}");
        assert_eq!(
            sorted_rows(&env, sql, Some("-1")),
            *want,
            "{sql} without the limit"
        );
    }
}
