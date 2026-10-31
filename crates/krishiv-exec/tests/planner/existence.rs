// SPDX-License-Identifier: Apache-2.0
//! Correlated `EXISTS` / `IN` inside `OR`, `NOT` and select lists: one
//! existence join per subquery (P2-T115 unit 2a). Expected rows follow
//! Spark 4.2: `exists` is two-valued, so a NULL key never matches.

use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::block_on;
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer};

use super::{Env, env};

/// `t(a, b, s)` of [`env`]: `a` is 0, 1, 2, NULL, 4, 5, 6, NULL, 8, 9 for
/// `b` = 0, 10, ..., 90 and `s` = `s0` .. `s9`.
fn rows(env: &Env, sql: &str) -> Vec<String> {
    env.rows(sql).1
}

#[test]
fn exists_under_or_keeps_the_rows_either_side_matches() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT a FROM t x WHERE EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b >= 50) \
             OR x.s = 's0' ORDER BY a"
        ),
        ["0", "5", "6", "8", "9"]
    );
}

#[test]
fn a_select_list_exists_is_false_for_a_null_key() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT a, EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b >= 50) AS e \
             FROM t x ORDER BY a"
        ),
        [
            "NULL|false",
            "NULL|false",
            "0|false",
            "1|false",
            "2|false",
            "4|false",
            "5|true",
            "6|true",
            "8|true",
            "9|true",
        ]
    );
}

#[test]
fn in_under_or_and_not_in_under_or_follow_spark() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT a FROM t WHERE a IN (SELECT a FROM t WHERE b >= 50) OR s = 's1' ORDER BY a"
        ),
        ["1", "5", "6", "8", "9"]
    );
    // The subquery holds no NULL: `NOT IN` is true for the other non-NULL
    // keys, and `s3` (a NULL key) is kept by the other disjunct only.
    assert_eq!(
        rows(
            &env,
            "SELECT a FROM t WHERE a NOT IN (SELECT a FROM t WHERE b >= 80) OR s = 's3' ORDER BY a"
        ),
        ["NULL", "0", "1", "2", "4", "5", "6"]
    );
    // A NULL in the subquery makes `NOT IN` NULL for every unmatched key.
    assert_eq!(
        rows(
            &env,
            "SELECT a FROM t WHERE a NOT IN (SELECT a FROM t WHERE b >= 50) OR s = 's1' ORDER BY a"
        ),
        ["1"]
    );
}

#[test]
fn not_exists_under_or_keeps_the_unmatched_rows() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT a FROM t x WHERE NOT EXISTS (SELECT 1 FROM t y WHERE y.a = x.a) \
             OR x.s = 's0' ORDER BY a"
        ),
        ["NULL", "NULL", "0"]
    );
}

#[test]
fn an_outer_only_correlated_conjunct_decides_exists() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT a FROM t x WHERE EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND x.b > 50) \
             OR x.s = 's0' ORDER BY a"
        ),
        ["0", "6", "8", "9"]
    );
}

#[test]
fn two_existence_joins_under_one_or() {
    let env = env();
    let sql = "SELECT a FROM t x WHERE a IS NOT NULL \
               AND EXISTS (SELECT 1 FROM t y WHERE y.a = x.a) \
               AND (EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b >= 80) \
                    OR EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b = 0)) ORDER BY a";
    assert_eq!(rows(&env, sql), ["0", "8", "9"]);
    let plan = format!("{}", env.analyze(sql));
    assert_eq!(plan.matches("ExistenceJoin").count(), 2, "{plan}");
    assert_eq!(plan.matches("LeftSemi").count(), 1, "{plan}");
}

#[test]
fn a_subquery_written_twice_is_one_join_per_expression() {
    let env = env();
    let sql = "SELECT a FROM t x WHERE EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b >= 80) \
               OR NOT EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b >= 80) ORDER BY a";
    assert_eq!(rows(&env, sql).len(), 10);
    let plan = format!("{}", env.analyze(sql));
    assert_eq!(plan.matches("ExistenceJoin").count(), 2, "{plan}");
}

#[test]
fn a_user_column_named_exists_never_collides() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT s, EXISTS (SELECT 1 FROM t y WHERE y.a = x.a AND y.b >= 80) AS `exists` \
             FROM t x WHERE b >= 80 ORDER BY s"
        ),
        ["s8|true", "s9|true"]
    );
}

#[test]
fn forms_spark_folds_before_any_join_are_refused() {
    let env = env();
    for sql in [
        "SELECT a FROM t WHERE NULL IN (SELECT a FROM t) OR s = 's0'",
        "SELECT a FROM t WHERE a IN (SELECT 1) OR s = 's0'",
        "SELECT a FROM t x WHERE NOT (x.a NOT IN (SELECT a FROM t)) OR s = 's0'",
    ] {
        let plan =
            krishiv_sql::parse_and_lower(sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap();
        let e = block_on(DefaultAnalyzer.analyze(&plan, &env.ctx)).unwrap_err();
        assert_eq!(
            e.error.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}: {e}"
        );
    }
}

#[test]
fn an_existence_join_drops_no_left_row_through_a_limit() {
    let env = env();
    assert_eq!(
        rows(
            &env,
            "SELECT s FROM (SELECT s, EXISTS (SELECT 1 FROM t y WHERE y.a = x.a) AS e FROM t x) \
             WHERE e OR s = 's3' ORDER BY s LIMIT 20"
        )
        .len(),
        9
    );
}

/// The data of the design note's Spark 4.2.0 probe (`c`, `s`, `w`) as CTEs.
const PROBE: &str = "WITH c AS (SELECT * FROM VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, NULL), (NULL, 'e') AS c(ck, g)), \
     s AS (SELECT * FROM VALUES (1, 10), (1, 11), (2, NULL), (NULL, 5) AS s(sk, v)), \
     w AS (SELECT * FROM VALUES (3, 1), (2, 2), (NULL, 3) AS w(wk, v)) ";

fn probe(env: &Env, body: &str) -> Vec<String> {
    let mut r = rows(env, &format!("{PROBE}{body}"));
    r.sort();
    r
}

/// Rows of Spark 4.2.0 for each shape of the design note's section 2 table
/// (`E_s` is `EXISTS (SELECT 1 FROM s WHERE s.sk = c.ck)`, `E_w` likewise).
#[test]
fn the_spark_probe_rows_match() {
    let env = env();
    let es = "EXISTS (SELECT 1 FROM s WHERE s.sk = c.ck)";
    let ew = "EXISTS (SELECT 1 FROM w WHERE w.wk = c.ck)";
    let filter = |p: &str| format!("SELECT ck FROM c WHERE {p}");
    let cases: Vec<(String, Vec<&str>)> = vec![
        (filter(&format!("{es} OR {ew}")), vec!["1", "2", "3"]),
        (
            filter(&format!("ck > 0 AND {es} AND ({ew} OR g = 'a')")),
            vec!["1", "2"],
        ),
        (
            filter(&format!("NOT {es} OR g = 'c'")),
            vec!["3", "4", "NULL"],
        ),
        (
            filter("ck IN (SELECT sk FROM s WHERE s.v > c.ck) OR g = 'b'"),
            vec!["1", "2"],
        ),
        (
            filter("ck IN (SELECT sk FROM s) OR g IS NULL"),
            vec!["1", "2", "4"],
        ),
        (
            filter("ck NOT IN (SELECT wk FROM w WHERE w.v >= 1) OR g = 'a'"),
            vec!["1"],
        ),
        (
            filter("EXISTS (SELECT 1 FROM s WHERE s.sk = c.ck AND s.v > 10) OR g = 'c'"),
            vec!["1", "3"],
        ),
        (
            filter("EXISTS (SELECT 1 FROM s WHERE s.sk > c.ck) OR g = 'e'"),
            vec!["1", "NULL"],
        ),
        (
            filter(&format!("({es} AND g = 'a') OR ({ew} AND g = 'c')")),
            vec!["1", "3"],
        ),
        (
            filter(&format!("{es} OR NOT {es}")),
            vec!["1", "2", "3", "4", "NULL"],
        ),
        (
            filter(&format!("{es} OR {ew} OR ck IS NULL")),
            vec!["1", "2", "3", "NULL"],
        ),
        (
            filter("(ck, g) IN (SELECT sk, 'a' FROM s WHERE s.v > 0) OR g = 'c'"),
            vec!["1", "3"],
        ),
        (
            format!("SELECT CASE WHEN {es} THEN 'y' ELSE 'n' END FROM c"),
            vec!["n", "n", "n", "y", "y"],
        ),
        (
            format!("SELECT ck, {es} AS e FROM c"),
            vec!["1|true", "2|true", "3|false", "4|false", "NULL|false"],
        ),
        (
            format!("SELECT ck FROM c GROUP BY ck HAVING {es} OR ck = 3"),
            vec!["1", "2", "3"],
        ),
        // A join condition: the existence join sits under the side it reads.
        (
            "SELECT c.ck FROM c JOIN w ON c.ck = w.wk AND (EXISTS (SELECT 1 FROM s WHERE \
             s.sk = c.ck) OR w.v = 2)"
                .to_owned(),
            vec!["2"],
        ),
        (
            "SELECT c.ck, w.wk FROM c LEFT JOIN w ON c.ck = w.wk AND EXISTS (SELECT 1 FROM s \
             WHERE s.sk = c.ck)"
                .to_owned(),
            vec!["1|NULL", "2|2", "3|NULL", "4|NULL", "NULL|NULL"],
        ),
    ];
    for (sql, want) in cases {
        let mut want: Vec<String> = want.into_iter().map(str::to_owned).collect();
        want.sort();
        assert_eq!(probe(&env, &sql), want, "{sql}");
    }
}

#[test]
fn shapes_that_need_unnesting_stay_refused() {
    let env = env();
    for body in [
        // Correlation below an aggregate (T115 units 2 and 3).
        "SELECT ck FROM c WHERE EXISTS (SELECT max(v) FROM s WHERE s.sk = c.ck) OR g = 'zz'",
        "SELECT ck FROM c WHERE EXISTS (SELECT sk FROM s WHERE s.sk = c.ck GROUP BY sk \
         HAVING count(*) > 1) OR g = 'c'",
        // A join condition that reads both sides.
        "SELECT c.ck FROM c JOIN w ON c.ck = w.wk AND EXISTS (SELECT 1 FROM s WHERE s.sk = c.ck \
         AND s.v = w.v)",
    ] {
        let sql = format!("{PROBE}{body}");
        let plan =
            krishiv_sql::parse_and_lower(&sql, &krishiv_sql::ParserOptions::default(), SourceId(0))
                .unwrap();
        let e = block_on(DefaultAnalyzer.analyze(&plan, &env.ctx)).unwrap_err();
        assert_eq!(
            e.error.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{body}: {e}"
        );
    }
}

/// Re-analysing a rewritten plan changes nothing and mints no second
/// `exists` id, also when the pruned plan leaves an `exists` unread.
#[test]
fn existence_plans_are_a_fixed_point_of_analysis() {
    let env = env();
    for sql in [
        "SELECT a FROM t x WHERE EXISTS (SELECT 1 FROM t y WHERE y.a = x.a) OR x.s = 's0'",
        "SELECT a, EXISTS (SELECT 1 FROM t y WHERE y.a = x.a) AS e FROM t x",
        "SELECT a FROM (SELECT a, EXISTS (SELECT 1 FROM t y WHERE y.a = x.a) AS e FROM t x)",
        "SELECT a FROM t WHERE a NOT IN (SELECT a FROM t) OR s = 's1'",
    ] {
        let first = env.analyzed(sql);
        let again = block_on(DefaultAnalyzer.analyze(&first.plan, &env.ctx)).unwrap();
        assert!(again.plan.structurally_eq(&first.plan), "{sql}");
        // Pruned (the `exists` of a projection nobody reads is dropped
        // from the output, the join stays): analysis still works.
        let optimized = krishiv_optimizer::optimize(&first.plan).unwrap();
        let third = block_on(DefaultAnalyzer.analyze(&optimized, &env.ctx)).unwrap();
        assert!(third.plan.structurally_eq(&optimized), "{sql}");
    }
}
