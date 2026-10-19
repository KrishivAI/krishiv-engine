// SPDX-License-Identifier: Apache-2.0
//! One query per rule and edge case: its optimised plan snapshotted, and
//! (when the result is deterministic) the same rows as the unoptimised plan
//! on generated data.

use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_optimizer::{TopN, optimize};
use krishiv_plan::{
    Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, LogicalPlan, Plan, PlanNode,
    ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

use crate::support::{self, data, eval};

/// `(name, query)` over `t(a int, b string, c bigint)` and `u(a int, d string)`.
const CASES: &[(&str, &str)] = &[
    // Constant folding and simplification.
    (
        "fold_arith",
        "SELECT a FROM t WHERE a > 1 + 2 * 3 AND c < 9223372036854775807 + 1",
    ),
    (
        "fold_int_overflow",
        "SELECT a FROM t WHERE a > 2147483647 + 1",
    ),
    (
        "fold_negative",
        "SELECT a FROM t WHERE a > -(2) AND a < -(-2147483647 - 1)",
    ),
    (
        "fold_cast",
        "SELECT a FROM t WHERE CAST('2020-01-31' AS DATE) > CAST('2020-01-01' AS DATE) AND a > CAST('7' AS INT)",
    ),
    (
        "fold_bad_cast",
        "SELECT a FROM t WHERE a > CAST('x' AS INT) OR a > TRY_CAST('x' AS INT)",
    ),
    (
        "fold_cast_same_type",
        "SELECT a FROM t WHERE CAST(a AS INT) = 1",
    ),
    (
        "fold_compare",
        "SELECT a FROM t WHERE 'b' > 'a' AND 2L >= 2L AND 1.50 = 1.5 AND true AND a = a",
    ),
    (
        "fold_doubles",
        "SELECT a FROM t WHERE CAST('NaN' AS DOUBLE) = CAST('NaN' AS DOUBLE) AND -0.0D = 0.0D AND 1.0F < 2.0F",
    ),
    (
        "fold_false",
        "SELECT a FROM t WHERE 1 = 2 OR (a > 1 AND false)",
    ),
    (
        "fold_null",
        "SELECT a FROM t WHERE (a = 1 OR CAST(NULL AS INT) = 1) AND NOT (CAST(NULL AS INT) > a)",
    ),
    (
        "fold_null_logic",
        "SELECT a FROM t WHERE (CAST(NULL AS BOOLEAN) AND CAST(NULL AS BOOLEAN)) IS NULL AND (CAST(NULL AS BOOLEAN) OR CAST(NULL AS BOOLEAN)) IS NULL",
    ),
    (
        "fold_null_safe",
        "SELECT a FROM t WHERE CAST(NULL AS INT) <=> CAST(NULL AS INT) AND NOT (1 <=> CAST(NULL AS INT)) AND 1 <=> 1",
    ),
    (
        "fold_in",
        "SELECT a FROM t WHERE 2 IN (1, 2) AND NOT 3 IN (1, 2) AND (3 IN (1, CAST(NULL AS INT))) IS NULL",
    ),
    (
        "fold_between",
        "SELECT a FROM t WHERE 2 BETWEEN 1 AND 3 AND NOT 5 BETWEEN 1 AND 3 AND (2 BETWEEN CAST(NULL AS INT) AND 3) IS NULL",
    ),
    (
        "fold_not",
        "SELECT a FROM t WHERE NOT a < 1 AND NOT a <= 2 AND NOT a > 3 AND NOT a >= 4 AND NOT NOT b = 'x' AND NOT b = 'y'",
    ),
    (
        "fold_case",
        "SELECT CASE WHEN false THEN 1 WHEN a > 1 THEN 2 WHEN true THEN 3 ELSE 4 END AS x, CASE WHEN 1 = 1 THEN 'p' ELSE 'q' END AS y, CASE WHEN false THEN 1 END AS z FROM t",
    ),
    (
        "fold_case_kept",
        "SELECT CASE WHEN a > 1 THEN 2 ELSE 3 END AS x FROM t",
    ),
    (
        "fold_isnull",
        "SELECT a FROM t WHERE 1 IS NOT NULL AND CAST(NULL AS INT) IS NULL AND a IS NOT NULL",
    ),
    (
        "fold_dedupe",
        "SELECT a FROM t WHERE (a > 1 AND a > 1) OR (b = 'x' OR b = 'x')",
    ),
    // Predicate pushdown.
    (
        "push_project",
        "SELECT x FROM (SELECT a + 1 AS x, b FROM t) WHERE x > 2 AND b = 'k'",
    ),
    (
        "push_aggregate",
        "SELECT k, n FROM (SELECT a + 1 AS k, count(*) AS n FROM t GROUP BY a + 1) WHERE k > 2 AND n > 1",
    ),
    (
        "push_global_aggregate",
        "SELECT n FROM (SELECT count(*) AS n FROM t) WHERE n > 1",
    ),
    (
        "push_sort_alias",
        "SELECT * FROM (SELECT a, b FROM t ORDER BY a) AS x (p, q) WHERE p = 1",
    ),
    (
        "push_cte",
        "WITH v AS (SELECT a, b FROM t) SELECT * FROM v WHERE a = 1",
    ),
    (
        "push_cross",
        "SELECT t.a FROM t, u WHERE t.a = u.a AND t.b = 'x' AND u.d = 'y' AND 1 = 1",
    ),
    (
        "push_or_extract",
        "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE (t.b = 'x' AND u.d = 'y') OR (t.b = 'z' AND u.d = 'w')",
    ),
    (
        "push_or_no_extract",
        "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE t.b = 'x' OR u.d = 'y'",
    ),
    (
        "push_left",
        "SELECT t.a FROM t LEFT JOIN u ON t.a = u.a AND u.d = 'y' AND t.b = 'x' WHERE t.c > 1",
    ),
    (
        "push_left_rejects",
        "SELECT t.a FROM t LEFT JOIN u ON t.a = u.a WHERE u.d = 'y'",
    ),
    (
        "push_left_keeps",
        "SELECT t.a FROM t LEFT JOIN u ON t.a = u.a WHERE u.d IS NULL OR t.b = 'x'",
    ),
    (
        "push_right",
        "SELECT t.a FROM t RIGHT JOIN u ON t.a = u.a AND t.b = 'x' AND u.d = 'y' WHERE u.a > 1 AND t.c > 1",
    ),
    (
        "push_full",
        "SELECT t.a FROM t FULL JOIN u ON t.a = u.a AND t.b = 'x' WHERE t.c > 1",
    ),
    (
        "push_full_right",
        "SELECT t.a FROM t FULL JOIN u ON t.a = u.a WHERE u.d = 'y'",
    ),
    (
        "push_full_both",
        "SELECT t.a FROM t FULL JOIN u ON t.a = u.a WHERE u.d = 'y' AND t.b = 'x'",
    ),
    (
        "push_semi",
        "SELECT a FROM t LEFT SEMI JOIN u ON t.a = u.a AND u.d = 'y' AND t.b = 'x' WHERE t.c > 1",
    ),
    (
        "push_anti",
        "SELECT a FROM t LEFT ANTI JOIN u ON t.a = u.a AND u.d = 'y' AND t.b = 'x' WHERE t.c > 1",
    ),
    (
        "push_self_join",
        "SELECT x.a FROM t x JOIN t y ON x.a = y.c WHERE y.b = 'q'",
    ),
    (
        "push_subquery",
        "SELECT a FROM t WHERE a IN (SELECT a FROM u WHERE d = 'y') AND EXISTS (SELECT 1 FROM u WHERE u.a = t.a)",
    ),
    (
        "push_correlated_join",
        "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE t.c > (SELECT count(*) FROM u u2 WHERE u2.d = t.b AND u2.a = u.a)",
    ),
    // Restricting an aggregate to the keys its join can match.
    (
        "restrict_scalar_avg",
        "SELECT a FROM t WHERE b = 'x' AND c > (SELECT avg(u.a) FROM u WHERE u.a = t.a)",
    ),
    (
        "restrict_count_bug",
        "SELECT a, (SELECT count(*) FROM u WHERE u.a = t.a) AS n FROM t WHERE b = 'x'",
    ),
    (
        "restrict_duplicate_keys",
        "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE t.b = 'x' AND t.c > (SELECT max(u2.a) FROM u u2 WHERE u2.a = t.a)",
    ),
    (
        "restrict_two_keys",
        "SELECT a FROM t WHERE b = 'x' AND c > (SELECT count(*) FROM u WHERE u.a = t.a AND u.d = t.b)",
    ),
    (
        "restrict_in_having",
        "SELECT a FROM t WHERE b = 'x' AND a IN (SELECT a FROM u GROUP BY a HAVING count(*) > 1)",
    ),
    (
        "restrict_equality_only",
        "SELECT t.a FROM t JOIN (SELECT a, count(*) AS n FROM u GROUP BY a) g ON t.a = g.a AND t.c > g.n WHERE t.b = 'x'",
    ),
    (
        "restrict_right_outer",
        "SELECT t.a, g.n FROM (SELECT a, count(*) AS n FROM u GROUP BY a) g RIGHT JOIN (SELECT a FROM t WHERE b = 'x') t ON g.a = t.a",
    ),
    (
        "restrict_renamed_key",
        "SELECT t.a FROM t JOIN (SELECT k, n FROM (SELECT a AS k, count(*) AS n FROM u GROUP BY a) WHERE n > 1) g ON t.a = g.k WHERE t.b = 'x' AND t.c < (SELECT max(c) FROM t)",
    ),
    // An aggregate over an aggregate: restricted at both levels at once.
    (
        "restrict_nested",
        "SELECT x1.c1 AS c0, count(*) AS c1 FROM (SELECT a + 1 AS c0, d AS c1 FROM u WHERE d >= 'x') x1 LEFT SEMI JOIN (SELECT x4.c0, count(*) AS c1 FROM (SELECT c AS c0, count(*) AS c1 FROM t WHERE NOT (coalesce(a, 3) = 3) GROUP BY c) x4 GROUP BY x4.c0) x3 ON x1.c0 = x3.c0 AND x3.c1 IN (3, -2) WHERE x1.c1 IS NOT NULL GROUP BY x1.c1",
    ),
    (
        "restrict_below_semi",
        "SELECT t.a FROM t JOIN (SELECT a, count(*) AS n FROM u WHERE d IN (SELECT b FROM t WHERE c > 0) GROUP BY a) g ON t.a = g.a WHERE t.b = 'x'",
    ),
    (
        "restrict_not_global",
        "SELECT t.a FROM t JOIN (SELECT count(*) AS n FROM u) g ON t.c = g.n WHERE t.b = 'x'",
    ),
    (
        "restrict_not_nondeterministic_input",
        "SELECT t.a FROM t JOIN (SELECT a, count(*) AS n FROM u WHERE rand() > 0.5 GROUP BY a) g ON t.a = g.a WHERE t.b = 'x'",
    ),
    (
        "restrict_not_subquery_key",
        "SELECT t.a FROM t JOIN (SELECT a, count(*) AS n FROM u GROUP BY a) g ON t.a + (SELECT max(a) FROM u) = g.a WHERE t.b = 'x'",
    ),
    (
        "restrict_not_inequality",
        "SELECT t.a FROM t JOIN (SELECT a, count(*) AS n FROM u GROUP BY a) g ON t.a < g.a WHERE t.b = 'x'",
    ),
    (
        "restrict_not_null_safe",
        "SELECT t.a FROM t JOIN (SELECT a, count(*) AS n FROM u GROUP BY a) g ON t.a <=> g.a WHERE t.b = 'x'",
    ),
    (
        "restrict_not_preserved",
        "SELECT g.a FROM (SELECT a, count(*) AS n FROM u GROUP BY a) g LEFT JOIN t ON t.a = g.a AND t.b = 'x'",
    ),
    (
        "restrict_not_nondeterministic_aggregate",
        "SELECT t.a, g.f FROM t JOIN (SELECT a, nondeterministic_first(d) AS f FROM u GROUP BY a) g ON t.a = g.a WHERE t.b = 'x'",
    ),
    (
        "restrict_not_nondeterministic_domain",
        "SELECT t.a FROM (SELECT a FROM t WHERE b = 'x' AND rand() > 0.5) t JOIN (SELECT a, count(*) AS n FROM u GROUP BY a) g ON t.a = g.a",
    ),
    (
        "restrict_not_floating_key",
        "SELECT t.a FROM t JOIN (SELECT CAST(a AS DOUBLE) AS k, count(*) AS n FROM u GROUP BY CAST(a AS DOUBLE)) g ON CAST(t.a AS DOUBLE) = g.k WHERE t.b = 'x'",
    ),
    // Projections and limits.
    (
        "collapse",
        "SELECT y + 1 AS z FROM (SELECT x * 2 AS y FROM (SELECT a + 1 AS x FROM t))",
    ),
    (
        "collapse_twice_used",
        "SELECT y + y AS z FROM (SELECT a + 1 AS y FROM t)",
    ),
    (
        "collapse_aggregate",
        "SELECT n * 2 AS m FROM (SELECT a, count(*) AS n FROM t GROUP BY a)",
    ),
    (
        "limit_limit",
        "SELECT * FROM (SELECT a FROM t LIMIT 5) LIMIT 3",
    ),
    (
        "limit_limit_larger",
        "SELECT * FROM (SELECT a FROM t LIMIT 3) LIMIT 5",
    ),
    (
        "limit_top_n",
        "SELECT a, a + 1 AS x FROM t ORDER BY a LIMIT 4",
    ),
    (
        "limit_left",
        "SELECT t.a, u.d FROM t LEFT JOIN u ON t.a = u.a LIMIT 2",
    ),
    (
        "limit_right",
        "SELECT t.a, u.d FROM t RIGHT JOIN u ON t.a = u.a LIMIT 2",
    ),
    (
        "limit_left_bounded",
        "SELECT x.a FROM (SELECT a FROM t LIMIT 1) x LEFT JOIN u ON x.a = u.a LIMIT 2",
    ),
    (
        "push_with_cte",
        "SELECT * FROM (WITH v AS (SELECT a FROM t) SELECT a FROM v) WHERE a = 1",
    ),
    (
        "dedupe_cast_outer",
        "SELECT a FROM t WHERE CAST(c AS INT) = a AND CAST(c AS INT) = a AND EXISTS (SELECT 1 FROM u WHERE u.a = t.a AND u.a = t.a)",
    ),
    // Nondeterminism: nothing moves past, merges with or inlines `rand()`.
    (
        "rand_project",
        "SELECT a FROM (SELECT a, rand() AS r FROM t) WHERE r > 0.5 AND a > 1",
    ),
    (
        "rand_prefix",
        "SELECT t.a FROM t JOIN u ON t.a = u.a WHERE rand() > 0.5 AND t.b = 'x'",
    ),
    (
        "rand_filters",
        "SELECT a FROM (SELECT a FROM t WHERE rand() > 0.5) WHERE a > 1 AND rand() > 0.5 AND rand() > 0.5",
    ),
    (
        "rand_on",
        "SELECT t.a FROM t JOIN u ON t.a = u.a AND rand() > 0.5",
    ),
    (
        "rand_collapse",
        "SELECT r + 1 AS s FROM (SELECT rand() AS r FROM t)",
    ),
    (
        "rand_limit",
        "SELECT r FROM (SELECT rand() AS r FROM t) LIMIT 1",
    ),
    (
        "first_aggregate",
        "SELECT k, f FROM (SELECT a AS k, nondeterministic_first(b) AS f FROM t GROUP BY a) WHERE k > 1",
    ),
    (
        "collapse_passthrough",
        "SELECT x + 1 AS y, b FROM (SELECT a + 1 AS x, b FROM t)",
    ),
    (
        "collapse_subquery",
        "SELECT x, (SELECT max(a) FROM u) AS m FROM (SELECT a + 1 AS x FROM t)",
    ),
    (
        "collapse_aggregate_subquery",
        "SELECT n, (SELECT max(a) FROM u) AS m FROM (SELECT count(*) AS n FROM t GROUP BY a)",
    ),
    (
        "limit_global_aggregate",
        "SELECT x.n FROM (SELECT count(*) AS n FROM t) x LEFT JOIN u ON x.n = u.a LIMIT 2",
    ),
    (
        "limit_semi",
        "SELECT y.a FROM (SELECT a FROM t LEFT SEMI JOIN u ON t.a = u.a ORDER BY a LIMIT 1) y LEFT JOIN u ON y.a = u.a LIMIT 2",
    ),
    (
        "case_no_else_is_null",
        "SELECT a FROM t WHERE (CASE WHEN a IS NULL THEN 1 WHEN false THEN 2 END) IS NULL",
    ),
    (
        "case_no_else_not_null",
        "SELECT x FROM (SELECT CASE WHEN a IS NULL THEN 1 WHEN false THEN 2 END AS x FROM t) WHERE x IS NOT NULL",
    ),
    (
        "group_by_expression",
        "SELECT NOT (a < 1) AS x FROM t GROUP BY a < 1",
    ),
    (
        "group_by_expression_collapse",
        "SELECT NOT k AS x FROM (SELECT a < 1 AS k FROM t GROUP BY a < 1)",
    ),
    // Column pruning.
    ("prune_count", "SELECT count(*) AS n FROM t"),
    (
        "prune_join",
        "SELECT t.b FROM t JOIN u ON t.a = u.a WHERE t.c > 1",
    ),
    (
        "prune_alias_columns",
        "SELECT q FROM (SELECT a, b, c FROM t) AS x (p, q, r)",
    ),
    (
        "prune_aggregate",
        "SELECT k FROM (SELECT a AS k, count(*) AS n, max(b) AS m FROM t GROUP BY a)",
    ),
    (
        "prune_exists",
        "SELECT a FROM t WHERE EXISTS (SELECT * FROM u WHERE u.a = t.a)",
    ),
];

#[test]
fn rules() {
    let ctx = support::ctx();
    for (name, sql) in CASES {
        let plan = support::resolve(sql, &ctx);
        let optimized = optimize(&plan).unwrap();
        assert_eq!(
            optimized.root().resolved_output(),
            plan.root().resolved_output(),
            "{name} keeps its output"
        );
        let again = optimize(&optimized).unwrap();
        assert!(
            again.structurally_eq(&optimized),
            "{name} is a fixed point:\n{}",
            again.root()
        );
        insta::assert_snapshot!(*name, optimized.root().to_string(), sql);
        // Limits and nondeterministic calls choose rows; the interpreter
        // keeps numbers in unbounded fixed point, without NaN or integral
        // overflow (whose ANSI error or legacy wrap it does not model).
        let skip = [
            "LIMIT",
            "rand()",
            "nondeterministic_first(",
            "DOUBLE",
            "1.0F",
        ];
        let overflows = ["fold_arith", "fold_int_overflow", "fold_negative"];
        if !skip.iter().any(|s| sql.contains(s)) && !overflows.contains(name) {
            for seed in 1..=8 {
                let db = data::small(seed);
                let want = eval::rows(&db, plan.root());
                assert_eq!(
                    eval::rows(&db, optimized.root()),
                    want,
                    "{name}, seed {seed}"
                );
            }
        }
    }
}

#[test]
fn top_n_is_detected() {
    let ctx = support::ctx();
    let plan = support::resolve("SELECT a, b FROM t ORDER BY a LIMIT 4", &ctx);
    let optimized = optimize(&plan).unwrap();
    let top = TopN::detect(optimized.root()).unwrap();
    assert_eq!((top.limit, top.order.len()), (4, 1));
    assert!(TopN::detect(top.input).is_none());
    let plan = support::resolve("SELECT a FROM t LIMIT 4", &ctx);
    assert!(TopN::detect(optimize(&plan).unwrap().root()).is_none());
}

#[test]
fn unresolved_plans_are_rejected() {
    let plan = Plan::try_new(LogicalPlan::table(["t"])).unwrap();
    let err = optimize(&plan).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
}

#[test]
fn deep_plans_run_on_a_big_stack() {
    let mut sql = "SELECT a, b FROM t".to_owned();
    for _ in 0..100 {
        sql = format!("SELECT a, b FROM ({sql}) WHERE a > 1");
    }
    let plan = support::resolve(&sql, &support::ctx());
    let optimized = optimize(&plan).unwrap();
    assert_eq!(
        optimized.root().to_string(),
        "Filter (a#0 > 1)\n+- Scan spark_catalog.default.t [a#0, b#1]\n"
    );
}

#[test]
fn renaming_aliases_keep_their_input() {
    let plan = support::resolve("SELECT a, b, c FROM t", &support::ctx());
    let renamed = LogicalPlan::new(PlanNode::SubqueryAlias {
        input: Arc::clone(plan.root()),
        alias: "x".into(),
        columns: vec!["p".into(), "q".into(), "r".into()],
    });
    let project = LogicalPlan::new(PlanNode::Project {
        input: Arc::new(renamed),
        exprs: vec![Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Column(plan.root().resolved_output().unwrap()[0].clone()),
        )))],
    });
    let optimized = optimize(&Plan::try_new(project).unwrap()).unwrap();
    insta::assert_snapshot!(optimized.root().to_string(), @r"
    Project [a#0]
    +- SubqueryAlias x (p, q, r)
       +- Scan spark_catalog.default.t [a#0, b#1, c#2]
    ");
}

#[test]
fn limits_pass_long_projection_runs_at_once() {
    let mut sql = "SELECT a AS x FROM t".to_owned();
    for _ in 0..150 {
        sql = format!("SELECT x + x AS x FROM ({sql})");
    }
    let plan = support::resolve(&format!("{sql} LIMIT 2"), &support::ctx());
    let optimized = optimize(&plan).unwrap();
    assert!(optimize(&optimized).unwrap().structurally_eq(&optimized));
    let mut p = optimized.root();
    while let PlanNode::Project { input, .. } = &p.node {
        p = input;
    }
    assert!(
        matches!(p.node, PlanNode::Limit { .. }),
        "{}",
        optimized.root()
    );
}

#[test]
fn shared_expressions_stay_linear() {
    let base = support::resolve("SELECT a FROM t", &support::ctx());
    let PlanNode::Project { input: scan, exprs } = &base.root().node else {
        panic!("{}", base.root());
    };
    let call = |op: Builtin, args: Vec<Arc<Expr>>, data_type: DataType| {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(op),
                kind: FunctionKind::Scalar,
                args,
                distinct: false,
                filter: None,
                data_type,
                nullable: true,
                deterministic: true,
            },
        ))))
    };
    // `c = c + c`, 40 times: a DAG of 40 distinct nodes, 2^40 paths.
    let mut e = Arc::clone(&exprs[0]);
    for _ in 0..40 {
        e = call(
            Builtin::Add,
            vec![Arc::clone(&e), Arc::clone(&e)],
            DataType::Integer,
        );
    }
    let zero = Arc::new(Expr::literal(Literal::Integer(0)));
    let gt = call(Builtin::Gt, vec![e, zero], DataType::Boolean);
    let both = call(Builtin::And, vec![Arc::clone(&gt), gt], DataType::Boolean);
    let filter = LogicalPlan::new(PlanNode::Filter {
        input: Arc::clone(scan),
        predicate: both,
    });
    let started = std::time::Instant::now();
    let optimized = optimize(&Plan::try_new(filter).unwrap()).unwrap();
    assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
    assert!(matches!(optimized.root().node, PlanNode::Filter { .. }));
}
