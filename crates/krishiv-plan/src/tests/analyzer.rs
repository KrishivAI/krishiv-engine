// SPDX-License-Identifier: Apache-2.0
//! The analyzer on hand-built plans: the TPC-H plans and DataFrame
//! `plan_id` cases against Spark 4.2 (output names and nullability, error
//! classes), the access-policy hook, pins, determinism, idempotence,
//! linearity on shared subtrees and the depth cap.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::redact::RedactedSql;
use krishiv_types::DataType;
use proptest::prelude::*;
use serde_json::Value;

use super::depth::{on_small_stack, self_joins};
use super::dsl::{add, c, eq, f, gt, int, s, star};
use super::tpch::QUERIES;
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AccessDecision, AccessPolicy, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _,
    BoxFuture, DefaultAnalyzer, MemoryCatalog, PolicyVersion, Principal, Privilege, QueryContext,
    ResolutionConf, RowColumnPolicy, Securable, SparkProfile, SqlFrontend, TableColumn,
};
use crate::walk::{Node, visit_distinct};
use crate::{
    ColumnRef, Cte, Expr, ExprNode, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan,
    MAX_DEPTH, Origin, Plan, PlanId, PlanNode, ReadSource, SourceId, SqlArgs, Star, TableVersion,
};

fn goldens() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/analyzer-goldens.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The fixture's tables; `lineitem` at version 7.
fn catalog() -> MemoryCatalog {
    let g = goldens();
    let mut cat = MemoryCatalog::new("spark_catalog");
    for (name, cols) in g["tables"].as_object().unwrap() {
        let columns = cols
            .as_array()
            .unwrap()
            .iter()
            .map(|f| TableColumn {
                name: f["name"].as_str().unwrap().to_owned(),
                data_type: krishiv_types::parse_data_type(f["type"].as_str().unwrap()).unwrap(),
                nullable: f["nullable"].as_bool().unwrap(),
            })
            .collect();
        let version = (name == "lineitem").then_some(TableVersion(7));
        cat = cat.with_table(&["default"], name, columns, version);
    }
    cat
}

fn ctx() -> AnalysisContext {
    context(catalog(), ResolutionConf::default())
}

fn run(plan: LogicalPlan, ctx: &AnalysisContext) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, ctx))
}

fn class(r: &Result<AnalyzedPlan, AnalysisError>) -> &'static str {
    r.as_ref()
        .map_or_else(|e| e.error.error_class().name(), |_| "OK")
}

/// Names and nullability, or the error class and parameters.
fn summary(r: &Result<AnalyzedPlan, AnalysisError>) -> Value {
    match r {
        Ok(a) => serde_json::json!(
            a.output
                .iter()
                .map(|c| (c.name.clone(), c.nullable))
                .collect::<Vec<_>>()
        ),
        Err(e) => serde_json::json!([
            e.error.error_class().name(),
            e.error
                .message_parameters()
                .iter()
                .map(|(k, v)| (*k, v))
                .collect::<std::collections::BTreeMap<_, _>>()
        ]),
    }
}

fn spark(case: &Value) -> Value {
    if case["ok"].as_bool().unwrap() {
        serde_json::json!(
            case["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| (
                    f["name"].as_str().unwrap(),
                    f["nullable"].as_bool().unwrap()
                ))
                .collect::<Vec<_>>()
        )
    } else {
        serde_json::json!([case["error_class"], case["message_parameters"]])
    }
}

#[test]
fn tpch_plans_resolve_as_spark_does() {
    let g = goldens();
    let ctx = ctx();
    for (i, q) in QUERIES.iter().enumerate() {
        let r = run(q(), &ctx);
        assert_eq!(summary(&r), spark(&g["tpch"][i]), "q{:02}", i + 1);
        let a = r.unwrap();
        assert!(a.plan.root().is_resolved(), "q{:02}", i + 1);
        if i == 14 {
            let text = a.plan.root().to_string();
            assert!(text.contains("CteRef revenue [supplier_no#"), "{text}");
        }
        // Resolving a resolved plan changes nothing.
        let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
        assert!(
            again.plan.structurally_eq(&a.plan),
            "q{:02} is a fixed point",
            i + 1
        );
        assert_eq!(again.output, a.output);
    }
}

fn df(table: &str, id: i64) -> LogicalPlan {
    LogicalPlan::table([table]).with_origin(Origin::plan(PlanId(id)))
}

fn col(name: &str, id: i64) -> Expr {
    Expr::new(ExprNode::Column(ColumnRef {
        name: vec![name.to_owned()],
        plan_id: Some(PlanId(id)),
    }))
}

fn with_id(p: LogicalPlan, id: i64) -> LogicalPlan {
    p.with_origin(Origin::plan(PlanId(id)))
}

/// The fixture's DataFrame cases, built as the Python client builds them;
/// `base` offsets every `plan_id` (renumbering).
fn dataframes(base: i64) -> Vec<(&'static str, LogicalPlan)> {
    let (t, u) = (df("t", base + 1), df("u", base + 2));
    let shared = Arc::new(t.clone());
    let join_two = with_id(
        t.clone().join(
            u.clone(),
            JoinKind::Inner,
            JoinCondition::On(Arc::new(eq(col("a", base + 1), col("a", base + 2)))),
        ),
        base + 3,
    );
    let with_columns = |input: LogicalPlan, name: &str, e: Expr, id: i64| {
        with_id(
            LogicalPlan::new(PlanNode::WithColumns {
                input: Arc::new(input),
                columns: vec![(name.to_owned(), Arc::new(e))],
            }),
            id,
        )
    };
    let self_join = |condition| {
        with_id(
            LogicalPlan::new(PlanNode::Join {
                left: Arc::clone(&shared),
                right: Arc::clone(&shared),
                kind: JoinKind::Inner,
                condition,
            }),
            base + 3,
        )
    };
    let added = Expr::builtin(crate::Builtin::Add, vec![col("a", base + 1), int(1)]);
    let a = || t.clone().project(vec![c("a")]);
    let asc = |name: &str| vec![crate::SortOrder::asc(c(name))];
    vec![
        ("same_df", t.clone().project(vec![col("a", base + 1)])),
        ("other_df", t.clone().project(vec![col("a", base + 2)])),
        (
            "through_filter",
            with_id(t.clone().filter(gt(col("a", base + 1), int(1))), base + 3)
                .project(vec![col("b", base + 1)]),
        ),
        (
            "projected_away",
            with_id(t.clone().project(vec![c("a")]), base + 3).project(vec![col("b", base + 1)]),
        ),
        (
            "join_two",
            join_two
                .clone()
                .project(vec![col("b", base + 1), col("d", base + 2)]),
        ),
        (
            "self_join",
            with_id(
                self_join(JoinCondition::Using(vec!["a".to_owned()])),
                base + 4,
            )
            .project(vec![col("b", base + 1)]),
        ),
        (
            "self_join_cond",
            self_join(JoinCondition::On(Arc::new(eq(
                col("a", base + 1),
                col("a", base + 1),
            )))),
        ),
        (
            "star",
            join_two.project(vec![Expr::new(ExprNode::Star(Star {
                qualifier: Vec::new(),
                plan_id: Some(PlanId(base + 2)),
            }))]),
        ),
        (
            "with_column",
            with_columns(
                with_columns(t.clone(), "a", added, base + 3),
                "e",
                Literal::Integer(1).into(),
                base + 4,
            ),
        ),
        ("missing", t.clone().project(vec![col("x", base + 1)])),
        (
            "outer",
            t.clone()
                .filter(Expr::subquery(
                    crate::SubqueryKind::Exists,
                    with_id(
                        u.filter(eq(col("a", base + 2), col("a", base + 1))),
                        base + 3,
                    ),
                ))
                .project(vec![col("b", base + 1)]),
        ),
        (
            "missing_through_filter",
            a().filter(gt(c("a"), int(1))).sort(asc("b")),
        ),
        ("missing_through_limit", a().limit(int(1)).sort(asc("b"))),
        ("missing_two_sorts", a().sort(asc("b")).sort(asc("c"))),
        (
            "missing_two_projects",
            t.clone()
                .project(vec![c("a"), c("b")])
                .project(vec![c("a")])
                .sort(asc("c")),
        ),
        (
            "missing_in_filter",
            t.clone()
                .project(vec![c("a"), c("c")])
                .filter(eq(c("b"), s("x"))),
        ),
        (
            "missing_grouping",
            t.clone()
                .aggregate(
                    vec![c("a")],
                    vec![c("a"), f("max", vec![c("c")]).alias("m")],
                )
                .project(vec![c("m")])
                .sort(asc("a")),
        ),
        ("missing_after_alias", a().alias("x").sort(asc("b"))),
        (
            "lateral_alias",
            t.project(vec![
                add(col("a", base + 1), int(1)).alias("k"),
                add(c("k"), int(1)).alias("m"),
            ]),
        ),
    ]
}

#[test]
fn plan_id_references_resolve_as_spark_connect_does() {
    let g = goldens();
    let ctx = ctx();
    let mut raised = BTreeSet::new();
    for (case, (id, plan)) in g["plan_ids"].as_array().unwrap().iter().zip(dataframes(0)) {
        assert_eq!(case["id"], id);
        let r = run(plan, &ctx);
        assert_eq!(summary(&r), spark(case), "{id}");
        raised.insert(class(&r));
    }
    assert!(raised.contains("CANNOT_RESOLVE_DATAFRAME_COLUMN"));
    assert!(raised.contains("AMBIGUOUS_COLUMN_REFERENCE"));
}

/// The select list of the generated-names statement of the corpus.
fn names_query() -> Vec<Expr> {
    use crate::Builtin::{EqNullSafe, IsNotNull, IsNull, Negative, Remainder, Subtract};
    let lit = |v: Literal| Expr::from(v);
    let date = lit(Literal::Date(18263));
    let decimal = lit(Literal::Decimal {
        unscaled: 15,
        data_type: krishiv_types::DecimalType::try_new(2, 1).unwrap(),
    });
    let op = |b, args| Expr::builtin(b, args);
    vec![
        add(c("a"), int(1)),
        op(Negative, vec![c("a")]),
        eq(c("a"), int(1)),
        op(crate::Builtin::Not, vec![eq(c("a"), int(1))]),
        op(crate::Builtin::Not, vec![eq(c("a"), int(1))]),
        c("a").cast(DataType::String),
        s("x"),
        decimal,
        int(2),
        lit(Literal::Long(10)),
        op(IsNull, vec![c("a")]),
        op(IsNotNull, vec![c("a")]),
        op(crate::Builtin::In, vec![c("a"), int(1), int(2)]),
        op(crate::Builtin::Between, vec![c("a"), int(1), int(2)]),
        op(
            crate::Builtin::When,
            vec![eq(c("a"), int(1)), s("y"), s("z")],
        ),
        f("substring", vec![c("b"), int(1), int(2)]),
        op(
            Subtract,
            vec![
                op(crate::Builtin::Multiply, vec![c("a"), int(2)]),
                op(crate::Builtin::Divide, vec![c("c"), int(3)]),
            ],
        ),
        op(crate::Builtin::Like, vec![c("b"), s("x%")]),
        op(Remainder, vec![c("a"), int(2)]),
        lit(Literal::Boolean(true)),
        date.clone(),
        op(EqNullSafe, vec![c("a"), int(1)]),
        op(
            crate::Builtin::Or,
            vec![
                op(
                    crate::Builtin::And,
                    vec![
                        op(crate::Builtin::GtEq, vec![c("a"), int(1)]),
                        op(crate::Builtin::Lt, vec![c("a"), int(2)]),
                    ],
                ),
                gt(c("c"), int(3)),
            ],
        ),
        f("extract", vec![s("year"), date]),
    ]
}

/// Statements of the resolution corpus, built as SQL lowering builds them
/// (the SQL front end checks the whole corpus through the parser).
fn corpus() -> Vec<(&'static str, ResolutionConf, LogicalPlan)> {
    let t = || LogicalPlan::table(["t"]);
    let default = ResolutionConf::default();
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..default
    };
    let on = |l: &str, r: &str| JoinCondition::On(Arc::new(eq(c(l), c(r))));
    let cte = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "t".to_owned(),
            plan: Arc::new(LogicalPlan::table(["u"]).project(vec![int(1).alias("z")])),
        }],
    });
    let renamed = LogicalPlan::new(PlanNode::SubqueryAlias {
        input: Arc::new(t()),
        alias: "x".to_owned(),
        columns: vec!["p".to_owned(), "q".to_owned()],
    });
    let scalar = Expr::subquery(
        crate::SubqueryKind::Scalar,
        LogicalPlan::table(["u"]).project(vec![c("a"), c("d")]),
    );
    let using_kind = |kind| {
        t().join(
            LogicalPlan::table(["u"]),
            kind,
            JoinCondition::Using(vec!["a".to_owned()]),
        )
        .project(vec![star()])
    };
    let using = |cols: &[&str]| {
        t().join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::Using(cols.iter().map(|c| (*c).to_owned()).collect()),
        )
        .project(vec![star()])
    };
    vec![
        ("SELECT x FROM t", default, t().project(vec![c("x")])),
        (
            "SELECT colz FROM wide",
            default,
            LogicalPlan::table(["wide"]).project(vec![c("colz")]),
        ),
        ("SELECT t.x FROM t", default, t().project(vec![c("t.x")])),
        ("SELECT v.a FROM t", default, t().project(vec![c("v.a")])),
        (
            "SELECT a FROM t JOIN u ON t.a = u.a",
            default,
            t().join(LogicalPlan::table(["u"]), JoinKind::Inner, on("t.a", "u.a"))
                .project(vec![c("a")]),
        ),
        ("SELECT a.b FROM t", default, t().project(vec![c("a.b")])),
        ("SELECT A FROM t", sensitive, t().project(vec![c("A")])),
        (
            "SELECT A, T.B FROM t",
            default,
            t().project(vec![c("A"), c("T.B")]),
        ),
        (
            "SELECT nosuch(a) FROM t",
            default,
            t().project(vec![f("nosuch", vec![c("a")])]),
        ),
        (
            "SELECT * FROM missing_table",
            default,
            LogicalPlan::table(["missing_table"]).project(vec![star()]),
        ),
        (
            "SELECT * FROM default.missing_table",
            default,
            LogicalPlan::table(["default", "missing_table"]).project(vec![star()]),
        ),
        (
            "WITH t AS (SELECT 1 AS z FROM u) SELECT a FROM t",
            default,
            cte,
        ),
        (
            "SELECT t.a FROM (SELECT a FROM t) s",
            default,
            t().project(vec![c("a")]).alias("s").project(vec![c("t.a")]),
        ),
        (
            "SELECT x.* FROM t",
            default,
            t().project(vec![Expr::new(ExprNode::Star(Star {
                qualifier: vec!["x".to_owned()],
                plan_id: None,
            }))]),
        ),
        ("SELECT * FROM t JOIN u USING (a)", default, using(&["a"])),
        (
            "SELECT * FROM t RIGHT JOIN u USING (a)",
            default,
            using_kind(JoinKind::RightOuter),
        ),
        (
            "SELECT * FROM t FULL JOIN u USING (a)",
            default,
            using_kind(JoinKind::FullOuter),
        ),
        (
            "SELECT a + 1, -a, a = 1, NOT (a = 1), a <> 1, CAST(a AS STRING), 'x', 1.5, 2, 10L, a IS NULL, \
             a IS NOT NULL, a IN (1, 2), a BETWEEN 1 AND 2, CASE WHEN a = 1 THEN 'y' ELSE 'z' END, \
             substring(b, 1, 2), a * 2 - c / 3, b LIKE 'x%', a % 2, TRUE, DATE '2020-01-02', a <=> 1, \
             a >= 1 AND a < 2 OR c > 3, extract(year FROM DATE '2020-01-02') FROM t",
            default,
            t().project(names_query()),
        ),
        (
            "SELECT CAST(a AS BIGINT) + 1 AS x, TRY_CAST(b AS INT) FROM t",
            default,
            t().project(vec![
                add(c("a").cast(DataType::Long), int(1)).alias("x"),
                Expr::new(ExprNode::Cast {
                    expr: Arc::new(c("b")),
                    to: DataType::Integer,
                    mode: crate::CastMode::Try,
                }),
            ]),
        ),
        (
            "SELECT count(a) FILTER (WHERE b = 'x') FROM t",
            default,
            t().project(vec![Expr::new(ExprNode::Function(crate::FunctionCall {
                name: "count".to_owned(),
                args: vec![Arc::new(c("a"))],
                distinct: false,
                filter: Some(Arc::new(eq(c("b"), s("x")))),
            }))]),
        ),
        ("SELECT * FROM t JOIN u USING (z)", default, using(&["z"])),
        ("SELECT * FROM t JOIN u USING (b)", default, using(&["b"])),
        (
            "SELECT a, b FROM t GROUP BY a",
            default,
            t().aggregate(vec![c("a")], vec![c("a"), c("b")]),
        ),
        (
            "SELECT sum(max(a)) FROM t",
            default,
            t().project(vec![f("sum", vec![f("max", vec![c("a")])])]),
        ),
        (
            "SELECT a FROM t WHERE sum(a) > 1",
            default,
            t().filter(gt(f("sum", vec![c("a")]), int(1)))
                .project(vec![c("a")]),
        ),
        (
            "SELECT sum(a), b FROM t",
            default,
            t().project(vec![f("sum", vec![c("a")]), c("b")]),
        ),
        (
            "SELECT (SELECT a, d FROM u) FROM t",
            default,
            t().project(vec![scalar]),
        ),
        (
            "SELECT * FROM t AS x (p, q)",
            default,
            renamed.project(vec![star()]),
        ),
        (
            "SELECT a FROM t GROUP BY a HAVING b = 'x'",
            default,
            t().aggregate(vec![c("a")], vec![c("a")])
                .filter(eq(c("b"), s("x"))),
        ),
    ]
}

#[test]
fn resolution_errors_and_names_match_spark() {
    let g = goldens();
    let cases = g["corpus"].as_array().unwrap();
    let mut raised = BTreeSet::new();
    for (sql, conf, plan) in corpus() {
        let case = cases.iter().find(|c| c["sql"] == sql).unwrap();
        let ctx = context(catalog(), conf);
        let r = run(plan, &ctx);
        assert_eq!(summary(&r), spark(case), "{sql}");
        raised.insert(class(&r));
    }
    // The analyzer's Spark error classes, each matched against Spark above
    // (the DataFrame ones in `plan_id_references_resolve_as_spark_connect_does`).
    for expected in [
        "AMBIGUOUS_REFERENCE",
        "ASSIGNMENT_ARITY_MISMATCH",
        "CANNOT_RESOLVE_STAR_EXPAND",
        "INVALID_EXTRACT_BASE_FIELD_TYPE",
        "INVALID_SUBQUERY_EXPRESSION.SCALAR_SUBQUERY_RETURN_MORE_THAN_ONE_OUTPUT_COLUMN",
        "INVALID_WHERE_CONDITION",
        "MISSING_AGGREGATION",
        "MISSING_GROUP_BY",
        "NESTED_AGGREGATE_FUNCTION",
        "TABLE_OR_VIEW_NOT_FOUND",
        "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        "UNRESOLVED_ROUTINE",
        "UNRESOLVED_USING_COLUMN_FOR_JOIN",
    ] {
        assert!(raised.contains(expected), "{expected} not raised");
    }
}

/// `e = e + e` 30 times (2^30 leaves as a tree): bound, validated and
/// named in time linear in its 31 distinct nodes; the generated name is
/// bounded.
#[test]
fn shared_expressions_stay_linear() {
    let doubled = || {
        (0..30).fold(Arc::new(c("a")), |e, _| {
            Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
                name: "+".to_owned(),
                args: vec![Arc::clone(&e), e],
                distinct: false,
                filter: None,
            })))
        })
    };
    let grouped = LogicalPlan::table(["t"]).aggregate(
        vec![Arc::unwrap_or_clone(doubled())],
        vec![f("count", vec![star()])],
    );
    let a = timed("grouped by a shared expression", || run(grouped, &ctx())).unwrap();
    assert_eq!(a.output[0].name, "count(1)");
    let named = LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::table(["t"])),
        exprs: vec![doubled()],
    });
    let a = timed("named after a shared expression", || run(named, &ctx())).unwrap();
    assert!(
        a.output[0].name.len() < 3 * (64 << 10),
        "{}",
        a.output[0].name.len()
    );
}

proptest! {
    /// Renumbered `plan_id`s resolve alike: same output, same key, same
    /// resolved plan.
    #[test]
    fn plan_ids_renumber_without_effect(base in 0_i64..1_000_000) {
        let ctx = ctx();
        for ((id, a), (_, b)) in dataframes(0).into_iter().zip(dataframes(base)) {
            let (a, b) = (Plan::try_new(a).unwrap(), Plan::try_new(b).unwrap());
            prop_assert_eq!(a.structural_hash(), b.structural_hash(), "{}", id);
            let (ra, rb) = (block_on(DefaultAnalyzer.analyze(&a, &ctx)), block_on(DefaultAnalyzer.analyze(&b, &ctx)));
            prop_assert_eq!(summary(&ra), summary(&rb), "{}", id);
            if let (Ok(ra), Ok(rb)) = (ra, rb) {
                prop_assert!(ra.plan.structurally_eq(&rb.plan), "{}", id);
                prop_assert_eq!(ra.input_key, rb.input_key);
            }
        }
    }

    /// Resolution is deterministic: the same input gives byte-identical
    /// resolved plans, whatever hasher seeds the process uses.
    #[test]
    fn resolution_is_deterministic(q in 0_usize..22) {
        let ctx = ctx();
        let a = run(QUERIES[q](), &ctx).unwrap();
        let b = run(QUERIES[q](), &ctx).unwrap();
        prop_assert_eq!(a.plan.canonical_encoding(), b.plan.canonical_encoding());
        prop_assert_eq!(a.input_key, b.input_key);
        prop_assert_eq!(a.context.digest(), b.context.digest());
    }
}

fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{what}: {:?}",
        start.elapsed()
    );
    out
}

fn distinct_relations(p: &Plan) -> usize {
    let mut n = 0;
    let _ = visit_distinct(Node::Plan(p.root()), |node| {
        n += usize::from(matches!(node, Node::Plan(_)));
        ControlFlow::Continue(())
    });
    n
}

/// `p.join(p)` 30 times (2^31 - 1 relations as a tree) resolves in time
/// and size linear in its 31 distinct relations: each right side is a
/// `Shift` of the shared left side.
#[test]
fn self_joins_stay_linear() {
    let plan = (0..30).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
        let p = Arc::new(p);
        let side = |alias: &str| {
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input: Arc::clone(&p),
                alias: alias.to_owned(),
                columns: Vec::new(),
            })
        };
        side("l")
            .join(
                side("r"),
                JoinKind::Inner,
                JoinCondition::On(Arc::new(eq(c("l.a"), c("r.a")))),
            )
            .project(vec![c("l.a")])
    });
    let a = timed("30 self-joins", || run(plan, &ctx()).unwrap());
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx())).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
    assert!(
        distinct_relations(&a.plan) <= 30 * 5 + 2,
        "{}",
        distinct_relations(&a.plan)
    );
    assert_eq!(a.plan.root().to_string().matches("Shift +").count(), 30);
    // The 31 distinct relations of `p.join(p)` x30, without projections.
    let wide = timed("wide self-joins", || run(self_joins(12), &ctx()).unwrap());
    assert_eq!(wide.output.len(), 3 << 12);
    let ids: BTreeSet<u64> = wide.output.iter().map(|c| c.id.0).collect();
    assert_eq!(ids.len(), 3 << 12, "each instance has its own attributes");
}

/// Self-join instances reserve id spans that double per level: 70 levels
/// would pass 2^64, so the plan is too complex, never an overflow; a
/// crafted `Shift` far out of range is rejected too.
#[test]
fn instance_ids_never_overflow() {
    let deep = (0..70).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
        let p = Arc::new(p);
        let side = |alias: &str| {
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input: Arc::clone(&p),
                alias: alias.to_owned(),
                columns: Vec::new(),
            })
        };
        side("l")
            .join(side("r"), JoinKind::Cross, JoinCondition::None)
            .project(vec![c("l.a")])
    });
    assert_eq!(class(&run(deep, &ctx())), "FAILED_TO_PARSE_TOO_COMPLEX");
    let crafted = LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(LogicalPlan::table(["t"])),
        offset: u64::MAX - 1,
    });
    assert_eq!(class(&run(crafted, &ctx())), "FAILED_TO_PARSE_TOO_COMPLEX");
}

/// `WITH c0 AS (...), c1 AS (SELECT * FROM c0 JOIN c0), ...` 30 deep:
/// each reference is a `CteRef`, so nothing is copied.
#[test]
fn cte_chains_stay_linear() {
    let mut ctes = vec![Cte {
        name: "c0".to_owned(),
        plan: Arc::new(LogicalPlan::table(["u"]).project(vec![c("a")])),
    }];
    for i in 1..30 {
        let prev = format!("c{}", i - 1);
        let body = LogicalPlan::table([prev.as_str()])
            .alias("l")
            .join(
                LogicalPlan::table([prev.as_str()]).alias("r"),
                JoinKind::Inner,
                JoinCondition::On(Arc::new(eq(c("l.a"), c("r.a")))),
            )
            .project(vec![c("l.a")]);
        ctes.push(Cte {
            name: format!("c{i}"),
            plan: Arc::new(body),
        });
    }
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["c29"])),
        ctes,
    });
    let a = timed("CTE chain", || run(plan, &ctx()).unwrap());
    assert!(
        distinct_relations(&a.plan) < 30 * 8,
        "{}",
        distinct_relations(&a.plan)
    );
}

/// `df = df.where(df.exists())` 30 times: each plan is reached both as a
/// child and inside a subquery, 2^30 paths. Resolution reuses a relation
/// without outer references across scopes, so it stays linear.
#[test]
fn subqueries_over_shared_plans_stay_linear() {
    let plan = (0..30).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
        let p = Arc::new(p);
        LogicalPlan::new(PlanNode::Filter {
            input: Arc::clone(&p),
            predicate: Arc::new(Expr::new(ExprNode::Subquery {
                kind: crate::SubqueryKind::Exists,
                plan: p,
            })),
        })
    });
    let a = timed("30 nested EXISTS over one plan", || {
        run(plan, &ctx()).unwrap()
    });
    assert!(
        distinct_relations(&a.plan) <= 30 * 4,
        "{}",
        distinct_relations(&a.plan)
    );
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx())).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
}

/// A `Shift` is the fresh instance: its input resolves as itself even
/// where the input's ids meet an enclosing query's (which makes a direct
/// reference to the input a fresh instance). So the attributes bound over
/// the shift are the ones the retained node reads, `id + offset` of what
/// the input states, and re-analysis is a fixed point.
#[test]
fn shift_input_is_never_re_instanced() {
    let ctx = ctx();
    let p = run(LogicalPlan::table(["t"]).project(vec![c("a")]), &ctx).unwrap();
    let p = Arc::clone(p.plan.root());
    let stated: Vec<_> = p.resolved_output().unwrap();
    // `p WHERE EXISTS (SELECT a FROM <instance of p>)`: in the subquery,
    // `p`'s ids are those of the enclosing query.
    let shift = LogicalPlan::new(PlanNode::Shift {
        input: Arc::clone(&p),
        offset: 100,
    });
    let plan = LogicalPlan::new(PlanNode::Filter {
        input: Arc::clone(&p),
        predicate: Arc::new(Expr::subquery(
            crate::SubqueryKind::Exists,
            shift.project(vec![c("a")]),
        )),
    });
    let a = run(plan, &ctx).unwrap();
    let PlanNode::Filter { predicate, .. } = &a.plan.root().node else {
        panic!("{:?}", a.plan)
    };
    let ExprNode::Subquery { plan: sub, .. } = &predicate.node else {
        panic!("{predicate:?}")
    };
    let PlanNode::Project { input, exprs } = &sub.node else {
        panic!("{sub:?}")
    };
    assert!(matches!(input.node, PlanNode::Shift { offset: 100, .. }));
    let read: Vec<_> = exprs
        .iter()
        .map(|e| match &e.node {
            ExprNode::Resolved(crate::ResolvedExpr::Column(c)) => c.id,
            other => panic!("{other:?}"),
        })
        .collect();
    let shifted: Vec<_> = stated.iter().map(|c| crate::AttrId(c.id.0 + 100)).collect();
    assert_eq!(read, shifted, "bound over the attributes the shift reads");
    let output: Vec<_> = input
        .resolved_output()
        .unwrap()
        .iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(output, shifted);
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
    assert_eq!(again.output, a.output);
}

/// A deep copy: every relation and expression a new `Arc`.
fn unshared(p: &LogicalPlan) -> LogicalPlan {
    Arc::unwrap_or_clone(
        Arc::new(p.clone())
            .transform_up(&mut |n: Arc<LogicalPlan>| {
                let n = Arc::unwrap_or_clone(n);
                let n = Arc::new(n).map_expressions(|e| {
                    Ok::<_, std::convert::Infallible>(Arc::new(Arc::unwrap_or_clone(e)))
                })?;
                Ok::<_, std::convert::Infallible>(Arc::new(Arc::unwrap_or_clone(n)))
            })
            .unwrap(),
    )
}

/// Equal input keys give equal resolved plans, however the input shares
/// its subtrees.
#[test]
fn resolution_ignores_how_the_input_shares() {
    let ctx = ctx();
    for (i, q) in QUERIES.iter().enumerate() {
        let (shared, copied) = (run(q(), &ctx).unwrap(), run(unshared(&q()), &ctx).unwrap());
        assert!(shared.plan.structurally_eq(&copied.plan), "q{:02}", i + 1);
    }
    let t = Arc::new(LogicalPlan::table(["t"]).alias("x"));
    let join = |l: Arc<LogicalPlan>, r: Arc<LogicalPlan>| {
        LogicalPlan::new(PlanNode::Join {
            left: l,
            right: r,
            kind: JoinKind::Cross,
            condition: JoinCondition::None,
        })
    };
    let shared = join(Arc::clone(&t), Arc::clone(&t));
    let copied = join(Arc::new(LogicalPlan::table(["t"]).alias("x")), t);
    let (a, b) = (
        Plan::try_new(shared).unwrap(),
        Plan::try_new(copied).unwrap(),
    );
    assert_eq!(a.structural_hash(), b.structural_hash());
    let (ra, rb) = (
        block_on(DefaultAnalyzer.analyze(&a, &ctx)).unwrap(),
        block_on(DefaultAnalyzer.analyze(&b, &ctx)).unwrap(),
    );
    assert!(ra.plan.structurally_eq(&rb.plan));
    assert_eq!(ra.plan.root().resolved_output(), Some(ra.output.clone()));
}

/// A policy that records its calls and answers with `decision`.
struct Policy {
    decision: Result<AccessDecision, KrishivError>,
    calls: Mutex<Vec<String>>,
}

impl AccessPolicy for Policy {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(9)
    }

    fn check<'a>(
        &'a self,
        p: &'a Principal,
        s: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(format!("{} {privilege} {s}", p.name));
        }
        Box::pin(std::future::ready(self.decision.clone()))
    }
}

fn with_policy(decision: Result<AccessDecision, KrishivError>) -> (AnalysisContext, Arc<Policy>) {
    let policy = Arc::new(Policy {
        decision,
        calls: Mutex::new(Vec::new()),
    });
    let mut ctx = ctx();
    ctx.policy = Arc::clone(&policy) as Arc<dyn AccessPolicy>;
    (ctx, policy)
}

#[test]
fn the_policy_is_asked_once_per_table_and_fails_closed() {
    let deny = Ok(AccessDecision::Deny {
        reason: "Canary-7f3a9c".to_owned(),
    });
    let (ctx, policy) = with_policy(deny);
    let q21 = QUERIES[20]();
    let e = run(q21.clone(), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::KrishivInsufficientPermissions
    );
    assert_eq!(
        e.error.to_string(),
        "[KRISHIV_INSUFFICIENT_PERMISSIONS] The principal `tester` lacks the SELECT privilege on \
         table `spark_catalog`.`default`.`lineitem`. SQLSTATE: 42501"
    );
    // The first denial stops the query.
    assert_eq!(policy.calls.lock().unwrap().len(), 1);
    // Four tables (lineitem read three times): four calls.
    let (ctx, policy) = with_policy(Ok(AccessDecision::Allow));
    run(q21.clone(), &ctx).unwrap();
    let calls = policy.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 4, "{calls:?}");
    // A policy error fails closed too, without its text.
    let (ctx, _) = with_policy(Err(krishiv_common::error::KrishivUnsupportedFeature {
        feature: "Canary-7f3a9c".to_owned(),
    }
    .into()));
    let e = run(q21, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::KrishivInsufficientPermissions
    );
    assert!(!e.to_string().contains("Canary"));
    // Row filters and column masks are rejected until they can be applied.
    let (ctx, _) = with_policy(Ok(AccessDecision::AllowWith(RowColumnPolicy::default())));
    assert_eq!(
        class(&run(QUERIES[0](), &ctx)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

/// A policy that denies one identifier, exactly.
struct DenyOne(String);

impl AccessPolicy for DenyOne {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        s: &'a Securable,
        _: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if s.to_string() == self.0 {
            AccessDecision::Deny {
                reason: String::new(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

fn scan(name: &[&str], columns: &[(&str, DataType)]) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Table {
            name: name.iter().map(|s| (*s).to_owned()).collect(),
            options: std::collections::BTreeMap::new(),
        },
        output: columns
            .iter()
            .enumerate()
            .map(|(i, (n, t))| crate::BoundColumn {
                id: crate::AttrId(i as u64),
                name: (*n).to_owned(),
                data_type: t.clone(),
                nullable: true,
            })
            .collect(),
        version: None,
    }))
}

/// Every relation that reads data reaches the policy, under the catalog's
/// own identifier; a scan must match its table; a denied principal learns
/// nothing about existence; catalog failures carry no provider text.
#[test]
fn no_relation_bypasses_the_policy() {
    let deny = |ident: &str| {
        let mut c = ctx();
        c.policy = Arc::new(DenyOne(format!("table {ident}")));
        c
    };
    let lineitem = "`spark_catalog`.`default`.`lineitem`";
    let files = |paths: Vec<krishiv_common::secret::OptionValue>| {
        LogicalPlan::new(PlanNode::Read(ReadSource::Files {
            format: "parquet".to_owned(),
            paths,
            options: std::collections::BTreeMap::new(),
        }))
    };
    let key = [("l_orderkey", DataType::Long)];
    let shifted = LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(scan(&["LINEITEM"], &key)),
        offset: 10,
    });
    for plan in [
        LogicalPlan::table(["LINEITEM"]),
        LogicalPlan::table(["Default", "LineItem"]),
        scan(&["lineitem"], &key),
        scan(&["LINEITEM"], &key),
        scan(&["spark_catalog", "Default", "LineItem"], &key),
        shifted,
    ] {
        assert_eq!(
            class(&run(plan, &deny(lineitem))),
            "KRISHIV_INSUFFICIENT_PERMISSIONS"
        );
    }
    // A denied name is denied whether it exists or not.
    let nosuch = "`spark_catalog`.`default`.`nosuch`";
    assert_eq!(
        class(&run(LogicalPlan::table(["nosuch"]), &deny(nosuch))),
        "KRISHIV_INSUFFICIENT_PERMISSIONS"
    );
    // Deny-all reaches every kind of relation.
    let (all, _) = with_policy(Ok(AccessDecision::Deny {
        reason: String::new(),
    }));
    let path = || vec![krishiv_common::secret::OptionValue::from_user("path", "/d")];
    for plan in [
        LogicalPlan::table(["t"]),
        LogicalPlan::table(["nosuch"]),
        scan(&["t"], &[("a", DataType::Integer)]),
    ] {
        assert_eq!(class(&run(plan, &all)), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    }
    // Without a file resolver nothing is read by path; with one, every path
    // is asked about first (`tests::files`).
    let file_scan = LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: path(),
            options: std::collections::BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }));
    for plan in [files(path()), file_scan] {
        assert_eq!(class(&run(plan, &all)), "KRISHIV_UNSUPPORTED_FEATURE");
    }
    // Scans that name no table, a missing one, or invent columns fail closed.
    let ctx = ctx();
    assert_eq!(class(&run(scan(&[], &[]), &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
    assert_eq!(
        class(&run(scan(&["nosuch"], &[]), &ctx)),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    let invented = scan(&["t"], &[("secret_column", DataType::Integer)]);
    assert_eq!(class(&run(invented, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    let retyped = scan(&["t"], &[("a", DataType::Long)]);
    assert_eq!(class(&run(retyped, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// Every scan of a table is checked, not only the first; a spelling of a
/// name does not dodge a rule on another, existing table or not.
#[test]
fn every_scan_and_spelling_is_checked() {
    let ctx = ctx();
    let valid = scan(&["t"], &[("a", DataType::Integer)]);
    let invented = scan(&["t"], &[("secret_column", DataType::Integer)]);
    let crossed = valid
        .clone()
        .join(invented, JoinKind::Cross, JoinCondition::None);
    assert_eq!(class(&run(crossed, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    let mut versioned = scan(&["t"], &[("secret_column", DataType::Integer)]);
    if let PlanNode::Scan(s) = &mut versioned.node {
        s.version = Some(crate::TableVersion(3));
    }
    let crossed = valid.join(versioned, JoinKind::Cross, JoinCondition::None);
    assert_eq!(class(&run(crossed, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    let deny = |ident: &str| {
        let mut c = ctx.clone();
        c.policy = Arc::new(DenyOne(format!("table {ident}")));
        c
    };
    let nosuch = "`spark_catalog`.`default`.`nosuch`";
    for spelling in [["NoSuch"], ["NOSUCH"]] {
        assert_eq!(
            class(&run(LogicalPlan::table(spelling), &deny(nosuch))),
            "KRISHIV_INSUFFICIENT_PERMISSIONS"
        );
    }
}

/// A shift over a plan that is not resolved is not a resolved plan.
#[test]
fn a_shift_over_an_unresolved_plan_is_rejected() {
    let shifted = LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(LogicalPlan::table(["t"])),
        offset: 10,
    });
    assert_eq!(class(&run(shifted, &ctx())), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// A catalog that fails with a message of its own.
struct Failing;

impl crate::analyzer::Catalog for Failing {
    fn name(&self) -> &str {
        "spark_catalog"
    }

    fn epoch(&self) -> u64 {
        0
    }

    fn load_table<'a>(
        &'a self,
        _: &'a crate::analyzer::TableIdent,
        _: Option<TableVersion>,
        _: crate::analyzer::NameForm,
    ) -> BoxFuture<'a, Result<Option<crate::analyzer::TableMeta>, KrishivError>> {
        let e = krishiv_common::error::KrishivUnsupportedFeature {
            feature: "Canary-7f3a9c".to_owned(),
        };
        Box::pin(std::future::ready(Err(e.into())))
    }
}

#[test]
fn catalog_failures_hide_the_provider_text() {
    let mut ctx = ctx();
    ctx.catalogs = Arc::new(crate::analyzer::testing::OneCatalog(
        Arc::new(Failing),
        vec!["default".to_owned()],
    ));
    let e = run(LogicalPlan::table(["t"]), &ctx).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::KrishivCatalogFailure);
    assert!(!format!("{e} {e:?}").contains("Canary"));
}

#[test]
fn already_resolved_plans_are_checked_too() {
    let resolved = run(QUERIES[5](), &ctx()).unwrap().plan;
    let (ctx, policy) = with_policy(Ok(AccessDecision::Deny {
        reason: String::new(),
    }));
    let e = block_on(DefaultAnalyzer.analyze(&resolved, &ctx)).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::KrishivInsufficientPermissions
    );
    assert_eq!(policy.calls.lock().unwrap().len(), 1);
}

#[test]
fn pins_securables_and_fingerprint() {
    let a = run(QUERIES[20](), &ctx()).unwrap();
    let pins: Vec<_> = a.pins.iter().map(|(t, v)| (t.to_string(), v.0)).collect();
    assert_eq!(
        pins,
        [("`spark_catalog`.`default`.`lineitem`".to_owned(), 7)]
    );
    // lineitem is read once at the pin; its other two uses (in subqueries)
    // are instances of that scan.
    let text = a.plan.root().to_string();
    assert_eq!(text.matches("@ version 7").count(), 1, "{text}");
    assert_eq!(text.matches("Shift +").count(), 2, "{text}");
    let used: std::collections::BTreeMap<String, Vec<String>> = a
        .securables
        .iter()
        .map(|s| (s.securable.to_string(), s.columns.iter().cloned().collect()))
        .collect();
    assert_eq!(used.len(), 4);
    assert_eq!(
        used["table `spark_catalog`.`default`.`supplier`"],
        ["s_name", "s_nationkey", "s_suppkey"]
    );
    // `EXISTS (SELECT * FROM lineitem ...)` references every column.
    assert_eq!(used["table `spark_catalog`.`default`.`lineitem`"].len(), 16);
    let q6 = run(QUERIES[5](), &ctx()).unwrap();
    assert_eq!(
        q6.securables[0].columns.iter().collect::<Vec<_>>(),
        ["l_discount", "l_extendedprice", "l_quantity", "l_shipdate"]
    );
    let mut other = ctx();
    other.conf.case_sensitive = true;
    let b = run(QUERIES[20](), &other).unwrap();
    assert_ne!(a.context.digest(), b.context.digest());
    assert_eq!(a.context.catalog_epochs.len(), 1);
}

/// A resolver or registry wrapper that reports another epoch/version.
struct Epoch<T>(T, u64);

impl crate::analyzer::CatalogResolver for Epoch<crate::analyzer::testing::OneCatalog> {
    fn current_catalog(&self) -> &str {
        self.0.current_catalog()
    }

    fn current_namespace(&self) -> &[String] {
        self.0.current_namespace()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn crate::analyzer::Catalog>> {
        self.0.catalog(name)
    }

    fn epoch(&self) -> u64 {
        self.1
    }
}

impl crate::analyzer::FunctionRegistry for Epoch<crate::analyzer::BuiltinFunctions> {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> crate::analyzer::FunctionLookup {
        self.0.lookup(name, profile)
    }

    fn version(&self) -> u64 {
        self.1
    }
}

/// Every input of resolution enters the context fingerprint: changing any
/// one changes the digest.
#[test]
fn every_context_input_changes_the_key() {
    use crate::analyzer::BuiltinFunctions;
    use crate::analyzer::testing::OneCatalog;
    use crate::analyzer::{PrincipalKind, fingerprint};
    let digest = |c: &AnalysisContext, epochs: &[(&str, u64)]| {
        let e = epochs.iter().map(|(n, v)| ((*n).to_owned(), *v)).collect();
        fingerprint(c, e).digest()
    };
    let base = ctx();
    let one = |name: &str, ns: &str| -> Arc<dyn crate::analyzer::CatalogResolver> {
        Arc::new(OneCatalog(
            Arc::new(MemoryCatalog::new(name)),
            vec![ns.to_owned()],
        ))
    };
    let mut variants: Vec<(&str, AnalysisContext)> = Vec::new();
    let mut v = |what, f: &dyn Fn(&mut AnalysisContext)| {
        let mut c = base.clone();
        f(&mut c);
        variants.push((what, c));
    };
    v("profile", &|c| c.profile = SparkProfile::V4_1);
    v("caseSensitive", &|c| c.conf.case_sensitive ^= true);
    v("ansi", &|c| c.conf.ansi ^= true);
    v("groupByAliases", &|c| c.conf.group_by_aliases ^= true);
    v("groupByOrdinal", &|c| c.conf.group_by_ordinal ^= true);
    v("orderByOrdinal", &|c| c.conf.order_by_ordinal ^= true);
    v("lateralColumnAlias", &|c| {
        c.conf.lateral_column_alias ^= true;
    });
    v("current catalog", &|c| c.catalogs = one("other", "default"));
    v("current namespace", &|c| {
        c.catalogs = one("spark_catalog", "db2");
    });
    v("catalogs epoch", &|c| {
        let inner = OneCatalog(
            Arc::new(MemoryCatalog::new("spark_catalog")),
            vec!["default".to_owned()],
        );
        c.catalogs = Arc::new(Epoch(inner, 9));
    });
    v("functions", &|c| {
        c.functions = Arc::new(Epoch(BuiltinFunctions::new(), 9));
    });
    v("sql front end", &|c| c.sql = Some(Arc::new(Frontend)));
    v("principal", &|c| {
        c.principal = Arc::new(Principal::new("other", PrincipalKind::User));
    });
    v("principal kind", &|c| {
        c.principal = Arc::new(Principal::new("tester", PrincipalKind::Service));
    });
    v("groups", &|c| {
        let mut p = Principal::new("tester", PrincipalKind::User);
        p.groups = vec!["admins".to_owned()];
        c.principal = Arc::new(p);
    });
    v("tenant", &|c| {
        let mut p = Principal::new("tester", PrincipalKind::User);
        p.tenant = Some("acme".to_owned());
        c.principal = Arc::new(p);
    });
    v("policy", &|c| c.policy = Arc::new(DenyOne(String::new())));
    let key = digest(&base, &[("spark_catalog", 1)]);
    assert_ne!(
        key,
        digest(&base, &[("spark_catalog", 2)]),
        "catalog epochs"
    );
    for (what, c) in &variants {
        assert_ne!(key, digest(c, &[("spark_catalog", 1)]), "{what}");
    }
    assert_eq!(base.catalogs.epoch(), 0);
}

#[test]
fn resolution_reads_only_its_configs() {
    let read = Mutex::new(Vec::new());
    let conf = ResolutionConf::from_conf(|k| {
        if let Ok(mut r) = read.lock() {
            r.push(k.to_owned());
        }
        (k == "spark.sql.caseSensitive").then_some("TRUE")
    });
    assert!(conf.case_sensitive);
    assert_eq!(read.into_inner().unwrap(), ResolutionConf::KEYS);
    assert_eq!(
        ResolutionConf::from_conf(|_| None),
        ResolutionConf::default()
    );
    // Analysis has no configuration source besides `AnalysisContext::conf`
    // (the type holds none): the same context gives the same result.
    let c = ctx();
    assert!(
        run(QUERIES[0](), &c)
            .unwrap()
            .plan
            .structurally_eq(&run(QUERIES[0](), &c).unwrap().plan)
    );
}

/// SQL text resolves through the front end, which keeps its redacted form.
struct Frontend;

impl SqlFrontend for Frontend {
    fn lower(&self, query: &str, _: SourceId) -> Result<Plan, AnalysisError> {
        if query == "bad" {
            return Err(AnalysisError {
                error: krishiv_common::error::ParseEmptyStatement {}.into(),
                origin: Origin::UNKNOWN,
            });
        }
        Plan::try_new(QUERIES[5]()).map_err(|error| AnalysisError {
            error,
            origin: Origin::UNKNOWN,
        })
    }

    fn redact(&self, query: &str) -> RedactedSql {
        RedactedSql::opaque(query)
    }

    fn fingerprint(&self) -> u64 {
        7
    }
}

#[test]
fn sql_nodes_resolve_through_the_front_end() {
    let sql = |q: &str, args| {
        LogicalPlan::new(PlanNode::Sql {
            query: q.to_owned(),
            args,
        })
    };
    let mut ctx = ctx();
    assert_eq!(
        class(&run(sql("x", SqlArgs::None), &ctx)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    ctx.sql = Some(Arc::new(Frontend));
    let a = run(sql("q6", SqlArgs::None), &ctx).unwrap();
    assert_eq!(a.output[0].name, "revenue");
    assert_eq!(a.query_text.unwrap().as_str(), "*********(redacted)");
    let filtered = sql("q6", SqlArgs::None)
        .filter(Expr::builtin(crate::Builtin::IsNotNull, vec![c("revenue")]));
    assert!(run(filtered, &ctx).unwrap().query_text.is_none());
    assert_eq!(
        class(&run(sql("bad", SqlArgs::None), &ctx)),
        "PARSE_EMPTY_STATEMENT"
    );
    let args = SqlArgs::Positional(vec![Arc::new(int(1))]);
    assert_eq!(
        class(&run(sql("q6", args), &ctx)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn unsupported_inputs_fail_explicitly() {
    let ctx = ctx();
    let files = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        paths: vec![krishiv_common::secret::OptionValue::from_user("path", "/d")],
        options: std::collections::BTreeMap::new(),
    }));
    let sha = LogicalPlan::table(["t"]).project(vec![Expr::function("factorial", vec![c("b")])]);
    for plan in [files, sha] {
        assert_eq!(class(&run(plan, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    }
    // An untyped NULL is the void type; an interval is not supported yet.
    let null = LogicalPlan::table(["t"]).project(vec![Literal::Null(None).into()]);
    let out = run(null, &ctx).unwrap().output;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].data_type, krishiv_types::DataType::Null);
    assert!(out[0].nullable);
    let interval = LogicalPlan::table(["t"]).project(vec![
        Literal::YearMonthInterval {
            months: 1,
            start: crate::YearMonthField::Year,
            end: crate::YearMonthField::Month,
        }
        .into(),
    ]);
    assert_eq!(class(&run(interval, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    // With ordinals off, an integer key is a constant.
    let off = context(
        catalog(),
        ResolutionConf {
            group_by_ordinal: false,
            ..ResolutionConf::default()
        },
    );
    let constant =
        LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![f("count", vec![star()])]);
    assert_eq!(class(&run(constant, &off)), "OK");
    // A crafted, already resolved scan of files is checked like a read: with
    // no file resolver, it is not read at all (`tests::files` covers the
    // policy).
    let (deny, _) = with_policy(Ok(AccessDecision::Deny {
        reason: String::new(),
    }));
    let scan = LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: vec![krishiv_common::secret::OptionValue::from_user("path", "/d")],
            options: std::collections::BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }));
    assert_eq!(class(&run(scan, &deny)), "KRISHIV_UNSUPPORTED_FEATURE");
    // A function newer than the session's Spark version does not exist.
    let mut old = ctx.clone();
    old.profile = SparkProfile::V3_5;
    let randstr = LogicalPlan::table(["t"]).project(vec![Expr::function("randstr", vec![int(3)])]);
    assert_eq!(class(&run(randstr.clone(), &old)), "UNRESOLVED_ROUTINE");
    assert_eq!(class(&run(randstr, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    // The argument count is checked first, with Spark's `expectedNum` text.
    let upper =
        LogicalPlan::table(["t"]).project(vec![Expr::function("upper", vec![int(1), int(2)])]);
    let err = run(upper, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
    );
    let text = err.error.to_string();
    assert!(
        text.contains("The `upper` requires 1 parameters but the actual number is 2"),
        "{text}"
    );
}

/// A column a projection dropped comes back from below it, as Spark's
/// missing references do, down to an aggregate's grouping attributes
/// (which the aggregate then outputs); the result resolves to itself.
#[test]
fn missing_references_reach_grouping_attributes() {
    let ctx = ctx();
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![f("max", vec![c("c")]).alias("m")])
        .project(vec![c("m")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let a = run(plan, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["m"]
    );
    assert_eq!(
        a.plan.root().to_string(),
        "Project [m#3]\n\
         +- Sort [a#0 ASC NULLS FIRST], true\n   \
         +- Project [m#3, a#0]\n      \
         +- Aggregate [a#0], [max(c#2) AS m#3, a#0]\n         \
         +- Scan spark_catalog.default.t [a#0, b#1, c#2]\n"
    );
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
}

/// In a subquery, a column a projection dropped hides an outer column of
/// its name (Spark resolves missing references before outer ones); a name
/// only the enclosing query has still resolves there.
#[test]
fn missing_references_come_before_outer_ones() {
    let ctx = ctx();
    let inner = |name: &str| {
        LogicalPlan::table(["u"])
            .project(vec![c("a")])
            .filter(eq(c(name), c("a")))
    };
    let query = |name: &str| {
        LogicalPlan::table(["t"])
            .project(vec![c("a").alias("d"), c("b")])
            .filter(Expr::subquery(crate::SubqueryKind::Exists, inner(name)))
    };
    let shown = |name: &str| run(query(name), &ctx).unwrap().plan.root().to_string();
    // `d` is u's own column, dropped below the filter, not the outer alias.
    let own = shown("d");
    assert!(own.contains("Filter (CAST(d#"), "{own}");
    assert!(!own.contains("outer("), "{own}");
    // `b` exists only in the enclosing query: the subquery is correlated,
    // a semi join on it.
    let outer = shown("b");
    assert!(outer.contains("Join LeftSemi, (CAST(b#1 "), "{outer}");
}

/// A subexpression shared through `Arc` inside a lateral alias item and a
/// sort key over a dropped column is bound and walked once.
#[test]
fn shared_subexpressions_in_rewrites() {
    let ctx = ctx();
    let twice = |name: &str| {
        let x = Arc::new(c(name));
        Expr::new(ExprNode::Function(crate::FunctionCall {
            name: "+".to_owned(),
            args: vec![Arc::clone(&x), x],
            distinct: false,
            filter: None,
        }))
    };
    let lateral =
        LogicalPlan::table(["t"]).project(vec![f("max", vec![c("a")]).alias("m"), twice("m")]);
    let a = run(lateral, &ctx).unwrap();
    assert_eq!(
        a.output[1].name,
        "(lateralAliasReference(m) + lateralAliasReference(m))"
    );
    let sorted = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .sort(vec![crate::SortOrder::asc(twice("c"))]);
    let a = run(sorted, &ctx).unwrap();
    assert!(
        a.plan
            .root()
            .to_string()
            .contains("Sort [(c#2 + c#2) ASC NULLS FIRST]"),
        "{}",
        a.plan.root()
    );
}

/// A registry entry prints its identity, kind and inputs (never its result
/// closure).
#[test]
fn function_entries_print_their_signature() {
    use crate::analyzer::{FunctionLookup, FunctionRegistry as _};
    let FunctionLookup::Found(entry) = crate::analyzer::BuiltinFunctions::new()
        .lookup(&["substring".to_owned()], SparkProfile::default())
    else {
        panic!("substring is registered");
    };
    let text = format!("{entry:?}");
    assert!(
        text.contains("inputs: Expects([OneOf([String, Binary]), Exact(Integer), Exact(Integer)])"),
        "{text}"
    );
    // Built-ins are unqualified; a name only Spark has is known.
    let b = crate::analyzer::BuiltinFunctions::default();
    let qualified = ["builtin".to_owned(), "upper".to_owned()];
    assert!(matches!(
        b.lookup(&qualified, SparkProfile::V4_2),
        FunctionLookup::NotFound
    ));
    let uuid = ["uuid".to_owned()];
    assert!(matches!(
        b.lookup(&uuid, SparkProfile::V4_2),
        FunctionLookup::KnownUnsupported
    ));
    assert_eq!(b.version(), 0);
}

#[test]
fn deep_plans_resolve_on_a_small_stack_and_added_nodes_count() {
    on_small_stack(|| {
        let ctx = ctx();
        // Nested scalar subqueries: the resolver recurses per level.
        let deep =
            (0..MAX_DEPTH / 3 - 2).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
                LogicalPlan::table(["t"])
                    .project(vec![Expr::subquery(crate::SubqueryKind::Scalar, p)])
            });
        assert_eq!(class(&run(deep, &ctx)), "OK");
        // A column list on an alias adds a projection: one level too many.
        let wide = (0..MAX_DEPTH - 2).fold(LogicalPlan::table(["t"]), |p, _| {
            p.filter(gt(c("a"), int(0)))
        });
        let renamed = LogicalPlan::new(PlanNode::SubqueryAlias {
            input: Arc::new(wide),
            alias: "x".to_owned(),
            columns: vec!["p".to_owned(), "q".to_owned(), "r".to_owned()],
        });
        assert_eq!(class(&run(renamed, &ctx)), "FAILED_TO_PARSE_TOO_COMPLEX");
    });
}

#[test]
fn principals_securables_and_contexts_print_safely() {
    let local = Principal::local_user();
    assert!(local.id().starts_with("user:``/"));
    let mut service = Principal::new("etl", crate::analyzer::PrincipalKind::Service);
    service.tenant = Some("acme".to_owned());
    service.groups = vec!["b".to_owned(), "a".to_owned(), "a".to_owned()];
    assert_eq!(service.id(), "service:`acme`/`etl`[`a`.`b`]");
    let ident = crate::analyzer::TableIdent {
        catalog: "c".to_owned(),
        namespace: vec!["d".to_owned()],
        name: "v`w".to_owned(),
    };
    let url = krishiv_common::secret::OptionValue::from_user("path", "s3a://k:Canary-7f3a9c@b/p");
    let shown: Vec<String> = [
        Securable::View(ident),
        Securable::Function(vec!["f".to_owned()]),
        Securable::Path(url),
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    assert_eq!(
        shown,
        [
            "view `c`.`d`.`v``w`",
            "function `f`",
            "path 's3a://k:*********(redacted)@b/p'"
        ]
    );
    assert!(format!("{:?}", ctx()).starts_with("AnalysisContext"));
    // Errors keep the span of the node that raised them.
    let sql = "SELECT x FROM t";
    let at = Origin::sql(crate::SourceId(0), crate::Span::new(7, 8));
    let plan = LogicalPlan::table(["t"]).project(vec![c("x").with_origin(at)]);
    let e = run(plan, &ctx()).unwrap_err();
    let q = e.query_context(&RedactedSql::opaque(sql)).unwrap();
    assert!(
        matches!(
            q,
            QueryContext::Sql {
                start: 7,
                stop: 7,
                ..
            }
        ),
        "{q:?}"
    );
    // A DataFrame error names the relation that raised it.
    let df = LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(5)));
    let df_error = run(df.project(vec![col("x", 5)]), &ctx()).unwrap_err();
    assert_eq!(
        df_error.query_context(&RedactedSql::opaque(sql)),
        Some(QueryContext::DataFrame { plan_id: PlanId(5) })
    );
    assert_eq!(
        KrishivError::from(e).error_class(),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
}

/// A catalog whose table gets a new version on every load, as under a
/// concurrent writer.
struct Committing(MemoryCatalog, Mutex<i64>);

impl crate::analyzer::Catalog for Committing {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn epoch(&self) -> u64 {
        self.0.epoch()
    }

    fn load_table<'a>(
        &'a self,
        id: &'a crate::analyzer::TableIdent,
        version: Option<TableVersion>,
        form: crate::analyzer::NameForm,
    ) -> BoxFuture<'a, Result<Option<crate::analyzer::TableMeta>, KrishivError>> {
        let version = version.or_else(|| {
            let mut v = self.1.lock().ok()?;
            *v += 1;
            Some(TableVersion(*v))
        });
        self.0.load_table(id, version, form)
    }
}

#[test]
fn one_snapshot_per_table_per_query() {
    let committing = Committing(catalog(), Mutex::new(0));
    let mut ctx = ctx();
    ctx.catalogs = Arc::new(crate::analyzer::testing::OneCatalog(
        Arc::new(committing),
        vec!["default".to_owned()],
    ));
    // `t` and `T` load twice; both read the version pinned first.
    let plan = LogicalPlan::table(["t"])
        .alias("x")
        .join(
            LogicalPlan::table(["T"]).alias("y"),
            JoinKind::Cross,
            JoinCondition::None,
        )
        .project(vec![c("x.a"), c("y.a")]);
    let a = run(plan, &ctx).unwrap();
    assert_eq!(a.pins.values().map(|v| v.0).collect::<Vec<_>>(), [1]);
    assert_eq!(a.plan.root().to_string().matches("@ version 1").count(), 2);
}

#[test]
fn stars_in_calls_and_elsewhere() {
    let ctx = ctx();
    let concat = LogicalPlan::table(["u"]).project(vec![f("concat", vec![star()])]);
    assert_eq!(run(concat, &ctx).unwrap().output[0].name, "concat(a, d)");
    let filtered = LogicalPlan::table(["t"]).filter(star());
    assert_eq!(
        class(&run(filtered, &ctx)),
        "INVALID_USAGE_OF_STAR_OR_REGEX"
    );
    // A USING column two left-side columns could mean.
    let joined = LogicalPlan::table(["t"])
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Cross,
            JoinCondition::None,
        )
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::Using(vec!["a".to_owned()]),
        );
    assert_eq!(class(&run(joined, &ctx)), "AMBIGUOUS_REFERENCE");
    // An already resolved scan of files is checked against the files, which
    // needs a file resolver (see `tests::files`).
    let scan = LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: vec![krishiv_common::secret::OptionValue::from_user("path", "/d")],
            options: std::collections::BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }));
    assert_eq!(class(&run(scan, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn types_follow_the_catalog() {
    let a = run(
        LogicalPlan::table(["t"]).project(vec![c("c"), c("b")]),
        &ctx(),
    )
    .unwrap();
    let types: Vec<_> = a.output.iter().map(|c| c.data_type.clone()).collect();
    assert_eq!(types, [DataType::Long, DataType::String]);
}

/// `resolved_output` on hand-written resolved shapes.
#[test]
fn resolved_output_follows_the_contract() {
    let col = |id: u64, name: &str| crate::BoundColumn {
        id: crate::AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: false,
    };
    let scan = |ids: &[u64]| {
        Arc::new(LogicalPlan::new(PlanNode::Scan(crate::Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: std::collections::BTreeMap::new(),
            },
            output: ids.iter().map(|i| col(*i, "a")).collect(),
            version: None,
        })))
    };
    let join = |kind| {
        LogicalPlan::new(PlanNode::Join {
            left: scan(&[1]),
            right: scan(&[2]),
            kind,
            condition: JoinCondition::None,
        })
    };
    let nullable = |p: LogicalPlan| -> Vec<bool> {
        p.resolved_output()
            .unwrap()
            .iter()
            .map(|c| c.nullable)
            .collect()
    };
    assert_eq!(nullable(join(JoinKind::LeftOuter)), [false, true]);
    assert_eq!(nullable(join(JoinKind::RightOuter)), [true, false]);
    assert_eq!(nullable(join(JoinKind::FullOuter)), [true, true]);
    assert_eq!(nullable(join(JoinKind::LeftSemi)), [false]);
    let alias = |columns: Vec<String>| {
        LogicalPlan::new(PlanNode::SubqueryAlias {
            input: scan(&[1]),
            alias: "x".to_owned(),
            columns,
        })
    };
    assert_eq!(
        alias(vec!["z".to_owned()]).resolved_output().unwrap()[0].name,
        "z"
    );
    assert!(
        alias(vec!["y".to_owned(), "z".to_owned()])
            .resolved_output()
            .is_none()
    );
    let shift = |offset| {
        LogicalPlan::new(PlanNode::Shift {
            input: scan(&[3]),
            offset,
        })
    };
    assert_eq!(shift(4).resolved_output().unwrap()[0].id, crate::AttrId(7));
    assert!(shift(u64::MAX).resolved_output().is_none());
    let filtered = LogicalPlan::new(PlanNode::Filter {
        input: scan(&[1]),
        predicate: Arc::new(Literal::Boolean(true).into()),
    });
    assert_eq!(filtered.resolved_output().unwrap().len(), 1);
    assert!(LogicalPlan::table(["t"]).resolved_output().is_none());
    assert!(
        LogicalPlan::table(["t"])
            .project(vec![c("a")])
            .resolved_output()
            .is_none()
    );
}

/// `SELECT` without `FROM` reads one row of no columns; the plan resolves
/// to itself, prints as Spark's `OneRowRelation` and has a key of its own.
#[test]
fn one_row_relation_has_no_columns() {
    let ctx = ctx();
    let one = LogicalPlan::new(PlanNode::OneRowRelation);
    let a = run(one.clone(), &ctx).unwrap();
    assert!(a.output.is_empty());
    assert_eq!(a.plan.root().resolved_output(), Some(Vec::new()));
    assert_eq!(a.plan.root().to_string(), "OneRowRelation\n");
    let again = run(a.plan.root().as_ref().clone(), &ctx).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
    let projected = one.project(vec![int(1)]);
    let p = run(projected, &ctx).unwrap();
    assert_eq!(p.output.len(), 1);
    assert_ne!(p.plan.structural_hash(), a.plan.structural_hash());
}

#[test]
fn try_arithmetic_casts_its_arguments_in_try_mode() {
    let ctx = ctx();
    let project = |name: &str| {
        let call = Expr::function(name, vec![s("x"), int(1)]);
        run(LogicalPlan::table(["t"]).project(vec![call]), &ctx).unwrap()
    };
    let try_add = project("try_add");
    let text = format!("{:?}", try_add.plan);
    assert!(text.contains("mode: Try"), "{text}");
    // `mod` casts as the operator does (ANSI here).
    let text = format!("{:?}", project("mod").plan);
    assert!(
        text.contains("mode: Ansi") && !text.contains("mode: Try"),
        "{text}"
    );
    // A TRY cast of a string may be NULL; `try_*` results are nullable anyway.
    assert!(try_add.output[0].nullable);
}
