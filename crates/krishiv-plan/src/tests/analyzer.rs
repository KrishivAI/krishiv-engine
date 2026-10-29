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
use super::dsl::{add, c, count_star, date, dec, eq, f, ge, gt, int, le, lt, s, star};
use super::tpch::QUERIES;
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AccessDecision, AccessPolicy, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _,
    BoxFuture, DefaultAnalyzer, MemoryCatalog, PolicyVersion, Principal, Privilege, QueryContext,
    ResolutionConf, RowColumnPolicy, Securable, SparkProfile, SqlFrontend, TableColumn,
};
use crate::walk::{Node, visit_distinct};
use crate::{
    ColumnRef, Cte, Expr, ExprNode, JoinCondition, JoinKind, Literal, LogicalPlan, MAX_DEPTH,
    Origin, Plan, PlanId, PlanNode, ReadSource, ResolvedExpr, SourceId, SqlArgs, Star,
    TableVersion,
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

/// Ids of every resolved input column under `e`, in walk order.
fn resolved_ids(e: &Arc<Expr>) -> Vec<crate::AttrId> {
    let mut ids = Vec::new();
    let _ = visit_distinct(Node::Expr(e), |n| {
        if let Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
        {
            ids.push(c.id);
        }
        ControlFlow::Continue(())
    });
    ids
}

/// The grouping expressions of an aggregate node.
fn grouping_keys(grouping: &crate::Grouping) -> Vec<Arc<Expr>> {
    match grouping {
        crate::Grouping::GroupBy(keys)
        | crate::Grouping::Rollup(keys)
        | crate::Grouping::Cube(keys) => keys.clone(),
        crate::Grouping::GroupingSets { exprs, .. } => exprs.clone(),
    }
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
                except: Vec::new(),
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

/// `TIMESTAMP` and `TIMESTAMP_NTZ` literals are named as Spark's
/// `Literal.sql` prints them (`SELECT TIMESTAMP '2024-01-01 00:00:00'` in
/// Spark 4.2); SQL does not lower timestamp literals yet, so the plan is
/// built by hand.
#[test]
fn timestamp_literals_are_named_as_spark_sql() {
    let lit = |v: Literal| Expr::from(v);
    let r = run(
        LogicalPlan::table(["t"]).project(vec![
            lit(Literal::Timestamp(1_704_067_200_000_000)),
            lit(Literal::TimestampNtz(1_704_067_200_123_456)),
            lit(Literal::Timestamp(-1)),
        ]),
        &ctx(),
    );
    let names: Vec<String> = r.unwrap().output.iter().map(|c| c.name.clone()).collect();
    assert_eq!(
        names,
        [
            "TIMESTAMP '2024-01-01 00:00:00'",
            "TIMESTAMP_NTZ '2024-01-01 00:00:00.123456'",
            "TIMESTAMP '1969-12-31 23:59:59.999999'",
        ]
    );
}

/// The select list of the generated-names statement of the corpus.
fn names_query() -> Vec<Expr> {
    use crate::Builtin::{EqNullSafe, IsNotNull, IsNull, Remainder, Subtract};
    let lit = |v: Literal| Expr::from(v);
    let date = lit(Literal::Date(18263));
    let decimal = lit(Literal::Decimal {
        unscaled: 15,
        data_type: krishiv_types::DecimalType::try_new(2, 1).unwrap(),
    });
    let op = |b, args| Expr::builtin(b, args);
    vec![
        add(c("a"), int(1)),
        // Unary `-` lowers as `"-"` (see `builtin_op`): the function
        // `negative(x)` is a registry function instead.
        Expr::function("-", vec![c("a")]),
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
                except: Vec::new(),
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
                qualifier: Vec::new(),
                arg_names: Vec::new(),
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

/// `mask` quotes the expanded call (with the omitted defaults materialized)
/// in input-type errors, as Spark does (`mask(b, a, x, n, NULL)`).
#[test]
fn mask_errors_quote_the_expanded_call() {
    let r = run(
        LogicalPlan::table(["t"]).project(vec![f("mask", vec![c("b"), c("a")])]),
        &ctx(),
    );
    let e = r.unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    let p: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    // `a` is an `int`, not a `STRING`; the error quotes the call with the
    // defaults for `lowerChar`, `digitChar` and `otherChar`.
    assert_eq!(p["sqlExpr"], "\"mask(b, a, x, n, NULL)\"");
    assert_eq!(p["inputSql"], "\"a\"");
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
                qualifier: Vec::new(),
                arg_names: Vec::new(),
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
    use crate::analyzer::{PrincipalKind, SessionVariable, VariableSnapshot, fingerprint};
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
    v("variables", &|c| {
        c.variables = Arc::new(VariableSnapshot {
            variables: [(
                "v".to_owned(),
                SessionVariable {
                    data_type: DataType::Integer,
                    value: Literal::Integer(1),
                },
            )]
            .into_iter()
            .collect(),
            version: 1,
        });
    });
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
    // The timestamp type and session zone flow through `from_conf`: unset
    // is UTC, fixed offsets resolve, regions and malformed zones are `None`.
    let ntz =
        ResolutionConf::from_conf(|k| (k == "spark.sql.timestampType").then_some("TIMESTAMP_NTZ"));
    assert!(ntz.timestamp_ntz);
    assert_eq!(ntz.session_zone, Some(0));
    let plus =
        ResolutionConf::from_conf(|k| (k == "spark.sql.session.timeZone").then_some("+05:30"));
    assert_eq!(plus.session_zone, Some(19_800));
    assert!(!plus.timestamp_ntz);
    let region = ResolutionConf::from_conf(|k| {
        (k == "spark.sql.session.timeZone").then_some("America/Los_Angeles")
    });
    assert_eq!(region.session_zone, None);
    let bad =
        ResolutionConf::from_conf(|k| (k == "spark.sql.session.timeZone").then_some("Mars/Base"));
    assert_eq!(bad.session_zone, None);
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
        // `gone` names a missing table: the unused-CTE table check reads it
        // from the lowered plan Bind stores, without ever resolving it.
        let plan = if query == "gone" {
            LogicalPlan::table(["gone"]).project(vec![c("a")])
        } else {
            QUERIES[5]()
        };
        Plan::try_new(plan).map_err(|error| AnalysisError {
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

    fn keywords(&self) -> Vec<(String, bool)> {
        Vec::new()
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
    // `randstr` is implemented: it resolves on current versions.
    assert_eq!(class(&run(randstr, &ctx)), "OK");
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

/// The cast-named functions (`float(x)`, `int(x)`, ...): Spark analyzes
/// each call as `CAST(x AS target)`, so the analyzer rewrites the call
/// to a cast during analysis and the plan only ever holds the `Cast`
/// node. The output column is named by the argument, as Spark names it.
#[test]
fn cast_named_functions_rewrite_to_casts() {
    use krishiv_types::DecimalType;

    let ctx = ctx();
    let double = |v: f64| Expr::from(Literal::Double(v));
    let long = |v: i64| Expr::from(Literal::Long(v));
    let boolean = |v: bool| Expr::from(Literal::Boolean(v));
    let decimal = |p: u32, s: u32| DataType::Decimal(DecimalType::try_new(p, s).unwrap());
    // (call, type, output name, nullable).
    let cases: Vec<(Expr, DataType, &str, bool)> = vec![
        (
            f("float", vec![double(15.0)]),
            DataType::Float,
            "15.0",
            false,
        ),
        (f("double", vec![s("1.5")]), DataType::Double, "1.5", true),
        (f("int", vec![s("7")]), DataType::Integer, "7", true),
        (f("boolean", vec![int(1)]), DataType::Boolean, "1", false),
        (f("tinyint", vec![long(1)]), DataType::Byte, "1", false),
        (f("smallint", vec![long(1)]), DataType::Short, "1", false),
        (
            f("bigint", vec![boolean(true)]),
            DataType::Long,
            "true",
            false,
        ),
        (
            f("decimal", vec![s("1.234")]),
            decimal(10, 0),
            "1.234",
            true,
        ),
        (f("string", vec![int(42)]), DataType::String, "42", false),
        (f("binary", vec![s("x")]), DataType::Binary, "x", false),
        (
            f("date", vec![s("2024-01-02")]),
            DataType::Date,
            "2024-01-02",
            true,
        ),
        (
            f("timestamp", vec![s("2024-01-02 12:34:56")]),
            DataType::Timestamp,
            "2024-01-02 12:34:56",
            true,
        ),
        // A cast of a column keeps the column's name.
        (f("int", vec![c("c")]), DataType::Integer, "c", true),
        // Names resolve ASCII case-insensitively.
        (f("FLOAT", vec![int(1)]), DataType::Float, "1", false),
    ];
    for (call, data_type, name, nullable) in cases {
        let plan = LogicalPlan::table(["t"]).project(vec![call]);
        let a = run(plan, &ctx).unwrap();
        assert_eq!(a.output.len(), 1);
        assert_eq!(a.output[0].data_type, data_type);
        assert_eq!(a.output[0].name, name);
        assert_eq!(a.output[0].nullable, nullable);
        // The plan holds a cast, never a function call.
        let text = a.plan.root().to_string();
        assert!(text.contains("CAST("), "{text}");
        assert!(!text.contains("float(") && !text.contains("int("), "{text}");
    }
    // Resolving a resolved plan changes nothing.
    let plan = LogicalPlan::table(["t"]).project(vec![f("int", vec![s("7")])]);
    let a = run(plan, &ctx).unwrap();
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
}

/// The arity error of a cast-named function quotes the call, with
/// Spark's `expectedNum` text.
#[test]
fn cast_named_functions_check_arity() {
    let ctx = ctx();
    for (args, actual) in [(Vec::new(), "0"), (vec![int(1), int(2)], "2")] {
        let plan = LogicalPlan::table(["t"]).project(vec![f("float", args)]);
        let err = run(plan, &ctx).unwrap_err();
        assert_eq!(
            err.error.error_class().name(),
            "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
        );
        let params = err.error.message_parameters();
        let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
        assert_eq!(get("functionName"), "`float`");
        assert_eq!(get("expectedNum"), "1");
        assert_eq!(get("actualNum"), actual);
    }
}

/// A pair the cast path does not support fails as `CAST(x AS T)` fails,
/// quoting the argument as Spark does.
#[test]
fn cast_named_functions_fail_like_casts() {
    let ctx = ctx();
    let binary = Expr::from(Literal::Binary(vec![1]));
    let plan = LogicalPlan::table(["t"]).project(vec![f("boolean", vec![binary])]);
    let err = run(plan, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION"
    );
    let params = err.error.message_parameters();
    let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
    assert_eq!(get("sqlExpr"), "\"X'01'\"");
    assert_eq!(get("srcType"), "\"BINARY\"");
    assert_eq!(get("targetType"), "\"BOOLEAN\"");
    // A suggestion or a config hint fails as `CAST` does, quoting the
    // argument.
    let plan = LogicalPlan::table(["t"]).project(vec![f("bigint", vec![date("2024-01-02")])]);
    let err = run(plan, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITH_FUNC_SUGGESTION"
    );
    let params = err.error.message_parameters();
    let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
    assert_eq!(get("sqlExpr"), "\"CAST(2024-01-02 AS DATE)\"");
    assert_eq!(get("functionNames"), "`UNIX_DATE`");
    let plan = LogicalPlan::table(["t"]).project(vec![f("binary", vec![int(1)])]);
    let err = run(plan, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITH_CONF_SUGGESTION"
    );
    let params = err.error.message_parameters();
    let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
    assert_eq!(get("sqlExpr"), "\"1\"");
    assert_eq!(get("srcType"), "\"INT\"");
    assert_eq!(get("targetType"), "\"BINARY\"");
    // Without ANSI the same pair is allowed, as `CAST` allows it.
    let legacy = context(
        catalog(),
        ResolutionConf {
            ansi: false,
            ..ResolutionConf::default()
        },
    );
    let plan = LogicalPlan::table(["t"]).project(vec![f("binary", vec![int(1)])]);
    let a = run(plan, &legacy).unwrap();
    assert_eq!(a.output[0].data_type, DataType::Binary);
    assert_eq!(a.output[0].name, "1");
    // `timestamp_ntz` is not a function.
    let plan = LogicalPlan::table(["t"]).project(vec![f("timestamp_ntz", vec![int(1)])]);
    assert_eq!(class(&run(plan, &ctx)), "UNRESOLVED_ROUTINE");
    // A malformed string analyzes in both modes (it fails at run time):
    // nullability follows the cast, which may yield NULL.
    for ansi in [true, false] {
        let ctx = context(
            catalog(),
            ResolutionConf {
                ansi,
                ..ResolutionConf::default()
            },
        );
        let plan = LogicalPlan::table(["t"]).project(vec![f("int", vec![s("x")])]);
        let a = run(plan, &ctx).unwrap();
        assert_eq!(a.output[0].data_type, DataType::Integer);
        assert_eq!(a.output[0].name, "x");
        assert!(a.output[0].nullable);
    }
}

/// `DISTINCT` and `FILTER` have no cast form: a cast-named function with
/// either is unsupported syntax on the function, as for any other
/// non-aggregate function.
#[test]
fn cast_named_functions_reject_distinct_and_filter() {
    let ctx = ctx();
    let distinct = Expr::new(ExprNode::Function(crate::FunctionCall {
        name: "float".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("a"))],
        distinct: true,
        filter: None,
    }));
    let r = run(LogicalPlan::table(["t"]).project(vec![distinct]), &ctx);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX",
            {"prettyName": "`float`", "syntax": "DISTINCT"}
        ])
    );
    let filtered = Expr::new(ExprNode::Function(crate::FunctionCall {
        name: "float".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("a"))],
        distinct: false,
        filter: Some(Arc::new(Literal::Boolean(true).into())),
    }));
    let r = run(LogicalPlan::table(["t"]).project(vec![filtered]), &ctx);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX",
            {"prettyName": "`float`", "syntax": "FILTER"}
        ])
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

/// Session variables resolve after columns and outer references: a name
/// a column has is the column, a qualified `session.v` skips the columns,
/// and anything else stays an unresolved column.
#[test]
fn session_variables_resolve_after_columns() {
    use crate::analyzer::{SessionVariable, VariableSnapshot};
    use std::collections::BTreeMap;
    let catalog = || {
        MemoryCatalog::new("spark_catalog").with_table(
            &["default"],
            "t",
            vec![TableColumn {
                name: "var1".to_owned(),
                data_type: DataType::String,
                nullable: true,
            }],
            None,
        )
    };
    let variables = || {
        let mut ctx = context(catalog(), ResolutionConf::default());
        ctx.variables = Arc::new(VariableSnapshot {
            variables: BTreeMap::from([(
                "var1".to_owned(),
                SessionVariable {
                    data_type: DataType::Integer,
                    value: Literal::Integer(1),
                },
            )]),
            version: 3,
        });
        ctx
    };
    // A name a column has is the column, even when a variable has it too.
    let plan = LogicalPlan::table(["t"]).project(vec![c("var1")]);
    let a = run(plan, &variables()).unwrap();
    assert_eq!(a.output[0].data_type, DataType::String);
    // A qualified `session.v` skips the columns for the variable.
    for name in ["session.var1", "system.session.var1", "SESSION.VAR1"] {
        let plan = LogicalPlan::table(["t"]).project(vec![c(name)]);
        let a = run(plan, &variables()).unwrap();
        assert_eq!(a.output[0].data_type, DataType::Integer, "{name}");
    }
    // Anything else stays an unresolved column, never a variable error.
    // (Outer references before variables are covered end to end, where a
    // correlated `EXISTS` reads the outer column.)
    for name in ["nope", "ses.var1", "b.ses.var1", "var1.var1"] {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![c(name)]);
        assert_eq!(
            class(&run(plan, &variables())),
            "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION",
            "{name}"
        );
    }
}

/// A changed variable version changes the context fingerprint, so a
/// cached resolved plan is never reused after a variable changed.
#[test]
fn variable_versions_change_the_fingerprint() {
    use crate::analyzer::{SessionVariable, VariableSnapshot, fingerprint};
    use std::collections::BTreeMap;
    let digest = |c: &AnalysisContext| fingerprint(c, BTreeMap::new()).digest();
    let base = ctx();
    let key = digest(&base);
    let mut changed = base.clone();
    changed.variables = Arc::new(VariableSnapshot {
        variables: BTreeMap::from([(
            "var1".to_owned(),
            SessionVariable {
                data_type: DataType::Integer,
                value: Literal::Integer(1),
            },
        )]),
        version: base.variables.version + 1,
    });
    assert_ne!(key, digest(&changed));
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
            qualifier: Vec::new(),
            arg_names: Vec::new(),
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
        FunctionLookup::Found(_)
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

/// Typed datetime literals resolve in the analyzer with the session zone.
#[test]
fn typed_literals_resolve_with_the_session_zone() {
    use crate::{ExprNode, Literal, LogicalPlan, Plan, PlanNode, ResolvedExpr, TypedLiteralKind};
    let project = |kind: TypedLiteralKind, value: &str, conf: ResolutionConf| {
        let expr = Expr::new(ExprNode::TypedLiteral {
            kind,
            value: value.to_owned(),
        });
        let plan = LogicalPlan::new(crate::PlanNode::Project {
            input: std::sync::Arc::new(LogicalPlan::new(crate::PlanNode::OneRowRelation)),
            exprs: vec![std::sync::Arc::new(expr)],
        });
        let plan = Plan::try_new(plan).unwrap();
        let ctx = context(MemoryCatalog::new("spark_catalog"), conf);
        block_on(DefaultAnalyzer.analyze(&plan, &ctx))
    };
    // The instant a one-column project holds, unwrapping its alias.
    let micros_of = |plan: &LogicalPlan| {
        let PlanNode::Project { exprs, .. } = &plan.node else {
            panic!("not a project: {plan}");
        };
        let inner = match &exprs[0].node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
            ExprNode::Literal(_) => &exprs[0],
            other => panic!("not a literal: {other:?}"),
        };
        match &inner.node {
            ExprNode::Literal(Literal::Timestamp(v)) => *v,
            other => panic!("not a timestamp: {other:?}"),
        }
    };
    let utc = ResolutionConf {
        session_zone: Some(0),
        ..ResolutionConf::default()
    };
    let plus = ResolutionConf {
        session_zone: Some(19_800),
        ..ResolutionConf::default()
    };
    let ntz_conf = ResolutionConf {
        timestamp_ntz: true,
        session_zone: Some(0),
        ..ResolutionConf::default()
    };
    let region = ResolutionConf {
        session_zone: None,
        ..ResolutionConf::default()
    };
    // Plain `TIMESTAMP` is an instant by default, wall-clock under NTZ.
    let a = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    assert_eq!(a.output[0].name, "TIMESTAMP '2024-01-01 10:00:00'");
    let a = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", ntz_conf).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp_ntz");
    assert_eq!(a.output[0].name, "TIMESTAMP_NTZ '2024-01-01 10:00:00'");
    // An explicit zone stays an instant under NTZ and ignores the session.
    let a = project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00+05:30",
        ntz_conf,
    )
    .unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    // The session zone moves wall-clock instants (`+05:30` is 04:30 UTC):
    // Spark 4.2 gives 1_704_103_200_000_000 µs in UTC and
    // 1_704_083_400_000_000 µs in +05:30.
    let utc_plan = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", utc).unwrap();
    let plus_plan = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", plus).unwrap();
    assert_eq!(micros_of(utc_plan.plan.root()), 1_704_103_200_000_000);
    assert_eq!(micros_of(plus_plan.plan.root()), 1_704_083_400_000_000);
    let a = project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00+00:00",
        plus,
    )
    .unwrap();
    assert_eq!(a.output[0].name, "TIMESTAMP '2024-01-01 15:30:00'");
    // `DATE` resolves without a zone.
    let a = project(TypedLiteralKind::Date, "2016-03-12", utc).unwrap();
    assert_eq!(a.output[0].name, "DATE '2016-03-12'");
    assert_eq!(a.output[0].data_type.simple_string(), "date");
    // Malformed text is `INVALID_TYPED_LITERAL`.
    for (kind, value) in [
        (TypedLiteralKind::Timestamp, "2024-13-01 10:00:00"),
        (TypedLiteralKind::TimestampNtz, "10:00"),
        (TypedLiteralKind::Date, "2020-02-30"),
        (TypedLiteralKind::Date, "not-a-date"),
        (TypedLiteralKind::Date, "10:00"),
    ] {
        let e = project(kind, value, utc).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidTypedLiteral,
            "{value}"
        );
    }
    // Clock-dependent spellings need the query clock.
    for (kind, value) in [
        (TypedLiteralKind::Timestamp, "now"),
        (TypedLiteralKind::TimestampLtz, "10:00"),
        (TypedLiteralKind::Timestamp, "10:00"),
    ] {
        let e = project(kind, value, utc).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{value}"
        );
    }
    // A region session serves explicit zones but not wall-clock time.
    project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00+05:30",
        region,
    )
    .unwrap();
    let e = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", region).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A region id in the value needs the time-zone database.
    let e = project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00 America/Los_Angeles",
        utc,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

/// A conditional with `FILTER` names the filter once: the shown call takes
/// only the function's own arguments, not the filter child.
#[test]
fn conditional_with_filter_names_the_filter_once() {
    let ctx = ctx();
    let call = Expr::new(ExprNode::Function(crate::FunctionCall {
        name: "nvl".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("a")), Arc::new(int(1))],
        distinct: false,
        filter: Some(Arc::new(eq(c("b"), s("x")))),
    }));
    let a = run(LogicalPlan::table(["t"]).project(vec![call]), &ctx).unwrap();
    assert_eq!(a.output[0].name, "nvl(a, 1) FILTER (WHERE (b = x))");
}

/// Built-in functions plus `rand`, a nondeterministic `DOUBLE` function as
/// in Spark (the engine does not implement it yet; `sqrt`'s signature
/// stands in for its typing).
struct WithRand(crate::analyzer::BuiltinFunctions);

impl crate::analyzer::FunctionRegistry for WithRand {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> crate::analyzer::FunctionLookup {
        use crate::analyzer::FunctionLookup;
        if name.len() == 1 && name[0].eq_ignore_ascii_case("rand") {
            return match self.0.lookup(&["sqrt".to_owned()], profile) {
                FunctionLookup::Found(mut e) => {
                    e.deterministic = false;
                    FunctionLookup::Found(e)
                }
                other => other,
            };
        }
        self.0.lookup(name, profile)
    }

    fn version(&self) -> u64 {
        self.0.version()
    }
}

/// `OFFSET rand(1)` is `IS_UNFOLDABLE` (cited as `_nondeterministic`), not
/// `DATA_TYPE` for its `DOUBLE`: Spark checks foldability first (Spark 4.2.0
/// raises the same for `OFFSET rand()`).
#[test]
fn a_nondeterministic_count_is_unfoldable_before_its_type() {
    let mut cx = ctx();
    cx.functions = Arc::new(WithRand(crate::analyzer::BuiltinFunctions::new()));
    let rand = || Expr::function("rand", vec![Expr::from(Literal::Integer(1))]);
    for (plan, name) in [
        (LogicalPlan::table(["t"]).offset(rand()), "offset"),
        (LogicalPlan::table(["t"]).limit(rand()), "limit"),
    ] {
        let e = run(plan, &cx).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidLimitLikeExpressionIsUnfoldable,
            "{e:?}"
        );
        let msg = e.error.to_string();
        assert!(
            msg.contains("\"_nondeterministic\"") && msg.contains(name),
            "{msg}"
        );
    }
}

/// An unused CTE is not resolved: its column errors never fail the query.
/// Spark's golden `cte` case 50 keeps an unused `cte3` whose projection
/// over the duplicate-case columns of `cte2` would be ambiguous.
#[test]
fn unused_cte_definitions_are_not_resolved() {
    let ctx = ctx();
    let valid = LogicalPlan::table(["t"]).project(vec![c("a")]);
    let ambiguous = LogicalPlan::table(["t"])
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::None,
        )
        .project(vec![c("a")]);
    assert_eq!(class(&run(ambiguous.clone(), &ctx)), "AMBIGUOUS_REFERENCE");
    // The main query reads only the valid definition; the ambiguous one
    // is never referenced, so the query resolves.
    let unused = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["good"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(valid),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(ambiguous),
            },
        ],
    });
    let a = run(unused, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    // Referencing the bad definition fails.
    let used = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["bad"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(
                    LogicalPlan::table(["t"])
                        .join(
                            LogicalPlan::table(["u"]),
                            JoinKind::Inner,
                            JoinCondition::None,
                        )
                        .project(vec![c("a")]),
                ),
            },
        ],
    });
    assert_eq!(class(&run(used, &ctx)), "AMBIGUOUS_REFERENCE");
}

/// `SELECT 1 GROUP BY `1`` groups by the select-list alias `1`, even
/// though the item has no explicit alias (Spark golden `group-by-alias`
/// case 11). With `groupByAliases` off the same key is unresolved.
#[test]
fn group_by_resolves_implicit_select_aliases() {
    let plan = LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["1"]
    );
    let conf = ResolutionConf {
        group_by_aliases: false,
        ..ResolutionConf::default()
    };
    let no_alias = context(catalog(), conf);
    let plan = LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    assert_eq!(
        class(&run(plan, &no_alias)),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
}

/// A grouping expression may not contain an aggregate call, however it was
/// written: directly, or through a select-list alias (`SELECT COUNT(b) AS k
/// ... GROUP BY k`, Spark golden `group-by-alias`). Without the check the
/// alias resolves to the aggregate, whose semantic key is already in the
/// grouping-key set `check_aggregation` skips, so the plan wrongly builds.
#[test]
fn group_by_rejects_aggregate_functions() {
    let aliased = LogicalPlan::table(["t"])
        .aggregate(vec![c("k")], vec![f("count", vec![c("b")]).alias("k")]);
    assert_eq!(class(&run(aliased, &ctx())), "GROUP_BY_AGGREGATE");
    let direct = LogicalPlan::table(["t"]).aggregate(vec![f("count", vec![c("b")])], vec![c("a")]);
    assert_eq!(class(&run(direct, &ctx())), "GROUP_BY_AGGREGATE");
}

/// A `GROUP BY` key naming a select-list alias that reads lateral column
/// aliases is `UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY` (Spark
/// golden `column-resolution-aggregate`: `SELECT k AS lca, lca + 1 AS col
/// ... GROUP BY k, col`). Grouping by a lateral alias itself stays fine.
#[test]
fn group_by_rejects_aliases_reading_lateral_aliases() {
    let items = || vec![c("a").alias("lca"), add(c("lca"), int(1)).alias("col")];
    let lateral = LogicalPlan::table(["t"]).aggregate(vec![c("col")], items());
    assert_eq!(
        class(&run(lateral, &ctx())),
        "UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY"
    );
    let direct = LogicalPlan::table(["t"]).aggregate(vec![c("lca")], items());
    run(direct, &ctx()).unwrap();
}

/// A plan shared under two inner `WITH` scopes with different shadowing
/// names is visited under each scope: visits are keyed by scope contents,
/// not scope length, so a shadowing scope cannot hide a read the other
/// scope reports and drop a definition the input needs.
#[test]
fn cte_refs_visit_shared_plans_under_each_scope() {
    let shared = Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")]));
    let inner = |name: &str| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::clone(&shared),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(LogicalPlan::table(["u"]).project(vec![c("a")])),
            }],
        })
    };
    // The right branch shadows `x`, so `shared` is visited under its scope
    // first (a join pops its right child first): keying visits by scope
    // length alone would skip the left branch's read and drop the outer `x`.
    let input = inner("y").join(inner("x"), JoinKind::Inner, JoinCondition::None);
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(input),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "a"]
    );
}

/// The same shared subquery expression under two inner `WITH` scopes with
/// different shadowing names is visited under each scope too: expression
/// visits are keyed by scope contents as well, or the shadowed visit hides
/// the read the other scope's subquery reports.
#[test]
fn cte_refs_visit_shared_exprs_under_each_scope() {
    let sub: Arc<Expr> = Arc::new(Expr::subquery(
        crate::SubqueryKind::Scalar,
        LogicalPlan::table(["x"]).project(vec![c("a")]),
    ));
    let branch = |name: &str| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::new(PlanNode::Project {
                input: Arc::new(LogicalPlan::table(["t"])),
                exprs: vec![Arc::clone(&sub)],
            })),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(LogicalPlan::table(["u"]).project(vec![c("a")])),
            }],
        })
    };
    // As above, the right branch shadows `x` and is visited first.
    let input = branch("y").join(branch("x"), JoinKind::Inner, JoinCondition::None);
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(input),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["scalarsubquery()", "scalarsubquery()"]
    );
}

/// An inner `WITH` name shadows the outer one: reads inside the inner scope
/// resolve to the inner definition, so an outer definition nothing else
/// reads is dropped (even when it would not resolve), while a read outside
/// the inner scope still needs the outer definition.
#[test]
fn inner_with_names_shadow_outer_ctes() {
    let bad = || {
        LogicalPlan::table(["t"])
            .join(
                LogicalPlan::table(["u"]),
                JoinKind::Inner,
                JoinCondition::None,
            )
            .project(vec![c("a")])
    };
    assert_eq!(class(&run(bad(), &ctx())), "AMBIGUOUS_REFERENCE");
    let good = || LogicalPlan::table(["t"]).project(vec![c("a")]);
    let inner = || {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "x".to_owned(),
                plan: Arc::new(good()),
            }],
        })
    };
    // Only the inner `x` is read: the outer one is dropped, so its
    // ambiguity never fails the query.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(inner()),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    let a = run(shadowed, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    // A read outside the inner scope still needs the outer definition.
    let outer_read = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(inner().join(
            LogicalPlan::table(["x"]).project(vec![c("a")]),
            JoinKind::Inner,
            JoinCondition::None,
        )),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    assert_eq!(class(&run(outer_read, &ctx())), "AMBIGUOUS_REFERENCE");
}

/// A definition's reads see only the outer CTEs and the earlier definitions:
/// a forward reference never marks a later definition, so the read falls
/// through to table binding.
#[test]
fn cte_forward_references_fall_through_to_tables() {
    // `a` reads `b`, which is defined later: `b` is dropped, and resolving
    // `a` reads a missing table.
    let forward = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["a"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "a".to_owned(),
                plan: Arc::new(LogicalPlan::table(["b"]).project(vec![c("a")])),
            },
            Cte {
                name: "b".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
        ],
    });
    assert_eq!(class(&run(forward, &ctx())), "TABLE_OR_VIEW_NOT_FOUND");
    // The same definitions in dependency order resolve.
    let ordered = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["b"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "a".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "b".to_owned(),
                plan: Arc::new(LogicalPlan::table(["a"]).project(vec![c("a")])),
            },
        ],
    });
    let a = run(ordered, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
}

/// CTE-name matching follows `spark.sql.caseSensitive`: case-insensitive by
/// default, exact when set.
#[test]
fn cte_name_matching_follows_case_sensitivity() {
    let plan = || {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "X".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            }],
        })
    };
    let a = run(plan(), &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..ResolutionConf::default()
    };
    assert_eq!(
        class(&run(plan(), &context(catalog(), sensitive))),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

/// `GROUP BY ALL` groups by every select item without an aggregate; a lone
/// `all` that resolves to a real column stays that column (Spark's
/// `ResolveReferencesInAggregate`).
#[test]
fn group_by_all_expands_over_non_aggregates() {
    use crate::{ExprNode, Grouping};

    fn grouping_of(plan: &LogicalPlan) -> Vec<String> {
        let mut keys = Vec::new();
        let _ = visit_distinct(Node::Plan(plan), |node| {
            if let Node::Plan(p) = node
                && let PlanNode::Aggregate { grouping, .. } = &p.node
                && let Grouping::GroupBy(ks) = grouping
            {
                keys.extend(ks.iter().map(|k| match &k.node {
                    ExprNode::Resolved(crate::ResolvedExpr::Column(c)) => c.name.clone(),
                    other => format!("{other:?}"),
                }));
            }
            ControlFlow::Continue(())
        });
        keys
    }

    // Only SQL has a `GROUP BY ALL`: the plan carries its span.
    let sql = || Origin::sql(crate::SourceId(0), crate::Span::new(0, 1));
    // `a` groups; `count(*)` does not.
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("all")], vec![c("a"), f("count", vec![star()])])
        .with_origin(sql());
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "count(1)"]
    );
    assert_eq!(grouping_of(a.plan.root()).len(), 1);
    // An item mixing a bare column with an aggregate cannot be inferred.
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("all")], vec![add(c("a"), count_star())])
        .with_origin(sql());
    assert_eq!(class(&run(plan, &ctx())), "UNRESOLVED_ALL_IN_GROUP_BY");
    // A column actually named `all` takes precedence over the expansion.
    let plan = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("all"), c("b")])
        .aggregate(vec![c("all")], vec![c("all"), f("count", vec![star()])])
        .with_origin(sql());
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(grouping_of(a.plan.root()).len(), 1);
    // A DataFrame `groupBy("all")` has no SQL span: an unknown column.
    let plan =
        LogicalPlan::table(["t"]).aggregate(vec![c("all")], vec![c("a"), f("count", vec![star()])]);
    assert_eq!(
        class(&run(plan, &ctx())),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
}

/// `ORDER BY ALL` sorts by every output column; a column actually named
/// `all` takes precedence (Spark's `ResolveReferencesInSort`).
#[test]
fn order_by_all_sorts_by_every_output_column() {
    use crate::SortOrder;

    let plan = LogicalPlan::table(["t"])
        .project(vec![c("a"), c("b")])
        .sort(vec![SortOrder::asc(c("all"))]);
    let a = run(plan, &ctx()).unwrap();
    let mut orders = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            orders.extend(order.iter().map(|o| o.expr.to_string()));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(orders.len(), 2);
    let plan = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("all"), c("b")])
        .sort(vec![SortOrder::asc(c("all"))]);
    let a = run(plan, &ctx()).unwrap();
    let mut orders = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            orders.extend(order.iter().map(|o| o.expr.to_string()));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(orders.len(), 1);
    // `SORT BY ALL` / `sortWithinPartitions("all")` is a non-global sort:
    // `all` is an unknown column there.
    let plan = LogicalPlan::new(PlanNode::Sort {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a"), c("b")])),
        order: vec![SortOrder::asc(c("all"))],
        global: false,
    });
    assert_eq!(
        class(&run(plan, &ctx())),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
}

/// `* EXCEPT` drops the named columns; overlapping entries and nested
/// fields fail with Spark's classes, unknown names resolve as usual.
#[test]
fn star_except_drops_overlaps_and_rejects_nested() {
    use crate::Star;

    let except = |cols: &[&str]| {
        Expr::new(ExprNode::Star(Star {
            qualifier: Vec::new(),
            except: cols
                .iter()
                .map(|c| c.split('.').map(str::to_owned).collect())
                .collect(),
            plan_id: None,
        }))
    };
    let names = |plan: LogicalPlan| {
        run(plan, &ctx()).map(|a| a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
    };
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["b"])]);
    assert_eq!(names(plan).unwrap(), ["a", "c"]);
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["a", "a"])]);
    assert_eq!(class(&run(plan, &ctx())), "EXCEPT_OVERLAPPING_COLUMNS");
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["a", "a", "a.b"])]);
    assert_eq!(class(&run(plan, &ctx())), "EXCEPT_OVERLAPPING_COLUMNS");
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["a.b"])]);
    assert_eq!(class(&run(plan, &ctx())), "KRISHIV_UNSUPPORTED_FEATURE");
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["z"])]);
    assert_eq!(
        class(&run(plan, &ctx())),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
}

/// `NATURAL JOIN` joins on the common column names in left order and
/// projects like `USING`; with no common columns it is a cross join.
#[test]
fn natural_join_uses_common_columns_in_left_order() {
    let names = |plan: LogicalPlan, ctx: &AnalysisContext| {
        run(plan, ctx).map(|a| a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
    };
    let join = || {
        LogicalPlan::table(["t"]).join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::Natural,
        )
    };
    assert_eq!(
        names(join().project(vec![star()]), &ctx()).unwrap(),
        ["a", "b", "c", "d"]
    );
    // Case-insensitive by default: `A` still meets `a`.
    let upper = LogicalPlan::table(["t"]).join(
        LogicalPlan::table(["u"]).project(vec![c("a").alias("A"), c("d")]),
        JoinKind::Inner,
        JoinCondition::Natural,
    );
    assert_eq!(
        names(upper.project(vec![star()]), &ctx()).unwrap(),
        ["a", "b", "c", "d"]
    );
    // Case-sensitive: `a` and `A` share nothing, so the join is a cross.
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..ResolutionConf::default()
    };
    let upper = LogicalPlan::table(["t"]).join(
        LogicalPlan::table(["u"]).project(vec![c("a").alias("A"), c("d")]),
        JoinKind::Inner,
        JoinCondition::Natural,
    );
    let a = run(upper.project(vec![star()]), &context(catalog(), sensitive)).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "b", "c", "A", "d"]
    );
}

/// `UNRESOLVED_COLUMN` proposals rank quoted candidates by edit distance
/// to the unquoted name, ties in name order (Spark's
/// `orderSuggestedIdentifiersBySimilarity`, checked against Spark 4.2.0).
#[test]
fn unresolved_column_proposals_rank_like_spark() {
    let renamed = LogicalPlan::table(["wide"]).project(vec![
        c("col1").alias("country"),
        c("col2").alias("city"),
        c("col3").alias("name"),
        c("col4").alias("id"),
        c("col5").alias("power"),
    ]);
    let r = run(renamed.project(vec![c("non_exist")]), &ctx());
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            {
                "objectName": "`non_exist`",
                "proposal": "`id`, `name`, `power`, `city`, `country`",
            }
        ])
    );
    let renamed = LogicalPlan::table(["t"]).project(vec![
        c("a").alias("id"),
        c("b").alias("name"),
        c("c").alias("data"),
    ]);
    let r = run(renamed.project(vec![c("invalid_column")]), &ctx());
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            {
                "objectName": "`invalid_column`",
                "proposal": "`id`, `name`, `data`",
            }
        ])
    );
}
/// An unresolved inline table over `rows` with `columns`.
fn inline_table(rows: Vec<Vec<Expr>>, columns: &[&str]) -> LogicalPlan {
    LogicalPlan::new(PlanNode::InlineTable {
        rows: rows
            .into_iter()
            .map(|r| r.into_iter().map(Arc::new).collect())
            .collect(),
        columns: columns.iter().map(|c| (*c).to_owned()).collect(),
    })
}

fn null() -> Expr {
    Literal::Null(None).into()
}

fn long(v: i64) -> Expr {
    Literal::Long(v).into()
}

fn boolean(v: bool) -> Expr {
    Literal::Boolean(v).into()
}

/// Whether an `InlineTable` node remains anywhere under `plan`.
fn has_inline_table(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = visit_distinct(Node::Plan(plan), |node| {
        if let Node::Plan(p) = node
            && matches!(p.node, PlanNode::InlineTable { .. })
        {
            found = true;
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    found
}

fn datatypes(a: &AnalyzedPlan) -> Vec<DataType> {
    a.output.iter().map(|c| c.data_type.clone()).collect()
}

/// A single row is just the `Project`: names, types and nullability come
/// from the row, and no `InlineTable` node remains.
#[test]
fn inline_table_of_one_row_is_a_project() {
    let ctx = ctx();
    let r = run(
        inline_table(vec![vec![s("one"), int(1)]], &["col1", "col2"]),
        &ctx,
    );
    assert_eq!(
        summary(&r),
        serde_json::json!([["col1", false], ["col2", false]])
    );
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    assert!(!has_inline_table(a.plan.root()));
    assert!(
        matches!(a.plan.root().node, PlanNode::Project { .. }),
        "{}",
        a.plan.root()
    );
    assert_eq!(datatypes(&a), [DataType::String, DataType::Integer]);
}

/// Several rows are one `Expand` over `OneRowRelation` with a projection
/// per row, in row order; each column widens to the wider type and is
/// nullable where any row's value is.
#[test]
fn inline_table_rows_widen_and_nullable() {
    let ctx = ctx();
    let r = run(
        inline_table(
            vec![
                vec![s("one"), int(1)],
                vec![s("two"), long(2)],
                vec![s("three"), null()],
            ],
            &["a", "b"],
        ),
        &ctx,
    );
    assert_eq!(summary(&r), serde_json::json!([["a", false], ["b", true]]));
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    assert!(!has_inline_table(a.plan.root()));
    assert_eq!(datatypes(&a), [DataType::String, DataType::Long]);
    let PlanNode::Expand {
        input,
        projections,
        output,
    } = &a.plan.root().node
    else {
        panic!("an expand of row projections:\n{}", a.plan.root());
    };
    assert!(matches!(input.node, PlanNode::OneRowRelation));
    assert_eq!(projections.len(), 3);
    assert_eq!(output.len(), 2);
    for row in projections {
        assert_eq!(row.len(), 2);
    }
}

/// An all-NULL column is `VOID`.
#[test]
fn inline_table_of_nulls_is_void() {
    let ctx = ctx();
    let r = run(
        inline_table(
            vec![vec![s("one"), null()], vec![s("two"), null()]],
            &["a", "b"],
        ),
        &ctx,
    );
    assert_eq!(summary(&r), serde_json::json!([["a", false], ["b", true]]));
    assert_eq!(datatypes(&r.unwrap()), [DataType::String, DataType::Null]);
}

/// A column whose values share no type names it, quoted, in
/// `INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE` (`INT` meets
/// `STRING` at `BIGINT`, so only the second column fails here).
#[test]
fn inline_table_of_incompatible_types_fails() {
    let ctx = ctx();
    let r = run(
        inline_table(
            vec![vec![int(1), int(2)], vec![s("x"), boolean(true)]],
            &["a", "b"],
        ),
        &ctx,
    );
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE",
            {"colName": "`b`"}
        ])
    );
}

/// An aggregate, a nondeterministic call, a window and a subquery cannot
/// be evaluated in an inline table.
#[test]
fn inline_table_of_unevaluable_expressions_fails() {
    let ctx = ctx();
    for (cell, expr) in [
        (f("count", vec![int(1)]), "\"count(1)\""),
        (f("rand", vec![int(5)]), "\"rand(5)\""),
    ] {
        let r = run(inline_table(vec![vec![cell]], &["a"]), &ctx);
        assert_eq!(
            summary(&r),
            serde_json::json!([
                "INVALID_INLINE_TABLE.CANNOT_EVALUATE_EXPRESSION_IN_INLINE_TABLE",
                {"expr": expr}
            ]),
            "{expr}"
        );
    }
    // An unknown routine is `UNRESOLVED_ROUTINE`, as elsewhere.
    let r = run(
        inline_table(vec![vec![f("nosuchfunc", vec![int(1)])]], &["a"]),
        &ctx,
    );
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_ROUTINE",
            {
                "routineName": "`nosuchfunc`",
                "searchPath": "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]"
            }
        ])
    );
    // A window and a subquery cannot be evaluated either.
    let window = Expr::new(ExprNode::Window(crate::WindowExpr {
        function: Arc::new(f("row_number", Vec::new())),
        window: crate::WindowSpecRef::Spec(crate::WindowSpec {
            partition_by: Vec::new(),
            order_by: Vec::new(),
            frame: None,
        }),
    }));
    let r = run(inline_table(vec![vec![window]], &["a"]), &ctx);
    assert_eq!(
        class(&r),
        "INVALID_INLINE_TABLE.CANNOT_EVALUATE_EXPRESSION_IN_INLINE_TABLE"
    );
    let subquery = Expr::new(ExprNode::Subquery {
        kind: crate::SubqueryKind::Scalar,
        plan: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
    });
    let r = run(inline_table(vec![vec![subquery]], &["a"]), &ctx);
    assert_eq!(
        class(&r),
        "INVALID_INLINE_TABLE.CANNOT_EVALUATE_EXPRESSION_IN_INLINE_TABLE"
    );
}

/// The clock functions evaluate once per query, so they may stay.
#[test]
fn inline_table_of_clock_calls_resolves() {
    let ctx = ctx();
    let r = run(
        inline_table(vec![vec![f("now", Vec::new())]], &["ct"]),
        &ctx,
    );
    assert_eq!(summary(&r), serde_json::json!([["ct", false]]));
    let a = r.unwrap();
    assert_eq!(datatypes(&a), [DataType::Timestamp]);
    assert!(!has_inline_table(a.plan.root()));
}

/// Degenerate hand-built tables (the parser rejects these): no rows is an
/// error.
#[test]
fn inline_table_edge_cases() {
    let ctx = ctx();
    let r = run(
        LogicalPlan::new(PlanNode::InlineTable {
            rows: Vec::new(),
            columns: Vec::new(),
        }),
        &ctx,
    );
    assert_eq!(class(&r), "KRISHIV_UNSUPPORTED_FEATURE");
    // Rewriting an inline table's expressions visits every cell (and an
    // unchanged rewrite shares the plan).
    let p = Arc::new(inline_table(vec![vec![int(1)]], &["a"]));
    let q = Arc::clone(&p)
        .map_expressions(|e| Ok::<_, ()>(Arc::clone(&e)))
        .expect("an infallible rewrite");
    assert!(Arc::ptr_eq(&p, &q));
}

/// A column name resolves against nothing in an inline table: it is
/// `UNRESOLVED_COLUMN`, as in Spark (not `CANNOT_EVALUATE_...`), also
/// inside a larger expression.
#[test]
fn inline_table_of_unknown_column_is_unresolved() {
    let ctx = ctx();
    for cell in [c("x"), add(c("x"), int(1))] {
        let r = run(inline_table(vec![vec![int(0)], vec![cell]], &["a"]), &ctx);
        assert_eq!(
            summary(&r),
            serde_json::json!(["UNRESOLVED_COLUMN.WITHOUT_SUGGESTION", {"objectName": "`x`"}])
        );
    }
}

/// A large inline table analyzes in time linear in its rows: ten thousand
/// rows become one `Expand` with ten thousand projections, not a deep tree.
#[test]
fn inline_table_of_many_rows_resolves() {
    let ctx = ctx();
    let rows: Vec<Vec<Expr>> = (0..10_000).map(|i| vec![int(i), s("v")]).collect();
    let r = run(inline_table(rows, &["a", "b"]), &ctx);
    assert_eq!(summary(&r), serde_json::json!([["a", false], ["b", false]]));
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    assert!(!has_inline_table(a.plan.root()));
    let PlanNode::Expand { projections, .. } = &a.plan.root().node else {
        panic!("one expand:\n{}", a.plan.root());
    };
    assert_eq!(projections.len(), 10_000);
}

/// A hand-built row of the wrong width fails rather than NULL-fills or
/// truncates (the parser rejects these).
#[test]
fn inline_table_of_ragged_rows_fails() {
    let ctx = ctx();
    for rows in [vec![vec![int(1)], vec![]], vec![vec![int(1), int(2)]]] {
        let r = run(inline_table(rows, &["a"]), &ctx);
        assert_eq!(class(&r), "KRISHIV_UNSUPPORTED_FEATURE");
    }
}

/// A DataFrame reference (`df["a"]`) names a column no input has.
#[test]
fn inline_table_of_dataframe_reference_fails() {
    let ctx = ctx();
    let cell = Expr::new(ExprNode::Column(ColumnRef {
        name: vec!["a".to_owned()],
        plan_id: Some(PlanId(1)),
    }));
    let r = run(inline_table(vec![vec![cell]], &["a"]), &ctx);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "CANNOT_RESOLVE_DATAFRAME_COLUMN",
            {"name": "\"a\""}
        ])
    );
}

/// Without ANSI a string never meets another type in an inline table; with
/// ANSI it does (`INT` and `STRING` meet at `BIGINT`).
#[test]
fn inline_table_string_promotion_follows_ansi() {
    let rows = || inline_table(vec![vec![int(1)], vec![s("x")]], &["a"]);
    let mut off = ctx();
    off.conf.ansi = false;
    let r = run(rows(), &off);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE",
            {"colName": "`a`"}
        ])
    );
    let mut on = ctx();
    on.conf.ansi = true;
    let r = run(rows(), &on);
    assert_eq!(datatypes(&r.unwrap()), [DataType::Long]);
    // The remaining widening is the set operations' either way.
    let r = run(
        inline_table(vec![vec![int(1)], vec![long(2)]], &["a"]),
        &off,
    );
    assert_eq!(datatypes(&r.unwrap()), [DataType::Long]);
}

/// An unused CTE over a missing relation still fails the query with
/// `TABLE_OR_VIEW_NOT_FOUND` (Spark golden `cte` cases 16 and 24): table
/// existence is checked even though the definition is never resolved.
/// Column errors of an unused definition still never fail the query.
#[test]
fn unused_cte_tables_still_fail() {
    let ctx = ctx();
    let missing = |name: &str| LogicalPlan::table([name]).project(vec![c("a")]);
    // An unused definition over a missing table fails.
    let unused_missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(missing("nope")),
        }],
    });
    let e = run(unused_missing, &ctx).unwrap_err();
    assert_eq!(e.error.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(
        params.get("relationName").map(String::as_str),
        Some("`nope`")
    );
    // The main query reading only the valid definition still resolves when
    // the other definition is ambiguous (column errors are skipped), but
    // fails when the other definition reads a missing table.
    let ambiguous = LogicalPlan::table(["t"])
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::None,
        )
        .project(vec![c("a")]);
    assert_eq!(class(&run(ambiguous.clone(), &ctx)), "AMBIGUOUS_REFERENCE");
    let ok = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["good"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(ambiguous),
            },
        ],
    });
    run(ok, &ctx).unwrap();
    let missing_bad = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["good"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(missing("gone")),
            },
        ],
    });
    assert_eq!(class(&run(missing_bad, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
}

/// A forward reference inside a CTE falls through to tables: the later
/// definition is not visible, so reading it is a missing table (Spark
/// golden `cte` case 24, `with r1 as (select * from r2), r2 as ...`).
#[test]
fn cte_forward_references_are_missing_tables() {
    let ctx = ctx();
    let forward = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "r1".to_owned(),
                plan: Arc::new(LogicalPlan::table(["r2"]).project(vec![c("a")])),
            },
            Cte {
                name: "r2".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
        ],
    });
    let e = run(forward, &ctx).unwrap_err();
    assert_eq!(e.error.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("relationName").map(String::as_str), Some("`r2`"));
    // In dependency order the same definitions resolve.
    let ordered = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["b"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "a".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "b".to_owned(),
                plan: Arc::new(LogicalPlan::table(["a"]).project(vec![c("a")])),
            },
        ],
    });
    run(ordered, &ctx).unwrap();
}

/// Unused CTE bodies nested inside an unused definition are checked too:
/// a missing table at any depth fails the query, while an ambiguous
/// column at any depth does not.
#[test]
fn unused_cte_tables_are_checked_at_any_depth() {
    let ctx = ctx();
    let nested_missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "inner".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["missing_deep"]).project(vec![c("a")])),
                }],
            })),
        }],
    });
    assert_eq!(class(&run(nested_missing, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
    // A self-reference inside an unused definition is a missing table, not
    // infinite recursion (Spark golden `cte` cases 4 and 5).
    let self_ref = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "s".to_owned(),
            plan: Arc::new(LogicalPlan::table(["s"]).project(vec![c("a")])),
        }],
    });
    assert_eq!(class(&run(self_ref, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
}

/// CTE name matching is case-insensitive by default: an unused definition
/// shadowing a table with different casing still hides it from table
/// checks, and exact when `caseSensitive` is set.
#[test]
fn unused_cte_shadowing_follows_case_sensitivity() {
    let ctx = ctx();
    // `T` (upper) shadows table `t` case-insensitively: reading `t` in the
    // main query is the CTE, and the unused `T` body over `t` reads the
    // table (which exists), so no table error.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["T"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "T".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    run(shadowed, &ctx).unwrap();
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..ResolutionConf::default()
    };
    let sensitive_ctx = context(catalog(), sensitive);
    // Case-sensitive: `T` does not shadow `t`, so the main query reads the
    // table and the CTE body is unused but over an existing table too.
    run(
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "T".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            }],
        }),
        &sensitive_ctx,
    )
    .unwrap();
    // Case-sensitive: an unused `T` over a missing `T` still fails, since
    // no shadowing applies to its own body.
    let missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "T".to_owned(),
            plan: Arc::new(LogicalPlan::table(["T_missing"]).project(vec![c("a")])),
        }],
    });
    assert_eq!(
        class(&run(missing, &sensitive_ctx)),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

/// `GROUP BY` resolves a backquoted select-list alias (even an implicit one
/// like `` `1` `` for `SELECT 1`): with `groupByAliases` on the key groups
/// by the alias expression, with it off the same key is unresolved (Spark
/// golden `group-by-alias` cases 11 and 32, `having` cases 24 and 25).
/// Hand-built plans use plain names: backquotes are SQL syntax only.
#[test]
fn group_by_resolves_backquoted_select_aliases() {
    let ctx = ctx();
    // Explicit alias (`SELECT a AS k ... GROUP BY k`, backquoted in SQL).
    let aliased = LogicalPlan::table(["t"]).aggregate(vec![c("k")], vec![c("a").alias("k")]);
    run(aliased, &ctx).unwrap();
    // Implicit alias: `SELECT 1 GROUP BY `1`` groups by the literal.
    let implicit = LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    let a = run(implicit, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["1"]
    );
    // With aliases off the same keys are unresolved, with a suggestion when
    // the input has columns and without one over `OneRowRelation`.
    let off = context(
        catalog(),
        ResolutionConf {
            group_by_aliases: false,
            ..ResolutionConf::default()
        },
    );
    let aliased_off = LogicalPlan::table(["t"]).aggregate(vec![c("k")], vec![c("a").alias("k")]);
    assert_eq!(
        class(&run(aliased_off, &off)),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let implicit_off =
        LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    assert_eq!(
        class(&run(implicit_off, &off)),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
    // `SELECT 1 AS `2`, 2 ... GROUP BY `2`` (Spark golden `having` cases 24
    // and 25): the key resolves to the input column, which takes precedence
    // over the select-list aliases (constants need no grouping).
    let backquoted = LogicalPlan::table(["t"])
        .aggregate(vec![c("b")], vec![int(1).alias("b"), int(2).alias("b")]);
    run(backquoted, &ctx).unwrap();
}

/// `GROUP BY` resolves the first matching select-list alias: duplicate
/// aliases do not fail the grouping itself (Spark golden `group-by-alias`
/// case 27 succeeds).
#[test]
fn group_by_picks_the_first_matching_alias() {
    let ctx = ctx();
    // Checked on a Spark 4.2.0 server: `SELECT a AS x, a AS x FROM t GROUP BY x`
    // groups by the first `x` and succeeds; with `b AS x` second the key
    // still resolves, and the query fails only because `b` is not grouped.
    let dup = LogicalPlan::table(["t"])
        .aggregate(vec![c("x")], vec![c("a").alias("x"), c("a").alias("x")]);
    run(dup, &ctx).unwrap();
    let ungrouped = LogicalPlan::table(["t"])
        .aggregate(vec![c("x")], vec![c("a").alias("x"), c("b").alias("x")]);
    assert_eq!(class(&run(ungrouped, &ctx)), "MISSING_AGGREGATION");
    let dup_group = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")]);
    // Resolves to the first alias rather than raising `AMBIGUOUS_REFERENCE`.
    let a = run(dup_group, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "a"]
    );
}

/// `HAVING` and `ORDER BY` over an aggregate read grouping attributes when
/// the select list repeats a name (Spark golden `having` cases 24 and 25,
/// `group-by-alias` case 28's `HAVING`): without leniency the duplicate
/// outputs are `AMBIGUOUS_REFERENCE` where Spark resolves to the grouping
/// column.
#[test]
fn having_and_order_by_prefer_grouping_over_duplicate_outputs() {
    let ctx = ctx();
    // `SELECT 1 AS a, 2 AS a FROM t GROUP BY a HAVING a > 0`: `a` in the
    // grouping and the `HAVING` resolves to the grouping column (input
    // `a`), not to either duplicate output.
    let having = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")])
        .filter(gt(c("a"), int(0)));
    let a = run(having, &ctx).unwrap();
    // The predicate and the grouping key share the input column's id, which
    // neither duplicate output carries.
    let mut predicates = Vec::new();
    let mut groupings = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node {
            match &p.node {
                PlanNode::Filter { predicate, .. } => {
                    predicates.extend(resolved_ids(predicate));
                }
                PlanNode::Aggregate { grouping, .. } => {
                    groupings.extend(grouping_keys(grouping).iter().flat_map(resolved_ids));
                }
                _ => {}
            }
        }
        ControlFlow::Continue(())
    });
    assert_eq!(predicates.len(), 1);
    assert_eq!(groupings, predicates);
    let outputs: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert!(!outputs.contains(&predicates[0]));
    // Without a grouping column of that name, the same shape stays
    // ambiguous: two outputs, nothing below to prefer.
    let ambiguous = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![int(1)], vec![int(1).alias("a"), int(2).alias("a")]);
    let ambiguous = ambiguous.filter(gt(c("a"), int(0)));
    assert_eq!(class(&run(ambiguous, &ctx)), "AMBIGUOUS_REFERENCE");
    // `ORDER BY` over the aggregate behaves the same way.
    let order = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let a = run(order, &ctx).unwrap();
    // The sort key is the grouping column (input `a`), not either duplicate
    // output.
    let mut keys = Vec::new();
    let mut groupings = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node {
            match &p.node {
                PlanNode::Sort { order, .. } => {
                    keys.extend(order.iter().flat_map(|o| resolved_ids(&o.expr)));
                }
                PlanNode::Aggregate { grouping, .. } => {
                    groupings.extend(grouping_keys(grouping).iter().flat_map(resolved_ids));
                }
                _ => {}
            }
        }
        ControlFlow::Continue(())
    });
    assert_eq!(keys.len(), 1);
    assert_eq!(groupings, keys);
    let outputs: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert!(!outputs.contains(&keys[0]));
    let order_ambiguous = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![int(1)], vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    // Spark 4.2.0 reports `UNRESOLVED_COLUMN` (not `AMBIGUOUS_REFERENCE`)
    // for `SELECT 1 AS a, 2 AS a GROUP BY 1 ORDER BY a`, proposing the
    // duplicate outputs.
    let e = run(order_ambiguous, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`a`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`, `a`"));
}

/// `ORDER BY` over an aggregate with an unknown column stays unresolved:
/// with no grouping attribute of that name either, the lookup error passes
/// through unchanged (Spark 4.2.0 reports `UNRESOLVED_COLUMN`, proposing
/// the aggregate's outputs).
#[test]
fn order_by_unknown_column_over_aggregate_is_unresolved() {
    let ctx = ctx();
    // `SELECT a FROM t GROUP BY a ORDER BY missing`: `missing` is neither
    // an output nor a grouping attribute.
    let order = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![c("a")])
        .sort(vec![crate::SortOrder::asc(c("missing"))]);
    let e = run(order, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(
        params.get("objectName").map(String::as_str),
        Some("`missing`")
    );
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`"));
}

/// `ORDER BY` over duplicate select outputs looks below them (existing
/// leniency): with a single input column of that name it resolves there,
/// otherwise it stays ambiguous rather than unresolved.
#[test]
fn order_by_duplicates_fall_back_then_stay_unresolved() {
    let ctx = ctx();
    // `SELECT 1 AS a, 2 AS a FROM t ORDER BY a`: the input `a` wins over
    // the duplicate outputs.
    let order = LogicalPlan::table(["t"])
        .project(vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    run(order, &ctx).unwrap();
    // No input column of that name: unresolved (Spark 4.2.0:
    // `SELECT 1 AS a, 2 AS a ORDER BY a` is UNRESOLVED_COLUMN, proposing the
    // duplicate outputs).
    let ambiguous = LogicalPlan::new(PlanNode::OneRowRelation)
        .project(vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let e = run(ambiguous, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`a`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`, `a`"));
}

/// `LIMIT` (and `OFFSET`) over a column is `IS_UNFOLDABLE`, not
/// `UNRESOLVED_COLUMN` (Spark golden `limit` case 9): columns resolve
/// against the input and then fail folding, exactly as nondeterministic
/// calls do.
#[test]
fn limit_over_a_column_is_unfoldable() {
    let ctx = ctx();
    for (plan, name) in [
        (LogicalPlan::table(["t"]).limit(gt(c("a"), int(3))), "limit"),
        (LogicalPlan::table(["t"]).limit(c("a")), "limit"),
        (LogicalPlan::table(["t"]).offset(c("b")), "offset"),
    ] {
        let e = run(plan, &ctx).unwrap_err();
        assert_eq!(
            e.error.error_class().name(),
            "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE",
            "{name}"
        );
        let params: std::collections::BTreeMap<_, _> = e
            .error
            .message_parameters()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        assert_eq!(params.get("name").map(String::as_str), Some(name));
    }
    // Foldable non-integers are a type error, foldable negatives and nulls
    // are value errors, in Spark's order (foldable, then type, then value).
    let e = run(LogicalPlan::table(["t"]).limit(s("a")), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.DATA_TYPE"
    );
    let e = run(LogicalPlan::table(["t"]).limit(int(-1)), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    );
    let e = run(
        LogicalPlan::table(["t"]).limit(Expr::new(ExprNode::Literal(Literal::Null(Some(
            krishiv_types::DataType::Integer,
        ))))),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL"
    );
    // A foldable integer limit resolves to a literal.
    let a = run(LogicalPlan::table(["t"]).limit(add(int(1), int(2))), &ctx).unwrap();
    assert!(
        a.plan.root().to_string().contains("Limit"),
        "{}",
        a.plan.root()
    );
}

/// `LIMIT` resolves against the select list: an alias the projection
/// defines is visible (then fails folding when it names a column), and
/// duplicates in the projection stay ambiguous rather than resolving
/// below it.
#[test]
fn limit_resolves_against_the_select_list() {
    let ctx = ctx();
    // `SELECT a AS k FROM t LIMIT k`: `k` resolves to the projection
    // (not `UNRESOLVED_COLUMN`), then fails as unfoldable since it names
    // a column.
    let aliased = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("k")])
        .limit(c("k"));
    assert_eq!(
        class(&run(aliased, &ctx)),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    // Duplicates in the projection are ambiguous, even though the input
    // has a single column of that name.
    let dup = LogicalPlan::table(["t"])
        .project(vec![int(1).alias("a"), int(2).alias("a")])
        .limit(c("a"));
    assert_eq!(class(&run(dup, &ctx)), "AMBIGUOUS_REFERENCE");
    // Checked on a Spark 4.2.0 server: `SELECT a AS x, b AS x FROM t LIMIT x`
    // is `AMBIGUOUS_REFERENCE`; the same attribute selected twice
    // (`SELECT a, a FROM t LIMIT a`) is one attribute, so `LIMIT a` is
    // unfoldable; and a name the select list lacks (`LIMIT b`) is
    // unresolved: the limit reads the select list, not the table below it.
    let twice = LogicalPlan::table(["t"])
        .project(vec![c("a"), c("a")])
        .limit(c("a"));
    assert_eq!(
        class(&run(twice, &ctx)),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    let below = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .limit(c("b"));
    let e = run(below, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`b`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`"));
}

/// `GROUP BY` ordinals follow `spark.sql.groupByOrdinal`: with it on, an
/// integer literal names the n-th select item; with it off, the same
/// literal is a constant grouping key.
#[test]
fn group_by_ordinals_follow_their_config() {
    let ctx = ctx();
    // `SELECT a, COUNT(b) FROM t GROUP BY 1`: groups by the first item.
    let ordinal =
        LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![c("a"), f("count", vec![c("b")])]);
    let a = run(ordinal, &ctx).unwrap();
    assert_eq!(a.output.len(), 2);
    // Out of range is `GROUP_BY_POS_OUT_OF_RANGE`.
    let out_of_range =
        LogicalPlan::table(["t"]).aggregate(vec![int(5)], vec![c("a"), f("count", vec![c("b")])]);
    assert_eq!(class(&run(out_of_range, &ctx)), "GROUP_BY_POS_OUT_OF_RANGE");
    // With ordinals off, `GROUP BY 1` groups by the constant 1 (one group).
    let off = context(
        catalog(),
        ResolutionConf {
            group_by_ordinal: false,
            ..ResolutionConf::default()
        },
    );
    let constant =
        LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![f("count", vec![star()])]);
    let a = run(constant, &off).unwrap();
    assert_eq!(a.output.len(), 1);
    // Star with an ordinal key is `STAR_GROUP_BY_POS`.
    let star = LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![star()]);
    assert_eq!(class(&run(star, &ctx)), "STAR_GROUP_BY_POS");
}

/// `ORDER BY` ordinals follow `spark.sql.orderByOrdinal`: with it on, an
/// integer literal names the n-th output column; with it off, the same
/// literal sorts by its constant value.
#[test]
fn order_by_ordinals_follow_their_config() {
    let ctx = ctx();
    // `SELECT a, b FROM t ORDER BY 2`: sorts by the second output (`b`).
    let ordinal = LogicalPlan::table(["t"])
        .project(vec![c("a"), c("b")])
        .sort(vec![crate::SortOrder::asc(int(2))]);
    let a = run(ordinal, &ctx).unwrap();
    assert!(
        a.plan.root().to_string().contains("b#"),
        "{}",
        a.plan.root()
    );
    // Out of range is `ORDER_BY_POS_OUT_OF_RANGE`.
    let out_of_range = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .sort(vec![crate::SortOrder::asc(int(5))]);
    assert_eq!(class(&run(out_of_range, &ctx)), "ORDER_BY_POS_OUT_OF_RANGE");
    // With ordinals off, `ORDER BY 2` sorts by the constant 2.
    let off = context(
        catalog(),
        ResolutionConf {
            order_by_ordinal: false,
            ..ResolutionConf::default()
        },
    );
    run(
        LogicalPlan::table(["t"])
            .project(vec![c("a")])
            .sort(vec![crate::SortOrder::asc(int(2))]),
        &off,
    )
    .unwrap();
}

/// `GROUP BY` attribute precedence and `HAVING`/`ORDER BY` over the
/// aggregate: an input column wins over select-list aliases of that name,
/// and later clauses resolve to the grouping column rather than either
/// duplicate output.
#[test]
fn group_by_attributes_take_precedence_over_aliases() {
    let ctx = ctx();
    // `SELECT 1 AS a, 2 AS a FROM t(a) GROUP BY a`: the key is the input
    // column, not either alias.
    let key = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")]);
    // The key is the input column: the grouping expression references an id
    // neither duplicate output carries.
    let a = run(key, &ctx).unwrap();
    let mut groupings = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Aggregate { grouping, .. } = &p.node
        {
            groupings.extend(grouping_keys(grouping).iter().flat_map(resolved_ids));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(groupings.len(), 1);
    let outputs: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert!(!outputs.contains(&groupings[0]));
    // Without an input column of that name, the first alias wins for the
    // grouping itself (Spark golden `group-by-alias` case 27 succeeds).
    let no_input = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")]);
    run(no_input, &ctx).unwrap();
}

/// CTE shadowing across nesting levels: an inner `WITH` name hides the
/// outer one inside its scope, and reads outside still need the outer
/// definition (which is checked for missing tables even when unused
/// elsewhere).
#[test]
fn cte_shadowing_applies_at_every_level() {
    let ctx = ctx();
    let good = || LogicalPlan::table(["t"]).project(vec![c("a")]);
    let bad = || {
        LogicalPlan::table(["t"])
            .join(
                LogicalPlan::table(["u"]),
                JoinKind::Inner,
                JoinCondition::None,
            )
            .project(vec![c("a")])
    };
    // Inner `x` shadows outer `x`: only the inner (valid) definition is
    // read, so the outer ambiguity never fails the query.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "x".to_owned(),
                plan: Arc::new(good()),
            }],
        })),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    run(shadowed, &ctx).unwrap();
    // A read outside the inner scope still needs the outer definition: its
    // ambiguity fails the query.
    let outer_read = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(
            LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(good()),
                }],
            })
            .join(
                LogicalPlan::table(["x"]).project(vec![c("a")]),
                JoinKind::Inner,
                JoinCondition::None,
            ),
        ),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    assert_eq!(class(&run(outer_read, &ctx)), "AMBIGUOUS_REFERENCE");
}

/// `LIMIT` and `OFFSET` share Spark's checks: foldable, then type, then
/// value, with the same error classes and parameters as the golden
/// `limit.sql.out` shows.
#[test]
fn limit_and_offset_share_sparks_checks() {
    let ctx = ctx();
    // Foldable non-literals pass through as literals for the planner.
    let folded = LogicalPlan::table(["t"]).limit(add(int(1), int(2)));
    let a = run(folded, &ctx).unwrap();
    assert!(
        a.plan.root().to_string().contains("Limit"),
        "{}",
        a.plan.root()
    );
    // Offsets behave like limits (name `offset` in the parameters).
    let e = run(LogicalPlan::table(["t"]).offset(c("a")), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    assert_eq!(
        e.error
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "name")
            .map(|(_, v)| v.as_str()),
        Some("offset")
    );
    // A subquery limit is unfoldable (cited at the expression).
    let sub = LogicalPlan::table(["t"]).limit(crate::tests::dsl::scalar(
        LogicalPlan::table(["u"]).project(vec![c("a")]),
    ));
    assert_eq!(
        class(&run(sub, &ctx)),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
}

/// `ORDER BY` over a non-aggregate with duplicate outputs resolves below
/// them (existing leniency), and `GROUP BY` ordinal out-of-range and
/// star-group-by-pos keep their own classes.
#[test]
fn order_by_leniency_and_group_by_star_rules() {
    let ctx = ctx();
    // Duplicate outputs with no input column of that name are unresolved
    // (Spark 4.2.0: `SELECT a AS k, b AS k FROM t ORDER BY k`), proposing
    // the duplicate outputs.
    let order = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("x"), c("b").alias("x")])
        .sort(vec![crate::SortOrder::asc(c("x"))]);
    let e = run(order, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`x`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`x`, `x`"));
    // Star grouped by all of its columns resolves.
    let star_ok = LogicalPlan::table(["t"]).aggregate(vec![c("a"), c("b"), c("c")], vec![star()]);
    run(star_ok, &ctx).unwrap();
}

/// A CTE name may contain dots when backquoted (Spark golden `cte` case
/// 44, `` WITH `a.b.c` AS ... ``): it is a single identifier, resolved as
/// one CTE rather than a qualified table.
#[test]
fn cte_names_may_contain_dots() {
    let ctx = ctx();
    let dotted = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["a.b.c"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "a.b.c".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    let a = run(dotted, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    // Without the definition, the same dotted name is a missing table.
    assert_eq!(
        class(&run(
            LogicalPlan::table(["a.b.c"]).project(vec![c("a")]),
            &ctx
        )),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

/// An inner `WITH` inside an unused CTE is still checked for missing
/// tables, and shadowing inside it hides what the outer scope would
/// otherwise report.
#[test]
fn unused_cte_inner_with_scopes_shadow() {
    let ctx = ctx();
    // Unused outer over an inner `WITH x ...` that reads `x`: the inner
    // read is the inner CTE, not a table, so no error.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
                }],
            })),
        }],
    });
    run(shadowed, &ctx).unwrap();
    // The same shape over a missing inner table fails.
    let missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["missing_inner"]).project(vec![c("a")])),
                }],
            })),
        }],
    });
    assert_eq!(class(&run(missing, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
}

/// `GROUP BY` with a subquery key and `HAVING`/`ORDER BY` over it resolve
/// without special-casing: scalar subqueries are foldable inputs, not
/// columns, so they never trigger alias or leniency paths.
#[test]
fn group_by_with_subquery_keys_resolves() {
    let ctx = ctx();
    // `SELECT a, COUNT(*) FROM t GROUP BY a, (SELECT ... LIMIT 1)` is
    // covered by golden `group-by-alias` case 15 (passes): the subquery
    // key is not a column, alias, or duplicate output.
    let sub = crate::tests::dsl::scalar(
        LogicalPlan::table(["u"])
            .project(vec![c("a")])
            .limit(int(1)),
    );
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("a"), sub], vec![c("a"), f("count", vec![star()])]);
    run(plan, &ctx).unwrap();
}

/// `ORDER BY` over a `HAVING` filter resolves through the filter to the
/// aggregate below it: duplicate select-list names are looked up below
/// with leniency first, and when that is ambiguous too the aggregate's
/// grouping attributes still get their try. A `HAVING` that grew the
/// aggregate leaves a projection the sort never resolves through, and
/// without a grouping column of the name the duplicates stay
/// `UNRESOLVED_COLUMN` (as over the aggregate directly).
#[test]
fn order_by_through_having_reports_duplicates_as_unresolved() {
    let ctx = ctx();
    // `SELECT 1 AS z, 2 AS z, a FROM t GROUP BY a HAVING a > 0 ORDER BY z`:
    // the `HAVING` needs nothing new (no growth), so the sort retries
    // through the filter against the aggregate, whose grouping has no `z`:
    // unresolved, proposing the duplicate outputs.
    let through = LogicalPlan::table(["t"])
        .aggregate(
            vec![c("a")],
            vec![int(1).alias("z"), int(2).alias("z"), c("a")],
        )
        .filter(gt(c("a"), int(0)))
        .sort(vec![crate::SortOrder::asc(c("z"))]);
    let e = run(through, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`z`"));
    assert_eq!(
        params.get("proposal").map(String::as_str),
        Some("`z`, `z`, `a`")
    );
    // `SELECT 1 AS a, 2 AS a FROM t GROUP BY b HAVING b = 'x' ORDER BY a`:
    // the `HAVING` grows the aggregate (a projection the sort never
    // resolves through), and no grouping column is named `a`: unresolved.
    let grown = LogicalPlan::table(["t"])
        .aggregate(vec![c("b")], vec![int(1).alias("a"), int(2).alias("a")])
        .filter(eq(c("b"), s("x")))
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let e = run(grown, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`a`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`, `a`"));
}

/// A `Sql` node inside an unused CTE is still checked for missing tables:
/// the node's reads live in the lowered plan, which the
/// unused-definition walk follows. An unused `Sql` over existing tables
/// passes.
#[test]
fn unused_cte_sql_tables_are_checked() {
    // An unused definition holding SQL text over existing tables passes:
    // the walk follows the lowered plan Bind stored for the `Sql` node.
    let mut good_ctx = ctx();
    good_ctx.sql = Some(Arc::new(Frontend));
    let unused_sql = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::Sql {
                query: "q6".to_owned(),
                args: SqlArgs::None,
            })),
        }],
    });
    run(unused_sql, &good_ctx).unwrap();
    // The same shape over a missing table fails with
    // `TABLE_OR_VIEW_NOT_FOUND` at the lowered read.
    let mut bad_ctx = ctx();
    bad_ctx.sql = Some(Arc::new(Frontend));
    let unused_missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::Sql {
                query: "gone".to_owned(),
                args: SqlArgs::None,
            })),
        }],
    });
    let e = run(unused_missing, &bad_ctx).unwrap_err();
    assert_eq!(e.error.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(
        params.get("relationName").map(String::as_str),
        Some("`gone`")
    );
}

/// `ORDER BY` with several keys converts only the ambiguous one: a
/// resolvable key before it still binds, and the duplicate stays
/// `UNRESOLVED_COLUMN` (Spark 4.2.0 proposes the duplicate outputs, as
/// for a single key). (multi-key case)
#[test]
fn order_by_second_duplicate_key_is_unresolved() {
    let ctx = ctx();
    // `SELECT 1 AS x, 2 AS x, b FROM t ORDER BY b, x`: `b` resolves to
    // the input column while `x` stays unresolved. (multi-key sort)
    let order = LogicalPlan::table(["t"])
        .project(vec![int(1).alias("x"), int(2).alias("x"), c("b")])
        .sort(vec![
            crate::SortOrder::asc(c("b")),
            crate::SortOrder::asc(c("x")),
        ]);
    let e = run(order, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`x`"));
}

/// `ORDER BY` through a `HAVING` filter also resolves a grouping
/// expression written out in full: with no growth the sort retries
/// against the aggregate below the filter, where the expression is the
/// aggregate's output (as over the aggregate directly) or is added to
/// it (as over the aggregate directly).
#[test]
fn order_by_through_having_resolves_full_grouping_expression() {
    let ctx = ctx();
    // `SELECT a + 1 AS k FROM t GROUP BY a + 1 HAVING k > 0 ORDER BY a + 1`:
    // the `HAVING` needs nothing new, so the sort retries through the
    // filter against the aggregate, resolving to the `k` output.
    let plan = LogicalPlan::table(["t"])
        .aggregate(
            vec![add(c("a"), int(1))],
            vec![add(c("a"), int(1)).alias("k")],
        )
        .filter(gt(c("k"), int(0)))
        .sort(vec![crate::SortOrder::asc(add(c("a"), int(1)))]);
    let a = run(plan, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["k"]
    );
    // The sort key is the `k` output.
    let mut keys = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            keys.extend(order.iter().flat_map(|o| resolved_ids(&o.expr)));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(keys, vec![a.output[0].id]);
    // `SELECT count(*) AS c FROM t GROUP BY a + 1 HAVING count(*) > 0
    // ORDER BY a + 1`: the grouping expression is not an output, so the
    // retry adds it to the aggregate below the filter and sorts by it. The
    // added column stays internal: the sort restores the select list above
    // what it added, so the query still outputs only `c` (as Spark does).
    let grown = LogicalPlan::table(["t"])
        .aggregate(
            vec![add(c("a"), int(1))],
            vec![f("count", vec![star()]).alias("c")],
        )
        .filter(gt(f("count", vec![star()]), int(0)))
        .sort(vec![crate::SortOrder::asc(add(c("a"), int(1)))]);
    let b = run(grown, &ctx).unwrap();
    assert_eq!(
        b.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["c"]
    );
    let mut grown_keys = Vec::new();
    let mut grown_names = Vec::new();
    let _ = visit_distinct(Node::Plan(b.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            for o in order {
                let _ = visit_distinct(Node::Expr(&o.expr), |n| {
                    if let Node::Expr(x) = n
                        && let ExprNode::Resolved(ResolvedExpr::Column(col)) = &x.node
                    {
                        grown_keys.push(col.id);
                        grown_names.push(col.name.clone());
                    }
                    ControlFlow::Continue(())
                });
            }
        }
        ControlFlow::Continue(())
    });
    // The sort key is the grouping expression the retry added below the
    // filter: one column, named for `a + 1`, and not one of the outputs.
    assert_eq!(grown_keys.len(), 1);
    assert_eq!(grown_names, ["(a + 1)"]);
    assert!(!b.output.iter().any(|o| o.id == grown_keys[0]));
}

/// `ORDER BY` over duplicate outputs above windows still reports them
/// as unresolved: the sort reads the select list only, never resolving
/// through the windows to the aggregate below.
#[test]
fn order_by_over_windows_reports_duplicates_as_unresolved() {
    let ctx = ctx();
    let ranked = Expr::new(ExprNode::Window(crate::WindowExpr {
        function: Arc::new(f("row_number", Vec::new())),
        window: crate::WindowSpecRef::Spec(crate::WindowSpec {
            partition_by: Vec::new(),
            // `row_number` needs an ordered window (Spark rejects `OVER ()`
            // with `WINDOW_FUNCTION_FRAME_NOT_ORDERED`, as does the
            // analyzer); the ordering does not affect the sort below.
            order_by: vec![crate::SortOrder::asc(c("a"))],
            frame: None,
        }),
    }));
    // `SELECT 1 AS x, 2 AS x, a, row_number() OVER (ORDER BY a) FROM t
    // GROUP BY a HAVING a > 0 ORDER BY x`: the `HAVING` needs nothing new, so the
    // sort retries through the filter, but the aggregate holds windows,
    // so the duplicates stay `UNRESOLVED_COLUMN`.
    let plan = LogicalPlan::table(["t"])
        .aggregate(
            vec![c("a")],
            vec![
                int(1).alias("x"),
                int(2).alias("x"),
                c("a"),
                ranked.alias("r"),
            ],
        )
        .filter(gt(c("a"), int(0)))
        .sort(vec![crate::SortOrder::asc(c("x"))]);
    let e = run(plan, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`x`"));
}

/// `ORDER BY` passes a sort key's own errors through unchanged: only an
/// `AMBIGUOUS_REFERENCE` over duplicate outputs becomes
/// `UNRESOLVED_COLUMN`; any other failure (here a function's arity) is
/// reported as raised, as Spark 4.2 reports it.
#[test]
fn order_by_passes_sort_key_errors_through() {
    let ctx = ctx();
    // `SELECT a FROM t ORDER BY upper(1, 2)`: the key is not a name error,
    // so the sort reports the function's arity failure.
    let order = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .sort(vec![crate::SortOrder::asc(Expr::function(
            "upper",
            vec![int(1), int(2)],
        ))]);
    let err = run(order, &ctx).unwrap_err();
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

/// Spark's strength reduction for an integral expression compared with a
/// decimal literal (`DecimalPrecisionTypeCoercion.integralAndDecimalLiteral`
/// in Spark 4.2): the comparison narrows the literal to the `BIGINT` bound
/// (`floor`/`ceil`), and the output name shows the rewritten comparison.
/// `select 1 > 0.00001` names its column `(1 > 0)`, as the golden
/// `operators` statement 55 records.
#[test]
fn integral_compared_with_decimal_literal_narrows_to_bigint() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    // (expression, Spark 4.2 output name): every comparison direction, the
    // literal on either side, and negative literals.
    let cases = [
        (gt(int(1), dec("0.00001")), "(1 > 0)"),
        (ge(int(1), dec("0.00001")), "(1 >= 1)"),
        (lt(int(1), dec("0.00001")), "(1 < 1)"),
        (le(int(1), dec("0.00001")), "(1 <= 0)"),
        (gt(dec("0.00001"), int(1)), "(1 > 1)"),
        (ge(dec("0.00001"), int(1)), "(0 >= 1)"),
        (lt(dec("0.00001"), int(1)), "(0 < 1)"),
        (le(dec("0.00001"), int(1)), "(1 <= 1)"),
        (gt(int(1), dec("-0.00001")), "(1 > -1)"),
        (ge(int(1), dec("-0.00001")), "(1 >= 0)"),
        (lt(int(1), dec("-0.00001")), "(1 < 0)"),
        (le(int(1), dec("-0.00001")), "(1 <= -1)"),
        (gt(dec("-0.00001"), int(1)), "(0 > 1)"),
        (ge(dec("-0.00001"), int(1)), "(-1 >= 1)"),
        (lt(dec("-0.00001"), int(1)), "(-1 < 1)"),
        (le(dec("-0.00001"), int(1)), "(0 <= 1)"),
        (gt(long(5), dec("0.5")), "(5 > 0)"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
}

/// A decimal literal outside `BIGINT` range decides an integral comparison
/// outright (`true`/`false`), as Spark's rule does.
#[test]
fn integral_compared_with_huge_decimal_literal_is_decided() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    let cases = [
        (gt(int(1), dec("99999999999999999999.9")), "false"),
        (ge(int(1), dec("99999999999999999999.9")), "false"),
        (lt(int(1), dec("99999999999999999999.9")), "true"),
        (le(int(1), dec("99999999999999999999.9")), "true"),
        (gt(dec("99999999999999999999.9"), int(1)), "true"),
        (ge(dec("99999999999999999999.9"), int(1)), "true"),
        (lt(dec("99999999999999999999.9"), int(1)), "false"),
        (le(dec("99999999999999999999.9"), int(1)), "false"),
        (gt(int(1), dec("-99999999999999999999.9")), "true"),
        (lt(int(1), dec("-99999999999999999999.9")), "false"),
        (gt(dec("-99999999999999999999.9"), int(1)), "false"),
        (lt(dec("-99999999999999999999.9"), int(1)), "true"),
        // The `BIGINT` bounds themselves still narrow instead of deciding.
        (
            gt(int(1), dec("9223372036854775807")),
            "(1 > 9223372036854775807)",
        ),
        (
            lt(int(1), dec("-9223372036854775808")),
            "(1 < -9223372036854775808)",
        ),
        (gt(int(1), dec("9223372036854775807.5")), "false"),
        (lt(int(1), dec("9223372036854775807.5")), "true"),
        (gt(int(1), dec("-9223372036854775808.5")), "true"),
        (lt(int(1), dec("-9223372036854775808.5")), "false"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
}

/// The narrowing fires only for an integral expression against a decimal
/// literal: `=`, two decimals, floating-point and untyped `NULL` operands
/// keep their written form (up to the usual implicit casts, which the name
/// does not show).
#[test]
fn non_integral_or_non_literal_comparisons_keep_their_form() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    let float = Expr::new(ExprNode::Literal(Literal::Float(0.5)));
    let cases = [
        (eq(int(1), dec("0.00001")), "(1 = 0.00001)"),
        (gt(dec("1.5"), dec("0.00001")), "(1.5 > 0.00001)"),
        (gt(int(1), float), "(1 > 0.5)"),
        (gt(int(1), int(2)), "(1 > 2)"),
        (gt(null(), dec("0.00001")), "(NULL > 0.00001)"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
}

/// Columns narrow like literals do: the rewritten comparison, not the
/// written one, names the output.
#[test]
fn integral_columns_compared_with_decimal_literals_narrow() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::table(["t"]).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    let cases = [
        (gt(c("a"), dec("0.5")), "(a > 0)"),
        (lt(c("c"), dec("2.5")), "(c < 3)"),
        (ge(c("a"), dec("-0.5")), "(a >= 0)"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
    // Narrower integral types narrow the same way.
    let plan = LogicalPlan::table(["n"]).project(vec![gt(c("ti"), dec("0.5"))]);
    assert_eq!(run(plan, &ctx).unwrap().output[0].name, "(ti > 0)");
    // The narrowed comparison stays boolean.
    let plan = LogicalPlan::table(["t"]).project(vec![gt(c("a"), dec("0.5"))]);
    assert_eq!(datatypes(&run(plan, &ctx).unwrap()), [DataType::Boolean]);
}
