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
    ResolutionConf, RowColumnPolicy, Securable, SessionZone, SparkProfile, SqlFrontend,
    TableColumn,
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
            .map(|f| {
                TableColumn::new(
                    f["name"].as_str().unwrap().to_owned(),
                    krishiv_types::parse_data_type(f["type"].as_str().unwrap()).unwrap(),
                    f["nullable"].as_bool().unwrap(),
                )
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

fn df(table: &str, id: i64) -> LogicalPlan {
    LogicalPlan::table([table]).with_origin(Origin::plan(PlanId(id)))
}

fn col(name: &str, id: i64) -> Expr {
    Expr::new(ExprNode::Column(ColumnRef {
        plan_id: Some(PlanId(id)),
        ..ColumnRef::new(vec![name.to_owned()])
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
                regex: None,
                replace: Vec::new(),
                position: None,
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
                regex: None,
                replace: Vec::new(),
                position: None,
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
                within_group: Vec::new(),
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

/// Every single-part table read of an unresolved plan, in order, and whether
/// any `WITH` survived elimination.
fn single_reads(plan: &Arc<LogicalPlan>) -> Vec<String> {
    use crate::walk::{Node, visit_distinct};
    let mut reads = Vec::new();
    let mut with = false;
    let _ = visit_distinct(Node::Plan(plan), &mut |n| {
        let p = match n {
            Node::Plan(p) => p,
            Node::Expr(_) => return ControlFlow::Continue(()),
        };
        match &p.node {
            PlanNode::Read(ReadSource::Table { name, .. }) if name.len() == 1 => {
                reads.push(name[0].clone());
            }
            PlanNode::WithCte { .. } => with = true,
            _ => {}
        }
        ControlFlow::Continue(())
    });
    assert!(!with, "a WITH survived elimination");
    reads
}

mod part1;
mod part2;
mod part3;
mod part4;
mod part5;
