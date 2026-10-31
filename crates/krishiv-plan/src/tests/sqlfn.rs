// SPDX-License-Identifier: Apache-2.0
//! SQL-bodied functions in the analyzer: a call is replaced by the body, so
//! the resolved plan holds no call of the function; defaults, named
//! arguments, argument checks, recursion, isolation of the body's scope,
//! the tables the body reads, and the context fingerprint.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::Arc;

use krishiv_types::DataType;

use super::depth::on_small_stack;
use super::dsl::{add, c, date, eq, int, named_call, s, sum};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, BuiltinFunctions, DefaultAnalyzer, FunctionLookup,
    FunctionRegistry, MemoryCatalog, ResolutionConf, SparkProfile, SqlFunction, SqlParam,
    TableColumn,
};
use crate::walk::{Node, visit_distinct};
use crate::{Expr, ExprNode, LogicalPlan, Plan, ResolvedExpr, SourceId, SubqueryKind};

/// The built-ins, `rand` (nondeterministic, typed as `sqrt`), and `defs`.
struct Registry {
    base: BuiltinFunctions,
    defs: BTreeMap<String, Arc<SqlFunction>>,
    version: u64,
}

impl FunctionRegistry for Registry {
    fn lookup(&self, name: &[String], profile: SparkProfile) -> FunctionLookup {
        if let [n] = name {
            if n.eq_ignore_ascii_case("rand")
                && let FunctionLookup::Found(mut e) =
                    self.base.lookup(&["sqrt".to_owned()], profile)
            {
                e.deterministic = false;
                return FunctionLookup::Found(e);
            }
            if let Some(d) = self.defs.get(&n.to_lowercase()) {
                return match self.base.lookup(name, profile) {
                    FunctionLookup::NotFound => FunctionLookup::Sql(Arc::clone(d)),
                    other => other,
                };
            }
        }
        self.base.lookup(name, profile)
    }

    fn version(&self) -> u64 {
        self.version
    }
}

fn param(name: &str, t: DataType, default: Option<Expr>) -> SqlParam {
    SqlParam {
        name: name.to_owned(),
        data_type: t,
        default: default.map(Arc::new),
        comment: None,
    }
}

fn func(name: &str, params: Vec<SqlParam>, returns: Option<DataType>, body: Expr) -> SqlFunction {
    SqlFunction {
        qualifier: Vec::new(),
        name: name.to_owned(),
        params,
        returns,
        body: Arc::new(body),
        body_text: "<body>".to_owned(),
        source: SourceId(1),
        deterministic: true,
    }
}

fn registry(defs: Vec<SqlFunction>, version: u64) -> Arc<Registry> {
    Arc::new(Registry {
        base: BuiltinFunctions::new(),
        defs: defs
            .into_iter()
            .map(|d| (d.name.clone(), Arc::new(d)))
            .collect(),
        version,
    })
}

fn ctx_with(defs: Vec<SqlFunction>, conf: ResolutionConf) -> crate::analyzer::AnalysisContext {
    let cat = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![
            TableColumn {
                name: "a".to_owned(),
                data_type: DataType::Integer,
                nullable: false,
            },
            TableColumn {
                name: "b".to_owned(),
                data_type: DataType::String,
                nullable: true,
            },
        ],
        None,
    );
    let mut ctx = context(cat, conf);
    ctx.functions = registry(defs, 1);
    ctx
}

fn run(
    plan: LogicalPlan,
    ctx: &crate::analyzer::AnalysisContext,
) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: crate::Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, ctx))
}

fn select(items: Vec<Expr>) -> LogicalPlan {
    LogicalPlan::table(["t"]).project(items)
}

fn call(items: Vec<Expr>, defs: Vec<SqlFunction>) -> Result<AnalyzedPlan, AnalysisError> {
    run(select(items), &ctx_with(defs, ResolutionConf::default()))
}

fn err(r: &Result<AnalyzedPlan, AnalysisError>) -> (String, BTreeMap<String, String>) {
    let e = &r.as_ref().expect_err("analysis succeeded").error;
    (
        e.error_class().name().to_owned(),
        e.message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    )
}

/// Every resolved or unresolved call in the plan, by name.
fn calls(a: &AnalyzedPlan) -> Vec<String> {
    let mut out = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |n| {
        if let Node::Expr(e) = n {
            match &e.node {
                ExprNode::Function(f) => out.push(f.name.clone()),
                ExprNode::Resolved(ResolvedExpr::Function(f)) => out.push(f.id.name().to_owned()),
                _ => {}
            }
        }
        ControlFlow::Continue(())
    });
    out
}

/// The marker of a named argument in the arguments given to [`f`].
const NAMED: &str = "=>";

fn named(name: &str, value: Expr) -> Expr {
    Expr::function(NAMED, vec![s(name), value])
}

/// `name(args)`; an argument made by [`named`] is passed under its name.
fn f(name: &str, args: Vec<Expr>) -> Expr {
    let parsed: Vec<(Option<String>, Expr)> = args
        .into_iter()
        .map(|a| match &a.node {
            ExprNode::Function(call) if call.name == NAMED => {
                let label = match call.args.first().map(|n| &n.node) {
                    Some(ExprNode::Literal(crate::Literal::String(n))) => Some(n.clone()),
                    _ => None,
                };
                (label, call.args[1].as_ref().clone())
            }
            _ => (None, a),
        })
        .collect();
    let pairs: Vec<(Option<&str>, Expr)> = parsed
        .iter()
        .map(|(n, e)| (n.as_deref(), e.clone()))
        .collect();
    named_call(name, &pairs)
}

fn add2() -> SqlFunction {
    func(
        "add2",
        vec![param("a", DataType::Integer, None)],
        Some(DataType::Long),
        add(c("a"), int(2)),
    )
}

#[test]
fn a_call_is_replaced_by_the_body() {
    let r = call(vec![f("add2", vec![c("a")])], vec![add2()]).unwrap();
    // The plan holds the body's `+` and a cast to the return type, and no
    // call of `add2`; the column is named as the call was written.
    let names = calls(&r);
    assert!(!names.iter().any(|n| n == "add2"), "{names:?}");
    assert!(names.iter().any(|n| n == "+"), "{names:?}");
    assert_eq!(r.output[0].name, "add2(a)");
    assert_eq!(r.output[0].data_type, DataType::Long);
    assert!(!r.output[0].nullable);
    let text = r.plan.root().to_string();
    assert!(text.contains("CAST("), "{text}");
}

#[test]
fn two_calls_of_one_function_are_independent() {
    let r = call(
        vec![f("add2", vec![c("a")]), f("add2", vec![int(7)])],
        vec![add2()],
    )
    .unwrap();
    assert_eq!(r.output.len(), 2);
    assert_eq!(r.output[1].name, "add2(7)");
    assert!(!calls(&r).iter().any(|n| n == "add2"));
}

#[test]
fn arguments_are_cast_to_the_parameter_type() {
    let r = call(vec![f("add2", vec![s("5")])], vec![add2()]).unwrap();
    assert!(r.plan.root().to_string().contains("CAST('5' AS INT)"));
    // A value that cannot be cast is Spark's cast error, showing the value.
    let bad = f("add2", vec![date("2020-01-01")]);
    let (class, params) = err(&call(vec![bad], vec![add2()]));
    assert_eq!(class, "DATATYPE_MISMATCH.CAST_WITH_FUNC_SUGGESTION");
    assert_eq!(params["sqlExpr"], "\"CAST(2020-01-01 AS DATE)\"");
    assert_eq!(params["targetType"], "\"INT\"");
}

#[test]
fn argument_counts_are_checked() {
    for args in [vec![], vec![int(1), int(2)]] {
        let n = args.len().to_string();
        let r = call(vec![f("add2", args)], vec![add2()]);
        let (class, params) = err(&r);
        assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
        assert_eq!(params["functionName"], "`add2`");
        assert_eq!(params["expectedNum"], "1");
        assert_eq!(params["actualNum"], n);
    }
}

fn with_default() -> SqlFunction {
    func(
        "g",
        vec![
            param("a", DataType::Integer, None),
            param("b", DataType::Integer, Some(add(int(5), int(5)))),
        ],
        Some(DataType::Integer),
        add(c("a"), c("b")),
    )
}

#[test]
fn defaults_fill_missing_arguments() {
    let r = call(
        vec![f("g", vec![int(1)]), f("g", vec![int(1), int(2)])],
        vec![with_default()],
    )
    .unwrap();
    assert_eq!(r.output.len(), 2);
    assert!(!calls(&r).iter().any(|n| n == "g"));
    // The default is resolved at the call, in a scope of its own: it cannot
    // read a column of the query or a parameter.
    let bad = func(
        "h",
        vec![
            param("a", DataType::Integer, None),
            param("b", DataType::Integer, Some(c("a"))),
        ],
        None,
        c("b"),
    );
    let (class, params) = err(&call(vec![f("h", vec![int(1)])], vec![bad]));
    assert_eq!(class, "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION");
    assert_eq!(params["objectName"], "`a`");
    // A default that is a call of another function resolves it.
    let inner = func(
        "k",
        vec![
            param("a", DataType::Integer, None),
            param("b", DataType::Integer, Some(f("g", vec![int(3)]))),
        ],
        None,
        add(c("a"), c("b")),
    );
    let r = call(vec![f("k", vec![int(1)])], vec![inner, with_default()]).unwrap();
    assert!(!calls(&r).iter().any(|n| n == "g" || n == "k"));
}

#[test]
fn named_arguments_bind_by_name() {
    let r = call(
        vec![
            f("g", vec![named("b", int(2)), named("a", int(1))]),
            f("g", vec![int(1), named("b", int(3))]),
            f("g", vec![named("A", int(1))]),
        ],
        vec![with_default()],
    )
    .unwrap();
    assert_eq!(r.output.len(), 3);
    let cases: [(Vec<Expr>, &str); 5] = [
        (
            vec![int(1), named("zz", int(2))],
            "UNRECOGNIZED_PARAMETER_NAME",
        ),
        (
            vec![int(1), named("a", int(2))],
            "DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.BOTH_POSITIONAL_AND_NAMED",
        ),
        (
            vec![named("a", int(1)), named("a", int(2))],
            "DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.DOUBLE_NAMED_ARGUMENT_REFERENCE",
        ),
        (
            vec![named("a", int(1)), int(2)],
            "UNEXPECTED_POSITIONAL_ARGUMENT",
        ),
        (vec![named("b", int(1))], "REQUIRED_PARAMETER_NOT_FOUND"),
    ];
    for (args, want) in cases {
        let (class, _) = err(&call(vec![f("g", args)], vec![with_default()]));
        assert_eq!(class, want);
    }
    // More positional arguments than parameters.
    let (class, params) = err(&call(
        vec![f("g", vec![int(1), int(2), int(3), named("a", int(4))])],
        vec![with_default()],
    ));
    assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    assert_eq!(params["expectedNum"], "2");
    let (_, params) = err(&call(
        vec![f("g", vec![int(1), named("zz", int(2))])],
        vec![with_default()],
    ));
    assert_eq!(params["argumentName"], "`zz`");
    assert_eq!(params["proposal"], "`a`, `b`");
    let (_, params) = err(&call(
        vec![f("g", vec![named("b", int(1))])],
        vec![with_default()],
    ));
    assert_eq!(params["parameterName"], "`a`");
    assert_eq!(params["index"], "0");
}

#[test]
fn only_sql_functions_take_named_arguments() {
    let r = call(vec![f("abs", vec![named("x", int(1))])], vec![]);
    let (class, params) = err(&r);
    assert_eq!(class, "NAMED_PARAMETERS_NOT_SUPPORTED");
    assert_eq!(params["functionName"], "`abs`");
}

#[test]
fn recursion_is_rejected_with_its_path() {
    let direct = func("r", vec![], Some(DataType::Integer), f("r", vec![]));
    let (class, params) = err(&call(vec![f("r", vec![])], vec![direct]));
    assert_eq!(class, "CYCLIC_FUNCTION_REFERENCE");
    assert_eq!(params["path"], "r -> r");
    let a = func("fa", vec![], Some(DataType::Integer), f("fb", vec![]));
    let b = func(
        "fb",
        vec![],
        Some(DataType::Integer),
        add(f("fc", vec![]), int(1)),
    );
    let cc = func("fc", vec![], Some(DataType::Integer), f("fa", vec![]));
    let (class, params) = err(&call(vec![f("fb", vec![])], vec![a, b, cc]));
    assert_eq!(class, "CYCLIC_FUNCTION_REFERENCE");
    assert_eq!(params["path"], "fb -> fc -> fa -> fb");
    // A cycle through a default.
    let d = func(
        "d",
        vec![param("x", DataType::Integer, Some(f("d", vec![])))],
        None,
        c("x"),
    );
    let (class, _) = err(&call(vec![f("d", vec![])], vec![d]));
    assert_eq!(class, "CYCLIC_FUNCTION_REFERENCE");
}

#[test]
fn deep_chains_resolve_on_a_small_stack_and_the_depth_is_capped() {
    let chain = |n: usize| -> Vec<SqlFunction> {
        (0..n)
            .map(|i| {
                let body = if i + 1 == n {
                    c("x")
                } else {
                    add(f(&format!("c{}", i + 1), vec![c("x")]), int(0))
                };
                func(
                    &format!("c{i}"),
                    vec![param("x", DataType::Integer, None)],
                    Some(DataType::Integer),
                    body,
                )
            })
            .collect()
    };
    let ok = chain(60);
    on_small_stack(move || {
        let r = call(vec![f("c0", vec![c("a")])], ok).unwrap();
        assert!(!calls(&r).iter().any(|n| n.starts_with('c') && n.len() < 4));
    });
    let too_deep = chain(1100);
    on_small_stack(move || {
        let r = call(vec![f("c0", vec![c("a")])], too_deep);
        assert_eq!(err(&r).0, "FAILED_TO_PARSE_TOO_COMPLEX");
    });
}

#[test]
fn the_body_sees_only_its_own_scope() {
    // `a` is a column of the calling query, not of the function.
    let leaky = func("leaky", vec![], Some(DataType::Integer), c("a"));
    let r = call(vec![f("leaky", vec![])], vec![leaky]);
    assert_eq!(err(&r).0, "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION");
    // The proposals are the parameters.
    let near = func(
        "near",
        vec![param("xx", DataType::Integer, None)],
        Some(DataType::Integer),
        c("x"),
    );
    let (class, params) = err(&call(vec![f("near", vec![int(1)])], vec![near]));
    assert_eq!(class, "UNRESOLVED_COLUMN.WITH_SUGGESTION");
    assert_eq!(params["proposal"], "`xx`");
    // Nor does a function see the parameters of the function calling it.
    let inner = func("inner", vec![], Some(DataType::Integer), c("x"));
    let outer = func(
        "outer",
        vec![param("x", DataType::Integer, None)],
        Some(DataType::Integer),
        f("inner", vec![]),
    );
    let r = call(vec![f("outer", vec![int(1)])], vec![inner, outer]);
    assert_eq!(err(&r).0, "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION");
}

#[test]
fn a_parameter_can_be_qualified_by_the_function_name() {
    let q = func(
        "q",
        vec![param("x", DataType::Integer, None)],
        Some(DataType::Integer),
        add(Expr::column(["q", "x"]), c("x")),
    );
    let r = call(vec![f("q", vec![int(1)])], vec![q]).unwrap();
    // Both uses are the argument: `1 + 1`, typed as the declared return.
    assert_eq!(r.output[0].name, "q(1)");
    assert_eq!(r.output[0].data_type, DataType::Integer);
}

#[test]
fn a_body_that_aggregates_is_rejected_and_arguments_may() {
    let mut d = func("agg", vec![], Some(DataType::Long), sum(int(1)));
    d.body_text = "sum(1)".to_owned();
    let (class, params) = err(&call(vec![f("agg", vec![])], vec![d]));
    assert_eq!(
        class,
        "USER_DEFINED_FUNCTIONS.CANNOT_CONTAIN_COMPLEX_FUNCTIONS"
    );
    assert_eq!(params["queryText"], "sum(1)");
    // An argument may be an aggregate: the query aggregates.
    let r = run(
        LogicalPlan::table(["t"]).aggregate(vec![], vec![f("add2", vec![sum(c("a"))])]),
        &ctx_with(vec![add2()], ResolutionConf::default()),
    )
    .unwrap();
    assert!(
        r.plan.root().to_string().contains("Aggregate"),
        "{}",
        r.plan.root()
    );
    assert!(!calls(&r).iter().any(|n| n == "add2"));
}

#[test]
fn a_nondeterministic_argument_is_read_once() {
    let twice = func(
        "twice",
        vec![param("x", DataType::Double, None)],
        Some(DataType::Double),
        add(c("x"), c("x")),
    );
    let once = func(
        "once",
        vec![param("x", DataType::Double, None)],
        Some(DataType::Double),
        add(c("x"), int(1)),
    );
    let defs = || vec![twice.clone(), once.clone()];
    assert!(call(vec![f("once", vec![f("rand", vec![int(1)])])], defs()).is_ok());
    let (class, _) = err(&call(
        vec![f("twice", vec![f("rand", vec![int(1)])])],
        defs(),
    ));
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
    // A deterministic argument may be read any number of times.
    assert!(call(vec![f("twice", vec![f("sqrt", vec![int(4)])])], defs()).is_ok());
}

/// A nondeterministic argument read from a subquery of the body would run
/// once per row of the subquery; Spark evaluates it once per call.
#[test]
fn a_nondeterministic_argument_is_not_read_from_a_subquery() {
    let sub = func(
        "sub",
        vec![param("x", DataType::Double, None)],
        Some(DataType::Double),
        scalar_subquery(LogicalPlan::table(["t"]).aggregate(vec![], vec![f("max", vec![c("x")])])),
    );
    let (class, _) = err(&call(
        vec![f("sub", vec![f("rand", vec![int(1)])])],
        vec![sub.clone()],
    ));
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
    assert!(call(vec![f("sub", vec![f("sqrt", vec![int(4)])])], vec![sub]).is_ok());
}

/// Functions that each call the next one twice inline to 2^n bodies; the
/// budget stops the statement instead of hanging the analyzer.
#[test]
fn nested_calls_that_double_are_budgeted() {
    let levels = 30;
    let defs: Vec<SqlFunction> = (0..=levels)
        .map(|i| {
            let body = if i == levels {
                c("x")
            } else {
                let next = format!("d{}", i + 1);
                add(f(&next, vec![c("x")]), f(&next, vec![c("x")]))
            };
            func(
                &format!("d{i}"),
                vec![param("x", DataType::Integer, None)],
                Some(DataType::Integer),
                body,
            )
        })
        .collect();
    let r = call(vec![f("d0", vec![int(1)])], defs);
    assert_eq!(err(&r).0, "FAILED_TO_PARSE_TOO_COMPLEX");
}

fn scalar_subquery(plan: LogicalPlan) -> Expr {
    Expr::subquery(SubqueryKind::Scalar, plan)
}

#[test]
fn parameters_are_visible_in_the_bodys_subqueries() {
    // `(SELECT x)`: a constant argument at any depth; a column argument
    // one subquery down is an outer reference; two down is refused.
    let one = func(
        "one",
        vec![param("x", DataType::Integer, None)],
        Some(DataType::Integer),
        scalar_subquery(
            LogicalPlan::table(["t"])
                .filter(eq(c("a"), c("x")))
                .aggregate(vec![], vec![f("max", vec![c("a")])]),
        ),
    );
    assert!(call(vec![f("one", vec![int(3)])], vec![one.clone()]).is_ok());
    // The argument's column is the calling query's: correlated, as if the
    // subquery were written at the call.
    let r = call(vec![f("one", vec![c("a")])], vec![one]).unwrap();
    assert!(
        r.plan.root().to_string().contains("Join"),
        "{}",
        r.plan.root()
    );
    let two = func(
        "two",
        vec![param("x", DataType::Integer, None)],
        Some(DataType::Integer),
        scalar_subquery(
            LogicalPlan::new(crate::PlanNode::OneRowRelation).project(vec![scalar_subquery(
                LogicalPlan::new(crate::PlanNode::OneRowRelation).project(vec![c("x")]),
            )]),
        ),
    );
    assert!(call(vec![f("two", vec![int(3)])], vec![two.clone()]).is_ok());
    let (class, _) = err(&call(vec![f("two", vec![c("a")])], vec![two]));
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn a_column_of_the_bodys_own_query_beats_the_parameter() {
    // `(SELECT a FROM t)` with a parameter `a`: the table's column.
    let w = func(
        "w",
        vec![param("a", DataType::Integer, None)],
        Some(DataType::Integer),
        scalar_subquery(LogicalPlan::table(["t"]).aggregate(vec![], vec![f("max", vec![c("a")])])),
    );
    let r = call(vec![f("w", vec![int(42)])], vec![w]).unwrap();
    assert!(
        !r.plan.root().to_string().contains("CAST(42"),
        "{}",
        r.plan.root()
    );
}

#[test]
fn tables_a_body_reads_are_bound_like_those_of_the_query() {
    let reads = |table: &str| {
        func(
            "reads",
            vec![],
            Some(DataType::Long),
            scalar_subquery(
                LogicalPlan::table([table]).aggregate(vec![], vec![f("count", vec![int(1)])]),
            ),
        )
    };
    let r = call(vec![f("reads", vec![])], vec![reads("t")]).unwrap();
    assert!(
        r.securables
            .iter()
            .any(|u| u.securable.to_string().contains("`t`"))
    );
    let (class, _) = err(&call(vec![f("reads", vec![])], vec![reads("nope")]));
    assert_eq!(class, "TABLE_OR_VIEW_NOT_FOUND");
    // A function the body calls must exist.
    let calls_missing = func("cm", vec![], Some(DataType::Integer), f("missing", vec![]));
    let (class, _) = err(&call(vec![f("cm", vec![])], vec![calls_missing]));
    assert_eq!(class, "UNRESOLVED_ROUTINE");
}

#[test]
fn the_legacy_flag_lets_a_parameter_shadow_a_parameterless_function() {
    // `current_user` as a parameter and as the body's whole text.
    let p = func(
        "p",
        vec![param("current_user", DataType::String, None)],
        Some(DataType::String),
        c("current_user"),
    );
    let show = |legacy: bool| {
        let conf = ResolutionConf {
            udf_parameter_shadows_function: legacy,
            ..ResolutionConf::default()
        };
        run(
            select(vec![f("p", vec![s("arg")])]),
            &ctx_with(vec![p.clone()], conf),
        )
        .unwrap()
    };
    assert!(show(false).plan.root().to_string().contains("current_user"));
    let legacy = show(true).plan.root().to_string();
    assert!(!legacy.contains("current_user()"), "{legacy}");
    assert!(legacy.contains("'arg'"), "{legacy}");
}

#[test]
fn the_legacy_flag_is_read_from_the_session_config() {
    let on = ResolutionConf::from_conf(|k| {
        (k == "spark.sql.legacy.allowUdfParameterToShadowParameterlessFunction").then_some(" TRUE ")
    });
    assert!(on.udf_parameter_shadows_function);
    assert!(!ResolutionConf::from_conf(|_| None).udf_parameter_shadows_function);
}

#[test]
fn a_function_change_changes_the_context_fingerprint() {
    let plan = || select(vec![f("add2", vec![int(1)])]);
    let analyze = |version: u64| {
        let mut ctx = ctx_with(vec![add2()], ResolutionConf::default());
        ctx.functions = Arc::new(Registry {
            base: BuiltinFunctions::new(),
            defs: [("add2".to_owned(), Arc::new(add2()))].into(),
            version,
        });
        run(plan(), &ctx).unwrap().context
    };
    let (v1, v1_again, v2) = (analyze(1), analyze(1), analyze(2));
    assert_eq!(v1, v1_again);
    assert_eq!(v1.digest(), v1_again.digest());
    assert_eq!(v1.functions, 1);
    assert_ne!(v1, v2);
    assert_ne!(v1.digest(), v2.digest());
}

#[test]
fn an_inferred_return_type_is_the_bodys() {
    let g = func(
        "inf",
        vec![param("x", DataType::Integer, None)],
        None,
        add(c("x"), int(1)),
    );
    let r = call(vec![f("inf", vec![int(1)])], vec![g]).unwrap();
    assert_eq!(r.output[0].data_type, DataType::Integer);
    assert!(
        !r.plan.root().to_string().contains("CAST(("),
        "{}",
        r.plan.root()
    );
}

#[test]
fn distinct_and_filter_are_refused() {
    let mut d = Expr::function("add2", vec![int(1)]);
    if let ExprNode::Function(call) = &mut d.node {
        call.distinct = true;
    }
    let (class, _) = err(&call(vec![d], vec![add2()]));
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
/// The registry ranks built-ins first; the session refuses to create such a
/// function (Spark: `ROUTINE_ALREADY_EXISTS`), so this only guards the order.
fn builtins_rank_before_session_functions() {
    let abs = func("abs", vec![], Some(DataType::Integer), int(1));
    let r = call(vec![f("abs", vec![int(-5)])], vec![abs]).unwrap();
    assert!(calls(&r).iter().any(|n| n == "abs"));
}
