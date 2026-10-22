// SPDX-License-Identifier: Apache-2.0
//! SQL lowering and the analyzer against Spark 4.2.
//!
//! - Analyzed, they and a name-resolution corpus give Spark's output names
//!   and nullability, or Spark's error class, message parameters and query
//!   context (`tests/fixtures/spark-4.2.0/analyzer-goldens.json`, generated
//!   by `tests/scripts/gen_analyzer_goldens.py` from a real Spark run), and
//!   output types (Spark's type coercion, P2-T071b). The resolved TPC-H
//!   plans are insta snapshots.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(clippy::unwrap_in_result, reason = "a malformed fixture is a test bug")]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, QueryContext,
    ResolutionConf, TableColumn,
};
use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, FunctionId, JoinCondition, JoinKind, LogicalPlan, PlanNode,
    ResolvedExpr, SourceId,
};
use krishiv_sql::{ParserOptions, parse_and_lower, redact_sql};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn goldens() -> Value {
    fixture("analyzer-goldens.json")
}

fn fixture(name: &str) -> Value {
    let path = root().join("tests/fixtures/spark-4.2.0").join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn tpch(id: &str) -> String {
    let path = root().join(format!("bench/queries/tpch/{id}.sql"));
    let text = std::fs::read_to_string(path).unwrap();
    text.trim().trim_end_matches(';').to_owned()
}

/// The fixture's tables in `spark_catalog.default`.
fn catalog(g: &Value) -> MemoryCatalog {
    let mut c = MemoryCatalog::new("spark_catalog");
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
        c = c.with_table(&["default"], name, columns, None);
    }
    c
}

fn conf(c: &Value) -> ResolutionConf {
    let map = c.as_object().cloned().unwrap_or_default();
    ResolutionConf::from_conf(|k| map.get(k).and_then(Value::as_str))
}

fn analyze(g: &Value, sql: &str, conf: ResolutionConf) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0)).map_err(|e| {
        AnalysisError {
            error: e.error,
            origin: krishiv_plan::Origin::UNKNOWN,
        }
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &context(catalog(g), conf)))
}

/// Mismatches between `got` and Spark's `want`, as text.
fn compare(sql: &str, got: &Result<AnalyzedPlan, AnalysisError>, want: &Value) -> Option<String> {
    let text = redact_sql(sql);
    match (got, want["ok"].as_bool().unwrap()) {
        (Ok(a), true) => {
            let got: Vec<(String, String, bool)> = a
                .output
                .iter()
                .map(|c| (c.name.clone(), c.data_type.simple_string(), c.nullable))
                .collect();
            let want: Vec<(String, String, bool)> = want["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| {
                    (
                        f["name"].as_str().unwrap().to_owned(),
                        f["type"].as_str().unwrap().to_owned(),
                        f["nullable"].as_bool().unwrap(),
                    )
                })
                .collect();
            (got != want).then(|| format!("output {got:?}, Spark {want:?}"))
        }
        (Err(e), false) => {
            let class = e.error.error_class().name();
            let params: serde_json::Map<String, Value> = e
                .error
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), Value::String(v.clone())))
                .collect();
            let mut diff = Vec::new();
            if class != want["error_class"] {
                diff.push(format!("class {class}, Spark {}", want["error_class"]));
            } else if Value::Object(params.clone()) != want["message_parameters"] {
                diff.push(format!(
                    "params {params:?}, Spark {}",
                    want["message_parameters"]
                ));
            }
            if let Some(ctx) = want.get("context").filter(|c| c.get("start").is_some()) {
                let got = e.query_context(&text);
                let got = match got {
                    Some(QueryContext::Sql {
                        start,
                        stop,
                        fragment,
                    }) => Some((start as u64, stop as u64, fragment)),
                    _ => None,
                };
                let want = (
                    ctx["start"].as_u64().unwrap(),
                    ctx["stop"].as_u64().unwrap(),
                    ctx["fragment"].as_str().unwrap().to_owned(),
                );
                if got.as_ref() != Some(&want) {
                    diff.push(format!("context {got:?}, Spark {want:?}"));
                }
            }
            (!diff.is_empty()).then(|| diff.join("; "))
        }
        (Ok(a), false) => Some(format!(
            "resolved to {:?}, Spark raised {}",
            a.output.iter().map(|c| &c.name).collect::<Vec<_>>(),
            want["error_class"]
        )),
        (Err(e), true) => Some(format!("failed with {e}, Spark resolved it")),
    }
}

#[test]
fn tpch_resolves_as_spark_does() {
    let g = goldens();
    let mut failures = Vec::new();
    for case in g["tpch"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let got = analyze(&g, &tpch(id), ResolutionConf::default());
        if let Some(diff) = compare(&tpch(id), &got, case) {
            failures.push(format!("{id}: {diff}"));
            continue;
        }
        let a = got.unwrap();
        assert!(a.plan.root().is_resolved(), "{id}");
        assert_eq!(
            a.plan.root().resolved_output(),
            Some(a.output.clone()),
            "{id}"
        );
        insta::assert_snapshot!(
            format!("tpch_{}", id.to_lowercase()),
            a.plan.root().to_string()
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `e`'s arguments when it is the operator `op`.
fn operator(e: &Expr, op: Builtin) -> Option<&[Arc<Expr>]> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.id == FunctionId::Builtin(op) => {
            Some(&f.args)
        }
        _ => None,
    }
}

/// The attribute `e` reads directly, under any casts.
fn column(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        ExprNode::Cast { expr, .. } => column(expr),
        _ => None,
    }
}

/// Whether `e` equates a column of `left` with one of `right`: `l = r`,
/// or the NULL-aware `(l = r) OR isnull(l = r)` of a `NOT IN`.
fn equi_key(e: &Expr, left: &BTreeSet<AttrId>, right: &BTreeSet<AttrId>) -> bool {
    if let Some([eq, unknown]) = operator(e, Builtin::Or) {
        return operator(unknown, Builtin::IsNull).is_some_and(|a| a.len() == 1 && a[0] == *eq)
            && equi_key(eq, left, right);
    }
    let Some([a, b]) = operator(e, Builtin::Eq) else {
        return false;
    };
    let (Some(a), Some(b)) = (column(a), column(b)) else {
        return false;
    };
    (left.contains(&a) && right.contains(&b)) || (left.contains(&b) && right.contains(&a))
}

/// Every join of `plan` that is not a cross product, by its kind, and
/// whether its condition has an equi-key conjunct; and whether any
/// expression still reads an enclosing query's column.
fn joins(plan: &Arc<LogicalPlan>, out: &mut Vec<(JoinKind, bool)>, outer: &mut bool) {
    fn exprs(e: &Arc<Expr>, out: &mut Vec<(JoinKind, bool)>, outer: &mut bool) {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => *outer = true,
            ExprNode::Subquery { plan, .. } => joins(plan, out, outer),
            _ => {}
        }
        e.for_each_child(&mut |c| exprs(c, out, outer));
    }
    if let PlanNode::Join {
        left,
        right,
        kind,
        condition: JoinCondition::On(on),
    } = &plan.node
    {
        let ids = |p: &LogicalPlan| -> BTreeSet<AttrId> {
            p.resolved_output().unwrap().iter().map(|c| c.id).collect()
        };
        let (l, r) = (ids(left), ids(right));
        let mut stack = vec![Arc::clone(on)];
        let mut keyed = false;
        while let Some(c) = stack.pop() {
            match operator(&c, Builtin::And) {
                Some(args) => stack.extend(args.iter().cloned()),
                None => keyed |= equi_key(&c, &l, &r),
            }
        }
        out.push((*kind, keyed));
    }
    plan.for_each_expr(&mut |e| exprs(e, out, outer));
    plan.for_each_child(&mut |c| joins(c, out, outer));
}

/// TPC-H's subqueries become joins with equi keys (or the NULL-aware key
/// of a `NOT IN`), so none needs a nested loop, and no correlation is
/// left behind (P2-T071c).
#[test]
fn tpch_subqueries_become_equi_joins() {
    let g = goldens();
    let mut kinds = BTreeSet::new();
    for case in g["tpch"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let a = analyze(&g, &tpch(id), ResolutionConf::default()).unwrap();
        let (mut found, mut outer) = (Vec::new(), false);
        joins(a.plan.root(), &mut found, &mut outer);
        assert!(!outer, "{id}: an outer reference is left");
        for (kind, keyed) in found {
            assert!(keyed, "{id}: a {kind:?} join without an equi key");
            kinds.insert(format!("{kind:?}"));
        }
    }
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        ["LeftAnti", "LeftOuter", "LeftSemi"]
    );
}

/// Shapes Krishiv rejects explicitly (`KRISHIV_UNSUPPORTED_FEATURE`) where
/// Spark resolves or fails with another class, never with a wrong result:
/// `+` and `-` with a date or timestamp operand (`date_add`, intervals;
/// owner: P2-T025a), operators' string coercions with
/// `spark.sql.ansi.enabled` off (owner: P2-T071d), and correlated
/// subqueries beyond TPC-H's shapes (owner: P2-T115, general unnesting).
const GAPS: [&str; 12] = [
    "date and timestamp arithmetic",
    "implicit string casts with spark.sql.ansi.enabled off",
    "correlated EXISTS and IN subqueries inside OR, NOT or other expressions",
    "correlated scalar subqueries without an aggregate",
    "correlated scalar subqueries with GROUP BY",
    "a correlated scalar subquery whose value over no rows is not NULL or a count",
    "correlated scalar subqueries whose correlation is not an equality",
    "correlated predicates with subqueries",
    "outer references in a correlated scalar subquery's value",
    "outer references in a subquery's select list",
    "outer references below a join, aggregate, limit or set operation",
    "correlated subqueries outside a filter or a projection",
];

/// Statements whose output is nullable where Spark's is not, by design
/// (ADR 0023): a STRING is valid UTF-8, so `TRY_CAST` of binary bytes that
/// are not yields NULL.
const NULLABLE_BY_DESIGN: [&str; 1] = ["SELECT TRY_CAST(X'01' AS STRING) FROM n"];

/// Whether `got` fails with one of the [`GAPS`].
fn known_gap(got: &Result<AnalyzedPlan, AnalysisError>) -> bool {
    got.as_ref().err().is_some_and(|e| {
        e.error.error_class().name() == "KRISHIV_UNSUPPORTED_FEATURE"
            && e.error
                .message_parameters()
                .iter()
                .any(|(k, v)| *k == "feature" && GAPS.contains(&v.as_str()))
    })
}

#[test]
fn resolution_corpus_matches_spark() {
    let g = goldens();
    let mut failures = Vec::new();
    let mut raised = std::collections::BTreeSet::new();
    let mut gaps = 0;
    let mut by_design = 0;
    for case in g["corpus"].as_array().unwrap() {
        let sql = case["sql"].as_str().unwrap();
        let conf = conf(&case["conf"]);
        let got = analyze(&g, sql, conf);
        if known_gap(&got) {
            gaps += 1;
            continue;
        }
        let mut case = case.clone();
        if NULLABLE_BY_DESIGN.contains(&sql) {
            by_design += 1;
            case["output"][0]["nullable"] = serde_json::Value::Bool(true);
        }
        if let Some(diff) = compare(sql, &got, &case) {
            failures.push(format!("{sql} {}: {diff}", case["conf"]));
        }
        if let Err(e) = &got {
            raised.insert(e.error.error_class().name());
        }
        // A resolved plan states its output and resolves to itself.
        if let Ok(a) = got {
            let ctx = context(catalog(&g), conf);
            let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
            if !again.plan.structurally_eq(&a.plan)
                || a.plan.root().resolved_output() != Some(a.output)
            {
                failures.push(format!("{sql}: not a fixed point\n{}", again.plan.root()));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The gaps stay the statements they are now.
    assert_eq!(gaps, 132, "statements in documented gaps");
    assert_eq!(by_design, NULLABLE_BY_DESIGN.len());
    // Classes of lateral aliases, ordinals and type coercion, each matched
    // against Spark above.
    for class in [
        "AMBIGUOUS_LATERAL_COLUMN_ALIAS",
        "GROUP_BY_POS_AGGREGATE",
        "GROUP_BY_POS_OUT_OF_RANGE",
        "ORDER_BY_POS_OUT_OF_RANGE",
        "STAR_GROUP_BY_POS",
        "UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_AGGREGATE_FUNC",
        "UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY",
        // Type coercion and checks.
        "DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES",
        "DATATYPE_MISMATCH.BINARY_OP_WRONG_TYPE",
        "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION",
        "DATATYPE_MISMATCH.CAST_WITH_CONF_SUGGESTION",
        "DATATYPE_MISMATCH.CAST_WITH_FUNC_SUGGESTION",
        "DATATYPE_MISMATCH.DATA_DIFF_TYPES",
        "DATATYPE_MISMATCH.FILTER_NOT_BOOLEAN",
        "DATATYPE_MISMATCH.IN_SUBQUERY_DATA_TYPE_MISMATCH",
        "DATATYPE_MISMATCH.IN_SUBQUERY_LENGTH_MISMATCH",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
        "INVALID_AGGREGATE_FILTER.NOT_BOOLEAN",
        "INVALID_LIMIT_LIKE_EXPRESSION.DATA_TYPE",
        "JOIN_CONDITION_IS_NOT_BOOLEAN_TYPE",
        // Subqueries.
        "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.CORRELATED_REFERENCE",
    ] {
        assert!(raised.contains(class), "{class} not raised");
    }
}

/// The columns a `Sort` or `Filter` of `plan` reads that its input does
/// not output: a plan like that analyzes but cannot be planned.
fn dangling(plan: &Arc<LogicalPlan>, out: &mut Vec<String>) {
    fn reads(e: &Arc<Expr>, ids: &mut BTreeSet<AttrId>) {
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node {
            ids.insert(c.id);
        }
        e.for_each_child(&mut |k| reads(k, ids));
    }
    if let PlanNode::Sort { input, .. } | PlanNode::Filter { input, .. } = &plan.node {
        let have: BTreeSet<AttrId> = input
            .resolved_output()
            .unwrap()
            .iter()
            .map(|c| c.id)
            .collect();
        let mut used = BTreeSet::new();
        plan.for_each_expr(&mut |e| reads(e, &mut used));
        for id in used.difference(&have) {
            out.push(format!("{id:?} in\n{plan}"));
        }
    }
    plan.for_each_child(&mut |c| dangling(c, out));
}

/// Constant folding of the arguments that need a constant (`round`'s and
/// `ceil`'s scale, `date_part`'s field, `LIMIT`), ORDER BY and HAVING over
/// grouping expressions: Spark's outcome for each statement
/// (`analyzer-folding-goldens.json`, generated with
/// `gen_analyzer_goldens.py --section folding`).
#[test]
fn folding_and_grouped_ordering_match_spark() {
    let g = fixture("analyzer-folding-goldens.json");
    let mut failures = Vec::new();
    let mut raised = BTreeSet::new();
    for case in g["corpus"].as_array().unwrap() {
        let sql = case["sql"].as_str().unwrap();
        let got = analyze(&g, sql, conf(&case["conf"]));
        if let Some(diff) = compare(sql, &got, case) {
            failures.push(format!("{sql} {}: {diff}", case["conf"]));
        }
        match &got {
            Ok(a) => {
                let mut bad = Vec::new();
                dangling(a.plan.root(), &mut bad);
                failures.extend(bad.into_iter().map(|b| format!("{sql}: dangling {b}")));
            }
            Err(e) => {
                raised.insert(e.error.error_class().name());
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    for class in [
        "ARITHMETIC_OVERFLOW",
        "DIVIDE_BY_ZERO",
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT",
        "NON_FOLDABLE_ARGUMENT",
        "UNRESOLVED_COLUMN.WITH_SUGGESTION",
    ] {
        assert!(raised.contains(class), "{class} not raised");
    }
}

/// SQL text inside a plan resolves through the front end, which keeps it
/// redacted; its parse errors keep their span and never echo a secret.
#[test]
fn sql_text_in_plans_goes_through_the_front_end() {
    use std::sync::Arc;

    use krishiv_plan::analyzer::SqlFrontend as _;
    use krishiv_plan::{LogicalPlan, Plan, PlanNode, SqlArgs};
    use krishiv_sql::Frontend;

    let g = goldens();
    let mut ctx = context(catalog(&g), ResolutionConf::default());
    ctx.sql = Some(Arc::new(Frontend::default()));
    let sql = |q: &str| {
        Plan::try_new(LogicalPlan::new(PlanNode::Sql {
            query: q.to_owned(),
            args: SqlArgs::None,
        }))
        .unwrap()
    };
    let text = "SELECT a FROM t WITH ('password' = 'Canary7f3a9c')";
    let a = block_on(DefaultAnalyzer.analyze(&sql(text), &ctx)).unwrap();
    assert_eq!(a.output[0].name, "a");
    let shown = a.query_text.unwrap().to_string();
    assert!(!shown.contains("Canary7f3a9c"), "{shown}");
    assert_eq!(a.plan.root().resolved_output(), Some(a.output));
    let e = block_on(DefaultAnalyzer.analyze(
        &sql("SELECT a FROM t WITH (password = Canary7f3a9c x)"),
        &ctx,
    ))
    .unwrap_err();
    assert!(e.origin.sql.is_some());
    assert!(!format!("{e} {e:?}").contains("Canary7f3a9c"));
    let ansi = Frontend {
        options: ParserOptions {
            enforce_reserved_keywords: true,
            ..ParserOptions::default()
        },
    };
    assert_ne!(ansi.fingerprint(), Frontend::default().fingerprint());
}

/// A correlation key column met twice is one grouping key: every
/// relation of the rewritten plan outputs each attribute once.
#[test]
fn repeated_correlation_keys_stay_unique() {
    fn check(p: &Arc<LogicalPlan>) {
        let out = p.resolved_output().unwrap();
        let ids: BTreeSet<AttrId> = out.iter().map(|c| c.id).collect();
        assert_eq!(ids.len(), out.len(), "{p}");
        p.for_each_child(&mut |c| check(c));
    }
    let g = goldens();
    let sql = "SELECT a FROM t WHERE (SELECT max(u.a) FROM u WHERE u.a = t.a AND u.a = CAST(t.c AS INT)) > 1";
    let a = analyze(&g, sql, ResolutionConf::default()).unwrap();
    check(a.plan.root());
    assert!(a.plan.root().to_string().contains("Aggregate [a#3], "));
}
