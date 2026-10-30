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
    ResolutionConf, SessionZone, TableColumn,
};
use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, FunctionId, JoinCondition, JoinKind, Literal, LogicalPlan,
    NullOrdering, PlanNode, ResolvedExpr, SortDirection, SourceId,
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
    // The gaps stay the statements they are now (128 since P2-T115 unit 2a
    // resolved four correlated `EXISTS` / `IN` under `OR` and `NOT`).
    assert_eq!(gaps, 128, "statements in documented gaps");
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

/// Every `Sort` of `plan`, in walk order: its keys' direction and null
/// ordering, and whether it is global (`ORDER BY`) or not (`SORT BY`).
fn sorts_of(plan: &Arc<LogicalPlan>, out: &mut Vec<(Vec<(SortDirection, NullOrdering)>, bool)>) {
    fn exprs(e: &Arc<Expr>, out: &mut Vec<(Vec<(SortDirection, NullOrdering)>, bool)>) {
        if let ExprNode::Subquery { plan, .. } = &e.node {
            sorts_of(plan, out);
        }
        e.for_each_child(&mut |k| exprs(k, out));
    }
    if let PlanNode::Sort { order, global, .. } = &plan.node {
        out.push((
            order.iter().map(|o| (o.direction, o.nulls)).collect(),
            *global,
        ));
    }
    plan.for_each_child(&mut |c| sorts_of(c, out));
    plan.for_each_expr(&mut |e| exprs(e, out));
}

/// Sort keys of `plan` that are not bound: unresolved, reading an outer
/// query, or naming no attribute of the sort's input.
fn unbound_sorts(plan: &Arc<LogicalPlan>, out: &mut Vec<String>) {
    if let PlanNode::Sort { order, input, .. } = &plan.node {
        let have: BTreeSet<AttrId> = input
            .resolved_output()
            .unwrap()
            .iter()
            .map(|c| c.id)
            .collect();
        for o in order {
            if !o.expr.is_resolved() {
                out.push(format!("unresolved sort key in\n{plan}"));
            }
            let mut stack = vec![&o.expr];
            while let Some(e) = stack.pop() {
                match &e.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                        if !have.contains(&c.id) {
                            out.push(format!("unbound sort key {c:?} in\n{plan}"));
                        }
                    }
                    ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
                        out.push(format!("outer sort key {c:?} in\n{plan}"));
                    }
                    ExprNode::Column(_)
                    | ExprNode::Star(_)
                    | ExprNode::Function(_)
                    | ExprNode::Alias { .. } => {
                        out.push(format!("unresolved sort key in\n{plan}"));
                    }
                    _ => {}
                }
                e.for_each_child(&mut |k| stack.push(k));
            }
        }
    }
    plan.for_each_child(&mut |c| unbound_sorts(c, out));
    plan.for_each_expr(&mut |e| {
        if let ExprNode::Subquery { plan, .. } = &e.node {
            unbound_sorts(plan, out);
        }
    });
}

/// The sort keys of the single `Sort` of a resolved plan: the column each
/// key reads when it is one, else its display.
fn top_sort_keys(plan: &Arc<LogicalPlan>) -> Vec<(String, SortDirection, NullOrdering)> {
    let mut stack = vec![Arc::clone(plan)];
    let mut found = None;
    while let Some(p) = stack.pop() {
        if let PlanNode::Sort { order, .. } = &p.node {
            found = Some(order.clone());
            break;
        }
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    let order = found.unwrap();
    order
        .iter()
        .map(|o| (key_name(&o.expr), o.direction, o.nulls))
        .collect()
}

fn key_name(e: &Arc<Expr>) -> String {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => c.name.clone(),
        _ => format!("{e}"),
    }
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
                // Grouped ordering keeps the lowered sorts: the same sorts,
                // keys, directions and null ordering, all bound.
                let lowered = parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
                    .expect("a passing case lowers");
                let mut low = Vec::new();
                sorts_of(lowered.root(), &mut low);
                let mut high = Vec::new();
                sorts_of(a.plan.root(), &mut high);
                if low.len() != high.len() {
                    failures.push(format!(
                        "{sql}: {} lowered sorts, {} resolved",
                        low.len(),
                        high.len()
                    ));
                } else {
                    for (i, (l, h)) in low.iter().zip(&high).enumerate() {
                        if l.0.len() != h.0.len() {
                            failures.push(format!(
                                "{sql}: sort {i} has {} keys lowered, {} resolved",
                                l.0.len(),
                                h.0.len()
                            ));
                        }
                        if l.0 != h.0 {
                            failures.push(format!(
                                "{sql}: sort {i} directions/nulls {l:?} lowered, {h:?} resolved"
                            ));
                        }
                        if l.1 != h.1 {
                            failures.push(format!(
                                "{sql}: sort {i} global {} lowered, {} resolved",
                                l.1, h.1
                            ));
                        }
                    }
                }
                let mut unbound = Vec::new();
                unbound_sorts(a.plan.root(), &mut unbound);
                failures.extend(unbound.into_iter().map(|b| format!("{sql}: {b}")));
            }
            Err(e) => {
                raised.insert(e.error.error_class().name());
            }
        }
    }
    // Representative grouped orderings resolve to their grouping or
    // aggregate attributes, with direction and nulls kept.
    for (sql, want) in [
        (
            "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x + 1",
            vec![(
                "m".to_owned(),
                SortDirection::Ascending,
                NullOrdering::NullsFirst,
            )],
        ),
        (
            "SELECT x + 1 m, count(*) c FROM g GROUP BY x + 1 ORDER BY x + 1 DESC, c",
            vec![
                (
                    "m".to_owned(),
                    SortDirection::Descending,
                    NullOrdering::NullsLast,
                ),
                (
                    "c".to_owned(),
                    SortDirection::Ascending,
                    NullOrdering::NullsFirst,
                ),
            ],
        ),
        (
            "SELECT x m, count(*) c FROM g GROUP BY x HAVING count(*) > 3 ORDER BY sum(l)",
            vec![(
                "sum(l)".to_owned(),
                SortDirection::Ascending,
                NullOrdering::NullsFirst,
            )],
        ),
    ] {
        let got = analyze(&g, sql, ResolutionConf::default()).expect("representative resolves");
        assert_eq!(top_sort_keys(got.plan.root()), want, "{sql}");
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
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE",
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL",
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE",
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
    let consecutive = Frontend {
        options: ParserOptions {
            consecutive_string_literals: true,
            ..ParserOptions::default()
        },
    };
    assert_ne!(consecutive.fingerprint(), Frontend::default().fingerprint());
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

/// A foldable `LIMIT` that is negative or NULL fails analysis as Spark's
/// `checkLimitLikeClause` does (it used to reach the planner).
/// The `LIMIT` count of `plan`, wherever it sits.
fn limit_of(plan: &Arc<LogicalPlan>) -> Option<Arc<Expr>> {
    if let PlanNode::Limit { limit, .. } = &plan.node {
        return Some(Arc::clone(limit));
    }
    let mut found = None;
    plan.for_each_child(&mut |c| {
        if found.is_none() {
            found = limit_of(c);
        }
    });
    found
}

#[test]
fn negative_and_null_limits_fail_analysis() {
    let g = goldens();
    let err = |sql: &str| {
        analyze(&g, sql, ResolutionConf::default())
            .unwrap_err()
            .error
    };
    let e = err("SELECT a FROM t LIMIT -1");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    );
    assert_eq!(e.sql_state(), Some("42K0E"));
    assert!(e.message().contains("\"-1\""), "{}", e.message());
    assert!(e.message().contains("but got -1"), "{}", e.message());
    let e = err("SELECT a FROM t LIMIT CAST(NULL AS INT)");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL"
    );
    let a = analyze(&g, "SELECT a FROM t LIMIT 0", ResolutionConf::default()).unwrap();
    let limit = limit_of(a.plan.root()).expect("a Limit node");
    assert!(
        matches!(&limit.node, ExprNode::Literal(Literal::Integer(0))),
        "{limit:?}"
    );
}

/// `LIMIT` folds arithmetic before Spark's checks, and rejects counts that
/// cannot be evaluated without rows.
#[test]
fn limit_folds_arithmetic_and_rejects_unfoldable_counts() {
    let g = goldens();
    let err = |sql: &str| {
        analyze(&g, sql, ResolutionConf::default())
            .unwrap_err()
            .error
    };
    // `1 - 2` is a foldable negative count, not a passing limit.
    let e = err("SELECT a FROM t LIMIT 1 - 2");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    );
    assert_eq!(e.sql_state(), Some("42K0E"));
    assert!(e.message().contains("but got -1"), "{}", e.message());
    // `NULL + 1` stays NULL, and a cast of arithmetic folds through the kernels.
    let e = err("SELECT a FROM t LIMIT CAST(NULL AS INT) + 1");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL"
    );
    let e = err("SELECT a FROM t LIMIT CAST(1 - 2 AS INT)");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    );
    assert!(e.message().contains("but got -1"), "{}", e.message());
    let e = err("SELECT a FROM t LIMIT CAST(CAST(NULL AS INT) + 1 AS INT)");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL"
    );
    // Unary minus over a constant folds as well.
    let e = err("SELECT a FROM t LIMIT -(1)");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    );
    let e = err("SELECT a FROM t LIMIT -(CAST(NULL AS INT))");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL"
    );
    // A scalar subquery binds but is not foldable.
    let e = err("SELECT a FROM t LIMIT (SELECT 1)");
    assert_eq!(
        e.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    assert_eq!(e.sql_state(), Some("42K0E"));
}

/// The window-function goldens (`tests/fixtures/spark-4.2.0/window-goldens.json`,
/// generated from a real Spark run) resolve to Spark's output names, types
/// and nullability over the same schemas, and their resolved plans are a
/// fixed point. Rows and errors run end to end in `krishiv-session`'s
/// `window_goldens` test.
#[test]
fn window_goldens_resolve_to_a_fixed_point() {
    let path = root().join("tests/fixtures/spark-4.2.0/window-goldens.json");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut c = MemoryCatalog::new("spark_catalog");
    for (name, t) in g["tables"].as_object().unwrap() {
        let columns = t["columns"]
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
    let mut failures = Vec::new();
    let mut resolved = 0;
    let mut pending = Vec::new();
    for case in g["cases"].as_array().unwrap() {
        if !case["ok"].as_bool().unwrap() {
            continue;
        }
        let sql = case["sql"].as_str().unwrap();
        let conf = conf(&case["conf"]);
        let ctx = context(c.clone(), conf);
        let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0)).unwrap();
        let a = match block_on(DefaultAnalyzer.analyze(&plan, &ctx)) {
            Ok(a) => a,
            // Frames with value bounds are a later part of P2-T055.
            Err(e) if e.error.error_class().name() == "KRISHIV_UNSUPPORTED_FEATURE" => {
                pending.push(case["id"].as_str().unwrap());
                continue;
            }
            Err(e) => {
                failures.push(format!("{sql}: {}", e.error));
                continue;
            }
        };
        resolved += 1;
        let want = case["output"].as_array().unwrap();
        let got: Vec<(String, String, bool)> = a
            .output
            .iter()
            .map(|c| (c.name.clone(), c.data_type.simple_string(), c.nullable))
            .collect();
        let want: Vec<(String, String, bool)> = want
            .iter()
            .map(|w| {
                (
                    w["name"].as_str().unwrap().to_owned(),
                    w["type"].as_str().unwrap().to_owned(),
                    w["nullable"].as_bool().unwrap(),
                )
            })
            .collect();
        if got != want {
            failures.push(format!("{sql}: {got:?}\n  Spark {want:?}"));
        }
        let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
        if !again.plan.structurally_eq(&a.plan) || a.plan.root().resolved_output() != Some(a.output)
        {
            failures.push(format!("{sql}: not a fixed point\n{}", again.plan.root()));
        }
    }
    assert_eq!(pending, ["range_offset"]);
    assert_eq!(resolved, 97);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Typed datetime literals resolve in the analyzer with the session zone
/// and timestamp type: plain `TIMESTAMP` follows
/// `spark.sql.timestampType` (and a zone in the value), and instants
/// without a zone read `spark.sql.session.timeZone`. The golden files
/// (`literals.sql`, `timestamp.sql`, `timestampNTZ/timestamp.sql`,
/// `datetime-*.sql`) record these names, rows and `INVALID_TYPED_LITERAL`
/// errors.
#[test]
fn typed_timestamp_literals_resolve_like_spark() {
    let g = goldens();
    let analyze_with = |sql: &str, timestamp_ntz: bool, zone: Option<i32>| {
        let conf = ResolutionConf {
            timestamp_ntz,
            session_zone: zone.map(SessionZone::Fixed),
            ..ResolutionConf::default()
        };
        let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0)).map_err(|e| {
            AnalysisError {
                error: e.error,
                origin: krishiv_plan::Origin::UNKNOWN,
            }
        })?;
        block_on(DefaultAnalyzer.analyze(&plan, &context(catalog(&g), conf)))
    };
    let utc = Some(0);
    // Plain `TIMESTAMP` is an instant by default, wall-clock under `TIMESTAMP_NTZ`.
    let a = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00'", false, utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    assert_eq!(a.output[0].name, "TIMESTAMP '2024-01-01 10:00:00'");
    let a = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00'", true, utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp_ntz");
    assert_eq!(a.output[0].name, "TIMESTAMP_NTZ '2024-01-01 10:00:00'");
    // ... unless the value names a zone, which stays an instant.
    let a = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00+05:30'", true, utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    // The session zone moves instants without a zone (`+05:30` is 04:30 UTC):
    // Spark 4.2 gives 1_704_103_200_000_000 µs in UTC and
    // 1_704_083_400_000_000 µs in +05:30.
    let utc_plan = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00'", false, utc).unwrap();
    let plus_plan = analyze_with(
        "SELECT TIMESTAMP '2024-01-01 10:00:00'",
        false,
        Some(19_800),
    )
    .unwrap();
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
    assert_eq!(micros_of(utc_plan.plan.root()), 1_704_103_200_000_000);
    assert_eq!(micros_of(plus_plan.plan.root()), 1_704_083_400_000_000);
    // `TIMESTAMP_LTZ` and `TIMESTAMP_NTZ` keep their types.
    let a = analyze_with("SELECT TIMESTAMP_LTZ '2024-01-01 10:00:00'", true, utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    let a = analyze_with("SELECT TIMESTAMP_NTZ '2024-01-01 10:00:00'", false, utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp_ntz");
    // The session zone is unset by default (UTC): plain `TIMESTAMP` works.
    let a = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00'", false, Some(0)).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    // An explicit zone renders in a non-UTC session zone, not in UTC:
    // 10:00 UTC is 15:30 in +05:30.
    let a = analyze_with(
        "SELECT TIMESTAMP '2024-01-01 10:00:00+00:00'",
        false,
        Some(19_800),
    )
    .unwrap();
    assert_eq!(a.output[0].name, "TIMESTAMP '2024-01-01 15:30:00'");
    // Spark's golden statements (`literals.sql`, `timestamp.sql`).
    for (sql, name, ty) in [
        (
            "select tImEstAmp '2016-03-11 20:54:00.000'",
            "TIMESTAMP '2016-03-11 20:54:00'",
            "timestamp",
        ),
        (
            "select timestamp '2019-01-01\t'",
            "TIMESTAMP '2019-01-01 00:00:00'",
            "timestamp",
        ),
        ("select dAte '2016-03-12'", "DATE '2016-03-12'", "date"),
    ] {
        let a = analyze_with(sql, false, utc).unwrap();
        assert_eq!(a.output[0].name, name, "{sql}");
        assert_eq!(a.output[0].data_type.simple_string(), ty, "{sql}");
    }
    // Invalid text is `INVALID_TYPED_LITERAL` at the literal.
    for sql in [
        "SELECT TIMESTAMP '2024-13-01 10:00:00'",
        "SELECT TIMESTAMP_NTZ '10:00'",
        "SELECT DATE '2020-02-30'",
        "SELECT DATE '10:00'",
    ] {
        let e = analyze_with(sql, false, utc).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            krishiv_common::error::ErrorClass::InvalidTypedLiteral,
            "{sql}"
        );
    }
    // Clock-dependent spellings need the query clock.
    for sql in [
        "SELECT TIMESTAMP 'now'",
        "SELECT TIMESTAMP_LTZ '10:00'",
        "SELECT TIMESTAMP '10:00'",
    ] {
        let e = analyze_with(sql, false, utc).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            krishiv_common::error::ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
        assert!(e.error.to_string().contains("query clock"), "{sql}: {e}");
    }
    // A region session zone serves explicit offsets but not wall-clock time.
    let e = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00+05:30'", false, None).unwrap();
    assert_eq!(e.output[0].data_type.simple_string(), "timestamp");
    let e = analyze_with("SELECT TIMESTAMP '2024-01-01 10:00:00'", false, None).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature,
        "region session without zone"
    );
    // A region id in the value resolves against the bundled database:
    // 10:00 in Los Angeles (UTC-08:00 in January) is 18:00 UTC.
    let a = analyze_with(
        "SELECT TIMESTAMP '2024-01-01 10:00:00 America/Los_Angeles'",
        false,
        utc,
    )
    .unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    assert_eq!(micros_of(a.plan.root()), 1_704_132_000_000_000);
}

/// Foldable expressions the analyzer evaluates for typing: casts to string
/// and date, datetime functions of literals and function-form `concat`.
/// Spark folds them at analysis time, so a `LIMIT`, a legacy day count and
/// a `round` scale read them; only what Spark itself calls not foldable (a
/// column, a nondeterministic call) stays `IS_UNFOLDABLE`.
#[test]
fn foldable_limits_day_counts_and_scales_match_spark() {
    let g = goldens();
    let ansi = ResolutionConf::default();
    let legacy = conf(&serde_json::json!({ "spark.sql.ansi.enabled": "false" }));
    // A `LIMIT` over a foldable expression resolves, whatever the analyzer
    // evaluates it to (Spark returns rows for both).
    for sql in [
        "SELECT * FROM t LIMIT length(CAST(12 AS STRING))",
        "SELECT * FROM t LIMIT year(DATE'2024-01-01') - 2020",
        "SELECT * FROM t LIMIT length(concat('a', 'b'))",
    ] {
        assert!(
            analyze(&g, sql, ansi).is_ok(),
            "{sql}: Spark returns rows, the analyzer must resolve it"
        );
    }
    // A column in a `LIMIT` is what Spark calls not foldable.
    let err = analyze(&g, "SELECT * FROM t LIMIT a", ansi).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    // Foldable but beyond what the folder evaluates: Spark folds them, so
    // they resolve here and the downstream folders read them
    // (`current_date` is deterministic to Spark, so foldable, but the
    // folder has no query clock; `md5` is deterministic too, but outside
    // the folder's pure families).
    for sql in [
        "SELECT * FROM t LIMIT year(current_date)",
        "SELECT * FROM t LIMIT length(md5('a'))",
    ] {
        assert!(
            analyze(&g, sql, ansi).is_ok(),
            "{sql}: Spark folds it, so it must resolve"
        );
    }
    // The legacy day count reads through a cast to string, shown as the
    // integer Spark's `StringLiteralCoercion` folds it to.
    let a = analyze(&g, "SELECT date_add(dt, CAST(1 AS STRING)) FROM n", legacy)
        .expect("Spark folds the legacy day count");
    assert_eq!(a.output[0].name, "date_add(dt, 1)");
    assert_eq!(a.output[0].data_type.simple_string(), "date");
    // `round` reads its scale through function-form `concat`.
    let a = analyze(&g, "SELECT round(dc, length(concat('a','b'))) FROM n", ansi)
        .expect("Spark folds the round scale");
    assert_eq!(a.output[0].data_type.simple_string(), "decimal(11,2)");
    // A `concat` argument that needs its own cast folds through it too.
    let a = analyze(
        &g,
        "SELECT round(dc, length(concat('a', CAST(1 AS STRING)))) FROM n",
        ansi,
    )
    .expect("Spark folds the round scale through the cast");
    assert_eq!(a.output[0].data_type.simple_string(), "decimal(11,2)");
}
