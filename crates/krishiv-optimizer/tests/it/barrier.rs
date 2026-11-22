// SPDX-License-Identifier: Apache-2.0
//! Planted leaks against the security barrier (design `P3-E14-security-barrier.md` §7.2): every
//! scan of `t` is wrapped as `Barrier(Project(masks) -> Filter(policy(c)) -> Scan)`, with
//! `policy` and `mask` opaque deterministic test UDFs (not leakproof) and `leak` the test UDF the
//! query's `abs(..)` calls become. After `optimize` to its fixed point and once more, nothing
//! that is not leakproof runs inside a barrier except the policy's own conjuncts, the mask
//! projection stays above the policy filter, and a leakproof conjunct on an unmasked column did
//! move in.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::{Principal, PrincipalKey, PrincipalKind};
use krishiv_optimizer::{Options, Statistics, optimize, optimize_with_options, optimize_with_udfs};
use krishiv_plan::analyzer::{AnalysisContext, Securable, TableIdent};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, JoinCondition,
    JoinKind, LogicalPlan, Plan, PlanNode, PolicyRef, ReadSource, ResolvedExpr, ResolvedFunction,
    leakproof,
};
use krishiv_types::DataType;
use krishiv_udf_sdk::UdfLibrary;

use crate::support::{self, data, eval};
use crate::{qerror, tpcds};

fn udf(name: &str, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Udf {
                name: name.to_owned(),
                version: 1,
            },
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn col(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

fn is_udf(e: &Expr, name: &str) -> bool {
    matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Function(f))
        if matches!(&f.id, FunctionId::Udf { name: n, .. } if n == name))
}

/// Whether `e` calls the test UDF `name` anywhere.
fn calls(e: &Expr, name: &str) -> bool {
    let mut found = is_udf(e, name);
    e.for_each_child(&mut |c| found |= calls(c, name));
    found
}

fn policy() -> PolicyRef {
    PolicyRef::new(
        Securable::Table(TableIdent {
            catalog: "spark_catalog".to_owned(),
            namespace: vec!["default".to_owned()],
            name: "t".to_owned(),
        }),
        PrincipalKey::of(&Principal::new("ana", PrincipalKind::User)),
        1,
    )
}

/// `plan` with its `abs(..)` calls turned into `leak(..)` and every scan of `t` behind a barrier
/// masking the columns named in `masked`.
///
/// Every table goes behind a barrier when `all`, its policy over its `c` or first column. With
/// `plant`, each barrier sits under a `Filter(isnotnull(leak(key)))` (the key unmasked), a leak
/// that keeps every row and must stay above the barrier.
fn guarded_tables(plan: &Plan, masked: &[&str], all: bool, plant: bool) -> Arc<LogicalPlan> {
    let mut next = 1_000_000_u64;
    Arc::clone(plan.root())
        .transform_up(&mut |p| {
            let p = leaks(p)?;
            let PlanNode::Scan(scan) = &p.node else {
                return Ok(p);
            };
            if !matches!(&scan.source, ReadSource::Table { name, .. }
                if all || name.last().is_some_and(|n| n == "t"))
            {
                return Ok(p);
            }
            let mut inner = scan.clone();
            let mut items = Vec::new();
            for c in &mut inner.output {
                if masked.contains(&c.name.as_str()) {
                    let outer = c.clone();
                    next += 1;
                    c.id = AttrId(next);
                    items.push(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Alias {
                            expr: udf("mask", vec![col(c)], c.data_type.clone()),
                            output: outer,
                        },
                    ))));
                } else {
                    items.push(col(c));
                }
            }
            let Some(key) = inner
                .output
                .iter()
                .find(|c| c.name == "c")
                .or_else(|| inner.output.first())
                .cloned()
            else {
                return Ok(p);
            };
            let scan = LogicalPlan::new(PlanNode::Scan(inner));
            let filtered = scan.filter(udf("policy", vec![col(&key)], DataType::Boolean));
            let masked_key = masked.contains(&key.name.as_str());
            let masked = LogicalPlan::new(PlanNode::Project {
                input: Arc::new(filtered),
                exprs: items,
            });
            let barrier = masked.security_barrier(policy());
            if !plant || masked_key {
                return Ok(Arc::new(barrier));
            }
            let leak = udf("leak", vec![col(&key)], key.data_type.clone());
            Ok::<_, std::convert::Infallible>(Arc::new(barrier.filter(Arc::new(Expr::new(
                ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::IsNotNull),
                    kind: FunctionKind::Scalar,
                    args: vec![leak],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: DataType::Boolean,
                    nullable: false,
                    deterministic: true,
                })),
            )))))
        })
        .unwrap_or_else(|e| match e {})
}

/// `p`'s own `abs(..)` calls as `leak(..)`.
fn leaks(p: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, std::convert::Infallible> {
    p.map_expressions(|e| {
        e.transform_up(&mut |x| {
            Ok(match &x.node {
                ExprNode::Resolved(ResolvedExpr::Function(f))
                    if matches!(&f.id, FunctionId::Registry { name, .. } if name == "abs") =>
                {
                    udf("leak", f.args.clone(), f.data_type.clone())
                }
                _ => x,
            })
        })
    })
}

/// The whole of `sql`'s plan behind a barrier, as a definer view's body would be: its own
/// filters, leaks included, start inside. Optimized to its fixed point and once more.
fn definer(sql: &str) -> Arc<LogicalPlan> {
    let plan = support::resolve(sql, &support::ctx());
    let body = Arc::clone(plan.root())
        .transform_up(&mut leaks)
        .unwrap_or_else(|e| match e {});
    let root = Plan::try_new(LogicalPlan::clone(&body).security_barrier(policy())).unwrap();
    let once = optimize(&root).unwrap();
    let twice = optimize(&once).unwrap();
    assert!(
        once.structurally_eq(&twice),
        "{sql}: not a fixed point\n{once}\n{twice}"
    );
    let mut inputs = Vec::new();
    barriers(twice.root(), &mut inputs);
    assert_eq!(inputs.len(), 1, "{sql}\n{twice}");
    inputs.remove(0)
}

/// Whether anything under `p` (`p` included) calls `name`.
fn subtree_calls(p: &Arc<LogicalPlan>, name: &str) -> bool {
    let mut all = Vec::new();
    nodes(p, &mut all);
    all.iter().any(|n| {
        let mut found = false;
        n.for_each_expr(&mut |e| found |= calls(e, name));
        found
    })
}

/// The conjunct texts of the filters under `p` (`p` included).
fn filter_texts(p: &Arc<LogicalPlan>) -> Vec<String> {
    let mut all = Vec::new();
    nodes(p, &mut all);
    all.iter()
        .filter_map(|n| match &n.node {
            PlanNode::Filter { predicate, .. } => Some(
                conjuncts(predicate)
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect()
}

/// Every barrier input under `p`.
fn barriers(p: &Arc<LogicalPlan>, out: &mut Vec<Arc<LogicalPlan>>) {
    if let PlanNode::SecurityBarrier { input, .. } = &p.node {
        out.push(Arc::clone(input));
    }
    p.for_each_child(&mut |c| barriers(c, out));
}

/// Every node under `p`, `p` included.
fn nodes(p: &Arc<LogicalPlan>, out: &mut Vec<Arc<LogicalPlan>>) {
    out.push(Arc::clone(p));
    p.for_each_child(&mut |c| nodes(c, out));
}

fn conjuncts(e: &Expr) -> Vec<&Expr> {
    krishiv_plan::conjuncts(e)
}

/// The checks of §7.2 on one barrier's input; returns its conjuncts' text.
fn check_inside(input: &Arc<LogicalPlan>, sql: &str) -> Vec<String> {
    let mut all = Vec::new();
    nodes(input, &mut all);
    let mut texts = Vec::new();
    let mut policy_filter = None;
    for n in &all {
        n.for_each_expr(&mut |e| {
            assert!(!calls(e, "leak"), "{sql}: leak inside a barrier: {e}");
        });
        let preds: Vec<&Expr> = match &n.node {
            PlanNode::Filter { predicate, .. } => conjuncts(predicate),
            PlanNode::Join {
                condition: krishiv_plan::JoinCondition::On(c),
                ..
            } => conjuncts(c),
            _ => Vec::new(),
        };
        for c in preds {
            if is_udf(c, "policy") {
                policy_filter = Some(Arc::clone(n));
                continue;
            }
            assert!(
                leakproof(c, true),
                "{sql}: a conjunct that is not leakproof inside a barrier: {c}"
            );
            texts.push(c.to_string());
        }
    }
    let Some(policy_filter) = policy_filter else {
        // A contradiction that moved in folds `policy(..) AND false` to `false`: the barrier
        // returns no row, so nothing runs on a row the policy would reject.
        assert!(
            all.iter()
                .any(|n| matches!(&n.node, PlanNode::Filter { predicate, .. }
                if predicate.to_string() == "false")),
            "{sql}: the policy filter is gone\n{input}"
        );
        return texts;
    };
    // The mask projection stays above the policy filter.
    let mut below = Vec::new();
    nodes(&policy_filter, &mut below);
    for n in &below {
        n.for_each_expr(&mut |e| assert!(!calls(e, "mask"), "{sql}: a mask below the policy: {e}"));
    }
    texts
}

/// Optimizes `sql` with `t` guarded, to its fixed point and once more, and checks every barrier;
/// returns the text of the leakproof conjuncts that moved in.
fn run(sql: &str, masked: &[&str]) -> (Arc<LogicalPlan>, Vec<String>) {
    run_over(sql, masked, false)
}

fn run_over(sql: &str, masked: &[&str], all: bool) -> (Arc<LogicalPlan>, Vec<String>) {
    run_in(
        sql,
        &support::ctx(),
        &qerror::stats_of(&data::small(3)),
        masked,
        all,
    )
}

/// [`run_over`] over the tables of `ctx`, in two arms: `optimize` (no session configuration) and
/// `optimize_with_udfs` with `stats` and a default session configuration, where the rules that
/// need one (`ShareCte`, `ShareFilteredScan`, `SubqueryAsWindow`, `DerivedKeysAbove`,
/// `PayloadAboveAggregate`) run too. Both arms are checked; the first arm's root and moved
/// conjuncts are returned.
fn run_in(
    sql: &str,
    ctx: &AnalysisContext,
    stats: &dyn Statistics,
    masked: &[&str],
    all: bool,
) -> (Arc<LogicalPlan>, Vec<String>) {
    let plan = support::resolve(sql, ctx);
    let guarded = Plan::try_new(guarded_tables(&plan, masked, all, false)).unwrap();
    let plain = |p: &Plan| optimize(p).unwrap();
    let configured =
        |p: &Plan| optimize_with_udfs(p, stats, &UdfLibrary::new(), &Config::new()).unwrap();
    let mut first = None;
    for (arm, opt) in [
        ("plain", &plain as &dyn Fn(&Plan) -> Plan),
        ("configured", &configured),
    ] {
        let once = opt(&guarded);
        let twice = opt(&once);
        assert!(
            once.structurally_eq(&twice),
            "{sql} ({arm}): not a fixed point\n{once}\n{twice}"
        );
        let root = Arc::clone(twice.root());
        let mut inputs = Vec::new();
        barriers(&root, &mut inputs);
        assert!(
            !inputs.is_empty(),
            "{sql} ({arm}): the barrier is gone\n{twice}"
        );
        check_g2(&root, &format!("{sql} ({arm})"));
        let what = format!("{sql} ({arm})");
        let moved: Vec<String> = inputs.iter().flat_map(|i| check_inside(i, &what)).collect();
        first.get_or_insert((root, moved));
    }
    first.unwrap()
}

/// G2 (design §2): nothing computed inside a barrier is shared with anything outside it. No
/// relation (`Arc`) is reachable both through a barrier and without crossing one (a `WITH`
/// definition, a `Shift` body or a shared scan read from both sides), and no `CteRef` sits inside
/// a barrier (its definition is computed outside). A whole barrier shared by a `CteRef` is fine.
fn check_g2(root: &Arc<LogicalPlan>, what: &str) {
    fn outside(p: &Arc<LogicalPlan>, seen: &mut HashSet<*const LogicalPlan>) {
        if matches!(p.node, PlanNode::SecurityBarrier { .. }) || !seen.insert(Arc::as_ptr(p)) {
            return;
        }
        p.for_each_child(&mut |c| outside(c, seen));
    }
    let mut out = HashSet::new();
    outside(root, &mut out);
    let mut inputs = Vec::new();
    barriers(root, &mut inputs);
    for input in &inputs {
        let mut inside = Vec::new();
        nodes(input, &mut inside);
        for n in &inside {
            assert!(
                !out.contains(&Arc::as_ptr(n)),
                "{what}: a relation inside a barrier is also read outside it:\n{n}\n{root}"
            );
            assert!(
                !matches!(n.node, PlanNode::CteRef(_)),
                "{what}: a CteRef inside a barrier:\n{root}"
            );
        }
    }
}

#[test]
fn leaks_above_a_barrier_stay_above_it() {
    for sql in [
        "SELECT a FROM t WHERE abs(a) = 1",
        "SELECT a FROM t WHERE 1 / (a - 42) > 0",
        "SELECT a FROM t WHERE abs(c) = 1 AND a > 0",
        "SELECT t.a FROM t JOIN u ON t.a = u.a AND abs(t.c) = 1",
        "SELECT a, count(*) FROM t GROUP BY a HAVING abs(a) = 1",
        "SELECT abs(a) FROM t ORDER BY abs(c)",
        "SELECT a FROM (SELECT a FROM t UNION ALL SELECT a FROM t) WHERE abs(a) = 1",
        "SELECT a FROM t WHERE a IN (SELECT a FROM u WHERE abs(a) = 1)",
    ] {
        run(sql, &["b"]);
    }
}

#[test]
fn a_leakproof_conjunct_moves_in_and_a_masked_one_does_not() {
    let (_, moved) = run("SELECT a FROM t WHERE a = 1 AND b = 'x'", &["b"]);
    assert!(
        moved.iter().any(|c| c.contains('1')),
        "a = 1 moved in: {moved:?}"
    );
    // `b = 'x'` over the masked column would be `mask(b) = 'x'` inside: it is not there (the
    // leakproof check would have failed) and stays above.
    assert!(
        !moved.iter().any(|c| c.contains("'x'") || c.contains("x")),
        "{moved:?}"
    );
}

#[test]
fn a_leakproof_conjunct_moves_through_a_union_of_barriers() {
    let (_, moved) = run(
        "SELECT a FROM (SELECT a FROM t UNION ALL SELECT a FROM t) WHERE a = 2",
        &["b"],
    );
    assert_eq!(
        moved.iter().filter(|c| c.contains('2')).count(),
        2,
        "{moved:?}"
    );
}

#[test]
fn a_leak_inside_a_barrier_stays_above_the_join_it_sits_on() {
    let input = definer("SELECT t.a FROM t JOIN u ON t.a = u.a WHERE abs(t.c) = 1 AND t.b = 'x'");
    let mut all = Vec::new();
    nodes(&input, &mut all);
    let joins: Vec<_> = all
        .iter()
        .filter(|n| matches!(n.node, PlanNode::Join { .. }))
        .collect();
    assert!(!joins.is_empty());
    for j in joins {
        assert!(
            !subtree_calls(j, "leak"),
            "the leak crossed or joined the join:\n{j:?}"
        );
        // Its leakproof twin did move below the join.
        let mut below = Vec::new();
        j.for_each_child(&mut |c| below.extend(filter_texts(c)));
        assert!(below.iter().any(|c| c.contains("x")), "{below:?}");
    }
    assert!(subtree_calls(&input, "leak"), "the leak is kept");
}

#[test]
fn a_leak_inside_a_barrier_stays_above_a_union() {
    let input = definer(
        "SELECT a FROM (SELECT a FROM t UNION ALL SELECT a FROM u) WHERE abs(a) = 1 AND a = 2",
    );
    let mut all = Vec::new();
    nodes(&input, &mut all);
    let unions: Vec<_> = all
        .iter()
        .filter(|n| matches!(n.node, PlanNode::Union { .. }))
        .collect();
    assert_eq!(unions.len(), 1);
    assert!(!subtree_calls(unions[0], "leak"), "{:?}", unions[0]);
    let mut below = Vec::new();
    unions[0].for_each_child(&mut |c| below.extend(filter_texts(c)));
    assert_eq!(
        below.iter().filter(|c| c.contains('2')).count(),
        2,
        "{below:?}"
    );
}

#[test]
fn a_one_sided_leak_in_a_join_condition_inside_a_barrier_stays_there() {
    let input = definer("SELECT t.a FROM t JOIN u ON t.a = u.a AND abs(t.c) = 1 AND t.b = 'y'");
    let mut all = Vec::new();
    nodes(&input, &mut all);
    for n in &all {
        if let PlanNode::Join { left, right, .. } = &n.node {
            assert!(!subtree_calls(left, "leak") && !subtree_calls(right, "leak"));
            assert!(subtree_calls(n, "leak"), "the leak stays in the condition");
            assert!(
                filter_texts(left).iter().any(|c| c.contains('y')),
                "the twin moved"
            );
        }
    }
}

#[test]
fn collation_keys_are_lowered_inside_a_barrier() {
    let input =
        definer("SELECT b COLLATE UTF8_LCASE AS k, count(*) FROM t GROUP BY b COLLATE UTF8_LCASE");
    assert!(
        format!("{input:?}").contains("__collation_key"),
        "{input:?}"
    );
}

/// The invariant over real queries (§7.3, structural half): every TPC-H query with every table
/// behind a barrier, a comment column masked, optimized to its fixed point and once more.
#[test]
fn every_tpch_query_keeps_its_barriers_whole() {
    let mut moved = 0;
    for q in 1..=22 {
        let sql = support::tpch_sql(q);
        let (_, m) = run_over(&sql, &["c_comment", "o_comment", "p_name"], true);
        moved += m.len();
    }
    // Leakproof filters (dates, keys, flags) do move in.
    assert!(moved > 20, "{moved}");
}

/// The structural half of §7.3 over TPC-DS: every query with every table behind a barrier, in
/// both arms of [`run_in`] (the configured one with the q-error report's TPC-DS statistics, where
/// `ShareCte` and `ShareFilteredScan` find their q04/q11/q28-style candidates).
#[test]
fn every_tpcds_query_keeps_its_barriers_whole() {
    let schemas = qerror::tpcds_schemas();
    let db = qerror::synthetic_tpcds(&schemas, 11, 2 * qerror::scale());
    let stats = qerror::stats_of(&db);
    let ctx = support::ctx_over(tpcds::catalog());
    let (mut moved, mut swept) = (0, 0);
    for q in 1..=99 {
        let sql = tpcds::sql(q);
        // As the rule differential: a query the test analyzer cannot resolve is left out.
        if support::try_resolve(&sql, &ctx).is_none() {
            continue;
        }
        let (_, m) = run_in(&sql, &ctx, &stats, &["i_item_desc", "c_last_name"], true);
        moved += m.len();
        swept += 1;
    }
    assert!(swept >= 90, "{swept} of 99 TPC-DS queries swept");
    assert!(moved > 99, "{moved}");
}

/// A grouped CTE read twice: `ShareCte` shares its two instances (the control of the tests below).
const SHARED_TWICE: &str = "WITH s AS (SELECT a, count(*) AS n FROM t GROUP BY a) \
     SELECT * FROM s x JOIN s y ON x.a = y.a";

/// Whether a node under `p` (`p` included) is a security barrier.
fn holds_a_barrier(p: &Arc<LogicalPlan>) -> bool {
    let mut all = Vec::new();
    nodes(p, &mut all);
    all.iter()
        .any(|n| matches!(n.node, PlanNode::SecurityBarrier { .. }))
}

/// The `CteRef`s under `p`.
fn cte_refs(p: &Arc<LogicalPlan>) -> usize {
    let mut all = Vec::new();
    nodes(p, &mut all);
    all.iter()
        .filter(|n| matches!(n.node, PlanNode::CteRef(_)))
        .count()
}

/// `ShareCte` may share two equal instances whose barrier is inside them (the whole barrier is
/// in the one definition), and does: the two reads of `s` become `CteRef`s of one definition
/// that holds the barrier, and G2 holds.
#[test]
fn share_cte_shares_equal_barriers_whole() {
    run(SHARED_TWICE, &["b"]);
    let plan = support::resolve(SHARED_TWICE, &support::ctx());
    let guarded = Plan::try_new(guarded_tables(&plan, &["b"], false, false)).unwrap();
    let stats = qerror::stats_of(&data::small(3));
    let on = optimize_with_udfs(&guarded, &stats, &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(cte_refs(on.root()) >= 2, "the two reads are shared\n{on}");
    let PlanNode::WithCte { ctes, .. } = &on.root().node else {
        panic!("no WITH definition\n{on}");
    };
    assert!(
        ctes.iter().any(|c| holds_a_barrier(&c.plan)),
        "the barrier is whole inside a definition\n{on}"
    );
    check_g2(on.root(), SHARED_TWICE);
}

/// Every node of `p` again, in new `Arc`s: an equal plan that shares no relation with `p`.
fn deep_copy(p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    Arc::clone(p)
        .transform_up(&mut |n| Ok::<_, std::convert::Infallible>(Arc::new(LogicalPlan::clone(&n))))
        .unwrap_or_else(|e| match e {})
}

/// Two instances of one grouped body, `Shift +0` of a copy (behind a barrier when `guard`) and
/// `Shift +4` of the original, joined on their key. The copy is a separate relation, so the
/// input holds no relation both inside and outside a barrier (G2's input side).
fn two_instances(guard: bool) -> Plan {
    let plan = support::resolve("SELECT a, count(*) AS n FROM t GROUP BY a", &support::ctx());
    let body = Arc::clone(plan.root());
    let shift = |input: Arc<LogicalPlan>, offset| {
        Arc::new(LogicalPlan::new(PlanNode::Shift { input, offset }))
    };
    let mut left = shift(deep_copy(&body), 0);
    if guard {
        left = Arc::new(LogicalPlan::clone(&left).security_barrier(policy()));
    }
    let right = shift(body, 4);
    let first = |p: &Arc<LogicalPlan>| p.resolved_output().unwrap()[0].clone();
    let (l, r) = (first(&left), first(&right));
    let condition = builtin(Builtin::Eq, vec![col(&l), col(&r)], false);
    Plan::try_new(Arc::new(LogicalPlan::new(PlanNode::Join {
        left,
        right,
        kind: JoinKind::Inner,
        condition: JoinCondition::On(condition),
    })))
    .unwrap()
}

/// An instance inside a barrier equal to one outside it is not shared (G2): `ShareCte` takes no
/// member under a barrier, so the guarded aggregate and the open one stay two computations; the
/// same two instances with no barrier are shared (control).
#[test]
fn share_cte_does_not_share_a_subtree_inside_a_barrier_with_one_outside() {
    let stats = qerror::stats_of(&data::small(3));
    let opt = |p: &Plan| optimize_with_udfs(p, &stats, &UdfLibrary::new(), &Config::new()).unwrap();
    let guarded = two_instances(true);
    check_g2(guarded.root(), "input");
    let on = opt(&guarded);
    assert!(on.structurally_eq(&opt(&on)), "not a fixed point\n{on}");
    assert_eq!(
        cte_refs(on.root()),
        0,
        "an aggregate inside the barrier was shared with the one outside\n{on}"
    );
    check_g2(on.root(), "inside and outside");
    let open = opt(&two_instances(false));
    assert!(
        cte_refs(open.root()) >= 2,
        "control: the open instances are shared\n{open}"
    );
}

/// The key columns of every policy call under `p` (the guarded scans' keys).
fn policy_keys(p: &Arc<LogicalPlan>) -> HashSet<AttrId> {
    let mut all = Vec::new();
    nodes(p, &mut all);
    fn walk(e: &Expr, keys: &mut HashSet<AttrId>) {
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
            && is_udf(e, "policy")
            && let Some(ExprNode::Resolved(ResolvedExpr::Column(c))) =
                f.args.first().map(|a| &a.node)
        {
            keys.insert(c.id);
        }
        e.for_each_child(&mut |c| walk(c, keys));
    }
    let mut keys = HashSet::new();
    for n in &all {
        n.for_each_expr(&mut |e| walk(e, &mut keys));
    }
    keys
}

/// The invariant at run time (§7.3, run-time half): every TPC-H query the interpreter runs, with
/// every table behind a barrier whose policy rejects the rows with a key that is a multiple of 7
/// and a leak planted above each barrier, optimized (with and without statistics, to the fixed
/// point and once more), returns the unoptimized barrier plan's rows, and no expression that is
/// not leakproof runs on a row the policy rejects (the interpreter's tagged rows panic).
#[test]
fn every_tpch_query_returns_the_barrier_rows_under_a_rejecting_policy() {
    let db = data::tpch(7, qerror::scale());
    let stats = qerror::stats_of(&db);
    let (mut compared, mut nonempty) = (0, 0);
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &support::ctx());
        let guarded = Plan::try_new(guarded_tables(
            &plan,
            &["c_comment", "o_comment", "p_name"],
            true,
            true,
        ))
        .unwrap();
        let keys = policy_keys(guarded.root());
        // The unoptimized plan's own run checks the same rule (its leaks sit above the barriers).
        let want = eval::rows_guarded(&db, guarded.root(), &keys);
        compared += 1;
        nonempty += usize::from(!want.is_empty());
        let plain = optimize(&guarded).unwrap();
        let with_stats = optimize_with_options(&guarded, &stats, &Options::default()).unwrap();
        let configured =
            optimize_with_udfs(&guarded, &stats, &UdfLibrary::new(), &Config::new()).unwrap();
        for (arm, once) in [
            ("plain", plain),
            ("statistics", with_stats),
            ("configured", configured),
        ] {
            let twice = if arm == "configured" {
                optimize_with_udfs(&once, &stats, &UdfLibrary::new(), &Config::new()).unwrap()
            } else {
                optimize_with_options(&once, &stats, &Options::default()).unwrap()
            };
            check_g2(twice.root(), &format!("q{q} {arm}"));
            for p in [&once, &twice] {
                let got = eval::rows_guarded(&db, p.root(), &keys);
                assert_eq!(got.len(), want.len(), "q{q} {arm}\n{p}");
                assert_eq!(got, want, "q{q} {arm}");
            }
        }
    }
    assert_eq!(compared, 22);
    assert!(nonempty >= 15, "{nonempty} of 22 queries return rows");
}

fn builtin(op: Builtin, args: Vec<Arc<Expr>>, nullable: bool) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(op),
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Boolean,
            nullable,
            deterministic: true,
        },
    ))))
}

/// The run-time check itself: a leak merged into the policy's `Filter` (what a broken pushdown
/// arm would produce) runs on the rows the policy rejects, and the interpreter refuses it, even
/// though the leak keeps every row and the policy's conjunct comes first.
#[test]
#[should_panic(expected = "not leakproof runs on a row the policy rejects")]
fn a_leak_beside_the_policy_fails_at_run_time() {
    let plan = support::resolve("SELECT a FROM t", &support::ctx());
    let guarded = guarded_tables(&plan, &[], false, false)
        .transform_up(&mut |p| {
            let PlanNode::Filter { input, predicate } = &p.node else {
                return Ok::<_, std::convert::Infallible>(p);
            };
            let PlanNode::Scan(scan) = &input.node else {
                return Ok(p);
            };
            let a = col(&scan.output[0]);
            let leak = builtin(
                Builtin::IsNotNull,
                vec![udf("leak", vec![a], DataType::Integer)],
                false,
            );
            let both = builtin(Builtin::And, vec![Arc::clone(predicate), leak], true);
            Ok(Arc::new(LogicalPlan::clone(input).filter(both)))
        })
        .unwrap_or_else(|e| match e {});
    let keys = policy_keys(&guarded);
    assert_eq!(keys.len(), 1);
    eval::rows_guarded(&data::small(3), &guarded, &keys);
}
