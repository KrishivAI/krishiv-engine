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

use krishiv_common::{Principal, PrincipalKey, PrincipalKind};
use krishiv_optimizer::{Options, optimize, optimize_with_options};
use krishiv_plan::analyzer::{Securable, TableIdent};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, LogicalPlan, Plan,
    PlanNode, PolicyRef, ReadSource, ResolvedExpr, ResolvedFunction, leakproof,
};
use krishiv_types::DataType;

use crate::qerror;
use crate::support::{self, data, eval};

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
    let policy_filter = policy_filter.unwrap_or_else(|| panic!("{sql}: the policy filter is gone"));
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
    let plan = support::resolve(sql, &support::ctx());
    let guarded = Plan::try_new(guarded_tables(&plan, masked, all, false)).unwrap();
    let once = optimize(&guarded).unwrap();
    let twice = optimize(&once).unwrap();
    assert!(
        once.structurally_eq(&twice),
        "{sql}: not a fixed point\n{once}\n{twice}"
    );
    let root = Arc::clone(twice.root());
    let mut inputs = Vec::new();
    barriers(&root, &mut inputs);
    assert!(!inputs.is_empty(), "{sql}: the barrier is gone\n{twice}");
    let moved = inputs.iter().flat_map(|i| check_inside(i, sql)).collect();
    (root, moved)
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
        for (arm, once) in [("plain", plain), ("statistics", with_stats)] {
            let twice = optimize_with_options(&once, &stats, &Options::default()).unwrap();
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
