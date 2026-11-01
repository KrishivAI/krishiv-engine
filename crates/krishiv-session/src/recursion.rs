// SPDX-License-Identifier: Apache-2.0
//! Recursive CTEs: [`PlanNode::UnionLoop`] runs here, before physical
//! planning (the coordinator-driven loop of the T180 design).
//!
//! Each loop, innermost first, runs its anchor, then its recursive term
//! again and again with [`PlanNode::UnionLoopRef`] reading the previous
//! round's rows, until a round is empty. Every round is an ordinary query
//! run: its rows become an in-memory table scanned under a hidden name (the
//! relation cache's mechanism), and the loop's node is replaced by a scan of
//! the rows of all rounds. The level and row limits are the
//! `spark.sql.cteRecursionLevelLimit` / `cteRecursionRowLimit` settings;
//! `MAX RECURSION LEVEL n` overrides the first.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::config::{CTE_RECURSION_LEVEL_LIMIT, CTE_RECURSION_ROW_LIMIT, ConfigValue};
use krishiv_common::error::{KrishivError, RecursionLevelLimitExceeded, RecursionRowLimitExceeded};
use krishiv_exec::planner::MemoryTable;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{BoundColumn, LogicalPlan, Plan, PlanNode, ReadSource, Scan};
use krishiv_types::batch::Batch;

use crate::cache::CACHE_PREFIX;
use crate::session::{QueryEnv, Session};

/// The level and row limits of a loop: `MAX RECURSION LEVEL` over
/// `spark.sql.cteRecursionLevelLimit`, and `spark.sql.cteRecursionRowLimit`
/// (-1: none).
fn limits(env: &QueryEnv, max_level: Option<u32>) -> (u32, usize) {
    let conf = |entry| env.config.get(entry).and_then(ConfigValue::as_int);
    let bound = |v: Option<i64>, default: i64| match v.unwrap_or(default) {
        n if n < 0 => i64::MAX,
        n => n,
    };
    let level = max_level.unwrap_or_else(|| {
        u32::try_from(bound(conf(&CTE_RECURSION_LEVEL_LIMIT), 100)).unwrap_or(u32::MAX)
    });
    let rows =
        usize::try_from(bound(conf(&CTE_RECURSION_ROW_LIMIT), 1_000_000)).unwrap_or(usize::MAX);
    (level, rows)
}

/// A fresh hidden table name. Process-wide: concurrent actions on one
/// analysis share its catalog.
fn hidden_name() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{CACHE_PREFIX}loop{n}")
}

fn loop_schema(output: &[BoundColumn]) -> SchemaRef {
    Arc::new(Schema::new(
        output
            .iter()
            .map(|c| c.data_type.to_arrow_field(c.name.clone(), true))
            .collect::<Vec<_>>(),
    ))
}

/// A loop that is not run: a scan of an empty table of its schema.
fn empty_loop(
    node: &Arc<LogicalPlan>,
    env: &QueryEnv,
    pins: &mut Vec<String>,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let PlanNode::UnionLoop { output, .. } = &node.node else {
        return Err(crate::session::internal("not a recursive CTE".to_owned()));
    };
    let name = hidden_name();
    pins.push(name.clone());
    env.catalog.pin_cached(
        name.clone(),
        Arc::new(MemoryTable::new(loop_schema(output), Vec::new())),
    );
    Ok(scan_of(node.origin, name, output.clone()))
}

/// The first loop (children before parents) that holds no other loop.
fn innermost(p: &Arc<LogicalPlan>, seen: &mut HashSet<usize>) -> Option<Arc<LogicalPlan>> {
    found_in(p, seen, &mut Vec::new()).map(|(node, _)| node)
}

/// As [`innermost`], with the CTE definitions in scope at the loop (those of
/// the enclosing `WITH`s that hold no loop): its rounds run as plans of
/// their own and still read them.
fn found_in(
    p: &Arc<LogicalPlan>,
    seen: &mut HashSet<usize>,
    defs: &mut Vec<krishiv_plan::Cte>,
) -> Option<(Arc<LogicalPlan>, Vec<krishiv_plan::Cte>)> {
    if !seen.insert(Arc::as_ptr(p) as usize) {
        return None;
    }
    let before = defs.len();
    if let PlanNode::WithCte { ctes, .. } = &p.node {
        defs.extend(ctes.iter().filter(|c| !has_loop(&c.plan)).cloned());
    }
    let mut kids = Vec::new();
    p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
    p.for_each_expr(&mut |e| subquery_plans(e, &mut kids));
    for k in &kids {
        if let Some(found) = found_in(k, seen, defs) {
            return Some(found);
        }
    }
    // A loop that reads the current round of an enclosing one waits for it:
    // that round's rows run it (`Session::run_rounds`).
    let here = (matches!(p.node, PlanNode::UnionLoop { .. }) && free_refs(p, defs).is_empty())
        .then(|| (Arc::clone(p), defs.clone()));
    defs.truncate(before);
    here
}

/// The plans of the subqueries inside `e`.
fn subquery_plans(e: &Arc<krishiv_plan::Expr>, out: &mut Vec<Arc<LogicalPlan>>) {
    if let krishiv_plan::ExprNode::Subquery { plan, .. } = &e.node {
        out.push(Arc::clone(plan));
    }
    e.for_each_child(&mut |c| subquery_plans(c, out));
}

/// `plan` with every relation `f` rewrites replaced, in subquery plans too.
fn rewrite(
    plan: Arc<LogicalPlan>,
    f: &mut dyn FnMut(Arc<LogicalPlan>) -> Arc<LogicalPlan>,
) -> Arc<LogicalPlan> {
    fn in_expr(
        e: Arc<krishiv_plan::Expr>,
        f: &mut dyn FnMut(Arc<LogicalPlan>) -> Arc<LogicalPlan>,
    ) -> Arc<krishiv_plan::Expr> {
        let e = e
            .map_children(|c| Ok::<_, std::convert::Infallible>(in_expr(c, f)))
            .unwrap_or_else(|never| match never {});
        if let krishiv_plan::ExprNode::Subquery { kind, plan } = &e.node {
            let replaced = rewrite(Arc::clone(plan), f);
            if !Arc::ptr_eq(&replaced, plan) {
                return Arc::new(krishiv_plan::Expr {
                    node: krishiv_plan::ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: replaced,
                    },
                    origin: e.origin,
                });
            }
        }
        e
    }
    plan.transform_up(&mut |n| {
        let n = f(n);
        Ok::<_, std::convert::Infallible>(
            n.map_expressions(|e| Ok::<_, std::convert::Infallible>(in_expr(e, f)))
                .unwrap_or_else(|never| match never {}),
        )
    })
    .unwrap_or_else(|never| match never {})
}

/// `plan` with `node` (a loop) replaced by `scan`.
fn replace_loop(
    plan: Arc<LogicalPlan>,
    node: &Arc<LogicalPlan>,
    scan: &Arc<LogicalPlan>,
) -> Arc<LogicalPlan> {
    rewrite(plan, &mut |n| {
        if Arc::ptr_eq(&n, node) {
            Arc::clone(scan)
        } else {
            n
        }
    })
}

/// The loop references `p` reads that no loop inside `p` defines: the
/// rounds of enclosing loops. A read of a CTE counts as a read of what its
/// definition reads; `scope` holds the definitions around `p`.
fn free_refs(p: &Arc<LogicalPlan>, scope: &[krishiv_plan::Cte]) -> HashSet<u32> {
    struct Walk {
        defs: Vec<krishiv_plan::Cte>,
        bound: Vec<u32>,
        free: HashSet<u32>,
        seen: HashSet<usize>,
    }
    fn walk(p: &Arc<LogicalPlan>, w: &mut Walk) {
        if !w.seen.insert(Arc::as_ptr(p) as usize) {
            return;
        }
        let before = w.defs.len();
        let mut bound_here = false;
        match &p.node {
            PlanNode::UnionLoopRef { id, .. } if !w.bound.contains(id) => {
                w.free.insert(*id);
            }
            PlanNode::UnionLoop { id, .. } => {
                w.bound.push(*id);
                bound_here = true;
            }
            PlanNode::CteRef(r) => {
                if let Some(i) = w
                    .defs
                    .iter()
                    .rposition(|c| c.name.eq_ignore_ascii_case(&r.name))
                {
                    let def = Arc::clone(&w.defs[i].plan);
                    for id in free_refs(&def, &w.defs[..i]) {
                        if !w.bound.contains(&id) {
                            w.free.insert(id);
                        }
                    }
                }
            }
            PlanNode::WithCte { ctes, .. } => w.defs.extend(ctes.iter().cloned()),
            _ => {}
        }
        let mut kids = Vec::new();
        p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
        p.for_each_expr(&mut |e| subquery_plans(e, &mut kids));
        for k in &kids {
            walk(k, w);
        }
        w.defs.truncate(before);
        if bound_here {
            w.bound.pop();
        }
    }
    let mut w = Walk {
        defs: scope.to_vec(),
        bound: Vec::new(),
        free: HashSet::new(),
        seen: HashSet::new(),
    };
    walk(p, &mut w);
    w.free
}

/// Whether `p` holds a recursive CTE.
pub(crate) fn has_loop(p: &Arc<LogicalPlan>) -> bool {
    innermost(p, &mut HashSet::new()).is_some()
}

/// `node` replaced by a scan of the hidden table `name`.
fn scan_of(
    origin: krishiv_plan::Origin,
    name: String,
    output: Vec<BoundColumn>,
) -> Arc<LogicalPlan> {
    Arc::new(
        LogicalPlan::new(PlanNode::Scan(Scan {
            source: ReadSource::Table {
                name: vec![name],
                options: BTreeMap::new(),
            },
            output,
            version: None,
        }))
        .with_origin(origin),
    )
}

/// The row count each loop needs when every read of its CTE sits under a
/// `LIMIT n` (through projections and aliases only): Spark pushes the
/// largest into the loop, so an unbounded recursion under `LIMIT n` ends and
/// `LIMIT ALL` (here `i32::MAX`) lifts the row limit. A loop with a read
/// outside any limit (another recursive CTE's included) is absent. Reads in
/// the loop's own recursive term are not outside readers: there the name is
/// defined as the loop's reference, which shadows the CTE.
fn pushed_limits(root: &Arc<LogicalPlan>) -> HashMap<u32, usize> {
    /// Per loop id: the largest limit over its reads, `None` once one read
    /// has none.
    type Reads = HashMap<u32, Option<usize>>;

    fn through(p: &Arc<LogicalPlan>) -> &Arc<LogicalPlan> {
        let mut d = p;
        while let PlanNode::SubqueryAlias { input, .. } | PlanNode::Project { input, .. } = &d.node
        {
            d = input;
        }
        d
    }

    fn plan(
        p: &Arc<LogicalPlan>,
        limit: Option<usize>,
        defs: &mut Vec<(String, Option<u32>)>,
        reads: &mut Reads,
        seen: &mut HashSet<(usize, Option<usize>)>,
    ) {
        if !seen.insert((Arc::as_ptr(p) as usize, limit)) {
            return;
        }
        match &p.node {
            PlanNode::CteRef(r) => {
                let def = defs
                    .iter()
                    .rev()
                    .find(|(n, _)| n.eq_ignore_ascii_case(&r.name));
                if let Some((_, Some(id))) = def {
                    reads
                        .entry(*id)
                        .and_modify(|m| *m = m.zip(limit).map(|(a, b)| a.max(b)))
                        .or_insert(limit);
                }
            }
            PlanNode::WithCte { input, ctes } => {
                let before = defs.len();
                for c in ctes {
                    plan(&c.plan, None, defs, reads, seen);
                    let id = match &through(&c.plan).node {
                        PlanNode::UnionLoop { id, .. } => Some(*id),
                        _ => None,
                    };
                    defs.push((c.name.clone(), id));
                }
                plan(input, limit, defs, reads, seen);
                defs.truncate(before);
            }
            PlanNode::Limit { input, limit: l } => {
                let n = match &l.node {
                    krishiv_plan::ExprNode::Literal(krishiv_plan::Literal::Integer(n)) => {
                        usize::try_from(*n).ok()
                    }
                    _ => None,
                };
                let next = match (n, limit) {
                    (Some(n), Some(m)) => Some(n.min(m)),
                    (Some(n), None) => Some(n),
                    (None, m) => m,
                };
                plan(input, next, defs, reads, seen);
                p.for_each_expr(&mut |e| expr(e, defs, reads, seen));
            }
            PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } => {
                plan(input, limit, defs, reads, seen);
                p.for_each_expr(&mut |e| expr(e, defs, reads, seen));
            }
            _ => {
                p.for_each_child(&mut |c| plan(c, None, defs, reads, seen));
                p.for_each_expr(&mut |e| expr(e, defs, reads, seen));
            }
        }
    }

    fn expr(
        e: &Arc<krishiv_plan::Expr>,
        defs: &mut Vec<(String, Option<u32>)>,
        reads: &mut Reads,
        seen: &mut HashSet<(usize, Option<usize>)>,
    ) {
        if let krishiv_plan::ExprNode::Subquery { plan: p, .. } = &e.node {
            plan(p, None, defs, reads, seen);
        }
        e.for_each_child(&mut |c| expr(c, defs, reads, seen));
    }

    let mut reads = Reads::new();
    plan(root, None, &mut Vec::new(), &mut reads, &mut HashSet::new());
    reads
        .into_iter()
        .filter_map(|(id, n)| n.map(|n| (id, n)))
        .collect()
}

impl Session {
    /// `analyzed` with every recursive CTE evaluated and replaced by a scan
    /// of its rows.
    ///
    /// Returns the hidden tables the result scans: the caller unpins them
    /// (`QueryCatalog::unpin_cached`) once the physical plan has resolved
    /// them. With `run` false (a plan only inspected: EXPLAIN, estimates) no
    /// loop runs and each is replaced by a scan of an empty table of its
    /// schema.
    pub(crate) async fn expand_loops(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        run: bool,
    ) -> Result<(AnalyzedPlan, Vec<String>), KrishivError> {
        let mut pins = Vec::new();
        match self
            .expand_pinning(analyzed, &[], env, run, &mut pins)
            .await
        {
            Ok(expanded) => Ok((expanded, pins)),
            Err(e) => {
                env.catalog.unpin_cached(&pins);
                Err(e)
            }
        }
    }

    /// `outer`: the CTE definitions in scope around `analyzed`'s plan.
    async fn expand_pinning(
        &self,
        analyzed: &AnalyzedPlan,
        outer: &[krishiv_plan::Cte],
        env: &QueryEnv,
        run: bool,
        pins: &mut Vec<String>,
    ) -> Result<AnalyzedPlan, KrishivError> {
        let mut plan = Arc::clone(analyzed.plan.root());
        let limit = pushed_limits(&plan);
        while let Some((node, defs)) = found_in(&plan, &mut HashSet::new(), &mut outer.to_vec()) {
            let scan = if run {
                self.run_loop(&node, &defs, analyzed, env, pins, &limit)
                    .await?
            } else {
                empty_loop(&node, env, pins)?
            };
            plan = replace_loop(Arc::clone(&plan), &node, &scan);
        }
        Ok(AnalyzedPlan {
            plan: Plan::try_new(plan)?,
            ..analyzed.clone()
        })
    }

    async fn run_plan(
        &self,
        plan: Arc<LogicalPlan>,
        defs: &[krishiv_plan::Cte],
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        let plan = if defs.is_empty() {
            plan
        } else {
            Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: plan,
                ctes: defs.to_vec(),
            }))
        };
        let sub = AnalyzedPlan {
            plan: Plan::try_new(plan)?,
            ..analyzed.clone()
        };
        let (_, physical) = Box::pin(self.physical(&sub, env)).await?;
        let mut stream = self.run(&physical, env).await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            out.push(batch?);
        }
        Ok(out)
    }

    async fn run_loop(
        &self,
        node: &Arc<LogicalPlan>,
        defs: &[krishiv_plan::Cte],
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        pins: &mut Vec<String>,
        limit: &HashMap<u32, usize>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let round_name = hidden_name();
        let result = self
            .run_rounds(node, defs, analyzed, env, limit, &round_name, pins)
            .await;
        // The rounds' table is only read while the loop runs.
        env.catalog.unpin_cached(std::slice::from_ref(&round_name));
        result
    }

    /// One round: `next` with its nested loops expanded, then run.
    async fn run_round(
        &self,
        next: Arc<LogicalPlan>,
        defs: &[krishiv_plan::Cte],
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        pins: &mut Vec<String>,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        let next = if has_loop(&next) {
            let sub = AnalyzedPlan {
                plan: Plan::try_new(next)?,
                ..analyzed.clone()
            };
            let expanded = Box::pin(self.expand_pinning(&sub, defs, env, true, pins)).await?;
            Arc::clone(expanded.plan.root())
        } else {
            next
        };
        self.run_plan(next, defs, analyzed, env).await
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one loop run: its plan, scope, limits and pins"
    )]
    async fn run_rounds(
        &self,
        node: &Arc<LogicalPlan>,
        defs: &[krishiv_plan::Cte],
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        limit: &HashMap<u32, usize>,
        round_name: &str,
        pins: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let PlanNode::UnionLoop {
            id,
            anchor,
            recursive,
            max_level,
            output,
        } = &node.node
        else {
            return Err(crate::session::internal("not a recursive CTE".to_owned()));
        };
        let schema = loop_schema(output);
        let (level_limit, row_limit) = limits(env, *max_level);
        let mut held = self.recursion_reservation();
        let mut last = self
            .run_plan(Arc::clone(anchor), defs, analyzed, env)
            .await?;
        let mut all: Vec<RecordBatch> = Vec::new();
        let (mut total, mut level) = (0_usize, 0_u32);
        loop {
            let rows: usize = last.iter().map(RecordBatch::num_rows).sum();
            if rows == 0 {
                break;
            }
            total += rows;
            let wanted = limit.get(id).copied();
            if total > row_limit && wanted.is_none_or(|n| total - rows >= n) {
                return Err(RecursionRowLimitExceeded {
                    row_limit: row_limit.to_string(),
                }
                .into());
            }
            let fixed = last
                .iter()
                .map(|b| RecordBatch::try_new(Arc::clone(&schema), b.columns().to_vec()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| crate::session::internal(format!("a recursive round: {e}")))?;
            // The rows kept for the result count against the memory pool, so
            // `krishiv.memory.limit` bounds a recursion (it does not spill).
            held.try_grow(fixed.iter().map(RecordBatch::get_array_memory_size).sum())?;
            all.extend(fixed.iter().cloned());
            if wanted.is_some_and(|n| total >= n) {
                break;
            }
            level += 1;
            if level > level_limit {
                return Err(RecursionLevelLimitExceeded {
                    level_limit: level_limit.to_string(),
                }
                .into());
            }
            let table = MemoryTable::new(
                Arc::clone(&schema),
                fixed.into_iter().map(Batch::new).collect(),
            );
            env.catalog
                .pin_cached(round_name.to_owned(), Arc::new(table));
            let name = round_name.to_owned();
            let next = rewrite(Arc::clone(recursive), &mut |n| match &n.node {
                PlanNode::UnionLoopRef { id: r, output } if r == id => {
                    scan_of(n.origin, name.clone(), output.clone())
                }
                _ => n,
            });
            // Loops inside the term that read this round run now.
            let mut round_pins = Vec::new();
            let result = self
                .run_round(next, defs, analyzed, env, &mut round_pins)
                .await;
            env.catalog.unpin_cached(&round_pins);
            last = result?;
        }
        let result = hidden_name();
        pins.push(result.clone());
        env.catalog.pin_cached(
            result.clone(),
            Arc::new(MemoryTable::new(
                schema,
                all.into_iter().map(Batch::new).collect(),
            )),
        );
        // The rows stay accounted while the table is pinned.
        env.catalog.hold_reservation(result.clone(), held);
        Ok(scan_of(node.origin, result, output.clone()))
    }
}
