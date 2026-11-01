// SPDX-License-Identifier: Apache-2.0
//! Recursive CTEs: [`PlanNode::UnionLoop`] runs here, before physical
//! planning (the coordinator-driven loop of the T180 design).
//!
//! Each loop, innermost first, runs its anchor, then its recursive term
//! again and again with [`PlanNode::UnionLoopRef`] reading the previous
//! round's rows, until a round is empty. Every round is an ordinary query
//! run: its rows become an in-memory table scanned under a hidden name (the
//! relation cache's mechanism), and the loop's node is replaced by a scan of
//! the rows of all rounds. The level and row limits are Spark's defaults
//! (`spark.sql.cteRecursionLevelLimit` 100, `cteRecursionRowLimit`
//! 1,000,000); `MAX RECURSION LEVEL n` overrides the first.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::{KrishivError, RecursionLevelLimitExceeded, RecursionRowLimitExceeded};
use krishiv_exec::planner::MemoryTable;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{BoundColumn, LogicalPlan, Plan, PlanNode, ReadSource, Scan};
use krishiv_types::batch::Batch;

use crate::cache::CACHE_PREFIX;
use crate::session::{QueryEnv, Session};

/// `spark.sql.cteRecursionLevelLimit`.
const LEVEL_LIMIT: u32 = 100;
/// `spark.sql.cteRecursionRowLimit`.
const ROW_LIMIT: usize = 1_000_000;

/// The first loop (children before parents) that holds no other loop.
fn innermost(p: &Arc<LogicalPlan>, seen: &mut HashSet<usize>) -> Option<Arc<LogicalPlan>> {
    if !seen.insert(Arc::as_ptr(p) as usize) {
        return None;
    }
    let mut kids = Vec::new();
    p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
    for k in &kids {
        if let Some(found) = innermost(k, seen) {
            return Some(found);
        }
    }
    matches!(p.node, PlanNode::UnionLoop { .. }).then(|| Arc::clone(p))
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

/// The row count an outer `LIMIT` over the loop `id` needs: Spark pushes it
/// into the loop, so an unbounded recursion under `LIMIT n` ends. Found
/// only when the query is `LIMIT n` over projections of one reference to
/// the CTE whose definition is the loop.
fn pushed_limit(root: &Arc<LogicalPlan>) -> Option<(u32, usize)> {
    let mut p = root;
    let mut limit: Option<usize> = None;
    let mut defs: Vec<&krishiv_plan::Cte> = Vec::new();
    loop {
        match &p.node {
            PlanNode::Limit { input, limit: l } => {
                if let krishiv_plan::ExprNode::Literal(krishiv_plan::Literal::Integer(n)) = &l.node
                {
                    let n = usize::try_from(*n).ok()?;
                    limit = Some(limit.map_or(n, |m| m.min(n)));
                }
                p = input;
            }
            PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } => p = input,
            PlanNode::WithCte { input, ctes } => {
                defs.extend(ctes.iter());
                p = input;
            }
            PlanNode::CteRef(r) => {
                let def = defs
                    .iter()
                    .rev()
                    .find(|c| c.name.eq_ignore_ascii_case(&r.name))?;
                let mut d = &def.plan;
                while let PlanNode::SubqueryAlias { input, .. } | PlanNode::Project { input, .. } =
                    &d.node
                {
                    d = input;
                }
                let PlanNode::UnionLoop { id, .. } = &d.node else {
                    return None;
                };
                return Some((*id, limit?));
            }
            _ => return None,
        }
    }
}

impl Session {
    /// `analyzed` with every recursive CTE evaluated and replaced by a scan
    /// of its rows.
    pub(crate) async fn expand_loops(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<AnalyzedPlan, KrishivError> {
        let mut plan = Arc::clone(analyzed.plan.root());
        let mut tables = 0_u32;
        let limit = pushed_limit(&plan);
        while let Some(node) = innermost(&plan, &mut HashSet::new()) {
            let scan = self
                .run_loop(&node, analyzed, env, &mut tables, limit)
                .await?;
            plan = Arc::clone(&plan)
                .transform_up(&mut |n| {
                    Ok::<_, std::convert::Infallible>(if Arc::ptr_eq(&n, &node) {
                        Arc::clone(&scan)
                    } else {
                        n
                    })
                })
                .unwrap_or(plan);
        }
        Ok(AnalyzedPlan {
            plan: Plan::try_new(plan)?,
            ..analyzed.clone()
        })
    }

    async fn run_plan(
        &self,
        plan: Arc<LogicalPlan>,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
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
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        tables: &mut u32,
        limit: Option<(u32, usize)>,
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
        let schema: SchemaRef = Arc::new(Schema::new(
            output
                .iter()
                .map(|c| c.data_type.to_arrow_field(c.name.clone(), true))
                .collect::<Vec<_>>(),
        ));
        let level_limit = max_level.unwrap_or(LEVEL_LIMIT);
        let hidden = |tables: &mut u32| {
            *tables += 1;
            format!("{CACHE_PREFIX}loop{tables}")
        };
        let round_name = hidden(tables);
        let mut last = self.run_plan(Arc::clone(anchor), analyzed, env).await?;
        let mut all: Vec<RecordBatch> = Vec::new();
        let (mut total, mut level) = (0_usize, 0_u32);
        loop {
            let rows: usize = last.iter().map(RecordBatch::num_rows).sum();
            if rows == 0 {
                break;
            }
            total += rows;
            let wanted = limit.filter(|(i, _)| i == id).map(|(_, n)| n);
            if total > ROW_LIMIT && wanted.is_none_or(|n| total - rows >= n) {
                return Err(RecursionRowLimitExceeded {
                    row_limit: ROW_LIMIT.to_string(),
                }
                .into());
            }
            let fixed = last
                .iter()
                .map(|b| RecordBatch::try_new(Arc::clone(&schema), b.columns().to_vec()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| crate::session::internal(format!("a recursive round: {e}")))?;
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
            env.catalog.pin_cached(round_name.clone(), Arc::new(table));
            let name = round_name.clone();
            let next = Arc::clone(recursive)
                .transform_up(&mut |n| {
                    Ok::<_, std::convert::Infallible>(match &n.node {
                        PlanNode::UnionLoopRef { id: r, output } if r == id => {
                            scan_of(n.origin, name.clone(), output.clone())
                        }
                        _ => n,
                    })
                })
                .unwrap_or_else(|_| Arc::clone(recursive));
            last = self.run_plan(next, analyzed, env).await?;
        }
        let result = hidden(tables);
        env.catalog.pin_cached(
            result.clone(),
            Arc::new(MemoryTable::new(
                schema,
                all.into_iter().map(Batch::new).collect(),
            )),
        );
        Ok(scan_of(node.origin, result, output.clone()))
    }
}
