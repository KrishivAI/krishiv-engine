// SPDX-License-Identifier: Apache-2.0
//! Recursive common table expressions as [`PlanNode::UnionLoop`] relations
//! (Spark's `UnionLoop`).
//!
//! Lowering builds `UnionLoop { anchor, recursive }` where `recursive` is a
//! `WithCte` defining the CTE's name as a [`PlanNode::UnionLoopRef`], so
//! every name-scope walker treats the name as a definition inside the
//! recursive term. Here the anchor resolves first; the reference then takes
//! the anchor's columns (all nullable), the recursive term resolves over
//! it, and the two are widened as `UNION ALL` inputs are.

use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::resolve::{Attr, Env, Rel, Resolver};
use std::collections::{HashMap, HashSet};

use krishiv_common::error::InvalidRecursiveReferencePlace;

use crate::{BoundColumn, ExprNode, JoinKind, LogicalPlan, Origin, PlanNode};

impl Resolver<'_> {
    pub(super) fn union_loop(
        &mut self,
        p: &Arc<LogicalPlan>,
        id: u32,
        anchor: &Arc<LogicalPlan>,
        recursive: &Arc<LogicalPlan>,
        max_level: Option<u32>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let a = self.rel(anchor, env)?;
        let refs: Vec<Attr> = a
            .out
            .iter()
            .map(|x| Attr {
                col: BoundColumn {
                    id: self.fresh(),
                    nullable: true,
                    ..x.col.clone()
                },
                qual: Arc::from(Vec::new()),
                source: None,
            })
            .collect();
        self.loops.push((id, refs.clone()));
        let r = self.rel(recursive, env);
        self.loops.pop();
        let r = r?;
        // From inside the wrapper: its own definition of the name is the
        // reference being checked, not a shadow of it.
        if let PlanNode::WithCte { input, ctes } = &r.plan.node
            && let [def] = ctes.as_slice()
        {
            check_places(input, &def.name, o)?;
        }
        let kids = self.widen_inputs(&[a, r], "UNION", o)?;
        // The reference was typed by the anchor: a recursive term whose
        // columns widen the anchor's would feed rows of another type back.
        if kids[1]
            .out
            .iter()
            .zip(&refs)
            .any(|(x, y)| x.col.data_type != y.col.data_type)
        {
            return Err(unsupported(
                "a recursive term whose column types differ from the anchor's",
                o,
            ));
        }
        let mut out: Vec<Attr> = kids[0].out.to_vec();
        for (a, b) in out.iter_mut().zip(kids[1].out.iter()) {
            a.col.nullable |= b.col.nullable;
        }
        let node = PlanNode::UnionLoop {
            id,
            anchor: Arc::clone(&kids[0].plan),
            recursive: Arc::clone(&kids[1].plan),
            max_level,
            output: out.iter().map(|a| a.col.clone()).collect(),
        };
        let kids: Vec<&Rel> = kids.iter().collect();
        Ok(self.make(o, node, out, &kids))
    }

    pub(super) fn loop_ref(&mut self, p: &Arc<LogicalPlan>, id: u32) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let Some((_, refs)) = self.loops.iter().rev().find(|(i, _)| *i == id) else {
            return Err(unsupported("a recursive reference outside its CTE", o));
        };
        let out = refs.clone();
        let node = PlanNode::UnionLoopRef {
            id,
            output: out.iter().map(|a| a.col.clone()).collect(),
        };
        Ok(self.make(o, node, out, &[]))
    }
}

/// Spark's `INVALID_RECURSIVE_REFERENCE.PLACE`: the recursive reference
/// (a `CteRef` over `ids`) may not sit under an aggregate, a window or a
/// sort, inside a subquery expression, on the right side of a left outer,
/// semi or anti join, the left side of a right outer join, or in a full
/// outer join.
fn check_places(plan: &Arc<LogicalPlan>, name: &str, o: Origin) -> Result<(), AnalysisError> {
    fn has(p: &Arc<LogicalPlan>, name: &str, memo: &mut HashMap<usize, bool>) -> bool {
        let key = Arc::as_ptr(p) as usize;
        if let Some(b) = memo.get(&key) {
            return *b;
        }
        let mut found = matches!(&p.node, PlanNode::CteRef(r) if r.name.eq_ignore_ascii_case(name));
        if let PlanNode::WithCte { input, ctes } = &p.node {
            // A definition of `name` hides the recursive CTE from what
            // follows it: the later definitions and the body.
            let mut hidden = false;
            for c in ctes {
                found |= !hidden && has(&c.plan, name, memo);
                hidden |= c.name.eq_ignore_ascii_case(name);
            }
            found |= !hidden && has(input, name, memo);
        } else {
            p.for_each_child(&mut |c| found |= has(c, name, memo));
        }
        p.for_each_expr(&mut |e| found |= expr_has(e, name, memo));
        memo.insert(key, found);
        found
    }
    fn expr_has(e: &Arc<crate::Expr>, name: &str, memo: &mut HashMap<usize, bool>) -> bool {
        let mut found = matches!(&e.node, ExprNode::Subquery { plan, .. } if has(plan, name, memo));
        e.for_each_child(&mut |c| found |= expr_has(c, name, memo));
        found
    }
    let mut memo = HashMap::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    let bad = || AnalysisError::new(InvalidRecursiveReferencePlace {}, o);
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p) as usize) {
            continue;
        }
        let mut in_expr = false;
        p.for_each_expr(&mut |e| in_expr |= expr_has(e, name, &mut memo));
        if in_expr {
            return Err(bad());
        }
        match &p.node {
            PlanNode::Aggregate { input, .. }
            | PlanNode::Window { input, .. }
            | PlanNode::Sort { input, .. }
                if has(input, name, &mut memo) =>
            {
                return Err(bad());
            }
            PlanNode::Join {
                left, right, kind, ..
            } => {
                let (l, r) = (has(left, name, &mut memo), has(right, name, &mut memo));
                let wrong = match kind {
                    JoinKind::FullOuter => l || r,
                    JoinKind::RightOuter => l,
                    JoinKind::LeftOuter | JoinKind::LeftSemi | JoinKind::LeftAnti => r,
                    _ => false,
                };
                if wrong {
                    return Err(bad());
                }
            }
            _ => {}
        }
        if let PlanNode::WithCte { input, ctes } = &p.node {
            // What a redefinition of `name` hides is not the recursive
            // reference's scope.
            let mut hidden = false;
            for c in ctes {
                if !hidden {
                    stack.push(Arc::clone(&c.plan));
                }
                hidden |= c.name.eq_ignore_ascii_case(name);
            }
            if !hidden {
                stack.push(Arc::clone(input));
            }
        } else {
            p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        }
    }
    Ok(())
}
