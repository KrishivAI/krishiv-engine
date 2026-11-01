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

use std::rc::Rc;
use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::resolve::{Attr, Env, Rel, Resolver};
use std::collections::{HashMap, HashSet};

use krishiv_common::error::{CannotMergeIncompatibleDataType, InvalidRecursiveReferencePlace};
use krishiv_types::DataType;

use super::coerce::quoted_type;

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
        // The reference was typed by the anchor, so rows of another type
        // cannot feed back. Spark refuses a recursive term that is not the
        // anchor's type (`CANNOT_MERGE_INCOMPATIBLE_DATA_TYPE`); a term of a
        // narrower numeric type it casts to the anchor's, which is not done
        // here yet.
        for (term, anchor) in kids[1].out.iter().zip(&refs) {
            let (t, a) = (&term.col.data_type, &anchor.col.data_type);
            if t == a {
                continue;
            }
            if numeric_rank(t)
                .zip(numeric_rank(a))
                .is_some_and(|(t, a)| t < a)
            {
                return Err(unsupported(
                    "a recursive term whose column is narrower than the anchor's",
                    o,
                ));
            }
            return Err(AnalysisError::new(
                CannotMergeIncompatibleDataType {
                    left: quoted_type(Some(a)),
                    right: quoted_type(Some(t)),
                },
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

/// The position of a numeric type in the widening order, `None` for others.
fn numeric_rank(t: &DataType) -> Option<u8> {
    Some(match t {
        DataType::Byte => 0,
        DataType::Short => 1,
        DataType::Integer => 2,
        DataType::Long => 3,
        DataType::Decimal(_) => 4,
        DataType::Float => 5,
        DataType::Double => 6,
        _ => return None,
    })
}

/// The CTE definitions in scope at a node of [`check_places`]'s walk,
/// innermost first: a read of a CTE counts as a read of what its definition
/// reads.
struct Scope {
    name: String,
    plan: Arc<LogicalPlan>,
    /// The definitions in scope of this one's own body.
    outer: Option<Rc<Scope>>,
}

type Defs = Option<Rc<Scope>>;

/// The definition `name` resolves to in `defs`, with the scope its body sees.
fn lookup(defs: &Defs, name: &str) -> Option<(Arc<LogicalPlan>, Defs)> {
    let mut cur = defs.as_ref();
    while let Some(d) = cur {
        if d.name.eq_ignore_ascii_case(name) {
            return Some((Arc::clone(&d.plan), d.outer.clone()));
        }
        cur = d.outer.as_ref();
    }
    None
}

/// Spark's `INVALID_RECURSIVE_REFERENCE.PLACE`: the recursive reference
/// (a `CteRef` of `name` that no definition inside the term answers, or a
/// CTE defined inside the term that reads it) may not sit under an
/// aggregate, a window or a sort, inside a subquery expression, on the right
/// side of a left outer, semi or anti join, the left side of a right outer
/// join, or in a full outer join.
fn check_places(plan: &Arc<LogicalPlan>, name: &str, o: Origin) -> Result<(), AnalysisError> {
    type Memo = HashMap<usize, bool>;
    fn has(p: &Arc<LogicalPlan>, name: &str, defs: &Defs, memo: &mut Memo) -> bool {
        let key = Arc::as_ptr(p) as usize;
        if let Some(b) = memo.get(&key) {
            return *b;
        }
        let mut found = false;
        match &p.node {
            PlanNode::CteRef(r) => {
                found = match lookup(defs, &r.name) {
                    // A definition inside the term: what it reads.
                    Some((plan, outer)) => has(&plan, name, &outer, memo),
                    // None inside: the recursive reference, if it is the name.
                    None => r.name.eq_ignore_ascii_case(name),
                };
            }
            PlanNode::WithCte { input, ctes } => {
                let mut scope = defs.clone();
                for c in ctes {
                    found |= has(&c.plan, name, &scope, memo);
                    scope = Some(Rc::new(Scope {
                        name: c.name.clone(),
                        plan: Arc::clone(&c.plan),
                        outer: scope,
                    }));
                }
                found |= has(input, name, &scope, memo);
            }
            _ => p.for_each_child(&mut |c| found |= has(c, name, defs, memo)),
        }
        p.for_each_expr(&mut |e| found |= expr_has(e, name, defs, memo));
        memo.insert(key, found);
        found
    }
    fn expr_has(e: &Arc<crate::Expr>, name: &str, defs: &Defs, memo: &mut Memo) -> bool {
        let mut found =
            matches!(&e.node, ExprNode::Subquery { plan, .. } if has(plan, name, defs, memo));
        e.for_each_child(&mut |c| found |= expr_has(c, name, defs, memo));
        found
    }
    let mut memo = HashMap::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<(Arc<LogicalPlan>, Defs)> = vec![(Arc::clone(plan), None)];
    let bad = || AnalysisError::new(InvalidRecursiveReferencePlace {}, o);
    while let Some((p, defs)) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p) as usize) {
            continue;
        }
        let mut in_expr = false;
        p.for_each_expr(&mut |e| in_expr |= expr_has(e, name, &defs, &mut memo));
        if in_expr {
            return Err(bad());
        }
        match &p.node {
            PlanNode::Aggregate { input, .. }
            | PlanNode::Window { input, .. }
            | PlanNode::Sort { input, .. }
                if has(input, name, &defs, &mut memo) =>
            {
                return Err(bad());
            }
            PlanNode::Join {
                left, right, kind, ..
            } => {
                let (l, r) = (
                    has(left, name, &defs, &mut memo),
                    has(right, name, &defs, &mut memo),
                );
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
            let mut scope = defs.clone();
            for c in ctes {
                stack.push((Arc::clone(&c.plan), scope.clone()));
                scope = Some(Rc::new(Scope {
                    name: c.name.clone(),
                    plan: Arc::clone(&c.plan),
                    outer: scope,
                }));
            }
            stack.push((Arc::clone(input), scope));
        } else {
            p.for_each_child(&mut |c| stack.push((Arc::clone(c), defs.clone())));
        }
    }
    Ok(())
}
