// SPDX-License-Identifier: Apache-2.0
//! The `WITH` definitions of a stored view's text, resolved under the CTE precedence policy the
//! view was created with (P3-T062 review r4 Y1), never the reader's.
//!
//! Spark resolves a view's text under the SQL configs captured when the view was created
//! (`View.effectiveSQLConf(view.desc.viewSQLConfigs, ..)`, v4.2.0 `ViewResolution.scala` lines
//! 46-52): `spark.sql.legacy.ctePrecedencePolicy` is one of them when it was set, and the default
//! (`CORRECTED`) otherwise. [`inline_view_ctes`] applies that policy to the text alone, before the
//! text is placed in the reader's plan, so no pass the reader's policy runs over the whole plan
//! (`LEGACY` inlining, the `EXCEPTION` check) ever meets a definition of the view's.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::{FailedToParseTooComplex, KrishivError};

use super::CtePrecedencePolicy;
use super::bind::same;
use super::legacy::{CtePolicy, eliminate, hop, nested_conflict};
use crate::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource};

/// The text `plan` of a stored view created under `view` (its captured policy, `CORRECTED` when
/// none was captured), to be read by a query under `reader`:
/// * `LEGACY` view: every definition inlined as `LEGACY` does;
/// * `EXCEPTION` view: a nested definition shadowing an outer one is
///   `AMBIGUOUS_ALIAS_IN_NESTED_CTE`, then as `CORRECTED`;
/// * `CORRECTED` view: under a `CORRECTED` reader the text comes back as it is (the reader's
///   analysis resolves its definitions with the same scoping); under any other reader every
///   definition is inlined with the `CORRECTED` scoping (a reference reads the innermost
///   definition in scope), so the reader's whole-plan pass finds none of them.
///
/// Inlined, a text keeps no read of a definition: what is left of its single-part reads are
/// tables, which the caller qualifies to the view's namespace.
///
/// # Errors
/// `AMBIGUOUS_ALIAS_IN_NESTED_CTE` as above; `FAILED_TO_PARSE_TOO_COMPLEX` for a text nested too
/// deep.
pub fn inline_view_ctes(
    plan: &Arc<LogicalPlan>,
    view: CtePrecedencePolicy,
    reader: CtePrecedencePolicy,
    case_sensitive: bool,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let policy = CtePolicy {
        precedence: view,
        case_sensitive,
    };
    match view {
        CtePrecedencePolicy::Legacy => eliminate(plan, policy),
        CtePrecedencePolicy::Exception | CtePrecedencePolicy::Corrected => {
            if view == CtePrecedencePolicy::Exception
                && let Some(conflict) = nested_conflict(plan, policy).map_err(|e| e.error)?
            {
                return Err(conflict.error);
            }
            if reader == CtePrecedencePolicy::Corrected {
                return Ok(Arc::clone(plan));
            }
            let mut walk = Corrected {
                cs: case_sensitive,
                defs: Vec::new(),
                scope: 0,
                next_scope: 0,
                plans: HashMap::new(),
                exprs: HashMap::new(),
            };
            let out = walk.plan(plan, 0)?;
            crate::check_depth(&out)?;
            Ok(out)
        }
    }
}

/// [`inline_view_ctes`] under the `CORRECTED` scoping: `defs` holds the definitions in scope,
/// oldest first (the newest match wins), each already inlined. A node is rewritten once per
/// scope: `scope` names the state of `defs` (a fresh number each time a definition enters it),
/// and the memo is keyed by node address and scope, holding the node so its address is not reused.
struct Corrected {
    cs: bool,
    defs: Vec<(String, Arc<LogicalPlan>)>,
    scope: u64,
    next_scope: u64,
    plans: HashMap<(usize, u64), (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
    exprs: HashMap<(usize, u64), (Arc<Expr>, Arc<Expr>)>,
}

impl Corrected {
    fn plan(
        &mut self,
        plan: &Arc<LogicalPlan>,
        depth: usize,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let key = (Arc::as_ptr(plan) as usize, self.scope);
        if let Some((_, out)) = self.plans.get(&key) {
            return Ok(Arc::clone(out));
        }
        if depth > crate::MAX_DEPTH {
            return Err(FailedToParseTooComplex {}.into());
        }
        let out = hop(depth, || self.plan_body(plan, depth))?;
        self.plans.insert(key, (Arc::clone(plan), Arc::clone(&out)));
        Ok(out)
    }

    fn plan_body(
        &mut self,
        plan: &Arc<LogicalPlan>,
        depth: usize,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        match &plan.node {
            PlanNode::Read(ReadSource::Table { name, .. }) => {
                let found = match name.as_slice() {
                    [one] => self
                        .defs
                        .iter()
                        .rev()
                        .find(|(n, _)| same(n, one, self.cs))
                        .map(|(_, body)| Arc::clone(body)),
                    _ => None,
                };
                Ok(found.unwrap_or_else(|| Arc::clone(plan)))
            }
            PlanNode::WithCte { input, ctes } => {
                let (outer, outer_scope) = (self.defs.len(), self.scope);
                let mut out = Ok(Arc::clone(input));
                for c in ctes {
                    // A definition sees the definitions before it, not itself.
                    match self.plan(&c.plan, depth + 1) {
                        Ok(body) => {
                            self.defs.push((c.name.clone(), body));
                            self.next_scope += 1;
                            self.scope = self.next_scope;
                        }
                        Err(e) => {
                            out = Err(e);
                            break;
                        }
                    }
                }
                if out.is_ok() {
                    out = self.plan(input, depth + 1);
                }
                self.defs.truncate(outer);
                self.scope = outer_scope;
                out
            }
            _ => {
                let rebuilt = Arc::clone(plan).map_children(|c| self.plan(&c, depth + 1))?;
                rebuilt.map_expressions(|e| self.expr(&e, depth + 1))
            }
        }
    }

    fn expr(&mut self, e: &Arc<Expr>, depth: usize) -> Result<Arc<Expr>, KrishivError> {
        let key = (Arc::as_ptr(e) as usize, self.scope);
        if let Some((_, out)) = self.exprs.get(&key) {
            return Ok(Arc::clone(out));
        }
        if depth > crate::MAX_DEPTH {
            return Err(FailedToParseTooComplex {}.into());
        }
        let out = hop(depth, || {
            let e = Arc::clone(e).map_children(|c| self.expr(&c, depth + 1))?;
            if let ExprNode::Subquery { kind, plan } = &e.node {
                // A subquery reads the definitions of the query around it.
                let next = self.plan(plan, depth + 1)?;
                if !Arc::ptr_eq(&next, plan) {
                    return Ok(Arc::new(Expr {
                        node: ExprNode::Subquery {
                            kind: kind.clone(),
                            plan: next,
                        },
                        origin: e.origin,
                    }));
                }
            }
            Ok(e)
        })?;
        self.exprs.insert(key, (Arc::clone(e), Arc::clone(&out)));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::inline_view_ctes;
    use crate::analyzer::CtePrecedencePolicy::{Corrected, Exception, Legacy};
    use crate::{Cte, Expr, LogicalPlan, PlanNode, ReadSource};

    fn with(input: LogicalPlan, ctes: Vec<(&str, LogicalPlan)>) -> LogicalPlan {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(input),
            ctes: ctes
                .into_iter()
                .map(|(name, plan)| Cte {
                    name: name.to_owned(),
                    plan: Arc::new(plan),
                })
                .collect(),
        })
    }

    fn read(name: &str) -> LogicalPlan {
        LogicalPlan::table([name]).project(vec![Expr::column(["a"])])
    }

    /// The table names `plan` reads, and whether a `WITH` is left in it.
    fn reads(plan: &LogicalPlan, names: &mut Vec<String>, with: &mut bool) {
        match &plan.node {
            PlanNode::Read(ReadSource::Table { name, .. }) => names.push(name.join(".")),
            PlanNode::WithCte { .. } => *with = true,
            _ => {}
        }
        plan.for_each_child(&mut |c| reads(c, names, with));
    }

    fn inlined(plan: &Arc<LogicalPlan>, view: crate::analyzer::CtePrecedencePolicy) -> Vec<String> {
        let out = inline_view_ctes(plan, view, Legacy, false).unwrap();
        let (mut names, mut left) = (Vec::new(), false);
        reads(&out, &mut names, &mut left);
        assert!(!left, "{out:?}");
        names
    }

    /// Review r4 Y1: `WITH c AS (SELECT a FROM t), d AS (WITH e AS (SELECT * FROM c) SELECT * FROM
    /// e) SELECT * FROM d`, the view text whose nested definition reads an outer one, keeps no
    /// read of `c` under its own (CORRECTED) policy, whatever the reader's.
    #[test]
    fn a_views_definitions_resolve_under_its_own_policy() {
        let dvc = Arc::new(with(
            read("d"),
            vec![
                ("c", read("t")),
                ("d", with(read("e"), vec![("e", read("c"))])),
            ],
        ));
        assert_eq!(inlined(&dvc, Corrected), ["t"]);
        assert_eq!(inlined(&dvc, Exception), ["t"]);
        // Under a CORRECTED reader the text is left for the reader's own (same) scoping.
        let kept = inline_view_ctes(&dvc, Corrected, Corrected, false).unwrap();
        assert!(Arc::ptr_eq(&kept, &dvc));
        // Under LEGACY the nested definition's body does not see `c`: the read stays a table's.
        assert_eq!(inlined(&dvc, Legacy), ["c"]);

        // `WITH c AS (SELECT a FROM t), d AS (WITH c AS (SELECT a FROM u) SELECT * FROM c) SELECT
        // * FROM d`: the innermost `c` wins under CORRECTED, the outer one under LEGACY, and
        // EXCEPTION refuses the shadowing.
        let shadowing = Arc::new(with(
            read("d"),
            vec![
                ("c", read("t")),
                ("d", with(read("c"), vec![("c", read("u"))])),
            ],
        ));
        assert_eq!(inlined(&shadowing, Corrected), ["u"]);
        assert_eq!(inlined(&shadowing, Legacy), ["t"]);
        let e = inline_view_ctes(&shadowing, Exception, Corrected, false).unwrap_err();
        assert_eq!(e.error_class().name(), "AMBIGUOUS_ALIAS_IN_NESTED_CTE");

        // A definition reading its own name reads the table of that name.
        let own = Arc::new(with(read("t"), vec![("t", read("t"))]));
        assert_eq!(inlined(&own, Corrected), ["t"]);
    }
}
