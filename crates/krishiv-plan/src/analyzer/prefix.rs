// SPDX-License-Identifier: Apache-2.0
//! The analysis prefix cache (design P3-T013a, PR-35a): a client that grows a plan one step at a time
//! (`withColumn`, then `df.columns`, again and again) resends the whole relation tree, whose prefix is
//! structurally equal to the one analysed the call before. The resolution of that prefix (the "last
//! analysed root" of a session) is kept, and the next analysis starts from it.
//!
//! A warm analysis equals the cold one exactly, ids included. That holds because the prefix is the first
//! relation the resolver finishes (the first-child spine of the new root), starts at the same first free id,
//! and reads nothing but tables Bind found unchanged. Anything else is a miss. Authorization is never served
//! from here: Bind asks the policy for every securable on every call, before Resolve runs.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use super::bind::{BoundTable, addr};
use super::catalog::TableMeta;
use super::resolve::{Env, Rel, Resolver};
use super::{ContextFingerprint, Privilege, Securable};
use crate::{AttrId, LogicalPlan, PlanNode};

/// The resolution of an analysed root, and what a later analysis must find unchanged to reuse it.
#[derive(Clone)]
pub(super) struct PrefixEntry {
    /// The context the entry was made in (set by the analyzer, which knows the catalog epochs).
    pub(super) fingerprint: Option<ContextFingerprint>,
    /// The identity key of the root.
    key: u128,
    /// The input root itself: a later prefix must equal it, origins included (an identity key leaves
    /// out origins, and a resolved plan keeps the SQL spans and view source ids of its input).
    input: Arc<LogicalPlan>,
    /// The resolution of every relation under the root in the root's environment, and of every
    /// relation without outer references in the root's CTE scope, by identity key: a later step may look
    /// below the prefix root (QUALIFY reads a grandchild) or read a relation equal to one inside it (a
    /// self-join instance), and finds what a cold run finds.
    memo: HashMap<u128, Rel>,
    closed: HashMap<u128, Rel>,
    /// The first free id when the root was resolved, and the next one after.
    start_id: u64,
    end_id: u64,
    declared: HashMap<AttrId, Arc<str>>,
    comments: HashMap<AttrId, Arc<str>>,
    pass_exprs: HashMap<AttrId, Arc<crate::Expr>>,
    /// The securables Bind found, in order, and the columns the root read of each.
    securables: Vec<(Securable, Privilege)>,
    columns: Vec<BTreeSet<String>>,
    /// The tables the root reads: as written, the metadata resolved, and the securable index.
    tables: Vec<(Vec<String>, Arc<TableMeta>, usize)>,
    sql_calls: usize,
    /// The observation names the root registered (`Dataset.observe`), with the identity key of the
    /// relation that holds each: a warm run must still refuse a second relation using the name.
    observed: Vec<(String, u128)>,
}

impl Resolver<'_> {
    /// The entry for the root just resolved, if it can serve a later analysis.
    pub(super) fn export_prefix(
        &self,
        root: &Arc<LogicalPlan>,
        start_id: u64,
    ) -> Option<PrefixEntry> {
        if self.uncacheable
            || self.b.reads_files
            || !self.ctx.variables.locals.is_empty()
            || self.pending_nondeterministic.is_some()
            || self.sql_calls != 0
        {
            return None;
        }
        let key = *self.keys.get(&addr(&**root))?;
        let env = self.root_env.as_ref()?;
        let by_key = |m: &HashMap<(usize, usize), Rel>, scope: usize| -> HashMap<u128, Rel> {
            m.iter()
                .filter(|((_, s), _)| *s == scope)
                .filter_map(|((a, _), r)| self.keys.get(a).map(|k| (*k, r.clone())))
                .collect()
        };
        let memo = by_key(&self.memo, env.id);
        let closed = by_key(&self.closed, env.ctes_id);
        let mut tables = Vec::new();
        let mut seen = HashSet::new();
        for name in &self.read_tables {
            if !seen.insert(name) {
                continue;
            }
            let Some(BoundTable::Found { meta, securable }) = self.b.tables.get(name) else {
                return None;
            };
            tables.push((name.clone(), Arc::clone(meta), *securable));
        }
        Some(PrefixEntry {
            fingerprint: None,
            key,
            input: Arc::clone(root),
            memo,
            closed,
            start_id,
            end_id: self.next_id,
            declared: self.declared.clone(),
            comments: self.comments.clone(),
            pass_exprs: self.pass_exprs.clone(),
            securables: self.b.securables.clone(),
            columns: self.columns.clone(),
            tables,
            sql_calls: self.sql_calls,
            observed: self
                .observed
                .iter()
                .filter_map(|(name, a)| Some((name.clone(), *self.keys.get(a)?)))
                .collect(),
        })
    }

    /// Starts this resolution from `entry` when the input is that root plus steps above it and nothing the
    /// root read has moved; otherwise leaves the resolver untouched (a miss).
    pub(super) fn seed_prefix(
        &mut self,
        entry: &PrefixEntry,
        root: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) {
        let Some(prefix) = self.find_prefix(entry, root) else {
            return;
        };
        let b = self.b;
        let same_tables = entry.tables.iter().all(|(name, meta, index)| {
            matches!(b.tables.get(name),
                Some(BoundTable::Found { meta: m, securable }) if securable == index && **m == **meta)
        });
        let n = entry.securables.len();
        if !same_tables
            || b.securables.len() < n
            || b.securables[..n] != entry.securables[..]
            || self.next_id != entry.start_id
            || self.uncacheable
            || b.reads_files
            || !self.ctx.variables.locals.is_empty()
            || !self.loops.is_empty()
            || !self.lambdas.is_empty()
            || self.frame.is_some()
            || !self.fn_path.is_empty()
            || self.expr_depth != 0
        {
            return;
        }
        // Origins too (spans, view source ids): the resolution keeps the input's.
        if *prefix != *entry.input {
            return;
        }
        let mut nodes = Vec::new();
        let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(&prefix), |n| {
            if let crate::walk::Node::Plan(p) = n {
                nodes.push(addr(p));
            }
            std::ops::ControlFlow::Continue(())
        });
        for a in nodes {
            let (Some(k), Some(rep)) = (self.keys.get(&a), self.canon.get(&a)) else {
                continue;
            };
            let at = addr(&**rep);
            if let Some(r) = entry.memo.get(k) {
                self.memo.insert((at, env.id), r.clone());
            }
            if let Some(r) = entry.closed.get(k) {
                self.closed.insert((at, env.ctes_id), r.clone());
            }
        }
        for (name, key) in &entry.observed {
            let holder = self
                .keys
                .iter()
                .filter(|(_, k)| *k == key)
                .find_map(|(a, _)| self.canon.get(a).map(|rep| addr(&**rep)));
            if let Some(at) = holder {
                self.observed.insert(name.clone(), at);
            }
        }
        self.warm_started = true;
        self.next_id = entry.end_id;
        self.declared = entry.declared.clone();
        self.comments = entry.comments.clone();
        self.pass_exprs = entry.pass_exprs.clone();
        self.sql_calls += entry.sql_calls;
        for (mine, theirs) in self.columns.iter_mut().zip(&entry.columns) {
            mine.clone_from(theirs);
        }
    }

    /// The node of the input that is the entry's root: on the first-child spine from `root`, which the
    /// resolver finishes first, and with no equal relation anywhere else (an equal one would be a memo
    /// hit in a cold run and a fresh instance here).
    fn find_prefix(
        &self,
        entry: &PrefixEntry,
        root: &Arc<LogicalPlan>,
    ) -> Option<Arc<LogicalPlan>> {
        let mut node = Arc::clone(root);
        let prefix = loop {
            if self.keys.get(&addr(&*node)) == Some(&entry.key) {
                break node;
            }
            match &node.node {
                PlanNode::WithCte { .. }
                | PlanNode::UnionLoop { .. }
                | PlanNode::Shift { .. }
                | PlanNode::Join {
                    kind: crate::JoinKind::LateralInner | crate::JoinKind::LateralLeft,
                    ..
                } => return None,
                _ => {}
            }
            let mut first = None;
            node.for_each_child(&mut |c| {
                if first.is_none() {
                    first = Some(Arc::clone(c));
                }
            });
            node = first?;
        };
        let mut inside = HashSet::new();
        let mut inside_keys = HashSet::new();
        let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(&prefix), |n| {
            if let crate::walk::Node::Plan(p) = n {
                inside.insert(addr(p));
                if let Some(k) = self.keys.get(&addr(p)) {
                    inside_keys.insert(*k);
                }
            }
            std::ops::ControlFlow::Continue(())
        });
        let mut clash = false;
        let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(root), |n| {
            if let crate::walk::Node::Plan(p) = n
                && !inside.contains(&addr(p))
                && self
                    .keys
                    .get(&addr(p))
                    .is_some_and(|k| inside_keys.contains(k))
            {
                clash = true;
                return std::ops::ControlFlow::Break(());
            }
            std::ops::ControlFlow::Continue(())
        });
        (!clash).then_some(prefix)
    }
}
