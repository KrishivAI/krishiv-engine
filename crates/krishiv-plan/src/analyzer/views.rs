// SPDX-License-Identifier: Apache-2.0
//! The view side table (P3-T063, design `P3-T063-analyze-api.md` §4, review m4): permanent and temporary views
//! are inlined before binding, so there is no IR node for a view read. View inlining records each one it
//! inlines ([`ViewExpansion`]), and [`locate_views`] finds where each landed in the resolved plan. The column
//! lineage pass reads the table: it expands through a view whose base tables were checked against the caller,
//! and shows any other view as a leaf.
//!
//! A view read is identified by the `SubqueryAlias` the inlining built: its alias and its origin (the read's,
//! which resolution keeps). Each expansion has its own entry. Two reads of one view through one shared
//! DataFrame (`df.join(df)`) are inlined twice with the same (alias, origin); the analyzer may resolve the two
//! equal subtrees to one node (the second side under a `Shift`), so a key can have more entries than nodes (only
//! the first entry gets ids), and a consumer matches a node by (alias, origin), never by ids. A view whose query
//! repeats a column name is inlined with a rename by position inside it, a second `SubqueryAlias` of the same
//! key: it is not a read of its own, and only the outer node is located.

use crate::plan::{LogicalPlan, PlanNode};
use crate::walk::{Node, PtrSet, children};
use crate::{AttrId, Origin};

/// How a view's base tables are authorized (T062 sets [`Self::Definer`]; today every permanent view is
/// read as its invoker reads its query).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ViewSecurity {
    /// A temporary view: its query is the caller's.
    Temporary,
    /// `SQL SECURITY INVOKER`: base tables are checked against the caller.
    Invoker,
    /// `SQL SECURITY DEFINER`: base tables are checked against the view's owner.
    Definer,
}

/// One view read the inlining expanded (one entry per expansion, see the module doc).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewExpansion {
    /// The view's name parts.
    pub name: Vec<String>,
    /// The alias of the `SubqueryAlias` the inlining built.
    pub alias: String,
    /// That node's origin (the read's), which resolution keeps.
    pub origin: Origin,
    /// How its base tables are authorized.
    pub security: ViewSecurity,
    /// Whether its base securables were authorized for the caller.
    pub bases_authorized: bool,
    /// The output attributes of this expansion's node in the resolved plan; `None` until located, or when
    /// resolution left no node for it.
    pub output: Option<Vec<AttrId>>,
}

impl ViewExpansion {
    /// A view read inlined as `SubqueryAlias alias` at `origin`, not located yet.
    #[must_use]
    pub fn new(
        name: Vec<String>,
        alias: String,
        origin: Origin,
        security: ViewSecurity,
        bases_authorized: bool,
    ) -> Self {
        Self {
            name,
            alias,
            origin,
            security,
            bases_authorized,
            output: None,
        }
    }

    /// Whether the view's query may be shown: only a view whose base tables were checked against the
    /// caller (temporary and `INVOKER` views), never a `DEFINER` view.
    #[must_use]
    pub fn expands(&self) -> bool {
        match self.security {
            ViewSecurity::Temporary | ViewSecurity::Invoker => self.bases_authorized,
            ViewSecurity::Definer => false,
        }
    }

    /// Whether the resolved node `alias` at `origin` is a read of this view.
    #[must_use]
    pub fn is_at(&self, alias: &str, origin: Origin) -> bool {
        self.alias == alias && self.origin == origin
    }
}

/// `views` with their [`output`](ViewExpansion::output) set from the resolved plan `root`: the `SubqueryAlias`
/// nodes with an entry's (alias, origin), in plan order, go to that key's entries in recording order, one node
/// each. A node nested under a node of the same key is not another read: it is the rename by position the
/// inlining adds inside a view whose query repeats a column name, so it gets no entry. An entry with no node
/// left keeps `None`: when no entry of its key is located either, no node carries the view (a consumer that
/// must not show a view's query fails closed on it). A node with no entry left (a function body inlined at two
/// call sites) stays a read of the key's views.
#[must_use]
pub fn locate_views(root: &LogicalPlan, mut views: Vec<ViewExpansion>) -> Vec<ViewExpansion> {
    if views.is_empty() {
        return views;
    }
    // The pre-order walk of `walk::visit_distinct`, each node carrying its innermost enclosing view node (an
    // index into `enclosing`, whose entries chain to their own enclosing node).
    let mut enclosing: Vec<(&str, Origin, Option<usize>)> = Vec::new();
    let mut seen: PtrSet<*const ()> = PtrSet::default();
    let mut stack = vec![(Node::Plan(root), None)];
    let mut next = Vec::new();
    while let Some((node, outer)) = stack.pop() {
        if !seen.insert(node.key()) {
            continue;
        }
        let mut inner = outer;
        if let Node::Plan(p) = node
            && let PlanNode::SubqueryAlias { alias, .. } = &p.node
            && views.iter().any(|v| v.is_at(alias, p.origin))
        {
            if !nested(&enclosing, outer, alias, p.origin)
                && let Some(v) = views
                    .iter_mut()
                    .find(|v| v.output.is_none() && v.is_at(alias, p.origin))
            {
                v.output = p
                    .resolved_output()
                    .map(|out| out.into_iter().map(|c| c.id).collect());
            }
            enclosing.push((alias, p.origin, outer));
            inner = Some(enclosing.len() - 1);
        }
        children(node, &mut next);
        stack.extend(next.drain(..).rev().map(|c| (c, inner)));
    }
    views
}

/// Whether the view node chain from `at` up holds a node `alias` at `origin`.
fn nested(
    enclosing: &[(&str, Origin, Option<usize>)],
    mut at: Option<usize>,
    alias: &str,
    origin: Origin,
) -> bool {
    while let Some(&(a, o, up)) = at.and_then(|i| enclosing.get(i)) {
        if a == alias && o == origin {
            return true;
        }
        at = up;
    }
    false
}
