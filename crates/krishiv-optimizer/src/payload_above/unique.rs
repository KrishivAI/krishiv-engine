// SPDX-License-Identifier: Apache-2.0
//! FD2 for `PayloadAboveAggregate`: how a relation is proven unique on a set
//! of its columns ([`unique`]), or which base relations' columns a count
//! would have to prove unique first ([`Key::Wanted`], design P2b A5 (d)).

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Builtin, Grouping, JoinCondition, JoinKind, LogicalPlan, PlanNode, ReadSource,
};

use super::{identity_key, ids};
use crate::expr::{Attrs, builtin, conjuncts, output_of, strip_alias};
use crate::late_payload::bare_column;
use crate::stats::Statistics;

/// A base relation's columns a key count may prove unique: the scan and
/// its output column names, sorted.
#[derive(Clone)]
pub(super) struct Want {
    /// The scan node.
    pub(super) scan: Arc<LogicalPlan>,
    /// Its output column names, sorted, without repeats.
    pub(super) columns: Vec<String>,
}

impl Want {
    /// `<relation> [<columns>]`, for notes.
    pub(super) fn text(&self) -> String {
        format!(
            "{} [{}]",
            super::relation_name(&self.scan),
            self.columns.join(", ")
        )
    }

    fn same(&self, other: &Self) -> bool {
        self.columns == other.columns
            && match (&self.scan.node, &other.scan.node) {
                (PlanNode::Scan(a), PlanNode::Scan(b)) => a.source == b.source,
                _ => false,
            }
    }
}

/// What is known of a relation's uniqueness on some columns.
pub(super) enum Key {
    /// Proven, and how.
    Proven(String),
    /// Proven once each of these is (never empty).
    Wanted(Vec<Want>),
}

impl Key {
    /// Both `self` and `other` hold: proven when both are, else what each
    /// still wants.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Proven(a), Self::Proven(b)) => Self::Proven(format!("{a} joined to {b}")),
            (a, b) => {
                let mut wanted = a.into_wanted();
                for w in b.into_wanted() {
                    push_want(&mut wanted, w);
                }
                Self::Wanted(wanted)
            }
        }
    }

    /// What it still wants (nothing when proven).
    pub(super) fn into_wanted(self) -> Vec<Want> {
        match self {
            Self::Proven(_) => Vec::new(),
            Self::Wanted(w) => w,
        }
    }
}

/// Adds `w` to `wanted` unless it is there.
pub(super) fn push_want(wanted: &mut Vec<Want>, w: Want) {
    if !wanted.iter().any(|x| x.same(&w)) {
        wanted.push(w);
    }
}

/// How `p` is proven unique on a subset of `cols` among the rows whose
/// columns of that subset are not NULL; which base relations' columns a
/// count must prove unique first ([`Key::Wanted`]); or `None`:
/// - a scan whose table declares such a set ([`Statistics::unique_keys`]),
///   else a count of the scan's columns in `cols` (wanted);
/// - an aggregate grouped by columns that `cols` all hold (the plan);
/// - through filters, `Shift`s and projections that pass the columns through;
/// - an inner join of `A` unique on its part of `cols` with `B` unique on the
///   columns its equalities compare (each `A` row meets at most one `B` row),
///   either way round: a proof first, else the way that wants fewer counts.
pub(super) fn unique(p: &Arc<LogicalPlan>, cols: &Attrs, stats: &dyn Statistics) -> Option<Key> {
    if cols.is_empty() {
        return None;
    }
    match &p.node {
        PlanNode::Scan(s) => {
            let named: HashSet<&str> = s
                .output
                .iter()
                .filter(|c| cols.contains(&c.id))
                .map(|c| c.name.as_str())
                .collect();
            if named.is_empty() {
                return None;
            }
            if let Some(k) = stats
                .unique_keys(s)
                .into_iter()
                .find(|k| !k.is_empty() && k.iter().all(|n| named.contains(n.as_str())))
            {
                return Some(Key::Proven(format!("statistics (key [{}])", k.join(", "))));
            }
            // Only a base relation's columns are counted: a scan of a table
            // or of files.
            if !matches!(
                s.source,
                ReadSource::Table { .. } | ReadSource::Files { .. }
            ) {
                return None;
            }
            let mut columns: Vec<String> = named.into_iter().map(str::to_owned).collect();
            columns.sort();
            Some(Key::Wanted(vec![Want {
                scan: Arc::clone(p),
                columns,
            }]))
        }
        PlanNode::Filter { input, .. } => unique(input, cols, stats),
        PlanNode::Shift { input, offset } => {
            let below: Attrs = cols
                .iter()
                .filter_map(|id| id.0.checked_sub(*offset).map(AttrId))
                .collect();
            unique(input, &below, stats)
        }
        PlanNode::Project { input, exprs } => {
            let mut below = Attrs::default();
            for e in exprs {
                if let Some(out) = output_of(e)
                    && cols.contains(&out.id)
                    && let Some(c) = bare_column(strip_alias(e))
                {
                    below.insert(c.id);
                }
            }
            unique(input, &below, stats)
        }
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            aggregates,
            ..
        } => {
            let mut key_ids = Attrs::default();
            for k in keys {
                key_ids.insert(bare_column(k)?.id);
            }
            // The output columns that are grouping keys, as their key.
            let mut held = Attrs::default();
            for e in aggregates {
                if let Some(out) = output_of(e)
                    && cols.contains(&out.id)
                    && let Some(c) = bare_column(strip_alias(e))
                    && key_ids.contains(&c.id)
                {
                    held.insert(c.id);
                }
            }
            (!key_ids.is_empty() && key_ids.is_subset(&held)).then(|| {
                let v: Vec<String> = keys
                    .iter()
                    .filter_map(|k| bare_column(k).map(|c| c.name.clone()))
                    .collect();
                Key::Proven(format!("the plan (grouped by [{}])", v.join(", ")))
            })
        }
        PlanNode::Join {
            left,
            right,
            kind: JoinKind::Inner,
            condition: JoinCondition::On(on),
        } => {
            let (l_ids, r_ids) = (ids(left)?, ids(right)?);
            let mut l_keys = Attrs::default();
            let mut r_keys = Attrs::default();
            for conj in conjuncts(on) {
                if let Some((Builtin::Eq, [a, b], _)) = builtin(&conj)
                    && let (Some(a), Some(b)) = (bare_column(a), bare_column(b))
                    && identity_key(&a.data_type)
                    && a.data_type == b.data_type
                {
                    if l_ids.contains(&a.id) && r_ids.contains(&b.id) {
                        l_keys.insert(a.id);
                        r_keys.insert(b.id);
                    } else if l_ids.contains(&b.id) && r_ids.contains(&a.id) {
                        l_keys.insert(b.id);
                        r_keys.insert(a.id);
                    }
                }
            }
            let l_cols: Attrs = cols.intersection(&l_ids).copied().collect();
            let r_cols: Attrs = cols.intersection(&r_ids).copied().collect();
            let both = |a: Option<Key>, b: &dyn Fn() -> Option<Key>| Some(a?.and(b()?));
            let first = both(unique(left, &l_cols, stats), &|| {
                unique(right, &r_keys, stats)
            });
            if let Some(Key::Proven(_)) = first {
                return first;
            }
            let second = both(unique(right, &r_cols, stats), &|| {
                unique(left, &l_keys, stats)
            });
            match (first, second) {
                (_, Some(Key::Proven(p))) => Some(Key::Proven(p)),
                (Some(Key::Wanted(a)), Some(Key::Wanted(b))) => {
                    Some(Key::Wanted(if b.len() < a.len() { b } else { a }))
                }
                (a, b) => a.or(b),
            }
        }
        _ => None,
    }
}
