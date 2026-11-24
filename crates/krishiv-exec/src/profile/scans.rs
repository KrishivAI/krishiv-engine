// SPDX-License-Identifier: Apache-2.0
//! The tables a query scanned and the columns it compared with values (P3-T025): the input of the partition
//! advisor, kept in the profile's `scans` member (minor 6).
//!
//! A column counts when a filter of the plan (a `Filter` node, or a filter pushed into a scan) holds, as a
//! top-level `AND` conjunct, `column = value` or `column IN (values)` (a cast of the column included). Joins
//! and ranges do not count: only an equality on a value lets a partitioned table skip whole partitions.
//!
//! A table that any scan under a security barrier reads ([`PhysicalPlan::guarded`]) lists no column, whichever
//! filter holds the comparison: a barrier keeps attribute ids, so a filter above it on a column it passes through
//! names the guarded table's column too, and the policy's own conjuncts must not reach the advisor or later
//! readers of the history (design `P3-E14-security-barrier.md` §5).

use std::collections::{BTreeMap, BTreeSet};

use krishiv_optimizer::TableStats;
use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, FunctionId, ReadSource, ResolvedExpr, Scan, conjuncts,
};

use crate::physical::{PhysNode, PhysicalPlan, ScanId};

/// One table a query scanned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileScan {
    /// The table, as `catalog.schema.name`, each component backquoted (backquotes doubled) unless it is a plain
    /// identifier (letters, digits and `_`, not all digits), so distinct component sequences name distinct tables.
    pub table: String,
    /// The table's row count when the query was planned, if known.
    pub rows: Option<u64>,
    /// The columns the query compared with values, by name.
    pub equality: Vec<ScanColumn>,
}

/// A column of a [`ProfileScan`] the query compared with values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanColumn {
    /// The column name.
    pub name: String,
    /// Its distinct non-null values when the query was planned, if known.
    pub distinct: Option<u64>,
}

/// The scans of `plan`, one per named table (a table read twice is one entry with the union of its columns), in
/// order of first appearance. `stats` gives a scan's table statistics, when known.
#[must_use]
pub fn profile_scans(
    plan: &PhysicalPlan,
    stats: &dyn Fn(&Scan) -> Option<TableStats>,
) -> Vec<ProfileScan> {
    let mut scans = Vec::new();
    let mut predicates = Vec::new();
    for root in plan.roots() {
        collect(root, &mut scans, &mut predicates);
    }
    let guarded: BTreeSet<String> = scans
        .iter()
        .filter(|(id, _)| plan.guarded().binary_search(id).is_ok())
        .filter_map(|(_, scan)| match &scan.source {
            ReadSource::Table { name, .. } => Some(table_name(name)),
            _ => None,
        })
        .collect();
    let mut compared = BTreeSet::new();
    for p in predicates {
        for c in conjuncts(p) {
            if let Some(id) = compared_column(c) {
                compared.insert(id);
            }
        }
    }
    let mut out: Vec<ProfileScan> = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (_, scan) in scans {
        let ReadSource::Table { name, .. } = &scan.source else {
            continue;
        };
        let table = table_name(name);
        let hidden = guarded.contains(&table);
        let table_stats = stats(scan);
        let at = *seen.entry(table.clone()).or_insert_with(|| {
            out.push(ProfileScan {
                table,
                rows: table_stats.as_ref().map(|s| s.rows),
                equality: Vec::new(),
            });
            out.len() - 1
        });
        for c in scan
            .output
            .iter()
            .filter(|c| !hidden && compared.contains(&c.id))
        {
            if out[at].equality.iter().all(|e| e.name != c.name) {
                let distinct = table_stats
                    .as_ref()
                    .and_then(|s| s.columns.get(&c.name))
                    .and_then(|s| s.distinct);
                out[at].equality.push(ScanColumn {
                    name: c.name.clone(),
                    distinct,
                });
            }
        }
    }
    for s in &mut out {
        s.equality.sort_by(|a, b| a.name.cmp(&b.name));
    }
    out
}

/// The components of a table name joined by `.`, each of letters, digits and `_` (not all digits) left bare and any
/// other backquoted with its backquotes doubled (Spark's `quoteIfNeeded`): `["cat", "db", "a.b"]` is
/// ``cat.db.`a.b` ``, never the same text as `["cat", "db.a", "b"]`.
fn table_name(parts: &[String]) -> String {
    let quoted: Vec<String> = parts
        .iter()
        .map(|p| {
            let bare = !p.is_empty()
                && p.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !p.chars().all(|c| c.is_ascii_digit());
            if bare {
                p.clone()
            } else {
                format!("`{}`", p.replace('`', "``"))
            }
        })
        .collect();
    quoted.join(".")
}

fn collect<'a>(
    node: &'a PhysNode,
    scans: &mut Vec<(ScanId, &'a Scan)>,
    predicates: &mut Vec<&'a Expr>,
) {
    match node {
        PhysNode::Scan(s) => {
            scans.push((s.id, &s.scan));
            if let Some(f) = &s.filter {
                predicates.push(&f.predicate);
            }
        }
        PhysNode::Filter { predicate, .. } => predicates.push(predicate),
        _ => {}
    }
    for child in node.children() {
        collect(child, scans, predicates);
    }
}

/// The column `e` compares with a value by `=` or `IN`, if it does.
fn compared_column(e: &Expr) -> Option<AttrId> {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return None;
    };
    let FunctionId::Builtin(b) = &f.id else {
        return None;
    };
    let args: Vec<&Expr> = f.args.iter().map(AsRef::as_ref).collect();
    match (b, args.as_slice()) {
        (Builtin::Eq | Builtin::EqNullSafe, [a, c]) => match (column(a), column(c)) {
            (Some(id), None) if is_value(c) => Some(id),
            (None, Some(id)) if is_value(a) => Some(id),
            _ => None,
        },
        (Builtin::In, [a, list @ ..]) if !list.is_empty() && list.iter().all(|x| is_value(x)) => {
            column(a)
        }
        _ => None,
    }
}

fn column(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        ExprNode::Cast { expr, .. } => column(expr),
        _ => None,
    }
}

/// A literal, or a cast of one.
fn is_value(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(_) => true,
        ExprNode::Cast { expr, .. } => is_value(expr),
        _ => false,
    }
}
