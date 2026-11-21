// SPDX-License-Identifier: Apache-2.0
//! Whether an analyzed plan reads a permanent table: the scans of a plan and
//! of the subqueries in its expressions, each shared expression node visited once.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource, Scan};

#[cfg(feature = "delta")]
use super::QueryCatalog;
use crate::store::CatalogStore as _;

/// Whether the analyzed `plan` reads the permanent table `namespace.name` (directly, or through a view or a
/// subquery): `INSERT OVERWRITE` of a table that is also read would delete what it reads.
pub(crate) fn reads_stored_table(
    plan: &LogicalPlan,
    namespace: &str,
    name: &str,
    current: &str,
) -> bool {
    let mut scans = Vec::new();
    collect_scans(plan, &mut scans);
    scans.iter().any(|scan| {
        let ReadSource::Table { name: parts, .. } = &scan.source else {
            return false;
        };
        let (ns, table) = match parts.as_slice() {
            [table] => (current, table),
            [.., ns, table] => (ns.as_str(), table),
            [] => return false,
        };
        ns.eq_ignore_ascii_case(namespace) && table.eq_ignore_ascii_case(name)
    })
}

/// Whether the analyzed `plan` reads the files at `path` (a Delta table read by path), the path
/// compared without a trailing `/`.
#[cfg(feature = "delta")]
pub(crate) fn reads_files_at(plan: &LogicalPlan, path: &str) -> bool {
    let mut scans = Vec::new();
    collect_scans(plan, &mut scans);
    let want = path.trim_end_matches('/');
    scans.iter().any(|scan| match &scan.source {
        ReadSource::Files { paths, .. } => paths.iter().any(|p| {
            p.as_plain()
                .is_some_and(|t| t.trim_end_matches('/') == want)
        }),
        _ => false,
    })
}

/// Whether the analyzed `plan` (or a subquery or view in it) reads the Delta table whose root is
/// `root` (as [`QueryCatalog::delta_root`] names it): any scan resolving to that root, whatever the
/// spelling of its path (`/t`, `/t/`, `file:/t`) or whether it names the catalog table or its
/// location. A Delta scan whose root cannot be resolved counts as a read of it (the safe side: the
/// conflict check then reads the whole table).
#[cfg(feature = "delta")]
pub(crate) fn reads_delta_root(plan: &LogicalPlan, root: &str, catalog: &QueryCatalog) -> bool {
    let mut scans = Vec::new();
    collect_scans(plan, &mut scans);
    scans.iter().any(|scan| match catalog.delta_root(scan) {
        Some(Ok(r)) => r == root,
        Some(Err(_)) => true,
        None => false,
    })
}

/// Whether the analyzed `plan` (or a subquery or view in it) scans `source`: the same table name
/// (parts compared ignoring ASCII case, as written: `t` and `default.t` do not match) or the same
/// paths as written. Used for row-level targets that are not Delta tables (Iceberg), where the
/// read filter is the condition anyway and the table's own copy-on-write validation checks it, so a
/// spelling it misses loses nothing; a Delta target compares roots ([`reads_delta_root`]).
pub(crate) fn reads_source(plan: &LogicalPlan, source: &ReadSource) -> bool {
    let mut scans = Vec::new();
    collect_scans(plan, &mut scans);
    scans.iter().any(|scan| match (&scan.source, source) {
        (ReadSource::Table { name: a, .. }, ReadSource::Table { name: b, .. }) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
        }
        (ReadSource::Files { paths: a, .. }, ReadSource::Files { paths: b, .. }) => a == b,
        _ => false,
    })
}

/// The scans of `plan` and of the subqueries in its expressions.
///
/// Expressions are DAGs (the analyzer shares a subexpression by `Arc`, as a
/// nested `decode` does in every search condition), so each expression node
/// is visited once: a walk that revisits shared nodes is exponential in the
/// nesting depth.
pub(super) fn collect_scans<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a Scan>) {
    collect_scans_in(plan, out, &mut HashSet::new());
}

fn collect_scans_in<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a Scan>, seen: &mut HashSet<usize>) {
    fn in_expr<'a>(e: &'a Arc<Expr>, out: &mut Vec<&'a Scan>, seen: &mut HashSet<usize>) {
        if !seen.insert(Arc::as_ptr(e).addr()) {
            return;
        }
        if let ExprNode::Subquery { plan, .. } = &e.node {
            collect_scans_in(plan, out, seen);
        }
        e.for_each_child(&mut |c| in_expr(c, out, seen));
    }
    if let PlanNode::Scan(scan) = &plan.node {
        out.push(scan);
    }
    plan.for_each_expr(&mut |e| in_expr(e, out, seen));
    plan.for_each_child(&mut |c| collect_scans_in(c, out, seen));
}

impl super::Expander<'_> {
    /// The name a read is asked about: a bare name that the path placed in a stored schema while `system` is
    /// current is `spark_catalog.<schema>.<name>`, where it is read, not `system.<name>`.
    pub(super) fn read_name(
        &self,
        written: &[String],
        namespace: &str,
        table: &str,
    ) -> Vec<String> {
        if written.len() == 1 && self.view_namespace.is_none() && self.outside_session() {
            vec![
                super::CATALOG.to_owned(),
                namespace.to_owned(),
                table.to_owned(),
            ]
        } else {
            written.to_vec()
        }
    }

    /// Whether the current catalog was another than `spark_catalog` (`system` or a declared one) as
    /// the query snapshot saw it (the live catalog outside a query's expansion): a two-part name is
    /// then no stored namespace, and a stored view's names are `spark_catalog`'s.
    fn outside_session(&self) -> bool {
        self.stored.map_or_else(
            || {
                !self
                    .catalog
                    .current_catalog()
                    .eq_ignore_ascii_case(super::CATALOG)
            },
            |s| s.outside_session,
        )
    }

    /// The first schema of `path` (in the catalog store) holding the table or view `table`.
    pub(super) fn path_schema_of(&self, path: &[Vec<String>], table: &str) -> Option<String> {
        path.iter().find_map(|entry| match entry.as_slice() {
            [c, n] if c.eq_ignore_ascii_case(super::CATALOG) => {
                let (n_key, t_key) = if self.case_sensitive {
                    (n.clone(), table.to_owned())
                } else {
                    (n.to_lowercase(), table.to_lowercase())
                };
                self.catalog.store.table(&n_key, &t_key).map(|_| n.clone())
            }
            _ => None,
        })
    }

    /// Whether a two-part name is `namespace.table` of the session catalog's store: inside a
    /// stored view always, outside one only while the current catalog is the session catalog. A
    /// two-part name is a namespace of the CURRENT catalog (`system` has none; a declared one's
    /// are its own), so there it is no stored view (and it is authorized as `system.<ns>.<table>`, never as the store's).
    pub(super) fn names_a_stored_namespace(&self) -> bool {
        self.view_namespace.is_some() || !self.outside_session()
    }

    /// The name of a table a stored view reads from `namespace`: `spark_catalog.namespace.table` while another
    /// catalog than `spark_catalog` is current (the view's tables belong to the catalog that defines it), else `namespace.table` as ever.
    pub(super) fn view_read_name(&self, namespace: String, table: String) -> Vec<String> {
        if self.outside_session() {
            vec![super::CATALOG.to_owned(), namespace, table]
        } else {
            vec![namespace, table]
        }
    }

    /// `plan`, a read of `namespace.table` inside a stored view, as `spark_catalog.namespace.table` while another
    /// catalog than `spark_catalog` is current: a view's table belongs to the catalog that defines it.
    pub(super) fn anchored(&self, plan: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        if self.view_namespace.is_some()
            && self.outside_session()
            && let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [namespace, table] = name.as_slice()
            && !crate::system::is_system(namespace)
            && !namespace.eq_ignore_ascii_case(crate::catalog::GLOBAL_NAMESPACE)
        {
            let name = vec![super::CATALOG.to_owned(), namespace.clone(), table.clone()];
            let read = ReadSource::Table {
                name,
                options: options.clone(),
            };
            return Arc::new(LogicalPlan::new(PlanNode::Read(read)).with_origin(plan.origin));
        }
        plan
    }
}

impl super::QueryCatalog {
    /// Whether `system` is the current catalog of this query's snapshot.
    pub(crate) fn in_system(&self) -> bool {
        crate::system::is_system(&self.catalog)
    }

    /// Whether `spark_catalog` is the current catalog of this query's snapshot.
    pub(crate) fn in_session_catalog(&self) -> bool {
        self.catalog.eq_ignore_ascii_case(super::CATALOG)
    }

    /// The `system` catalog of this query, with the session's temporary relations in front of it.
    pub(super) fn system_catalog(self: &Arc<Self>) -> crate::system::SystemCatalog {
        crate::system::SystemCatalog::of(&self.system).with_temps(Arc::clone(self))
    }
}

impl super::SessionCatalog {
    /// `USE catalog.namespace` in one step, so a query never sees the catalog changed and the namespace not.
    pub(crate) fn use_position(&self, catalog: &str, namespace: &str) {
        *self
            .position
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            (catalog.to_owned(), namespace.to_owned());
        self.epoch.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl super::SessionCatalog {
    /// [`sql_path`](Self::sql_path) with the current position given: the path of a query is built from the
    /// same read of the position as its catalog and namespace.
    pub(crate) fn sql_path_in(
        &self,
        default_conf: &str,
        catalog: &str,
        namespace: &[String],
    ) -> Vec<Vec<String>> {
        let current: Vec<String> = std::iter::once(catalog.to_owned())
            .chain(namespace.iter().cloned())
            .collect();
        let entries = self
            .stored_path()
            .unwrap_or_else(|| crate::sql_path::default_entries(default_conf));
        entries.iter().map(|e| e.parts(&current)).collect()
    }
}

impl super::Expander<'_> {
    /// The part of `path` that can shadow the bare `table`: the entries before `system` when the name is a
    /// system table and the path holds `system` (the first matching entry wins), else all of it.
    pub(super) fn before_system<'p>(path: &'p [Vec<String>], table: &str) -> &'p [Vec<String>] {
        let system = path
            .iter()
            .position(|e| matches!(e.as_slice(), [c] if crate::system::is_system(c)));
        match system {
            Some(at)
                if crate::system::is_system_table(&[
                    crate::system::SYSTEM.to_owned(),
                    table.to_lowercase(),
                ]) =>
            {
                &path[..at]
            }
            _ => path,
        }
    }

    /// `plan`, a bare read inside a view created while `system` was current (its frozen path holds
    /// `[system]`), as `system.<table>` when no schema before `system` in the path holds a stored table of that name and
    /// the name is a system table.
    pub(super) fn system_table_in_path(&self, plan: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
        let path = self.view_path.as_deref()?;
        let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node else {
            return None;
        };
        let [bare] = name.as_slice() else {
            return None;
        };
        // The first matching entry of the path wins: only the schemas before `system` can shadow it.
        let system = path
            .iter()
            .position(|e| matches!(e.as_slice(), [c] if crate::system::is_system(c)));
        let qualified = vec![crate::system::SYSTEM.to_owned(), bare.to_lowercase()];
        let system = system?;
        if !(crate::system::is_system_table(&qualified)
            && self.path_schema_of(&path[..system], bare).is_none())
        {
            return None;
        }
        let read = ReadSource::Table {
            name: qualified,
            options: options.clone(),
        };
        Some(Arc::new(
            LogicalPlan::new(PlanNode::Read(read)).with_origin(plan.origin),
        ))
    }
}
