// SPDX-License-Identifier: Apache-2.0
//! Stage 1, Bind: SQL text, functions, tables, pins and access decisions.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use krishiv_common::error::{
    KrishivInsufficientPermissions, KrishivUnsupportedFeature, UnresolvedRoutine,
};
use krishiv_common::redact::RedactedSql;

use super::catalog::quoted;
use super::{
    AccessDecision, AnalysisContext, AnalysisError, Catalog, FunctionLookup, Privilege, Securable,
    TableIdent, TableKind, TableMeta,
};
use crate::walk::{self, Node};
use crate::{
    ExprNode, LogicalPlan, Origin, Plan, PlanNode, ReadSource, SourceId, SqlArgs, TableVersion,
};

/// A table name as written in the plan, resolved.
#[derive(Debug, Clone)]
pub(super) enum BoundTable {
    /// Loaded; `securable` indexes [`Bindings::securables`].
    Found {
        meta: Arc<TableMeta>,
        securable: usize,
    },
    /// No such table.
    Missing,
}

/// What Bind found: the input of Resolve besides the plan and context.
#[derive(Debug, Default)]
pub(super) struct Bindings {
    /// Tables by name parts as written.
    pub(super) tables: HashMap<Vec<String>, BoundTable>,
    /// Securables of already-resolved scans, by name parts.
    pub(super) scans: HashMap<Vec<String>, usize>,
    /// The lowered plan of each `PlanNode::Sql`, by node address.
    pub(super) sql: HashMap<usize, Arc<LogicalPlan>>,
    pub(super) pins: BTreeMap<TableIdent, TableVersion>,
    pub(super) epochs: BTreeMap<String, u64>,
    pub(super) securables: Vec<(Securable, Privilege)>,
    pub(super) query_text: Option<RedactedSql>,
    /// Spark's search path, for `TABLE_OR_VIEW_NOT_FOUND` and
    /// `UNRESOLVED_ROUTINE`.
    pub(super) search_path: String,
}

pub(super) fn addr<T: ?Sized>(p: &T) -> usize {
    std::ptr::from_ref(p).cast::<u8>() as usize
}

pub(super) fn unsupported(feature: impl Into<String>, origin: Origin) -> AnalysisError {
    AnalysisError::new(
        KrishivUnsupportedFeature {
            feature: feature.into(),
        },
        origin,
    )
}

/// Whether `a` and `b` name the same thing: Java's `equalsIgnoreCase`
/// unless `case_sensitive`.
pub(super) fn same(a: &str, b: &str, case_sensitive: bool) -> bool {
    if case_sensitive {
        return a == b;
    }
    a.len() == b.len() && a.eq_ignore_ascii_case(b)
        || a.chars().count() == b.chars().count()
            && a.chars().zip(b.chars()).all(|(x, y)| {
                x == y
                    || x.to_uppercase().eq(y.to_uppercase())
                    || x.to_lowercase().eq(y.to_lowercase())
            })
}

pub(super) async fn bind(plan: &Plan, ctx: &AnalysisContext) -> Result<Bindings, AnalysisError> {
    let mut b = Bindings {
        search_path: search_path(ctx),
        ..Bindings::default()
    };
    // SQL text first: its parse errors come before anything else.
    let mut sql = Vec::new();
    let _ = walk::visit_distinct(Node::Plan(plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sql { query, args } = &p.node
        {
            sql.push((p, query, args));
        }
        ControlFlow::Continue(())
    });
    let mut roots = vec![Arc::clone(plan.root())];
    for (p, query, args) in sql {
        let root = lower_sql(ctx, query, args, p.origin, &b)?;
        if std::ptr::eq(p, &**plan.root()) {
            b.query_text = ctx.sql.as_ref().map(|f| f.redact(query));
        }
        b.sql.insert(addr(p), Arc::clone(&root));
        roots.push(root);
    }
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    // Every CTE scope stays alive for the walk, so no scope address is
    // reused while `seen` holds it.
    let mut scopes: Vec<Arc<[String]>> = vec![Arc::from(Vec::new())];
    for root in &roots {
        let mut stack = vec![(Node::Plan(root), Arc::clone(&scopes[0]))];
        while let Some((node, env)) = stack.pop() {
            if !seen.insert((node.key() as usize, addr(&*env))) {
                continue;
            }
            let mut next = Vec::new();
            visit(node, &env, ctx, &mut next, &mut found)?;
            scopes.extend(next.iter().map(|(_, e)| Arc::clone(e)));
            stack.extend(next.into_iter().rev());
        }
    }
    load(ctx, &mut b, found).await?;
    Ok(b)
}

enum Found {
    Table(Vec<String>, Origin),
    Scan(Vec<String>, Origin),
    Path(Securable, Origin),
}

/// One node of the walk: reports what it reads, checks its functions, and
/// queues its children with their CTE scope.
fn visit<'a>(
    node: Node<'a>,
    env: &Arc<[String]>,
    ctx: &AnalysisContext,
    next: &mut Vec<(Node<'a>, Arc<[String]>)>,
    found: &mut Vec<Found>,
) -> Result<(), AnalysisError> {
    let cs = ctx.conf.case_sensitive;
    match node {
        Node::Plan(p) => {
            match &p.node {
                PlanNode::Read(ReadSource::Table { name, .. }) => {
                    let shadowed = name.len() == 1 && env.iter().any(|c| same(c, &name[0], cs));
                    if !shadowed {
                        found.push(Found::Table(name.clone(), p.origin));
                    }
                }
                PlanNode::Read(ReadSource::Files { paths, .. }) => paths
                    .iter()
                    .for_each(|v| found.push(Found::Path(Securable::Path(v.clone()), p.origin))),
                // A plan resolved elsewhere is checked like a fresh one.
                PlanNode::Scan(s) => match &s.source {
                    ReadSource::Table { name, .. } => {
                        found.push(Found::Scan(name.clone(), p.origin));
                    }
                    ReadSource::Files { paths, .. } => paths.iter().for_each(|v| {
                        found.push(Found::Path(Securable::Path(v.clone()), p.origin));
                    }),
                },
                PlanNode::WithCte { input, ctes } => {
                    p.for_each_expr(&mut |e| next.push((Node::Expr(e), Arc::clone(env))));
                    let mut scope = env.to_vec();
                    for c in ctes {
                        next.push((Node::Plan(&c.plan), Arc::from(scope.clone())));
                        scope.push(c.name.clone());
                    }
                    next.push((Node::Plan(input), Arc::from(scope)));
                    return Ok(());
                }
                _ => {}
            }
            p.for_each_expr(&mut |e| next.push((Node::Expr(e), Arc::clone(env))));
            p.for_each_child(&mut |c| next.push((Node::Plan(c), Arc::clone(env))));
        }
        Node::Expr(e) => {
            if let ExprNode::Function(f) = &e.node {
                check_function(ctx, &f.name, e.origin)?;
            }
            if let ExprNode::Subquery { plan, .. } = &e.node {
                next.push((Node::Plan(plan), Arc::clone(env)));
            }
            e.for_each_child(&mut |c| next.push((Node::Expr(c), Arc::clone(env))));
        }
    }
    Ok(())
}

fn check_function(ctx: &AnalysisContext, name: &str, origin: Origin) -> Result<(), AnalysisError> {
    match ctx.functions.lookup(name, ctx.profile) {
        FunctionLookup::Found(_) => Ok(()),
        FunctionLookup::KnownUnsupported => Err(unsupported(format!("function {name}"), origin)),
        FunctionLookup::NotFound => Err(AnalysisError::new(
            UnresolvedRoutine {
                routine_name: quoted(&[name.to_owned()]),
                search_path: search_path(ctx),
            },
            origin,
        )),
    }
}

fn search_path(ctx: &AnalysisContext) -> String {
    let mut current = vec![ctx.catalogs.current_catalog().to_owned()];
    current.extend(ctx.catalogs.current_namespace().iter().cloned());
    format!(
        "[`system`.`builtin`, `system`.`session`, {}]",
        quoted(&current)
    )
}

fn lower_sql(
    ctx: &AnalysisContext,
    query: &str,
    args: &SqlArgs,
    origin: Origin,
    b: &Bindings,
) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if !matches!(args, SqlArgs::None) {
        return Err(unsupported("SQL parameters", origin));
    }
    let Some(frontend) = &ctx.sql else {
        return Err(unsupported("SQL text without a SQL front end", origin));
    };
    let source = SourceId(u32::try_from(b.sql.len()).unwrap_or(u32::MAX));
    Ok(Arc::clone(frontend.lower(query, source)?.root()))
}

/// Where name `parts` live: the catalog and the identity to load.
fn locate(ctx: &AnalysisContext, parts: &[String]) -> Option<(Arc<dyn Catalog>, TableIdent)> {
    let (last, init) = parts.split_last()?;
    let named = (parts.len() > 1)
        .then(|| ctx.catalogs.catalog(&parts[0]))
        .flatten();
    let (catalog, namespace) = match named {
        Some(c) => (c, init[1..].to_vec()),
        None => {
            let c = ctx.catalogs.catalog(ctx.catalogs.current_catalog())?;
            let ns = if init.is_empty() {
                ctx.catalogs.current_namespace().to_vec()
            } else {
                init.to_vec()
            };
            (c, ns)
        }
    };
    let ident = TableIdent {
        catalog: catalog.name().to_owned(),
        namespace,
        name: last.clone(),
    };
    Some((catalog, ident))
}

/// Loads each distinct table once, pins one version per table, and asks
/// the policy once per securable.
async fn load(
    ctx: &AnalysisContext,
    b: &mut Bindings,
    found: Vec<Found>,
) -> Result<(), AnalysisError> {
    let cs = ctx.conf.case_sensitive;
    let (mut names, mut scans, mut paths) = (Vec::new(), Vec::new(), Vec::new());
    for f in found {
        match f {
            Found::Table(n, o) => names.push((n, o)),
            Found::Scan(n, o) => scans.push((n, o)),
            Found::Path(s, o) => paths.push((s, o)),
        }
    }
    let mut index: HashMap<TableIdent, usize> = HashMap::new();
    let mut origins: Vec<Origin> = Vec::new();
    for (name, origin) in names {
        if b.tables.contains_key(&name) {
            continue;
        }
        let Some((catalog, ident)) = locate(ctx, &name) else {
            b.tables.insert(name, BoundTable::Missing);
            continue;
        };
        let err = |e| AnalysisError::new(e, origin);
        let mut meta = catalog.load_table(&ident, None, cs).await.map_err(err)?;
        if let Some(m) = &meta
            && let Some(&pinned) = b.pins.get(&m.ident)
            && m.version != Some(pinned)
        {
            // Another spelling of a table already pinned: read the pin.
            meta = catalog
                .load_table(&ident, Some(pinned), cs)
                .await
                .map_err(err)?;
        }
        b.epochs.insert(catalog.name().to_owned(), catalog.epoch());
        let Some(meta) = meta else {
            b.tables.insert(name, BoundTable::Missing);
            continue;
        };
        if let Some(v) = meta.version {
            b.pins.entry(meta.ident.clone()).or_insert(v);
        }
        let securable = *index.entry(meta.ident.clone()).or_insert_with(|| {
            let securable = match meta.kind {
                TableKind::Table => Securable::Table(meta.ident.clone()),
                TableKind::View => Securable::View(meta.ident.clone()),
            };
            b.securables.push((securable, Privilege::Select));
            origins.push(origin);
            b.securables.len() - 1
        });
        b.tables.insert(
            name,
            BoundTable::Found {
                meta: Arc::new(meta),
                securable,
            },
        );
    }
    for (name, origin) in scans {
        if let Some((_, ident)) = locate(ctx, &name) {
            let i = *index.entry(ident.clone()).or_insert_with(|| {
                b.securables
                    .push((Securable::Table(ident), Privilege::Select));
                origins.push(origin);
                b.securables.len() - 1
            });
            b.scans.insert(name, i);
        }
    }
    for (securable, origin) in paths {
        b.securables.push((securable, Privilege::Select));
        origins.push(origin);
    }
    for ((securable, privilege), origin) in b.securables.iter().zip(origins) {
        let decision = ctx
            .policy
            .check(&ctx.principal, securable, *privilege)
            .await;
        let denied = || {
            AnalysisError::new(
                KrishivInsufficientPermissions {
                    principal: quoted(std::slice::from_ref(&ctx.principal.name)),
                    privilege: privilege.to_string(),
                    securable: securable.to_string(),
                },
                origin,
            )
        };
        match decision {
            Ok(AccessDecision::Allow) => {}
            Ok(AccessDecision::AllowWith(_)) => {
                return Err(unsupported(
                    format!("row filters and column masks (on {securable})"),
                    origin,
                ));
            }
            // Fail closed: a denial or any policy error stops the query,
            // without the policy's reason or error text.
            Ok(AccessDecision::Deny { .. }) | Err(_) => return Err(denied()),
        }
    }
    Ok(())
}
