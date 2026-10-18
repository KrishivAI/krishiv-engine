// SPDX-License-Identifier: Apache-2.0
//! Stage 1, Bind: SQL text, functions, tables, pins and access decisions.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use krishiv_common::error::{
    KrishivCatalogFailure, KrishivInsufficientPermissions, KrishivUnsupportedFeature,
    TableOrViewNotFound, UnresolvedRoutine, UnsupportedDatasourceForDirectQuery,
};
use krishiv_common::redact::RedactedSql;
use krishiv_common::secret::OptionValue;

use super::catalog::quoted;
use super::{
    AccessDecision, AnalysisContext, AnalysisError, Catalog, FunctionLookup, Privilege, Securable,
    SourceKind, TableColumn, TableIdent, TableKind, TableMeta,
};
use crate::walk::{self, Node};
use crate::{
    BoundColumn, ExprNode, LogicalPlan, Origin, Plan, PlanNode, ReadSource, SourceId, SqlArgs,
    TableVersion,
};

/// A table name as written in the plan, resolved.
#[derive(Debug, Clone)]
pub(super) enum BoundTable {
    /// Loaded; `securable` indexes [`Bindings::securables`].
    Found {
        meta: Arc<TableMeta>,
        securable: usize,
    },
    /// No such table, but `` format.`path` `` names files: the index of
    /// their [`Bindings::files`] entry.
    Files(usize),
    /// No such table.
    Missing,
}

/// A read by path and the columns its files have (nullable).
#[derive(Debug)]
pub(super) struct BoundFiles {
    pub(super) source: ReadSource,
    pub(super) columns: Vec<TableColumn>,
}

/// What Bind found: the input of Resolve besides the plan and context.
#[derive(Debug, Default)]
pub(super) struct Bindings {
    /// Tables by name parts as written.
    pub(super) tables: HashMap<Vec<String>, BoundTable>,
    /// Reads by path, each once.
    pub(super) files: Vec<BoundFiles>,
    /// Whether the plan reads files by path.
    pub(super) reads_files: bool,
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
/// unless `case_sensitive`. Java compares UTF-16 units and Rust `char`s;
/// they differ only for characters whose case mapping is several
/// characters (`ß` and the like), which no identifier comparison of Spark's
/// relies on.
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

enum Found<'a> {
    Table(&'a [String], Origin),
    /// An already-resolved scan: its name, version and columns.
    Scan(
        &'a [String],
        Option<TableVersion>,
        &'a [BoundColumn],
        Origin,
    ),
    Path(Securable, Origin),
    /// A read by path, after its paths.
    Files(&'a ReadSource, Origin),
    /// An already-resolved scan of files, after its paths.
    FileScan(&'a ReadSource, &'a [BoundColumn], Origin),
}

/// One node of the walk: reports what it reads, checks its functions, and
/// queues its children with their CTE scope.
fn visit<'a>(
    node: Node<'a>,
    env: &Arc<[String]>,
    ctx: &AnalysisContext,
    next: &mut Vec<(Node<'a>, Arc<[String]>)>,
    found: &mut Vec<Found<'a>>,
) -> Result<(), AnalysisError> {
    let cs = ctx.conf.case_sensitive;
    match node {
        Node::Plan(p) => {
            match &p.node {
                PlanNode::Read(ReadSource::Table { name, .. }) => {
                    let shadowed = name.len() == 1 && env.iter().any(|c| same(c, &name[0], cs));
                    if !shadowed {
                        found.push(Found::Table(name, p.origin));
                    }
                }
                PlanNode::Read(source @ ReadSource::Files { paths, .. }) => {
                    paths.iter().for_each(|v| {
                        found.push(Found::Path(Securable::Path(v.clone()), p.origin));
                    });
                    found.push(Found::Files(source, p.origin));
                }
                // A plan resolved elsewhere is checked like a fresh one.
                PlanNode::Scan(s) => match &s.source {
                    ReadSource::Table { name, .. } => {
                        found.push(Found::Scan(name, s.version, &s.output, p.origin));
                    }
                    source @ ReadSource::Files { paths, .. } => {
                        paths.iter().for_each(|v| {
                            found.push(Found::Path(Securable::Path(v.clone()), p.origin));
                        });
                        found.push(Found::FileScan(source, &s.output, p.origin));
                    }
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
    // The canonical operators exist in every profile; the analyzer types
    // them itself.
    if crate::Builtin::from_name(name).is_some() {
        return Ok(());
    }
    match ctx
        .functions
        .lookup(std::slice::from_ref(&name.to_owned()), ctx.profile)
    {
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

/// `ident` as a catalog resolving case-insensitively sees it.
fn fold(ident: &TableIdent, case_sensitive: bool) -> TableIdent {
    if case_sensitive {
        return ident.clone();
    }
    TableIdent {
        catalog: ident.catalog.clone(),
        namespace: ident.namespace.iter().map(|s| s.to_lowercase()).collect(),
        name: ident.name.to_lowercase(),
    }
}

/// Asks the policy about `securable` once per query; a denial or any policy
/// error fails closed, without the policy's reason or error text, and row
/// filters and column masks are rejected until they can be applied.
async fn authorize(
    ctx: &AnalysisContext,
    b: &mut Bindings,
    checked: &mut Vec<(Securable, usize)>,
    securable: Securable,
    origin: Origin,
) -> Result<usize, AnalysisError> {
    if let Some((_, i)) = checked.iter().find(|(s, _)| *s == securable) {
        return Ok(*i);
    }
    let privilege = Privilege::Select;
    let decision = ctx
        .policy
        .check(&ctx.principal, &securable, privilege)
        .await;
    match decision {
        Ok(AccessDecision::Allow) => {}
        Ok(AccessDecision::AllowWith(_)) => {
            return Err(unsupported(
                format!("row filters and column masks (on {securable})"),
                origin,
            ));
        }
        Ok(AccessDecision::Deny { .. }) | Err(_) => {
            return Err(AnalysisError::new(
                KrishivInsufficientPermissions {
                    principal: quoted(std::slice::from_ref(&ctx.principal.name)),
                    privilege: privilege.to_string(),
                    securable: securable.to_string(),
                },
                origin,
            ));
        }
    }
    b.securables.push((securable.clone(), privilege));
    checked.push((securable, b.securables.len() - 1));
    Ok(b.securables.len() - 1)
}

/// For every table a plan reads (`Read` or an already-resolved `Scan`): asks
/// the policy about the name as located, before any metadata is read (so a
/// denied principal learns nothing about existence); loads the table once,
/// pinning one version per table; and asks the policy again under the
/// catalog's own identifier when it differs. A scan must name a table that
/// exists, at the pinned version, with columns the table has.
async fn load(
    ctx: &AnalysisContext,
    b: &mut Bindings,
    found: Vec<Found<'_>>,
) -> Result<(), AnalysisError> {
    let cs = ctx.conf.case_sensitive;
    let mut checked: Vec<(Securable, usize)> = Vec::new();
    // Scans already validated, by table: a scan of one table may differ from
    // another in version and columns, and each is checked on its own.
    type Seen<'a> = (Option<TableVersion>, &'a [BoundColumn]);
    let mut validated: HashMap<&[String], Vec<Seen<'_>>> = HashMap::new();
    for f in found {
        let (name, version, columns, origin, scan) = match f {
            Found::Path(securable, origin) => {
                authorize(ctx, b, &mut checked, securable, origin).await?;
                continue;
            }
            Found::Files(source, origin) => {
                if origin.sql.is_some()
                    && let ReadSource::Files { format, paths, .. } = source
                {
                    sql_on_files(ctx, b, format, paths, origin)?;
                }
                bind_files(ctx, b, source, origin).await?;
                continue;
            }
            Found::FileScan(source, columns, origin) => {
                let i = bind_files(ctx, b, source, origin).await?;
                let known = columns.iter().all(|c| {
                    b.files[i]
                        .columns
                        .iter()
                        .any(|m| m.name == c.name && m.data_type == c.data_type)
                });
                if !known {
                    return Err(unsupported("a scan that does not match its files", origin));
                }
                continue;
            }
            Found::Table(name, origin) => (name, None, &[][..], origin, false),
            Found::Scan(name, version, columns, origin) => (name, version, columns, origin, true),
        };
        if !scan && b.tables.contains_key(name)
            || scan
                && validated.get(name).is_some_and(|v| {
                    v.iter()
                        .any(|(ver, cols)| *ver == version && *cols == columns)
                })
        {
            continue;
        }
        let search_path = b.search_path.clone();
        let not_found = || {
            AnalysisError::new(
                TableOrViewNotFound {
                    relation_name: quoted(name),
                    search_path: search_path.clone(),
                },
                origin,
            )
        };
        let Some((catalog, ident)) = locate(ctx, name) else {
            if scan {
                return Err(not_found());
            }
            let bound = path_read(ctx, b, &mut checked, name, origin).await?;
            b.tables.insert(name.to_vec(), bound);
            continue;
        };
        // Asked as written, and folded the way a case-insensitive catalog
        // folds, so no spelling of a name dodges a rule on another (the
        // catalog's own identifier is asked below once the table is loaded).
        for first in [ident.clone(), fold(&ident, cs)] {
            authorize(ctx, b, &mut checked, Securable::Table(first), origin).await?;
        }
        let failed = |_| {
            AnalysisError::new(
                KrishivCatalogFailure {
                    catalog: quoted(std::slice::from_ref(&ident.catalog)),
                    relation_name: quoted(name),
                },
                origin,
            )
        };
        b.epochs.insert(catalog.name().to_owned(), catalog.epoch());
        let mut meta = catalog
            .load_table(&ident, version, cs)
            .await
            .map_err(failed)?;
        if let Some(m) = &meta
            && let Some(&pinned) = b.pins.get(&m.ident)
            && m.version != Some(pinned)
        {
            if version.is_some() {
                return Err(unsupported("two versions of one table in a query", origin));
            }
            // Another spelling of a table already pinned: read the pin.
            meta = catalog
                .load_table(&ident, Some(pinned), cs)
                .await
                .map_err(failed)?;
        }
        let Some(meta) = meta else {
            if scan {
                return Err(not_found());
            }
            let bound = path_read(ctx, b, &mut checked, name, origin).await?;
            b.tables.insert(name.to_vec(), bound);
            continue;
        };
        let securable = match meta.kind {
            TableKind::Table => Securable::Table(meta.ident.clone()),
            TableKind::View => Securable::View(meta.ident.clone()),
        };
        let securable = authorize(ctx, b, &mut checked, securable, origin).await?;
        if let Some(v) = meta.version {
            b.pins.entry(meta.ident.clone()).or_insert(v);
        }
        if scan {
            let known = columns.iter().all(|c| {
                meta.columns
                    .iter()
                    .any(|m| m.name == c.name && m.data_type == c.data_type)
            });
            if !known || version != meta.version {
                return Err(unsupported("a scan that does not match its table", origin));
            }
            b.scans.insert(name.to_vec(), securable);
            validated.entry(name).or_default().push((version, columns));
        } else {
            b.tables.insert(
                name.to_vec(),
                BoundTable::Found {
                    meta: Arc::new(meta),
                    securable,
                },
            );
        }
    }
    Ok(())
}

/// A table name no catalog has: Spark's `ResolveSQLOnFile` reads
/// `` format.`path` `` as files when `spark.sql.runSQLOnFiles` is on and
/// `format` is a file source; a data source that reads no files is
/// `UNSUPPORTED_DATASOURCE_FOR_DIRECT_QUERY`; anything else stays missing.
async fn path_read(
    ctx: &AnalysisContext,
    b: &mut Bindings,
    checked: &mut Vec<(Securable, usize)>,
    name: &[String],
    origin: Origin,
) -> Result<BoundTable, AnalysisError> {
    let ([format, path], Some(files)) = (name, &ctx.files) else {
        return Ok(BoundTable::Missing);
    };
    if !ctx.conf.run_sql_on_files {
        return Ok(BoundTable::Missing);
    }
    match files.source_kind(format) {
        SourceKind::File => {}
        SourceKind::Other => {
            return Err(AnalysisError::new(
                UnsupportedDatasourceForDirectQuery {
                    data_source_type: format.clone(),
                },
                origin,
            ));
        }
        SourceKind::Unknown => return Ok(BoundTable::Missing),
    }
    let path = OptionValue::from_user("path", path);
    authorize(ctx, b, checked, Securable::Path(path.clone()), origin).await?;
    let source = ReadSource::Files {
        format: format.clone(),
        paths: vec![path],
        options: BTreeMap::new(),
    };
    Ok(BoundTable::Files(
        bind_files(ctx, b, &source, origin).await?,
    ))
}

/// A read by path written in SQL (`` format.`a/path` ``, which lowering
/// keeps typed): Spark's `ResolveSQLOnFile` reads files only when
/// `spark.sql.runSQLOnFiles` is on and `format` is a file source; otherwise
/// the name is a missing table, or `UNSUPPORTED_DATASOURCE_FOR_DIRECT_QUERY`
/// for a data source that reads no files.
fn sql_on_files(
    ctx: &AnalysisContext,
    b: &Bindings,
    format: &str,
    paths: &[OptionValue],
    origin: Origin,
) -> Result<(), AnalysisError> {
    let kind = match (&ctx.files, ctx.conf.run_sql_on_files) {
        (Some(f), true) => f.source_kind(format),
        // Without a resolver, `bind_files` reports the read unsupported.
        (None, true) => return Ok(()),
        (_, false) => SourceKind::Unknown,
    };
    match kind {
        SourceKind::File => Ok(()),
        SourceKind::Other => Err(AnalysisError::new(
            UnsupportedDatasourceForDirectQuery {
                data_source_type: format.to_owned(),
            },
            origin,
        )),
        SourceKind::Unknown => {
            let mut name = vec![format.to_owned()];
            name.extend(paths.iter().map(ToString::to_string));
            Err(AnalysisError::new(
                TableOrViewNotFound {
                    relation_name: quoted(&name),
                    search_path: b.search_path.clone(),
                },
                origin,
            ))
        }
    }
}

/// Infers the columns of the read by path `source` once per query (its
/// paths are authorized already) and returns its [`Bindings::files`] index.
async fn bind_files(
    ctx: &AnalysisContext,
    b: &mut Bindings,
    source: &ReadSource,
    origin: Origin,
) -> Result<usize, AnalysisError> {
    b.reads_files = true;
    if let Some(i) = b.files.iter().position(|f| f.source == *source) {
        return Ok(i);
    }
    let (
        ReadSource::Files {
            format,
            paths,
            options,
        },
        Some(files),
    ) = (source, &ctx.files)
    else {
        return Err(unsupported("reading files by path", origin));
    };
    let columns = files
        .infer(format, paths, options)
        .await
        .map_err(|e| AnalysisError::new(e, origin))?
        .into_iter()
        // Spark reads every file column as nullable.
        .map(|c| TableColumn {
            nullable: true,
            ..c
        })
        .collect();
    b.files.push(BoundFiles {
        source: source.clone(),
        columns,
    });
    Ok(b.files.len() - 1)
}
