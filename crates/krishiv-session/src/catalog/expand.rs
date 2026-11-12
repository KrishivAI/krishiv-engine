// SPDX-License-Identifier: Apache-2.0
//! View expansion: the walk that inlines permanent and temporary views into a plan.

use super::*;

/// [`expand_temp_views`]: the views on the current expansion path, by
/// lower-case name, for cycle detection.
pub(super) struct Expander<'a> {
    pub(super) catalog: &'a SessionCatalog,
    pub(super) case_sensitive: bool,
    pub(super) path: Vec<String>,
    pub(super) stored: Option<&'a StoredViews<'a>>,
    /// Inside a stored view: the namespace its bare table names belong to.
    pub(super) view_namespace: Option<String>,
    /// Inside a stored view: the SQL path frozen with it.
    pub(super) view_path: Option<Vec<Vec<String>>>,
    /// The catalog-store lookups so far, in plan order, without repeats.
    pub(super) reads: Vec<ReadCandidate>,
    /// The first failure inside a stored view's text, kept for [`Expansion::deferred`].
    pub(super) deferred: Option<KrishivError>,
    /// The stored views inlined so far, for [`Expansion::stored_reads`].
    pub(super) stored_reads: Vec<crate::view_schema::StoredRead>,
}

impl Expander<'_> {
    /// `plan` with its views inlined; `scope` holds the `WITH`
    /// definitions visible where `plan` reads.
    pub(super) fn go(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        // Snapshot pins and cached snapshots are unreachable from user
        // input however a read spells them: a read of one behaves exactly
        // as a missing table, with or without read options. (The pins
        // snapshot views read are scans, which pass through untouched
        // below.)
        if let PlanNode::Read(ReadSource::Table { name, .. }) = &plan.node
            && let [view] = name.as_slice()
            && !scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
            && is_reserved_view_name(view)
        {
            return Err(TableOrViewNotFound {
                relation_name: quote(view),
                search_path: search_path(),
            }
            .into());
        }
        // Inside a stored view a bare table name is the view's own namespace's, never a temporary view.
        if let Some(namespace) = &self.view_namespace
            && let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [bare] = name.as_slice()
            && !scope
                .iter()
                .any(|c| same_name(c, bare, self.case_sensitive))
        {
            if let Some(system) = self.system_table_in_path(&plan) {
                return Ok(system);
            }
            // First schema of the frozen path holding the name (none held: the first reports it).
            let namespace = self
                .view_path
                .as_deref()
                .and_then(|path| {
                    self.path_schema_of(path, bare).or_else(|| {
                        path.iter().find_map(|entry| match entry.as_slice() {
                            [c, n] if c.eq_ignore_ascii_case(CATALOG) => Some(n.clone()),
                            _ => None,
                        })
                    })
                })
                .unwrap_or_else(|| namespace.clone());
            let qualified = ReadSource::Table {
                name: self.view_read_name(namespace, bare.clone()),
                options: options.clone(),
            };
            let plan =
                Arc::new(LogicalPlan::new(PlanNode::Read(qualified)).with_origin(plan.origin));
            return self.expand_stored(plan, scope);
        }
        if let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && options.is_empty()
        {
            // A bare name (not shadowed by a `WITH` definition) is a temporary view; `global_temp.name`
            // is a global one.
            let found = match name.as_slice() {
                [view]
                    if !scope
                        .iter()
                        .any(|c| same_name(c, view, self.case_sensitive)) =>
                {
                    // (Hidden names are rejected above, before this view lookup.)
                    Some((view.as_str(), self.catalog.get(view, self.case_sensitive)))
                }
                [namespace, view] if namespace.eq_ignore_ascii_case(GLOBAL_NAMESPACE) => Some((
                    view.as_str(),
                    self.catalog
                        .get(&global_key(view), self.case_sensitive)
                        .filter(|_| !is_reserved_view_name(view)),
                )),
                _ => None,
            };
            if let Some((view, Some(entry))) = found
                && let TempTable::View {
                    plan: definition,
                    columns,
                    ..
                } = &*entry
            {
                // Debug-quoted parts keep the part boundaries: `a.b` (one part) and `a`.`b` differ.
                let key = format!(
                    "{:?}",
                    name.iter().map(|p| p.to_lowercase()).collect::<Vec<_>>()
                );
                if self.path.contains(&key) {
                    return Err(InternalError {
                        message: format!("temporary view `{}` reads itself", name.join(".")),
                    }
                    .into());
                }
                self.path.push(key);
                // A view's definitions are its own: outer `WITH` names do not
                // reach into it.
                let input = self.go(Arc::clone(definition), &mut Vec::new());
                self.path.pop();
                let columns = columns.iter().map(|c| c.name.clone()).collect();
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::SubqueryAlias {
                        input: input?,
                        // A global view's qualifier is `global_temp.v`, two parts (see
                        // `ALIAS_PART_SEP`).
                        alias: if name.len() == 2 {
                            format!("{GLOBAL_NAMESPACE}{ALIAS_PART_SEP}{view}")
                        } else {
                            view.to_owned()
                        },
                        columns,
                    })
                    .with_origin(plan.origin),
                ));
            }
        }
        if let PlanNode::WithCte { input, ctes } = &plan.node {
            let (input, ctes) = (Arc::clone(input), ctes.clone());
            return self.go_with_cte(plan.origin, input, ctes, scope);
        }
        let plan = self.children_and_exprs(plan, scope)?;
        self.expand_stored(plan, scope)
    }

    /// `plan`, or when it reads a stored view, that view's query as a subquery named for the view.
    pub(super) fn expand_stored(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut [String],
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let Some(stored) = self.stored else {
            return Ok(plan);
        };
        let plan = self.anchored(plan);
        let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node else {
            return Ok(plan);
        };
        if !options.is_empty()
            || scope
                .iter()
                .any(|c| name.len() == 1 && same_name(c, &name[0], self.case_sensitive))
        {
            return Ok(plan);
        }
        // A bare name that is a temporary table or view never reaches a permanent view (Spark's resolution order).
        if self.view_namespace.is_none()
            && name.len() == 1
            && self.catalog.exists(&name[0], self.case_sensitive)
        {
            return Ok(plan);
        }
        let (namespace, table) = match name.as_slice() {
            [table] => {
                // Outside a stored view, the first schema of the path holding the name; inside one,
                // the view's own namespace.
                let path = self.view_path.as_deref().unwrap_or(&stored.path);
                let path = Self::before_system(path, table);
                let from_path = (self.view_namespace.is_none() || self.view_path.is_some())
                    .then(|| self.path_schema_of(path, table))
                    .flatten();
                match (from_path, &self.view_namespace) {
                    (Some(namespace), _) => (namespace, table),
                    (None, Some(_)) => (stored.namespace.clone(), table),
                    // No schema of the path holds the name: no view to inline.
                    (None, None) => return Ok(plan),
                }
            }
            [namespace, table] if self.names_a_stored_namespace() => (namespace.clone(), table),
            [catalog, namespace, table] if catalog.eq_ignore_ascii_case(CATALOG) => {
                (namespace.clone(), table)
            }
            _ => return Ok(plan),
        };
        let (namespace, table_key) = if self.case_sensitive {
            (namespace, table.clone())
        } else {
            (namespace.to_lowercase(), table.to_lowercase())
        };
        let entry = self.catalog.store.table(&namespace, &table_key);
        let candidate = ReadCandidate {
            written: self.read_name(name, &namespace, &table_key),
            view: entry
                .as_ref()
                .filter(|e| e.view.is_some())
                .map(|e| (e.namespace.clone(), e.name.clone())),
        };
        if !self.reads.contains(&candidate) {
            self.reads.push(candidate);
        }
        let Some((entry, view)) = entry.as_ref().and_then(|e| Some((e, e.view.as_ref()?))) else {
            return Ok(plan);
        };
        match self.inline_stored(&plan, entry, view, &namespace, &table_key, scope) {
            Ok(done) => Ok(done),
            Err(e) => {
                // Keep the read as it is; the failure is raised once the reads are authorized.
                self.deferred.get_or_insert(e);
                Ok(plan)
            }
        }
    }

    /// The query of the stored view `entry`, inlined for the read `plan`.
    pub(super) fn inline_stored(
        &mut self,
        plan: &Arc<LogicalPlan>,
        entry: &Arc<CatalogTable>,
        view: &ViewDef,
        namespace: &str,
        table_key: &str,
        _scope: &mut [String],
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let Some(stored) = self.stored else {
            return Ok(Arc::clone(plan));
        };
        let table = match &plan.node {
            PlanNode::Read(ReadSource::Table { name, .. }) => {
                name.last().cloned().unwrap_or_default()
            }
            _ => return Ok(Arc::clone(plan)),
        };
        let key = format!("{namespace}.{table_key}");
        if self.path.contains(&key) {
            return Err(InternalError {
                message: format!("view `{key}` reads itself"),
            }
            .into());
        }
        let statement =
            krishiv_sql::parse_statement(&view.sql, &stored.parser).map_err(|e| e.error)?;
        let krishiv_sql::ast::Statement::Query(query) = statement else {
            return Err(InternalError {
                message: format!("the text of view `{key}` is not a query"),
            }
            .into());
        };
        let lowered = krishiv_sql::lower_identified(
            &query,
            (stored.next_source)(),
            stored.parser.case_sensitive,
            &view.identifier_spans,
        )
        .map_err(|e| e.error)?;
        self.path.push(key);
        let previous = self.view_namespace.replace(view.namespace.clone());
        let previous_path = self.view_path.replace(view.path.clone());
        let input = self.go(Arc::clone(lowered.root()), &mut Vec::new());
        self.view_namespace = previous;
        self.view_path = previous_path;
        self.path.pop();
        let mut input = input?;
        let mode = crate::view_schema::effective_mode(view.schema_mode, stored.binding_enabled);
        let projection = crate::view_schema::projection(mode);
        let mut columns: Vec<String> = entry.columns.iter().map(|c| c.name.clone()).collect();
        self.stored_reads.push(crate::view_schema::StoredRead {
            entry: Arc::clone(entry),
            query: Arc::clone(&input),
            mode,
            renamed: projection != crate::view_schema::ReadProjection::Output
                && view.query_columns.len() == columns.len()
                && crate::view_schema::repeats_names(&view.query_columns),
        });
        if projection == crate::view_schema::ReadProjection::Output {
            // `EVOLUTION`: the view is its query's output, whatever the stored schema says.
            columns.clear();
        } else if view.query_columns.len() == columns.len() {
            // The stored columns, by their query names: a base table that gained columns since does not change
            // what the view returns. In their stored types (the binding check ran on the analyzed query), or in
            // the query's types under `TYPE EVOLUTION`.
            let names;
            (input, names) =
                crate::view_schema::read_names(input, &view.query_columns, &table, plan.origin);
            let cast = projection == crate::view_schema::ReadProjection::Cast;
            let exprs = names
                .iter()
                .zip(&entry.columns)
                .map(|(q, c)| {
                    let to = cast.then(|| c.data_type.clone());
                    crate::view_schema::read_column(q, to, Some(c.comment.clone()), &c.name)
                })
                .collect();
            input = Arc::new(
                LogicalPlan::new(PlanNode::Project { input, exprs }).with_origin(plan.origin),
            );
        }
        // Otherwise (a view stored without its query's column names) the columns are only renamed by position.
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input,
                alias: table,
                columns,
            })
            .with_origin(plan.origin),
        ))
    }

    /// `WITH` query parts (of `origin`) with their views inlined: each
    /// definition expands with only the preceding definitions in scope, as
    /// in Spark, so a body still reads a view of its own name.
    pub(super) fn go_with_cte(
        &mut self,
        origin: Origin,
        input: Arc<LogicalPlan>,
        ctes: Vec<Cte>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let outer = scope.len();
        let mut expanded = Vec::with_capacity(ctes.len());
        for cte in ctes {
            let definition = self.go(Arc::clone(&cte.plan), scope)?;
            scope.push(cte.name.clone());
            expanded.push(Cte {
                name: cte.name,
                plan: definition,
            });
        }
        let input = self.go(input, scope)?;
        scope.truncate(outer);
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::WithCte {
                input,
                ctes: expanded,
            })
            .with_origin(origin),
        ))
    }

    /// `plan` with its child relations and the subquery plans in its
    /// expressions expanded.
    pub(super) fn children_and_exprs(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let plan = plan.map_children(|child| self.go(child, scope))?;
        plan.map_expressions(|expr| self.expand_expr(expr, scope))
    }

    /// `expr` with the views read by the subquery plans inside it inlined.
    pub(super) fn expand_expr(
        &mut self,
        expr: Arc<Expr>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<Expr>, KrishivError> {
        let mut expr = expr.map_children(|child| self.expand_expr(child, scope))?;
        // Inside a stored view the calls of its query resolve through the path it was created under.
        if matches!(expr.node, ExprNode::Function(_))
            && let (Some(qualify), Some(namespace), Some(path)) = (
                self.stored.and_then(|s| s.qualify),
                &self.view_namespace,
                &self.view_path,
            )
            && let Some(next) = qualify(&expr, namespace, path)
        {
            expr = next;
        }
        if let ExprNode::Subquery { kind, plan } = &expr.node {
            let expanded = self.go(Arc::clone(plan), scope)?;
            if !Arc::ptr_eq(&expanded, plan) {
                return Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: expanded,
                    },
                    origin: expr.origin,
                }));
            }
        }
        Ok(expr)
    }
}
