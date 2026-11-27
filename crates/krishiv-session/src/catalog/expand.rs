// SPDX-License-Identifier: Apache-2.0
//! View expansion: the walk that inlines permanent and temporary views into a plan.

use super::*;
use crate::sql_functions::CallRule;
use krishiv_plan::analyzer::views::{ViewExpansion, ViewSecurity};
use krishiv_plan::{CastMode, Expr, ExprNode};

/// [`expand_temp_views`]: the views on the current expansion path, by
/// lower-case name, for cycle detection.
pub(super) struct Expander<'a> {
    pub(super) catalog: &'a SessionCatalog,
    pub(super) case_sensitive: bool,
    pub(super) path: Vec<String>,
    pub(super) stored: Option<&'a StoredViews<'a>>,
    /// How many temporary views the walk has inlined.
    pub(super) inlined_temp: usize,
    /// Inside a stored view: the namespace its bare table names belong to.
    pub(super) view_namespace: Option<String>,
    /// Inside a stored view: the SQL path frozen with it.
    pub(super) view_path: Option<Vec<Vec<String>>>,
    /// Inside a definer view's text: the functions bound when it was created (review r8 item 1).
    pub(super) view_functions: Option<crate::store::BoundFunctions>,
    /// The catalog-store lookups so far, in plan order, without repeats.
    pub(super) reads: Vec<ReadCandidate>,
    /// The first failure inside a stored view's text, kept for [`Expansion::deferred`].
    pub(super) deferred: Option<KrishivError>,
    /// Whether [`Self::deferred`] arose inside a definer view's text (review B2: the reader is told
    /// only its class).
    pub(super) deferred_in_definer: bool,
    /// The stored views inlined so far, for [`Expansion::stored_reads`].
    pub(super) stored_reads: Vec<crate::view_schema::StoredRead>,
    /// The views inlined so far, for [`Expansion::views`] (the lineage side table).
    pub(super) views: Vec<ViewExpansion>,
    /// The definer views inlined so far, for [`Expansion::definers`].
    pub(super) definers: Vec<DefinerRead>,
    /// The definer views whose text the expansion is inside, innermost last (indexes of
    /// [`Self::definers`]): a lookup there is the innermost one's, asked of its definer.
    pub(super) in_definer: Vec<usize>,
    /// How many stored views' texts the expansion is inside: their calls are qualified by
    /// [`StoredViews::qualify`] under a view's rule (a stored function's body, expanded in its
    /// namespace, is qualified under a body's rule, [`StoredViews::stored_body`]).
    pub(super) in_stored_view: usize,
    /// The keeps of the temporary view entries inlined so far, for [`Expansion::keeps`].
    pub(super) keeps: Vec<Keep>,
    /// The stored views inlined so far, each with the definer view whose text holds it (an index
    /// of [`Self::definers`], `None` outside one), for [`Expansion::reached`].
    reached: Vec<(Option<usize>, ReachedView)>,
    /// The inlined query of each temporary view read, for [`Expansion::temp_texts`].
    temp_texts: Vec<Arc<LogicalPlan>>,
    /// The stored views whose text the expansion is inside (indexes of [`Self::reached`]).
    open_views: Vec<usize>,
}

/// A stored view an expansion inlined (review r10 M10): inside a definer view's trusted text it is
/// checked against the version the definer view pinned, and its own reads are asked of its owner
/// and author too.
#[derive(Debug, Clone)]
pub(crate) struct ReachedView {
    /// The namespace and name the catalog stores.
    pub(crate) view: (String, String),
    /// [`ViewDef::revision`] of the entry inlined.
    pub(crate) revision: u64,
    /// [`ViewDef::author`] of the entry inlined.
    pub(crate) author: String,
    /// The catalog-store lookups of its text and of the invoker views inside it, not those inside
    /// a definer view in it (which are that definer's).
    pub(crate) reads: Vec<ReadCandidate>,
    /// Its text as inlined (`None` when it failed).
    pub(crate) query: Option<Arc<LogicalPlan>>,
}

impl<'a> Expander<'a> {
    /// An expansion of a plan written inside `namespace` (none: the query's own) under `view_path`.
    pub(super) fn new(
        catalog: &'a SessionCatalog,
        case_sensitive: bool,
        stored: Option<&'a StoredViews<'a>>,
        namespace: Option<String>,
        view_path: Option<Vec<Vec<String>>>,
    ) -> Self {
        Self {
            catalog,
            case_sensitive,
            path: Vec::new(),
            stored,
            inlined_temp: 0,
            view_namespace: namespace,
            view_path,
            view_functions: None,
            reads: Vec::new(),
            deferred: None,
            deferred_in_definer: false,
            stored_reads: Vec::new(),
            views: Vec::new(),
            definers: Vec::new(),
            in_definer: Vec::new(),
            in_stored_view: 0,
            keeps: Vec::new(),
            reached: Vec::new(),
            temp_texts: Vec::new(),
            open_views: Vec::new(),
        }
    }

    /// What the expansion found, with its expanded `plan`.
    pub(super) fn finish(mut self, plan: Arc<LogicalPlan>) -> Expansion {
        for (definer, view) in &self.reached {
            if let Some(d) = definer.and_then(|i| self.definers.get_mut(i)) {
                d.views.push(view.clone());
            }
        }
        Expansion {
            plan,
            reads: self.reads,
            deferred: self.deferred,
            deferred_in_definer: self.deferred_in_definer,
            stored_reads: self.stored_reads,
            views: self.views,
            definers: self.definers,
            keeps: self.keeps,
            reached: self.reached.into_iter().map(|(_, v)| v).collect(),
            temp_texts: self.temp_texts,
        }
    }
}

/// A view read the expansion inlined as `SubqueryAlias alias` at `origin`, for the view side table: a
/// temporary view or a permanent `INVOKER` view, read with its base tables checked against the caller (the
/// analyzer binds them as the query's own reads), or a permanent `DEFINER` view, whose base tables are checked
/// against its definer (P3-T062 §2.4). The expansion has no memo, so a read shared by two parents of the plan
/// (`df.join(df)`) is inlined, and recorded, twice.
fn view_read(
    name: Vec<String>,
    alias: &str,
    origin: Origin,
    security: ViewSecurity,
) -> ViewExpansion {
    ViewExpansion::new(name, alias.to_owned(), origin, security, true)
}

impl Expander<'_> {
    /// Refuses the time travel clause where the relation it rides on resolves to something Spark cannot travel:
    /// a `WITH` definition and a temporary view (named as written), and a path (`format`.`path`). A table and a
    /// permanent view are refused where they resolve (`expand_stored`, and the read check for a table).
    fn refuse_time_travel_here(
        &self,
        plan: &LogicalPlan,
        scope: &[String],
    ) -> Result<(), KrishivError> {
        let PlanNode::Read(source) = &plan.node else {
            return Ok(());
        };
        let relation_id = match source {
            ReadSource::Files {
                format,
                paths,
                options,
                ..
            } if options.contains_key(krishiv_plan::TIME_TRAVEL_OPTION) => {
                let path = paths
                    .first()
                    .and_then(|p| p.as_plain())
                    .unwrap_or("<redacted>");
                format!("{}.{}", quote(format), quote(path))
            }
            ReadSource::Table { name, options }
                if options.contains_key(krishiv_plan::TIME_TRAVEL_OPTION) =>
            {
                let bare = match name.as_slice() {
                    [bare] => bare,
                    [namespace, view] if namespace.eq_ignore_ascii_case(GLOBAL_NAMESPACE) => {
                        if self
                            .catalog
                            .get(&global_key(view), self.case_sensitive)
                            .is_none()
                        {
                            return Ok(());
                        }
                        return Err(krishiv_common::error::UnsupportedFeatureTimeTravel {
                            relation_id: format!("{}.{}", quote(GLOBAL_NAMESPACE), quote(view)),
                        }
                        .into());
                    }
                    _ => return Ok(()),
                };
                let cte = scope
                    .iter()
                    .any(|c| same_name(c, bare, self.case_sensitive));
                let temporary =
                    self.view_namespace.is_none() && self.catalog.exists(bare, self.case_sensitive);
                if !cte && !temporary {
                    return Ok(());
                }
                quote(bare)
            }
            _ => return Ok(()),
        };
        Err(krishiv_common::error::UnsupportedFeatureTimeTravel { relation_id }.into())
    }

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
                search_path: crate::catalog::search_path_in(
                    &self
                        .view_namespace
                        .clone()
                        .unwrap_or_else(|| self.catalog.current_namespace()),
                ),
            }
            .into());
        }
        // A `SQL` relation (Spark Connect's `spark.sql(..)` query, sent back as a root relation) is
        // lowered by the analyzer, after this walk: its text's views would never be inlined. Lower it
        // here and expand that; text that does not parse, or reads no view, stays as the analyzer's
        // own to report and lower.
        if let PlanNode::Sql {
            query,
            args: krishiv_plan::SqlArgs::None,
        } = &plan.node
            && let Some(stored) = self.stored
            && let Ok(krishiv_sql::ast::Statement::Query(parsed)) =
                krishiv_sql::parse_statement(query, &stored.parser)
            && let Ok(lowered) = krishiv_sql::lower_with(
                &parsed,
                (stored.next_source)(),
                stored.parser.case_sensitive,
            )
        {
            // A trial walk on a copy of the walk's inputs: only a text that reads a view is replaced
            // (its reads and checks are then recorded by the real walk, once).
            let mut trial = Expander::new(
                self.catalog,
                self.case_sensitive,
                self.stored,
                self.view_namespace.clone(),
                self.view_path.clone(),
            );
            trial.go(Arc::clone(lowered.root()), &mut scope.clone())?;
            if trial.inlined_temp > 0 || !trial.reached.is_empty() {
                return self.go(Arc::clone(lowered.root()), scope);
            }
            return Ok(plan);
        }
        // After the reserved-name guard: a read of a hidden name is a missing table, options or not.
        self.refuse_time_travel_here(&plan, scope)?;
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
                    keeps,
                    ..
                } = &*entry
            {
                self.inlined_temp += 1;
                self.keeps.extend(keeps.0.iter().cloned());
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
                let input = input?;
                self.temp_texts.push(Arc::clone(&input));
                let columns = columns.iter().map(|c| c.name.clone()).collect();
                // A global view's qualifier is `global_temp.v`, two parts (see `ALIAS_PART_SEP`).
                let alias = if name.len() == 2 {
                    format!("{GLOBAL_NAMESPACE}{ALIAS_PART_SEP}{view}")
                } else {
                    view.to_owned()
                };
                self.views.push(view_read(
                    name.clone(),
                    &alias,
                    plan.origin,
                    ViewSecurity::Temporary,
                ));
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::SubqueryAlias {
                        input,
                        alias,
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
        let plan = self.qualified_table_function(plan);
        self.expand_stored(plan, scope)
    }

    /// Where the calls of the plan being expanded resolve, when not through the reader's session:
    /// inside a stored view's text, its namespace and frozen path (a view's rule); else, in a stored
    /// function's body, the function's namespace and frozen path (a body's rule, review r7 R7-1).
    /// The last field: the rule, a definer view's text calling only the functions it bound when
    /// it was created (review r8 item 1).
    fn call_scope(&self) -> Option<(&str, &[Vec<String>], CallRule<'_>)> {
        let namespace = self.view_namespace.as_deref()?;
        if self.in_stored_view > 0 {
            let rule = match &self.view_functions {
                Some(bound) => CallRule::Bound(bound),
                None => CallRule::View,
            };
            return Some((namespace, self.view_path.as_deref()?, rule));
        }
        self.stored.filter(|s| s.stored_body).map(|_| {
            (
                namespace,
                self.view_path.as_deref().unwrap_or_default(),
                CallRule::Body,
            )
        })
    }

    /// `plan`, or inside a stored view's text or a stored function's body, when it calls a table
    /// function by a bare name, the call qualified as its scalar calls are (review r6 R6-1, r7
    /// R7-1, [`StoredViews::qualify_table`]).
    fn qualified_table_function(&self, plan: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        let PlanNode::TableFunction { name, args } = &plan.node else {
            return plan;
        };
        let (Some(qualify), Some((namespace, path, rule))) =
            (self.stored.and_then(|s| s.qualify_table), self.call_scope())
        else {
            return plan;
        };
        match qualify(name, namespace, path, rule) {
            Some(name) => Arc::new(
                LogicalPlan::new(PlanNode::TableFunction {
                    name,
                    args: args.clone(),
                })
                .with_origin(plan.origin),
            ),
            None => plan,
        }
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
        let travel = options.contains_key(krishiv_plan::TIME_TRAVEL_OPTION);
        // Other options beside the clause still reach the refusal of a view with it (Spark reports the clause).
        if (!travel
            && options
                .keys()
                .any(|k| k != krishiv_plan::TIME_TRAVEL_OPTION))
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
        // A lookup in a stored view's text is its owner's too (review r10 M10): recorded for each
        // view whose text holds it, up to the innermost definer view's text.
        let innermost = self.in_definer.last().copied();
        for &open in &self.open_views {
            if let Some((definer, view)) = self.reached.get_mut(open)
                && *definer == innermost
                && !view.reads.contains(&candidate)
            {
                view.reads.push(candidate.clone());
            }
        }
        // Inside a definer view's text a lookup is the definer's to be allowed, not the reader's.
        let reads = match self.in_definer.last() {
            Some(&i) => match self.definers.get_mut(i) {
                Some(d) => &mut d.reads,
                None => &mut self.reads,
            },
            None => &mut self.reads,
        };
        if !reads.contains(&candidate) {
            reads.push(candidate);
        }
        let Some((entry, view)) = entry.as_ref().and_then(|e| Some((e, e.view.as_ref()?))) else {
            return Ok(plan);
        };
        if travel {
            // Spark's `failTimeTravel` names the view in full. Raised once the reads are authorized
            // (as a failure inside the view's text is), so a caller who may not read the view is
            // denied exactly as for a missing one: no existence oracle.
            if self.deferred.is_none() {
                self.deferred_in_definer = !self.in_definer.is_empty() || view.definer.is_some();
                self.deferred = Some(
                    krishiv_common::error::UnsupportedFeatureTimeTravel {
                        relation_id: format!(
                            "{}.{}.{}",
                            quote(CATALOG),
                            quote(&entry.namespace),
                            quote(&entry.name)
                        ),
                    }
                    .into(),
                );
            }
            return Ok(plan);
        }
        match self.inline_stored(&plan, entry, view, &namespace, &table_key, scope) {
            Ok(done) => Ok(done),
            Err(e) => {
                // Keep the read as it is; the failure is raised once the reads are authorized.
                if self.deferred.is_none() {
                    // A failure of a definer view's own text (its `WITH`s under its policy, a name in it) is
                    // the definer's too, though its text is no longer on the path once it fails.
                    self.deferred_in_definer =
                        !self.in_definer.is_empty() || view.definer.is_some();
                    self.deferred = Some(e);
                }
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
        if let crate::store::ViewKind::Metric(def) = &view.kind {
            // Its `sql` is YAML, never a query.
            return self.inline_metric(plan, entry, view, def, namespace, table_key, key, stored);
        }
        if self.path.contains(&key) {
            return Err(InternalError {
                message: format!("view `{key}` reads itself"),
            }
            .into());
        }
        // A bare `TIMESTAMP` in the text is the type it was when the view was created, never the
        // reader's (Spark parses and resolves a view under its captured `view.sqlConfig.*`).
        let parser = krishiv_sql::ParserOptions {
            timestamp_ntz: view.timestamp_ntz.unwrap_or(false),
            ..stored.parser
        };
        let statement = krishiv_sql::parse_statement(&view.sql, &parser).map_err(|e| e.error)?;
        let krishiv_sql::ast::Statement::Query(query) = statement else {
            return Err(InternalError {
                message: format!("the text of view `{key}` is not a query"),
            }
            .into());
        };
        let source = (stored.next_source)();
        let lowered = krishiv_sql::lower_identified(
            &query,
            source,
            stored.parser.case_sensitive,
            &view.identifier_spans,
        )
        .map_err(|e| e.error)?;
        // The view's own `WITH`s resolve under the policy captured with it, in its text alone, before
        // expansion (review r4 Y1; Spark: `View.effectiveSQLConf(viewSQLConfigs, ..)`): no reader's policy
        // or definition changes what its text reads, and a read the text's own policy leaves to a table is
        // qualified to the view's namespace below like any other.
        let text = krishiv_plan::analyzer::inline_view_ctes(
            lowered.root(),
            view.cte_precedence.unwrap_or_default(),
            stored.cte_precedence,
            self.case_sensitive,
        )?;
        let name = vec![
            CATALOG.to_owned(),
            namespace.to_owned(),
            table_key.to_owned(),
        ];
        // Its version and its own lookups, for the definer view whose text holds it (review r10 M10).
        self.reached.push((
            self.in_definer.last().copied(),
            ReachedView {
                view: (entry.namespace.clone(), entry.name.clone()),
                revision: view.revision,
                author: view.author.clone(),
                reads: Vec::new(),
                query: None,
            },
        ));
        let reached = self.reached.len() - 1;
        self.open_views.push(reached);
        // A definer view's text is read with its definer's rights: its lookups are recorded as its own.
        let definer = view.definer.as_ref().map(|definer| {
            self.definers.push(DefinerRead {
                name: name.clone(),
                definer: definer.clone(),
                query: None,
                reads: Vec::new(),
                view: 0,
                namespace: view.namespace.clone(),
                path: view.path.clone(),
                bound: view.functions.clone().unwrap_or_default(),
                views: Vec::new(),
            });
            self.definers.len() - 1
        });
        self.in_definer.extend(definer);
        self.path.push(key);
        let previous = self.view_namespace.replace(view.namespace.clone());
        let previous_path = self.view_path.replace(view.path.clone());
        // A definer view calls what it bound when created; an invoker view resolves at each read.
        let previous_functions = std::mem::replace(
            &mut self.view_functions,
            view.definer
                .as_ref()
                .map(|_| view.functions.clone().unwrap_or_default()),
        );
        self.in_stored_view += 1;
        let input = self.go(text, &mut Vec::new());
        self.in_stored_view -= 1;
        self.open_views.pop();
        self.view_namespace = previous;
        self.view_path = previous_path;
        self.view_functions = previous_functions;
        self.path.pop();
        if definer.is_some() {
            self.in_definer.pop();
        }
        let mut input = input?;
        if let Some((_, r)) = self.reached.get_mut(reached) {
            r.query = Some(Arc::clone(&input));
        }
        if let Some(d) = definer.and_then(|i| self.definers.get_mut(i)) {
            d.query = Some(Arc::clone(&input));
            d.view = self.views.len();
        }
        let mode = crate::view_schema::effective_mode(view.schema_mode, stored.binding_enabled);
        let projection = crate::view_schema::projection(mode);
        let mut columns: Vec<String> = entry.columns.iter().map(|c| c.name.clone()).collect();
        self.stored_reads.push(crate::view_schema::StoredRead {
            entry: Arc::clone(entry),
            query: Arc::clone(&input),
            mode,
            in_definer: definer.is_some() || !self.in_definer.is_empty(),
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
        self.views.push(view_read(
            name,
            &table,
            plan.origin,
            if definer.is_some() {
                ViewSecurity::Definer
            } else {
                ViewSecurity::Invoker
            },
        ));
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

    /// A read of the metric view `entry` (Spark's `MetricViewPlaceholder`, design note
    /// `P3-metric-views.md` §4): its YAML read again under the parse options captured with it, the
    /// source with its filter expanded in the reader's context (the reader's namespace, path and
    /// temporary views; inside an ordinary view, that view's namespace and path, as Spark 4.2.1),
    /// and the dimensions and measures parsed, left for the analyzer's rule.
    #[expect(
        clippy::too_many_arguments,
        reason = "the parts of the read inline_stored already took apart"
    )]
    fn inline_metric(
        &mut self,
        plan: &Arc<LogicalPlan>,
        entry: &Arc<CatalogTable>,
        view: &ViewDef,
        def: &crate::store::MetricViewDef,
        namespace: &str,
        table_key: &str,
        key: String,
        stored: &StoredViews<'_>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        if self.path.contains(&key) {
            return Err(InternalError {
                message: format!("view `{key}` reads itself"),
            }
            .into());
        }
        let mut spec = crate::metric_view::parse_metric_view_yaml(&view.sql, &def.parser)?;
        crate::metric_view::bare_expressions(&mut spec, &def.parser);
        let mut text = String::from("SELECT *");
        spec.push_from(&mut text);
        let query = krishiv_sql::parse_query(&text, &def.parser).map_err(|e| e.error)?;
        let lowered = krishiv_sql::lower_identified(
            &query,
            (stored.next_source)(),
            def.parser.case_sensitive,
            &[],
        )
        .map_err(|e| e.error)?;
        // The source with its filter: below the `SELECT *`.
        let source = match &lowered.root().node {
            PlanNode::Project { input, .. } => Arc::clone(input),
            _ => Arc::clone(lowered.root()),
        };
        let source = with_cast_modes(&source, def.ansi);
        // Its version and its own lookups, for the definer view whose text holds it (review r10 M10),
        // as a SQL view's; the side table of `analyzer/views.rs` keys by a `SubqueryAlias` this read
        // has none of, so lineage shows its source as opaque instead (`lineage/walk.rs`).
        self.reached.push((
            self.in_definer.last().copied(),
            ReachedView {
                view: (entry.namespace.clone(), entry.name.clone()),
                revision: view.revision,
                author: view.author.clone(),
                reads: Vec::new(),
                query: None,
            },
        ));
        let reached = self.reached.len() - 1;
        self.open_views.push(reached);
        self.path.push(key);
        let source = self.go(source, &mut Vec::new());
        self.path.pop();
        self.open_views.pop();
        let source = source?;
        if let Some((_, r)) = self.reached.get_mut(reached) {
            r.query = Some(Arc::clone(&source));
        }
        let mut columns = entry.columns.iter();
        let mut parsed = |c: &crate::metric_view::MetricViewColumn| {
            let item =
                krishiv_sql::parse_named_expression(&c.expr, &def.parser).map_err(|e| e.error)?;
            let expr = krishiv_sql::lower_expression(&item.expr, (stored.next_source)())
                .map_err(|e| e.error)?;
            let expr = cast_modes(Arc::new(expr), def.ansi);
            let stored_column = columns.next();
            Ok::<_, KrishivError>(krishiv_plan::MetricColumn {
                name: stored_column.map_or_else(|| c.name.clone(), |s| s.name.clone()),
                expr,
                data_type: stored_column
                    .map_or(krishiv_types::DataType::Null, |s| s.data_type.clone()),
                nullable: stored_column.is_none_or(|s| s.nullable),
            })
        };
        let dimensions = spec
            .dimensions
            .iter()
            .map(&mut parsed)
            .collect::<Result<Vec<_>, _>>()?;
        let measures = spec
            .measures
            .iter()
            .map(&mut parsed)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::MetricViewRead {
                input: source,
                name: vec![
                    CATALOG.to_owned(),
                    namespace.to_owned(),
                    table_key.to_owned(),
                ],
                dimensions,
                measures,
                output: Vec::new(),
                ansi: Some(def.ansi),
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
        // Inside a stored view (or a stored function's body) the calls of its query resolve through
        // the path it was created under.
        if matches!(expr.node, ExprNode::Function(_))
            && let (Some(qualify), Some((namespace, path, rule))) =
                (self.stored.and_then(|s| s.qualify), self.call_scope())
            && let Some(next) = qualify(&expr, namespace, path, rule)
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

/// The view side table (P3-T063, review m4): each view inlined here is recorded, once per expansion, and
/// located in the analyzed plan by its (alias, origin).
#[cfg(test)]
#[path = "expand_tests.rs"]
mod view_side_table;

/// `e` with each `CAST` it writes in the mode `ansi` gives (a metric view's casts take the mode
/// captured when it was created, as Spark's parser builds them; `TRY_CAST` stays).
fn cast_modes(e: Arc<Expr>, ansi: bool) -> Arc<Expr> {
    let done: Result<Arc<Expr>, std::convert::Infallible> = e.transform_up(&mut |x| {
        Ok(match &x.node {
            ExprNode::Cast {
                expr,
                to,
                mode: CastMode::Default,
            } => Arc::new(
                Expr::new(ExprNode::Cast {
                    expr: Arc::clone(expr),
                    to: to.clone(),
                    mode: if ansi {
                        CastMode::Ansi
                    } else {
                        CastMode::Legacy
                    },
                })
                .with_origin(x.origin),
            ),
            _ => x,
        })
    });
    done.unwrap_or_else(|never| match never {})
}

/// [`cast_modes`] of every expression of `plan` (a metric view's source and filter).
fn with_cast_modes(plan: &Arc<LogicalPlan>, ansi: bool) -> Arc<LogicalPlan> {
    let done: Result<Arc<LogicalPlan>, std::convert::Infallible> =
        Arc::clone(plan).transform_up(&mut |n| n.map_expressions(|e| Ok(cast_modes(e, ansi))));
    done.unwrap_or_else(|never| match never {})
}
