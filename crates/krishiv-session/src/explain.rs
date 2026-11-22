// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN [EXTENDED | FORMATTED | COST | CODEGEN]` and
//! `DataFrame::explain`: the plans of a query as Spark 4.2 lays them out
//! (the section headers, which sections each mode prints, and Spark's
//! handling of a query that does not plan). The nodes themselves are
//! Krishiv's operators, not Spark's (ADR 0023).

use std::fmt::Write as _;
use std::sync::Arc;

use krishiv_common::error::{ErrorClass, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::{ExplainNode, PhysicalPlan, ProviderStatistics, TableProviderResolver};
use krishiv_optimizer::PlanStats;
use krishiv_plan::{LogicalPlan, Plan};
use krishiv_sql::ast::{Explain, SetKind, Statement, TimeZone};

use crate::dataframe::{Analysis, DataFrame, ExplainMode};
use crate::session::Session;

pub(crate) mod withheld;

impl From<krishiv_sql::ast::ExplainMode> for ExplainMode {
    fn from(mode: krishiv_sql::ast::ExplainMode) -> Self {
        use krishiv_sql::ast::ExplainMode as Sql;
        match mode {
            Sql::Extended => Self::Extended,
            Sql::Formatted => Self::Formatted,
            Sql::Codegen => Self::Codegen,
            Sql::Cost => Self::Cost,
            _ => Self::Simple,
        }
    }
}

/// Spark's text for a query that does not plan under the simple and
/// formatted modes (its message is not part of it).
const PLANNING_ERROR: &str = "Error occurred during query planning: ";

/// The stages of a query that planned.
struct Planned {
    analysis: Arc<Analysis>,
    optimized: Plan,
    physical: ExplainNode,
}

impl Session {
    /// Answers `EXPLAIN` as `spark.sql` does: eagerly, as a one-row
    /// `plan STRING` frame. A command (`SET`, `RESET`, `CREATE TEMPORARY
    /// VIEW`, `DROP VIEW`) explains as its own plan text. A query that
    /// fails to analyze or plan is part of the text, not an error; a
    /// syntax error is an error.
    pub(crate) async fn explain_statement(
        &self,
        e: &Explain,
        env: crate::session::QueryEnv,
        text: &str,
        redacted: &krishiv_common::redact::RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        // Krishiv's `EXPLAIN FORMAT JSON` (P3-T063 N2): the explained
        // statement's analyze document. It is analyzed, never run; its
        // reads are checked as a run checks them.
        if e.mode == krishiv_sql::ast::ExplainMode::Json {
            let json = self
                .analyze_statement((*e.statement).clone(), text, redacted)
                .await?
                .to_json();
            return self.one_row(&json).await;
        }
        if !matches!(&*e.statement, Statement::Query(_)) {
            let text = command_plan(&e.statement)?;
            return self.one_row(&text).await;
        }
        let Statement::Query(q) = &*e.statement else {
            unreachable!("an explained query");
        };
        let plan = krishiv_sql::lower_identified(
            q,
            krishiv_plan::SourceId(0),
            env.parser.case_sensitive,
            &env.identifier_spans,
        )
        .map_err(|e| e.error)?;
        // The per-query store check (`check_store_options`: `MANAGE` on the cluster while the
        // session changed a server-scoped store setting) is asked by every analysis: the explained
        // query's (its trial, then a withheld fallback, which folds a refusal into the text) and
        // the result frame's. Asked once here, after lowering (a query that does not lower gets
        // its lowering error) and before either, so one statement gives one refusal report: one
        // `denied`, one audit record, one count. The same refusal the result frame gave, so the
        // caller sees the same error.
        let r = self
            .check_store_options(&*env.analysis.policy, &env.config, std::iter::empty())
            .await;
        crate::access::audit_read(self, &env, r)?;
        let explained = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let text = match e.mode {
            // Krishiv's extensions (ADR 0023): a query that fails to
            // analyze, plan or run is an error, not part of the text.
            krishiv_sql::ast::ExplainMode::Analyze => {
                // The query runs here, so its row shows the EXPLAIN statement and its class.
                // The profile asks what its definer views' queries read (`DataFrame::profiled`).
                explained.analyzed_in_with_text(env, Some(redacted)).await?;
                explained
                    .profiled(false)
                    .await?
                    .1
                    .to_text(krishiv_exec::profile::TextOptions::default())
            }
            krishiv_sql::ast::ExplainMode::Verbose => explained.verbose().await?.to_text(),
            mode => {
                explained
                    .explain_for_metadata_text(mode.into(), true)
                    .await?
            }
        };
        // Spark's planning failure is one row per line of its text (`ExplainCommand`), as the
        // metric-view refusals give it (`failed`).
        if let Some(rest) = text.strip_prefix(PLANNING_ERROR)
            && rest.starts_with("\n[INTERNAL_ERROR]")
        {
            let rows = std::iter::once(PLANNING_ERROR)
                .chain(rest.lines().skip(1))
                .map(|l| vec![Some(l.to_owned())])
                .collect();
            return self
                .string_frame(crate::analyze::command_schema::EXPLAIN, rows)
                .await;
        }
        self.one_row(&text).await
    }
}

/// The plan text of an explained command: the command itself, under the
/// physical-plan header Spark prints.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an `EXPLAIN` of an `EXPLAIN`.
fn command_plan(statement: &Statement) -> Result<String, KrishivError> {
    let command = match statement {
        Statement::Set(s) => match &s.kind {
            SetKind::All => "SET".to_owned(),
            SetKind::Verbose => "SET -v".to_owned(),
            SetKind::Show { key, .. } => format!("SET {key}"),
            SetKind::Assign { key, value, .. } => format!("SET {key}={value}"),
            SetKind::Path { .. } => "SET PATH".to_owned(),
            SetKind::TimeZone { zone, .. } => match zone {
                TimeZone::Local => "SET TIME ZONE LOCAL".to_owned(),
                TimeZone::Named(zone) => format!("SET TIME ZONE '{zone}'"),
                _ => {
                    return Err(KrishivUnsupportedFeature {
                        feature: "EXPLAIN of this SET form in sessions".to_owned(),
                    }
                    .into());
                }
            },
            _ => {
                return Err(KrishivUnsupportedFeature {
                    feature: "EXPLAIN of this SET form in sessions".to_owned(),
                }
                .into());
            }
        },
        Statement::Reset(r) => match &r.key {
            None => "RESET".to_owned(),
            Some(key) => format!("RESET {key}"),
        },
        Statement::CreateView(v) => {
            let mut text = if v.or_replace {
                "CREATE OR REPLACE TEMPORARY VIEW ".to_owned()
            } else {
                "CREATE TEMPORARY VIEW ".to_owned()
            };
            text.push_str(&format!("`{}`", v.name.value.replace('`', "``")));
            if !v.columns.is_empty() {
                let columns: Vec<String> = v
                    .columns
                    .iter()
                    .map(|c| format!("`{}`", c.name.value.replace('`', "``")))
                    .collect();
                text.push_str(&format!(" ({})", columns.join(", ")));
            }
            text.push_str(" AS ...");
            text
        }
        Statement::DropView(v) => {
            if v.if_exists {
                format!("DROP VIEW IF EXISTS `{}`", v.name.value.replace('`', "``"))
            } else {
                format!("DROP VIEW `{}`", v.name.value.replace('`', "``"))
            }
        }
        Statement::DeclareVariable(v) => {
            let mut text = if v.or_replace {
                "DECLARE OR REPLACE ".to_owned()
            } else {
                "DECLARE ".to_owned()
            };
            text.push_str(
                &v.names
                    .iter()
                    .map(|n| {
                        n.parts
                            .iter()
                            .map(|p| format!("`{}`", p.value.replace('`', "``")))
                            .collect::<Vec<_>>()
                            .join(".")
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            text.push_str(" AS ...");
            text
        }
        Statement::SetVariable(_) => "SET VARIABLE ...".to_owned(),
        Statement::DropVariable(v) => {
            let name = v
                .name
                .parts
                .iter()
                .map(|p| format!("`{}`", p.value.replace('`', "``")))
                .collect::<Vec<_>>()
                .join(".");
            if v.if_exists {
                format!("DROP TEMPORARY VARIABLE IF EXISTS {name}")
            } else {
                format!("DROP TEMPORARY VARIABLE {name}")
            }
        }
        Statement::Explain(_) => {
            return Err(KrishivUnsupportedFeature {
                feature: "EXPLAIN of an EXPLAIN in sessions".to_owned(),
            }
            .into());
        }
        _ => {
            return Err(KrishivUnsupportedFeature {
                feature: "EXPLAIN of this statement in sessions".to_owned(),
            }
            .into());
        }
    };
    Ok(format!("== Physical Plan ==\n{command}\n"))
}

/// A frame of one row, the string column `plan`.
impl Session {
    async fn one_row(&self, text: &str) -> Result<DataFrame, KrishivError> {
        self.string_frame(
            crate::analyze::command_schema::EXPLAIN,
            vec![vec![Some(text.to_owned())]],
        )
        .await
    }
}

/// What a statement would cost, estimated before it runs (ADR 0041).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Estimate {
    /// The bytes its table scans would read, estimated from the tables' statistics (rows times Spark's default
    /// column widths, as `EXPLAIN COST`); `None` when a table has no statistics: unknown.
    pub scan_bytes: Option<u64>,
    /// The estimated rows of the result; `None` when unknown.
    pub rows: Option<u64>,
    /// The physical plan outline, one line per node (engine-generated text, no cell content).
    pub plan: Vec<String>,
}

impl Session {
    /// The estimate of what the SQL `query` would scan, before it runs anything (ADR 0041): for a query, the
    /// query; for `INSERT ... SELECT`, the source query (the write itself is not planned, so nothing is written).
    /// `None` for any other statement (DDL, commands, other writes): there is nothing to estimate.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let e = futures::executor::block_on(s.estimate_sql("SELECT a FROM t"))?;
    /// assert!(e.is_some());
    /// assert!(futures::executor::block_on(s.estimate_sql("SET a.b = 1"))?.is_none());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Parse, analysis, access-policy and planning errors of the query, as for `sql`.
    pub async fn estimate_sql(&self, query: &str) -> Result<Option<Estimate>, KrishivError> {
        let (session, query) = (self.clone(), query.to_owned());
        self.on_executor(async move { session.estimate_sql_here(&query).await })
            .await
    }

    async fn estimate_sql_here(&self, query: &str) -> Result<Option<Estimate>, KrishivError> {
        let env = self.query_env();
        let query = self.substituted(query);
        let resolved = self.resolve_identifier_clauses(&query, &env.parser).await?;
        let redacted = krishiv_sql::redact_sql(&resolved);
        let statement = krishiv_sql::parse_statement(&resolved, &env.parser).map_err(|e| {
            crate::session::sql_context(e.error, &redacted, e.span.start, e.span.end)
        })?;
        // `EXPLAIN ANALYZE` runs its statement: what it reads is what the
        // statement reads.
        let mut statement = &statement;
        while let Statement::Explain(e) = statement
            && e.mode == krishiv_sql::ast::ExplainMode::Analyze
        {
            statement = &e.statement;
        }
        let source = match statement {
            Statement::Query(q) => q,
            Statement::Insert(i) if i.source.is_none() && i.clauses.len() == 1 => {
                &i.clauses[0].query
            }
            _ => return Ok(None),
        };
        let plan = krishiv_sql::lower(source, krishiv_plan::SourceId(0)).map_err(|e| {
            crate::session::sql_context(e.error, &redacted, e.span.start, e.span.end)
        })?;
        DataFrame::new(self.clone(), Arc::clone(plan.root()))
            .estimate_inner()
            .await
            .map(Some)
    }
}

impl DataFrame {
    /// Plans this statement as an action would and estimates what it reads, without running it.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let e = futures::executor::block_on(s.table("t").estimate())?;
    /// assert!(!e.plan.is_empty());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis, access-policy and planning errors, as for an action.
    pub async fn estimate(&self) -> Result<Estimate, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.estimate_inner().await })
            .await
    }

    async fn estimate_inner(&self) -> Result<Estimate, KrishivError> {
        let analysis = self.analysis().await?;
        let (optimized, physical) = self
            .session
            .physical_inspected(&analysis.analyzed, &analysis.env)
            .await?;
        Ok(estimate_of(&optimized, &physical, &analysis.env.catalog))
    }
}

/// The estimate of an optimized plan and its physical plan, from the statistics `catalog` gives.
pub(crate) fn estimate_of(
    optimized: &Plan,
    physical: &PhysicalPlan,
    catalog: &Arc<crate::catalog::QueryCatalog>,
) -> Estimate {
    let stats = ProviderStatistics(Arc::clone(catalog) as Arc<dyn TableProviderResolver>);
    let estimates = PlanStats::compute(optimized.root(), &stats);
    // Each scan counts once per path that reaches it: the two sides of a self-join share one subtree and read
    // it twice.
    let mut order: Vec<Arc<LogicalPlan>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![(Arc::clone(optimized.root()), false)];
    while let Some((p, expanded)) = stack.pop() {
        if expanded {
            order.push(p);
            continue;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        stack.push((Arc::clone(&p), true));
        for c in plan_children(&p) {
            stack.push((c, false));
        }
    }
    let mut paths: std::collections::HashMap<*const LogicalPlan, u64> =
        std::collections::HashMap::new();
    paths.insert(Arc::as_ptr(optimized.root()), 1);
    let mut scan_bytes = Some(0.0_f64);
    for p in order.iter().rev() {
        let n = paths.get(&Arc::as_ptr(p)).copied().unwrap_or(0);
        if matches!(p.node, krishiv_plan::PlanNode::Scan(_)) {
            scan_bytes = match (
                scan_bytes,
                estimates.has_statistics(p),
                estimates.size_bytes(p),
            ) {
                (Some(total), true, Some(size)) => Some(total + size * (n as f64)),
                _ => None,
            };
        }
        for c in plan_children(p) {
            let slot = paths.entry(Arc::as_ptr(&c)).or_insert(0);
            *slot = slot.saturating_add(n);
        }
    }
    let root = optimized.root();
    let rows = estimates
        .has_statistics(root)
        .then(|| estimates.rows(root))
        .map(f64_to_u64);
    let plan = physical
        .explain_all()
        .to_string()
        .lines()
        .map(str::to_owned)
        .collect();
    Estimate {
        scan_bytes: scan_bytes.map(f64_to_u64),
        rows,
        plan,
    }
}

impl DataFrame {
    /// The text of `mode`. With `lenient` (SQL `EXPLAIN`) a query that does
    /// not analyze or plan is described in the text, as Spark does;
    /// otherwise (`DataFrame.explain`) it is the error.
    pub(crate) async fn explain_text(
        &self,
        mode: ExplainMode,
        lenient: bool,
    ) -> Result<String, KrishivError> {
        let parsed = self.plan.to_string();
        let analysis = match self.analysis().await {
            Ok(a) => a,
            Err(e) if lenient => return Ok(failed(mode, &parsed, None, &e)),
            Err(e) => return Err(e),
        };
        // Before any section shows the analyzed plan: it holds the queries of its definer views
        // (P3-T062 §2.4).
        match self
            .session
            .authorize_shown_definer_reads(&analysis.analyzed)
            .await
        {
            Ok(()) => {}
            Err(e) if lenient => return Ok(failed(mode, &parsed, None, &e)),
            Err(e) => return Err(e),
        }
        let (optimized, physical, scanned) = match self
            .session
            .physical_inspected(&analysis.analyzed, &analysis.env)
            .await
        {
            Ok((optimized, physical)) => {
                let scanned: Vec<String> =
                    krishiv_exec::profile::profile_scans(&physical, &|_| None)
                        .into_iter()
                        .map(|s| s.table)
                        .collect();
                (optimized, physical.explain_all(), scanned)
            }
            Err(e) if lenient => return Ok(failed(mode, &parsed, Some(&analysis), &e)),
            Err(e) => return Err(e),
        };
        let planned = Planned {
            analysis,
            optimized,
            physical,
        };
        let stats = ProviderStatistics(
            Arc::clone(&planned.analysis.env.catalog) as Arc<dyn TableProviderResolver>
        );
        let mut text = krishiv_exec::profile::dense_ids(&render(mode, &parsed, &planned, &stats));
        // Which estimates came from distinct-value samples and which stayed
        // the defaults (only when a table was sampled, so a query that
        // samples nothing prints Spark's text).
        let samples = planned.analysis.env.catalog.sample_notes(false);
        if mode == ExplainMode::Cost && !samples.is_empty() {
            let _ = write!(
                text,
                "\n== Distinct-Value Samples ==\n{}\n",
                samples.join("\n")
            );
        }
        // Only when the cache is on for the query, so the plan text stays Spark's by default.
        if let Some(note) = self.session.result_cache_note(&planned.analysis) {
            let _ = write!(text, "\n== Result Cache ==\n{note}\n");
        }
        // Only with a history that has advice for a table the query reads (P3-T025); read here, after
        // planning, so no query's planning pays for it.
        if let Some(section) = self.session.partition_advice_section(&scanned).await {
            let _ = write!(text, "\n{section}");
        }
        Ok(text)
    }
}

impl DataFrame {
    /// The text of `mode` for a caller that may hold only `Browse` on what the plan reads (Connect
    /// `AnalyzePlan.explain`, SQL `EXPLAIN`; ADR 0046 §3b, design P3 PR-14d D1). A principal whose
    /// ordinary (`Select`) analysis passes gets [`Self::explain_text`], unchanged. Otherwise the
    /// plan is shown with the data withheld ([`Self::explain_withheld`]). The trial analysis logs
    /// no audit record: its denial is an answer, not a refusal. `cost` keeps `Select`: its numbers
    /// are the data.
    pub(crate) async fn explain_for_metadata_text(
        &self,
        mode: ExplainMode,
        lenient: bool,
    ) -> Result<String, KrishivError> {
        if mode == ExplainMode::Cost {
            return self.explain_text(mode, lenient).await;
        }
        match self.trial_analysis().await {
            // An access refusal (a denial, a policy error, a row filter or mask on `Select`):
            // the withheld plan. Any other failure is the ordinary explain's to show.
            Err(e)
                if matches!(
                    e.error_class(),
                    ErrorClass::KrishivInsufficientPermissions
                        | ErrorClass::KrishivUnsupportedFeature
                ) =>
            {
                self.explain_withheld(mode, lenient).await
            }
            _ => self.explain_text(mode, lenient).await,
        }
    }

    /// The plan of a metadata analysis (catalog tables need `Browse`), planned with no table
    /// statistics, no planning hints and no distinct-value samples, and without the result-cache
    /// note; a note line says so. A plan that reads a permanent view still needs `Select` on it:
    /// the plan shows the inlined body, and a view's text is data (ADR 0046, update 2026-11-12).
    /// A file listing or footer failure is shown without its
    /// paths, which can carry partition values.
    async fn explain_withheld(
        &self,
        mode: ExplainMode,
        lenient: bool,
    ) -> Result<String, KrishivError> {
        let parsed = self.plan.to_string();
        let analysis = match self.metadata_analysis().await {
            Ok(a) => a,
            Err(e) if lenient => return Ok(failed(mode, &parsed, None, &e)),
            Err(e) => return Err(e),
        };
        if let Err(e) = self.session.authorize_bodies(&analysis).await {
            return if lenient {
                Ok(failed(mode, &parsed, None, &e))
            } else {
                Err(e)
            };
        }
        // Its definer views' queries too (P3-T062 §2.4), before any section shows the plan.
        if let Err(e) = self
            .session
            .authorize_shown_definer_reads(&analysis.analyzed)
            .await
        {
            return if lenient {
                Ok(failed(mode, &parsed, None, &e))
            } else {
                Err(e)
            };
        }
        let (optimized, physical) = match self
            .session
            .physical_withheld(&analysis.analyzed, &analysis.env)
            .await
        {
            Ok((optimized, physical)) => (optimized, physical.explain_all()),
            Err(e) => {
                let e = withheld_error(&e);
                return if lenient {
                    Ok(failed(mode, &parsed, Some(&analysis), &e))
                } else {
                    Err(e)
                };
            }
        };
        let stats = ProviderStatistics(withheld::resolver(
            Arc::clone(&analysis.env.catalog) as Arc<dyn TableProviderResolver>
        ));
        let planned = Planned {
            analysis,
            optimized,
            physical,
        };
        let mut text = render(mode, &parsed, &planned, &stats);
        let _ = write!(text, "\n{WITHHELD_NOTE}\n");
        Ok(text)
    }
}

/// The last section of a withheld explain.
pub(crate) const WITHHELD_NOTE: &str =
    "== Note ==\nStatistics withheld: the plan is the one chosen with no table statistics.";

/// A planning failure of a withheld explain, without its text: listing a table's files or reading
/// their footers can fail with a path, and a path can carry partition values (`country=US`). Its
/// class and SQLSTATE stay; every message parameter is redacted.
fn withheld_error(e: &KrishivError) -> KrishivError {
    e.redacted()
}

fn physical_section(physical: &ExplainNode) -> String {
    format!("== Physical Plan ==\n{physical}\n")
}

fn analyzed_section(parsed_analysis: &Analysis) -> String {
    let schema = parsed_analysis
        .analyzed
        .output
        .iter()
        .map(|c| format!("{}: {}", c.name, c.data_type.simple_string()))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "== Analyzed Logical Plan ==\n{schema}\n{}",
        parsed_analysis.analyzed.plan.root()
    )
}

/// The optimized plan with `Statistics(sizeInBytes=.., rowCount=..)` after
/// each node, as Spark's `EXPLAIN COST` prints it: the estimates of one
/// pass over the plan (`PlanStats`), rows scaled by the optimizer's
/// selectivities and sizes by Spark's default column widths.
fn costed(optimized: &Plan, stats: &ProviderStatistics) -> String {
    let estimates = PlanStats::compute(optimized.root(), stats);
    let note = |p: &LogicalPlan| estimates.describe(p);
    optimized.root().display_with(&note).to_string()
}

fn render(mode: ExplainMode, parsed: &str, p: &Planned, stats: &ProviderStatistics) -> String {
    match mode {
        ExplainMode::Extended => [
            format!("== Parsed Logical Plan ==\n{parsed}"),
            analyzed_section(&p.analysis),
            format!("== Optimized Logical Plan ==\n{}", p.optimized.root()),
            format!("== Physical Plan ==\n{}", p.physical),
        ]
        .join("\n"),
        ExplainMode::Cost => format!(
            "== Optimized Logical Plan ==\n{}\n{}",
            costed(&p.optimized, stats),
            physical_section(&p.physical)
        ),
        ExplainMode::Formatted => formatted(&p.physical),
        ExplainMode::Codegen => "Found 0 WholeStageCodegen subtrees.\n".to_owned(),
        _ => physical_section(&p.physical),
    }
}

/// Spark's `FORMATTED`: the outline with operator ids (children first),
/// then `(id) Name` with the node's arguments.
fn formatted(root: &ExplainNode) -> String {
    let mut ids = Vec::new();
    number(root, &mut ids);
    let mut out = String::from("== Physical Plan ==\n");
    outline(root, &mut String::new(), 0, &mut out);
    out.push_str("\n\n");
    for (id, node) in ids.iter().enumerate() {
        let _ = write!(out, "({}) {}", id + 1, node.name);
        if !node.detail.is_empty() {
            let _ = write!(out, "\nArguments: {}", node.detail);
        }
        out.push_str("\n\n");
    }
    out
}

/// The nodes in id order: children before their parent.
fn number<'a>(node: &'a ExplainNode, ids: &mut Vec<&'a ExplainNode>) {
    for child in &node.children {
        number(child, ids);
    }
    ids.push(node);
}

/// Writes the tree of `node` as `Name (id)`; `before` nodes precede its
/// subtree in id order.
fn outline(node: &ExplainNode, prefix: &mut String, before: usize, out: &mut String) {
    let _ = writeln!(out, "{} ({})", node.name, before + subtree_size(node));
    let last = node.children.len().saturating_sub(1);
    let mut start = before;
    for (i, child) in node.children.iter().enumerate() {
        let (head, tail) = if i == last {
            ("+- ", "   ")
        } else {
            (":- ", ":  ")
        };
        out.push_str(prefix);
        out.push_str(head);
        let len = prefix.len();
        prefix.push_str(tail);
        outline(child, prefix, start, out);
        prefix.truncate(len);
        start += subtree_size(child);
    }
}

fn subtree_size(node: &ExplainNode) -> usize {
    1 + node.children.iter().map(subtree_size).sum::<usize>()
}

/// What Spark prints for a query that failed: `analysis` is `Some` when
/// the failure came after analysis.
fn failed(
    mode: ExplainMode,
    parsed: &str,
    analysis: Option<&Arc<Analysis>>,
    error: &KrishivError,
) -> String {
    // A metric-view read or `measure(..)` left in the plan: Spark's planner fails before any
    // section prints, and `ExplainCommand` gives the failure's lines whatever the mode.
    if krishiv_plan::analyzer::is_unplannable(error) {
        return format!("{PLANNING_ERROR}\n{error}");
    }
    match mode {
        ExplainMode::Extended => {
            let mut out = format!("== Parsed Logical Plan ==\n{parsed}\n");
            match analysis {
                None => {
                    let _ = writeln!(out, "== Analyzed Logical Plan ==\n{error}");
                }
                Some(a) => {
                    let _ = write!(
                        out,
                        "{}\n== Optimized Logical Plan ==\n{error}\n\n== Physical Plan ==\n{error}\n",
                        analyzed_section(a)
                    );
                }
            }
            out
        }
        ExplainMode::Cost => format!(
            "{error}\n\n== Optimized Logical Plan ==\n{error}\n\n== Physical Plan ==\n{error}\n\n"
        ),
        ExplainMode::Codegen => format!("{error}\n"),
        // An access refusal (its parameters already redacted where they are a definer's) is told
        // as the other forms tell it, never as an empty line (review r7 n6).
        _ if error.error_class() == ErrorClass::KrishivInsufficientPermissions => {
            format!("{PLANNING_ERROR}\n{error}")
        }
        _ => PLANNING_ERROR.to_owned(),
    }
}

/// The relations under `p`: its inputs and the plans of the subquery expressions it holds (scalar, `EXISTS`, `IN`),
/// which `for_each_child` does not visit.
fn plan_children(p: &Arc<LogicalPlan>) -> Vec<Arc<LogicalPlan>> {
    let mut out = Vec::new();
    p.for_each_child(&mut |c| out.push(Arc::clone(c)));
    let mut exprs = Vec::new();
    p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
    let mut seen = std::collections::HashSet::new();
    while let Some(e) = exprs.pop() {
        if !seen.insert(Arc::as_ptr(&e)) {
            continue;
        }
        if let krishiv_plan::ExprNode::Subquery { plan, .. } = &e.node {
            out.push(Arc::clone(plan));
        }
        e.for_each_child(&mut |c| exprs.push(Arc::clone(c)));
    }
    out
}

/// `v` as a whole number of at least 0 (saturating).
fn f64_to_u64(v: f64) -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to the u64 range first"
    )]
    let n = v.max(0.0).min(u64::MAX as f64) as u64;
    n
}
