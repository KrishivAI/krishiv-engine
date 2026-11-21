// SPDX-License-Identifier: Apache-2.0
//! `CREATE VIEW ... WITH METRICS LANGUAGE YAML AS $$...$$` (P3 4.2-MV step 4, design note
//! `P3-metric-views.md` §1, §3): Spark 4.2's `CreateMetricViewCommand` on the session catalog.
//!
//! The YAML is read ([`crate::metric_view`]), then the view's schema comes from analyzing the
//! aggregate Spark's `ResolveMetricView` builds at create: every dimension and measure over the
//! filtered source, grouped by the dimensions. Krishiv writes that aggregate as SQL text around the
//! YAML's expressions and analyzes it; an error raised inside one expression is told against that
//! expression's own text (Spark parses each one alone), any other against the whole statement.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use krishiv_common::error::{
    CreateViewColumnArityMismatchNotEnoughDataColumns,
    CreateViewColumnArityMismatchTooManyDataColumns, ErrorClass, ErrorQueryContext, InternalError,
    KrishivError, MissingAttributesResolvedAttributeAppearInOperation,
    MissingAttributesResolvedAttributeMissingFromInput, SchemaNotFound, TableOrViewAlreadyExists,
};
use krishiv_common::names::check_column_name_duplication;
use krishiv_sql::ast::{ColumnName, CreateMetricView, ExprKind};

use super::{check_new_name, now_ms, options_of, unsupported};
use crate::dataframe::DataFrame;
use crate::metric_view::{MetricViewSource, MetricViewSpec, parse_metric_view_yaml};
use crate::session::Session;
use crate::store::{
    CatalogStore, CatalogTable, CreateMode, MetricColumn, MetricColumnKind, MetricViewDef,
    TableLayout, ViewDef, ViewKind,
};

/// Spark keeps at most this many UTF-16 units of `metric_view.from.sql` and `metric_view.where`
/// (`MetricViewCanonical.scala` 206-207).
const PROPERTY_LIMIT: usize = 1024;

/// The aggregate's SQL text and where each YAML expression sits in it.
struct Aggregate {
    text: String,
    /// The range of each expression in `text`, with the expression as written.
    parts: Vec<(Range<usize>, String)>,
}

impl Aggregate {
    /// `SELECT d AS n, .., m AS n FROM source [WHERE filter] GROUP BY 1, ..`: grouped by the
    /// dimensions' ordinals, as Spark groups by their attributes (`ResolveMetricView` 177-181),
    /// so a dimension reading another one's alias is `LATERAL_COLUMN_ALIAS_IN_GROUP_BY`. The
    /// ordinals in `ungrouped` (1-based) are left out: Spark's grouping attribute of a dimension
    /// that is an aggregate is no aggregate, so its create succeeds (`yaml_dimension_is_aggregate`).
    fn build(spec: &MetricViewSpec, ungrouped: &[usize]) -> Self {
        let mut out = Self {
            text: String::from("SELECT "),
            parts: Vec::new(),
        };
        let mut columns = spec.dimensions.iter().chain(&spec.measures).peekable();
        if columns.peek().is_none() {
            // A view of no columns still has its source analyzed.
            out.text.push('1');
        }
        for (i, c) in columns.enumerate() {
            if i > 0 {
                out.text.push_str(", ");
            }
            out.push_expr(&c.expr);
            // On a line of its own: an expression ending in a `--` comment (each parses alone)
            // leaves its alias and the rest of the statement.
            out.text.push_str("\n AS `");
            out.text.push_str(&c.name.replace('`', "``"));
            out.text.push('`');
        }
        if let (Some(range), Some(filter)) = (spec.push_from(&mut out.text), &spec.filter) {
            out.parts.push((range, filter.clone()));
        }
        let keys = (1..=spec.dimensions.len()).filter(|n| !ungrouped.contains(n));
        for (i, n) in keys.enumerate() {
            out.text.push_str(if i == 0 { " GROUP BY " } else { ", " });
            out.text.push_str(&n.to_string());
        }
        out
    }

    fn push_expr(&mut self, expr: &str) {
        let start = self.text.len();
        self.text.push_str(expr);
        self.parts.push((start..self.text.len(), expr.to_owned()));
    }

    /// `e` with the context Spark gives it: inside one expression, a span of that expression's
    /// text; anywhere else, the whole statement.
    fn located(&self, e: KrishivError, statement: &str) -> KrishivError {
        let Some(c) = e.query_contexts().first().cloned() else {
            return e;
        };
        let inner = self
            .parts
            .iter()
            .find(|(r, _)| r.start <= c.start && c.stop < r.end);
        let e = e.without_query_contexts();
        match inner {
            Some((r, text)) => {
                let (start, stop) = (c.start - r.start, c.stop - r.start);
                e.with_query_context(ErrorQueryContext::sql(
                    krishiv_sql::redact_sql(text).fragment(start..stop + 1),
                    start,
                    stop,
                ))
            }
            None => whole_statement(e, statement),
        }
    }
}

/// `e` with the whole statement as its context, as Spark gives errors of a metric view's plan.
fn whole_statement(e: KrishivError, statement: &str) -> KrishivError {
    let shown = krishiv_sql::redact_sql(statement);
    e.with_query_context(ErrorQueryContext::sql(
        shown.fragment(0..statement.len()),
        0,
        statement.len().saturating_sub(1),
    ))
}

/// The dimension ordinal `e` refuses as `GROUP_BY_POS_AGGREGATE`, if it is one not yet left out.
fn aggregate_ordinal(e: &KrishivError, dimensions: usize, ungrouped: &[usize]) -> Option<usize> {
    if e.error_class() != ErrorClass::GroupByPosAggregate {
        return None;
    }
    let (_, index) = e.message_parameters().iter().find(|(k, _)| *k == "index")?;
    let n: usize = index.parse().ok()?;
    ((1..=dimensions).contains(&n) && !ungrouped.contains(&n)).then_some(n)
}

/// The first `limit` UTF-16 units of `s` (Java's `take`); a surrogate pair cut in half becomes
/// U+FFFD, since a Rust string cannot hold a lone surrogate.
fn take_utf16(s: &str, limit: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().take(limit).collect();
    String::from_utf16_lossy(&units)
}

/// Spark's `MetricView.getProperties`: where the view reads from, and its filter.
fn view_properties(spec: &MetricViewSpec) -> [(String, String); 2] {
    let (kind, key, value) = match &spec.source {
        MetricViewSource::Asset(name) => ("ASSET", "metric_view.from.name", name.clone()),
        MetricViewSource::Sql(query) => (
            "SQL",
            "metric_view.from.sql",
            take_utf16(query, PROPERTY_LIMIT),
        ),
    };
    [
        ("metric_view.from.type".to_owned(), kind.to_owned()),
        (key.to_owned(), value),
    ]
}

impl Session {
    /// Spark's error for a dimension that is a literal: its grouping attribute never reaches the
    /// aggregate's input (a Spark bug, reproduced by class, design note "Golden results" 5). The
    /// class says whether the source has a column of the dimension's name.
    async fn constant_dimension(
        &self,
        spec: &MetricViewSpec,
        dimension: &str,
        env: &crate::session::QueryEnv,
    ) -> Result<KrishivError, KrishivError> {
        let source = MetricViewSpec {
            source: spec.source.clone(),
            filter: None,
            dimensions: Vec::new(),
            measures: Vec::new(),
        };
        let text = Aggregate::build(&source, &[])
            .text
            .replacen("SELECT 1", "SELECT *", 1);
        let query = krishiv_sql::parse_query(&text, &env.parser).map_err(|e| e.error)?;
        let plan = krishiv_sql::lower_identified(
            &query,
            self.next_source_id(),
            self.parser_options().case_sensitive,
            &[],
        )
        .map_err(|e| e.error)?;
        let analysis = DataFrame::new(self.clone(), Arc::clone(plan.root()))
            .analyzed_in(env.clone())
            .await?;
        let input: Vec<String> = analysis
            .analyzed
            .output
            .iter()
            .map(|o| o.name.clone())
            .collect();
        let quoted = |n: &str| format!("\"{n}\"");
        let missing = quoted(dimension);
        let shown_input = input
            .iter()
            .map(|n| quoted(n))
            .collect::<Vec<_>>()
            .join(", ");
        let operator = "!Aggregate".to_owned();
        let same_name = input.iter().any(|n| n.eq_ignore_ascii_case(dimension));
        Ok(if same_name {
            MissingAttributesResolvedAttributeAppearInOperation {
                missing_attributes: missing.clone(),
                input: shown_input,
                operator,
                operation: missing,
            }
            .into()
        } else {
            MissingAttributesResolvedAttributeMissingFromInput {
                missing_attributes: missing,
                input: shown_input,
                operator,
            }
            .into()
        })
    }

    /// `CREATE VIEW ... WITH METRICS LANGUAGE YAML AS $$...$$` in the session catalog.
    pub(crate) async fn exec_create_metric_view(
        &self,
        v: &CreateMetricView,
        original: &str,
    ) -> Result<DataFrame, KrishivError> {
        if v.with_metrics.is_none() || v.language.is_none() {
            // Spark 4.2 raises `MISSING_CLAUSES_FOR_OPERATION` without registering it, so the
            // client sees the failure to build it (4.2.1 golden); 4.3 registers it.
            return Err(InternalError {
                message: "Cannot find main error class 'MISSING_CLAUSES_FOR_OPERATION'".to_owned(),
            }
            .into());
        }
        let env = self.query_env();
        let spec = parse_metric_view_yaml(&v.body, &env.parser)?;
        let statement = original.get(v.span.start..v.span.end).unwrap_or(original);
        let at_statement = |e: KrishivError| whole_statement(e, statement);
        // Spark parses each expression alone (`MetricViewPlanner.buildInputColumns`) before
        // anything resolves; a syntax error there is told against the whole statement.
        let mut constant = None;
        for c in spec.dimensions.iter().chain(&spec.measures) {
            // `MetricViewCanonical` 127-131: a plain JVM exception, outside Spark's error classes.
            let length = c.expr.encode_utf16().count();
            if length > PROPERTY_LIMIT {
                return Err(InternalError {
                    message: format!(
                        "Expression length {length} exceeds maximum allowed size {PROPERTY_LIMIT} for column '{}'",
                        c.name
                    ),
                }
                .into());
            }
            let item = krishiv_sql::parse_named_expression(&c.expr, &env.parser)
                .map_err(|e| at_statement(e.error))?;
            if constant.is_none()
                && matches!(item.expr.kind, ExprKind::Literal(_))
                && spec.dimensions.iter().any(|d| std::ptr::eq(d, c))
            {
                constant = Some(c);
            }
        }
        if let Some(filter) = &spec.filter {
            krishiv_sql::parse_named_expression(filter, &env.parser)
                .map_err(|e| at_statement(e.error))?;
        }
        let name = self.table_name(&v.name, &env)?;
        self.refuse_collated_namespace(&name)?;
        check_new_name(&name.namespace)?;
        check_new_name(&name.name)?;
        // Spark names the view as its `TableIdentifier` prints: database and name.
        let shown = format!(
            "`{}`.`{}`",
            name.namespace.replace('`', "``"),
            name.name.replace('`', "``")
        );
        if !self.state.catalog.store.has_namespace(&name.namespace) {
            return Err(SchemaNotFound {
                schema_name: format!("`spark_catalog`.`{}`", name.namespace.replace('`', "``")),
            }
            .into());
        }

        // The create aggregate groups by ordinals, whatever `spark.sql.groupByOrdinal` says.
        let mut grouping = env.clone();
        grouping.analysis.conf.group_by_ordinal = true;
        let mut ungrouped = Vec::new();
        let (plan, analysis) = loop {
            let aggregate = Aggregate::build(&spec, &ungrouped);
            let query = krishiv_sql::parse_query(&aggregate.text, &env.parser)
                .map_err(|e| at_statement(e.error))?;
            let plan = krishiv_sql::lower_identified(
                &query,
                self.next_source_id(),
                self.parser_options().case_sensitive,
                &[],
            )
            .map_err(|e| aggregate.located(e.error, statement))?;
            let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
            let redacted = krishiv_sql::redact_sql(&aggregate.text);
            match df
                .analyzed_in_with_text(grouping.clone(), Some(&redacted))
                .await
            {
                Ok(analysis) => break (plan, analysis),
                Err(e) => match aggregate_ordinal(&e, spec.dimensions.len(), &ungrouped) {
                    // At most one more analysis per dimension: `aggregate_ordinal` never names
                    // an ordinal twice.
                    Some(n) => {
                        ungrouped.push(n);
                        debug_assert!(ungrouped.len() <= spec.dimensions.len());
                    }
                    None => return Err(aggregate.located(e, statement)),
                },
            }
        };
        if let Some(c) = constant {
            // Spark groups by the dimensions' attributes, and a literal's never reaches the
            // aggregate's input (a Spark bug, reproduced by class, design note "Golden results" 5).
            return Err(at_statement(
                self.constant_dimension(&spec, &c.name, &env).await?,
            ));
        }
        // Then the temporary objects (`ViewHelper.verifyTemporaryObjectsNotExists`).
        if let Some(temp) = crate::catalog::reads_temporary(
            &self.state.catalog,
            plan.root(),
            env.analysis.conf.case_sensitive,
        ) {
            return Err(crate::view_session_refs::refusal(
                &shown,
                format!("`{temp}`"),
                "VIEW",
            ));
        }
        self.refuse_session_objects(&shown, plan.root(), &env)
            .await?;

        let kinds = spec
            .dimensions
            .iter()
            .map(|d| (MetricColumnKind::Dimension, d))
            .chain(spec.measures.iter().map(|m| (MetricColumnKind::Measure, m)));
        let mut columns = crate::char_write::output_columns(&analysis.analyzed, None);
        if spec.dimensions.is_empty() && spec.measures.is_empty() {
            // `Aggregate::build`'s placeholder column.
            columns.clear();
        }
        for (column, (kind, c)) in columns.iter_mut().zip(kinds) {
            column.metric = Some(MetricColumn {
                kind,
                expr: c.expr.clone(),
            });
        }
        if !v.columns.is_empty() {
            let mut declared = Vec::with_capacity(v.columns.len());
            for alias in &v.columns {
                let ColumnName::Ident(ident) = &alias.name else {
                    return Err(unsupported("IDENTIFIER() column names"));
                };
                declared.push(ident.value.clone());
            }
            let quote = |n: &str| format!("`{}`", n.replace('`', "``"));
            let list = |names: &mut dyn Iterator<Item = &String>| {
                names.map(|n| quote(n)).collect::<Vec<_>>().join(", ")
            };
            let data: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
            match declared.len().cmp(&columns.len()) {
                std::cmp::Ordering::Greater => {
                    return Err(CreateViewColumnArityMismatchNotEnoughDataColumns {
                        view_name: shown.clone(),
                        view_columns: list(&mut declared.iter()),
                        data_columns: list(&mut data.iter()),
                    }
                    .into());
                }
                std::cmp::Ordering::Less => {
                    return Err(CreateViewColumnArityMismatchTooManyDataColumns {
                        view_name: shown.clone(),
                        view_columns: list(&mut declared.iter()),
                        data_columns: list(&mut data.iter()),
                    }
                    .into());
                }
                std::cmp::Ordering::Equal => {}
            }
            for (column, (alias, n)) in columns.iter_mut().zip(v.columns.iter().zip(declared)) {
                column.name = n;
                if alias.comment.is_some() {
                    column.comment.clone_from(&alias.comment);
                }
            }
        }
        check_column_name_duplication(
            columns.iter().map(|c| c.name.as_str()),
            env.analysis.conf.case_sensitive,
        )?;

        // `properties ++ metricView.getProperties`: the view's own keys win.
        let mut properties = options_of(&v.tblproperties, original);
        properties.extend(view_properties(&spec));
        if let Some(filter) = &spec.filter {
            properties.insert(
                "metric_view.where".to_owned(),
                take_utf16(filter, PROPERTY_LIMIT),
            );
        }
        let namespace = env
            .catalog
            .namespace()
            .first()
            .cloned()
            .unwrap_or_else(|| self.state.catalog.current_namespace());
        let view = CatalogTable {
            namespace: name.namespace.clone(),
            name: name.name.clone(),
            managed: false,
            provider: String::new(),
            provider_written: String::new(),
            location: PathBuf::new(),
            columns,
            options: BTreeMap::new(),
            properties,
            comment: v.comment.clone(),
            created_ms: now_ms(),
            root: PathBuf::new(),
            view: Some(ViewDef {
                sql: v.body.clone(),
                namespace,
                path: env.catalog.sql_path_snapshot(),
                query_columns: Vec::new(),
                identifier_spans: Vec::new(),
                written: v.body.clone(),
                schema_mode: None,
                definer: None,
                cte_precedence: None,
                functions: None,
                revision: 0,
                author: self.state.principal.name.clone(),
                kind: ViewKind::Metric(MetricViewDef {
                    parser: env.parser,
                    ansi: env.analysis.conf.ansi,
                }),
            }),
            layout: TableLayout::default(),
            column_stats: BTreeMap::new(),
            incarnation: 0,
            data_version: 0,
            stamp: 0,
        };
        // Spark's v1 command passes only `IF NOT EXISTS` on (`ignoreIfExists = allowExisting`):
        // `OR REPLACE` over an existing view is `TABLE_OR_VIEW_ALREADY_EXISTS` (golden).
        if !v.if_not_exists
            && self
                .state
                .catalog
                .store
                .table(&name.namespace, &name.name)
                .is_some()
        {
            return Err(TableOrViewAlreadyExists {
                relation_name: shown,
            }
            .into());
        }
        let mode = if v.if_not_exists {
            CreateMode::IfNotExists
        } else {
            CreateMode::ErrorIfExists
        };
        self.state.catalog.store.create_table(view, mode)?;
        self.empty_frame().await
    }
}

#[cfg(test)]
mod tests {
    use super::take_utf16;

    #[test]
    fn properties_are_cut_in_utf16_units() {
        assert_eq!(take_utf16("abc", 2), "ab");
        // U+1F600 is two UTF-16 units: a cut after the first leaves a lone surrogate.
        assert_eq!(take_utf16("a\u{1F600}", 2), "a\u{FFFD}");
        assert_eq!(take_utf16("a\u{1F600}", 3), "a\u{1F600}");
    }
}
