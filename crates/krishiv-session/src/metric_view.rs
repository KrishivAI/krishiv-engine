// SPDX-License-Identifier: Apache-2.0
//! Metric views (P3 4.2-MV): the YAML body of `CREATE VIEW ... WITH METRICS LANGUAGE YAML AS $$...$$`
//! read into a [`MetricViewSpec`], as Spark 4.2's `MetricViewFactory.fromYAML` reads it
//! (`yaml.rs`: Jackson's data binding over YAML events, its errors and their locations).
//!
//! Every failure is `INVALID_METRIC_VIEW_YAML` (42K0L) with Spark's message: the validation
//! messages bare (`Invalid YAML version: 0.2`, `Source cannot be empty`, `Invalid source: ...`),
//! everything else after `Failed to parse YAML: `. A malformed document (a YAML syntax error) keeps
//! that prefix, but the rest of its text is the event parser's, not SnakeYAML's: the one documented
//! difference (design note `P3-metric-views.md` §2).

mod yaml;

use krishiv_common::error::{InvalidMetricViewYaml, KrishivError};
use krishiv_sql::ParserOptions;

/// A metric view's definition, version 0.1, in the order Spark keeps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetricViewSpec {
    /// `source`: a table or view name, or a query.
    pub(crate) source: MetricViewSource,
    /// `filter`: a boolean expression over the source, as written.
    pub(crate) filter: Option<String>,
    /// `dimensions`, in YAML order.
    pub(crate) dimensions: Vec<MetricViewColumn>,
    /// `measures`, in YAML order (they follow the dimensions in the view's columns).
    pub(crate) measures: Vec<MetricViewColumn>,
}

impl MetricViewSpec {
    /// Appends ` FROM source [WHERE (filter)]` to `text` and returns where the filter's text sits
    /// in it. The source and the filter each end a line of their own, so one ending in a `--`
    /// comment (an asset name parses with one after it, as Spark's `parseMultipartIdentifier`
    /// reads it) leaves the parenthesis and the clauses built after it in place.
    pub(crate) fn push_from(&self, text: &mut String) -> Option<std::ops::Range<usize>> {
        text.push_str(" FROM ");
        match &self.source {
            MetricViewSource::Asset(name) => {
                text.push_str(name);
                text.push('\n');
            }
            MetricViewSource::Sql(query) => {
                text.push('(');
                text.push_str(query);
                text.push_str("\n)");
            }
        }
        let filter = self.filter.as_ref()?;
        text.push_str(" WHERE (");
        let start = text.len();
        text.push_str(filter);
        let range = start..text.len();
        text.push_str("\n)");
        Some(range)
    }
}

/// Spark's `Source`: a name that parses as a multipart identifier is an asset, else a query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MetricViewSource {
    /// `AssetSource`: a table, view or metric view, the text as written.
    Asset(String),
    /// `SQLSource`: a query, the text as written.
    Sql(String),
}

/// One dimension or measure: its name and its SQL expression, as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetricViewColumn {
    pub(crate) name: String,
    pub(crate) expr: String,
}

/// Reads a metric view's YAML body (the text between the dollar quotes). `options` are the parse
/// options Spark's `CatalystSqlParser` would see, used to tell a name from a query in `source`.
///
/// # Errors
/// `INVALID_METRIC_VIEW_YAML` with Spark's message.
pub(crate) fn parse_metric_view_yaml(
    text: &str,
    options: &ParserOptions,
) -> Result<MetricViewSpec, KrishivError> {
    yaml::read(text, options).map_err(|message| InvalidMetricViewYaml { message }.into())
}

#[cfg(test)]
mod tests;
