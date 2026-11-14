// SPDX-License-Identifier: Apache-2.0
//! The analyze API (P3-T063 N1b, design `P3-T063-analyze-api.md` §2, §3.3.3, §3.5): what a statement is,
//! what it needs, what it returns and where its columns come from, without running it.
//!
//! [`Session::analyze`] and [`Session::analyze_plan`] stop at the analyzed plan: nothing is optimized or
//! planned physically (the step that evaluates a recursive CTE's `UnionLoop`), no row is read, no UDF is
//! invoked, nothing spills. Reads go through the session's access policy as on a run (a denied read fails
//! with the same 42501); targets are listed, not authorized.
//!
//! Not listed: the directory of a managed table. A run of `DROP TABLE`, `TRUNCATE TABLE` or `INSERT` on a
//! managed table also checks `MODIFY` (and for `INSERT`, `CREATE`) on the table's directory, but only when
//! the directory exists and on its resolved path; listing it would take file-system I/O and would not be
//! deterministic. The integration test `securables_equal_what_a_run_authorizes` pins these cases.
//!
//! [`Analysis::to_json`] writes the `krishiv.analyze.v1` document
//! (`docs/reference/analyze/krishiv.analyze.v1.schema.json`, described by
//! `krishiv_plan::analyzer::lineage::schema`). It is deterministic: securables in first-use order, lineage
//! by ordinal, sources, filters, columns and warnings sorted, and no process-local value (attribute ids)
//! in it. Securables print through `Securable`'s `Display`, the 42501 text (ADR 0038 redaction, reused).
//!
//! Known gaps, owned by N3: a statement whose reads are not whole queries (a multi-table `INSERT` with a
//! shared `FROM`, `UPDATE`, `DELETE` with a subquery predicate, `EXECUTE IMMEDIATE`, scripts, variable
//! expressions) is `partial` with only its targets and the warning `UNSUPPORTED_READ_SHAPE`; `MERGE`
//! reports what `statement_queries` gives today (its source query), not the subqueries of its `ON`,
//! clause conditions or `SET` values; `EXECUTE IMMEDIATE` is always `SESSION`.
//!
//! Output schema and lineage are a query's (`Statement::Query`). A `SHOW`, `DESCRIBE`, `EXPLAIN`, `SET` or
//! `RESET` command has the result columns the session returns for it ([`command_schema`]; `RESET` and
//! `SET PATH` have none) and no lineage. Any other statement (writes, DDL) has neither, but its reads are
//! listed.
//!
//! Provisional: every command column is reported nullable. Spark's command attributes are mostly
//! non-null; pinning them at the Spark tags is a spark-semantics follow-up, and the command frames and
//! [`command_schema`] then change together.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::lineage::schema::{
    LeafName, SCHEMA_VERSION, leaf_key, leaf_name, securable_kind, securable_name,
};
use krishiv_plan::analyzer::lineage::{
    ColumnLineage, FilterKind, Leaf, LineageWarning, Source, lineage,
};
use krishiv_plan::analyzer::{AnalyzedPlan, SecurableUse};
use krishiv_sql::ExplainKind;
use krishiv_sql::ast::Statement;
use krishiv_types::DataType;

use crate::access::StatementInfo;
use crate::session::{Session, internal};

mod command_schema;
#[cfg(test)]
mod tests;

use command_schema::command_schema;

/// What a statement is, needs, returns and derives its columns from (see the module doc).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Analysis {
    /// Its class, explain kind and securables (as [`Session::statement_info`]; an `ADMIN` statement
    /// lists `Manage` on the cluster).
    pub info: StatementInfo,
    /// Whether its reads are not fully known: then it is not provably read-only.
    pub partial: bool,
    /// The result columns: a query's, or a `SHOW`, `DESCRIBE`, `EXPLAIN`, `SET` or `RESET` command's (see
    /// the module doc); else empty.
    pub output_schema: Vec<OutputColumn>,
    /// Per output ordinal of a query, its sources and transform (else empty).
    pub lineage: Vec<ColumnLineage>,
    /// The columns that decide which rows or groups (a query's), sorted.
    pub filters: BTreeSet<(FilterKind, Leaf)>,
    /// Fixed warning codes, sorted.
    pub warnings: BTreeSet<AnalysisWarning>,
}

/// One result column.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OutputColumn {
    /// Its name.
    pub name: String,
    /// Its type.
    pub data_type: DataType,
    /// Whether it may be null.
    pub nullable: bool,
    /// The `CHAR(n)`/`VARCHAR(n)` it was declared with, if any.
    pub declared_type: Option<String>,
}

impl OutputColumn {
    /// A column `name` of `data_type`.
    #[must_use]
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
            declared_type: None,
        }
    }
}

/// A fixed analysis warning code; never text with user data. Declared in code order, so a sorted set is
/// sorted by code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum AnalysisWarning {
    /// A user-defined function hides how a column derives (its outputs are `opaque`).
    LineageOpaqueUdf,
    /// The statement's reads are not whole queries: it is `partial`.
    UnsupportedReadShape,
    /// User code ran at analysis (a Python UDTF's `analyze`; unused until PR-48b).
    UserCodeRanAtAnalysis,
    /// A view was not expanded: its columns are leaves.
    ViewNotExpanded,
}

impl AnalysisWarning {
    /// The stable code (`LINEAGE_OPAQUE_UDF`, ...).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::LineageOpaqueUdf => "LINEAGE_OPAQUE_UDF",
            Self::UnsupportedReadShape => "UNSUPPORTED_READ_SHAPE",
            Self::UserCodeRanAtAnalysis => "USER_CODE_RAN_AT_ANALYSIS",
            Self::ViewNotExpanded => "VIEW_NOT_EXPANDED",
        }
    }

    /// Every warning, in code order.
    pub const ALL: [Self; 4] = [
        Self::LineageOpaqueUdf,
        Self::UnsupportedReadShape,
        Self::UserCodeRanAtAnalysis,
        Self::ViewNotExpanded,
    ];

    /// The warning of `code` (`LineageWarning` is non-exhaustive in another crate, so it maps by its
    /// code; the schema's tests keep every lineage code among [`Self::ALL`]'s). An unknown code is an
    /// internal error, never a dropped warning.
    fn of_lineage(w: LineageWarning) -> Result<Self, KrishivError> {
        Self::ALL
            .into_iter()
            .find(|a| a.code() == w.code())
            .ok_or_else(|| internal(format!("lineage warning {} has no analysis code", w.code())))
    }
}

impl Session {
    /// Analyzes the SQL statement `sql` without running it: its class, the securables it needs (reads
    /// with their columns, found by analysis and checked against the session's access policy; then
    /// targets, listed), its output schema, column lineage and filter columns (see [`Analysis`]).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow_schema::{DataType, Field, Schema};
    /// use krishiv_session::Session;
    ///
    /// let s = Session::new()?;
    /// # s.register_batches("t", Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])), vec![])?;
    /// let a = futures::executor::block_on(s.analyze("SELECT a + 1 AS b FROM t"))?;
    /// assert_eq!(a.output_schema[0].name, "b");
    /// assert!(a.to_json().contains("\"transform\":\"expression\""));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Parse and analysis errors, as running the statement gives them (`KRISHIV_INSUFFICIENT_PERMISSIONS`
    /// for a denied read); an internal error when the lineage pass meets a plan it cannot read.
    pub async fn analyze(&self, sql: &str) -> Result<Analysis, KrishivError> {
        let (session, sql) = (self.clone(), sql.to_owned());
        self.on_executor(async move { session.analyze_here(&sql).await })
            .await
    }

    /// [`Self::analyze`] of a DataFrame's plan: class `READ`, its reads, output schema and lineage (feature
    /// `unstable-ir`, as every IR entry point).
    ///
    /// # Errors
    /// As [`Self::analyze`].
    #[cfg(any(test, feature = "unstable-ir"))]
    pub async fn analyze_plan(
        &self,
        plan: &krishiv_plan::LogicalPlan,
    ) -> Result<Analysis, KrishivError> {
        let (session, plan) = (self.clone(), std::sync::Arc::new(plan.clone()));
        self.on_executor(async move {
            let df = crate::dataframe::DataFrame::new(session.clone(), plan);
            let analysis = df.analyzed_in_with_text(session.query_env(), None).await?;
            let mut securables: Vec<SecurableUse> = Vec::new();
            crate::access::merge_reads(&mut securables, &analysis.analyzed.securables);
            let info = StatementInfo {
                class: krishiv_sql::StatementClass::Read,
                explain: None,
                securables,
            };
            Analysis::of_query(info, &analysis.analyzed)
        })
        .await
    }

    async fn analyze_here(&self, sql: &str) -> Result<Analysis, KrishivError> {
        let parts = self.statement_parts(sql).await?;
        if parts.partial {
            return Ok(Analysis {
                info: parts.info,
                partial: true,
                output_schema: Vec::new(),
                lineage: Vec::new(),
                filters: BTreeSet::new(),
                warnings: BTreeSet::from([AnalysisWarning::UnsupportedReadShape]),
            });
        }
        match (&parts.statement, &parts.analyzed[..]) {
            (Statement::Query(_), [analyzed]) => Analysis::of_query(parts.info, analyzed),
            (statement, _) => Ok(Analysis {
                info: parts.info,
                partial: false,
                output_schema: command_schema(statement).unwrap_or_default(),
                lineage: Vec::new(),
                filters: BTreeSet::new(),
                warnings: BTreeSet::new(),
            }),
        }
    }
}

impl Analysis {
    /// The analysis of a query: its output schema and lineage.
    fn of_query(info: StatementInfo, analyzed: &AnalyzedPlan) -> Result<Self, KrishivError> {
        let l = lineage(analyzed)?;
        let output_schema = analyzed
            .output
            .iter()
            .enumerate()
            .map(|(i, c)| OutputColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
                declared_type: analyzed.output_declared.get(i).cloned().flatten(),
            })
            .collect();
        Ok(Self {
            info,
            partial: false,
            output_schema,
            lineage: l.columns,
            filters: l.filters,
            warnings: l
                .warnings
                .into_iter()
                .map(AnalysisWarning::of_lineage)
                .collect::<Result<_, _>>()?,
        })
    }

    /// The `krishiv.analyze.v1` JSON document (compact; see the module doc).
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut o = String::from("{");
        member(&mut o, "schema_version", &string(SCHEMA_VERSION), true);
        member(
            &mut o,
            "statement_class",
            &string(self.info.class.name()),
            false,
        );
        let explain = match self.info.explain {
            None => "null".to_owned(),
            Some(ExplainKind::Plan) => string("PLAN"),
            Some(ExplainKind::Analyze) => string("ANALYZE"),
        };
        member(&mut o, "explain", &explain, false);
        member(&mut o, "partial", bool_text(self.partial), false);
        let securables = list(self.info.securables.iter().map(securable_json));
        member(&mut o, "securables", &securables, false);
        let schema = list(self.output_schema.iter().enumerate().map(|(i, c)| {
            let mut e = String::from("{");
            member(&mut e, "ordinal", &i.to_string(), true);
            member(&mut e, "name", &string(&c.name), false);
            member(&mut e, "type", &string(&c.data_type.simple_string()), false);
            member(&mut e, "type_json", &string(&c.data_type.json()), false);
            member(&mut e, "nullable", bool_text(c.nullable), false);
            let declared = c
                .declared_type
                .as_deref()
                .map_or_else(|| "null".to_owned(), string);
            member(&mut e, "declared_type", &declared, false);
            e.push('}');
            e
        }));
        member(&mut o, "output_schema", &schema, false);
        let lineage = list(self.lineage.iter().map(|c| {
            let mut e = String::from("{");
            member(&mut e, "ordinal", &c.ordinal.to_string(), true);
            member(&mut e, "output", &string(&c.name), false);
            member(
                &mut e,
                "sources",
                &list(c.sources.iter().map(source_json)),
                false,
            );
            member(&mut e, "transform", &string(c.transform.name()), false);
            e.push('}');
            e
        }));
        member(&mut o, "lineage", &lineage, false);
        let filters = list(self.filters.iter().map(|(kind, leaf)| {
            let mut e = String::from("{");
            member(&mut e, "kind", &string(kind.name()), true);
            leaf_members(&mut e, leaf);
            e.push('}');
            e
        }));
        member(&mut o, "filters", &filters, false);
        let warnings = list(self.warnings.iter().map(|w| string(w.code())));
        member(&mut o, "warnings", &warnings, false);
        o.push('}');
        o
    }
}

fn securable_json(u: &SecurableUse) -> String {
    let mut e = String::from("{");
    member(&mut e, "kind", &string(securable_kind(&u.securable)), true);
    let name = list(securable_name(&u.securable).iter().map(|p| string(p)));
    member(&mut e, "name", &name, false);
    member(&mut e, "display", &string(&u.securable.to_string()), false);
    member(
        &mut e,
        "privilege",
        &string(&u.privilege.to_string()),
        false,
    );
    member(
        &mut e,
        "columns",
        &list(u.columns.iter().map(|c| string(c))),
        false,
    );
    e.push('}');
    e
}

fn source_json(s: &Source) -> String {
    let mut e = String::from("{");
    leaf_members_first(&mut e, &s.leaf);
    let via = s
        .via_view
        .as_ref()
        .map_or_else(|| "null".to_owned(), |v| list(v.iter().map(|p| string(p))));
    member(&mut e, "via_view", &via, false);
    e.push('}');
    e
}

/// A leaf's two members, after other members.
fn leaf_members(e: &mut String, leaf: &Leaf) {
    leaf_fields(e, leaf, false);
}

/// A leaf's two members, first in the object.
fn leaf_members_first(e: &mut String, leaf: &Leaf) {
    leaf_fields(e, leaf, true);
}

/// The key and name come from `krishiv-plan`'s exhaustive matches ([`leaf_key`], [`leaf_name`]), so a
/// new leaf form is named in the schema description before it can be written.
fn leaf_fields(e: &mut String, leaf: &Leaf, first: bool) {
    let (name, column) = leaf_name(leaf);
    let value = match name {
        LeafName::Parts(parts) => list(parts.iter().map(|s| string(s))),
        LeafName::Text(text) => string(text),
    };
    member(e, leaf_key(leaf), &value, first);
    member(e, "column", &string(column), false);
}

fn member(out: &mut String, key: &str, value: &str, first: bool) {
    if !first {
        out.push(',');
    }
    out.push_str(&string(key));
    out.push(':');
    out.push_str(value);
}

const fn bool_text(b: bool) -> &'static str {
    if b { "true" } else { "false" }
}

fn list(items: impl Iterator<Item = String>) -> String {
    let items: Vec<String> = items.collect();
    format!("[{}]", items.join(","))
}

/// A JSON string literal.
fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
