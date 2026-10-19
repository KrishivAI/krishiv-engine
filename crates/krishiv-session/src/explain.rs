// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN [EXTENDED | FORMATTED | COST | CODEGEN]` and
//! `DataFrame::explain`: the plans of a query as Spark 4.2 lays them out
//! (the section headers, which sections each mode prints, and Spark's
//! handling of a query that does not plan). The nodes themselves are
//! Krishiv's operators, not Spark's (ADR 0023).

use std::fmt::Write as _;
use std::sync::Arc;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::planner::{ExplainNode, ProviderStatistics, TableProviderResolver};
use krishiv_optimizer::estimated_rows;
use krishiv_plan::Plan;
use krishiv_sql::ast::{Explain, Statement};

use crate::dataframe::{Analysis, DataFrame, ExplainMode};
use crate::functions::lit;
use crate::session::{Session, internal};

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
    /// `plan STRING` frame. A query that fails to analyze or plan is part
    /// of the text, not an error; a syntax error is an error.
    pub(crate) async fn explain_statement(&self, e: &Explain) -> Result<DataFrame, KrishivError> {
        let Statement::Query(q) = &*e.statement else {
            return Err(KrishivUnsupportedFeature {
                feature: "EXPLAIN of an EXPLAIN in sessions".to_owned(),
            }
            .into());
        };
        let plan = krishiv_sql::lower(q, krishiv_plan::SourceId(0)).map_err(|e| e.error)?;
        let explained = DataFrame::new(self.clone(), Arc::clone(plan.root()));
        let text = explained.explain_text(e.mode.into(), true).await?;
        self.one_row(&text).await
    }

    /// A frame of one row, the string column `plan`.
    async fn one_row(&self, text: &str) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let Statement::Query(q) =
            krishiv_sql::parse_statement("SELECT 1", &env.parser).map_err(|e| e.error)?
        else {
            return Err(internal("SELECT 1 is not a query".to_owned()));
        };
        let plan = krishiv_sql::lower(&q, krishiv_plan::SourceId(0)).map_err(|e| e.error)?;
        let df = DataFrame::new(self.clone(), Arc::clone(plan.root()))
            .select([lit(text.to_owned()).alias("plan")]);
        df.analyzed_in(env).await?;
        Ok(df)
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
        let (optimized, physical) = match self.session.physical(&analysis.analyzed, &analysis.env) {
            Ok((optimized, physical)) => (optimized, physical.explain().clone()),
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
        Ok(render(mode, &parsed, &planned, &stats))
    }
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

/// The optimized plan with the estimated row count of its root
/// (`Statistics(rowCount=..)`; byte sizes are not estimated).
fn costed(optimized: &Plan, stats: &ProviderStatistics) -> String {
    let rows = estimated_rows(optimized.root(), stats);
    let text = optimized.root().to_string();
    let (first, rest) = text.split_once('\n').unwrap_or((&text, ""));
    format!("{first}, Statistics(rowCount={rows:.0})\n{rest}")
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
        _ => PLANNING_ERROR.to_owned(),
    }
}
