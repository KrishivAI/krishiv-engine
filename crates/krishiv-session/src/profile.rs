// SPDX-License-Identifier: Apache-2.0
//! `EXPLAIN ANALYZE` and `EXPLAIN VERBOSE`, Krishiv extensions (Spark has
//! neither; ADR 0023): a query's profile, and the logical plan after each
//! optimizer rule that changed it.

use std::fmt::Write as _;

use arrow_array::RecordBatch;
use futures::StreamExt as _;
use krishiv_common::error::KrishivError;
use krishiv_exec::profile::QueryProfile;
use serde_json::{Value, json};

use crate::dataframe::DataFrame;
use crate::session::internal;

/// The `format` member of an [`ExplainVerbose`] document.
pub const VERBOSE_FORMAT: &str = "krishiv.explain-verbose";
/// Its version.
pub const VERBOSE_VERSION: u64 = 1;

/// One optimizer rule application that changed the logical plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerStep {
    /// The rule batch (`Rewrite`, `JoinOrder`, ...).
    pub batch: String,
    /// The round of the batch, from 1.
    pub round: u64,
    /// The rule.
    pub rule: String,
    /// The logical plan after the rule.
    pub plan: String,
    /// The lines the rule removed (`- `) and added (`+ `).
    pub diff: String,
}

/// What `EXPLAIN VERBOSE` prints: the plan at every stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainVerbose {
    /// The parsed (unresolved) plan.
    pub parsed: String,
    /// The analyzed plan.
    pub analyzed: String,
    /// The optimizer's steps that changed the plan, in order.
    pub steps: Vec<OptimizerStep>,
    /// The optimized plan.
    pub optimized: String,
    /// The physical plan.
    pub physical: String,
}

impl ExplainVerbose {
    /// The text `EXPLAIN VERBOSE` prints.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "== Parsed Logical Plan ==\n{}\n== Analyzed Logical Plan ==\n{}\n",
            self.parsed, self.analyzed
        );
        for s in &self.steps {
            let _ = write!(
                out,
                "== Optimizer: {} round {}, {} ==\n{}\n",
                s.batch, s.round, s.rule, s.diff
            );
        }
        let _ = write!(
            out,
            "== Optimized Logical Plan ==\n{}\n== Physical Plan ==\n{}",
            self.optimized, self.physical
        );
        out
    }

    /// The JSON document (format [`VERBOSE_FORMAT`], version
    /// [`VERBOSE_VERSION`]; experimental: may change before 1.0).
    #[must_use]
    pub fn to_json(&self) -> String {
        json!({
            "format": VERBOSE_FORMAT,
            "version": VERBOSE_VERSION,
            "parsed": self.parsed,
            "analyzed": self.analyzed,
            "steps": self.steps.iter().map(|s| json!({
                "batch": s.batch,
                "round": s.round,
                "rule": s.rule,
                "plan": s.plan,
                "diff": s.diff,
            })).collect::<Vec<_>>(),
            "optimized": self.optimized,
            "physical": self.physical,
        })
        .to_string()
    }

    /// Reads a document [`to_json`](Self::to_json) wrote.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` naming the problem: malformed JSON, another format
    /// or version, a missing or mistyped member.
    pub fn from_json(text: &str) -> Result<Self, KrishivError> {
        let bad = |m: &str| internal(format!("invalid EXPLAIN VERBOSE document: {m}"));
        let v: Value = serde_json::from_str(text).map_err(|e| bad(&e.to_string()))?;
        let s = |v: &Value, k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| bad(&format!("{k} is not a string")))
        };
        let n = |v: &Value, k: &str| {
            v.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| bad(&format!("{k} is not an unsigned integer")))
        };
        if s(&v, "format")? != VERBOSE_FORMAT {
            return Err(bad("unknown format"));
        }
        if n(&v, "version")? != VERBOSE_VERSION {
            return Err(bad("unsupported version"));
        }
        let steps = v
            .get("steps")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("steps is not an array"))?
            .iter()
            .map(|st| {
                Ok(OptimizerStep {
                    batch: s(st, "batch")?,
                    round: n(st, "round")?,
                    rule: s(st, "rule")?,
                    plan: s(st, "plan")?,
                    diff: s(st, "diff")?,
                })
            })
            .collect::<Result<_, KrishivError>>()?;
        Ok(Self {
            parsed: s(&v, "parsed")?,
            analyzed: s(&v, "analyzed")?,
            steps,
            optimized: s(&v, "optimized")?,
            physical: s(&v, "physical")?,
        })
    }
}

/// Plans with more lines than this are diffed as all-removed, all-added
/// rather than line by line (the diff is quadratic).
const DIFF_MAX_LINES: usize = 2000;

/// The lines of `before` that `after` drops (`- `) and adds (`+ `), in
/// order: a longest-common-subsequence line diff.
pub(crate) fn line_diff(before: &str, after: &str) -> String {
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    let mut out = String::new();
    if a.len() > DIFF_MAX_LINES || b.len() > DIFF_MAX_LINES {
        for l in &a {
            let _ = writeln!(out, "- {l}");
        }
        for l in &b {
            let _ = writeln!(out, "+ {l}");
        }
        return out;
    }
    // lcs[i][j]: the common subsequence of a[i..] and b[j..].
    let mut lcs = vec![vec![0_u32; b.len() + 1]; a.len() + 1];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            i += 1;
            j += 1;
        } else if i < a.len() && (j == b.len() || lcs[i + 1][j] >= lcs[i][j + 1]) {
            let _ = writeln!(out, "- {}", a[i]);
            i += 1;
        } else {
            let _ = writeln!(out, "+ {}", b[j]);
            j += 1;
        }
    }
    out
}

impl DataFrame {
    /// Runs the query and returns its profile (`EXPLAIN ANALYZE`): the
    /// physical plan with each operator's rows, batches, compute and I/O
    /// time (summed over drivers, with the largest driver's share), peak
    /// memory, spills, runtime filters and hash-table statistics, and each
    /// pipeline's drivers, threads and wall time. The rows are counted and
    /// dropped. Probe lengths are collected too, which costs a little.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![3, 1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// let profile = futures::executor::block_on(s.table("t").explain_analyze())?;
    /// assert!(profile.to_text(Default::default()).starts_with("== Physical Plan (analyzed) =="));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis, planning and execution errors.
    pub async fn explain_analyze(&self) -> Result<QueryProfile, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.profiled(false).await.map(|(_, p)| p) })
            .await
    }

    /// As [`Self::explain_analyze`], also returning the rows.
    ///
    /// # Errors
    /// As [`Self::explain_analyze`].
    pub async fn collect_with_profile(
        &self,
    ) -> Result<(Vec<RecordBatch>, QueryProfile), KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.profiled(true).await })
            .await
    }

    /// Runs the query with detailed metrics; keeps its rows if `keep`.
    pub(crate) async fn profiled(
        &self,
        keep: bool,
    ) -> Result<(Vec<RecordBatch>, QueryProfile), KrishivError> {
        let a = self.analysis().await?;
        let (_, physical) = self.session.physical(&a.analyzed, &a.env).await?;
        let (mut stream, query) = self.session.run_profiled(&physical, &a.env).await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if keep {
                out.push(batch);
            }
        }
        drop(stream);
        // Every driver has merged its counters once the query finished.
        query.finished().await;
        Ok((out, QueryProfile::new(&physical, query.metrics(), true)))
    }

    /// The plan at every stage (`EXPLAIN VERBOSE`): parsed, analyzed, after
    /// each optimizer rule that changed it (with the lines it changed),
    /// optimized and physical.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.sql("SELECT a FROM (SELECT a FROM t) WHERE a > 1"))?;
    /// let v = futures::executor::block_on(df.explain_verbose())?;
    /// assert!(v.steps.iter().any(|s| s.rule == "PushDownFilter"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors.
    pub async fn explain_verbose(&self) -> Result<ExplainVerbose, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.verbose().await })
            .await
    }

    pub(crate) async fn verbose(&self) -> Result<ExplainVerbose, KrishivError> {
        let parsed = self.plan.to_string();
        let a = self.analysis().await?;
        let analyzed = a.analyzed.plan.root().to_string();
        let (optimized, physical, rules) = self
            .session
            .physical_traced(&a.analyzed, &a.env, true)
            .await?;
        let mut before = analyzed.clone();
        let mut steps = Vec::new();
        for r in rules {
            let plan = r.plan.to_string();
            if plan == before {
                continue;
            }
            steps.push(OptimizerStep {
                batch: r.batch.to_owned(),
                round: r.round as u64,
                rule: r.rule.to_owned(),
                diff: line_diff(&before, &plan),
                plan: plan.clone(),
            });
            before = plan;
        }
        Ok(ExplainVerbose {
            parsed,
            analyzed,
            steps,
            optimized: optimized.root().to_string(),
            physical: physical.explain().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_diffs() {
        assert_eq!(line_diff("a\nb\nc", "a\nc"), "- b\n");
        assert_eq!(line_diff("a\nc", "a\nb\nc"), "+ b\n");
        assert_eq!(line_diff("x\ny", "x\nz"), "- y\n+ z\n");
        assert_eq!(line_diff("same", "same"), "");
        let big: String = (0..=DIFF_MAX_LINES).map(|i| format!("{i}\n")).collect();
        let d = line_diff(&big, "0");
        assert!(d.starts_with("- 0\n- 1\n") && d.ends_with("+ 0\n"));
    }

    #[test]
    fn verbose_json_round_trips_and_rejects_bad_input() {
        let v = ExplainVerbose {
            parsed: "p".into(),
            analyzed: "a".into(),
            steps: vec![OptimizerStep {
                batch: "Rewrite".into(),
                round: 1,
                rule: "PushDownFilter".into(),
                plan: "b".into(),
                diff: "- a\n+ b\n".into(),
            }],
            optimized: "b".into(),
            physical: "P".into(),
        };
        let back = ExplainVerbose::from_json(&v.to_json()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, v);
        let text = v.to_text();
        assert!(text.contains("== Optimizer: Rewrite round 1, PushDownFilter ==\n- a\n+ b\n"));
        for (bad, want) in [
            ("{", "invalid EXPLAIN VERBOSE document"),
            (r#"{"format":"x","version":1}"#, "unknown format"),
            (
                r#"{"format":"krishiv.explain-verbose","version":9}"#,
                "unsupported version",
            ),
            (
                r#"{"format":"krishiv.explain-verbose","version":1,"steps":{}}"#,
                "steps is not an array",
            ),
            (
                r#"{"format":"krishiv.explain-verbose","version":1,"steps":[{"batch":1}]}"#,
                "batch is not a string",
            ),
        ] {
            let e = ExplainVerbose::from_json(bad).err().map(|e| e.to_string());
            assert!(
                e.as_deref().is_some_and(|e| e.contains(want)),
                "{bad}: {e:?}"
            );
        }
    }
}
