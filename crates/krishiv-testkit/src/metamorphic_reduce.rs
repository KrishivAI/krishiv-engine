// SPDX-License-Identifier: Apache-2.0
//! Statement-level reduction of a metamorphic finding and Spark's answer on the reduced one
//! (P2-T110b, acceptance 3 and 5).
//!
//! [`crate::metamorphic::reduce`] shrinks the predicate. This module shrinks what the predicate
//! runs over: delta debugging (`ddmin`) of the rows of the `nulls` view, so a finding is reported
//! as a few rows, one predicate and the handful of statements the oracle issues. Those statements
//! are then run on Spark 4.2 ([`PythonSpark`], `tests/scripts/spark_answers.py`); a finding whose
//! statements the engine answers as Spark does is an *oracle artefact* (the relation does not hold
//! for that case in Spark either), and one it answers differently keeps Spark's rows beside its
//! own, which is the evidence for filing it.

use std::collections::HashMap;
use std::fmt;
use std::io::Write as _;
use std::process::{Command, Stdio};

use crate::metamorphic::{
    Finding, NULLS_ROWS, Oracle, Pred, Raw, Rows, SOURCES, canonical, check, nulls_setup, number,
};
use crate::query_oracle::QueryRunner;

/// Minimises `items` with delta debugging: the smallest subset (in the order given) on which
/// `fails` still holds, found by removing chunks of decreasing size. `fails` must hold for the
/// whole of `items`; the result is never empty.
pub fn ddmin(items: &[usize], fails: &mut dyn FnMut(&[usize]) -> bool) -> Vec<usize> {
    let mut cur = items.to_vec();
    let mut n = 2_usize;
    while cur.len() >= 2 {
        let chunk = cur.len().div_ceil(n);
        let mut reduced = false;
        let mut start = 0;
        while start < cur.len() {
            let end = (start + chunk).min(cur.len());
            let rest: Vec<usize> = cur[..start].iter().chain(&cur[end..]).copied().collect();
            if !rest.is_empty() && fails(&rest) {
                cur = rest;
                n = n.saturating_sub(1).max(2);
                reduced = true;
                break;
            }
            start = end;
        }
        if !reduced {
            if n >= cur.len() {
                break;
            }
            n = (n * 2).min(cur.len());
        }
    }
    cur
}

/// Reduces the rows of the `nulls` view a finding over source 0 needs, with the predicate fixed.
/// Other sources read the generated TPC-H files, which are not reduced: all rows are returned.
/// The runner is left with the full view again.
///
/// # Errors
/// The engine's failure to define the view.
pub fn reduce_rows(
    r: &mut dyn QueryRunner,
    f: &Finding,
    small: &Pred,
) -> Result<Vec<usize>, String> {
    let all: Vec<usize> = (0..NULLS_ROWS.len()).collect();
    if f.source != 0 {
        return Ok(all);
    }
    let kept = ddmin(&all, &mut |rows| {
        r.run(&nulls_setup(rows)).is_ok()
            && matches!(check(r, f.oracle, f.source, small), Ok(Some(_)))
    });
    r.run(&nulls_setup(&all))?;
    Ok(kept)
}

/// A runner that remembers every statement it ran.
struct Recording<'a> {
    inner: &'a mut dyn QueryRunner,
    log: Vec<String>,
    answers: Vec<Result<Rows, String>>,
}

impl QueryRunner for Recording<'_> {
    fn run(&mut self, sql: &str) -> Result<Rows, String> {
        let out = self.inner.run(sql);
        self.log.push(sql.to_owned());
        self.answers.push(out.clone());
        out
    }
}

/// The queries `oracle` issues for `p` over `source`, with the engine's answer to each.
pub fn statements(
    r: &mut dyn QueryRunner,
    oracle: Oracle,
    source: usize,
    p: &Pred,
) -> Vec<(String, Result<Rows, String>)> {
    let mut rec = Recording {
        inner: r,
        log: Vec::new(),
        answers: Vec::new(),
    };
    drop(check(&mut rec, oracle, source, p));
    rec.log.into_iter().zip(rec.answers).collect()
}

/// A way to run statements on Spark.
pub trait SparkAnswers {
    /// Runs `setup` (view definitions), then each of `queries`; one answer per query, in order.
    ///
    /// # Errors
    /// Spark could not be started or its output was unreadable.
    fn answer(
        &mut self,
        setup: &[String],
        queries: &[String],
    ) -> Result<Vec<Result<Rows, String>>, String>;
}

/// The Spark 4.2 oracle: `tests/scripts/spark_answers.py` run by `command` (for example
/// `env -u SPARK_HOME uv run --no-project --python 3.12 --with pyspark==4.2.0 python -I`), which reads
/// `{"data": DIR, "setup": [...], "queries": [...]}` on stdin and prints
/// `[{"rows": [[...]]} | {"error": "..."}]`.
#[derive(Debug, Clone)]
pub struct PythonSpark {
    /// The command and its arguments, script path last.
    pub command: Vec<String>,
    /// The TPC-H Parquet directory the script registers as views.
    pub data: std::path::PathBuf,
}

impl PythonSpark {
    /// Reads the script's output.
    ///
    /// # Errors
    /// When the output is not the documented JSON or has the wrong number of answers.
    pub fn parse(text: &str, expected: usize) -> Result<Vec<Result<Rows, String>>, String> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("spark output: {e}"))?;
        let items = v.as_array().ok_or("spark output is not a list")?;
        if items.len() != expected {
            return Err(format!(
                "spark answered {} of {expected} queries",
                items.len()
            ));
        }
        items
            .iter()
            .map(|a| match (a.get("rows"), a.get("error")) {
                (Some(rows), _) => {
                    let rows = rows.as_array().ok_or("`rows` is not a list")?;
                    rows.iter()
                        .map(|row| {
                            row.as_array()
                                .ok_or_else(|| "a row is not a list".to_owned())?
                                .iter()
                                .map(|c| {
                                    c.as_str()
                                        .map(str::to_owned)
                                        .ok_or_else(|| "a cell is not text".to_owned())
                                })
                                .collect()
                        })
                        .collect::<Result<Rows, String>>()
                        .map(Ok)
                }
                (None, Some(e)) => Ok(Err(e.as_str().unwrap_or("error").to_owned())),
                _ => Err("an answer has neither `rows` nor `error`".to_owned()),
            })
            .collect()
    }
}

impl SparkAnswers for PythonSpark {
    fn answer(
        &mut self,
        setup: &[String],
        queries: &[String],
    ) -> Result<Vec<Result<Rows, String>>, String> {
        let (program, args) = self.command.split_first().ok_or("an empty Spark command")?;
        let input = serde_json::json!({
            "data": self.data.display().to_string(),
            "setup": setup,
            "queries": queries,
        });
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("{program}: {e}"))?;
        child
            .stdin
            .take()
            .ok_or("no stdin")?
            .write_all(input.to_string().as_bytes())
            .map_err(|e| e.to_string())?;
        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        if !out.status.success() {
            return Err(format!("the Spark script exited with {}", out.status));
        }
        Self::parse(&String::from_utf8_lossy(&out.stdout), queries.len())
    }
}

/// How the engine's answers compare with Spark's on a reduced finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Same answers on every statement: the relation does not hold for this case in Spark
    /// either, so it is an oracle artefact, not an engine bug.
    OracleArtefact,
    /// The engine answers differently from Spark on `query`: Spark's rows are the evidence.
    EngineDiffers {
        /// The statement.
        query: String,
        /// The engine's answer.
        engine: Result<Rows, String>,
        /// Spark's answer.
        spark: Result<Rows, String>,
    },
    /// Spark could not be asked.
    Unavailable(String),
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OracleArtefact => f.write_str("oracle artefact: Spark 4.2 answers the same"),
            Self::EngineDiffers {
                query,
                engine,
                spark,
            } => {
                write!(
                    f,
                    "engine differs from Spark 4.2 on `{query}`: engine {engine:?}, spark {spark:?}"
                )
            }
            Self::Unavailable(why) => write!(f, "Spark not consulted: {why}"),
        }
    }
}

fn same_cell(a: &str, b: &str) -> bool {
    match (number(a), number(b)) {
        (Some(x), Some(y)) => {
            x == y
                || (x.is_nan() && y.is_nan())
                // The tolerance is for two finite numbers: with an infinity in play the bound itself is
                // infinite and every pair would pass.
                || (x.is_finite() && y.is_finite() && (x - y).abs() <= 1e-9 * x.abs().max(y.abs()))
        }
        _ => a == b,
    }
}

fn same_answer(a: &Result<Rows, String>, b: &Result<Rows, String>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => {
            let (x, y) = (canonical(x.clone()), canonical(y.clone()));
            x.len() == y.len()
                && x.iter().zip(&y).all(|(r, s)| {
                    r.len() == s.len() && r.iter().zip(s).all(|(c, d)| same_cell(c, d))
                })
        }
        // Both failed: the error text differs by engine, the class of outcome is the same.
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

/// Asks Spark the statements of the reduced finding (`small` over the rows `kept` of the `nulls`
/// view) and compares its answers with the engine's.
pub fn spark_verdict(
    r: &mut dyn QueryRunner,
    spark: &mut dyn SparkAnswers,
    f: &Finding,
    small: &Pred,
    kept: &[usize],
) -> Verdict {
    // The engine answers over the same reduced rows Spark gets, then gets the full view back.
    if f.source == 0
        && let Err(e) = r.run(&nulls_setup(kept))
    {
        return Verdict::Unavailable(format!("the engine could not define the reduced view: {e}"));
    }
    let stmts = statements(r, f.oracle, f.source, small);
    if f.source == 0 {
        let all: Vec<usize> = (0..NULLS_ROWS.len()).collect();
        drop(r.run(&nulls_setup(&all)));
    }
    let queries: Vec<String> = stmts.iter().map(|(q, _)| q.clone()).collect();
    let setup = if f.source == 0 {
        vec![nulls_setup(kept)]
    } else {
        Vec::new()
    };
    let theirs = match spark.answer(&setup, &queries) {
        Ok(a) => a,
        Err(e) => return Verdict::Unavailable(e),
    };
    for ((query, engine), spark) in stmts.into_iter().zip(theirs) {
        if !same_answer(&engine, &spark) {
            return Verdict::EngineDiffers {
                query,
                engine,
                spark,
            };
        }
    }
    Verdict::OracleArtefact
}

/// Answers from a fixed table, for tests and for replaying recorded Spark output.
#[derive(Debug, Default)]
pub struct Canned(pub HashMap<String, Result<Rows, String>>);

impl SparkAnswers for Canned {
    fn answer(
        &mut self,
        _setup: &[String],
        queries: &[String],
    ) -> Result<Vec<Result<Rows, String>>, String> {
        queries
            .iter()
            .map(|q| {
                self.0
                    .get(q)
                    .cloned()
                    .ok_or_else(|| format!("no canned answer for {q}"))
            })
            .collect()
    }
}

/// The reduced finding as text: the rows, the predicate, the statements and the verdict.
#[must_use]
pub fn render(f: &Finding, small: &Pred, kept: &[usize], verdict: &Verdict) -> String {
    let rows = if f.source == 0 {
        format!("rows {kept:?} of {} in `nulls`; ", NULLS_ROWS.len())
    } else {
        String::new()
    };
    format!(
        "{:?} over `{}`: {}\n  {rows}predicate: {small}\n  {verdict}\n",
        f.oracle, SOURCES[f.source].from, f.detail
    )
}

/// The predicate of a regression file as the oracles take it.
#[must_use]
pub fn raw(text: &str) -> impl fmt::Display {
    Raw(text.to_owned())
}

#[cfg(test)]
mod tests {
    use super::same_cell;

    #[test]
    fn the_relative_tolerance_is_for_finite_numbers_only() {
        assert!(same_cell("1.0000000001", "1.0000000002"));
        assert!(same_cell("Infinity", "Infinity"));
        assert!(same_cell("NaN", "NaN"));
        assert!(!same_cell("0", "Infinity"));
        assert!(!same_cell("Infinity", "-Infinity"));
        assert!(!same_cell("5", "Infinity"));
    }
}
