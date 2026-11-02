// SPDX-License-Identifier: Apache-2.0
//! Expression-level differential testing against Spark 4.2 (P2-T101).
//!
//! A fixture (`tests/fixtures/spark-4.2.0/expr-differential.json`, written by
//! `tests/scripts/gen_expr_differential.py`) holds seeded random scalar expressions over one typed
//! table, with Spark's answer for each: the column's Spark type and its rows as `HiveResult`
//! prints them, or the Spark error class. Every sub-expression of a generated expression is a
//! case of its own and lists its children, so a mismatch is narrowed to a *minimal* one (a case
//! that disagrees while all of its children agree) with no second run of Spark.
//!
//! [`run`] replays the cases on an [`Evaluator`]; [`SessionEvaluator`] is `krishiv-session` under
//! the golden runner's rendering, so a row prints as the goldens do.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::PathBuf;

use serde_json::Value;

use crate::golden::run::Engine as _;
use crate::golden::session_engine::SessionEngine;
use crate::golden::setup::{SessionSetup, TestKind};

/// What one expression produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expect {
    /// The column's Spark type string (`int`, `decimal(10,2)`) and one rendered cell per row.
    Rows {
        /// `DataType.simpleString`.
        schema: String,
        /// The cells, in the table's `id` order.
        rows: Vec<String>,
    },
    /// The error class the query failed with.
    Error(String),
}

impl fmt::Display for Expect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rows { schema, rows } => write!(f, "{schema} {}", rows.join(" | ")),
            Self::Error(class) => write!(f, "error {class}"),
        }
    }
}

/// One expression, with Spark's answer.
#[derive(Debug, Clone)]
pub struct Case {
    /// The expression's SQL.
    pub sql: String,
    /// `spark.sql.ansi.enabled` for the case.
    pub ansi: bool,
    /// The SQL of the direct sub-expressions (cases of the same `ansi`).
    pub kids: Vec<String>,
    /// Spark's answer.
    pub expect: Expect,
}

/// A parsed fixture.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// The statement that creates the table `t` the expressions read.
    pub setup: String,
    /// All cases: roots and their sub-expressions.
    pub cases: Vec<Case>,
}

fn str_field<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string field `{key}`"))
}

impl Fixture {
    /// Parses the generator's JSON.
    ///
    /// # Errors
    /// When the text is not JSON or a case lacks a field.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let setup = str_field(&v, "setup")?.to_owned();
        let raw = v
            .get("cases")
            .and_then(Value::as_array)
            .ok_or("missing `cases`")?;
        let mut cases = Vec::with_capacity(raw.len());
        for c in raw {
            let expect = if let Some(class) = c.get("error").and_then(Value::as_str) {
                Expect::Error(class.to_owned())
            } else {
                let rows = c
                    .get("rows")
                    .and_then(Value::as_array)
                    .ok_or("case without rows or error")?
                    .iter()
                    .map(|r| r.as_str().map(str::to_owned).ok_or("row is not text"))
                    .collect::<Result<Vec<_>, _>>()?;
                Expect::Rows {
                    schema: str_field(c, "schema")?.to_owned(),
                    rows,
                }
            };
            let kids = c
                .get("kids")
                .and_then(Value::as_array)
                .map(|k| {
                    k.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            cases.push(Case {
                sql: str_field(c, "sql")?.to_owned(),
                ansi: c.get("ansi").and_then(Value::as_bool).unwrap_or(true),
                kids,
                expect,
            });
        }
        Ok(Self { setup, cases })
    }
}

/// Runs one expression of the table `t`.
pub trait Evaluator {
    /// The expression's rows (ordered by `id`) or error class under the given ANSI mode.
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect;
}

/// A case whose answer differs from Spark's while every child's agrees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// The expression.
    pub sql: String,
    /// The ANSI mode it ran under.
    pub ansi: bool,
    /// Spark's answer.
    pub spark: Expect,
    /// The engine's answer.
    pub got: Expect,
}

impl Mismatch {
    /// The stable key a known-failure list uses: `ansi|sql`.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{}|{}", u8::from(self.ansi), self.sql)
    }
}

/// Replays every case and returns the minimal mismatches, sorted by key.
pub fn run(fixture: &Fixture, ev: &mut dyn Evaluator) -> Vec<Mismatch> {
    let mut bad: BTreeMap<(&str, bool), Option<Expect>> = BTreeMap::new();
    for c in &fixture.cases {
        let got = ev.eval(&c.sql, c.ansi);
        bad.insert((&c.sql, c.ansi), (got != c.expect).then_some(got));
    }
    let mut out: Vec<Mismatch> = fixture
        .cases
        .iter()
        .filter_map(|c| {
            let got = bad.get(&(c.sql.as_str(), c.ansi))?.clone()?;
            let kid_bad = c
                .kids
                .iter()
                .any(|k| bad.get(&(k.as_str(), c.ansi)).is_some_and(Option::is_some));
            (!kid_bad).then(|| Mismatch {
                sql: c.sql.clone(),
                ansi: c.ansi,
                spark: c.expect.clone(),
                got,
            })
        })
        .collect();
    out.sort_by_key(Mismatch::key);
    out.dedup_by_key(|m| m.key());
    out
}

/// `krishiv-session` as an [`Evaluator`].
#[derive(Debug)]
pub struct SessionEvaluator {
    engine: SessionEngine,
    ansi: Option<bool>,
}

impl SessionEvaluator {
    /// A session with the fixture's table.
    ///
    /// # Errors
    /// When the session or the table's setup statement fails.
    pub fn new(setup: &str) -> Result<Self, String> {
        let mut engine = SessionEngine::new().map_err(|e| e.message)?;
        engine
            .reset(&SessionSetup::plain(TestKind::Regular))
            .map_err(|e| e.message)?;
        engine.execute(setup).map_err(|e| e.message)?;
        Ok(Self { engine, ansi: None })
    }
}

impl Evaluator for SessionEvaluator {
    fn eval(&mut self, sql: &str, ansi: bool) -> Expect {
        if self.ansi != Some(ansi) {
            let set = format!("SET spark.sql.ansi.enabled={ansi}");
            if let Err(e) = self.engine.execute(&set) {
                return Expect::Error(format!("HARNESS: {}", e.message));
            }
            self.ansi = Some(ansi);
        }
        match self
            .engine
            .execute(&format!("SELECT {sql} AS c FROM t ORDER BY id"))
        {
            Ok(out) => Expect::Rows {
                schema: out
                    .schema
                    .strip_prefix("struct<c:")
                    .and_then(|s| s.strip_suffix('>'))
                    .unwrap_or(&out.schema)
                    .to_owned(),
                rows: out.rows.into_iter().map(|r| r.join("\t")).collect(),
            },
            Err(e) => Expect::Error(
                e.class
                    .unwrap_or_else(|| format!("UNCLASSIFIED: {}", e.message)),
            ),
        }
    }
}

/// The ratchet against a known-failure list of [`Mismatch::key`]s.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Ratchet {
    /// Mismatches the list does not hold: new failures.
    pub new: Vec<String>,
    /// Listed keys that now agree with Spark (or are gone): to be removed from the list.
    pub fixed: Vec<String>,
}

/// Compares `found` with `known` (sorted keys).
#[must_use]
pub fn ratchet(found: &[Mismatch], known: &[String]) -> Ratchet {
    let keys: Vec<String> = found.iter().map(Mismatch::key).collect();
    ratchet_keys(&keys, known)
}

/// [`ratchet`] over plain keys.
#[must_use]
pub fn ratchet_keys(found: &[String], known: &[String]) -> Ratchet {
    let found: std::collections::BTreeSet<&String> = found.iter().collect();
    let known: std::collections::BTreeSet<&String> = known.iter().collect();
    Ratchet {
        new: found.difference(&known).map(|k| (*k).clone()).collect(),
        fixed: known.difference(&found).map(|k| (*k).clone()).collect(),
    }
}

/// Parses a known-failure list (a JSON array of keys).
///
/// # Errors
/// When the text is not a JSON array of strings.
pub fn parse_known(text: &str) -> Result<Vec<String>, String> {
    serde_json::from_str(text).map_err(|e| e.to_string())
}

/// Renders a known-failure list: one key per line, sorted.
#[must_use]
pub fn render_known(found: &[Mismatch]) -> String {
    let keys: Vec<String> = found.iter().map(Mismatch::key).collect();
    render_known_keys(&keys)
}

/// [`render_known`] over plain keys.
#[must_use]
pub fn render_known_keys(keys: &[String]) -> String {
    let mut sorted = keys.to_vec();
    sorted.sort();
    let mut out = String::from("[\n");
    for (n, k) in sorted.iter().enumerate() {
        let sep = if n + 1 == sorted.len() { "" } else { "," };
        out.push_str(&format!("  {}{sep}\n", Value::from(k.as_str())));
    }
    out.push_str("]\n");
    out
}

/// Usage text.
pub const USAGE: &str = "usage: expr-diff --fixture FILE [--known FILE] [--out FILE]";

/// Runs the replay for `args` (without the program name); `Ok(true)` when no mismatch is new.
///
/// # Errors
/// A usage, I/O or fixture error, as text.
pub fn cli(
    args: impl IntoIterator<Item = String>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<bool, String> {
    let (mut fixture, mut known, mut out) = (None, None, None);
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let v = args
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| USAGE.to_owned())?;
        match flag.as_str() {
            "--fixture" => fixture = Some(v),
            "--known" => known = Some(v),
            "--out" => out = Some(v),
            _ => return Err(USAGE.to_owned()),
        }
    }
    let path = fixture.ok_or_else(|| USAGE.to_owned())?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let fixture = Fixture::parse(&text)?;
    let mut ev = SessionEvaluator::new(&fixture.setup)?;
    let found = run(&fixture, &mut ev);
    for m in &found {
        writeln!(
            &mut *stdout,
            "{}\n  spark: {}\n  got:   {}",
            m.key(),
            m.spark,
            m.got
        )
        .map_err(|e| e.to_string())?;
    }
    let known = match known {
        Some(p) => {
            parse_known(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)?
        }
        None => Vec::new(),
    };
    let r = ratchet(&found, &known);
    writeln!(
        &mut *stderr,
        "{} cases, {} minimal mismatches, {} new, {} fixed",
        fixture.cases.len(),
        found.len(),
        r.new.len(),
        r.fixed.len()
    )
    .map_err(|e| e.to_string())?;
    if let Some(p) = out {
        let rows: Vec<serde_json::Value> = found
            .iter()
            .map(|m| {
                serde_json::json!({
                    "key": m.key(), "spark": m.spark.to_string(), "got": m.got.to_string(),
                    "new": r.new.contains(&m.key()),
                })
            })
            .collect();
        let text = serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?;
        std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(r.new.is_empty())
}
