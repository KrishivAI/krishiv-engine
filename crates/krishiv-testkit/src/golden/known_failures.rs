// SPDX-License-Identifier: Apache-2.0
//! The frozen statement-level baseline of the session engine's golden run
//! (`tests/spark-golden/known-failures.json`): every statement that does not pass, each with a
//! category, and the ratchet that keeps the list from growing.
//!
//! ```json
//! {"format_version": 1, "spark_tag": "v4.2.0", "frozen_at": "2026-10-22",
//!  "categories": {
//!    "ddl": {"kind": "unsupported", "task": "P2-T075a", "summary": "DDL statements"},
//!    "float-text": {"kind": "divergence", "adr": "0023", "summary": "..."},
//!    "wrong-sum": {"kind": "bug", "owner": "U-agg", "summary": "..."}},
//!  "cases": {"group-by": {"ddl": [0, 1], "wrong-sum": [7]}}}
//! ```
//!
//! A category is one of three kinds, each with a mandatory reference: an unsupported feature
//! names the plan task that adds it (`P2-T075a`), a known divergence names the ADR that records
//! it (ADR 0023, the Spark-divergence register), a bug names its owner in the internal status
//! list. Indices are statement positions in the golden file (0-based). The ratchet
//! ([`KnownFailures::check`]) fails on a statement that fails but is not listed (a regression)
//! and on a listed statement that passes (the list must shrink with it).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::GoldenSet;
use super::classify::{Failure, FileNames, Rule, classify, is_command};
use super::parse::Statement;
use super::run::{RunResult, load_statements};

/// Format version this loader reads.
pub const KNOWN_FAILURES_FORMAT_VERSION: u64 = 1;

/// The ADR a known divergence must cite.
pub const DIVERGENCE_ADR: &str = "0023";

/// What a category is, with its mandatory reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CategoryKind {
    /// A feature Krishiv does not have yet; the plan task that adds it.
    Unsupported {
        /// Plan task ID, `P<phase>-T<number>[letter]`.
        task: String,
    },
    /// A deliberate difference from Spark recorded in an ADR.
    Divergence {
        /// ADR number ([`DIVERGENCE_ADR`]).
        adr: String,
    },
    /// A defect; who owns the fix.
    Bug {
        /// Owner as named in the internal status list.
        owner: String,
    },
}

/// A failure category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Category {
    /// Kind and reference.
    pub kind: CategoryKind,
    /// One line on what fails and why.
    pub summary: String,
}

/// A problem with the known-failures file.
#[derive(Debug, thiserror::Error)]
pub enum KnownFailuresError {
    /// Reading the file failed.
    #[error("reading {path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        source: std::io::Error,
    },
    /// Not valid JSON, or the wrong shape.
    #[error("invalid known-failures file: {0}")]
    Invalid(String),
}

/// The frozen baseline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnownFailures {
    /// Spark tag the baseline was taken on.
    pub spark_tag: String,
    /// Date of the freeze (`YYYY-MM-DD`).
    pub frozen_at: String,
    /// Categories by key.
    pub categories: BTreeMap<String, Category>,
    /// Classification rules for new failures, in order ([`super::classify`]).
    pub rules: Vec<Rule>,
    /// Per golden file: statement index → category key.
    pub cases: BTreeMap<String, BTreeMap<u32, String>>,
}

/// What [`KnownFailures::update`] changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Update {
    /// Statements added, with their category.
    pub added: Vec<(Drift, String)>,
    /// Listed statements removed because they pass (or no longer exist).
    pub removed: Vec<Drift>,
    /// Failing statements no rule classifies; nothing is written while there are any.
    pub unclassified: Vec<Drift>,
}

/// A statement whose outcome disagrees with the baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drift {
    /// Golden file.
    pub file: String,
    /// Statement index.
    pub index: u32,
    /// Why it failed (empty for a statement that now passes).
    pub detail: String,
}

/// The ratchet's verdict.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ratchet {
    /// Statements that fail but are not listed: regressions (or new statements).
    pub newly_failing: Vec<Drift>,
    /// Listed statements that pass: remove them from the list.
    pub newly_passing: Vec<Drift>,
    /// Listed files the run did not run (renamed, removed or excluded).
    pub missing_files: Vec<String>,
}

impl Ratchet {
    /// Whether the run matches the baseline exactly.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.newly_failing.is_empty()
            && self.newly_passing.is_empty()
            && self.missing_files.is_empty()
    }

    /// A readable account of every disagreement.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut s = String::new();
        if !self.newly_failing.is_empty() {
            s.push_str(&format!(
                "{} statement(s) fail that the baseline does not list (fix them, or list them \
                 with a category in tests/spark-golden/known-failures.json):\n",
                self.newly_failing.len()
            ));
            for d in &self.newly_failing {
                s.push_str(&format!("  {}#{}: {}\n", d.file, d.index, d.detail));
            }
        }
        if !self.newly_passing.is_empty() {
            s.push_str(&format!(
                "{} listed statement(s) now pass; remove them from \
                 tests/spark-golden/known-failures.json:\n",
                self.newly_passing.len()
            ));
            for d in &self.newly_passing {
                s.push_str(&format!("  {}#{}\n", d.file, d.index));
            }
        }
        for f in &self.missing_files {
            s.push_str(&format!("listed file `{f}` did not run\n"));
        }
        s
    }
}

fn is_task_id(s: &str) -> bool {
    // `P<digit>-T<digits>[a-z]`
    let Some(rest) = s.strip_prefix('P') else {
        return false;
    };
    let Some((phase, task)) = rest.split_once("-T") else {
        return false;
    };
    let digits = task.trim_end_matches(|c: char| c.is_ascii_lowercase());
    !phase.is_empty()
        && phase.chars().all(|c| c.is_ascii_digit())
        && !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit())
        && task.len() - digits.len() <= 1
}

impl KnownFailures {
    /// Default path in the repo.
    #[must_use]
    pub fn default_path() -> PathBuf {
        crate::tests_dir()
            .join("spark-golden")
            .join("known-failures.json")
    }

    /// Load from `path`.
    pub fn load(path: &Path) -> Result<Self, KnownFailuresError> {
        let text = std::fs::read_to_string(path).map_err(|source| KnownFailuresError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text)
    }

    /// Parse JSON text, validating every category and case.
    pub fn parse(text: &str) -> Result<Self, KnownFailuresError> {
        let bad = |m: String| KnownFailuresError::Invalid(m);
        let root: Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        if root.get("format_version").and_then(Value::as_u64) != Some(KNOWN_FAILURES_FORMAT_VERSION)
        {
            return Err(bad("unsupported or missing `format_version`".into()));
        }
        let text_field = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_owned);
        let spark_tag =
            text_field(&root, "spark_tag").ok_or_else(|| bad("missing `spark_tag`".into()))?;
        let frozen_at =
            text_field(&root, "frozen_at").ok_or_else(|| bad("missing `frozen_at`".into()))?;
        let mut categories = BTreeMap::new();
        for (key, c) in root
            .get("categories")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("missing `categories` object".into()))?
        {
            let summary = text_field(c, "summary")
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| bad(format!("category `{key}` without a `summary`")))?;
            let reference = |k: &str| {
                text_field(c, k)
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| bad(format!("category `{key}` without `{k}`")))
            };
            let kind = match c.get("kind").and_then(Value::as_str) {
                Some("unsupported") => {
                    let task = reference("task")?;
                    if !is_task_id(&task) {
                        return Err(bad(format!("category `{key}`: `{task}` is not a task ID")));
                    }
                    CategoryKind::Unsupported { task }
                }
                Some("divergence") => {
                    let adr = reference("adr")?;
                    if adr != DIVERGENCE_ADR {
                        return Err(bad(format!(
                            "category `{key}`: a divergence cites ADR {DIVERGENCE_ADR}, not `{adr}`"
                        )));
                    }
                    CategoryKind::Divergence { adr }
                }
                Some("bug") => CategoryKind::Bug {
                    owner: reference("owner")?,
                },
                other => {
                    return Err(bad(format!("category `{key}`: unknown kind {other:?}")));
                }
            };
            categories.insert(key.clone(), Category { kind, summary });
        }
        let mut rules = Vec::new();
        for r in root
            .get("rules")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("missing `rules` array".into()))?
        {
            let rule = Rule::parse(r).map_err(bad)?;
            if !categories.contains_key(&rule.category) {
                return Err(bad(format!(
                    "a rule names the unknown category `{}`",
                    rule.category
                )));
            }
            rules.push(rule);
        }
        let mut cases = BTreeMap::new();
        for (file, by_category) in root
            .get("cases")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("missing `cases` object".into()))?
        {
            let mut indices = BTreeMap::new();
            for (key, list) in by_category
                .as_object()
                .ok_or_else(|| bad(format!("`{file}`: expected an object of categories")))?
            {
                if !categories.contains_key(key) {
                    return Err(bad(format!("`{file}`: unknown category `{key}`")));
                }
                for i in list
                    .as_array()
                    .ok_or_else(|| bad(format!("`{file}`/`{key}`: expected an array")))?
                {
                    let i = i
                        .as_u64()
                        .and_then(|i| u32::try_from(i).ok())
                        .ok_or_else(|| bad(format!("`{file}`/`{key}`: bad index {i}")))?;
                    if indices.insert(i, key.clone()).is_some() {
                        return Err(bad(format!("`{file}`#{i} is listed twice")));
                    }
                }
            }
            if indices.is_empty() {
                return Err(bad(format!("`{file}` lists no statements")));
            }
            cases.insert(file.clone(), indices);
        }
        Ok(Self {
            spark_tag,
            frozen_at,
            categories,
            rules,
            cases,
        })
    }

    /// The JSON form (stable key order; one file per line in the pretty form).
    #[must_use]
    pub fn to_json(&self) -> Value {
        let categories: Map<String, Value> = self
            .categories
            .iter()
            .map(|(k, c)| {
                let v = match &c.kind {
                    CategoryKind::Unsupported { task } => {
                        json!({"kind": "unsupported", "task": task, "summary": c.summary})
                    }
                    CategoryKind::Divergence { adr } => {
                        json!({"kind": "divergence", "adr": adr, "summary": c.summary})
                    }
                    CategoryKind::Bug { owner } => {
                        json!({"kind": "bug", "owner": owner, "summary": c.summary})
                    }
                };
                (k.clone(), v)
            })
            .collect();
        let cases: Map<String, Value> = self
            .cases
            .iter()
            .map(|(file, indices)| {
                let mut by: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
                for (i, k) in indices {
                    by.entry(k).or_default().push(*i);
                }
                (file.clone(), json!(by))
            })
            .collect();
        json!({
            "format_version": KNOWN_FAILURES_FORMAT_VERSION,
            "spark_tag": self.spark_tag,
            "frozen_at": self.frozen_at,
            "categories": categories,
            "rules": self.rules.iter().map(Rule::to_json).collect::<Vec<_>>(),
            "cases": cases,
        })
    }

    /// The file text: pretty JSON, with each golden file's cases on one line so that a change
    /// to one file is a one-line diff.
    #[must_use]
    pub fn to_text(&self) -> String {
        let doc = self.to_json();
        let mut head = doc.clone();
        if let Some(o) = head.as_object_mut() {
            o.remove("cases");
        }
        let head = serde_json::to_string_pretty(&head).unwrap_or_default();
        let head = head.trim_end().trim_end_matches('}').trim_end();
        let mut out = format!("{head},\n  \"cases\": {{\n");
        let files = doc.get("cases").and_then(Value::as_object);
        let n = files.map_or(0, Map::len);
        for (i, (file, by)) in files.into_iter().flatten().enumerate() {
            let sep = if i + 1 == n { "" } else { "," };
            out.push_str(&format!("    {}: {by}{sep}\n", Value::String(file.clone())));
        }
        out.push_str("  }\n}\n");
        out
    }

    /// [`Self::update`] for a run over `set`: the statements come from the set's files and the
    /// function groups from the Spark function matrix ([`crate::fn_matrix`]).
    pub fn update_with_set(&mut self, run: &RunResult, set: &GoldenSet) -> Result<Update, String> {
        let mut statements = BTreeMap::new();
        for pair in &set.pairs {
            statements.insert(
                pair.name.clone(),
                load_statements(pair).map_err(|e| e.to_string())?,
            );
        }
        let families: BTreeMap<String, String> = crate::fn_matrix::FnMatrix::load_default()
            .map_err(|e| e.to_string())?
            .functions
            .into_iter()
            .map(|f| (f.name.to_ascii_lowercase(), f.family))
            .collect();
        Ok(self.update(run, &statements, &families))
    }

    /// The ratchet over the file at `path`: `Ok` with a one-line summary when `run` fails
    /// exactly the listed statements and every category is used, else `Err` with every
    /// disagreement.
    pub fn check_file(path: &Path, run: &RunResult) -> Result<String, String> {
        let known = Self::load(path).map_err(|e| e.to_string())?;
        let r = known.check(run);
        if !r.is_clean() {
            return Err(r.describe());
        }
        let unused = known.unused_categories();
        if !unused.is_empty() {
            return Err(format!(
                "categories without cases (remove them): {unused:?}"
            ));
        }
        Ok(format!(
            "ratchet: the run matches the {} known failures of {}",
            known.listed(),
            path.display()
        ))
    }

    /// [`Self::update_with_set`] on the file at `path`, rewriting it; `Err` (and the file
    /// untouched) when a failure matches no rule.
    pub fn update_file(path: &Path, run: &RunResult, set: &GoldenSet) -> Result<String, String> {
        let mut known = Self::load(path).map_err(|e| e.to_string())?;
        let up = known.update_with_set(run, set)?;
        if !up.unclassified.is_empty() {
            let mut msg = format!(
                "{} failure(s) match no rule; add a rule or category to {} (nothing written):\n",
                up.unclassified.len(),
                path.display()
            );
            for d in &up.unclassified {
                msg.push_str(&format!("  {}#{}: {}\n", d.file, d.index, d.detail));
            }
            return Err(msg);
        }
        std::fs::write(path, known.to_text()).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(format!(
            "ratchet: {} added, {} removed, {} listed in {}",
            up.added.len(),
            up.removed.len(),
            known.listed(),
            path.display()
        ))
    }

    /// Bring the baseline in line with `run`: drop the listed statements that pass (or are
    /// gone), keep the categories of the ones still failing, and classify new failures with
    /// [`Self::rules`]. `statements` holds each run file's statements (for the rules);
    /// `families` maps function names to their Spark function group. Categories left without
    /// cases are dropped with their rules. Nothing changes while a failure is unclassified.
    pub fn update(
        &mut self,
        run: &RunResult,
        statements: &BTreeMap<String, Vec<Statement>>,
        families: &BTreeMap<String, String>,
    ) -> Update {
        let mut up = Update::default();
        let mut next: BTreeMap<String, BTreeMap<u32, String>> = BTreeMap::new();
        let empty = Vec::new();
        for f in &run.files {
            let stmts = statements.get(&f.name).unwrap_or(&empty);
            let names = FileNames::of(stmts);
            let listed = self.cases.get(&f.name);
            let mut kept = BTreeMap::new();
            for c in &f.cases {
                let drift = || Drift {
                    file: f.name.clone(),
                    index: c.index,
                    detail: c.detail.clone(),
                };
                let old = listed.and_then(|l| l.get(&c.index));
                if c.passed() {
                    if old.is_some() {
                        up.removed.push(drift());
                    }
                    continue;
                }
                if let Some(k) = old {
                    kept.insert(c.index, k.clone());
                    continue;
                }
                let query = usize::try_from(c.index)
                    .ok()
                    .and_then(|i| stmts.get(i))
                    .map_or("", |s| s.query.as_str());
                let failure = Failure {
                    file: &f.name,
                    query,
                    detail: &c.detail,
                    command: is_command(query),
                    depends: names.mentioned_by(query),
                };
                match classify(&self.rules, &failure, families) {
                    Some(k) => {
                        up.added.push((drift(), k.to_owned()));
                        kept.insert(c.index, k.to_owned());
                    }
                    None => up.unclassified.push(drift()),
                }
            }
            if let Some(l) = listed {
                let present: BTreeSet<u32> = f.cases.iter().map(|c| c.index).collect();
                for i in l.keys().filter(|i| !present.contains(i)) {
                    up.removed.push(Drift {
                        file: f.name.clone(),
                        index: *i,
                        detail: String::new(),
                    });
                }
            }
            if !kept.is_empty() {
                next.insert(f.name.clone(), kept);
            }
        }
        if !up.unclassified.is_empty() {
            return up;
        }
        let used: BTreeSet<String> = next.values().flat_map(BTreeMap::values).cloned().collect();
        self.categories.retain(|k, _| used.contains(k));
        self.rules.retain(|r| used.contains(&r.category));
        self.cases = next;
        up
    }

    /// Categories no listed statement uses (stale after fixes; [`Self::update`] drops them).
    #[must_use]
    pub fn unused_categories(&self) -> Vec<&str> {
        let used: BTreeSet<&str> = self
            .cases
            .values()
            .flat_map(BTreeMap::values)
            .map(String::as_str)
            .collect();
        self.categories
            .keys()
            .map(String::as_str)
            .filter(|k| !used.contains(k))
            .collect()
    }

    /// Number of listed statements.
    #[must_use]
    pub fn listed(&self) -> usize {
        self.cases.values().map(BTreeMap::len).sum()
    }

    /// Listed statements per category key.
    #[must_use]
    pub fn counts(&self) -> BTreeMap<&str, usize> {
        let mut out = BTreeMap::new();
        for k in self.cases.values().flat_map(BTreeMap::values) {
            *out.entry(k.as_str()).or_default() += 1;
        }
        out
    }

    /// Compare `run` with the baseline.
    #[must_use]
    pub fn check(&self, run: &RunResult) -> Ratchet {
        let mut r = Ratchet::default();
        let empty = BTreeMap::new();
        let mut ran = BTreeSet::new();
        for f in &run.files {
            ran.insert(f.name.as_str());
            let listed = self.cases.get(&f.name).unwrap_or(&empty);
            for c in &f.cases {
                match (c.passed(), listed.contains_key(&c.index)) {
                    (false, false) => r.newly_failing.push(Drift {
                        file: f.name.clone(),
                        index: c.index,
                        detail: c.detail.clone(),
                    }),
                    (true, true) => r.newly_passing.push(Drift {
                        file: f.name.clone(),
                        index: c.index,
                        detail: String::new(),
                    }),
                    _ => {}
                }
            }
            // A listed index the file does not have (the file shrank) counts as passing.
            let present: BTreeSet<u32> = f.cases.iter().map(|c| c.index).collect();
            for i in listed.keys().filter(|i| !present.contains(i)) {
                r.newly_passing.push(Drift {
                    file: f.name.clone(),
                    index: *i,
                    detail: String::new(),
                });
            }
        }
        r.missing_files = self
            .cases
            .keys()
            .filter(|f| !ran.contains(f.as_str()))
            .cloned()
            .collect();
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::run::{Case, FileResult};

    type R = Result<(), Box<dyn std::error::Error>>;

    const DOC: &str = r#"{"format_version":1,"spark_tag":"v4.2.0","frozen_at":"2026-10-22",
      "categories":{
        "ddl":{"kind":"unsupported","task":"P2-T075a","summary":"DDL"},
        "text":{"kind":"divergence","adr":"0023","summary":"float text"},
        "sum":{"kind":"bug","owner":"U-agg","summary":"wrong sum"}},
      "rules":[{"category":"ddl","command":true},{"category":"sum","detail":"rows differ"}],
      "cases":{"a":{"ddl":[0,1],"sum":[3]},"b":{"text":[2]}}}"#;

    fn case(index: u32, passed: bool) -> Case {
        Case {
            index,
            result: Some(passed),
            detail: if passed {
                String::new()
            } else {
                "rows differ".into()
            },
            ..Case::default()
        }
    }

    fn file(name: &str, cases: Vec<Case>) -> FileResult {
        FileResult {
            name: name.into(),
            cases,
            ..FileResult::default()
        }
    }

    #[test]
    fn parses_and_round_trips() -> R {
        let k = KnownFailures::parse(DOC)?;
        assert_eq!(k.listed(), 4);
        assert_eq!(k.counts().get("ddl"), Some(&2));
        assert!(k.unused_categories().is_empty());
        let k2 = KnownFailures::parse(&DOC.replace(r#","b":{"text":[2]}"#, ""))?;
        assert_eq!(k2.unused_categories(), ["text"]);
        assert_eq!(
            k.categories["sum"].kind,
            CategoryKind::Bug {
                owner: "U-agg".into()
            }
        );
        let again = KnownFailures::parse(&k.to_json().to_string())?;
        assert_eq!(again, k);
        Ok(())
    }

    #[test]
    fn rejects_bad_references_and_shapes() {
        let cases = [
            (r#""task":"T075a""#, "is not a task ID"),
            (r#""task":"P2-T075ab""#, "is not a task ID"),
            (r#""task":"P2T075""#, "is not a task ID"),
            (r#""task":"P-T075""#, "is not a task ID"),
        ];
        for (task, msg) in cases {
            let doc = DOC.replace(r#""task":"P2-T075a""#, task);
            let err = KnownFailures::parse(&doc)
                .map(drop)
                .unwrap_err()
                .to_string();
            assert!(err.contains(msg), "{err}");
        }
        let checks = [
            (
                DOC.replace(r#""adr":"0023""#, r#""adr":"0012""#),
                "cites ADR 0023",
            ),
            (DOC.replace(r#","owner":"U-agg""#, ""), "without `owner`"),
            (
                DOC.replace(r#""kind":"bug""#, r#""kind":"later""#),
                "unknown kind",
            ),
            (
                DOC.replace(r#""summary":"wrong sum""#, r#""summary":" ""#),
                "without a `summary`",
            ),
            (
                DOC.replace(r#""b":{"text":[2]}"#, r#""b":{"nope":[2]}"#),
                "unknown category",
            ),
            (DOC.replace(r#""sum":[3]"#, r#""sum":[1]"#), "listed twice"),
            (DOC.replace(r#""sum":[3]"#, r#""sum":[-1]"#), "bad index"),
            (
                DOC.replace(r#""format_version":1"#, r#""format_version":2"#),
                "format_version",
            ),
            (
                DOC.replace(r#""b":{"text":[2]}"#, r#""b":{}"#),
                "lists no statements",
            ),
            (
                DOC.replace(
                    r#"{"category":"ddl","command":true}"#,
                    r#"{"category":"zzz","command":true}"#,
                ),
                "unknown category `zzz`",
            ),
            (
                DOC.replace(
                    r#"{"category":"ddl","command":true}"#,
                    r#"{"category":"ddl"}"#,
                ),
                "no condition",
            ),
            (DOC.replace(r#""rules":"#, r#""rulez":"#), "missing `rules`"),
        ];
        for (doc, msg) in checks {
            let err = KnownFailures::parse(&doc)
                .map(drop)
                .unwrap_err()
                .to_string();
            assert!(err.contains(msg), "{msg}: {err}");
        }
    }

    #[test]
    fn ratchet_flags_new_failures_and_new_passes() -> R {
        let k = KnownFailures::parse(DOC)?;
        let run = RunResult {
            files: vec![
                file(
                    "a",
                    vec![
                        case(0, false),
                        case(1, true),
                        case(2, false),
                        case(3, false),
                    ],
                ),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        let r = k.check(&run);
        assert_eq!(
            r.newly_failing,
            [Drift {
                file: "a".into(),
                index: 2,
                detail: "rows differ".into()
            }]
        );
        assert_eq!(r.newly_passing.len(), 1);
        assert_eq!(
            (r.newly_passing[0].file.as_str(), r.newly_passing[0].index),
            ("a", 1)
        );
        assert!(r.missing_files.is_empty());
        assert!(!r.is_clean());
        let text = r.describe();
        assert!(
            text.contains("a#2: rows differ") && text.contains("remove them"),
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn ratchet_is_clean_on_the_baseline_and_flags_missing_files() -> R {
        let k = KnownFailures::parse(DOC)?;
        let mut run = RunResult {
            files: vec![
                file(
                    "a",
                    vec![
                        case(0, false),
                        case(1, false),
                        case(2, true),
                        case(3, false),
                    ],
                ),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        assert!(k.check(&run).is_clean());
        run.files.truncate(1);
        let r = k.check(&run);
        assert_eq!(r.missing_files, ["b"]);
        assert!(r.describe().contains("listed file `b` did not run"));
        // A listed statement the file no longer has must be removed too.
        run.files[0].cases.truncate(3);
        assert_eq!(k.check(&run).newly_passing.len(), 1);
        Ok(())
    }

    fn stmt(q: &str) -> Statement {
        Statement {
            query: q.into(),
            schema: String::new(),
            expected: crate::golden::parse::Expected::Rows(String::new()),
        }
    }

    #[test]
    fn update_keeps_categories_drops_passes_and_classifies_new_failures() -> R {
        let mut k = KnownFailures::parse(DOC)?;
        let run = RunResult {
            files: vec![
                // a#0 still fails (kept as ddl), a#1 passes (removed), a#2 is new (a command: ddl),
                // a#3 is gone (removed); b is not run (dropped with `text`).
                file("a", vec![case(0, false), case(1, true), case(2, false)]),
                // c#0 is new and matches the `sum` rule by its detail.
                file("c", vec![case(0, false)]),
            ],
            ..RunResult::default()
        };
        let statements = BTreeMap::from([
            (
                "a".to_owned(),
                vec![
                    stmt("CREATE VIEW v AS SELECT 1"),
                    stmt("SELECT 1"),
                    stmt("SET x=1"),
                ],
            ),
            ("c".to_owned(), vec![stmt("SELECT 2")]),
        ]);
        let up = k.update(&run, &statements, &BTreeMap::new());
        assert!(up.unclassified.is_empty());
        assert_eq!(
            up.added
                .iter()
                .map(|(d, c)| (d.file.as_str(), d.index, c.as_str()))
                .collect::<Vec<_>>(),
            [("a", 2, "ddl"), ("c", 0, "sum")]
        );
        assert_eq!(
            up.removed
                .iter()
                .map(|d| (d.file.as_str(), d.index))
                .collect::<Vec<_>>(),
            [("a", 1), ("a", 3)]
        );
        assert_eq!(k.listed(), 3);
        assert!(!k.categories.contains_key("text"));
        assert!(k.check(&run).is_clean());
        // The written text parses back to the same baseline, one line per file.
        let text = k.to_text();
        assert_eq!(KnownFailures::parse(&text)?, k);
        assert!(text.contains("\n    \"a\": {\"ddl\":[0,2]},\n"), "{text}");
        Ok(())
    }

    #[test]
    fn update_changes_nothing_while_a_failure_is_unclassified() -> R {
        let mut k = KnownFailures::parse(DOC)?;
        let before = k.clone();
        let mut run = RunResult::default();
        let mut odd = case(5, false);
        odd.detail = "error X: odd".into();
        run.files = vec![file("a", vec![case(0, false), odd])];
        let statements = BTreeMap::from([("a".to_owned(), vec![stmt("SELECT 1"); 6])]);
        let up = k.update(&run, &statements, &BTreeMap::new());
        assert_eq!(up.unclassified.len(), 1);
        assert_eq!(up.unclassified[0].index, 5);
        assert_eq!(k, before);
        Ok(())
    }

    #[test]
    fn update_with_set_reads_statements_and_function_groups() -> R {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir_all(dir.path().join("inputs"))?;
        std::fs::create_dir_all(dir.path().join("results"))?;
        std::fs::write(dir.path().join("inputs/z.sql"), "select upper('x');\n")?;
        std::fs::write(
            dir.path().join("results/z.sql.out"),
            "-- !query\nselect upper('x')\n-- !query schema\nstruct<upper(x):string>\n-- !query output\nX\n",
        )?;
        let set = GoldenSet::scan(dir.path())?;
        let mut k = KnownFailures::parse(&DOC.replace(
            r#"{"category":"sum","detail":"rows differ"}"#,
            r#"{"category":"sum","family":["string"]}"#,
        ))?;
        let mut run = RunResult::default();
        let mut c = case(0, false);
        c.detail =
            "error KRISHIV_UNSUPPORTED_FEATURE: The feature function upper is not supported".into();
        run.files = vec![file("z", vec![c])];
        let up = k.update_with_set(&run, &set)?;
        assert_eq!(up.added.len(), 1, "{up:?}");
        assert_eq!(up.added[0].1, "sum");
        Ok(())
    }

    #[test]
    fn files_are_checked_and_updated_in_place() -> R {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("known.json");
        let missing = KnownFailures::check_file(&path, &RunResult::default()).unwrap_err();
        assert!(missing.contains("reading"), "{missing}");
        std::fs::write(&path, DOC)?;
        let clean = RunResult {
            files: vec![
                file("a", vec![case(0, false), case(1, false), case(3, false)]),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        let ok = KnownFailures::check_file(&path, &clean)?;
        assert!(ok.contains("matches the 4 known failures"), "{ok}");
        let drift = RunResult {
            files: vec![file("b", vec![case(2, false)])],
            ..RunResult::default()
        };
        let err = KnownFailures::check_file(&path, &drift).unwrap_err();
        assert!(err.contains("listed file `a` did not run"), "{err}");
        std::fs::write(&path, DOC.replace(r#","b":{"text":[2]}"#, ""))?;
        let only_a = RunResult {
            files: vec![file(
                "a",
                vec![case(0, false), case(1, false), case(3, false)],
            )],
            ..RunResult::default()
        };
        let err = KnownFailures::check_file(&path, &only_a).unwrap_err();
        assert!(err.contains("categories without cases"), "{err}");

        // Update: `a#5` matches no rule, so nothing is written; then it passes and `a#1`
        // passes too, and the file shrinks.
        let set = GoldenSet::default();
        std::fs::write(&path, DOC)?;
        let mut odd = case(5, false);
        odd.detail = "error X".into();
        let with_odd = RunResult {
            files: vec![
                file(
                    "a",
                    vec![case(0, false), case(1, false), case(3, false), odd],
                ),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        let err = KnownFailures::update_file(&path, &with_odd, &set).unwrap_err();
        assert!(err.contains("a#5: error X"), "{err}");
        assert_eq!(std::fs::read_to_string(&path)?, DOC);
        let fewer = RunResult {
            files: vec![
                file("a", vec![case(0, false), case(1, true), case(3, false)]),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        let msg = KnownFailures::update_file(&path, &fewer, &set)?;
        assert!(msg.contains("0 added, 1 removed, 3 listed"), "{msg}");
        assert!(KnownFailures::check_file(&path, &fewer).is_ok());
        Ok(())
    }
}
