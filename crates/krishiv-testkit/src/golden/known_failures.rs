// SPDX-License-Identifier: Apache-2.0
//! The frozen statement-level baseline of the session engine's golden run.
//! It is split into one file per key so that two PRs never touch the same
//! file unless they touch the same key:
//!
//! - `tests/spark-golden/known-failures.json` holds only the format version,
//!   the Spark tag and the freeze date;
//! - `tests/spark-golden/known-failures-categories/` holds one file per
//!   category, with the classification rules that assign it;
//! - `tests/spark-golden/known-failures-cases/` holds one file per golden
//!   input file with that file's failing statement indices by category.
//!
//! Totals are computed from the run, never stored.
//!
//! ```json
//! {"format_version": 1, "spark_tag": "v4.2.0", "frozen_at": "2026-10-22"}
//! ```
//! ```json
//! {"format_version": 1, "category": "ddl", "kind": "unsupported", "task": "P2-T075a",
//!  "summary": "DDL statements", "rules": [{"order": 10, "command": true}]}
//! ```
//! ```json
//! {"format_version": 1, "file": "group-by", "cases": {"ddl": [0, 1], "wrong-sum": [7]}}
//! ```
//! A category's kind is `unsupported` (with `task`), `divergence` (with `adr`) or
//! `bug` (with `owner`).
//!
//! Rules are tried in ascending `order` across all category files, and the first that
//! matches a failure assigns its category; rules with equal `order` are tried by
//! category key, then by position in the file. Leave gaps between orders so that a
//! new rule can go anywhere without renumbering the others.
//!
//! A category is one of three kinds, each with a mandatory reference: an unsupported feature
//! names the plan task that adds it (`P2-T075a`), a known divergence names the ADR that records
//! it (ADR 0023, the Spark-divergence register), a bug names its owner in the internal status
//! list. Indices are statement positions in the golden file (0-based). The ratchet
//! ([`KnownFailures::check`]) fails on a statement that fails but is not listed (a regression)
//! and on a listed statement that passes (the list must shrink with it).
//!
//! [`KnownFailures::parse`] reads the legacy single-file form (header,
//! categories, rules and cases in one document, as the unit tests use);
//! [`KnownFailures::load`] reads the split on-disk form, where the
//! categories and cases directories next to the header file supply them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::GoldenSet;
use super::classify::{Failure, FileNames, Rule, classify, is_command};
use super::parse::Statement;
use super::run::{RunResult, load_statements};

/// Format version this loader reads.
pub const KNOWN_FAILURES_FORMAT_VERSION: u64 = 1;

/// Name of the per-file cases directory next to the header file.
pub const CASES_DIR_NAME: &str = "known-failures-cases";

/// Name of the per-category directory next to the header file.
pub const CATEGORIES_DIR_NAME: &str = "known-failures-categories";

/// The ADR a known divergence must cite.
pub const DIVERGENCE_ADR: &str = "0023";

impl CategoryKind {
    /// The kind's name in the baseline file.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "unsupported",
            Self::Divergence { .. } => "divergence",
            Self::Bug { .. } => "bug",
        }
    }
}

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
    /// With `reclassify`: listed statements whose category changed, with the old and the new one.
    pub moved: Vec<(Drift, String, String)>,
    /// The moves that also change the kind of category (`unsupported`, `divergence`, `bug`), as
    /// `(old category, new category)`: a bug re-homed as a divergence deserves a look.
    pub crossed_kinds: BTreeSet<(String, String)>,
}

impl Update {
    /// The moved statements counted per `old -> new` category pair, largest first, one per line.
    #[must_use]
    pub fn moved_summary(&self) -> String {
        let mut pairs: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for (_, old, new) in &self.moved {
            *pairs.entry((old.as_str(), new.as_str())).or_default() += 1;
        }
        let mut rows: Vec<_> = pairs.into_iter().collect();
        rows.sort_by_key(|((old, new), n)| (std::cmp::Reverse(*n), *old, *new));
        rows.into_iter()
            .map(|((old, new), n)| format!("  {n:>5}  {old} -> {new}\n"))
            .collect()
    }
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
                 with a category under tests/spark-golden/known-failures-cases/):\n",
                self.newly_failing.len()
            ));
            for d in &self.newly_failing {
                s.push_str(&format!("  {}#{}: {}\n", d.file, d.index, d.detail));
            }
        }
        if !self.newly_passing.is_empty() {
            s.push_str(&format!(
                "{} listed statement(s) now pass; remove them from \
                 tests/spark-golden/known-failures-cases/:\n",
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

/// One golden file's cases (`category -> statement indices`), validated
/// against the categories: every category known, every index a `u32`, no
/// index listed twice, at least one statement.
fn file_cases(
    file: &str,
    by_category: &Map<String, Value>,
    categories: &BTreeMap<String, Category>,
) -> Result<BTreeMap<u32, String>, String> {
    let mut indices = BTreeMap::new();
    for (key, list) in by_category {
        if !categories.contains_key(key) {
            return Err(format!("`{file}`: unknown category `{key}`"));
        }
        for i in list
            .as_array()
            .ok_or_else(|| format!("`{file}`/`{key}`: expected an array"))?
        {
            let i = i
                .as_u64()
                .and_then(|i| u32::try_from(i).ok())
                .ok_or_else(|| format!("`{file}`/`{key}`: bad index {i}"))?;
            if indices.insert(i, key.clone()).is_some() {
                return Err(format!("`{file}`#{i} is listed twice"));
            }
        }
    }
    if indices.is_empty() {
        return Err(format!("`{file}` lists no statements"));
    }
    Ok(indices)
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

fn invalid(message: String) -> KnownFailuresError {
    KnownFailuresError::Invalid(message)
}

fn text_field(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

fn check_format_version(v: &Value) -> bool {
    v.get("format_version").and_then(Value::as_u64) == Some(KNOWN_FAILURES_FORMAT_VERSION)
}

/// The Spark tag and freeze date of a header document.
fn parse_meta(root: &Value) -> Result<(String, String), KnownFailuresError> {
    if !check_format_version(root) {
        return Err(invalid("unsupported or missing `format_version`".into()));
    }
    let spark_tag =
        text_field(root, "spark_tag").ok_or_else(|| invalid("missing `spark_tag`".into()))?;
    let frozen_at =
        text_field(root, "frozen_at").ok_or_else(|| invalid("missing `frozen_at`".into()))?;
    Ok((spark_tag, frozen_at))
}

/// One category from its JSON object (`kind`, the reference it needs, `summary`).
fn parse_category(key: &str, c: &Value) -> Result<Category, KnownFailuresError> {
    let summary = text_field(c, "summary")
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid(format!("category `{key}` without a `summary`")))?;
    let reference = |k: &str| {
        text_field(c, k)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| invalid(format!("category `{key}` without `{k}`")))
    };
    let kind = match c.get("kind").and_then(Value::as_str) {
        Some("unsupported") => {
            let task = reference("task")?;
            if !is_task_id(&task) {
                return Err(invalid(format!(
                    "category `{key}`: `{task}` is not a task ID"
                )));
            }
            CategoryKind::Unsupported { task }
        }
        Some("divergence") => {
            let adr = reference("adr")?;
            if adr != DIVERGENCE_ADR {
                return Err(invalid(format!(
                    "category `{key}`: a divergence cites ADR {DIVERGENCE_ADR}, not `{adr}`"
                )));
            }
            CategoryKind::Divergence { adr }
        }
        Some("bug") => CategoryKind::Bug {
            owner: reference("owner")?,
        },
        other => {
            return Err(invalid(format!("category `{key}`: unknown kind {other:?}")));
        }
    };
    Ok(Category { kind, summary })
}

/// A category as the JSON object that holds it (`kind`, its reference, `summary`).
fn category_json(c: &Category) -> Map<String, Value> {
    let value = match &c.kind {
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
    match value {
        Value::Object(o) => o,
        _ => Map::new(),
    }
}

/// The key of a per-key file: its path relative to the directory without `.json`.
fn file_key(rel: &Path) -> Result<String, KnownFailuresError> {
    rel.to_str()
        .and_then(|s| s.strip_suffix(".json"))
        .map(|s| s.replace('\\', "/"))
        .ok_or_else(|| invalid(format!("{}: not a JSON file name", rel.display())))
}

/// Every category file of `dir` as the categories and the rules of all of them in
/// matching order (see the module docs).
fn load_categories(
    dir: &Path,
) -> Result<(BTreeMap<String, Category>, Vec<Rule>), KnownFailuresError> {
    let files =
        crate::manifest_dir::read_json_dir(dir).map_err(|source| KnownFailuresError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
    let mut categories = BTreeMap::new();
    let mut ordered: Vec<(u32, String, usize, Rule)> = Vec::new();
    for (rel, text) in &files {
        let name = rel.display();
        let stem = file_key(rel)?;
        let value: Value =
            serde_json::from_str(text).map_err(|e| invalid(format!("{name}: {e}")))?;
        if !check_format_version(&value) {
            return Err(invalid(format!(
                "{name}: unsupported or missing `format_version`"
            )));
        }
        let key = text_field(&value, "category")
            .ok_or_else(|| invalid(format!("{name}: missing `category`")))?;
        if key != stem {
            return Err(invalid(format!(
                "{name}: names category `{key}`, not its file"
            )));
        }
        let category = parse_category(&key, &value)?;
        let rules = match value.get("rules") {
            None => &[][..],
            Some(v) => v
                .as_array()
                .ok_or_else(|| invalid(format!("{name}: `rules` is not an array")))?,
        };
        for (i, rule) in rules.iter().enumerate() {
            let mut object = rule
                .as_object()
                .cloned()
                .ok_or_else(|| invalid(format!("{name}: rule {i} is not an object")))?;
            let order = object
                .remove("order")
                .and_then(|o| o.as_u64())
                .and_then(|o| u32::try_from(o).ok())
                .ok_or_else(|| invalid(format!("{name}: rule {i} without a numeric `order`")))?;
            object.insert("category".to_owned(), Value::String(key.clone()));
            let mut rule = Rule::parse(&Value::Object(object))
                .map_err(|e| invalid(format!("{name}: rule {i}: {e}")))?;
            rule.order = order;
            ordered.push((order, key.clone(), i, rule));
        }
        categories.insert(key, category);
    }
    ordered.sort_by(|a, b| (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2)));
    Ok((categories, ordered.into_iter().map(|o| o.3).collect()))
}

impl KnownFailures {
    /// Default path of the header file in the repo.
    #[must_use]
    pub fn default_path() -> PathBuf {
        crate::tests_dir()
            .join("spark-golden")
            .join("known-failures.json")
    }

    /// Default path of the per-file cases directory in the repo.
    #[must_use]
    pub fn default_cases_dir() -> PathBuf {
        crate::tests_dir().join("spark-golden").join(CASES_DIR_NAME)
    }

    /// Default path of the per-category directory in the repo.
    #[must_use]
    pub fn default_categories_dir() -> PathBuf {
        crate::tests_dir()
            .join("spark-golden")
            .join(CATEGORIES_DIR_NAME)
    }

    /// Load the split baseline: the header file at `path`, the categories
    /// and rules in the [`CATEGORIES_DIR_NAME`] directory next to it, and the
    /// per-file cases in the [`CASES_DIR_NAME`] directory next to it. Each
    /// directory, when it exists, supplies everything of its kind (the
    /// header's own `categories`, `rules` and `cases`, if any, are ignored);
    /// otherwise the header's embedded ones apply (the legacy single-file
    /// form the unit tests use).
    pub fn load(path: &Path) -> Result<Self, KnownFailuresError> {
        let text = std::fs::read_to_string(path).map_err(|source| KnownFailuresError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let categories_dir = path
            .parent()
            .map(|p| p.join(CATEGORIES_DIR_NAME))
            .filter(|d| d.is_dir());
        let mut known = if let Some(dir) = &categories_dir {
            let root: Value = serde_json::from_str(&text).map_err(|e| invalid(e.to_string()))?;
            let (spark_tag, frozen_at) = parse_meta(&root)?;
            let (categories, rules) = load_categories(dir)?;
            Self {
                spark_tag,
                frozen_at,
                categories,
                rules,
                cases: BTreeMap::new(),
            }
        } else {
            Self::parse(&text)?
        };
        let Some(dir) = path.parent().map(|p| p.join(CASES_DIR_NAME)) else {
            return Ok(known);
        };
        if !dir.is_dir() {
            return Ok(known);
        }
        // The split on-disk form: the cases directory supplies every file's
        // cases, replacing the header's own (unit-test) ones.
        let mut cases = BTreeMap::new();
        let bad = KnownFailuresError::Invalid;
        let files =
            crate::manifest_dir::read_json_dir(&dir).map_err(|source| KnownFailuresError::Io {
                path: dir.clone(),
                source,
            })?;
        for (rel, text) in &files {
            let stem = rel
                .to_str()
                .and_then(|s| s.strip_suffix(".json"))
                .map(|s| s.replace('\\', "/"))
                .ok_or_else(|| bad(format!("{}: not a case file", rel.display())))?;
            let value: Value =
                serde_json::from_str(text).map_err(|e| bad(format!("{}: {e}", rel.display())))?;
            if value.get("format_version").and_then(Value::as_u64)
                != Some(KNOWN_FAILURES_FORMAT_VERSION)
            {
                return Err(bad(format!(
                    "{}: unsupported or missing `format_version`",
                    rel.display()
                )));
            }
            let file = value
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| bad(format!("{}: missing `file`", rel.display())))?;
            if file != stem {
                return Err(bad(format!(
                    "{}: names `{file}`, not its golden file",
                    rel.display()
                )));
            }
            let indices = file_cases(
                file,
                value
                    .get("cases")
                    .and_then(Value::as_object)
                    .ok_or_else(|| bad(format!("`{file}`: missing `cases` object")))?,
                &known.categories,
            )
            .map_err(bad)?;
            if cases.insert(file.to_owned(), indices).is_some() {
                return Err(bad(format!("`{file}` is listed twice")));
            }
        }
        known.cases = cases;
        Ok(known)
    }

    /// Parse JSON text (the legacy single-file form), validating every
    /// category and case. A document without `cases` parses to a baseline
    /// without cases.
    pub fn parse(text: &str) -> Result<Self, KnownFailuresError> {
        let bad = invalid;
        let root: Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        let (spark_tag, frozen_at) = parse_meta(&root)?;
        let mut categories = BTreeMap::new();
        for (key, c) in root
            .get("categories")
            .and_then(Value::as_object)
            .ok_or_else(|| bad("missing `categories` object".into()))?
        {
            categories.insert(key.clone(), parse_category(key, c)?);
        }
        let mut rules = Vec::new();
        for (i, r) in root
            .get("rules")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("missing `rules` array".into()))?
            .iter()
            .enumerate()
        {
            let mut rule = Rule::parse(r).map_err(bad)?;
            if !categories.contains_key(&rule.category) {
                return Err(bad(format!(
                    "a rule names the unknown category `{}`",
                    rule.category
                )));
            }
            // The document order is the match order; gaps leave room to insert.
            rule.order = u32::try_from((i + 1) * 10).map_err(|e| bad(e.to_string()))?;
            rules.push(rule);
        }
        let mut cases = BTreeMap::new();
        // A header file without `cases` (the split on-disk form) parses to
        // an empty baseline; `load` fills the cases from the directory.
        if let Some(object) = root.get("cases") {
            for (file, by_category) in object
                .as_object()
                .ok_or_else(|| bad("`cases` is not an object".into()))?
            {
                let indices = file_cases(
                    file,
                    by_category.as_object().ok_or_else(|| {
                        bad(format!("`{file}`: expected an object of categories"))
                    })?,
                    &categories,
                )
                .map_err(bad)?;
                cases.insert(file.clone(), indices);
            }
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
        let mut header = self.header_json();
        if let Some(object) = header.as_object_mut() {
            let cases: Map<String, Value> = self
                .cases
                .iter()
                .map(|(file, indices)| (file.clone(), Self::indices_json(indices)))
                .collect();
            object.insert("cases".to_owned(), Value::Object(cases));
        }
        header
    }

    /// The legacy single-file header JSON: everything but the per-file
    /// cases, in one document.
    #[must_use]
    pub fn header_json(&self) -> Value {
        let categories: Map<String, Value> = self
            .categories
            .iter()
            .map(|(k, c)| (k.clone(), Value::Object(category_json(c))))
            .collect();
        json!({
            "format_version": KNOWN_FAILURES_FORMAT_VERSION,
            "spark_tag": self.spark_tag,
            "frozen_at": self.frozen_at,
            "categories": categories,
            "rules": self.rules.iter().map(Rule::to_json).collect::<Vec<_>>(),
        })
    }

    /// The on-disk header file text (`known-failures.json`): the format
    /// version, the Spark tag and the freeze date; pretty JSON with a
    /// trailing newline.
    #[must_use]
    pub fn meta_text(&self) -> String {
        let doc = json!({
            "format_version": KNOWN_FAILURES_FORMAT_VERSION,
            "spark_tag": self.spark_tag,
            "frozen_at": self.frozen_at,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n"
    }

    /// One category's file (`known-failures-categories/<key>.json`): the
    /// category and the rules that assign it, each with its `order`; `None`
    /// for a key that is not a category.
    #[must_use]
    pub fn category_file_json(&self, key: &str) -> Option<Value> {
        let category = self.categories.get(key)?;
        let rules: Vec<Value> = self
            .rules
            .iter()
            .filter(|r| r.category == key)
            .map(|r| {
                let mut rule = r.to_json().clone();
                if let Some(o) = rule.as_object_mut() {
                    o.remove("category");
                    o.insert("order".to_owned(), json!(r.order));
                }
                rule
            })
            .collect();
        let mut doc = category_json(category);
        doc.insert(
            "format_version".to_owned(),
            json!(KNOWN_FAILURES_FORMAT_VERSION),
        );
        doc.insert("category".to_owned(), json!(key));
        doc.insert("rules".to_owned(), Value::Array(rules));
        Some(Value::Object(doc))
    }

    /// The category file text: pretty JSON with a trailing newline, or
    /// `None` for a key that is not a category.
    #[must_use]
    pub fn category_file_text(&self, key: &str) -> Option<String> {
        self.category_file_json(key)
            .map(|doc| serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n")
    }

    /// One golden file's indices as `category -> sorted indices`, the value
    /// the per-file case file holds under `cases`.
    fn indices_json(indices: &BTreeMap<u32, String>) -> Value {
        let mut by: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
        for (i, k) in indices {
            by.entry(k).or_default().push(*i);
        }
        for list in by.values_mut() {
            list.sort_unstable();
        }
        json!(by)
    }

    /// One golden file's case file (`known-failures-cases/<file>.json`):
    /// the file's failing indices by category, or `None` when the file
    /// lists nothing.
    #[must_use]
    pub fn case_file_json(&self, file: &str) -> Option<Value> {
        self.cases.get(file).map(|indices| {
            json!({
                "format_version": KNOWN_FAILURES_FORMAT_VERSION,
                "file": file,
                "cases": Self::indices_json(indices),
            })
        })
    }

    /// The case file text: pretty JSON with a trailing newline, or `None`
    /// when the file lists nothing.
    #[must_use]
    pub fn case_file_text(&self, file: &str) -> Option<String> {
        self.case_file_json(file)
            .map(|doc| serde_json::to_string_pretty(&doc).unwrap_or_default() + "\n")
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
    pub fn update_with_set(
        &mut self,
        run: &RunResult,
        set: &GoldenSet,
        reclassify: Option<&str>,
    ) -> Result<Update, String> {
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
        Ok(self.update(run, &statements, &families, reclassify))
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

    /// [`Self::update_with_set`] on the file at `path`, rewriting the header
    /// and syncing the per-category and per-file cases directories next to
    /// it (only changed files are rewritten; category and case files with
    /// nothing left are deleted); `Err` (and nothing written) when a failure
    /// matches no rule.
    pub fn update_file(
        path: &Path,
        run: &RunResult,
        set: &GoldenSet,
        reclassify: Option<&str>,
    ) -> Result<String, String> {
        let mut known = Self::load(path).map_err(|e| e.to_string())?;
        if let Some(category) = reclassify
            && !known.categories.contains_key(category)
        {
            return Err(format!(
                "--reclassify: unknown category `{category}` (known: {})",
                known
                    .categories
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let up = known.update_with_set(run, set, reclassify)?;
        if !up.unclassified.is_empty() {
            let mut msg = format!(
                "{} failure(s) match no rule; add a rule or category under {} (nothing written):\n",
                up.unclassified.len(),
                path.with_file_name(CATEGORIES_DIR_NAME).display()
            );
            for d in &up.unclassified {
                msg.push_str(&format!("  {}#{}: {}\n", d.file, d.index, d.detail));
            }
            return Err(msg);
        }
        crate::manifest_dir::write_if_changed(path, &known.meta_text())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let parent = path
            .parent()
            .ok_or_else(|| format!("{}: no parent directory", path.display()))?;
        let mut categories = BTreeMap::new();
        for key in known.categories.keys() {
            let text = known
                .category_file_text(key)
                .ok_or_else(|| format!("category `{key}` vanished while writing"))?;
            categories.insert(PathBuf::from(format!("{key}.json")), text);
        }
        crate::manifest_dir::sync_json_dir(&parent.join(CATEGORIES_DIR_NAME), &categories)
            .map_err(|e| e.to_string())?;
        let dir = parent.join(CASES_DIR_NAME);
        let mut wanted = BTreeMap::new();
        for file in known.cases.keys() {
            let text = known
                .case_file_text(file)
                .ok_or_else(|| format!("`{file}` vanished while writing"))?;
            wanted.insert(PathBuf::from(format!("{file}.json")), text);
        }
        crate::manifest_dir::sync_json_dir(&dir, &wanted).map_err(|e| e.to_string())?;
        let mut msg = format!(
            "ratchet: {} added, {} removed, {} listed in {}",
            up.added.len(),
            up.removed.len(),
            known.listed(),
            path.display()
        );
        if let Some(category) = reclassify {
            msg.push_str(&format!(
                "\nreclassified {} statement(s) of `{category}`:\n{}",
                up.moved.len(),
                up.moved_summary()
            ));
            for (old, new) in &up.crossed_kinds {
                msg.push_str(&format!(
                    "NOTE: moves from `{old}` to `{new}` change the kind of category; check them\n"
                ));
            }
        }
        Ok(msg)
    }

    /// Bring the baseline in line with `run`: drop the listed statements that pass (or are
    /// gone), keep the categories of the ones still failing, and classify new failures with
    /// [`Self::rules`]. `statements` holds each run file's statements (for the rules);
    /// `families` maps function names to their Spark function group. Categories left without
    /// cases are dropped with their rules. Nothing changes while a failure is unclassified.
    ///
    /// With `reclassify = Some(category)` the still-failing statements listed under that category
    /// are classified again by the current rules instead of keeping it. Categories are sticky by
    /// design (a fix elsewhere must not silently re-home a statement), so this is an explicit run
    /// (`golden-report --ratchet update --reclassify CATEGORY`), never the default, and it touches
    /// one category only.
    pub fn update(
        &mut self,
        run: &RunResult,
        statements: &BTreeMap<String, Vec<Statement>>,
        families: &BTreeMap<String, String>,
        reclassify: Option<&str>,
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
                if let Some(k) = old
                    && reclassify != Some(k.as_str())
                {
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
                match (classify(&self.rules, &failure, families), old) {
                    (Some(k), Some(was)) => {
                        if k != was {
                            let kind = |c: &str| self.categories.get(c).map(|c| c.kind.label());
                            if kind(k) != kind(was) {
                                up.crossed_kinds.insert((was.clone(), k.to_owned()));
                            }
                            up.moved.push((drift(), was.clone(), k.to_owned()));
                        }
                        kept.insert(c.index, k.to_owned());
                    }
                    (Some(k), None) => {
                        up.added.push((drift(), k.to_owned()));
                        kept.insert(c.index, k.to_owned());
                    }
                    (None, _) => up.unclassified.push(drift()),
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
        let up = k.update(&run, &statements, &BTreeMap::new(), None);
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
    fn reclassify_rehomes_the_listed_statements_by_the_current_rules() -> R {
        let mut k = KnownFailures::parse(DOC)?;
        // a#0 is listed as `ddl` but is a plain query whose failure the `sum` rule now names.
        let run = RunResult {
            files: vec![file(
                "a",
                vec![case(0, false), case(1, false), case(3, false)],
            )],
            ..RunResult::default()
        };
        let statements = BTreeMap::from([(
            "a".to_owned(),
            vec![
                stmt("SELECT sum(x) FROM t"),
                stmt("CREATE VIEW v AS SELECT 1"),
                stmt("SELECT 2"),
                stmt("SELECT 3"),
            ],
        )]);
        // Without `reclassify` the categories are kept.
        let mut sticky = k.clone();
        let up = sticky.update(&run, &statements, &BTreeMap::new(), None);
        assert!(up.moved.is_empty());
        assert_eq!(sticky.cases["a"][&0], "ddl");
        // With it, a#0 moves to `sum` (rule order: `ddl` needs a command), a#1 stays `ddl`, a#3 stays `sum`.
        let up = k.update(&run, &statements, &BTreeMap::new(), Some("ddl"));
        assert_eq!(k.cases["a"][&0], "sum");
        assert_eq!(k.cases["a"][&1], "ddl");
        assert_eq!(up.moved.len(), 1);
        assert_eq!(up.moved[0].1, "ddl");
        assert_eq!(up.moved[0].2, "sum");
        assert!(up.added.is_empty());
        assert_eq!(up.moved_summary(), "      1  ddl -> sum\n");
        // `ddl` is unsupported and `sum` a bug: the move crosses kinds and is flagged.
        assert_eq!(
            up.crossed_kinds.iter().collect::<Vec<_>>(),
            [&("ddl".to_owned(), "sum".to_owned())]
        );
        let fresh = KnownFailures::parse(DOC)?;
        assert_eq!(fresh.categories["ddl"].kind.label(), "unsupported");
        assert_eq!(fresh.categories["sum"].kind.label(), "bug");
        assert_eq!(fresh.categories["text"].kind.label(), "divergence");
        // A statement no rule classifies blocks the rewrite, as for a new failure.
        let mut blocked = KnownFailures::parse(DOC)?;
        let run2 = RunResult {
            files: vec![FileResult {
                cases: vec![Case {
                    index: 0,
                    result: Some(false),
                    detail: "something else".into(),
                    ..Case::default()
                }],
                ..file("a", vec![])
            }],
            ..RunResult::default()
        };
        let up = blocked.update(&run2, &statements, &BTreeMap::new(), Some("ddl"));
        assert_eq!(up.unclassified.len(), 1);
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
        let up = k.update(&run, &statements, &BTreeMap::new(), None);
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
        let up = k.update_with_set(&run, &set, None)?;
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
        let err = KnownFailures::update_file(&path, &with_odd, &set, None).unwrap_err();
        assert!(err.contains("a#5: error X"), "{err}");
        assert_eq!(std::fs::read_to_string(&path)?, DOC);
        let fewer = RunResult {
            files: vec![
                file("a", vec![case(0, false), case(1, true), case(3, false)]),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        let msg = KnownFailures::update_file(&path, &fewer, &set, None)?;
        assert!(msg.contains("0 added, 1 removed, 3 listed"), "{msg}");
        assert!(KnownFailures::check_file(&path, &fewer).is_ok());
        Ok(())
    }

    /// Writes the split on-disk form of `k` into `dir`: `known.json`, one file per category and
    /// one per golden file.
    fn write_split(dir: &Path, k: &KnownFailures) -> R {
        std::fs::write(dir.join("known.json"), k.meta_text())?;
        let categories = dir.join(super::CATEGORIES_DIR_NAME);
        std::fs::create_dir_all(&categories)?;
        for key in k.categories.keys() {
            let text = k.category_file_text(key).ok_or("missing category file")?;
            std::fs::write(categories.join(format!("{key}.json")), text)?;
        }
        let cases = dir.join(super::CASES_DIR_NAME);
        for file in k.cases.keys() {
            let text = k.case_file_text(file).ok_or("missing case file")?;
            std::fs::create_dir_all(cases.join(file).parent().ok_or("no parent")?)?;
            std::fs::write(cases.join(format!("{file}.json")), text)?;
        }
        Ok(())
    }

    #[test]
    fn header_category_and_case_files_round_trip_through_the_split_form() -> R {
        let k = KnownFailures::parse(DOC)?;
        // The header holds the version, tag and date only.
        let meta = k.meta_text();
        assert!(
            ["categories", "rules", "cases"]
                .iter()
                .all(|w| !meta.contains(w)),
            "{meta}"
        );
        assert!(
            KnownFailures::parse(&meta).is_err(),
            "the header alone is not a legacy document"
        );
        // Header plus one category file per category and one case file per listed file loads
        // back to the same baseline.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("known.json");
        write_split(dir.path(), &k)?;
        let cases = dir.path().join(super::CASES_DIR_NAME);
        // A nested golden name lands in a subdirectory, like the real baseline.
        std::fs::create_dir_all(cases.join("nonansi"))?;
        std::fs::write(
            cases.join("nonansi/array.json"),
            "{\"format_version\":1,\"file\":\"nonansi/array\",\"cases\":{\"zzz\":[0]}}",
        )?;
        let err = KnownFailures::load(&path).unwrap_err();
        assert!(err.to_string().contains("unknown category"), "{err}");
        std::fs::remove_file(cases.join("nonansi/array.json"))?;
        std::fs::remove_dir(cases.join("nonansi"))?;
        assert_eq!(KnownFailures::load(&path)?, k);
        assert_eq!(KnownFailures::load(&path)?.listed(), 4);
        Ok(())
    }

    #[test]
    fn case_files_must_name_their_file_and_version() -> R {
        let k = KnownFailures::parse(DOC)?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("known.json");
        write_split(dir.path(), &k)?;
        let cases = dir.path().join(super::CASES_DIR_NAME);
        std::fs::create_dir_all(&cases)?;
        let write_case = |name: &str, text: &str| -> R {
            std::fs::write(cases.join(name), text)?;
            Ok(())
        };
        write_case(
            "a.json",
            &k.case_file_text("a")
                .ok_or("missing")?
                .replace("\"a\"", "\"b\""),
        )?;
        let err = KnownFailures::load(&path).unwrap_err();
        assert!(err.to_string().contains("not its golden file"), "{err}");
        write_case("a.json", &k.case_file_text("a").ok_or("missing")?)?;
        assert!(KnownFailures::load(&path).is_ok());
        write_case(
            "a.json",
            "{\"format_version\":2,\"file\":\"a\",\"cases\":{}}",
        )?;
        let err = KnownFailures::load(&path).unwrap_err();
        assert!(err.to_string().contains("format_version"), "{err}");
        write_case(
            "a.json",
            "{\"format_version\":1,\"file\":\"a\",\"cases\":{\"ddl\":[]}}",
        )?;
        let err = KnownFailures::load(&path).unwrap_err();
        assert!(err.to_string().contains("no statements"), "{err}");
        Ok(())
    }

    fn category_dir(
        rules: &[(&str, &str)],
    ) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("known.json"),
            r#"{"format_version":1,"spark_tag":"v4.2.0","frozen_at":"2026-10-22"}"#,
        )?;
        let categories = dir.path().join(super::CATEGORIES_DIR_NAME);
        std::fs::create_dir_all(&categories)?;
        for (name, text) in rules {
            std::fs::write(categories.join(name), text)?;
        }
        Ok(dir)
    }

    #[test]
    fn rules_match_in_global_order_across_category_files() -> R {
        let file = |key: &str, orders: &[u32]| {
            let rules: Vec<String> = orders
                .iter()
                .map(|o| format!(r#"{{"order":{o},"detail":"x{o}"}}"#))
                .collect();
            (
                format!("{key}.json"),
                format!(
                    r#"{{"format_version":1,"category":"{key}","kind":"bug","owner":"o",
                        "summary":"s","rules":[{}]}}"#,
                    rules.join(",")
                ),
            )
        };
        let (a, b) = (file("a", &[30, 5]), file("b", &[10, 30]));
        let dir = category_dir(&[(&a.0, &a.1), (&b.0, &b.1)])?;
        let k = KnownFailures::load(&dir.path().join("known.json"))?;
        // Ascending `order`; equal orders go by category key, then position.
        let got: Vec<(&str, u32)> = k
            .rules
            .iter()
            .map(|r| (r.category.as_str(), r.order))
            .collect();
        assert_eq!(got, [("a", 5), ("b", 10), ("a", 30), ("b", 30)]);
        assert!(k.cases.is_empty());
        Ok(())
    }

    #[test]
    fn category_files_must_name_their_category_and_number_their_rules() -> R {
        let ok = r#"{"format_version":1,"category":"a","kind":"bug","owner":"o","summary":"s",
                     "rules":[{"order":1,"command":true}]}"#;
        let cases = [
            (
                ok.replace(r#""category":"a""#, r#""category":"b""#),
                "not its file",
            ),
            (
                ok.replace(r#""format_version":1"#, r#""format_version":2"#),
                "format_version",
            ),
            (ok.replace(r#""order":1,"#, ""), "without a numeric `order`"),
            (
                ok.replace(r#""order":1"#, r#""order":-1"#),
                "without a numeric `order`",
            ),
            (
                ok.replace(r#"{"order":1,"command":true}"#, r#"{"order":1}"#),
                "no condition",
            ),
            (
                ok.replace(r#""kind":"bug""#, r#""kind":"later""#),
                "unknown kind",
            ),
            (ok.replace(r#""owner":"o","#, ""), "without `owner`"),
            (
                ok.replace(r#""rules":[{"order":1,"command":true}]"#, r#""rules":{}"#),
                "not an array",
            ),
        ];
        for (text, msg) in cases {
            let dir = category_dir(&[("a.json", &text)])?;
            let err = KnownFailures::load(&dir.path().join("known.json")).unwrap_err();
            assert!(err.to_string().contains(msg), "{msg}: {err}");
        }
        let dir = category_dir(&[("a.json", ok)])?;
        assert_eq!(
            KnownFailures::load(&dir.path().join("known.json"))?
                .rules
                .len(),
            1
        );
        Ok(())
    }

    /// The checked-in baseline is exactly what update mode writes, so a hand-edited or stale
    /// per-key file fails here without running the whole golden suite.
    #[test]
    fn the_checked_in_baseline_is_in_canonical_form() -> R {
        let path = KnownFailures::default_path();
        let known = KnownFailures::load(&path)?;
        assert_eq!(std::fs::read_to_string(&path)?, known.meta_text());
        let on_disk =
            |dir: PathBuf, key: &str| std::fs::read_to_string(dir.join(format!("{key}.json")));
        for key in known.categories.keys() {
            let want = known.category_file_text(key).ok_or("missing category")?;
            let got = on_disk(KnownFailures::default_categories_dir(), key)?;
            assert_eq!(got, want, "category `{key}` is not in canonical form");
        }
        for file in known.cases.keys() {
            let want = known.case_file_text(file).ok_or("missing case file")?;
            let got = on_disk(KnownFailures::default_cases_dir(), file)?;
            assert_eq!(got, want, "case file `{file}` is not in canonical form");
        }
        Ok(())
    }

    #[test]
    fn update_file_writes_the_split_and_rewrites_nothing_twice() -> R {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("known.json");
        std::fs::write(&path, DOC)?;
        let set = GoldenSet::default();
        let fewer = RunResult {
            files: vec![
                file("a", vec![case(0, false), case(1, true), case(3, false)]),
                file("b", vec![case(2, false)]),
            ],
            ..RunResult::default()
        };
        let unknown = KnownFailures::update_file(&path, &fewer, &set, Some("nope")).unwrap_err();
        assert!(unknown.contains("unknown category `nope`"), "{unknown}");
        assert!(unknown.contains("ddl"), "{unknown}");
        KnownFailures::update_file(&path, &fewer, &set, None)?;
        // The header holds no cases, categories or rules; each category and each listed file
        // has its own file.
        let header: Value =
            serde_json::from_str(&std::fs::read_to_string(&path)?).map_err(|e| e.to_string())?;
        for key in ["cases", "categories", "rules"] {
            assert!(header.get(key).is_none(), "{header}");
        }
        let cases = dir.path().join(super::CASES_DIR_NAME);
        let categories = dir.path().join(super::CATEGORIES_DIR_NAME);
        assert!(cases.join("a.json").is_file());
        assert!(cases.join("b.json").is_file());
        for key in ["ddl", "sum", "text"] {
            assert!(categories.join(format!("{key}.json")).is_file(), "{key}");
        }
        let loaded = KnownFailures::load(&path)?;
        assert_eq!(loaded.listed(), 3);
        assert_eq!(loaded.rules.len(), 2);
        // A second update changes no bytes and touches no file.
        let before: BTreeMap<String, (Vec<u8>, std::time::SystemTime)> = ["known.json"]
            .into_iter()
            .map(|n| dir.path().join(n))
            .chain([cases.join("a.json"), cases.join("b.json")])
            .chain(["ddl", "sum", "text"].map(|k| categories.join(format!("{k}.json"))))
            .map(|p| {
                Ok((
                    p.to_string_lossy().into_owned(),
                    (std::fs::read(&p)?, std::fs::metadata(&p)?.modified()?),
                ))
            })
            .collect::<Result<_, std::io::Error>>()?;
        KnownFailures::update_file(&path, &fewer, &set, None)?;
        for (name, (bytes, mtime)) in &before {
            let p = Path::new(name);
            assert_eq!(&std::fs::read(p)?, bytes, "{name} was rewritten");
            assert_eq!(
                &std::fs::metadata(p)?.modified()?,
                mtime,
                "{name} was touched"
            );
        }
        // A category with no statements left loses its file, and only its file.
        let without_b = RunResult {
            files: vec![file(
                "a",
                vec![case(0, false), case(1, true), case(3, false)],
            )],
            ..RunResult::default()
        };
        KnownFailures::update_file(&path, &without_b, &set, None)?;
        assert!(!categories.join("text.json").exists());
        assert!(categories.join("ddl.json").is_file());
        assert!(!cases.join("b.json").exists());
        assert_eq!(KnownFailures::load(&path)?.categories.len(), 2);
        Ok(())
    }
}
