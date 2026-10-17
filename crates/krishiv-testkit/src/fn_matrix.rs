// SPDX-License-Identifier: Apache-2.0
//! Spark function-matrix fixtures and the coverage-matrix report (P2-T106a, ADR 0029).
//!
//! Fixtures live in `tests/fixtures/spark-<ver>/function-matrix/`: a `manifest.json` plus one JSON-lines
//! shard per function family (`cases/<family>.jsonl`). They are produced by executing a real Spark with
//! `tests/scripts/gen_function_matrix.py`; never edit them by hand (ADR 0008, review I12).
//!
//! Each fixture line is one mode-independent case (SQL + session conf); it carries one expectation per
//! ANSI mode, which [`FnMatrix`] expands into one [`MatrixCase`] per (case, mode). The coverage matrix
//! ([`CoverageReport`]) is the share of those cases an engine passes, per family and per mode.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::hash_fixture::FixtureError;

/// Fixture format version this loader reads.
pub const FORMAT_VERSION: u64 = 1;

/// Session conf key of the ANSI mode.
pub const ANSI_CONF: &str = "spark.sql.ansi.enabled";

/// ANSI mode a case runs under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mode {
    /// `spark.sql.ansi.enabled=true` (Spark 4.x default).
    AnsiOn,
    /// `spark.sql.ansi.enabled=false`.
    AnsiOff,
}

impl Mode {
    /// Both modes, in fixture order.
    pub const ALL: [Self; 2] = [Self::AnsiOn, Self::AnsiOff];

    /// Fixture key of the mode (`ansi_on` / `ansi_off`).
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::AnsiOn => "ansi_on",
            Self::AnsiOff => "ansi_off",
        }
    }

    /// Value of [`ANSI_CONF`] in this mode.
    #[must_use]
    pub const fn ansi_value(self) -> &'static str {
        match self {
            Self::AnsiOn => "true",
            Self::AnsiOff => "false",
        }
    }
}

/// One result column as Spark reported it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Column name (Spark's auto-generated alias for expressions).
    pub name: String,
    /// Spark type, `DataType.simpleString()`.
    pub data_type: String,
    /// Nullability.
    pub nullable: bool,
}

/// A Spark error: condition (error class), SQLSTATE, parameters and message text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparkError {
    /// Error condition, e.g. `DIVIDE_BY_ZERO`; `UNCLASSIFIED.<Exception>` when Spark gave none.
    pub error_class: String,
    /// SQLSTATE, when Spark reported one.
    pub sqlstate: Option<String>,
    /// Message text without the JVM stack trace or logical plan; expression ids normalised to `#N`.
    pub message: String,
    /// Message parameters.
    pub message_parameters: BTreeMap<String, String>,
    /// `parse`, `analysis` or `execution`, from the type of Spark's exception.
    pub phase: String,
    /// `false` for `_LEGACY_ERROR_TEMP_*`, unclassified or SQLSTATE-less errors: not stable API
    /// (ADR 0008), so compare only that the case errors, and its phase.
    pub stable: bool,
}

/// Expected outcome of one case in one mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expected {
    /// The query succeeded.
    Rows {
        /// Result schema.
        schema: Vec<Field>,
        /// Rows; each value is Spark's own rendering (see `render`), `None` for SQL NULL.
        rows: Vec<Vec<Option<String>>>,
        /// Per column: `cast` (`CAST(v AS STRING)`), `hex` (binary as `hex(v)`), `wkb_hex`
        /// (`hex(st_asbinary(v))`) or `utf8_hex` (hex of the string bytes; used when not valid UTF-8).
        render: Vec<String>,
        /// Rows are in Spark's order (top-level ORDER BY); otherwise sorted, compare as a multiset.
        ordered: bool,
        /// More than the generator's `max_rows` rows; only the first ones are recorded.
        truncated: bool,
        /// Typed cells, row-aligned with `rows` (manifest `typed_cell_format`): exact float bits,
        /// decimal unscaled+scale, epoch micros, days, hex bytes, nested JSON. `None` when Arrow cannot
        /// carry a column type (see `typed_unavailable`).
        typed_rows: Option<Vec<Vec<Value>>>,
        /// Why `typed_rows` is absent (e.g. `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION`).
        typed_unavailable: Option<String>,
    },
    /// The query failed; `schema` is present when analysis succeeded and execution failed.
    Error {
        /// Result schema, if analysis succeeded.
        schema: Option<Vec<Field>>,
        /// The error.
        error: SparkError,
    },
    /// The result depends on time, randomness or environment: only the schema is checked.
    Nondeterministic {
        /// Result schema.
        schema: Vec<Field>,
    },
}

/// Matrix slice a cell counts towards in the coverage report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// A seed or edge case whose expectation is a value (or schema, if nondeterministic).
    Value,
    /// A seed or edge case whose expectation is an error.
    Error,
    /// An input-type probe (`:type:` kind), mostly `DATATYPE_MISMATCH` errors.
    TypeProbe,
}

impl MatrixCase {
    /// The report slice of this cell.
    #[must_use]
    pub fn category(&self) -> Category {
        if self.kind.contains(":type:") {
            Category::TypeProbe
        } else if self.expected.is_error() {
            Category::Error
        } else {
            Category::Value
        }
    }
}

impl Expected {
    /// Whether Spark raised an error.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }
}

/// One (case, mode) cell of the matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixCase {
    /// `<base_id>/<mode>`; stable across regenerations (content hash of function, SQL and conf).
    pub id: String,
    /// Mode-independent id, `<function>#<hash>`.
    pub base_id: String,
    /// Function under test (a `SHOW FUNCTIONS` name).
    pub function: String,
    /// Function family (Spark `ExpressionInfo` group without `_funcs`).
    pub family: String,
    /// How the case was derived: `example<n>`, `seed<n>`, or `<seed>:edge:<label>@arg<i>` /
    /// `<seed>:type:<label>@arg<i>` / `<seed>:edge:<label>@rows`.
    pub kind: String,
    /// ANSI mode.
    pub mode: Mode,
    /// SQL statement.
    pub sql: String,
    /// Full session conf for the run (includes [`ANSI_CONF`] and the session time zone).
    pub conf: BTreeMap<String, String>,
    /// Spark's outcome.
    pub expected: Expected,
}

/// A function listed in the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionEntry {
    /// `SHOW FUNCTIONS` name.
    pub name: String,
    /// Family.
    pub family: String,
    /// `function`, `table_function` or `operator`.
    pub registry: String,
    /// Mode-independent case count.
    pub cases: usize,
}

/// The loaded function matrix.
#[derive(Debug, Clone)]
pub struct FnMatrix {
    /// Spark version the fixtures were generated against.
    pub spark_version: String,
    /// Every name `SHOW FUNCTIONS` returned.
    pub show_functions: Vec<String>,
    /// Functions in the manifest, sorted by name.
    pub functions: Vec<FunctionEntry>,
    /// All (case, mode) cells, ordered by family, function, id, mode.
    pub cases: Vec<MatrixCase>,
}

impl FnMatrix {
    /// Default fixture directory (Spark 4.2.0).
    #[must_use]
    pub fn default_dir() -> PathBuf {
        crate::tests_dir().join("fixtures/spark-4.2.0/function-matrix")
    }

    /// Load the default fixtures.
    pub fn load_default() -> Result<Self, FixtureError> {
        Self::load(Self::default_dir())
    }

    /// Load `manifest.json` and every shard it lists from `dir`, validating counts.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, FixtureError> {
        let dir = dir.as_ref();
        let manifest = read(&dir.join("manifest.json"))?;
        let mut shards = BTreeMap::new();
        for shard in shard_paths(&manifest)? {
            let text = read(&dir.join(&shard))?;
            shards.insert(shard, text);
        }
        Self::parse(&manifest, &shards)
    }

    /// Parse a manifest and its shards (`shard path -> JSON-lines text`).
    pub fn parse(manifest: &str, shards: &BTreeMap<String, String>) -> Result<Self, FixtureError> {
        let root: Value = serde_json::from_str(manifest).map_err(|e| invalid(&e))?;
        let version = root.get("format_version").and_then(Value::as_u64);
        if version != Some(FORMAT_VERSION) {
            return Err(FixtureError::Invalid(format!(
                "unsupported format_version {version:?} (expected {FORMAT_VERSION})"
            )));
        }
        let spark_version = str_field(&root, "spark_version")?;
        let show_functions = array(&root, "show_functions")?
            .iter()
            .map(|v| as_str(v, "show_functions[]"))
            .collect::<Result<Vec<_>, _>>()?;
        let functions = array(&root, "functions")?
            .iter()
            .map(parse_function)
            .collect::<Result<Vec<_>, _>>()?;
        let base_conf = string_map(root.get("base_conf").unwrap_or(&Value::Null))?;
        let mut cases = Vec::new();
        for fam in array(&root, "families")? {
            let shard = str_field(fam, "shard")?;
            let family = str_field(fam, "family")?;
            let text = shards
                .get(&shard)
                .ok_or_else(|| FixtureError::Invalid(format!("missing shard `{shard}`")))?;
            let before = cases.len();
            for (n, line) in text
                .lines()
                .enumerate()
                .filter(|(_, l)| !l.trim().is_empty())
            {
                let v: Value = serde_json::from_str(line)
                    .map_err(|e| FixtureError::Invalid(format!("{shard}:{}: {e}", n + 1)))?;
                expand_case(&v, &family, &base_conf, &mut cases)
                    .map_err(|e| FixtureError::Invalid(format!("{shard}:{}: {e}", n + 1)))?;
            }
            let want = usize_field(fam, "cases")? * Mode::ALL.len();
            if cases.len() - before != want {
                return Err(FixtureError::Invalid(format!(
                    "shard `{shard}` has {} mode cases, manifest says {want}",
                    cases.len() - before
                )));
            }
        }
        let matrix = Self {
            spark_version,
            show_functions,
            functions,
            cases,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    /// Cross-checks: every `SHOW FUNCTIONS` name is a manifest function with at least one case, and
    /// per-function case counts and ids agree with the shards.
    fn validate(&self) -> Result<(), FixtureError> {
        let names: BTreeSet<&str> = self.functions.iter().map(|f| f.name.as_str()).collect();
        let shown: BTreeSet<&str> = self.show_functions.iter().map(String::as_str).collect();
        if names != shown {
            return Err(FixtureError::Invalid(format!(
                "manifest functions differ from SHOW FUNCTIONS: missing {:?}, extra {:?}",
                shown.difference(&names).collect::<Vec<_>>(),
                names.difference(&shown).collect::<Vec<_>>()
            )));
        }
        let fam_of: BTreeMap<&str, &str> = self
            .functions
            .iter()
            .map(|f| (f.name.as_str(), f.family.as_str()))
            .collect();
        let mut per_fn: BTreeMap<&str, usize> = BTreeMap::new();
        let mut ids = BTreeSet::new();
        for c in &self.cases {
            if fam_of.get(c.function.as_str()) != Some(&c.family.as_str()) {
                return Err(FixtureError::Invalid(format!(
                    "case `{}` is in shard `{}`, not its function's family",
                    c.id, c.family
                )));
            }
            *per_fn.entry(c.function.as_str()).or_default() += 1;
            if !ids.insert(c.id.as_str()) {
                return Err(FixtureError::Invalid(format!(
                    "duplicate case id `{}`",
                    c.id
                )));
            }
        }
        for f in &self.functions {
            let got = per_fn.get(f.name.as_str()).copied().unwrap_or(0);
            if f.cases == 0 || got != f.cases * Mode::ALL.len() {
                return Err(FixtureError::Invalid(format!(
                    "function `{}`: {got} mode cases, manifest says {} x {}",
                    f.name,
                    f.cases,
                    Mode::ALL.len()
                )));
            }
        }
        Ok(())
    }

    /// Names of all families, sorted.
    #[must_use]
    pub fn families(&self) -> BTreeSet<&str> {
        self.functions.iter().map(|f| f.family.as_str()).collect()
    }
}

fn shard_paths(manifest: &str) -> Result<Vec<String>, FixtureError> {
    let root: Value = serde_json::from_str(manifest).map_err(|e| invalid(&e))?;
    array(&root, "families")?
        .iter()
        .map(|f| str_field(f, "shard"))
        .collect()
}

fn parse_function(v: &Value) -> Result<FunctionEntry, FixtureError> {
    Ok(FunctionEntry {
        name: str_field(v, "name")?,
        family: str_field(v, "family")?,
        registry: str_field(v, "registry")?,
        cases: usize_field(v, "cases")?,
    })
}

/// Expands one fixture line (family = its shard's) into one case per mode. The line's `conf` holds
/// overrides of the manifest `base_conf`; the mode adds [`ANSI_CONF`].
fn expand_case(
    v: &Value,
    family: &str,
    base_conf: &BTreeMap<String, String>,
    out: &mut Vec<MatrixCase>,
) -> Result<(), FixtureError> {
    let base_id = str_field(v, "id")?;
    let function = str_field(v, "function")?;
    let kind = str_field(v, "kind")?;
    let sql = str_field(v, "sql")?;
    let mut case_conf = base_conf.clone();
    case_conf.extend(string_map(v.get("conf").unwrap_or(&Value::Null))?);
    for mode in Mode::ALL {
        let mut raw = v.get(mode.key()).ok_or_else(|| {
            FixtureError::Invalid(format!("case `{base_id}` lacks `{}`", mode.key()))
        })?;
        if let Some(other) = raw.get("same_as").and_then(Value::as_str) {
            raw = v
                .get(other)
                .filter(|r| r.get("same_as").is_none())
                .ok_or_else(|| FixtureError::Invalid(format!("bad same_as `{other}`")))?;
        }
        let mut conf = case_conf.clone();
        conf.insert(ANSI_CONF.to_owned(), mode.ansi_value().to_owned());
        out.push(MatrixCase {
            id: format!("{base_id}/{}", mode.key()),
            base_id: base_id.clone(),
            function: function.clone(),
            family: family.to_owned(),
            kind: kind.clone(),
            mode,
            sql: sql.clone(),
            conf,
            expected: parse_expected(raw)?,
        });
    }
    Ok(())
}

fn parse_expected(v: &Value) -> Result<Expected, FixtureError> {
    let schema = v.get("schema").map(parse_schema).transpose()?;
    if let Some(e) = v.get("error") {
        return Ok(Expected::Error {
            schema,
            error: SparkError {
                error_class: str_field(e, "error_class")?,
                sqlstate: e.get("sqlstate").and_then(Value::as_str).map(str::to_owned),
                message: str_field(e, "message")?,
                message_parameters: string_map(
                    e.get("message_parameters").unwrap_or(&Value::Null),
                )?,
                phase: str_field(e, "phase")?,
                stable: e.get("stability").and_then(Value::as_str) != Some("unstable"),
            },
        });
    }
    let schema = schema.ok_or_else(|| FixtureError::Invalid("result without `schema`".into()))?;
    if v.get("nondeterministic").and_then(Value::as_bool) == Some(true) {
        return Ok(Expected::Nondeterministic { schema });
    }
    let rows = array(v, "rows")?
        .iter()
        .map(|r| {
            r.as_array()
                .ok_or_else(|| FixtureError::Invalid("row is not an array".into()))?
                .iter()
                .map(|c| match c {
                    Value::Null => Ok(None),
                    Value::String(s) => Ok(Some(s.clone())),
                    other => Err(FixtureError::Invalid(format!(
                        "cell {other} is not a string"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let render = array(v, "render")?
        .iter()
        .map(|r| as_str(r, "render[]"))
        .collect::<Result<Vec<_>, _>>()?;
    let typed_rows = v
        .get("typed_rows")
        .map(|t| {
            t.as_array()
                .ok_or_else(|| FixtureError::Invalid("`typed_rows` is not an array".into()))?
                .iter()
                .map(|r| {
                    r.as_array()
                        .cloned()
                        .ok_or_else(|| FixtureError::Invalid("typed row is not an array".into()))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let typed_unavailable = v
        .get("typed_unavailable")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if typed_rows.is_none() == typed_unavailable.is_none() {
        return Err(FixtureError::Invalid(
            "exactly one of `typed_rows` / `typed_unavailable` expected".into(),
        ));
    }
    if typed_rows
        .as_ref()
        .is_some_and(|t| t.len() != rows.len() || t.iter().any(|r| r.len() != schema.len()))
    {
        return Err(FixtureError::Invalid(
            "typed rows misaligned with rows".into(),
        ));
    }
    if render.len() != schema.len() || rows.iter().any(|r| r.len() != schema.len()) {
        return Err(FixtureError::Invalid(
            "row/render width differs from schema".into(),
        ));
    }
    Ok(Expected::Rows {
        schema,
        rows,
        render,
        ordered: bool_field(v, "ordered")?,
        truncated: bool_field(v, "truncated")?,
        typed_rows,
        typed_unavailable,
    })
}

fn parse_schema(v: &Value) -> Result<Vec<Field>, FixtureError> {
    v.as_array()
        .ok_or_else(|| FixtureError::Invalid("`schema` is not an array".into()))?
        .iter()
        .map(|f| {
            Ok(Field {
                name: str_field(f, "name")?,
                data_type: str_field(f, "type")?,
                nullable: bool_field(f, "nullable")?,
            })
        })
        .collect()
}

/// Outcome of running one [`MatrixCase`] on an engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Matched Spark.
    Pass,
    /// Ran and differed from Spark.
    Fail,
    /// Not run (unsupported, or no engine).
    NotRun,
}

/// Pass counts for one slice of the matrix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Cases.
    pub total: usize,
    /// Cases that passed.
    pub passed: usize,
    /// Cases that ran and failed.
    pub failed: usize,
}

impl Tally {
    fn add(&mut self, o: Outcome) {
        self.total += 1;
        match o {
            Outcome::Pass => self.passed += 1,
            Outcome::Fail => self.failed += 1,
            Outcome::NotRun => {}
        }
    }

    /// Percentage of passed cases (0 for an empty tally).
    #[must_use]
    pub fn percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            // Case counts are far below 2^52, so the conversions are exact.
            100.0 * self.passed as f64 / self.total as f64
        }
    }
}

/// Coverage of one family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyCoverage {
    /// Family name.
    pub family: String,
    /// Functions in the family.
    pub functions: usize,
    /// Functions whose every case passed in both modes.
    pub functions_green: usize,
    /// All cases.
    pub all: Tally,
    /// Cases per mode, in [`Mode::ALL`] order.
    pub by_mode: [Tally; 2],
    /// Cases per [`Category`]: values, errors, type probes.
    pub by_category: [Tally; 3],
}

/// The coverage matrix: % green per family and mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    /// Spark version of the fixtures.
    pub spark_version: String,
    /// Name of the engine/build the outcomes came from.
    pub engine: String,
    /// One row per family, sorted by family name.
    pub families: Vec<FamilyCoverage>,
    /// Totals.
    pub total: FamilyCoverage,
}

impl CoverageReport {
    /// Compute coverage of `matrix` given an outcome per case.
    pub fn compute(
        matrix: &FnMatrix,
        engine: &str,
        mut outcome: impl FnMut(&MatrixCase) -> Outcome,
    ) -> Self {
        let mut fams: BTreeMap<&str, (FamilyCoverage, BTreeMap<&str, bool>)> = BTreeMap::new();
        for f in &matrix.functions {
            fams.entry(f.family.as_str())
                .or_insert_with(|| (empty(&f.family), BTreeMap::new()))
                .1
                .insert(f.name.as_str(), true);
        }
        for c in &matrix.cases {
            let o = outcome(c);
            let (cov, green) = fams
                .entry(c.family.as_str())
                .or_insert_with(|| (empty(&c.family), BTreeMap::new()));
            cov.all.add(o);
            cov.by_mode[mode_index(c.mode)].add(o);
            cov.by_category[c.category() as usize].add(o);
            let g = green.entry(c.function.as_str()).or_insert(true);
            *g &= o == Outcome::Pass;
        }
        let mut total = empty("total");
        let families: Vec<FamilyCoverage> = fams
            .into_values()
            .map(|(mut cov, green)| {
                cov.functions = green.len();
                cov.functions_green = green.values().filter(|g| **g).count();
                total.functions += cov.functions;
                total.functions_green += cov.functions_green;
                merge(&mut total.all, cov.all);
                for (t, m) in total.by_mode.iter_mut().zip(cov.by_mode) {
                    merge(t, m);
                }
                for (t, m) in total.by_category.iter_mut().zip(cov.by_category) {
                    merge(t, m);
                }
                cov
            })
            .collect();
        Self {
            spark_version: matrix.spark_version.clone(),
            engine: engine.to_owned(),
            families,
            total,
        }
    }

    /// Render the dashboard as Markdown (for STATUS.md and CI summaries). Value cells, error cells and
    /// type-mismatch probes are shown separately so error-heavy probes cannot inflate the headline.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let t = &self.total;
        let [values, errors, probes] = &t.by_category;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "### Spark function matrix (Spark {}, engine `{}`): {} green\n",
            self.spark_version,
            self.engine,
            pct(&t.all)
        );
        let _ = writeln!(
            s,
            "Value cells {}, error cells {}, type probes {}. {} / {} cells passing ({} failing, rest \
             not run); {} / {} functions fully green; ANSI on {}, ANSI off {}.\n",
            cell(values),
            cell(errors),
            cell(probes),
            t.all.passed,
            t.all.total,
            t.all.failed,
            t.functions_green,
            t.functions,
            pct(&t.by_mode[0]),
            pct(&t.by_mode[1])
        );
        s.push_str(
            "| Family | Functions green | Value cells | Error cells | Type probes | ANSI on | ANSI off | All |\n",
        );
        s.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
        for f in self.families.iter().chain(std::iter::once(t)) {
            let b = if std::ptr::eq(f, t) { "**" } else { "" };
            let _ = writeln!(
                s,
                "| {b}{}{b} | {} / {} | {} | {} | {} | {} | {} | {b}{}{b} |",
                f.family,
                f.functions_green,
                f.functions,
                cell(&f.by_category[0]),
                cell(&f.by_category[1]),
                cell(&f.by_category[2]),
                pct(&f.by_mode[0]),
                pct(&f.by_mode[1]),
                pct(&f.all)
            );
        }
        s
    }
}

/// `passed/total (pct)`.
fn cell(t: &Tally) -> String {
    format!("{}/{} ({})", t.passed, t.total, pct(t))
}

/// Formats a percentage: `0%`, `100%`, otherwise one decimal (never rounds a partial pass up to 100%).
fn pct(t: &Tally) -> String {
    let p = t.percent();
    if t.passed == 0 {
        "0%".to_owned()
    } else if t.passed == t.total {
        "100%".to_owned()
    } else {
        format!("{:.1}%", (p * 10.0).floor() / 10.0)
    }
}

fn empty(family: &str) -> FamilyCoverage {
    FamilyCoverage {
        family: family.to_owned(),
        functions: 0,
        functions_green: 0,
        all: Tally::default(),
        by_mode: [Tally::default(); 2],
        by_category: [Tally::default(); 3],
    }
}

fn merge(into: &mut Tally, t: Tally) {
    into.total += t.total;
    into.passed += t.passed;
    into.failed += t.failed;
}

const fn mode_index(m: Mode) -> usize {
    match m {
        Mode::AnsiOn => 0,
        Mode::AnsiOff => 1,
    }
}

/// Engine results consumed by the report: `{"engine": "...", "passed": [ids], "failed": [ids]}`.
/// Ids not listed are [`Outcome::NotRun`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EngineResults {
    /// Engine/build label.
    pub engine: String,
    /// Passing case ids (`<base_id>/<mode>`).
    pub passed: BTreeSet<String>,
    /// Failing case ids.
    pub failed: BTreeSet<String>,
}

impl EngineResults {
    /// Parse the results JSON.
    pub fn parse(text: &str) -> Result<Self, FixtureError> {
        let root: Value = serde_json::from_str(text).map_err(|e| invalid(&e))?;
        let ids = |key: &str| -> Result<BTreeSet<String>, FixtureError> {
            match root.get(key) {
                None => Ok(BTreeSet::new()),
                Some(v) => v
                    .as_array()
                    .ok_or_else(|| FixtureError::Invalid(format!("`{key}` is not an array")))?
                    .iter()
                    .map(|s| as_str(s, key))
                    .collect(),
            }
        };
        let res = Self {
            engine: str_field(&root, "engine")?,
            passed: ids("passed")?,
            failed: ids("failed")?,
        };
        if let Some(id) = res.passed.intersection(&res.failed).next() {
            return Err(FixtureError::Invalid(format!(
                "case `{id}` is both passed and failed"
            )));
        }
        Ok(res)
    }

    /// Rejects ids that are not cells of `matrix` (a results file from older fixtures).
    pub fn check_ids(&self, matrix: &FnMatrix) -> Result<(), FixtureError> {
        let known: BTreeSet<&str> = matrix.cases.iter().map(|c| c.id.as_str()).collect();
        let unknown: Vec<&String> = self
            .passed
            .iter()
            .chain(&self.failed)
            .filter(|id| !known.contains(id.as_str()))
            .collect();
        match unknown.first() {
            None => Ok(()),
            Some(first) => Err(FixtureError::Invalid(format!(
                "{} result ids are not in the fixtures (stale results?), e.g. `{first}`",
                unknown.len()
            ))),
        }
    }

    /// Outcome of `case`.
    #[must_use]
    pub fn outcome(&self, case: &MatrixCase) -> Outcome {
        if self.passed.contains(&case.id) {
            Outcome::Pass
        } else if self.failed.contains(&case.id) {
            Outcome::Fail
        } else {
            Outcome::NotRun
        }
    }
}

fn read(path: &Path) -> Result<String, FixtureError> {
    std::fs::read_to_string(path).map_err(|source| FixtureError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn invalid(e: &serde_json::Error) -> FixtureError {
    FixtureError::Invalid(e.to_string())
}

fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>, FixtureError> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| FixtureError::Invalid(format!("missing array `{key}`")))
}

fn as_str(v: &Value, what: &str) -> Result<String, FixtureError> {
    v.as_str()
        .map(str::to_owned)
        .ok_or_else(|| FixtureError::Invalid(format!("`{what}` is not a string")))
}

fn str_field(v: &Value, key: &str) -> Result<String, FixtureError> {
    v.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| FixtureError::Invalid(format!("missing string field `{key}`")))
}

fn bool_field(v: &Value, key: &str) -> Result<bool, FixtureError> {
    v.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| FixtureError::Invalid(format!("missing bool field `{key}`")))
}

fn usize_field(v: &Value, key: &str) -> Result<usize, FixtureError> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| FixtureError::Invalid(format!("missing count field `{key}`")))
}

fn string_map(v: &Value) -> Result<BTreeMap<String, String>, FixtureError> {
    match v {
        Value::Null => Ok(BTreeMap::new()),
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| Ok((k.clone(), as_str(v, k)?)))
            .collect(),
        _ => Err(FixtureError::Invalid(
            "expected an object of strings".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    const MANIFEST: &str = r#"{"format_version":1,"spark_version":"4.2.0",
      "base_conf":{"spark.sql.session.timeZone":"UTC"},
      "show_functions":["abs","explode"],
      "functions":[{"name":"abs","family":"math","registry":"function","cases":2},
                   {"name":"explode","family":"generator","registry":"function","cases":1}],
      "families":[{"family":"generator","shard":"cases/generator.jsonl","functions":1,"cases":1},
                  {"family":"math","shard":"cases/math.jsonl","functions":1,"cases":2}]}"#;
    const MATH: &str = concat!(
        r#"{"id":"abs#1","function":"abs","kind":"example0","sql":"SELECT abs(-1)","#,
        r#""conf":{},"ansi_on":{"schema":[{"name":"abs(-1)","type":"int","nullable":false}],"#,
        r#""rows":[["1"]],"typed_rows":[[1]],"render":["cast"],"ordered":false,"truncated":false},"ansi_off":{"same_as":"ansi_on"}}"#,
        "\n",
        r#"{"id":"abs#2","function":"abs","kind":"example0:edge:int_min@arg0","sql":"SELECT abs(CAST(-2147483648 AS INT))","#,
        r#""conf":{"spark.sql.session.timeZone":"America/Los_Angeles"},"ansi_on":{"schema":[{"name":"x","type":"int","nullable":false}],"#,
        r#""error":{"error_class":"ARITHMETIC_OVERFLOW","sqlstate":"22003","message":"[ARITHMETIC_OVERFLOW] overflow.","#,
        r#""message_parameters":{"message":"overflow"},"phase":"execution"}},"#,
        r#""ansi_off":{"schema":[{"name":"x","type":"int","nullable":false}],"rows":[["-2147483648"]],"typed_unavailable":"X","render":["cast"],"ordered":false,"truncated":false}}"#,
        "\n"
    );
    const GEN: &str = concat!(
        r#"{"id":"explode#1","function":"explode","kind":"example0","sql":"SELECT explode(array(10, 20))","#,
        r#""conf":{},"ansi_on":{"schema":[{"name":"col","type":"int","nullable":false}],"rows":[["10"],["20"]],"typed_rows":[[10],[20]],"#,
        r#""render":["cast"],"ordered":false,"truncated":false},"ansi_off":{"same_as":"ansi_on"}}"#
    );

    fn shards(math: &str) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("cases/math.jsonl".to_owned(), math.to_owned()),
            ("cases/generator.jsonl".to_owned(), GEN.to_owned()),
        ])
    }

    #[test]
    fn parses_and_expands_modes() -> R {
        let m = FnMatrix::parse(MANIFEST, &shards(MATH))?;
        assert_eq!(m.cases.len(), 6);
        let on = m
            .cases
            .iter()
            .find(|c| c.id == "abs#2/ansi_on")
            .ok_or("missing")?;
        assert!(on.expected.is_error());
        assert_eq!(on.conf.get(ANSI_CONF).map(String::as_str), Some("true"));
        let tz = |c: &MatrixCase| c.conf.get("spark.sql.session.timeZone").cloned();
        assert_eq!(tz(on).as_deref(), Some("America/Los_Angeles"));
        assert_eq!(on.family, "math");
        let off = m
            .cases
            .iter()
            .find(|c| c.id == "abs#1/ansi_off")
            .ok_or("missing")?;
        assert_eq!(tz(off).as_deref(), Some("UTC"));
        assert!(
            matches!(&off.expected, Expected::Rows { rows, .. } if rows == &[vec![Some("1".to_owned())]])
        );
        assert_eq!(
            m.families().into_iter().collect::<Vec<_>>(),
            ["generator", "math"]
        );
        Ok(())
    }

    #[test]
    fn rejects_count_mismatch_and_bad_shapes() {
        let one_line = MATH.lines().next().unwrap_or_default();
        assert!(FnMatrix::parse(MANIFEST, &shards(one_line)).is_err());
        assert!(FnMatrix::parse("{}", &shards(MATH)).is_err());
        let v2 = MANIFEST.replace("\"format_version\":1", "\"format_version\":2");
        assert!(FnMatrix::parse(&v2, &shards(MATH)).is_err());
        let missing_fn = MANIFEST.replace(r#"["abs","explode"]"#, r#"["abs","explode","sum"]"#);
        assert!(FnMatrix::parse(&missing_fn, &shards(MATH)).is_err());
        let bad_same = MATH.replace(r#"{"same_as":"ansi_on"}"#, r#"{"same_as":"nope"}"#);
        assert!(FnMatrix::parse(MANIFEST, &shards(&bad_same)).is_err());
        let no_typed = MATH.replace(r#""typed_rows":[[1]],"#, "");
        assert!(FnMatrix::parse(MANIFEST, &shards(&no_typed)).is_err());
        let misaligned = MATH.replace(r#""typed_rows":[[1]]"#, r#""typed_rows":[[1],[2]]"#);
        assert!(FnMatrix::parse(MANIFEST, &shards(&misaligned)).is_err());
    }

    #[test]
    fn rejects_duplicate_ids_and_wrong_family_shards() {
        let dup = MATH.replace(r#""id":"abs#2""#, r#""id":"abs#1""#);
        let err = FnMatrix::parse(MANIFEST, &shards(&dup)).map(|_| ());
        assert!(format!("{err:?}").contains("duplicate case id"), "{err:?}");
        // Same per-shard counts, but `abs#1` sits in the generator shard and `explode#1` in math.
        let mut lines = MATH.lines();
        let (abs1, abs2) = (lines.next().unwrap_or(""), lines.next().unwrap_or(""));
        let swapped = BTreeMap::from([
            ("cases/math.jsonl".to_owned(), format!("{abs2}\n{GEN}\n")),
            ("cases/generator.jsonl".to_owned(), abs1.to_owned()),
        ]);
        let err = FnMatrix::parse(MANIFEST, &swapped).map(|_| ());
        assert!(
            format!("{err:?}").contains("not its function's family"),
            "{err:?}"
        );
    }

    #[test]
    fn categories_and_error_stability() -> R {
        let legacy = MATH.replace(
            r#""phase":"execution"}"#,
            r#""phase":"execution","stability":"unstable"}"#,
        );
        let m = FnMatrix::parse(MANIFEST, &shards(&legacy))?;
        let get = |id: &str| m.cases.iter().find(|c| c.id == id).ok_or("missing");
        assert_eq!(get("abs#2/ansi_on")?.category(), Category::Error);
        assert_eq!(get("abs#2/ansi_off")?.category(), Category::Value);
        assert!(
            matches!(&get("abs#2/ansi_on")?.expected, Expected::Error { error, .. } if !error.stable)
        );
        let probe = MATH.replace("example0:edge:int_min@arg0", "example0:type:binary@arg0");
        let m = FnMatrix::parse(MANIFEST, &shards(&probe))?;
        let c = m
            .cases
            .iter()
            .find(|c| c.id == "abs#2/ansi_on")
            .ok_or("missing")?;
        assert_eq!(c.category(), Category::TypeProbe);
        Ok(())
    }

    #[test]
    fn engine_results_reject_overlap_and_stale_ids() -> R {
        let both = r#"{"engine":"k","passed":["abs#1/ansi_on"],"failed":["abs#1/ansi_on"]}"#;
        assert!(EngineResults::parse(both).is_err());
        let m = FnMatrix::parse(MANIFEST, &shards(MATH))?;
        let stale = EngineResults::parse(r#"{"engine":"k","passed":["gone#1/ansi_on"]}"#)?;
        assert!(stale.check_ids(&m).is_err());
        let ok = EngineResults::parse(r#"{"engine":"k","failed":["abs#1/ansi_off"]}"#)?;
        ok.check_ids(&m)?;
        Ok(())
    }

    #[test]
    fn empty_engine_report_is_zero_percent() -> R {
        let m = FnMatrix::parse(MANIFEST, &shards(MATH))?;
        let r = CoverageReport::compute(&m, "empty", |_| Outcome::NotRun);
        assert_eq!(r.total.all.total, 6);
        let md = r.render_markdown();
        assert!(md.contains("engine `empty`): 0% green"), "{md}");
        assert!(
            md.contains(
                "| **total** | 0 / 2 | 0/5 (0%) | 0/1 (0%) | 0/0 (0%) | 0% | 0% | **0%** |"
            ),
            "{md}"
        );
        Ok(())
    }

    #[test]
    fn report_counts_passes_per_family_and_mode() -> R {
        let m = FnMatrix::parse(MANIFEST, &shards(MATH))?;
        let res = EngineResults::parse(
            r#"{"engine":"k","passed":["explode#1/ansi_on","explode#1/ansi_off","abs#1/ansi_on"],"failed":["abs#2/ansi_on"]}"#,
        )?;
        let r = CoverageReport::compute(&m, &res.engine, |c| res.outcome(c));
        let generator = &r.families[0];
        assert_eq!(
            (
                generator.family.as_str(),
                generator.functions_green,
                generator.all.passed
            ),
            ("generator", 1, 2)
        );
        let math = &r.families[1];
        assert_eq!(
            (math.all.passed, math.all.failed, math.functions_green),
            (1, 1, 0)
        );
        assert_eq!((math.by_mode[0].passed, math.by_mode[1].passed), (1, 0));
        assert_eq!(r.total.all.passed, 3);
        let md = r.render_markdown();
        assert!(md.contains("): 50.0% green"), "{md}");
        assert!(
            md.contains(
                "| generator | 1 / 1 | 2/2 (100%) | 0/0 (0%) | 0/0 (0%) | 100% | 100% | 100% |"
            ),
            "{md}"
        );
        assert!(
            md.contains("Value cells 3/5 (60.0%), error cells 0/1 (0%)"),
            "{md}"
        );
        Ok(())
    }

    #[test]
    fn percent_never_rounds_up_to_full() {
        let t = Tally {
            total: 10_000,
            passed: 9_999,
            failed: 1,
        };
        assert_eq!(pct(&t), "99.9%");
    }

    /// Acceptance (P2-T106a): the committed fixtures load, every `SHOW FUNCTIONS` name is covered, every
    /// case parses, and the dashboard renders 0% green against the (still empty) engine.
    #[test]
    fn committed_fixtures_load_every_case() -> R {
        let m = FnMatrix::load_default()?;
        assert_eq!(m.spark_version, "4.2.0");
        assert!(
            m.show_functions.len() > 500,
            "{} functions",
            m.show_functions.len()
        );
        assert_eq!(m.functions.len(), m.show_functions.len());
        assert!(m.cases.len() >= m.functions.len() * Mode::ALL.len());
        for mode in Mode::ALL {
            assert_eq!(
                m.cases.iter().filter(|c| c.mode == mode).count() * 2,
                m.cases.len()
            );
        }
        let md = CoverageReport::compute(&m, "krishiv (no functions yet)", |_| Outcome::NotRun)
            .render_markdown();
        assert!(md.contains(": 0% green"), "{md}");
        Ok(())
    }
}
