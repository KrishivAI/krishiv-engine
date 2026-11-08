// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask docs-sql [--check]`: one Markdown page per SQL function under `docs/reference/sql/`,
//! plus an index. The text (usage, examples, group, release) is Spark 4.2.0's own, dumped by
//! `tests/scripts/gen_function_docs.py` into `vendored/spark-function-docs.jsonl`; the Krishiv status,
//! determinism, kind and argument counts come from the function registry.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use krishiv_functions::registry::{Execution, FunctionDef, Registry, builtins};
use krishiv_functions::signature::FunctionKind;
use serde::Deserialize;

/// Where the pages go, relative to the workspace root.
pub const OUT_DIR: &str = "docs/reference/sql/functions";

const DOCS_JSONL: &str =
    include_str!("../../crates/krishiv-functions/vendored/spark-function-docs.jsonl");

/// One line of the Spark documentation dump.
#[derive(Debug, Deserialize)]
struct SparkDoc {
    name: String,
    registry: String,
    group: String,
    since: String,
    deprecated: String,
    usage: String,
    examples: String,
}

/// The file stem of a function's page: the name itself when it is `[a-z0-9_]+`, else `op-` and the
/// words of its symbols (`<=>` is `op-lt-eq-gt`).
#[must_use]
pub fn slug(name: &str) -> String {
    if !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        return name.to_owned();
    }
    let words: Vec<String> = name
        .chars()
        .map(|c| match c {
            '!' => "bang".to_owned(),
            '%' => "percent".to_owned(),
            '&' => "amp".to_owned(),
            '*' => "star".to_owned(),
            '+' => "plus".to_owned(),
            '-' => "minus".to_owned(),
            '/' => "slash".to_owned(),
            '<' => "lt".to_owned(),
            '=' => "eq".to_owned(),
            '>' => "gt".to_owned(),
            '^' => "caret".to_owned(),
            '|' => "pipe".to_owned(),
            '~' => "tilde".to_owned(),
            ':' => "colon".to_owned(),
            '[' => "lbracket".to_owned(),
            ']' => "rbracket".to_owned(),
            c if c.is_ascii_alphanumeric() || c == '_' => c.to_ascii_lowercase().to_string(),
            c => format!("u{:x}", u32::from(c)),
        })
        .collect();
    format!("op-{}", words.join("-"))
}

/// Splits Spark's usage text into its signature (before the first ` - `) and description.
fn split_usage(usage: &str) -> (String, String) {
    let usage = usage.trim();
    match usage.split_once(" - ") {
        Some((sig, rest)) => (sig.trim().to_owned(), dedent(rest)),
        None => (String::new(), dedent(usage)),
    }
}

/// Removes the common left margin of continuation lines (Spark indents them to align with the
/// source) and trims trailing whitespace.
fn dedent(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let margin = lines
        .iter()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        out.push(if i == 0 {
            l.trim()
        } else {
            l.get(margin..).unwrap_or("").trim_end()
        });
    }
    out.join("\n").trim().to_owned()
}

/// Spark's example block as `(sql, result)` pairs.
fn parse_examples(block: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut sql_open = false;
    for line in block.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("> ") {
            out.push((rest.trim().to_owned(), String::new()));
            sql_open = !rest.trim_end().ends_with(';');
        } else if t.is_empty() || t.ends_with("Examples:") {
            // spacing and the heading
        } else if let Some(last) = out.last_mut() {
            if sql_open {
                last.0.push('\n');
                last.0.push_str(t);
                sql_open = !t.ends_with(';');
            } else {
                if !last.1.is_empty() {
                    last.1.push('\n');
                }
                last.1.push_str(t);
            }
        }
    }
    out
}

/// The id of a function page's `n`th example in the captured-output file.
fn example_id(name: &str, n: usize) -> String {
    format!("{}#{}", slug(name), n + 1)
}

/// The file with the engine's answers to the examples, written by `cargo xtask docs-examples`.
pub const CAPTURED: &str = "docs/reference/sql/functions/_examples.json";

fn captured_examples() -> serde_json::Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."));
    std::fs::read_to_string(root.join(CAPTURED))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// The lines under an example's input: the engine's captured answer, the error class of a failed
/// example, a note for an answer that differs between runs, or (not yet captured) Spark's own.
fn shown_result(captured: Option<&serde_json::Value>, spark: &str) -> String {
    let Some(c) = captured else {
        return if spark.is_empty() {
            String::new()
        } else {
            format!("{spark}\n")
        };
    };
    if c.get("volatile").and_then(serde_json::Value::as_bool) == Some(true) {
        return "-- the result differs on every run\n".to_owned();
    }
    if c.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
        let rows = c.get("rows").and_then(serde_json::Value::as_array);
        return rows
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .fold(String::new(), |mut a, r| {
                a.push_str(&r.replace('\t', " | "));
                a.push('\n');
                a
            });
    }
    format!(
        "Error: {}\n",
        c.get("class")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unclassified")
    )
}

/// The characters of an answer without whitespace, sorted, for comparing against Spark's documented
/// one: row order of an unordered query, row breaks and spacing do not count, a changed digit does.
fn squash(text: &str) -> Vec<char> {
    let mut v: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    v.sort_unstable();
    v
}

/// Whether the engine's captured answer differs from the result Spark documents for the example.
/// Examples without a documented result, failed examples and volatile ones never differ.
fn differs_from_spark(captured: Option<&serde_json::Value>, spark: &str) -> bool {
    let Some(c) = captured else { return false };
    if spark.trim().is_empty()
        || spark.trim_start().starts_with('+')
        || c.get("volatile").and_then(serde_json::Value::as_bool) == Some(true)
        || c.get("ok").and_then(serde_json::Value::as_bool) != Some(true)
    {
        return false;
    }
    let ours = shown_result(Some(c), "");
    squash(&ours.replace(" | ", "\t")) != squash(spark)
}

/// Every example of every function page as `(id, sql)`.
///
/// # Errors
/// When the vendored documentation dump is malformed.
pub fn examples() -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for line in DOCS_JSONL.lines().filter(|l| !l.trim().is_empty()) {
        let doc: SparkDoc =
            serde_json::from_str(line).map_err(|e| format!("spark-function-docs.jsonl: {e}"))?;
        for (n, (sql, _)) in parse_examples(&doc.examples).into_iter().enumerate() {
            out.push((example_id(&doc.name, n), sql));
        }
    }
    Ok(out)
}

fn kind_name(kind: FunctionKind) -> &'static str {
    match kind {
        FunctionKind::Scalar => "scalar",
        FunctionKind::Aggregate => "aggregate",
        FunctionKind::Window => "window",
        FunctionKind::Generator => "generator",
        _ => "other",
    }
}

/// Names Spark lists as functions that Krishiv's parser and analyzer own as syntax (operators and
/// keywords), so the function registry has no entry for them.
fn is_syntax(name: &str) -> bool {
    !name.starts_with(|c: char| c.is_ascii_alphabetic())
        || matches!(
            name,
            "and"
                | "or"
                | "not"
                | "between"
                | "case"
                | "when"
                | "cast"
                | "in"
                | "like"
                | "ilike"
                | "rlike"
        )
}

/// Names implemented outside the function registry: the planner's built-in operators
/// (`krishiv_plan::Builtin`, e.g. `concat`, `div`), the grouping functions the analyzer lowers,
/// the table functions `range` and `sql_keywords`, and `collate`, which the SQL lowering handles.
fn plan_owned(name: &str) -> bool {
    krishiv_plan::Builtin::from_name(name).is_some()
        || matches!(
            name.to_ascii_lowercase().as_str(),
            "grouping" | "grouping_id" | "range" | "sql_keywords" | "collate"
        )
}

fn status(name: &str, def: Option<&FunctionDef>) -> &'static str {
    match def {
        None if is_syntax(name) => "Implemented (SQL syntax)",
        None if plan_owned(name) => "Implemented",
        None => "Not implemented yet",
        Some(d) if matches!(d.execution, Execution::Cast) => "Implemented (rewritten to a `CAST`)",
        Some(_) => "Implemented",
    }
}

fn page(
    doc: &SparkDoc,
    reg: &Registry,
    captured: &serde_json::Value,
    groups: &BTreeMap<String, Vec<String>>,
) -> String {
    let def = match reg.lookup(&doc.name, krishiv_functions::signature::SparkProfile::V4_2) {
        krishiv_functions::registry::Lookup::Found(d) => Some(d),
        _ => None,
    };
    let (sig, desc) = split_usage(&doc.usage);
    let spark = reg.spark_function(&doc.name);
    let mut m = String::new();
    let _ = write!(
        m,
        "# `{}`\n\n<!-- @generated by `cargo xtask docs-sql`; do not edit. -->\n\n",
        doc.name
    );
    if !desc.is_empty() {
        let _ = write!(m, "{desc}\n\n");
    }
    let implemented = def.is_some() || is_syntax(&doc.name) || plan_owned(&doc.name);
    let krishiv = if implemented {
        "Krishiv 0.2.0 and above"
    } else {
        "not available in Krishiv yet"
    };
    let spark_line = spark.map_or_else(
        || doc.since.clone(),
        |s| format!("{} and later", s.since.line()),
    );
    let _ = write!(m, "**Applies to:** {krishiv}. Spark: {spark_line}.\n\n");
    if !sig.is_empty() {
        let _ = write!(m, "## Syntax\n\n```text\n{sig}\n```\n\n");
    }
    if !doc.deprecated.trim().is_empty() {
        let _ = write!(m, "**Deprecated.** {}\n\n", doc.deprecated.trim());
    }
    m.push_str("## Details\n\n| | |\n|---|---|\n");
    let _ = writeln!(m, "| Krishiv | {} |", status(&doc.name, def));
    if !doc.group.is_empty() {
        let _ = writeln!(m, "| Group | `{}` |", doc.group);
    }
    let kind = def
        .map(|d| d.kind)
        .or_else(|| spark.and_then(|s| s.kind))
        .map_or("unknown", kind_name);
    if doc.registry == "table_function" && def.is_none() {
        m.push_str("| Kind | table function |\n");
    } else {
        let _ = writeln!(m, "| Kind | {kind} ({}) |", doc.registry.replace('_', " "));
    }
    if !doc.since.is_empty() {
        let _ = writeln!(m, "| Added in Spark | {} |", doc.since);
    }
    if let Some(s) = spark {
        let _ = writeln!(m, "| In Spark profiles | {} and later |", s.since.line());
        let args = s.arity.expected().trim();
        let _ = writeln!(
            m,
            "| Arguments | {} |",
            if args.is_empty() {
                "-".to_owned()
            } else {
                args.replace('|', "\\|")
            }
        );
        if let Some(d) = s.deterministic {
            let _ = writeln!(m, "| Deterministic | {} |", if d { "yes" } else { "no" });
        }
    }
    m.push('\n');
    let examples = parse_examples(&doc.examples);
    if !examples.is_empty() {
        m.push_str("## Examples\n\n");
        for (n, (sql, spark_result)) in examples.iter().enumerate() {
            let id = example_id(&doc.name, n);
            let _ = writeln!(m, "```sql\n-- {}, example {}", doc.name, n + 1);
            for l in sql.lines() {
                let _ = writeln!(m, "> {l}");
            }
            m.push_str(&shown_result(captured.get(&id), spark_result));
            if differs_from_spark(captured.get(&id), spark_result) {
                let _ = writeln!(m, "-- Spark documents: {}", spark_result.replace('\n', " "));
            }
            m.push_str("```\n\n");
        }
    }
    m.push_str("## Version history\n\n");
    if !doc.since.is_empty() {
        let _ = writeln!(m, "- Added in Spark {}.", doc.since);
    }
    if implemented {
        m.push_str("- Added in Krishiv 0.2.0.\n");
    }
    m.push('\n');
    let related: Vec<&String> = groups
        .get(&doc.group)
        .into_iter()
        .flatten()
        .filter(|n| **n != doc.name)
        .take(8)
        .collect();
    if !related.is_empty() {
        m.push_str("## Related\n\n");
        for n in related {
            let _ = writeln!(m, "- [`{n}`]({}.md)", slug(n));
        }
    }
    m
}

/// Every generated file: path (relative to the workspace root) and content.
///
/// # Errors
/// When the vendored documentation dump is malformed or two names share a page.
pub fn render() -> Result<BTreeMap<PathBuf, String>, String> {
    let reg = builtins();
    let captured = captured_examples();
    let mut docs: Vec<SparkDoc> = Vec::new();
    for line in DOCS_JSONL.lines().filter(|l| !l.trim().is_empty()) {
        docs.push(
            serde_json::from_str(line).map_err(|e| format!("spark-function-docs.jsonl: {e}"))?,
        );
    }
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for d in &docs {
        groups
            .entry(d.group.clone())
            .or_default()
            .push(d.name.clone());
    }
    let mut out = BTreeMap::new();
    let mut index = String::from(
        "# SQL function reference\n\n<!-- @generated by `cargo xtask docs-sql`; do not edit. -->\n\n\
         Every Spark built-in function and operator, with Spark 4.2.0's description and examples and Krishiv's status.\n\n\
         | Function | Kind | Krishiv | Description |\n|---|---|---|---|\n",
    );
    let mut implemented = 0_usize;
    for doc in &docs {
        let file = PathBuf::from(OUT_DIR).join(format!("{}.md", slug(&doc.name)));
        if out
            .insert(file.clone(), page(doc, reg, &captured, &groups))
            .is_some()
        {
            return Err(format!("two functions share the page {}", file.display()));
        }
        let def = matches!(
            reg.lookup(&doc.name, krishiv_functions::signature::SparkProfile::V4_2),
            krishiv_functions::registry::Lookup::Found(_)
        );
        let def = def || is_syntax(&doc.name) || plan_owned(&doc.name);
        implemented += usize::from(def);
        let (_, desc) = split_usage(&doc.usage);
        let first = desc.lines().next().unwrap_or("").replace('|', "\\|");
        let kind = reg
            .spark_function(&doc.name)
            .and_then(|s| s.kind)
            .map_or("-", kind_name);
        let _ = writeln!(
            index,
            "| [`{}`]({}.md) | {kind} | {} | {first} |",
            doc.name.replace('|', "\\|"),
            slug(&doc.name),
            if def { "yes" } else { "not yet" }
        );
    }
    let _ = write!(
        index,
        "\n{implemented} of {} functions are implemented.\n",
        docs.len()
    );
    out.insert(PathBuf::from(OUT_DIR).join("index.md"), index);
    out.insert(
        PathBuf::from(OUT_DIR).join("_meta.json"),
        "{\n  \"index\": \"SQL functions\"\n}\n".to_owned(),
    );
    Ok(out)
}

/// Writes the pages (removing stale ones), or with `check` compares them with the files on disk.
///
/// # Errors
/// On I/O errors, or with `check` when a file is missing, stale or extra.
pub fn run(root: &Path, check: bool) -> Result<(), String> {
    let want = render()?;
    let dir = root.join(OUT_DIR);
    let mut have: Vec<PathBuf> = Vec::new();
    if dir.is_dir() {
        for e in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
            let p = e.map_err(|e| e.to_string())?.path();
            if p.extension().is_some_and(|x| x == "md") {
                have.push(PathBuf::from(OUT_DIR).join(p.file_name().unwrap_or_default()));
            }
        }
    }
    if check {
        let mut bad = Vec::new();
        for (path, text) in &want {
            if std::fs::read_to_string(root.join(path)).ok().as_deref() != Some(text) {
                bad.push(path.display().to_string());
            }
        }
        bad.extend(
            have.iter()
                .filter(|p| !want.contains_key(*p))
                .map(|p| format!("{} (extra)", p.display())),
        );
        return if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "docs/reference/sql/functions is stale ({} files, first: {}); run `cargo xtask docs-sql`",
                bad.len(),
                bad[0]
            ))
        };
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for p in have.iter().filter(|p| !want.contains_key(*p)) {
        std::fs::remove_file(root.join(p)).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    for (path, text) in &want {
        std::fs::write(root.join(path), text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}
