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
pub const OUT_DIR: &str = "docs/reference/sql";

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

fn status(name: &str, def: Option<&FunctionDef>) -> &'static str {
    match def {
        None if is_syntax(name) => "Implemented (SQL syntax)",
        None => "Not in the function registry yet",
        Some(d) if matches!(d.execution, Execution::Cast) => "Implemented (rewritten to a `CAST`)",
        Some(_) => "Implemented",
    }
}

fn page(doc: &SparkDoc, reg: &Registry) -> String {
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
    if !sig.is_empty() {
        let _ = write!(m, "```text\n{sig}\n```\n\n");
    }
    if !desc.is_empty() {
        let _ = write!(m, "{desc}\n\n");
    }
    if !doc.deprecated.trim().is_empty() {
        let _ = write!(m, "**Deprecated.** {}\n\n", doc.deprecated.trim());
    }
    m.push_str("| | |\n|---|---|\n");
    let _ = writeln!(m, "| Krishiv | {} |", status(&doc.name, def));
    if !doc.group.is_empty() {
        let _ = writeln!(m, "| Group | `{}` |", doc.group);
    }
    let kind = def
        .map(|d| d.kind)
        .or_else(|| spark.and_then(|s| s.kind))
        .map_or("unknown", kind_name);
    let _ = writeln!(m, "| Kind | {kind} ({}) |", doc.registry.replace('_', " "));
    if !doc.since.is_empty() {
        let _ = writeln!(m, "| Added in Spark | {} |", doc.since);
    }
    if let Some(s) = spark {
        let _ = writeln!(m, "| In Spark profiles | {} and later |", s.since.line());
        let _ = writeln!(
            m,
            "| Arguments | {} |",
            s.arity.expected().replace('|', "\\|")
        );
        if let Some(d) = s.deterministic {
            let _ = writeln!(m, "| Deterministic | {} |", if d { "yes" } else { "no" });
        }
    }
    m.push('\n');
    let examples = parse_examples(&doc.examples);
    if !examples.is_empty() {
        m.push_str("## Examples\n\n");
        for (sql, result) in examples {
            let _ = write!(m, "```sql\n{sql}\n```\n\n");
            if !result.is_empty() {
                let _ = write!(m, "Result:\n\n```text\n{result}\n```\n\n");
            }
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
    let mut docs: Vec<SparkDoc> = Vec::new();
    for line in DOCS_JSONL.lines().filter(|l| !l.trim().is_empty()) {
        docs.push(
            serde_json::from_str(line).map_err(|e| format!("spark-function-docs.jsonl: {e}"))?,
        );
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
        if out.insert(file.clone(), page(doc, reg)).is_some() {
            return Err(format!("two functions share the page {}", file.display()));
        }
        let def = matches!(
            reg.lookup(&doc.name, krishiv_functions::signature::SparkProfile::V4_2),
            krishiv_functions::registry::Lookup::Found(_)
        );
        let def = def || is_syntax(&doc.name);
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
                "docs/reference/sql is stale ({} files, first: {}); run `cargo xtask docs-sql`",
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
