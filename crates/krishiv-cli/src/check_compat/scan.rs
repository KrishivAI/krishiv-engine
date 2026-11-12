// SPDX-License-Identifier: Apache-2.0
//! The scanners: SQL text, Python source (SQL passed to `spark.sql` and friends,
//! and calls into the `pyspark.sql.functions` module as imported) and notebook
//! code cells. Nothing here executes or parses beyond what it needs to find calls.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Value, json};

use super::{Status, matrix};

/// One reported call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Finding {
    /// File the call is in.
    pub(crate) file: String,
    /// 1-based line in the file (for notebooks, the line in the `.ipynb` JSON).
    pub(crate) line: usize,
    /// The function name, lower case.
    pub(crate) name: String,
    /// `Partial`, `NotYet` or `Unknown` (`Supported` is never reported).
    pub(crate) status: Status,
    /// 0-based index of the notebook cell, for `.ipynb` files.
    pub(crate) cell: Option<usize>,
}

impl Finding {
    /// The text report line.
    pub(crate) fn text(&self) -> String {
        let what = match self.status {
            Status::Unknown => "unknown, not supported",
            s => s.label(),
        };
        format!("{}:{}: {}: {what}", self.file, self.line, self.name)
    }

    /// The JSON object of the stable report format.
    pub(crate) fn json(&self) -> Value {
        let mut v = json!({
            "file": self.file, "line": self.line, "name": self.name, "status": self.status.label(),
        });
        if let (Some(c), Some(o)) = (self.cell, v.as_object_mut()) {
            o.insert("cell".to_string(), json!(c));
        }
        v
    }
}

/// SQL words that may be followed by `(` and are not function calls.
const SQL_WORDS: &[&str] = &[
    "and",
    "or",
    "not",
    "in",
    "exists",
    "values",
    "over",
    "partition",
    "order",
    "by",
    "from",
    "join",
    "on",
    "where",
    "having",
    "group",
    "as",
    "table",
    "into",
    "select",
    "using",
    "union",
    "all",
    "distinct",
    "with",
    "when",
    "then",
    "else",
    "end",
    "is",
    "like",
    "between",
    "limit",
    "view",
    "function",
    "if",
    "for",
    "set",
    "insert",
    "overwrite",
    "create",
    "replace",
    "temp",
    "temporary",
    "range",
    "rows",
    "unbounded",
    "preceding",
    "following",
    "current",
    "row",
    "lateral",
    "cluster",
    "distribute",
    "sort",
    "returns",
    "div",
    "qualify",
    "outer",
    "rlike",
    "ilike",
    "regexp",
    "tblproperties",
    "options",
    "comment",
    "partitioned",
    "stored",
    "location",
    "struct",
    "array",
    "map",
    "interval",
    "filter",
    "within",
    "case",
    "any",
    "some",
    "except",
    "intersect",
    "pivot",
    "unpivot",
    "tablesample",
    "default",
    "primary",
    "key",
    "constraint",
    "unique",
    "check",
    "foreign",
    "references",
    "update",
    "delete",
    "merge",
    "matched",
    "source",
    "target",
    "cross",
    "natural",
    "inner",
    "outer",
    "left",
    "right",
    "full",
    "semi",
    "anti",
    "of",
    "to",
    "at",
    "exclude",
    "include",
    "nulls",
    "first",
    "last",
    "asc",
    "desc",
    "rollup",
    "cube",
    "grouping",
    "sets",
    "cast",
    "try_cast",
    "identifier",
    "return",
    "repeatable",
    "columns",
    "varchar",
    "character",
    "numeric",
    "geometry",
    "geography",
    "variable",
    "variables",
    "recursive",
    "until",
    "while",
    "do",
    "rlike",
    "regexp",
    "ilike",
    "escape",
    "minus",
    "xor",
    "explain",
];

/// PySpark names reachable as `F.<name>(` that have no SQL function page, so the
/// matrix cannot judge them (DataFrame helpers, camelCase and `_distinct` aliases).
const PY_ONLY: &[&str] = &[
    "col",
    "column",
    "lit",
    "expr",
    "udf",
    "pandas_udf",
    "broadcast",
    "asc",
    "desc",
    "asc_nulls_first",
    "asc_nulls_last",
    "desc_nulls_first",
    "desc_nulls_last",
    "call_udf",
    "call_function",
    "udtf",
    "countdistinct",
    "count_distinct",
    "sumdistinct",
    "sum_distinct",
    "create_map",
    "approxcountdistinct",
];

/// Scans one file by extension.
pub(crate) fn scan_file(path: &Path) -> std::io::Result<Vec<Finding>> {
    let text = std::fs::read_to_string(path)?;
    let name = path.display().to_string();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    Ok(match ext {
        "sql" => scan_sql(&name, &text, 1),
        "py" => scan_python(&name, &text, 1),
        "ipynb" => scan_notebook(&name, &text)?,
        _ => Vec::new(),
    })
}

/// Drops repeats of one name on one line, keeping the first.
fn dedup(out: &mut Vec<Finding>) {
    let mut seen = BTreeSet::new();
    out.retain(|f| seen.insert((f.line, f.name.clone(), f.cell)));
}

/// True when the `(` at `open` closes into `AS (`: a CTE column list, not a call.
fn is_cte_column_list(bytes: &[u8], open: usize) -> bool {
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    break;
                }
            }
            q @ (b'\'' | b'"' | b'`') => {
                i += 1;
                while i < bytes.len() && bytes[i] != q {
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    let rest = bytes.get(i + 1..).unwrap_or(&[]);
    let rest = rest.trim_ascii_start();
    if rest.first() == Some(&b'(') {
        return true; // `t(col) (SELECT ...)`
    }
    if rest.len() >= 3 && rest[..3].eq_ignore_ascii_case(b"max") {
        return true; // `r(level) MAX RECURSION LEVEL 5 AS (`
    }
    match (rest.get(..2), rest.get(2..)) {
        (Some(w), Some(tail)) if w.eq_ignore_ascii_case(b"as") => {
            tail.trim_ascii_start().first() == Some(&b'(')
        }
        _ => false,
    }
}

/// Reports the function calls in SQL text; `first_line` numbers its first line.
pub(crate) fn scan_sql(file: &str, text: &str, first_line: usize) -> Vec<Finding> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut line = first_line;
    let mut prev_word = String::new();
    let mut prev2_word = String::new();
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'\n' => {
                line += 1;
                i += 1;
            }
            b'-' if bytes.get(i + 1) == Some(&b'-') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    if bytes[i] == b'\n' {
                        line += 1;
                    }
                    i += 1;
                }
                i = (i + 2).min(bytes.len());
            }
            b'\'' | b'"' | b'`' => {
                i += 1;
                while i < bytes.len() && bytes[i] != c {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    if bytes.get(i) == Some(&b'\n') {
                        line += 1;
                    }
                    i += 1;
                }
                i += 1;
                prev_word.clear();
            }
            _ if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = text[start..i].to_ascii_lowercase();
                let mut j = i;
                while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                    j += 1;
                }
                let dotted = start > 0 && bytes[start - 1] == b'.';
                let is_call = bytes.get(j) == Some(&b'(')
                    && !dotted
                    && !SQL_WORDS.contains(&word.as_str())
                    && match prev_word.as_str() {
                        // `AS v(a, b)` alias list, `CAST(x AS DECIMAL(10, 2))` type, CTE
                        // names, `SHOW TBLPROPERTIES t(..)`, and DDL / DML targets such
                        // as `INSERT INTO t(a, b)`.
                        // `LATERAL VIEW explode(..)` is a generator call, not a view name.
                        "view" if prev2_word == "lateral" => true,
                        "as" | "recursive" | "with" | "tblproperties" | "into" | "table"
                        | "view" | "function" | "exists" => false,
                        // `FROM t(1)` is a table or a table function: only a name the
                        // matrix knows (`FROM sentences(..)`, `IS DISTINCT FROM f(x)`).
                        "from" | "join" => matrix().contains_key(&word),
                        _ => true,
                    }
                    // `) tab(col)`, `x xx(a, b)`: a table alias with a column list.
                    && !bytes[..start]
                        .trim_ascii_end()
                        .last()
                        .is_some_and(|b| *b == b')' || b.is_ascii_digit())
                    && !(!prev_word.is_empty() && !SQL_WORDS.contains(&prev_word.as_str()))
                    && !is_cte_column_list(bytes, j);
                if let Some(f) = is_call.then(|| judge(file, line, &word)).flatten() {
                    out.push(f);
                }
                prev2_word = std::mem::replace(&mut prev_word, word);
            }
            _ => {
                // After `(` an argument starts: the word before it says nothing about
                // the next one (`upper(sentences(x))`).
                if !c.is_ascii_whitespace() {
                    prev_word.clear();
                }
                i += 1;
            }
        }
    }
    dedup(&mut out);
    out
}

/// A finding for `name` unless the matrix lists it as supported.
fn judge(file: &str, line: usize, name: &str) -> Option<Finding> {
    let status = match matrix().get(name) {
        Some(Status::Supported) => return None,
        Some(s) => *s,
        None => Status::Unknown,
    };
    Some(Finding {
        file: file.to_string(),
        line,
        name: name.to_string(),
        status,
        cell: None,
    })
}

/// A Python string literal: where it starts (0-based line), its text and whether
/// the code around it says it is SQL.
struct Literal {
    line: usize,
    text: String,
    sql: bool,
}

/// Prefix letters of a Python string literal (`f"..."`, `rb'...'`).
fn is_str_prefix(s: &str) -> bool {
    !s.is_empty() && s.len() <= 2 && s.chars().all(|c| "rbfuRBFU".contains(c))
}

/// Calls whose literal argument is SQL (`spark.sql("...")`, `df.selectExpr(..)`,
/// `F.expr(..)`), and those where only the first argument is (`df.filter("...")`).
fn sql_signal(callee: &str, prev: Option<char>) -> bool {
    match callee.to_ascii_lowercase().as_str() {
        "sql" | "expr" | "selectexpr" => true,
        "filter" | "where" => prev == Some('('),
        _ => false,
    }
}

/// A literal that starts like a full SQL statement, in upper case as embedded SQL is
/// written (prose such as "SELECT the rows" or "CREATE a table" does not qualify).
fn starts_like_statement(s: &str) -> bool {
    let t = s.trim_start();
    if t.starts_with("SELECT ") {
        return t.contains(" FROM ");
    }
    if t.starts_with("WITH ") {
        return t.contains(" AS (") && t.contains("SELECT ");
    }
    [
        "INSERT INTO ",
        "INSERT OVERWRITE ",
        "CREATE TABLE ",
        "CREATE OR REPLACE ",
        "CREATE VIEW ",
        "CREATE TEMP",
        "CREATE FUNCTION ",
        "MERGE INTO ",
        "ALTER TABLE ",
    ]
    .iter()
    .any(|k| t.starts_with(k))
}

/// An f-string's text with each `{expr}` field replaced by a placeholder word
/// (`{{` and `}}` are literal braces), so a call inside a field is not read as SQL.
fn blank_fstring_fields(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if depth == 0 && chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if depth == 0 && chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                if depth == 0 {
                    out.push('x');
                }
                depth += 1;
            }
            '}' if depth > 0 => depth -= 1,
            '\n' => out.push('\n'),
            _ if depth > 0 => {}
            _ => out.push(c),
        }
    }
    out
}

/// Python source split into its code (comments removed, string bodies blanked,
/// newlines kept) and its string literals.
fn lex_python(text: &str) -> (String, Vec<Literal>) {
    let b: Vec<char> = text.chars().collect();
    let mut code = String::new();
    let mut lits = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let (mut i, mut line) = (0, 0);
    while i < b.len() {
        match b[i] {
            '\n' => {
                line += 1;
                code.push('\n');
                i += 1;
            }
            '#' => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
            }
            '(' => {
                let ident: String = code
                    .chars()
                    .rev()
                    .skip_while(|c| *c == ' ' || *c == '\t')
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                stack.push(ident);
                code.push('(');
                i += 1;
            }
            ')' => {
                stack.pop();
                code.push(')');
                i += 1;
            }
            q @ ('\'' | '"') => {
                let run: String = code
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                let run: String = run.chars().rev().collect();
                let is_f = run.contains(['f', 'F']);
                if is_str_prefix(&run) {
                    code.truncate(code.len() - run.len());
                }
                let prev = code.trim_end().chars().next_back();
                let triple = b.get(i + 1) == Some(&q) && b.get(i + 2) == Some(&q);
                let n = if triple { 3 } else { 1 };
                i += n;
                let start_line = line;
                let mut s = String::new();
                code.push(q);
                while i < b.len() {
                    if b[i] == '\\' {
                        if b.get(i + 1) == Some(&'\n') {
                            s.push('\n');
                            code.push('\n');
                            line += 1;
                        } else {
                            s.push(' ');
                            code.push(' ');
                        }
                        i += 2;
                        continue;
                    }
                    if triple {
                        if b[i] == q && b.get(i + 1) == Some(&q) && b.get(i + 2) == Some(&q) {
                            break;
                        }
                    } else if b[i] == q || b[i] == '\n' {
                        break;
                    }
                    if b[i] == '\n' {
                        line += 1;
                        code.push('\n');
                    } else {
                        code.push(' ');
                    }
                    s.push(b[i]);
                    i += 1;
                }
                code.push(q);
                i = (i + n).min(b.len());
                if is_f {
                    s = blank_fstring_fields(&s);
                }
                let sql =
                    stack.last().is_some_and(|c| sql_signal(c, prev)) || starts_like_statement(&s);
                lits.push(Literal {
                    line: start_line,
                    text: s,
                    sql,
                });
            }
            c => {
                code.push(c);
                i += 1;
            }
        }
    }
    (code, lits)
}

/// How `pyspark.sql.functions` is reachable in a Python file or notebook.
#[derive(Default)]
pub(crate) struct Imports {
    /// Module names and aliases: `F`, `sf`, `functions`, `pyspark.sql.functions`.
    modules: BTreeSet<String>,
    /// Directly imported functions: local name to function name.
    direct: BTreeMap<String, String>,
    /// Whether the code imports `pyspark`, so its SQL strings are Spark SQL.
    pyspark: bool,
}

/// Splits an import list (`a, b as c, (d)`) into (name, local) pairs.
fn import_items(list: &str) -> Vec<(String, String)> {
    list.replace(['(', ')'], " ")
        .split(',')
        .filter_map(|part| {
            let mut w = part.split_whitespace();
            let name = w.next()?.to_string();
            let local = match (w.next(), w.next()) {
                (Some("as"), Some(l)) => l.to_string(),
                _ => name.clone(),
            };
            Some((name, local))
        })
        .collect()
}

impl Imports {
    /// Reads the `pyspark` function imports of `code`; returns the lines they sit on.
    fn read(&mut self, code: &str) -> BTreeSet<usize> {
        const FN_MODS: [&str; 2] = ["pyspark.sql.functions", "pyspark.sql.connect.functions"];
        const SQL_MODS: [&str; 2] = ["pyspark.sql", "pyspark.sql.connect"];
        let lines: Vec<&str> = code.lines().collect();
        let mut import_lines = BTreeSet::new();
        let mut n = 0;
        while n < lines.len() {
            let t = lines[n].trim();
            let first = n;
            let is_import = t.starts_with("import ") || t.starts_with("from ");
            let mut stmt = t.to_string();
            if is_import && stmt.contains('(') && !stmt.contains(')') {
                while n + 1 < lines.len() && !stmt.contains(')') {
                    n += 1;
                    stmt.push(' ');
                    stmt.push_str(lines[n].trim());
                }
            }
            if is_import {
                import_lines.extend(first..=n);
            }
            if is_import && stmt.contains("pyspark") {
                self.pyspark = true;
            }
            if let Some(rest) = stmt.strip_prefix("import ") {
                for (name, local) in import_items(rest) {
                    if FN_MODS.contains(&name.as_str()) {
                        self.modules.insert(local);
                    }
                }
            } else if let Some(rest) = stmt.strip_prefix("from ")
                && let Some((module, list)) = rest.split_once(" import ")
            {
                let module = module.trim();
                if SQL_MODS.contains(&module) {
                    for (name, local) in import_items(list) {
                        if name == "functions" {
                            self.modules.insert(local);
                        }
                    }
                } else if FN_MODS.contains(&module) {
                    for (name, local) in import_items(list) {
                        if name != "*" {
                            self.direct.insert(local, name);
                        }
                    }
                }
            }
            n += 1;
        }
        import_lines
    }
}

/// Source text position helpers for the call scan: an identifier chain `a.b.c`
/// starting at `start` in `chars`; returns the chain and the index after it.
fn chain_at(chars: &[char], start: usize) -> (String, usize) {
    let mut i = start;
    while i < chars.len()
        && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == '.')
    {
        i += 1;
    }
    (chars[start..i].iter().collect(), i)
}

/// Calls into the functions module on one line of code.
fn scan_module_calls(
    file: &str,
    n: usize,
    code_line: &str,
    imports: &Imports,
    out: &mut Vec<Finding>,
) {
    let chars: Vec<char> = code_line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let starts_ident = c.is_ascii_alphabetic() || c == '_';
        let prev_ok = i == 0
            || !(chars[i - 1].is_ascii_alphanumeric()
                || chars[i - 1] == '_'
                || chars[i - 1] == '.');
        if !(starts_ident && prev_ok) {
            i += 1;
            continue;
        }
        let (chain, end) = chain_at(&chars, i);
        let mut j = end;
        while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
            j += 1;
        }
        let is_call = chars.get(j) == Some(&'(');
        let before: String = chars[..i].iter().collect();
        let defining = before.trim_end().ends_with("def") || before.trim_end().ends_with("class");
        let name = match chain.rsplit_once('.') {
            Some((prefix, name)) if imports.modules.contains(prefix) => Some(name.to_string()),
            Some(_) => None,
            None => imports.direct.get(&chain).cloned(),
        };
        if let (true, false, Some(name)) = (is_call, defining, name) {
            let lower = name.to_ascii_lowercase();
            if !PY_ONLY.contains(&lower.as_str()) {
                out.extend(judge(file, n, &lower));
            }
        }
        i = end.max(i + 1);
    }
}

/// Python source: SQL literals and calls into `pyspark.sql.functions`.
pub(crate) fn scan_python(file: &str, text: &str, first_line: usize) -> Vec<Finding> {
    scan_python_with(file, text, first_line, &mut Imports::default())
}

/// [`scan_python`] with import state that carries over (notebook cells).
pub(crate) fn scan_python_with(
    file: &str,
    text: &str,
    first_line: usize,
    imports: &mut Imports,
) -> Vec<Finding> {
    let (code, lits) = lex_python(text);
    let import_lines = imports.read(&code);
    let mut out = Vec::new();
    for (n, l) in code.lines().enumerate() {
        if !import_lines.contains(&n) {
            scan_module_calls(file, first_line + n, l, imports, &mut out);
        }
    }
    // SQL strings are Spark SQL only where the code shows Spark: a pyspark import or a
    // `spark` / `SparkSession` name (`duckdb.sql(...)` is not ours to judge).
    let spark_code = imports.pyspark
        || code
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|w| matches!(w, "spark" | "SparkSession" | "sqlContext"));
    for lit in lits.iter().filter(|l| l.sql && spark_code) {
        out.extend(scan_sql(file, &lit.text, first_line + lit.line));
    }
    out.sort_by(|a, b| (a.line, &a.name).cmp(&(b.line, &b.name)));
    dedup(&mut out);
    out
}

/// File lines of each source line of each notebook cell, read from the raw JSON
/// (`serde_json::Value` keeps no positions). Indexed by cell, then cell line.
fn source_line_maps(raw: &str) -> Vec<Vec<usize>> {
    #[derive(PartialEq, Clone, Copy)]
    enum Key {
        None,
        Cells,
        Source,
    }
    struct Elem {
        line: usize,
        newlines: usize,
        tail_newline: bool,
    }
    fn map_of(elems: &[Elem]) -> Vec<usize> {
        let mut map = Vec::new();
        let mut pending = true;
        for e in elems {
            if pending {
                map.push(e.line);
                pending = false;
            }
            for j in 1..=e.newlines {
                if j == e.newlines && e.tail_newline {
                    pending = true;
                } else {
                    map.push(e.line);
                }
            }
        }
        map
    }
    let b = raw.as_bytes();
    let (mut i, mut line, mut depth) = (0, 1, 0usize);
    let (mut in_cells, mut key) = (false, Key::None);
    let mut elems: Option<Vec<Elem>> = None;
    let mut maps: Vec<Vec<usize>> = Vec::new();
    while i < b.len() {
        match b[i] {
            b'\n' => line += 1,
            b'"' => {
                let (start, line_at) = (i + 1, line);
                let mut e = Elem {
                    line: line_at,
                    newlines: 0,
                    tail_newline: false,
                };
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' {
                        e.tail_newline = b.get(i + 1) == Some(&b'n');
                        e.newlines += usize::from(e.tail_newline);
                        i += 2;
                        continue;
                    }
                    e.tail_newline = false;
                    i += 1;
                }
                let text = raw.get(start..i).unwrap_or("");
                let is_key = b
                    .get(i + 1..)
                    .is_some_and(|r| r.trim_ascii_start().first() == Some(&b':'));
                if is_key {
                    key = match (depth, text) {
                        (1, "cells") => Key::Cells,
                        (3, "source") if in_cells => Key::Source,
                        _ => Key::None,
                    };
                } else if key == Key::Source && depth == 3 {
                    if let Some(m) = maps.last_mut() {
                        *m = map_of(&[e]);
                    }
                    key = Key::None;
                } else if depth == 4
                    && let Some(v) = elems.as_mut()
                {
                    v.push(e);
                }
            }
            b'[' => {
                if key == Key::Cells && depth == 1 {
                    in_cells = true;
                } else if key == Key::Source && depth == 3 {
                    elems = Some(Vec::new());
                }
                key = Key::None;
                depth += 1;
            }
            b'{' => {
                if in_cells && depth == 2 {
                    maps.push(Vec::new());
                }
                key = Key::None;
                depth += 1;
            }
            b']' | b'}' => {
                depth = depth.saturating_sub(1);
                if b[i] == b']'
                    && depth == 3
                    && let (Some(v), Some(m)) = (elems.take(), maps.last_mut())
                {
                    *m = map_of(&v);
                }
                if b[i] == b']' && depth == 1 {
                    in_cells = false;
                }
            }
            _ => {}
        }
        i += 1;
    }
    maps
}

/// Notebook code cells: `%%sql` / `%sql` cells as SQL, the rest as Python (imports
/// carry across cells). Lines are numbered in the `.ipynb` file; each finding
/// carries its cell index.
fn scan_notebook(file: &str, text: &str) -> std::io::Result<Vec<Finding>> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let cells = v
        .get("cells")
        .and_then(Value::as_array)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no `cells` array"))?;
    let maps = source_line_maps(text);
    let mut imports = Imports::default();
    let mut out = Vec::new();
    for (idx, cell) in cells.iter().enumerate() {
        if cell.get("cell_type").and_then(Value::as_str) != Some("code") {
            continue;
        }
        let src = match cell.get("source") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
            _ => continue,
        };
        let mut findings = if let Some(rest) = src
            .strip_prefix("%%sql")
            .or_else(|| src.strip_prefix("%sql"))
        {
            scan_sql(file, rest, 1)
        } else {
            scan_python_with(file, &src, 1, &mut imports)
        };
        let map = maps.get(idx);
        for f in &mut findings {
            f.line = map
                .and_then(|m| m.get(f.line.saturating_sub(1)).or(m.last()))
                .copied()
                .unwrap_or(1);
            f.cell = Some(idx);
        }
        out.extend(findings);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_compat::parse_matrix;

    fn names(f: &[Finding]) -> Vec<(&str, &str)> {
        f.iter()
            .map(|f| (f.name.as_str(), f.status.label()))
            .collect()
    }

    #[test]
    fn supported_functions_and_table_calls_are_not_findings() {
        let f = scan_sql(
            "t.sql",
            "SELECT abs(x), count(*) FROM t(1) WHERE x IN (1, 2)",
            1,
        );
        assert!(f.is_empty(), "{f:?}");
    }

    #[test]
    fn not_yet_and_unknown_names_are_reported_with_their_line() {
        let f = scan_sql(
            "t.sql",
            "SELECT 1,\n  approx_top_k_estimate(s), nope_fn(1)",
            1,
        );
        assert_eq!(
            names(&f),
            [("approx_top_k_estimate", "not yet"), ("nope_fn", "unknown")]
        );
        assert_eq!(f[0].line, 2);
    }

    #[test]
    fn comments_and_strings_hide_calls() {
        let f = scan_sql("t.sql", "-- nope(1)\n/* nope(2) */ SELECT 'nope(3)'", 1);
        assert!(f.is_empty(), "{f:?}");
    }

    #[test]
    fn python_scans_sql_strings_and_functions_calls() {
        let src = "import pyspark.sql.functions as F\ndf.select(F.abs('x'), F.col('c'), F.mystery('x'))\nspark.sql(\"SELECT other(a) FROM t\")\n# F.skipped('x')\n";
        let f = scan_python("a.py", src, 1);
        assert_eq!(names(&f), [("mystery", "unknown"), ("other", "unknown")]);
        assert_eq!((f[0].line, f[1].line), (2, 3));
    }

    #[test]
    fn notebooks_read_code_cells_only() {
        let nb = r#"{"cells":[{"cell_type":"code","source":["%%sql\n","SELECT nopefn(1)"]},
            {"cell_type":"markdown","source":["zzz(2)"]},
            {"cell_type":"code","source":["import pyspark.sql.functions as F\n","F.mystery('a')"]}]}"#;
        let f = scan_notebook("n.ipynb", nb).unwrap();
        assert_eq!(names(&f), [("nopefn", "unknown"), ("mystery", "unknown")]);
        assert!(scan_notebook("n.ipynb", "{").is_err());
    }

    #[test]
    fn casts_types_and_syntax_words_are_not_findings() {
        let sql = "SELECT CAST(a AS DECIMAL(10, 2)), int(b), date('2020-01-01'), TRY_CAST(c AS VARCHAR(10)),\n\
                   CAST(d AS NUMERIC(10,2)) FROM t WHERE x LIKE 'a' AND when_it(1) IS NULL";
        let f = scan_sql("t.sql", sql, 1);
        assert_eq!(names(&f), [("when_it", "unknown")]);
        let f = scan_sql(
            "t.sql",
            "ALTER TABLE s ADD COLUMNS (c INT);\nSELECT IDENTIFIER('t')",
            1,
        );
        assert!(f.is_empty(), "{f:?}");
    }

    #[test]
    fn alias_and_cte_column_lists_are_not_calls() {
        let sql = "WITH totals(k, v) AS (SELECT 1, 2), other AS (SELECT 3)\n\
                   SELECT * FROM VALUES (1, 2) AS v(a, b), totals";
        let f = scan_sql("t.sql", sql, 1);
        assert!(f.is_empty(), "{f:?}");
    }

    #[test]
    fn calls_after_from_are_reported_when_the_matrix_knows_them() {
        let sql = "SELECT * FROM not_a_table(1);\nSELECT * FROM sentences(1);\n\
                   SELECT extract(year FROM xpath_float(a, b)), a IS DISTINCT FROM sentences(b) FROM t(1)";
        let f = scan_sql("t.sql", sql, 1);
        assert_eq!(
            names(&f),
            [
                ("sentences", "not yet"),
                ("xpath_float", "not yet"),
                ("sentences", "not yet")
            ]
        );
    }

    #[test]
    fn python_strings_are_sql_only_with_a_signal() {
        let src = "\"\"\"Reads the batch from the loader and calls prepare(x) with defaults.\"\"\"\n\
                   msg = \"%s(%d)\" % (a, b)\n\
                   print(\"select the rows from the table (see docs)\")\n\
                   df = spark.sql(\"SELECT mystery_fn(a) FROM t\")\n\
                   q = \"\"\"\nSELECT other_fn(a) FROM t\"\"\"\n\
                   d.selectExpr(\"a\", \"third_fn(b)\")\n\
                   d.filter(F.col(\"x\") == \"fourth(1)\")\n";
        let f = scan_python("a.py", src, 1);
        assert_eq!(
            names(&f),
            [
                ("mystery_fn", "unknown"),
                ("other_fn", "unknown"),
                ("third_fn", "unknown")
            ]
        );
        assert_eq!(f[1].line, 6);
    }

    #[test]
    fn functions_module_is_resolved_from_imports() {
        let torch = "import torch.nn.functional as F\nx = F.relu(x)\nfunctions.softmax(x)\n";
        assert!(scan_python("a.py", torch, 1).is_empty());
        let a = "from pyspark.sql import functions as sf\ndf.select(sf.sentences('x'), sf.when(c, 1))\n";
        assert_eq!(
            names(&scan_python("a.py", a, 1)),
            [("sentences", "not yet")]
        );
        let b = "from pyspark.sql.functions import sentences, col as c, xpath_float as xf\n\
                 df.select(sentences('x'), c('a'), xf('y'), other(1))\n";
        assert_eq!(
            names(&scan_python("a.py", b, 1)),
            [("sentences", "not yet"), ("xpath_float", "not yet")]
        );
        let c = "import pyspark.sql.functions\npyspark.sql.functions.sentences('x')\n";
        assert_eq!(
            names(&scan_python("a.py", c, 1)),
            [("sentences", "not yet")]
        );
        let d = "from pyspark.sql.functions import (\n    sentences,\n    col,\n)\nsentences(1)\n";
        assert_eq!(
            names(&scan_python("a.py", d, 1)),
            [("sentences", "not yet")]
        );
    }

    #[test]
    fn hash_inside_a_string_does_not_hide_a_call() {
        let src = "import pyspark.sql.functions as F\ndf.select(F.lit(\"#\"), F.sentences(\"x\")) # F.xpath_float(1)\n";
        assert_eq!(
            names(&scan_python("a.py", src, 1)),
            [("sentences", "not yet")]
        );
    }

    #[test]
    fn pyspark_only_names_are_not_unknown() {
        let src = "import pyspark.sql.functions as F\nF.countDistinct('a'); F.count_distinct('a'); F.sum_distinct('a'); F.create_map('a'); F.mystery('a')\n";
        assert_eq!(
            names(&scan_python("a.py", src, 1)),
            [("mystery", "unknown")]
        );
    }

    #[test]
    fn backslash_continued_strings_keep_lines() {
        let src = "a = \"x\\\ny\"\nb = \"p\\\nq\"\nspark.sql(\"SELECT zz_fn(1)\")\n";
        let f = scan_python("a.py", src, 1);
        assert_eq!((f[0].name.as_str(), f[0].line), ("zz_fn", 5));
    }

    #[test]
    fn same_line_repeats_are_one_finding_in_sql_and_python() {
        let f = scan_sql("t.sql", "SELECT nopefn(1), nopefn(2)", 1);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn notebook_lines_are_file_lines_and_carry_the_cell() {
        let nb = "{\n \"cells\": [\n  {\n   \"cell_type\": \"code\",\n   \"source\": [\n    \"import pyspark.sql.functions as F\\n\",\n    \"x = 1\\n\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"source\": [\n    \"y = 2\\n\",\n    \"df.select(F.sentences('a'))\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"source\": \"%sql SELECT nopefn(1)\"\n  }\n ]\n}\n";
        let f = scan_notebook("n.ipynb", nb).unwrap();
        assert_eq!(
            f.iter()
                .map(|f| (f.name.as_str(), f.line, f.cell))
                .collect::<Vec<_>>(),
            [("sentences", 14, Some(1)), ("nopefn", 19, Some(2))]
        );
    }

    #[test]
    fn the_embedded_matrix_parses_and_has_no_partial_casts() {
        let m = parse_matrix(include_str!("functions.json")).unwrap();
        assert!(m.len() > 500);
        assert_eq!(m.get("abs"), Some(&Status::Supported));
        assert_eq!(m.get("cast"), Some(&Status::Supported));
        assert!(parse_matrix("{\"a\": \"weird\"}").is_err());
        assert!(parse_matrix("[").is_err());
    }

    #[test]
    fn fstring_fields_are_not_sql_calls() {
        let src =
            "spark.sql(f\"SELECT {fmt(a, b)} AS c, {{literal}}, mystery_fn(1) FROM {tbl(x)}\")\n";
        assert_eq!(
            names(&scan_python("a.py", src, 1)),
            [("mystery_fn", "unknown")]
        );
    }

    #[test]
    fn sql_strings_need_spark_in_the_file() {
        let duck = "import duckdb\nduckdb.sql(\"SELECT mystery_fn(1)\")\nq = \"SELECT other_fn(a) FROM t\"\n";
        assert!(scan_python("a.py", duck, 1).is_empty());
        let prose = "import pyspark\nmsg = \"SELECT the rows from the table (see docs)\"\nmsg2 = \"CREATE a view (now)\"\n";
        assert!(scan_python("a.py", prose, 1).is_empty());
        let stmt = "import pyspark\nq = \"SELECT other_fn(a) FROM t\"\n";
        assert_eq!(
            names(&scan_python("a.py", stmt, 1)),
            [("other_fn", "unknown")]
        );
    }
}
