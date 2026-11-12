// SPDX-License-Identifier: Apache-2.0
//! The scanners: SQL text, Python source (string literals that read as SQL and
//! `F.name(` calls) and notebook code cells.

use std::path::Path;

use serde_json::{Value, json};

use super::{Status, matrix};

/// One reported call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Finding {
    /// File the call is in.
    pub(crate) file: String,
    /// 1-based line (relative to the cell for notebook code cells).
    pub(crate) line: usize,
    /// The function name, lower case.
    pub(crate) name: String,
    /// `partial`, `not yet` or `unknown`.
    pub(crate) status: &'static str,
}

impl Finding {
    /// The text report line.
    pub(crate) fn text(&self) -> String {
        let what = match self.status {
            "unknown" => "unknown, not supported",
            s => s,
        };
        format!("{}:{}: {}: {what}", self.file, self.line, self.name)
    }

    /// The JSON object of the stable report format.
    pub(crate) fn json(&self) -> Value {
        json!({"file": self.file, "line": self.line, "name": self.name, "status": self.status})
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
    "window",
    "returns",
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
];

/// PySpark names reachable as `F.<name>(` that are not SQL functions.
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
    "input_file_name",
    "spark_partition_id",
    "monotonically_increasing_id",
    "from_json",
    "to_json",
    "transform",
    "udtf",
    "array_sort",
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

/// Reports the function calls in SQL text; `first_line` numbers its first line.
pub(crate) fn scan_sql(file: &str, text: &str, first_line: usize) -> Vec<Finding> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut line = first_line;
    let mut prev_word = String::new();
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
                let call = bytes.get(j) == Some(&b'(')
                    && !dotted
                    && !SQL_WORDS.contains(&word.as_str())
                    && !matches!(
                        prev_word.as_str(),
                        "from" | "join" | "into" | "table" | "view" | "function" | "exists"
                    );
                if let Some(f) = call.then(|| judge(file, line, &word)).flatten() {
                    out.push(f);
                }
                prev_word = word;
            }
            _ => {
                if !c.is_ascii_whitespace() && c != b'(' {
                    prev_word.clear();
                }
                i += 1;
            }
        }
    }
    out
}

/// A finding for `name` unless the matrix lists it as supported.
fn judge(file: &str, line: usize, name: &str) -> Option<Finding> {
    let status = match matrix().get(name) {
        Some(Status::Supported) => return None,
        Some(Status::Partial) => "partial",
        Some(Status::NotYet) => "not yet",
        None => "unknown",
    };
    Some(Finding {
        file: file.to_string(),
        line,
        name: name.to_string(),
        status,
    })
}

/// Python source: string literals that read as SQL, and `F.name(` /
/// `functions.name(` calls.
pub(crate) fn scan_python(file: &str, text: &str, first_line: usize) -> Vec<Finding> {
    let mut out = Vec::new();
    for (n, l) in text.lines().enumerate() {
        let code = l.split('#').next().unwrap_or("");
        for prefix in ["F.", "functions."] {
            let mut rest = code;
            while let Some(p) = rest.find(prefix) {
                let before = rest[..p].chars().next_back();
                let after = &rest[p + prefix.len()..];
                rest = after;
                if before.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
                    continue;
                }
                let name: String = after
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if name.is_empty() || !after[name.len()..].starts_with('(') {
                    continue;
                }
                let lower = name.to_ascii_lowercase();
                if !PY_ONLY.contains(&lower.as_str()) {
                    out.extend(judge(file, first_line + n, &lower));
                }
            }
        }
    }
    for (start_line, lit) in string_literals(text) {
        if reads_as_sql(&lit) {
            out.extend(scan_sql(file, &lit, first_line + start_line));
        }
    }
    out.sort_by(|a, b| (a.line, &a.name).cmp(&(b.line, &b.name)));
    out.dedup();
    out
}

/// A literal that holds a SQL statement or expression with calls.
fn reads_as_sql(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    [
        "select ", "from ", "insert ", "create ", "with ", "over (", "over(",
    ]
    .iter()
    .any(|k| l.contains(k))
        || l.contains('(') && s.len() < 200 && !s.contains(' ')
}

/// Python string literals as (0-based line, content); handles `'`, `"` and
/// triple quotes, and skips comments.
fn string_literals(text: &str) -> Vec<(usize, String)> {
    let b: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let (mut i, mut line) = (0, 0);
    while i < b.len() {
        match b[i] {
            '\n' => {
                line += 1;
                i += 1;
            }
            '#' => {
                while i < b.len() && b[i] != '\n' {
                    i += 1;
                }
            }
            q @ ('\'' | '"') => {
                let triple = b.get(i + 1) == Some(&q) && b.get(i + 2) == Some(&q);
                let n = if triple { 3 } else { 1 };
                i += n;
                let start_line = line;
                let mut s = String::new();
                while i < b.len() {
                    if b[i] == '\\' {
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
                    }
                    s.push(b[i]);
                    i += 1;
                }
                i += n;
                out.push((start_line, s));
            }
            _ => i += 1,
        }
    }
    out
}

/// Notebook code cells: `%%sql` cells as SQL, the rest as Python. Lines are
/// numbered in the `.ipynb` file (the cell's first source line).
fn scan_notebook(file: &str, text: &str) -> std::io::Result<Vec<Finding>> {
    let v: Value = serde_json::from_str(text)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let cells = v
        .get("cells")
        .and_then(Value::as_array)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no `cells` array"))?;
    let mut out = Vec::new();
    for cell in cells {
        if cell.get("cell_type").and_then(Value::as_str) != Some("code") {
            continue;
        }
        let src = match cell.get("source") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
            _ => continue,
        };
        // Cell line numbers are relative to the cell: the notebook JSON does
        // not keep source positions, so the first line of the cell is 1.
        let findings = if let Some(rest) = src.strip_prefix("%%sql") {
            scan_sql(file, rest, 1)
        } else {
            scan_python(file, &src, 1)
        };
        out.extend(findings);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(f: &[Finding]) -> Vec<(&str, &str)> {
        f.iter().map(|f| (f.name.as_str(), f.status)).collect()
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
        let src = "df.select(F.abs('x'), F.col('c'), F.mystery('x'))\nspark.sql(\"SELECT other(a) FROM t\")\n# F.skipped('x')\n";
        let f = scan_python("a.py", src, 1);
        assert_eq!(names(&f), [("mystery", "unknown"), ("other", "unknown")]);
        assert_eq!((f[0].line, f[1].line), (1, 2));
    }

    #[test]
    fn notebooks_read_code_cells_only() {
        let nb = r#"{"cells":[{"cell_type":"code","source":["%%sql\n","SELECT nopefn(1)"]},
            {"cell_type":"markdown","source":["zzz(2)"]},
            {"cell_type":"code","source":"F.mystery('a')"}]}"#;
        let f = scan_notebook("n.ipynb", nb).unwrap();
        assert_eq!(names(&f), [("nopefn", "unknown"), ("mystery", "unknown")]);
        assert!(scan_notebook("n.ipynb", "{").is_err());
    }
}
