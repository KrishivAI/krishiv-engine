// SPDX-License-Identifier: Apache-2.0
//! Output renderers: one per `--format` mode behind [`Renderer`].
//!
//! Value text comes from the shared Spark-style renderer in
//! `krishiv-session::agent` (SQL NULL as `NULL`, decimals at full scale,
//! ...); renderers only lay the strings out. `--format table` follows the
//! `spark-sql` CLI layout (bordered table plus `<n> row(s)`); no mode
//! follows `DataFrame.show()` (which pads and truncates differently and
//! prints nested nulls as `null`).
//!
//! Agent output (`--agent`, or auto-detected when stdout is not a TTY
//! unless `--format` is given) lives in `krishiv-session::agent`
//! (`krishiv.agent-result` v1): compact injection-safe tables plus errors
//! as one JSON object per failure.

use std::fmt::Write as _;

/// One output mode: lays already-formatted cell strings out.
pub(crate) trait Renderer {
    /// Short name (`table`, `csv`, ...).
    fn name(&self) -> &'static str;
    /// `rows` (one string per cell, SQL NULL as `NULL`) with `names` as
    /// the header; `nulls` marks the NULL cells (CSV and TSV print them
    /// as empty fields, as Spark writes them).
    fn render(&self, names: &[String], rows: &[Vec<String>], nulls: &[Vec<bool>]) -> String;
}

/// Aligned `spark-sql`-style table with a row count.
pub(crate) struct Table;
/// RFC 4180 CSV with a header row (NULL is an empty field).
pub(crate) struct Csv;
/// Tab-separated values with a header row.
pub(crate) struct Tsv;
/// One JSON object per row.
pub(crate) struct Json;
/// GitHub-flavoured Markdown table.
pub(crate) struct Markdown;
/// `column = value` lines, blank-separated rows (a CLI convenience, not
/// a Spark layout).
pub(crate) struct Line;

impl Renderer for Table {
    fn name(&self) -> &'static str {
        "table"
    }

    fn render(&self, names: &[String], rows: &[Vec<String>], _nulls: &[Vec<bool>]) -> String {
        let mut widths: Vec<usize> = names.iter().map(|n| n.chars().count()).collect();
        for row in rows {
            for (w, c) in widths.iter_mut().zip(row.iter()) {
                *w = (*w).max(c.chars().count());
            }
        }
        let mut rule = String::from("+");
        for w in &widths {
            let _ = write!(rule, "{}+", "-".repeat(w + 2));
        }
        let line = |cells: &[String]| {
            let mut s = String::from("|");
            for (c, w) in cells.iter().zip(&widths) {
                let pad = w.saturating_sub(c.chars().count());
                let _ = write!(s, " {c}{} |", " ".repeat(pad));
            }
            s
        };
        let mut out = String::new();
        let _ = writeln!(out, "{rule}\n{}\n{rule}", line(names));
        for row in rows {
            let _ = writeln!(out, "{}", line(row));
        }
        let _ = writeln!(out, "{rule}");
        let _ = writeln!(
            out,
            "{} row{}",
            rows.len(),
            if rows.len() == 1 { "" } else { "s" }
        );
        out
    }
}

/// One CSV field: quoted when empty or holding a comma, quote or
/// newline (NULL never reaches here: the delimiter renderer prints NULL
/// cells as empty fields).
pub(crate) fn csv_field(s: &str) -> String {
    if s.is_empty() || s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

/// One TSV field: tabs and newlines become spaces (NULL stays empty), so
/// rows stay one line each.
pub(crate) fn tsv_field(s: &str) -> String {
    if s.is_empty() {
        String::new()
    } else {
        s.replace(['\t', '\n', '\r'], " ")
    }
}

impl Renderer for Csv {
    fn name(&self) -> &'static str {
        "csv"
    }

    fn render(&self, names: &[String], rows: &[Vec<String>], nulls: &[Vec<bool>]) -> String {
        render_delimited(names, rows, nulls, ",", csv_field)
    }
}

impl Renderer for Tsv {
    fn name(&self) -> &'static str {
        "tsv"
    }

    fn render(&self, names: &[String], rows: &[Vec<String>], nulls: &[Vec<bool>]) -> String {
        render_delimited(names, rows, nulls, "\t", tsv_field)
    }
}

/// Header plus delimiter-joined rows; NULL cells are empty fields while
/// the empty string quotes (CSV) or is an empty field like NULL (TSV: the two are not told apart), as
/// Spark writes them.
fn render_delimited(
    names: &[String],
    rows: &[Vec<String>],
    nulls: &[Vec<bool>],
    sep: &str,
    field: fn(&str) -> String,
) -> String {
    let mut out = String::new();
    out.push_str(&names.iter().map(|n| field(n)).collect::<Vec<_>>().join(sep));
    out.push('\n');
    for (row, null_row) in rows.iter().zip(nulls.iter()) {
        out.push_str(
            &row.iter()
                .zip(null_row.iter())
                .map(|(c, n)| if *n { String::new() } else { field(c) })
                .collect::<Vec<_>>()
                .join(sep),
        );
        out.push('\n');
    }
    out
}

/// A JSON string literal.
pub(crate) fn json_string(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl Renderer for Json {
    fn name(&self) -> &'static str {
        "json"
    }

    fn render(&self, names: &[String], rows: &[Vec<String>], nulls: &[Vec<bool>]) -> String {
        // Plain strings: NULL cells print as `null`, everything else
        // quoted. The SQL layer calls [`render_json_typed`] instead, which
        // keeps numbers and booleans bare.
        let mut out = String::new();
        for (row, null_row) in rows.iter().zip(nulls.iter()) {
            out.push('{');
            for (i, c) in row.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&json_string(&names[i]));
                out.push(':');
                if null_row[i] {
                    out.push_str("null");
                } else {
                    out.push_str(&json_string(c));
                }
            }
            out.push_str("}\n");
        }
        out
    }
}

/// One JSON object per row with numbers and booleans unquoted, as
/// `krishiv sql -o json` has always printed: `numeric[i]` marks integer
/// and decimal columns, `float[i]` float columns (finite values print
/// bare, NaN and infinities print quoted — JSON has neither), `bool_col[i]`
/// boolean columns. Strings print quoted even when they read `true`.
pub(crate) fn render_json_typed(
    names: &[String],
    numeric: &[bool],
    float: &[bool],
    bool_col: &[bool],
    rows: &[Vec<String>],
    nulls: &[Vec<bool>],
) -> String {
    let mut out = String::new();
    for (row, null_row) in rows.iter().zip(nulls.iter()) {
        out.push('{');
        for (i, c) in row.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&json_string(&names[i]));
            out.push(':');
            if null_row[i] {
                out.push_str("null");
            } else if numeric[i] || bool_col[i] {
                out.push_str(c);
            } else if float[i] {
                if c.parse::<f64>().is_ok_and(f64::is_finite) {
                    out.push_str(c);
                } else {
                    out.push_str(&json_string(c));
                }
            } else {
                out.push_str(&json_string(c));
            }
        }
        out.push_str("}\n");
    }
    out
}

impl Renderer for Markdown {
    fn name(&self) -> &'static str {
        "markdown"
    }

    fn render(&self, names: &[String], rows: &[Vec<String>], _nulls: &[Vec<bool>]) -> String {
        let pipes = |cells: &[String]| {
            cells
                .iter()
                .map(|c| format!(" {} ", c.replace('|', "\\|").replace('\n', "<br>")))
                .collect::<Vec<_>>()
                .join("|")
        };
        let mut out = String::from("|");
        out.push_str(&pipes(names));
        out.push_str("|\n|");
        out.push_str(
            &names
                .iter()
                .map(|_| " --- ".to_owned())
                .collect::<Vec<_>>()
                .join("|"),
        );
        out.push_str("|\n");
        for row in rows {
            out.push('|');
            out.push_str(&pipes(row));
            out.push_str("|\n");
        }
        out
    }
}

impl Renderer for Line {
    fn name(&self) -> &'static str {
        "line"
    }

    fn render(&self, names: &[String], rows: &[Vec<String>], _nulls: &[Vec<bool>]) -> String {
        let mut out = String::new();
        for row in rows {
            for (n, c) in names.iter().zip(row.iter()) {
                let _ = writeln!(out, "{n} = {c}");
            }
            out.push('\n');
        }
        out
    }
}

/// Output modes for result sets (one renderer per mode in this module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputMode {
    /// Aligned `spark-sql`-style table with a row count.
    Table,
    /// RFC 4180 CSV with a header row.
    Csv,
    /// Tab-separated values with a header row.
    Tsv,
    /// One JSON object per row.
    Json,
    /// GitHub-flavoured Markdown table.
    Markdown,
    /// `column = value` lines, blank-separated rows.
    Line,
}

impl OutputMode {
    /// Short name (`table`, `csv`, ...).
    pub(crate) fn name(&self) -> &'static str {
        match self {
            OutputMode::Table => Table.name(),
            OutputMode::Csv => Csv.name(),
            OutputMode::Tsv => Tsv.name(),
            OutputMode::Json => Json.name(),
            OutputMode::Markdown => Markdown.name(),
            OutputMode::Line => Line.name(),
        }
    }
}

/// Renders already-typed rows in `mode`: `numeric[i]` marks integer and
/// decimal columns, `float[i]` float columns, `bool_col[i]` boolean
/// columns, `nulls` the NULL cells.
pub(crate) fn render_mode(
    mode: OutputMode,
    names: &[String],
    rows: &[Vec<String>],
    numeric: &[bool],
    float: &[bool],
    bool_col: &[bool],
    nulls: &[Vec<bool>],
) -> String {
    match mode {
        OutputMode::Table => Table.render(names, rows, nulls),
        OutputMode::Csv => Csv.render(names, rows, nulls),
        OutputMode::Tsv => Tsv.render(names, rows, nulls),
        OutputMode::Json => render_json_typed(names, numeric, float, bool_col, rows, nulls),
        OutputMode::Markdown => Markdown.render(names, rows, nulls),
        OutputMode::Line => Line.render(names, rows, nulls),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["a".to_owned(), "b".to_owned()]
    }

    fn rows() -> Vec<Vec<String>> {
        vec![
            vec!["1".to_owned(), "x".to_owned()],
            vec!["NULL".to_owned(), "2.5".to_owned()],
        ]
    }

    #[test]
    fn every_mode_renders_a_header_and_two_rows() {
        let names = names();
        let rows = rows();
        let nulls = vec![vec![false, false], vec![false, false]];
        let table = Table.render(&names, &rows, &nulls);
        assert!(table.contains("| a    | b   |"), "{table}");
        assert!(table.ends_with("2 rows\n"), "{table}");
        let csv = Csv.render(&names, &rows, &nulls);
        assert_eq!(csv, "a,b\n1,x\nNULL,2.5\n");
        let tsv = Tsv.render(&names, &rows, &nulls);
        assert_eq!(tsv, "a\tb\n1\tx\nNULL\t2.5\n");
        let md = Markdown.render(&names, &rows, &nulls);
        assert!(
            md.starts_with("| a | b |\n| --- | --- |\n| 1 | x |\n"),
            "{md}"
        );
        let line = Line.render(&names, &rows, &nulls);
        assert_eq!(line, "a = 1\nb = x\n\na = NULL\nb = 2.5\n\n");
        let json = Json.render(&names, &rows, &nulls);
        assert_eq!(
            json,
            "{\"a\":\"1\",\"b\":\"x\"}\n{\"a\":\"NULL\",\"b\":\"2.5\"}\n"
        );
    }

    #[test]
    fn nulls_are_empty_fields_in_csv_and_tsv() {
        let names = names();
        let rows = vec![vec!["".to_owned(), "NULL".to_owned()]];
        let nulls = vec![vec![false, true]];
        // Empty string quotes, NULL stays empty: Spark's distinction.
        assert_eq!(Csv.render(&names, &rows, &nulls), "a,b\n\"\",\n");
        assert_eq!(Tsv.render(&names, &rows, &nulls), "a\tb\n\t\n");
        assert_eq!(
            Json.render(&names, &rows, &nulls),
            "{\"a\":\"\",\"b\":null}\n"
        );
    }

    #[test]
    fn fields_escape_per_mode() {
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        // The empty string quotes; NULL never reaches `csv_field` (the
        // delimiter renderer prints NULL cells as empty fields).
        assert_eq!(csv_field(""), "\"\"");
        assert_eq!(tsv_field("a\tb\nc"), "a b c");
        assert_eq!(tsv_field(""), "");
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        let md = Markdown.render(&["a".to_owned()], &[vec!["x|y".to_owned()]], &[vec![false]]);
        assert!(md.contains("x\\|y"), "{md}");
    }

    #[test]
    fn typed_json_quotes_only_strings() {
        let names = names();
        let rows = rows();
        let nulls = vec![vec![false, false], vec![true, false]];
        let got = render_json_typed(
            &names,
            &[true, false],
            &[false, false],
            &[false, false],
            &rows,
            &nulls,
        );
        assert_eq!(got, "{\"a\":1,\"b\":\"x\"}\n{\"a\":null,\"b\":\"2.5\"}\n");
        let nan = render_json_typed(
            &names,
            &[false, false],
            &[false, true],
            &[false, false],
            &[vec!["s".to_owned(), "NaN".to_owned()]],
            &[vec![false, false]],
        );
        // A string that reads `true` stays quoted: only boolean columns
        // print bare.
        let strue = render_json_typed(
            &names,
            &[false, false],
            &[false, false],
            &[false, false],
            &[vec!["true".to_owned(), "x".to_owned()]],
            &[vec![false, false]],
        );
        assert_eq!(strue, "{\"a\":\"true\",\"b\":\"x\"}\n");
        assert_eq!(nan, "{\"a\":\"s\",\"b\":\"NaN\"}\n");
    }
}
