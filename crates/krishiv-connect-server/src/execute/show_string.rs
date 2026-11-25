// SPDX-License-Identifier: Apache-2.0
//! `show_string` and `html_string` (Spark's `Dataset.showString` / `htmlString`, v4.3.0
//! `sql/core/.../classic/Dataset.scala`): the ROOT relation a PySpark `df.show()` sends.
//!
//! The input relation runs as the plan of an ordinary query; each column is turned into Spark's
//! pretty string (`ToPrettyString`: `NULL` for a null, the cast to string otherwise, a binary as
//! `[01 02]`), at most `num_rows + 1` rows are collected, and the table is laid out here. The
//! answer is one row of one string column, `show_string` (`html_string`).

use std::fmt::Write as _;
use std::sync::Arc;

use arrow_array::{Array, BinaryArray, LargeBinaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::plan::OpType;
use krishiv_connect::proto::v4_3::relation::RelType;
use krishiv_session::DataFrame;

/// The largest row count Spark clamps to (`ByteArrayMethods.MAX_ROUNDED_ARRAY_LENGTH - 1`).
const MAX_ROWS: i64 = 2_147_483_632 - 1;

/// What a `show_string` / `html_string` ROOT relation asks.
pub(super) struct Request {
    /// The relation to show, as a plan of its own.
    pub(super) plan: pb::Plan,
    /// `num_rows`.
    pub(super) num_rows: i32,
    /// `truncate`: cut cells to this many characters, 0 for no cut.
    pub(super) truncate: i32,
    /// `vertical`; false for `html_string`.
    pub(super) vertical: bool,
    /// `html_string` rather than `show_string`.
    pub(super) html: bool,
}

/// The request of `plan`, when its ROOT relation is a `show_string` or an `html_string`.
pub(super) fn of(plan: &pb::Plan) -> Option<Request> {
    let Some(OpType::Root(relation)) = &plan.op_type else {
        return None;
    };
    let (input, num_rows, truncate, vertical, html) = match relation.rel_type.as_ref()? {
        RelType::ShowString(s) => (
            s.input.as_deref()?,
            s.num_rows,
            s.truncate,
            s.vertical,
            false,
        ),
        RelType::HtmlString(h) => (h.input.as_deref()?, h.num_rows, h.truncate, false, true),
        _ => return None,
    };
    Some(Request {
        plan: pb::Plan {
            op_type: Some(OpType::Root(Box::new(input.clone()))),
        },
        num_rows,
        truncate,
        vertical,
        html,
    })
}

/// The one-column schema of the answer.
pub(super) fn answer_schema(html: bool) -> SchemaRef {
    let name = if html { "html_string" } else { "show_string" };
    Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]))
}

/// The one-row batch holding `text`.
///
/// # Errors
/// Arrow's, when the batch cannot be built.
pub(super) fn answer_batch(html: bool, text: String) -> Result<RecordBatch, KrishivError> {
    RecordBatch::try_new(
        answer_schema(html),
        vec![Arc::new(StringArray::from(vec![Some(text)]))],
    )
    .map_err(|e| {
        InternalError {
            message: format!("building the show_string row: {e}"),
        }
        .into()
    })
}

/// Runs `df` and lays out its first rows as `request` asks.
///
/// # Errors
/// The query's errors.
pub(super) async fn render(
    df: &DataFrame,
    request: &Request,
    max_batch_size: usize,
) -> Result<String, KrishivError> {
    let num_rows = i64::from(request.num_rows).clamp(0, MAX_ROWS);
    let schema = df.schema_boxed().await?;
    let names: Vec<String> = schema.fields().iter().map(|f| escape(f.name())).collect();
    // One extra row tells whether there is more data.
    let limit = i32::try_from(num_rows + 1).unwrap_or(i32::MAX);
    let mut exprs = Vec::new();
    for (i, field) in schema.fields().iter().enumerate() {
        // Columns are bound by position (`c{i}`): result columns may share a name.
        let quoted = format!("`c{i}`");
        exprs.push(match field.data_type() {
            DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_) => format!("{quoted} AS `c{i}`"),
            dt => format!("{} AS `c{i}`", pretty(&quoted, dt, 0)),
        });
    }
    let truncate = usize::try_from(request.truncate).unwrap_or(0);
    // Spark's (maxBatchSize * 0.7) cap bounds what is held; cells are cut as they are read.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Spark's (maxBatchSize * 0.7).toLong on a positive size"
    )]
    let cap = (max_batch_size as f64 * 0.7) as usize;
    let mut held = 0usize;
    let mut rows: Vec<Vec<String>> = Vec::new();
    {
        // An input with no columns still has rows to count.
        let positional = df.to_df((0..schema.fields().len()).map(|i| format!("c{i}")));
        let limited = positional.limit(limit);
        let mut stream = if exprs.is_empty() {
            limited.execute_stream_boxed().await?
        } else {
            limited.select_expr(&exprs)?.execute_stream_boxed().await?
        };
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            for r in 0..batch.num_rows() {
                let mut row = Vec::with_capacity(batch.num_columns());
                // A row and each cell cost their headers even when empty.
                held = held.saturating_add(std::mem::size_of::<Vec<String>>());
                for c in 0..batch.num_columns() {
                    let cell = cut(&escape(&cell_text(batch.column(c).as_ref(), r)?), truncate);
                    held = held
                        .saturating_add(cell.len())
                        .saturating_add(std::mem::size_of::<String>());
                    row.push(cell);
                }
                if held > cap {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!(
                            "a show_string result over the {cap}-byte cap \
                             (0.7 x spark.connect.grpc.arrow.maxBatchSize)"
                        ),
                    }
                    .into());
                }
                rows.push(row);
            }
        }
    }
    rows.truncate(usize::try_from(num_rows + 1).unwrap_or(usize::MAX));
    let n = usize::try_from(num_rows).unwrap_or(usize::MAX);
    Ok(if request.html {
        html_table(&names, &rows, n)
    } else {
        table(&names, &rows, n, truncate, request.vertical)
    })
}

/// A SQL expression for the pretty string of `expr` (`ToPrettyString`): `NULL` for a null at any depth, a
/// binary as `[01 02]`, an array `[a, b]`, a struct `{a, b}`, a map `{k -> v}`, anything else its cast.
fn pretty(expr: &str, dt: &DataType, depth: usize) -> String {
    let body = match dt {
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            let v = format!("x{depth}");
            format!(
                "concat('[', concat_ws(', ', transform({expr}, {v} -> {})), ']')",
                pretty(&v, f.data_type(), depth + 1)
            )
        }
        DataType::Struct(fields) => {
            let cells: Vec<String> = fields
                .iter()
                .map(|f| {
                    let name = f.name().replace('`', "``");
                    pretty(&format!("{expr}.`{name}`"), f.data_type(), depth)
                })
                .collect();
            format!("concat('{{', concat_ws(', ', {}), '}}')", cells.join(", "))
        }
        DataType::Map(entries, _) => {
            let v = format!("x{depth}");
            let (k, val) = match entries.data_type() {
                DataType::Struct(f) if f.len() == 2 => (f[0].data_type(), f[1].data_type()),
                _ => (&DataType::Utf8, &DataType::Utf8),
            };
            format!(
                "concat('{{', concat_ws(', ', transform(map_entries({expr}), {v} -> concat({}, ' -> ', {}))), '}}')",
                pretty(&format!("{v}.`key`"), k, depth + 1),
                pretty(&format!("{v}.`value`"), val, depth + 1)
            )
        }
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => format!(
            "CASE WHEN length({expr}) = 0 THEN '[]' ELSE concat('[', concat_ws(' ', transform(sequence(1, length({expr})), b{depth} -> hex(substring({expr}, b{depth}, 1)))), ']') END"
        ),
        _ => format!("CAST({expr} AS STRING)"),
    };
    format!("CASE WHEN {expr} IS NULL THEN 'NULL' ELSE {body} END")
}

/// The pretty string of one cell of a column the query produced.
fn cell_text(column: &dyn Array, row: usize) -> Result<String, KrishivError> {
    let bytes = |b: &[u8]| {
        let mut s = String::from("[");
        for (i, byte) in b.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            let _ = write!(s, "{byte:02X}");
        }
        s.push(']');
        s
    };
    if column.is_null(row) {
        return Ok("NULL".to_owned());
    }
    if let Some(a) = column.as_any().downcast_ref::<StringArray>() {
        return Ok(a.value(row).to_owned());
    }
    if let Some(a) = column.as_any().downcast_ref::<BinaryArray>() {
        return Ok(bytes(a.value(row)));
    }
    if let Some(a) = column.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(bytes(a.value(row)));
    }
    if let Some(a) = column
        .as_any()
        .downcast_ref::<arrow_array::BinaryViewArray>()
    {
        return Ok(bytes(a.value(row)));
    }
    if let Some(a) = column
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
    {
        return Ok(bytes(a.value(row)));
    }
    if let Some(a) = column
        .as_any()
        .downcast_ref::<arrow_array::LargeStringArray>()
    {
        return Ok(a.value(row).to_owned());
    }
    if let Some(a) = column
        .as_any()
        .downcast_ref::<arrow_array::StringViewArray>()
    {
        return Ok(a.value(row).to_owned());
    }
    Err(InternalError {
        message: format!("a show_string cell of type {}", column.data_type()),
    }
    .into())
}

/// `SchemaUtils.escapeMetaCharacters`.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{c}' => out.push_str("\\f"),
            '\u{8}' => out.push_str("\\b"),
            '\u{b}' => out.push_str("\\v"),
            '\u{7}' => out.push_str("\\a"),
            c => out.push(c),
        }
    }
    out
}

/// The length in UTF-16 units, as Java's `String.length`.
fn java_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// Cuts `s` to `truncate` units when over it: the first `truncate - 3` and `...`, no ellipsis for
/// a width under 4. 0 cuts nothing.
fn cut(s: &str, truncate: usize) -> String {
    if truncate == 0 || java_len(s) <= truncate {
        return s.to_owned();
    }
    let keep = if truncate < 4 { truncate } else { truncate - 3 };
    let units: Vec<u16> = s.encode_utf16().take(keep).collect();
    let mut out = String::from_utf16_lossy(&units);
    if truncate >= 4 {
        out.push_str("...");
    }
    out
}

/// `Utils.stringHalfWidth`: the length plus one per full-width character.
fn half_width(s: &str) -> usize {
    let wide = s
        .chars()
        .filter(|c| {
            matches!(u32::from(*c),
                0x1100..=0x115F | 0x2E80..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF
                | 0xFE10..=0xFE19 | 0xFE30..=0xFE6F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6)
        })
        .count();
    java_len(s) + wide
}

/// Pads `s` with `fill` to `width` half-widths, on the left when `left`.
fn pad(s: &str, width: usize, left: bool, fill: char) -> String {
    let extra = width.saturating_sub(half_width(s));
    let padding: String = std::iter::repeat_n(fill, extra).collect();
    if left {
        format!("{padding}{s}")
    } else {
        format!("{s}{padding}")
    }
}

/// `Dataset.showString` over the header and the rows (at most `num_rows + 1` of them).
fn table(
    names: &[String],
    rows: &[Vec<String>],
    num_rows: usize,
    truncate: usize,
    vertical: bool,
) -> String {
    let more = rows.len() > num_rows;
    let rows = &rows[..rows.len().min(num_rows)];
    let mut sb = String::new();
    if vertical {
        let name_w = names.iter().map(|n| half_width(n)).fold(3, usize::max);
        let data_w = rows
            .iter()
            .flat_map(|r| r.iter().map(|c| half_width(c)))
            .fold(3, usize::max);
        for (i, row) in rows.iter().enumerate() {
            sb.push_str(&pad(
                &format!("-RECORD {i}"),
                name_w + data_w + 5,
                false,
                '-',
            ));
            sb.push('\n');
            for (name, cell) in names.iter().zip(row) {
                let _ = writeln!(
                    sb,
                    " {} | {} ",
                    pad(name, name_w, false, ' '),
                    pad(cell, data_w, false, ' ')
                );
            }
        }
    } else {
        let mut widths: Vec<usize> = vec![3; names.len()];
        for row in std::iter::once(names).chain(rows.iter().map(Vec::as_slice)) {
            for (w, cell) in widths.iter_mut().zip(row) {
                *w = (*w).max(half_width(cell));
            }
        }
        let separator = {
            let mut s = String::from("+");
            for w in &widths {
                s.push_str(&"-".repeat(*w));
                s.push('+');
            }
            s.push('\n');
            s
        };
        let line = |row: &[String]| {
            let mut s = String::from("|");
            for (cell, w) in row.iter().zip(&widths) {
                s.push_str(&pad(cell, *w, truncate > 0, ' '));
                s.push('|');
            }
            s.push('\n');
            s
        };
        sb.push_str(&separator);
        sb.push_str(&line(names));
        sb.push_str(&separator);
        for row in rows {
            sb.push_str(&line(row));
        }
        sb.push_str(&separator);
    }
    if vertical && rows.is_empty() {
        sb.push_str("(0 rows)");
    } else if more {
        let _ = write!(
            sb,
            "only showing top {num_rows} {}",
            if num_rows == 1 { "row" } else { "rows" }
        );
    }
    sb
}

/// `StringEscapeUtils.escapeHtml4` for the characters that matter in text and attributes.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `Dataset.htmlString`.
fn html_table(names: &[String], rows: &[Vec<String>], num_rows: usize) -> String {
    let more = rows.len() > num_rows;
    let rows = &rows[..rows.len().min(num_rows)];
    let mut sb = String::from("<table border='1'>\n<tr>");
    for n in names {
        let _ = write!(sb, "<th>{}</th>", html_escape(n));
    }
    sb.push_str("</tr>\n");
    for row in rows {
        sb.push_str("<tr>");
        for c in row {
            let _ = write!(sb, "<td>{}</td>", html_escape(c));
        }
        sb.push_str("</tr>\n");
    }
    sb.push_str("</table>\n");
    if more {
        let _ = writeln!(
            sb,
            "only showing top {num_rows} {}",
            if num_rows == 1 { "row" } else { "rows" }
        );
    }
    sb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn a_table_is_right_aligned_with_a_minimum_width_of_three() {
        let out = table(
            &s(&["a", "name"]),
            &[s(&["1", "x"]), s(&["22", "yy"])],
            20,
            20,
            false,
        );
        assert_eq!(
            out,
            "+---+----+\n|  a|name|\n+---+----+\n|  1|   x|\n| 22|  yy|\n+---+----+\n"
        );
    }

    #[test]
    fn truncate_zero_left_aligns() {
        let out = table(&s(&["a"]), &[s(&["1"])], 20, 0, false);
        assert_eq!(out, "+---+\n|a  |\n+---+\n|1  |\n+---+\n");
    }

    #[test]
    fn more_rows_than_asked_prints_the_footer() {
        let rows = [s(&["1"]), s(&["2"])];
        assert!(table(&s(&["a"]), &rows, 1, 20, false).ends_with("+---+\nonly showing top 1 row"));
        assert!(table(&s(&["a"]), &rows, 2, 20, false).ends_with("+---+\n"));
    }

    #[test]
    fn vertical_form_and_the_empty_set() {
        let out = table(&s(&["id", "v"]), &[s(&["1", "x"])], 20, 20, true);
        assert_eq!(out, "-RECORD 0--\n id  | 1   \n v   | x   \n");
        assert_eq!(table(&s(&["a"]), &[], 20, 20, true), "(0 rows)");
    }

    #[test]
    fn cells_are_escaped_and_cut_like_spark() {
        assert_eq!(escape("a\nb\tc"), "a\\nb\\tc");
        assert_eq!(cut("abcdefghij", 6), "abc...");
        assert_eq!(cut("abcdefghij", 3), "abc");
        assert_eq!(cut("abc", 3), "abc");
        assert_eq!(cut("abcdefghij", 0), "abcdefghij");
    }

    #[test]
    fn a_wide_character_takes_two_columns() {
        let out = table(&s(&["a"]), &[s(&["日本"])], 20, 20, false);
        assert_eq!(out, "+----+\n|   a|\n+----+\n|日本|\n+----+\n");
    }

    #[test]
    fn html_escapes_and_has_a_footer() {
        let out = html_table(&s(&["a"]), &[s(&["<1>"]), s(&["2"])], 1);
        assert_eq!(
            out,
            "<table border='1'>\n<tr><th>a</th></tr>\n<tr><td>&lt;1&gt;</td></tr>\n</table>\nonly showing top 1 row\n"
        );
    }
}
