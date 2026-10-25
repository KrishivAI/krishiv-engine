// SPDX-License-Identifier: Apache-2.0
//! Plain-text rendering of an error with its SQL query context: the message
//! followed by one `== SQL (line, position) ==` block per SQL context, the
//! source line and a caret under the span.
//!
//! The attached spans follow the Spark 4.2.0 goldens (see
//! `crates/krishiv-session/tests/error_context.rs`); the summary text itself
//! is engine-defined and not verified against a live Spark run.

use super::{ErrorContextType, ErrorQueryContext, KrishivError};

/// How many characters of the line around the span are shown on each side.
const MAX_EXTRA: usize = 32;
const ELLIPSIS: &str = "...";

/// `error`'s message (its `Display`), followed for each SQL query context
/// by Spark's summary of it:
///
/// ```text
/// [UNRESOLVED_COLUMN.WITH_SUGGESTION] A column ... SQLSTATE: 42703
/// == SQL (line 1, position 8) ==
/// SELECT x FROM (SELECT 1 AS a) t
///        ^
/// ```
///
/// The header gives the line and the 1-based character position of the
/// span's start. Each line of the span is shown with a caret line under the
/// covered characters; tabs before the span are kept in the caret line so
/// it lines up. At most 32 characters of the line before and after the span
/// are shown, with `...` for what is cut. The text has no trailing newline
/// and no colour.
///
/// `sql` is the text the context's byte offsets refer to, shown as it is, so
/// pass the redacted text (`RedactedSql::as_str`) when it may hold secrets.
/// A context is rendered only when `sql` holds exactly its (redacted)
/// fragment at `start..=stop`; so a secret inside the span never reaches
/// the output, and neither does text from the wrong place when redaction
/// has moved the offsets. Otherwise (a span outside the text, an empty or
/// reversed span, a DataFrame context) only the message is given for that
/// context. Never panics; the result depends only on the arguments.
///
/// The engine attaches only direct-query contexts (with empty `object_type`
/// and `object_name`); a context naming an object renders an
/// `== SQL of TYPE NAME ...` header.
///
/// `Display` and the message parameters of `error` are not changed.
#[must_use]
pub fn render_with_context(error: &KrishivError, sql: &str) -> String {
    let mut out = error.to_string();
    for context in error.query_contexts() {
        if let Some(summary) = summary(context, sql) {
            out.push('\n');
            out.push_str(&summary);
        }
    }
    out
}

/// The summary of one SQL context, without a trailing newline; `None` when
/// there is nothing valid to show.
fn summary(context: &ErrorQueryContext, sql: &str) -> Option<String> {
    if context.context_type != ErrorContextType::Sql || context.start > context.stop {
        return None;
    }
    let end = context.stop.checked_add(1)?;
    let span = context.fragment.as_str();
    if sql.get(context.start..end) != Some(span) {
        return None;
    }

    let before = &sql[..context.start];
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    let line_no = before.bytes().filter(|b| *b == b'\n').count() + 1;
    let prefix = &sql[line_start..context.start];
    let position = prefix.chars().count() + 1;
    // A span ending at a line break takes nothing after it.
    let suffix = if span.ends_with('\n') {
        ""
    } else {
        let rest = &sql[end..];
        &rest[..rest.find('\n').unwrap_or(rest.len())]
    };

    let (prefix, cut_before) = last_chars(prefix, MAX_EXTRA);
    let (suffix, cut_after) = first_chars(suffix, MAX_EXTRA);
    let lead = if cut_before { ELLIPSIS } else { "" };
    let trail = if cut_after { ELLIPSIS } else { "" };
    let marked_from = lead.chars().count() + prefix.chars().count();
    let marked_to = marked_from + span.chars().count();
    let text = format!("{lead}{prefix}{span}{suffix}{trail}");

    // Trailing empty lines are dropped.
    let mut lines: Vec<&str> = text.split('\n').collect();
    while lines.last() == Some(&"") {
        lines.pop();
    }
    if lines.is_empty() {
        return None;
    }

    let object = if context.object_type.is_empty() || context.object_name.is_empty() {
        String::new()
    } else {
        format!(" of {} {}", context.object_type, context.object_name)
    };
    let mut out = format!("== SQL{object} (line {line_no}, position {position}) ==");
    let mut cursor = 0;
    for raw in &lines {
        let shown = raw.strip_suffix('\r').unwrap_or(raw);
        out.push('\n');
        out.push_str(shown);
        out.push('\n');
        for (k, c) in shown.chars().enumerate() {
            let at = cursor + k;
            if at < marked_from {
                out.push(if c == '\t' { '\t' } else { ' ' });
            } else if at < marked_to {
                out.push('^');
            } else {
                break;
            }
        }
        cursor += raw.chars().count() + 1;
    }
    Some(out)
}

/// The last `n` characters of `s`, and whether any were cut.
fn last_chars(s: &str, n: usize) -> (&str, bool) {
    let count = s.chars().count();
    if count <= n {
        return (s, false);
    }
    let from = s.char_indices().nth(count - n).map_or(0, |(i, _)| i);
    (&s[from..], true)
}

/// The first `n` characters of `s`, and whether any were cut.
fn first_chars(s: &str, n: usize) -> (&str, bool) {
    match s.char_indices().nth(n) {
        Some((i, _)) => (&s[..i], true),
        None => (s, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{DivideByZero, UnresolvedColumnWithoutSuggestion};

    fn error() -> KrishivError {
        UnresolvedColumnWithoutSuggestion {
            object_name: "`x`".to_owned(),
        }
        .into()
    }

    /// `error()` with the SQL context `start..=stop` of `sql`.
    fn at(sql: &str, start: usize, stop: usize) -> KrishivError {
        error().with_query_context(ErrorQueryContext::sql(&sql[start..=stop], start, stop))
    }

    /// The summary lines after the message.
    fn rendered(e: &KrishivError, sql: &str) -> String {
        let out = render_with_context(e, sql);
        out.strip_prefix(&e.to_string())
            .unwrap_or_else(|| panic!("message first: {out:?}"))
            .to_owned()
    }

    #[test]
    fn single_line_span_is_marked_by_carets() {
        let sql = "SELECT x FROM t";
        assert_eq!(
            rendered(&at(sql, 7, 7), sql),
            "\n== SQL (line 1, position 8) ==\nSELECT x FROM t\n       ^"
        );
        assert_eq!(
            rendered(&at(sql, 14, 14), sql),
            "\n== SQL (line 1, position 15) ==\nSELECT x FROM t\n              ^"
        );
        assert_eq!(
            rendered(&at(sql, 9, 12), sql),
            "\n== SQL (line 1, position 10) ==\nSELECT x FROM t\n         ^^^^"
        );
    }

    #[test]
    fn span_at_start_and_end_of_text() {
        let sql = "xyz";
        assert_eq!(
            rendered(&at(sql, 0, 2), sql),
            "\n== SQL (line 1, position 1) ==\nxyz\n^^^"
        );
        assert_eq!(
            rendered(&at(sql, 2, 2), sql),
            "\n== SQL (line 1, position 3) ==\nxyz\n  ^"
        );
    }

    #[test]
    fn later_line_has_its_own_line_and_position() {
        let sql = "SELECT 1,\n  nope(2)\nFROM t\n";
        let start = sql.find("nope").unwrap();
        assert_eq!(
            rendered(&at(sql, start, start + 6), sql),
            "\n== SQL (line 2, position 3) ==\n  nope(2)\n  ^^^^^^^"
        );
    }

    #[test]
    fn multi_line_span_marks_each_line_it_covers() {
        // Every line of the span gets its own caret line.
        let sql = "SELECT a +\n  b +\n  c FROM t";
        let (start, stop) = (7, sql.find(" FROM").unwrap() - 1);
        assert_eq!(
            rendered(&at(sql, start, stop), sql),
            "\n== SQL (line 1, position 8) ==\n\
             SELECT a +\n       ^^^\n  b +\n^^^^^\n  c FROM t\n^^^"
        );
    }

    #[test]
    fn positions_count_characters_not_bytes() {
        let sql = "select 'é' + x";
        let start = sql.find('x').unwrap();
        assert_eq!(start, 14);
        assert_eq!(
            rendered(&at(sql, start, start), sql),
            "\n== SQL (line 1, position 14) ==\nselect 'é' + x\n             ^"
        );
        // A span of multi-byte characters: `stop` is the last byte of 'é'.
        let sql = "select éé, 1";
        assert_eq!(
            rendered(&at(sql, 7, 10), sql),
            "\n== SQL (line 1, position 8) ==\nselect éé, 1\n       ^^"
        );
    }

    #[test]
    fn tabs_before_the_span_are_kept_so_the_carets_line_up() {
        let sql = "SELECT\tx\t, y";
        assert_eq!(
            rendered(&at(sql, 11, 11), sql),
            "\n== SQL (line 1, position 12) ==\nSELECT\tx\t, y\n      \t \t  ^"
        );
    }

    #[test]
    fn crlf_line_breaks_do_not_show_the_carriage_return() {
        let sql = "SELECT 1,\r\n  nope\r\nFROM t";
        let start = sql.find("nope").unwrap();
        assert_eq!(
            rendered(&at(sql, start, start + 3), sql),
            "\n== SQL (line 2, position 3) ==\n  nope\n  ^^^^"
        );
    }

    #[test]
    fn a_span_across_crlf_breaks_keeps_the_carets_aligned() {
        // The carriage returns are counted but not shown, so the carets of
        // the lines after the first still sit under their characters.
        let sql = "ab\r\ncd";
        assert_eq!(
            rendered(&at(sql, 0, 5), sql),
            "\n== SQL (line 1, position 1) ==\nab\n^^\ncd\n^^"
        );
    }

    #[test]
    fn long_lines_are_cut_to_32_characters_around_the_span() {
        let sql = format!("{}x{}", "a".repeat(40), "b".repeat(40));
        let out = rendered(&at(&sql, 40, 40), &sql);
        let shown = format!("...{}x{}...", "a".repeat(32), "b".repeat(32));
        assert_eq!(
            out,
            format!(
                "\n== SQL (line 1, position 41) ==\n{shown}\n{}^",
                " ".repeat(35)
            )
        );
        // Nothing is cut at exactly 32 characters.
        let sql = format!("{}x{}", "a".repeat(32), "b".repeat(32));
        let out = rendered(&at(&sql, 32, 32), &sql);
        assert!(out.contains(&format!("\n{sql}\n")), "{out}");
    }

    #[test]
    fn a_span_ending_at_a_line_break_takes_nothing_after_it() {
        let sql = "SELECT 1\n  FROM t";
        assert_eq!(
            rendered(&at(sql, 7, 8), sql),
            "\n== SQL (line 1, position 8) ==\nSELECT 1\n       ^"
        );
    }

    #[test]
    fn cutting_long_lines_keeps_whole_characters() {
        let sql = format!("{}x{}", "é".repeat(40), "ü".repeat(40));
        let out = rendered(&at(&sql, 80, 80), &sql);
        let shown = format!("...{}x{}...", "é".repeat(32), "ü".repeat(32));
        assert!(out.contains(&format!("\n{shown}\n")), "{out}");
        assert!(out.contains("(line 1, position 41)"), "{out}");
    }

    #[test]
    fn a_span_of_line_breaks_alone_has_no_line_to_show() {
        let sql = "\n\nabc";
        let e = error().with_query_context(ErrorQueryContext::sql("\n\n", 0, 1));
        assert_eq!(render_with_context(&e, sql), e.to_string());
    }

    #[test]
    fn object_header_names_the_view() {
        let sql = "SELECT x";
        let mut context = ErrorQueryContext::sql("x", 7, 7);
        context.object_type = "VIEW".to_owned();
        context.object_name = "`default`.`v`".to_owned();
        let e = error().with_query_context(context);
        assert!(
            rendered(&e, sql).starts_with("\n== SQL of VIEW `default`.`v` (line 1, position 8) ==")
        );
    }

    #[test]
    fn every_sql_context_is_rendered_in_order() {
        let sql = "SELECT x, y";
        let e = at(sql, 7, 7).with_query_context(ErrorQueryContext::sql("y", 10, 10));
        assert_eq!(
            rendered(&e, sql),
            "\n== SQL (line 1, position 8) ==\nSELECT x, y\n       ^\
             \n== SQL (line 1, position 11) ==\nSELECT x, y\n          ^"
        );
    }

    #[test]
    fn nothing_to_show_renders_the_message_only() {
        let sql = "SELECT x FROM t";
        let message = error().to_string();
        let plain = |e: &KrishivError, sql: &str| render_with_context(e, sql) == message;
        // No context at all.
        assert!(plain(&error(), sql));
        // A DataFrame context.
        assert!(plain(
            &error().with_query_context(ErrorQueryContext::dataframe()),
            sql
        ));
        // Outside the text, and an empty text.
        let outside = error().with_query_context(ErrorQueryContext::sql("x", 40, 40));
        assert!(plain(&outside, sql));
        assert!(plain(&at(sql, 7, 7), ""));
        // Reaching one past the end, and the largest offsets.
        let past = error().with_query_context(ErrorQueryContext::sql("t?", 14, 15));
        assert!(plain(&past, sql));
        let huge = error().with_query_context(ErrorQueryContext::sql("", usize::MAX, usize::MAX));
        assert!(plain(&huge, sql));
        let huge = error().with_query_context(ErrorQueryContext::sql("", 0, usize::MAX));
        assert!(plain(&huge, sql));
        // An empty span (`stop` before `start`), as the session records one.
        let empty = error().with_query_context(ErrorQueryContext::sql("", 7, 6));
        assert!(plain(&empty, sql));
        // Offsets inside a multi-byte character.
        let split = "select é";
        let inside = error().with_query_context(ErrorQueryContext::sql("?", 8, 8));
        assert!(plain(&inside, split));
    }

    #[test]
    fn a_secret_in_the_span_never_reaches_the_output() {
        let raw = "SELECT 1 FROM t WHERE password = 'hunter2' AND y";
        let start = raw.find("'hunter2'").unwrap();
        let stop = start + "'hunter2'".len() - 1;
        // The context holds the redacted fragment, as the session records it.
        let e = error().with_query_context(ErrorQueryContext::sql(
            crate::secret::REDACTED,
            start,
            stop,
        ));
        let out = render_with_context(&e, raw);
        assert!(!out.contains("hunter2"), "{out}");
        assert_eq!(out, e.to_string());
    }

    #[test]
    fn redacted_text_is_shown_as_given() {
        // The redacted text with the secret after the span: offsets still hold.
        let text = format!("SELECT x, '{}' FROM t", crate::secret::REDACTED);
        let out = rendered(&at(&text, 7, 7), &text);
        assert!(out.contains(crate::secret::REDACTED), "{out}");
        assert!(out.ends_with("\n       ^"), "{out}");
        // A secret before the span moved it: the text no longer holds the
        // fragment there, so nothing is shown.
        let raw = "SELECT 'hunter2', x";
        let moved = format!("SELECT '{}', x", crate::secret::REDACTED);
        let start = raw.len() - 1;
        let e = at(raw, start, start);
        assert!(rendered(&e, raw).ends_with("\n                  ^"));
        assert_eq!(render_with_context(&e, &moved), e.to_string());
    }

    #[test]
    fn fragment_must_match_the_text() {
        let e = error().with_query_context(ErrorQueryContext::sql("y", 7, 7));
        assert_eq!(render_with_context(&e, "SELECT x"), e.to_string());
    }

    #[test]
    fn message_and_parameters_are_unchanged() {
        let sql = "SELECT 1/0";
        let e = KrishivError::from(DivideByZero {
            config: "\"spark.sql.ansi.enabled\"".to_owned(),
        })
        .with_query_context(ErrorQueryContext::sql("1/0", 7, 9));
        let out = render_with_context(&e, sql);
        assert!(out.starts_with(&e.to_string()), "{out}");
        assert!(e.to_string().ends_with("SQLSTATE: 22012"));
        assert_eq!(
            e.message_parameters(),
            [("config", "\"spark.sql.ansi.enabled\"".to_owned())]
        );
        assert_eq!(
            out,
            format!("{e}\n== SQL (line 1, position 8) ==\nSELECT 1/0\n       ^^^")
        );
    }

    #[test]
    fn rendering_is_deterministic() {
        let sql = "SELECT 'é',\n\tx";
        let e = at(sql, 14, 14);
        assert_eq!(render_with_context(&e, sql), render_with_context(&e, sql));
    }
}
