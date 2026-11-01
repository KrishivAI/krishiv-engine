// SPDX-License-Identifier: Apache-2.0
//! Checks of the spans of a `krishiv-sql` syntax tree, shared by the
//! corpus test and the parser fuzz target.
//!
//! [`check_spans`] walks the tree's `Debug` rendering (so it covers every
//! node type without a visitor): every struct node must carry a span; every
//! span must be non-empty, start and end on token boundaries and lie within
//! its parent's span; and the source text of every expression, parsed on
//! its own, must give the same expression. [`respaced`] supports the
//! differential check that the same tokens with other whitespace and
//! comments parse to an `==` tree.

use krishiv_sql::ast::{ExprKind, Query, SetExprKind, Span, Statement};
use krishiv_sql::lexer::tokenize;
use krishiv_sql::{ParserOptions, parse_query};

/// Enum struct variants: written `Name { .. }` in `Debug`, but not nodes.
const SPANLESS: &[&str] = &[
    "SetOp",
    "Wildcard",
    "IsNull",
    "IsTruth",
    "IsDistinctFrom",
    "Between",
    "InList",
    "InSubquery",
    "Like",
    "LikeQuantified",
    "RLike",
    "Cast",
    "Case",
    "Lambda",
    "Number",
    "Typed",
    "Primitive",
    "Percent",
    "Bucket",
    "DynamicTable",
    "Table",
    "TableFunction",
];

/// A `Debug` range `a..b` (a trailing comma allowed).
fn range(r: &str) -> Option<(usize, usize)> {
    let (a, b) = r.trim_end_matches(',').split_once("..")?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// The span of a predicate's operator, from its `op: a..b` line (a set
/// operation's `op` names the operation instead).
fn op_span(line: &str) -> Option<(usize, usize)> {
    line.strip_prefix("op: ").and_then(range)
}

struct Frame {
    name: String,
    first_line: usize,
    span: Option<(usize, usize)>,
    children: Vec<(usize, usize)>,
}

/// An optional span, written over three lines: `order_span: Some(`, the
/// range, `)`.
const OPTIONAL_SPAN: &str = "order_span: Some(";

/// The node's `Debug` lines without spans (a predicate's `op` span and a
/// query's `order_span` included), trimmed, as a comparable text.
fn without_spans<'a>(lines: impl Iterator<Item = &'a str>) -> String {
    let mut out = String::new();
    let mut ordinal = false;
    let mut skip = 0;
    for line in lines.map(|l| l.trim().trim_end_matches(',')) {
        if skip > 0 || line.starts_with(OPTIONAL_SPAN) {
            skip = if skip > 0 { skip - 1 } else { 2 };
            continue;
        }
        if line.starts_with("span: ") || line == "order_span: None" || op_span(line).is_some() {
            continue;
        }
        let line = line.rsplit_once(": Expr {").map_or(line, |_| "Expr {");
        // A `?` marker's ordinal is its position in the whole statement,
        // so an expression parsed alone numbers it from 0.
        out.push_str(if ordinal { "_" } else { line });
        ordinal = line.ends_with("Parameter(");
        out.push('\n');
    }
    out
}

/// The expression of `SELECT <text>` and, as the select list may read a
/// leading `ALL` differently, the one inside `SELECT (<text>)`, rendered
/// like [`without_spans`].
fn reparsed(text: &str, options: &ParserOptions) -> Vec<Option<String>> {
    let parse = |sql: String, nested: bool, options: &ParserOptions| -> Option<String> {
        let q = parse_query(&sql, options).ok()?;
        let SetExprKind::Select(s) = q.body.kind else {
            return None;
        };
        let item = s.projection.first().filter(|i| i.alias.is_none())?;
        let expr = match &item.expr.kind {
            ExprKind::Nested(inner) if nested => &**inner,
            _ if nested => return None,
            _ => &item.expr,
        };
        Some(without_spans(format!("{expr:#?}").lines()))
    };
    let mut out = vec![
        parse(format!("SELECT {text}"), false, options),
        parse(format!("SELECT ({text})"), true, options),
    ];
    // A backquoted name outside a select item is a column even where the select list would read
    // it as a regex (`spark.sql.parser.quotedRegexColumnNames`).
    if options.quoted_regex_column_names {
        let plain = ParserOptions {
            quoted_regex_column_names: false,
            ..*options
        };
        out.push(parse(format!("SELECT {text}"), false, &plain));
        out.push(parse(format!("SELECT ({text})"), true, &plain));
    }
    out
}

/// Checks the spans of `q`, parsed from `sql` with `options` (see the
/// module docs).
///
/// # Errors
/// A description of the first span that is wrong.
pub fn check_spans(sql: &str, q: &Query, options: &ParserOptions) -> Result<(), String> {
    check_tree(sql, &format!("{q:#?}"), q.span, options, "query")
}

/// Checks the spans of `s`, parsed from `sql` with `options`: the same
/// checks as [`check_spans`], for a whole statement. Expressions inside
/// the statement re-parse on their own as for queries.
///
/// # Errors
/// A description of the first span that is wrong.
pub fn check_statement_spans(
    sql: &str,
    s: &Statement,
    options: &ParserOptions,
) -> Result<(), String> {
    check_tree(sql, &format!("{s:#?}"), s.span(), options, "statement")
}

fn check_tree(
    sql: &str,
    debug: &str,
    root: Span,
    options: &ParserOptions,
    what: &str,
) -> Result<(), String> {
    let tokens = tokenize(sql);
    let starts: Vec<usize> = tokens.iter().map(|t| t.span.start).collect();
    let ends: Vec<usize> = tokens.iter().map(|t| t.span.end).collect();
    let lines: Vec<&str> = debug.lines().collect();
    let mut stack = vec![Frame {
        name: "root".into(),
        first_line: 0,
        span: None,
        children: vec![],
    }];
    for (i, line) in lines.iter().map(|l| l.trim()).enumerate() {
        if let Some(r) = line.strip_prefix("span: ") {
            let span = range(r).ok_or("bad span")?;
            if let Some(top) = stack.last_mut() {
                top.span = Some(span);
            }
        } else if let Some((a, b)) = op_span(line).or_else(|| {
            // The range inside `order_span: Some(..)`.
            let after = i
                .checked_sub(1)
                .is_some_and(|p| lines[p].trim() == OPTIONAL_SPAN);
            after.then(|| range(line)).flatten()
        }) {
            // An operator's or a clause's span: on tokens, and inside its
            // node.
            if !(a < b && starts.contains(&a) && ends.contains(&b)) {
                return Err(format!("operator span {a}..{b} not on tokens"));
            }
            stack.last_mut().ok_or("unbalanced")?.children.push((a, b));
        } else if line.ends_with('{') || line.ends_with('(') || line.ends_with('[') {
            let struct_like = line.ends_with('{');
            let name = line.trim_end_matches(['{', '(', '[']).trim();
            let name = name.rsplit(": ").next().unwrap_or(name).to_owned();
            stack.push(Frame {
                name: if struct_like { name } else { String::new() },
                first_line: i,
                span: None,
                children: vec![],
            });
        } else if line.starts_with(['}', ')', ']']) {
            let f = stack.pop().ok_or("unbalanced")?;
            let parent = stack.last_mut().ok_or("unbalanced")?;
            match f.span {
                Some((a, b)) => {
                    if !(a < b && starts.contains(&a) && ends.contains(&b)) {
                        return Err(format!("{} span {a}..{b} not on tokens", f.name));
                    }
                    if let Some(c) = f.children.iter().find(|(ca, cb)| *ca < a || *cb > b) {
                        return Err(format!(
                            "{} span {a}..{b} does not cover child {c:?}",
                            f.name
                        ));
                    }
                    // A keyword argument and a `TABLE(...)` argument are no
                    // select-list expressions on their own.
                    let own = lines[f.first_line + 1];
                    if f.name == "Expr"
                        && !own.contains("kind: Keyword(")
                        && !own.contains("kind: TableArgument(")
                    {
                        let text = sql.get(a..b).ok_or("span out of range")?;
                        let node = without_spans(lines[f.first_line..=i].iter().copied());
                        if !reparsed(text, options).contains(&Some(node)) {
                            return Err(format!("expression `{text}` does not parse to its node"));
                        }
                    }
                    parent.children.push((a, b));
                }
                None if !f.name.is_empty() && !SPANLESS.contains(&f.name.as_str()) => {
                    return Err(format!("struct {} has no span", f.name));
                }
                None => parent.children.extend(f.children),
            }
        }
    }
    let first = tokens.iter().find(|t| !t.kind.is_trivia());
    if first.map(|t| t.span.start) != Some(root.start) || !ends.contains(&root.end) {
        return Err(format!(
            "{what} span {}..{} does not start at the first token",
            root.start, root.end
        ));
    }
    Ok(())
}

/// `sql`'s tokens (comments dropped) joined by a comment and a newline.
#[must_use]
pub fn respaced(sql: &str) -> String {
    let parts: Vec<&str> = tokenize(sql)
        .iter()
        .filter(|t| !t.kind.is_trivia())
        .map(|t| t.span.text(sql))
        .collect();
    parts.join(" /* x */\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_accepts_good_and_rejects_bad_spans() {
        let d = ParserOptions::default();
        let sql = "SELECT a + 1 AS x, f(b) FROM t WHERE c IN ((SELECT 1), 2)";
        let mut q = parse_query(sql, &d).unwrap();
        check_spans(sql, &q, &d).unwrap();
        q.span.end = 3;
        assert!(check_spans(sql, &q, &d).is_err());
        assert_eq!(respaced("SELECT -- c\n1"), "SELECT /* x */\n1");
        let between = "SELECT a FROM t WHERE a NOT BETWEEN 1 AND 2";
        let mut q = parse_query(between, &d).unwrap();
        check_spans(between, &q, &d).unwrap();
        let SetExprKind::Select(s) = &mut q.body.kind else {
            unreachable!()
        };
        let ExprKind::Between { op, .. } = &mut s.selection.as_mut().unwrap().kind else {
            unreachable!()
        };
        op.end = op.start + 1;
        assert!(check_spans(between, &q, &d).is_err());
        let ordered = "SELECT a FROM t ORDER BY a";
        let mut q = parse_query(ordered, &d).unwrap();
        check_spans(ordered, &q, &d).unwrap();
        let span = q.order_span.as_mut().unwrap();
        span.end = span.start + 1;
        assert!(check_spans(ordered, &q, &d).is_err());
        let params = "SELECT ? - ?";
        check_spans(params, &parse_query(params, &d).unwrap(), &d).unwrap();
    }
}
