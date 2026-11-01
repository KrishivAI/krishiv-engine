// SPDX-License-Identifier: Apache-2.0
//! `IDENTIFIER(expr)` in the name positions of a statement.
//!
//! Spark evaluates the constant string `expr` while it resolves the statement and uses the result as
//! the name. Here every clause is evaluated first, before the statement is parsed, and replaced in the
//! text by the identifier it names. Everything after that sees an ordinary statement: the access check
//! (`authorize_statement`) is made on the real names, which is why this cannot be left to the parser
//! (a dynamic name has no securable until it is evaluated).

use std::borrow::Cow;

use krishiv_common::error::{
    IdentifierTooManyNameParts, KrishivError, NotAConstantStringNotConstant,
    NotAConstantStringNull, NotAConstantStringWrongType,
};
use krishiv_plan::Literal;
use krishiv_sql::ast::{Expr, ExprKind};
use krishiv_sql::lexer::{Sym, TokenKind, tokenize};
use krishiv_sql::{ParserOptions, parse_multipart_identifier, parse_named_expression};

use crate::command::padded_select;
use crate::session::Session;

/// Functions whose value can change between calls: not constant, as Spark's `foldable` sees them.
const NON_DETERMINISTIC: &[&str] = &[
    "rand",
    "random",
    "randn",
    "uuid",
    "shuffle",
    "monotonically_increasing_id",
    "spark_partition_id",
    "input_file_name",
];

impl Session {
    /// `sql` with every `IDENTIFIER(expr)` replaced by the (quoted) identifier `expr` evaluates to.
    /// Text without the clause is returned as it is.
    pub(crate) async fn resolve_identifier_clauses<'a>(
        &self,
        sql: &'a str,
        options: &ParserOptions,
    ) -> Result<Cow<'a, str>, KrishivError> {
        let legacy = options.legacy_identifier_clause;
        // The legacy mode reads `IDENTIFIER(expr)` as a function, which the grammar accepts only
        // where a table or an expression may stand: a statement that does not parse as written
        // (a column list, a rename target) keeps the parser's error.
        if legacy && krishiv_sql::parse_statement(sql, options).is_err() {
            return Ok(Cow::Borrowed(sql));
        }
        let tokens: Vec<_> = tokenize(sql)
            .into_iter()
            .filter(|t| !t.kind.is_trivia())
            .collect();
        // A script is left alone: its clauses may name locals, which only the interpreter can see.
        let word = |k: usize| {
            tokens
                .get(k)
                .and_then(|t| sql.get(t.span.start..t.span.end))
                .unwrap_or_default()
        };
        if word(0).eq_ignore_ascii_case("BEGIN")
            || (tokens
                .get(1)
                .is_some_and(|t| t.kind == TokenKind::Sym(Sym::Colon))
                && word(2).eq_ignore_ascii_case("BEGIN"))
        {
            return Ok(Cow::Borrowed(sql));
        }
        let mut out = String::new();
        let mut copied = 0;
        let mut i = 0;
        let mut in_hint = false;
        while i < tokens.len() {
            match tokens[i].kind {
                TokenKind::HintStart => in_hint = true,
                TokenKind::HintEnd => in_hint = false,
                _ => {}
            }
            let t = tokens[i];
            let is_clause = t.kind == TokenKind::Word
                && sql
                    .get(t.span.start..t.span.end)
                    .is_some_and(|w| w.eq_ignore_ascii_case("IDENTIFIER"))
                && tokens
                    .get(i + 1)
                    .is_some_and(|n| n.kind == TokenKind::Sym(Sym::LParen));
            // `:IDENTIFIER(...)` is a named parameter marker, not a clause.
            let after_colon = i > 0 && tokens[i - 1].kind == TokenKind::Sym(Sym::Colon);
            // Positions whose grammar takes a plain identifier or a keyword, never the clause
            // (hint arguments, `SHOW <scope> FUNCTIONS`, `EXTRACT(<field> FROM ...)`): the parser
            // refuses them, as Spark does.
            let prev = |n: usize| {
                i.checked_sub(n)
                    .map(word)
                    .filter(|w| !w.is_empty())
                    .unwrap_or_default()
            };
            let mut plain_position = in_hint
                || prev(1).eq_ignore_ascii_case("SHOW")
                || (prev(1) == "(" && prev(2).eq_ignore_ascii_case("EXTRACT"));
            // The legacy mode reads `IDENTIFIER(expr)` as a function: it names a whole table or
            // column, never a part of a dotted name, an alias or a `USING` column.
            if legacy {
                plain_position = plain_position
                    || prev(1) == "."
                    || prev(1).eq_ignore_ascii_case("AS")
                    || (prev(1) == "(" && prev(2).eq_ignore_ascii_case("USING"))
                    // A unit is a keyword there: the call stays an unresolved routine.
                    || (prev(1) == "("
                        && ["TIMESTAMPADD", "TIMESTAMPDIFF"]
                            .iter()
                            .any(|f| prev(2).eq_ignore_ascii_case(f)));
            }
            if !is_clause || after_colon || plain_position {
                i += 1;
                continue;
            }
            // The matching `)`: the first token scanned is the `(`, so `depth` never underflows.
            let mut depth = 0_usize;
            let mut close = None;
            for (j, u) in tokens.iter().enumerate().skip(i + 1) {
                match u.kind {
                    TokenKind::Sym(Sym::LParen) => depth += 1,
                    TokenKind::Sym(Sym::RParen) => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(close) = close else {
                // Unbalanced: the parser reports it.
                break;
            };
            // Nothing between the brackets: left to the parser to refuse.
            if close <= i + 2 {
                i = close + 1;
                continue;
            }
            if legacy
                && tokens
                    .get(close + 1)
                    .is_some_and(|n| n.kind == TokenKind::Sym(Sym::Dot))
            {
                i = close + 1;
                continue;
            }
            let last = tokens[close];
            let inner_start = tokens[i + 2].span.start;
            let inner_end = tokens[close - 1].span.end;
            let inner = sql.get(inner_start..inner_end).unwrap_or_default();
            if inner.trim().is_empty() {
                i = close + 1;
                continue;
            }
            let Some((name, parts)) = self.identifier_of(inner, options).await? else {
                i = close + 1;
                continue;
            };
            // An alias or a window name is one part.
            if parts > 1
                && (prev(1).eq_ignore_ascii_case("AS") || prev(1).eq_ignore_ascii_case("OVER"))
            {
                return Err(IdentifierTooManyNameParts {
                    identifier: name,
                    limit: "1".to_owned(),
                }
                .into());
            }
            out.push_str(sql.get(copied..t.span.start).unwrap_or_default());
            out.push_str(&name);
            copied = last.span.end;
            i = close + 1;
        }
        if copied == 0 {
            return Ok(Cow::Borrowed(sql));
        }
        out.push_str(sql.get(copied..).unwrap_or_default());
        Ok(Cow::Owned(out))
    }

    /// The identifier the constant string expression `inner` names, quoted part by part.
    async fn identifier_of(
        &self,
        inner: &str,
        options: &ParserOptions,
    ) -> Result<Option<(String, usize)>, KrishivError> {
        let name = || "IDENTIFIER".to_owned();
        // Not an expression (`CREATE TABLE identifier (a INT)`): a plain name, left to the parser.
        let item = match parse_named_expression(inner, options) {
            Ok(item) if item.alias.is_none() => item,
            _ => return Ok(None),
        };
        if !is_constant(&item.expr) {
            return Err(NotAConstantStringNotConstant {
                name: name(),
                expr: inner.trim().to_owned(),
            }
            .into());
        }
        let redacted = krishiv_sql::redact_sql(inner);
        let select = padded_select("SELECT ", inner, 0, "");
        let span = krishiv_sql::ast::Span::new(0, inner.len());
        let (data_type, value) = self
            .eval_variable_value(&select, &redacted, span, span)
            .await?;
        let text = match value {
            Literal::String(s) => s,
            Literal::Null(_) => {
                return Err(NotAConstantStringNull {
                    name: name(),
                    expr: inner.trim().to_owned(),
                }
                .into());
            }
            _ => {
                return Err(NotAConstantStringWrongType {
                    name: name(),
                    expr: inner.trim().to_owned(),
                    data_type: data_type.simple_string(),
                }
                .into());
            }
        };
        let parsed = parse_multipart_identifier(&text, options).map_err(|e| e.error)?;
        let name = parsed
            .parts
            .iter()
            .map(|p| format!("`{}`", p.value.replace('`', "``")))
            .collect::<Vec<_>>()
            .join(".");
        Ok(Some((name, parsed.parts.len())))
    }
}

/// Whether the expression is constant for `IDENTIFIER`: no column, subquery or non-deterministic
/// function (a session variable is a bare name, so a bare name is allowed; evaluating it decides).
fn is_constant(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Subquery(_) | ExprKind::Exists(_) | ExprKind::InSubquery { .. } => return false,
        ExprKind::Function(f) => {
            let non_det = f.name.parts.last().is_some_and(|p| {
                NON_DETERMINISTIC
                    .iter()
                    .any(|n| p.value.eq_ignore_ascii_case(n))
            });
            if non_det {
                return false;
            }
        }
        _ => {}
    }
    let mut constant = true;
    e.for_each_child(&mut |c| constant = constant && is_constant(c));
    constant
}
