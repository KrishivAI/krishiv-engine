// SPDX-License-Identifier: Apache-2.0
//! A parse-only [`Engine`] over `krishiv-sql`, and the DQL scope its parse
//! number is measured on.
//!
//! [`ParseOnlyEngine`] parses each statement under the session's configs
//! (folder defaults, `--SET`, `--CONFIG_DIM`, and `SET k=v` statements run
//! in the file). It executes nothing: a statement that parses fails to
//! execute without an error class, and one that does not parse fails with
//! the parser's error class, so the negative number counts syntax errors
//! matched.
//!
//! [`in_dql_scope`] selects the query statements the DQL parser covers.
//! Excluded are statements that are not queries (DDL, DML, commands) and
//! queries using streaming relations (`STREAM`, `WATERMARK`, `CHANGES`).

use std::collections::HashMap;

use krishiv_sql::lexer::{Sym, Token, TokenKind, tokenize};
use krishiv_sql::{ParserOptions, parse_query};

use super::run::{Engine, EngineError, QueryOutput};
use super::setup::SessionSetup;

/// Parses with `krishiv-sql`; executes nothing.
#[derive(Debug, Default, Clone)]
pub struct ParseOnlyEngine {
    session: Vec<(String, String)>,
    conf: HashMap<String, String>,
}

impl ParseOnlyEngine {
    /// The parser options of the current session.
    #[must_use]
    pub fn options(&self) -> ParserOptions {
        ParserOptions::from_conf(|k| self.conf.get(k).map(String::as_str))
    }

    /// Applies a `SET key = value` or `RESET` statement; whether it was one.
    pub fn apply_set(&mut self, sql: &str) -> bool {
        let sql = sql.trim().trim_end_matches(';').trim();
        let word = sql.split_whitespace().next().unwrap_or_default();
        if word.eq_ignore_ascii_case("RESET") {
            // A bare `RESET` restores every key; `RESET key` restores only
            // that key (to its session value, or unset).
            match sql.get(5..).map(str::trim).filter(|k| !k.is_empty()) {
                None => self.conf = self.session.iter().cloned().collect(),
                Some(key) => match self.session.iter().find(|(k, _)| k == key) {
                    Some((k, v)) => {
                        self.conf.insert(k.clone(), v.clone());
                    }
                    None => {
                        self.conf.remove(key);
                    }
                },
            }
            return true;
        }
        if !word.eq_ignore_ascii_case("SET") {
            return false;
        }
        if let Some((k, v)) = sql.get(3..).and_then(|rest| rest.split_once('=')) {
            self.conf.insert(k.trim().to_owned(), v.trim().to_owned());
        }
        true
    }
}

impl Engine for ParseOnlyEngine {
    fn name(&self) -> &str {
        "krishiv-sql parse-only (DQL subset)"
    }

    fn reset(&mut self, setup: &SessionSetup) -> Result<(), EngineError> {
        self.session.clone_from(&setup.configs);
        self.conf = self.session.iter().cloned().collect();
        Ok(())
    }

    fn parse(&mut self, sql: &str) -> Result<(), EngineError> {
        parse_query(sql, &self.options())
            .map(drop)
            .map_err(|e| EngineError {
                class: Some(e.error.error_class().name().to_owned()),
                sql_state: e.error.sql_state().map(str::to_owned),
                message: e.error.to_string(),
            })
    }

    fn execute(&mut self, sql: &str) -> Result<QueryOutput, EngineError> {
        if !self.apply_set(sql) {
            self.parse(sql)?;
        }
        Err(EngineError {
            class: None,
            sql_state: None,
            message: "parse-only engine: not executed".to_owned(),
        })
    }
}

/// Whether `sql` is a query in the DQL parser's scope (see the module
/// docs).
#[must_use]
pub fn in_dql_scope(sql: &str) -> bool {
    let mut tokens = tokenize(sql);
    tokens.retain(|t| !t.kind.is_trivia());
    let word = |i: usize, kw: &str| {
        tokens
            .get(i)
            .is_some_and(|t| t.kind == TokenKind::Word && t.span.text(sql).eq_ignore_ascii_case(kw))
    };
    let sym = |i: usize, s: Sym| tokens.get(i).is_some_and(|t| t.kind == TokenKind::Sym(s));
    let starts_query = ["SELECT", "WITH", "VALUES", "TABLE", "FROM", "MAP", "REDUCE"]
        .iter()
        .any(|k| word(0, k))
        || sym(0, Sym::LParen);
    if !starts_query {
        return false;
    }
    let mut depth = 0_i32;
    for (i, t) in tokens.iter().enumerate() {
        match t.kind {
            TokenKind::Sym(Sym::LParen) => depth += 1,
            TokenKind::Sym(Sym::RParen) => depth -= 1,
            TokenKind::Unrecognized => return false,
            TokenKind::Word if out_of_scope_word(sql, &tokens, i, depth) => return false,
            _ => {}
        }
    }
    true
}

fn out_of_scope_word(sql: &str, tokens: &[Token], i: usize, depth: i32) -> bool {
    let word = |j: usize, kw: &str| {
        tokens
            .get(j)
            .is_some_and(|t| t.kind == TokenKind::Word && t.span.text(sql).eq_ignore_ascii_case(kw))
    };
    let w = word;
    (["WATERMARK", "STREAM", "CHANGES"].iter().any(|k| w(i, k)))
        || (depth == 0
            && ["INSERT", "UPDATE", "DELETE", "MERGE"]
                .iter()
                .any(|k| w(i, k)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_selects_plain_queries() {
        for sql in [
            "SELECT 1",
            "with t as (select 1) select * from t",
            "(SELECT 1) UNION (SELECT 2)",
            "FROM t SELECT a",
            "VALUES 1",
            "SELECT a | b FROM t",
            "SELECT CAST(x AS STRUCT<a: INT>)",
            "SELECT a COLLATE utf8_lcase FROM t",
            "SELECT * FROM t TABLESAMPLE (1 ROWS)",
            "FROM t |> SELECT a",
            "TABLE t | SELECT a",
            "SELECT * FROM IDENTIFIER('t')",
            "SELECT * FROM f(TABLE(t))",
            "SELECT * FROM a JOIN b APPROX NEAREST 2 BY DISTANCE x",
            "SELECT v:a.b[0]::INT FROM t",
            "SELECT TRANSFORM(a) USING 'cat' FROM t",
            "MAP a USING 'cat' FROM t",
        ] {
            assert!(in_dql_scope(sql), "{sql}");
        }
        for sql in [
            "CREATE TABLE t (a INT)",
            "INSERT INTO t VALUES 1",
            "WITH t AS (SELECT 1) INSERT INTO u SELECT * FROM t",
            "SELECT $1",
        ] {
            assert!(!in_dql_scope(sql), "{sql}");
        }
        assert!(in_dql_scope("SELECT /* unclosed"));
        assert!(!in_dql_scope("DESC /* unclosed"));
    }

    #[test]
    fn engine_tracks_set_statements() {
        let mut e = ParseOnlyEngine::default();
        e.reset(&SessionSetup::plain(super::super::setup::TestKind::of("x")))
            .unwrap();
        assert!(e.parse("SELECT 1 AS select").is_ok());
        assert!(
            e.execute("SET spark.sql.ansi.enforceReservedKeywords = true")
                .is_err()
        );
        let err = e.parse("SELECT 1 AS select").unwrap_err();
        assert_eq!(err.class.as_deref(), Some("PARSE_SYNTAX_ERROR"));
        assert!(e.execute("SELECT 1 AS select").unwrap_err().class.is_some());
        assert!(e.execute("RESET").is_err());
        assert!(e.parse("SELECT 1 AS select").is_ok());
        assert_eq!(e.name(), "krishiv-sql parse-only (DQL subset)");
    }

    #[test]
    fn keyed_reset_restores_only_that_key() {
        let mut e = ParseOnlyEngine::default();
        assert!(e.apply_set("SET spark.sql.ansi.doubleQuotedIdentifiers = true"));
        assert!(e.apply_set("SET spark.sql.ansi.enforceReservedKeywords = true;"));
        assert!(e.options().double_quoted_identifiers);
        assert!(e.apply_set("RESET spark.sql.ansi.doubleQuotedIdentifiers"));
        assert!(!e.options().double_quoted_identifiers);
        assert!(e.options().enforce_reserved_keywords);
        assert!(e.apply_set("RESET"));
        assert!(!e.options().enforce_reserved_keywords);
    }
}
