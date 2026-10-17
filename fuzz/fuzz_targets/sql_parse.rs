// SPDX-License-Identifier: Apache-2.0
//! The SQL parser on arbitrary input. The first byte picks the parser
//! options, the rest is the statement. Parsing must return a tree or an
//! error (never panic, overflow the stack or take super-linear time or
//! memory: run with `-timeout` and `-rss_limit_mb`); an error's span must
//! lie within the input; and a tree must pass the span checks and parse to
//! an equal tree with other whitespace and comments.
#![no_main]

use krishiv_sql::lexer::{Sym, TokenKind, tokenize};
use krishiv_sql::{ParserOptions, parse_query};
use krishiv_testkit::golden::sql_spans::{check_spans, respaced};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&bits, rest)) = data.split_first() else {
        return;
    };
    let sql = String::from_utf8_lossy(rest);
    let options = ParserOptions {
        enforce_reserved_keywords: bits & 1 != 0,
        double_quoted_identifiers: bits & 2 != 0,
        legacy_setops_precedence: bits & 4 != 0,
        escaped_string_literals: bits & 8 != 0,
        legacy_bang_equals_not: bits & 16 != 0,
    };
    match parse_query(&sql, &options) {
        Ok(q) => {
            // The checker re-parses every expression and renders the tree
            // with `Debug`, which costs more than linear time in the depth;
            // check statements of a typical size and nesting only.
            if sql.len() <= 4096
                && nesting(&sql) <= 32
                && let Err(e) = check_spans(&sql, &q, &options)
            {
                panic!("{e}");
            }
            let again = parse_query(&respaced(&sql), &options);
            assert!(matches!(again, Ok(r) if r == q), "differential mismatch");
        }
        Err(e) => assert!(e.span.end <= sql.len()),
    }
});

/// The deepest parenthesis nesting of `sql`.
fn nesting(sql: &str) -> usize {
    let (mut depth, mut max) = (0_usize, 0_usize);
    for t in tokenize(sql) {
        match t.kind {
            TokenKind::Sym(Sym::LParen) => depth += 1,
            TokenKind::Sym(Sym::RParen) => depth = depth.saturating_sub(1),
            _ => {}
        }
        max = max.max(depth);
    }
    max
}
