// SPDX-License-Identifier: Apache-2.0
//! Parser unit tests. Tree shapes (spans included) are `insta` snapshots,
//! one per feature group; errors, options, depth and lexing are asserted
//! directly.

use std::fmt::Write as _;

use krishiv_common::error::ErrorClass;

use super::ast::{BinaryOp, ExprKind, SetExprKind, SetKind, Span, Statement};
use super::lexer::{TokenKind, tokenize, unescape_string};
use super::*;

mod commands;
mod function;
mod golden_syntax;
mod probe;
mod sample;
mod tail;

/// The tree of `sql`, without Spark's post-parse checks (tree shapes).
fn ok(sql: &str) -> ast::Query {
    match parser::parse_query_unchecked(sql, ParserOptions::default()) {
        Ok(q) => q,
        Err(e) => panic!("{sql}: {}", e.error),
    }
}

fn err(sql: &str, options: &ParserOptions) -> ParseError {
    match parse_query(sql, options) {
        Ok(q) => panic!("{sql} parsed: {q:?}"),
        Err(e) => e,
    }
}

fn snapshot(statements: &[&str]) -> String {
    let mut out = String::new();
    for sql in statements {
        writeln!(out, "-- {sql}\n{:#?}", ok(sql)).unwrap();
    }
    out
}

/// Runs `f` on a thread with a 2 MiB stack, the default of spawned threads
/// and of tokio workers.
fn on_small_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

mod part1;
mod part2;
mod part3;
