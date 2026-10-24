// SPDX-License-Identifier: Apache-2.0
//! The SQL parser on arbitrary input. The first byte picks the parser
//! options, the rest is the statement. Parsing must return a tree or an
//! error (never panic, overflow the stack or take super-linear time or
//! memory: run with `-timeout` and `-rss_limit_mb`); an error's span must
//! lie within the input; and a tree must pass the span checks and parse to
//! an equal tree with other whitespace and comments. A tree is then
//! lowered and analyzed against a small catalog (an error or a resolved
//! plan, never a panic), and the text is redacted.
//! A redaction oracle writes a secret built from the input into every
//! option form: it never survives `redact_sql` or reaches an error.
#![no_main]

use krishiv_sql::lexer::{Sym, TokenKind, tokenize};
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, lower, parse_query, redact_sql};
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
        legacy_interval: bits & 32 != 0,
        timestamp_ntz: bits & 64 != 0,
        legacy_identifier_clause: bits & 128 != 0,
        single_character_pipe_operator: true,
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
            if let Ok(plan) = lower(&q, SourceId(0)) {
                assert!(!plan.root().is_resolved(), "a lowered plan is unresolved");
                let col = |name: &str| TableColumn {
                    name: name.to_owned(),
                    data_type: krishiv_types::DataType::Integer,
                    nullable: true,
                };
                let catalog = MemoryCatalog::new("spark_catalog")
                    .with_table(&["default"], "t", vec![col("a"), col("b")], None);
                let ctx = context(catalog, ResolutionConf::default());
                if let Ok(a) = block_on(DefaultAnalyzer.analyze(&plan, &ctx)) {
                    assert!(a.plan.root().is_resolved(), "unresolved node left");
                    assert_eq!(a.plan.root().resolved_output(), Some(a.output));
                }
            }
        }
        Err(e) => assert!(e.span.end <= sql.len()),
    }
    let _ = redact_sql(&sql).fragment(0..sql.len());
    redaction_oracle(rest);
});

/// A secret of at least 8 letters and digits from `data`, written into
/// option values in every form, is hidden by `redact_sql` and absent from
/// parse and lowering errors.
fn redaction_oracle(data: &[u8]) {
    let secret: String = data
        .iter()
        .map(|b| char::from(b'a' + b % 26))
        .take(24)
        .collect();
    if secret.len() < 8 {
        return;
    }
    let secret = format!("Zq{secret}");
    for template in [
        "SELECT a FROM t WITH ('password' = '{s}')",
        "SELECT a FROM t WITH (password = {s})",
        "SELECT a FROM t WITH (password '{s}' '{s}', k = 'v')",
        "SELECT a FROM t WITH (fs.s3a.secret.key = 'x' '{s}'",
        "SELECT a FROM t WITH ('url' = 'https://u:{s}@h/')",
        "SELECT 'https://u:{s}\\u0040h/' AS s FROM t",
        "SET spark.hadoop.fs.s3a.secret.key={s}",
        "SELECT a FROM t WITH (password = {s} {s})",
    ] {
        let sql = template.replace("{s}", &secret);
        assert!(!redact_sql(&sql).as_str().contains(&secret), "{sql}");
        if let Err(e) = parse_query(&sql, &ParserOptions::default()) {
            assert!(!format!("{e} {e:?}").contains(&secret), "{sql}");
        }
    }
}

/// The deepest parenthesis nesting of `sql`, plus its longest run of
/// prefix operators (each nests the expression after it).
fn nesting(sql: &str) -> usize {
    let (mut depth, mut max, mut run, mut longest) = (0_usize, 0_usize, 0_usize, 0_usize);
    for t in tokenize(sql).iter().filter(|t| !t.kind.is_trivia()) {
        match t.kind {
            TokenKind::Sym(Sym::LParen) => depth += 1,
            TokenKind::Sym(Sym::RParen) => depth = depth.saturating_sub(1),
            _ => {}
        }
        let prefix = matches!(
            t.kind,
            TokenKind::Sym(Sym::Plus | Sym::Minus | Sym::Tilde | Sym::Bang)
        ) || (t.kind == TokenKind::Word && t.span.text(sql).eq_ignore_ascii_case("NOT"));
        run = if prefix { run + 1 } else { 0 };
        longest = longest.max(run);
        max = max.max(depth);
    }
    max + longest
}
