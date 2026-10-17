# krishiv-sql

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2; the DQL parser is in (P2-T070), the analyzer is next.

## Purpose
Spark SQL dialect parser and analyzer (resolution, type coercion).
`parse_query` → `ast::Query` (every node spanned; `==` ignores spans). Lexer `lexer.rs`, keyword
classes `keywords.rs`, recursive descent `parser.rs` + precedence climbing `parser/expr.rs`,
`types.rs` resolves written types through `krishiv_types::parse_data_type`.

## Invariants
- Own parser (no sqlparser-rs): hand-written lexer + recursive descent.
- Nesting is capped at `MAX_DEPTH` (= the IR's) with `FAILED_TO_PARSE_TOO_COMPLEX`; operator chains
  are iterative, and statements nested deeper than `INLINE_NESTING` brackets parse on a thread with a
  big stack, so no input overflows the caller's stack. Keep new recursive rules inside `nested`.
- `lexer::tokenize` is lossless and never fails (comments, unterminated quotes and unclosed comments are
  typed tokens): redaction (ADR 0038) relies on it. Errors never echo a string literal's text.
- Facts Spark's `AstBuilder` checks after parsing stay in the tree (`Keyword` vs string arguments,
  interval `trailing` parts, quoted typed-literal names, table options with spans); the post-parse
  validation (P2-T079) raises those errors, not the parser.
- `FROM ... SELECT ...` stores its `FROM` once (`FromStatement`); never copy subtrees per use.
- Ambiguities ANTLR resolves by prediction are decided deterministically; every decision is checked
  against a real Spark run (`tests/fixtures/spark-4.2.0/sql-parse-probe.json`,
  `tests/scripts/gen_sql_parse_probe.py`) and the golden corpus (`krishiv-testkit` `tests/sql_parse.rs`).
- Unresolved plans mirror Spark Connect relations/expressions (ADR 0003).
- Error messages/classes match Spark; `insta` snapshots for errors.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0008, 0012

## Test
`cargo nextest run -p krishiv-sql --no-tests=pass` · `cargo clippy -p krishiv-sql --all-targets -- -D warnings`
· corpus: `cargo nextest run -p krishiv-testkit --test sql_parse` · parse rate:
`cargo xtask golden --engine sql-parse` · fuzz: `cd fuzz && cargo +nightly fuzz run sql_parse`
· bench: `cargo bench -p krishiv-sql --bench parse`
