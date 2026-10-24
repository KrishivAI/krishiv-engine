# krishiv-sql

**Layer:** L3 (may depend only on crates in L3 or lower; `cargo xtask layering`).  
**Real code arrives:** P2; the DQL parser (P2-T070) and lowering into the IR (P2-T071a) are in.

## Purpose
Spark SQL dialect parser and analyzer (resolution, type coercion).
`parse_query` → `ast::Query` (every node spanned; `==` ignores spans). Lexer `lexer.rs`, keyword
classes `keywords.rs`, recursive descent `parser.rs` + precedence climbing `parser/expr.rs`,
`types.rs` resolves written types through `krishiv_types::parse_data_type`.
Type strings use the same grammar (`parser/types.rs`): `parse_data_type` (Spark's `parseDataType`),
`parse_table_schema` (`parseTableSchema`) and `parse_ddl_type` (`DataType.fromDDL`, the schema
argument of `from_json` and friends); `validate.rs` holds the checks Spark's `AstBuilder` makes on
the tree (unknown types, missing sizes, decimal bounds, ...).
`lower` / `parse_and_lower` turn a query into an unresolved `krishiv_plan::Plan` (`lower.rs`, purely
syntactic, origins = spans); `redact_sql` turns query text into a `RedactedSql` from lexer tokens
(`redact.rs`).

## Key files
Paths are relative to this crate.
- `src/lexer.rs` lossless tokenizer; `src/keywords.rs` keyword classes
- `src/parser.rs` statement parser (recursive descent); `src/parser/expr.rs` precedence climbing
- `src/parser/types.rs` type strings (`parse_data_type`, `parse_table_schema`, `parse_ddl_type`)
- `src/ast.rs` spanned AST; `src/error.rs` parser errors with Spark classes
- `src/validate/mod.rs` checks Spark's `AstBuilder` makes after parsing (`literal.rs`, `types.rs`, `interval.rs`)
- `src/lower.rs` AST to unresolved `krishiv_plan::Plan`; `src/redact.rs` `redact_sql` to `RedactedSql`

## Extension points
- New construct: token in `keywords.rs` or `lexer.rs`, grammar in `parser.rs` or `parser/expr.rs`, checks in `validate/`, lowering in `lower.rs`.
- A construct without an IR node fails with `KRISHIV_UNSUPPORTED_FEATURE` naming it; recursive rules stay inside `nested`.

## Invariants
- Own parser (no sqlparser-rs): hand-written lexer + recursive descent.
- Nesting is capped at `MAX_DEPTH` (= the IR's) with `FAILED_TO_PARSE_TOO_COMPLEX` at the whole statement's
  span (exact limit and limit+1 tested per shape; no `stacker`); operator chains
  are iterative, and statements nested deeper than `INLINE_NESTING` brackets parse on a thread with a
  big stack, so no input overflows the caller's stack. Keep new recursive rules inside `nested`.
- `lexer::tokenize` is lossless and never fails (comments, unterminated quotes and unclosed comments are
  typed tokens): redaction (ADR 0038) relies on it. Errors never echo a string literal's text.
- Facts Spark's `AstBuilder` checks after parsing stay in the tree (`Keyword` vs string arguments,
  interval `trailing` parts, quoted typed-literal names, table options with spans); `validate/` raises
  those errors (P2-T079) after every top rule parses, in `AstBuilder`'s visit order, on a big stack for
  deep trees. Lowering reads literal values through `validate::literal`, so checked literals lower.
- Type errors follow ANTLR's prediction: a complex type whose `<...>` fails falls back to the bare
  type when that reads on (error at the `<`; after `::` the statement is re-parsed with the bare
  type when the complex one fails later); optional type parts are taken only when complete.
- `FROM ... SELECT ...` stores its `FROM` once (`FromStatement`); never copy subtrees per use.
- Ambiguities ANTLR resolves by prediction are decided deterministically; every decision is checked
  against a real Spark run (`tests/fixtures/spark-4.2.0/sql-parse-probe.json`,
  `tests/scripts/gen_sql_parse_probe.py`) and the golden corpus (`krishiv-testkit` `tests/sql_parse.rs`).
- `OVER w` resolves as Spark's `WindowsSubstitution`: the innermost `WINDOW` clause around decides (derived
  tables in its `FROM` see it, subquery expressions do not); a name it lacks is `MISSING_WINDOW_SPECIFICATION`
  at lowering; with no clause around, the name stays in the IR for the analyzer. `n PRECEDING` is `-(n)`.
- Lowering mirrors Spark's `AstBuilder` shapes (comma joins fold left into inner joins, balanced `AND`/`OR`,
  `<>` as `not(==)`, `HAVING` as a filter over the aggregate); the TPC-H oracle is `krishiv-plan`'s hand-built
  plans. Constructs the IR lacks fail with `KRISHIV_UNSUPPORTED_FEATURE` naming the construct, never silently.
  Recursion follows the tree (bounded by the parser) and hops to a fresh stack every `STEP` levels; join lists
  and dotted names are loops; the plan is checked against `MAX_DEPTH`.
- Query text reaches sinks only as `RedactedSql` (ADR 0038 update): option values of sensitive keys, URL
  credentials in literals and quoted identifiers, and comments (pattern redaction) are masked; nothing else.
- Error messages/classes match Spark; `insta` snapshots for errors.
- `#![forbid(unsafe_code)]`.
- No `unwrap`/`expect`/`panic` in library code; errors carry a Spark error class + SQLSTATE.

## ADRs (in `../krishiv-internal/adr/`)
0003, 0008, 0012, 0038

## Tests
`cargo nextest run -p krishiv-sql --no-tests=pass` · `cargo clippy -p krishiv-sql --all-targets -- -D warnings`
· corpus: `cargo nextest run -p krishiv-testkit --test sql_parse` (DDL/DML slice 1:
`--test ddl_commands_parse`) · parse rate:
`cargo xtask golden --engine sql-parse` · fuzz: `cd fuzz && cargo +nightly fuzz run sql_parse`
· bench: `cargo bench -p krishiv-sql --bench parse` / `--bench lower`
