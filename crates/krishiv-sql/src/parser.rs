// SPDX-License-Identifier: Apache-2.0
//! Recursive-descent parser for queries, relations and clauses; expressions
//! are in [`expr`] (precedence climbing over explicit stacks).
//!
//! Failures are [`Fail::Syntax`] (the furthest token reached is reported as
//! `PARSE_SYNTAX_ERROR`; a caller may try another alternative) or
//! [`Fail::Fatal`] (an error Spark raises at once, never retried). The only
//! retried alternative is a parenthesized subquery, and failed query
//! positions are remembered, so parsing stays near-linear on any input.
//!
//! Depth: every recursive rule runs inside [`Parser::nested`], and loops that
//! build left-deep trees (operator chains, set operations, postfix chains)
//! account for the height they add with [`Parser::measured`] and
//! [`Parser::grow`], so no tree nests deeper than [`MAX_DEPTH`]. A statement
//! nested deeper than [`INLINE_NESTING`] brackets parses on a thread whose
//! stack fits that depth.

mod expr;
mod types;

use std::collections::HashMap;

use crate::ast::{
    Alias, BucketSpec, ColumnConstraint, ColumnConstraintKind, ColumnDef, ColumnName,
    ConstraintCharacteristic, CreateNamespace, CreateTable, CreateTableLike, CreateView, Cte,
    DataTypeSpec, DdlName, DropBehavior, DropNamespace, DropTable, DropView, DynamicName, Explain,
    ExplainMode, Expr, ExprKind, FileFormat, FileFormatKind, ForeignKeyConstraint, ForeignKeyRef,
    FromClause, FromStatement, GeneratedColumn, GeneratedKind, GroupBy, GroupItem, GroupItemKind,
    GroupWith, GroupingSet, Hint, Ident, IdentityColumn, IdentityHow, Join, JoinConstraint,
    JoinKind, LateralView, Limit, Literal, NamedWindow, NamespaceKind, NumKind, ObjectName,
    OptionKey, OrderedIdent, PartitionColumn, PartitionField, PartitionFieldKind,
    PartitionTransform, Pivot, Query, Relation, RelationExt, Reset, RowDelimited, RowFormat,
    RowFormatKind, RowSerde, Select, SelectItem, Set, SetExpr, SetExprKind, SetKind, SetOperator,
    SetQuantifier, SkewSpec, SortItem, Span, Statement, StructField, TableAlias, TableConstraint,
    TableConstraintKind, TableElement, TableFactor, TableFactorKind, TableOption, TemporalClause,
    TemporalKind, TimeZone, Unpivot, UnpivotColumns, Use, Values, ViewColumn, With,
};
use crate::error::{self, ParseError};
use crate::keywords::{CLAUSE_WORD, RESERVED_WORD, STRICT_WORD, classify, is_kw};
use crate::lexer::{Sym, Token, TokenKind, tokenize, unescape_string, unquote_ident};
use crate::{MAX_DEPTH, ParserOptions, validate};

/// Why a rule did not match.
pub(crate) enum Fail {
    /// The tokens do not fit; the caller may try another alternative.
    Syntax,
    /// An error to report as is.
    Fatal(Box<ParseError>),
}

/// Parse state to rewind to when a speculative parse fails.
/// The `CREATE TABLE` clauses Spark rejects when written twice, in the
/// order it checks them (`DUPLICATE_CLAUSES` names the first).
const CREATE_TABLE_CLAUSES: [&str; 10] = [
    "TBLPROPERTIES",
    "OPTIONS",
    "PARTITIONED BY",
    "STORED AS/BY",
    "ROW FORMAT",
    "COMMENT",
    "DEFAULT COLLATION",
    "CLUSTERED BY",
    "CLUSTER BY",
    "LOCATION",
];

struct Checkpoint {
    pos: usize,
    far: usize,
    extra: Option<usize>,
    missing: Option<(usize, &'static str)>,
    matched: Option<(usize, Sym, Close)>,
    deferred: Option<ParseError>,
}

/// The inline recovery ANTLR shows a failed `match` of a closing token with
/// (measured on Spark 4.2): dropping the token in the way when the one after
/// it fits ("extra input"), and inserting the closing token when the token in
/// the way fits after it ("missing").
#[derive(Clone, Copy)]
pub(crate) enum Close {
    /// Only dropping the token in the way.
    Drop,
    /// Dropping, else inserting.
    DropOrInsert,
    /// After a comma list of this many elements: dropping, and at the end of
    /// the input (which Spark drops as well) once the list has two elements.
    List(usize),
}

pub(crate) type PR<T> = Result<T, Fail>;

pub(crate) struct Parser<'s> {
    src: &'s str,
    toks: Vec<Token>,
    /// [`classify`] bits per token.
    classes: Vec<u8>,
    pos: usize,
    opts: ParserOptions,
    /// Furthest token a rule failed at.
    far: usize,
    /// Token after a complete statement, when it is the last one (Spark's
    /// "extra input" hint).
    extra: Option<usize>,
    /// A token before which ANTLR would insert a missing token, and that
    /// token (Spark's "missing" hint).
    missing: Option<(usize, &'static str)>,
    /// The closing token ANTLR's `match` wanted at the furthest failure, and
    /// which recoveries Spark shows there; see [`Parser::error_with_recovery`].
    matched: Option<(usize, Sym, Close)>,
    /// Current recursion depth and the deepest tree level reached.
    depth: usize,
    peak: usize,
    /// Positions where a query failed to parse, with the furthest token
    /// reached.
    failed_queries: HashMap<usize, usize>,
    /// For each token, the first token at or after it that is not `(`.
    after_parens: Vec<usize>,
    /// Token positions of the `?` parameter markers, in order.
    markers: Vec<usize>,
    /// The first error Spark raises after parsing (in `AstBuilder`) that
    /// the tree cannot carry; reported if the statement parses.
    deferred: Option<ParseError>,
    /// The keyword tokens of the complex types read after `::`, in order.
    cast_types: Vec<usize>,
    /// A complex type keyword after `::` to read as a bare type.
    force_bare: Option<usize>,
}

/// Parses one query statement (`query ;* EOF`).
pub(crate) fn parse_query(src: &str, opts: ParserOptions) -> Result<Query, ParseError> {
    parse_with::<QueryRule>(src, opts)
}

/// Parses one query statement without Spark's post-parse checks: the tree
/// the checks see (parser tests of tree shapes).
#[cfg(test)]
pub(crate) fn parse_query_unchecked(src: &str, opts: ParserOptions) -> Result<Query, ParseError> {
    parse_with::<UncheckedQueryRule>(src, opts)
}

/// [`QueryRule`] without its checks.
#[cfg(test)]
struct UncheckedQueryRule;

#[cfg(test)]
impl TopRule for UncheckedQueryRule {
    type Out = Query;

    fn parse(p: &mut Parser<'_>) -> PR<Query> {
        p.statement()
    }

    fn validate(_: &Query, _: &str, _: ParserOptions) -> Result<(), ParseError> {
        Ok(())
    }
}

/// Parses one statement: a query, or `EXPLAIN` of a statement.
pub(crate) fn parse_statement(src: &str, opts: ParserOptions) -> Result<Statement, ParseError> {
    parse_with::<StatementRule>(src, opts)
}

/// The top rule of a statement.
trait TopRule {
    /// What it parses.
    type Out: Send + Sync;
    /// Whether an unclosed bracketed comment is reported even when the
    /// input has a syntax error (Spark's type-string entry points do so;
    /// a statement reports the syntax error).
    const UNCLOSED_FIRST: bool = false;
    /// Parses it.
    fn parse(p: &mut Parser<'_>) -> PR<Self::Out>;
    /// The checks Spark's `AstBuilder` makes on the parsed tree.
    fn validate(out: &Self::Out, src: &str, opts: ParserOptions) -> Result<(), ParseError>;
}

/// [`Parser::statement`].
struct QueryRule;

impl TopRule for QueryRule {
    type Out = Query;
    /// One query statement.
    fn parse(p: &mut Parser<'_>) -> PR<Query> {
        p.statement()
    }

    fn validate(out: &Self::Out, src: &str, opts: ParserOptions) -> Result<(), ParseError> {
        validate::query(out, src, opts)
    }
}

/// [`Parser::explainable`].
struct StatementRule;

impl TopRule for StatementRule {
    type Out = Statement;
    /// One statement.
    fn parse(p: &mut Parser<'_>) -> PR<Statement> {
        p.explainable()
    }

    fn validate(out: &Self::Out, src: &str, opts: ParserOptions) -> Result<(), ParseError> {
        validate::statement(out, src, opts)
    }
}

/// Parses one statement without the builder's checks, for tests of the
/// tree of a statement Spark parses and then rejects.
#[cfg(test)]
pub(crate) fn parse_statement_unchecked(
    src: &str,
    opts: ParserOptions,
) -> Result<Statement, ParseError> {
    parse_with::<UncheckedStatementRule>(src, opts)
}

/// [`Parser::explainable`], without [`validate::statement`].
#[cfg(test)]
struct UncheckedStatementRule;

#[cfg(test)]
impl TopRule for UncheckedStatementRule {
    type Out = Statement;
    fn parse(p: &mut Parser<'_>) -> PR<Statement> {
        p.explainable()
    }

    fn validate(_: &Self::Out, _: &str, _: ParserOptions) -> Result<(), ParseError> {
        Ok(())
    }
}

/// Parses one data type (`singleDataType`).
pub(crate) fn parse_data_type(src: &str, opts: ParserOptions) -> Result<DataTypeSpec, ParseError> {
    parse_with::<DataTypeRule>(src, opts)
}

/// Parses a table schema (`singleTableSchema`).
pub(crate) fn parse_table_schema(
    src: &str,
    opts: ParserOptions,
) -> Result<Vec<StructField>, ParseError> {
    parse_with::<TableSchemaRule>(src, opts)
}

/// [`Parser::single_data_type`].
struct DataTypeRule;

impl TopRule for DataTypeRule {
    type Out = DataTypeSpec;
    const UNCLOSED_FIRST: bool = true;
    /// One data type.
    fn parse(p: &mut Parser<'_>) -> PR<DataTypeSpec> {
        p.single_data_type()
    }

    fn validate(out: &Self::Out, _src: &str, _opts: ParserOptions) -> Result<(), ParseError> {
        validate::data_type(out)
    }
}

/// [`Parser::single_table_schema`].
struct TableSchemaRule;

impl TopRule for TableSchemaRule {
    type Out = Vec<StructField>;
    const UNCLOSED_FIRST: bool = true;
    /// Columns.
    fn parse(p: &mut Parser<'_>) -> PR<Vec<StructField>> {
        p.single_table_schema()
    }

    fn validate(out: &Self::Out, _src: &str, _opts: ParserOptions) -> Result<(), ParseError> {
        validate::columns(out)
    }
}

/// Parses a multipart identifier (`multipartIdentifier`).
pub(crate) fn parse_multipart_identifier(
    src: &str,
    opts: ParserOptions,
) -> Result<ObjectName, ParseError> {
    parse_with::<MultipartIdentifierRule>(src, opts)
}

/// Parses one named expression (`namedExpression`), as Spark's
/// `CatalystSqlParser.parseExpression` does for `F.expr`,
/// `df.selectExpr("...")` (once per argument string) and
/// `df.filter("...")`: one expression with an optional alias, then the end
/// of the input. The expression grammar is that of [`parse_query`]
/// (spans, the [`MAX_DEPTH`] recursion limit and error classes); a
/// statement or trailing tokens are a `PARSE_SYNTAX_ERROR`, as in Spark.
pub(crate) fn parse_named_expression(
    src: &str,
    opts: ParserOptions,
) -> Result<SelectItem, ParseError> {
    parse_with::<NamedExpressionRule>(src, opts)
}

/// [`Parser::single_named_expression`].
struct NamedExpressionRule;

impl TopRule for NamedExpressionRule {
    type Out = SelectItem;
    /// One named expression and the end of the input.
    fn parse(p: &mut Parser<'_>) -> PR<SelectItem> {
        p.single_named_expression()
    }

    fn validate(out: &Self::Out, src: &str, opts: ParserOptions) -> Result<(), ParseError> {
        validate::expression(&out.expr, src, opts)
    }
}

/// [`Parser::single_multipart_identifier`].
struct MultipartIdentifierRule;

impl TopRule for MultipartIdentifierRule {
    type Out = ObjectName;
    /// A dotted name.
    fn parse(p: &mut Parser<'_>) -> PR<ObjectName> {
        p.single_multipart_identifier()
    }

    fn validate(_: &Self::Out, _src: &str, _opts: ParserOptions) -> Result<(), ParseError> {
        Ok(())
    }
}

/// Lexes `src` and parses it with the top rule `R`.
fn parse_with<R: TopRule>(src: &str, opts: ParserOptions) -> Result<R::Out, ParseError> {
    // Java's `String.trim`: Spark reports an empty statement only for these.
    if src.trim_matches(|c| c <= ' ').is_empty() {
        return Err(error::empty_statement(Span::new(0, src.len())));
    }
    let mut toks = tokenize(src);
    // Spark reports an unclosed bracketed comment only once the statement
    // parsed (`exitSingleStatement`), so a syntax error is not masked.
    let unclosed = toks
        .iter()
        .find(|t| t.kind == TokenKind::UnclosedComment)
        .map(|t| t.span);
    toks.retain(|t| !t.kind.is_trivia() && t.kind != TokenKind::UnclosedComment);
    let parsed = parse_tokens::<R>(src, toks, opts);
    // Spark raises its post-parse (`AstBuilder`) errors only after the
    // unclosed-comment check, so that error wins over them.
    match (parsed, unclosed) {
        (Err(e), Some(span)) if e.deferred || R::UNCLOSED_FIRST => {
            Err(error::unclosed_comment(span))
        }
        (Err(e), _) => Err(e.error),
        (Ok(_), Some(span)) => Err(error::unclosed_comment(span)),
        (Ok(parsed), None) => Ok(parsed),
    }
}

/// A failed parse, and whether Spark raises it after parsing.
struct Failed {
    error: ParseError,
    deferred: bool,
}

impl From<ParseError> for Failed {
    /// An error raised while parsing.
    fn from(error: ParseError) -> Self {
        Self {
            error,
            deferred: false,
        }
    }
}

/// Parses the non-trivia tokens of one statement.
fn parse_tokens<R: TopRule>(
    src: &str,
    toks: Vec<Token>,
    opts: ParserOptions,
) -> Result<R::Out, Failed> {
    if nesting_bound(src, &toks) <= INLINE_NESTING {
        return run::<R>(src, toks, opts);
    }
    // A deeply nested statement parses on its own thread, whose stack fits
    // `MAX_DEPTH` levels whatever the caller's stack is.
    let stack = BIG_STACK;
    let at = statement_span(src, &toks);
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("krishiv-sql-parse".to_owned())
            .stack_size(stack)
            .spawn_scoped(scope, || run::<R>(src, toks, opts))
            .map_or_else(
                |_| Err(error::too_complex(at).into()),
                |h| {
                    h.join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                },
            )
    })
}

/// The stack of the threads that parse and check deep statements: it fits
/// [`MAX_DEPTH`] levels.
const BIG_STACK: usize = if cfg!(debug_assertions) {
    64 << 20
} else {
    16 << 20
};

/// The span of a whole statement, from its first token to its last: where
/// Spark reports `FAILED_TO_PARSE_TOO_COMPLEX` (its query context is the
/// statement, not the token parsing stopped at).
fn statement_span(src: &str, toks: &[Token]) -> Span {
    match (toks.first(), toks.last()) {
        (Some(first), Some(last)) => Span::new(first.span.start, last.span.end),
        _ => Span::new(0, src.len()),
    }
}

/// Bracket nesting (`(`, `[`, `CASE`, `<` of a complex type) parsed on the
/// caller's stack; each level takes a few recursion levels.
const INLINE_NESTING: usize = 16;

/// An upper bound of the bracket nesting of `toks`, plus one per lambda
/// arrow: every recursion of the parser is inside one of them.
fn nesting_bound(src: &str, toks: &[Token]) -> usize {
    let (mut depth, mut angles, mut max, mut arrows) = (0_usize, 0_usize, 0_usize, 0_usize);
    let mut prev_complex = false;
    for t in toks {
        let word = |kw: &str| t.kind == TokenKind::Word && is_kw(t.span.text(src), kw);
        match t.kind {
            TokenKind::Sym(Sym::LParen | Sym::LBracket) => depth += 1,
            TokenKind::Sym(Sym::Lt) if prev_complex => {
                depth += 1;
                angles += 1;
            }
            TokenKind::Sym(Sym::RParen | Sym::RBracket) => depth = depth.saturating_sub(1),
            TokenKind::Sym(Sym::Gt) if angles > 0 => {
                angles -= 1;
                depth = depth.saturating_sub(1);
            }
            TokenKind::Sym(Sym::Arrow) => arrows += 1,
            _ if word("CASE") => depth += 1,
            _ if word("END") => depth = depth.saturating_sub(1),
            _ => {}
        }
        max = max.max(depth);
        prev_complex = word("ARRAY") || word("MAP") || word("STRUCT");
    }
    max + arrows
}

/// Parses `toks` with the top rule `R` on the current thread.
fn run<R: TopRule>(src: &str, toks: Vec<Token>, opts: ParserOptions) -> Result<R::Out, Failed> {
    let mut p = Parser::new(src, toks, opts, None);
    let failed = match R::parse(&mut p) {
        Ok(out) => {
            let deep = p.peak > INLINE_NESTING;
            return p
                .finish(out)
                .and_then(|out| checked::<R>(out, src, opts, deep));
        }
        Err(Fail::Fatal(e)) => return Err((*e).into()),
        Err(Fail::Syntax) => p.error_with_recovery::<R>(opts),
    };
    // ANTLR's full-context prediction keeps a bare `ARRAY` after `::` as an
    // alternative to `ARRAY<...>` until one of them fails: when the bare
    // type reads on further (`x::array<int> > y` as a comparison), Spark
    // takes it.
    for keyword in retry_keywords(&p.cast_types, p.far) {
        let mut retry = Parser::new(src, p.toks.clone(), opts, Some(keyword));
        match R::parse(&mut retry) {
            Ok(out) => {
                let deep = retry.peak > INLINE_NESTING;
                return retry
                    .finish(out)
                    .and_then(|out| checked::<R>(out, src, opts, deep));
            }
            Err(Fail::Syntax) if retry.far > p.far => {
                return Err(retry.error_with_recovery::<R>(opts).into());
            }
            Err(_) => {}
        }
    }
    Err(failed.into())
}

/// The most `::` complex-cast keywords [`run`] retries as bare types.
const MAX_CAST_RETRIES: usize = 16;

/// The cast keywords worth a bare-type retry after a parse that read up to
/// token `far` failed, in
/// position order. Backtracking records a cast once per attempt, so they
/// are deduplicated. A keyword at or after the furthest token read cannot
/// have shaped the failure. Over the cap, the casts nearest the failure
/// are kept: forcing one bare changes the parse only from that token on,
/// so the cast that matters is the one the failure sits behind.
pub(crate) fn retry_keywords(cast_types: &[usize], far: usize) -> Vec<usize> {
    let mut keywords: Vec<usize> = cast_types.iter().copied().filter(|&k| k < far).collect();
    keywords.sort_unstable();
    keywords.dedup();
    let excess = keywords.len().saturating_sub(MAX_CAST_RETRIES);
    keywords.drain(..excess);
    keywords
}

/// Whether `key` is a bare `SET`/`RESET` key as Spark accepts it:
/// `[a-zA-Z_\d\.:]+`.
fn valid_bare_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':'))
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// `out` after Spark's post-parse checks ([`TopRule::validate`]), whose
/// errors Spark raises after parsing. The checks follow the tree, so a
/// `deep` tree is checked on a thread whose stack fits [`MAX_DEPTH`]
/// levels.
fn checked<R: TopRule>(
    out: R::Out,
    src: &str,
    opts: ParserOptions,
    deep: bool,
) -> Result<R::Out, Failed> {
    let result = if deep {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .name("krishiv-sql-check".to_owned())
                .stack_size(BIG_STACK)
                .spawn_scoped(scope, || R::validate(&out, src, opts))
                .map_or_else(
                    |_| Err(error::too_complex(Span::new(0, src.len()))),
                    |h| {
                        h.join()
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                    },
                )
        })
    } else {
        R::validate(&out, src, opts)
    };
    match result {
        Ok(()) => Ok(out),
        Err(error) => Err(Failed {
            error,
            deferred: true,
        }),
    }
}

/// Words after which a parenthesis opens a query rather than an expression.
const QUERY_START: &[&str] = &["SELECT", "WITH", "VALUES", "TABLE", "FROM"];

impl<'s> Parser<'s> {
    /// A parser at the start of `toks`; `force_bare` is the token of a
    /// complex type keyword after `::` to read as a bare type.
    fn new(src: &'s str, toks: Vec<Token>, opts: ParserOptions, force_bare: Option<usize>) -> Self {
        let classes = toks
            .iter()
            .map(|t| {
                if t.kind == TokenKind::Word {
                    classify(t.span.text(src))
                } else {
                    0
                }
            })
            .collect();
        let mut after_parens = vec![0; toks.len()];
        let mut next = toks.len();
        for (i, t) in toks.iter().enumerate().rev() {
            if t.kind != TokenKind::Sym(Sym::LParen) {
                next = i;
            }
            after_parens[i] = next;
        }
        let markers = (0..toks.len())
            .filter(|&i| toks[i].kind == TokenKind::Sym(Sym::Question))
            .collect();
        Self {
            src,
            toks,
            classes,
            pos: 0,
            opts,
            far: 0,
            extra: None,
            missing: None,
            matched: None,
            depth: 0,
            peak: 0,
            failed_queries: HashMap::new(),
            after_parens,
            markers,
            deferred: None,
            cast_types: vec![],
            force_bare,
        }
    }

    /// The outcome of a parse that produced `out`: an error Spark raises
    /// after parsing, if one was deferred.
    fn finish<T>(self, out: T) -> Result<T, Failed> {
        match self.deferred {
            Some(error) => Err(Failed {
                error,
                deferred: true,
            }),
            None => Ok(out),
        }
    }

    // ---- tokens -------------------------------------------------------

    fn tok(&self, n: usize) -> Option<Token> {
        self.toks.get(self.pos + n).copied()
    }

    fn kind(&self, n: usize) -> Option<TokenKind> {
        self.tok(n).map(|t| t.kind)
    }

    /// The [`classify`] bits of token `n` (0 for a non-word).
    fn class(&self, n: usize) -> u8 {
        self.classes.get(self.pos + n).copied().unwrap_or(0)
    }

    fn text(&self, n: usize) -> &'s str {
        self.tok(n).map_or("", |t| t.span.text(self.src))
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    /// Start of the current token (end of input if none).
    fn start(&self) -> usize {
        self.tok(0).map_or(self.src.len(), |t| t.span.start)
    }

    /// The span from `start` to the end of the last consumed token.
    fn span_from(&self, start: usize) -> Span {
        let end = self
            .pos
            .checked_sub(1)
            .and_then(|i| self.toks.get(i))
            .map_or(start, |t| t.span.end);
        Span::new(start, end.max(start))
    }

    fn kw_at(&self, n: usize, kw: &str) -> bool {
        self.kind(n) == Some(TokenKind::Word) && is_kw(self.text(n), kw)
    }

    fn any_kw_at(&self, n: usize, kws: &[&str]) -> bool {
        kws.iter().any(|k| self.kw_at(n, k))
    }

    fn sym_at(&self, n: usize, s: Sym) -> bool {
        self.kind(n) == Some(TokenKind::Sym(s))
    }

    /// Records a failure at the current token.
    fn fail<T>(&mut self) -> PR<T> {
        if self.pos > self.far {
            self.matched = None;
        }
        self.far = self.far.max(self.pos);
        Err(Fail::Syntax)
    }

    fn fatal<T>(e: ParseError) -> PR<T> {
        Err(Fail::Fatal(Box::new(e)))
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        let hit = self.kw_at(0, kw);
        self.pos += usize::from(hit);
        hit
    }

    /// The current parse state, for a speculative parse.
    fn checkpoint(&mut self) -> Checkpoint {
        Checkpoint {
            pos: self.pos,
            far: self.far,
            extra: self.extra,
            missing: self.missing,
            matched: self.matched,
            deferred: self.deferred.take(),
        }
    }

    /// Rewinds to `checkpoint`, abandoning the speculative parse.
    fn restore(&mut self, checkpoint: Checkpoint) {
        self.pos = checkpoint.pos;
        self.far = checkpoint.far;
        self.extra = checkpoint.extra;
        self.missing = checkpoint.missing;
        self.matched = checkpoint.matched;
        self.deferred = checkpoint.deferred;
    }

    /// Keeps the speculative parse begun at `checkpoint`: an error deferred
    /// before it still wins over one deferred during it.
    fn commit(&mut self, checkpoint: Checkpoint) {
        if checkpoint.deferred.is_some() {
            self.deferred = checkpoint.deferred;
        }
    }

    fn expect_kw(&mut self, kw: &str) -> PR<()> {
        if self.eat_kw(kw) { Ok(()) } else { self.fail() }
    }

    fn eat_sym(&mut self, s: Sym) -> bool {
        let hit = self.sym_at(0, s);
        self.pos += usize::from(hit);
        hit
    }

    fn expect_sym(&mut self, s: Sym) -> PR<()> {
        if self.eat_sym(s) { Ok(()) } else { self.fail() }
    }

    /// `eat_sym` for a closing token that ANTLR matches after a complete
    /// element (no prediction decision in the way), where its error recovery
    /// applies: a failure here may carry Spark's `extra input` hint and, per
    /// `how`, its `missing` hint.
    fn expect_closing(&mut self, s: Sym, how: Close) -> PR<()> {
        if self.eat_sym(s) {
            return Ok(());
        }
        let r = self.fail();
        if self.pos == self.far {
            self.matched = Some((self.pos, s, how));
        }
        r
    }

    fn comma_sep<T>(&mut self, mut f: impl FnMut(&mut Self) -> PR<T>) -> PR<Vec<T>> {
        let mut v = vec![f(self)?];
        while self.eat_sym(Sym::Comma) {
            v.push(f(self)?);
        }
        Ok(v)
    }

    /// `( f, ... )`.
    fn paren_list<T>(&mut self, f: impl FnMut(&mut Self) -> PR<T>) -> PR<Vec<T>> {
        self.expect_sym(Sym::LParen)?;
        let v = self.comma_sep(f)?;
        self.expect_closing(Sym::RParen, Close::List(v.len()))?;
        Ok(v)
    }

    #[inline(never)]
    fn syntax_error(&self) -> ParseError {
        let found = self.toks.get(self.far);
        let (display, span) = found.map_or_else(
            || {
                (
                    "end of input".to_owned(),
                    Span::new(self.src.len(), self.src.len()),
                )
            },
            |t| {
                // An unterminated quote is reported at the quote, as Spark does.
                let span = if t.kind == TokenKind::Unterminated {
                    let len = t
                        .span
                        .text(self.src)
                        .chars()
                        .next()
                        .map_or(1, char::len_utf8);
                    Span::new(t.span.start, t.span.start + len)
                } else {
                    t.span
                };
                (self.token_display(*t), span)
            },
        );
        let hint = match self.missing {
            _ if self.extra == Some(self.far) => format!(": extra input {display}"),
            Some((at, token)) if at == self.far => format!(": missing {token}"),
            _ => String::new(),
        };
        error::syntax_error(display, hint, span)
    }

    /// The syntax error at the furthest failure, with the hint of ANTLR's
    /// inline recovery when the failure is at a `match` ([`expect_closing`]):
    /// ANTLR first drops the token in the way if the one after it fits
    /// ("extra input"), else inserts the wanted token if the token in the way
    /// fits after it ("missing"). Whether a token fits is decided by parsing
    /// the statement again with the edit made: it fits when the parse
    /// consumes it, and for a dropped token only when the wanted token is next.
    ///
    /// [`expect_closing`]: Parser::expect_closing
    #[inline(never)]
    fn error_with_recovery<R: TopRule>(&mut self, opts: ParserOptions) -> ParseError {
        self.dangling_clause_word::<R>(opts);
        self.organization_clause_failure::<R>(opts);
        if let Some((at, sym, how)) = self.matched
            && at == self.far
            && self.extra != Some(at)
            && self.missing.is_none_or(|(m, _)| m != at)
        {
            let fits = |edited: Vec<Token>, past: usize| {
                let mut p = Parser::new(self.src, edited, opts, self.force_bare);
                match R::parse(&mut p) {
                    Ok(_) => true,
                    Err(Fail::Syntax) => p.far > past,
                    Err(Fail::Fatal(_)) => false,
                }
            };
            let mut dropped = self.toks.clone();
            if at < dropped.len() {
                dropped.remove(at);
                // The token after the dropped one must be the wanted one: it
                // is what the state expects (operators that continue the
                // element aside, which Spark's hints do not show either).
                let next = self.toks.get(at + 1).map(|t| t.kind);
                if next == Some(TokenKind::Sym(sym)) && fits(dropped, at) {
                    self.extra = Some(at);
                }
            } else if matches!(how, Close::List(n) if n >= 2) {
                self.extra = Some(at);
            }
            if self.extra != Some(at) && matches!(how, Close::DropOrInsert) {
                let text = match sym {
                    Sym::RBracket => Some("']'"),
                    Sym::RParen => Some("')'"),
                    _ => None,
                };
                let mut inserted = self.toks.clone();
                let start = self.toks.get(at).map_or(self.src.len(), |t| t.span.start);
                let token = Token {
                    kind: TokenKind::Sym(sym),
                    span: Span::new(start, start),
                };
                inserted.insert(at, token);
                if let Some(text) = text
                    && fits(inserted, at + 1)
                {
                    self.missing = Some((at, text));
                }
            }
        }
        self.syntax_error()
    }

    /// A statement that ends with `BY` or `HAVING` right after text that is a
    /// statement without it (`... t ORDER BY`, `... GROUP BY a HAVING`):
    /// ANTLR's prediction for the optional clause fails at the end of the
    /// input and takes the exit, so the error is the clause's last token
    /// dropped as "extra input" instead of the end of the input.
    fn dangling_clause_word<R: TopRule>(&mut self, opts: ParserOptions) {
        let n = self.toks.len();
        let last_is = |p: &Self, kw: &str| {
            p.toks
                .last()
                .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(p.src), kw))
        };
        if n >= 2
            && self.far == n
            && (last_is(self, "BY") || last_is(self, "HAVING"))
            && self.extra.is_none()
        {
            let mut p = Parser::new(self.src, self.toks[..n - 1].to_vec(), opts, self.force_bare);
            if R::parse(&mut p).is_ok() {
                self.far = n - 1;
                self.extra = Some(n - 1);
            }
        }
    }

    /// A failure at the end of the input inside an `ORDER BY`, `SORT BY` or
    /// `LIMIT` clause that follows a complete query (`... t ORDER BY a,`; `(SELECT 1 ORDER BY a,` also gets its `missing` hint):
    /// ANTLR's prediction takes the keyword as an alias and reports it, with
    /// no hint, instead of the end of the input. Failing right after the
    /// `BY` reports the `BY`.
    fn organization_clause_failure<R: TopRule>(&mut self, opts: ParserOptions) {
        let n = self.toks.len();
        if self.far != n || self.extra.is_some() {
            return;
        }
        let word = |p: &Self, i: usize, kw: &str| {
            p.toks
                .get(i)
                .is_some_and(|t| t.kind == TokenKind::Word && is_kw(t.span.text(p.src), kw))
        };
        let mut depth = 0usize;
        let mut clause = None;
        for i in (0..n).rev() {
            match self.toks[i].kind {
                TokenKind::Sym(Sym::RParen) => depth += 1,
                TokenKind::Sym(Sym::LParen) => depth = depth.saturating_sub(1),
                _ if depth == 0
                    && ((word(self, i, "ORDER") || word(self, i, "SORT"))
                        && word(self, i + 1, "BY")
                        && i + 2 < n
                        || word(self, i, "LIMIT") && i + 1 < n) =>
                {
                    clause = Some(i);
                    break;
                }
                _ => {}
            }
        }
        let Some(k) = clause.filter(|&k| k > 0) else {
            return;
        };
        let mut p = Parser::new(self.src, self.toks[..k].to_vec(), opts, self.force_bare);
        // The query before the clause is complete, or ends where the clause
        // starts (an unclosed `(`), which then gets its recovery hint.
        match R::parse(&mut p) {
            Ok(_) => self.far = k,
            Err(Fail::Syntax) if p.far == k => {
                self.far = k;
                self.matched = p.matched;
            }
            Err(_) => {}
        }
    }

    /// A token as an error message shows it: quoted text like Spark's, except
    /// that the text of a string literal, which may be a secret, is never
    /// echoed (an unterminated one shows its quote, as Spark does).
    fn token_display(&self, t: Token) -> String {
        let text = t.span.text(self.src);
        match t.kind {
            TokenKind::Unterminated => {
                format!("'{}'", text.chars().next().unwrap_or('\''))
            }
            TokenKind::DoubleQuoted if self.opts.double_quoted_identifiers => format!("'{text}'"),
            // Spark shows the opening tag of a dollar-quoted string.
            TokenKind::DollarString => {
                let tag = text.get(1..).and_then(|t| t.find('$')).map_or(1, |i| i + 2);
                format!("'{}'", text.get(..tag).unwrap_or("$"))
            }
            k if k.is_string_like() => "a string literal".to_owned(),
            _ => {
                let shown = text
                    .replace('\n', "\\n")
                    .replace('\r', "\\r")
                    .replace('\t', "\\t");
                format!("'{shown}'")
            }
        }
    }

    /// Records an error Spark raises after parsing; the first one wins.
    fn defer(&mut self, e: ParseError) {
        if self.deferred.is_none() {
            self.deferred = Some(e);
        }
    }

    // ---- depth ----------------------------------------------------------

    /// Goes one level deeper (the caller decrements `depth` after).
    fn enter(&mut self) -> PR<()> {
        self.depth += 1;
        let r = self.grow(0);
        if r.is_err() {
            self.depth -= 1;
        }
        r
    }

    /// Runs `f` one level deeper.
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> PR<T>) -> PR<T> {
        self.enter()?;
        let r = f(self);
        self.depth -= 1;
        r
    }

    /// Runs `f` and returns how many levels its tree reaches below the
    /// current depth.
    fn measured<T>(&mut self, f: impl FnOnce(&mut Self) -> PR<T>) -> PR<(T, usize)> {
        let outer = self.peak;
        self.peak = self.depth;
        let r = f(self);
        let height = self.peak - self.depth;
        self.peak = self.peak.max(outer);
        r.map(|t| (t, height))
    }

    /// Records a tree reaching `height` levels below the current depth.
    fn grow(&mut self, height: usize) -> PR<()> {
        let level = self.depth + height;
        if level > MAX_DEPTH {
            return Self::fatal(error::too_complex(statement_span(self.src, &self.toks)));
        }
        self.peak = self.peak.max(level);
        Ok(())
    }

    // ---- identifiers ----------------------------------------------------

    /// The identifier at token `n`, if it is one; `strong` accepts every
    /// keyword (a position where only an identifier fits) unless keywords
    /// are enforced.
    fn ident_at(&self, n: usize, strong: bool) -> Option<Ident> {
        let t = self.tok(n)?;
        let text = t.span.text(self.src);
        let (value, quoted) = match t.kind {
            TokenKind::Word => {
                let keyword_ok = strong && !self.opts.enforce_reserved_keywords;
                if !keyword_ok && self.class(n) & RESERVED_WORD != 0 {
                    return None;
                }
                (text.to_owned(), false)
            }
            TokenKind::QuotedIdent => (unquote_ident(text), true),
            TokenKind::DoubleQuoted if self.opts.double_quoted_identifiers => (
                text.get(1..text.len() - 1)
                    .unwrap_or_default()
                    .replace("\"\"", "\""),
                true,
            ),
            _ => return None,
        };
        Some(Ident {
            value,
            quoted,
            span: t.span,
        })
    }

    /// Whether token `n` is an identifier ([`Self::ident_at`] without
    /// building it).
    fn is_ident_at(&self, n: usize, strong: bool) -> bool {
        match self.kind(n) {
            Some(TokenKind::Word) => {
                (strong && !self.opts.enforce_reserved_keywords)
                    || self.class(n) & RESERVED_WORD == 0
            }
            Some(TokenKind::QuotedIdent) => true,
            Some(TokenKind::DoubleQuoted) => self.opts.double_quoted_identifiers,
            _ => false,
        }
    }

    /// Consumes an identifier (`strong` as for [`Self::ident_at`]).
    fn ident_with(&mut self, strong: bool) -> PR<Ident> {
        let Some(id) = self.ident_at(0, strong) else {
            return self.fail();
        };
        // Spark's `exitUnquotedIdentifier`: only ASCII letters, digits, `_`.
        if !id.quoted
            && !id
                .value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Self::fatal(error::invalid_identifier(&id.value, id.span));
        }
        self.bump();
        Ok(id)
    }

    fn ident(&mut self) -> PR<Ident> {
        self.ident_with(false)
    }

    fn any_ident(&mut self) -> PR<Ident> {
        self.ident_with(true)
    }

    /// An identifier that is not followed by `- word` (Spark's
    /// `errorCapturingIdentifier`: `test-table` is `INVALID_IDENTIFIER`).
    fn name_part(&mut self, strong: bool) -> PR<Ident> {
        let id = self.ident_with(strong)?;
        if self.sym_at(0, Sym::Minus) && self.is_ident_at(1, true) {
            // The name is the whole `a-b-c`; Spark's position is the first
            // `-` (the start of the extra part it rejects).
            let (start, dash) = (id.span.start, self.start());
            while self.sym_at(0, Sym::Minus) && self.is_ident_at(1, true) {
                self.pos += 2;
            }
            let name = self.span_from(start).text(self.src);
            return Self::fatal(error::invalid_identifier(name, self.span_from(dash)));
        }
        Ok(id)
    }

    fn object_name(&mut self) -> PR<ObjectName> {
        let start = self.start();
        let mut parts = vec![self.name_part(false)?];
        while self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
            self.bump();
            parts.push(self.name_part(true)?);
        }
        Ok(ObjectName {
            parts,
            span: self.span_from(start),
        })
    }

    /// A table name. Unless keywords are enforced, any keyword names a
    /// table here (`FROM order`, `FROM from t`), as Spark's grammar allows.
    fn table_name(&mut self) -> PR<ObjectName> {
        let start = self.start();
        let mut parts = vec![self.name_part(true)?];
        while self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
            self.bump();
            parts.push(self.name_part(true)?);
        }
        Ok(ObjectName {
            parts,
            span: self.span_from(start),
        })
    }

    /// Whether the trailing input is one token to Spark, so ANTLR
    /// reports it as "extra input": the last token when nothing follows
    /// it. An unterminated quote with text after it is several tokens to
    /// Spark, so no single extraneous token.
    fn trailing_is_single_token(&self) -> bool {
        self.tok(0)
            .filter(|_| self.pos + 1 == self.toks.len())
            .is_some_and(|t| {
                t.kind != TokenKind::Unterminated || t.span.text(self.src).chars().count() == 1
            })
    }

    /// `singleMultipartIdentifier`: a dotted name and the end of the input.
    fn single_multipart_identifier(&mut self) -> PR<ObjectName> {
        let name = self.table_name()?;
        if self.pos < self.toks.len() {
            // ANTLR drops one extraneous token before the end, as "extra
            // input".
            if self.trailing_is_single_token() {
                self.extra = Some(self.pos);
            }
            return self.fail();
        }
        Ok(name)
    }

    /// One `namedExpression` and the end of the input: one expression with
    /// an optional alias, as in Spark's `parseExpression` (one extraneous
    /// token reads as "extra input", as ANTLR drops it).
    fn single_named_expression(&mut self) -> PR<SelectItem> {
        let item = self.select_item()?;
        if self.pos < self.toks.len() {
            if self.trailing_is_single_token() {
                self.extra = Some(self.pos);
            }
            return self.fail();
        }
        Ok(item)
    }

    fn ident_list(&mut self) -> PR<Vec<Ident>> {
        self.paren_list(Self::any_ident)
    }

    /// Whether token `n` ends a list item: end of input, `,`, `)` or `;`.
    fn item_end_at(&self, n: usize) -> bool {
        matches!(
            self.kind(n),
            None | Some(TokenKind::Sym(Sym::Comma | Sym::RParen | Sym::Semi))
        )
    }

    /// An alias without `AS`. Unless keywords are enforced, a keyword that
    /// ends the item is one too (`SELECT a FROM` names `a` `FROM`, as
    /// ANTLR reads it); `strict` admits the strict-non-reserved words
    /// (`UNION`, `JOIN`, ...), which only a column alias may be.
    fn implicit_alias(&mut self, strict: bool) -> PR<Option<Ident>> {
        let keyword = !self.opts.enforce_reserved_keywords
            && self.kind(0) == Some(TokenKind::Word)
            && self.item_end_at(1)
            && (strict || self.class(0) & STRICT_WORD == 0);
        let clause = self.kind(0) == Some(TokenKind::Word) && self.class(0) & CLAUSE_WORD != 0;
        if !self.is_ident_at(0, keyword) || (clause && !keyword) {
            return Ok(None);
        }
        self.ident_with(keyword).map(Some)
    }

    /// `[AS] name`.
    fn opt_alias_name(&mut self) -> PR<Option<Ident>> {
        let strict_next = self.kind(1) == Some(TokenKind::Word) && self.class(1) & STRICT_WORD != 0;
        if self.kw_at(0, "AS") && strict_next && !self.opts.enforce_reserved_keywords {
            // `t AS union`: a strict-non-reserved word is no table alias
            // (`strictIdentifier`), so ANTLR reads `AS` as the alias.
            return self.ident_with(true).map(Some);
        }
        if !self.item_end_at(1) && self.eat_kw("AS") {
            return self.any_ident().map(Some);
        }
        self.implicit_alias(false)
    }

    // ---- statements and queries ------------------------------------------

    fn statement(&mut self) -> PR<Query> {
        let q = self.query()?;
        self.command_end()?;
        Ok(q)
    }

    /// Trailing `;`s and the end of the input after a complete statement.
    /// Anything else is a syntax error at the token parsing stopped at;
    /// one extraneous token before the end (or any after `;`, read by
    /// ANTLR's `SEMICOLON*` loop) is reported as "extra input", as in
    /// Spark.
    fn command_end(&mut self) -> PR<()> {
        let mut semi = false;
        while self.eat_sym(Sym::Semi) {
            semi = true;
        }
        if self.pos < self.toks.len() {
            if semi || self.trailing_is_single_token() {
                self.extra = Some(self.pos);
            }
            return self.fail();
        }
        Ok(())
    }

    /// `(EXPLAIN [mode])* query ;* EOF`. The `EXPLAIN` prefixes are read
    /// in a loop and the tree is built from the inside out, so a long
    /// chain neither recurses nor nests deeper than [`MAX_DEPTH`].
    fn explainable(&mut self) -> PR<Statement> {
        let mut explains = Vec::new();
        while self.kw_at(0, "EXPLAIN") {
            let start = self.start();
            self.bump();
            if explains.len() == MAX_DEPTH {
                return Self::fatal(error::too_complex(statement_span(self.src, &self.toks)));
            }
            let word = if self.kind(0) == Some(TokenKind::Word) {
                self.text(0).to_ascii_uppercase()
            } else {
                String::new()
            };
            let mode = match word.as_str() {
                "EXTENDED" => Some(ExplainMode::Extended),
                "FORMATTED" => Some(ExplainMode::Formatted),
                "CODEGEN" => Some(ExplainMode::Codegen),
                "COST" => Some(ExplainMode::Cost),
                // Spark parses `LOGICAL`, then rejects it after parsing.
                "LOGICAL" => Some(ExplainMode::Simple),
                // Spark reads `EXPLAIN ANALYZE` as the start of `ANALYZE
                // TABLE` (unsupported here: its syntax error is at the next
                // token). Before anything else it is Krishiv's
                // `EXPLAIN ANALYZE <query>`, a syntax error in Spark.
                "ANALYZE" if self.kw_at(1, "TABLE") || self.kw_at(1, "TABLES") => {
                    self.bump();
                    return self.fail();
                }
                "ANALYZE" => Some(ExplainMode::Analyze),
                // A syntax error in Spark; Krishiv's `EXPLAIN VERBOSE`.
                "VERBOSE" => Some(ExplainMode::Verbose),
                _ => None,
            };
            let logical = word == "LOGICAL";
            if mode.is_some() {
                self.bump();
            }
            explains.push((start, mode.unwrap_or(ExplainMode::Simple), logical));
        }
        // `SET`, `RESET`, `CREATE`, `DROP` and `USE` start no query:
        // without an `EXPLAIN` the statement is that command, and under one
        // it is the explained statement. Anything else is a query, which
        // fails under `EXPLAIN` as in Spark when it is a command (a command
        // explains; it never reads as a query).
        let statement = if self.kw_at(0, "SET") {
            self.set_command()?
        } else if self.kw_at(0, "RESET") {
            self.reset_command()?
        } else if self.kw_at(0, "CREATE") {
            self.create_command()?
        } else if self.kw_at(0, "DROP") {
            self.drop_command()?
        } else if self.kw_at(0, "USE") {
            self.use_command()?
        } else {
            Statement::Query(Box::new(self.statement()?))
        };
        let end = statement.span().end;
        // The outermost `EXPLAIN LOGICAL` is reported first, before any
        // error Spark raises while building the explained query.
        if let Some(&(start, ..)) = explains.iter().find(|e| e.2) {
            self.deferred = Some(error::invalid_statement(
                "EXPLAIN LOGICAL",
                Span::new(start, end),
            ));
        }
        let mut statement = statement;
        for (start, mode, _) in explains.into_iter().rev() {
            statement = Statement::Explain(Explain {
                mode,
                statement: Box::new(statement),
                span: Span::new(start, end),
            });
        }
        Ok(statement)
    }

    /// `SET`, `SET -v`, `SET key`, `SET key = value` or `SET TIME ZONE
    /// ...`. A bare key holds no whitespace and matches Spark's
    /// `[a-zA-Z_\d\.:]+`; anything else is `INVALID_SET_SYNTAX`. Only
    /// backquotes quote a key or value: a double-quoted key is a syntax
    /// error, a single-quoted one fails the key check. The value is the
    /// raw rest of the statement, trimmed (decoded only when it is one
    /// backquoted token); anything after its `;` is extra input, as for
    /// queries.
    fn set_command(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("SET")?;
        // `SET` with nothing after it lists every explicitly set key.
        if self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi) {
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Set(Set {
                kind: SetKind::All,
                span,
            }));
        }
        // `SET -v`: only `-v` with nothing after it.
        if self.sym_at(0, Sym::Minus) {
            return self.set_verbose_command(start);
        }
        // `SET TIME ZONE ...`.
        if self.kw_at(0, "TIME") && self.kw_at(1, "ZONE") {
            return self.set_timezone_command(start);
        }
        // A backquoted key.
        if let Some((key, key_span)) = self.quoted_key()? {
            if self.eat_sym(Sym::Eq) {
                let (value, value_span) = self.set_value()?;
                let span = self.span_from(start);
                self.command_end()?;
                return Ok(Statement::Set(Set {
                    kind: SetKind::Assign {
                        key,
                        key_span,
                        value,
                        value_span,
                    },
                    span,
                }));
            }
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Set(Set {
                kind: SetKind::Show { key, key_span },
                span,
            }));
        }
        // A bare key runs to the first `=` (an assignment) or to the `;`
        // or end of input (a lookup). A double-quoted token there is a
        // syntax error, as in Spark; the text is the key as written, which
        // must match `[a-zA-Z_\d\.:]+`.
        let mut eq = None;
        let mut semi = self.toks.len();
        for (i, t) in self.toks.iter().enumerate().skip(self.pos) {
            match t.kind {
                TokenKind::Sym(Sym::Eq) => {
                    eq = Some(i);
                    break;
                }
                TokenKind::Sym(Sym::Semi) => {
                    semi = i;
                    break;
                }
                TokenKind::Unterminated | TokenKind::DoubleQuoted => {
                    self.pos = i;
                    return self.fail();
                }
                _ => {}
            }
        }
        if let Some(e) = eq {
            let from = self.start();
            let (key, key_span) = self.trimmed_span(from, self.toks[e].span.start);
            if !valid_bare_key(&key) {
                return Self::fatal(error::invalid_set_syntax(self.span_from(start)));
            }
            self.pos = e + 1;
            let (value, value_span) = self.set_value()?;
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Set(Set {
                kind: SetKind::Assign {
                    key,
                    key_span,
                    value,
                    value_span,
                },
                span,
            }));
        }
        let from = self.start();
        let to = (self.pos..semi)
            .filter_map(|i| self.toks.get(i))
            .map(|t| t.span.end)
            .max()
            .unwrap_or(from);
        let (key, key_span) = self.trimmed_span(from, to);
        if !valid_bare_key(&key) {
            return Self::fatal(error::invalid_set_syntax(self.span_from(start)));
        }
        self.pos = semi;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Set(Set {
            kind: SetKind::Show { key, key_span },
            span,
        }))
    }

    /// `SET -...` after the `SET` at `start`: only `-v` with nothing after
    /// it; anything else is `INVALID_SET_SYNTAX`.
    fn set_verbose_command(&mut self, start: usize) -> PR<Statement> {
        self.bump();
        if self.kind(0) == Some(TokenKind::Word) && self.text(0).eq_ignore_ascii_case("v") {
            self.bump();
            if self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi) {
                let span = self.span_from(start);
                self.command_end()?;
                return Ok(Statement::Set(Set {
                    kind: SetKind::Verbose,
                    span,
                }));
            }
        }
        Self::fatal(error::invalid_set_syntax(self.span_from(start)))
    }

    /// `SET TIME ZONE LOCAL | '<zone>'` after the `SET` at `start`. An
    /// `INTERVAL` zone is a supported grammar form the engine does not
    /// implement, so it fails with `KRISHIV_UNSUPPORTED_FEATURE` naming
    /// the form.
    fn set_timezone_command(&mut self, start: usize) -> PR<Statement> {
        self.pos += 2;
        let (zone, zone_span) = match self.kind(0) {
            Some(TokenKind::Word) if self.text(0).eq_ignore_ascii_case("LOCAL") => {
                let span = self.tok(0).map_or(self.span_from(start), |t| t.span);
                self.bump();
                (TimeZone::Local, span)
            }
            Some(TokenKind::String) => {
                let Some(t) = self.tok(0) else {
                    return self.fail();
                };
                let zone =
                    unescape_string(t.span.text(self.src), !self.opts.escaped_string_literals);
                self.bump();
                (TimeZone::Named(zone), t.span)
            }
            Some(TokenKind::DoubleQuoted) if !self.opts.double_quoted_identifiers => {
                // A string when `"` does not quote identifiers.
                let Some(t) = self.tok(0) else {
                    return self.fail();
                };
                let zone =
                    unescape_string(t.span.text(self.src), !self.opts.escaped_string_literals);
                self.bump();
                (TimeZone::Named(zone), t.span)
            }
            _ if self.kw_at(0, "INTERVAL") => {
                let from = self.start();
                let mut end = self.src.len();
                for t in self.toks.iter().skip(self.pos) {
                    if t.kind == TokenKind::Sym(Sym::Semi) {
                        end = t.span.start;
                        break;
                    }
                }
                let rest = self.src.get(from..end).unwrap_or_default();
                let form = format!("SET TIME ZONE {}", rest.trim_matches(|c: char| c <= ' '));
                return Self::fatal(error::unsupported_set_time_zone(
                    &form,
                    self.span_from(start),
                ));
            }
            _ => return self.fail(),
        };
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Set(Set {
            kind: SetKind::TimeZone { zone, zone_span },
            span,
        }))
    }

    /// The raw rest of a `SET key =` assignment, trimmed: decoded only
    /// when it is one backquoted token, which is how Spark lets a value
    /// hold a semicolon. An unterminated quote fails where the quote
    /// is, as in Spark. The cursor ends at the `;` or end of input.
    fn set_value(&mut self) -> PR<(String, Span)> {
        let from_pos = self.pos;
        let mut end = self.toks.len();
        for (i, t) in self.toks.iter().enumerate().skip(self.pos) {
            if t.kind == TokenKind::Sym(Sym::Semi) {
                end = i;
                break;
            }
            if t.kind == TokenKind::Unterminated {
                self.pos = i;
                return self.fail();
            }
        }
        let from = self.tok(0).map_or(self.src.len(), |t| t.span.start);
        let to = (self.pos..end)
            .filter_map(|i| self.toks.get(i))
            .map(|t| t.span.end)
            .max()
            .unwrap_or(from);
        let (value, value_span) = self.trimmed_span(from, to);
        self.pos = end;
        // One backquoted token for the whole value is decoded, as Spark
        // allows backquoted keys and values.
        if from_pos + 1 == end
            && let Some(t) = self.toks.get(from_pos)
            && t.kind == TokenKind::QuotedIdent
            && t.span.range() == value_span.range()
        {
            return Ok((unquote_ident(t.span.text(self.src)), value_span));
        }
        Ok((value, value_span))
    }

    /// The backquoted key at the cursor, decoded; `None` when the token
    /// is no backquoted key. An unterminated quote fails where the quote
    /// is, as in Spark.
    fn quoted_key(&mut self) -> PR<Option<(String, Span)>> {
        let Some(t) = self.tok(0) else {
            return Ok(None);
        };
        if t.kind == TokenKind::Unterminated {
            return self.fail();
        }
        if t.kind != TokenKind::QuotedIdent {
            return Ok(None);
        }
        self.bump();
        Ok(Some((unquote_ident(t.span.text(self.src)), t.span)))
    }

    /// `text[from..to]` trimmed the way Spark trims statement text, with
    /// its span.
    fn trimmed_span(&self, from: usize, to: usize) -> (String, Span) {
        let raw = self.src.get(from..to).unwrap_or_default();
        let trimmed = raw.trim_matches(|c: char| c <= ' ');
        let leading = raw.len() - raw.trim_start_matches(|c: char| c <= ' ').len();
        (
            trimmed.to_owned(),
            Span::new(from + leading, from + leading + trimmed.len()),
        )
    }

    /// `RESET` or `RESET key`: the key is one backquoted key or a bare
    /// key matching `[a-zA-Z_\d\.:]+`; anything else is
    /// `INVALID_RESET_COMMAND_FORMAT`.
    fn reset_command(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("RESET")?;
        if self.pos >= self.toks.len() || self.sym_at(0, Sym::Semi) {
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Reset(Reset {
                key: None,
                key_span: None,
                span,
            }));
        }
        if let Some((key, key_span)) = self.quoted_key()? {
            let span = self.span_from(start);
            self.command_end()?;
            return Ok(Statement::Reset(Reset {
                key: Some(key),
                key_span: Some(key_span),
                span,
            }));
        }
        let mut semi = self.toks.len();
        for (i, t) in self.toks.iter().enumerate().skip(self.pos) {
            if t.kind == TokenKind::Sym(Sym::Semi) {
                semi = i;
                break;
            }
            if t.kind == TokenKind::Unterminated {
                self.pos = i;
                return self.fail();
            }
        }
        let from = self.start();
        let to = (self.pos..semi)
            .filter_map(|i| self.toks.get(i))
            .map(|t| t.span.end)
            .max()
            .unwrap_or(from);
        let (key, key_span) = self.trimmed_span(from, to);
        if !valid_bare_key(&key) {
            return Self::fatal(error::invalid_reset_command_format(self.span_from(start)));
        }
        self.pos = semi;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::Reset(Reset {
            key: Some(key),
            key_span: Some(key_span),
            span,
        }))
    }

    /// `CREATE ...`: a temporary view, a table (or `TABLE ... LIKE`), or
    /// a namespace. Anything else (`CREATE VIEW`, `CREATE FUNCTION`,
    /// `CREATE GLOBAL ...`, ...) fails where it stands, as in Spark.
    fn create_command(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("CREATE")?;
        let or_replace = if self.kw_at(0, "OR") && self.kw_at(1, "REPLACE") {
            self.pos += 2;
            true
        } else {
            false
        };
        if or_replace {
            if self.eat_kw("TEMP") || self.eat_kw("TEMPORARY") {
                return self.create_temp_view_command(start, true);
            }
            if self.kw_at(0, "TABLE") {
                return self.create_table_command(start, true, false, false);
            }
            return self.fail();
        }
        let temporary = self.eat_kw("TEMP") || self.eat_kw("TEMPORARY");
        let external = self.eat_kw("EXTERNAL");
        if self.kw_at(0, "TABLE") {
            return self.create_table_command(start, false, temporary, external);
        }
        if self.kw_at(0, "VIEW") {
            if temporary && !external {
                return self.create_temp_view_command(start, false);
            }
            return self.fail();
        }
        if !temporary && !external {
            return self.create_namespace_command(start);
        }
        self.fail()
    }

    /// `CREATE [OR REPLACE] TEMPORARY VIEW [IF NOT EXISTS] name
    /// [(column [COMMENT '...'], ...)] AS query` (`TEMP` for `TEMPORARY`),
    /// after the `CREATE [OR REPLACE] TEMPORARY` at `start`. A qualified
    /// name parses and fails with `INVALID_TEMP_OBJ_QUALIFIER`;
    /// `OR REPLACE` with `IF NOT EXISTS` parses and fails with
    /// `CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE`, and `IF NOT EXISTS`
    /// without `OR REPLACE` parses and fails with `_LEGACY_ERROR_TEMP_0053`,
    /// as in Spark. The flag checks run here, on the parsed statement, so
    /// they come before the post-parse checks of the view's query and
    /// before execution (an `EXPLAIN` of such a statement fails too).
    fn create_temp_view_command(&mut self, start: usize, or_replace: bool) -> PR<Statement> {
        self.expect_kw("VIEW")?;
        let if_not_exists =
            if self.kw_at(0, "IF") && self.kw_at(1, "NOT") && self.kw_at(2, "EXISTS") {
                self.pos += 3;
                true
            } else {
                false
            };
        // Temporary views take one name part; more parts fail after the
        // statement parsed, where the name is complete.
        let first = self.any_ident()?;
        let mut extra = Vec::new();
        while self.eat_sym(Sym::Dot) {
            extra.push(self.any_ident()?);
        }
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(Self::view_column)?
        } else {
            Vec::new()
        };
        self.expect_kw("AS")?;
        let query = Box::new(self.query()?);
        let span = self.span_from(start);
        self.command_end()?;
        if !extra.is_empty() {
            let mut parts = vec![&first];
            parts.extend(extra.iter());
            let object_name: Vec<String> = parts.iter().map(|p| quote_ident(&p.value)).collect();
            let qualifier: Vec<String> = parts[..parts.len() - 1]
                .iter()
                .map(|p| quote_ident(&p.value))
                .collect();
            return Self::fatal(error::invalid_temp_obj_qualifier(
                "view",
                object_name.join("."),
                qualifier.join("."),
                span,
            ));
        }
        if if_not_exists && or_replace {
            return Self::fatal(error::create_view_with_if_not_exists_and_replace(
                quote_ident(&first.value),
                span,
            ));
        }
        if if_not_exists && !or_replace {
            return Self::fatal(error::temp_view_if_not_exists(span));
        }
        Ok(Statement::CreateView(CreateView {
            or_replace,
            if_not_exists,
            name: first,
            columns,
            query,
            span,
        }))
    }

    /// `name [COMMENT '...']` in a `CREATE VIEW` column list.
    fn view_column(&mut self) -> PR<ViewColumn> {
        let start = self.start();
        let name = self.any_ident()?;
        let comment = if self.eat_kw("COMMENT") {
            match self.string_lit() {
                Some(comment) => Some(comment),
                None => return self.fail(),
            }
        } else {
            None
        };
        Ok(ViewColumn {
            name,
            comment,
            span: self.span_from(start),
        })
    }

    /// `DROP ...`: a view, a table, or a namespace. Anything else fails
    /// where it stands, as in Spark.
    fn drop_command(&mut self) -> PR<Statement> {
        let start = self.start();
        self.expect_kw("DROP")?;
        if self.kw_at(0, "VIEW") {
            return self.drop_view_command(start);
        }
        if self.kw_at(0, "TABLE") {
            return self.drop_table_command(start);
        }
        self.drop_namespace_command(start)
    }

    /// `DROP VIEW [IF EXISTS] name`, after the `DROP` at `start`.
    fn drop_view_command(&mut self, start: usize) -> PR<Statement> {
        self.expect_kw("VIEW")?;
        let if_exists = if self.kw_at(0, "IF") && self.kw_at(1, "EXISTS") {
            self.pos += 2;
            true
        } else {
            false
        };
        let name = self.any_ident()?;
        let span = self.span_from(start);
        self.command_end()?;
        Ok(Statement::DropView(DropView {
            if_exists,
            name,
            span,
        }))
    }

    // ---- DDL: tables, namespaces, USE ------------------------------------

    /// A table, namespace or `USE` name: dotted parts or
    /// `IDENTIFIER(expr)`, which Spark's `identifierReference` allows in
    /// both identifier modes.
    fn ddl_name(&mut self) -> PR<DdlName> {
        if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            return self.identifier_name().map(DdlName::Dynamic);
        }
        self.table_name().map(DdlName::Parts)
    }

    /// A column name: `name` or `IDENTIFIER('name')` (never in legacy
    /// identifier mode, where it fails at the `(` as in Spark).
    fn column_name(&mut self) -> PR<ColumnName> {
        if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            if self.opts.legacy_identifier_clause {
                self.pos += 1;
                return self.fail();
            }
            return self.identifier_name().map(ColumnName::Dynamic);
        }
        self.name_part(true).map(ColumnName::Ident)
    }

    /// `IDENTIFIER(expr)`.
    fn identifier_name(&mut self) -> PR<DynamicName> {
        let start = self.start();
        self.expect_kw("IDENTIFIER")?;
        self.expect_sym(Sym::LParen)?;
        let expr = Box::new(self.expr()?);
        self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
        Ok(DynamicName {
            expr,
            span: self.span_from(start),
        })
    }

    /// `DATABASE`, `SCHEMA` or `NAMESPACE`.
    fn namespace_kind(&mut self) -> PR<NamespaceKind> {
        if self.eat_kw("DATABASE") {
            Ok(NamespaceKind::Database)
        } else if self.eat_kw("SCHEMA") {
            Ok(NamespaceKind::Schema)
        } else if self.eat_kw("NAMESPACE") {
            Ok(NamespaceKind::Namespace)
        } else {
            self.fail()
        }
    }

    /// The string literal after `LOCATION`, `COMMENT` or `STORED BY`
    /// (adjacent strings concatenate, as in Spark).
    fn string_after(&mut self) -> PR<String> {
        match self.string_lit() {
            Some(s) => Ok(s),
            None => self.fail(),
        }
    }

    /// `CREATE [OR REPLACE] [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS]
    /// name [(elements)] [USING provider] [clauses] [AS query]`, after the
    /// `CREATE [OR REPLACE]` at `start` (`OR REPLACE` takes no
    /// `TEMPORARY`/`EXTERNAL`/`IF NOT EXISTS`/`LIKE`, as in Spark).
    fn create_table_command(
        &mut self,
        start: usize,
        or_replace: bool,
        temporary: bool,
        external: bool,
    ) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("TABLE")?;
            let if_not_exists =
                if !or_replace && p.kw_at(0, "IF") && p.kw_at(1, "NOT") && p.kw_at(2, "EXISTS") {
                    p.pos += 3;
                    true
                } else {
                    false
                };
            let name = p.ddl_name()?;
            // `... LIKE source ...`: no columns and none of the other
            // clauses.
            if !or_replace && p.eat_kw("LIKE") {
                let source = p.ddl_name()?;
                let mut like = CreateTableLike {
                    if_not_exists,
                    name,
                    source,
                    using: None,
                    row_format: None,
                    file_format: None,
                    location: None,
                    tblproperties: Vec::new(),
                    span: Span::new(start, start),
                };
                loop {
                    if p.eat_kw("USING") {
                        like.using = Some(p.table_name()?);
                    } else if p.kw_at(0, "ROW") && p.kw_at(1, "FORMAT") {
                        like.row_format = Some(p.row_format()?);
                    } else if p.kw_at(0, "STORED") {
                        like.file_format = Some(p.file_format()?);
                    } else if p.eat_kw("LOCATION") {
                        like.location = Some(p.string_after()?);
                    } else if p.eat_kw("TBLPROPERTIES") {
                        like.tblproperties.extend(p.property_list(false)?);
                    } else {
                        break;
                    }
                }
                like.span = p.span_from(start);
                p.command_end()?;
                return Ok(Statement::CreateTableLike(Box::new(like)));
            }
            // `(columns)` or `(query)`: a parenthesized query reads as the
            // CTAS query (the query consumes its own parens), anything else
            // as a column list. ANTLR predicts a query when the `(` opens
            // one and rewinds to the column list otherwise, so `WITH` still
            // reads as a column name. No clauses follow a parenthesized
            // query.
            let (elements, paren_query) = if p.sym_at(0, Sym::LParen) && p.query_follows(1) {
                let save = p.checkpoint();
                match p.query() {
                    Ok(query) => {
                        p.commit(save);
                        (Vec::new(), Some(query))
                    }
                    Err(Fail::Syntax) => {
                        p.restore(save);
                        (p.paren_list(Self::table_element)?, None)
                    }
                    Err(fatal) => return Err(fatal),
                }
            } else if p.sym_at(0, Sym::LParen) {
                (p.paren_list(Self::table_element)?, None)
            } else {
                (Vec::new(), None)
            };
            let using = if paren_query.is_none() && p.eat_kw("USING") {
                Some(p.table_name()?)
            } else {
                None
            };
            let mut options = Vec::new();
            let mut partitioned_by = Vec::new();
            let mut cluster_by = Vec::new();
            let mut bucket = None;
            let mut skew = None;
            let mut row_format = None;
            let mut file_format = None;
            let mut location = None;
            let mut comment = None;
            let mut collation = None;
            let mut tblproperties = Vec::new();
            // Occurrences per clause, in Spark's duplicate-check order.
            let mut seen = [0_usize; CREATE_TABLE_CLAUSES.len()];
            loop {
                if paren_query.is_some() {
                    break;
                }
                let clause = if p.kw_at(0, "TBLPROPERTIES") {
                    Some("TBLPROPERTIES")
                } else if p.kw_at(0, "OPTIONS") {
                    Some("OPTIONS")
                } else if p.kw_at(0, "PARTITIONED") && p.kw_at(1, "BY") {
                    Some("PARTITIONED BY")
                } else if p.kw_at(0, "STORED") {
                    Some("STORED AS/BY")
                } else if p.kw_at(0, "ROW") && p.kw_at(1, "FORMAT") {
                    Some("ROW FORMAT")
                } else if p.kw_at(0, "COMMENT") {
                    Some("COMMENT")
                } else if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                    Some("DEFAULT COLLATION")
                } else if p.kw_at(0, "CLUSTERED") {
                    Some("CLUSTERED BY")
                } else if p.kw_at(0, "CLUSTER") && p.kw_at(1, "BY") {
                    Some("CLUSTER BY")
                } else if p.kw_at(0, "LOCATION") {
                    Some("LOCATION")
                } else {
                    None
                };
                if let Some(i) =
                    clause.and_then(|c| CREATE_TABLE_CLAUSES.iter().position(|k| *k == c))
                {
                    seen[i] += 1;
                }
                if p.eat_kw("OPTIONS") {
                    options.extend(p.property_list(true)?);
                } else if p.kw_at(0, "PARTITIONED") && p.kw_at(1, "BY") {
                    p.pos += 2;
                    partitioned_by.extend(p.paren_list(Self::partition_field)?);
                } else if p.kw_at(0, "SKEWED") {
                    skew = Some(p.skew_spec()?);
                } else if p.kw_at(0, "CLUSTERED") {
                    bucket = Some(p.bucket_spec()?);
                } else if p.kw_at(0, "CLUSTER") && p.kw_at(1, "BY") {
                    p.pos += 2;
                    cluster_by.extend(p.paren_list(Self::table_name)?);
                } else if p.kw_at(0, "ROW") && p.kw_at(1, "FORMAT") {
                    row_format = Some(p.row_format()?);
                } else if p.kw_at(0, "STORED") {
                    file_format = Some(p.file_format()?);
                } else if p.eat_kw("LOCATION") {
                    location = Some(p.string_after()?);
                } else if p.eat_kw("COMMENT") {
                    comment = Some(p.string_after()?);
                } else if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                    p.pos += 2;
                    collation = Some(p.table_name()?);
                } else if p.eat_kw("TBLPROPERTIES") {
                    tblproperties.extend(p.property_list(false)?);
                } else {
                    break;
                }
            }
            // `[AS] query`: a CTAS (or the parenthesized query above).
            let as_query = (if let Some(query) = paren_query {
                Some(query)
            } else if p.eat_kw("AS")
                || p.kw_at(0, "SELECT")
                || p.kw_at(0, "WITH")
                || p.kw_at(0, "VALUES")
                || p.kw_at(0, "TABLE")
                || p.kw_at(0, "FROM")
                || (p.sym_at(0, Sym::LParen) && p.query_follows(1))
            {
                Some(p.query()?)
            } else {
                None
            })
            .map(Box::new);
            let duplicate_clause = CREATE_TABLE_CLAUSES
                .iter()
                .zip(seen)
                .find(|(_, n)| *n > 1)
                .map(|(c, _)| *c);
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::CreateTable(Box::new(CreateTable {
                or_replace,
                temporary,
                external,
                if_not_exists,
                name,
                elements,
                using,
                options,
                partitioned_by,
                cluster_by,
                bucket,
                skew,
                row_format,
                file_format,
                location,
                comment,
                collation,
                tblproperties,
                as_query,
                duplicate_clause,
                span,
            })))
        })
    }

    /// A `CREATE TABLE` element: a table constraint, or a column
    /// definition. The constraint keywords are never column names, so a
    /// constraint-starting element is committed to.
    fn table_element(&mut self) -> PR<TableElement> {
        if self.kw_at(0, "CONSTRAINT")
            || self.any_kw_at(0, &["PRIMARY", "FOREIGN", "CHECK", "UNIQUE"])
        {
            return self.table_constraint().map(TableElement::Constraint);
        }
        self.table_column()
            .map(|c| TableElement::Column(Box::new(c)))
    }

    /// `name type [NOT NULL] [DEFAULT ...] [GENERATED ...] [COMMENT '...']
    /// [constraints]`.
    fn table_column(&mut self) -> PR<ColumnDef> {
        let start = self.start();
        let name = self.column_name()?;
        let data_type = self.data_type()?;
        let mut not_null = false;
        let mut default = None;
        let mut generated = None;
        let mut comment = None;
        let mut constraints = Vec::new();
        // Descriptors seen so far, and the first one written twice.
        let mut seen: Vec<&'static str> = Vec::new();
        let mut duplicate_descriptor = None;
        loop {
            let descriptor = if self.kw_at(0, "NOT") {
                Some("NOT NULL")
            } else if self.kw_at(0, "DEFAULT") {
                Some("DEFAULT")
            } else if self.kw_at(0, "GENERATED") {
                Some("GENERATED ALWAYS AS")
            } else if self.kw_at(0, "COMMENT") {
                Some("COMMENT")
            } else if self.kw_at(0, "CONSTRAINT")
                || self.any_kw_at(0, &["PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "REFERENCES"])
            {
                Some("CONSTRAINT")
            } else {
                None
            };
            if let Some(d) = descriptor {
                if seen.contains(&d) {
                    duplicate_descriptor = duplicate_descriptor.or(Some(d));
                } else {
                    seen.push(d);
                }
            }
            if self.eat_kw("NOT") {
                not_null = true;
                if !self.eat_kw("NULL") {
                    // ANTLR inserts the missing `NULL` when what follows
                    // fits after it.
                    if self.kind(0).is_none()
                        || self.sym_at(0, Sym::Comma)
                        || self.sym_at(0, Sym::RParen)
                        || self.kw_at(0, "COMMENT")
                    {
                        self.missing = Some((self.pos, "'NULL'"));
                    }
                    return self.fail();
                }
            } else if self.eat_kw("DEFAULT") {
                default = Some(self.expr()?);
            } else if self.kw_at(0, "GENERATED") {
                generated = Some(self.generated_column()?);
            } else if self.eat_kw("COMMENT") {
                comment = Some(match self.string_lit() {
                    Some(comment) => comment,
                    None => return self.fail(),
                });
            } else if self.kw_at(0, "CONSTRAINT")
                || self.any_kw_at(0, &["PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "REFERENCES"])
            {
                constraints.push(self.column_constraint()?);
            } else {
                break;
            }
        }
        Ok(ColumnDef {
            name,
            data_type,
            not_null,
            default,
            generated,
            comment,
            constraints,
            duplicate_descriptor,
            span: self.span_from(start),
        })
    }

    /// `GENERATED ALWAYS AS (expr)` or `GENERATED (ALWAYS | BY DEFAULT)
    /// AS IDENTITY [(START WITH n INCREMENT BY m)]`.
    fn generated_column(&mut self) -> PR<GeneratedColumn> {
        let start = self.start();
        self.expect_kw("GENERATED")?;
        let kind = if self.eat_kw("ALWAYS") {
            self.expect_kw("AS")?;
            if self.kw_at(0, "IDENTITY") {
                let ident = self.start();
                self.bump();
                GeneratedKind::Identity(self.identity_options(IdentityHow::Always, ident)?)
            } else {
                self.expect_sym(Sym::LParen)?;
                let expr = self.expr()?;
                self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
                GeneratedKind::Expr(expr)
            }
        } else if self.kw_at(0, "BY") {
            self.pos += 1;
            self.expect_kw("DEFAULT")?;
            self.expect_kw("AS")?;
            let ident = self.start();
            self.expect_kw("IDENTITY")?;
            GeneratedKind::Identity(self.identity_options(IdentityHow::ByDefault, ident)?)
        } else {
            return self.fail();
        };
        Ok(GeneratedColumn {
            kind,
            span: self.span_from(start),
        })
    }

    /// `[(START WITH n INCREMENT BY m)]` after `IDENTITY` at `ident`.
    fn identity_options(&mut self, how: IdentityHow, ident: usize) -> PR<IdentityColumn> {
        let (mut start, mut step) = (None, None);
        let mut duplicate_option = None;
        let mut count = 0;
        if self.eat_sym(Sym::LParen) {
            loop {
                if self.kw_at(0, "START") && self.kw_at(1, "WITH") {
                    self.pos += 2;
                    if start.is_some() {
                        duplicate_option = duplicate_option.or(Some("START"));
                    }
                    start = Some(self.identity_count()?);
                    count += 1;
                } else if self.kw_at(0, "INCREMENT") && self.kw_at(1, "BY") {
                    self.pos += 2;
                    if step.is_some() {
                        duplicate_option = duplicate_option.or(Some("STEP"));
                    }
                    step = Some(self.identity_count()?);
                    count += 1;
                } else {
                    break;
                }
            }
            self.expect_closing(Sym::RParen, Close::List(count))?;
        }
        Ok(IdentityColumn {
            how,
            start,
            step,
            duplicate_option,
            span: self.span_from(ident),
        })
    }

    /// A `START WITH` / `INCREMENT BY` count: an integer, optionally
    /// negative.
    fn identity_count(&mut self) -> PR<i64> {
        let negative = self.sym_at(0, Sym::Minus);
        let at = usize::from(negative);
        if self.kind(at) != Some(TokenKind::Number(NumKind::Int)) {
            return self.fail();
        }
        let text = if negative {
            format!("-{}", self.text(at))
        } else {
            self.text(at).to_owned()
        };
        match text.parse() {
            Ok(n) => {
                self.pos += at + 1;
                Ok(n)
            }
            Err(_) => self.fail(),
        }
    }

    /// `[CONSTRAINT name] (PRIMARY KEY ... | FOREIGN KEY ... | UNIQUE ...
    /// | CHECK ...) [characteristics]`.
    fn table_constraint(&mut self) -> PR<TableConstraint> {
        let start = self.start();
        let name = if self.eat_kw("CONSTRAINT") {
            Some(self.name_part(true)?)
        } else {
            None
        };
        let kind = if self.kw_at(0, "PRIMARY") && self.kw_at(1, "KEY") {
            self.pos += 2;
            TableConstraintKind::PrimaryKey(self.paren_list(|p| p.name_part(true))?)
        } else if self.kw_at(0, "FOREIGN") && self.kw_at(1, "KEY") {
            let fk = self.start();
            self.pos += 2;
            let columns = self.paren_list(|p| p.name_part(true))?;
            let (table, ref_columns) = self.foreign_key_ref()?;
            TableConstraintKind::ForeignKey(ForeignKeyConstraint {
                columns,
                table,
                ref_columns,
                span: self.span_from(fk),
            })
        } else if self.eat_kw("UNIQUE") {
            TableConstraintKind::Unique(self.paren_list(|p| p.name_part(true))?)
        } else if self.eat_kw("CHECK") {
            self.expect_sym(Sym::LParen)?;
            let expr = self.expr()?;
            self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
            TableConstraintKind::Check(expr)
        } else {
            return self.fail();
        };
        let characteristics = self.constraint_characteristics()?;
        Ok(TableConstraint {
            name,
            kind,
            characteristics,
            span: self.span_from(start),
        })
    }

    /// `[CONSTRAINT name] (PRIMARY KEY | UNIQUE | REFERENCES ... | CHECK
    /// ...) [characteristics]` on a column.
    fn column_constraint(&mut self) -> PR<ColumnConstraint> {
        let start = self.start();
        let name = if self.eat_kw("CONSTRAINT") {
            Some(self.name_part(true)?)
        } else {
            None
        };
        let kind = if self.kw_at(0, "PRIMARY") && self.kw_at(1, "KEY") {
            self.pos += 2;
            ColumnConstraintKind::PrimaryKey
        } else if self.eat_kw("UNIQUE") {
            ColumnConstraintKind::Unique
        } else if self.eat_kw("REFERENCES") {
            let start = self.start();
            let table = self.table_name()?;
            let columns = if self.sym_at(0, Sym::LParen) {
                self.paren_list(|p| p.name_part(true))?
            } else {
                Vec::new()
            };
            ColumnConstraintKind::References(ForeignKeyRef {
                table,
                columns,
                span: self.span_from(start),
            })
        } else if self.eat_kw("CHECK") {
            self.expect_sym(Sym::LParen)?;
            let expr = self.expr()?;
            self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
            ColumnConstraintKind::Check(expr)
        } else {
            return self.fail();
        };
        let characteristics = self.constraint_characteristics()?;
        Ok(ColumnConstraint {
            name,
            kind,
            characteristics,
            span: self.span_from(start),
        })
    }

    /// `REFERENCES table [(columns)]` after `FOREIGN KEY (columns)`.
    fn foreign_key_ref(&mut self) -> PR<(ObjectName, Vec<Ident>)> {
        self.expect_kw("REFERENCES")?;
        let table = self.table_name()?;
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(|p| p.name_part(true))?
        } else {
            Vec::new()
        };
        Ok((table, columns))
    }

    /// `ENFORCED`, `NOT ENFORCED`, `RELY` and `NORELY`, in any order.
    fn constraint_characteristics(&mut self) -> PR<Vec<ConstraintCharacteristic>> {
        let mut out = Vec::new();
        loop {
            if self.eat_kw("ENFORCED") {
                out.push(ConstraintCharacteristic::Enforced(true));
            } else if self.kw_at(0, "NOT") && self.kw_at(1, "ENFORCED") {
                self.pos += 2;
                out.push(ConstraintCharacteristic::Enforced(false));
            } else if self.eat_kw("RELY") {
                out.push(ConstraintCharacteristic::Rely(true));
            } else if self.kw_at(0, "NORELY") {
                self.bump();
                out.push(ConstraintCharacteristic::Rely(false));
            } else {
                break;
            }
        }
        Ok(out)
    }

    /// `(key [=] value, ...)` for `OPTIONS` (full expressions) and
    /// `TBLPROPERTIES` / `DBPROPERTIES` (literals only): `expressions`
    /// selects which.
    fn property_list(&mut self, expressions: bool) -> PR<Vec<TableOption>> {
        self.paren_list(|p| {
            let start = p.start();
            let key = p.option_key()?;
            let has_eq = p.eat_sym(Sym::Eq);
            let value = if has_eq || !(p.sym_at(0, Sym::Comma) || p.sym_at(0, Sym::RParen)) {
                if expressions {
                    Some(p.expr()?)
                } else {
                    Some(p.property_value()?)
                }
            } else {
                None
            };
            Ok(TableOption {
                key,
                value,
                span: p.span_from(start),
            })
        })
    }

    /// A `TBLPROPERTIES` / `DBPROPERTIES` value: a number, string or
    /// boolean literal.
    fn property_value(&mut self) -> PR<Expr> {
        let start = self.start();
        let kind = match self.kind(0) {
            Some(TokenKind::Number(kind @ (NumKind::Int | NumKind::Decimal))) => {
                let text = self.text(0).to_owned();
                self.bump();
                Literal::Number { text, kind }
            }
            Some(TokenKind::String) => Literal::String(self.string_lit().unwrap_or_default()),
            Some(TokenKind::DoubleQuoted) if !self.opts.double_quoted_identifiers => {
                Literal::String(self.string_lit().unwrap_or_default())
            }
            Some(TokenKind::Word) if self.kw_at(0, "TRUE") || self.kw_at(0, "FALSE") => {
                let value = self.kw_at(0, "TRUE");
                self.bump();
                Literal::Boolean(value)
            }
            _ => return self.fail(),
        };
        Ok(Expr {
            kind: ExprKind::Literal(kind),
            span: self.span_from(start),
        })
    }

    /// One `PARTITIONED BY` element: `name [(args)]`, or `name type [NOT
    /// NULL] [COMMENT '...']` for a single plain name. A
    /// `PARTITIONED BY` column takes no `COLLATE`: one reads as an
    /// identity transform instead, which fails at the `COLLATE`.
    /// (Applied and dotted dynamic names are not supported.)
    fn partition_field(&mut self) -> PR<PartitionField> {
        let start = self.start();
        let name = if self.kw_at(0, "IDENTIFIER") && self.sym_at(1, Sym::LParen) {
            DdlName::Dynamic(self.identifier_name()?)
        } else {
            let first = self.name_part(true)?;
            if self.sym_at(0, Sym::LParen) {
                return self.partition_apply_transform(start, first);
            }
            let mut parts = vec![first];
            while self.eat_sym(Sym::Dot) {
                parts.push(self.name_part(true)?);
            }
            let span = self.span_from(start);
            DdlName::Parts(ObjectName { parts, span })
        };
        // An identity transform, or a column when a single name takes a
        // type.
        if let DdlName::Parts(name) = &name
            && name.parts.len() == 1
            && matches!(
                self.kind(0),
                Some(TokenKind::Word | TokenKind::QuotedIdent | TokenKind::DoubleQuoted)
                if self.kind(0) != Some(TokenKind::DoubleQuoted)
                    || self.opts.double_quoted_identifiers
            )
        {
            let save = self.pos;
            let (far, extra, missing, matched) = (self.far, self.extra, self.missing, self.matched);
            match self.data_type() {
                Ok(data_type) => {
                    if data_type.collation.is_none() {
                        let mut not_null = false;
                        if self.eat_kw("NOT") {
                            not_null = true;
                            if !self.eat_kw("NULL") {
                                if self.kind(0).is_none()
                                    || self.sym_at(0, Sym::Comma)
                                    || self.sym_at(0, Sym::RParen)
                                    || self.kw_at(0, "COMMENT")
                                {
                                    self.missing = Some((self.pos, "'NULL'"));
                                }
                                return self.fail();
                            }
                        }
                        let comment = if self.eat_kw("COMMENT") {
                            match self.string_lit() {
                                Some(comment) => Some(comment),
                                None => return self.fail(),
                            }
                        } else {
                            None
                        };
                        let span = self.span_from(start);
                        return Ok(PartitionField {
                            kind: PartitionFieldKind::Column(PartitionColumn {
                                name: ColumnName::Ident(name.parts[0].clone()),
                                data_type,
                                not_null,
                                comment,
                                span,
                            }),
                            span,
                        });
                    }
                    self.pos = save;
                    (self.far, self.extra, self.missing, self.matched) =
                        (far, extra, missing, matched);
                }
                Err(Fail::Syntax) => {
                    self.pos = save;
                    (self.far, self.extra, self.missing, self.matched) =
                        (far, extra, missing, matched);
                }
                Err(fatal) => return Err(fatal),
            }
        }
        // An identity transform.
        let span = self.span_from(start);
        Ok(PartitionField {
            kind: PartitionFieldKind::Transform(PartitionTransform {
                name,
                args: Vec::new(),
                span,
            }),
            span,
        })
    }

    /// `name(args)` in `PARTITIONED BY`.
    fn partition_apply_transform(&mut self, start: usize, first: Ident) -> PR<PartitionField> {
        let name_span = first.span;
        self.expect_sym(Sym::LParen)?;
        let args = self.comma_sep(Self::primary_expr)?;
        self.expect_closing(Sym::RParen, Close::List(args.len()))?;
        let span = self.span_from(start);
        Ok(PartitionField {
            kind: PartitionFieldKind::Transform(PartitionTransform {
                name: DdlName::Parts(ObjectName {
                    parts: vec![first],
                    span: name_span,
                }),
                args,
                span,
            }),
            span,
        })
    }

    /// `CLUSTERED BY (...) [SORTED BY (...)] INTO n BUCKETS`.
    fn bucket_spec(&mut self) -> PR<BucketSpec> {
        let start = self.start();
        self.expect_kw("CLUSTERED")?;
        self.expect_kw("BY")?;
        let clustered_by = self.paren_list(|p| p.name_part(true))?;
        let mut sorted_by = Vec::new();
        if self.kw_at(0, "SORTED") && self.kw_at(1, "BY") {
            self.pos += 2;
            sorted_by = self.paren_list(Self::ordered_ident)?;
        }
        self.expect_kw("INTO")?;
        let buckets = self.bucket_count()?;
        self.expect_kw("BUCKETS")?;
        Ok(BucketSpec {
            clustered_by,
            sorted_by,
            buckets,
            span: self.span_from(start),
        })
    }

    /// `name [ASC | DESC]`.
    fn ordered_ident(&mut self) -> PR<OrderedIdent> {
        let start = self.start();
        let name = self.name_part(true)?;
        let asc = if self.eat_kw("ASC") {
            Some(true)
        } else if self.eat_kw("DESC") {
            Some(false)
        } else {
            None
        };
        Ok(OrderedIdent {
            name,
            asc,
            span: self.span_from(start),
        })
    }

    /// `INTO n`: a plain integer token.
    fn bucket_count(&mut self) -> PR<u64> {
        let text = match self.kind(0) {
            Some(TokenKind::Number(NumKind::Int)) => self.text(0).to_owned(),
            _ => return self.fail(),
        };
        match text.parse() {
            Ok(n) => {
                self.bump();
                Ok(n)
            }
            Err(_) => self.fail(),
        }
    }

    /// `SKEWED BY (...) ON (...) [STORED AS DIRECTORIES]`.
    fn skew_spec(&mut self) -> PR<SkewSpec> {
        let start = self.start();
        self.expect_kw("SKEWED")?;
        self.expect_kw("BY")?;
        let skewed_by = self.paren_list(|p| p.name_part(true))?;
        self.expect_kw("ON")?;
        // `(...)` holds one value group, `((...), ...)` several.
        let values = if self.sym_at(0, Sym::LParen) && self.sym_at(1, Sym::LParen) {
            self.paren_list(|p| p.paren_list(Self::primary_expr))?
        } else {
            vec![self.paren_list(Self::primary_expr)?]
        };
        let mut stored_as_directories = false;
        if self.eat_kw("STORED") {
            self.expect_kw("AS")?;
            self.expect_kw("DIRECTORIES")?;
            stored_as_directories = true;
        }
        Ok(SkewSpec {
            skewed_by,
            values,
            stored_as_directories,
            span: self.span_from(start),
        })
    }

    /// `ROW FORMAT SERDE ...` or `ROW FORMAT DELIMITED ...`.
    fn row_format(&mut self) -> PR<RowFormat> {
        let start = self.start();
        self.expect_kw("ROW")?;
        self.expect_kw("FORMAT")?;
        let kind = if self.eat_kw("SERDE") {
            let serde = self.start();
            let class = self.string_after()?;
            let properties = if self.kw_at(0, "WITH") && self.kw_at(1, "SERDEPROPERTIES") {
                self.pos += 2;
                self.property_list(false)?
            } else {
                Vec::new()
            };
            RowFormatKind::Serde(RowSerde {
                class,
                properties,
                span: self.span_from(serde),
            })
        } else if self.kw_at(0, "DELIMITED") {
            let delimited_start = self.start();
            self.bump();
            // `ROW FORMAT DELIMITED` on its own: every clause is empty.
            let mut delimited = RowDelimited {
                fields_terminated_by: None,
                escaped_by: None,
                collection_items_terminated_by: None,
                map_keys_terminated_by: None,
                lines_terminated_by: None,
                null_defined_as: None,
                span: Span::new(delimited_start, delimited_start),
            };
            if self.kw_at(0, "FIELDS") && self.kw_at(1, "TERMINATED") && self.kw_at(2, "BY") {
                self.pos += 3;
                delimited.fields_terminated_by = Some(self.string_after()?);
                if self.kw_at(0, "ESCAPED") && self.kw_at(1, "BY") {
                    self.pos += 2;
                    delimited.escaped_by = Some(self.string_after()?);
                }
            }
            if self.kw_at(0, "COLLECTION")
                && self.kw_at(1, "ITEMS")
                && self.kw_at(2, "TERMINATED")
                && self.kw_at(3, "BY")
            {
                self.pos += 4;
                delimited.collection_items_terminated_by = Some(self.string_after()?);
            }
            if self.kw_at(0, "MAP")
                && self.kw_at(1, "KEYS")
                && self.kw_at(2, "TERMINATED")
                && self.kw_at(3, "BY")
            {
                self.pos += 4;
                delimited.map_keys_terminated_by = Some(self.string_after()?);
            }
            if self.kw_at(0, "LINES") && self.kw_at(1, "TERMINATED") && self.kw_at(2, "BY") {
                self.pos += 3;
                delimited.lines_terminated_by = Some(self.string_after()?);
            }
            if self.kw_at(0, "NULL") && self.kw_at(1, "DEFINED") && self.kw_at(2, "AS") {
                self.pos += 3;
                delimited.null_defined_as = Some(self.string_after()?);
            }
            delimited.span = self.span_from(delimited_start);
            RowFormatKind::Delimited(delimited)
        } else {
            return self.fail();
        };
        Ok(RowFormat {
            kind,
            span: self.span_from(start),
        })
    }

    /// `STORED AS name` or `STORED BY 'handler'`.
    fn file_format(&mut self) -> PR<FileFormat> {
        let start = self.start();
        self.expect_kw("STORED")?;
        let kind = if self.eat_kw("AS") {
            FileFormatKind::StoredAs(self.table_name()?)
        } else if self.eat_kw("BY") {
            FileFormatKind::StoredBy(self.string_after()?)
        } else {
            return self.fail();
        };
        Ok(FileFormat {
            kind,
            span: self.span_from(start),
        })
    }

    /// `CREATE DATABASE|SCHEMA|NAMESPACE [IF NOT EXISTS] name
    /// [COMMENT ...] [LOCATION ...] [DEFAULT COLLATION ...]
    /// [WITH DBPROPERTIES|PROPERTIES (...)]`, after the `CREATE` at
    /// `start`.
    fn create_namespace_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            let kind = p.namespace_kind()?;
            let if_not_exists = if p.kw_at(0, "IF") && p.kw_at(1, "NOT") && p.kw_at(2, "EXISTS") {
                p.pos += 3;
                true
            } else {
                false
            };
            let name = p.ddl_name()?;
            let mut comment = None;
            let mut location = None;
            let mut collation = None;
            let mut properties = Vec::new();
            loop {
                if p.eat_kw("COMMENT") {
                    comment = Some(p.string_after()?);
                } else if p.eat_kw("LOCATION") {
                    location = Some(p.string_after()?);
                } else if p.kw_at(0, "DEFAULT") && p.kw_at(1, "COLLATION") {
                    p.pos += 2;
                    collation = Some(p.table_name()?);
                } else if p.eat_kw("WITH") {
                    if p.eat_kw("DBPROPERTIES") || p.eat_kw("PROPERTIES") {
                        properties.extend(p.property_list(false)?);
                    } else {
                        return p.fail();
                    }
                } else {
                    break;
                }
            }
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::CreateNamespace(CreateNamespace {
                kind,
                if_not_exists,
                name,
                comment,
                location,
                collation,
                properties,
                span,
            }))
        })
    }

    /// `DROP TABLE [IF EXISTS] name [PURGE]`, after the `DROP` at `start`.
    fn drop_table_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            p.expect_kw("TABLE")?;
            let if_exists = if p.kw_at(0, "IF") && p.kw_at(1, "EXISTS") {
                p.pos += 2;
                true
            } else {
                false
            };
            let name = p.ddl_name()?;
            let purge = p.eat_kw("PURGE");
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::DropTable(DropTable {
                if_exists,
                name,
                purge,
                span,
            }))
        })
    }

    /// `DROP DATABASE|SCHEMA|NAMESPACE [IF EXISTS] name
    /// [RESTRICT|CASCADE]`, after the `DROP` at `start`.
    fn drop_namespace_command(&mut self, start: usize) -> PR<Statement> {
        self.nested(|p| {
            let kind = p.namespace_kind()?;
            let if_exists = if p.kw_at(0, "IF") && p.kw_at(1, "EXISTS") {
                p.pos += 2;
                true
            } else {
                false
            };
            let name = p.ddl_name()?;
            let behavior = if p.eat_kw("RESTRICT") {
                Some(DropBehavior::Restrict)
            } else if p.eat_kw("CASCADE") {
                Some(DropBehavior::Cascade)
            } else {
                None
            };
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::DropNamespace(DropNamespace {
                kind,
                if_exists,
                name,
                behavior,
                span,
            }))
        })
    }

    /// `USE name` or `USE DATABASE|SCHEMA|NAMESPACE name`.
    fn use_command(&mut self) -> PR<Statement> {
        self.nested(|p| {
            let start = p.start();
            p.expect_kw("USE")?;
            let kind = if p.any_kw_at(0, &["DATABASE", "SCHEMA", "NAMESPACE"]) {
                Some(p.namespace_kind()?)
            } else {
                None
            };
            let name = p.ddl_name()?;
            let span = p.span_from(start);
            p.command_end()?;
            Ok(Statement::Use(Use { kind, name, span }))
        })
    }

    /// A query; a failure at a position is remembered, so the subquery
    /// attempts of nested parentheses never repeat work.
    pub(crate) fn query(&mut self) -> PR<Query> {
        let pos = self.pos;
        if let Some(&far) = self.failed_queries.get(&pos) {
            self.far = self.far.max(far);
            return Err(Fail::Syntax);
        }
        let r = self.query_at();
        if matches!(r, Err(Fail::Syntax)) {
            self.failed_queries.insert(pos, self.far);
        }
        r
    }

    fn query_at(&mut self) -> PR<Query> {
        self.nested(|p| {
            let start = p.start();
            let with = if p.kw_at(0, "WITH") {
                Some(p.with_clause()?)
            } else {
                None
            };
            let body = p.query_term()?;
            let mut q = Self::bare_query(body, start);
            q.with = with;
            p.organization(&mut q)?;
            q.span = p.span_from(start);
            Ok(q)
        })
    }

    fn bare_query(body: SetExpr, start: usize) -> Query {
        Query {
            with: None,
            span: Span::new(start, body.span.end),
            body,
            order_by: vec![],
            order_span: None,
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            windows: vec![],
            limit: None,
            offset: None,
        }
    }

    /// `queryOrganization`.
    #[inline(never)]
    fn organization(&mut self, q: &mut Query) -> PR<()> {
        if self.kw_at(0, "ORDER") && self.kw_at(1, "BY") {
            let start = self.start();
            self.pos += 2;
            q.order_by = self.comma_sep(|p| p.sort_item(true))?;
            q.order_span = Some(self.span_from(start));
        }
        for (kw, list) in [
            ("CLUSTER", &mut q.cluster_by),
            ("DISTRIBUTE", &mut q.distribute_by),
        ] {
            if self.kw_at(0, kw) && self.kw_at(1, "BY") {
                self.pos += 2;
                *list = self.comma_sep(Self::expr)?;
            }
        }
        if self.kw_at(0, "SORT") && self.kw_at(1, "BY") {
            self.pos += 2;
            q.sort_by = self.comma_sep(|p| p.sort_item(false))?;
        }
        q.windows = self.window_clause()?;
        let start = self.start();
        if self.eat_kw("LIMIT") {
            let value = if self.eat_kw("ALL") {
                None
            } else {
                Some(self.expr()?)
            };
            q.limit = Some(Limit {
                value,
                span: self.span_from(start),
            });
        }
        if self.eat_kw("OFFSET") {
            q.offset = Some(self.expr()?);
        }
        Ok(())
    }

    #[inline(never)]
    fn with_clause(&mut self) -> PR<With> {
        let start = self.start();
        self.expect_kw("WITH")?;
        let recursive = self.eat_kw("RECURSIVE");
        let ctes = self.comma_sep(|p| {
            let start = p.start();
            let name = p.name_part(true)?;
            let columns = if p.sym_at(0, Sym::LParen) {
                p.ident_list()?
            } else {
                vec![]
            };
            let mut max_recursion_level = None;
            if p.kw_at(0, "MAX") && p.kw_at(1, "RECURSION") && p.kw_at(2, "LEVEL") {
                p.pos += 3;
                if p.kind(0) != Some(TokenKind::Number(crate::ast::NumKind::Int)) {
                    return p.fail();
                }
                max_recursion_level = Some(p.text(0).to_owned());
                p.bump();
            }
            p.eat_kw("AS");
            p.expect_sym(Sym::LParen)?;
            let query = Box::new(p.query()?);
            p.expect_sym(Sym::RParen)?;
            Ok(Cte {
                name,
                columns,
                max_recursion_level,
                query,
                span: p.span_from(start),
            })
        })?;
        Ok(With {
            recursive,
            ctes,
            span: self.span_from(start),
        })
    }

    /// Set operations: `INTERSECT` binds tighter than `UNION` / `EXCEPT`
    /// unless `spark.sql.legacy.setopsPrecedence.enabled`; all are
    /// left-associative.
    fn query_term(&mut self) -> PR<SetExpr> {
        self.set_ops(0)
    }

    fn set_ops(&mut self, min_prec: u8) -> PR<SetExpr> {
        let (mut left, mut height) = self.measured(Self::query_primary)?;
        loop {
            let (op, prec) = if self.kw_at(0, "INTERSECT") {
                (SetOperator::Intersect, 2)
            } else if self.kw_at(0, "UNION") {
                (SetOperator::Union, 1)
            } else if self.any_kw_at(0, &["EXCEPT", "MINUS"]) {
                (SetOperator::Except, 1)
            } else {
                break;
            };
            let prec = if self.opts.legacy_setops_precedence {
                1
            } else {
                prec
            };
            if prec < min_prec {
                break;
            }
            self.bump();
            let quantifier = self.set_quantifier();
            let (right, right_height) = self.measured(|p| p.nested(|p| p.set_ops(prec + 1)))?;
            height = (height + 1).max(right_height);
            self.grow(height)?;
            let span = left.span.to(right.span);
            left = SetExpr {
                kind: SetExprKind::SetOp {
                    op,
                    quantifier,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn set_quantifier(&mut self) -> Option<SetQuantifier> {
        if self.eat_kw("DISTINCT") {
            Some(SetQuantifier::Distinct)
        } else if self.eat_kw("ALL") {
            Some(SetQuantifier::All)
        } else {
            None
        }
    }

    fn query_primary(&mut self) -> PR<SetExpr> {
        let start = self.start();
        let kind = if self.kw_at(0, "SELECT") {
            SetExprKind::Select(Box::new(self.select(start, true)?))
        } else if self.kw_at(0, "FROM") {
            return self.query_from_first();
        } else if self.kw_at(0, "VALUES") {
            SetExprKind::Values(self.values()?)
        } else if self.eat_kw("TABLE") {
            SetExprKind::Table(self.object_name()?)
        } else if self.eat_sym(Sym::LParen) {
            let q = self.query()?;
            self.expect_sym(Sym::RParen)?;
            SetExprKind::Query(Box::new(q))
        } else {
            return self.fail();
        };
        Ok(SetExpr {
            kind,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    fn values(&mut self) -> PR<Values> {
        let start = self.start();
        self.expect_kw("VALUES")?;
        let rows = self.comma_sep(Self::expr)?;
        let alias = self.table_alias()?;
        Ok(Values {
            rows,
            alias,
            span: self.span_from(start),
        })
    }

    /// `FROM relations [SELECT ... [organization]]*`. The `FROM` clause is
    /// parsed and stored once, however many bodies read it.
    #[inline(never)]
    fn query_from_first(&mut self) -> PR<SetExpr> {
        let start = self.start();
        let from = self.clause_from()?;
        let mut bodies = vec![];
        while self.kw_at(0, "SELECT") {
            let body_start = self.start();
            let select = self.select(body_start, false)?;
            let span = select.span;
            let mut q = Self::bare_query(
                SetExpr {
                    kind: SetExprKind::Select(Box::new(select)),
                    span,
                },
                body_start,
            );
            self.organization(&mut q)?;
            q.span = self.span_from(body_start);
            bodies.push(q);
        }
        let span = self.span_from(start);
        Ok(SetExpr {
            kind: SetExprKind::FromStatement(Box::new(FromStatement { from, bodies, span })),
            span,
        })
    }

    /// `FROM relation, ... lateralView* pivot? unpivot?`.
    #[inline(never)]
    fn clause_from(&mut self) -> PR<FromClause> {
        let start = self.start();
        self.expect_kw("FROM")?;
        let relations = self.comma_sep(Self::relation)?;
        let lateral_views = self.lateral_views()?;
        let pivot = if self.kw_at(0, "PIVOT") {
            Some(self.pivot()?)
        } else {
            None
        };
        let unpivot = if self.kw_at(0, "UNPIVOT") {
            Some(self.unpivot()?)
        } else {
            None
        };
        Ok(FromClause {
            relations,
            lateral_views,
            pivot,
            unpivot,
            span: self.span_from(start),
        })
    }

    /// A query specification from `SELECT` on; `with_from` reads a `FROM`
    /// clause (a body of a `FROM ... SELECT` statement has none).
    fn select(&mut self, start: usize, with_from: bool) -> PR<Select> {
        self.expect_kw("SELECT")?;
        let mut hints = vec![];
        while self.kind(0) == Some(TokenKind::HintStart) {
            self.hints(&mut hints)?;
        }
        // `SELECT all, b`: `ALL` / `DISTINCT` before `,` or `FROM` is a column.
        let column_next = !self.opts.enforce_reserved_keywords
            && (self.sym_at(1, Sym::Comma) || self.kw_at(1, "FROM") || self.tok(1).is_none());
        let quantifier = if column_next {
            None
        } else {
            self.set_quantifier()
        };
        let projection = self.comma_sep(Self::select_item)?;
        let from = if with_from && self.kw_at(0, "FROM") {
            Some(self.clause_from()?)
        } else {
            None
        };
        let lateral_views = self.lateral_views()?;
        let selection = self.opt_clause("WHERE")?;
        let group_by = if self.kw_at(0, "GROUP") && self.kw_at(1, "BY") {
            Some(self.group_by()?)
        } else {
            None
        };
        let having = self.opt_clause("HAVING")?;
        let windows = self.window_clause()?;
        let qualify = self.opt_clause("QUALIFY")?;
        Ok(Select {
            hints,
            quantifier,
            projection,
            from,
            lateral_views,
            selection,
            group_by,
            having,
            windows,
            qualify,
            span: self.span_from(start),
        })
    }

    fn opt_clause(&mut self, kw: &str) -> PR<Option<Expr>> {
        if self.eat_kw(kw) {
            self.expr().map(Some)
        } else {
            Ok(None)
        }
    }

    /// `/*+ name[(params)] [,] ... */`.
    #[inline(never)]
    fn hints(&mut self, out: &mut Vec<Hint>) -> PR<()> {
        self.bump();
        loop {
            let start = self.start();
            let name = self.any_ident()?;
            let params = if self.sym_at(0, Sym::LParen) {
                self.paren_list(Self::primary_expr)?
            } else {
                vec![]
            };
            out.push(Hint {
                name,
                params,
                span: self.span_from(start),
            });
            if self.kind(0) == Some(TokenKind::HintEnd) {
                self.bump();
                return Ok(());
            }
            // `COMMA? hintStatement`: a comma needs another hint.
            self.eat_sym(Sym::Comma);
        }
    }

    fn select_item(&mut self) -> PR<SelectItem> {
        let start = self.start();
        let expr = self.expr()?;
        let alias = if !self.item_end_at(1) && self.eat_kw("AS") {
            Some(if self.sym_at(0, Sym::LParen) {
                Alias::List(self.ident_list()?)
            } else {
                Alias::Name(self.name_part(true)?)
            })
        } else if self.sym_at(0, Sym::LParen) {
            Some(Alias::List(self.ident_list()?))
        } else {
            self.implicit_alias(true)?.map(Alias::Name)
        };
        Ok(SelectItem {
            expr,
            alias,
            span: self.span_from(start),
        })
    }

    fn sort_item(&mut self, allow_all: bool) -> PR<SortItem> {
        let start = self.start();
        let expr = if allow_all
            && !self.opts.enforce_reserved_keywords
            && self.kw_at(0, "ALL")
            && !self.sym_at(1, Sym::Dot)
            && !self.sym_at(1, Sym::LParen)
        {
            self.column_named_all()
        } else {
            self.expr()?
        };
        let asc = if self.eat_kw("ASC") {
            Some(true)
        } else if self.eat_kw("DESC") {
            Some(false)
        } else {
            None
        };
        let mut nulls_first = None;
        if self.eat_kw("NULLS") {
            nulls_first = Some(self.eat_kw("FIRST"));
            if nulls_first == Some(false) {
                self.expect_kw("LAST")?;
            }
        }
        Ok(SortItem {
            expr,
            asc,
            nulls_first,
            span: self.span_from(start),
        })
    }

    /// `ALL` in `ORDER BY ALL` / `GROUP BY ALL`: a column named `ALL`.
    fn column_named_all(&mut self) -> Expr {
        let span = self.tok(0).map_or_else(Span::default, |t| t.span);
        self.bump();
        Expr {
            kind: crate::ast::ExprKind::Column(vec![Ident {
                value: span.text(self.src).to_owned(),
                quoted: false,
                span,
            }]),
            span,
        }
    }

    // ---- GROUP BY -------------------------------------------------------

    #[inline(never)]
    fn group_by(&mut self) -> PR<GroupBy> {
        let start = self.start();
        self.pos += 2;
        let items = self.comma_sep(Self::group_item)?;
        let with = if self.kw_at(0, "WITH") && self.any_kw_at(1, &["ROLLUP", "CUBE"]) {
            let rollup = self.kw_at(1, "ROLLUP");
            self.pos += 2;
            Some(if rollup {
                GroupWith::Rollup
            } else {
                GroupWith::Cube
            })
        } else if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
            self.pos += 2;
            Some(GroupWith::GroupingSets(
                self.paren_list(Self::grouping_set)?,
            ))
        } else {
            None
        };
        Ok(GroupBy {
            items,
            with,
            span: self.span_from(start),
        })
    }

    fn group_item(&mut self) -> PR<GroupItem> {
        let start = self.start();
        let kind = if self.any_kw_at(0, &["ROLLUP", "CUBE"]) && self.sym_at(1, Sym::LParen) {
            let rollup = self.kw_at(0, "ROLLUP");
            self.bump();
            let sets = self.paren_list(Self::grouping_set)?;
            if rollup {
                GroupItemKind::Rollup(sets)
            } else {
                GroupItemKind::Cube(sets)
            }
        } else if self.kw_at(0, "GROUPING") && self.kw_at(1, "SETS") {
            self.pos += 2;
            GroupItemKind::GroupingSets(self.paren_list(|p| {
                let analytics = p.any_kw_at(0, &["ROLLUP", "CUBE"]) && p.sym_at(1, Sym::LParen)
                    || p.kw_at(0, "GROUPING") && p.kw_at(1, "SETS");
                if analytics {
                    p.nested(Self::group_item)
                } else {
                    let set = p.grouping_set()?;
                    Ok(GroupItem {
                        span: set.span,
                        kind: GroupItemKind::Set(set),
                    })
                }
            })?)
        } else if !self.opts.enforce_reserved_keywords
            && self.kw_at(0, "ALL")
            && !self.sym_at(1, Sym::Dot)
            && !self.sym_at(1, Sym::LParen)
        {
            GroupItemKind::Expr(self.column_named_all())
        } else {
            GroupItemKind::Expr(self.expr()?)
        };
        Ok(GroupItem {
            kind,
            span: self.span_from(start),
        })
    }

    /// `(a, b)`, `()` or `a`. A parenthesized list is a set; `(a + b) * 2`
    /// is an expression, continued from `(a + b)` without re-parsing it.
    fn grouping_set(&mut self) -> PR<GroupingSet> {
        let start = self.start();
        if !self.sym_at(0, Sym::LParen) || self.query_follows(1) {
            let e = self.expr()?;
            return Ok(self.grouping_set_of(vec![e], start));
        }
        self.bump();
        if self.eat_sym(Sym::RParen) {
            return Ok(self.grouping_set_of(vec![], start));
        }
        let (first, height) = self.measured(Self::expr)?;
        if self.eat_sym(Sym::Comma) {
            let mut exprs = vec![first];
            exprs.extend(self.comma_sep(Self::expr)?);
            self.expect_sym(Sym::RParen)?;
            return Ok(self.grouping_set_of(exprs, start));
        }
        self.expect_sym(Sym::RParen)?;
        if self.item_end_at(0) {
            return Ok(self.grouping_set_of(vec![first], start));
        }
        let nested = Expr {
            span: self.span_from(start),
            kind: crate::ast::ExprKind::Nested(Box::new(first)),
        };
        let e = self.expr_from(nested, height)?;
        Ok(self.grouping_set_of(vec![e], start))
    }

    fn grouping_set_of(&self, exprs: Vec<Expr>, start: usize) -> GroupingSet {
        GroupingSet {
            exprs,
            span: self.span_from(start),
        }
    }

    // ---- windows ----------------------------------------------------------

    fn window_clause(&mut self) -> PR<Vec<NamedWindow>> {
        if !self.eat_kw("WINDOW") {
            return Ok(vec![]);
        }
        self.comma_sep(|p| {
            let start = p.start();
            let name = p.name_part(true)?;
            p.expect_kw("AS")?;
            let window = p.window_ref()?;
            Ok(NamedWindow {
                name,
                window,
                span: p.span_from(start),
            })
        })
    }

    // ---- relations ----------------------------------------------------------

    fn relation(&mut self) -> PR<Relation> {
        self.nested(|p| {
            let start = p.start();
            let lateral = p.eat_kw("LATERAL");
            let primary = p.table_factor()?;
            let mut extensions = vec![];
            loop {
                if p.kw_at(0, "PIVOT") && p.sym_at(1, Sym::LParen) {
                    extensions.push(RelationExt::Pivot(p.pivot()?));
                } else if p.kw_at(0, "UNPIVOT")
                    && (p.sym_at(1, Sym::LParen) || p.any_kw_at(1, &["INCLUDE", "EXCLUDE"]))
                {
                    extensions.push(RelationExt::Unpivot(p.unpivot()?));
                } else if let Some(j) = p.join()? {
                    extensions.push(RelationExt::Join(j));
                } else {
                    break;
                }
            }
            Ok(Relation {
                lateral,
                primary,
                extensions,
                span: p.span_from(start),
            })
        })
    }

    /// Whether token `n` and the parentheses after it open a query.
    fn query_follows(&self, n: usize) -> bool {
        let at = self.pos + n;
        let after = self.after_parens.get(at).copied().unwrap_or(at);
        self.any_kw_at(after - self.pos, QUERY_START)
    }

    fn table_alias(&mut self) -> PR<Option<TableAlias>> {
        let start = self.start();
        // `AS (columns)`: a column list without a table name, as in
        // `VALUES (1) AS (c)`.
        if self.kw_at(0, "AS") && self.sym_at(1, Sym::LParen) && !self.query_follows(2) {
            self.pos += 1;
            let columns = self.ident_list()?;
            return Ok(Some(TableAlias {
                name: None,
                columns,
                span: self.span_from(start),
            }));
        }
        let Some(name) = self.opt_alias_name()? else {
            return Ok(None);
        };
        let columns = if self.sym_at(0, Sym::LParen) {
            self.ident_list()?
        } else {
            vec![]
        };
        Ok(Some(TableAlias {
            name: Some(name),
            columns,
            span: self.span_from(start),
        }))
    }

    fn table_factor(&mut self) -> PR<TableFactor> {
        let start = self.start();
        let kind = if self.kw_at(0, "VALUES") && !self.item_end_at(1) {
            let values = self.values()?;
            return Ok(TableFactor {
                kind: TableFactorKind::Values(values.rows),
                alias: values.alias,
                span: values.span,
            });
        } else if self.sym_at(0, Sym::LParen) {
            let save = self.pos;
            let mut sub = None;
            if self.query_follows(1) {
                self.bump();
                let nested = self.sym_at(0, Sym::LParen);
                match self.query() {
                    Ok(q) if self.sym_at(0, Sym::RParen) || !nested => {
                        // `(SELECT ...` is committed to by ANTLR's prediction.
                        self.expect_closing(Sym::RParen, Close::DropOrInsert)?;
                        sub = Some(q);
                    }
                    Err(Fail::Fatal(e)) => return Err(Fail::Fatal(e)),
                    _ => self.pos = save,
                }
            }
            match sub {
                Some(q) => TableFactorKind::Subquery(Box::new(q)),
                None => {
                    self.expect_sym(Sym::LParen)?;
                    let r = self.relation()?;
                    self.expect_sym(Sym::RParen)?;
                    TableFactorKind::Nested(Box::new(r))
                }
            }
        } else {
            let name = self.table_name()?;
            if self.sym_at(0, Sym::LParen) {
                self.bump();
                let args = self.function_args()?;
                self.expect_sym(Sym::RParen)?;
                TableFactorKind::TableFunction { name, args }
            } else {
                let temporal = self.temporal_clause()?;
                let options = self.table_options()?;
                TableFactorKind::Table {
                    name,
                    temporal,
                    options,
                }
            }
        };
        let alias = self.table_alias()?;
        Ok(TableFactor {
            kind,
            alias,
            span: self.span_from(start),
        })
    }

    /// `[FOR] (VERSION | SYSTEM_VERSION) AS OF version` or `[FOR]
    /// (TIMESTAMP | SYSTEM_TIME) AS OF expression`, if one starts here.
    #[inline(never)]
    fn temporal_clause(&mut self) -> PR<Option<TemporalClause>> {
        let start = self.start();
        let n = usize::from(self.kw_at(0, "FOR"));
        let kind = if self.any_kw_at(n, &["VERSION", "SYSTEM_VERSION"]) {
            TemporalKind::Version
        } else if self.any_kw_at(n, &["TIMESTAMP", "SYSTEM_TIME"]) {
            TemporalKind::Timestamp
        } else {
            return Ok(None);
        };
        if !(self.kw_at(n + 1, "AS") && self.kw_at(n + 2, "OF")) {
            return Ok(None);
        }
        self.pos += n + 3;
        let value = match kind {
            TemporalKind::Version => {
                let at = self.start();
                let literal = match self.kind(0) {
                    Some(TokenKind::Number(NumKind::Int)) => {
                        let text = self.text(0).to_owned();
                        self.bump();
                        Literal::Number {
                            text,
                            kind: NumKind::Int,
                        }
                    }
                    _ => match self.string_lit() {
                        Some(v) => Literal::String(v),
                        None => return self.fail(),
                    },
                };
                Expr {
                    kind: ExprKind::Literal(literal),
                    span: self.span_from(at),
                }
            }
            _ => self.value()?,
        };
        Ok(Some(TemporalClause {
            kind,
            value,
            span: self.span_from(start),
        }))
    }

    /// `WITH (key [=] value, ...)`, the options of a table.
    #[inline(never)]
    fn table_options(&mut self) -> PR<Vec<TableOption>> {
        if !(self.kw_at(0, "WITH") && self.sym_at(1, Sym::LParen)) {
            return Ok(vec![]);
        }
        self.bump();
        self.paren_list(|p| {
            let start = p.start();
            let key = p.option_key()?;
            let equals = p.eat_sym(Sym::Eq);
            let literal = matches!(
                p.kind(0),
                Some(TokenKind::Number(NumKind::Int | NumKind::Decimal) | TokenKind::String)
            ) || p.any_kw_at(0, &["TRUE", "FALSE"])
                || (p.kind(0) == Some(TokenKind::DoubleQuoted)
                    && !p.opts.double_quoted_identifiers);
            let value = if equals || literal {
                if !literal {
                    return p.fail();
                }
                Some(p.primary_expr()?)
            } else {
                None
            };
            Ok(TableOption {
                key,
                value,
                span: p.span_from(start),
            })
        })
    }

    /// `name(.name)*` or a string.
    fn option_key(&mut self) -> PR<OptionKey> {
        let start = self.start();
        if let Some(text) = self.string_lit() {
            let span = self.span_from(start);
            return Ok(OptionKey {
                parts: vec![Ident {
                    value: text,
                    quoted: true,
                    span,
                }],
                is_string: true,
                span,
            });
        }
        let mut parts = vec![self.name_part(true)?];
        while self.eat_sym(Sym::Dot) {
            parts.push(self.name_part(true)?);
        }
        Ok(OptionKey {
            parts,
            is_string: false,
            span: self.span_from(start),
        })
    }

    /// A join, if one starts here.
    #[inline(never)]
    fn join(&mut self) -> PR<Option<Join>> {
        let start = self.start();
        let natural = self.kw_at(0, "NATURAL");
        let n = usize::from(natural);
        let at = |p: &Self, i: usize, kw: &str| p.kw_at(n + i, kw);
        let (kind, words) = if at(self, 0, "JOIN") {
            (JoinKind::Inner, 0)
        } else if at(self, 0, "INNER") {
            (JoinKind::Inner, 1)
        } else if at(self, 0, "CROSS") {
            (JoinKind::Cross, 1)
        } else if at(self, 0, "LEFT") && at(self, 1, "SEMI") || at(self, 0, "SEMI") {
            (JoinKind::LeftSemi, 1 + usize::from(at(self, 0, "LEFT")))
        } else if at(self, 0, "LEFT") && at(self, 1, "ANTI") || at(self, 0, "ANTI") {
            (JoinKind::LeftAnti, 1 + usize::from(at(self, 0, "LEFT")))
        } else if let Some(kind) = [
            ("LEFT", JoinKind::Left),
            ("RIGHT", JoinKind::Right),
            ("FULL", JoinKind::Full),
        ]
        .into_iter()
        .find_map(|(kw, k)| at(self, 0, kw).then_some(k))
        {
            (kind, 1 + usize::from(at(self, 1, "OUTER")))
        } else {
            if natural {
                self.bump();
                return self.fail();
            }
            return Ok(None);
        };
        self.pos += n + words;
        self.expect_kw("JOIN")?;
        let lateral = self.eat_kw("LATERAL");
        let right = Box::new(self.table_factor()?);
        let constraint = if natural {
            JoinConstraint::None
        } else if self.eat_kw("ON") {
            JoinConstraint::On(self.expr()?)
        } else if self.eat_kw("USING") {
            JoinConstraint::Using(self.ident_list()?)
        } else {
            JoinConstraint::None
        };
        Ok(Some(Join {
            kind,
            natural,
            lateral,
            right,
            constraint,
            span: self.span_from(start),
        }))
    }

    #[inline(never)]
    fn pivot(&mut self) -> PR<Pivot> {
        let start = self.start();
        self.expect_kw("PIVOT")?;
        self.expect_sym(Sym::LParen)?;
        let aggregates = self.comma_sep(Self::select_item)?;
        self.expect_kw("FOR")?;
        let columns = if self.sym_at(0, Sym::LParen) {
            self.paren_list(|p| p.name_part(true))?
        } else {
            vec![self.name_part(true)?]
        };
        self.expect_kw("IN")?;
        let values = self.paren_list(|p| {
            let start = p.start();
            let expr = p.expr()?;
            let alias = p.opt_alias_name()?.map(Alias::Name);
            Ok(SelectItem {
                expr,
                alias,
                span: p.span_from(start),
            })
        })?;
        self.expect_sym(Sym::RParen)?;
        let alias = self.opt_alias_name()?;
        Ok(Pivot {
            aggregates,
            columns,
            values,
            alias,
            span: self.span_from(start),
        })
    }

    #[inline(never)]
    fn unpivot(&mut self) -> PR<Unpivot> {
        let start = self.start();
        self.expect_kw("UNPIVOT")?;
        let mut include_nulls = None;
        if self.any_kw_at(0, &["INCLUDE", "EXCLUDE"]) {
            include_nulls = Some(self.kw_at(0, "INCLUDE"));
            self.bump();
            self.expect_kw("NULLS")?;
        }
        self.expect_sym(Sym::LParen)?;
        let multi = self.sym_at(0, Sym::LParen);
        let value_columns = if multi {
            self.ident_list()?
        } else {
            vec![self.any_ident()?]
        };
        self.expect_kw("FOR")?;
        let name_column = self.any_ident()?;
        self.expect_kw("IN")?;
        let columns = self.paren_list(|p| {
            let start = p.start();
            let columns = if multi {
                p.paren_list(Self::object_name)?
            } else {
                vec![p.object_name()?]
            };
            let alias = p.opt_alias_name()?;
            Ok(UnpivotColumns {
                columns,
                alias,
                span: p.span_from(start),
            })
        })?;
        self.expect_sym(Sym::RParen)?;
        let alias = self.opt_alias_name()?;
        Ok(Unpivot {
            include_nulls,
            value_columns,
            name_column,
            columns,
            alias,
            span: self.span_from(start),
        })
    }

    /// `LATERAL VIEW [OUTER] f(args) table [AS] col, ...`*.
    #[inline(never)]
    fn lateral_views(&mut self) -> PR<Vec<LateralView>> {
        let mut out = vec![];
        while self.kw_at(0, "LATERAL") && self.kw_at(1, "VIEW") {
            let start = self.start();
            self.pos += 2;
            let outer = self.eat_kw("OUTER");
            let function = self.object_name()?;
            self.expect_sym(Sym::LParen)?;
            let args = if self.sym_at(0, Sym::RParen) {
                vec![]
            } else {
                self.comma_sep(Self::expr)?
            };
            self.expect_sym(Sym::RParen)?;
            // Spark's grammar needs the table name: `AS t` names it `AS`.
            let table = self.any_ident()?;
            let columns = if self.eat_kw("AS") {
                self.comma_sep(Self::any_ident)?
            } else if let Some(first) = self.implicit_alias(false)? {
                let mut columns = vec![first];
                while self.eat_sym(Sym::Comma) {
                    columns.push(self.ident()?);
                }
                columns
            } else {
                vec![]
            };
            out.push(LateralView {
                outer,
                function,
                args,
                table,
                columns,
                span: self.span_from(start),
            });
        }
        Ok(out)
    }
}
