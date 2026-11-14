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

mod case;
mod ddl;
mod expr;
mod function;
mod names;
pub(crate) use function::check_temporary_function_name;
mod script;
mod types;

use std::collections::HashMap;

use crate::ast::{
    Alias, BucketOn, BucketSpec, ColumnConstraint, ColumnConstraintKind, ColumnDef, ColumnName,
    ConstraintCharacteristic, CreateNamespace, CreateTable, CreateTableLike, CreateView, Cte,
    DataTypeSpec, DdlName, DeclareVariable, DropBehavior, DropNamespace, DropTable, DropVariable,
    DropView, DynamicName, ExecuteImmediate, Explain, ExplainMode, Expr, ExprKind, FileFormat,
    FileFormatKind, ForeignKeyConstraint, ForeignKeyRef, FromClause, FromStatement,
    GeneratedColumn, GeneratedKind, GroupBy, GroupItem, GroupItemKind, GroupWith, GroupingSet,
    Hint, Ident, IdentityColumn, IdentityHow, Join, JoinConstraint, JoinKind, LateralView, Limit,
    Literal, NamedWindow, NamespaceKind, NearestBy, NearestDirection, NumKind, ObjectName,
    OptionKey, OrderedIdent, PartitionColumn, PartitionField, PartitionFieldKind,
    PartitionTransform, Pipe, PipeAggregate, PipeAssignment, PipeOp, PipeOpKind, PipeSelect,
    PipeWhere, Pivot, Query, QueryOrganization, Relation, RelationExt, Reset, RowDelimited,
    RowFormat, RowFormatKind, RowSerde, SampleMethod, SampleMethodKind, SampleType, Select,
    SelectItem, Set, SetAssignment, SetExpr, SetExprKind, SetKind, SetOperator, SetQuantifier,
    SetVariable, SetVariableKind, SkewSpec, SortItem, Span, Statement, StructField, TableAlias,
    TableConstraint, TableConstraintKind, TableElement, TableFactor, TableFactorKind, TableOption,
    TableSample, TemporalClause, TemporalKind, TimeZone, TransformKind, TransformOutput,
    TransformSelect, Unpivot, UnpivotColumns, Use, UsingArgument, Values, ViewColumn, With,
};
use crate::error::{self, ParseError};
use crate::keywords::{CLAUSE_WORD, RESERVED_WORD, STRICT_WORD, classify, is_kw};
use crate::lexer::{Sym, Token, TokenKind, tokenize, unescape_string, unquote_ident};
use crate::{MAX_DEPTH, ParserOptions, validate};
use krishiv_common::sql_path::PathElement as PathEl;

/// Why a rule did not match.
pub(crate) enum Fail {
    /// The tokens do not fit; the caller may try another alternative.
    Syntax,
    /// An error to report as is.
    Fatal(Box<ParseError>),
}

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

/// The `CREATE TABLE ... LIKE` clauses Spark rejects when written twice,
/// in the order it checks them (it calls `USING` `PROVIDER` here).
const CREATE_TABLE_LIKE_CLAUSES: [&str; 5] = [
    "PROVIDER",
    "STORED AS/BY",
    "ROW FORMAT",
    "LOCATION",
    "TBLPROPERTIES",
];

/// The `CREATE NAMESPACE` clauses Spark rejects when written twice, in
/// the order it checks them (`visitCreateNamespace`).
const CREATE_NAMESPACE_CLAUSES: [&str; 5] = [
    "COMMENT",
    "DEFAULT COLLATION",
    "LOCATION",
    "WITH PROPERTIES",
    "WITH DBPROPERTIES",
];

/// Parse state to rewind to when a speculative parse fails.
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
    /// For each `(` token, the position of its matching `)`; `usize::MAX`
    /// for any other token and for an unclosed `(`.
    closers: Vec<usize>,
    /// Token positions of the `?` parameter markers, in order.
    markers: Vec<usize>,
    /// How many script bodies the cursor is inside: a statement there ends at its `;`.
    in_script: usize,
    /// How many select items enclose the cursor (Spark's `NamedExpression` ancestors): a
    /// backquoted name there may be a column regex.
    in_named: usize,
    /// Where the expression of the innermost select-list item starts: an
    /// alias may follow the expression there (see `Parser::not_is_column`).
    item_start: Option<usize>,
    /// The part of the innermost `CASE` being read and where it starts: the
    /// words that close it decide whether `NOT <word>` is the operator
    /// (see `Parser::not_takes_operand`).
    case_part: Option<(usize, CasePart)>,
    /// What reading a `CASE` expression from a token position gave, so a
    /// speculative `CASE` (`Parser::case_or_column`) is read once per
    /// position and backtracking stays linear.
    case_memo: HashMap<usize, CaseMemo>,
    /// The `CASE` token a simple `CASE`'s operand must read as a column,
    /// set while that operand is read again (`CASE case - 1 WHEN 0 ...`).
    case_column_at: Option<usize>,
    /// How many script statements enclose the cursor (compounds, branches and loop bodies).
    script_depth: usize,
    /// The first error Spark raises after parsing (in `AstBuilder`) that
    /// the tree cannot carry; reported if the statement parses.
    deferred: Option<ParseError>,
    /// The keyword tokens of the complex types read after `::`, in order.
    cast_types: Vec<usize>,
    /// A complex type keyword after `::` to read as a bare type.
    force_bare: Option<usize>,
    /// Reading the bodies of a multi-table insert: `INSERT INTO|OVERWRITE`
    /// starts the next clause, never an implicit alias.
    insert_bodies: bool,
}

/// A part of a `CASE` expression.
#[derive(Clone, Copy)]
pub(crate) enum CasePart {
    /// A simple `CASE`'s operand.
    Operand,
    /// A `WHEN` condition.
    Condition,
    /// A `THEN` result.
    Result,
    /// The `ELSE` result.
    Else,
}

impl CasePart {
    /// The words that end the part.
    pub(crate) fn closers(self) -> &'static [&'static str] {
        match self {
            Self::Operand => &["WHEN"],
            Self::Condition => &["THEN"],
            Self::Result => &["WHEN", "ELSE", "END"],
            Self::Else => &["END"],
        }
    }
}

/// The outcome of reading a `CASE` expression at a token position.
pub(crate) enum CaseMemo {
    /// It parsed once; a second parse here stores the tree. Only a
    /// speculative re-read parses a position twice, so an unambiguous
    /// `CASE` is never copied (nested ones would copy every subtree).
    Seen,
    /// It parsed to this expression, ending before this token, its tree
    /// reaching this many levels below the `CASE`.
    Parsed(crate::ast::ExprKind, usize, usize),
    /// It failed; the furthest failure and its recovery hints.
    Failed {
        far: usize,
        matched: Option<(usize, Sym, Close)>,
        missing: Option<(usize, &'static str)>,
    },
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
pub(crate) fn nesting_bound(src: &str, toks: &[Token]) -> usize {
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

#[cfg(test)]
thread_local! {
    /// Tests: parsers made on this thread (each is one parse attempt over the
    /// statement), so a test bounds the work of a parse by count, not time.
    pub(crate) static PARSERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
        #[cfg(test)]
        PARSERS.set(PARSERS.get() + 1);
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
        let mut closers = vec![usize::MAX; toks.len()];
        let mut open = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            match t.kind {
                TokenKind::Sym(Sym::LParen) => open.push(i),
                TokenKind::Sym(Sym::RParen) => {
                    if let Some(o) = open.pop() {
                        closers[o] = i;
                    }
                }
                _ => {}
            }
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
            closers,
            markers,
            in_script: 0,
            in_named: 0,
            item_start: None,
            case_part: None,
            case_memo: HashMap::new(),
            case_column_at: None,
            script_depth: 0,
            deferred: None,
            cast_types: vec![],
            force_bare,
            insert_bodies: false,
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

    /// Where a statement that reads the rest of its text raw stops: at the `;` that ends it inside a
    /// script, else at the end of the input. A token index, and its byte offset.
    fn raw_end(&self) -> (usize, usize) {
        let end = if self.in_script > 0 {
            (self.pos..self.toks.len())
                .find(|&i| self.toks[i].kind == TokenKind::Sym(Sym::Semi))
                .unwrap_or(self.toks.len())
        } else {
            self.toks.len()
        };
        (
            end,
            self.toks.get(end).map_or(self.src.len(), |t| t.span.start),
        )
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
        if self.identifier_literal_at(n) {
            return self.identifier_literal(n);
        }
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
        if let Some(len) = self.identifier_literal_len(0) {
            let mut id = id;
            // A name that is not one part is an error already; one part
            // is the identifier.
            if let Some(name) = self.check_identifier_literal(&id.value, id.span, 1)
                && let [part] = name.parts.as_slice()
            {
                id.value.clone_from(&part.value);
            }
            self.pos += len;
            return Ok(id);
        }
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
        let mut parts = self.name_parts(false)?;
        while self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
            self.bump();
            parts.extend(self.name_parts(true)?);
        }
        Ok(ObjectName {
            parts,
            span: self.span_from(start),
        })
    }

    /// A session variable name: any keyword names one unless keywords are
    /// enforced (`DECLARE all = 1`), as in Spark's grammar.
    #[inline(never)]
    fn variable_name(&mut self) -> PR<ObjectName> {
        let start = self.start();
        let mut parts = self.name_parts(true)?;
        while self.sym_at(0, Sym::Dot) && self.is_ident_at(1, true) {
            self.bump();
            parts.extend(self.name_parts(true)?);
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
        let mut parts = self.name_parts(true)?;
        // With only the legacy function syntax, `IDENTIFIER(` is never a
        // later part of a name.
        while self.sym_at(0, Sym::Dot)
            && self.is_ident_at(1, true)
            && !(self.opts.legacy_identifier_clause
                && self.kw_at(1, "IDENTIFIER")
                && self.sym_at(2, Sym::LParen))
        {
            self.bump();
            parts.extend(self.name_parts(true)?);
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
        if let Some(done) = self.script_alias(strict)? {
            return Ok(done);
        }
        // `APPROX NEAREST ...` after a join's relation is its ranking clause.
        if self.any_kw_at(0, &["APPROX", "EXACT"]) && self.kw_at(1, "NEAREST") {
            return Ok(None);
        }
        if self.insert_bodies
            && self.kw_at(0, "INSERT")
            && (self.kw_at(1, "INTO") || self.kw_at(1, "OVERWRITE"))
        {
            return Ok(None);
        }
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

    // ---- DDL: tables, namespaces, USE ------------------------------------

    // ---- GROUP BY -------------------------------------------------------

    // ---- windows ----------------------------------------------------------

    // ---- relations ----------------------------------------------------------
}

mod query;

mod create;

mod commands;

mod errors;
