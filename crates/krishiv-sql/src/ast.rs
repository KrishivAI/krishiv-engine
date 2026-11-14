// SPDX-License-Identifier: Apache-2.0
//! The syntax tree of a Spark SQL query.
//!
//! Every struct node carries the [`Span`] of the source text it was parsed
//! from; enums either wrap spanned nodes or are the `kind` of a spanned
//! struct ([`Expr`], [`SetExpr`], [`TableFactor`], ...). Two spans always
//! compare equal, so the derived `==` of every node ignores source
//! positions: the same query written with different whitespace or comments
//! gives `==` trees. Compare [`Span::range`] for positional equality.
//!
//! The tree is syntax only: names are not resolved, types not checked, and
//! literals keep their source text (Spark's `AstBuilder` checks such as
//! literal ranges happen when the tree is lowered).

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;

/// A byte range of the source text.
///
/// `==` is always true and `Hash` writes nothing, so that the derived
/// equality and hashing of AST nodes ignore positions. Types outside the
/// tree whose position matters ([`crate::ParseError`],
/// [`crate::lexer::Token`]) compare [`Span::range`] instead.
#[derive(Clone, Copy, Default, Eq)]
pub struct Span {
    /// First byte.
    pub start: usize,
    /// One past the last byte.
    pub end: usize,
}

impl Span {
    /// The span `start..end`.
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// The byte range.
    #[must_use]
    pub const fn range(self) -> Range<usize> {
        self.start..self.end
    }

    /// The smallest span covering `self` and `other`.
    #[must_use]
    pub fn to(self, other: Self) -> Self {
        Self::new(self.start.min(other.start), self.end.max(other.end))
    }

    /// The source text of the span (empty if out of range).
    #[must_use]
    pub fn text(self, src: &str) -> &str {
        src.get(self.range()).unwrap_or_default()
    }
}

impl PartialEq for Span {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Hash for Span {
    fn hash<H: Hasher>(&self, _: &mut H) {}
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// An identifier, case kept as written.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    /// The name (unquoted and unescaped).
    pub value: String,
    /// Written with backquotes (or double quotes as identifiers).
    pub quoted: bool,
    /// Source text.
    pub span: Span,
}

/// A dotted name such as `catalog.db.table`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectName {
    /// The parts, outermost first.
    pub parts: Vec<Ident>,
    /// Source text.
    pub span: Span,
}

/// A statement: what [`crate::parse_statement`] accepts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Statement {
    /// A query.
    Query(Box<Query>),
    /// `EXPLAIN [mode] statement`.
    Explain(Explain),
    /// `SET`, `SET -v`, `SET key` or `SET key = value`.
    Set(Set),
    /// `RESET` or `RESET key`.
    Reset(Reset),
    /// `CREATE [OR REPLACE] TEMPORARY VIEW name [(columns)] AS query`.
    CreateView(CreateView),
    /// `DROP VIEW [IF EXISTS] name`.
    DropView(DropView),
    /// `CREATE [OR REPLACE] [TEMPORARY] FUNCTION name (...) ... RETURN body`.
    CreateFunction(Box<CreateFunction>),
    /// `DROP [TEMPORARY] FUNCTION [IF EXISTS] name`.
    DropFunction(DropFunction),
    /// `CREATE [OR REPLACE] [TEMPORARY] [EXTERNAL] TABLE ...`.
    CreateTable(Box<CreateTable>),
    /// `CREATE TABLE ... LIKE ...`.
    CreateTableLike(Box<CreateTableLike>),
    /// `DROP TABLE [IF EXISTS] name [PURGE]`.
    DropTable(DropTable),
    /// `CREATE DATABASE|SCHEMA|NAMESPACE [IF NOT EXISTS] name ...`.
    CreateNamespace(CreateNamespace),
    /// `DROP DATABASE|SCHEMA|NAMESPACE [IF EXISTS] name ...`.
    DropNamespace(DropNamespace),
    /// `USE name` or `USE DATABASE|SCHEMA|NAMESPACE name`.
    Use(Use),
    /// `DECLARE [OR REPLACE] [VARIABLE | VAR] name [, ...] [type]
    /// [{DEFAULT | =} expr]`.
    DeclareVariable(DeclareVariable),
    /// `SET VAR[IABLE] v = expr [, ...]` or `SET VAR[IABLE] (v, ...) =
    /// (query)`.
    SetVariable(SetVariable),
    /// `DROP TEMPORARY VAR[IABLE] [IF EXISTS] name`.
    DropVariable(DropVariable),
    /// `EXECUTE IMMEDIATE query [INTO v, ...] [USING args]`.
    ExecuteImmediate(ExecuteImmediate),
    /// A SQL script: `[NOT ATOMIC] BEGIN ... END`.
    Script(Box<Compound>),
    /// `CREATE [OR REPLACE] [GLOBAL] [TEMPORARY] VIEW ...`: every view
    /// shape the legacy [`CreateView`] node cannot express (a permanent
    /// view, or a temporary view with a qualified or computed name, a
    /// datasource provider, or the `GLOBAL` qualifier).
    CreatePermanentView(Box<CreatePermanentView>),
    /// `DROP VIEW [IF EXISTS] name` with a qualified or computed name.
    DropPermanentView(DropPermanentView),
    /// `ALTER TABLE name ...`.
    AlterTable(Box<AlterTable>),
    /// `ALTER VIEW name ...`.
    AlterView(Box<AlterView>),
    /// `ALTER DATABASE|SCHEMA|NAMESPACE name ...`.
    AlterNamespace(AlterNamespace),
    /// `INSERT INTO|OVERWRITE ...`, or `FROM ... INSERT INTO ... INSERT
    /// INTO ...` (a multi-table insert).
    Insert(Box<Insert>),
    /// `DELETE FROM name [WHERE ...]`.
    Delete(Delete),
    /// `UPDATE name SET ... [WHERE ...]`.
    Update(Box<Update>),
    /// `MERGE INTO ...`.
    Merge(Box<Merge>),
    /// `TRUNCATE TABLE name [PARTITION ...]`.
    Truncate(Truncate),
    /// `SHOW ...`.
    Show(Show),
    /// `DESCRIBE ...` (`DESC` for `DESCRIBE`).
    Describe(Describe),
    /// `CACHE ...`, `UNCACHE ...` or `CLEAR CACHE`.
    Cache(Cache),
    /// `REFRESH ...`.
    Refresh(Refresh),
    /// `ANALYZE TABLE ... COMPUTE STATISTICS ...`.
    Analyze(Analyze),
    /// `MSCK REPAIR TABLE ...` or `REPAIR TABLE ...`.
    Repair(Repair),
    /// `ADD JAR|FILE|ARCHIVE ...`.
    AddResource(AddResource),
    /// `LIST JAR|FILE|ARCHIVE [...]`.
    ListResource(ListResource),
    /// `CREATE MATERIALIZED VIEW` or `CREATE STREAMING TABLE`.
    CreatePipelineDataset(Box<CreatePipelineDataset>),
    /// `CREATE FLOW name AS INSERT ...`.
    CreateFlow(Box<CreateFlow>),
    /// `LOAD DATA ... INTO TABLE ...`.
    LoadData(Box<LoadData>),
    /// `INSERT OVERWRITE [LOCAL] DIRECTORY ...`.
    InsertDirectory(Box<InsertDirectory>),
}

impl Statement {
    /// Source text, from the first keyword to the end of the statement.
    #[must_use]
    pub const fn span(&self) -> Span {
        match self {
            Self::Query(q) => q.span,
            Self::Explain(e) => e.span,
            Self::Set(s) => s.span,
            Self::Reset(r) => r.span,
            Self::CreateView(v) => v.span,
            Self::DropView(v) => v.span,
            Self::CreateFunction(f) => f.span,
            Self::DropFunction(f) => f.span,
            Self::CreateTable(t) => t.span,
            Self::CreateTableLike(t) => t.span,
            Self::DropTable(t) => t.span,
            Self::CreateNamespace(n) => n.span,
            Self::DropNamespace(n) => n.span,
            Self::Use(u) => u.span,
            Self::DeclareVariable(d) => d.span,
            Self::SetVariable(s) => s.span,
            Self::DropVariable(d) => d.span,
            Self::ExecuteImmediate(e) => e.span,
            Self::Script(c) => c.span,
            Self::CreatePermanentView(v) => v.span,
            Self::DropPermanentView(v) => v.span,
            Self::AlterTable(a) => a.span,
            Self::AlterView(a) => a.span,
            Self::AlterNamespace(a) => a.span,
            Self::Insert(i) => i.span,
            Self::Delete(d) => d.span,
            Self::Update(u) => u.span,
            Self::Merge(m) => m.span,
            Self::Truncate(t) => t.span,
            Self::Show(s) => s.span,
            Self::Describe(d) => d.span,
            Self::Cache(c) => c.span,
            Self::Refresh(r) => r.span,
            Self::Analyze(a) => a.span,
            Self::Repair(r) => r.span,
            Self::AddResource(r) => r.span,
            Self::ListResource(r) => r.span,
            Self::CreatePipelineDataset(c) => c.span,
            Self::CreateFlow(c) => c.span,
            Self::LoadData(l) => l.span,
            Self::InsertDirectory(i) => i.span,
        }
    }
}

/// `EXPLAIN [EXTENDED | FORMATTED | CODEGEN | COST] statement`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Explain {
    /// The mode keyword.
    pub mode: ExplainMode,
    /// The statement explained (an `EXPLAIN` may explain an `EXPLAIN`).
    pub statement: Box<Statement>,
    /// Source text, from `EXPLAIN` to the end of the statement.
    pub span: Span,
}

/// The mode of an [`Explain`], as Spark names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExplainMode {
    /// No keyword: the physical plan.
    Simple,
    /// `EXTENDED`: the parsed, analyzed, optimized and physical plans.
    Extended,
    /// `FORMATTED`: the physical plan outline and node details.
    Formatted,
    /// `CODEGEN`: the physical plan and its generated code.
    Codegen,
    /// `COST`: the optimized plan with statistics.
    Cost,
    /// `ANALYZE` (a Krishiv extension; Spark has none): runs the query and
    /// prints the physical plan with each operator's metrics.
    Analyze,
    /// `VERBOSE` (a Krishiv extension): the logical plan after each
    /// optimizer rule that changed it, then the physical plan.
    Verbose,
}

/// `SET`, `SET -v`, `SET key`, `SET key = value` or `SET TIME ZONE ...`:
/// the key of an assignment or lookup is decoded only when backquoted,
/// and the value of an assignment is the raw rest of the statement,
/// trimmed (decoded only when it is one backquoted token).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Set {
    /// What it is.
    pub kind: SetKind,
    /// Source text, from `SET` to the end of the statement.
    pub span: Span,
}

/// The kinds of [`Set`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SetKind {
    /// `SET`: every explicitly set key.
    All,
    /// `SET -v`: every key with its meaning and version.
    Verbose,
    /// `SET key`: the one row of `key`.
    Show {
        /// The key, decoded when backquoted.
        key: String,
        /// The key as written.
        key_span: Span,
    },
    /// `SET key = value`.
    Assign {
        /// The key, decoded when backquoted.
        key: String,
        /// The key as written.
        key_span: Span,
        /// The value: the raw rest of the statement, trimmed, or the
        /// decoded value when it is one backquoted token.
        value: String,
        /// The value as written (trimmed).
        value_span: Span,
    },
    /// `SET TIME ZONE ...`: the zone to use.
    TimeZone {
        /// The zone.
        zone: TimeZone,
        /// The zone as written.
        zone_span: Span,
    },
    /// `SET PATH = element, ...`: the session's SQL path.
    Path {
        /// The elements, in order.
        elements: Vec<krishiv_common::sql_path::PathElement>,
    },
}

/// `SET TIME ZONE LOCAL` or `SET TIME ZONE '<zone id>'`: the zone to use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TimeZone {
    /// `LOCAL`: the system's time zone.
    Local,
    /// A zone id as written (unescaped).
    Named(String),
}

/// `RESET` or `RESET key`: the key is decoded when backquoted, and a bare
/// key holds no whitespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Reset {
    /// The key to reset; `None` resets every key the session set.
    pub key: Option<String>,
    /// The key as written.
    pub key_span: Option<Span>,
    /// Source text, from `RESET` to the end of the statement.
    pub span: Span,
}

/// `CREATE [OR REPLACE] TEMPORARY VIEW name [(columns)] [DEFAULT COLLATION
/// name] AS query` (`TEMP` for `TEMPORARY`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateView {
    /// `OR REPLACE`.
    pub or_replace: bool,
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// The view name (temporary views take one part).
    pub name: Ident,
    /// The declared column aliases; empty for none.
    pub columns: Vec<ViewColumn>,
    /// `DEFAULT COLLATION name`.
    pub collation: Option<Ident>,
    /// The query.
    pub query: Box<Query>,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// `name [COMMENT '...']` in a `CREATE VIEW` column list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ViewColumn {
    /// The column alias.
    pub name: Ident,
    /// Its comment, if written.
    pub comment: Option<String>,
    /// Source text.
    pub span: Span,
}

/// `DROP VIEW [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropView {
    /// `IF EXISTS`.
    pub if_exists: bool,
    /// The view name (temporary views take one part).
    pub name: Ident,
    /// Source text, from `DROP` to the end of the statement.
    pub span: Span,
}

/// `CREATE [OR REPLACE] [TEMPORARY] FUNCTION [IF NOT EXISTS] name
/// ([parameters]) [RETURNS {type | TABLE [(columns)]}] [characteristics]
/// RETURN {expression | query}`: a SQL-bodied user function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateFunction {
    /// `OR REPLACE`.
    pub or_replace: bool,
    /// `TEMPORARY` (`TEMP`).
    pub temporary: bool,
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// The function name.
    pub name: ObjectName,
    /// The parameters: column definitions, as Spark's grammar reads them
    /// (`NOT NULL`, `GENERATED` and constraints parse and are rejected
    /// after parsing).
    pub params: Vec<ColumnDef>,
    /// The text between the parentheses of the parameter list (Spark's
    /// `NOT_NULL_ON_FUNCTION_PARAMETERS` quotes it).
    pub params_span: Span,
    /// `RETURNS ...`; omitted, the type is that of the body.
    pub returns: Option<FunctionReturns>,
    /// The routine characteristics, in whatever order they were written.
    pub characteristics: RoutineCharacteristics,
    /// The body after `RETURN`.
    pub body: FunctionBody,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// The `RETURNS` clause of a [`CreateFunction`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionReturns {
    /// What it returns.
    pub kind: FunctionReturnsKind,
    /// Source text, from `RETURNS` to the end of the clause.
    pub span: Span,
}

/// The kinds of [`FunctionReturns`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FunctionReturnsKind {
    /// `RETURNS type`: a scalar function.
    Type(DataTypeSpec),
    /// `RETURNS TABLE [(columns)]`: a table function; without a column
    /// list the columns are those of the body.
    Table(Option<Vec<ReturnColumn>>),
}

/// `name type [NOT NULL] [COMMENT '...']` in `RETURNS TABLE (...)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReturnColumn {
    /// The column name.
    pub name: Ident,
    /// Its type.
    pub data_type: DataTypeSpec,
    /// `NOT NULL` (parsed; Spark rejects it after parsing).
    pub not_null: bool,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// Source text.
    pub span: Span,
}

/// The routine characteristics of a [`CreateFunction`]: each at most once
/// (a repeat is kept in [`Self::duplicate`] and fails after parsing, as in
/// Spark).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct RoutineCharacteristics {
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// `LANGUAGE name`.
    pub language: Option<Ident>,
    /// `DETERMINISTIC` (`true`) or `NOT DETERMINISTIC` (`false`).
    pub deterministic: Option<bool>,
    /// `CONTAINS SQL` or `READS SQL DATA`.
    pub data_access: Option<SqlDataAccess>,
    /// `SQL SECURITY INVOKER` or `DEFINER`.
    pub security: Option<SqlSecurity>,
    /// The first characteristic written twice, as Spark names its clause
    /// in `DUPLICATE_CLAUSES`.
    pub duplicate: Option<&'static str>,
    /// Source text of the characteristics (empty when there are none).
    pub span: Span,
}

/// `CONTAINS SQL` or `READS SQL DATA`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SqlDataAccess {
    /// `CONTAINS SQL`.
    ContainsSql,
    /// `READS SQL DATA`.
    ReadsSqlData,
}

/// `SQL SECURITY INVOKER` or `SQL SECURITY DEFINER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SqlSecurity {
    /// `SQL SECURITY INVOKER`.
    Invoker,
    /// `SQL SECURITY DEFINER`.
    Definer,
}

/// What follows `RETURN` in a [`CreateFunction`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionBody {
    /// An expression or a query.
    pub kind: FunctionBodyKind,
    /// Source text of the body, which the function's plan keeps so errors
    /// raised while the function is used point into it.
    pub span: Span,
}

/// The kinds of [`FunctionBody`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FunctionBodyKind {
    /// `RETURN expression`.
    Expr(Box<Expr>),
    /// `RETURN query`: a table function's body (a scalar function's is a
    /// scalar subquery).
    Query(Box<Query>),
}

/// `DROP [TEMPORARY] FUNCTION [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropFunction {
    /// `TEMPORARY` (`TEMP`).
    pub temporary: bool,
    /// `IF EXISTS`.
    pub if_exists: bool,
    /// The function name.
    pub name: ObjectName,
    /// Source text, from `DROP` to the end of the statement.
    pub span: Span,
}

/// Calls `f` on each expression of the inline window specification `window`
/// (a named window holds none).
fn for_each_window_expr<'a>(window: &'a WindowRef, f: &mut impl FnMut(&'a Expr)) {
    let WindowRef::Spec(spec) = window else {
        return;
    };
    spec.partition_by.iter().for_each(&mut *f);
    for item in &spec.order_by {
        f(&item.expr);
    }
    if let Some(frame) = &spec.frame {
        for bound in [&frame.start].into_iter().chain(frame.end.as_ref()) {
            match &bound.kind {
                FrameBoundKind::Preceding(expr) | FrameBoundKind::Following(expr) => f(expr),
                _ => {}
            }
        }
    }
}

/// A compound statement: `[label:] BEGIN [NOT ATOMIC] ... END [label]`. Declarations, handlers and
/// statements are kept in source order; where each may stand is checked after parsing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Compound {
    /// The label before `BEGIN`.
    pub label: Option<Ident>,
    /// The label after `END`.
    pub end_label: Option<Ident>,
    /// Everything between `BEGIN` and `END`, in source order.
    pub body: Vec<ScriptStatement>,
    /// Source text, from `BEGIN` (or the label) to `END` (or its label).
    pub span: Span,
}

/// One element of a script body.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScriptStatement {
    /// Any SQL statement the parser accepts, except a script.
    Sql(Box<Statement>),
    /// `DECLARE [OR REPLACE] [VARIABLE] names [type] [DEFAULT expr]`, a local variable.
    Declare(Box<DeclareVariable>),
    /// `DECLARE name CONDITION [FOR SQLSTATE 's']`.
    Condition(DeclareCondition),
    /// `DECLARE {EXIT | CONTINUE} HANDLER FOR conditions body`.
    Handler(Box<Handler>),
    /// `SET name = expr [, ...]` or `SET (names) = (query)`: an assignment, never a configuration key.
    Assign(SetVariable),
    /// `IF ... END IF`.
    If(Box<IfStatement>),
    /// `CASE ... END CASE`.
    Case(Box<CaseStatement>),
    /// `[label:] WHILE cond DO ... END WHILE`.
    While(Box<WhileStatement>),
    /// `[label:] REPEAT ... UNTIL cond END REPEAT`.
    Repeat(Box<RepeatStatement>),
    /// `[label:] LOOP ... END LOOP`.
    Loop(Box<LoopStatement>),
    /// `[label:] FOR [var AS] query DO ... END FOR`.
    For(Box<ForStatement>),
    /// `LEAVE label`.
    Leave(LabelJump),
    /// `ITERATE label`.
    Iterate(LabelJump),
    /// A nested `BEGIN ... END`.
    Compound(Box<Compound>),
    /// `DECLARE ... CURSOR`, `OPEN`, `FETCH` or `CLOSE`: parsed so that the refusal comes from the
    /// checks, as in Spark.
    Cursor(Span),
}

impl ScriptStatement {
    /// The source text of the element.
    #[must_use]
    pub fn span(&self) -> Span {
        match self {
            Self::Sql(s) => s.span(),
            Self::Declare(d) => d.span,
            Self::Condition(c) => c.span,
            Self::Handler(h) => h.span,
            Self::Assign(a) => a.span,
            Self::If(i) => i.span,
            Self::Case(c) => c.span,
            Self::While(w) => w.span,
            Self::Repeat(r) => r.span,
            Self::Loop(l) => l.span,
            Self::For(f) => f.span,
            Self::Leave(j) | Self::Iterate(j) => j.span,
            Self::Compound(c) => c.span,
            Self::Cursor(span) => *span,
        }
    }
}

/// `DECLARE name CONDITION [FOR SQLSTATE [VALUE] 's']`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclareCondition {
    /// The condition name (one part; a qualified name is a syntax error).
    pub name: Ident,
    /// The SQLSTATE text and where it is.
    pub sqlstate: Option<(String, Span)>,
    /// Source text.
    pub span: Span,
}

/// `DECLARE {EXIT | CONTINUE} HANDLER FOR conditions body`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Handler {
    /// `CONTINUE` was written (Spark 4.2 refuses it after parsing).
    pub continue_handler: bool,
    /// The conditions it handles.
    pub conditions: Vec<HandlerCondition>,
    /// What it runs.
    pub body: HandlerBody,
    /// Source text.
    pub span: Span,
}

/// One condition of a [`Handler`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HandlerCondition {
    /// `SQLEXCEPTION`.
    SqlException(Span),
    /// `NOT FOUND`.
    NotFound(Span),
    /// `SQLSTATE [VALUE] 's'`: the text and where it is.
    SqlState(String, Span),
    /// A declared condition or an error class.
    Name(Ident),
}

/// What a [`Handler`] runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HandlerBody {
    /// `BEGIN ... END`.
    Compound(Box<Compound>),
    /// One SQL statement.
    Sql(Box<Statement>),
    /// `SET ...` as an assignment.
    Assign(SetVariable),
}

/// `IF cond THEN ... [ELSEIF cond THEN ...]... [ELSE ...] END IF`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IfStatement {
    /// The `IF` and `ELSEIF` branches, in order.
    pub branches: Vec<ConditionalBranch>,
    /// The `ELSE` body, if written.
    pub else_body: Option<Vec<ScriptStatement>>,
    /// Source text.
    pub span: Span,
}

/// `cond THEN body` of an [`IfStatement`] or a searched [`CaseStatement`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConditionalBranch {
    /// The condition (for a simple `CASE`, the value compared with the subject).
    pub condition: Expr,
    /// The statements it runs.
    pub body: Vec<ScriptStatement>,
}

/// `CASE [subject] WHEN ... THEN ... [ELSE ...] END CASE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CaseStatement {
    /// The subject of a simple `CASE`; none for a searched one.
    pub subject: Option<Expr>,
    /// The `WHEN` branches, in order.
    pub branches: Vec<ConditionalBranch>,
    /// The `ELSE` body, if written.
    pub else_body: Option<Vec<ScriptStatement>>,
    /// Source text.
    pub span: Span,
}

/// `[label:] WHILE cond DO ... END WHILE [label]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WhileStatement {
    /// The label before `WHILE`.
    pub label: Option<Ident>,
    /// The label after `END WHILE`.
    pub end_label: Option<Ident>,
    /// The loop condition.
    pub condition: Expr,
    /// The body.
    pub body: Vec<ScriptStatement>,
    /// Source text.
    pub span: Span,
}

/// `[label:] REPEAT ... UNTIL cond END REPEAT [label]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepeatStatement {
    /// The label before `REPEAT`.
    pub label: Option<Ident>,
    /// The label after `END REPEAT`.
    pub end_label: Option<Ident>,
    /// The body.
    pub body: Vec<ScriptStatement>,
    /// The `UNTIL` condition.
    pub condition: Expr,
    /// Source text.
    pub span: Span,
}

/// `[label:] LOOP ... END LOOP [label]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoopStatement {
    /// The label before `LOOP`.
    pub label: Option<Ident>,
    /// The label after `END LOOP`.
    pub end_label: Option<Ident>,
    /// The body.
    pub body: Vec<ScriptStatement>,
    /// Source text.
    pub span: Span,
}

/// `[label:] FOR [var AS] query DO ... END FOR [label]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForStatement {
    /// The label before `FOR`.
    pub label: Option<Ident>,
    /// The label after `END FOR`.
    pub end_label: Option<Ident>,
    /// The variable naming the row (`var AS`).
    pub variable: Option<Ident>,
    /// The query iterated.
    pub query: Query,
    /// The body.
    pub body: Vec<ScriptStatement>,
    /// Source text.
    pub span: Span,
}

/// `LEAVE label` or `ITERATE label`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabelJump {
    /// The label.
    pub label: Ident,
    /// Source text.
    pub span: Span,
}

mod query_nodes;
pub use query_nodes::*;

mod ddl_nodes;
pub use ddl_nodes::*;

mod pipeline_nodes;
pub use pipeline_nodes::*;
