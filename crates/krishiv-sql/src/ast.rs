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

/// A query: `[WITH ...] body [ORDER BY ...] [LIMIT ...] [OFFSET ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Query {
    /// The `WITH` clause.
    pub with: Option<With>,
    /// The set of rows.
    pub body: SetExpr,
    /// `ORDER BY`.
    pub order_by: Vec<SortItem>,
    /// `CLUSTER BY`.
    pub cluster_by: Vec<Expr>,
    /// `DISTRIBUTE BY`.
    pub distribute_by: Vec<Expr>,
    /// `SORT BY`.
    pub sort_by: Vec<SortItem>,
    /// A `WINDOW` clause after the query organization.
    pub windows: Vec<NamedWindow>,
    /// `LIMIT`.
    pub limit: Option<Limit>,
    /// `OFFSET`.
    pub offset: Option<Expr>,
    /// Source text.
    pub span: Span,
}

/// `LIMIT n` or `LIMIT ALL`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Limit {
    /// The row count; `None` for `ALL`.
    pub value: Option<Expr>,
    /// Source text.
    pub span: Span,
}

/// `WITH [RECURSIVE] cte, ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct With {
    /// `RECURSIVE`.
    pub recursive: bool,
    /// The common table expressions.
    pub ctes: Vec<Cte>,
    /// Source text.
    pub span: Span,
}

/// `name [(columns)] [MAX RECURSION LEVEL n] AS (query)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cte {
    /// The name.
    pub name: Ident,
    /// Column aliases.
    pub columns: Vec<Ident>,
    /// `MAX RECURSION LEVEL`, as written.
    pub max_recursion_level: Option<String>,
    /// The query.
    pub query: Box<Query>,
    /// Source text.
    pub span: Span,
}

/// `DISTINCT` or `ALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SetQuantifier {
    /// `ALL`.
    All,
    /// `DISTINCT`.
    Distinct,
}

/// A set operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SetOperator {
    /// `UNION`.
    Union,
    /// `INTERSECT`.
    Intersect,
    /// `EXCEPT` or `MINUS`.
    Except,
}

/// A query term: a `SELECT`, a set operation, `VALUES`, `TABLE t`, or a
/// parenthesized query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetExpr {
    /// What it is.
    pub kind: SetExprKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`SetExpr`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SetExprKind {
    /// `SELECT ...`.
    Select(Box<Select>),
    /// `FROM ... SELECT ...`.
    FromStatement(Box<FromStatement>),
    /// `(query)`.
    Query(Box<Query>),
    /// `left op [quantifier] right`.
    SetOp {
        /// The operator.
        op: SetOperator,
        /// `ALL` or `DISTINCT`, if written.
        quantifier: Option<SetQuantifier>,
        /// Left input.
        left: Box<SetExpr>,
        /// Right input.
        right: Box<SetExpr>,
    },
    /// `VALUES ...`.
    Values(Values),
    /// `TABLE name`.
    Table(ObjectName),
}

/// `VALUES row, ... [alias]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Values {
    /// One expression per row (a row constructor for several columns).
    pub rows: Vec<Expr>,
    /// The table alias.
    pub alias: Option<TableAlias>,
    /// Source text.
    pub span: Span,
}

/// A `SELECT` query specification.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Select {
    /// `/*+ ... */` hints.
    pub hints: Vec<Hint>,
    /// `DISTINCT` or `ALL`.
    pub quantifier: Option<SetQuantifier>,
    /// The select list.
    pub projection: Vec<SelectItem>,
    /// The `FROM` clause (none in a body of a `FROM ... SELECT` statement).
    pub from: Option<FromClause>,
    /// `LATERAL VIEW`s after the `FROM` clause's.
    pub lateral_views: Vec<LateralView>,
    /// `WHERE`.
    pub selection: Option<Expr>,
    /// `GROUP BY`.
    pub group_by: Option<GroupBy>,
    /// `HAVING`.
    pub having: Option<Expr>,
    /// `WINDOW`.
    pub windows: Vec<NamedWindow>,
    /// `QUALIFY`.
    pub qualify: Option<Expr>,
    /// Source text.
    pub span: Span,
}

/// `FROM relation, ... [LATERAL VIEW ...] [PIVOT ...] [UNPIVOT ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FromClause {
    /// The relations.
    pub relations: Vec<Relation>,
    /// `LATERAL VIEW`s.
    pub lateral_views: Vec<LateralView>,
    /// `PIVOT` at the end of the clause.
    pub pivot: Option<Pivot>,
    /// `UNPIVOT` at the end of the clause.
    pub unpivot: Option<Unpivot>,
    /// Source text.
    pub span: Span,
}

/// `FROM ... SELECT ... [SELECT ...]`: every body reads the one `FROM`
/// clause (several bodies are a `UNION ALL`, no body selects every column),
/// so it is stored once.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FromStatement {
    /// The shared `FROM` clause.
    pub from: FromClause,
    /// The bodies, each a `SELECT` without `FROM` and its organization.
    pub bodies: Vec<Query>,
    /// Source text.
    pub span: Span,
}

/// One hint of a `/*+ ... */` comment: `name [(params)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Hint {
    /// The hint name.
    pub name: Ident,
    /// Its parameters.
    pub params: Vec<Expr>,
    /// Source text.
    pub span: Span,
}

/// `expr [[AS] alias]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SelectItem {
    /// The expression.
    pub expr: Expr,
    /// The alias.
    pub alias: Option<Alias>,
    /// Source text.
    pub span: Span,
}

/// An output alias.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Alias {
    /// `AS name`.
    Name(Ident),
    /// `AS (a, b)` (for generators).
    List(Vec<Ident>),
}

/// `GROUP BY items [WITH ROLLUP | WITH CUBE | GROUPING SETS (...)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupBy {
    /// The grouping items.
    pub items: Vec<GroupItem>,
    /// The trailing grouping analytics.
    pub with: Option<GroupWith>,
    /// Source text.
    pub span: Span,
}

/// Trailing grouping analytics.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GroupWith {
    /// `WITH ROLLUP`.
    Rollup,
    /// `WITH CUBE`.
    Cube,
    /// `GROUPING SETS (...)` after the items.
    GroupingSets(Vec<GroupingSet>),
}

/// A `GROUP BY` item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupItem {
    /// What it is.
    pub kind: GroupItemKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`GroupItem`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GroupItemKind {
    /// An expression (`ALL` is a column named `ALL`).
    Expr(Expr),
    /// `ROLLUP (...)`.
    Rollup(Vec<GroupingSet>),
    /// `CUBE (...)`.
    Cube(Vec<GroupingSet>),
    /// `GROUPING SETS (...)`.
    GroupingSets(Vec<GroupItem>),
    /// A grouping set inside `GROUPING SETS`.
    Set(GroupingSet),
}

/// `(a, b)`, `()` or a single expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupingSet {
    /// The expressions.
    pub exprs: Vec<Expr>,
    /// Source text.
    pub span: Span,
}

/// `name AS window`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamedWindow {
    /// The name.
    pub name: Ident,
    /// The window.
    pub window: WindowRef,
    /// Source text.
    pub span: Span,
}

/// A window after `OVER` or `AS`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WindowRef {
    /// A named window, `w` or `(w)`.
    Named(Ident),
    /// An inline window specification.
    Spec(Box<WindowSpec>),
}

/// The keyword of a window's partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Partitioning {
    /// `PARTITION BY`.
    Partition,
    /// `DISTRIBUTE BY`.
    Distribute,
    /// `CLUSTER BY` (no `ORDER BY` may follow).
    Cluster,
}

/// `([PARTITION BY ...] [ORDER BY ...] [frame])`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowSpec {
    /// The partitioning keyword, when there is a partitioning.
    pub partitioning: Option<Partitioning>,
    /// The partitioning expressions.
    pub partition_by: Vec<Expr>,
    /// `ORDER BY` or `SORT BY`.
    pub order_by: Vec<SortItem>,
    /// The frame.
    pub frame: Option<WindowFrame>,
    /// Source text.
    pub span: Span,
}

/// `ROWS` or `RANGE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameUnits {
    /// `ROWS`.
    Rows,
    /// `RANGE`.
    Range,
}

/// `units start` or `units BETWEEN start AND end`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowFrame {
    /// The units.
    pub units: FrameUnits,
    /// Start bound.
    pub start: FrameBound,
    /// End bound (`BETWEEN` form).
    pub end: Option<FrameBound>,
    /// Source text.
    pub span: Span,
}

/// A frame bound.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FrameBound {
    /// What it is.
    pub kind: FrameBoundKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`FrameBound`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FrameBoundKind {
    /// `UNBOUNDED PRECEDING`.
    UnboundedPreceding,
    /// `UNBOUNDED FOLLOWING`.
    UnboundedFollowing,
    /// `CURRENT ROW`.
    CurrentRow,
    /// `expr PRECEDING`.
    Preceding(Expr),
    /// `expr FOLLOWING`.
    Following(Expr),
}

/// `expr [ASC | DESC] [NULLS FIRST | NULLS LAST]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortItem {
    /// The sort key.
    pub expr: Expr,
    /// `ASC` (`true`) or `DESC` (`false`).
    pub asc: Option<bool>,
    /// `NULLS FIRST` (`true`) or `NULLS LAST` (`false`).
    pub nulls_first: Option<bool>,
    /// Source text.
    pub span: Span,
}

/// `[LATERAL] primary extension*` in a `FROM` clause.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Relation {
    /// `LATERAL`.
    pub lateral: bool,
    /// The leftmost relation.
    pub primary: TableFactor,
    /// Joins, pivots and unpivots, left to right.
    pub extensions: Vec<RelationExt>,
    /// Source text.
    pub span: Span,
}

/// What follows a relation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RelationExt {
    /// A join.
    Join(Join),
    /// `PIVOT (...)`.
    Pivot(Pivot),
    /// `UNPIVOT (...)`.
    Unpivot(Unpivot),
}

/// `[AS] name [(columns)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableAlias {
    /// The alias.
    pub name: Ident,
    /// Column aliases.
    pub columns: Vec<Ident>,
    /// Source text.
    pub span: Span,
}

/// A relation primary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableFactor {
    /// What it is.
    pub kind: TableFactorKind,
    /// The alias.
    pub alias: Option<TableAlias>,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`TableFactor`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TableFactorKind {
    /// A table or view, with time travel and options.
    Table {
        /// The name.
        name: ObjectName,
        /// `VERSION AS OF` / `TIMESTAMP AS OF`.
        temporal: Option<TemporalClause>,
        /// `WITH (key = value, ...)`.
        options: Vec<TableOption>,
    },
    /// `(query)`.
    Subquery(Box<Query>),
    /// `(relation)`.
    Nested(Box<Relation>),
    /// `VALUES ...` (its alias is the factor's alias).
    Values(Vec<Expr>),
    /// A table-valued function call.
    TableFunction {
        /// The function name.
        name: ObjectName,
        /// The arguments.
        args: Vec<FunctionArg>,
    },
}

/// What a time-travel clause selects by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TemporalKind {
    /// `[FOR] VERSION | SYSTEM_VERSION AS OF version`.
    Version,
    /// `[FOR] TIMESTAMP | SYSTEM_TIME AS OF expression`.
    Timestamp,
}

/// Time travel: `VERSION AS OF 1`, `TIMESTAMP AS OF '2020-01-01'`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TemporalClause {
    /// Version or timestamp.
    pub kind: TemporalKind,
    /// The version (an integer or string literal) or timestamp.
    pub value: Expr,
    /// Source text.
    pub span: Span,
}

/// A table option, `key [=] value`. Keys and values keep their spans, so
/// secrets in options can be masked from the source text.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableOption {
    /// The key.
    pub key: OptionKey,
    /// The value (an integer, decimal, boolean or string literal).
    pub value: Option<Expr>,
    /// Source text.
    pub span: Span,
}

/// An option key: a dotted name or a string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptionKey {
    /// The name parts; one part holding the text for a string key.
    pub parts: Vec<Ident>,
    /// Written as a string literal.
    pub is_string: bool,
    /// Source text.
    pub span: Span,
}

/// A join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JoinKind {
    /// `[INNER] JOIN`.
    Inner,
    /// `CROSS JOIN`.
    Cross,
    /// `LEFT [OUTER] JOIN`.
    Left,
    /// `RIGHT [OUTER] JOIN`.
    Right,
    /// `FULL [OUTER] JOIN`.
    Full,
    /// `[LEFT] SEMI JOIN`.
    LeftSemi,
    /// `[LEFT] ANTI JOIN`.
    LeftAnti,
}

/// The join condition.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum JoinConstraint {
    /// None written.
    None,
    /// `ON expr`.
    On(Expr),
    /// `USING (columns)`.
    Using(Vec<Ident>),
}

/// `[NATURAL] kind JOIN [LATERAL] right [constraint]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Join {
    /// The join type.
    pub kind: JoinKind,
    /// `NATURAL`.
    pub natural: bool,
    /// `LATERAL` before the right side.
    pub lateral: bool,
    /// The right side.
    pub right: Box<TableFactor>,
    /// The condition.
    pub constraint: JoinConstraint,
    /// Source text.
    pub span: Span,
}

/// `PIVOT (aggregates FOR columns IN (values)) [alias]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pivot {
    /// The aggregate expressions.
    pub aggregates: Vec<SelectItem>,
    /// The pivot columns.
    pub columns: Vec<Ident>,
    /// The pivot values.
    pub values: Vec<SelectItem>,
    /// The alias.
    pub alias: Option<Ident>,
    /// Source text.
    pub span: Span,
}

/// `UNPIVOT [INCLUDE | EXCLUDE NULLS] (values FOR name IN (columns)) [alias]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Unpivot {
    /// `INCLUDE NULLS` (`true`) or `EXCLUDE NULLS` (`false`).
    pub include_nulls: Option<bool>,
    /// The value columns (several in the multi-value form).
    pub value_columns: Vec<Ident>,
    /// The name column.
    pub name_column: Ident,
    /// The unpivoted column sets.
    pub columns: Vec<UnpivotColumns>,
    /// The alias.
    pub alias: Option<Ident>,
    /// Source text.
    pub span: Span,
}

/// `col [alias]` or `(col, ...) [alias]` in an `UNPIVOT`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnpivotColumns {
    /// The columns.
    pub columns: Vec<ObjectName>,
    /// The alias.
    pub alias: Option<Ident>,
    /// Source text.
    pub span: Span,
}

/// `LATERAL VIEW [OUTER] func(args) table [AS] columns`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LateralView {
    /// `OUTER`.
    pub outer: bool,
    /// The generator function.
    pub function: ObjectName,
    /// Its arguments.
    pub args: Vec<Expr>,
    /// The table alias.
    pub table: Ident,
    /// The column aliases.
    pub columns: Vec<Ident>,
    /// Source text.
    pub span: Span,
}

/// The type suffix of a numeric literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NumKind {
    /// Digits only.
    Int,
    /// `L`.
    Long,
    /// `S`.
    Short,
    /// `Y`.
    Byte,
    /// Digits with a point.
    Decimal,
    /// With an exponent.
    Exponent,
    /// `D`.
    Double,
    /// `F`.
    Float,
    /// `BD`.
    BigDecimal,
}

/// A literal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Literal {
    /// `NULL`.
    Null,
    /// `TRUE` or `FALSE`.
    Boolean(bool),
    /// Adjacent string literals, concatenated and unescaped.
    String(String),
    /// A number: its text (with a leading `-` if negated, without the
    /// suffix) and kind.
    Number {
        /// The digits.
        text: String,
        /// The suffix.
        kind: NumKind,
    },
    /// `type 'value'`, e.g. `DATE '2020-01-01'` or `X'1F'`; the type name is
    /// an identifier as written (quoted names too: Spark rejects those when
    /// building the plan).
    Typed {
        /// The type name.
        type_name: Ident,
        /// The string value.
        value: String,
    },
    /// An `INTERVAL` with units.
    Interval(Box<Interval>),
}

/// `INTERVAL value unit ...` or `INTERVAL value from TO to`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Interval {
    /// `value unit` pairs; the unit is lower-case.
    pub units: Vec<(IntervalValue, String)>,
    /// `value from TO to`.
    pub range: Option<(IntervalValue, String, String)>,
    /// A second interval written after this one (`INTERVAL 1 DAY '1-2' YEAR
    /// TO MONTH`): Spark parses it and then rejects the literal.
    pub trailing: Option<Box<Interval>>,
    /// Source text.
    pub span: Span,
}

/// An interval value: `[+|-] (number | string)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntervalValue {
    /// A leading `-`.
    pub negative: bool,
    /// The number text or the unescaped string.
    pub value: String,
    /// The value was a string literal.
    pub is_string: bool,
    /// Source text.
    pub span: Span,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    /// `+`.
    Plus,
    /// `-`.
    Minus,
    /// `~`.
    BitNot,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the operator named")]
#[non_exhaustive]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    NullSafeEq,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    IntegerDivide,
    Concat,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    ShiftRightUnsigned,
}

/// The right side of `IS [NOT] TRUE | FALSE | UNKNOWN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the keyword named")]
#[non_exhaustive]
pub enum Truth {
    True,
    False,
    Unknown,
}

/// The quantifier of `LIKE ANY | SOME | ALL (...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the keyword named")]
#[non_exhaustive]
pub enum LikeQuantifier {
    Any,
    Some,
    All,
}

/// A data type as written (Spark's `dataType` rule).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataTypeSpec {
    /// What it is.
    pub kind: DataTypeKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`DataTypeSpec`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DataTypeKind {
    /// A named type with integer arguments: `name` is lower-case and holds
    /// every keyword (`timestamp with local time zone`, `interval day to
    /// second`).
    Primitive {
        /// The type name.
        name: String,
        /// Its arguments as written (`ANY` for geometry types).
        args: Vec<String>,
    },
    /// `ARRAY<element>`, or bare `ARRAY`.
    Array(Option<Box<DataTypeSpec>>),
    /// `MAP<key, value>`, or bare `MAP`.
    Map(Option<Box<(DataTypeSpec, DataTypeSpec)>>),
    /// `STRUCT<fields>`, `STRUCT<>`, or bare `STRUCT` (`None`).
    Struct(Option<Vec<StructField>>),
}

/// `name [:] type [NOT NULL] [COMMENT '...']` in a `STRUCT`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    /// The field name.
    pub name: Ident,
    /// Its type.
    pub data_type: DataTypeSpec,
    /// `NOT NULL`.
    pub not_null: bool,
    /// `COMMENT`.
    pub comment: Option<String>,
    /// Source text.
    pub span: Span,
}

/// `IGNORE NULLS` or `RESPECT NULLS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the keyword named")]
#[non_exhaustive]
pub enum NullTreatment {
    Ignore,
    Respect,
}

/// A function argument, `expr` or `name => expr`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionArg {
    /// The parameter name.
    pub name: Option<Ident>,
    /// The value.
    pub expr: Expr,
    /// Source text.
    pub span: Span,
}

/// A function call. Spark's special forms (`EXTRACT`, `SUBSTRING ... FROM`,
/// `TRIM`, `POSITION`, `OVERLAY`, `TIMESTAMPADD`, ...) are calls of the
/// function they denote, with the arguments in that function's order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    /// The function name.
    pub name: ObjectName,
    /// `DISTINCT` or `ALL`.
    pub quantifier: Option<SetQuantifier>,
    /// The arguments.
    pub args: Vec<FunctionArg>,
    /// `WITHIN GROUP (ORDER BY ...)`.
    pub within_group: Vec<SortItem>,
    /// `FILTER (WHERE ...)`.
    pub filter: Option<Box<Expr>>,
    /// `IGNORE NULLS` or `RESPECT NULLS`.
    pub null_treatment: Option<NullTreatment>,
    /// `OVER window`.
    pub over: Option<WindowRef>,
    /// Source text.
    pub span: Span,
}

/// `WHEN condition THEN result`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WhenClause {
    /// The condition (or the value compared with the `CASE` operand).
    pub condition: Expr,
    /// The result.
    pub result: Expr,
    /// Source text.
    pub span: Span,
}

/// An expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Expr {
    /// What it is.
    pub kind: ExprKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`Expr`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExprKind {
    /// A literal.
    Literal(Literal),
    /// The positional parameter marker `?`, with its 0-based ordinal in
    /// the statement.
    Parameter(usize),
    /// A keyword argument of a special form: the unit of `TIMESTAMPADD(DAY,
    /// ...)` or the field of `EXTRACT(YEAR FROM ...)` (unlike a string
    /// literal, which Spark rejects there).
    Keyword(Ident),
    /// A column, possibly qualified: `a`, `t.a`.
    Column(Vec<Ident>),
    /// `*` or `qualifier.*`, with `EXCEPT (columns)`.
    Wildcard {
        /// The qualifier.
        qualifier: Vec<Ident>,
        /// Excluded columns.
        except: Vec<ObjectName>,
    },
    /// `expr.field`.
    Field(Box<Expr>, Ident),
    /// `expr[index]`.
    Subscript(Box<Expr>, Box<Expr>),
    /// A unary operator.
    Unary(UnaryOp, Box<Expr>),
    /// A binary operator; chains of `AND` / `OR` are balanced trees, as
    /// Spark's `AstBuilder` builds them.
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    /// `NOT expr` or `!expr`.
    Not(Box<Expr>),
    /// `expr IS [NOT] NULL`.
    IsNull {
        /// The operand.
        expr: Box<Expr>,
        /// `NOT`.
        negated: bool,
    },
    /// `expr IS [NOT] TRUE | FALSE | UNKNOWN`.
    IsTruth {
        /// The operand.
        expr: Box<Expr>,
        /// The truth value.
        truth: Truth,
        /// `NOT`.
        negated: bool,
    },
    /// `left IS [NOT] DISTINCT FROM right`.
    IsDistinctFrom {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
        /// `NOT`.
        negated: bool,
    },
    /// `expr [NOT] BETWEEN low AND high`.
    Between {
        /// The operand.
        expr: Box<Expr>,
        /// Lower bound.
        low: Box<Expr>,
        /// Upper bound.
        high: Box<Expr>,
        /// `NOT`.
        negated: bool,
    },
    /// `expr [NOT] IN (list)`.
    InList {
        /// The operand.
        expr: Box<Expr>,
        /// The list.
        list: Vec<Expr>,
        /// `NOT`.
        negated: bool,
    },
    /// `expr [NOT] IN (query)`.
    InSubquery {
        /// The operand.
        expr: Box<Expr>,
        /// The query.
        query: Box<Query>,
        /// `NOT`.
        negated: bool,
    },
    /// `expr [NOT] LIKE | ILIKE pattern [ESCAPE 'c']`.
    Like {
        /// The operand.
        expr: Box<Expr>,
        /// The pattern.
        pattern: Box<Expr>,
        /// The escape string.
        escape: Option<String>,
        /// `NOT`.
        negated: bool,
        /// `ILIKE`.
        case_insensitive: bool,
    },
    /// `expr [NOT] LIKE | ILIKE ANY | SOME | ALL (patterns)`.
    LikeQuantified {
        /// The operand.
        expr: Box<Expr>,
        /// The quantifier.
        quantifier: LikeQuantifier,
        /// The patterns.
        patterns: Vec<Expr>,
        /// `NOT`.
        negated: bool,
        /// `ILIKE`.
        case_insensitive: bool,
    },
    /// `expr [NOT] RLIKE | REGEXP pattern`.
    RLike {
        /// The operand.
        expr: Box<Expr>,
        /// The pattern.
        pattern: Box<Expr>,
        /// `NOT`.
        negated: bool,
    },
    /// `EXISTS (query)`.
    Exists(Box<Query>),
    /// A scalar subquery.
    Subquery(Box<Query>),
    /// `CAST(expr AS type)`, `TRY_CAST(...)` or `expr::type`.
    Cast {
        /// The operand.
        expr: Box<Expr>,
        /// The target type.
        data_type: Box<DataTypeSpec>,
        /// `TRY_CAST`.
        try_cast: bool,
    },
    /// `CASE [operand] WHEN ... [ELSE ...] END`.
    Case {
        /// The operand of a simple `CASE`.
        operand: Option<Box<Expr>>,
        /// The branches.
        branches: Vec<WhenClause>,
        /// `ELSE`.
        else_result: Option<Box<Expr>>,
    },
    /// A function call.
    Function(Box<Function>),
    /// `x -> body` or `(x, y) -> body`.
    Lambda {
        /// The parameters.
        params: Vec<Ident>,
        /// The body.
        body: Box<Expr>,
    },
    /// `(a, b AS c)`.
    Row(Vec<SelectItem>),
    /// `STRUCT(a, b AS c)`.
    Struct(Vec<SelectItem>),
    /// `(expr)`.
    Nested(Box<Expr>),
}
