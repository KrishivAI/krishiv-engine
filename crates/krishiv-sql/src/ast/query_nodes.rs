// SPDX-License-Identifier: Apache-2.0
//! AST nodes of queries and expressions: select, FROM, joins, windows and expression kinds, plus the ANALYZE, REPAIR and resource utility nodes.

use super::*;

/// Which columns an [`Analyze`] computes statistics for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AnalyzeColumns {
    /// The whole table.
    All,
    /// `FOR ALL COLUMNS`.
    ForAll,
    /// `FOR COLUMNS c, ...`.
    ForColumns(Vec<Ident>),
}

/// `MSCK REPAIR TABLE ...` or `REPAIR TABLE ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Repair {
    /// Whether `MSCK` was written.
    pub msck: bool,
    /// The table.
    pub table: DdlName,
    /// `ADD`, `DROP` or `SYNC PARTITIONS`, if written.
    pub action: Option<RepairAction>,
    /// Source text, from `MSCK` or `REPAIR` to the end of the statement.
    pub span: Span,
}

/// `ADD|DROP|SYNC PARTITIONS` in a [`Repair`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RepairAction {
    /// `ADD PARTITIONS`.
    Add,
    /// `DROP PARTITIONS`.
    Drop,
    /// `SYNC PARTITIONS`.
    Sync,
}

/// A resource kind for [`AddResource`] and [`ListResource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ResourceType {
    /// `JAR`.
    Jar,
    /// `FILE`.
    File,
    /// `ARCHIVE`.
    Archive,
}

/// `ADD JAR[S]|FILE[S]|ARCHIVE[S] paths`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AddResource {
    /// What to add.
    pub kind: ResourceType,
    /// The paths, quoted or not (quoted ones unescaped); may be empty.
    pub paths: Vec<String>,
    /// Source text, from `ADD` to the end of the statement.
    pub span: Span,
}

/// `LIST JAR[S]|FILE[S]|ARCHIVE[S] [paths]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ListResource {
    /// What to list.
    pub kind: ResourceType,
    /// The paths, quoted or not (quoted ones unescaped); empty for all.
    pub paths: Vec<String>,
    /// Source text, from `LIST` to the end of the statement.
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
    /// The `ORDER BY` clause, as Spark's errors cite the sort.
    pub order_span: Option<Span>,
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
    /// The query-organization clauses (`ORDER BY` to `OFFSET`), as Spark's
    /// errors cite a `SORT BY`, `DISTRIBUTE BY` or `CLUSTER BY`; `None`
    /// without any.
    pub organization_span: Option<Span>,
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
    /// `TABLE IDENTIFIER(expr)`.
    DynamicTable(DynamicName),
    /// `input |> operator |> ...`.
    Pipe(Pipe),
    /// `SELECT TRANSFORM (...) USING 'script' ...`, `MAP ...` or `REDUCE ...`.
    Transform(Box<TransformSelect>),
}

/// A script transformation, parsed and not run: `SELECT TRANSFORM (inputs)
/// [ROW FORMAT ...] [RECORDWRITER 'class'] USING 'script' [AS output]
/// [ROW FORMAT ...] [RECORDREADER 'class'] [FROM ...] [WHERE ...] ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransformSelect {
    /// Which keyword introduced it.
    pub kind: TransformKind,
    /// `DISTINCT` or `ALL`, which Spark parses and rejects.
    pub quantifier: Option<SetQuantifier>,
    /// How rows are written to the script.
    pub input_format: Option<RowFormat>,
    /// `RECORDWRITER 'class'`.
    pub record_writer: Option<String>,
    /// The script, `USING 'script'`.
    pub script: String,
    /// `AS (columns)`: names, or names with types.
    pub output: Option<TransformOutput>,
    /// How rows are read from the script.
    pub output_format: Option<RowFormat>,
    /// `RECORDREADER 'class'`.
    pub record_reader: Option<String>,
    /// The inputs as the select list (no aliases), with the `FROM` clause
    /// and the filtering and grouping clauses after the transform.
    pub select: Select,
    /// Source text.
    pub span: Span,
}

/// The keyword that starts a [`TransformSelect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the keyword named")]
#[non_exhaustive]
pub enum TransformKind {
    SelectTransform,
    Map,
    Reduce,
}

/// The `AS` clause of a [`TransformSelect`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransformOutput {
    /// Column names; the columns are strings.
    Names(Vec<Ident>),
    /// Column names with types.
    Columns(Vec<StructField>),
}

/// A query term followed by pipe operators (`|>`, or `|` before an
/// operator keyword). Operators apply left to right, each to the result of
/// the one before; a set operation or a further pipe over the result is a
/// new node around this one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pipe {
    /// The query the operators start from.
    pub input: Box<SetExpr>,
    /// The operators, in source order (never empty).
    pub ops: Vec<PipeOp>,
    /// Source text.
    pub span: Span,
}

/// One pipe operator, from its `|>` to its last token.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeOp {
    /// What it is.
    pub kind: PipeOpKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`PipeOp`] (Spark's `operatorPipeRightSide`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PipeOpKind {
    /// `SELECT ... [GROUP BY ...] [WINDOW ...]`.
    Select(Box<PipeSelect>),
    /// `EXTEND expr [AS name], ...`.
    Extend(Vec<SelectItem>),
    /// `SET name = expr, ...`.
    Set(Vec<PipeAssignment>),
    /// `DROP name, ...`.
    Drop(Vec<ObjectName>),
    /// `AS name`.
    As(Ident),
    /// `WHERE condition`.
    Where(PipeWhere),
    /// `PIVOT (...)`, with the `UNPIVOT` Spark parses after it and rejects.
    Pivot(Box<Pivot>, Option<Box<Unpivot>>),
    /// `UNPIVOT (...)`, with the `PIVOT` Spark parses after it and rejects.
    Unpivot(Box<Unpivot>, Option<Box<Pivot>>),
    /// `TABLESAMPLE (...)`.
    Sample(TableSample),
    /// `[type] JOIN right [ON ... | USING (...) | NEAREST BY ...]`.
    Join(Box<Join>),
    /// `UNION | EXCEPT | MINUS | INTERSECT [ALL | DISTINCT] right`; the
    /// right side is a query primary.
    SetOp {
        /// The operator.
        op: SetOperator,
        /// `ALL` or `DISTINCT`, if written.
        quantifier: Option<SetQuantifier>,
        /// The right input.
        right: Box<SetExpr>,
    },
    /// `ORDER BY`, `LIMIT`, ... (possibly none: the operator is then bare).
    Organization(QueryOrganization),
    /// `AGGREGATE [aggregates] [GROUP BY ...]`.
    Aggregate(Box<PipeAggregate>),
}

/// The `SELECT` of a pipe operator.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeSelect {
    /// `/*+ ... */` hints.
    pub hints: Vec<Hint>,
    /// `DISTINCT` or `ALL`.
    pub quantifier: Option<SetQuantifier>,
    /// The select list.
    pub projection: Vec<SelectItem>,
    /// `GROUP BY`.
    pub group_by: Option<GroupBy>,
    /// `WINDOW`.
    pub windows: Vec<NamedWindow>,
    /// Source text.
    pub span: Span,
}

/// `name = expr` of a pipe `SET`. Spark's grammar reads a dotted key and
/// rejects it afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeAssignment {
    /// The key's parts (one for a valid key).
    pub key: Vec<Ident>,
    /// The new value.
    pub value: Expr,
    /// Source text.
    pub span: Span,
}

/// `WHERE condition` of a pipe operator, with the `WINDOW` clause Spark
/// parses after it and rejects.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeWhere {
    /// The condition.
    pub condition: Expr,
    /// The (invalid) `WINDOW` clause.
    pub windows: Vec<NamedWindow>,
    /// Source text.
    pub span: Span,
}

/// `AGGREGATE [aggregates] [GROUP BY items]`: grouping items may carry
/// aliases ([`GroupItemKind::Named`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeAggregate {
    /// The aggregate expressions.
    pub aggregates: Vec<SelectItem>,
    /// `GROUP BY`.
    pub group_by: Option<GroupBy>,
    /// Source text.
    pub span: Span,
}

/// Spark's `queryOrganization`: the trailing clauses of a query, or the
/// body of a pipe operator that starts with one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryOrganization {
    /// `ORDER BY`.
    pub order_by: Vec<SortItem>,
    /// The `ORDER BY` clause, as Spark's errors cite the sort.
    pub order_span: Option<Span>,
    /// `CLUSTER BY`.
    pub cluster_by: Vec<Expr>,
    /// `DISTRIBUTE BY`.
    pub distribute_by: Vec<Expr>,
    /// `SORT BY`.
    pub sort_by: Vec<SortItem>,
    /// `WINDOW`.
    pub windows: Vec<NamedWindow>,
    /// `LIMIT`.
    pub limit: Option<Limit>,
    /// `OFFSET`.
    pub offset: Option<Expr>,
    /// Source text (the operator's `|>` when no clause is written).
    pub span: Span,
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
    /// `spark.sql.legacy.parser.havingWithoutGroupByAsWhere`: a `HAVING` with no `GROUP BY` over
    /// a select list with no aggregate call filters the rows, as `WHERE` does.
    pub having_as_where: bool,
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
    /// `expr [AS] alias`, in the `GROUP BY` of a pipe `AGGREGATE`.
    Named(SelectItem),
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

/// `[AS] name [(columns)]`, or `AS (columns)` without a name, as in
/// `VALUES (1) AS (c)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableAlias {
    /// The alias; `None` for `AS (columns)`.
    pub name: Option<Ident>,
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
    /// `TABLESAMPLE`, after a table, a subquery or a parenthesized relation.
    pub sample: Option<TableSample>,
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
    /// A table or view named `IDENTIFIER(expr)`.
    DynamicTable {
        /// The name expression.
        name: DynamicName,
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
        /// `IDENTIFIER(expr)` written as the name, which `name` then holds
        /// as the word `IDENTIFIER`.
        dynamic_name: Option<DynamicName>,
        /// The arguments (`TABLE(...)` ones included).
        args: Vec<FunctionArg>,
    },
}

/// `TABLESAMPLE [SYSTEM | BERNOULLI] ([method]) [REPEATABLE (seed)]`, as a
/// relation suffix or a pipe operator. Spark's builder rejects an empty
/// method, `SYSTEM` with `REPEATABLE` or with any method but a percentage,
/// and the byte and column forms; the tree keeps what it needs to say so.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableSample {
    /// `SYSTEM` or `BERNOULLI`, if written.
    pub sample_type: Option<SampleType>,
    /// What to sample; `None` for `TABLESAMPLE ()`.
    pub method: Option<SampleMethod>,
    /// The `REPEATABLE` seed, as written.
    pub seed: Option<String>,
    /// Source text.
    pub span: Span,
}

/// The sampling type keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the keyword named")]
#[non_exhaustive]
pub enum SampleType {
    System,
    Bernoulli,
}

/// The method inside `TABLESAMPLE (...)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SampleMethod {
    /// What it is.
    pub kind: SampleMethodKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`SampleMethod`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleMethodKind {
    /// `[-] number PERCENT`.
    Percent {
        /// A leading `-`.
        negative: bool,
        /// The integer or decimal, as written.
        value: String,
    },
    /// `expr ROWS`.
    Rows(Expr),
    /// `BUCKET numerator OUT OF denominator [ON column | ON function()]`.
    Bucket {
        /// The numerator, as written.
        numerator: String,
        /// The denominator, as written.
        denominator: String,
        /// The `ON` target.
        on: Option<BucketOn>,
    },
    /// A bare expression, which Spark reads as a byte length (`10M`) and
    /// rejects.
    Bytes(Expr),
}

/// The target of `BUCKET ... ON`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BucketOn {
    /// A column.
    Column(Ident),
    /// `name()`.
    Function(ObjectName),
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
    /// `[APPROX | EXACT] NEAREST [k] BY DISTANCE | SIMILARITY expr`.
    NearestBy(NearestBy),
}

/// The ranking clause of a top-k join.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NearestBy {
    /// `APPROX` (`true`) or `EXACT`.
    pub approx: bool,
    /// The number of matches per left row, as written (one if absent).
    pub num_results: Option<String>,
    /// `DISTANCE` (smaller is nearer) or `SIMILARITY` (larger is nearer).
    pub direction: NearestDirection,
    /// The ranking expression.
    pub ranking: Expr,
    /// Source text.
    pub span: Span,
}

/// The direction of a [`NearestBy`] ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[expect(missing_docs, reason = "each variant is the keyword named")]
#[non_exhaustive]
pub enum NearestDirection {
    Distance,
    Similarity,
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

/// The path after the `:` of a variant extraction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantPath {
    /// The parts, in order (never empty).
    pub parts: Vec<VariantPathPart>,
    /// Source text.
    pub span: Span,
}

/// One part of a [`VariantPath`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantPathPart {
    /// What it is.
    pub kind: VariantPathPartKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`VariantPathPart`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VariantPathPartKind {
    /// A field name: the first part, or `.name`.
    Field(Ident),
    /// `['key']`.
    Key(String),
    /// `[0]`.
    Index(String),
    /// `[name]`, after the first part.
    Name(Ident),
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
    /// `COLLATE ...` after a `STRING`, `CHAR` or `VARCHAR` type.
    pub collation: Option<ObjectName>,
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
    /// A quoted identifier as a type name (`` `int` ``), which Spark parses
    /// and then rejects as `UNSUPPORTED_DATATYPE`.
    Quoted {
        /// The name, unquoted.
        name: String,
        /// Its arguments as written.
        args: Vec<String>,
    },
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
/// function they denote, with the arguments in that function's order
/// (`TRIM(c FROM s)` is `trim(c, s)`: the trim string first, as in Spark's
/// two-parameter `trim`, `ltrim` and `rtrim`).
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
    /// `IDENTIFIER(expr)` written as the function's name, which `name`
    /// then holds as the word `IDENTIFIER`.
    pub dynamic_name: Option<DynamicName>,
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
    /// The named parameter marker `:name`.
    NamedParameter(Box<Ident>),
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
        /// A backquoted name read as a regular expression over the column names
        /// (`spark.sql.parser.quotedRegexColumnNames`); the qualifier is its table.
        regex: Option<Box<str>>,
    },
    /// `expr.field`.
    Field(Box<Expr>, Ident),
    /// `expr COLLATE name`.
    Collate(Box<Expr>, ObjectName),
    /// `expr:path` on a variant (`v:a.b[0]`, `v:['key']`).
    VariantGet(Box<Expr>, Box<VariantPath>),
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
        /// From `NOT` (or the keyword) to the end: the predicate as Spark's
        /// errors cite it.
        op: Span,
    },
    /// `expr [NOT] IN (list)`.
    InList {
        /// The operand.
        expr: Box<Expr>,
        /// The list.
        list: Vec<Expr>,
        /// `NOT`.
        negated: bool,
        /// From `NOT` (or the keyword) to the end: the predicate as Spark's
        /// errors cite it.
        op: Span,
    },
    /// `expr [NOT] IN (query)`.
    InSubquery {
        /// The operand.
        expr: Box<Expr>,
        /// The query.
        query: Box<Query>,
        /// `NOT`.
        negated: bool,
        /// From `NOT` (or the keyword) to the end: the predicate as Spark's
        /// errors cite it.
        op: Span,
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
    /// `TABLE(...)`, an argument of a table-valued function.
    TableArgument(Box<TableArgument>),
}

/// `TABLE(name)` or `TABLE(query)` with an optional partitioning, an
/// argument of a table-valued function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableArgument {
    /// The relation passed.
    pub source: TableArgumentSource,
    /// Written `TABLE name`, which Spark rejects: the name needs
    /// parentheses.
    pub unparenthesized: bool,
    /// `WITH SINGLE PARTITION`, `PARTITION BY ...`, ...
    pub partitioning: Option<TableArgumentPartitioning>,
    /// Source text.
    pub span: Span,
}

/// What a [`TableArgument`] passes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TableArgumentSource {
    /// A table or view.
    Table(DdlName),
    /// A query.
    Query(Box<Query>),
}

/// `WITH SINGLE PARTITION | (PARTITION | DISTRIBUTE) BY ... [(ORDER | SORT)
/// BY ...]` after a [`TableArgument`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableArgumentPartitioning {
    /// `WITH SINGLE PARTITION`.
    pub single: bool,
    /// `DISTRIBUTE BY` rather than `PARTITION BY`.
    pub distribute: bool,
    /// The partitioning expressions.
    pub partition_by: Vec<Expr>,
    /// The first expression of an unparenthesized list of several, which
    /// Spark parses and rejects.
    pub extra_partition: Option<Span>,
    /// `SORT BY` rather than `ORDER BY`.
    pub sort: bool,
    /// The ordering.
    pub order_by: Vec<SortItem>,
    /// The first item of an unparenthesized list of several, which Spark
    /// parses and rejects.
    pub extra_order: Option<Span>,
    /// Source text.
    pub span: Span,
}
