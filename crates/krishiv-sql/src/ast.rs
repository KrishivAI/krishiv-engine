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
            Self::CreateTable(t) => t.span,
            Self::CreateTableLike(t) => t.span,
            Self::DropTable(t) => t.span,
            Self::CreateNamespace(n) => n.span,
            Self::DropNamespace(n) => n.span,
            Self::Use(u) => u.span,
            Self::DeclareVariable(d) => d.span,
            Self::SetVariable(s) => s.span,
            Self::DropVariable(d) => d.span,
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

/// A table, namespace or `USE` name in a DDL statement, or the table of a
/// `TABLE(...)` argument (Spark's `identifierReference`): dotted parts or
/// `IDENTIFIER(expr)`, which Spark resolves when it analyzes the plan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DdlName {
    /// `a.b.c`.
    Parts(ObjectName),
    /// `IDENTIFIER(expr)`, kept as written.
    Dynamic(DynamicName),
}

impl DdlName {
    /// Source text.
    #[must_use]
    pub const fn span(&self) -> Span {
        match self {
            Self::Parts(n) => n.span,
            Self::Dynamic(d) => d.span,
        }
    }
}

/// `IDENTIFIER(expr)`: a name computed from an expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicName {
    /// The expression.
    pub expr: Box<Expr>,
    /// Source text, from `IDENTIFIER` to `)`.
    pub span: Span,
}

/// A column name in `CREATE TABLE`: `name` or `IDENTIFIER('name')` (the
/// latter only when `spark.sql.legacy.identifierClause` is off).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColumnName {
    /// A plain identifier.
    Ident(Ident),
    /// `IDENTIFIER('name')`, kept as written.
    Dynamic(DynamicName),
}

/// `CREATE [OR REPLACE] [TEMPORARY] [EXTERNAL] TABLE [IF NOT EXISTS] name
/// [(elements)] [USING provider] [clauses] [AS query]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateTable {
    /// `OR REPLACE`.
    pub or_replace: bool,
    /// `TEMPORARY` (`TEMP`).
    pub temporary: bool,
    /// `EXTERNAL`.
    pub external: bool,
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// The table name.
    pub name: DdlName,
    /// The column definitions and table constraints.
    pub elements: Vec<TableElement>,
    /// `USING provider`.
    pub using: Option<ObjectName>,
    /// `OPTIONS (...)`.
    pub options: Vec<TableOption>,
    /// `PARTITIONED BY (...)`.
    pub partitioned_by: Vec<PartitionField>,
    /// `CLUSTER BY (...)`.
    pub cluster_by: Vec<ObjectName>,
    /// `CLUSTERED BY ... SORTED BY ... INTO n BUCKETS`.
    pub bucket: Option<BucketSpec>,
    /// `SKEWED BY ... ON ...`.
    pub skew: Option<SkewSpec>,
    /// `ROW FORMAT ...`.
    pub row_format: Option<RowFormat>,
    /// `STORED AS ...` or `STORED BY ...`.
    pub file_format: Option<FileFormat>,
    /// `LOCATION '...'`.
    pub location: Option<String>,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// `DEFAULT COLLATION ...`.
    pub collation: Option<ObjectName>,
    /// `TBLPROPERTIES (...)`.
    pub tblproperties: Vec<TableOption>,
    /// The `AS query` (or bare query) of a CTAS.
    pub as_query: Option<Box<Query>>,
    /// The first clause written more than once, in the order Spark checks
    /// them (`TBLPROPERTIES`, `OPTIONS`, `PARTITIONED BY`, `STORED AS/BY`,
    /// `ROW FORMAT`, `COMMENT`, `DEFAULT COLLATION`, `CLUSTERED BY`,
    /// `CLUSTER BY`, `LOCATION`): the statement then fails with
    /// `DUPLICATE_CLAUSES`. Repeated clauses are merged above (lists
    /// appended, the later single value kept).
    pub duplicate_clause: Option<&'static str>,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// A `CREATE TABLE` element: a column definition or a table constraint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TableElement {
    /// A column definition.
    Column(Box<ColumnDef>),
    /// A table constraint.
    Constraint(TableConstraint),
}

/// `name type [NOT NULL] [DEFAULT ...] [GENERATED ...] [COMMENT '...']
/// [constraints]` in `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColumnDef {
    /// The column name.
    pub name: ColumnName,
    /// Its type.
    pub data_type: DataTypeSpec,
    /// `NOT NULL`.
    pub not_null: bool,
    /// `DEFAULT expr`.
    pub default: Option<Expr>,
    /// `GENERATED ...`.
    pub generated: Option<GeneratedColumn>,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// Inline constraints.
    pub constraints: Vec<ColumnConstraint>,
    /// The first descriptor written twice, in source order (`NOT NULL`,
    /// `DEFAULT`, `GENERATED ALWAYS AS`, `COMMENT` or `CONSTRAINT`, as
    /// Spark names them): the statement then fails with
    /// `CREATE_TABLE_COLUMN_DESCRIPTOR_DUPLICATE`. The later occurrence's
    /// value is the one kept above.
    pub duplicate_descriptor: Option<&'static str>,
    /// Source text.
    pub span: Span,
}

/// `GENERATED ALWAYS AS (expr)` or `GENERATED ... AS IDENTITY ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeneratedColumn {
    /// What it is.
    pub kind: GeneratedKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`GeneratedColumn`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GeneratedKind {
    /// `GENERATED ALWAYS AS (expr)`.
    Expr(Expr),
    /// `GENERATED ... AS IDENTITY ...`.
    Identity(IdentityColumn),
}

/// `GENERATED (ALWAYS | BY DEFAULT) AS IDENTITY [(START WITH n
/// INCREMENT BY m)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdentityColumn {
    /// `ALWAYS` or `BY DEFAULT`.
    pub how: IdentityHow,
    /// `START WITH n`.
    pub start: Option<i64>,
    /// `INCREMENT BY n`.
    pub step: Option<i64>,
    /// The first option written twice, in source order (`START` or
    /// `STEP`, as Spark names them): the statement then fails with
    /// `IDENTITY_COLUMNS_DUPLICATED_SEQUENCE_GENERATOR_OPTION`.
    pub duplicate_option: Option<&'static str>,
    /// Source text.
    pub span: Span,
}

/// How an identity column generates values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IdentityHow {
    /// `ALWAYS`.
    Always,
    /// `BY DEFAULT`.
    ByDefault,
}

/// `[CONSTRAINT name] (PRIMARY KEY | FOREIGN KEY ... | UNIQUE | CHECK ...)
/// [characteristics]` on a `CREATE TABLE` column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColumnConstraint {
    /// The `CONSTRAINT name`, if written.
    pub name: Option<Ident>,
    /// What it is.
    pub kind: ColumnConstraintKind,
    /// `ENFORCED`/`NOT ENFORCED`/`RELY`/`NORELY`.
    pub characteristics: Vec<ConstraintCharacteristic>,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`ColumnConstraint`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColumnConstraintKind {
    /// `PRIMARY KEY`.
    PrimaryKey,
    /// `UNIQUE`.
    Unique,
    /// `REFERENCES table [(columns)]`.
    References(ForeignKeyRef),
    /// `CHECK (expr)`.
    Check(Expr),
}

/// `[CONSTRAINT name] (PRIMARY KEY ... | FOREIGN KEY ... | UNIQUE ... |
/// CHECK ...) [characteristics]` on a `CREATE TABLE` table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableConstraint {
    /// The `CONSTRAINT name`, if written.
    pub name: Option<Ident>,
    /// What it is.
    pub kind: TableConstraintKind,
    /// `ENFORCED`/`NOT ENFORCED`/`RELY`/`NORELY`.
    pub characteristics: Vec<ConstraintCharacteristic>,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`TableConstraint`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TableConstraintKind {
    /// `PRIMARY KEY (columns)`.
    PrimaryKey(Vec<Ident>),
    /// `FOREIGN KEY (columns) REFERENCES table [(columns)]`.
    ForeignKey(ForeignKeyConstraint),
    /// `UNIQUE (columns)`.
    Unique(Vec<Ident>),
    /// `CHECK (expr)`.
    Check(Expr),
}

/// `FOREIGN KEY (columns) REFERENCES table [(ref columns)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForeignKeyConstraint {
    /// The foreign key columns.
    pub columns: Vec<Ident>,
    /// The referenced table.
    pub table: ObjectName,
    /// The referenced columns; empty for none.
    pub ref_columns: Vec<Ident>,
    /// Source text.
    pub span: Span,
}

/// `REFERENCES table [(columns)]` on a `CREATE TABLE` column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ForeignKeyRef {
    /// The referenced table.
    pub table: ObjectName,
    /// The referenced columns; empty for none.
    pub columns: Vec<Ident>,
    /// Source text.
    pub span: Span,
}

/// `ENFORCED`, `NOT ENFORCED`, `RELY` or `NORELY` after a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConstraintCharacteristic {
    /// `ENFORCED` (`true`) or `NOT ENFORCED` (`false`).
    Enforced(bool),
    /// `RELY` (`true`) or `NORELY` (`false`).
    Rely(bool),
}

/// A `PARTITIONED BY` element: a column or a transform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionField {
    /// What it is.
    pub kind: PartitionFieldKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`PartitionField`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PartitionFieldKind {
    /// `name type [NOT NULL] [COMMENT '...']`.
    Column(PartitionColumn),
    /// `name` or `name(args)`.
    Transform(PartitionTransform),
}

/// A `PARTITIONED BY` column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionColumn {
    /// The column name.
    pub name: ColumnName,
    /// Its type.
    pub data_type: DataTypeSpec,
    /// `NOT NULL`.
    pub not_null: bool,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// Source text.
    pub span: Span,
}

/// A `PARTITIONED BY` transform.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionTransform {
    /// The transform name.
    pub name: DdlName,
    /// Its arguments; empty for an identity transform.
    pub args: Vec<Expr>,
    /// Source text.
    pub span: Span,
}

/// `CLUSTERED BY (columns) [SORTED BY (ordered columns)] INTO n BUCKETS`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BucketSpec {
    /// The clustered columns.
    pub clustered_by: Vec<Ident>,
    /// The sorted columns.
    pub sorted_by: Vec<OrderedIdent>,
    /// The bucket count.
    pub buckets: u64,
    /// Source text.
    pub span: Span,
}

/// `name [ASC | DESC]` in `SORTED BY`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderedIdent {
    /// The column.
    pub name: Ident,
    /// `ASC` (`true`), `DESC` (`false`), or neither.
    pub asc: Option<bool>,
    /// Source text.
    pub span: Span,
}

/// `SKEWED BY (columns) ON (...) [STORED AS DIRECTORIES]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SkewSpec {
    /// The skewed columns.
    pub skewed_by: Vec<Ident>,
    /// The skewed value groups.
    pub values: Vec<Vec<Expr>>,
    /// `STORED AS DIRECTORIES`.
    pub stored_as_directories: bool,
    /// Source text.
    pub span: Span,
}

/// `ROW FORMAT SERDE ...` or `ROW FORMAT DELIMITED ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowFormat {
    /// What it is.
    pub kind: RowFormatKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`RowFormat`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowFormatKind {
    /// `ROW FORMAT SERDE 'class' [WITH SERDEPROPERTIES (...)]`.
    Serde(RowSerde),
    /// `ROW FORMAT DELIMITED [...]`.
    Delimited(RowDelimited),
}

/// `SERDE 'class' [WITH SERDEPROPERTIES (...)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowSerde {
    /// The SerDe class.
    pub class: String,
    /// `WITH SERDEPROPERTIES (...)`.
    pub properties: Vec<TableOption>,
    /// Source text.
    pub span: Span,
}

/// `DELIMITED [FIELDS TERMINATED BY ...] ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowDelimited {
    /// `FIELDS TERMINATED BY '...'`.
    pub fields_terminated_by: Option<String>,
    /// `ESCAPED BY '...'`.
    pub escaped_by: Option<String>,
    /// `COLLECTION ITEMS TERMINATED BY '...'`.
    pub collection_items_terminated_by: Option<String>,
    /// `MAP KEYS TERMINATED BY '...'`.
    pub map_keys_terminated_by: Option<String>,
    /// `LINES TERMINATED BY '...'`.
    pub lines_terminated_by: Option<String>,
    /// `NULL DEFINED AS '...'`.
    pub null_defined_as: Option<String>,
    /// Source text.
    pub span: Span,
}

/// `STORED AS name` or `STORED BY 'handler'`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileFormat {
    /// What it is.
    pub kind: FileFormatKind,
    /// Source text.
    pub span: Span,
}

/// The kinds of [`FileFormat`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FileFormatKind {
    /// `STORED AS name`.
    StoredAs(ObjectName),
    /// `STORED BY 'handler'`.
    StoredBy(String),
}

/// `CREATE TABLE [IF NOT EXISTS] target LIKE source [...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateTableLike {
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// The new table.
    pub name: DdlName,
    /// The liked table.
    pub source: DdlName,
    /// `USING provider`.
    pub using: Option<ObjectName>,
    /// `ROW FORMAT ...`.
    pub row_format: Option<RowFormat>,
    /// `STORED AS ...` or `STORED BY ...`.
    pub file_format: Option<FileFormat>,
    /// `LOCATION '...'`.
    pub location: Option<String>,
    /// `TBLPROPERTIES (...)`.
    pub tblproperties: Vec<TableOption>,
    /// The first clause written more than once, in the order Spark checks
    /// them (`PROVIDER` for `USING`, `STORED AS/BY`, `ROW FORMAT`,
    /// `LOCATION`, `TBLPROPERTIES`): the statement then fails with
    /// `DUPLICATE_CLAUSES`. Repeated clauses are merged above (lists
    /// appended, the later single value kept).
    pub duplicate_clause: Option<&'static str>,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// `DROP TABLE [IF EXISTS] name [PURGE]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropTable {
    /// `IF EXISTS`.
    pub if_exists: bool,
    /// The table name.
    pub name: DdlName,
    /// `PURGE`.
    pub purge: bool,
    /// Source text, from `DROP` to the end of the statement.
    pub span: Span,
}

/// `DECLARE [OR REPLACE] [VARIABLE | VAR] name [, ...] [type] [{DEFAULT
/// | =} expr]`: one or several session variables sharing a type and a
/// default (`VAR` for `VARIABLE`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeclareVariable {
    /// `OR REPLACE`.
    pub or_replace: bool,
    /// The variable names (each 1-3 parts: `v`, `session.v` or
    /// `system.session.v`).
    pub names: Vec<ObjectName>,
    /// The declared type, if written.
    pub data_type: Option<DataTypeSpec>,
    /// The default (`DEFAULT expr` or `= expr`), if written.
    pub default: Option<Expr>,
    /// Source text, from `DECLARE` to the end of the statement.
    pub span: Span,
}

/// `SET VAR[IABLE] v = expr [, ...]` or `SET VAR[IABLE] (v, ...) =
/// (query)` (`VAR` for `VARIABLE`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetVariable {
    /// What it is.
    pub kind: SetVariableKind,
    /// Source text, from `SET` to the end of the statement.
    pub span: Span,
}

/// The kinds of [`SetVariable`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SetVariableKind {
    /// `v = expr [, ...]`.
    Single {
        /// The assignments, in order.
        assignments: Vec<SetAssignment>,
    },
    /// `(v, ...) = (query)`.
    Row {
        /// The target variables, in order.
        targets: Vec<ObjectName>,
        /// The query whose single row is assigned.
        query: Box<Query>,
    },
}

/// `v = expr` in a [`SetVariableKind::Single`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SetAssignment {
    /// The target variable.
    pub name: ObjectName,
    /// The value (`DEFAULT` reads the variable's default).
    pub value: Expr,
    /// Source text, from the name to the end of the expression.
    pub span: Span,
}

/// `DROP TEMPORARY VAR[IABLE] [IF EXISTS] name` (`VAR` for `VARIABLE`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropVariable {
    /// `IF EXISTS`.
    pub if_exists: bool,
    /// The variable name (`v`, `session.v` or `system.session.v`).
    pub name: ObjectName,
    /// Source text, from `DROP` to the end of the statement.
    pub span: Span,
}

/// `DATABASE`, `SCHEMA` or `NAMESPACE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NamespaceKind {
    /// `DATABASE`.
    Database,
    /// `SCHEMA`.
    Schema,
    /// `NAMESPACE`.
    Namespace,
}

/// `CREATE DATABASE|SCHEMA|NAMESPACE [IF NOT EXISTS] name [...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateNamespace {
    /// Which keyword.
    pub kind: NamespaceKind,
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// The namespace name.
    pub name: DdlName,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// `LOCATION '...'`.
    pub location: Option<String>,
    /// `DEFAULT COLLATION ...`.
    pub collation: Option<ObjectName>,
    /// `WITH DBPROPERTIES|PROPERTIES (...)`.
    pub properties: Vec<TableOption>,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// `DROP DATABASE|SCHEMA|NAMESPACE [IF EXISTS] name [RESTRICT|CASCADE]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropNamespace {
    /// Which keyword.
    pub kind: NamespaceKind,
    /// `IF EXISTS`.
    pub if_exists: bool,
    /// The namespace name.
    pub name: DdlName,
    /// `RESTRICT` or `CASCADE`.
    pub behavior: Option<DropBehavior>,
    /// Source text, from `DROP` to the end of the statement.
    pub span: Span,
}

/// `RESTRICT` or `CASCADE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DropBehavior {
    /// `RESTRICT`.
    Restrict,
    /// `CASCADE`.
    Cascade,
}

/// `USE name` or `USE DATABASE|SCHEMA|NAMESPACE name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Use {
    /// The namespace keyword, if written.
    pub kind: Option<NamespaceKind>,
    /// The name.
    pub name: DdlName,
    /// Source text, from `USE` to the end of the statement.
    pub span: Span,
}

/// `CREATE [OR REPLACE] [GLOBAL] [TEMPORARY] VIEW [IF NOT EXISTS] name
/// [(columns)] [USING provider] [OPTIONS (...)] [COMMENT '...']
/// [TBLPROPERTIES (...)] [WITH SCHEMA ...] [AS query]`: every view shape
/// the legacy temporary [`CreateView`] node cannot express. A datasource
/// temporary view (`USING ...` with no `AS`) has no query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreatePermanentView {
    /// `OR REPLACE`.
    pub or_replace: bool,
    /// `GLOBAL` (a global temporary view).
    pub global: bool,
    /// `TEMPORARY` (`TEMP`).
    pub temporary: bool,
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// The view name.
    pub name: DdlName,
    /// The declared column aliases; empty for none.
    pub columns: Vec<PermanentViewColumn>,
    /// `DEFAULT COLLATION ...`.
    pub collation: Option<ObjectName>,
    /// `USING provider` (a datasource temporary view).
    pub using: Option<ObjectName>,
    /// `OPTIONS (...)`.
    pub options: Vec<TableOption>,
    /// `COMMENT '...'`.
    pub comment: Option<String>,
    /// `TBLPROPERTIES (...)`.
    pub tblproperties: Vec<TableOption>,
    /// `WITH SCHEMA ...`.
    pub schema_binding: Option<SchemaBinding>,
    /// `PARTITIONED ON (...)` was written (Spark rejects it after
    /// parsing).
    pub partitioned_on: bool,
    /// The first clause written twice, by Spark's check order (`COMMENT`,
    /// `DEFAULT COLLATION`, `WITH SCHEMA`, `PARTITIONED ON`,
    /// `TBLPROPERTIES`), if any.
    pub duplicate_clause: Option<&'static str>,
    /// The `AS query`; `None` for a datasource view (which has none).
    pub query: Option<Box<Query>>,
    /// Source text, from `CREATE` to the end of the statement.
    pub span: Span,
}

/// `name [COMMENT '...']`, or in a datasource (`USING`) view `name type
/// [NOT NULL] [COMMENT '...']`, in a [`CreatePermanentView`] column list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PermanentViewColumn {
    /// The column alias.
    pub name: ColumnName,
    /// Its type (datasource views only).
    pub data_type: Option<DataTypeSpec>,
    /// `NOT NULL` (datasource views only).
    pub not_null: bool,
    /// Its comment, if written.
    pub comment: Option<String>,
    /// Source text.
    pub span: Span,
}

/// `WITH SCHEMA ...` on a view: how the view tracks the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SchemaBinding {
    /// `WITH SCHEMA BINDING`.
    Binding,
    /// `WITH SCHEMA EVOLUTION`.
    Evolution,
    /// `WITH SCHEMA COMPENSATION`.
    Compensation,
    /// `WITH SCHEMA TYPE EVOLUTION`.
    TypeEvolution,
}

/// `DROP VIEW [IF EXISTS] name` with a qualified or computed name (the
/// legacy [`DropView`] node takes one plain part).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropPermanentView {
    /// `IF EXISTS`.
    pub if_exists: bool,
    /// The view name.
    pub name: DdlName,
    /// Source text, from `DROP` to the end of the statement.
    pub span: Span,
}

/// `ALTER TABLE name ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlterTable {
    /// The table name.
    pub table: DdlName,
    /// What to change.
    pub action: AlterTableAction,
    /// Source text, from `ALTER` to the end of the statement.
    pub span: Span,
}

/// What an [`AlterTable`] changes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlterTableAction {
    /// `ADD [COLUMN|COLUMNS] ...`.
    AddColumns(Vec<ColumnDef>),
    /// `DROP [COLUMN|COLUMNS] ...`.
    DropColumns(Vec<ColumnName>),
    /// `RENAME [COLUMN] from TO to`.
    RenameColumn {
        /// The column to rename.
        from: ColumnName,
        /// The new name.
        to: ColumnName,
        /// Source text.
        span: Span,
    },
    /// `RENAME TO name`.
    RenameTo {
        /// The new table name.
        name: DdlName,
        /// Source text.
        span: Span,
    },
    /// `CHANGE [COLUMN] name ...` (the legacy column change).
    ChangeColumn {
        /// The column to change.
        name: ColumnName,
        /// How to change it.
        action: ChangeColumnAction,
        /// Source text.
        span: Span,
    },
    /// `ALTER [COLUMN] name ...`.
    AlterColumn {
        /// The column to change.
        name: ColumnName,
        /// How to change it.
        action: AlterColumnAction,
        /// Source text.
        span: Span,
    },
    /// `SET TBLPROPERTIES (...)`.
    SetTblProperties(Vec<TableOption>),
    /// `UNSET TBLPROPERTIES [IF EXISTS] (...)`.
    UnsetTblProperties {
        /// `IF EXISTS`.
        if_exists: bool,
        /// The keys to unset (values, if written, are ignored).
        keys: Vec<TableOption>,
        /// Source text.
        span: Span,
    },
    /// `SET SERDEPROPERTIES (...)`.
    SetSerdeProperties(Vec<TableOption>),
    /// `PARTITION (...) SET SERDEPROPERTIES (...)`.
    SetPartitionSerdeProperties {
        /// The partition.
        partition: PartitionSpec,
        /// The properties.
        properties: Vec<TableOption>,
        /// Source text.
        span: Span,
    },
    /// `SET LOCATION '...'`.
    SetLocation(String),
    /// `ADD [IF NOT EXISTS] PARTITION ... [LOCATION ...] ...`.
    AddPartition {
        /// `IF NOT EXISTS`.
        if_not_exists: bool,
        /// The partitions to add, each with its `LOCATION`.
        parts: Vec<PartitionWithLocation>,
        /// Source text.
        span: Span,
    },
    /// `DROP [IF EXISTS] PARTITION ... [PURGE]`.
    DropPartition {
        /// `IF EXISTS`.
        if_exists: bool,
        /// The partitions to drop.
        specs: Vec<PartitionSpec>,
        /// `PURGE`.
        purge: bool,
        /// Source text.
        span: Span,
    },
    /// `PARTITION ... RENAME TO PARTITION ...`.
    RenamePartition {
        /// The partition to rename.
        from: PartitionSpec,
        /// The new specification.
        to: PartitionSpec,
        /// Source text.
        span: Span,
    },
    /// `PARTITION ... SET LOCATION '...'`.
    SetPartitionLocation {
        /// The partition.
        partition: PartitionSpec,
        /// The new location.
        location: String,
        /// Source text.
        span: Span,
    },
    /// `RECOVER PARTITIONS`.
    RecoverPartitions,
}

/// The change of a legacy `ALTER TABLE ... CHANGE [COLUMN]` action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChangeColumnAction {
    /// `TYPE type`.
    Type(DataTypeSpec),
    /// `COMMENT '...'`.
    Comment(String),
    /// `FIRST`.
    First,
    /// `AFTER column`.
    After(ColumnName),
    /// `SET NOT NULL`.
    SetNotNull,
    /// `DROP NOT NULL`.
    DropNotNull,
}

/// The change of an `ALTER TABLE ... ALTER [COLUMN]` action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlterColumnAction {
    /// `COMMENT '...'`.
    Comment(String),
    /// `SET NOT NULL`.
    SetNotNull,
    /// `DROP NOT NULL`.
    DropNotNull,
    /// `SET DEFAULT expr`.
    SetDefault(Expr),
    /// `DROP DEFAULT`.
    DropDefault,
}

/// `ALTER VIEW name ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlterView {
    /// The view name.
    pub name: DdlName,
    /// What to change.
    pub action: AlterViewAction,
    /// Source text, from `ALTER` to the end of the statement.
    pub span: Span,
}

/// What an [`AlterView`] changes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlterViewAction {
    /// `AS query [WITH SCHEMA ...]`.
    SetQuery {
        /// The new query.
        query: Box<Query>,
        /// `WITH SCHEMA ...`, if written.
        schema_binding: Option<SchemaBinding>,
        /// Source text.
        span: Span,
    },
    /// `WITH SCHEMA ...` without a query.
    SetSchemaBinding(SchemaBinding),
    /// `SET TBLPROPERTIES (...)`.
    SetTblProperties(Vec<TableOption>),
    /// `UNSET TBLPROPERTIES [IF EXISTS] (...)`.
    UnsetTblProperties {
        /// `IF EXISTS`.
        if_exists: bool,
        /// The keys to unset (values, if written, are ignored).
        keys: Vec<TableOption>,
        /// Source text.
        span: Span,
    },
    /// `RENAME TO name`.
    RenameTo {
        /// The new view name.
        name: DdlName,
        /// Source text.
        span: Span,
    },
}

/// `ALTER DATABASE|SCHEMA|NAMESPACE name ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlterNamespace {
    /// Which keyword.
    pub kind: NamespaceKind,
    /// The namespace name.
    pub name: DdlName,
    /// What to change.
    pub action: AlterNamespaceAction,
    /// Source text, from `ALTER` to the end of the statement.
    pub span: Span,
}

/// What an [`AlterNamespace`] changes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AlterNamespaceAction {
    /// `SET PROPERTIES (...)` (namespaces say `PROPERTIES`, not
    /// `TBLPROPERTIES`).
    SetProperties(Vec<TableOption>),
    /// `SET LOCATION '...'`.
    SetLocation(String),
    /// `DEFAULT COLLATION ...`.
    DefaultCollation(ObjectName),
}

/// `PARTITION (key [= value], ...)` in a DDL or command statement: a
/// value column holds a full expression, and a bare key reads its value
/// dynamically.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionSpec {
    /// The keys, in order.
    pub parts: Vec<PartitionSpecPart>,
    /// Source text, from `PARTITION` to `)`.
    pub span: Span,
}

/// One `key [= value]` of a [`PartitionSpec`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionSpecPart {
    /// The partition column.
    pub key: Ident,
    /// Its value; `None` for a bare key.
    pub value: Option<Expr>,
    /// Source text.
    pub span: Span,
}

/// `PARTITION ... [LOCATION '...']` in `ALTER TABLE ... ADD PARTITION`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionWithLocation {
    /// The partition.
    pub spec: PartitionSpec,
    /// Its `LOCATION`, if written.
    pub location: Option<String>,
    /// Source text.
    pub span: Span,
}

/// `INSERT INTO|OVERWRITE ...`, or `FROM ... INSERT INTO ... INSERT INTO
/// ...` (a multi-table insert).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Insert {
    /// The shared `FROM` clause of a multi-table insert; `None` for a
    /// single-table `INSERT`.
    pub source: Option<FromClause>,
    /// The rows to write; one for a single-table `INSERT`.
    pub clauses: Vec<InsertClause>,
    /// Source text, from `INSERT` (or `FROM`) to the end of the statement.
    pub span: Span,
}

/// One `INSERT INTO|OVERWRITE ... query` of an [`Insert`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InsertClause {
    /// `OVERWRITE` (`false` for `INTO`).
    pub overwrite: bool,
    /// The target table.
    pub name: DdlName,
    /// The target columns; empty for none.
    pub columns: Vec<ColumnName>,
    /// `PARTITION (...)`.
    pub partition: Option<PartitionSpec>,
    /// `IF NOT EXISTS` after the partition (Spark rejects it after
    /// parsing for `INTO`, and for `OVERWRITE` with dynamic partitions).
    pub if_not_exists: bool,
    /// `REPLACE WHERE ...`.
    pub replace_where: Option<Expr>,
    /// `BY NAME`.
    pub by_name: bool,
    /// The rows to write.
    pub query: Box<Query>,
    /// Source text of the clause head, from `INSERT` to the query.
    pub head_span: Span,
    /// Source text, from `INSERT` to the end of the clause.
    pub span: Span,
}

/// `DELETE FROM name [WHERE ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Delete {
    /// The table to delete from.
    pub table: DdlName,
    /// Its alias, if written.
    pub alias: Option<Ident>,
    /// `WHERE ...`.
    pub where_clause: Option<Expr>,
    /// Source text, from `DELETE` to the end of the statement.
    pub span: Span,
}

/// `v = expr` in an [`Update`] or [`Merge`] action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UpdateAssignment {
    /// The target column.
    pub target: ObjectName,
    /// Its new value.
    pub value: Expr,
    /// Source text.
    pub span: Span,
}

/// `UPDATE name [AS alias] SET ... [WHERE ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Update {
    /// The table to update.
    pub table: DdlName,
    /// Its alias, if written.
    pub alias: Option<Ident>,
    /// The `SET` assignments.
    pub assignments: Vec<UpdateAssignment>,
    /// `WHERE ...`.
    pub where_clause: Option<Expr>,
    /// Source text, from `UPDATE` to the end of the statement.
    pub span: Span,
}

/// `MERGE INTO target USING source ON ... WHEN ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Merge {
    /// The target table.
    pub target: DdlName,
    /// Its alias, if written.
    pub target_alias: Option<Ident>,
    /// The source.
    pub source: MergeSource,
    /// `ON ...`.
    pub on: Expr,
    /// The `WHEN ...` clauses, in order.
    pub clauses: Vec<MergeClause>,
    /// Source text, from `MERGE` to the end of the statement.
    pub span: Span,
}

/// The source of a [`Merge`]: a table or a subquery, each with an
/// optional alias.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MergeSource {
    /// A table.
    Table {
        /// The table name.
        name: DdlName,
        /// Its alias, if written.
        alias: Option<Ident>,
        /// Source text.
        span: Span,
    },
    /// A subquery.
    Query {
        /// The query.
        query: Box<Query>,
        /// Its alias, if written.
        alias: Option<Ident>,
        /// Source text.
        span: Span,
    },
}

/// One `WHEN ... THEN ...` of a [`Merge`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeClause {
    /// Which rows it fires on.
    pub matched: MergeMatched,
    /// `AND ...`, if written.
    pub condition: Option<Expr>,
    /// What it does.
    pub action: MergeAction,
    /// Source text, from `WHEN` to the end of the clause.
    pub span: Span,
}

/// Which rows a [`MergeClause`] fires on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MergeMatched {
    /// `WHEN MATCHED`.
    Matched,
    /// `WHEN NOT MATCHED`.
    NotMatched,
    /// `WHEN NOT MATCHED BY TARGET`.
    NotMatchedByTarget,
    /// `WHEN NOT MATCHED BY SOURCE`.
    NotMatchedBySource,
}

/// What a [`MergeClause`] does.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MergeAction {
    /// `UPDATE SET ...`.
    Update(Vec<UpdateAssignment>),
    /// `UPDATE SET *` (only after `WHEN MATCHED`).
    UpdateStar,
    /// `DELETE`.
    Delete,
    /// `INSERT ...`.
    Insert(MergeInsert),
}

/// The `INSERT` of a [`MergeClause`] not-matched action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MergeInsert {
    /// `INSERT *`.
    Star,
    /// `INSERT (columns) VALUES (...)`.
    Values {
        /// The target columns (dotted names), at least one.
        columns: Vec<ObjectName>,
        /// The row to insert.
        values: Vec<Expr>,
        /// Source text.
        span: Span,
    },
}

/// `TRUNCATE TABLE name [PARTITION ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Truncate {
    /// The table to truncate.
    pub table: DdlName,
    /// `PARTITION ...`.
    pub partition: Option<PartitionSpec>,
    /// Source text, from `TRUNCATE` to the end of the statement.
    pub span: Span,
}

/// `SHOW ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Show {
    /// What to show.
    pub kind: ShowKind,
    /// Source text, from `SHOW` to the end of the statement.
    pub span: Span,
}

/// What a [`Show`] shows.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShowKind {
    /// `SHOW TABLES [IN name] [LIKE pattern]`.
    Tables {
        /// `IN name`, if written.
        namespace: Option<DdlName>,
        /// The pattern, if written.
        pattern: Option<ShowPattern>,
        /// Source text.
        span: Span,
    },
    /// `SHOW TABLE EXTENDED [IN name] LIKE pattern [PARTITION ...]`.
    TableExtended {
        /// `IN name`, if written.
        namespace: Option<DdlName>,
        /// The pattern (`LIKE` itself is optional).
        pattern: ShowPattern,
        /// `PARTITION ...`, if written.
        partition: Option<PartitionSpec>,
        /// Source text.
        span: Span,
    },
    /// `SHOW VIEWS [{FROM | IN} name] [LIKE pattern]`.
    Views {
        /// The namespace, if written.
        namespace: Option<DdlName>,
        /// The pattern, if written.
        pattern: Option<ShowPattern>,
        /// Source text.
        span: Span,
    },
    /// `SHOW COLUMNS {FROM | IN} table [{FROM | IN} namespace]`.
    Columns {
        /// The table.
        table: DdlName,
        /// The namespace, if written.
        namespace: Option<DdlName>,
        /// Source text.
        span: Span,
    },
    /// `SHOW [scope.] [USER|SYSTEM|ALL] FUNCTIONS [{FROM | IN} namespace]
    /// [[LIKE] {pattern | name}]`.
    Functions {
        /// The scope, if written (never a computed name: Spark rejects
        /// `IDENTIFIER(...)` here).
        scope: Option<ObjectName>,
        /// `USER`, `SYSTEM` or `ALL`, if written.
        system: Option<FunctionScope>,
        /// The namespace, if written.
        namespace: Option<DdlName>,
        /// The pattern, if written as a string.
        pattern: Option<ShowPattern>,
        /// The legacy pattern, if written as a (dotted) name: its last
        /// part is the pattern, the others the namespace. Spark rejects
        /// it beside a `FROM`/`IN` namespace.
        legacy_pattern: Option<ObjectName>,
        /// Source text.
        span: Span,
    },
    /// `SHOW DATABASES|SCHEMAS|NAMESPACES [LIKE pattern]`.
    Namespaces {
        /// Which keyword.
        kind: NamespaceKind,
        /// The pattern, if written.
        pattern: Option<ShowPattern>,
        /// Source text.
        span: Span,
    },
    /// `SHOW CREATE TABLE name [AS SERDE]`.
    CreateTable {
        /// The table or view.
        name: DdlName,
        /// `AS SERDE`.
        serde: bool,
        /// Source text.
        span: Span,
    },
    /// `SHOW TBLPROPERTIES name [(key)]`.
    TblProperties {
        /// The table or view.
        name: DdlName,
        /// The key, if written (a string literal or an identifier).
        key: Option<String>,
        /// The key as written.
        key_span: Option<Span>,
        /// Source text.
        span: Span,
    },
    /// `SHOW PARTITIONS name [PARTITION ...]`.
    Partitions {
        /// The table.
        name: DdlName,
        /// `PARTITION ...`, if written.
        partition: Option<PartitionSpec>,
        /// Source text.
        span: Span,
    },
    /// `SHOW CURRENT NAMESPACE|SCHEMA|DATABASE`.
    CurrentNamespace {
        /// Which keyword.
        kind: NamespaceKind,
        /// Source text.
        span: Span,
    },
}

/// `[LIKE] 'pattern'` in a [`Show`]: the `LIKE` keyword is optional.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShowPattern {
    /// Whether `LIKE` was written.
    pub like: bool,
    /// The pattern.
    pub pattern: String,
    /// Source text.
    pub span: Span,
}

/// `USER`, `SYSTEM` or `ALL` in a [`ShowKind::Functions`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FunctionScope {
    /// `USER`.
    User,
    /// `SYSTEM`.
    System,
    /// `ALL`.
    All,
}

/// `DESCRIBE ...` (`DESC` for `DESCRIBE`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Describe {
    /// What to describe.
    pub kind: DescribeKind,
    /// Source text, from `DESCRIBE` to the end of the statement.
    pub span: Span,
}

/// What a [`Describe`] describes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DescribeKind {
    /// `DESCRIBE [TABLE] [EXTENDED|FORMATTED] table [PARTITION ...]
    /// [column] [AS JSON]`.
    Table {
        /// `EXTENDED` or `FORMATTED`, if written.
        mode: Option<DescribeMode>,
        /// The table.
        table: DdlName,
        /// `PARTITION ...`, if written.
        partition: Option<PartitionSpec>,
        /// The column's name parts (a nested field is dotted); empty for
        /// none.
        column: Vec<ColumnName>,
        /// `AS JSON`.
        as_json: bool,
        /// Source text.
        span: Span,
    },
    /// `DESCRIBE FUNCTION [EXTENDED] name`.
    Function {
        /// `EXTENDED`, if written.
        extended: bool,
        /// The function.
        name: DdlName,
        /// Source text.
        span: Span,
    },
    /// `DESCRIBE DATABASE|SCHEMA|NAMESPACE [EXTENDED] name`.
    Namespace {
        /// Which keyword.
        kind: NamespaceKind,
        /// `EXTENDED`, if written.
        extended: bool,
        /// The namespace.
        name: DdlName,
        /// Source text.
        span: Span,
    },
    /// `DESCRIBE [QUERY] query`.
    Query {
        /// The query.
        query: Box<Query>,
        /// Source text.
        span: Span,
    },
}

/// `EXTENDED` or `FORMATTED` in a [`DescribeKind::Table`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DescribeMode {
    /// `EXTENDED`.
    Extended,
    /// `FORMATTED`.
    Formatted,
}

/// `CACHE ...`, `UNCACHE ...` or `CLEAR CACHE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cache {
    /// What to do.
    pub kind: CacheKind,
    /// Source text, from `CACHE`, `UNCACHE` or `CLEAR` to the end of the
    /// statement.
    pub span: Span,
}

/// What a [`Cache`] does.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CacheKind {
    /// `CACHE [LAZY] TABLE name [OPTIONS (...)] [[AS] query]`.
    Cache {
        /// `LAZY`.
        lazy: bool,
        /// The table.
        name: DdlName,
        /// `OPTIONS (...)`.
        options: Vec<TableOption>,
        /// The query caching the table; `None` for a plain `CACHE`.
        query: Option<Box<Query>>,
        /// Source text.
        span: Span,
    },
    /// `UNCACHE TABLE [IF EXISTS] name`.
    Uncache {
        /// `IF EXISTS`.
        if_exists: bool,
        /// The table.
        name: DdlName,
        /// Source text.
        span: Span,
    },
    /// `CLEAR CACHE`.
    Clear,
}

/// `REFRESH ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Refresh {
    /// What to refresh.
    pub kind: RefreshKind,
    /// Source text, from `REFRESH` to the end of the statement.
    pub span: Span,
}

/// What a [`Refresh`] refreshes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RefreshKind {
    /// `REFRESH TABLE name`.
    Table {
        /// The table.
        name: DdlName,
        /// Source text.
        span: Span,
    },
    /// `REFRESH FUNCTION name`.
    Function {
        /// The function.
        name: DdlName,
        /// Source text.
        span: Span,
    },
    /// `REFRESH path`: one string literal, or the rest of the statement as
    /// an unquoted path.
    Path {
        /// The path.
        path: String,
        /// Source text.
        span: Span,
    },
}

/// `ANALYZE TABLE name [PARTITION ...] COMPUTE STATISTICS ...`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Analyze {
    /// The table.
    pub table: DdlName,
    /// `PARTITION ...`, if written.
    pub partition: Option<PartitionSpec>,
    /// `NOSCAN`, if written.
    pub noscan: bool,
    /// Which columns to compute statistics for.
    pub columns: AnalyzeColumns,
    /// Source text, from `ANALYZE` to the end of the statement.
    pub span: Span,
}

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

impl Expr {
    /// Calls `f` on each direct child expression, in order. Subquery plans
    /// are not expressions and are not visited; window specifications after
    /// `OVER` are visited (their partitioning, ordering and frame-offset
    /// expressions).
    pub fn for_each_child<'a>(&'a self, f: &mut impl FnMut(&'a Self)) {
        match &self.kind {
            ExprKind::Literal(_)
            | ExprKind::Parameter(_)
            | ExprKind::Keyword(_)
            | ExprKind::Column(_)
            | ExprKind::Wildcard { .. }
            | ExprKind::Exists(_)
            | ExprKind::Subquery(_) => {}
            ExprKind::Field(inner, _)
            | ExprKind::Collate(inner, _)
            | ExprKind::VariantGet(inner, _)
            | ExprKind::Unary(_, inner)
            | ExprKind::Not(inner)
            | ExprKind::Nested(inner)
            | ExprKind::IsNull { expr: inner, .. }
            | ExprKind::IsTruth { expr: inner, .. }
            | ExprKind::Cast { expr: inner, .. }
            | ExprKind::Lambda { body: inner, .. } => f(inner),
            ExprKind::Subscript(first, second) | ExprKind::Binary(first, _, second) => {
                f(first);
                f(second);
            }
            ExprKind::IsDistinctFrom { left, right, .. } => {
                f(left);
                f(right);
            }
            ExprKind::Between {
                expr, low, high, ..
            } => {
                f(expr);
                f(low);
                f(high);
            }
            ExprKind::InList { expr, list, .. } => {
                f(expr);
                list.iter().for_each(&mut *f);
            }
            ExprKind::InSubquery { expr, .. } => f(expr),
            ExprKind::Like { expr, pattern, .. } | ExprKind::RLike { expr, pattern, .. } => {
                f(expr);
                f(pattern);
            }
            ExprKind::LikeQuantified { expr, patterns, .. } => {
                f(expr);
                patterns.iter().for_each(&mut *f);
            }
            ExprKind::Case {
                operand,
                branches,
                else_result,
            } => {
                if let Some(inner) = operand {
                    f(inner);
                }
                for branch in branches {
                    f(&branch.condition);
                    f(&branch.result);
                }
                if let Some(inner) = else_result {
                    f(inner);
                }
            }
            ExprKind::Function(function) => {
                for arg in &function.args {
                    f(&arg.expr);
                }
                if let Some(inner) = function.filter.as_ref() {
                    f(inner);
                }
                for item in &function.within_group {
                    f(&item.expr);
                }
                if let Some(over) = &function.over {
                    for_each_window_expr(over, f);
                }
            }
            ExprKind::Row(items) | ExprKind::Struct(items) => {
                for item in items {
                    f(&item.expr);
                }
            }
            ExprKind::TableArgument(t) => {
                if let Some(p) = &t.partitioning {
                    for e in &p.partition_by {
                        f(e);
                    }
                    for s in &p.order_by {
                        f(&s.expr);
                    }
                }
            }
        }
    }
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
