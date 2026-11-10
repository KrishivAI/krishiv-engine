// SPDX-License-Identifier: Apache-2.0
//! AST nodes of DDL and utility statements (names, tables, columns, partitions, options), the `Expr` helpers and the scripting statements.

use super::*;

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

/// `EXECUTE IMMEDIATE query [INTO v, ...] [USING [(]arg [AS alias], ...[)]]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecuteImmediate {
    /// The statement text: a string expression or a variable.
    pub query: Expr,
    /// The variables receiving the one result row.
    pub into: Vec<ObjectName>,
    /// The arguments binding the statement's parameter markers.
    pub using: Vec<UsingArgument>,
    /// Source text, from `EXECUTE` to the end of the statement.
    pub span: Span,
}

/// One `USING` argument of an [`ExecuteImmediate`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UsingArgument {
    /// The value.
    pub value: Expr,
    /// `AS alias`, naming the `:alias` marker it binds.
    pub alias: Option<Ident>,
    /// Source text.
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
    /// `WITH DBPROPERTIES|PROPERTIES (...)`; every list written, appended.
    pub properties: Vec<TableOption>,
    /// The first clause written twice, in Spark's check order (`COMMENT`,
    /// `DEFAULT COLLATION`, `LOCATION`, `WITH PROPERTIES`,
    /// `WITH DBPROPERTIES`), if any: `DUPLICATE_CLAUSES`.
    pub duplicate_clause: Option<&'static str>,
    /// Both `WITH PROPERTIES` and `WITH DBPROPERTIES` written.
    pub properties_and_dbproperties: bool,
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
    pub schema_binding: Option<SchemaBindingClause>,
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

/// A `WITH SCHEMA ...` clause as written: the mode and the clause's span
/// (Spark points `FEATURE_NOT_ENABLED` at it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaBindingClause {
    /// The mode.
    pub mode: SchemaBinding,
    /// Source text, from `WITH` to the end of the mode.
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
    /// `COMMENT ON TABLE name IS 'comment'` (`None` for `IS NULL`).
    SetComment(Option<String>),
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
        schema_binding: Option<SchemaBindingClause>,
        /// Source text.
        span: Span,
    },
    /// `WITH SCHEMA ...` without a query.
    SetSchemaBinding(SchemaBindingClause),
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
    /// `SET PROPERTIES (...)` or `SET DBPROPERTIES (...)` (namespaces say
    /// `PROPERTIES`, not `TBLPROPERTIES`).
    SetProperties(Vec<TableOption>),
    /// `SET LOCATION '...'`.
    SetLocation(String),
    /// `COMMENT ON SCHEMA name IS 'comment'` (`None` for `IS NULL`).
    SetComment(Option<String>),
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
    /// `WITH SCHEMA EVOLUTION`.
    pub with_schema_evolution: bool,
    /// The target table.
    pub name: DdlName,
    /// Its alias, if written (only with `REPLACE`).
    pub target_alias: Option<Ident>,
    /// `WITH (...)` target options.
    pub target_options: Vec<TableOption>,
    /// The target columns; empty for none.
    pub columns: Vec<ColumnName>,
    /// `PARTITION (...)`.
    pub partition: Option<PartitionSpec>,
    /// `IF NOT EXISTS` after the partition (Spark rejects it after
    /// parsing for `INTO`, and for `OVERWRITE` with dynamic partitions).
    pub if_not_exists: bool,
    /// `REPLACE WHERE ...`.
    pub replace_where: Option<Expr>,
    /// `REPLACE ON ...`.
    pub replace_on: Option<Expr>,
    /// `REPLACE USING (...)` columns.
    pub replace_using: Option<Vec<Ident>>,
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

impl Expr {
    /// Calls `f` on each direct child expression, in order. Subquery plans
    /// are not expressions and are not visited; window specifications after
    /// `OVER` are visited (their partitioning, ordering and frame-offset
    /// expressions).
    pub fn for_each_child<'a>(&'a self, f: &mut impl FnMut(&'a Self)) {
        match &self.kind {
            ExprKind::Literal(_)
            | ExprKind::Parameter(_)
            | ExprKind::NamedParameter(_)
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
