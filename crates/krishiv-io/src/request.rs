// SPDX-License-Identifier: Apache-2.0
//! [`ScanRequest`]: what a scan reads, built once per scan and immutable.
//!
//! The planner builds the **table-level** request (names and types of the
//! table schema). The framework maps it per file through the
//! [`SchemaMapper`](crate::SchemaMapper) into a **file-level** request (the
//! file's names and types), which is what a [`FileFormat`](crate::FileFormat)
//! sees: formats never resolve names, case rules, field ids or type widening.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, ListViewArray, MapArray,
    RecordBatch, RecordBatchOptions, StructArray,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};

use crate::error::{Result, internal};

/// Name of the row-index column a decoder appends when
/// [`MetadataCols::row_index`] is set (Spark's internal name for
/// `_metadata.row_index`): the row's position in the file, `Int64`, not null.
pub const ROW_INDEX_COLUMN: &str = "_tmp_metadata_row_index";

/// A column or nested struct field, from the top level down: `["a"]`,
/// `["a", "b"]`. Lists and maps are traversed implicitly (a path into a list
/// of structs names the struct's fields).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct ColumnPath(pub Vec<String>);

impl ColumnPath {
    /// A top-level column.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(vec![name.into()])
    }

    /// A nested field.
    #[must_use]
    pub fn nested<I, S>(parts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(parts.into_iter().map(Into::into).collect())
    }

    /// The top-level column name (empty for an empty path).
    #[must_use]
    pub fn root(&self) -> &str {
        self.0.first().map_or("", String::as_str)
    }
}

impl fmt::Display for ColumnPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.0.join(", "))
    }
}

/// A literal in a pushed predicate or a statistic. Values use the physical
/// representation of their Arrow type; equality compares floats by bits.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ScalarValue {
    /// `NULL`.
    Null,
    /// `Boolean`.
    Boolean(bool),
    /// `Int8`.
    Int8(i8),
    /// `Int16`.
    Int16(i16),
    /// `Int32`.
    Int32(i32),
    /// `Int64`.
    Int64(i64),
    /// `Float32`.
    Float32(f32),
    /// `Float64`.
    Float64(f64),
    /// `Decimal128(precision, scale)`: the value is `unscaled * 10^-scale`.
    Decimal128 {
        /// The unscaled value.
        unscaled: i128,
        /// Precision.
        precision: u8,
        /// Scale.
        scale: i8,
    },
    /// A string (`Utf8`, `LargeUtf8`, `Utf8View`).
    Utf8(String),
    /// Bytes (`Binary`, `LargeBinary`, `BinaryView`).
    Binary(Vec<u8>),
    /// `Date32`: days since the epoch.
    Date32(i32),
    /// `Timestamp(Microsecond, _)`: microseconds since the epoch.
    TimestampMicros(i64),
}

impl PartialEq for ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float32(a), Self::Float32(b)) => a.to_bits() == b.to_bits(),
            (Self::Float64(a), Self::Float64(b)) => a.to_bits() == b.to_bits(),
            (Self::Null, Self::Null) => true,
            (Self::Boolean(a), Self::Boolean(b)) => a == b,
            (Self::Int8(a), Self::Int8(b)) => a == b,
            (Self::Int16(a), Self::Int16(b)) => a == b,
            (Self::Int32(a), Self::Int32(b)) | (Self::Date32(a), Self::Date32(b)) => a == b,
            (Self::Int64(a), Self::Int64(b))
            | (Self::TimestampMicros(a), Self::TimestampMicros(b)) => a == b,
            (
                Self::Decimal128 {
                    unscaled: a,
                    precision: pa,
                    scale: sa,
                },
                Self::Decimal128 {
                    unscaled: b,
                    precision: pb,
                    scale: sb,
                },
            ) => a == b && pa == pb && sa == sb,
            (Self::Utf8(a), Self::Utf8(b)) => a == b,
            (Self::Binary(a), Self::Binary(b)) => a == b,
            _ => false,
        }
    }
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    /// `=`.
    Eq,
    /// `<>`.
    NotEq,
    /// `<`.
    Lt,
    /// `<=`.
    LtEq,
    /// `>`.
    Gt,
    /// `>=`.
    GtEq,
}

impl CmpOp {
    /// The operator of `NOT (a op b)` for non-null operands.
    #[must_use]
    pub fn negate(self) -> Self {
        match self {
            Self::Eq => Self::NotEq,
            Self::NotEq => Self::Eq,
            Self::Lt => Self::GtEq,
            Self::LtEq => Self::Gt,
            Self::Gt => Self::LtEq,
            Self::GtEq => Self::Lt,
        }
    }
}

/// The collation of a string comparison. Statistics and dictionaries are
/// ordered by bytes, so only [`Collation::Utf8Binary`] can prune.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum Collation {
    /// `UTF8_BINARY` (byte order; also every non-string comparison).
    #[default]
    Utf8Binary,
    /// Any other collation, by Spark name (`UTF8_LCASE`, `UNICODE_CI`, ...).
    Other(String),
}

/// A typed predicate tree pushed into a scan. Pruning keeps a **superset**
/// of the matching rows; the `Filter` above the scan is always kept.
/// Comparisons follow SQL: a `NULL` operand never matches.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PushdownExpr {
    /// `column op value`.
    Cmp {
        /// The column.
        column: ColumnPath,
        /// The operator.
        op: CmpOp,
        /// The literal (never [`ScalarValue::Null`]: that comparison is
        /// always unknown and is not pushed).
        value: ScalarValue,
        /// The comparison's collation.
        collation: Collation,
    },
    /// `column IN (values)`.
    In {
        /// The column.
        column: ColumnPath,
        /// The literals (non-null).
        values: Vec<ScalarValue>,
        /// The comparison's collation.
        collation: Collation,
    },
    /// `column IS NULL`.
    IsNull(ColumnPath),
    /// `column IS NOT NULL`.
    IsNotNull(ColumnPath),
    /// `startswith(column, prefix)` / `column LIKE 'prefix%'`.
    StartsWith {
        /// The column.
        column: ColumnPath,
        /// The prefix.
        prefix: String,
        /// The comparison's collation.
        collation: Collation,
    },
    /// Conjunction.
    And(Vec<PushdownExpr>),
    /// Disjunction.
    Or(Vec<PushdownExpr>),
    /// Negation.
    Not(Box<PushdownExpr>),
    /// A constant: `true` (every row matches) or `false` (none does). The
    /// schema mapper produces it, e.g. for a literal outside the file type.
    Constant(bool),
    /// `column`'s value, cast to the filter's key type, may be one of a
    /// runtime filter's keys (design P2-T163): statistics answer unknown;
    /// stages that see values (rows, dictionaries) evaluate it with
    /// [`KeyFilter::evaluate`](crate::KeyFilter::evaluate). Never in a plan.
    Membership {
        /// The column.
        column: ColumnPath,
        /// The filter's knowledge at one generation.
        filter: MembershipFilter,
    },
}

/// The [`KeyFilter`] of a [`PushdownExpr::Membership`], identified by
/// filter id and generation: two are equal when both are.
#[derive(Clone)]
pub struct MembershipFilter {
    /// The runtime filter.
    pub id: crate::FilterId,
    /// The generation `keys` were published as.
    pub generation: u64,
    /// The keys.
    pub keys: Arc<crate::KeyFilter>,
}

impl PartialEq for MembershipFilter {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.generation == other.generation
    }
}

impl fmt::Debug for MembershipFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RuntimeFilter {}@{}", self.id, self.generation)
    }
}

impl PushdownExpr {
    /// `column = value` with binary collation.
    #[must_use]
    pub fn eq(column: ColumnPath, value: ScalarValue) -> Self {
        Self::cmp(column, CmpOp::Eq, value)
    }

    /// `column op value` with binary collation.
    #[must_use]
    pub fn cmp(column: ColumnPath, op: CmpOp, value: ScalarValue) -> Self {
        Self::Cmp {
            column,
            op,
            value,
            collation: Collation::Utf8Binary,
        }
    }

    /// The expression with negations pushed to the leaves (negation normal
    /// form), so pruning never has to negate a three-valued answer. A
    /// negated `IN` becomes a conjunction of `<>`; a negated `StartsWith`
    /// or `Membership` stays a `Not` leaf (pruning answers unknown for it).
    #[must_use]
    pub fn to_nnf(&self) -> Self {
        self.nnf(false)
    }

    fn nnf(&self, negate: bool) -> Self {
        match (self, negate) {
            (Self::Not(inner), n) => inner.nnf(!n),
            (Self::And(xs), false) => Self::And(xs.iter().map(|x| x.nnf(false)).collect()),
            (Self::And(xs), true) => Self::Or(xs.iter().map(|x| x.nnf(true)).collect()),
            (Self::Or(xs), false) => Self::Or(xs.iter().map(|x| x.nnf(false)).collect()),
            (Self::Or(xs), true) => Self::And(xs.iter().map(|x| x.nnf(true)).collect()),
            (Self::Constant(b), n) => Self::Constant(*b != n),
            (Self::IsNull(c), true) => Self::IsNotNull(c.clone()),
            (Self::IsNotNull(c), true) => Self::IsNull(c.clone()),
            (
                Self::Cmp {
                    column,
                    op,
                    value,
                    collation,
                },
                true,
            ) => Self::Cmp {
                column: column.clone(),
                op: op.negate(),
                value: value.clone(),
                collation: collation.clone(),
            },
            (
                Self::In {
                    column,
                    values,
                    collation,
                },
                true,
            ) => Self::And(
                values
                    .iter()
                    .map(|v| Self::Cmp {
                        column: column.clone(),
                        op: CmpOp::NotEq,
                        value: v.clone(),
                        collation: collation.clone(),
                    })
                    .collect(),
            ),
            (leaf @ (Self::StartsWith { .. } | Self::Membership { .. }), true) => {
                Self::Not(Box::new(leaf.clone()))
            }
            (leaf, false) => leaf.clone(),
        }
    }

    /// Every column the expression reads.
    #[must_use]
    pub fn columns(&self) -> Vec<&ColumnPath> {
        let mut out = Vec::new();
        self.collect_columns(&mut out);
        out
    }

    fn collect_columns<'a>(&'a self, out: &mut Vec<&'a ColumnPath>) {
        match self {
            Self::Cmp { column, .. }
            | Self::In { column, .. }
            | Self::StartsWith { column, .. }
            | Self::Membership { column, .. }
            | Self::IsNull(column)
            | Self::IsNotNull(column) => out.push(column),
            Self::And(xs) | Self::Or(xs) => xs.iter().for_each(|x| x.collect_columns(out)),
            Self::Not(x) => x.collect_columns(out),
            Self::Constant(_) => {}
        }
    }
}

/// File-relative rows to read: sorted, disjoint, non-empty ranges of row
/// positions. Deletion vectors, positional deletes and index evaluators
/// produce it; formats that cannot skip rows still honour it by filtering.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RowSelection {
    ranges: Vec<Range<u64>>,
}

impl RowSelection {
    /// The selection of `ranges` (any order, may overlap or be empty).
    #[must_use]
    pub fn from_ranges(ranges: impl IntoIterator<Item = Range<u64>>) -> Self {
        let mut v: Vec<Range<u64>> = ranges.into_iter().filter(|r| r.start < r.end).collect();
        v.sort_by_key(|r| r.start);
        let mut out: Vec<Range<u64>> = Vec::with_capacity(v.len());
        for r in v {
            match out.last_mut() {
                Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                _ => out.push(r),
            }
        }
        Self { ranges: out }
    }

    /// Every row in `0..rows` except those listed (a deletion vector).
    #[must_use]
    pub fn excluding(rows: u64, deleted: impl IntoIterator<Item = u64>) -> Self {
        let mut del: Vec<u64> = deleted.into_iter().filter(|r| *r < rows).collect();
        del.sort_unstable();
        del.dedup();
        let mut ranges = Vec::with_capacity(del.len() + 1);
        let mut at = 0;
        for d in del {
            if d > at {
                ranges.push(at..d);
            }
            at = d + 1;
        }
        if at < rows {
            ranges.push(at..rows);
        }
        Self { ranges }
    }

    /// The selected ranges, sorted and disjoint.
    #[must_use]
    pub fn ranges(&self) -> &[Range<u64>] {
        &self.ranges
    }

    /// Whether row `row` is selected.
    #[must_use]
    pub fn contains(&self, row: u64) -> bool {
        let i = self.ranges.partition_point(|r| r.end <= row);
        self.ranges.get(i).is_some_and(|r| r.start <= row)
    }

    /// Selected rows within `window`, as ranges relative to `window.start`.
    #[must_use]
    pub fn within(&self, window: Range<u64>) -> Vec<Range<u64>> {
        let i = self.ranges.partition_point(|r| r.end <= window.start);
        self.ranges[i..]
            .iter()
            .take_while(|r| r.start < window.end)
            .map(|r| r.start.max(window.start) - window.start..r.end.min(window.end) - window.start)
            .collect()
    }

    /// Number of selected rows.
    #[must_use]
    pub fn row_count(&self) -> u64 {
        self.ranges.iter().map(|r| r.end - r.start).sum()
    }

    /// Rows selected by both.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while let (Some(a), Some(b)) = (self.ranges.get(i), other.ranges.get(j)) {
            let (s, e) = (a.start.max(b.start), a.end.min(b.end));
            if s < e {
                out.push(s..e);
            }
            if a.end < b.end {
                i += 1;
            } else {
                j += 1;
            }
        }
        Self { ranges: out }
    }
}

/// Whether a format may evaluate the predicate while decoding (late
/// materialisation). The decision rule (P2-T087) lives in the policy, never
/// in formats; formats without late materialisation ignore it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowFilterPolicy {
    /// Never evaluate rows in the format.
    Off,
    /// Always evaluate the predicate in the format.
    On,
    /// Let the policy decide per file or unit (the default).
    #[default]
    Auto,
}

/// The `_metadata` columns a scan returns. The constant ones are appended by
/// the framework; `row_index` needs the decoder (only it knows positions),
/// which appends [`ROW_INDEX_COLUMN`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per Spark `_metadata` field, as Spark exposes them"
)]
pub struct MetadataCols {
    /// `_metadata.file_path`.
    pub file_path: bool,
    /// `_metadata.file_name`.
    pub file_name: bool,
    /// `_metadata.file_size`.
    pub file_size: bool,
    /// `_metadata.file_block_start`.
    pub file_block_start: bool,
    /// `_metadata.file_block_length`.
    pub file_block_length: bool,
    /// `_metadata.file_modification_time`.
    pub file_modification_time: bool,
    /// `_metadata.row_index` (formats with [`Capabilities::row_index`](crate::Capabilities)).
    pub row_index: bool,
}

/// How table columns find their file columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMapping {
    /// By name, as Spark file sources: exact when `case_sensitive`
    /// (`spark.sql.caseSensitive`), else case-insensitive with an error on
    /// ambiguity.
    ByName {
        /// `spark.sql.caseSensitive`.
        case_sensitive: bool,
    },
    /// By field id ([`FIELD_ID_KEY`](crate::FIELD_ID_KEY) in both schemas'
    /// field metadata), as Iceberg: renames and reorders are free. A table
    /// field without an id is matched by name (Spark 4.2).
    FieldId,
    /// Delta column mapping: by `delta.columnMapping.physicalName` (name
    /// mode) or `delta.columnMapping.id` (id mode) of the table field.
    DeltaColumnMapping {
        /// Id mode (else name mode).
        by_id: bool,
    },
}

impl Default for FieldMapping {
    fn default() -> Self {
        Self::ByName {
            case_sensitive: false,
        }
    }
}

/// What a scan reads. Built once per scan by the planner, immutable.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScanRequest {
    /// The schema the projection and predicate refer to: the table schema
    /// (table level) or the file schema (file level, after mapping).
    pub table_schema: SchemaRef,
    /// Columns to read, nested paths allowed (nested-schema pruning). Empty
    /// reads no column: the batches carry row counts only.
    pub projection: Vec<ColumnPath>,
    /// Pushed predicate (pruning only: a superset of matching rows).
    pub predicate: Option<PushdownExpr>,
    /// Runtime filters (design P2-T163) on this level's columns, re-read
    /// at unit boundaries.
    pub dynamic: Vec<crate::ScanFilter>,
    /// File-relative rows to read.
    pub selection: Option<RowSelection>,
    /// Late-materialisation policy.
    pub row_filter: RowFilterPolicy,
    /// Most rows needed (a pushed `LIMIT` without a filter in between);
    /// a hint, never a reason to return fewer rows than exist.
    pub limit: Option<u64>,
    /// `_metadata` columns.
    pub metadata_cols: MetadataCols,
    /// Column resolution between table and file.
    pub mapping: FieldMapping,
}

impl ScanRequest {
    /// Reads `projection` of `table_schema`, with every other field at its
    /// default.
    #[must_use]
    pub fn new(table_schema: SchemaRef, projection: Vec<ColumnPath>) -> Self {
        Self {
            table_schema,
            projection,
            predicate: None,
            dynamic: Vec::new(),
            selection: None,
            row_filter: RowFilterPolicy::Auto,
            limit: None,
            metadata_cols: MetadataCols::default(),
            mapping: FieldMapping::default(),
        }
    }

    /// Reads every column of `table_schema`.
    #[must_use]
    pub fn all(table_schema: SchemaRef) -> Self {
        let projection = table_schema
            .fields()
            .iter()
            .map(|f| ColumnPath::new(f.name()))
            .collect();
        Self::new(table_schema, projection)
    }

    /// The schema of the decoded batches: the projection of `table_schema`
    /// (see [`project_schema`]), plus [`ROW_INDEX_COLUMN`] when requested.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a path does not exist in `table_schema`.
    pub fn output_schema(&self) -> Result<SchemaRef> {
        let projected = project_schema(&self.table_schema, &self.projection)?;
        if !self.metadata_cols.row_index {
            return Ok(projected);
        }
        let mut fields: Vec<Field> = projected
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        fields.push(Field::new(ROW_INDEX_COLUMN, DataType::Int64, false));
        Ok(Arc::new(Schema::new_with_metadata(
            fields,
            projected.metadata().clone(),
        )))
    }
}

/// The schema holding exactly the fields named by `paths`: top-level columns
/// in order of their first path, nested struct fields in schema order, a
/// path that stops at a struct keeps it whole. Lists and maps are
/// traversed implicitly.
///
/// # Errors
/// `INTERNAL_ERROR` if a path does not exist in `schema`.
pub fn project_schema(schema: &Schema, paths: &[ColumnPath]) -> Result<SchemaRef> {
    let mut order: Vec<usize> = Vec::new();
    for p in paths {
        let i = schema
            .index_of(p.root())
            .map_err(|_| internal(format!("projected column {p} is not in the schema")))?;
        if !order.contains(&i) {
            order.push(i);
        }
    }
    let mut fields = Vec::with_capacity(order.len());
    for i in order {
        let field = schema.field(i);
        let subs: Vec<&[String]> = paths
            .iter()
            .filter(|p| p.root() == field.name())
            .map(|p| &p.0[1..])
            .collect();
        fields.push(project_field(field, &subs)?);
    }
    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    )))
}

/// `batch` holding exactly the fields named by `paths`, as
/// [`project_schema`] shapes its schema: nested struct fields are pruned
/// (zero-copy) through structs, lists and maps.
///
/// # Errors
/// `INTERNAL_ERROR` if a path does not exist in the batch's schema.
pub fn project_batch(batch: &RecordBatch, paths: &[ColumnPath]) -> Result<RecordBatch> {
    let schema = project_schema(batch.schema_ref(), paths)?;
    let columns = schema
        .fields()
        .iter()
        .map(|f| {
            let i = batch
                .schema_ref()
                .index_of(f.name())
                .map_err(|e| internal(e.to_string()))?;
            prune_array(batch.column(i), f.data_type())
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| internal(format!("projected batch is invalid: {e}")))
}

/// `array` with its nested struct fields pruned to `want`'s.
fn prune_array(array: &ArrayRef, want: &DataType) -> Result<ArrayRef> {
    if array.data_type() == want {
        return Ok(Arc::clone(array));
    }
    let bad = |e: arrow_schema::ArrowError| internal(format!("cannot prune a nested column: {e}"));
    Ok(match (array.data_type(), want) {
        (DataType::Struct(_), DataType::Struct(fields)) => {
            let s = array.as_struct();
            let children = fields
                .iter()
                .map(|f| {
                    let c = s.column_by_name(f.name()).ok_or_else(|| {
                        internal(format!("no struct field {} to project", f.name()))
                    })?;
                    prune_array(c, f.data_type())
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(
                StructArray::try_new_with_length(
                    fields.clone(),
                    children,
                    s.nulls().cloned(),
                    s.len(),
                )
                .map_err(bad)?,
            )
        }
        (DataType::List(_), DataType::List(f)) => {
            let l = array.as_list::<i32>();
            let values = prune_array(l.values(), f.data_type())?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(bad)?,
            )
        }
        (DataType::LargeList(_), DataType::LargeList(f)) => {
            let l = array.as_list::<i64>();
            let values = prune_array(l.values(), f.data_type())?;
            Arc::new(
                LargeListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(bad)?,
            )
        }
        (DataType::Map(_, _), DataType::Map(f, sorted)) => {
            let m = array.as_map();
            let entries: ArrayRef = Arc::new(m.entries().clone());
            let entries = prune_array(&entries, f.data_type())?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(f),
                    m.offsets().clone(),
                    entries.as_struct().clone(),
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(bad)?,
            )
        }
        (DataType::ListView(_), DataType::ListView(f)) => {
            let l = array
                .as_any()
                .downcast_ref::<ListViewArray>()
                .ok_or_else(|| internal("a list view column is not a ListViewArray"))?;
            let values = prune_array(l.values(), f.data_type())?;
            Arc::new(
                ListViewArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    l.sizes().clone(),
                    values,
                    l.nulls().cloned(),
                )
                .map_err(bad)?,
            )
        }
        (DataType::FixedSizeList(_, n), DataType::FixedSizeList(f, _)) => {
            let l = array.as_fixed_size_list();
            let values = prune_array(l.values(), f.data_type())?;
            Arc::new(
                FixedSizeListArray::try_new(Arc::clone(f), *n, values, l.nulls().cloned())
                    .map_err(bad)?,
            )
        }
        (have, _) => {
            return Err(internal(format!("cannot prune a {have} column to {want}")));
        }
    })
}

/// `field` pruned to the sub-paths `subs` (an empty sub-path keeps it whole).
fn project_field(field: &Field, subs: &[&[String]]) -> Result<Field> {
    if subs.iter().any(|s| s.is_empty()) {
        return Ok(field.clone());
    }
    let data_type = project_type(field.data_type(), subs, field.name())?;
    Ok(field.clone().with_data_type(data_type))
}

fn project_type(data_type: &DataType, subs: &[&[String]], name: &str) -> Result<DataType> {
    match data_type {
        DataType::Struct(children) => {
            let mut kept = Vec::new();
            for child in children {
                let child_subs: Vec<&[String]> = subs
                    .iter()
                    .filter(|s| s.first().is_some_and(|n| n == child.name()))
                    .map(|s| &s[1..])
                    .collect();
                if !child_subs.is_empty() {
                    kept.push(project_field(child, &child_subs)?);
                }
            }
            let named: std::collections::BTreeSet<&String> =
                subs.iter().filter_map(|s| s.first()).collect();
            if kept.len() != named.len() {
                return Err(internal(format!(
                    "a nested projection of {name} names a field that does not exist"
                )));
            }
            Ok(DataType::Struct(Fields::from(kept)))
        }
        DataType::List(f) => Ok(DataType::List(Arc::new(project_field(f, subs)?))),
        DataType::LargeList(f) => Ok(DataType::LargeList(Arc::new(project_field(f, subs)?))),
        DataType::ListView(f) => Ok(DataType::ListView(Arc::new(project_field(f, subs)?))),
        DataType::FixedSizeList(f, n) => Ok(DataType::FixedSizeList(
            Arc::new(project_field(f, subs)?),
            *n,
        )),
        DataType::Map(entries, sorted) => {
            // Paths into a map name the value's fields.
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(internal(format!("map {name} has no key/value struct")));
            };
            let (Some(k), Some(v)) = (kv.first(), kv.get(1)) else {
                return Err(internal(format!("map {name} has no key/value struct")));
            };
            let v = project_field(v, subs)?;
            let kv = DataType::Struct(Fields::from(vec![k.as_ref().clone(), v]));
            Ok(DataType::Map(
                Arc::new(entries.as_ref().clone().with_data_type(kv)),
                *sorted,
            ))
        }
        _ => Err(internal(format!(
            "cannot project into non-nested column {name}"
        ))),
    }
}
