// SPDX-License-Identifier: Apache-2.0
//! Krishiv's own view of an Iceberg table's schema and partition specs: the
//! primitive types (as the table format defines them, not as Spark shows them),
//! the primitive columns pruning can see, and the partition fields with their
//! transforms. Built from iceberg-rust's types in `iceberg_v0`; nothing here
//! names `iceberg`.

use std::collections::BTreeMap;

use krishiv_io::ColumnPath;

/// An Iceberg primitive type (table spec "Primitive Types").
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IcebergType {
    /// `boolean`.
    Boolean,
    /// `int` (32-bit).
    Int,
    /// `long` (64-bit).
    Long,
    /// `float` (32-bit IEEE 754).
    Float,
    /// `double` (64-bit IEEE 754).
    Double,
    /// `decimal(P, S)`.
    Decimal {
        /// Precision.
        precision: u8,
        /// Scale.
        scale: u8,
    },
    /// `date` (days since the epoch).
    Date,
    /// `time` (microseconds since midnight; Spark refuses it).
    Time,
    /// `timestamp` (microseconds, no zone; Spark's `TIMESTAMP_NTZ`).
    Timestamp,
    /// `timestamptz` (microseconds, UTC; Spark's `TIMESTAMP`).
    Timestamptz,
    /// `timestamp_ns` (Spark refuses it).
    TimestampNs,
    /// `timestamptz_ns` (Spark refuses it).
    TimestamptzNs,
    /// `string` (UTF-8).
    String,
    /// `uuid` (Spark shows it as a string).
    Uuid,
    /// `fixed(L)`.
    Fixed(u64),
    /// `binary`.
    Binary,
}

impl IcebergType {
    /// Whether values of the type are IEEE floats (bounds exclude NaN).
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float | Self::Double)
    }
}

/// A primitive column of the schema reachable through structs only (columns
/// inside lists and maps have element-level metrics, not row-level ones, and
/// are left out).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaField {
    /// The Iceberg field id.
    pub id: i32,
    /// The column's path, in the schema's names.
    pub path: ColumnPath,
    /// The field's type in this schema.
    pub ty: IcebergType,
}

/// A partition transform (table spec "Partition Transforms").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Transform {
    /// The source value.
    Identity,
    /// `bucket[N]`: Murmur3 (x86, 32-bit, seed 0) of the value, modulo `N`.
    Bucket(u32),
    /// `truncate[W]`.
    Truncate(u32),
    /// Years since 1970.
    Year,
    /// Months since 1970-01.
    Month,
    /// Days since the epoch (a `date`).
    Day,
    /// Hours since the epoch.
    Hour,
    /// Always null.
    Void,
    /// A transform this reader does not know: never used for pruning.
    Unknown,
}

/// A partition field of a spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionFieldInfo {
    /// The partition field id.
    pub field_id: i32,
    /// The source column's field id.
    pub source_id: i32,
    /// The partition field's name.
    pub name: String,
    /// The transform.
    pub transform: Transform,
    /// The source column's type, when a schema of the table still has it.
    pub source_type: Option<IcebergType>,
    /// The type of the partition value (`None`: unknown, never pruned on).
    pub result_type: Option<IcebergType>,
}

impl PartitionFieldInfo {
    /// The field with its result type derived from `transform` and `source_type`
    /// (table spec: `bucket`, `year`, `month` and `hour` give `int`, `day` a
    /// `date`, the others the source type).
    #[must_use]
    pub fn new(
        field_id: i32,
        source_id: i32,
        name: String,
        transform: Transform,
        source_type: Option<IcebergType>,
    ) -> Self {
        let result_type = match transform {
            Transform::Bucket(_) | Transform::Year | Transform::Month | Transform::Hour => {
                Some(IcebergType::Int)
            }
            Transform::Day => Some(IcebergType::Date),
            Transform::Identity | Transform::Truncate(_) | Transform::Void => source_type.clone(),
            Transform::Unknown => None,
        };
        Self {
            field_id,
            source_id,
            name,
            transform,
            source_type,
            result_type,
        }
    }
}

/// A partition spec: its id and fields in order (the order of a data file's
/// partition tuple).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PartitionSpecInfo {
    /// The spec id.
    pub spec_id: i32,
    /// The fields, in the spec's order.
    pub fields: Vec<PartitionFieldInfo>,
}

impl PartitionSpecInfo {
    /// Whether the spec has no field (or only `void` fields): its files are
    /// unpartitioned.
    #[must_use]
    pub fn is_unpartitioned(&self) -> bool {
        self.fields
            .iter()
            .all(|f| matches!(f.transform, Transform::Void))
    }
}

/// The partition specs of a table, by id.
pub type PartitionSpecs = BTreeMap<i32, PartitionSpecInfo>;
