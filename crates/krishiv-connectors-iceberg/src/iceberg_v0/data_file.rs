// SPDX-License-Identifier: Apache-2.0
//! The manifest entry of a data file a write task wrote (P3-T027, PR-36a;
//! design note §6.1): where the file goes ([`DataLocations`], [`file_name`]),
//! its partition tuple ([`partition_tuple`]) and the iceberg `DataFile` built
//! from what the Parquet writer returned and the column metrics
//! (`plan::file_metrics`) ([`data_file`]).
//!
//! Locations are Java's `LocationProviders` (apache-iceberg 1.12.0), through
//! iceberg-rust's generators, which port them: data files under
//! `write.data.path`, else `write.folder-storage.path`, else
//! `<table location>/data`; with `write.object-storage.enabled` (Java's
//! `Boolean.parseBoolean`: `true` in any case) under a 20-bit Murmur3 hash of
//! the file name in three directories, after `write.object-storage.path` too,
//! partition directories kept unless
//! `write.object-storage.partitioned-paths` is false. Java refuses the two
//! deprecated properties (`write.object-storage.path`,
//! `write.folder-storage.path`) when it writes; Krishiv honours them, as
//! iceberg-rust does (a stated divergence). Partition directories are
//! `name=value` pairs, URL-encoded, with each transform's human string. A
//! table with a custom `write.location-provider.impl` (a Java class) is
//! refused. `write.object-storage.partitioned-paths` must be a boolean
//! (iceberg-rust's parse; Java reads anything but `true` as false).
//!
//! Every partition tuple is checked against its spec before iceberg-rust sees
//! it ([`check_partition`]): its path and summary code panics on a mismatch.
//! Column sizes are not recorded (`FileStatistics` has no chunk sizes yet;
//! Java writes them for every column with metrics).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, PartitionKey,
    PartitionSpec, PrimitiveType, SchemaRef, Struct, TableMetadata, Type,
};
use iceberg::writer::file_writer::location_generator::{
    DefaultLocationGenerator, LocationGenerator, ObjectStorageLocationGenerator,
};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::ScalarValue;

use super::scan::iceberg_type;
use super::storage::from_iceberg;
use crate::plan::bounds::encode;
use crate::plan::file_metrics::FileMetrics;

/// `write.object-storage.enabled` (Java's `TableProperties.OBJECT_STORE_ENABLED`).
pub(crate) const OBJECT_STORE_ENABLED: &str = "write.object-storage.enabled";

/// `write.location-provider.impl` (a Java `LocationProvider` class).
pub(crate) const LOCATION_PROVIDER_IMPL: &str = "write.location-provider.impl";

/// Where a write's data files go.
#[derive(Debug, Clone)]
pub(crate) enum DataLocations {
    /// `<data path>/<partition path>/<file>`.
    Default(DefaultLocationGenerator),
    /// Hash-prefixed paths (`write.object-storage.enabled`).
    ObjectStore(ObjectStorageLocationGenerator),
}

impl DataLocations {
    /// The data locations of the table `metadata` describes (module docs).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a custom location provider; a table
    /// property iceberg-rust cannot parse (`write.object-storage.partitioned-paths`
    /// that is not a boolean), as an internal error naming its message.
    pub(crate) fn for_table(metadata: &TableMetadata) -> Result<Self, KrishivError> {
        if let Some(class) = metadata.properties().get(LOCATION_PROVIDER_IMPL) {
            return Err(KrishivUnsupportedFeature {
                feature: format!(
                    "writing an Iceberg table whose {LOCATION_PROVIDER_IMPL} is {class}"
                ),
            }
            .into());
        }
        let object_store = metadata
            .properties()
            .get(OBJECT_STORE_ENABLED)
            .is_some_and(|v| v.eq_ignore_ascii_case("true"));
        Ok(if object_store {
            Self::ObjectStore(ObjectStorageLocationGenerator::new(metadata).map_err(from_iceberg)?)
        } else {
            Self::Default(DefaultLocationGenerator::new(metadata).map_err(from_iceberg)?)
        })
    }

    /// The location of the file `name` with the partition tuple `partition` of
    /// `spec` over `schema` (an unpartitioned spec: no partition directories).
    ///
    /// # Errors
    /// [`check_partition`]'s.
    pub(crate) fn location(
        &self,
        spec: &PartitionSpec,
        schema: &SchemaRef,
        partition: &Struct,
        name: &str,
    ) -> Result<String, KrishivError> {
        check_partition(spec, schema, partition)?;
        let key = PartitionKey::new(spec.clone(), Arc::clone(schema), partition.clone());
        Ok(match self {
            Self::Default(g) => g.generate_location(Some(&key), name),
            Self::ObjectStore(g) => g.generate_location(Some(&key), name),
        })
    }
}

/// Whether `partition` is a tuple of `spec` over `schema`: the spec binds to
/// the schema, one value per field, each null or a primitive of the field's
/// type. iceberg-rust's partition paths, summaries and manifest partition
/// summaries panic otherwise.
///
/// # Errors
/// An internal error naming the mismatch: tuples come from [`partition_tuple`]
/// over the same spec, so a mismatch is a bug.
pub(crate) fn check_partition(
    spec: &PartitionSpec,
    schema: &SchemaRef,
    partition: &Struct,
) -> Result<(), KrishivError> {
    let ty = spec.partition_type(schema).map_err(from_iceberg)?;
    let values: Vec<_> = partition.iter().collect();
    if ty.fields().len() != values.len() {
        return Err(internal(format!(
            "partition tuple of {} values for spec {} of {} fields",
            values.len(),
            spec.spec_id(),
            ty.fields().len()
        )));
    }
    for (field, value) in ty.fields().iter().zip(values) {
        let fits = match (field.field_type.as_ref(), value) {
            (Type::Primitive(_), None) => true,
            (Type::Primitive(p), Some(Literal::Primitive(v))) => p.compatible(v),
            _ => false,
        };
        if !fits {
            return Err(internal(format!(
                "partition value {value:?} of field {} is not a {}",
                field.name, field.field_type
            )));
        }
    }
    Ok(())
}

/// The name of the `n`th data file (from 1, as Java's counter) task `task` of
/// the write `write_id` wrote (design note §7.1, Java's `OutputFileFactory`
/// shape: `<partition>-<task>-<write id>-<n>.parquet`, the numbers zero-padded
/// to five digits as Java's).
pub(crate) fn file_name(partition: u32, task: u64, write_id: &str, n: u32) -> String {
    format!("{partition:05}-{task}-{write_id}-{n:05}.parquet")
}

/// The partition tuple of a file whose rows all have the transformed values
/// `values` (one per field of `spec`, in order; `None` a null), typed by the
/// spec's result types over `schema`.
///
/// # Errors
/// An internal error when `values` does not match the spec (count or types):
/// the write task computed them from this spec, so that is a bug.
pub(crate) fn partition_tuple(
    spec: &PartitionSpec,
    schema: &SchemaRef,
    values: &[Option<ScalarValue>],
) -> Result<Struct, KrishivError> {
    let ty = spec.partition_type(schema).map_err(from_iceberg)?;
    if ty.fields().len() != values.len() {
        return Err(internal(format!(
            "partition tuple of {} values for a spec of {} fields",
            values.len(),
            ty.fields().len()
        )));
    }
    let literals = ty
        .fields()
        .iter()
        .zip(values)
        .map(|(field, value)| {
            let Some(value) = value else {
                return Ok(None);
            };
            let Some(p) = field.field_type.as_primitive_type() else {
                return Err(internal(format!(
                    "partition field {} is not primitive",
                    field.name
                )));
            };
            Ok(Some(Literal::Primitive(datum(value, p)?.literal().clone())))
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    Ok(Struct::from_iter(literals))
}

/// What one written data file contributes to its manifest entry.
#[derive(Debug, Clone)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the Iceberg sink of PR-36a (later step) reads it")
)]
pub(crate) struct WrittenDataFile {
    /// Its full location.
    pub(crate) location: String,
    /// Its size in bytes.
    pub(crate) size: u64,
    /// Its rows.
    pub(crate) rows: u64,
    /// Its column metrics.
    pub(crate) metrics: FileMetrics,
    /// Its partition tuple ([`partition_tuple`]).
    pub(crate) partition: Struct,
    /// The partition spec it was written under.
    pub(crate) spec_id: i32,
    /// The row-group start offsets, when the writer reported them.
    pub(crate) split_offsets: Option<Vec<i64>>,
    /// The sort order its rows really follow; `None` claims none (written as
    /// order 0, unsorted, design note §6.1).
    pub(crate) sort_order_id: Option<i32>,
}

/// The iceberg `DataFile` of `file`, its bounds typed by the columns of `schema`.
///
/// # Errors
/// An internal error when a bound names a column `schema` does not have as a
/// primitive, or cannot be read as that column's type: the metrics came from
/// this schema, so that is a bug.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the Iceberg sink of PR-36a (later step) calls it")
)]
pub(crate) fn data_file(
    file: &WrittenDataFile,
    schema: &SchemaRef,
) -> Result<DataFile, KrishivError> {
    let bounds = |bounds: &BTreeMap<i32, Vec<u8>>| {
        bounds
            .iter()
            .map(|(id, bytes)| {
                let p = schema
                    .field_by_id(*id)
                    .and_then(|f| f.field_type.as_primitive_type().cloned())
                    .ok_or_else(|| internal(format!("bound of column {id}, not in the schema")))?;
                let d = Datum::try_from_bytes(bytes, p).map_err(from_iceberg)?;
                Ok((*id, d))
            })
            .collect::<Result<HashMap<_, _>, KrishivError>>()
    };
    let counts =
        |m: &BTreeMap<i32, u64>| m.iter().map(|(k, v)| (*k, *v)).collect::<HashMap<_, _>>();
    let mut b = DataFileBuilder::default();
    b.content(DataContentType::Data)
        .file_path(file.location.clone())
        .file_format(DataFileFormat::Parquet)
        .partition(file.partition.clone())
        .partition_spec_id(file.spec_id)
        .record_count(file.rows)
        .file_size_in_bytes(file.size)
        .value_counts(counts(&file.metrics.value_counts))
        .null_value_counts(counts(&file.metrics.null_counts))
        .nan_value_counts(counts(&file.metrics.nan_counts))
        .lower_bounds(bounds(&file.metrics.lower_bounds)?)
        .upper_bounds(bounds(&file.metrics.upper_bounds)?)
        .split_offsets(file.split_offsets.clone())
        .sort_order_id(file.sort_order_id.unwrap_or(0));
    b.build()
        .map_err(|e| internal(format!("data file {}: {e}", file.location)))
}

/// `value` as a datum of the primitive type `p`, through the single-value
/// serialisation both sides share.
fn datum(value: &ScalarValue, p: &PrimitiveType) -> Result<Datum, KrishivError> {
    let bytes = encode(value, &iceberg_type(p))
        .ok_or_else(|| internal(format!("partition value {value:?} is not a {p}")))?;
    Datum::try_from_bytes(&bytes, p.clone()).map_err(from_iceberg)
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}
