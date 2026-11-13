// SPDX-License-Identifier: Apache-2.0
//! The iceberg-rust side of scan planning: table metadata, the manifest list and
//! manifests turned into the Krishiv-owned values of [`crate::plan`]. Bytes come
//! through the registration's `FileIO` (so `krishiv-io`); parsing uses
//! iceberg-rust's public parsers (`ManifestList::parse_with_version`,
//! `Manifest::parse_avro`), and the planning itself (inheritance, pruning) is
//! Krishiv's.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFileFormat, FormatVersion, Literal, Manifest, ManifestContentType,
    ManifestList, ManifestStatus, NestedField, PrimitiveLiteral, PrimitiveType, Schema,
    SnapshotReference, SnapshotRetention, TableMetadata, Type,
};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::{ColumnPath, PushdownExpr};

use super::storage::from_iceberg;
use super::types::spark_schema;
use crate::plan::bounds::min_be_bytes;
use crate::plan::entry::{
    ContentFile, ContentKind, EntryStatus, FileFormatKind, FileScanTask, LiveEntry,
    ManifestContent, ManifestFileEntry, ManifestFilter, ParsedEntry, PartitionFieldSummary,
    PartitionValue, ScanPlan, inherit,
};
use crate::plan::prune::{PruneOptions, Pruner};
use crate::plan::snapshot::{
    MetadataView, RefInfo, RefKind, Resolved, SnapshotInfo, TableSnapshot, VersionSpec, resolve,
};
use crate::plan::types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};

/// Manifests read at once by one plan.
const MANIFEST_CONCURRENCY: usize = 8;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The Krishiv type of an Iceberg primitive type.
pub(crate) fn iceberg_type(p: &PrimitiveType) -> IcebergType {
    match p {
        PrimitiveType::Boolean => IcebergType::Boolean,
        PrimitiveType::Int => IcebergType::Int,
        PrimitiveType::Long => IcebergType::Long,
        PrimitiveType::Float => IcebergType::Float,
        PrimitiveType::Double => IcebergType::Double,
        PrimitiveType::Decimal { precision, scale } => IcebergType::Decimal {
            precision: u8::try_from(*precision).unwrap_or(u8::MAX),
            scale: u8::try_from(*scale).unwrap_or(u8::MAX),
        },
        PrimitiveType::Date => IcebergType::Date,
        PrimitiveType::Time => IcebergType::Time,
        PrimitiveType::Timestamp => IcebergType::Timestamp,
        PrimitiveType::Timestamptz => IcebergType::Timestamptz,
        PrimitiveType::TimestampNs => IcebergType::TimestampNs,
        PrimitiveType::TimestamptzNs => IcebergType::TimestamptzNs,
        PrimitiveType::String => IcebergType::String,
        PrimitiveType::Uuid => IcebergType::Uuid,
        PrimitiveType::Fixed(n) => IcebergType::Fixed(*n),
        PrimitiveType::Binary => IcebergType::Binary,
    }
}

/// The primitive columns of `schema` reachable through structs.
pub(crate) fn schema_fields(schema: &Schema) -> Vec<SchemaField> {
    fn walk(fields: &[Arc<NestedField>], prefix: &[String], out: &mut Vec<SchemaField>) {
        for f in fields {
            let mut path = prefix.to_vec();
            path.push(f.name.clone());
            match f.field_type.as_ref() {
                Type::Primitive(p) => out.push(SchemaField {
                    id: f.id,
                    path: ColumnPath(path),
                    ty: iceberg_type(p),
                }),
                Type::Struct(s) => walk(s.fields(), &path, out),
                Type::List(_) | Type::Map(_) | Type::Variant(_) => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(schema.as_struct().fields(), &[], &mut out);
    out
}

/// The table's partition specs; a source column's type comes from `read` or,
/// when a later schema dropped it, from the newest schema that has it.
pub(crate) fn partition_specs(metadata: &TableMetadata, read: &Schema) -> PartitionSpecs {
    let mut schemas: Vec<_> = metadata.schemas_iter().collect();
    schemas.sort_by_key(|s| std::cmp::Reverse(s.schema_id()));
    let source_type = |id: i32| {
        std::iter::once(read)
            .chain(schemas.iter().map(AsRef::as_ref))
            .find_map(|s| s.field_by_id(id))
            .and_then(|f| f.field_type.as_primitive_type().map(iceberg_type))
    };
    metadata
        .partition_specs_iter()
        .map(|spec| {
            let fields = spec
                .fields()
                .iter()
                .map(|f| {
                    let transform = match f.transform {
                        iceberg::spec::Transform::Identity => Transform::Identity,
                        iceberg::spec::Transform::Bucket(n) => Transform::Bucket(n),
                        iceberg::spec::Transform::Truncate(w) => Transform::Truncate(w),
                        iceberg::spec::Transform::Year => Transform::Year,
                        iceberg::spec::Transform::Month => Transform::Month,
                        iceberg::spec::Transform::Day => Transform::Day,
                        iceberg::spec::Transform::Hour => Transform::Hour,
                        iceberg::spec::Transform::Void => Transform::Void,
                        iceberg::spec::Transform::Unknown => Transform::Unknown,
                    };
                    PartitionFieldInfo::new(
                        f.field_id,
                        f.source_id,
                        f.name.clone(),
                        transform,
                        source_type(f.source_id),
                    )
                })
                .collect();
            (
                spec.spec_id(),
                PartitionSpecInfo {
                    spec_id: spec.spec_id(),
                    fields,
                },
            )
        })
        .collect()
}

/// The refs of `metadata`. iceberg-rust 0.11 has no accessor for the map (only
/// `snapshot_for_ref`, which panics on a dangling ref), so they are read from
/// the metadata's own serialisation.
fn refs(metadata: &TableMetadata) -> Result<BTreeMap<String, RefInfo>, KrishivError> {
    let mut json = serde_json::to_value(metadata)
        .map_err(|e| internal(format!("iceberg-rust: serialising table metadata: {e}")))?;
    let Some(refs) = json.get_mut("refs").map(serde_json::Value::take) else {
        return Ok(BTreeMap::new());
    };
    let refs: HashMap<String, SnapshotReference> = serde_json::from_value(refs)
        .map_err(|e| internal(format!("iceberg-rust: table metadata refs: {e}")))?;
    Ok(refs
        .into_iter()
        .map(|(name, r)| {
            let kind = match r.retention {
                SnapshotRetention::Branch { .. } => RefKind::Branch,
                SnapshotRetention::Tag { .. } => RefKind::Tag,
            };
            (
                name,
                RefInfo {
                    snapshot_id: r.snapshot_id,
                    kind,
                },
            )
        })
        .collect())
}

/// What resolution needs of `metadata` (refs only when `with_refs`).
pub(crate) fn metadata_view(
    metadata: &TableMetadata,
    with_refs: bool,
) -> Result<MetadataView, KrishivError> {
    Ok(MetadataView {
        snapshots: metadata
            .snapshots()
            .map(|s| {
                (
                    s.snapshot_id(),
                    SnapshotInfo {
                        snapshot_id: s.snapshot_id(),
                        sequence_number: s.sequence_number(),
                        timestamp_ms: s.timestamp_ms(),
                        schema_id: s.schema_id(),
                        manifest_list: s.manifest_list().to_owned(),
                    },
                )
            })
            .collect(),
        current_snapshot_id: metadata.current_snapshot_id(),
        current_schema_id: metadata.current_schema_id(),
        refs: if with_refs {
            refs(metadata)?
        } else {
            BTreeMap::new()
        },
        log: metadata
            .history()
            .iter()
            .map(|l| (l.timestamp_ms, l.snapshot_id))
            .collect(),
    })
}

/// The snapshot `version` of `metadata` reads.
///
/// # Errors
/// Iceberg's time-travel messages ([`resolve`]); an internal error for a schema
/// id the metadata lacks; `KRISHIV_UNSUPPORTED_FEATURE` for a column type Spark
/// refuses.
pub(crate) fn table_snapshot(
    metadata: &TableMetadata,
    version: &VersionSpec,
) -> Result<TableSnapshot, KrishivError> {
    let view = metadata_view(metadata, matches!(version, VersionSpec::Ref(_)))?;
    let Resolved {
        snapshot,
        schema_id,
    } = resolve(&view, version)?;
    let schema = metadata
        .schema_by_id(schema_id)
        .ok_or_else(|| internal(format!("the table metadata has no schema {schema_id}")))?;
    Ok(TableSnapshot {
        table_uuid: metadata.uuid().to_string(),
        format_version: metadata.format_version() as u8,
        snapshot_id: snapshot.as_ref().map(|s| s.snapshot_id),
        sequence_number: snapshot.as_ref().map(|s| s.sequence_number),
        timestamp_ms: snapshot.as_ref().map(|s| s.timestamp_ms),
        manifest_list: snapshot.map(|s| s.manifest_list),
        schema_id,
        schema: spark_schema(schema)?,
        fields: schema_fields(schema),
        partition_specs: partition_specs(metadata, schema),
    })
}

fn format_version(v: u8) -> Result<FormatVersion, KrishivError> {
    match v {
        1 => Ok(FormatVersion::V1),
        2 => Ok(FormatVersion::V2),
        3 => Ok(FormatVersion::V3),
        _ => Err(internal(format!("unknown Iceberg format version {v}"))),
    }
}

/// The entries of the manifest list at `location`.
pub(crate) async fn read_manifest_list(
    io: &FileIO,
    location: &str,
    version: u8,
) -> Result<Vec<ManifestFileEntry>, KrishivError> {
    let bytes = io
        .new_input(location)
        .map_err(from_iceberg)?
        .read()
        .await
        .map_err(from_iceberg)?;
    let list =
        ManifestList::parse_with_version(&bytes, format_version(version)?).map_err(from_iceberg)?;
    Ok(list
        .consume_entries()
        .into_iter()
        .map(|m| ManifestFileEntry {
            content: match m.content {
                ManifestContentType::Data => ManifestContent::Data,
                ManifestContentType::Deletes => ManifestContent::Deletes,
            },
            partitions: m.partitions.map(|ps| {
                ps.into_iter()
                    .map(|p| PartitionFieldSummary {
                        contains_null: p.contains_null,
                        contains_nan: p.contains_nan,
                        lower: p.lower_bound.map(iceberg::spec::ByteBuf::into_vec),
                        upper: p.upper_bound.map(iceberg::spec::ByteBuf::into_vec),
                    })
                    .collect()
            }),
            encrypted: m.key_metadata.is_some(),
            path: m.manifest_path,
            length: m.manifest_length,
            spec_id: m.partition_spec_id,
            sequence_number: m.sequence_number,
            min_sequence_number: m.min_sequence_number,
            added_snapshot_id: m.added_snapshot_id,
            added_files: m.added_files_count,
            existing_files: m.existing_files_count,
            deleted_files: m.deleted_files_count,
            first_row_id: m.first_row_id,
        })
        .collect())
}

/// A partition value in the spec's single-value serialisation.
fn primitive_bytes(p: &PrimitiveLiteral) -> Option<Vec<u8>> {
    Some(match p {
        PrimitiveLiteral::Boolean(b) => vec![u8::from(*b)],
        PrimitiveLiteral::Int(v) => v.to_le_bytes().to_vec(),
        PrimitiveLiteral::Long(v) => v.to_le_bytes().to_vec(),
        PrimitiveLiteral::Float(v) => v.0.to_le_bytes().to_vec(),
        PrimitiveLiteral::Double(v) => v.0.to_le_bytes().to_vec(),
        PrimitiveLiteral::String(s) => s.as_bytes().to_vec(),
        PrimitiveLiteral::Binary(b) => b.clone(),
        PrimitiveLiteral::Int128(v) => min_be_bytes(*v),
        PrimitiveLiteral::UInt128(v) => v.to_be_bytes().to_vec(),
        PrimitiveLiteral::AboveMax | PrimitiveLiteral::BelowMin => return None,
    })
}

/// The entries of a manifest (`bytes`), before inheritance.
///
/// # Errors
/// An internal error naming iceberg-rust's message for a manifest it cannot parse.
pub(crate) fn parse_manifest(bytes: &[u8], spec_id: i32) -> Result<Vec<ParsedEntry>, KrishivError> {
    let manifest = Manifest::parse_avro(bytes).map_err(from_iceberg)?;
    let (entries, _) = manifest.into_parts();
    Ok(entries
        .iter()
        .map(|e| {
            let f = e.data_file();
            let bounds = |m: &HashMap<i32, iceberg::spec::Datum>| {
                m.iter()
                    .filter_map(|(id, d)| d.to_bytes().ok().map(|b| (*id, b.into_vec())))
                    .collect()
            };
            let counts = |m: &HashMap<i32, u64>| m.iter().map(|(k, v)| (*k, *v)).collect();
            ParsedEntry {
                status: match e.status() {
                    ManifestStatus::Existing => EntryStatus::Existing,
                    ManifestStatus::Added => EntryStatus::Added,
                    ManifestStatus::Deleted => EntryStatus::Deleted,
                },
                snapshot_id: e.snapshot_id(),
                sequence_number: e.sequence_number(),
                file_sequence_number: e.file_sequence_number,
                file: ContentFile {
                    content: match f.content_type() {
                        DataContentType::Data => ContentKind::Data,
                        DataContentType::PositionDeletes => ContentKind::PositionDeletes,
                        DataContentType::EqualityDeletes => ContentKind::EqualityDeletes,
                    },
                    path: f.file_path().to_owned(),
                    format: match f.file_format() {
                        DataFileFormat::Parquet => FileFormatKind::Parquet,
                        DataFileFormat::Orc => FileFormatKind::Orc,
                        DataFileFormat::Avro => FileFormatKind::Avro,
                        DataFileFormat::Puffin => FileFormatKind::Puffin,
                    },
                    spec_id,
                    partition: f
                        .partition()
                        .iter()
                        .map(|v| match v {
                            None => PartitionValue::Null,
                            Some(Literal::Primitive(p)) => primitive_bytes(p)
                                .map_or(PartitionValue::Unreadable, PartitionValue::Value),
                            Some(_) => PartitionValue::Unreadable,
                        })
                        .collect(),
                    record_count: f.record_count(),
                    size: f.file_size_in_bytes(),
                    column_sizes: counts(f.column_sizes()),
                    value_counts: counts(f.value_counts()),
                    null_counts: counts(f.null_value_counts()),
                    nan_counts: counts(f.nan_value_counts()),
                    lower_bounds: bounds(f.lower_bounds()),
                    upper_bounds: bounds(f.upper_bounds()),
                    split_offsets: f.split_offsets().map(<[i64]>::to_vec),
                    sort_order_id: f.sort_order_id(),
                    first_row_id: f.first_row_id(),
                    equality_ids: f.equality_ids(),
                    referenced_data_file: f.referenced_data_file(),
                    content_offset: f.content_offset(),
                    content_size_in_bytes: f.content_size_in_bytes(),
                },
            }
        })
        .collect())
}

/// The live entries of one manifest, read, parsed and inherited.
async fn manifest_entries(
    io: &FileIO,
    manifest: &ManifestFileEntry,
) -> Result<Vec<LiveEntry>, KrishivError> {
    if manifest.encrypted {
        return Err(KrishivUnsupportedFeature {
            feature: "encrypted Iceberg manifests".to_owned(),
        }
        .into());
    }
    let bytes = io
        .new_input(&manifest.path)
        .map_err(from_iceberg)?
        .read()
        .await
        .map_err(from_iceberg)?;
    inherit(manifest, parse_manifest(&bytes, manifest.spec_id)?)
}

/// The files of `snapshot` that may hold rows matching `predicate`.
///
/// # Errors
/// The I/O error of a read (its class kept); an internal error for a manifest
/// list or manifest iceberg-rust cannot parse or that breaks the spec.
pub(crate) async fn plan(
    io: &FileIO,
    snapshot: &TableSnapshot,
    predicate: Option<&PushdownExpr>,
    filter: ManifestFilter,
    options: PruneOptions,
) -> Result<ScanPlan, KrishivError> {
    let ManifestFilter::All = filter;
    let Some(list) = &snapshot.manifest_list else {
        return Ok(ScanPlan::default());
    };
    let always = PushdownExpr::Constant(true);
    let pruner = Pruner::new(
        predicate.unwrap_or(&always),
        &snapshot.fields,
        &snapshot.partition_specs,
        options,
    );
    let manifests = read_manifest_list(io, list, snapshot.format_version).await?;
    let manifests_total = manifests.len();
    let kept: Vec<_> = manifests
        .into_iter()
        .filter(|m| m.may_have_live_files() && pruner.manifest_may_match(m))
        .collect();
    let manifests_pruned = manifests_total - kept.len();
    let per_manifest: Vec<Vec<LiveEntry>> = futures::stream::iter(kept.iter())
        .map(|m| manifest_entries(io, m))
        .buffered(MANIFEST_CONCURRENCY)
        .try_collect()
        .await?;
    let mut out = ScanPlan {
        snapshot_id: snapshot.snapshot_id,
        manifests_total,
        manifests_pruned,
        ..ScanPlan::default()
    };
    for entry in per_manifest.into_iter().flatten() {
        match entry {
            LiveEntry::Data(d) => {
                if pruner.file_may_match(&d.file) {
                    out.tasks.push(FileScanTask {
                        file: Arc::new(d),
                        residual: predicate.cloned(),
                    });
                } else {
                    out.files_pruned += 1;
                }
            }
            LiveEntry::Delete(d) => out.delete_files.push(Arc::new(d)),
        }
    }
    Ok(out)
}
