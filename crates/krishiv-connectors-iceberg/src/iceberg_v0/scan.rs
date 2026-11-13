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
use krishiv_common::error::{
    InternalError, KrishivCatalogLimitExceeded, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_io::{ColumnPath, PushdownExpr};

use super::manifest_cache::{Cached, CachedEntry, ManifestCache, ManifestKey};
use super::storage::from_iceberg;
use super::types::spark_schema;
use crate::plan::bounds::min_be_bytes;
use crate::plan::deletes::DeleteIndex;
use crate::plan::entry::{
    ContentFile, ContentKind, EntryStatus, FileFormatKind, FileScanTask, ManifestContent,
    ManifestFileEntry, ManifestFilter, ParsedEntry, PartitionFieldSummary, PartitionValue,
    ScanPlan, inherit,
};
use crate::plan::metrics::WriterMetrics;
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
    let refs: HashMap<String, SnapshotReference> = match json.get_mut("refs") {
        Some(refs) => serde_json::from_value(refs.take())
            .map_err(|e| internal(format!("iceberg-rust: table metadata refs: {e}")))?,
        None => HashMap::new(),
    };
    let mut out: BTreeMap<String, RefInfo> = refs
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
        .collect();
    // iceberg-rust's v1 serialisation has no refs (review F5); Java's parser
    // makes `main` a branch at the current snapshot when the metadata has none.
    if let Some(current) = metadata.current_snapshot_id() {
        out.entry("main".to_owned()).or_insert(RefInfo {
            snapshot_id: current,
            kind: RefKind::Branch,
        });
    }
    Ok(out)
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

/// The catalog parse limit (`krishiv.catalog.maxMetadataBytes`) on one
/// metadata file of `size` bytes.
///
/// # Errors
/// `KRISHIV_CATALOG_LIMIT_EXCEEDED` naming the file.
pub(crate) fn check_size(
    what: &str,
    location: &str,
    size: u64,
    max: u64,
) -> Result<(), KrishivError> {
    if size <= max {
        return Ok(());
    }
    Err(KrishivCatalogLimitExceeded {
        source: format!("The Iceberg {what} {location}"),
        limit: format!("the metadata size limit of {max} bytes"),
        value: format!("{size} bytes"),
        config: "\"krishiv.catalog.maxMetadataBytes\" raised".to_owned(),
    }
    .into())
}

/// The bytes of the metadata file at `location`, refused before the read when
/// the store reports more than `max` bytes (and after it, if the store lied).
async fn read_capped(
    io: &FileIO,
    what: &str,
    location: &str,
    declared: Option<u64>,
    max: u64,
) -> Result<bytes::Bytes, KrishivError> {
    let input = io.new_input(location).map_err(from_iceberg)?;
    let size = match declared {
        Some(n) => n,
        None => input.metadata().await.map_err(from_iceberg)?.size,
    };
    check_size(what, location, size, max)?;
    let bytes = input.read().await.map_err(from_iceberg)?;
    check_size(
        what,
        location,
        u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        max,
    )?;
    Ok(bytes)
}

/// The `format-version` an Avro object container file's header names
/// (Iceberg's manifest-list and manifest writers set it), if any.
pub(crate) fn avro_format_version(bytes: &[u8]) -> Option<u8> {
    fn long(b: &[u8], at: &mut usize) -> Option<i64> {
        let mut v: u64 = 0;
        for shift in (0..64).step_by(7) {
            let byte = *b.get(*at)?;
            *at += 1;
            v |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some((v >> 1) as i64 ^ -((v & 1) as i64));
            }
        }
        None
    }
    fn blob<'b>(b: &'b [u8], at: &mut usize) -> Option<&'b [u8]> {
        let n = usize::try_from(long(b, at)?).ok()?;
        let out = b.get(*at..at.checked_add(n)?)?;
        *at += n;
        Some(out)
    }
    if !bytes.starts_with(b"Obj\x01") {
        return None;
    }
    let mut at = 4;
    loop {
        let count = long(bytes, &mut at)?;
        if count == 0 {
            return None;
        }
        if count < 0 {
            long(bytes, &mut at)?; // the block's byte size
        }
        for _ in 0..count.unsigned_abs() {
            let key = blob(bytes, &mut at)?;
            let value = blob(bytes, &mut at)?;
            if key == b"format-version" {
                return std::str::from_utf8(value).ok()?.parse().ok();
            }
        }
    }
}

/// The entries of the manifest list at `location`, parsed with the format
/// version its own header names (a snapshot written before the table's
/// upgrade has an older list; review F9), else `version`. A list larger than
/// `max_bytes` is refused before it is read.
pub(crate) async fn read_manifest_list(
    io: &FileIO,
    location: &str,
    version: u8,
    max_bytes: u64,
) -> Result<Vec<ManifestFileEntry>, KrishivError> {
    let bytes = read_capped(io, "manifest list", location, None, max_bytes).await?;
    let version = avro_format_version(&bytes).unwrap_or(version);
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

/// The live entries of one manifest, read, parsed and inherited, or from the
/// manifest cache of `env`.
async fn manifest_entries(
    io: &FileIO,
    manifest: &ManifestFileEntry,
    env: PlanEnv<'_>,
) -> Result<Arc<[CachedEntry]>, KrishivError> {
    let cache = env.cache.filter(|(_, limit)| *limit > 0);
    let key = cache.map(|_| ManifestKey::of(manifest));
    if let (Some((c, _)), Some(k)) = (cache, &key)
        && let Some(hit) = c.get(k)
    {
        check_size(
            "manifest",
            &manifest.path,
            hit.read_bytes,
            env.max_metadata_bytes,
        )?;
        return Ok(hit.entries);
    }
    if manifest.encrypted {
        return Err(KrishivUnsupportedFeature {
            feature: "encrypted Iceberg manifests".to_owned(),
        }
        .into());
    }
    let declared = u64::try_from(manifest.length).ok().filter(|n| *n > 0);
    let bytes = read_capped(
        io,
        "manifest",
        &manifest.path,
        declared,
        env.max_metadata_bytes,
    )
    .await?;
    let entries: Arc<[CachedEntry]> = inherit(manifest, parse_manifest(&bytes, manifest.spec_id)?)?
        .into_iter()
        .map(CachedEntry::from)
        .collect();
    if let (Some((c, limit)), Some(k)) = (cache, key) {
        let cached = Cached {
            entries: Arc::clone(&entries),
            read_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        };
        c.insert(k, cached, limit);
    }
    Ok(entries)
}

/// What a plan reads under: the catalog parse limit on each manifest list and
/// manifest (compressed bytes; the decoded entries are not charged to a memory
/// reservation) and the writers' metrics modes, with the decoded-manifest
/// cache and its byte limit (`None`: no cache).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanEnv<'a> {
    pub(crate) max_metadata_bytes: u64,
    pub(crate) metrics: &'a WriterMetrics,
    pub(crate) cache: Option<(&'a ManifestCache, u64)>,
}

/// The files of `snapshot` that may hold rows matching `predicate`.
///
/// # Errors
/// The I/O error of a read (its class kept); `KRISHIV_CATALOG_LIMIT_EXCEEDED` for
/// a manifest list or manifest over the size limit; an internal error for one
/// iceberg-rust cannot parse or that breaks the spec;
/// `KRISHIV_ICEBERG_INVALID_DELETES` for deletes that cannot be assigned
/// ([`DeleteIndex`]).
pub(crate) async fn plan(
    io: &FileIO,
    snapshot: &TableSnapshot,
    predicate: Option<&PushdownExpr>,
    filter: ManifestFilter,
    options: PruneOptions,
    env: PlanEnv<'_>,
) -> Result<ScanPlan, KrishivError> {
    let (mut out, index) = plan_files(io, snapshot, predicate, filter, options, env).await?;
    if !index.is_empty() {
        for t in &mut out.tasks {
            t.deletes = Arc::new(index.for_data_file(&t.file)?);
        }
    }
    Ok(out)
}

/// [`plan`] without assigning deletes to the tasks: the index assigns them
/// to the files a scan reads (the provider, after filter pruning).
///
/// # Errors
/// As [`plan`], except the errors of assigning deletes to one data file.
pub(crate) async fn plan_files(
    io: &FileIO,
    snapshot: &TableSnapshot,
    predicate: Option<&PushdownExpr>,
    filter: ManifestFilter,
    options: PruneOptions,
    env: PlanEnv<'_>,
) -> Result<(ScanPlan, DeleteIndex), KrishivError> {
    let ManifestFilter::All = filter;
    let Some(list) = &snapshot.manifest_list else {
        return Ok((ScanPlan::default(), DeleteIndex::default()));
    };
    let always = PushdownExpr::Constant(true);
    let pruner = Pruner::new(
        predicate.unwrap_or(&always),
        &snapshot.fields,
        &snapshot.partition_specs,
        options,
    )
    .with_metrics(env.metrics);
    // A snapshot from before an upgrade to v2 has no sequence number; its list
    // is v1 when the header does not say.
    let fallback = if snapshot.sequence_number == Some(0) {
        1
    } else {
        snapshot.format_version
    };
    let manifests = read_manifest_list(io, list, fallback, env.max_metadata_bytes).await?;
    let manifests_total = manifests.len();
    let kept: Vec<_> = manifests
        .into_iter()
        .filter(|m| m.may_have_live_files() && pruner.manifest_may_match(m))
        .collect();
    let manifests_pruned = manifests_total - kept.len();
    let per_manifest: Vec<Arc<[CachedEntry]>> = futures::stream::iter(kept.iter())
        .map(|m| manifest_entries(io, m, env))
        .buffered(MANIFEST_CONCURRENCY)
        .try_collect()
        .await?;
    let mut out = ScanPlan {
        snapshot_id: snapshot.snapshot_id,
        manifests_total,
        manifests_pruned,
        ..ScanPlan::default()
    };
    for entry in per_manifest.iter().flat_map(|m| m.iter()).cloned() {
        match entry {
            CachedEntry::Data(d) => {
                if pruner.file_may_match(&d.file) {
                    out.tasks.push(FileScanTask {
                        file: d,
                        residual: predicate.cloned(),
                        deletes: Arc::default(),
                    });
                } else {
                    out.files_pruned += 1;
                }
            }
            CachedEntry::Delete(d) => out.delete_files.push(d),
        }
    }
    let index = DeleteIndex::new(&out.delete_files, &snapshot.partition_specs)?;
    Ok((out, index))
}
