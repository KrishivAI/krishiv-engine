// SPDX-License-Identifier: Apache-2.0
//! [`DeltaTable`], the provider of one Delta table at one version for one query (design note
//! P3-T021 section 3.4): the kernel reads the log, Krishiv reads the data.
//!
//! [`DeltaTable::open`] builds the kernel snapshot on the log pool, checks Krishiv's reader
//! allow-list ([`crate::features`]) before anything else is read, maps the logical schema to
//! the engine's Arrow forms (column-mapping metadata kept on every field) and lists the live
//! files once (each a [`ScanUnit`]): that listing gives the optimizer its row count
//! ([`TableProvider::statistics`]) and the scan its units. A scan with a pushed filter lists
//! again through the kernel with the translated predicate ([`predicate`]), which prunes by
//! partition values and statistics.
//!
//! # Snapshot cache
//! ADR 0037's snapshot cache is not here yet (PR-08b part 2): `open` builds a snapshot per
//! call. Its key must include the store's credential scope (the `StoreId` and the resolved
//! credential identity) besides the table root and version, so a snapshot built under one
//! principal's `IoContext` is never served under another's; table authorization runs before
//! the lookup.

pub mod predicate;
pub mod scan;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, FieldRef, Schema, SchemaRef};
use delta_kernel::expressions::Scalar;
use delta_kernel::scan::state::DvInfo;
use delta_kernel::schema::{DataType, StructField};
use delta_kernel::table_features::ColumnMappingMode;
use delta_kernel::{KernelError, Snapshot, SnapshotRef};
use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::ScanPushdown;
use krishiv_exec::planner::TableProvider;
use krishiv_io::{FieldMapping, ScanOptions};
use krishiv_optimizer::TableStats;

use crate::engine::DeltaEngine;
use crate::engine::schema::{scalar_to_array, to_arrow_field};
use crate::error::from_kernel;
use crate::features::{ReadProtocol, check_read, snapshot_error};

pub use self::scan::{DeltaScan, DeltaScanSource};

/// One live data file of the snapshot: one scan unit.
#[derive(Debug, Clone)]
pub struct ScanUnit {
    /// The path the log names, relative to the table root (URL-encoded) or absolute.
    pub path: String,
    /// The file's size in bytes.
    pub size: u64,
    /// The file's `numRecords`, when its statistics have it (rows before its deletion vector).
    pub records: Option<u64>,
    /// The `add` action's modification time, ms since the epoch.
    pub modification_time: i64,
    /// The raw partition values, by physical column name.
    pub partition_values: HashMap<String, String>,
    /// Its deletion vector, if any.
    pub dv: DvInfo,
}

/// A partition column: the table field and the name its values are stored under.
#[derive(Debug, Clone)]
pub(crate) struct PartitionColumn {
    /// The table field (the engine's Arrow form).
    pub(crate) field: FieldRef,
    /// The kernel type values are parsed as.
    pub(crate) kernel_type: DataType,
    /// The physical name (the key of `partitionValues`).
    pub(crate) physical: String,
}

/// What a table and its scans share.
#[derive(Debug)]
pub(crate) struct Inner {
    pub(crate) engine: DeltaEngine,
    pub(crate) name: String,
    pub(crate) snapshot: SnapshotRef,
    pub(crate) schema: SchemaRef,
    pub(crate) partitions: Vec<PartitionColumn>,
    pub(crate) mapping: FieldMapping,
    pub(crate) units: Arc<[ScanUnit]>,
    pub(crate) scan_options: ScanOptions,
}

/// A Delta table at one version for one query (see the module docs).
#[derive(Debug, Clone)]
pub struct DeltaTable {
    inner: Arc<Inner>,
}

impl DeltaTable {
    /// Opens the table at `engine`'s root at `version` (the latest when `None`) and lists its
    /// live files. Scan settings follow `config`.
    ///
    /// # Errors
    /// The allow-list's refusals ([`crate::features`]), before any file of the table's data is
    /// read; `DELTA_INVALID_PROTOCOL_VERSION`; the kernel's errors otherwise
    /// ([`crate::error`]).
    pub async fn open(
        engine: DeltaEngine,
        version: Option<u64>,
        config: &Config,
    ) -> Result<Self, KrishivError> {
        let name = engine.table_root().as_str().to_owned();
        let root = name.clone();
        let (snapshot, units) = engine
            .delta_blocking(move |e| -> Result<_, KrishivError> {
                let mut builder =
                    Snapshot::builder_for(&root).with_cancellation_token(e.kernel_cancellation());
                if let Some(v) = version {
                    builder = builder.at_version(v);
                }
                let snapshot = builder
                    .build(e)
                    .map_err(|err| snapshot_error(&err, &root))?;
                // The allow-list before the file listing: a refused table reads no more.
                check_read(&root, &ReadProtocol::of(&snapshot))?;
                let units =
                    list_units(e, &snapshot, None).map_err(|err| from_kernel(&err, &root))?;
                Ok((snapshot, units))
            })
            .await??;
        let tc = snapshot.table_configuration();
        let mode = tc.column_mapping_mode();
        let logical = snapshot.schema();
        let fields = logical
            .fields()
            .map(|f| engine_field(f, &name))
            .collect::<Result<Vec<_>, _>>()?;
        let schema: SchemaRef = Arc::new(Schema::new(fields));
        let partitions = tc
            .metadata()
            .partition_columns()
            .iter()
            .map(|p| {
                let (i, kf) = logical
                    .fields()
                    .enumerate()
                    .find(|(_, f)| f.name() == p)
                    .ok_or_else(|| {
                        internal(format!("partition column {p} is not in the schema"))
                    })?;
                Ok(PartitionColumn {
                    field: Arc::clone(&schema.fields()[i]),
                    kernel_type: kf.data_type().clone(),
                    physical: kf.physical_name(mode).to_owned(),
                })
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        let mapping = match mode {
            ColumnMappingMode::None => FieldMapping::ByName {
                case_sensitive: true,
            },
            ColumnMappingMode::Id => FieldMapping::DeltaColumnMapping { by_id: true },
            ColumnMappingMode::Name => FieldMapping::DeltaColumnMapping { by_id: false },
        };
        Ok(Self {
            inner: Arc::new(Inner {
                engine,
                name,
                snapshot,
                schema,
                partitions,
                mapping,
                units: units.into(),
                scan_options: ScanOptions::from_config(config),
            }),
        })
    }

    /// The snapshot's version.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.inner.snapshot.version()
    }

    /// The table's columns in the engine's Arrow forms.
    #[must_use]
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.inner.schema)
    }

    /// The kernel snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &SnapshotRef {
        &self.inner.snapshot
    }

    /// The live files, in the kernel's order.
    #[must_use]
    pub fn units(&self) -> &[ScanUnit] {
        &self.inner.units
    }

    /// The scan, typed.
    #[must_use]
    pub fn delta_scan(&self) -> DeltaScan {
        DeltaScan::new(Arc::clone(&self.inner))
    }
}

impl TableProvider for DeltaTable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        Ok(Arc::new(self.delta_scan()))
    }

    /// The live rows: every file's `numRecords` less its deletion vector's cardinality, when
    /// every file has the statistic. Column statistics are not exposed in P3 (T024).
    fn statistics(&self) -> Option<TableStats> {
        let mut rows = 0_u64;
        for u in self.inner.units.iter() {
            let deleted = u.dv.cardinality().ok()?.unwrap_or(0);
            rows = rows.saturating_add(u.records?.saturating_sub(deleted));
        }
        Some(TableStats {
            rows,
            ..TableStats::default()
        })
    }

    /// One unit per live file.
    fn scan_units(&self) -> Option<u64> {
        u64::try_from(self.inner.units.len()).ok()
    }
}

/// The live files of `snapshot`, pruned by `predicate`. Drains the kernel's lazy iterator, so
/// it runs inside a `delta_blocking` call.
pub(crate) fn list_units(
    engine: &DeltaEngine,
    snapshot: &SnapshotRef,
    predicate: Option<delta_kernel::expressions::PredicateRef>,
) -> Result<Vec<ScanUnit>, KernelError> {
    let scan = Arc::clone(snapshot)
        .scan_builder()
        .with_predicate(predicate)
        .with_cancellation_token(engine.kernel_cancellation())
        .build()?;
    let mut files: Vec<delta_kernel::scan::state::ScanFile> = Vec::new();
    for metadata in scan.scan_metadata(engine)? {
        files = metadata?.visit_scan_files(files, Vec::push)?;
    }
    files
        .into_iter()
        .map(|f| {
            let size = u64::try_from(f.size).map_err(|_| {
                KernelError::generic(format!(
                    "the Delta log gives the data file {} a negative size ({})",
                    f.path, f.size
                ))
            })?;
            Ok(ScanUnit {
                path: f.path,
                size,
                records: f.stats.map(|s| s.num_records),
                modification_time: f.modification_time,
                partition_values: f.partition_values,
                dv: f.dv_info,
            })
        })
        .collect()
}

/// The value of a partition column stored as `raw`, as the kernel's scan parses it: an
/// absent value is NULL, an empty one is the empty string or bytes (NULL for other types).
pub(crate) fn partition_scalar(raw: Option<&String>, t: &DataType) -> Result<Scalar, KernelError> {
    match (raw, t.as_primitive_opt()) {
        (Some(v), Some(p)) if v.is_empty() => Ok(match p {
            delta_kernel::schema::PrimitiveType::String => Scalar::String(String::new()),
            delta_kernel::schema::PrimitiveType::Binary => Scalar::Binary(Vec::new()),
            _ => Scalar::Null(t.clone()),
        }),
        (Some(v), Some(p)) => p.parse_scalar(v),
        (Some(_), None) => Err(KernelError::generic(format!(
            "unexpected partition column type {t:?}"
        ))),
        (None, _) => Ok(Scalar::Null(t.clone())),
    }
}

/// The one-row column of `column`'s value `raw`, in the table field's type.
pub(crate) fn partition_array(
    column: &PartitionColumn,
    raw: Option<&String>,
    table: &str,
) -> Result<arrow_array::ArrayRef, KrishivError> {
    let scalar = partition_scalar(raw, &column.kernel_type).map_err(|e| from_kernel(&e, table))?;
    let array = scalar_to_array(&scalar, 1).map_err(|e| from_kernel(&e, table))?;
    arrow_cast::cast(&array, column.field.data_type())
        .map_err(|e| internal(format!("partition value of {}: {e}", column.field.name())))
}

/// The engine's Arrow field of the kernel field `field`: the kernel's form (with its metadata
/// and field ids) in the engine's canonical types (`Utf8View`, `BinaryView`, `Decimal64` up to
/// precision 18, the engine's Variant and map entries).
pub(crate) fn engine_field(field: &StructField, table: &str) -> Result<Field, KrishivError> {
    let arrow = to_arrow_field(field).map_err(|e| from_kernel(&e, table))?;
    canonical(field.data_type(), &arrow)
}

fn canonical(kernel: &DataType, arrow: &Field) -> Result<Field, KrishivError> {
    let t = match (kernel, arrow.data_type()) {
        (DataType::Primitive(_), leaf) => krishiv_types::DataType::from_arrow(leaf)?.to_arrow(),
        (DataType::Variant(_), _) => krishiv_types::DataType::Variant.to_arrow(),
        (DataType::Struct(k), ArrowType::Struct(a)) => ArrowType::Struct(
            k.fields()
                .zip(a.iter())
                .map(|(kf, af)| canonical(kf.data_type(), af).map(Arc::new))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
        ),
        (DataType::Array(k), ArrowType::List(e)) => {
            ArrowType::List(Arc::new(canonical(k.element_type(), e)?))
        }
        (DataType::Map(k), ArrowType::Map(entries, sorted)) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return Err(internal(format!("map {} without entries", arrow.name())));
            };
            let (Some(key), Some(value)) = (kv.first(), kv.get(1)) else {
                return Err(internal(format!(
                    "map {} without key and value",
                    arrow.name()
                )));
            };
            let kv = vec![
                canonical(k.key_type(), key)?,
                canonical(k.value_type(), value)?,
            ];
            ArrowType::Map(
                Arc::new(Field::new("entries", ArrowType::Struct(kv.into()), false)),
                *sorted,
            )
        }
        _ => {
            return Err(internal(format!(
                "kernel type {kernel:?} has Arrow form {:?}",
                arrow.data_type()
            )));
        }
    };
    Ok(arrow.clone().with_data_type(t))
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod tests;
