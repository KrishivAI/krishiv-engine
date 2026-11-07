// SPDX-License-Identifier: Apache-2.0
//! `GEOMETRY` and `GEOGRAPHY` columns in Parquet (P2b row 4), as Spark 4.2 reads and writes them
//! (`research/2026-11-07-spark-parquet-geo.md`, a real Spark 4.2.0 run).
//!
//! A column is `BINARY` annotated `GEOMETRY(crs)` or `GEOGRAPHY(crs, SPHERICAL)`; each row holds
//! the WKB only, and the SRID comes from the schema, so the engine's in-memory form is the
//! struct `(srid, wkb)` with the schema's SRID in every row.
//!
//! - **Read.** The SRID is the footer's Spark schema (`geometry(EPSG:3857)`) when it has one
//!   for the column, else the logical type's CRS (`srid_of_crs`: none is `OGC:CRS84`; an unknown
//!   CRS, `srid:N` and PROJJSON are `ST_INVALID_CRS_VALUE`). Every value is validated and
//!   re-encoded little endian (`geo::rewrite`); a malformed one fails the scan with
//!   `WKB_PARSE_ERROR`. A plain `BINARY` column stays `binary`. Only top-level columns are
//!   converted; a file with one nested in a struct, array or map is `KRISHIV_UNSUPPORTED_FEATURE`
//!   (never silently read as binary).
//! - **Write.** A top-level column of fixed SRID is written as `BINARY` with the logical type and
//!   the CRS string Spark writes (`SRID:0`, `OGC:CRS84`, `EPSG:3857`, ... explicitly; arrow-rs's
//!   own conversion would write `srid:0`, which Spark 4.2 refuses to read). `GEOMETRY(ANY)` and
//!   `GEOGRAPHY(ANY)` are `UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE`, as in Spark. Not written: the
//!   bounding-box statistics of Spark's files (they need the parquet crate's `geospatial`
//!   feature and its 11 crates; only a reader that prunes by them needs them).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::BinaryViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StructArray};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType as ArrowType, Field, FieldRef, Fields, Schema, SchemaRef};
use krishiv_types::arrow::{SPATIAL_SRID_FIELD, SPATIAL_WKB_FIELD};
use krishiv_types::geo::{ByteOrder, SpatialKind, rewrite, srid_of_crs};
use krishiv_types::{DataType, Srid};
use parquet::basic::{LogicalType, Repetition, Type as PhysicalType};
use parquet::file::metadata::KeyValue;
use parquet::schema::types::{SchemaDescriptor, Type, TypePtr};

use crate::error::{FormatError, FormatResult, unsupported_column};

/// Spark's footer key for the schema JSON.
const ROW_METADATA_KEY: &str = "org.apache.spark.sql.parquet.row.metadata";

/// A spatial column: its kind and the SRID of every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Spatial {
    pub(crate) geography: bool,
    pub(crate) srid: i32,
}

impl Spatial {
    fn kind(self) -> SpatialKind {
        if self.geography {
            SpatialKind::Geography
        } else {
            SpatialKind::Geometry
        }
    }

    fn data_type(self) -> DataType {
        let srid = Srid::Fixed(self.srid);
        if self.geography {
            DataType::Geography { srid }
        } else {
            DataType::Geometry { srid }
        }
    }
}

/// The spatial columns of a file, by top-level name.
#[derive(Debug, Default)]
pub(crate) struct Geos {
    pub(crate) roots: HashMap<String, Spatial>,
}

/// The type the footer's Spark schema gives top-level column `name`, when it has one.
fn declared(footer: Option<&Vec<KeyValue>>, name: &str) -> Option<DataType> {
    let json = footer?
        .iter()
        .find(|kv| kv.key == ROW_METADATA_KEY)?
        .value
        .as_deref()?;
    let DataType::Struct(s) = DataType::from_json(json).ok()? else {
        return None;
    };
    s.fields()
        .iter()
        .find(|f| f.name() == name)
        .map(|f| f.data_type().clone())
}

impl Geos {
    /// `fields` (the reader's top-level fields) of a file with `descr` and `footer`, a column
    /// annotated `GEOMETRY` or `GEOGRAPHY` typed as the engine's spatial type.
    ///
    /// # Errors
    /// `ST_INVALID_CRS_VALUE` for a CRS Spark does not read.
    pub(super) fn engine_fields(
        descr: &SchemaDescriptor,
        footer: Option<&Vec<KeyValue>>,
        fields: Vec<FieldRef>,
    ) -> FormatResult<(Vec<FieldRef>, Self)> {
        for t in descr.root_schema().get_fields() {
            if t.is_group() && nested_spatial(t) {
                return Err(FormatError::Other(
                    krishiv_common::error::KrishivUnsupportedFeature {
                        feature: format!(
                            "reading Parquet GEOMETRY or GEOGRAPHY nested in the column `{}` \
                             (only top-level columns)",
                            t.name()
                        ),
                    }
                    .into(),
                ));
            }
        }
        let mut roots = HashMap::new();
        let mut out = Vec::with_capacity(fields.len());
        for f in fields {
            let logical = descr
                .root_schema()
                .get_fields()
                .iter()
                .find(|t| t.name() == f.name() && !t.is_group())
                .and_then(|t| t.get_basic_info().logical_type_ref().cloned());
            let (geography, crs) = match &logical {
                Some(LogicalType::Geometry(g)) => (false, g.crs.clone()),
                Some(LogicalType::Geography(g)) => (true, g.crs.clone()),
                _ => {
                    out.push(f);
                    continue;
                }
            };
            // The footer's Spark schema wins, as in `ParquetFileFormat`.
            let spatial = match declared(footer, f.name()) {
                Some(DataType::Geometry {
                    srid: Srid::Fixed(srid),
                }) if !geography => Spatial {
                    geography: false,
                    srid,
                },
                Some(DataType::Geography {
                    srid: Srid::Fixed(srid),
                }) if geography => Spatial {
                    geography: true,
                    srid,
                },
                _ => Spatial {
                    geography,
                    srid: srid_of_crs(crs.as_deref(), geography).map_err(FormatError::Other)?,
                },
            };
            out.push(Arc::new(
                spatial
                    .data_type()
                    .to_arrow_field(f.name().clone(), f.is_nullable()),
            ));
            roots.insert(f.name().clone(), spatial);
        }
        Ok((out, Self { roots }))
    }
}

/// Whether a leaf below group `t` is annotated `GEOMETRY` or `GEOGRAPHY`.
fn nested_spatial(t: &Type) -> bool {
    if t.is_group() {
        return t.get_fields().iter().any(|c| nested_spatial(c));
    }
    matches!(
        t.get_basic_info().logical_type_ref(),
        Some(LogicalType::Geometry(_) | LogicalType::Geography(_))
    )
}

/// `batch` with each spatial column of `roots` (a `BINARY` of WKB) as the in-memory struct of
/// `engine`, the file's engine schema.
///
/// # Errors
/// `WKB_PARSE_ERROR` (as a failed read) for a value that is not well-formed WKB.
pub(super) fn to_engine(
    batch: &RecordBatch,
    engine: &Schema,
    roots: &HashMap<String, Spatial>,
) -> FormatResult<RecordBatch> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (f, c) in schema.fields().iter().zip(batch.columns()) {
        match (roots.get(f.name()), engine.field_with_name(f.name())) {
            (Some(spatial), Ok(target)) => {
                columns.push(spatial_column(c, *spatial)?);
                fields.push(Arc::new(target.clone()));
            }
            _ => {
                columns.push(Arc::clone(c));
                fields.push(Arc::clone(f));
            }
        }
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| FormatError::Corrupt(e.to_string()))
}

/// The WKB bytes of row `i` of a binary column, `None` for NULL.
fn bytes(a: &ArrayRef, i: usize) -> FormatResult<Option<&[u8]>> {
    if a.is_null(i) {
        return Ok(None);
    }
    Ok(Some(match a.data_type() {
        ArrowType::Binary => a.as_binary::<i32>().value(i),
        ArrowType::LargeBinary => a.as_binary::<i64>().value(i),
        ArrowType::BinaryView => a.as_binary_view().value(i),
        other => {
            return Err(FormatError::Corrupt(format!(
                "a GEOMETRY column read as {other}"
            )));
        }
    }))
}

fn spatial_column(a: &ArrayRef, spatial: Spatial) -> FormatResult<ArrayRef> {
    let mut wkb = BinaryViewBuilder::with_capacity(a.len());
    for i in 0..a.len() {
        match bytes(a, i)? {
            Some(v) => {
                let v = rewrite(v, spatial.kind(), ByteOrder::Ndr).map_err(FormatError::Failed)?;
                wkb.append_value(v);
            }
            // The struct is NULL here; its WKB child holds an empty placeholder.
            None => wkb.append_value([]),
        }
    }
    let srid = Int32Array::from(vec![spatial.srid; a.len()]);
    let nulls = a.nulls().cloned();
    let fields = Fields::from(vec![
        Field::new(SPATIAL_SRID_FIELD, ArrowType::Int32, false),
        Field::new(SPATIAL_WKB_FIELD, ArrowType::BinaryView, false),
    ]);
    StructArray::try_new(
        fields,
        vec![Arc::new(srid) as ArrayRef, Arc::new(wkb.finish())],
        nulls,
    )
    .map(|s| Arc::new(s) as ArrayRef)
    .map_err(|e| FormatError::Corrupt(e.to_string()))
}

/// The top-level spatial columns of `schema` to write, by position.
///
/// # Errors
/// `UNSUPPORTED_DATA_TYPE_FOR_DATASOURCE` for `GEOMETRY(ANY)` / `GEOGRAPHY(ANY)`.
pub(super) fn spatial_columns(schema: &Schema) -> FormatResult<Vec<(usize, Spatial)>> {
    let mut out = Vec::new();
    for (i, f) in schema.fields().iter().enumerate() {
        let (geography, srid) = match DataType::from_arrow_field(f) {
            Ok(DataType::Geometry { srid }) => (false, srid),
            Ok(DataType::Geography { srid }) => (true, srid),
            _ => continue,
        };
        match srid {
            Srid::Fixed(srid) => out.push((i, Spatial { geography, srid })),
            Srid::Any => return Err(unsupported_column("Parquet", f)),
        }
    }
    Ok(out)
}

/// `schema` with each column of `spatial` stored as a plain `BINARY` of WKB.
pub(super) fn stored_schema(schema: &SchemaRef, spatial: &[(usize, Spatial)]) -> SchemaRef {
    if spatial.is_empty() {
        return Arc::clone(schema);
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if spatial.iter().any(|(j, _)| *j == i) {
                Field::new(f.name(), ArrowType::BinaryView, f.is_nullable())
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// `batch` with the struct of each column of `spatial` replaced by its WKB, as `stored`.
///
/// # Errors
/// `WriteFailed` when a column is not the spatial struct.
pub(super) fn to_stored(
    batch: &RecordBatch,
    spatial: &[(usize, Spatial)],
    stored: &SchemaRef,
) -> FormatResult<RecordBatch> {
    let bad = |what: &str| FormatError::WriteFailed(what.to_owned());
    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if !spatial.iter().any(|(j, _)| *j == i) {
                return Ok(Arc::clone(c));
            }
            let s = c
                .as_struct_opt()
                .ok_or_else(|| bad("a spatial column that is not a struct"))?;
            let wkb = s
                .column_by_name(SPATIAL_WKB_FIELD)
                .ok_or_else(|| bad("a spatial struct without WKB"))?;
            let data = wkb
                .to_data()
                .into_builder()
                .nulls(NullBuffer::union(wkb.nulls(), s.nulls()))
                .build()
                .map_err(|e| FormatError::WriteFailed(e.to_string()))?;
            Ok(arrow_array::make_array(data))
        })
        .collect::<FormatResult<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        Arc::clone(stored),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| FormatError::WriteFailed(e.to_string()))
}

/// The Parquet schema `descr` with each column of `spatial` annotated `GEOMETRY(crs)` or
/// `GEOGRAPHY(crs, SPHERICAL)` with the CRS string Spark writes.
///
/// # Errors
/// `WriteFailed` when the schema cannot be rebuilt.
pub(super) fn annotate(
    descr: SchemaDescriptor,
    schema: &Schema,
    spatial: &[(usize, Spatial)],
) -> FormatResult<SchemaDescriptor> {
    if spatial.is_empty() {
        return Ok(descr);
    }
    let fail = |e: parquet::errors::ParquetError| FormatError::WriteFailed(e.to_string());
    let root = descr.root_schema();
    let children = root
        .get_fields()
        .iter()
        .map(|t| -> FormatResult<TypePtr> {
            let found = schema
                .fields()
                .iter()
                .position(|f| f.name() == t.name())
                .and_then(|i| spatial.iter().find(|(j, _)| *j == i));
            let Some((_, s)) = found else {
                return Ok(Arc::clone(t));
            };
            let crs = Srid::crs(s.srid, s.geography).map(str::to_owned);
            let logical = if s.geography {
                LogicalType::geography(crs, None)
            } else {
                LogicalType::geometry(crs)
            };
            let info = t.get_basic_info();
            let repetition = if info.has_repetition() {
                info.repetition()
            } else {
                Repetition::OPTIONAL
            };
            Type::primitive_type_builder(t.name(), PhysicalType::BYTE_ARRAY)
                .with_repetition(repetition)
                .with_logical_type(Some(logical))
                .build()
                .map(Arc::new)
                .map_err(fail)
        })
        .collect::<FormatResult<Vec<_>>>()?;
    let root = Type::group_type_builder(root.name())
        .with_fields(children)
        .build()
        .map_err(fail)?;
    Ok(SchemaDescriptor::new(Arc::new(root)))
}
