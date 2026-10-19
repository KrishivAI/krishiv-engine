// SPDX-License-Identifier: Apache-2.0
//! The shared [`SchemaMapper`]: schema evolution and case rules, once for
//! every format.
//!
//! For one file it resolves each projected table column to a file column
//! ([`FieldMapping`]: by name with Spark's case rules, by field id, or by
//! Delta column mapping), then:
//! - maps the **request** to the file ([`SchemaMapper::file_request`]): the
//!   projection in file names (nested pruning kept), and the predicate
//!   **re-typed to the file's types before pruning** (a literal outside the
//!   file type's range makes the comparison constant, never a cast error:
//!   `int_col = 3e9` on an `INT32` column matches nothing);
//! - maps every decoded **batch** back ([`SchemaMapper::map_batch`]): a
//!   missing column, struct field or nested field is null (recursively
//!   through structs, lists and maps), legal widenings are cast, and the
//!   result has the table's names and types made nullable (Spark reads file
//!   sources with `asNullable`).
//!
//! Legal type changes (file → table): integer widening, integers and
//! `float` to `double`, integers and decimals to a decimal that holds them,
//! `date` to `timestamp_ntz` (Spark 4's Parquet type widening), the string,
//! binary and timestamp representations of one Spark type (zone labels;
//! `timestamp_ntz` and `timestamp` read as each other with the stored
//! microseconds unchanged, as Spark 4.2 does), and integer
//! narrowing (`int` read as `smallint`, as Spark's Parquet reader allows;
//! a value that does not fit is a `CAST_OVERFLOW` error, where Spark wraps
//! silently). Anything else is `FAILED_READ_FILE.PARQUET_COLUMN_DATA_TYPE_MISMATCH`.
//! Case-insensitive resolution matching several fields is
//! `FAILED_READ_FILE.NO_HINT` caused by Spark's duplicate-field error
//! (`_LEGACY_ERROR_TEMP_2093`), as Spark 4.2 reports it. A stored struct none
//! of whose requested fields exists reads as a struct of nulls (null where
//! the stored struct is null), as in Spark 4.2.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeListArray,
    LargeStringArray, ListArray, MapArray, PrimitiveArray, RecordBatch, RecordBatchOptions,
    StringArray, StringViewArray, StructArray, new_null_array,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, Schema, SchemaRef, TimeUnit};
use krishiv_common::error::{
    CastOverflow, FailedReadFileNoHint, KrishivError, LegacyErrorTemp2093,
};

use crate::error::{FormatError, Result, internal};
use crate::request::{
    ColumnPath, FieldMapping, PushdownExpr, ROW_INDEX_COLUMN, ScalarValue, ScanRequest,
    project_schema,
};

/// Arrow field-metadata key of a field id (Parquet's, as arrow-rs reads it).
pub const FIELD_ID_KEY: &str = "PARQUET:field_id";
const DELTA_PHYSICAL_NAME: &str = "delta.columnMapping.physicalName";
const DELTA_ID: &str = "delta.columnMapping.id";

/// How one value is converted from its file type to its table type.
#[derive(Debug, Clone)]
enum Conv {
    /// Same type (up to nullability and metadata).
    Same,
    /// A representation change, widening or narrowing of a leaf.
    Cast,
    /// Per table child: its position in the decoded file struct and its
    /// conversion, or `None` (missing: null).
    Struct(Vec<Option<(usize, Conv)>>),
    /// Element conversion.
    List(Box<Conv>),
    /// Key and value conversions.
    Map(Box<Conv>, Box<Conv>),
}

/// One output column.
#[derive(Debug, Clone)]
enum Plan {
    /// Missing in the file.
    Null,
    /// Position in the decoded batch and conversion.
    Take(usize, Conv),
}

/// Resolution rules of the mapper.
#[derive(Debug)]
struct Resolver {
    table: SchemaRef,
    file: SchemaRef,
    mapping: FieldMapping,
    path: String,
}

/// Maps a table-level request to one file and its batches back (see the
/// module docs).
#[derive(Debug, Clone)]
pub struct SchemaMapper {
    resolver: Arc<Resolver>,
    output: SchemaRef,
    file_output: SchemaRef,
    file_projection: Vec<ColumnPath>,
    columns: Vec<Plan>,
    row_index: bool,
}

impl SchemaMapper {
    /// Resolves `req` (table level) against `file_schema` of the file
    /// printed as `path`.
    ///
    /// # Errors
    /// `FAILED_READ_FILE.PARQUET_COLUMN_DATA_TYPE_MISMATCH` for an illegal
    /// type change of a projected column; `FAILED_READ_FILE.NO_HINT` caused
    /// by `_LEGACY_ERROR_TEMP_2093` for an ambiguous case-insensitive match.
    pub fn new(req: &ScanRequest, file_schema: SchemaRef, path: &str) -> Result<Self> {
        let resolver = Arc::new(Resolver {
            table: Arc::clone(&req.table_schema),
            file: file_schema,
            mapping: req.mapping.clone(),
            path: path.to_owned(),
        });
        let projected = project_schema(&req.table_schema, &req.projection)?;
        let mut file_projection = Vec::new();
        let mut file_order: Vec<String> = Vec::new();
        let mut columns = Vec::with_capacity(projected.fields().len());
        for tf in projected.fields() {
            let Some(j) = resolver.find(resolver.file.fields(), tf)? else {
                columns.push(Plan::Null);
                continue;
            };
            let ff = resolver.file.field(j);
            let resolved = resolver.resolve(tf, ff, &[tf.name().clone()], true)?;
            let Some((conv, subs)) = resolved else {
                columns.push(Plan::Null);
                continue;
            };
            let pos = file_order
                .iter()
                .position(|n| n == ff.name())
                .unwrap_or_else(|| {
                    file_order.push(ff.name().clone());
                    file_order.len() - 1
                });
            for sub in subs {
                let mut p = vec![ff.name().clone()];
                p.extend(sub);
                file_projection.push(ColumnPath(p));
            }
            columns.push(Plan::Take(pos, conv));
        }
        let row_index = req.metadata_cols.row_index;
        let file_output = with_row_index(
            project_schema(&resolver.file, &file_projection)?,
            row_index,
            false,
        );
        let output = with_row_index(Arc::new(as_nullable_schema(&projected)), row_index, true);
        Ok(Self {
            resolver,
            output,
            file_output,
            file_projection,
            columns,
            row_index,
        })
    }

    /// The schema of mapped batches: the table-level output, nullable.
    #[must_use]
    pub fn output_schema(&self) -> &SchemaRef {
        &self.output
    }

    /// The schema decoders must produce for [`Self::file_request`].
    #[must_use]
    pub fn file_output_schema(&self) -> &SchemaRef {
        &self.file_output
    }

    /// The file-level request: file names, nested pruning, the predicate
    /// re-typed to the file, and runtime filters on the file's columns
    /// (each lowers to the file's type when read).
    #[must_use]
    pub fn file_request(&self, req: &ScanRequest) -> ScanRequest {
        ScanRequest {
            table_schema: Arc::clone(&self.resolver.file),
            projection: self.file_projection.clone(),
            predicate: req.predicate.as_ref().and_then(|p| self.map_predicate(p)),
            dynamic: req
                .dynamic
                .iter()
                .filter_map(|d| {
                    // The file's column and type (`None`: missing, NULL);
                    // an unresolvable path is not filtered.
                    let (path, _) = d.column()?;
                    let column = self
                        .resolver
                        .resolve_path(path)?
                        .map(|(fc, ft, _)| (fc, ft));
                    Some(crate::ScanFilter::at(Arc::clone(d.filter()), column))
                })
                .collect(),
            selection: req.selection.clone(),
            row_filter: req.row_filter,
            limit: req.limit,
            metadata_cols: req.metadata_cols,
            mapping: FieldMapping::ByName {
                case_sensitive: true,
            },
        }
    }

    /// `expr` (table level) in the file's names and types, weakened where it
    /// cannot be expressed: dropped conjuncts, or `None` when nothing of it
    /// can be pushed. Sound: never excludes a row the original matches.
    #[must_use]
    pub fn map_predicate(&self, expr: &PushdownExpr) -> Option<PushdownExpr> {
        self.resolver.map_predicate(expr)
    }

    /// A decoded batch (in [`Self::file_output_schema`]) in the table's
    /// names and types.
    ///
    /// # Errors
    /// `CAST_OVERFLOW` for a narrowed value that does not fit;
    /// `INTERNAL_ERROR` if the batch does not have the file-level schema.
    pub fn map_batch(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let expected = self.file_output.fields().len();
        if batch.num_columns() != expected {
            return Err(internal(format!(
                "decoder produced {} columns, the request needs {expected}",
                batch.num_columns()
            )));
        }
        let rows = batch.num_rows();
        let mut out = Vec::with_capacity(self.output.fields().len());
        for (plan, field) in self.columns.iter().zip(self.output.fields()) {
            out.push(match plan {
                Plan::Null => new_null_array(field.data_type(), rows),
                Plan::Take(pos, conv) => convert(batch.column(*pos), conv, field.data_type())?,
            });
        }
        if self.row_index
            && let Some(c) = batch.columns().last()
        {
            out.push(Arc::clone(c));
        }
        RecordBatch::try_new_with_options(
            Arc::clone(&self.output),
            out,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(arrow_internal)
    }
}

#[expect(clippy::needless_pass_by_value, reason = "passed to `map_err` by name")]
fn arrow_internal(e: ArrowError) -> KrishivError {
    internal(format!("schema mapping produced an invalid batch: {e}"))
}

fn with_row_index(schema: SchemaRef, row_index: bool, _nullable: bool) -> SchemaRef {
    if !row_index {
        return schema;
    }
    let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(
        ROW_INDEX_COLUMN,
        DataType::Int64,
        false,
    )));
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn as_nullable_schema(s: &Schema) -> Schema {
    Schema::new_with_metadata(
        s.fields()
            .iter()
            .map(|f| as_nullable(f))
            .collect::<Vec<_>>(),
        s.metadata().clone(),
    )
}

fn as_nullable(f: &Field) -> Field {
    f.clone()
        .with_nullable(true)
        .with_data_type(as_nullable_type(f.data_type()))
}

fn as_nullable_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Struct(fs) => {
            DataType::Struct(fs.iter().map(|f| as_nullable(f)).collect::<Fields>())
        }
        DataType::List(f) => DataType::List(Arc::new(as_nullable(f))),
        DataType::LargeList(f) => DataType::LargeList(Arc::new(as_nullable(f))),
        DataType::Map(entries, sorted) => {
            // Map keys stay non-null (Arrow requires it); values become nullable.
            let DataType::Struct(kv) = entries.data_type() else {
                return dt.clone();
            };
            let kv: Fields = kv
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    if i == 0 {
                        f.as_ref().clone()
                    } else {
                        as_nullable(f)
                    }
                })
                .collect();
            DataType::Map(
                Arc::new(
                    entries
                        .as_ref()
                        .clone()
                        .with_data_type(DataType::Struct(kv)),
                ),
                *sorted,
            )
        }
        other => other.clone(),
    }
}

/// The Spark rendering of `dt` for messages.
fn spark_name(dt: &DataType) -> String {
    krishiv_types::DataType::from_arrow(dt).map_or_else(
        |_| dt.to_string().to_ascii_lowercase(),
        |t| t.simple_string(),
    )
}

/// Whether the file type `file` can be read as `table` by a leaf cast.
fn leaf_castable(file: &DataType, table: &DataType) -> bool {
    use DataType as D;
    let int_digits = |d: &D| match d {
        D::Int8 => Some(3),
        D::Int16 => Some(5),
        D::Int32 => Some(10),
        D::Int64 => Some(20),
        _ => None,
    };
    match (file, table) {
        (D::Utf8 | D::LargeUtf8 | D::Utf8View, D::Utf8 | D::LargeUtf8 | D::Utf8View)
        | (
            D::Binary | D::LargeBinary | D::BinaryView,
            D::Binary | D::LargeBinary | D::BinaryView,
        )
        | (D::Int8, D::Int16 | D::Int32 | D::Int64)
        | (D::Int16, D::Int32 | D::Int64 | D::Int8)
        | (D::Int32, D::Int64 | D::Int16 | D::Int8)
        | (D::Int8 | D::Int16 | D::Int32 | D::Float32, D::Float64)
        | (D::Date32, D::Timestamp(TimeUnit::Microsecond, None)) => true,
        // Zone labels only (Spark 4.2 reads NTZ as LTZ and LTZ as NTZ with
        // the stored microseconds unchanged).
        (D::Timestamp(u1, _), D::Timestamp(u2, _)) => u1 == u2,
        (D::Decimal128(p1, s1), D::Decimal128(p2, s2)) => {
            s2 >= s1 && i16::from(*p2) - i16::from(*s2) >= i16::from(*p1) - i16::from(*s1)
        }
        (f, D::Decimal128(p, s)) => {
            int_digits(f).is_some_and(|d| i16::from(*p) - i16::from(*s) >= d)
        }
        _ => false,
    }
}

impl Resolver {
    /// The file field among `children` that `table_field` maps to.
    fn find(&self, children: &Fields, table_field: &Field) -> Result<Option<usize>> {
        let by_meta = |key: &str, tf: &Field| -> Option<usize> {
            let id = tf.metadata().get(key)?;
            children
                .iter()
                .position(|f| f.metadata().get(FIELD_ID_KEY) == Some(id))
        };
        Ok(match &self.mapping {
            FieldMapping::ByName {
                case_sensitive: true,
            } => children.iter().position(|f| f.name() == table_field.name()),
            FieldMapping::ByName {
                case_sensitive: false,
            } => {
                let want = table_field.name().to_lowercase();
                let hits: Vec<usize> = children
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.name().to_lowercase() == want)
                    .map(|(i, _)| i)
                    .collect();
                if hits.len() > 1 {
                    let names: Vec<&str> =
                        hits.iter().map(|&i| children[i].name().as_str()).collect();
                    let duplicate = LegacyErrorTemp2093 {
                        required_field_name: table_field.name().clone(),
                        matched_orc_fields: format!("[{}]", names.join(", ")),
                    };
                    return Err(KrishivError::from(FailedReadFileNoHint {
                        path: self.path.clone(),
                    })
                    .with_cause(duplicate.into()));
                }
                hits.first().copied()
            }
            // A table field without an id is matched by name (Spark 4.2).
            FieldMapping::FieldId => match table_field.metadata().get(FIELD_ID_KEY) {
                Some(_) => by_meta(FIELD_ID_KEY, table_field),
                None => children.iter().position(|f| f.name() == table_field.name()),
            },
            FieldMapping::DeltaColumnMapping { by_id: true } => by_meta(DELTA_ID, table_field),
            FieldMapping::DeltaColumnMapping { by_id: false } => {
                let name = table_field
                    .metadata()
                    .get(DELTA_PHYSICAL_NAME)
                    .unwrap_or(table_field.name());
                children.iter().position(|f| f.name() == name)
            }
        })
    }

    /// The conversion of file field `ff` to table field `tf` and the file
    /// sub-paths to read (relative to `ff`). With `prune`, struct children
    /// are read selectively and positions refer to the pruned struct; a
    /// struct none of whose requested children exist is `None` (null).
    fn resolve(
        &self,
        tf: &Field,
        ff: &Field,
        at: &[String],
        prune: bool,
    ) -> Result<Option<(Conv, Vec<Vec<String>>)>> {
        let (t, f) = (tf.data_type(), ff.data_type());
        let whole = vec![Vec::new()];
        if t.equals_datatype(f) && !needs_struct_resolution(t) {
            return Ok(Some((Conv::Same, whole)));
        }
        match (t, f) {
            (DataType::Struct(tc), DataType::Struct(fc)) => {
                let mut picks = Vec::with_capacity(tc.len());
                for child in tc {
                    let mut child_at = at.to_vec();
                    child_at.push(child.name().clone());
                    let pick = match self.find(fc, child)? {
                        None => None,
                        Some(j) => self
                            .resolve(child, &fc[j], &child_at, prune)?
                            .map(|(c, subs)| (j, c, subs)),
                    };
                    picks.push(pick);
                }
                if prune && picks.iter().all(Option::is_none) {
                    // None of the requested fields is stored, but the struct
                    // is: Spark returns it with null fields, null only where
                    // the stored struct is null. Its first field is read for
                    // the struct's validity.
                    let Some(first) = fc.first() else {
                        return Ok(None);
                    };
                    let convs = vec![None; picks.len()];
                    return Ok(Some((
                        Conv::Struct(convs),
                        vec![vec![first.name().clone()]],
                    )));
                }
                let mut needed: Vec<usize> = picks.iter().flatten().map(|(j, ..)| *j).collect();
                needed.sort_unstable();
                needed.dedup();
                let mut subs = Vec::new();
                let convs = picks
                    .into_iter()
                    .map(|p| {
                        p.map(|(j, conv, child_subs)| {
                            if prune {
                                for s in child_subs {
                                    let mut path = vec![fc[j].name().clone()];
                                    path.extend(s);
                                    subs.push(path);
                                }
                            }
                            let pos = if prune {
                                needed.iter().position(|n| *n == j).unwrap_or(j)
                            } else {
                                j
                            };
                            (pos, conv)
                        })
                    })
                    .collect();
                Ok(Some((
                    Conv::Struct(convs),
                    if prune { subs } else { whole },
                )))
            }
            (DataType::List(te), DataType::List(fe))
            | (DataType::LargeList(te), DataType::LargeList(fe)) => {
                let elem = self
                    .resolve(te, fe, at, false)?
                    .map_or(Conv::Struct(Vec::new()), |(c, _)| c);
                Ok(Some((Conv::List(Box::new(elem)), whole)))
            }
            (DataType::Map(te, _), DataType::Map(fe, _)) => {
                let (DataType::Struct(tkv), DataType::Struct(fkv)) =
                    (te.data_type(), fe.data_type())
                else {
                    return Err(self.mismatch(at, t, f));
                };
                let (Some(tk), Some(tv), Some(fk), Some(fv)) =
                    (tkv.first(), tkv.get(1), fkv.first(), fkv.get(1))
                else {
                    return Err(self.mismatch(at, t, f));
                };
                let k = self
                    .resolve(tk, fk, at, false)?
                    .map_or(Conv::Same, |(c, _)| c);
                let v = self
                    .resolve(tv, fv, at, false)?
                    .map_or(Conv::Same, |(c, _)| c);
                Ok(Some((Conv::Map(Box::new(k), Box::new(v)), whole)))
            }
            _ if leaf_castable(f, t) => Ok(Some((Conv::Cast, whole))),
            _ => Err(self.mismatch(at, t, f)),
        }
    }

    fn mismatch(&self, at: &[String], t: &DataType, f: &DataType) -> KrishivError {
        FormatError::TypeMismatch {
            column: format!("[{}]", at.join(", ")),
            expected: spark_name(t),
            actual: spark_name(f),
        }
        .into_error(&self.path)
    }

    /// The file path, file type and table type of the table column `path`;
    /// `Ok(None)` when it is missing in the file; `Err` when it cannot be
    /// resolved (ambiguous, not a struct path).
    fn resolve_path(&self, path: &ColumnPath) -> Option<Option<(ColumnPath, DataType, DataType)>> {
        let mut tchildren = self.table.fields().clone();
        let mut fchildren = self.file.fields().clone();
        let mut out = Vec::with_capacity(path.0.len());
        let mut types = None;
        for (i, name) in path.0.iter().enumerate() {
            if i > 0 {
                let Some((DataType::Struct(tc), DataType::Struct(fc))) = types.take() else {
                    return None;
                };
                tchildren = tc;
                fchildren = fc;
            }
            let tf = tchildren.iter().find(|f| f.name() == name)?;
            let Ok(found) = self.find(&fchildren, tf) else {
                return None;
            };
            let Some(j) = found else {
                return Some(None);
            };
            let ff = &fchildren[j];
            out.push(ff.name().clone());
            types = Some((tf.data_type().clone(), ff.data_type().clone()));
        }
        let (t, f) = types?;
        Some(Some((ColumnPath(out), f, t)))
    }

    fn map_predicate(&self, expr: &PushdownExpr) -> Option<PushdownExpr> {
        self.map_node(&expr.to_nnf())
    }

    fn map_node(&self, e: &PushdownExpr) -> Option<PushdownExpr> {
        use PushdownExpr as P;
        match e {
            P::Constant(b) => Some(P::Constant(*b)),
            P::And(xs) => {
                let mapped: Vec<P> = xs.iter().filter_map(|x| self.map_node(x)).collect();
                (!mapped.is_empty()).then_some(P::And(mapped))
            }
            P::Or(xs) => xs
                .iter()
                .map(|x| self.map_node(x))
                .collect::<Option<Vec<_>>>()
                .map(P::Or),
            P::Not(x) => self.map_node(x).map(|m| P::Not(Box::new(m))),
            P::IsNull(c) => Some(match self.resolve_path(c)? {
                None => P::Constant(true),
                Some((fc, ..)) => P::IsNull(fc),
            }),
            P::IsNotNull(c) => Some(match self.resolve_path(c)? {
                None => P::Constant(false),
                Some((fc, ..)) => P::IsNotNull(fc),
            }),
            P::Cmp {
                column,
                op,
                value,
                collation,
            } => {
                let Some((fc, ft, tt)) = self.resolve_path(column)? else {
                    return Some(P::Constant(false));
                };
                Some(match retype(value, &tt, &ft, *op)? {
                    Retyped::Exact(v) => P::Cmp {
                        column: fc,
                        op: *op,
                        value: v,
                        collation: collation.clone(),
                    },
                    Retyped::Always => P::IsNotNull(fc),
                    Retyped::Never => P::Constant(false),
                })
            }
            P::In {
                column,
                values,
                collation,
            } => {
                let Some((fc, ft, tt)) = self.resolve_path(column)? else {
                    return Some(P::Constant(false));
                };
                let mut kept = Vec::with_capacity(values.len());
                for v in values {
                    match retype(v, &tt, &ft, crate::request::CmpOp::Eq)? {
                        Retyped::Exact(v) => kept.push(v),
                        Retyped::Never => {}
                        Retyped::Always => return None,
                    }
                }
                Some(if kept.is_empty() {
                    P::Constant(false)
                } else {
                    P::In {
                        column: fc,
                        values: kept,
                        collation: collation.clone(),
                    }
                })
            }
            P::Membership { column, filter } => {
                let Some((fc, ft, _)) = self.resolve_path(column)? else {
                    return Some(P::Constant(filter.keys.nulls_pass()));
                };
                // Values are cast to the key type; a file type that does
                // not widen to it is not filtered.
                crate::runtime_filter::widens_to(&ft, filter.keys.key_type()).then(|| {
                    P::Membership {
                        column: fc,
                        filter: filter.clone(),
                    }
                })
            }
            P::StartsWith {
                column,
                prefix,
                collation,
            } => {
                let Some((fc, ft, _)) = self.resolve_path(column)? else {
                    return Some(P::Constant(false));
                };
                matches!(
                    ft,
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                )
                .then(|| P::StartsWith {
                    column: fc,
                    prefix: prefix.clone(),
                    collation: collation.clone(),
                })
            }
        }
    }
}

/// Whether two equal-looking types still need struct resolution (field
/// names may differ in case or by id).
fn needs_struct_resolution(dt: &DataType) -> bool {
    match dt {
        DataType::Struct(_) | DataType::Map(..) => true,
        DataType::List(f) | DataType::LargeList(f) => needs_struct_resolution(f.data_type()),
        _ => false,
    }
}

/// A literal re-typed from the table type to the file type.
pub(crate) enum Retyped {
    /// The equivalent literal of the file type.
    Exact(ScalarValue),
    /// Every non-null value satisfies the comparison.
    Always,
    /// No value satisfies it.
    Never,
}

/// `value` (of table type `t`) as a literal of file type `f` for `op`;
/// `None` when the comparison cannot be expressed (it is then not pushed).
pub(crate) fn retype(
    value: &ScalarValue,
    t: &DataType,
    f: &DataType,
    op: crate::request::CmpOp,
) -> Option<Retyped> {
    use crate::request::CmpOp as Op;
    use ScalarValue as S;
    if t.equals_datatype(f) || representation_only(f, t) {
        return Some(Retyped::Exact(value.clone()));
    }
    // A value outside the file type's range, below (`true`) or above it.
    let out_of_range = |below: bool| match op {
        Op::Eq => Retyped::Never,
        Op::NotEq => Retyped::Always,
        Op::Lt | Op::LtEq => {
            if below {
                Retyped::Never
            } else {
                Retyped::Always
            }
        }
        Op::Gt | Op::GtEq => {
            if below {
                Retyped::Always
            } else {
                Retyped::Never
            }
        }
    };
    // A value strictly between two file values: only (in)equality is known.
    let inexact = || match op {
        Op::Eq => Some(Retyped::Never),
        Op::NotEq => Some(Retyped::Always),
        _ => None,
    };
    let as_int = |n: i128| -> Option<Retyped> {
        let (lo, hi) = int_range(f)?;
        if n < lo {
            return Some(out_of_range(true));
        }
        if n > hi {
            return Some(out_of_range(false));
        }
        int_scalar(f, n).map(Retyped::Exact)
    };
    match (value, f) {
        (S::Int8(n), _) => as_int(i128::from(*n)),
        (S::Int16(n), _) => as_int(i128::from(*n)),
        (S::Int32(n), _) => as_int(i128::from(*n)),
        (S::Int64(n), _) => as_int(i128::from(*n)),
        (S::Float64(x), DataType::Float32) => {
            if x.is_nan() {
                return None;
            }
            #[expect(clippy::cast_possible_truncation, reason = "checked exact below")]
            let y = *x as f32;
            if f64::from(y) == *x {
                Some(Retyped::Exact(S::Float32(y)))
            } else {
                inexact()
            }
        }
        (S::Float64(x), _) if int_range(f).is_some() => {
            if x.is_nan() || x.is_infinite() {
                return None;
            }
            if x.fract() != 0.0 {
                return inexact();
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "integral and range-checked by as_int"
            )]
            let n = *x as i128;
            as_int(n)
        }
        (
            S::Decimal128 {
                unscaled, scale, ..
            },
            _,
        ) => {
            let (fp, fs) = match f {
                DataType::Decimal128(p, s) => (*p, *s),
                _ if int_range(f).is_some() => (0, 0),
                _ => return None,
            };
            let diff = u32::try_from(i16::from(*scale) - i16::from(fs)).ok()?;
            let div = 10_i128.checked_pow(diff)?;
            if unscaled % div != 0 {
                return inexact();
            }
            let u = unscaled / div;
            if fp == 0 {
                return as_int(u);
            }
            let bound = 10_i128.checked_pow(u32::from(fp))?;
            if u <= -bound {
                Some(out_of_range(true))
            } else if u >= bound {
                Some(out_of_range(false))
            } else {
                Some(Retyped::Exact(S::Decimal128 {
                    unscaled: u,
                    precision: fp,
                    scale: fs,
                }))
            }
        }
        (S::TimestampMicros(us), DataType::Date32) => {
            const DAY: i64 = 86_400_000_000;
            if us % DAY != 0 {
                return inexact();
            }
            i32::try_from(us / DAY)
                .ok()
                .map(|d| Retyped::Exact(S::Date32(d)))
        }
        _ => None,
    }
}

fn representation_only(f: &DataType, t: &DataType) -> bool {
    use DataType as D;
    matches!(
        (f, t),
        (
            D::Utf8 | D::LargeUtf8 | D::Utf8View,
            D::Utf8 | D::LargeUtf8 | D::Utf8View
        ) | (
            D::Binary | D::LargeBinary | D::BinaryView,
            D::Binary | D::LargeBinary | D::BinaryView
        )
    ) || matches!(
        // Timestamps differ only by their zone label when their units agree
        // (NTZ and LTZ store the same microseconds); other units hold other
        // values and are not re-typed.
        (f, t),
        (D::Timestamp(u1, _), D::Timestamp(u2, _)) if u1 == u2
    )
}

fn int_range(dt: &DataType) -> Option<(i128, i128)> {
    Some(match dt {
        DataType::Int8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
        DataType::Int16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
        DataType::Int32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        DataType::Int64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        _ => return None,
    })
}

fn int_scalar(dt: &DataType, n: i128) -> Option<ScalarValue> {
    Some(match dt {
        DataType::Int8 => ScalarValue::Int8(i8::try_from(n).ok()?),
        DataType::Int16 => ScalarValue::Int16(i16::try_from(n).ok()?),
        DataType::Int32 => ScalarValue::Int32(i32::try_from(n).ok()?),
        DataType::Int64 => ScalarValue::Int64(i64::try_from(n).ok()?),
        _ => return None,
    })
}

/// `array` converted by `conv` to `target`.
fn convert(array: &ArrayRef, conv: &Conv, target: &DataType) -> Result<ArrayRef> {
    match conv {
        Conv::Same if array.data_type() == target => Ok(Arc::clone(array)),
        Conv::Same | Conv::Cast if !is_nested(target) => cast_leaf(array, target),
        Conv::Same | Conv::Cast => convert(array, &same_conv(target), target),
        Conv::Struct(children) => {
            let DataType::Struct(fields) = target else {
                return Err(internal("struct conversion to a non-struct type"));
            };
            let sa = array
                .as_struct_opt()
                .ok_or_else(|| internal("struct conversion of a non-struct array"))?;
            let mut out = Vec::with_capacity(fields.len());
            for (field, child) in fields.iter().zip(children) {
                out.push(match child {
                    None => new_null_array(field.data_type(), sa.len()),
                    Some((pos, c)) => convert(sa.column(*pos), c, field.data_type())?,
                });
            }
            let s = StructArray::try_new_with_length(
                fields.clone(),
                out,
                sa.nulls().cloned(),
                sa.len(),
            )
            .map_err(arrow_internal)?;
            Ok(Arc::new(s))
        }
        Conv::List(elem) => match target {
            DataType::List(field) => {
                let la = array
                    .as_list_opt::<i32>()
                    .ok_or_else(|| internal("list conversion of a non-list array"))?;
                let values = convert(la.values(), elem, field.data_type())?;
                Ok(Arc::new(
                    ListArray::try_new(
                        Arc::clone(field),
                        la.offsets().clone(),
                        values,
                        la.nulls().cloned(),
                    )
                    .map_err(arrow_internal)?,
                ))
            }
            DataType::LargeList(field) => {
                let la = array
                    .as_list_opt::<i64>()
                    .ok_or_else(|| internal("list conversion of a non-list array"))?;
                let values = convert(la.values(), elem, field.data_type())?;
                Ok(Arc::new(
                    LargeListArray::try_new(
                        Arc::clone(field),
                        la.offsets().clone(),
                        values,
                        la.nulls().cloned(),
                    )
                    .map_err(arrow_internal)?,
                ))
            }
            _ => Err(internal("list conversion to a non-list type")),
        },
        Conv::Map(k, v) => {
            let DataType::Map(entries, sorted) = target else {
                return Err(internal("map conversion to a non-map type"));
            };
            let DataType::Struct(kv) = entries.data_type() else {
                return Err(internal("map entries are not a struct"));
            };
            let ma = array
                .as_map_opt()
                .ok_or_else(|| internal("map conversion of a non-map array"))?;
            let (Some(kf), Some(vf)) = (kv.first(), kv.get(1)) else {
                return Err(internal("map entries need a key and a value"));
            };
            let keys = convert(ma.keys(), k, kf.data_type())?;
            let values = convert(ma.values(), v, vf.data_type())?;
            let entries_array = StructArray::try_new(kv.clone(), vec![keys, values], None)
                .map_err(arrow_internal)?;
            Ok(Arc::new(
                MapArray::try_new(
                    Arc::clone(entries),
                    ma.offsets().clone(),
                    entries_array,
                    ma.nulls().cloned(),
                    *sorted,
                )
                .map_err(arrow_internal)?,
            ))
        }
    }
}

fn is_nested(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Struct(_) | DataType::List(_) | DataType::LargeList(_) | DataType::Map(..)
    )
}

/// The identity conversion of a nested type (rebuilds it with the target's
/// field nullability and names).
fn same_conv(dt: &DataType) -> Conv {
    match dt {
        DataType::Struct(fs) => Conv::Struct(
            fs.iter()
                .enumerate()
                .map(|(i, f)| Some((i, same_conv(f.data_type()))))
                .collect(),
        ),
        DataType::List(f) | DataType::LargeList(f) => {
            Conv::List(Box::new(same_conv(f.data_type())))
        }
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(kv) if kv.len() == 2 => Conv::Map(
                Box::new(same_conv(kv[0].data_type())),
                Box::new(same_conv(kv[1].data_type())),
            ),
            _ => Conv::Same,
        },
        _ => Conv::Same,
    }
}

fn pow10(scale: i8) -> i128 {
    u32::try_from(scale).map_or(1, |s| 10_i128.pow(s.min(38)))
}

/// A leaf conversion (see the module docs for the legal ones).
fn cast_leaf(array: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    use DataType as D;
    let from = array.data_type();
    if from == target {
        return Ok(Arc::clone(array));
    }
    let out: ArrayRef = match (from, target) {
        (D::Utf8 | D::LargeUtf8 | D::Utf8View, _) => {
            let values: Vec<Option<&str>> = match from {
                D::Utf8 => array.as_string::<i32>().iter().collect(),
                D::LargeUtf8 => array.as_string::<i64>().iter().collect(),
                _ => array.as_string_view().iter().collect(),
            };
            match target {
                D::Utf8 => Arc::new(StringArray::from(values)),
                D::LargeUtf8 => Arc::new(LargeStringArray::from(values)),
                D::Utf8View => Arc::new(StringViewArray::from(values)),
                _ => return Err(internal(format!("cannot cast {from} to {target}"))),
            }
        }
        (D::Binary | D::LargeBinary | D::BinaryView, _) => {
            let values: Vec<Option<&[u8]>> = match from {
                D::Binary => array.as_binary::<i32>().iter().collect(),
                D::LargeBinary => array.as_binary::<i64>().iter().collect(),
                _ => array.as_binary_view().iter().collect(),
            };
            match target {
                D::Binary => Arc::new(BinaryArray::from(values)),
                D::LargeBinary => Arc::new(LargeBinaryArray::from(values)),
                D::BinaryView => Arc::new(BinaryViewArray::from(values)),
                _ => return Err(internal(format!("cannot cast {from} to {target}"))),
            }
        }
        (D::Timestamp(unit, _), D::Timestamp(_, tz)) => match unit {
            TimeUnit::Second => Arc::new(
                array
                    .as_primitive::<TimestampSecondType>()
                    .clone()
                    .with_timezone_opt(tz.clone()),
            ),
            TimeUnit::Millisecond => Arc::new(
                array
                    .as_primitive::<TimestampMillisecondType>()
                    .clone()
                    .with_timezone_opt(tz.clone()),
            ),
            TimeUnit::Microsecond => Arc::new(
                array
                    .as_primitive::<TimestampMicrosecondType>()
                    .clone()
                    .with_timezone_opt(tz.clone()),
            ),
            TimeUnit::Nanosecond => Arc::new(
                array
                    .as_primitive::<TimestampNanosecondType>()
                    .clone()
                    .with_timezone_opt(tz.clone()),
            ),
        },
        (D::Date32, D::Timestamp(TimeUnit::Microsecond, None)) => Arc::new(
            array
                .as_primitive::<Date32Type>()
                .try_unary::<_, TimestampMicrosecondType, KrishivError>(|d| {
                    i64::from(d).checked_mul(86_400_000_000).ok_or_else(|| {
                        CastOverflow {
                            value: d.to_string(),
                            source_type: "\"DATE\"".to_owned(),
                            target_type: "\"TIMESTAMP_NTZ\"".to_owned(),
                        }
                        .into()
                    })
                })?,
        ),
        (D::Float32, D::Float64) => Arc::new(
            array
                .as_primitive::<Float32Type>()
                .unary::<_, arrow_array::types::Float64Type>(f64::from),
        ),
        (D::Decimal128(_, s1), D::Decimal128(p2, s2)) => {
            let m = pow10(*s2 - *s1);
            // In range for every value of the file's precision; a corrupt
            // file can hold larger ones, which must not overflow.
            Arc::new(
                array
                    .as_primitive::<Decimal128Type>()
                    .try_unary::<_, Decimal128Type, KrishivError>(|v| {
                        v.checked_mul(m).ok_or_else(|| {
                            CastOverflow {
                                value: v.to_string(),
                                source_type: format!(
                                    "\"{}\"",
                                    spark_name(from).to_ascii_uppercase()
                                ),
                                target_type: format!(
                                    "\"{}\"",
                                    spark_name(target).to_ascii_uppercase()
                                ),
                            }
                            .into()
                        })
                    })?
                    .with_precision_and_scale(*p2, *s2)
                    .map_err(arrow_internal)?,
            )
        }
        _ => cast_int(array, target)?,
    };
    Ok(out)
}

/// Integer sources: widening to a wider integer, `double` or decimal, and
/// checked narrowing.
fn cast_int(array: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    let values: Vec<Option<i64>> = match array.data_type() {
        DataType::Int8 => array
            .as_primitive::<Int8Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Int16 => array
            .as_primitive::<Int16Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Int32 => array
            .as_primitive::<Int32Type>()
            .iter()
            .map(|v| v.map(i64::from))
            .collect(),
        DataType::Int64 => array.as_primitive::<Int64Type>().iter().collect(),
        other => return Err(internal(format!("cannot cast {other} to {target}"))),
    };
    let source = spark_name(array.data_type());
    let overflow = |v: i64| -> KrishivError {
        CastOverflow {
            value: v.to_string(),
            source_type: format!("\"{}\"", source.to_ascii_uppercase()),
            target_type: format!("\"{}\"", spark_name(target).to_ascii_uppercase()),
        }
        .into()
    };
    macro_rules! narrow {
        ($t:ty, $arrow:ty) => {{
            let mut out: Vec<Option<$t>> = Vec::with_capacity(values.len());
            for v in &values {
                out.push(match v {
                    None => None,
                    Some(v) => Some(<$t>::try_from(*v).map_err(|_| overflow(*v))?),
                });
            }
            Arc::new(PrimitiveArray::<$arrow>::from(out)) as ArrayRef
        }};
    }
    Ok(match target {
        DataType::Int8 => narrow!(i8, Int8Type),
        DataType::Int16 => narrow!(i16, Int16Type),
        DataType::Int32 => narrow!(i32, Int32Type),
        DataType::Int64 => Arc::new(PrimitiveArray::<Int64Type>::from(values)),
        DataType::Float64 => {
            #[expect(
                clippy::cast_precision_loss,
                reason = "only int8/16/32 sources are legal, exact in f64"
            )]
            let out: Vec<Option<f64>> = values.iter().map(|v| v.map(|v| v as f64)).collect();
            Arc::new(PrimitiveArray::<arrow_array::types::Float64Type>::from(out))
        }
        DataType::Decimal128(p, s) => {
            let m = pow10(*s);
            let out: Vec<Option<i128>> = values
                .iter()
                .map(|v| v.map(|v| i128::from(v) * m))
                .collect();
            Arc::new(
                PrimitiveArray::<Decimal128Type>::from(out)
                    .with_precision_and_scale(*p, *s)
                    .map_err(arrow_internal)?,
            )
        }
        other => return Err(internal(format!("cannot cast an integer to {other}"))),
    })
}
