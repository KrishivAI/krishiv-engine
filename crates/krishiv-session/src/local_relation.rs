// SPDX-License-Identifier: Apache-2.0
//! [`Session::local_relation`]: Arrow IPC data (Spark Connect's `LocalRelation`) as a
//! [`LocalRelation`] carrier over the session's relation store (design P3-T012a part 2a, section 1).
//!
//! A user schema is applied **eagerly**, as Spark does (`SparkConnectPlanner.buildLocalRelationFromRows`
//! evaluates `Dataset.toDF(names).to(schema)` while the plan is built): the columns are renamed by
//! position, cast through the session's own executor when their types differ, and checked against the
//! schema's non-nullable fields; only the result is stored. The cast and nullability rules and their
//! error classes follow the current style and are pending Spark goldens.

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, LargeListArray, ListArray, MapArray, RecordBatch,
    RecordBatchOptions, StructArray,
};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use krishiv_common::error::UnableToInferSchema;
use krishiv_common::error::{KrishivError, NotNullAssertViolation, NumColumnsMismatch};
use krishiv_memory::RelationStore;
use krishiv_plan::{Expr, LocalRelation, LogicalPlan, PlanNode, ReadSource};
use krishiv_types::batch::Batch;
use krishiv_types::{DataType, StructType};

use crate::carrier::{canonical_schema, recarried, viewed};
use crate::dataframe::DataFrame;
use crate::session::{Session, internal};

impl Session {
    /// The relation of `data` (an Arrow IPC stream with its own schema) as a plan carrier, with
    /// `user_schema` applied eagerly when given (Spark Connect's `LocalRelation`):
    ///
    /// - no user schema: the data as it is;
    /// - a user schema: the columns are renamed by position and cast to its types through this
    ///   session's executor (its configuration and memory pool) unless the types already match; a
    ///   non-nullable field holding a null is an error. Only the result is stored;
    /// - no data: an empty relation of the user schema.
    ///
    /// The batches live once in the session's relation store; the carrier holds a lease.
    ///
    /// # Errors
    /// `UNABLE_TO_INFER_SCHEMA` for neither data nor schema; `NUM_COLUMNS_MISMATCH` when the
    /// schema's column count differs from the data's; `NOT_NULL_ASSERT_VIOLATION` for a null in a
    /// non-nullable field; the cast errors of the executor; `INTERNAL_ERROR` for undecodable data.
    pub async fn local_relation(
        &self,
        data: Option<&[u8]>,
        user_schema: Option<StructType>,
    ) -> Result<LocalRelation, KrishivError> {
        let store = &self.state.relations;
        let target = user_schema.as_ref().map(arrow_schema_of);
        let Some(data) = data else {
            let Some(target) = target else {
                return Err(UnableToInferSchema {
                    format: "LocalRelation".to_owned(),
                }
                .into());
            };
            return insert(store, canonical_schema(Arc::new(target)), Vec::new());
        };
        let (decoded, batches) = RelationStore::decode_ipc(data)?;
        let schema = canonical_schema(decoded);
        let records = batches
            .iter()
            .map(|b| recarried(b.data(), &schema))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(target) = target else {
            return insert(store, schema, records);
        };
        let target = canonical_schema(Arc::new(target));
        if schema.fields().len() != target.fields().len() {
            return Err(NumColumnsMismatch {
                operator: "LocalRelation".to_owned(),
                first_num_columns: target.fields().len().to_string(),
                invalid_ordinal_num: "second".to_owned(),
                invalid_num_columns: schema.fields().len().to_string(),
            }
            .into());
        }
        let (schema, records) = carried(&schema, records, &target)?;
        let renamed = renamed(&schema, &target);
        let same = renamed
            .fields()
            .iter()
            .zip(target.fields())
            .all(|(a, b)| same_shape(a.data_type(), b.data_type()));
        let converted = if same {
            records
        } else if records.is_empty() {
            // No batches: the conversion is still checked, as for a zero-row batch.
            let empty = RecordBatch::new_empty(Arc::clone(&renamed));
            self.cast_batches(&renamed, &target, vec![empty])
                .await?
                .into_iter()
                .filter(|b| b.num_rows() > 0)
                .collect()
        } else {
            self.cast_batches(&renamed, &target, records).await?
        };
        let conformed = converted
            .iter()
            .map(|b| conform(b, &target))
            .collect::<Result<Vec<_>, _>>()?;
        insert(store, target, conformed)
    }

    /// `records` (of `renamed`, the data's types under the target's names) cast to the types of
    /// `target` by a projection run through this session's executor over a temporary relation.
    async fn cast_batches(
        &self,
        renamed: &SchemaRef,
        target: &SchemaRef,
        records: Vec<RecordBatch>,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        let store = &self.state.relations;
        // The temporary entry lives while this function does. The plan names it by hash only, so
        // no cache of an analysis keeps it alive afterwards.
        let named = records
            .iter()
            .map(|b| {
                RecordBatch::try_new_with_options(
                    Arc::clone(renamed),
                    b.columns().to_vec(),
                    &RecordBatchOptions::new().with_row_count(Some(b.num_rows())),
                )
                .map_err(|e| internal(e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let temp = insert(store, Arc::clone(renamed), named)?;
        let local = LocalRelation::from_decoded(
            Arc::clone(temp.schema()),
            *temp.relation().hash.as_bytes(),
            temp.batches(),
            temp.rows(),
            temp.bytes(),
        );
        let mut exprs = Vec::with_capacity(target.fields().len());
        for (have, want) in renamed.fields().iter().zip(target.fields()) {
            let column = Expr::column([have.name().clone()]);
            // A UDT column is its storage type inside the engine: cast to that; `conform` puts the UDT's tag back.
            let to = DataType::from_arrow_field(want)?.without_udts();
            exprs.push(
                if same_shape(have.data_type(), want.data_type()) {
                    column
                } else {
                    column.cast(to)
                }
                .alias(want.name().clone()),
            );
        }
        let plan = LogicalPlan::new(PlanNode::Read(ReadSource::Local(local))).project(exprs);
        let batches = DataFrame::new(self.clone(), Arc::new(plan))
            .collect()
            .await?;
        batches
            .iter()
            .map(|b| recarried(b, &canonical_schema(Arc::clone(b.schema_ref()))))
            .collect()
    }
}

/// `records` (of `schema`) with the encoding-only carriers of string and binary columns moved to
/// the views the engine runs on, by an Arrow cast of the buffers: a logical STRING-to-STRING cast
/// would be folded away by the optimizer, so this is done here and explicitly. Only columns whose
/// target is the same logical kind are touched; the rest keep their cast path.
fn carried(
    schema: &SchemaRef,
    records: Vec<RecordBatch>,
    target: &SchemaRef,
) -> Result<(SchemaRef, Vec<RecordBatch>), KrishivError> {
    let wanted: Vec<ArrowType> = schema
        .fields()
        .iter()
        .zip(target.fields())
        .map(|(have, want)| {
            let v = viewed(have.data_type());
            if &v != have.data_type() && v == viewed(want.data_type()) {
                v
            } else {
                have.data_type().clone()
            }
        })
        .collect();
    if wanted
        .iter()
        .zip(schema.fields())
        .all(|(w, f)| w == f.data_type())
    {
        return Ok((Arc::clone(schema), records));
    }
    let out = Arc::new(Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .zip(&wanted)
            .map(|(f, w)| f.as_ref().clone().with_data_type(w.clone()))
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    ));
    let batches = records
        .iter()
        .map(|b| {
            let columns = b
                .columns()
                .iter()
                .zip(&wanted)
                .map(|(c, w)| {
                    if c.data_type() == w {
                        Ok(Arc::clone(c))
                    } else {
                        arrow_cast::cast(c, w).map_err(|e| internal(e.to_string()))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            RecordBatch::try_new_with_options(
                Arc::clone(&out),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(b.num_rows())),
            )
            .map_err(|e| internal(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((out, batches))
}

/// The Arrow schema of a user-declared struct type.
fn arrow_schema_of(schema: &StructType) -> Schema {
    Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| f.data_type().to_arrow_field(f.name(), f.nullable()))
            .collect::<Vec<_>>(),
    )
}

/// `schema`'s types under `names`' field names (Spark's `toDF(names)`).
fn renamed(schema: &SchemaRef, names: &SchemaRef) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        schema
            .fields()
            .iter()
            .zip(names.fields())
            .map(|(f, n)| f.as_ref().clone().with_name(n.name().clone()))
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    ))
}

/// Stores `records` (of `schema`) and returns the carrier over the entry.
fn insert(
    store: &Arc<RelationStore>,
    schema: SchemaRef,
    records: Vec<RecordBatch>,
) -> Result<LocalRelation, KrishivError> {
    let rows: u64 = records.iter().map(|b| b.num_rows() as u64).sum();
    let bytes: u64 = records
        .iter()
        .map(|b| b.get_array_memory_size() as u64)
        .sum();
    let batches: Vec<Batch> = records.into_iter().map(Batch::new).collect();
    let lease = store.insert(&schema, &batches)?;
    Ok(LocalRelation::from_lease(schema, &lease, rows, bytes))
}

/// Whether `a` and `b` are the same type once nested field names are ignored (Spark renames
/// nested fields `col_i` before comparing); nullability is checked against the data.
fn same_shape(a: &ArrowType, b: &ArrowType) -> bool {
    match (a, b) {
        (ArrowType::List(x), ArrowType::List(y))
        | (ArrowType::LargeList(x), ArrowType::LargeList(y)) => {
            same_shape(x.data_type(), y.data_type())
        }
        (ArrowType::FixedSizeList(x, n), ArrowType::FixedSizeList(y, m)) => {
            n == m && same_shape(x.data_type(), y.data_type())
        }
        (ArrowType::Map(x, s), ArrowType::Map(y, t)) => {
            s == t && same_shape(x.data_type(), y.data_type())
        }
        (ArrowType::Struct(x), ArrowType::Struct(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|(p, q)| same_shape(p.data_type(), q.data_type()))
        }
        _ => a == b,
    }
}

/// `batch` relabelled to `target` (names, nested field names, nullability flags) over the same
/// buffers, after checking that no non-nullable field, nested ones included, holds a null.
fn conform(batch: &RecordBatch, target: &SchemaRef) -> Result<RecordBatch, KrishivError> {
    let columns = batch
        .columns()
        .iter()
        .zip(target.fields())
        .map(|(c, f)| {
            check_non_null(c, f, f.name())?;
            relabel(c, f.data_type())
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new_with_options(
        Arc::clone(target),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| internal(e.to_string()))
}

/// The error for a null in the non-nullable field at `path`.
fn null_in(path: &str) -> KrishivError {
    NotNullAssertViolation {
        walked_type_path: path.to_owned(),
    }
    .into()
}

/// Fails when `array` holds a null where `field` (and the fields under it) is non-nullable. A struct
/// child's nulls count only where the struct itself is valid; a list's elements are checked over
/// the whole values array (a null parent owns an empty range in the arrays Arrow writes).
///
/// Golden-pending (P3-T012a): a writer that leaves data under a null list slot, or a null grandchild
/// under a null struct two levels up, can raise a violation Spark would not; carrying the parent
/// validity mask down is settled with the nullability goldens.
fn check_non_null(array: &ArrayRef, field: &Field, path: &str) -> Result<(), KrishivError> {
    if !field.is_nullable() && array.logical_null_count() > 0 {
        return Err(null_in(path));
    }
    match (array.data_type(), field.data_type()) {
        (ArrowType::Struct(_), ArrowType::Struct(fields)) => {
            let s = array.as_struct();
            let parent = array.logical_nulls();
            for (child, f) in s.columns().iter().zip(fields.iter()) {
                let at = format!("{path}.{}", f.name());
                if !f.is_nullable()
                    && let Some(nulls) = child.logical_nulls()
                    && (0..child.len())
                        .any(|i| nulls.is_null(i) && parent.as_ref().is_none_or(|p| p.is_valid(i)))
                {
                    return Err(null_in(&at));
                }
                check_non_null_below(child, f, &at)?;
            }
            Ok(())
        }
        (ArrowType::List(_), ArrowType::List(f)) => check_non_null(
            array.as_list::<i32>().values(),
            f,
            &format!("{path}.element"),
        ),
        (ArrowType::LargeList(_), ArrowType::LargeList(f)) => check_non_null(
            array.as_list::<i64>().values(),
            f,
            &format!("{path}.element"),
        ),
        (ArrowType::FixedSizeList(..), ArrowType::FixedSizeList(f, _)) => check_non_null(
            array.as_fixed_size_list().values(),
            f,
            &format!("{path}.element"),
        ),
        (ArrowType::Map(..), ArrowType::Map(f, _)) => {
            let entries: ArrayRef = Arc::new(array.as_map().entries().clone());
            check_non_null(&entries, f, path)
        }
        _ => Ok(()),
    }
}

/// [`check_non_null`] for the fields under `field`, whose own nullability the caller checked.
fn check_non_null_below(array: &ArrayRef, field: &Field, path: &str) -> Result<(), KrishivError> {
    let lenient = Field::new(field.name(), field.data_type().clone(), true);
    check_non_null(array, &lenient, path)
}

/// `array` under the nested field names and nullability of `to` (same shape; the buffers are
/// shared).
fn relabel(array: &ArrayRef, to: &ArrowType) -> Result<ArrayRef, KrishivError> {
    if array.data_type() == to {
        return Ok(Arc::clone(array));
    }
    let wrong = || {
        internal(format!(
            "a local relation column of {} for {to}",
            array.data_type()
        ))
    };
    let relabelled: ArrayRef = match to {
        ArrowType::Struct(fields) => {
            let s = array.as_struct_opt().ok_or_else(wrong)?;
            let children = s
                .columns()
                .iter()
                .zip(fields.iter())
                .map(|(c, f)| relabel(c, f.data_type()))
                .collect::<Result<Vec<_>, _>>()?;
            Arc::new(
                StructArray::try_new(fields.clone(), children, s.nulls().cloned())
                    .map_err(|e| internal(e.to_string()))?,
            )
        }
        ArrowType::List(f) => {
            let l = array.as_list_opt::<i32>().ok_or_else(wrong)?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    relabel(l.values(), f.data_type())?,
                    l.nulls().cloned(),
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        ArrowType::LargeList(f) => {
            let l = array.as_list_opt::<i64>().ok_or_else(wrong)?;
            Arc::new(
                LargeListArray::try_new(
                    Arc::clone(f),
                    l.offsets().clone(),
                    relabel(l.values(), f.data_type())?,
                    l.nulls().cloned(),
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        ArrowType::FixedSizeList(f, size) => {
            let l = array.as_fixed_size_list_opt().ok_or_else(wrong)?;
            Arc::new(
                FixedSizeListArray::try_new(
                    Arc::clone(f),
                    *size,
                    relabel(l.values(), f.data_type())?,
                    l.nulls().cloned(),
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        ArrowType::Map(f, sorted) => {
            let m = array.as_map_opt().ok_or_else(wrong)?;
            let entries = relabel(&(Arc::new(m.entries().clone()) as ArrayRef), f.data_type())?;
            let entries = entries.as_struct_opt().ok_or_else(wrong)?.clone();
            Arc::new(
                MapArray::try_new(
                    Arc::clone(f),
                    m.offsets().clone(),
                    entries,
                    m.nulls().cloned(),
                    *sorted,
                )
                .map_err(|e| internal(e.to_string()))?,
            )
        }
        _ => return Err(wrong()),
    };
    Ok(relabelled)
}

#[cfg(test)]
mod tests;
