// SPDX-License-Identifier: Apache-2.0
//! Sort keys and their normalized, binary-comparable row encoding.
//!
//! arrow-row's `RowConverter` encodes the key columns of a batch into byte
//! strings whose `memcmp` order is the requested order, nulls and direction
//! included. Its float encoding is IEEE `totalOrder` (`-NaN < -inf < … <
//! -0.0 < 0.0 < … < NaN`), which differs from Spark's
//! `SQLOrderingUtil.compareDoubles` (every NaN equal and greater than every
//! other value, `-0.0 = 0.0`). Float key columns are therefore canonicalized
//! before encoding (every NaN to the positive quiet NaN, `-0.0` to `0.0`);
//! the output rows keep their original values.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{ArrayRef, GenericListArray, RecordBatch, StructArray};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::{DataType, IntervalUnit, Schema, SortOptions};
use krishiv_common::error::KrishivUnsupportedFeature;

use crate::error::{Result, internal};

/// One `ORDER BY` key: a column of the input and its direction.
///
/// Spark's defaults are `NULLS FIRST` for `ASC` and `NULLS LAST` for
/// `DESC` ([`asc`](Self::asc), [`desc`](Self::desc)). The planner projects
/// computed keys into columns first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortKey {
    /// The key column's index in the input.
    pub column: usize,
    /// `DESC`.
    pub descending: bool,
    /// `NULLS FIRST`.
    pub nulls_first: bool,
}

impl SortKey {
    /// `column ASC` (`NULLS FIRST`, Spark's default).
    pub fn asc(column: usize) -> Self {
        Self {
            column,
            descending: false,
            nulls_first: true,
        }
    }

    /// `column DESC` (`NULLS LAST`, Spark's default).
    pub fn desc(column: usize) -> Self {
        Self {
            column,
            descending: true,
            nulls_first: false,
        }
    }

    /// With an explicit `NULLS FIRST` (`true`) or `NULLS LAST`.
    #[must_use]
    pub fn nulls_first(mut self, nulls_first: bool) -> Self {
        self.nulls_first = nulls_first;
        self
    }
}

/// Encodes the key columns of batches into comparable rows.
#[derive(Debug)]
pub(crate) struct KeyEncoder {
    converter: RowConverter,
    /// Key column indices, in key order.
    columns: Vec<usize>,
}

impl KeyEncoder {
    /// An encoder for `keys` over batches of `schema`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a key type the sort cannot order
    /// as Spark does yet (maps, unions, `Float16`, day-time intervals, a struct
    /// or array with a non-default null placement);
    /// `INTERNAL_ERROR` for no keys or an out-of-range column.
    pub(crate) fn new(schema: &Schema, keys: &[SortKey]) -> Result<Arc<Self>> {
        if keys.is_empty() {
            // No keys: every row is equal (all rows in one stream, ties in
            // input order). One NULL per row stands in for the key.
            let converter = RowConverter::new(vec![SortField::new(DataType::Null)])
                .map_err(|e| internal(e.to_string()))?;
            return Ok(Arc::new(Self {
                converter,
                columns: Vec::new(),
            }));
        }
        let fields = keys
            .iter()
            .map(|k| {
                let field = schema.fields().get(k.column).ok_or_else(|| {
                    internal(format!("sort key column {} is out of range", k.column))
                })?;
                if !orderable(field.data_type())
                    || !nested_null_order_is_spark(field.data_type(), k)
                {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!("ORDER BY on a {} key", field.data_type()),
                    }
                    .into());
                }
                let options = SortOptions {
                    descending: k.descending,
                    nulls_first: k.nulls_first,
                };
                Ok(SortField::new_with_options(
                    field.data_type().clone(),
                    options,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let converter = RowConverter::new(fields).map_err(|e| internal(e.to_string()))?;
        Ok(Arc::new(Self {
            converter,
            columns: keys.iter().map(|k| k.column).collect(),
        }))
    }

    /// The rows of `batch`'s keys, in row order.
    pub(crate) fn encode(&self, batch: &RecordBatch) -> Result<Rows> {
        if self.columns.is_empty() {
            let nulls = arrow_array::new_null_array(&DataType::Null, batch.num_rows());
            return self
                .converter
                .convert_columns(&[nulls])
                .map_err(|e| internal(e.to_string()));
        }
        let columns = self
            .columns
            .iter()
            .map(|&i| {
                let column = batch
                    .columns()
                    .get(i)
                    .ok_or_else(|| internal(format!("sort key column {i} is out of range")))?;
                Ok(canonical(column))
            })
            .collect::<Result<Vec<_>>>()?;
        self.converter
            .convert_columns(&columns)
            .map_err(|e| internal(e.to_string()))
    }

    /// An empty row buffer for about `rows` rows of `bytes` key bytes.
    pub(crate) fn empty_rows(&self, rows: usize, bytes: usize) -> Rows {
        self.converter.empty_rows(rows, bytes)
    }
}

/// Whether the sort orders `t` as Spark does. Refused: `Float16`, maps, unions, floats it does not
/// canonicalize (in dictionaries) and day-time intervals (not durations).
fn orderable(t: &DataType) -> bool {
    match t {
        DataType::Dictionary(_, values) => {
            !matches!(**values, DataType::Float32 | DataType::Float64) && orderable(values)
        }
        DataType::Float16
        | DataType::Interval(IntervalUnit::DayTime | IntervalUnit::MonthDayNano)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(..)
        | DataType::Map(..)
        | DataType::Union(..)
        | DataType::RunEndEncoded(..) => false,
        DataType::Struct(fields) => fields.iter().all(|f| orderable(f.data_type())),
        DataType::List(f) | DataType::LargeList(f) => orderable(f.data_type()),
        _ => RowConverter::supports_fields(&[SortField::new(t.clone())]),
    }
}

/// Whether `key`'s null placement orders a nested `t` as Spark does. Spark
/// compares a struct's fields and an array's elements with the key's own
/// direction (a null field is the smallest, then reversed for `DESC`), and
/// `NULLS FIRST/LAST` moves only a null value of the key itself; the row
/// encoding applies the placement to the nested nulls as well, so it agrees
/// only for the default placement (`NULLS FIRST` for `ASC`, `LAST` for `DESC`).
fn nested_null_order_is_spark(t: &DataType, key: &SortKey) -> bool {
    !matches!(
        t,
        DataType::Struct(_) | DataType::List(_) | DataType::LargeList(_)
    ) || key.nulls_first != key.descending
}

/// Spark's float order as `totalOrder`: one positive NaN (`NAN`'s sign is
/// unspecified, `abs` clears it), no `-0.0` (`v + 0.0` makes it `0.0`).
fn canonical(column: &ArrayRef) -> ArrayRef {
    match column.data_type() {
        DataType::Float64 => Arc::new(
            column
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|v| if v.is_nan() { f64::NAN.abs() } else { v + 0.0 }),
        ),
        DataType::Float32 => Arc::new(
            column
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|v| if v.is_nan() { f32::NAN.abs() } else { v + 0.0 }),
        ),
        DataType::Struct(fields) if fields.iter().any(|f| has_float(f.data_type())) => {
            let array = column.as_struct();
            let children = array.columns().iter().map(canonical).collect();
            Arc::new(StructArray::new(
                fields.clone(),
                children,
                array.nulls().cloned(),
            ))
        }
        DataType::List(field) if has_float(field.data_type()) => {
            let array = column.as_list::<i32>();
            Arc::new(GenericListArray::<i32>::new(
                Arc::clone(field),
                array.offsets().clone(),
                canonical(array.values()),
                array.nulls().cloned(),
            ))
        }
        DataType::LargeList(field) if has_float(field.data_type()) => {
            let array = column.as_list::<i64>();
            Arc::new(GenericListArray::<i64>::new(
                Arc::clone(field),
                array.offsets().clone(),
                canonical(array.values()),
                array.nulls().cloned(),
            ))
        }
        _ => Arc::clone(column),
    }
}

/// Whether a float is anywhere inside `t` (struct fields, list elements).
fn has_float(t: &DataType) -> bool {
    match t {
        DataType::Float32 | DataType::Float64 => true,
        DataType::Struct(fields) => fields.iter().any(|f| has_float(f.data_type())),
        DataType::List(f) | DataType::LargeList(f) => has_float(f.data_type()),
        _ => false,
    }
}
