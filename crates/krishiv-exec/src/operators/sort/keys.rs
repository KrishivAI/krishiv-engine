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

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{ArrayRef, RecordBatch};
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
    /// as Spark does yet (nested types, `Float16`, day-time intervals);
    /// `INTERNAL_ERROR` for no keys or an out-of-range column.
    pub(crate) fn new(schema: &Schema, keys: &[SortKey]) -> Result<Arc<Self>> {
        if keys.is_empty() {
            return Err(internal("a sort needs at least one key".to_owned()));
        }
        let fields = keys
            .iter()
            .map(|k| {
                let field = schema.fields().get(k.column).ok_or_else(|| {
                    internal(format!("sort key column {} is out of range", k.column))
                })?;
                if !orderable(field.data_type()) {
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

/// Whether the sort orders `t` as Spark does. Refused: floats it does not
/// canonicalize (`Float16`, in dictionaries, in nested types: nested keys
/// are not implemented) and day-time intervals (not durations).
fn orderable(t: &DataType) -> bool {
    match t {
        DataType::Dictionary(_, values) => {
            !matches!(**values, DataType::Float32 | DataType::Float64) && orderable(values)
        }
        DataType::Float16
        | DataType::Interval(IntervalUnit::DayTime | IntervalUnit::MonthDayNano)
        | DataType::List(_)
        | DataType::LargeList(_)
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::FixedSizeList(..)
        | DataType::Struct(_)
        | DataType::Map(..)
        | DataType::Union(..)
        | DataType::RunEndEncoded(..) => false,
        _ => RowConverter::supports_fields(&[SortField::new(t.clone())]),
    }
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
        _ => Arc::clone(column),
    }
}
