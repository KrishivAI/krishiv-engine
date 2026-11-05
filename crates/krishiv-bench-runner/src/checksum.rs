// SPDX-License-Identifier: Apache-2.0
//! Order-independent checksum of a query result, for `RUN_CHECKED`.
//!
//! The structure is that of the harness's `bench/src/krishiv_bench/result_check.py`, so its comparison applies
//! unchanged: `{"kinds": [...], "values": [...]}`. First the row count (`rows`); then, per column, the number of
//! non-NULL values (`count`) and either, for a floating-point column, their sum (`float`, compared with a relative
//! tolerance because parallel sums differ in the last bits) or, for any other type, the sum of a hash of each value
//! (`exact`, as a decimal string); then, when the result has a column that is not floating point, the sum of a hash of
//! each row's non-float values (`exact`: ties the columns to the same row) and, per float column, the sum of the
//! value times a weight in (0, 1] drawn from that row hash (`float`: a measure moved to another row changes it).
//!
//! Row order does not matter. The hashes are `std`'s SipHash with fixed keys over Arrow's row encoding, so they are
//! the same in every run of one build: a run is compared with the verified run of the same binary, never across
//! engines. Floats are printed as JSON numbers, with Python's `NaN`/`Infinity` tokens for the non-finite ones.

use std::hash::{DefaultHasher, Hash, Hasher};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_row::{RowConverter, SortField};
use arrow_schema::DataType;

use crate::RunnerError;

/// Row hashes are reduced modulo this before they become a float's weight.
const WEIGHT_MODULUS: u64 = 1_000_003;

fn is_float(t: &DataType) -> bool {
    matches!(t, DataType::Float32 | DataType::Float64)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    bytes.hash(&mut h);
    h.finish()
}

fn encode_error(e: &arrow_schema::ArrowError) -> RunnerError {
    RunnerError::Config(format!("result checksum: {e}"))
}

/// The weight in (0, 1] of a row from its hash.
#[expect(
    clippy::cast_precision_loss,
    reason = "a weight below 2^20 is exact in an f64"
)]
fn weight(row_hash: u64) -> f64 {
    ((row_hash % WEIGHT_MODULUS) + 1) as f64 / WEIGHT_MODULUS as f64
}

/// A floating-point column's values as `f64`, NULLs as `None`.
fn floats(a: &ArrayRef) -> Vec<Option<f64>> {
    match a.data_type() {
        DataType::Float32 => a
            .as_primitive::<Float32Type>()
            .iter()
            .map(|v| v.map(f64::from))
            .collect(),
        _ => a.as_primitive::<Float64Type>().iter().collect(),
    }
}

/// What is accumulated per column.
struct Column {
    float: bool,
    /// Encodes this column alone (non-float columns).
    converter: Option<RowConverter>,
    count: u64,
    exact: u128,
    sum: f64,
    weighted: f64,
}

/// The checksum of a result: kinds and the values as JSON tokens.
#[derive(Debug, PartialEq)]
pub(crate) struct Checksum {
    kinds: Vec<&'static str>,
    values: Vec<String>,
}

impl Checksum {
    /// One line of JSON: `{"kinds":[...],"values":[...],"source":"run"}`.
    #[must_use]
    pub(crate) fn to_json(&self) -> String {
        let kinds: Vec<String> = self.kinds.iter().map(|k| format!("\"{k}\"")).collect();
        format!(
            "{{\"kinds\":[{}],\"values\":[{}],\"source\":\"run\"}}",
            kinds.join(","),
            self.values.join(",")
        )
    }
}

fn json_float(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_owned()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        format!("{v:?}")
    }
}

/// The checksum of `batches`.
///
/// # Errors
/// A column type Arrow's row encoding cannot take.
pub(crate) fn of(batches: &[RecordBatch]) -> Result<Checksum, RunnerError> {
    let Some(first) = batches.first() else {
        return Ok(Checksum {
            kinds: vec!["rows"],
            values: vec!["0".to_owned()],
        });
    };
    let types: Vec<DataType> = first
        .schema()
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    let mut columns = Vec::with_capacity(types.len());
    for t in &types {
        let float = is_float(t);
        let converter = if float {
            None
        } else {
            Some(RowConverter::new(vec![SortField::new(t.clone())]).map_err(|e| encode_error(&e))?)
        };
        columns.push(Column {
            float,
            converter,
            count: 0,
            exact: 0,
            sum: 0.0,
            weighted: 0.0,
        });
    }
    let other: Vec<usize> = (0..types.len())
        .filter(|i| columns.get(*i).is_some_and(|c| !c.float))
        .collect();
    let row_converter = if other.is_empty() {
        None
    } else {
        let fields = other
            .iter()
            .filter_map(|i| types.get(*i))
            .map(|t| SortField::new(t.clone()))
            .collect();
        Some(RowConverter::new(fields).map_err(|e| encode_error(&e))?)
    };
    let (mut rows, mut row_sum) = (0_u64, 0_u128);
    for batch in batches {
        rows += u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        let row_hashes: Vec<u64> = match &row_converter {
            Some(c) => {
                let cols: Vec<ArrayRef> = other
                    .iter()
                    .filter_map(|i| batch.columns().get(*i).cloned())
                    .collect();
                c.convert_columns(&cols)
                    .map_err(|e| encode_error(&e))?
                    .iter()
                    .map(|row| hash_bytes(row.as_ref()))
                    .collect()
            }
            None => Vec::new(),
        };
        for h in &row_hashes {
            row_sum = row_sum.wrapping_add(u128::from(*h));
        }
        for (state, array) in columns.iter_mut().zip(batch.columns()) {
            state.count += u64::try_from(array.len() - array.null_count()).unwrap_or(u64::MAX);
            if let Some(converter) = &state.converter {
                let encoded = converter
                    .convert_columns(&[ArrayRef::clone(array)])
                    .map_err(|e| encode_error(&e))?;
                for (k, row) in encoded.iter().enumerate() {
                    if array.is_valid(k) {
                        state.exact = state
                            .exact
                            .wrapping_add(u128::from(hash_bytes(row.as_ref())));
                    }
                }
            } else {
                for (k, v) in floats(array).into_iter().enumerate() {
                    if let Some(v) = v {
                        state.sum += v;
                        if let Some(h) = row_hashes.get(k) {
                            state.weighted += v * weight(*h);
                        }
                    }
                }
            }
        }
    }
    let mut kinds = vec!["rows"];
    let mut values = vec![rows.to_string()];
    for c in &columns {
        kinds.push("count");
        values.push(c.count.to_string());
        if c.float {
            kinds.push("float");
            values.push(if c.count == 0 {
                "null".to_owned()
            } else {
                json_float(c.sum)
            });
        } else {
            kinds.push("exact");
            values.push(format!("\"{}\"", c.exact));
        }
    }
    if row_converter.is_some() {
        kinds.push("exact");
        values.push(format!("\"{row_sum}\""));
        for c in columns.iter().filter(|c| c.float) {
            kinds.push("float");
            values.push(if c.count == 0 {
                "null".to_owned()
            } else {
                json_float(c.weighted)
            });
        }
    }
    Ok(Checksum { kinds, values })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int64Array, StringArray};
    use arrow_schema::{Field, Schema};

    use super::*;

    fn batch(keys: &[i64], names: &[Option<&str>], measures: &[Option<f64>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
            Field::new("m", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
                Arc::new(Float64Array::from(measures.to_vec())),
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}"))
    }

    fn json(b: &[RecordBatch]) -> String {
        of(b).unwrap_or_else(|e| unreachable!("{e}")).to_json()
    }

    #[test]
    fn rows_in_another_order_or_split_into_other_batches_agree() {
        let whole = batch(
            &[1, 2, 3],
            &[Some("a"), None, Some("c")],
            &[Some(1.5), Some(2.5), None],
        );
        let shuffled = batch(
            &[3, 1, 2],
            &[Some("c"), Some("a"), None],
            &[None, Some(1.5), Some(2.5)],
        );
        let (head, tail) = (whole.slice(0, 1), whole.slice(1, 2));
        assert_eq!(json(&[whole]), json(&[shuffled]));
        assert_eq!(
            json(&[batch(
                &[1, 2, 3],
                &[Some("a"), None, Some("c")],
                &[Some(1.5), Some(2.5), None]
            )]),
            json(&[head, tail])
        );
    }

    #[test]
    fn a_changed_value_a_lost_row_and_a_moved_measure_differ() {
        let base = json(&[batch(
            &[1, 2],
            &[Some("a"), Some("b")],
            &[Some(10.0), Some(20.0)],
        )]);
        // a string changed
        assert_ne!(
            base,
            json(&[batch(
                &[1, 2],
                &[Some("a"), Some("x")],
                &[Some(10.0), Some(20.0)]
            )])
        );
        // a row lost
        assert_ne!(base, json(&[batch(&[1], &[Some("a")], &[Some(10.0)])]));
        // the measures on the wrong keys: the plain sums agree, the row-weighted ones do not
        assert_ne!(
            base,
            json(&[batch(
                &[1, 2],
                &[Some("a"), Some("b")],
                &[Some(20.0), Some(10.0)]
            )])
        );
    }

    #[test]
    fn the_json_has_the_harness_structure() {
        let got = json(&[batch(&[1], &[Some("a")], &[Some(f64::NAN)])]);
        assert!(
            got.starts_with("{\"kinds\":[\"rows\",\"count\",\"exact\","),
            "{got}"
        );
        assert!(got.contains("NaN"), "{got}");
        assert!(got.ends_with("\"source\":\"run\"}"), "{got}");
        assert_eq!(
            json(&[]),
            "{\"kinds\":[\"rows\"],\"values\":[0],\"source\":\"run\"}"
        );
    }
}
