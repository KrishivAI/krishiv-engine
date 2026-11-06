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
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, Schema};

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
///
/// # Errors
/// The array is not the floating-point type its column was declared with (a batch whose type differs from the schema).
fn floats(a: &ArrayRef, declared: &DataType) -> Result<Vec<Option<f64>>, RunnerError> {
    let mismatch = || {
        RunnerError::Config(format!(
            "result checksum: a batch column is {}, the schema says {declared}",
            a.data_type()
        ))
    };
    if a.data_type() != declared {
        return Err(mismatch());
    }
    match declared {
        DataType::Float32 => Ok(a
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(mismatch)?
            .iter()
            .map(|v| v.map(f64::from))
            .collect()),
        _ => Ok(a
            .as_primitive_opt::<Float64Type>()
            .ok_or_else(mismatch)?
            .iter()
            .collect()),
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

/// The checksum of `batches`. The column types are those of the first batch; `schema` (the result's) gives them when
/// there is none, so a result of no batches has the same structure as one of empty batches, and as the harness's
/// checksum of an empty table.
///
/// # Errors
/// A column type Arrow's row encoding cannot take, or a batch whose columns differ from the first one's.
pub(crate) fn of(batches: &[RecordBatch], schema: &Schema) -> Result<Checksum, RunnerError> {
    let fields = match batches.first() {
        Some(first) => first.schema(),
        None => Arc::new(schema.clone()),
    };
    let types: Vec<DataType> = fields
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
        if batch.num_columns() != types.len() {
            return Err(RunnerError::Config(format!(
                "result checksum: a batch has {} columns, the schema {}",
                batch.num_columns(),
                types.len()
            )));
        }
        for h in &row_hashes {
            row_sum = row_sum.saturating_add(u128::from(*h));
        }
        for ((state, array), declared) in columns.iter_mut().zip(batch.columns()).zip(&types) {
            state.count += u64::try_from(array.len() - array.null_count()).unwrap_or(u64::MAX);
            if let Some(converter) = &state.converter {
                let encoded = converter
                    .convert_columns(&[ArrayRef::clone(array)])
                    .map_err(|e| encode_error(&e))?;
                for (k, row) in encoded.iter().enumerate() {
                    if array.is_valid(k) {
                        state.exact = state
                            .exact
                            .saturating_add(u128::from(hash_bytes(row.as_ref())));
                    }
                }
            } else {
                for (k, v) in floats(array, declared)?.into_iter().enumerate() {
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

    use arrow_array::{Float32Array, Float64Array, Int64Array, StringArray};
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
        of(b, &Schema::empty())
            .unwrap_or_else(|e| unreachable!("{e}"))
            .to_json()
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

    fn sum_of(b: &[RecordBatch], schema: &Schema) -> String {
        of(b, schema)
            .unwrap_or_else(|e| unreachable!("{e}"))
            .to_json()
    }

    #[test]
    fn no_batches_and_an_empty_batch_have_the_same_structure() {
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
            Field::new("m", DataType::Float64, true),
        ]);
        let empty = batch(&[], &[], &[]);
        assert_eq!(sum_of(&[], &schema), sum_of(&[empty], &schema));
        let got = sum_of(&[], &schema);
        assert!(
            got.starts_with(
                "{\"kinds\":[\"rows\",\"count\",\"exact\",\"count\",\"exact\",\"count\",\"float\","
            ),
            "{got}"
        );
        assert!(
            got.contains("\"values\":[0,0,\"0\",0,\"0\",0,null,"),
            "{got}"
        );
    }

    #[test]
    fn a_float_only_result_sums_without_a_row_hash() {
        let schema = Arc::new(Schema::new(vec![Field::new("m", DataType::Float64, true)]));
        let b = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(vec![
                Some(1.5),
                None,
                Some(2.0),
            ]))],
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let got = sum_of(&[b], &schema);
        assert_eq!(
            got,
            "{\"kinds\":[\"rows\",\"count\",\"float\"],\"values\":[3,2,3.5],\"source\":\"run\"}"
        );
    }

    #[test]
    fn float32_and_all_null_float_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float32, true),
            Field::new("n", DataType::Float64, true),
            Field::new("k", DataType::Int64, false),
        ]));
        let b = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Float32Array::from(vec![Some(0.5_f32), Some(1.5)])),
                Arc::new(Float64Array::from(vec![None::<f64>, None])),
                Arc::new(Int64Array::from(vec![1, 2])),
            ],
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let got = sum_of(&[b], &schema);
        // rows 2; f: count 2, sum 2.0; n: count 0, null; k: count 2; then the row hash and the two weighted floats
        assert!(
            got.starts_with("{\"kinds\":[\"rows\",\"count\",\"float\",\"count\",\"float\",\"count\",\"exact\",\"exact\",\"float\",\"float\"],"),
            "{got}"
        );
        assert!(got.contains("\"values\":[2,2,2.0,0,null,2,"), "{got}");
        assert!(got.matches("null").count() == 2, "{got}");
    }

    #[test]
    fn nested_and_dictionary_columns_are_checksummed() {
        use arrow_array::types::Int32Type;
        use arrow_array::{DictionaryArray, ListArray, StructArray};

        let lists: [Option<Vec<Option<i32>>>; 3] =
            [Some(vec![Some(1), Some(2)]), None, Some(vec![])];
        let words = ["a", "b", "a"];
        let numbers = [1_i64, 2, 3];
        let make = |order: [usize; 3], words: [&str; 3]| {
            let list = ListArray::from_iter_primitive::<Int32Type, _, _>(
                order.iter().map(|i| lists[*i].clone()),
            );
            let dict: DictionaryArray<Int32Type> = order.iter().map(|i| words[*i]).collect();
            let st = StructArray::from(vec![(
                Arc::new(Field::new("x", DataType::Int64, false)),
                Arc::new(Int64Array::from(
                    order.iter().map(|i| numbers[*i]).collect::<Vec<_>>(),
                )) as ArrayRef,
            )]);
            let schema = Arc::new(Schema::new(vec![
                Field::new("l", list.data_type().clone(), true),
                Field::new("d", dict.data_type().clone(), false),
                Field::new("s", st.data_type().clone(), false),
            ]));
            let b = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(list), Arc::new(dict), Arc::new(st)],
            )
            .unwrap_or_else(|e| unreachable!("{e}"));
            (b, schema)
        };
        let (b, schema) = make([0, 1, 2], words);
        let a = sum_of(&[b], &schema);
        let (shuffled, _) = make([2, 0, 1], words);
        assert_eq!(a, sum_of(&[shuffled], &schema), "row order does not matter");
        assert!(a.contains("\"values\":[3,2,"), "{a}"); // 3 rows; the list column has 2 non-NULL values
        let (changed, _) = make([0, 1, 2], ["a", "b", "b"]);
        assert_ne!(a, sum_of(&[changed], &schema));
    }

    #[test]
    fn a_later_batch_of_another_type_is_an_error_not_a_panic() {
        let schema = Arc::new(Schema::new(vec![Field::new("m", DataType::Float64, true)]));
        let first = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        )
        .unwrap_or_else(|e| unreachable!("{e}"));
        let other = Arc::new(Schema::new(vec![Field::new("m", DataType::Float32, true)]));
        let second = RecordBatch::try_new(other, vec![Arc::new(Float32Array::from(vec![1.0_f32]))])
            .unwrap_or_else(|e| unreachable!("{e}"));
        assert!(of(&[first, second], &schema).is_err());
    }

    proptest::proptest! {
        #[test]
        fn shuffling_rows_and_resplitting_batches_keeps_the_checksum(
            rows in proptest::collection::vec(
                (proptest::prelude::any::<i64>(), proptest::option::of("[a-c]{0,3}"), proptest::option::of(-1000.0_f64..1000.0)),
                0..40,
            ),
            seed in proptest::prelude::any::<u64>(),
            cut in 0_usize..40,
        ) {
            let schema = Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("s", DataType::Utf8, true),
                Field::new("m", DataType::Float64, true),
            ]);
            let make = |rs: &[(i64, Option<String>, Option<f64>)]| {
                let names: Vec<Option<&str>> = rs.iter().map(|r| r.1.as_deref()).collect();
                batch(
                    &rs.iter().map(|r| r.0).collect::<Vec<_>>(),
                    &names,
                    &rs.iter().map(|r| r.2).collect::<Vec<_>>(),
                )
            };
            let whole = sum_of(&[make(&rows)], &schema);
            let mut shuffled = rows;
            // a deterministic shuffle from the seed
            let mut x = seed | 1;
            for i in (1..shuffled.len()).rev() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                shuffled.swap(i, usize::try_from(x % (i as u64 + 1)).unwrap_or(0));
            }
            let at = cut.min(shuffled.len());
            let (head, tail) = shuffled.split_at(at);
            let split = sum_of(&[make(head), make(tail)], &schema);
            // sums of floats differ in the last bits by order: compare the exact parts and the counts
            let strip = |s: &str| s.split("\"values\":").nth(1).map(|v| {
                v.split(',').filter(|t| !t.contains('.') && !t.contains('e')).collect::<Vec<_>>().join(",")
            });
            proptest::prop_assert_eq!(strip(&whole), strip(&split));
        }
    }
}
