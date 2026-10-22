// SPDX-License-Identifier: Apache-2.0
//! Deterministic test data and a row-wise oracle for pushed predicates.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Float64Type, Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{
    ArrayRef, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, ListArray, MapArray, NullArray,
    RecordBatch, StringViewArray, StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};

use super::SuiteResult;

/// SplitMix64: a small deterministic generator.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// A generator from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (`n > 0`).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    /// A value in `0..n` as usize.
    pub fn index(&mut self, n: usize) -> usize {
        usize::try_from(self.below(n as u64)).unwrap_or(0)
    }
}

fn f(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, true)
}

/// Every Spark type the suite round-trips, nested types and a Variant
/// placeholder (`struct<metadata: binary, value: binary>` tagged with the
/// Parquet Variant extension name) included.
#[must_use]
pub fn all_types_schema() -> SchemaRef {
    let inner = Fields::from(vec![f("c", DataType::Utf8View)]);
    let st = Fields::from(vec![
        f("a", DataType::Int32),
        f("b", DataType::Struct(inner)),
    ]);
    let entries = Field::new(
        "entries",
        DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8View, false),
            f("value", DataType::Int32),
        ])),
        false,
    );
    let variant = f(
        "variant",
        DataType::Struct(Fields::from(vec![
            f("metadata", DataType::BinaryView),
            f("value", DataType::BinaryView),
        ])),
    )
    .with_metadata(HashMap::from([(
        "ARROW:extension:name".to_owned(),
        "arrow.parquet.variant".to_owned(),
    )]));
    Arc::new(Schema::new(vec![
        f("bool", DataType::Boolean),
        f("byte", DataType::Int8),
        f("short", DataType::Int16),
        f("int", DataType::Int32),
        f("long", DataType::Int64),
        f("float", DataType::Float32),
        f("double", DataType::Float64),
        f("dec", DataType::Decimal128(10, 2)),
        f("wide_dec", DataType::Decimal128(38, 6)),
        f("string", DataType::Utf8View),
        f("binary", DataType::BinaryView),
        f("date", DataType::Date32),
        f(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        ),
        f("ts_ntz", DataType::Timestamp(TimeUnit::Microsecond, None)),
        f("st", DataType::Struct(st)),
        f(
            "list",
            DataType::List(Arc::new(f("element", DataType::Int32))),
        ),
        f("map", DataType::Map(Arc::new(entries), false)),
        variant,
        Field::new("void", DataType::Null, true),
    ]))
}

/// Deterministic rows of `schema` (row `i` is a function of `i` and `seed`;
/// about one value in seven is null).
///
/// # Errors
/// A type without a generator.
pub fn batch(schema: &SchemaRef, rows: usize, seed: u64) -> SuiteResult<RecordBatch> {
    let cols = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(c, fld)| column(fld.data_type(), rows, seed.wrapping_add(c as u64 * 7919)))
        .collect::<SuiteResult<Vec<_>>>()?;
    RecordBatch::try_new(Arc::clone(schema), cols).map_err(|e| format!("test data: {e}"))
}

fn nulls(rows: usize, seed: u64) -> Vec<bool> {
    let mut r = Rng::new(seed ^ 0xA5A5);
    (0..rows).map(|_| r.below(7) != 3).collect()
}

/// A column of `dt`.
///
/// # Errors
/// A type without a generator.
pub fn column(dt: &DataType, rows: usize, seed: u64) -> SuiteResult<ArrayRef> {
    let valid = nulls(rows, seed);
    let mut r = Rng::new(seed);
    let vals = |m: u64| -> Vec<Option<i64>> {
        let mut g = Rng::new(seed.wrapping_mul(31));
        (0..rows)
            .map(|i| {
                valid[i].then(|| {
                    i64::try_from(g.below(m)).unwrap_or(0) - i64::try_from(m / 2).unwrap_or(0)
                })
            })
            .collect()
    };
    Ok(match dt {
        DataType::Boolean => Arc::new(BooleanArray::from(
            vals(2)
                .into_iter()
                .map(|v| v.map(|v| v == 0))
                .collect::<Vec<_>>(),
        )),
        DataType::Int8 => Arc::new(Int8Array::from(
            vals(250)
                .into_iter()
                .map(|v| v.and_then(|v| i8::try_from(v).ok()))
                .collect::<Vec<_>>(),
        )),
        DataType::Int16 => Arc::new(Int16Array::from(
            vals(60_000)
                .into_iter()
                .map(|v| v.and_then(|v| i16::try_from(v).ok()))
                .collect::<Vec<_>>(),
        )),
        DataType::Int32 => Arc::new(Int32Array::from(
            vals(2_000_000)
                .into_iter()
                .map(|v| v.and_then(|v| i32::try_from(v).ok()))
                .collect::<Vec<_>>(),
        )),
        DataType::Int64 => Arc::new(Int64Array::from(vals(1 << 40))),
        DataType::Float32 => Arc::new(Float32Array::from(
            special(vals(1000), seed)
                .into_iter()
                .map(|v| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "test data: rounding to f32 is the point"
                    )]
                    v.map(|v| v as f32)
                })
                .collect::<Vec<_>>(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(special(vals(1000), seed))),
        DataType::Decimal128(p, s) => Arc::new(
            Decimal128Array::from(
                vals(1_000_000)
                    .into_iter()
                    .map(|v| v.map(i128::from))
                    .collect::<Vec<_>>(),
            )
            .with_precision_and_scale(*p, *s)
            .map_err(|e| format!("decimal: {e}"))?,
        ),
        DataType::Utf8View => Arc::new(StringViewArray::from(
            vals(500)
                .iter()
                .map(|v| v.map(word))
                .collect::<Vec<_>>()
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        )),
        DataType::BinaryView => Arc::new(BinaryViewArray::from(
            vals(500)
                .iter()
                .map(|v| v.map(|v| v.to_le_bytes()[..3].to_vec()))
                .collect::<Vec<_>>()
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        )),
        DataType::Date32 => Arc::new(Date32Array::from(
            vals(40_000)
                .into_iter()
                .map(|v| v.and_then(|v| i32::try_from(v).ok()))
                .collect::<Vec<_>>(),
        )),
        DataType::Timestamp(_, tz) => {
            Arc::new(TimestampMicrosecondArray::from(vals(1 << 50)).with_timezone_opt(tz.clone()))
        }
        DataType::Struct(fields) => {
            let children = fields
                .iter()
                .enumerate()
                .map(|(i, c)| column(c.data_type(), rows, seed.wrapping_add(i as u64 + 1)))
                .collect::<SuiteResult<Vec<_>>>()?;
            Arc::new(
                StructArray::try_new(fields.clone(), children, Some(NullBuffer::from(valid)))
                    .map_err(|e| format!("struct: {e}"))?,
            )
        }
        DataType::List(fld) => {
            let lens: Vec<usize> = (0..rows)
                .map(|i| if valid[i] { r.index(4) } else { 0 })
                .collect();
            let total = lens.iter().sum();
            let values = column(fld.data_type(), total, seed.wrapping_add(1))?;
            Arc::new(
                ListArray::try_new(
                    Arc::clone(fld),
                    OffsetBuffer::from_lengths(lens),
                    values,
                    Some(NullBuffer::from(valid)),
                )
                .map_err(|e| format!("list: {e}"))?,
            )
        }
        DataType::Map(entries, sorted) => {
            let DataType::Struct(kv) = entries.data_type() else {
                return Err("map entries".into());
            };
            let lens: Vec<usize> = (0..rows)
                .map(|i| if valid[i] { r.index(3) } else { 0 })
                .collect();
            let total: usize = lens.iter().sum();
            let keys: Vec<String> = (0..total).map(|i| format!("k{i}")).collect();
            let keys = Arc::new(StringViewArray::from(
                keys.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let values = column(kv[1].data_type(), total, seed.wrapping_add(2))?;
            let entries_arr = StructArray::try_new(kv.clone(), vec![keys, values], None)
                .map_err(|e| format!("map entries: {e}"))?;
            Arc::new(
                MapArray::try_new(
                    Arc::clone(entries),
                    OffsetBuffer::from_lengths(lens),
                    entries_arr,
                    Some(NullBuffer::from(valid)),
                    *sorted,
                )
                .map_err(|e| format!("map: {e}"))?,
            )
        }
        DataType::Null => Arc::new(NullArray::new(rows)),
        other => return Err(format!("no generator for {other}")),
    })
}

/// Mixes NaN, -0.0 and 0.0 into float data.
fn special(v: Vec<Option<i64>>, seed: u64) -> Vec<Option<f64>> {
    let mut r = Rng::new(seed ^ 0xF10A7);
    v.into_iter()
        .map(|x| {
            x.map(|x| match r.below(23) {
                0 => f64::NAN,
                1 => -0.0,
                2 => 0.0,
                _ => x as f64 / 4.0,
            })
        })
        .collect()
}

fn word(v: i64) -> String {
    let letters = ["ab", "Ab", "abc", "b", "B", "ba", "zz", "é", "a\u{0}", ""];
    let i = usize::try_from(v.rem_euclid(letters.len() as i64)).unwrap_or(0);
    format!("{}{}", letters[i], v.rem_euclid(97))
}

/// A literal drawn from `a`'s values, plus edge literals.
pub fn literal(a: &ArrayRef, r: &mut Rng) -> Option<ScalarValue> {
    let n = a.len();
    if n == 0 {
        return None;
    }
    let pick = r.index(n);
    let edge = r.below(6) == 0;
    Some(match a.data_type() {
        DataType::Int32 => {
            if edge {
                [ScalarValue::Int32(i32::MIN), ScalarValue::Int32(i32::MAX)][r.index(2)].clone()
            } else {
                ScalarValue::Int32(a.as_primitive::<Int32Type>().value(pick))
            }
        }
        DataType::Int64 => {
            if edge {
                ScalarValue::Int64([3_000_000_000_i64, -3_000_000_000][r.index(2)])
            } else {
                ScalarValue::Int64(a.as_primitive::<Int64Type>().value(pick))
            }
        }
        DataType::Float64 => {
            if edge {
                ScalarValue::Float64([f64::NAN, -0.0, 0.0, 0.1][r.index(4)])
            } else {
                ScalarValue::Float64(a.as_primitive::<Float64Type>().value(pick))
            }
        }
        DataType::Utf8View => ScalarValue::Utf8(a.as_string_view().value(pick).to_owned()),
        DataType::Date32 => ScalarValue::Date32(a.as_primitive::<Date32Type>().value(pick)),
        DataType::Timestamp(..) => {
            ScalarValue::TimestampMicros(a.as_primitive::<TimestampMicrosecondType>().value(pick))
        }
        _ => return None,
    })
}

/// A random predicate over the top-level columns `cols` of `batch` (which
/// must exist).
pub fn predicate(batch: &RecordBatch, cols: &[&str], r: &mut Rng, depth: u32) -> PushdownExpr {
    if depth > 0 && r.below(3) == 0 {
        let a = predicate(batch, cols, r, depth - 1);
        let b = predicate(batch, cols, r, depth - 1);
        return match r.below(3) {
            0 => PushdownExpr::And(vec![a, b]),
            1 => PushdownExpr::Or(vec![a, b]),
            _ => PushdownExpr::Not(Box::new(a)),
        };
    }
    let name = cols[r.index(cols.len())];
    let Some(a) = batch.column_by_name(name) else {
        return PushdownExpr::Constant(true);
    };
    let col = ColumnPath::new(name);
    let collation = if a.data_type() == &DataType::Utf8View && r.below(4) == 0 {
        Collation::Other("UTF8_LCASE".to_owned())
    } else {
        Collation::Utf8Binary
    };
    match r.below(8) {
        0 => PushdownExpr::IsNull(col),
        1 => PushdownExpr::IsNotNull(col),
        2 => PushdownExpr::In {
            column: col,
            values: (0..3).filter_map(|_| literal(a, r)).collect(),
            collation,
        },
        3 if a.data_type() == &DataType::Utf8View && !a.is_empty() => {
            let s = a.as_string_view().value(r.index(a.len()));
            PushdownExpr::StartsWith {
                column: col,
                prefix: s.chars().take(r.index(3)).collect(),
                collation,
            }
        }
        4 if a.data_type() == &DataType::Utf8View && !a.is_empty() => {
            let s: Vec<char> = a.as_string_view().value(r.index(a.len())).chars().collect();
            let start = r.index(s.len() + 1);
            let len = r.index(s.len() - start + 1).max(1);
            PushdownExpr::Contains {
                column: col,
                needle: s[start..].iter().take(len).collect(),
                collation,
            }
        }
        _ => {
            let ops = [
                CmpOp::Eq,
                CmpOp::NotEq,
                CmpOp::Lt,
                CmpOp::LtEq,
                CmpOp::Gt,
                CmpOp::GtEq,
            ];
            match literal(a, r) {
                Some(value) => PushdownExpr::Cmp {
                    column: col,
                    op: ops[r.index(6)],
                    value,
                    collation,
                },
                None => PushdownExpr::IsNotNull(col),
            }
        }
    }
}

/// Spark's value order: NaN equals NaN and is the largest double; -0.0
/// equals 0.0.
fn cmp(a: &ScalarValue, b: &ScalarValue) -> Option<Ordering> {
    use ScalarValue as S;
    let num = |s: &S| -> Option<f64> {
        #[expect(clippy::cast_precision_loss, reason = "test oracle over small ints")]
        Some(match s {
            S::Int32(v) | S::Date32(v) => f64::from(*v),
            S::Int64(v) | S::TimestampMicros(v) => *v as f64,
            S::Float64(v) => *v,
            _ => return None,
        })
    };
    match (a, b) {
        (S::Utf8(x), S::Utf8(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        (S::Int64(x), S::Int64(y)) | (S::TimestampMicros(x), S::TimestampMicros(y)) => {
            Some(x.cmp(y))
        }
        _ => {
            let (x, y) = (num(a)?, num(b)?);
            Some(match (x.is_nan(), y.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
            })
        }
    }
}

fn value(a: &ArrayRef, i: usize) -> Option<ScalarValue> {
    use arrow_array::Array;
    if a.is_null(i) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Int32 => ScalarValue::Int32(a.as_primitive::<Int32Type>().value(i)),
        DataType::Int64 => ScalarValue::Int64(a.as_primitive::<Int64Type>().value(i)),
        DataType::Float64 => ScalarValue::Float64(a.as_primitive::<Float64Type>().value(i)),
        DataType::Utf8View => ScalarValue::Utf8(a.as_string_view().value(i).to_owned()),
        DataType::Date32 => ScalarValue::Date32(a.as_primitive::<Date32Type>().value(i)),
        DataType::Timestamp(..) => {
            ScalarValue::TimestampMicros(a.as_primitive::<TimestampMicrosecondType>().value(i))
        }
        _ => return None,
    })
}

fn fold(s: &ScalarValue, c: &Collation) -> ScalarValue {
    match (s, c) {
        (ScalarValue::Utf8(v), Collation::Other(_)) => ScalarValue::Utf8(v.to_lowercase()),
        _ => s.clone(),
    }
}

/// SQL three-valued evaluation of `e` on row `i` of `batch` (the oracle);
/// a column missing from `batch` reads as null.
pub fn eval(e: &PushdownExpr, batch: &RecordBatch, i: usize) -> Option<bool> {
    let null: ArrayRef = Arc::new(NullArray::new(batch.num_rows()));
    let col = |c: &ColumnPath| {
        batch
            .column_by_name(c.root())
            .cloned()
            .unwrap_or_else(|| Arc::clone(&null))
    };
    match e {
        PushdownExpr::Constant(b) => Some(*b),
        PushdownExpr::IsNull(c) => Some(arrow_array::Array::is_null(col(c).as_ref(), i)),
        PushdownExpr::IsNotNull(c) => Some(!arrow_array::Array::is_null(col(c).as_ref(), i)),
        PushdownExpr::Not(x) => eval(x, batch, i).map(|b| !b),
        PushdownExpr::And(xs) => xs
            .iter()
            .try_fold(true, |acc, x| match eval(x, batch, i) {
                Some(false) => Err(()),
                Some(true) => Ok(acc),
                None => Ok(false),
            })
            .map_or(Some(false), |all| all.then_some(true)),
        PushdownExpr::Or(xs) => {
            let vals: Vec<Option<bool>> = xs.iter().map(|x| eval(x, batch, i)).collect();
            if vals.contains(&Some(true)) {
                Some(true)
            } else if vals.contains(&None) {
                None
            } else {
                Some(false)
            }
        }
        PushdownExpr::Cmp {
            column,
            op,
            value: lit,
            collation,
        } => {
            let v = fold(&value(&col(column), i)?, collation);
            let o = cmp(&v, &fold(lit, collation))?;
            Some(match op {
                CmpOp::Eq => o == Ordering::Equal,
                CmpOp::NotEq => o != Ordering::Equal,
                CmpOp::Lt => o == Ordering::Less,
                CmpOp::LtEq => o != Ordering::Greater,
                CmpOp::Gt => o == Ordering::Greater,
                CmpOp::GtEq => o != Ordering::Less,
            })
        }
        PushdownExpr::In {
            column,
            values,
            collation,
        } => {
            let v = fold(&value(&col(column), i)?, collation);
            Some(
                values
                    .iter()
                    .any(|l| cmp(&v, &fold(l, collation)) == Some(Ordering::Equal)),
            )
        }
        PushdownExpr::StartsWith {
            column,
            prefix,
            collation,
        } => {
            let ScalarValue::Utf8(v) = fold(&value(&col(column), i)?, collation) else {
                return None;
            };
            let p = match collation {
                Collation::Other(_) => prefix.to_lowercase(),
                Collation::Utf8Binary => prefix.clone(),
            };
            Some(v.starts_with(&p))
        }
        PushdownExpr::Contains {
            column,
            needle,
            collation,
        } => {
            let ScalarValue::Utf8(v) = fold(&value(&col(column), i)?, collation) else {
                return None;
            };
            let n = match collation {
                Collation::Other(_) => needle.to_lowercase(),
                Collation::Utf8Binary => needle.clone(),
            };
            Some(v.contains(&n))
        }
        _ => None,
    }
}
