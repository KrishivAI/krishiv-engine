// SPDX-License-Identifier: Apache-2.0
//! Stage 2a of `design/P2-T080b-parquet-indexes.md`: what a stored Bloom
//! filter can answer. An equality or `IN` conjunct on a column whose file
//! values have an exact plain encoding becomes a set of 64-bit hashes
//! (XXH64 of the plain-encoded value, as Parquet's split-block filter
//! hashes it); the filter says a value is surely absent or maybe present.
//!
//! Only types where Spark's equality is byte equality of the stored value
//! take part: integers, dates, microsecond timestamps and integer-backed
//! decimals (the literal converted back to the file type exactly), and
//! strings and binaries under binary collation. Never floats (`0.0 = -0.0`
//! and NaN differ from their byte hashes), booleans, `INT96`, unsigned
//! integers, nested leaves or other collations.

use arrow_schema::DataType;
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::schema::types::ColumnDescriptor;

use crate::request::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};
use crate::runtime_filter::xxh64;

/// The column and literals of an equality conjunct under binary collation:
/// `col = v` or `col IN (v, ...)`.
pub(super) fn probe_values(c: &PushdownExpr) -> Option<(&ColumnPath, Vec<&ScalarValue>)> {
    match c {
        PushdownExpr::Cmp {
            column,
            op: CmpOp::Eq,
            value,
            collation: Collation::Utf8Binary,
        } => Some((column, vec![value])),
        PushdownExpr::In {
            column,
            values,
            collation: Collation::Utf8Binary,
        } if !values.is_empty() => Some((column, values.iter().collect())),
        _ => None,
    }
}

/// Whether the column `d` can be probed at all: not unsigned, and a
/// physical type with a plain encoding of its own.
pub(super) fn eligible(d: &ColumnDescriptor) -> bool {
    let unsigned = matches!(d.logical_type_ref(), Some(LogicalType::Integer(i)) if !i.is_signed)
        || matches!(
            d.converted_type(),
            ConvertedType::UINT_8
                | ConvertedType::UINT_16
                | ConvertedType::UINT_32
                | ConvertedType::UINT_64
        );
    !unsigned
        && d.max_rep_level() == 0
        && matches!(
            d.physical_type(),
            PhysicalType::INT32 | PhysicalType::INT64 | PhysicalType::BYTE_ARRAY
        )
}

/// The integer a literal stands for in a column of engine type `dt`.
fn integer(v: &ScalarValue, dt: &DataType, micros_per_unit: Option<i64>) -> Option<i64> {
    match (v, dt) {
        (
            ScalarValue::Int8(x),
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64,
        ) => Some(i64::from(*x)),
        (ScalarValue::Int16(x), DataType::Int16 | DataType::Int32 | DataType::Int64) => {
            Some(i64::from(*x))
        }
        (ScalarValue::Int32(x), DataType::Int32 | DataType::Int64)
        | (ScalarValue::Date32(x), DataType::Date32) => Some(i64::from(*x)),
        (ScalarValue::Int64(x), DataType::Int64) => Some(*x),
        (
            ScalarValue::TimestampMicros(m),
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _),
        ) => {
            // The file stores `micros_per_unit` microseconds per unit: a
            // literal between two stored values has no exact file value.
            let unit = micros_per_unit.filter(|u| *u > 0)?;
            (m % unit == 0).then_some(m / unit)
        }
        (
            ScalarValue::Decimal128 {
                unscaled, scale, ..
            },
            DataType::Decimal64(_, file_scale) | DataType::Decimal128(_, file_scale),
        ) if scale == file_scale => i64::try_from(*unscaled).ok(),
        _ => None,
    }
}

/// The plain-encoded bytes of every literal of a conjunct, as the file
/// stores them, hashed; `None` when any literal has no exact file value
/// (the conjunct then stays unknown).
pub(super) fn hashes(
    values: &[&ScalarValue],
    physical: PhysicalType,
    dt: &DataType,
    micros_per_unit: Option<i64>,
) -> Option<Vec<u64>> {
    values
        .iter()
        .map(|v| {
            let bytes: Vec<u8> = match physical {
                PhysicalType::INT32 => i32::try_from(integer(v, dt, micros_per_unit)?)
                    .ok()?
                    .to_le_bytes()
                    .to_vec(),
                PhysicalType::INT64 => integer(v, dt, micros_per_unit)?.to_le_bytes().to_vec(),
                PhysicalType::BYTE_ARRAY => match (v, dt) {
                    (ScalarValue::Utf8(s), DataType::Utf8 | DataType::Utf8View) => {
                        s.as_bytes().to_vec()
                    }
                    (ScalarValue::Binary(b), DataType::Binary | DataType::BinaryView) => b.clone(),
                    _ => return None,
                },
                _ => return None,
            };
            Some(xxh64(&bytes))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h32(v: i32) -> u64 {
        xxh64(&v.to_le_bytes())
    }

    #[test]
    fn integers_hash_in_the_files_width() {
        let dt = DataType::Int16;
        let got = hashes(
            &[&ScalarValue::Int16(7), &ScalarValue::Int8(-3)],
            PhysicalType::INT32,
            &dt,
            None,
        );
        assert_eq!(got, Some(vec![h32(7), h32(-3)]));
        // A literal wider than the file type has no exact value.
        assert_eq!(
            hashes(
                &[&ScalarValue::Int64(1 << 40)],
                PhysicalType::INT32,
                &DataType::Int32,
                None
            ),
            None
        );
        // Another type's literal does not take part.
        assert_eq!(
            hashes(
                &[&ScalarValue::Utf8("7".into())],
                PhysicalType::INT32,
                &DataType::Int32,
                None
            ),
            None
        );
    }

    #[test]
    fn timestamps_need_an_exact_file_value() {
        let dt = DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None);
        // A millisecond file: the micros literal must be whole milliseconds.
        let ok = hashes(
            &[&ScalarValue::TimestampMicros(5_000)],
            PhysicalType::INT64,
            &dt,
            Some(1000),
        );
        assert_eq!(ok, Some(vec![xxh64(&5_i64.to_le_bytes())]));
        let between = hashes(
            &[&ScalarValue::TimestampMicros(5_001)],
            PhysicalType::INT64,
            &dt,
            Some(1000),
        );
        assert_eq!(between, None);
        // Nanoseconds (no scale): no exact value.
        assert_eq!(
            hashes(
                &[&ScalarValue::TimestampMicros(5_000)],
                PhysicalType::INT64,
                &dt,
                None
            ),
            None
        );
    }

    #[test]
    fn decimals_need_the_files_scale() {
        let dt = DataType::Decimal128(10, 2);
        let same = ScalarValue::Decimal128 {
            unscaled: 1234,
            precision: 10,
            scale: 2,
        };
        let other = ScalarValue::Decimal128 {
            unscaled: 12,
            precision: 10,
            scale: 0,
        };
        assert_eq!(
            hashes(&[&same], PhysicalType::INT64, &dt, None),
            Some(vec![xxh64(&1234_i64.to_le_bytes())])
        );
        assert_eq!(hashes(&[&other], PhysicalType::INT64, &dt, None), None);
    }

    #[test]
    fn strings_hash_their_bytes_and_floats_never() {
        let s = ScalarValue::Utf8("abc".into());
        assert_eq!(
            hashes(&[&s], PhysicalType::BYTE_ARRAY, &DataType::Utf8View, None),
            Some(vec![xxh64(b"abc")])
        );
        // A string literal against a binary column is another type.
        assert_eq!(
            hashes(&[&s], PhysicalType::BYTE_ARRAY, &DataType::BinaryView, None),
            None
        );
        assert_eq!(
            hashes(
                &[&ScalarValue::Float64(1.0)],
                PhysicalType::INT64,
                &DataType::Float64,
                None
            ),
            None
        );
    }

    #[test]
    fn only_binary_collation_equalities_are_probes() {
        let eq = PushdownExpr::eq(ColumnPath::new("a"), ScalarValue::Int32(1));
        assert!(probe_values(&eq).is_some());
        let lt = PushdownExpr::cmp(ColumnPath::new("a"), CmpOp::Lt, ScalarValue::Int32(1));
        assert!(probe_values(&lt).is_none());
        let ci = PushdownExpr::Cmp {
            column: ColumnPath::new("a"),
            op: CmpOp::Eq,
            value: ScalarValue::Utf8("x".into()),
            collation: Collation::Other("UTF8_LCASE".into()),
        };
        assert!(probe_values(&ci).is_none());
        let empty_in = PushdownExpr::In {
            column: ColumnPath::new("a"),
            values: vec![],
            collation: Collation::Utf8Binary,
        };
        assert!(probe_values(&empty_in).is_none());
    }
}
