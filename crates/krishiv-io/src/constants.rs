// SPDX-License-Identifier: Apache-2.0
//! Per-file constant columns: typed values a table format keeps in its own
//! metadata, not (or not authoritatively) in the data files. Iceberg's
//! identity-partition values come from the manifest and win over the file's
//! column (FC S4); Delta keeps partition values out of the files (T021).
//!
//! A scan given [`FileConstants`] reads those columns from them, never from
//! the files, exactly as it reads Hive partition columns from the directories:
//! the values are carried with the file (keyed by its store and path), so a distributed
//! task's files find theirs however the splits were cut.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, StringViewArray, TimestampMicrosecondArray, new_null_array,
};
use arrow_schema::{DataType, Field, TimeUnit};
use krishiv_kernels::cast::{CastMode, CastOptions, FixedOffset, cast};
use object_store::path::Path;

use crate::StoreId;
use crate::error::{Result, internal};
use crate::request::ScalarValue;

/// The constant columns' values of each file, in the order of the scan's
/// constant columns (see the module docs).
#[derive(Debug, Clone, Default)]
pub struct FileConstants {
    by_file: HashMap<(StoreId, Path), Arc<[ScalarValue]>>,
}

impl FileConstants {
    /// No file.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the values of the file at `path` in `store` (one per constant
    /// column). Two stores may hold the same path: the store is part of the key.
    pub fn insert(&mut self, store: StoreId, path: Path, values: Vec<ScalarValue>) {
        self.by_file.insert((store, path), values.into());
    }

    /// The values of the file at `path` in `store`.
    #[must_use]
    pub fn get(&self, store: &StoreId, path: &Path) -> Option<&[ScalarValue]> {
        self.by_file
            .get(&(store.clone(), path.clone()))
            .map(AsRef::as_ref)
    }

    /// How many files have values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_file.len()
    }

    /// Whether no file has values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_file.is_empty()
    }
}

/// A column of `rows` copies of `value` of `field`'s type (all null for
/// [`ScalarValue::Null`]). The value is built in its own type and cast
/// losslessly (a string to a string view, a decimal to a narrower decimal
/// storage of the same precision and scale, a timestamp to its zone label).
///
/// # Errors
/// `INTERNAL_ERROR` when the value does not convert to the type without loss.
pub fn scalar_column(field: &Field, value: &ScalarValue, rows: usize) -> Result<ArrayRef> {
    use ScalarValue as S;
    let target = field.data_type();
    let natural: ArrayRef = match value {
        S::Null => return Ok(new_null_array(target, rows)),
        S::Boolean(v) => Arc::new(BooleanArray::from(vec![*v; rows])),
        S::Int8(v) => Arc::new(Int8Array::from(vec![*v; rows])),
        S::Int16(v) => Arc::new(Int16Array::from(vec![*v; rows])),
        S::Int32(v) => Arc::new(Int32Array::from(vec![*v; rows])),
        S::Int64(v) => Arc::new(Int64Array::from(vec![*v; rows])),
        S::Float32(v) => Arc::new(Float32Array::from(vec![*v; rows])),
        S::Float64(v) => Arc::new(Float64Array::from(vec![*v; rows])),
        S::Decimal128 {
            unscaled,
            precision,
            scale,
        } => Arc::new(
            Decimal128Array::from(vec![*unscaled; rows])
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| internal(format!("constant {value:?}: {e}")))?,
        ),
        S::Utf8(v) => match target {
            DataType::Utf8View => Arc::new(StringViewArray::from(vec![v.as_str(); rows])),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec![v.as_str(); rows])),
            _ => Arc::new(StringArray::from(vec![v.as_str(); rows])),
        },
        S::Binary(v) => match target {
            DataType::BinaryView => Arc::new(BinaryViewArray::from(vec![v.as_slice(); rows])),
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from(vec![v.as_slice(); rows])),
            _ => Arc::new(BinaryArray::from(vec![v.as_slice(); rows])),
        },
        S::Date32(v) => Arc::new(Date32Array::from(vec![*v; rows])),
        S::TimestampMicros(v) => {
            let zone = match target {
                DataType::Timestamp(TimeUnit::Microsecond, zone) => zone.clone(),
                _ => None,
            };
            Arc::new(TimestampMicrosecondArray::from(vec![*v; rows]).with_timezone_opt(zone))
        }
        S::TimestampNtzMicros(v) => Arc::new(TimestampMicrosecondArray::from(vec![*v; rows])),
    };
    if natural.data_type() == target {
        return Ok(natural);
    }
    let utc = FixedOffset(0);
    let opts = CastOptions {
        mode: CastMode::Ansi,
        time_zone: &utc,
    };
    let out = cast(&natural, target, &opts)
        .map_err(|e| internal(format!("constant {value:?} as {target}: {e}")))?;
    // The cast must be exact: a float with a fraction to an integer, or a
    // long past a double's 53 bits, would come back as another value. Every
    // row is the same value, so one row tells.
    let exact = || -> Result<bool> {
        if out.is_empty() {
            return Ok(true);
        }
        let back = cast(&out.slice(0, 1), natural.data_type(), &opts)
            .map_err(|e| internal(format!("constant {value:?} as {target}: {e}")))?;
        Ok(back.to_data() == natural.slice(0, 1).to_data())
    };
    if out.null_count() != 0 || !exact()? {
        return Err(internal(format!("constant {value:?} is not a {target}")));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal64Type, Int64Type};

    use super::*;

    #[test]
    fn constants_are_keyed_by_store_and_path() {
        let (a, b) = (StoreId::new("s3://a"), StoreId::new("s3://b"));
        let x = || Path::from("t/x.parquet");
        let mut c = FileConstants::new();
        assert!(c.is_empty());
        c.insert(a.clone(), x(), vec![ScalarValue::Int32(7)]);
        // The same path in another store is another file.
        c.insert(b.clone(), x(), vec![ScalarValue::Int32(9)]);
        c.insert(
            a.clone(),
            Path::from("t/y.parquet"),
            vec![ScalarValue::Null],
        );
        assert_eq!(c.len(), 3);
        assert!(matches!(c.get(&a, &x()), Some([ScalarValue::Int32(7)])));
        assert!(matches!(c.get(&b, &x()), Some([ScalarValue::Int32(9)])));
        assert!(c.get(&a, &Path::from("t/z.parquet")).is_none());
        assert!(c.get(&StoreId::new("s3://c"), &x()).is_none());
    }

    #[test]
    fn scalar_columns_take_the_field_type() {
        let col = |dt: DataType, v: ScalarValue| {
            scalar_column(&Field::new("c", dt, true), &v, 3).unwrap_or_else(|e| panic!("{e}"))
        };
        let a = col(DataType::Int64, ScalarValue::Int64(7));
        assert_eq!(a.as_primitive::<Int64Type>().values().as_ref(), [7, 7, 7]);
        let a = col(DataType::Utf8View, ScalarValue::Utf8("x".to_owned()));
        assert_eq!(a.data_type(), &DataType::Utf8View);
        assert_eq!(a.as_string_view().value(2), "x");
        let a = col(
            DataType::Decimal64(15, 2),
            ScalarValue::Decimal128 {
                unscaled: 1234,
                precision: 15,
                scale: 2,
            },
        );
        assert_eq!(a.as_primitive::<Decimal64Type>().value(0), 1234);
        let tz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let a = col(tz.clone(), ScalarValue::TimestampMicros(5));
        assert_eq!(a.data_type(), &tz);
        let a = col(DataType::Date32, ScalarValue::Null);
        assert_eq!((a.len(), a.null_count()), (3, 3));
        // A value the type cannot hold is an error, not a null.
        let e = scalar_column(
            &Field::new("c", DataType::Int8, true),
            &ScalarValue::Int64(1000),
            1,
        );
        assert!(e.is_err());
        // Nor is a value the cast would round.
        let lossy = |dt: DataType, v: ScalarValue| {
            scalar_column(&Field::new("c", dt, true), &v, 2).is_err()
        };
        assert!(lossy(DataType::Int64, ScalarValue::Float64(1.5)));
        assert!(lossy(DataType::Float64, ScalarValue::Int64((1 << 53) + 1)));
        assert!(!lossy(DataType::Int64, ScalarValue::Float64(2.0)));
        assert!(!lossy(DataType::Float64, ScalarValue::Int64(1 << 53)));
    }

    /// Every scalar kind builds its column, strings and binaries in the view,
    /// large or plain form the field asks for.
    #[test]
    fn scalar_columns_of_every_kind() {
        let col = |dt: DataType, v: ScalarValue| {
            let a = scalar_column(&Field::new("c", dt.clone(), true), &v, 2)
                .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(
                (a.data_type(), a.len(), a.null_count()),
                (&dt, 2, 0),
                "{v:?}"
            );
            a
        };
        assert!(
            col(DataType::Boolean, ScalarValue::Boolean(true))
                .as_boolean()
                .value(1)
        );
        col(DataType::Int8, ScalarValue::Int8(-1));
        col(DataType::Int16, ScalarValue::Int16(300));
        col(DataType::Float32, ScalarValue::Float32(1.5));
        col(DataType::Float64, ScalarValue::Float64(-2.5));
        col(DataType::Date32, ScalarValue::Date32(17_486));
        // A narrower field takes the value by a lossless cast.
        col(DataType::Int32, ScalarValue::Int16(7));
        let s = || ScalarValue::Utf8("x".to_owned());
        assert_eq!(col(DataType::Utf8, s()).as_string::<i32>().value(1), "x");
        assert_eq!(
            col(DataType::LargeUtf8, s()).as_string::<i64>().value(0),
            "x"
        );
        let b = || ScalarValue::Binary(vec![1, 2]);
        assert_eq!(
            col(DataType::Binary, b()).as_binary::<i32>().value(1),
            [1, 2]
        );
        assert_eq!(
            col(DataType::LargeBinary, b()).as_binary::<i64>().value(0),
            [1, 2]
        );
        assert_eq!(
            col(DataType::BinaryView, b()).as_binary_view().value(1),
            [1, 2]
        );
    }
}
