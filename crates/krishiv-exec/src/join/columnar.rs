// SPDX-License-Identifier: Apache-2.0
//! The columnar build payload of a row-layout hash join (design
//! `perf1-join-columnar-string-payload`, variant of 2026-11-20): build
//! columns kept as Arrow columns beside the key rows. A string or binary
//! view column is stored as plain `Utf8` / `Binary` (4-byte offsets and the
//! exact bytes: smaller than a 16-byte view per short value and than the
//! arrow-row blocks, with no shared page buffers to compact), and the probe
//! gathers the picked rows straight into a view column of the declared type.

use std::sync::Arc;

use arrow_array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::ByteViewType;
use arrow_array::{
    Array, ArrayRef, BinaryArray, GenericByteViewArray, OffsetSizeTrait, StringArray,
};
use arrow_schema::{DataType, Field};

use crate::error::{Result, internal};

/// The type an arena stores a build column of type `declared` as.
pub(crate) fn stored_type(declared: &DataType) -> DataType {
    match declared {
        DataType::Utf8View => DataType::Utf8,
        DataType::BinaryView => DataType::Binary,
        other => other.clone(),
    }
}

/// `field` with its stored type ([`stored_type`]).
pub(crate) fn stored_field(field: &Field) -> Field {
    field.clone().with_data_type(stored_type(field.data_type()))
}

/// Rows `rows` of build column `column`, in its stored type.
pub(crate) fn store(column: &ArrayRef, rows: &[usize]) -> Result<ArrayRef> {
    match column.data_type() {
        DataType::Utf8View => {
            let a = column.as_string_view();
            check_bytes(a, rows)?;
            Ok(Arc::new(
                rows.iter()
                    .map(|&i| a.is_valid(i).then(|| a.value(i)))
                    .collect::<StringArray>(),
            ))
        }
        DataType::BinaryView => {
            let a = column.as_binary_view();
            check_bytes(a, rows)?;
            Ok(Arc::new(
                rows.iter()
                    .map(|&i| a.is_valid(i).then(|| a.value(i)))
                    .collect::<BinaryArray>(),
            ))
        }
        _ => {
            let ids = rows
                .iter()
                .map(|&i| {
                    u32::try_from(i).map_err(|_| internal("a build batch too large".to_owned()))
                })
                .collect::<Result<arrow_array::UInt32Array>>()?;
            arrow_select::take::take(column.as_ref(), &ids, None)
                .map_err(|e| internal(e.to_string()))
        }
    }
}

/// Refuses a chunk whose picked values would overflow 32-bit offsets.
fn check_bytes<T: ByteViewType + ?Sized>(
    a: &GenericByteViewArray<T>,
    rows: &[usize],
) -> Result<()> {
    let lens = a.views();
    let total: u64 = rows
        .iter()
        // A view's low 32 bits are its value's length.
        .map(|&i| {
            lens.get(i)
                .map_or(0, |&v| u64::try_from(v & u128::from(u32::MAX)).unwrap_or(0))
        })
        .sum();
    if total > u64::from(i32::MAX.unsigned_abs()) {
        return Err(internal(format!(
            "a hash join build chunk holds {total} bytes of one string column (more than 2 GiB)"
        )));
    }
    Ok(())
}

/// The picked rows (`picks`: source, row) of one build column stored in
/// `sources`, as a column of type `declared`.
pub(crate) fn gather(
    declared: &DataType,
    sources: &[&dyn Array],
    picks: &[(usize, usize)],
) -> Result<ArrayRef> {
    let bad = || internal("a join row outside its arena".to_owned());
    match declared {
        DataType::Utf8View => {
            let s = sources
                .iter()
                .map(|a| a.as_string_opt::<i32>().ok_or_else(bad))
                .collect::<Result<Vec<_>>>()?;
            let mut b = StringViewBuilder::with_capacity(picks.len());
            for &(src, row) in picks {
                let a = s.get(src).ok_or_else(bad)?;
                match value(*a, row).ok_or_else(bad)? {
                    Some(v) => b.append_value(v),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        DataType::BinaryView => {
            let s = sources
                .iter()
                .map(|a| a.as_binary_opt::<i32>().ok_or_else(bad))
                .collect::<Result<Vec<_>>>()?;
            let mut b = BinaryViewBuilder::with_capacity(picks.len());
            for &(src, row) in picks {
                let a = s.get(src).ok_or_else(bad)?;
                match value(*a, row).ok_or_else(bad)? {
                    Some(v) => b.append_value(v),
                    None => b.append_null(),
                }
            }
            Ok(Arc::new(b.finish()))
        }
        _ => arrow_select::interleave::interleave(sources, picks)
            .map_err(|e| internal(e.to_string())),
    }
}

/// Row `row` of `a` (`None` outside it; `Some(None)` for a NULL).
fn value<O: OffsetSizeTrait, T: arrow_array::types::ByteArrayType<Offset = O>>(
    a: &arrow_array::GenericByteArray<T>,
    row: usize,
) -> Option<Option<&T::Native>> {
    (row < a.len()).then(|| a.is_valid(row).then(|| a.value(row)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringViewArray;

    #[test]
    fn views_round_trip_through_offsets() {
        let long = "a string longer than twelve bytes";
        let col: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            None,
            Some(long),
            Some(""),
        ]));
        let stored = store(&col, &[2, 1, 0, 3]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(stored.data_type(), &DataType::Utf8);
        let other = store(&col, &[0]).unwrap_or_else(|e| panic!("{e}"));
        let got = gather(
            &DataType::Utf8View,
            &[stored.as_ref(), other.as_ref()],
            &[(1, 0), (0, 0), (0, 1), (0, 3)],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let want: ArrayRef = Arc::new(StringViewArray::from(vec![
            Some("x"),
            Some(long),
            None,
            Some(""),
        ]));
        assert_eq!(got.as_ref(), want.as_ref());
        assert!(gather(&DataType::Utf8View, &[stored.as_ref()], &[(1, 0)]).is_err());
        assert!(gather(&DataType::Utf8View, &[stored.as_ref()], &[(0, 9)]).is_err());
    }

    #[test]
    fn binary_views_and_fixed_width_columns() {
        let bin: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from(vec![
            Some(&b"ab"[..]),
            None,
            Some(&[7u8; 20][..]),
        ]));
        let stored = store(&bin, &[2, 0, 1]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(stored.data_type(), &DataType::Binary);
        let got = gather(
            &DataType::BinaryView,
            &[stored.as_ref()],
            &[(0, 1), (0, 2), (0, 0)],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let want: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from(vec![
            Some(&b"ab"[..]),
            None,
            Some(&[7u8; 20][..]),
        ]));
        assert_eq!(got.as_ref(), want.as_ref());
        let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![Some(1), None, Some(3)]));
        let stored = store(&ints, &[2, 1]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(stored.data_type(), &DataType::Int32);
        let got = gather(&DataType::Int32, &[stored.as_ref()], &[(0, 1), (0, 0)])
            .unwrap_or_else(|e| panic!("{e}"));
        let want: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![None, Some(3)]));
        assert_eq!(got.as_ref(), want.as_ref());
        assert_eq!(stored_type(&DataType::Utf8), DataType::Utf8);
    }
}
