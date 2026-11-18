// SPDX-License-Identifier: Apache-2.0
//! String columns from raw value bytes with one UTF-8 pass (P3-A 2a).
//!
//! Row decode and accumulator output hold string values as bytes. Checking
//! each value with `from_utf8` cost 6-8% of ClickBench q14's CPU. Here the
//! values are copied into one buffer (the copy a builder made anyway) and
//! `GenericStringArray::try_new` validates the whole buffer once and then
//! checks that every offset is a char boundary: the whole buffer valid and
//! every value starting on a boundary means every value is valid. The bytes
//! are still checked, so a corrupt spill file or exchange frame still fails.
//! A `Utf8View` column is then built from the checked values (only those
//! past 12 bytes are copied again), so no `unsafe` is needed.

use std::sync::Arc;

use arrow_array::{ArrayRef, GenericStringArray, OffsetSizeTrait, StringViewArray};
use arrow_buffer::{Buffer, NullBufferBuilder, OffsetBuffer, ScalarBuffer};
use arrow_schema::DataType;

use crate::error::{Result, internal};

/// The string column of type `data_type` (`Utf8`, `LargeUtf8` or
/// `Utf8View`) holding `values`.
///
/// # Errors
/// An internal error when a value is not UTF-8, when the values pass the
/// offset range of `Utf8`, or for any other `data_type`.
pub(crate) fn text_array(data_type: &DataType, values: &[Option<&[u8]>]) -> Result<ArrayRef> {
    let total = || values.iter().flatten().map(|v| v.len()).sum::<usize>();
    Ok(match data_type {
        DataType::Utf8 => Arc::new(string_array::<i32>(values, total())?),
        DataType::LargeUtf8 => Arc::new(string_array::<i64>(values, total())?),
        DataType::Utf8View => {
            // Validated once through a string array, then copied into views:
            // the view buffers hold only values past the 12 inline bytes, as
            // a builder makes them, so filters and spill compaction find no
            // unused bytes to collect.
            let total = total();
            if i32::try_from(total).is_ok() {
                Arc::new(StringViewArray::from_iter(
                    string_array::<i32>(values, total)?.iter(),
                ))
            } else {
                Arc::new(StringViewArray::from_iter(
                    string_array::<i64>(values, total)?.iter(),
                ))
            }
        }
        t => return Err(internal(format!("no text decoding for {t}"))),
    })
}

/// `values` as a string array with `O` offsets; `total` is their byte count.
fn string_array<O: OffsetSizeTrait>(
    values: &[Option<&[u8]>],
    total: usize,
) -> Result<GenericStringArray<O>> {
    let too_long = || internal(format!("{total} string bytes pass the offset range"));
    O::from_usize(total).ok_or_else(too_long)?;
    let mut data = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(values.len() + 1);
    offsets.push(O::usize_as(0));
    let mut nulls = NullBufferBuilder::new(values.len());
    for v in values {
        match v {
            Some(b) => {
                data.extend_from_slice(b);
                nulls.append_non_null();
            }
            None => nulls.append_null(),
        }
        // `data.len() <= total`, which fits `O` (checked above).
        offsets.push(O::usize_as(data.len()));
    }
    // Monotonic from 0 by construction.
    let offsets = OffsetBuffer::new(ScalarBuffer::from(offsets));
    GenericStringArray::try_new(offsets, Buffer::from_vec(data), nulls.finish())
        .map_err(|e| internal(format!("a row string is not UTF-8: {e}")))
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, LargeStringArray, StringArray};

    use super::*;

    const TYPES: [DataType; 3] = [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View];

    /// The column as `Option<String>`s, whatever its string type.
    fn strings(a: &ArrayRef) -> Vec<Option<String>> {
        match a.data_type() {
            DataType::Utf8 => a
                .as_string::<i32>()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect(),
            DataType::LargeUtf8 => a
                .as_string::<i64>()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect(),
            DataType::Utf8View => a
                .as_string_view()
                .iter()
                .map(|v| v.map(str::to_owned))
                .collect(),
            t => panic!("not a string type: {t}"),
        }
    }

    #[test]
    fn valid_values_nulls_empty_and_long_round_trip() {
        let long = "a value well past the twelve inline bytes, with ü and 日本";
        let values: [Option<&[u8]>; 6] = [
            Some(b"short"),
            None,
            Some(b""),
            Some(long.as_bytes()),
            Some("é".as_bytes()),
            None,
        ];
        let want: Vec<Option<String>> = values
            .iter()
            .map(|v| v.map(|b| String::from_utf8(b.to_vec()).unwrap()))
            .collect();
        for t in TYPES {
            let a = text_array(&t, &values).unwrap();
            assert_eq!(a.data_type(), &t);
            assert_eq!(a.null_count(), 2, "{t}");
            assert_eq!(strings(&a), want, "{t}");
        }
    }

    #[test]
    fn empty_input_is_an_empty_column() {
        for t in TYPES {
            let a = text_array(&t, &[]).unwrap();
            assert_eq!((a.len(), a.data_type()), (0, &t));
        }
    }

    #[test]
    fn invalid_bytes_are_refused() {
        let e = "é".as_bytes(); // 0xC3 0xA9
        let cases: [&[Option<&[u8]>]; 5] = [
            // A two-byte char split across two values: the whole buffer is
            // valid UTF-8, the boundary check catches it.
            &[Some(&e[..1]), Some(&e[1..])],
            // A lone continuation byte starting a value.
            &[Some(b"ok"), Some(&[0xA9, b'x'])],
            // Invalid in the middle of a value.
            &[Some(&[b'a', 0xFF, b'b'])],
            // A truncated char at the end of the last value.
            &[Some(b"x"), Some(&e[..1])],
            // A split char around a NULL.
            &[Some(&e[..1]), None, Some(&e[1..])],
        ];
        for t in TYPES {
            for (i, c) in cases.iter().enumerate() {
                let err = text_array(&t, c).unwrap_err().to_string();
                assert!(err.contains("not UTF-8"), "{t} case {i}: {err}");
            }
        }
    }

    #[test]
    fn view_buffers_hold_only_long_values() {
        let long = "a value of more than twelve bytes";
        let values: [Option<&[u8]>; 4] = [
            Some(b"short"),
            Some(long.as_bytes()),
            None,
            Some(b"twelve bytes"),
        ];
        let a = text_array(&DataType::Utf8View, &values).unwrap();
        let v = a.as_string_view();
        let held: usize = v.data_buffers().iter().map(arrow_buffer::Buffer::len).sum();
        assert_eq!(held, long.len());
        assert_eq!(v.total_buffer_bytes_used(), long.len());
    }

    #[test]
    fn other_types_are_refused() {
        assert!(text_array(&DataType::Binary, &[Some(b"x")]).is_err());
    }

    #[test]
    fn matches_the_per_value_path() {
        // A deterministic mix of ASCII, multibyte, empty, NULL and long values.
        let pool = [
            "",
            "a",
            "ü",
            "日本語",
            "twelve bytes",
            "thirteen byte",
            "x".repeat(40).leak(),
        ];
        let values: Vec<Option<&[u8]>> = (0..500usize)
            .map(|i| (i % 7 != 3).then(|| pool[(i * 31 + i / 5) % pool.len()].as_bytes()))
            .collect();
        let per_value: Vec<Option<&str>> = values
            .iter()
            .map(|v| v.map(|b| std::str::from_utf8(b).unwrap()))
            .collect();
        let a = text_array(&DataType::Utf8, &values).unwrap();
        assert_eq!(a.as_string::<i32>(), &StringArray::from(per_value.clone()));
        let a = text_array(&DataType::LargeUtf8, &values).unwrap();
        assert_eq!(
            a.as_string::<i64>(),
            &LargeStringArray::from(per_value.clone())
        );
        let a = text_array(&DataType::Utf8View, &values).unwrap();
        assert_eq!(a.as_string_view(), &StringViewArray::from(per_value));
    }
}
