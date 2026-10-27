// SPDX-License-Identifier: Apache-2.0
//! The one `value` column of the `text` format, read and written.

use arrow_array::Array;

/// A `value` cell: the string at `i` of one of the string arrays, `None`
/// for NULL.
pub(crate) fn str_value(a: &dyn Array, i: usize) -> Option<&str> {
    use arrow_array::cast::AsArray as _;
    use arrow_schema::DataType as T;
    if a.is_null(i) {
        return None;
    }
    Some(match a.data_type() {
        T::Utf8View => a.as_string_view().value(i),
        T::Utf8 => a.as_string::<i32>().value(i),
        _ => a.as_string::<i64>().value(i),
    })
}
