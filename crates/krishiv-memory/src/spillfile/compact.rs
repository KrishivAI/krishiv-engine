// SPDX-License-Identifier: Apache-2.0
//! Compacting view arrays before a spill write.
//!
//! A `Utf8View` / `BinaryView` array shares its data buffers with every
//! slice and filter of the array it came from, and Arrow IPC writes a view
//! array's data buffers whole. Spilling a small slice of such a column would
//! therefore write (and reserve scratch for, and read back) every byte of the
//! shared buffers. The spill writer first copies such columns down to the
//! bytes their views reference, so a frame's size follows its rows.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::ByteViewType;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, GenericByteViewArray, GenericListArray, MapArray,
    OffsetSizeTrait, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType};

/// Bytes of one view.
const VIEW_BYTES: usize = 16;

/// Bytes [`compact`] allocates for `array` (0 if it is compact already):
/// the copied views, their referenced bytes and validity, and the rebased
/// offsets of the lists above them.
pub(crate) fn copy_bytes(array: &dyn Array) -> usize {
    match array.data_type() {
        DataType::Utf8View => view_copy(array.as_string_view()),
        DataType::BinaryView => view_copy(array.as_binary_view()),
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .map(|c| copy_bytes(c))
            .sum(),
        DataType::List(_) => list_copy(array.as_list::<i32>()),
        DataType::LargeList(_) => list_copy(array.as_list::<i64>()),
        DataType::FixedSizeList(..) => copy_bytes(array.as_fixed_size_list().values()),
        DataType::Map(..) => {
            let map = array.as_map();
            let (first, len) = value_range(map.value_offsets());
            let entries = copy_bytes(&map.entries().slice(first, len));
            if entries == 0 {
                0
            } else {
                entries + offsets_bytes::<i32>(map.len())
            }
        }
        _ => 0,
    }
}

/// `array` with every view array in it (nested ones included) holding only
/// the bytes its views reference, or `None` if it is compact already.
///
/// # Errors
/// An Arrow error if a nested array cannot be rebuilt (a malformed input).
pub(crate) fn compact(array: &ArrayRef) -> Result<Option<ArrayRef>, ArrowError> {
    if copy_bytes(array.as_ref()) == 0 {
        return Ok(None);
    }
    let out: ArrayRef = match array.data_type() {
        DataType::Utf8View => Arc::new(array.as_string_view().gc()),
        DataType::BinaryView => Arc::new(array.as_binary_view().gc()),
        DataType::Struct(_) => {
            let s = array.as_struct();
            let columns = s
                .columns()
                .iter()
                .map(|c| Ok(compact(c)?.unwrap_or_else(|| Arc::clone(c))))
                .collect::<Result<Vec<_>, ArrowError>>()?;
            let (fields, _, nulls) = s.clone().into_parts();
            Arc::new(StructArray::try_new_with_length(
                fields,
                columns,
                nulls,
                s.len(),
            )?)
        }
        DataType::List(_) => compact_list(array.as_list::<i32>())?,
        DataType::LargeList(_) => compact_list(array.as_list::<i64>())?,
        DataType::FixedSizeList(..) => {
            let l = array.as_fixed_size_list();
            let values = compact(l.values())?.unwrap_or_else(|| Arc::clone(l.values()));
            let (field, size, _, nulls) = l.clone().into_parts();
            Arc::new(FixedSizeListArray::try_new_with_length(
                field,
                size,
                values,
                nulls,
                l.len(),
            )?)
        }
        DataType::Map(_, ordered) => {
            let m = array.as_map();
            let (offsets, first, len) = rebased(m.value_offsets());
            let entries: ArrayRef = Arc::new(m.entries().slice(first, len));
            let entries = compact(&entries)?.unwrap_or(entries);
            let (field, _, _, nulls, _) = m.clone().into_parts();
            Arc::new(MapArray::try_new(
                field,
                offsets,
                entries.as_struct().clone(),
                nulls,
                *ordered,
            )?)
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

/// The copy of a view array that holds more buffer bytes than its views
/// reference; 0 if it holds no more.
fn view_copy<T: ByteViewType + ?Sized>(a: &GenericByteViewArray<T>) -> usize {
    let held: usize = a.data_buffers().iter().map(arrow_buffer::Buffer::len).sum();
    let used = a.total_buffer_bytes_used();
    if held <= used {
        return 0;
    }
    let validity = a.nulls().map_or(0, |_| a.len().div_ceil(8));
    a.len()
        .saturating_mul(VIEW_BYTES)
        .saturating_add(used)
        .saturating_add(validity)
}

fn list_copy<O: OffsetSizeTrait>(l: &GenericListArray<O>) -> usize {
    let (first, len) = value_range(l.value_offsets());
    let values = copy_bytes(&l.values().slice(first, len));
    if values == 0 {
        0
    } else {
        values + offsets_bytes::<O>(l.len())
    }
}

fn compact_list<O: OffsetSizeTrait>(l: &GenericListArray<O>) -> Result<ArrayRef, ArrowError> {
    let (offsets, first, len) = rebased(l.value_offsets());
    let values = l.values().slice(first, len);
    let values = compact(&values)?.unwrap_or(values);
    let (field, _, _, nulls) = l.clone().into_parts();
    Ok(Arc::new(GenericListArray::<O>::try_new(
        field, offsets, values, nulls,
    )?))
}

fn offsets_bytes<O: OffsetSizeTrait>(len: usize) -> usize {
    len.saturating_add(1).saturating_mul(size_of::<O>())
}

/// The first value and the number of values `offsets` cover.
fn value_range<O: OffsetSizeTrait>(offsets: &[O]) -> (usize, usize) {
    let first = offsets.first().map_or(0, |o| o.as_usize());
    let last = offsets.last().map_or(0, |o| o.as_usize());
    (first, last.saturating_sub(first))
}

/// `offsets` shifted to start at 0, and the value range they covered.
fn rebased<O: OffsetSizeTrait>(offsets: &[O]) -> (OffsetBuffer<O>, usize, usize) {
    let (first, len) = value_range(offsets);
    let base = offsets.first().copied().unwrap_or(O::zero());
    let shifted: ScalarBuffer<O> = offsets.iter().map(|&o| o - base).collect();
    // Shifting keeps the offsets monotonic and starting at 0.
    (OffsetBuffer::new(shifted), first, len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BinaryViewArray, Int32Array, LargeListArray, ListArray, StringViewArray};
    use arrow_schema::Field;

    /// Long values (not inlined in their views), `n` of them.
    fn long_strings(n: usize) -> StringViewArray {
        StringViewArray::from_iter_values((0..n).map(|i| format!("a value long enough {i:08}")))
    }

    fn held(a: &StringViewArray) -> usize {
        a.data_buffers().iter().map(arrow_buffer::Buffer::len).sum()
    }

    #[test]
    fn a_sliced_view_array_is_copied_down_to_its_referenced_bytes() {
        let a = long_strings(10_000);
        let whole: ArrayRef = Arc::new(a.clone());
        assert_eq!(copy_bytes(&whole), 0, "an unsliced array is compact");
        assert!(compact(&whole).unwrap().is_none());

        let slice: ArrayRef = Arc::new(a.slice(100, 10));
        let expect = 10 * VIEW_BYTES + slice.as_string_view().total_buffer_bytes_used();
        assert_eq!(copy_bytes(&slice), expect);
        let out = compact(&slice).unwrap().unwrap();
        let out = out.as_string_view();
        assert!(held(out) <= 10 * 40, "kept {} bytes", held(out));
        assert!(held(&a) > 100 * held(out));
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            slice.as_string_view().iter().collect::<Vec<_>>()
        );

        let bin = BinaryViewArray::from_iter_values((0..1000).map(|i| vec![7u8; 20 + i % 3]));
        let bin: ArrayRef = Arc::new(bin.slice(5, 2));
        let out = compact(&bin).unwrap().unwrap();
        assert_eq!(out.as_binary_view().value(0), bin.as_binary_view().value(0));
        assert!(
            out.as_binary_view()
                .data_buffers()
                .iter()
                .map(arrow_buffer::Buffer::len)
                .sum::<usize>()
                < 100
        );
    }

    #[test]
    fn nested_view_arrays_are_compacted() {
        let values: ArrayRef = Arc::new(long_strings(1000));
        let field = Arc::new(Field::new("item", DataType::Utf8View, true));
        let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(10, 100));
        let list = ListArray::new(
            Arc::clone(&field),
            offsets.clone(),
            Arc::clone(&values),
            None,
        );
        let large = LargeListArray::new(
            Arc::clone(&field),
            OffsetBuffer::from_lengths(std::iter::repeat_n(10, 100)),
            Arc::clone(&values),
            None,
        );
        let ints: ArrayRef = Arc::new(Int32Array::from_iter_values(0..100));
        let structs = StructArray::from(vec![
            (Arc::new(Field::new("i", DataType::Int32, false)), ints),
            (
                Arc::new(Field::new("s", DataType::Utf8View, false)),
                Arc::new(long_strings(100)) as ArrayRef,
            ),
        ]);
        let fixed = FixedSizeListArray::new(Arc::clone(&field), 10, Arc::clone(&values), None);
        let map_entries = StructArray::from(vec![
            (
                Arc::new(Field::new("keys", DataType::Utf8View, false)),
                Arc::clone(&values),
            ),
            (
                Arc::new(Field::new("values", DataType::Utf8View, true)),
                Arc::clone(&values),
            ),
        ]);
        let map = MapArray::new(
            Arc::new(Field::new(
                "entries",
                map_entries.data_type().clone(),
                false,
            )),
            offsets,
            map_entries,
            None,
            false,
        );
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(list),
            Arc::new(large),
            Arc::new(structs),
            Arc::new(fixed),
            Arc::new(map),
        ];
        for a in arrays {
            let slice = a.slice(40, 2);
            assert!(copy_bytes(&slice) > 0, "{}", a.data_type());
            let out = compact(&slice).unwrap().unwrap();
            assert_eq!(out.to_data(), slice.to_data(), "{}", a.data_type());
            assert_eq!(copy_bytes(&out), 0, "{} is compact", a.data_type());
            assert!(
                out.get_array_memory_size() * 10 < slice.get_array_memory_size(),
                "{}: {} vs {}",
                a.data_type(),
                out.get_array_memory_size(),
                slice.get_array_memory_size()
            );
        }
        let plain: ArrayRef = Arc::new(Int32Array::from_iter_values(0..10));
        assert_eq!(copy_bytes(&plain), 0);
        assert!(compact(&plain).unwrap().is_none());
    }
}
