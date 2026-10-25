// SPDX-License-Identifier: Apache-2.0
//! Malformed list offsets reach the list kernels as errors, never panics
//! or wrapped indices: a negative offset fails every higher-order reshape.

use std::sync::Arc;

use arrow_array::{Array, BooleanArray, Int32Array, ListArray};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use krishiv_kernels::list::{exists_list, filter_list, forall_list, transform_list};

fn field() -> Arc<Field> {
    Arc::new(Field::new("element", DataType::Int32, true))
}

/// One list over two values with a negative first offset. The checked
/// constructors refuse such offsets, so this uses the unchecked one: the
/// kernels must still reject the array.
fn malformed() -> ListArray {
    // SAFETY: the test deliberately builds an invalid offset buffer to
    // verify the kernels report it instead of wrapping or panicking; the
    // array is only passed to the fallible kernel entry points below.
    let offsets = unsafe { OffsetBuffer::new_unchecked(ScalarBuffer::from(vec![-1_i32, 2])) };
    let values = Arc::new(Int32Array::from(vec![1, 2]));
    ListArray::new(field(), offsets, values, None)
}

#[test]
fn negative_offsets_are_an_error() {
    let input = malformed();
    let keep = BooleanArray::from(vec![Some(true), Some(true), Some(true)]);
    assert!(filter_list(&input, &keep).is_err());
    assert!(exists_list(&input, &keep).is_err());
    assert!(forall_list(&input, &keep).is_err());
    let values: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 2]));
    assert!(transform_list(&input, &values, true).is_err());
}
