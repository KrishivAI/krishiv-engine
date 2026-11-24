// SPDX-License-Identifier: Apache-2.0
//! Concatenation of the value chunks an aggregate state keeps per group
//! (`collect_list`, `collect_set`, `listagg`, `mode`).
//!
//! A chunk of a string or binary view column shares its source's whole
//! list of data buffers (a slice or a `take` keeps the list as it is), so
//! the chunks of many groups mostly share a few lists. Arrow's `concat`
//! appends every input's list in full: `k` chunks of one `b`-buffer list
//! give a child of `k * b` buffers, which the next merge round multiplies
//! again (a 2 MB `collect_list` test reached 12 GB this way; none of it
//! reserved, since a state's reservation counts the values). Here each
//! distinct list is added once and the views are renumbered; a child whose
//! buffers then hold more than twice the bytes its values use (a spill run
//! takes a subset of the groups) is compacted, so spill files and
//! exchanges carry only the values.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{BinaryViewType, ByteViewType, StringViewType};
use arrow_array::{Array, ArrayRef, GenericByteViewArray};
use arrow_buffer::{Buffer, NullBufferBuilder, ScalarBuffer};
use arrow_data::ByteView;
use arrow_schema::DataType;
use arrow_select::concat::concat;

use crate::error::{Result, internal};

/// The longest value a view holds inline (no data buffer).
const MAX_INLINE: u32 = 12;

/// `chunks` concatenated in order, all of type `t` (an empty array of `t`
/// for none). View types share each distinct data-buffer list once; other
/// types go through Arrow's `concat`.
pub(crate) fn concat_chunks(chunks: &[ArrayRef], t: &DataType) -> Result<ArrayRef> {
    if let Some(c) = chunks.iter().find(|c| c.data_type() != t) {
        return Err(internal(format!(
            "an aggregate chunk of {} for {t}",
            c.data_type()
        )));
    }
    match (t, chunks) {
        (_, []) => Ok(arrow_array::new_empty_array(t)),
        (_, [one]) => Ok(compacted(Arc::clone(one))),
        (DataType::Utf8View, _) => {
            concat_views::<StringViewType>(chunks).map(|a| compacted(Arc::new(a)))
        }
        (DataType::BinaryView, _) => {
            concat_views::<BinaryViewType>(chunks).map(|a| compacted(Arc::new(a)))
        }
        _ => {
            let refs: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
            concat(&refs).map_err(|e| internal(e.to_string()))
        }
    }
}

/// The views of `chunks` over the union of their distinct buffer lists
/// (a list is known by its address: chunks of one source share it).
fn concat_views<T: ByteViewType>(chunks: &[ArrayRef]) -> Result<GenericByteViewArray<T>> {
    let total: usize = chunks.iter().map(Array::len).sum();
    let mut views: Vec<u128> = Vec::with_capacity(total);
    let mut nulls = NullBufferBuilder::new(total);
    let mut buffers: Vec<Buffer> = Vec::new();
    let mut bases: HashMap<usize, u32> = HashMap::new();
    for c in chunks {
        let a = c.as_byte_view::<T>();
        let list = a.data_buffers();
        let base = if list.is_empty() {
            0
        } else if let Some(&b) = bases.get(&(list.as_ptr() as usize)) {
            b
        } else {
            let b = u32::try_from(buffers.len())
                .map_err(|_| internal("an aggregate state past 2^32 buffers".to_owned()))?;
            buffers.extend(list.iter().cloned());
            bases.insert(list.as_ptr() as usize, b);
            b
        };
        for &v in a.views() {
            let mut view = ByteView::from(v);
            if view.length > MAX_INLINE {
                view.buffer_index = view
                    .buffer_index
                    .checked_add(base)
                    .ok_or_else(|| internal("an aggregate state past 2^32 buffers".to_owned()))?;
            }
            views.push(view.as_u128());
        }
        match a.nulls() {
            Some(n) => nulls.append_buffer(n),
            None => nulls.append_n_non_nulls(a.len()),
        }
    }
    GenericByteViewArray::<T>::try_new(ScalarBuffer::from(views), buffers, nulls.finish())
        .map_err(|e| internal(e.to_string()))
}

/// `a` compacted (its values copied into fresh buffers) when it is a view
/// array whose buffers hold more than twice the bytes its values use.
fn compacted(a: ArrayRef) -> ArrayRef {
    fn wasteful<T: ByteViewType>(a: &GenericByteViewArray<T>) -> bool {
        let held: usize = a.data_buffers().iter().map(Buffer::len).sum();
        held > a.total_buffer_bytes_used().saturating_mul(2)
    }
    match a.data_type() {
        DataType::Utf8View if wasteful(a.as_string_view()) => {
            Arc::new(a.as_string_view().gc()) as ArrayRef
        }
        DataType::BinaryView if wasteful(a.as_binary_view()) => {
            Arc::new(a.as_binary_view().gc()) as ArrayRef
        }
        _ => a,
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, StringViewArray};

    use super::*;

    /// A view column of `n` long values `prefix-<i>` (one data buffer).
    fn long_values(prefix: &str, n: usize) -> ArrayRef {
        Arc::new(StringViewArray::from_iter((0..n).map(|i| {
            (i % 5 != 0).then(|| format!("{prefix}-{i:06}-long-enough"))
        })))
    }

    fn texts(a: &ArrayRef) -> Vec<Option<String>> {
        a.as_string_view()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect()
    }

    #[test]
    fn chunks_of_one_source_share_its_buffers_once() {
        let src = long_values("a", 1000);
        let chunks: Vec<ArrayRef> = (0..1000).map(|i| src.slice(i, 1)).collect();
        let out = concat_chunks(&chunks, &DataType::Utf8View).unwrap();
        assert_eq!(texts(&out), texts(&src));
        let a = out.as_string_view();
        // Arrow's concat would hold 1000 copies of the source's list.
        assert_eq!(
            a.data_buffers().len(),
            src.as_string_view().data_buffers().len()
        );
    }

    #[test]
    fn repeated_rounds_keep_the_buffer_count_bounded() {
        // Each round regroups the previous round's child, as a merge of
        // spilled states does; with Arrow's concat the list grows
        // multiplicatively.
        let mut child = concat_chunks(
            &[long_values("a", 400), long_values("b", 400)],
            &DataType::Utf8View,
        )
        .unwrap();
        let want = texts(&child);
        for _ in 0..4 {
            let chunks: Vec<ArrayRef> = (0..child.len()).map(|i| child.slice(i, 1)).collect();
            child = concat_chunks(&chunks, &DataType::Utf8View).unwrap();
            assert_eq!(texts(&child), want);
            assert!(child.as_string_view().data_buffers().len() <= 2);
        }
    }

    #[test]
    fn a_subset_of_a_large_source_is_compacted() {
        let src = long_values("a", 1000);
        let chunks: Vec<ArrayRef> = (0..10).map(|i| src.slice(i * 100, 1)).collect();
        let out = concat_chunks(&chunks, &DataType::Utf8View).unwrap();
        let a = out.as_string_view();
        let held: usize = a.data_buffers().iter().map(Buffer::len).sum();
        assert!(held <= 2 * a.total_buffer_bytes_used(), "{held}");
        let want: Vec<Option<String>> = chunks.iter().flat_map(texts).collect();
        assert_eq!(texts(&out), want);
    }

    #[test]
    fn inline_values_nulls_and_binary_views() {
        let short: ArrayRef = Arc::new(StringViewArray::from(vec![Some("x"), None, Some("yz")]));
        let long = long_values("c", 6);
        let out = concat_chunks(
            &[Arc::clone(&short), Arc::clone(&long), short.slice(1, 2)],
            &DataType::Utf8View,
        )
        .unwrap();
        let mut want = texts(&short);
        want.extend(texts(&long));
        want.extend(texts(&short.slice(1, 2)));
        assert_eq!(texts(&out), want);
        assert_eq!(
            out.null_count(),
            want.iter().filter(|v| v.is_none()).count()
        );

        let bin: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from_iter(
            (0..20).map(|i| Some(format!("binary-value-{i:04}").into_bytes())),
        ));
        let parts: Vec<ArrayRef> = (0..20).map(|i| bin.slice(i, 1)).collect();
        let out = concat_chunks(&parts, &DataType::BinaryView).unwrap();
        assert_eq!(
            out.as_binary_view().iter().collect::<Vec<_>>(),
            bin.as_binary_view().iter().collect::<Vec<_>>()
        );
        assert_eq!(out.as_binary_view().data_buffers().len(), 1);
    }

    #[test]
    fn other_types_concatenate_and_mismatches_are_refused() {
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let out = concat_chunks(&[ints.slice(0, 1), ints.slice(2, 1)], &DataType::Int64).unwrap();
        assert_eq!(
            out.as_primitive::<arrow_array::types::Int64Type>().values(),
            &[1, 3]
        );
        assert_eq!(concat_chunks(&[], &DataType::Int64).unwrap().len(), 0);
        assert!(concat_chunks(&[ints], &DataType::Utf8View).is_err());
    }
}
