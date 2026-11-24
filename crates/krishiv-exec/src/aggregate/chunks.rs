// SPDX-License-Identifier: Apache-2.0
//! Concatenation of the value chunks an aggregate state keeps per group
//! (`collect_list`, `collect_set`, `listagg`, `mode`).
//!
//! A chunk of a string or binary view column shares its source's whole
//! list of data buffers (with the pinned Arrow major, a slice or a `take`
//! keeps the list as it is; the tests below pin that), so the chunks of
//! many groups mostly share a few lists. Arrow's `concat` appends every
//! input's list in full: `k` chunks of one `b`-buffer list give a child of
//! `k * b` buffers, which the next merge round multiplies again (a 2 MB
//! `collect_list` test reached 12 GB this way; none of it reserved, since a
//! state's reservation counts the values). Here each distinct list is added
//! once and the views are renumbered; a child whose buffers then hold more
//! than twice the bytes its values use (a spill run takes a subset of the
//! groups) is compacted, so spill files and exchanges carry only the values.
//! A struct, list, fixed-size list or map with a view type inside is
//! rebuilt from its children concatenated the same way, so a view leaf at
//! any depth keeps the bound.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{BinaryViewType, ByteViewType, StringViewType};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, GenericByteViewArray, GenericListArray, MapArray,
    OffsetSizeTrait, StructArray,
};
use arrow_buffer::{Buffer, NullBuffer, NullBufferBuilder, OffsetBuffer, ScalarBuffer};
use arrow_data::ByteView;
use arrow_schema::{DataType, FieldRef};
use arrow_select::concat::concat;

use crate::error::{Result, internal};

/// The longest value a view holds inline (no data buffer).
const MAX_INLINE: u32 = 12;

/// `chunks` concatenated in order, all of type `t` (an empty array of `t`
/// for none). View types share each distinct data-buffer list once, also
/// inside structs, lists and maps; other types go through Arrow's `concat`.
pub(crate) fn concat_chunks(chunks: &[ArrayRef], t: &DataType) -> Result<ArrayRef> {
    if let Some(c) = chunks.iter().find(|c| c.data_type() != t) {
        return Err(internal(format!(
            "an aggregate chunk of {} for {t}",
            c.data_type()
        )));
    }
    if chunks.is_empty() {
        return Ok(arrow_array::new_empty_array(t));
    }
    if !has_view(t) {
        if let [one] = chunks {
            return Ok(Arc::clone(one));
        }
        let refs: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
        return concat(&refs).map_err(|e| internal(e.to_string()));
    }
    match t {
        DataType::Utf8View | DataType::BinaryView if chunks.len() == 1 => {
            Ok(compacted(Arc::clone(&chunks[0])))
        }
        DataType::Utf8View => {
            concat_views::<StringViewType>(chunks).map(|a| compacted(Arc::new(a)))
        }
        DataType::BinaryView => {
            concat_views::<BinaryViewType>(chunks).map(|a| compacted(Arc::new(a)))
        }
        DataType::Struct(fields) => {
            let mut columns = Vec::with_capacity(fields.len());
            for (i, f) in fields.iter().enumerate() {
                let parts: Vec<ArrayRef> = chunks
                    .iter()
                    .map(|c| Arc::clone(c.as_struct().column(i)))
                    .collect();
                columns.push(concat_chunks(&parts, f.data_type())?);
            }
            let a = StructArray::try_new(fields.clone(), columns, nulls_of(chunks))
                .map_err(|e| internal(e.to_string()))?;
            Ok(Arc::new(a))
        }
        DataType::List(f) => concat_lists::<i32>(chunks, f),
        DataType::LargeList(f) => concat_lists::<i64>(chunks, f),
        DataType::FixedSizeList(f, size) => {
            let parts: Vec<ArrayRef> = chunks
                .iter()
                .map(|c| Arc::clone(c.as_fixed_size_list().values()))
                .collect();
            let values = concat_chunks(&parts, f.data_type())?;
            let a = FixedSizeListArray::try_new(Arc::clone(f), *size, values, nulls_of(chunks))
                .map_err(|e| internal(e.to_string()))?;
            Ok(Arc::new(a))
        }
        DataType::Map(f, ordered) => {
            let mut parts = Vec::with_capacity(chunks.len());
            let offsets = renumbered::<i32>(
                chunks.iter().map(|c| c.as_map().value_offsets()),
                |i, start, len| parts.push(chunks[i].as_map().entries().slice(start, len)),
            )?;
            let parts: Vec<ArrayRef> = parts.into_iter().map(|e| Arc::new(e) as ArrayRef).collect();
            let entries = concat_chunks(&parts, f.data_type())?;
            let a = MapArray::try_new(
                Arc::clone(f),
                offsets,
                entries.as_struct().clone(),
                nulls_of(chunks),
                *ordered,
            )
            .map_err(|e| internal(e.to_string()))?;
            Ok(Arc::new(a))
        }
        // A dictionary, union or list view with a view type inside: types
        // an aggregate state does not keep (`collect::normalize`).
        _ => {
            let refs: Vec<&dyn Array> = chunks.iter().map(AsRef::as_ref).collect();
            concat(&refs).map_err(|e| internal(e.to_string()))
        }
    }
}

/// Whether a string or binary view type is `t` or inside it.
fn has_view(t: &DataType) -> bool {
    match t {
        DataType::Utf8View | DataType::BinaryView => true,
        DataType::Struct(fields) => fields.iter().any(|f| has_view(f.data_type())),
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => has_view(f.data_type()),
        DataType::Dictionary(_, v) => has_view(v),
        _ => false,
    }
}

/// The lists of `chunks` (of element field `f`) over their values
/// concatenated with [`concat_chunks`].
fn concat_lists<O: OffsetSizeTrait>(chunks: &[ArrayRef], f: &FieldRef) -> Result<ArrayRef> {
    let mut parts = Vec::with_capacity(chunks.len());
    let offsets = renumbered::<O>(
        chunks.iter().map(|c| c.as_list::<O>().value_offsets()),
        |i, start, len| parts.push(chunks[i].as_list::<O>().values().slice(start, len)),
    )?;
    let values = concat_chunks(&parts, f.data_type())?;
    let a = GenericListArray::<O>::try_new(Arc::clone(f), offsets, values, nulls_of(chunks))
        .map_err(|e| internal(e.to_string()))?;
    Ok(Arc::new(a))
}

/// The offsets of lists whose per-chunk offsets are `each`, over their
/// value ranges laid end to end; `range(chunk, start, len)` is told each
/// chunk's range of its own values.
fn renumbered<'a, O: OffsetSizeTrait>(
    each: impl Iterator<Item = &'a [O]>,
    mut range: impl FnMut(usize, usize, usize),
) -> Result<OffsetBuffer<O>> {
    let overflow = || internal("an aggregate state past its list offsets".to_owned());
    let mut out: Vec<O> = vec![O::usize_as(0)];
    let mut base = 0_usize;
    for (i, offs) in each.enumerate() {
        let (Some(first), Some(last)) = (offs.first(), offs.last()) else {
            continue;
        };
        let start = first.as_usize();
        let len = last.as_usize().checked_sub(start).ok_or_else(overflow)?;
        for o in &offs[1..] {
            let at = o
                .as_usize()
                .checked_sub(start)
                .and_then(|d| base.checked_add(d))
                .and_then(O::from_usize)
                .ok_or_else(overflow)?;
            out.push(at);
        }
        range(i, start, len);
        base = base.checked_add(len).ok_or_else(overflow)?;
    }
    Ok(OffsetBuffer::new(ScalarBuffer::from(out)))
}

/// The validity of `chunks` laid end to end (`None` when every row is valid).
fn nulls_of(chunks: &[ArrayRef]) -> Option<NullBuffer> {
    let total: usize = chunks.iter().map(Array::len).sum();
    let mut nulls = NullBufferBuilder::new(total);
    for c in chunks {
        match c.nulls() {
            Some(n) => nulls.append_buffer(n),
            None => nulls.append_n_non_nulls(c.len()),
        }
    }
    nulls.finish()
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
    fn take_chunks_share_the_source_list_too() {
        // `CollectAcc::update` builds its chunks with `take`; if an Arrow
        // upgrade stops sharing the list there, this fails.
        let src = long_values("t", 600);
        let chunks: Vec<ArrayRef> = (0..300_u32)
            .map(|i| {
                let idx = arrow_array::UInt32Array::from(vec![i * 2, i * 2 + 1]);
                arrow_select::take::take(src.as_ref(), &idx, None).unwrap()
            })
            .collect();
        let out = concat_chunks(&chunks, &DataType::Utf8View).unwrap();
        assert_eq!(texts(&out), texts(&src));
        assert_eq!(
            out.as_string_view().data_buffers().len(),
            src.as_string_view().data_buffers().len()
        );
    }

    /// The buffers of every view leaf of `a`, at any depth.
    fn leaf_buffers(a: &dyn Array) -> usize {
        match a.data_type() {
            DataType::Utf8View => a.as_string_view().data_buffers().len(),
            DataType::Struct(_) => a
                .as_struct()
                .columns()
                .iter()
                .map(|c| leaf_buffers(c))
                .sum(),
            DataType::List(_) => leaf_buffers(a.as_list::<i32>().values()),
            DataType::LargeList(_) => leaf_buffers(a.as_list::<i64>().values()),
            DataType::FixedSizeList(..) => leaf_buffers(a.as_fixed_size_list().values()),
            DataType::Map(..) => leaf_buffers(a.as_map().entries()),
            _ => 0,
        }
    }

    /// Regroups `child` into one-row chunks four times, as merge rounds
    /// do, checking the values and that the view leaves keep at most the
    /// buffers they started with.
    fn rounds_stay_bounded(mut child: ArrayRef) {
        let bound = leaf_buffers(child.as_ref());
        let t = child.data_type().clone();
        let want = child.to_data();
        for round in 0..4 {
            let chunks: Vec<ArrayRef> = (0..child.len()).map(|i| child.slice(i, 1)).collect();
            child = concat_chunks(&chunks, &t).unwrap();
            assert_eq!(child.to_data(), want, "round {round} of {t}");
            let n = leaf_buffers(child.as_ref());
            assert!(n <= bound, "round {round} of {t}: {n} buffers");
        }
    }

    #[test]
    fn nested_view_leaves_keep_the_buffer_count_bounded() {
        let s = long_values("s", 300);
        let n = s.len();
        let ints: ArrayRef = Arc::new(Int64Array::from_iter(
            (0..n).map(|i| (i % 7 != 0).then(|| i64::try_from(i).unwrap())),
        ));
        let item = Arc::new(arrow_schema::Field::new("item", DataType::Utf8View, true));
        let fields = arrow_schema::Fields::from(vec![
            arrow_schema::Field::new("s", DataType::Utf8View, true),
            arrow_schema::Field::new("i", DataType::Int64, true),
        ]);
        let nulls = NullBuffer::from((0..n).map(|i| i % 11 != 0).collect::<Vec<_>>());
        let st: ArrayRef = Arc::new(
            StructArray::try_new(fields, vec![Arc::clone(&s), Arc::clone(&ints)], Some(nulls))
                .unwrap(),
        );
        rounds_stay_bounded(Arc::clone(&st));

        // A list of strings, three values a row (empty and NULL rows too).
        let offsets =
            OffsetBuffer::<i32>::from_lengths((0..100).map(|i| if i % 9 == 0 { 0 } else { 3 }));
        let used = usize::try_from(offsets.last()).unwrap();
        let list_nulls = NullBuffer::from((0..100).map(|i| i % 13 != 1).collect::<Vec<_>>());
        let list: ArrayRef = Arc::new(
            GenericListArray::<i32>::try_new(
                Arc::clone(&item),
                offsets,
                s.slice(0, used),
                Some(list_nulls),
            )
            .unwrap(),
        );
        rounds_stay_bounded(Arc::clone(&list));
        let large: ArrayRef = Arc::new(
            GenericListArray::<i64>::try_new(
                Arc::clone(&item),
                OffsetBuffer::<i64>::from_lengths((0..100).map(|_| 3)),
                Arc::clone(&s),
                None,
            )
            .unwrap(),
        );
        rounds_stay_bounded(large);

        // A list of structs and a fixed-size list.
        let st_field = Arc::new(arrow_schema::Field::new(
            "item",
            st.data_type().clone(),
            true,
        ));
        let list_of_structs: ArrayRef = Arc::new(
            GenericListArray::<i32>::try_new(
                st_field,
                OffsetBuffer::<i32>::from_lengths((0..150).map(|_| 2)),
                Arc::clone(&st),
                None,
            )
            .unwrap(),
        );
        rounds_stay_bounded(list_of_structs);
        let fixed: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(Arc::clone(&item), 3, Arc::clone(&s), None).unwrap(),
        );
        rounds_stay_bounded(fixed);

        // A map from strings to integers.
        let entries = StructArray::try_new(
            arrow_schema::Fields::from(vec![
                arrow_schema::Field::new("key", DataType::Utf8View, false),
                arrow_schema::Field::new("value", DataType::Int64, true),
            ]),
            vec![
                Arc::new(StringViewArray::from_iter_values(
                    (0..n).map(|i| format!("key-{i:06}-long-enough")),
                )) as ArrayRef,
                Arc::clone(&ints),
            ],
            None,
        )
        .unwrap();
        let entries_field = Arc::new(arrow_schema::Field::new(
            "entries",
            entries.data_type().clone(),
            false,
        ));
        let map: ArrayRef = Arc::new(
            MapArray::try_new(
                entries_field,
                OffsetBuffer::<i32>::from_lengths((0..100).map(|_| 3)),
                entries,
                None,
                false,
            )
            .unwrap(),
        );
        rounds_stay_bounded(map);

        // A subset of a large struct source is compacted at its leaf.
        let chunks: Vec<ArrayRef> = (0..3).map(|i| st.slice(i * 100, 1)).collect();
        let out = concat_chunks(&chunks, st.data_type()).unwrap();
        let leaf = out.as_struct().column(0).as_string_view().clone();
        let held: usize = leaf.data_buffers().iter().map(Buffer::len).sum();
        assert!(held <= 2 * leaf.total_buffer_bytes_used(), "{held}");
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
