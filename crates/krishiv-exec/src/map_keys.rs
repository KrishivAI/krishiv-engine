// SPDX-License-Identifier: Apache-2.0
//! Spark's `InsertMapSortInGroupingExpressions` for grouping and join keys: two
//! maps with the same entries are one key whatever order the entries were
//! built in, so a key column holding a map has the entries of every map
//! sorted by key before it is encoded, and the key that comes back out is
//! that sorted map (what Spark's `MapSort` returns).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, GenericListArray, MapArray, StructArray};
use arrow_row::{RowConverter, SortField};
use arrow_schema::DataType;
use arrow_select::take::take;

use crate::error::{Result, internal};

/// Whether a value of type `t` holds a map anywhere.
pub(crate) fn holds_map(t: &DataType) -> bool {
    match t {
        DataType::Map(..) => true,
        DataType::Struct(f) => f.iter().any(|f| holds_map(f.data_type())),
        DataType::List(f) | DataType::LargeList(f) => holds_map(f.data_type()),
        _ => false,
    }
}

/// `column` with the entries of every map in it sorted by key.
///
/// # Errors
/// `INTERNAL_ERROR` if the keys of a map cannot be row-encoded.
pub(crate) fn sort_maps(column: &ArrayRef) -> Result<ArrayRef> {
    if !holds_map(column.data_type()) {
        return Ok(Arc::clone(column));
    }
    match column.data_type() {
        DataType::Map(..) => sort_map(column.as_map()),
        DataType::Struct(fields) => {
            let s = column.as_struct();
            let children = s
                .columns()
                .iter()
                .map(sort_maps)
                .collect::<Result<Vec<_>>>()?;
            let out = StructArray::try_new(fields.clone(), children, s.nulls().cloned())
                .map_err(|e| internal(e.to_string()))?;
            Ok(Arc::new(out))
        }
        DataType::List(field) => {
            let l = column.as_list::<i32>();
            let values = sort_maps(l.values())?;
            let out = GenericListArray::<i32>::try_new(
                Arc::clone(field),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )
            .map_err(|e| internal(e.to_string()))?;
            Ok(Arc::new(out))
        }
        DataType::LargeList(field) => {
            let l = column.as_list::<i64>();
            let values = sort_maps(l.values())?;
            let out = GenericListArray::<i64>::try_new(
                Arc::clone(field),
                l.offsets().clone(),
                values,
                l.nulls().cloned(),
            )
            .map_err(|e| internal(e.to_string()))?;
            Ok(Arc::new(out))
        }
        _ => Ok(Arc::clone(column)),
    }
}

/// One map column with the entries of each map in key order.
fn sort_map(m: &MapArray) -> Result<ArrayRef> {
    let DataType::Map(field, _) = m.data_type() else {
        return Err(internal("a map array of another type".to_owned()));
    };
    let keys = m.keys();
    let rows = RowConverter::new(vec![SortField::new(keys.data_type().clone())])
        .and_then(|c| c.convert_columns(std::slice::from_ref(keys)))
        .map_err(|e| internal(e.to_string()))?;
    // A permutation of the entries that moves none out of its map.
    let mut order: Vec<u32> = (0..keys.len())
        .map(|i| u32::try_from(i).map_err(|_| internal("more than 2^32 map entries".to_owned())))
        .collect::<Result<_>>()?;
    for w in m.offsets().windows(2) {
        let (from, to) = (usize::try_from(w[0]), usize::try_from(w[1]));
        let (Ok(from), Ok(to)) = (from, to) else {
            return Err(internal("a negative map offset".to_owned()));
        };
        order[from..to].sort_by(|a, b| rows.row(*a as usize).cmp(&rows.row(*b as usize)));
    }
    let order = arrow_array::UInt32Array::from(order);
    let take_col =
        |c: &ArrayRef| take(c.as_ref(), &order, None).map_err(|e| internal(e.to_string()));
    let entries = m.entries();
    let columns = entries
        .columns()
        .iter()
        .map(|c| take_col(c).and_then(|c| sort_maps(&c)))
        .collect::<Result<Vec<_>>>()?;
    let DataType::Struct(entry_fields) = field.data_type() else {
        return Err(internal("map entries that are not a struct".to_owned()));
    };
    let entries = StructArray::try_new(entry_fields.clone(), columns, entries.nulls().cloned())
        .map_err(|e| internal(e.to_string()))?;
    let out = MapArray::try_new(
        Arc::clone(field),
        m.offsets().clone(),
        entries,
        m.nulls().cloned(),
        // The input's flag: the type must not change.
        matches!(m.data_type(), DataType::Map(_, true)),
    )
    .map_err(|e| internal(e.to_string()))?;
    Ok(Arc::new(out))
}

#[cfg(test)]
mod tests {
    use arrow_array::builder::{Int32Builder, MapBuilder};

    use super::*;

    fn maps(rows: &[Option<&[(i32, i32)]>]) -> ArrayRef {
        let mut b = MapBuilder::new(None, Int32Builder::new(), Int32Builder::new());
        for row in rows {
            match row {
                Some(entries) => {
                    for (k, v) in *entries {
                        b.keys().append_value(*k);
                        b.values().append_value(*v);
                    }
                    b.append(true).unwrap();
                }
                None => b.append(false).unwrap(),
            }
        }
        Arc::new(b.finish())
    }

    fn entries(a: &ArrayRef, row: usize) -> Vec<(i32, i32)> {
        let m = a.as_map();
        let v = m.value(row);
        let k = v.column(0).as_primitive::<arrow_array::types::Int32Type>();
        let x = v.column(1).as_primitive::<arrow_array::types::Int32Type>();
        (0..v.len()).map(|i| (k.value(i), x.value(i))).collect()
    }

    #[test]
    fn entries_come_out_in_key_order_and_nulls_stay() {
        let a = maps(&[Some(&[(2, 3), (1, 2)]), None, Some(&[]), Some(&[(1, 3)])]);
        let s = sort_maps(&a).unwrap();
        assert_eq!(entries(&s, 0), [(1, 2), (2, 3)]);
        assert!(s.is_null(1));
        assert_eq!(entries(&s, 2), []);
        assert_eq!(entries(&s, 3), [(1, 3)]);
        assert_eq!(s.data_type(), a.data_type());
    }

    #[test]
    fn the_keys_sorted_flag_of_the_input_is_kept() {
        let a = maps(&[Some(&[(2, 3), (1, 2)])]);
        let m = a.as_map();
        let DataType::Map(field, _) = m.data_type() else {
            panic!("a map");
        };
        let flagged: ArrayRef = Arc::new(
            MapArray::try_new(
                Arc::clone(field),
                m.offsets().clone(),
                m.entries().clone(),
                None,
                true,
            )
            .unwrap(),
        );
        let s = sort_maps(&flagged).unwrap();
        assert_eq!(s.data_type(), flagged.data_type());
        assert_eq!(entries(&s, 0), [(1, 2), (2, 3)]);
    }

    #[test]
    fn a_sliced_map_array_sorts_only_its_own_maps() {
        let a = maps(&[Some(&[(9, 9), (8, 8)]), Some(&[(2, 3), (1, 2)])]).slice(1, 1);
        let s = sort_maps(&a).unwrap();
        assert_eq!(entries(&s, 0), [(1, 2), (2, 3)]);
    }

    #[test]
    fn maps_inside_lists_and_structs_sort() {
        let m = maps(&[Some(&[(2, 3), (1, 2)])]);
        let item = Arc::new(arrow_schema::Field::new(
            "item",
            m.data_type().clone(),
            true,
        ));
        let list: ArrayRef = Arc::new(
            GenericListArray::<i32>::try_new(
                Arc::clone(&item),
                arrow_buffer::OffsetBuffer::from_lengths([1]),
                Arc::clone(&m),
                None,
            )
            .unwrap(),
        );
        let s = sort_maps(&list).unwrap();
        assert_eq!(entries(s.as_list::<i32>().values(), 0), [(1, 2), (2, 3)]);
        let st: ArrayRef = Arc::new(
            StructArray::try_new(
                vec![arrow_schema::Field::new("a", m.data_type().clone(), true)].into(),
                vec![m],
                None,
            )
            .unwrap(),
        );
        let s = sort_maps(&st).unwrap();
        assert_eq!(entries(s.as_struct().column(0), 0), [(1, 2), (2, 3)]);
    }

    #[test]
    fn a_column_without_maps_is_returned_as_is() {
        let a: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1]));
        assert!(Arc::ptr_eq(&sort_maps(&a).unwrap(), &a));
    }
}
