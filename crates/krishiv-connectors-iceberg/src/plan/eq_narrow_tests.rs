// SPDX-License-Identifier: Apache-2.0
//! Tests of the equality-delete metrics check, Java's
//! `DeleteFileIndex.canContainEqDeletesForFile` (PR-25d step 2): each of its
//! rules, the types whose bounds are not compared, and the property that a
//! delete whose key equals a row of the data file is never skipped.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use proptest::prelude::*;

use super::deletes::{DeleteIndex, can_contain_eq_deletes};
use super::entry::{
    ContentFile, ContentKind, DataFileEntry, DeleteFileEntry, DeleteKind, EntryStatus,
};
use super::tests::file;
use super::types::{IcebergType, SchemaField};

const ID: i32 = 1;

fn types(ty: IcebergType) -> HashMap<i32, IcebergType> {
    HashMap::from([(ID, ty)])
}

/// A file whose column `ID` holds `values` (`None` is null), with honest
/// metrics: long bounds of the non-null values, value and null counts.
fn long_file(values: &[Option<i64>]) -> ContentFile {
    let mut f = file(values.len() as u64);
    let present: Vec<i64> = values.iter().flatten().copied().collect();
    if let (Some(lo), Some(hi)) = (present.iter().min(), present.iter().max()) {
        f.lower_bounds.insert(ID, lo.to_le_bytes().to_vec());
        f.upper_bounds.insert(ID, hi.to_le_bytes().to_vec());
    }
    f.value_counts.insert(ID, values.len() as u64);
    f.null_counts
        .insert(ID, values.iter().filter(|v| v.is_none()).count() as u64);
    f
}

fn deletes(mut f: ContentFile) -> ContentFile {
    f.content = ContentKind::EqualityDeletes;
    f.equality_ids = Some(vec![ID]);
    f
}

fn may(data: &ContentFile, delete: &ContentFile) -> bool {
    can_contain_eq_deletes(&types(IcebergType::Long), data, delete)
}

#[test]
fn disjoint_ranges_skip_the_delete() {
    let data = long_file(&[Some(10), Some(20)]);
    assert!(!may(&data, &deletes(long_file(&[Some(1), Some(9)]))));
    assert!(!may(&data, &deletes(long_file(&[Some(21)]))));
    assert!(may(&data, &deletes(long_file(&[Some(5), Some(10)]))));
    assert!(may(&data, &deletes(long_file(&[Some(20), Some(30)]))));
    // Overlapping ranges keep a delete whose keys fall between the rows.
    assert!(may(&data, &deletes(long_file(&[Some(15)]))));
}

#[test]
fn null_counts_decide_all_null_columns() {
    let all_null = long_file(&[None, None]);
    let no_null = long_file(&[Some(1)]);
    // Data all null, deletes hold no null.
    assert!(!may(&all_null, &deletes(no_null.clone())));
    // Deletes all null, data holds no null.
    assert!(!may(&no_null, &deletes(all_null)));
    // Both hold a null: the null key deletes rows, whatever the ranges.
    let data = long_file(&[None, Some(100)]);
    assert!(may(&data, &deletes(long_file(&[None, Some(1)]))));
    // Unknown null counts may hold a null.
    let mut unknown = long_file(&[Some(100)]);
    unknown.null_counts.clear();
    unknown.value_counts.clear();
    let mut d = deletes(long_file(&[Some(1)]));
    d.null_counts.clear();
    assert!(may(&unknown, &d));
}

#[test]
fn missing_bounds_or_fields_keep_the_delete() {
    let data = long_file(&[Some(10)]);
    let mut d = deletes(long_file(&[Some(1)]));
    d.upper_bounds.clear();
    assert!(may(&data, &d));
    // A field the schema does not have (dropped): no check at all.
    let d = deletes(long_file(&[Some(1)]));
    assert!(can_contain_eq_deletes(&HashMap::new(), &data, &d));
    // A delete without equality ids is offered.
    let mut d = d;
    d.equality_ids = None;
    assert!(may(&data, &d));
}

#[test]
fn inexact_bound_types_are_not_compared() {
    let float = |v: f64| v.to_le_bytes().to_vec();
    let with = |lo: Vec<u8>, hi: Vec<u8>| {
        let mut f = file(1);
        f.lower_bounds.insert(ID, lo);
        f.upper_bounds.insert(ID, hi);
        f.value_counts.insert(ID, 1);
        f.null_counts.insert(ID, 0);
        f
    };
    // Bounds leave NaN out: a NaN key may match NaN rows.
    let data = with(float(1.0), float(2.0));
    let d = deletes(with(float(5.0), float(5.0)));
    assert!(can_contain_eq_deletes(
        &types(IcebergType::Double),
        &data,
        &d
    ));
    // A string upper bound may be a prefix truncated without rounding up.
    let data = with(b"a".to_vec(), b"abc".to_vec());
    let d = deletes(with(b"abcd".to_vec(), b"abcd".to_vec()));
    assert!(can_contain_eq_deletes(
        &types(IcebergType::String),
        &data,
        &d
    ));
    assert!(can_contain_eq_deletes(
        &types(IcebergType::Binary),
        &data,
        &d
    ));
    let data = with(vec![0; 16], vec![1; 16]);
    let d = deletes(with(vec![9; 16], vec![9; 16]));
    assert!(can_contain_eq_deletes(&types(IcebergType::Uuid), &data, &d));
}

/// An `int` promoted to `long`: the old data file's 4-byte bounds compare
/// with the delete's 8-byte ones.
#[test]
fn promoted_bounds_compare_at_the_table_type() {
    let mut data = file(1);
    data.lower_bounds.insert(ID, 10i32.to_le_bytes().to_vec());
    data.upper_bounds.insert(ID, 20i32.to_le_bytes().to_vec());
    data.null_counts.insert(ID, 0);
    assert!(!may(&data, &deletes(long_file(&[Some(30)]))));
    assert!(may(&data, &deletes(long_file(&[Some(15)]))));
}

/// Through the index: a disjoint global equality delete is not assigned once
/// the schema's fields are given, and is without them.
#[test]
fn the_index_skips_disjoint_equality_deletes_with_fields() {
    let entry = |f: ContentFile, seq: i64| DataFileEntry {
        file: f,
        status: EntryStatus::Added,
        snapshot_id: 1,
        data_sequence_number: Some(seq),
        file_sequence_number: Some(seq),
    };
    let mut d = deletes(long_file(&[Some(1)]));
    d.path = "eq.parquet".to_owned();
    let d = Arc::new(DeleteFileEntry {
        entry: entry(d, 2),
        kind: DeleteKind::EqualityDelete,
    });
    let specs = BTreeMap::from([(0, super::types::PartitionSpecInfo::default())]);
    let data = entry(long_file(&[Some(10)]), 1);
    let plain = DeleteIndex::new(std::slice::from_ref(&d), &specs).unwrap();
    assert_eq!(
        plain.for_data_file(&data).unwrap().equality_deletes.len(),
        1
    );
    let fields = [SchemaField {
        id: ID,
        path: krishiv_io::ColumnPath::new("id"),
        ty: IcebergType::Long,
    }];
    let narrowed = DeleteIndex::new(&[d], &specs).unwrap().with_fields(&fields);
    assert!(
        narrowed
            .for_data_file(&data)
            .unwrap()
            .equality_deletes
            .is_empty()
    );
}

proptest! {
    /// A delete file holding a key equal to a value of the data file (null
    /// equals null) is never skipped, whichever metrics either file lacks.
    #[test]
    fn a_matching_delete_is_never_skipped(
        values in prop::collection::vec(prop::option::of(-20..20i64), 1..8),
        keys in prop::collection::vec(prop::option::of(-20..20i64), 1..5),
        drop in prop::collection::vec(any::<bool>(), 6),
    ) {
        let mut data = long_file(&values);
        let mut delete = deletes(long_file(&keys));
        let strip = |f: &mut ContentFile, d: &[bool]| {
            if d[0] { f.lower_bounds.clear(); }
            if d[1] { f.upper_bounds.clear(); }
            if d[2] { f.null_counts.clear(); }
        };
        strip(&mut data, &drop[..3]);
        strip(&mut delete, &drop[3..]);
        let matches = keys.iter().any(|k| values.contains(k));
        if matches {
            prop_assert!(may(&data, &delete), "{:?} {:?} {:?}", values, keys, drop);
        }
    }
}
