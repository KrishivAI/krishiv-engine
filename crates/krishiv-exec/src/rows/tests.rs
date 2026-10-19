// SPDX-License-Identifier: Apache-2.0
//! The row tables agree with the arrow-row [`KeyTable`] on which keys are
//! equal, for every supported key type, NULLs, NaN and `-0.0`, and short
//! and long strings.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryArray, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray,
};
use arrow_schema::DataType;
use proptest::prelude::*;

use super::*;
use crate::hash::{HashSeed, KeyEncoder, KeyTable};

/// Group ids from the arrow-row path: the reference.
fn reference(columns: &[ArrayRef]) -> Vec<u32> {
    let types: Vec<DataType> = columns.iter().map(|c| c.data_type().clone()).collect();
    let enc = KeyEncoder::new(&types).unwrap_or_else(|e| panic!("{e}"));
    let rows = enc.encode(columns).unwrap_or_else(|e| panic!("{e}"));
    let mut t = KeyTable::new();
    let mut ids = Vec::new();
    t.find_or_insert(&rows, &HashSeed::new(Some(7)), &mut ids)
        .unwrap_or_else(|e| panic!("{e}"));
    ids
}

/// Group ids (dense, first-seen order) from `index` over row pages, fed in
/// batches of `batch` rows; every key is then found again by `find`.
fn grouped(columns: &[ArrayRef], index: &mut dyn RowIndex, batch: usize) -> Vec<u32> {
    let types: Vec<DataType> = columns.iter().map(|c| c.data_type().clone()).collect();
    let nullable = vec![true; types.len()];
    let layout = RowLayout::new(&types, &nullable).unwrap_or_else(|e| panic!("{e}"));
    let mut pages = RowPages::new(&layout);
    let n = columns.first().map_or(0, |c| c.len());
    let mut all = Vec::new();
    let mut ids = Vec::new();
    for start in (0..n).step_by(batch.max(1)) {
        let len = batch.min(n - start);
        let slice: Vec<ArrayRef> = columns.iter().map(|c| c.slice(start, len)).collect();
        let keys = layout.encode(&slice, 11).unwrap_or_else(|e| panic!("{e}"));
        index
            .find_or_insert(&layout, &keys, &mut pages, &mut ids)
            .unwrap_or_else(|e| panic!("{e}"));
        all.extend_from_slice(&ids);
        let mut found = Vec::new();
        index.find(&layout, &keys, &pages, &mut found);
        assert_eq!(found, ids, "find disagrees with find_or_insert");
    }
    assert_eq!(index.len(), pages.len());
    assert!(index.memory_size() > 0 && !index.is_empty() || n == 0);
    densify(&all)
}

/// Renumbers ids densely in first-seen order.
fn densify(ids: &[u32]) -> Vec<u32> {
    let mut map = std::collections::HashMap::new();
    ids.iter()
        .map(|id| {
            let next = u32::try_from(map.len()).unwrap_or(u32::MAX);
            *map.entry(*id).or_insert(next)
        })
        .collect()
}

fn check(columns: &[ArrayRef]) {
    let want = reference(columns);
    for batch in [1, 7, 4096] {
        assert_eq!(
            grouped(columns, &mut SaltedTable::new(4), batch),
            want,
            "salted {batch}"
        );
        assert_eq!(
            grouped(columns, &mut HashbrownTable::new(), batch),
            want,
            "hashbrown {batch}"
        );
    }
}

fn strings(v: &[Option<String>]) -> [ArrayRef; 3] {
    let refs: Vec<Option<&str>> = v.iter().map(Option::as_deref).collect();
    [
        Arc::new(StringViewArray::from(refs.clone())),
        Arc::new(StringArray::from(refs.clone())),
        Arc::new(LargeStringArray::from(refs)),
    ]
}

#[test]
fn nulls_nan_and_signed_zero_group_like_spark() {
    let f: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(0.0),
        Some(-0.0),
        Some(f64::NAN),
        Some(-f64::NAN),
        None,
        None,
        Some(1.5),
    ]));
    let s = strings(&[
        Some("a".into()),
        Some("a".into()),
        Some("long string over twelve".into()),
        Some("long string over twelve".into()),
        None,
        None,
        Some(String::new()),
    ]);
    for s in s {
        check(&[Arc::clone(&f), s]);
    }
    let f32s: ArrayRef = Arc::new(Float32Array::from(vec![Some(-0.0f32), Some(0.0), None]));
    check(&[f32s]);
}

#[test]
fn string_carriers_hash_alike() {
    let v = vec![
        Some("x".to_owned()),
        Some("a much longer value".to_owned()),
        None,
    ];
    let [view, utf8, large] = strings(&v);
    let enc = |a: ArrayRef| {
        let layout =
            RowLayout::new(&[a.data_type().clone()], &[true]).unwrap_or_else(|e| panic!("{e}"));
        layout
            .encode(&[a], 3)
            .unwrap_or_else(|e| panic!("{e}"))
            .hashes()
            .to_vec()
    };
    let h = enc(view);
    assert_eq!(h, enc(utf8));
    assert_eq!(h, enc(large));
}

#[test]
fn every_fixed_type_and_misses() {
    let cols: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            Some(true),
            None,
        ])),
        Arc::new(Int16Array::from(vec![1, 1, 1, 1])),
        Arc::new(
            Decimal128Array::from(vec![Some(i128::MAX), Some(-1), Some(i128::MAX), None])
                .with_precision_and_scale(38, 2)
                .unwrap_or_else(|e| panic!("{e}")),
        ),
        Arc::new(BinaryArray::from(vec![
            Some(&b"\0"[..]),
            Some(&b""[..]),
            Some(&b"\0"[..]),
            None,
        ])),
    ];
    check(&cols);
    // A key never inserted is not found by either index.
    let layout = RowLayout::new(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let present: ArrayRef = Arc::new(Int64Array::from_iter_values(0..5000));
    let absent: ArrayRef = Arc::new(Int64Array::from_iter_values(10_000..10_100));
    let keys = layout
        .encode(&[present], 1)
        .unwrap_or_else(|e| panic!("{e}"));
    let probe = layout
        .encode(&[absent], 1)
        .unwrap_or_else(|e| panic!("{e}"));
    let indexes: [Box<dyn RowIndex>; 2] = [
        Box::new(SaltedTable::new(5)),
        Box::new(HashbrownTable::new()),
    ];
    for mut index in indexes {
        let mut pages = RowPages::new(&layout);
        let mut ids = Vec::new();
        assert_eq!(
            index
                .find_or_insert(&layout, &keys, &mut pages, &mut ids)
                .ok(),
            Some(5000)
        );
        let mut out = Vec::new();
        index.find(&layout, &probe, &pages, &mut out);
        assert!(out.iter().all(|&id| id == NO_ROW));
        assert_eq!(pages.ids().count(), 5000);
        assert_eq!(pages.hash(ids[0]), keys.hashes()[0]);
        assert!(pages.memory_size() >= 5000 * layout.stride());
    }
}

#[test]
fn layout_rejects_what_it_cannot_store() {
    let list = DataType::List(Arc::new(arrow_schema::Field::new(
        "i",
        DataType::Int32,
        true,
    )));
    assert!(RowLayout::new(&[list], &[true]).is_err());
    assert!(RowLayout::new(&[DataType::Int32], &[]).is_err());
    let layout = RowLayout::new(&[DataType::Int32], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let with_null: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
    assert!(layout.encode(&[Arc::clone(&with_null)], 0).is_err());
    assert!(layout.encode(&[], 0).is_err());
    let wrong: ArrayRef = Arc::new(Float64Array::from(vec![1.0]));
    let layout = RowLayout::new(&[DataType::Int32, DataType::Utf8], &[true, true])
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(layout.encode(&[Arc::clone(&with_null), wrong], 0).is_err());
    let empty = layout
        .encode(
            &[
                with_null.slice(0, 0),
                Arc::new(StringArray::from(Vec::<&str>::new())),
            ],
            0,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(empty.is_empty() && empty.row(3).is_empty());
    let pages = RowPages::new(&layout);
    assert!(pages.is_empty() && pages.row(99).0.is_empty());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random two-column keys (int + string, many duplicates and NULLs)
    /// group exactly as the arrow-row reference does.
    #[test]
    fn groups_match_the_arrow_row_reference(
        rows in proptest::collection::vec(
            (proptest::option::of(0i32..20), proptest::option::of(0usize..6)),
            0..600,
        )
    ) {
        let words = ["", "a", "abcdefghijkl", "abcdefghijklm", "same prefix 1", "same prefix 2"];
        let ints: ArrayRef = Arc::new(Int32Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>()));
        let strs: Vec<Option<String>> = rows.iter().map(|r| r.1.map(|i| words[i].to_owned())).collect();
        for s in strings(&strs) {
            check(&[Arc::clone(&ints), s]);
        }
    }

    /// Many distinct single-int keys across growth and many pages.
    #[test]
    fn growth_keeps_every_key(n in 0usize..20_000, seed in 0u64..4) {
        let a: ArrayRef = Arc::new(Int64Array::from_iter_values(
            (0..n as i64).map(|i| i.wrapping_mul(0x9E37_79B9) ^ i64::try_from(seed).unwrap_or(0)),
        ));
        let want = reference(&[Arc::clone(&a)]);
        let salted = grouped(&[Arc::clone(&a)], &mut SaltedTable::new(4), 4096);
        prop_assert_eq!(&salted, &want);
        prop_assert_eq!(grouped(&[a], &mut HashbrownTable::new(), 4096), want);
    }
}

#[test]
fn fixed_keys_need_their_exact_type() {
    // Same width, other type: no silent reinterpretation of the bytes.
    let layout = RowLayout::new(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let floats: ArrayRef = Arc::new(Float64Array::from(vec![-0.0, 0.0]));
    assert!(layout.encode(&[floats], 0).is_err());
    let dec = DataType::Decimal128(38, 2);
    let layout = RowLayout::new(&[dec], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let other_scale: ArrayRef = Arc::new(
        Decimal128Array::from(vec![1i128])
            .with_precision_and_scale(38, 3)
            .unwrap_or_else(|e| panic!("{e}")),
    );
    assert!(layout.encode(&[other_scale], 0).is_err());
}

#[test]
fn push_refuses_rows_of_another_layout_without_writing() {
    let wide = RowLayout::new(&[DataType::Int64, DataType::Int64], &[false, false])
        .unwrap_or_else(|e| panic!("{e}"));
    let narrow = RowLayout::new(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let a: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
    let keys = narrow.encode(&[a], 0).unwrap_or_else(|e| panic!("{e}"));
    let mut pages = RowPages::new(&wide);
    assert!(pages.push(&keys, 0).is_err(), "row of another stride");
    let mut pages = RowPages::new(&narrow);
    assert!(pages.push(&keys, 9).is_err(), "row index out of range");
    assert!(pages.is_empty() && pages.memory_size() == 0);
    assert_eq!(pages.push(&keys, 1).ok(), Some(0));
    assert_eq!(pages.hash(0), keys.hashes()[1]);
}

#[test]
fn a_new_page_reserves_a_few_rows_and_a_page_heap_stays_bounded() {
    let layout = RowLayout::new(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let a: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    let keys = layout.encode(&[a], 0).unwrap_or_else(|e| panic!("{e}"));
    let mut pages = RowPages::new(&layout);
    assert_eq!(pages.push(&keys, 0).ok(), Some(0));
    assert!(
        pages.memory_size() < 64 * 1024,
        "one row must not reserve a whole page: {}",
        pages.memory_size()
    );

    let layout = RowLayout::new(&[DataType::Utf8], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let big = "x".repeat(10 << 20);
    let a: ArrayRef = Arc::new(StringArray::from(vec![big.as_str(), big.as_str()]));
    let keys = layout.encode(&[a], 0).unwrap_or_else(|e| panic!("{e}"));
    let mut pages = RowPages::new(&layout);
    let (first, second) = (pages.push(&keys, 0), pages.push(&keys, 1));
    assert!(first.is_ok() && second.is_ok());
    // 2 x 10 MiB would exceed the 16 MiB page heap: the second row opens a page.
    assert_eq!(pages.row(second.unwrap_or(0)).1.len(), 10 << 20);

    let huge = "y".repeat(17 << 20);
    let a: ArrayRef = Arc::new(StringArray::from(vec![huge.as_str()]));
    let keys = layout.encode(&[a], 0).unwrap_or_else(|e| panic!("{e}"));
    let id = pages
        .push(&keys, 0)
        .unwrap_or_else(|e| panic!("oversized key must be accepted: {e}"));
    assert_eq!(pages.row(id).1.len(), 17 << 20);
    // The oversized row's page takes no further rows.
    let small: ArrayRef = Arc::new(StringArray::from(vec!["z".repeat(20)]));
    let small_keys = layout.encode(&[small], 0).unwrap_or_else(|e| panic!("{e}"));
    let next = pages.push(&small_keys, 0).unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(next >> pages.shift_for_test(), id >> pages.shift_for_test());
    assert_eq!(pages.row(id).1.len(), 17 << 20);
}

#[test]
fn a_wide_row_page_starts_within_the_page_byte_budget() {
    let types = vec![DataType::Int64; 1000];
    let layout = RowLayout::new(&types, &vec![false; 1000]).unwrap_or_else(|e| panic!("{e}"));
    let cols: Vec<ArrayRef> = (0..1000)
        .map(|_| Arc::new(Int64Array::from(vec![1])) as ArrayRef)
        .collect();
    let keys = layout.encode(&cols, 0).unwrap_or_else(|e| panic!("{e}"));
    let mut pages = RowPages::new(&layout);
    assert_eq!(pages.push(&keys, 0).ok(), Some(0));
    assert!(layout.stride() > 8000);
    assert!(
        pages.memory_size() <= 256 << 10,
        "first reservation {} exceeds one page budget",
        pages.memory_size()
    );
}
