// SPDX-License-Identifier: Apache-2.0
//! The row tables agree with the arrow-row [`KeyTable`] on which keys are
//! equal, for every supported key type, NULLs, NaN and `-0.0`, and short
//! and long strings.

use std::sync::Arc;

use arrow_array::Array;
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
    let n = columns.first().map_or(0, Array::len);
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

/// Gathered build rows as a [`RowSource`] (what a join arena holds).
struct Gathered {
    rows: GatheredRows,
    hashes: Vec<u64>,
    stride: usize,
}

impl RowSource for Gathered {
    fn row(&self, id: u32) -> (&[u8], &[u8]) {
        let at = id as usize * self.stride;
        let row = self
            .rows
            .fixed
            .get(at..at + self.stride)
            .unwrap_or_default();
        let heap = self.rows.heap.as_ref().map_or(&[][..], |(o, b)| {
            let at = |i: usize| usize::try_from(o[i]).unwrap_or(0);
            &b[at(id as usize)..at(id as usize + 1)]
        });
        (row, heap)
    }

    fn hash(&self, id: u32) -> u64 {
        self.hashes[id as usize]
    }
}

/// Builds a [`JoinIndex`] over rows `idx` of `build` (a join stores only
/// rows whose `=` keys are not NULL) with `choice`, probes every row of
/// `probe`, and returns the heads found and the chain array (both in
/// positions of `idx`).
fn join_index(
    layout: &RowLayout,
    build: &[ArrayRef],
    idx: &[usize],
    probe: &[ArrayRef],
    choice: IndexChoice,
    touch: bool,
) -> (Vec<u32>, Vec<u32>, bool) {
    let b = layout.encode(build, 0).unwrap_or_else(|e| panic!("{e}"));
    let src = Gathered {
        rows: b.gather(layout, idx).unwrap_or_else(|e| panic!("{e}")),
        hashes: idx.iter().map(|&i| b.hashes()[i]).collect(),
        stride: layout.stride(),
    };
    let mut next = vec![NO_MATCH; idx.len()];
    let index = JoinIndex::build(layout, &src, idx.len(), 0xABCD, choice, &mut next)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(index.memory_size() > 0);
    assert!(index.memory_size() <= choice.bytes(idx.len()));
    let p = layout.encode(probe, 0).unwrap_or_else(|e| panic!("{e}"));
    let which: Vec<u32> = (0..u32::try_from(p.len()).unwrap_or(0)).collect();
    let mut heads = Vec::new();
    index.find_with(layout, &p, &which, &src, &mut heads, touch);
    assert_eq!(heads.len(), p.len());
    (heads, next, index.is_flat())
}

/// The chain of `head` through `next`.
fn chain(head: u32, next: &[u32]) -> Vec<u32> {
    std::iter::successors((head != NO_MATCH).then_some(head), |&r| {
        let n = next[r as usize];
        (n != NO_MATCH).then_some(n)
    })
    .collect()
}

/// Every probe row finds exactly the build rows the arrow-row reference
/// calls equal, newest first, whichever index kind (fast == slow).
fn check_join_index(build: &[ArrayRef], probe: &[ArrayRef], null_safe: &[bool]) {
    let types: Vec<DataType> = build.iter().map(|c| c.data_type().clone()).collect();
    let layout = RowLayout::for_join(&types, null_safe).unwrap_or_else(|e| panic!("{e}"));
    // The reference: group ids of build and probe rows together.
    let both: Vec<ArrayRef> = build
        .iter()
        .zip(probe)
        .map(|(b, p)| {
            arrow_select::concat::concat(&[b.as_ref(), p.as_ref()])
                .unwrap_or_else(|e| panic!("{e}"))
        })
        .collect();
    let groups = reference(&both);
    let nb = build.first().map_or(0, Array::len);
    // Rows with a NULL in an `=` key are never stored or probed by a join.
    let valid = |cols: &[ArrayRef], i: usize| {
        cols.iter()
            .zip(null_safe)
            .all(|(c, &safe)| safe || c.is_valid(i))
    };
    let stored: Vec<usize> = (0..nb).filter(|&i| valid(build, i)).collect();
    let (flat, flat_next, is_flat) =
        join_index(&layout, build, &stored, probe, IndexChoice::Flat, false);
    assert!(is_flat);
    for (choice, touch) in [
        (IndexChoice::Salted, false),
        (IndexChoice::Salted, true),
        (IndexChoice::Adaptive, false),
    ] {
        let (heads, next, flat_kind) = join_index(&layout, build, &stored, probe, choice, touch);
        assert_eq!(flat_kind, choice != IndexChoice::Salted);
        assert_eq!(heads, flat, "{choice:?} heads");
        assert_eq!(next, flat_next, "{choice:?} chains");
    }
    for (j, &head) in flat.iter().enumerate() {
        if !valid(probe, j) {
            continue;
        }
        let want: Vec<u32> = (0..stored.len())
            .rev()
            .filter(|&s| groups[stored[s]] == groups[nb + j])
            .map(|s| u32::try_from(s).unwrap_or(u32::MAX))
            .collect();
        assert_eq!(chain(head, &flat_next), want, "probe row {j}");
    }
}

#[test]
fn join_index_kinds_agree_on_spark_key_semantics() {
    let f = |v: Vec<Option<f64>>| -> ArrayRef { Arc::new(Float64Array::from(v)) };
    let build = f(vec![Some(0.0), Some(f64::NAN), None, Some(-0.0), Some(1.0)]);
    let probe = f(vec![Some(-0.0), Some(-f64::NAN), None, Some(2.0)]);
    for safe in [false, true] {
        check_join_index(
            std::slice::from_ref(&build),
            std::slice::from_ref(&probe),
            &[safe],
        );
    }
    // Empty build: every probe misses.
    let empty = f(vec![]);
    check_join_index(&[empty], &[probe], &[false]);
}

#[test]
fn a_large_build_takes_the_salted_directory() {
    let n = 1 << 20;
    assert!(flat_bytes(n) > FLAT_MAX_BYTES && salted_bytes(n) >= 16 * n);
    // Sizes saturate instead of overflowing.
    assert_eq!(salted_bytes(usize::MAX), usize::MAX);
    assert_eq!(flat_bytes(usize::MAX), usize::MAX);
    let a: ArrayRef = Arc::new(Int64Array::from_iter_values(
        0..i64::try_from(n).unwrap_or(0),
    ));
    let layout =
        RowLayout::for_join(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let probe: ArrayRef = Arc::new(Int64Array::from_iter_values([5, -1, 7]));
    let all: Vec<usize> = (0..n).collect();
    let (heads, _, flat) = join_index(
        &layout,
        &[Arc::clone(&a)],
        &all,
        &[Arc::clone(&probe)],
        IndexChoice::Adaptive,
        false,
    );
    assert!(!flat);
    assert_eq!(heads, [5, NO_MATCH, 7]);
    // `find` touches a directory this large first: the same answer.
    let b = layout.encode(&[a], 0).unwrap_or_else(|e| panic!("{e}"));
    let src = Gathered {
        rows: b.gather(&layout, &all).unwrap_or_else(|e| panic!("{e}")),
        hashes: b.hashes().to_vec(),
        stride: layout.stride(),
    };
    let mut next = vec![NO_MATCH; n];
    let index = JoinIndex::build(&layout, &src, n, 1, IndexChoice::Adaptive, &mut next)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(index.memory_size() >= TOUCH_MIN_BYTES);
    let p = layout.encode(&[probe], 0).unwrap_or_else(|e| panic!("{e}"));
    let mut found = Vec::new();
    index.find(&layout, &p, &[0, 1, 2], &src, &mut found);
    assert_eq!(found, [5, NO_MATCH, 7]);
}

#[test]
fn join_index_refuses_more_rows_than_chains() {
    let layout =
        RowLayout::for_join(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let src = Gathered {
        rows: GatheredRows {
            fixed: Vec::new(),
            heap: None,
        },
        hashes: Vec::new(),
        stride: layout.stride(),
    };
    let r = JoinIndex::build(&layout, &src, 3, 0, IndexChoice::Adaptive, &mut [0; 2]);
    assert!(r.is_err());
    let empty = JoinIndex::build(&layout, &src, 0, 0, IndexChoice::Salted, &mut []);
    let empty = empty.unwrap_or_else(|e| panic!("{e}"));
    assert!(empty.is_empty() && !empty.is_flat());
    assert_eq!(empty.len(), 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random int + string keys with duplicates and NULLs, `=` or `<=>`:
    /// the flat, salted and adaptive indexes return the same heads and
    /// chains, and exactly the reference's matches (fast == slow).
    #[test]
    fn join_index_kinds_match_the_arrow_row_reference(
        build in proptest::collection::vec(
            (proptest::option::of(0i64..12), proptest::option::of(0usize..6)),
            0..400,
        ),
        probe in proptest::collection::vec(
            (proptest::option::of(0i64..14), proptest::option::of(0usize..6)),
            0..200,
        ),
        safe in proptest::collection::vec(any::<bool>(), 2),
    ) {
        let words = ["", "a", "abcdefghijkl", "abcdefghijklm", "same prefix 1", "same prefix 2"];
        let cols = |rows: &[(Option<i64>, Option<usize>)]| -> Vec<[ArrayRef; 2]> {
            let ints: ArrayRef = Arc::new(Int64Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>()));
            let strs: Vec<Option<String>> = rows.iter().map(|r| r.1.map(|i| words[i].to_owned())).collect();
            strings(&strs).into_iter().map(|s| [Arc::clone(&ints), s]).collect()
        };
        for (b, p) in cols(&build).into_iter().zip(cols(&probe)) {
            check_join_index(&b, &p, &safe);
        }
    }
}

/// `decode` restores fixed-width keys (normalized floats, NULLs) for every
/// fixed type; a layout with views and a short row are errors.
#[test]
fn decode_restores_fixed_width_keys() {
    use arrow_array::{
        Date32Array, Decimal32Array, Decimal64Array, Int8Array, TimestampMicrosecondArray,
        UInt16Array, UInt64Array,
    };
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int8Array::from(vec![
            Some(-3),
            None,
            Some(i8::MAX),
            Some(0),
        ])),
        Arc::new(Int16Array::from(vec![
            Some(1),
            Some(-2),
            None,
            Some(i16::MIN),
        ])),
        Arc::new(UInt16Array::from(vec![
            Some(7),
            Some(u16::MAX),
            Some(0),
            None,
        ])),
        Arc::new(Int64Array::from(vec![
            None,
            Some(i64::MIN),
            Some(5),
            Some(-5),
        ])),
        Arc::new(UInt64Array::from(vec![
            Some(u64::MAX),
            None,
            Some(1),
            Some(2),
        ])),
        Arc::new(Date32Array::from(vec![
            Some(19_000),
            Some(-1),
            None,
            Some(0),
        ])),
        Arc::new(
            TimestampMicrosecondArray::from(vec![Some(1), None, Some(-7), Some(9)])
                .with_timezone("UTC"),
        ),
        Arc::new(
            Decimal32Array::from(vec![Some(12_345), None, Some(-1), Some(0)])
                .with_precision_and_scale(5, 1)
                .unwrap(),
        ),
        Arc::new(
            Decimal64Array::from(vec![Some(-99), Some(100), None, Some(1)])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ),
        Arc::new(
            Decimal128Array::from(vec![Some(i128::MAX / 7), None, Some(-3), Some(4)])
                .with_precision_and_scale(38, 3)
                .unwrap(),
        ),
        Arc::new(Float32Array::from(vec![
            Some(-0.0),
            Some(f32::NAN),
            None,
            Some(1.5),
        ])),
        Arc::new(Float64Array::from(vec![
            Some(f64::NAN),
            Some(-0.0),
            Some(2.5),
            None,
        ])),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
        ])),
    ];
    let types: Vec<DataType> = columns.iter().map(|c| c.data_type().clone()).collect();
    let mut nullable = vec![true; types.len()];
    let layout = RowLayout::new(&types, &nullable).unwrap();
    assert!(layout.is_fixed_width());
    let keys = layout.encode(&columns, 0).unwrap();
    // Rows in another order than encoded: decode follows the given rows.
    let order = [3, 0, 2, 1];
    let rows: Vec<&[u8]> = order.iter().map(|&i| keys.row(i)).collect();
    let decoded = layout.decode(&rows).unwrap();
    let take = arrow_array::UInt32Array::from(order.map(|i| u32::try_from(i).unwrap()).to_vec());
    for (c, d) in columns.iter().zip(&decoded) {
        let want = arrow_select::take::take(c, &take, None).unwrap();
        let want = match want.data_type() {
            // Stored normalized: -0.0 is 0.0, NaN is the canonical NaN.
            DataType::Float32 => {
                let v: Float32Array = want
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(|v| if v.is_nan() { f32::NAN } else { v + 0.0 }))
                    .collect();
                Arc::new(v) as ArrayRef
            }
            DataType::Float64 => {
                let v: Float64Array = want
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .map(|v| v.map(|v| if v.is_nan() { f64::NAN } else { v + 0.0 }))
                    .collect();
                Arc::new(v) as ArrayRef
            }
            _ => want,
        };
        assert_eq!(d.to_data(), want.to_data(), "{}", c.data_type());
    }
    // A non-nullable key decodes without a validity buffer.
    nullable[3] = false;
    let int64 = vec![Arc::new(Int64Array::from(vec![4, 5])) as ArrayRef];
    let one = RowLayout::new(&[DataType::Int64], &[false]).unwrap();
    let k = one.encode(&int64, 3).unwrap();
    let back = one.decode(&[k.row(1), k.row(0)]).unwrap();
    let want: ArrayRef = Arc::new(Int64Array::from(vec![5, 4]));
    assert_eq!(back[0].to_data(), want.to_data());
    // Strings may live in a heap: not decodable from rows alone.
    let views = RowLayout::new(&[DataType::Utf8], &[true]).unwrap();
    assert!(!views.is_fixed_width());
    let e = views.decode(&[]).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
    assert!(e.to_string().contains("Utf8"), "{e}");
    let e = one.decode(&[&[0u8; 8][..]]).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::InternalError
    );
    assert!(e.to_string().contains("8-byte key row"), "{e}");
}

/// Payload rows carry every supported type back exactly: floats bit for
/// bit (no normalization), NULLs, strings inline and on the heap; they
/// have no hash word, so a lone `Int64` payload takes 16 bytes.
#[test]
fn payload_rows_round_trip_exactly() {
    let ints: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(-7)]));
    let floats: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(-0.0),
        Some(f64::from_bits(0x7ff8_0000_0000_0001)),
        None,
    ]));
    let s = strings(&[
        Some("short".to_owned()),
        None,
        Some("a value longer than twelve bytes".to_owned()),
    ]);
    for strs in s {
        let cols = vec![Arc::clone(&ints), Arc::clone(&floats), strs];
        let types: Vec<DataType> = cols.iter().map(|c| c.data_type().clone()).collect();
        let layout = RowLayout::for_payload(&types).unwrap_or_else(|e| panic!("{e}"));
        let enc = layout.encode(&cols, 0).unwrap_or_else(|e| panic!("{e}"));
        let g = enc
            .gather(&layout, &[2, 0, 1])
            .unwrap_or_else(|e| panic!("{e}"));
        let stride = layout.stride();
        let heap_of = |j: usize| -> &[u8] {
            g.heap.as_ref().map_or(&[][..], |(o, b)| {
                let at = |k: usize| usize::try_from(o[k]).unwrap_or(0);
                &b[at(j)..at(j + 1)]
            })
        };
        let rows: Vec<(&[u8], &[u8])> = (0..3)
            .map(|j| (&g.fixed[j * stride..(j + 1) * stride], heap_of(j)))
            .collect();
        let back = layout.decode_rows(&rows).unwrap_or_else(|e| panic!("{e}"));
        for (c, b) in cols.iter().zip(&back) {
            let want = arrow_select::take::take(
                c.as_ref(),
                &arrow_array::UInt32Array::from(vec![2, 0, 1]),
                None,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(want.to_data(), b.to_data(), "{}", c.data_type());
        }
    }
    let one = RowLayout::for_payload(&[DataType::Int64]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(one.stride(), 16);
    let none = RowLayout::for_payload(&[DataType::Int32, DataType::Int32])
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(none.stride(), 16);
    let short = vec![(&[0u8; 3][..], &[][..])];
    assert!(one.decode_rows(&short).is_err());
}

/// An index sized for too few keys (a low estimate) grows from the stored
/// hashes and finds exactly what one sized for every row finds, for both
/// kinds; a high estimate only over-sizes.
#[test]
fn estimate_sized_indexes_grow_to_the_same_answer() {
    let n = 50_000usize;
    let a: ArrayRef = Arc::new(Int64Array::from_iter_values(
        (0..n).map(|i| i64::try_from(i % 20_000).unwrap_or(0)),
    ));
    let layout =
        RowLayout::for_join(&[DataType::Int64], &[false]).unwrap_or_else(|e| panic!("{e}"));
    let b = layout
        .encode(&[Arc::clone(&a)], 0)
        .unwrap_or_else(|e| panic!("{e}"));
    let all: Vec<usize> = (0..n).collect();
    let src = Gathered {
        rows: b.gather(&layout, &all).unwrap_or_else(|e| panic!("{e}")),
        hashes: b.hashes().to_vec(),
        stride: layout.stride(),
    };
    let probe = layout.encode(&[a], 0).unwrap_or_else(|e| panic!("{e}"));
    let which: Vec<u32> = (0..u32::try_from(n).unwrap_or(0)).collect();
    let run = |keys: usize, choice: IndexChoice| {
        let mut next = vec![NO_MATCH; n];
        let ix = JoinIndex::build_expecting(&layout, &src, n, keys, 9, choice, &mut next)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(ix.len(), 20_000);
        let mut heads = Vec::new();
        ix.find(&layout, &probe, &which, &src, &mut heads);
        (heads, next, ix.memory_size())
    };
    let exact = run(n, IndexChoice::Salted);
    for (keys, choice) in [
        (1, IndexChoice::Salted),
        (5_000, IndexChoice::Salted),
        (1, IndexChoice::Flat),
        (20_000, IndexChoice::Adaptive),
        (10 * n, IndexChoice::Adaptive),
    ] {
        let (heads, next, _) = run(keys, choice);
        assert_eq!(heads, exact.0, "{keys} {choice:?}");
        assert_eq!(next, exact.1, "{keys} {choice:?}");
    }
    // Sized by the estimate, the directory is smaller than by rows.
    assert!(run(20_000, IndexChoice::Salted).2 < exact.2);
}
