// SPDX-License-Identifier: Apache-2.0
//! `KeyFilter::merge`: a superset of every input's keys, commutative and
//! associative in what passes, equal to the filter of the union where the
//! forms are exact (bit for bit for Bloom filters of the planned size), and
//! stable through the v1 encoding.
#![cfg(not(loom))]

use arrow_array::{Array, Int64Array, StringArray};
use arrow_schema::DataType;
use proptest::prelude::*;

use super::super::{FilterForm, KeyFilter, KeyFilterBuilder, KeyFilterOptions};

fn filter(keys: &dyn Array, options: KeyFilterOptions) -> KeyFilter {
    let mut b = KeyFilterBuilder::new(keys.data_type().clone(), options)
        .unwrap_or_else(|| panic!("unsupported {}", keys.data_type()));
    b.add(keys, None).unwrap_or_else(|e| panic!("{e}"));
    b.finish()
}

fn ints(keys: &[Option<i64>], options: KeyFilterOptions) -> KeyFilter {
    filter(&Int64Array::from(keys.to_vec()), options)
}

fn merge(a: &KeyFilter, b: &KeyFilter, options: &KeyFilterOptions) -> KeyFilter {
    a.merge(b, options).unwrap_or_else(|e| panic!("{e}"))
}

fn pass(f: &KeyFilter, column: &dyn Array) -> Vec<bool> {
    f.evaluate(column)
        .unwrap_or_else(|| panic!("{} does not widen", column.data_type()))
        .iter()
        .map(|b| b == Some(true))
        .collect()
}

fn round_trip(f: &KeyFilter) {
    let back = KeyFilter::decode(&f.encode()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(&back, f, "decode(encode(f)) == f");
}

/// Options that reach every form: small caps, a Bloom of a planned size.
fn options(max_values: usize, exact: usize, planned: Option<u32>) -> KeyFilterOptions {
    KeyFilterOptions {
        max_values,
        exact_max_values: exact,
        bloom_bytes: planned,
        bloom_max_fpp: 1.0,
        ..KeyFilterOptions::default()
    }
}

fn keys() -> impl Strategy<Value = Vec<Option<i64>>> {
    prop::collection::vec(prop::option::weighted(0.95, -300i64..300), 0..120)
}

fn opts() -> impl Strategy<Value = KeyFilterOptions> {
    (
        0usize..20,
        0usize..40,
        prop::option::of(prop::sample::select(vec![32u32, 64, 256])),
    )
        .prop_map(|(v, e, p)| options(v, e, p))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn merge_keeps_every_key_and_ignores_order(
        a in keys(), b in keys(), c in keys(), o in opts()
    ) {
        let (fa, fb, fc) = (ints(&a, o), ints(&b, o), ints(&c, o));
        let ab = merge(&fa, &fb, &o);
        let probe = Int64Array::from((-320i64..320).map(Some).chain([None]).collect::<Vec<_>>());
        let got = pass(&ab, &probe);
        for (k, p) in probe.iter().zip(&got) {
            if a.contains(&k) || b.contains(&k) {
                prop_assert!(*p, "{k:?} of an input dropped by the merge");
            }
        }
        prop_assert_eq!(&got, &pass(&merge(&fb, &fa, &o), &probe), "commutative");
        let left = merge(&ab, &fc, &o);
        let right = merge(&fa, &merge(&fb, &fc, &o), &o);
        prop_assert_eq!(pass(&left, &probe), pass(&right, &probe), "associative");
        prop_assert_eq!(left.key_count(), right.key_count());
        prop_assert_eq!(left.is_empty(), a.iter().chain(&b).chain(&c).all(Option::is_none));
        round_trip(&left);
    }

    #[test]
    fn merged_partitions_equal_the_filter_of_their_union(
        parts in prop::collection::vec(keys(), 1..5), o in opts()
    ) {
        let all: Vec<Option<i64>> = parts.iter().flatten().copied().collect();
        let whole = ints(&all, o);
        let merged = parts
            .iter()
            .map(|p| ints(p, o))
            .reduce(|x, y| merge(&x, &y, &o))
            .unwrap_or_else(|| panic!("no parts"));
        prop_assert_eq!(merged.range_scalars(), whole.range_scalars());
        prop_assert_eq!(merged.nulls_pass(), whole.nulls_pass());
        prop_assert_eq!(merged.key_count(), whole.key_count());
        let probe = Int64Array::from((-320i64..320).map(Some).collect::<Vec<_>>());
        // An exact set merged from the parts passes exactly the keys, as the
        // union's exact form or dense range does. (A merged range form can
        // be looser: a part's dense range absorbs the rest.)
        let set = |f: &KeyFilter| matches!(f.form(), FilterForm::SmallSet(_) | FilterForm::HashSet(_));
        if set(&merged) && !matches!(whole.form(), FilterForm::Bloom(_)) {
            prop_assert_eq!(pass(&merged, &probe), pass(&whole, &probe));
        }
        if let (FilterForm::Bloom(m), FilterForm::Bloom(w), Some(_)) =
            (merged.form(), whole.form(), o.bloom_bytes)
        {
            // A planned size: the OR of the partials' Blooms is the Bloom of
            // the union, bit for bit.
            prop_assert_eq!(m, w);
            prop_assert_eq!(&merged.member, &whole.member);
        }
    }
}

#[test]
fn blooms_of_different_sizes_fall_back_to_the_range() {
    let wide = |planned| options(0, 0, Some(planned));
    let a = ints(&(0..100).map(Some).collect::<Vec<_>>(), wide(64));
    let b = ints(&(200..300).map(Some).collect::<Vec<_>>(), wide(128));
    assert_eq!(a.form(), FilterForm::Bloom(64));
    assert_eq!(b.form(), FilterForm::Bloom(128));
    let m = merge(&a, &b, &wide(64));
    assert_eq!(m.form(), FilterForm::Range);
    // Unplanned, a Bloom sized for its own keys takes no other keys.
    let own = options(0, 0, None);
    let c = ints(&(0..100).map(Some).collect::<Vec<_>>(), own);
    assert!(matches!(c.form(), FilterForm::Bloom(_)));
    assert_eq!(merge(&c, &c, &own).form(), FilterForm::Range);
    assert!(pass(&m, &Int64Array::from(vec![150])).iter().all(|p| *p));
}

#[test]
fn a_bloom_past_its_false_positive_cap_falls_back_to_the_range() {
    let o = KeyFilterOptions {
        bloom_max_fpp: 0.05,
        ..options(0, 0, Some(32))
    };
    let lo = ints(&(0..20).map(Some).collect::<Vec<_>>(), o);
    let hi = ints(&(1_000..1_020).map(Some).collect::<Vec<_>>(), o);
    assert_eq!(lo.form(), FilterForm::Bloom(32));
    // One block of 32 bytes holding 40 keys passes most random hashes.
    assert_eq!(merge(&lo, &hi, &o).form(), FilterForm::Range);
}

#[test]
fn exact_sets_past_the_cap_become_a_bloom_of_the_planned_size_or_the_range() {
    let a = ints(
        &(0..30).map(|k| Some(k * 7)).collect::<Vec<_>>(),
        options(0, 40, None),
    );
    let b = ints(
        &(0..30).map(|k| Some(k * 11 + 1)).collect::<Vec<_>>(),
        options(0, 40, None),
    );
    assert!(matches!(a.form(), FilterForm::HashSet(30)));
    assert_eq!(
        merge(&a, &b, &options(0, 40, None)).form(),
        FilterForm::Range
    );
    let planned = options(0, 40, Some(1_024));
    let m = merge(&a, &b, &planned);
    assert_eq!(m.form(), FilterForm::Bloom(1_024));
    let probe: Vec<Option<i64>> = (0..30)
        .flat_map(|k| [Some(k * 7), Some(k * 11 + 1)])
        .collect();
    assert!(pass(&m, &Int64Array::from(probe)).iter().all(|p| *p));
    // Within the cap, the union stays exact.
    let u = merge(&a, &b, &options(0, 100, None));
    assert!(matches!(u.form(), FilterForm::HashSet(_)));
    assert_eq!(pass(&u, &Int64Array::from(vec![3])), vec![false]);
}

#[test]
fn the_in_list_unites_while_it_fits() {
    let a = ints(&[Some(1), Some(5)], options(3, 0, None));
    let b = ints(&[Some(5), Some(9)], options(3, 0, None));
    let m = merge(&a, &b, &options(3, 0, None));
    assert_eq!(m.value_count(), Some(3));
    assert_eq!(merge(&a, &b, &options(2, 0, None)).value_count(), None);
}

#[test]
fn an_empty_side_keeps_the_other_and_nulls_pass_from_either() {
    let o = KeyFilterOptions::default();
    let empty_nulls = ints(&[None], o);
    assert!(empty_nulls.is_empty());
    let some = ints(&[Some(4), Some(8)], o);
    let m = merge(&empty_nulls, &some, &o);
    assert!(!m.is_empty());
    assert!(m.nulls_pass(), "a NULL key of either side passes NULLs");
    assert_eq!(m.key_count(), 2);
    assert_eq!(
        pass(&m, &Int64Array::from(vec![Some(4), Some(5), None])),
        vec![true, false, true]
    );
    // Without NULL keys, a null-safe build's NULL does not pass: plain keys
    // ignore NULLs, so this builder saw none.
    let both_empty = merge(&ints(&[], o), &ints(&[], o), &o);
    assert!(both_empty.is_empty());
    assert!(!both_empty.nulls_pass());
}

#[test]
fn strings_merge_into_one_exact_set() {
    let o = KeyFilterOptions::default();
    let a = filter(&StringArray::from(vec!["ant", "bee"]), o);
    let b = filter(&StringArray::from(vec!["bee", "cat"]), o);
    let m = merge(&a, &b, &o);
    assert_eq!(m.form(), FilterForm::SmallSet(3));
    assert_eq!(
        pass(&m, &StringArray::from(vec!["ant", "cat", "dog", "bee"])),
        vec![true, true, false, true]
    );
    round_trip(&m);
}

#[test]
fn filters_of_different_key_types_do_not_merge() {
    let o = KeyFilterOptions::default();
    let a = ints(&[Some(1)], o);
    let b = filter(&StringArray::from(vec!["x"]), o);
    assert!(a.merge(&b, &o).is_err());
    assert_eq!(a.key_type(), &DataType::Int64);
}
