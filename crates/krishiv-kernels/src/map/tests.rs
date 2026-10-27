// SPDX-License-Identifier: Apache-2.0
//! Tests for the map builders: `create` packs rows, [`super::dedup_entries`]
//! keeps first-seen order (raising or keeping last per policy), and
//! [`super::union_keys`] merges key sets the way `map_zip_with` reads them.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, StringArray};

use super::*;
use arrow_array::cast::AsArray;
use krishiv_common::error::ErrorClass;

fn strings(values: Vec<&str>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

fn ints(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

#[test]
fn create_packs_one_map_per_row() {
    let keys = vec![strings(vec!["a", "b"]), strings(vec!["c", "d"])];
    let values = vec![ints(vec![1, 2]), ints(vec![3, 4])];
    let map = create(&keys, &values, 2, false, false).unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map.value_length(0), 2);
    assert_eq!(map.value_length(1), 2);
}

#[test]
fn create_rejects_null_keys() {
    let keys = vec![Arc::new(StringArray::from(vec![None::<&str>])) as ArrayRef];
    let values = vec![ints(vec![1])];
    let err = create(&keys, &values, 1, false, false).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::NullMapKey);
}

#[test]
fn create_dedups_per_policy() {
    let keys = || vec![strings(vec!["a", "x"]), strings(vec!["a", "y"])];
    let values = || vec![ints(vec![1, 2]), ints(vec![3, 4])];
    // `EXCEPTION` raises, naming the duplicate key.
    let err = create(&keys(), &values(), 2, false, false).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
    assert!(err.to_string().contains("key a was found"), "{err}");
    // `LAST_WIN` keeps the last value.
    let map = create(&keys(), &values(), 2, false, true).unwrap();
    assert_eq!(map.value_length(0), 1);
    assert_eq!(map.value_length(1), 2);
    assert_eq!(
        map.values()
            .as_primitive::<arrow_array::types::Int32Type>()
            .values(),
        &[3, 2, 4]
    );
}

#[test]
fn dedup_raises_on_duplicates() {
    let keys = strings(vec!["b", "a", "b"]);
    let err = dedup_entries(keys.as_ref(), 3, false).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::DuplicatedMapKey);
    // Names the duplicate key, quoting the policy.
    assert!(err.to_string().contains("key b was found"), "{err}");
}

#[test]
fn dedup_last_win_reroutes_values() {
    let keys = strings(vec!["b", "a", "b"]);
    let (kept, taken) = dedup_entries(keys.as_ref(), 3, true).unwrap();
    assert_eq!((kept, taken), (vec![0, 1], vec![2, 1]));
}

#[test]
fn union_merges_in_first_seen_order() {
    let (k1, k2) = (strings(vec!["a", "b"]), strings(vec!["b", "c"]));
    let m = union_keys(k1.as_ref(), 2, k2.as_ref(), 2).unwrap();
    // `a` only left, `b` both (first entries), `c` only right.
    assert_eq!(
        m,
        vec![
            (0, Some(0), None),
            (1, Some(1), Some(0)),
            (3, None, Some(1)),
        ]
    );
}
