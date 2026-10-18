// SPDX-License-Identifier: Apache-2.0
//! Hash kernels: values recorded from Spark 4.2, the shapes the Spark
//! fixtures do not reach (sliced and large-offset arrays, interval types,
//! operand length errors), and property tests of the layout invariants.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, BinaryArray, Date32Array, DictionaryArray, DurationMicrosecondArray, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, IntervalYearMonthArray,
    LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, NullArray, StringArray,
    cast::AsArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::Operand;
use krishiv_kernels::hash::{crc32, hash, md5, sha1, sha2, xxhash64};
use proptest::prelude::*;

fn h(cols: &[&dyn Array]) -> Vec<i32> {
    let rows = cols.first().map_or(1, |c| c.len());
    hash(cols, rows).unwrap().values().to_vec()
}

fn x(cols: &[&dyn Array]) -> Vec<i64> {
    let rows = cols.first().map_or(1, |c| c.len());
    xxhash64(cols, rows).unwrap().values().to_vec()
}

#[test]
fn values_recorded_from_spark() {
    // hash(0), hash(1), hash(NULL) is the seed; xxhash64 likewise.
    let ints = Int32Array::from(vec![Some(0), Some(1), None]);
    assert_eq!(h(&[&ints]), [933_211_791, -559_580_957, 42]);
    assert_eq!(
        x(&[&ints]),
        [3_614_696_996_920_510_707, -6_698_625_589_789_238_999, 42]
    );
    // No columns: every row is the seed.
    assert_eq!(hash(&[], 3).unwrap().values().to_vec(), [42, 42, 42]);
    assert_eq!(xxhash64(&[], 2).unwrap().values().to_vec(), [42, 42]);
    // -0.0 hashes as 0.0 and every NaN alike.
    let f = Float64Array::from(vec![
        0.0,
        -0.0,
        f64::NAN,
        f64::from_bits(0x7ff8_0000_0000_0001),
    ]);
    let hf = h(&[&f]);
    assert_eq!((hf[0], hf[2]), (hf[1], hf[3]));
    let f = Float32Array::from(vec![0.0, -0.0, f32::NAN, f32::from_bits(0x7fc0_0001)]);
    let hf = x(&[&f]);
    assert_eq!((hf[0], hf[2]), (hf[1], hf[3]));
}

#[test]
fn narrow_types_hash_as_int_and_wide_as_long() {
    let (a, b, c) = (
        Int8Array::from(vec![-3]),
        Int16Array::from(vec![-3]),
        Int32Array::from(vec![-3]),
    );
    assert_eq!(h(&[&a]), h(&[&c]));
    assert_eq!(h(&[&b]), h(&[&c]));
    assert_eq!(x(&[&a]), x(&[&c]));
    let date = Date32Array::from(vec![-3]);
    let months = IntervalYearMonthArray::from(vec![-3]);
    assert_eq!(h(&[&date]), h(&[&c]));
    assert_eq!(h(&[&months]), h(&[&c]));
    let long = Int64Array::from(vec![-3]);
    let micros = DurationMicrosecondArray::from(vec![-3]);
    assert_eq!(h(&[&micros]), h(&[&long]));
    assert_ne!(h(&[&long]), h(&[&c]));
}

#[test]
fn chains_columns_and_skips_nulls() {
    let a = Int32Array::from(vec![Some(1), None, Some(1)]);
    let b = StringArray::from(vec![Some("a"), Some("a"), None]);
    let all = h(&[&a, &b]);
    // hash(NULL, 'a') == hash('a'); hash(1, NULL) == hash(1).
    assert_eq!(all[1], h(&[&b])[1]);
    assert_eq!(all[2], h(&[&a])[2]);
    let nulls = NullArray::new(3);
    assert_eq!(h(&[&nulls, &a]), h(&[&a]));
}

#[test]
fn string_and_binary_layouts_agree() {
    let s = StringArray::from(vec!["héllo", ""]);
    let ls = LargeStringArray::from(vec!["héllo", ""]);
    let b = BinaryArray::from(vec!["héllo".as_bytes(), b""]);
    let lb = LargeBinaryArray::from(vec!["héllo".as_bytes(), b""]);
    let want = (h(&[&s]), x(&[&s]));
    for other in [&ls as &dyn Array, &b, &lb] {
        assert_eq!((h(&[other]), x(&[other])), want);
    }
}

#[test]
fn sliced_and_large_lists_hash_alike() {
    let values = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let item = Arc::new(Field::new("item", DataType::Int32, true));
    let list = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::from_lengths([2, 0, 3, 1]),
        Arc::new(values.clone()),
        None,
    );
    let large = LargeListArray::new(
        item,
        OffsetBuffer::from_lengths([2, 0, 3, 1]),
        Arc::new(values),
        None,
    );
    let all = h(&[&list]);
    assert_eq!(all, h(&[&large]));
    // An empty list hashes to the seed; a slice hashes as its rows do.
    assert_eq!(all[1], 42);
    let tail = list.slice(2, 2);
    assert_eq!(h(&[&tail]), all[2..]);
    assert_eq!(x(&[&list.slice(1, 3)]), x(&[&list])[1..]);
    // Elements chain like columns: hash([1,2]) == hash(1, 2).
    let (one, two) = (Int32Array::from(vec![1]), Int32Array::from(vec![2]));
    assert_eq!(all[0], h(&[&one, &two])[0]);
}

#[test]
fn unsupported_types_and_bad_lengths_are_errors() {
    let keys = Int8Array::from(vec![0, 1]);
    let dict = DictionaryArray::<arrow_array::types::Int8Type>::try_new(
        keys,
        Arc::new(StringArray::from(vec!["a", "b"])),
    )
    .unwrap();
    let e = hash(&[&dict], 2).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // Time of day and 256-bit decimals are not Spark types.
    let time = arrow_array::Time32SecondArray::from(vec![1]);
    assert!(xxhash64(&[&time], 1).is_err());
    let wide = arrow_array::Decimal256Array::from(vec![arrow_buffer::i256::ZERO]);
    assert!(hash(&[&wide], 1).is_err());
    let a = Int32Array::from(vec![1, 2]);
    let e = hash(&[&a], 3).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn digests_match_known_values() {
    let input = StringArray::from(vec![Some("Spark"), Some(""), None]);
    let got = |a: StringArray| a.iter().map(|s| s.map(str::to_owned)).collect::<Vec<_>>();
    assert_eq!(
        got(md5(&input).unwrap()),
        [
            Some("8cde774d6f7333752ed72cacddb05126".to_owned()),
            Some("d41d8cd98f00b204e9800998ecf8427e".to_owned()),
            None
        ]
    );
    assert_eq!(
        sha1(&input).unwrap().value(0),
        "85f5955f4b27a9a4c2aab6ffe5d7189fc298b92c"
    );
    assert_eq!(
        crc32(&input).unwrap().iter().collect::<Vec<_>>(),
        [Some(1_557_323_817), Some(0), None]
    );
    // 0 is SHA-256; unsupported and NULL lengths give NULL.
    let s = |bits| sha2(&input, Operand::<Int32Type>::Scalar(bits)).unwrap();
    assert_eq!(s(Some(0)).value(0), s(Some(256)).value(0));
    assert_eq!(s(Some(224)).value(0).len(), 56);
    assert_eq!(s(Some(384)).value(0).len(), 96);
    assert_eq!(s(Some(512)).value(0).len(), 128);
    assert!(s(Some(1)).is_null(0) && s(None).is_null(0) && s(Some(-256)).is_null(0));
    assert!(s(Some(256)).is_null(2));
    // A bit-length column is per row.
    let bits = Int32Array::from(vec![Some(224), None, Some(256)]);
    let out = sha2(&input, Operand::Array(&bits)).unwrap();
    // Row 1 has a NULL length, row 2 a NULL input.
    assert_eq!(
        (out.value(0).len(), out.is_null(1), out.is_null(2)),
        (56, true, true)
    );
}

#[test]
fn digest_inputs_are_checked() {
    let input = StringArray::from(vec!["a", "b"]);
    let bits = Int32Array::from(vec![256]);
    let e = sha2(&input, Operand::Array(&bits)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let ints = Int32Array::from(vec![1]);
    assert!(md5(&ints).is_err() && sha1(&ints).is_err() && crc32(&ints).is_err());
    let large = LargeBinaryArray::from(vec![Some(&b"abc"[..])]);
    assert_eq!(crc32(&large).unwrap().value(0), 891_568_578);
    let ls = LargeStringArray::from(vec!["abc"]);
    assert_eq!(
        md5(&ls).unwrap().value(0),
        "900150983cd24fb0d6963f7d28e17f72"
    );
    assert_eq!(md5(&large).unwrap().value(0), md5(&ls).unwrap().value(0));
    let bin = BinaryArray::from(vec![&b"abc"[..]]);
    assert_eq!(md5(&bin).unwrap().value(0), md5(&ls).unwrap().value(0));
}

proptest! {
    #[test]
    fn xxhash64_ints_equal_the_byte_stream(v in any::<i32>(), w in any::<i64>()) {
        let i = Int32Array::from(vec![v]);
        let l = Int64Array::from(vec![w]);
        // The int and long fast paths are XXH64 of 4 and 8 little-endian bytes.
        let bi = BinaryArray::from(vec![v.to_le_bytes().as_slice()]);
        let bl = BinaryArray::from(vec![w.to_le_bytes().as_slice()]);
        prop_assert_eq!(x(&[&i]), x(&[&bi]));
        prop_assert_eq!(x(&[&l]), x(&[&bl]));
        prop_assert_eq!(h(&[&i]).len(), 1);
    }

    #[test]
    fn slicing_never_changes_a_row(rows in proptest::collection::vec(proptest::option::of("[a-z]{0,40}"), 1..40), cut in 0usize..40) {
        let a = StringArray::from(rows.clone());
        let cut = cut.min(rows.len());
        let whole = (h(&[&a]), x(&[&a]));
        let part = a.slice(cut, rows.len() - cut);
        let sliced = (h(&[&part]), x(&[&part]));
        prop_assert_eq!(&whole.0[cut..], sliced.0.as_slice());
        prop_assert_eq!(&whole.1[cut..], sliced.1.as_slice());
    }

    #[test]
    fn wide_decimals_hash_their_minimal_twos_complement(v in any::<i128>()) {
        let d = arrow_array::Decimal128Array::from(vec![v]).with_precision_and_scale(38, 0).unwrap();
        // BigInteger.toByteArray: the shortest big-endian two's complement.
        let be = v.to_be_bytes();
        let lead = if v < 0 { 0xff } else { 0 };
        let mut start = 0;
        while start < 15 && be[start] == lead && (be[start + 1] & 0x80 == lead & 0x80) {
            start += 1;
        }
        let bin = BinaryArray::from(vec![&be[start..]]);
        prop_assert_eq!(h(&[&d]), h(&[&bin]));
        prop_assert_eq!(x(&[&d]), x(&[&bin]));
        let small = i64::try_from(v % 1_000_000).unwrap();
        let narrow = arrow_array::Decimal128Array::from(vec![i128::from(small)]).with_precision_and_scale(18, 0).unwrap();
        let long = Int64Array::from(vec![small]);
        prop_assert_eq!(h(&[&narrow]), h(&[&long]));
    }
}

#[test]
fn list_of_strings_chain_into_the_row_hash() {
    let values = StringArray::from(vec!["a", "b"]);
    let list = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        OffsetBuffer::from_lengths([2]),
        Arc::new(values),
        None,
    );
    let (a, b) = (StringArray::from(vec!["a"]), StringArray::from(vec!["b"]));
    assert_eq!(h(&[&list]), h(&[&a, &b]));
    assert_eq!(list.value(0).as_string::<i32>().len(), 2);
}
