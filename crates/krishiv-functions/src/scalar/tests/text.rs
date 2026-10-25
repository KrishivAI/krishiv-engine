// SPDX-License-Identifier: Apache-2.0
//! The character-level and encoding string bindings: the argument shapes
//! each function binds, the storage carriers it reads, ANSI-keyed errors
//! (`elt`) and the errors of the encodings, over NULLs and both ANSI modes.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array, LargeStringArray, StringViewArray};
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;

use super::{
    BindArg, BindOptions, bind, bind_scalar, bind_with, bytes, class, col, int_values, ints, lit,
    owned, run, strs, text,
};

fn long(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn binary(a: &ArrayRef) -> Vec<Option<Vec<u8>>> {
    a.as_binary::<i32>()
        .iter()
        .map(|b| b.map(<[u8]>::to_vec))
        .collect()
}

fn bools(a: &ArrayRef) -> Vec<Option<bool>> {
    a.as_boolean().iter().collect()
}

#[test]
fn lengths_in_bits_and_bytes() {
    let s = strs(&[Some("abc"), None, Some("日本"), Some("")]);
    assert_eq!(
        int_values(&run("octet_length", &[&s])),
        [Some(3), None, Some(6), Some(0)]
    );
    assert_eq!(
        int_values(&run("bit_length", &[&s])),
        [Some(24), None, Some(48), Some(0)]
    );
    let b = bytes(&[Some(b"\x00\xff"), None]);
    assert_eq!(int_values(&run("octet_length", &[&b])), [Some(2), None]);
    assert_eq!(int_values(&run("bit_length", &[&b])), [Some(16), None]);
    // `len` is `length`.
    assert_eq!(
        int_values(&run("len", &[&s])),
        [Some(3), None, Some(2), Some(0)]
    );
    assert_eq!(
        int_values(&run("ascii", &[&s])),
        [Some(97), None, Some(26085), Some(0)]
    );
}

#[test]
fn chr_and_space_and_repeat() {
    let n = long(&[Some(65), Some(-3), None, Some(256)]);
    assert_eq!(
        text(&run("chr", &[&n])),
        owned(&[Some("A"), Some(""), None, Some("\0")])
    );
    assert_eq!(text(&run("char", &[&n])), text(&run("chr", &[&n])));
    let k = ints(&[Some(2), Some(0), None, Some(-1)]);
    assert_eq!(
        text(&run("space", &[&k])),
        owned(&[Some("  "), Some(""), None, Some("")])
    );
    let s = strs(&[Some("ab"), Some("ab"), Some("ab"), Some("é")]);
    assert_eq!(
        text(&run("repeat", &[&s, &k])),
        owned(&[Some("abab"), Some(""), None, Some("")])
    );
    // A literal count with a column of strings.
    let f = bind("repeat", &[&s, &k]);
    let out = f.invoke(&[col(&s), lit(&ints(&[Some(2)]))], 4).unwrap();
    assert_eq!(text(&out)[3].as_deref(), Some("éé"));
}

#[test]
fn left_and_right_on_strings_and_bytes() {
    let s = strs(&[Some("Spark SQL"), None, Some("日本語"), Some("")]);
    let n = ints(&[Some(5), Some(2), Some(2), Some(3)]);
    assert_eq!(
        text(&run("left", &[&s, &n])),
        owned(&[Some("Spark"), None, Some("日本"), Some("")])
    );
    assert_eq!(
        text(&run("right", &[&s, &n])),
        owned(&[Some("k SQL"), None, Some("本語"), Some("")])
    );
    let b = bytes(&[Some(b"\x01\x02\x03"), None]);
    let two = ints(&[Some(2), Some(2)]);
    let f = bind("left", &[&b, &two]);
    assert_eq!(f.return_type(), &DataType::Binary);
    let out = f.invoke(&[col(&b), col(&two)], 2).unwrap();
    assert_eq!(binary(&out), [Some(vec![1, 2]), None]);
}

#[test]
fn reverse_initcap_soundex_quote() {
    let s = strs(&[Some("abc"), None, Some("sPark sql"), Some("it's")]);
    assert_eq!(
        text(&run("reverse", &[&s])),
        owned(&[Some("cba"), None, Some("lqs kraPs"), Some("s'ti")])
    );
    assert_eq!(
        text(&run("initcap", &[&s])),
        owned(&[Some("Abc"), None, Some("Spark Sql"), Some("It's")])
    );
    assert_eq!(
        text(&run("soundex", &[&strs(&[Some("Miller"), None, Some("")])])),
        owned(&[Some("M460"), None, Some("")])
    );
    assert_eq!(
        text(&run("quote", &[&s])),
        owned(&[Some("'abc'"), None, Some("'sPark sql'"), Some("'it\\'s'")])
    );
    let b = bytes(&[Some(b"\x01\x02"), None]);
    let f = bind("reverse", &[&b]);
    assert_eq!(f.return_type(), &DataType::Binary);
    let out = f.invoke(&[col(&b)], 2).unwrap();
    assert_eq!(binary(&out), [Some(vec![2, 1]), None]);
}

#[test]
fn searching_and_splitting() {
    let hay = strs(&[Some("foobarbar"), Some("foobarbar"), None, Some("abc")]);
    let needle = strs(&[Some("bar"), Some("bar"), Some("a"), None]);
    let start = ints(&[Some(1), Some(5), Some(1), Some(1)]);
    assert_eq!(
        int_values(&run("locate", &[&needle, &hay])),
        [Some(4), Some(4), None, None]
    );
    assert_eq!(
        int_values(&run("position", &[&needle, &hay, &start])),
        [Some(4), Some(7), None, None]
    );
    assert_eq!(
        int_values(&run("instr", &[&hay, &needle])),
        [Some(4), Some(4), None, None]
    );
    let set = strs(&[Some("a,b,c"), Some("a,b,c"), None, Some("")]);
    let word = strs(&[Some("b"), Some("d"), Some("a"), Some("")]);
    assert_eq!(
        int_values(&run("find_in_set", &[&word, &set])),
        [Some(2), Some(0), None, Some(1)]
    );
    let url = strs(&[Some("www.apache.org"), Some("a.b"), None, Some("x")]);
    let dot = strs(&[Some("."); 4]);
    let n = ints(&[Some(2), Some(-1), Some(1), Some(1)]);
    assert_eq!(
        text(&run("substring_index", &[&url, &dot, &n])),
        owned(&[Some("www.apache"), Some("b"), None, Some("x")])
    );
    assert_eq!(
        text(&run("split_part", &[&url, &dot, &n])),
        owned(&[Some("apache"), Some("b"), None, Some("x")])
    );
    // Index 0 raises for a valid row only.
    let zero = ints(&[Some(0), Some(0), Some(0), Some(0)]);
    let f = bind("split_part", &[&url, &dot, &zero]);
    let err = f
        .invoke(&[col(&url), col(&dot), col(&zero)], 4)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::InvalidIndexOfZero);
    let nulls = strs(&[None, None]);
    let out = bind("split_part", &[&nulls, &nulls, &ints(&[Some(0), Some(0)])])
        .invoke(
            &[col(&nulls), col(&nulls), col(&ints(&[Some(0), Some(0)]))],
            2,
        )
        .unwrap();
    assert_eq!(text(&out), owned(&[None, None]));
}

#[test]
fn translate_overlay_levenshtein_luhn() {
    let s = strs(&[Some("translate"), None, Some("日本語")]);
    let from = strs(&[Some("rnlt"), Some("a"), Some("本")]);
    let to = strs(&[Some("1223"), Some("b"), Some("x")]);
    assert_eq!(
        text(&run("translate", &[&s, &from, &to])),
        owned(&[Some("31a2s2a3e"), None, Some("日x語")])
    );
    let (input, rep) = (
        strs(&[Some("Spark SQL"), None]),
        strs(&[Some("CORE"), Some("x")]),
    );
    let pos = ints(&[Some(7), Some(1)]);
    assert_eq!(
        text(&run("overlay", &[&input, &rep, &pos])),
        owned(&[Some("Spark CORE"), None])
    );
    let len = ints(&[Some(0), Some(0)]);
    assert_eq!(
        text(&run("overlay", &[&input, &rep, &pos, &len])),
        owned(&[Some("Spark CORESQL"), None])
    );
    let (bi, br) = (bytes(&[Some(b"Spark SQL")]), bytes(&[Some(b"CORE")]));
    let f = bind("overlay", &[&bi, &br, &ints(&[Some(7)])]);
    assert_eq!(f.return_type(), &DataType::Binary);
    let out = f
        .invoke(&[col(&bi), col(&br), col(&ints(&[Some(7)]))], 1)
        .unwrap();
    assert_eq!(binary(&out), [Some(b"Spark CORE".to_vec())]);
    let (a, b) = (
        strs(&[Some("kitten"), None]),
        strs(&[Some("sitting"), Some("x")]),
    );
    assert_eq!(int_values(&run("levenshtein", &[&a, &b])), [Some(3), None]);
    let limit = ints(&[Some(2), Some(2)]);
    assert_eq!(
        int_values(&run("levenshtein", &[&a, &b, &limit])),
        [Some(-1), None]
    );
    let cards = strs(&[
        Some("4111111111111111"),
        Some("4111111111111112"),
        None,
        Some(""),
    ]);
    assert_eq!(
        bools(&run("luhn_check", &[&cards])),
        [Some(true), Some(false), None, Some(false)]
    );
}

#[test]
fn elt_raises_only_under_ansi_and_only_for_evaluated_rows() {
    let idx = ints(&[Some(1), Some(2), None, Some(5)]);
    let (a, b) = (
        strs(&[Some("a"); 4]),
        strs(&[Some("b"), None, Some("b"), Some("b")]),
    );
    let lenient = BindOptions::new().with_ansi(false);
    let f = bind_with("elt", &[&idx, &a, &b], &lenient);
    assert_eq!(f.return_type(), &DataType::Utf8);
    let out = f.invoke(&[col(&idx), col(&a), col(&b)], 4).unwrap();
    assert_eq!(text(&out), owned(&[Some("a"), None, None, None]));
    let strict = bind_with("elt", &[&idx, &a, &b], &BindOptions::new());
    let err = strict
        .invoke(&[col(&idx), col(&a), col(&b)], 4)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::InvalidArrayIndex);
    let ok = ints(&[Some(1), Some(2), None]);
    let out = strict
        .invoke(&[col(&ok), col(&a.slice(0, 3)), col(&b.slice(0, 3))], 3)
        .unwrap();
    assert_eq!(text(&out), owned(&[Some("a"), None, None]));
    // Binary arguments pick a binary.
    let (x, y) = (bytes(&[Some(b"x")]), bytes(&[Some(b"y")]));
    let one = ints(&[Some(2)]);
    let f = bind("elt", &[&one, &x, &y]);
    assert_eq!(f.return_type(), &DataType::Binary);
    let out = f.invoke(&[col(&one), col(&x), col(&y)], 1).unwrap();
    assert_eq!(binary(&out), [Some(b"y".to_vec())]);
}

#[test]
fn concat_ws_skips_nulls() {
    let sep = strs(&[Some(","), None, Some("-")]);
    let (a, b) = (
        strs(&[Some("a"), Some("a"), None]),
        strs(&[Some("b"), Some("b"), None]),
    );
    assert_eq!(
        text(&run("concat_ws", &[&sep, &a, &b])),
        owned(&[Some("a,b"), None, Some("")])
    );
    assert_eq!(
        text(&run("concat_ws", &[&sep])),
        owned(&[Some(""), None, Some("")])
    );
}

#[test]
fn storage_carriers_are_read() {
    let view: ArrayRef = Arc::new(StringViewArray::from(vec![Some("abc"), None]));
    let large: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("abc"), None]));
    for a in [&view, &large] {
        assert_eq!(text(&run("reverse", &[a])), owned(&[Some("cba"), None]));
        assert_eq!(int_values(&run("octet_length", &[a])), [Some(3), None]);
        assert_eq!(text(&run("url_encode", &[a])), owned(&[Some("abc"), None]));
    }
    let bin: ArrayRef = Arc::new(arrow_array::BinaryViewArray::from(vec![
        Some(&b"abc"[..]),
        None,
    ]));
    assert_eq!(text(&run("base64", &[&bin])), owned(&[Some("YWJj"), None]));
}

#[test]
fn bad_shapes_are_rejected_at_binding() {
    let (s, n) = (strs(&[Some("a")]), ints(&[Some(1)]));
    for (name, args) in [
        ("ascii", vec![&n]),
        ("repeat", vec![&s, &s]),
        ("left", vec![&n, &n]),
        ("locate", vec![&s]),
        ("elt", vec![&s, &s]),
        ("mask", vec![&n]),
        ("base64", vec![&s]),
        ("decode", vec![&s, &s]),
        ("to_binary", vec![&n]),
        ("url_decode", vec![&n]),
    ] {
        let err = bind_scalar(name, &super::types(&args), &BindOptions::new()).unwrap_err();
        assert_eq!(class(&err), ErrorClass::KrishivUnsupportedFeature, "{name}");
    }
}

#[test]
fn base64_and_unbase64() {
    let b = bytes(&[Some(b"Spark SQL"), None, Some(b"")]);
    assert_eq!(
        text(&run("base64", &[&b])),
        owned(&[Some("U3BhcmsgU1FM"), None, Some("")])
    );
    let s = strs(&[Some("U3BhcmsgU1FM"), None, Some("")]);
    let out = run("unbase64", &[&s]);
    assert_eq!(
        binary(&out),
        [Some(b"Spark SQL".to_vec()), None, Some(Vec::new())]
    );
    let bad = strs(&[Some("1")]);
    let err = bind("unbase64", &[&bad])
        .invoke(&[col(&bad)], 1)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivInvalidBase64Input);
}

#[test]
fn to_binary_and_try_to_binary() {
    let s = strs(&[Some("68656C6C6F"), Some("GG"), None]);
    let f = bind("try_to_binary", &[&s]);
    let out = f.invoke(&[col(&s)], 3).unwrap();
    assert_eq!(binary(&out), [Some(b"hello".to_vec()), None, None]);
    let err = bind("to_binary", &[&s]).invoke(&[col(&s)], 3).unwrap_err();
    assert_eq!(class(&err), ErrorClass::ConversionInvalidInput);
    // The format is read per row, case-insensitively.
    let (t, fmt) = (
        strs(&[Some("YQ=="), Some("a"), Some("YQ==")]),
        strs(&[Some("BASE64"), Some("utf-8"), None]),
    );
    let out = run("to_binary", &[&t, &fmt]);
    assert_eq!(
        binary(&out),
        [Some(b"a".to_vec()), Some(b"a".to_vec()), None]
    );
}

#[test]
fn encode_and_decode() {
    let (s, cs) = (
        strs(&[Some("é"), None, Some("a")]),
        strs(&[Some("iso-8859-1"), Some("utf-8"), Some("UTF-16LE")]),
    );
    let out = run("encode", &[&s, &cs]);
    assert_eq!(binary(&out), [Some(vec![0xE9]), None, Some(vec![b'a', 0])]);
    let b = bytes(&[Some(&[0xE9][..]), None]);
    let latin = strs(&[Some("ISO-8859-1"), Some("utf-8")]);
    assert_eq!(
        text(&run("decode", &[&b, &latin])),
        owned(&[Some("é"), None])
    );
    let bad = strs(&[Some("WINDOWS-1252")]);
    let err = bind("encode", &[&strs(&[Some("a")]), &bad])
        .invoke(&[col(&strs(&[Some("a")])), col(&bad)], 1)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::InvalidParameterValueCharset);
    let ascii = strs(&[Some("us-ascii")]);
    let err = bind("decode", &[&bytes(&[Some(&[0x80][..])]), &ascii])
        .invoke(&[col(&bytes(&[Some(&[0x80][..])])), col(&ascii)], 1)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::MalformedCharacterCoding);
}

#[test]
fn utf8_validators_take_strings_and_binary() {
    // A string is always valid.
    let s = strs(&[Some("abc"), None]);
    assert_eq!(bools(&run("is_valid_utf8", &[&s])), [Some(true), None]);
    for name in ["make_valid_utf8", "validate_utf8", "try_validate_utf8"] {
        assert_eq!(
            text(&run(name, &[&s])),
            owned(&[Some("abc"), None]),
            "{name}"
        );
    }
    let b: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(&b"a"[..]),
        Some(&[0x80][..]),
        None,
    ]));
    assert_eq!(
        bools(&run("is_valid_utf8", &[&b])),
        [Some(true), Some(false), None]
    );
    assert_eq!(
        text(&run("make_valid_utf8", &[&b])),
        owned(&[Some("a"), Some("\u{FFFD}"), None])
    );
    assert_eq!(
        text(&run("try_validate_utf8", &[&b])),
        owned(&[Some("a"), None, None])
    );
    let err = bind("validate_utf8", &[&b])
        .invoke(&[col(&b)], 3)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::InvalidUtf8String);
    // Rows that are not evaluated cannot raise.
    let head = b.slice(0, 1);
    assert!(
        bind("validate_utf8", &[&head])
            .invoke(&[col(&head)], 1)
            .is_ok()
    );
}

#[test]
fn url_functions() {
    let s = strs(&[Some("https://a.b/c d"), None, Some("")]);
    let enc = run("url_encode", &[&s]);
    assert_eq!(
        text(&enc),
        owned(&[Some("https%3A%2F%2Fa.b%2Fc+d"), None, Some("")])
    );
    assert_eq!(text(&run("url_decode", &[&enc])), text(&s));
    let bad = strs(&[Some("%"), Some("%41")]);
    assert_eq!(
        text(&run("try_url_decode", &[&bad])),
        owned(&[None, Some("A")])
    );
    let err = bind("url_decode", &[&bad])
        .invoke(&[col(&bad)], 2)
        .unwrap_err();
    assert_eq!(class(&err), ErrorClass::CannotDecodeUrl);
}

#[test]
fn every_name_is_pure_and_sorted() {
    let names = super::scalar_names();
    for n in super::super::chars::NAMES
        .iter()
        .chain(super::super::codec::NAMES)
    {
        assert!(names.contains(n), "{n}");
        assert!(super::super::is_argument_pure(n), "{n}");
    }
}

#[test]
fn levenshtein_threshold_nulls() {
    let (a, b) = (
        strs(&[Some("kitten"), Some("kitten")]),
        strs(&[Some("kitten"), Some("sitting")]),
    );
    // A NULL read from a column is 0 (`null.asInstanceOf[Int]`).
    let limit = ints(&[None, None]);
    assert_eq!(
        int_values(&run("levenshtein", &[&a, &b, &limit])),
        [Some(0), Some(-1)]
    );
    // A NULL literal makes the whole call NULL.
    let null = ints(&[None]);
    let args = [
        BindArg::Column(DataType::Utf8),
        BindArg::Column(DataType::Utf8),
        BindArg::Literal(Arc::clone(&null)),
    ];
    let f = bind_scalar("levenshtein", &args, &BindOptions::new()).unwrap();
    let out = f.invoke(&[col(&a), col(&b), lit(&null)], 2).unwrap();
    assert_eq!(int_values(&out), [None, None]);
}
