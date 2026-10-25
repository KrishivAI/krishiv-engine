// SPDX-License-Identifier: Apache-2.0
//! The character-level and encoding string kernels: NULLs, the empty
//! string, multi-byte characters, boundaries, invalid input and the
//! algebraic properties. Expected values are Spark 4.2's (the golden files
//! and the function matrix).
#![forbid(unsafe_code)]

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, BinaryArray, BooleanArray, Int32Array, Int64Array, LargeStringArray, StringArray,
};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::Operand;
use krishiv_kernels::string::{self as s, MaskRule, MaskRules, StrOperand};
use proptest::prelude::*;

type S<'a> = StrOperand<'a, i32>;
type I<'a> = Operand<'a, Int32Type>;

fn strs(v: &[Option<&str>]) -> StringArray {
    StringArray::from(v.to_vec())
}

fn out(a: &StringArray) -> Vec<Option<&str>> {
    a.iter().collect()
}

fn lit(v: Option<&str>) -> S<'_> {
    StrOperand::Scalar(v)
}

fn int(v: i32) -> I<'static> {
    Operand::Scalar(Some(v))
}

fn ints(v: &[Option<i32>]) -> Int32Array {
    Int32Array::from(v.to_vec())
}

fn r1(text: &str) -> StringArray {
    strs(&[Some(text)])
}

fn one(f: impl Fn(&StringArray) -> StringArray, input: &str) -> String {
    f(&strs(&[Some(input)])).value(0).to_owned()
}

#[test]
fn ascii_is_the_first_code_point() {
    let a = strs(&[
        Some("abc"),
        Some(""),
        None,
        Some("Ä€"),
        Some("😀"),
        Some("\u{FFFD}"),
    ]);
    let got: Vec<_> = s::ascii(&a).iter().collect();
    assert_eq!(
        got,
        [
            Some(97),
            Some(0),
            None,
            Some(196),
            Some(0x1F600),
            Some(65533)
        ]
    );
}

#[test]
fn chr_takes_the_low_byte() {
    let a = Int64Array::from(vec![
        Some(65),
        Some(256),
        Some(-1),
        None,
        Some(i64::MAX),
        Some(0),
    ]);
    let got = s::chr::<i32>(&a).unwrap();
    assert_eq!(
        out(&got),
        [Some("A"), Some("\0"), Some(""), None, Some("ÿ"), Some("\0")]
    );
    assert_eq!(
        s::chr::<i32>(&Int64Array::from(vec![233_i64]))
            .unwrap()
            .value(0),
        "é"
    );
}

#[test]
fn space_repeat_right() {
    let n = ints(&[Some(3), Some(0), Some(-2), None]);
    let spaces = s::space::<i32>(Operand::Array(&n)).unwrap();
    assert_eq!(out(&spaces), [Some("   "), Some(""), Some(""), None]);
    let a = strs(&[Some("ab"), Some(""), None, Some("日本")]);
    let times = ints(&[Some(2), Some(5), Some(1), Some(2)]);
    let rep = s::repeat(StrOperand::Array(&a), Operand::Array(&times)).unwrap();
    assert_eq!(out(&rep), [Some("abab"), Some(""), None, Some("日本日本")]);
    let zero = s::repeat(
        lit(Some("x")),
        Operand::Array(&ints(&[Some(0), Some(-1), None])),
    )
    .unwrap();
    assert_eq!(out(&zero), [Some(""), Some(""), None]);
    let t = strs(&[
        Some("Spark SQL"),
        Some("Ä€日本語😀é"),
        Some("ab"),
        None,
        Some("abc"),
    ]);
    let n = ints(&[Some(3), Some(3), Some(5), Some(1), Some(0)]);
    let right = s::right(StrOperand::Array(&t), Operand::Array(&n)).unwrap();
    assert_eq!(
        out(&right),
        [Some("SQL"), Some("語😀é"), Some("ab"), None, Some("")]
    );
    let min = s::right(StrOperand::Array(&r1("Spark SQL")), int(i32::MIN)).unwrap();
    assert_eq!(out(&min), [Some("")]);
    let _ = s::space::<i32>(int(1)).unwrap_err();
}

#[test]
fn repeat_and_space_refuse_what_does_not_fit() {
    let e = s::repeat::<i32>(StrOperand::Array(&r1("abcd")), int(i32::MAX)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
}

#[test]
fn reverse_works_on_characters_and_bytes() {
    let a = strs(&[Some("abc"), Some(""), None, Some("Ä€日本語😀e\u{301}")]);
    let r = s::reverse(&a).unwrap();
    assert_eq!(
        out(&r),
        [Some("cba"), Some(""), None, Some("\u{301}e😀語本日€Ä")]
    );
    let b = BinaryArray::from(vec![Some(&[1_u8, 2, 3][..]), Some(&[][..]), None]);
    let r = s::reverse_binary(&b).unwrap();
    assert_eq!(
        r.iter().collect::<Vec<_>>(),
        [Some(&[3_u8, 2, 1][..]), Some(&[][..]), None]
    );
}

#[test]
fn initcap_title_cases_after_ascii_spaces() {
    let a = strs(&[
        Some("sPark sql"),
        Some(""),
        None,
        Some(" \t a\u{3000}\n "),
        Some("ǆ ß ΟΔΟΣ ΣΑΣ"),
        Some("İ"),
        Some("ab\tcd ef"),
    ]);
    let got = s::initcap(&a).unwrap();
    assert_eq!(
        out(&got),
        [
            Some("Spark Sql"),
            Some(""),
            None,
            Some(" \t A\u{3000}\n "),
            Some("ǅ Ss Οδος Σας"),
            Some("İ"),
            Some("Ab\tcd Ef"),
        ]
    );
}

#[test]
fn translate_maps_deletes_and_keeps_the_first_mapping() {
    let src = strs(&[Some("translate"), Some("日本語"), None, Some("")]);
    let from = lit(Some("rnlt"));
    let to = lit(Some("1223"));
    let got = s::translate(StrOperand::Array(&src), from, to).unwrap();
    assert_eq!(
        out(&got),
        [Some("31a2s2a3e"), Some("日本語"), None, Some("")]
    );
    // A shorter `to` deletes; a NUL in `to` deletes; the first of a repeated
    // `from` character wins.
    let got = s::translate(
        StrOperand::Array(&r1("abcabc")),
        lit(Some("abca")),
        lit(Some("x\0")),
    )
    .unwrap();
    assert_eq!(out(&got), [Some("xx")]);
    let got = s::translate(
        StrOperand::Array(&r1("日本語")),
        lit(Some("本")),
        lit(Some("x")),
    )
    .unwrap();
    assert_eq!(out(&got), [Some("日x語")]);
    let cols = strs(&[Some("a"), Some("b")]);
    let tos = strs(&[Some("1"), Some("2")]);
    let got = s::translate(
        lit(Some("ab")),
        StrOperand::Array(&cols),
        StrOperand::Array(&tos),
    )
    .unwrap();
    assert_eq!(out(&got), [Some("1b"), Some("a2")]);
}

#[test]
fn locate_follows_spark_including_its_empty_substring_quirk() {
    let rows = |sub: &[Option<&str>], text: &[Option<&str>], start: &[Option<i32>]| {
        let (a, b, c) = (strs(sub), strs(text), ints(start));
        let r = s::locate(
            StrOperand::Array(&a),
            StrOperand::Array(&b),
            Operand::Array(&c),
        )
        .unwrap();
        r.iter().collect::<Vec<_>>()
    };
    let got = rows(
        &[
            Some("bar"),
            Some("bar"),
            Some("bar"),
            Some(""),
            Some(""),
            Some("x"),
            None,
            Some("a"),
            Some("日"),
        ],
        &[
            Some("foobarbar"),
            Some("foobarbar"),
            Some("foobarbar"),
            Some("abc"),
            Some("abc"),
            Some(""),
            Some("a"),
            None,
            Some("a日b日"),
        ],
        &[
            Some(1),
            Some(5),
            Some(8),
            Some(3),
            Some(100),
            Some(1),
            Some(1),
            Some(1),
            Some(3),
        ],
    );
    assert_eq!(
        got,
        [
            Some(4),
            Some(7),
            Some(0),
            Some(1),
            Some(1),
            Some(0),
            None,
            None,
            Some(4)
        ]
    );
    // A start below 1 is 0; a NULL start is 0 even with NULL operands.
    let got = rows(
        &[Some("a"), None, Some("a")],
        &[Some("a"), None, Some("a")],
        &[Some(0), None, Some(-5)],
    );
    assert_eq!(got, [Some(0), Some(0), Some(0)]);
}

#[test]
fn substring_index_counts_overlapping_matches_from_either_end() {
    let case = |text: &str, delim: &str, n: i32| {
        let r = s::substring_index(StrOperand::Array(&r1(text)), lit(Some(delim)), int(n)).unwrap();
        r.value(0).to_owned()
    };
    assert_eq!(case("www.apache.org", ".", 2), "www.apache");
    assert_eq!(case("www.apache.org", ".", -2), "apache.org");
    assert_eq!(case("www.apache.org", ".", 5), "www.apache.org");
    assert_eq!(case("www.apache.org", ".", -5), "www.apache.org");
    assert_eq!(case("www.apache.org", ".", 0), "");
    assert_eq!(case("www.apache.org", "", 1), "");
    assert_eq!(case(".a", ".", 1), "");
    assert_eq!(case("a.", ".", -1), "");
    assert_eq!(case("aaa", "aa", 2), "a");
    assert_eq!(case("aaa", "aa", -2), "a");
    assert_eq!(case("a日b日c", "日", -1), "c");
    assert_eq!(case("ab", "abc", -1), "ab");
    assert_eq!(case("ab", "x", i32::MIN), "ab");
    // `Int.MIN` never has that many delimiters, so the whole string comes
    // back (Spark's JVM raises an unclassified exception here).
    assert_eq!(case("a.b.c", ".", i32::MIN), "a.b.c");
    assert_eq!(case("aaa", "a", i32::MIN), "aaa");
    let nulls =
        s::substring_index(StrOperand::Array(&strs(&[None])), lit(Some(".")), int(1)).unwrap();
    assert_eq!(out(&nulls), [None]);
}

#[test]
fn split_part_indexes_from_either_end() {
    let case = |text: &str, delim: &str, n: i32| {
        s::split_part(StrOperand::Array(&r1(text)), lit(Some(delim)), int(n))
            .map(|a| a.value(0).to_owned())
    };
    assert_eq!(case("11.12.13", ".", 2).unwrap(), "12");
    assert_eq!(case("11.12.13", ".", -1).unwrap(), "13");
    assert_eq!(case("11.12.13", ".", -3).unwrap(), "11");
    assert_eq!(case("11.12.13", ".", 4).unwrap(), "");
    assert_eq!(case("11.12.13", ".", -5).unwrap(), "");
    assert_eq!(case("11.12.13", "", 1).unwrap(), "11.12.13");
    assert_eq!(case("11ab12ab13", "ab", 1).unwrap(), "11");
    assert_eq!(case("a..b", ".", 2).unwrap(), "");
    assert_eq!(case("aaa", "aa", -1).unwrap(), "a");
    assert_eq!(case("x", "y", i32::MIN).unwrap(), "");
    // `Int.MIN` counts past the start, so there is never such a piece (Spark's
    // JVM raises an unclassified exception here).
    assert_eq!(case("a.b.c", ".", i32::MIN).unwrap(), "");
    assert_eq!(case("a.b.c", "", i32::MIN).unwrap(), "");
    let zero = case("a", ".", 0).unwrap_err();
    assert_eq!(zero.error_class(), ErrorClass::InvalidIndexOfZero);
    assert_eq!(zero.row(), Some(0));
    // A NULL operand is NULL, not an error, even with index 0.
    let nul = s::split_part(StrOperand::Array(&strs(&[None])), lit(Some(".")), int(0)).unwrap();
    assert_eq!(out(&nul), [None]);
    // Only the valid rows can raise.
    let a = strs(&[Some("a.b"), None]);
    let n = ints(&[Some(1), Some(0)]);
    let got = s::split_part(StrOperand::Array(&a), lit(Some(".")), Operand::Array(&n)).unwrap();
    assert_eq!(out(&got), [Some("a"), None]);
}

#[test]
fn find_in_set_and_overlay() {
    let w = strs(&[Some("ab"), Some("a,b"), Some(""), None, Some("c")]);
    let set = strs(&[
        Some("abc,b,ab,c,def"),
        Some("a,b"),
        Some(","),
        Some("a"),
        Some(""),
    ]);
    let got = s::find_in_set(StrOperand::Array(&w), StrOperand::Array(&set)).unwrap();
    assert_eq!(
        got.iter().collect::<Vec<_>>(),
        [Some(3), Some(0), Some(1), None, Some(0)]
    );
    let o = |text: &str, rep: &str, pos: i32, len: Option<i32>| {
        let r = s::overlay(
            StrOperand::Array(&r1(text)),
            lit(Some(rep)),
            int(pos),
            len.map(int),
        )
        .unwrap();
        r.value(0).to_owned()
    };
    assert_eq!(o("Spark SQL", "_", 6, None), "Spark_SQL");
    assert_eq!(o("Spark SQL", "CORE", 7, None), "Spark CORE");
    assert_eq!(o("Spark SQL", "ANSI ", 7, Some(0)), "Spark ANSI SQL");
    assert_eq!(o("Spark SQL", "xyz", 2, Some(4)), "Sxyz SQL");
    assert_eq!(o("日本語", "x", 2, None), "日x語");
    assert_eq!(o("abc", "X", 0, None), "Xabc");
    assert_eq!(o("abc", "X", 10, None), "abcX");
    assert_eq!(o("abc", "X", i32::MIN, None), "abcXabc");
    assert_eq!(o("abc", "XY", 2, Some(-1)), "aXY");
    let b = BinaryArray::from(vec![Some(&b"Spark SQL"[..]), None]);
    let r = BinaryArray::from(vec![Some(&b"CORE"[..]), Some(&b"x"[..])]);
    let got = s::overlay_binary(
        krishiv_kernels::string::BinOperand::Array(&b),
        krishiv_kernels::string::BinOperand::Array(&r),
        int(7),
        None,
    )
    .unwrap();
    assert_eq!(
        got.iter().collect::<Vec<_>>(),
        [Some(&b"Spark CORE"[..]), None]
    );
}

#[test]
fn levenshtein_soundex_luhn() {
    let (a, b) = (
        strs(&[Some("kitten"), Some(""), Some("日本"), None, Some("abc")]),
        strs(&[
            Some("sitting"),
            Some("ab"),
            Some("日"),
            Some("a"),
            Some("abc"),
        ]),
    );
    let d = s::levenshtein(StrOperand::Array(&a), StrOperand::Array(&b), None).unwrap();
    assert_eq!(
        d.iter().collect::<Vec<_>>(),
        [Some(3), Some(2), Some(1), None, Some(0)]
    );
    let limits = ints(&[Some(2), Some(2), Some(-1), Some(1), None]);
    let t = s::levenshtein(
        StrOperand::Array(&a),
        StrOperand::Array(&b),
        Some(Operand::Array(&limits)),
    )
    .unwrap();
    // 3 > 2; 2 <= 2; 1 > -1; NULL; a NULL limit is 0 and 0 <= 0.
    assert_eq!(
        t.iter().collect::<Vec<_>>(),
        [Some(-1), Some(2), Some(-1), None, Some(0)]
    );
    let x = strs(&[
        Some("Miller"),
        Some("Robert"),
        Some("Tymczak"),
        Some("Ashcraft"),
        Some(""),
        Some("123"),
        Some("é"),
        None,
        Some("a"),
    ]);
    let got = s::soundex(&x).unwrap();
    assert_eq!(
        out(&got),
        [
            Some("M460"),
            Some("R163"),
            Some("T522"),
            Some("A261"),
            Some(""),
            Some("123"),
            Some("é"),
            None,
            Some("A000")
        ]
    );
    let n = strs(&[
        Some("4111111111111111"),
        Some("6011000000000005"),
        Some(""),
        Some(" 1"),
        Some("0"),
        None,
        Some("٤١١١١١١١١١١١١١١١"),
    ]);
    let got: Vec<_> = s::luhn_check(&n).iter().collect();
    assert_eq!(
        got,
        [
            Some(true),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
            None,
            Some(true)
        ]
    );
    // Digits outside the BMP are not digits for Java's `charAt`.
    assert!(!s::luhn_check(&strs(&[Some("𝟎")])).value(0));
}

#[test]
fn mask_replaces_by_class() {
    let a = strs(&[Some("AbCD123-@$#"), Some(""), None, Some("é😀É٣")]);
    let default = s::mask(&a, MaskRules::default()).unwrap();
    assert_eq!(
        out(&default),
        [Some("XxXXnnn-@$#"), Some(""), None, Some("x😀Xn")]
    );
    let all = MaskRules {
        upper: MaskRule::With('Q'),
        lower: MaskRule::With('q'),
        digit: MaskRule::With('d'),
        other: MaskRule::With('o'),
    };
    let got = s::mask(&a, all).unwrap();
    assert_eq!(
        out(&got),
        [Some("QqQQdddoooo"), Some(""), None, Some("qooQd")]
    );
    let keep = MaskRules {
        upper: MaskRule::Keep,
        lower: MaskRule::Keep,
        digit: MaskRule::Keep,
        other: MaskRule::Keep,
    };
    assert_eq!(out(&s::mask(&a, keep).unwrap()), out(&a));
    // Modifier letters and ⓐ are letters in the Unicode properties but not in
    // category L{u,l}.
    let odd = strs(&[Some("ʰⓐⅣ")]);
    assert_eq!(
        out(&s::mask(&odd, MaskRules::default()).unwrap()),
        [Some("ʰⓐⅣ")]
    );
}

/// The toolchain's Unicode data is pinned: `initcap` and `mask` classify
/// with `char::{to_uppercase, to_lowercase, is_uppercase, is_lowercase}`,
/// so a toolchain upgrade that changes these mappings would silently change
/// the result. This test fails then instead, signalling the compensating
/// list and the documented approximation need review.
#[test]
fn toolchain_unicode_is_pinned() {
    // The compensating list is still needed on this toolchain: each sample
    // carries an Uppercase/Lowercase property without being `Lu`/`Ll`.
    for c in [
        '\u{00AA}',
        '\u{02B0}',
        '\u{0345}',
        '\u{1D2C}',
        '\u{2160}',
        '\u{2170}',
        '\u{24B6}',
        '\u{24D0}',
        '\u{2C7C}',
        '\u{A770}',
        '\u{AB5C}',
        '\u{1F130}',
    ] {
        assert!(
            c.is_uppercase() || c.is_lowercase(),
            "U+{:04X} lost its case property; review is_other_case",
            u32::from(c)
        );
    }
    // Plain `Lu`/`Ll` letters are not in the list: `mask` replaces them.
    let letters = strs(&[Some("AÉΩДaéωд")]);
    assert_eq!(
        out(&s::mask(&letters, MaskRules::default()).unwrap()),
        [Some("XXXXxxxx")]
    );
    // `initcap` on the divergent code points: title case is `to_uppercase`
    // with the tail lowered, except the digraphs and Georgian, which keep
    // their form. Greek with an iota subscript differs from ICU here.
    let words = strs(&[
        Some("ᾀ"),
        Some("ß"),
        Some("ǆ"),
        Some("ა"),
        Some("İ"),
        Some("ﬁ"),
        Some("ʰ"),
    ]);
    assert_eq!(
        out(&s::initcap(&words).unwrap()),
        [
            Some("Ἀι"),
            Some("Ss"),
            Some("ǅ"),
            Some("ა"),
            Some("İ"),
            Some("Fi"),
            Some("ʰ"),
        ]
    );
    // `mask` keeps the divergent forms under the default (`other`) rule.
    let odd = strs(&[Some("ʰⓐⅣᾀß")]);
    assert_eq!(
        out(&s::mask(&odd, MaskRules::default()).unwrap()),
        [Some("ʰⓐⅣxx")]
    );
}

#[test]
fn quote_elt_concat_ws() {
    let a = strs(&[Some("Spark"), Some("Don't"), Some(""), None]);
    let got = s::quote(&a).unwrap();
    assert_eq!(
        out(&got),
        [Some("'Spark'"), Some("'Don\\'t'"), Some("''"), None]
    );
    let (x, y) = (
        strs(&[Some("a"), None, Some("c")]),
        strs(&[Some("b"), Some("b"), None]),
    );
    let idx = ints(&[Some(1), Some(2), Some(2)]);
    let parts = [StrOperand::Array(&x), StrOperand::Array(&y)];
    let got = s::elt(Operand::Array(&idx), &parts, true).unwrap();
    assert_eq!(out(&got), [Some("a"), Some("b"), None]);
    let bad = ints(&[Some(0), Some(3), None]);
    let lenient = s::elt(Operand::Array(&bad), &parts, false).unwrap();
    assert_eq!(out(&lenient), [None, None, None]);
    let e = s::elt(Operand::Array(&bad), &parts, true).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidArrayIndex);
    assert_eq!(
        (e.row(), e.error().message_parameters()[0].1.as_str()),
        (Some(0), "0")
    );
    // Only a row that is evaluated raises.
    let skipped = ints(&[None, Some(1), Some(2)]);
    let got = s::elt(Operand::Array(&skipped), &parts, true).unwrap();
    assert_eq!(out(&got), [None, None, None]);
    let bins = BinaryArray::from(vec![Some(&b"x"[..]), Some(&b"y"[..])]);
    let one_bin = [krishiv_kernels::string::BinOperand::Array(&bins)];
    let picked = s::elt_binary(int(1), &one_bin, true).unwrap();
    assert_eq!(
        picked.iter().collect::<Vec<_>>(),
        [Some(&b"x"[..]), Some(&b"y"[..])]
    );
    let e = s::elt_binary(int(2), &one_bin, true).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidArrayIndex);
    let sep = strs(&[Some("/"), None, Some("-")]);
    let (p, q) = (
        strs(&[Some("foo"), Some("foo"), None]),
        strs(&[None, Some("bar"), None]),
    );
    let got = s::concat_ws(
        StrOperand::Array(&sep),
        &[StrOperand::Array(&p), StrOperand::Array(&q), lit(Some("z"))],
    )
    .unwrap();
    assert_eq!(out(&got), [Some("foo/z"), None, Some("z")]);
    let none = s::concat_ws(lit(Some(",")), &[StrOperand::Array(&strs(&[None]))]).unwrap();
    assert_eq!(out(&none), [Some("")]);
}

#[test]
fn byte_and_bit_lengths() {
    let a = strs(&[Some("abc"), Some(""), None, Some("日本")]);
    let bits: Vec<_> = s::bit_length(&a).iter().collect();
    assert_eq!(bits, [Some(24), Some(0), None, Some(48)]);
    let b = BinaryArray::from(vec![Some(&[1_u8][..]), None]);
    assert_eq!(
        s::bit_length(&b).iter().collect::<Vec<_>>(),
        [Some(8), None]
    );
}

// ---- encodings ----

fn bins(v: &[Option<&[u8]>]) -> BinaryArray {
    BinaryArray::from(v.to_vec())
}

#[test]
fn base64_round_trips_with_mime_line_breaks() {
    let b = bins(&[
        Some(b"Spark SQL"),
        Some(b""),
        None,
        Some(&[0, 0xFF, 0x7F, 0x80]),
        Some(b"a"),
        Some(b"ab"),
    ]);
    let got = s::base64(&b, true).unwrap();
    assert_eq!(
        out(&got),
        [
            Some("U3BhcmsgU1FM"),
            Some(""),
            None,
            Some("AP9/gA=="),
            Some("YQ=="),
            Some("YWI=")
        ]
    );
    // 57 bytes are 76 symbols: a break only when more follow.
    let line = vec![b'a'; 57];
    let two = vec![b'a'; 58];
    let got = s::base64(&bins(&[Some(&line), Some(&two)]), true).unwrap();
    assert_eq!(got.value(0).len(), 76);
    assert_eq!(got.value(1).len(), 76 + 2 + 4);
    assert_eq!(&got.value(1)[76..78], "\r\n");
    let plain = s::base64(&bins(&[Some(&two)]), false).unwrap();
    assert_eq!(plain.value(0).len(), 80);
}

#[test]
fn unbase64_is_java_mime_decoding() {
    let ok = |text: &str| {
        s::unbase64(&strs(&[Some(text)]), false)
            .map(|a| a.value(0).to_vec())
            .map_err(|e| e.error_class())
    };
    assert_eq!(ok("U3BhcmsgU1FM").unwrap(), b"Spark SQL");
    assert_eq!(ok("").unwrap(), b"");
    assert_eq!(ok("abc").unwrap(), [0x69, 0xB7]);
    assert_eq!(ok("a b c d").unwrap(), [0x69, 0xB7, 0x1D]);
    assert_eq!(ok("YQ==").unwrap(), b"a");
    assert_eq!(ok("\u{FFFD}(\u{FFFD}").unwrap(), b"");
    for bad in [
        "1",
        " \t a\u{3000}\n ",
        "=",
        "YQ=x",
        "YQ= =",
        "YQ==YQ",
        "e\u{301}",
    ] {
        assert_eq!(
            ok(bad).unwrap_err(),
            ErrorClass::KrishivInvalidBase64Input,
            "{bad:?}"
        );
    }
    // Every `mime_decode` reason is pinned, not just the class.
    let reason = |text: &str| {
        s::unbase64(&strs(&[Some(text)]), false)
            .unwrap_err()
            .error()
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "reason")
            .map(|(_, v)| v.clone())
            .unwrap_or_default()
    };
    assert_eq!(
        reason("1"),
        "Last unit does not have enough valid bits",
        "single symbol"
    );
    assert_eq!(
        reason("e\u{301}"),
        "Last unit does not have enough valid bits",
        "one valid symbol plus skipped"
    );
    assert_eq!(
        reason("="),
        "Input byte array has wrong 4-byte ending unit",
        "padding with no data"
    );
    assert_eq!(
        reason("YQ=x"),
        "Input byte array has wrong 4-byte ending unit",
        "lone padding before data"
    );
    assert_eq!(
        reason("YQ==YQ"),
        "Input byte array has incorrect ending byte",
        "data after padding"
    );
    let nulls = s::unbase64(&strs(&[None]), true).unwrap();
    assert!(nulls.is_null(0));
}

#[test]
fn to_binary_formats() {
    let conv = |text: &str, fmt: Option<&str>, try_: bool| {
        let fmt = fmt.map(|f| lit(Some(f)));
        s::to_binary(StrOperand::Array(&r1(text)), fmt, try_)
            .map(|a| a.iter().next().flatten().map(<[u8]>::to_vec))
    };
    assert_eq!(
        conv("737472696E67", None, false).unwrap().unwrap(),
        b"string"
    );
    assert_eq!(conv("1", Some("HEX"), false).unwrap().unwrap(), [1]);
    assert_eq!(conv("", None, false).unwrap().unwrap(), b"");
    assert_eq!(
        conv("大千", Some("Utf8"), false).unwrap().unwrap(),
        "大千".as_bytes()
    );
    assert_eq!(
        conv(" ab c=", Some("base64"), false).unwrap().unwrap(),
        [0x69, 0xB7]
    );
    assert_eq!(conv("", Some("base64"), false).unwrap().unwrap(), b"");
    for (text, fmt) in [
        ("GG", "hex"),
        ("01 AF", "hex"),
        ("a", "base64"),
        ("abcd=", "base64"),
        ("a===", "base64"),
        ("ab==f", "base64"),
        (" ab cdef= = ", "base64"),
    ] {
        let e = conv(text, Some(fmt), false).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::ConversionInvalidInput,
            "{text}"
        );
        assert_eq!(conv(text, Some(fmt), true).unwrap(), None, "{text}");
    }
    let e = conv("abc", Some("bogus"), false).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert_eq!(conv("abc", Some("bogus"), true).unwrap(), None);
    // The error quotes the string and the format as Spark does.
    let e = conv("GG", Some("hex"), false).unwrap_err();
    let params: Vec<_> = e
        .error()
        .message_parameters()
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    assert!(
        params.contains(&"str='GG'".to_owned()) && params.contains(&"fmt='HEX'".to_owned()),
        "{params:?}"
    );
    // NULL text or NULL format is NULL.
    assert_eq!(
        s::to_binary(
            StrOperand::Array(&strs(&[None])),
            Some(lit(Some("hex"))),
            false
        )
        .unwrap()
        .null_count(),
        1
    );
    assert_eq!(
        s::to_binary(StrOperand::Array(&r1("a")), Some(lit(None)), false)
            .unwrap()
            .null_count(),
        1
    );
}

#[test]
fn encode_and_decode_charsets() {
    let enc = |text: &str, cs: &str| {
        s::encode(StrOperand::Array(&r1(text)), lit(Some(cs)))
            .map(|a| a.value(0).to_vec())
            .map_err(|e| e.error_class())
    };
    assert_eq!(enc("abc", "utf-8").unwrap(), b"abc");
    assert_eq!(enc("é", "ISO-8859-1").unwrap(), [0xE9]);
    assert_eq!(
        enc("é", "us-ascii").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(
        enc("€", "iso-8859-1").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(
        enc("a😀", "UTF-16").unwrap(),
        [0xFE, 0xFF, 0, b'a', 0xD8, 0x3D, 0xDE, 0]
    );
    assert_eq!(enc("a", "utf-16be").unwrap(), [0, b'a']);
    assert_eq!(enc("a", "utf-16le").unwrap(), [b'a', 0]);
    assert_eq!(enc("a", "utf-32").unwrap(), [0, 0, 0, b'a']);
    assert_eq!(enc("", "utf-16").unwrap(), b"");
    assert_eq!(
        enc("abc", "WINDOWS-1252").unwrap_err(),
        ErrorClass::InvalidParameterValueCharset
    );
    assert_eq!(
        enc("", "bogus").unwrap_err(),
        ErrorClass::InvalidParameterValueCharset
    );
    let dec = |bytes: &[u8], cs: &str| {
        s::decode(
            krishiv_kernels::string::BinOperand::Array(&bins(&[Some(bytes)])),
            lit(Some(cs)),
        )
        .map(|a| a.value(0).to_owned())
        .map_err(|e| e.error_class())
    };
    assert_eq!(dec("日本".as_bytes(), "UTF-8").unwrap(), "日本");
    assert_eq!(dec(&[0xE9], "iso-8859-1").unwrap(), "é");
    assert_eq!(
        dec(&[0xE9], "us-ascii").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(
        dec(&[0xFF], "utf-8").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(dec(&[0xFE, 0xFF, 0, b'a'], "utf-16").unwrap(), "a");
    assert_eq!(dec(&[0xFF, 0xFE, b'a', 0], "utf-16").unwrap(), "a");
    assert_eq!(dec(&[0, b'a'], "utf-16").unwrap(), "a");
    assert_eq!(dec(&[0xFE, 0xFF], "utf-16be").unwrap(), "\u{FEFF}");
    assert_eq!(
        dec(&[0xD8, 0x3D], "utf-16be").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(
        dec(&[0], "utf-16le").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(dec(&[0, 0, 0, b'a'], "utf-32").unwrap(), "a");
    assert_eq!(
        dec(&[0, 0, 0xFE, 0xFF, 0, 0, 0, b'a'], "utf-32").unwrap(),
        "a"
    );
    assert_eq!(
        dec(&[0xFF, 0xFE, 0, 0, b'a', 0, 0, 0], "utf-32").unwrap(),
        "a"
    );
    assert_eq!(
        dec(&[0, 0x11, 0, 0], "utf-32").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(
        dec(&[0, 0, 0], "utf-32").unwrap_err(),
        ErrorClass::MalformedCharacterCoding
    );
    assert_eq!(dec(b"", "utf-16").unwrap(), "");
    assert_eq!(
        dec(b"a", "bogus").unwrap_err(),
        ErrorClass::InvalidParameterValueCharset
    );
    // NULL operands are NULL and never raise.
    let a = s::encode(StrOperand::Array(&r1("a")), lit(None)).unwrap();
    assert!(a.is_null(0));
    let cs = strs(&[Some("utf-8"), Some("bogus")]);
    let t = strs(&[Some("a"), None]);
    let got = s::encode(StrOperand::Array(&t), StrOperand::Array(&cs)).unwrap();
    assert_eq!(got.iter().collect::<Vec<_>>(), [Some(&b"a"[..]), None]);
}

#[test]
fn utf8_validation_of_binary() {
    let b = bins(&[
        Some(b"abc"),
        Some(b""),
        None,
        Some(&[0x80]),
        Some(&[0x61, 0xC2, 0x62]),
        Some("日本".as_bytes()),
        Some(&[0xE6, 0x97]),
    ]);
    let valid: Vec<_> = s::is_valid_utf8(&b).iter().collect();
    assert_eq!(
        valid,
        [
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(false)
        ]
    );
    let made = s::make_valid_utf8(&b).unwrap();
    assert_eq!(
        out(&made),
        [
            Some("abc"),
            Some(""),
            None,
            Some("\u{FFFD}"),
            Some("a\u{FFFD}b"),
            Some("日本"),
            Some("\u{FFFD}")
        ]
    );
    let tried = s::validate_utf8(&b, true).unwrap();
    assert_eq!(
        out(&tried),
        [Some("abc"), Some(""), None, None, None, Some("日本"), None]
    );
    let strict = s::validate_utf8(&b, false).unwrap_err();
    assert_eq!(strict.error_class(), ErrorClass::InvalidUtf8String);
    assert_eq!(strict.row(), Some(3));
    assert_eq!(strict.error().message_parameters()[0].1, "\\x80");
    // Rows after an error are never reached; rows before it are fine.
    assert!(s::validate_utf8(&b.slice(0, 3), false).is_ok());
    let _: BooleanArray = s::is_valid_utf8(&b);
}

#[test]
fn url_encoding_and_decoding() {
    let a = strs(&[
        Some("https://spark.apache.org/a b?x=1&y=é"),
        Some(""),
        None,
        Some("A-z_0.9*"),
    ]);
    let enc = s::url_encode(&a).unwrap();
    assert_eq!(
        out(&enc),
        [
            Some("https%3A%2F%2Fspark.apache.org%2Fa+b%3Fx%3D1%26y%3D%C3%A9"),
            Some(""),
            None,
            Some("A-z_0.9*")
        ]
    );
    let dec = |text: &str, try_: bool| {
        s::url_decode(&strs(&[Some(text)]), try_)
            .map(|a| a.iter().next().flatten().map(str::to_owned))
            .map_err(|e| (e.error_class(), e.error().message_parameters()[0].1.clone()))
    };
    assert_eq!(
        dec("https%3A%2F%2Fspark.apache.org", false)
            .unwrap()
            .unwrap(),
        "https://spark.apache.org"
    );
    assert_eq!(dec("a+b%20c", false).unwrap().unwrap(), "a b c");
    assert_eq!(dec("%C3%A9%e6%97%a5", false).unwrap().unwrap(), "é日");
    assert_eq!(dec("%FF%41", false).unwrap().unwrap(), "\u{FFFD}A");
    assert_eq!(dec("%+f%-0", false).unwrap().unwrap(), "\u{0F}\0");
    assert_eq!(dec("%٤١", false).unwrap().unwrap(), "A");
    assert_eq!(dec("", false).unwrap().unwrap(), "");
    for bad in ["%", "%4", "%zz", "a%2", "%-1", "%%41", "%4g"] {
        assert_eq!(
            dec(bad, false).unwrap_err(),
            (ErrorClass::CannotDecodeUrl, bad.to_owned()),
            "{bad}"
        );
        assert_eq!(dec(bad, true).unwrap(), None, "{bad}");
    }
    // Rows are independent: one bad row fails the call only without `try_`.
    let rows = strs(&[Some("%41"), Some("%"), None]);
    assert_eq!(
        out(&s::url_decode(&rows, true).unwrap()),
        [Some("A"), None, None]
    );
    assert_eq!(s::url_decode(&rows, false).unwrap_err().row(), Some(1));
}

#[test]
fn large_offsets_work_too() {
    let a = LargeStringArray::from(vec![Some("abc"), None]);
    let r = s::reverse(&a).unwrap();
    assert_eq!(r.iter().collect::<Vec<_>>(), [Some("cba"), None]);
    let q = s::quote(&a).unwrap();
    assert_eq!(q.value(0), "'abc'");
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 256 }, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn reverse_is_an_involution(text in ".{0,40}") {
        let twice = one(|a| s::reverse(a).unwrap(), &text);
        let twice = one(|a| s::reverse(a).unwrap(), &twice);
        prop_assert_eq!(twice, text);
    }

    #[test]
    fn left_and_right_have_the_requested_length(text in ".{0,30}", n in -3_i32..40) {
        let chars = text.chars().count();
        let want = usize::try_from(n).unwrap_or(0).min(chars);
        let r = s::right(StrOperand::Array(&r1(&text)), int(n)).unwrap();
        prop_assert_eq!(r.value(0).chars().count(), want);
        prop_assert!(text.ends_with(r.value(0)));
    }

    #[test]
    fn repeat_has_n_copies(text in ".{0,8}", n in -2_i32..6) {
        let r = s::repeat(StrOperand::Array(&r1(&text)), int(n)).unwrap();
        prop_assert_eq!(r.value(0), text.repeat(usize::try_from(n).unwrap_or(0)));
    }

    #[test]
    fn base64_then_unbase64_is_the_identity(bytes in proptest::collection::vec(any::<u8>(), 0..200)) {
        let encoded = s::base64(&bins(&[Some(&bytes)]), true).unwrap();
        let decoded = s::unbase64(&encoded, true).unwrap();
        prop_assert_eq!(decoded.value(0), &bytes[..]);
    }

    #[test]
    fn url_encode_then_decode_is_the_identity(text in ".{0,40}") {
        let enc = s::url_encode(&strs(&[Some(&text)])).unwrap();
        let dec = s::url_decode(&enc, false).unwrap();
        prop_assert_eq!(dec.value(0), text);
    }

    #[test]
    fn charsets_round_trip(text in "[^\u{feff}]{0,30}") {
        for cs in ["utf-8", "utf-16", "utf-16be", "utf-16le", "utf-32"] {
            let bytes = s::encode(StrOperand::Array(&r1(&text)), lit(Some(cs))).unwrap();
            let back = s::decode(
                krishiv_kernels::string::BinOperand::Array(&bins(&[Some(bytes.value(0))])),
                lit(Some(cs)),
            )
            .unwrap();
            prop_assert_eq!(back.value(0), text.as_str());
        }
    }

    #[test]
    fn levenshtein_is_a_metric(a in "[a-c日]{0,8}", b in "[a-c日]{0,8}", c in "[a-c日]{0,8}") {
        let d = |x: &str, y: &str| {
            s::levenshtein(StrOperand::Array(&r1(x)), lit(Some(y)), None).unwrap().value(0)
        };
        prop_assert_eq!(d(&a, &a), 0);
        prop_assert_eq!(d(&a, &b), d(&b, &a));
        prop_assert!(d(&a, &c) <= d(&a, &b) + d(&b, &c));
        let max = i32::try_from(a.chars().count().max(b.chars().count())).unwrap();
        prop_assert!(d(&a, &b) <= max);
    }

    #[test]
    fn locate_finds_what_str_find_finds(text in "[ab日]{0,12}", sub in "[ab日]{1,3}", start in 1_i32..14) {
        let got = s::locate(StrOperand::Array(&r1(&sub)), lit(Some(&text)), int(start)).unwrap().value(0);
        let skip = usize::try_from(start - 1).unwrap();
        let tail: String = text.chars().skip(skip).collect();
        let want = tail.find(&sub).map_or(0, |at| {
            i32::try_from(tail[..at].chars().count() + skip + 1).unwrap()
        });
        prop_assert_eq!(got, want);
    }

    #[test]
    fn split_part_pieces_join_back(text in "[a.]{0,12}") {
        let count = text.split('.').count();
        let mut pieces = Vec::new();
        for n in 1..=count {
            let n = i32::try_from(n).unwrap();
            pieces.push(s::split_part(StrOperand::Array(&r1(&text)), lit(Some(".")), int(n)).unwrap().value(0).to_owned());
        }
        prop_assert_eq!(pieces.join("."), text);
    }

    #[test]
    fn mask_keeps_the_length_of_bmp_text(text in "[a-zA-Z0-9 -]{0,20}") {
        let r = s::mask(&strs(&[Some(&text)]), MaskRules::default()).unwrap();
        prop_assert_eq!(r.value(0).chars().count(), text.chars().count());
        prop_assert!(r.value(0).chars().all(|c| matches!(c, 'X' | 'x' | 'n' | ' ' | '-')));
    }
}
