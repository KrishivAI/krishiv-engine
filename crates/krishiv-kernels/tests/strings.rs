// SPDX-License-Identifier: Apache-2.0
//! String kernels against values recorded from Spark 4.2 (4.0 and 4.1 gave
//! the same values), with `Utf8` and `LargeUtf8`, column and literal
//! operands, NULL rows, and property tests against independent oracles.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, BinaryArray, GenericListArray, Int32Array, LargeBinaryArray, LargeStringArray,
    ListArray, StringArray, cast::AsArray,
};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::string::{
    BinOperand, LikePattern, MatchKind, PadSide, StrOperand, TrimSide, byte_length, char_length,
    concat, concat_binary, like, literal_delimiter, lower, match_binary, match_str, pad,
    pad_binary, replace, split_literal, substring, substring_binary, trim, upper,
};
use krishiv_kernels::{KernelCtx, Operand, SparkVersion};
use proptest::prelude::*;

type S<'a> = StrOperand<'a, i32>;

fn strs(v: &[Option<&str>]) -> StringArray {
    StringArray::from(v.to_vec())
}

fn rows(a: &StringArray) -> Vec<Option<String>> {
    a.iter().map(|s| s.map(str::to_owned)).collect()
}

fn some(v: &[&str]) -> Vec<Option<String>> {
    v.iter().map(|s| Some((*s).to_owned())).collect()
}

fn ctx(v: SparkVersion) -> KernelCtx {
    KernelCtx::new().with_spark_version(v)
}

fn int(v: i32) -> Operand<'static, Int32Type> {
    Operand::Scalar(Some(v))
}

#[test]
fn case_mapping_matches_spark() {
    let c = ctx(SparkVersion::V4_2);
    // (input, upper, lower) recorded from Spark.
    let table = [
        ("SparkSql", "SPARKSQL", "sparksql"),
        ("straße", "STRASSE", "straße"),
        ("İ", "İ", "i\u{307}"),
        ("ΑΣ", "ΑΣ", "ας"),
        ("ΑΣ Σ", "ΑΣ Σ", "ας σ"),
        ("ǆ", "Ǆ", "ǆ"),
        ("ŉ", "ʼN", "ŉ"),
        ("ﬁ", "FI", "ﬁ"),
        ("", "", ""),
    ];
    let a = strs(&table.map(|(s, ..)| Some(s)));
    let (up, low) = (upper(&a, &c).unwrap(), lower(&a, &c).unwrap());
    for (i, (_, u, l)) in table.iter().enumerate() {
        assert_eq!((up.value(i), low.value(i)), (*u, *l));
    }
    let large = LargeStringArray::from(vec![Some("aÄ"), None]);
    let out = upper(&large, &c).unwrap();
    assert_eq!((out.value(0), out.is_null(1)), ("AÄ", true));
}

#[test]
fn unicode_17_letters_are_unmapped_before_spark_4_2() {
    // U+A7CE (capital) / U+A7CF (small) are Unicode 17 additions. Without
    // them, the sigma of "aΣ<U+A7CE>b" is final (nothing cased follows).
    let a = strs(&[Some("aΣ\u{A7CE}b"), Some("\u{A7CF}z")]);
    let new = lower(&a, &ctx(SparkVersion::V4_2)).unwrap();
    assert_eq!(
        rows(&new),
        some(&["aσ\u{A7CF}b", "\u{A7CF}z"]),
        "Unicode 17 cases U+A7CE"
    );
    for v in [SparkVersion::V4_0, SparkVersion::V4_1] {
        assert_eq!(
            rows(&lower(&a, &ctx(v)).unwrap()),
            some(&["aς\u{A7CE}b", "\u{A7CF}z"])
        );
        assert_eq!(
            rows(&upper(&a, &ctx(v)).unwrap()),
            some(&["AΣ\u{A7CE}B", "\u{A7CF}Z"])
        );
    }
}

#[test]
fn lengths() {
    let a = strs(&[
        Some("héllo"),
        Some("😀"),
        Some(""),
        None,
        Some("Spark SQL "),
    ]);
    let n = char_length(&a);
    assert_eq!(
        n.iter().collect::<Vec<_>>(),
        [Some(5), Some(1), Some(0), None, Some(10)]
    );
    let b = BinaryArray::from(vec![Some(&b"\xaa\xbb"[..]), None, Some(&b""[..])]);
    assert_eq!(
        byte_length(&b).iter().collect::<Vec<_>>(),
        [Some(2), None, Some(0)]
    );
    let large = LargeStringArray::from(vec!["é"]);
    assert_eq!(char_length(&large).value(0), 1);
}

#[test]
fn trimming_matches_spark() {
    let a = strs(&[Some("  a  "), Some("xxaxx"), None, Some(" \t a\u{3000}\n ")]);
    let t = |side, chars: Option<S<'_>>| rows(&trim(side, S::Array(&a), chars).unwrap());
    // Default: spaces only, no other whitespace.
    assert_eq!(
        t(TrimSide::Both, None),
        [Some("a"), Some("xxaxx"), None, Some("\t a\u{3000}\n")].map(|s| s.map(str::to_owned))
    );
    assert_eq!(t(TrimSide::Leading, None)[0].as_deref(), Some("a  "));
    assert_eq!(t(TrimSide::Trailing, None)[0].as_deref(), Some("  a"));
    let x = Some(S::Scalar(Some("x")));
    assert_eq!(t(TrimSide::Both, x)[1].as_deref(), Some("a"));
    assert_eq!(t(TrimSide::Leading, x)[1].as_deref(), Some("axx"));
    assert_eq!(t(TrimSide::Trailing, x)[1].as_deref(), Some("xxa"));
    // An empty trim string trims nothing; a NULL one gives NULL.
    assert_eq!(
        t(TrimSide::Both, Some(S::Scalar(Some(""))))[1].as_deref(),
        Some("xxaxx")
    );
    assert_eq!(t(TrimSide::Both, Some(S::Scalar(None))), vec![None; 4]);
    // A set of characters, and a column of trim strings.
    let set = strs(&[Some("a "), Some("xa"), Some("x"), Some("")]);
    let out = trim(TrimSide::Both, S::Array(&a), Some(S::Array(&set))).unwrap();
    assert_eq!(
        rows(&out),
        [Some(""), Some(""), None, Some(" \t a\u{3000}\n ")].map(|s| s.map(str::to_owned))
    );
}

#[test]
fn contains_family() {
    let a = strs(&[Some("Spark SQL"), Some(""), None, Some("é")]);
    let probe = |kind, needle: &str| -> Vec<Option<bool>> {
        match_str(kind, S::Array(&a), S::Scalar(Some(needle)))
            .unwrap()
            .iter()
            .collect()
    };
    let quad = |x| vec![Some(x), Some(false), None, Some(false)];
    assert_eq!(probe(MatchKind::StartsWith, "Spark"), quad(true));
    assert_eq!(probe(MatchKind::EndsWith, "Spark"), quad(false));
    assert_eq!(probe(MatchKind::Contains, "k S"), quad(true));
    assert_eq!(
        probe(MatchKind::StartsWith, ""),
        vec![Some(true), Some(true), None, Some(true)]
    );
    // Column against column, and a NULL needle.
    let b = strs(&[Some("SQL"), Some(""), Some("x"), None]);
    let out = match_str(MatchKind::EndsWith, S::Array(&a), S::Array(&b)).unwrap();
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [Some(true), Some(true), None, None]
    );
    assert_eq!(
        match_str(MatchKind::Contains, S::Array(&a), S::Scalar(None))
            .unwrap()
            .null_count(),
        4
    );
    // Binary.
    let bin = BinaryArray::from(vec![Some(&b"\xaa\xbb\xcc"[..]), Some(&b""[..]), None]);
    let t = |k, n: &'static [u8]| {
        match_binary(k, BinOperand::Array(&bin), BinOperand::Scalar(Some(n)))
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
    };
    assert_eq!(
        t(MatchKind::StartsWith, b"\xaa"),
        [Some(true), Some(false), None]
    );
    assert_eq!(
        t(MatchKind::EndsWith, b"\xcc"),
        [Some(true), Some(false), None]
    );
    assert_eq!(
        t(MatchKind::Contains, b"\xbb"),
        [Some(true), Some(false), None]
    );
    assert_eq!(t(MatchKind::Contains, b""), [Some(true), Some(true), None]);
    assert_eq!(
        t(MatchKind::Contains, b"\xcc\xdd"),
        [Some(false), Some(false), None]
    );
    let large = LargeBinaryArray::from(vec![&b"ab"[..]]);
    let out = match_binary(
        MatchKind::Contains,
        BinOperand::Array(&large),
        BinOperand::Scalar(Some(b"b")),
    )
    .unwrap();
    assert!(out.value(0));
}

#[test]
fn replacing() {
    let r = |s: &str, from: &str, to: &str| {
        let a = strs(&[Some(s)]);
        let out = replace(S::Array(&a), S::Scalar(Some(from)), S::Scalar(Some(to))).unwrap();
        out.value(0).to_owned()
    };
    assert_eq!(r("aaa", "aa", "b"), "ba");
    assert_eq!(r("abc", "", "x"), "abc");
    assert_eq!(r("abc", "b", ""), "ac");
    assert_eq!(r("abcabc", "bc", "é"), "aéaé");
    assert_eq!(r("abc", "z", "y"), "abc");
    assert_eq!(r("ABCabc", "abc", "Ä€日本語😀é"), "ABCÄ€日本語😀é");
    let a = strs(&[Some("a"), None]);
    let n = replace(S::Array(&a), S::Scalar(Some("a")), S::Scalar(None)).unwrap();
    assert_eq!(n.null_count(), 2);
    let n = replace(S::Array(&a), S::Scalar(Some("a")), S::Scalar(Some("b"))).unwrap();
    assert_eq!(rows(&n), [Some("b".to_owned()), None]);
}

#[test]
fn padding_matches_spark() {
    let p = |side, s: &str, len: i32, fill: Option<&str>| {
        let a = strs(&[Some(s)]);
        let out = pad(
            side,
            S::Array(&a),
            int(len),
            fill.map(|f| S::Scalar(Some(f))),
        )
        .unwrap();
        out.value(0).to_owned()
    };
    use PadSide::{Left, Right};
    assert_eq!(p(Left, "hi", 5, Some("??")), "???hi");
    assert_eq!(p(Right, "hi", 5, None), "hi   ");
    assert_eq!(p(Left, "héllo", 8, Some("é")), "éééhéllo");
    assert_eq!(p(Right, "😀", 3, Some("😀x")), "😀😀x");
    assert_eq!(p(Left, "hi", 0, Some("x")), "");
    assert_eq!(p(Left, "hi", -1, Some("x")), "");
    assert_eq!(p(Left, "hi", 5, Some("")), "hi");
    assert_eq!(p(Left, "hi", 1, Some("")), "h");
    assert_eq!(p(Left, "hi", 5, Some("Ä€日本語😀é")), "Ä€日hi");
    let a = strs(&[Some("a"), None]);
    let lens = Int32Array::from(vec![Some(3), Some(3)]);
    let out = pad(Left, S::Array(&a), Operand::Array(&lens), None).unwrap();
    assert_eq!(rows(&out), [Some("  a".to_owned()), None]);
    let null_len = pad(Left, S::Array(&a), Operand::Scalar(None), None).unwrap();
    assert_eq!(null_len.null_count(), 2);
}

#[test]
fn binary_padding_matches_spark() {
    let hex = |a: &arrow_array::GenericBinaryArray<i32>| -> String {
        a.value(0).iter().map(|b| format!("{b:02X}")).collect()
    };
    let p = |side, s: &[u8], len: i32, fill: Option<&[u8]>| {
        let a = BinaryArray::from(vec![s]);
        let out = pad_binary(
            side,
            BinOperand::Array(&a),
            int(len),
            fill.map(|f| BinOperand::Scalar(Some(f))),
        )
        .unwrap();
        hex(&out)
    };
    use PadSide::{Left, Right};
    assert_eq!(p(Left, b"\xaa", 4, None), "000000AA");
    assert_eq!(p(Left, b"\xaa\xbb\xcc", 2, Some(b"\xff")), "AABB");
    assert_eq!(p(Right, b"\xaa", 5, Some(b"")), "AA");
    // Spark raises NegativeArraySizeException for a negative length here;
    // the string overload and Krishiv give the empty value (ADR 0023).
    assert_eq!(p(Left, b"\xaa", -1, Some(b"\x11")), "");
    assert_eq!(p(Right, b"\xaa\xbb", 5, Some(b"\x11\x22")), "AABB112211");
    assert_eq!(p(Left, b"", 5, Some(b"\x1f\x2e")), "1F2E1F2E1F");
    let n = BinaryArray::from(vec![Some(&b"a"[..]), None]);
    let out = pad_binary(Left, BinOperand::Array(&n), int(3), None).unwrap();
    assert_eq!(out.null_count(), 1);
}

#[test]
fn substrings_match_spark() {
    let sub = |s: &str, pos: i32, len: Option<i32>| {
        let a = strs(&[Some(s)]);
        let out = substring(S::Array(&a), int(pos), len.map(int)).unwrap();
        out.value(0).to_owned()
    };
    assert_eq!(sub("Spark SQL", 0, None), "Spark SQL");
    assert_eq!(sub("Spark SQL", -3, None), "SQL");
    assert_eq!(sub("Spark SQL", -3, Some(2)), "SQ");
    assert_eq!(sub("Spark SQL", 2, Some(-1)), "");
    assert_eq!(sub("Spark SQL", -20, Some(12)), "S");
    assert_eq!(sub("héllo😀", 2, Some(3)), "éll");
    assert_eq!(sub("héllo😀", -2, None), "o😀");
    assert_eq!(sub("abc", 1, Some(i32::MAX)), "abc");
    assert_eq!(sub("abc", i32::MAX, None), "");
    assert_eq!(sub("abc", i32::MIN, Some(i32::MAX)), "ab");
    assert_eq!(sub("abc", -1, Some(i32::MAX)), "c");
    // Columns for the position and length; NULL operands.
    let a = strs(&[Some("abcdef"), Some("abcdef"), None, Some("abcdef")]);
    let pos = Int32Array::from(vec![Some(2), Some(-2), Some(1), None]);
    let len = Int32Array::from(vec![Some(3), None, Some(1), Some(1)]);
    let out = substring(
        S::Array(&a),
        Operand::Array(&pos),
        Some(Operand::Array(&len)),
    )
    .unwrap();
    assert_eq!(rows(&out), [Some("bcd".to_owned()), None, None, None]);
    // Binary substring counts bytes.
    let b = BinaryArray::from(vec![Some(&b"\xaa\xbb\xcc"[..]), None]);
    let out = substring_binary(BinOperand::Array(&b), int(-2), Some(int(1))).unwrap();
    assert_eq!((out.value(0), out.is_null(1)), (&b"\xbb"[..], true));
    let out = substring_binary(BinOperand::Array(&b), int(5), None).unwrap();
    assert_eq!(out.value(0), b"");
    let out = substring_binary(BinOperand::Array(&b), int(0), Some(int(2))).unwrap();
    assert_eq!(out.value(0), b"\xaa\xbb");
}

#[test]
fn concatenation() {
    let a = strs(&[Some("a"), None, Some("é")]);
    let b = strs(&[Some("b"), Some("x"), Some("")]);
    let out = concat(&[S::Array(&a), S::Array(&b), S::Scalar(Some("!"))]).unwrap();
    assert_eq!(
        rows(&out),
        [Some("ab!".to_owned()), None, Some("é!".to_owned())]
    );
    let out = concat(&[S::Array(&a), S::Scalar(None)]).unwrap();
    assert_eq!(out.null_count(), 3);
    assert_eq!(rows(&concat(&[S::Array(&b)]).unwrap()), rows(&b));
    let x = BinaryArray::from(vec![Some(&b"\xaa"[..]), None]);
    let out = concat_binary(&[BinOperand::Array(&x), BinOperand::Scalar(Some(b"\xbb"))]).unwrap();
    assert_eq!((out.value(0), out.is_null(1)), (&b"\xaa\xbb"[..], true));
    assert!(concat::<i32>(&[S::Scalar(Some("a"))]).is_err());
    let large = LargeStringArray::from(vec!["l"]);
    let out = concat(&[StrOperand::Array(&large), StrOperand::Scalar(Some("r"))]).unwrap();
    assert_eq!(out.value(0), "lr");
}

#[test]
fn like_matches_spark() {
    let a = strs(&[
        Some("Spark"),
        Some("spark"),
        Some("a_b"),
        Some("a%b"),
        None,
        Some(""),
    ]);
    let l = |pattern: &str, esc: char| -> Vec<Option<bool>> {
        like(S::Array(&a), S::Scalar(Some(pattern)), esc)
            .unwrap()
            .iter()
            .collect()
    };
    assert_eq!(
        l("_park", '\\'),
        [
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            None,
            Some(false)
        ]
    );
    assert_eq!(l("S%", '\\')[..2], [Some(true), Some(false)]);
    assert_eq!(l("", '\\')[5], Some(true));
    assert_eq!(l("a\\_b", '\\')[2..4], [Some(true), Some(false)]);
    assert_eq!(l("a$%b", '$')[2..4], [Some(false), Some(true)]);
    assert_eq!(l("%", '\\')[4], None);
    let e = like(S::Array(&a), S::Scalar(Some("a\\")), '\\').unwrap_err();
    assert_eq!(
        (e.row(), e.error_class()),
        (Some(0), ErrorClass::InvalidFormatEscAtTheEnd)
    );
    let e = like(S::Array(&a), S::Scalar(Some("a\\x")), '\\').unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidFormatEscInTheMiddle);
    // A column of patterns, repeated ones compiled once.
    let p = strs(&[
        Some("S%"),
        Some("S%"),
        Some("a_b"),
        Some("%b"),
        Some("x"),
        Some(""),
    ]);
    let out = like(S::Array(&a), S::Array(&p), '\\').unwrap();
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            None,
            Some(true)
        ]
    );
    // A NULL row never compiles (or raises on) its pattern.
    let bad = strs(&[Some("a\\"), Some("%")]);
    let one_null = strs(&[None, Some("x")]);
    let out = like(S::Array(&one_null), S::Array(&bad), '\\').unwrap();
    assert_eq!(out.iter().collect::<Vec<_>>(), [None, Some(true)]);
    let large = LargeStringArray::from(vec!["é"]);
    let out = like(
        StrOperand::Array(&large),
        StrOperand::Scalar(Some("_")),
        '\\',
    )
    .unwrap();
    assert!(out.value(0));
    assert!(LikePattern::compile("a%", '\\', 0).unwrap().matches("abc"));
}

fn list_rows(l: &ListArray) -> Vec<Option<Vec<String>>> {
    (0..l.len())
        .map(|i| {
            l.is_valid(i).then(|| {
                l.value(i)
                    .as_string::<i32>()
                    .iter()
                    .map(|s| s.unwrap().to_owned())
                    .collect()
            })
        })
        .collect()
}

#[test]
fn split_matches_spark() {
    let s = |input: &str, d: &str, limit: i32| {
        let a = strs(&[Some(input)]);
        let out = split_literal(S::Array(&a), d, int(limit)).unwrap();
        list_rows(&out).remove(0).unwrap()
    };
    assert_eq!(s("a,b,,", ",", -1), ["a", "b", "", ""]);
    assert_eq!(s("a,b,,", ",", 2), ["a", "b,,"]);
    assert_eq!(s("", ",", -1), [""]);
    assert_eq!(s("abc", "", -1), ["a", "b", "c"]);
    assert_eq!(s("a😀b", "😀", 0), ["a", "b"]);
    let a = strs(&[Some("a1b"), None, Some("1")]);
    let limits = Int32Array::from(vec![Some(-1), Some(-1), Some(2)]);
    let out = split_literal(S::Array(&a), "1", Operand::Array(&limits)).unwrap();
    assert_eq!(
        list_rows(&out),
        [
            Some(vec!["a".to_owned(), "b".to_owned()]),
            None,
            Some(vec![String::new(), String::new()])
        ]
    );
    let items: &GenericListArray<i32> = &out;
    assert!(!items.values().is_nullable());
    assert_eq!(literal_delimiter("\\.").as_deref(), Some("."));
    assert_eq!(literal_delimiter("[ABC]"), None);
}

#[test]
fn mismatched_operands_are_engine_errors() {
    let a = strs(&[Some("a"), Some("b")]);
    let b = strs(&[Some("a")]);
    let e = replace(S::Array(&a), S::Array(&b), S::Scalar(Some(""))).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    let e = like::<i32>(S::Scalar(Some("a")), S::Scalar(Some("a")), '\\').unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

fn small_text() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        prop_oneof![
            Just('a'),
            Just('b'),
            Just('%'),
            Just('_'),
            Just('\\'),
            Just('é'),
            Just('😀'),
            Just(' ')
        ],
        0..8,
    )
    .prop_map(|c| c.into_iter().collect())
}

/// `LIKE` by plain recursion over code points.
fn naive_like(p: &[char], s: &[char]) -> bool {
    match p.split_first() {
        None => s.is_empty(),
        Some(('%', rest)) => (0..=s.len()).any(|i| naive_like(rest, &s[i..])),
        Some(('_', rest)) => s.split_first().is_some_and(|(_, t)| naive_like(rest, t)),
        Some((c, rest)) => s
            .split_first()
            .is_some_and(|(d, t)| c == d && naive_like(rest, t)),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: if cfg!(miri) { 2 } else { 512 }, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn like_equals_naive_matching(p in "[ab%_é]{0,7}", s in "[abé😀]{0,7}") {
        let a = strs(&[Some(&s)]);
        let got = like(S::Array(&a), S::Scalar(Some(&p)), '\u{0}').unwrap().value(0);
        let pc: Vec<char> = p.chars().collect();
        let sc: Vec<char> = s.chars().collect();
        prop_assert_eq!(got, naive_like(&pc, &sc), "{:?} like {:?}", s, p);
    }

    #[test]
    fn substring_equals_i128_oracle(s in small_text(), pos in any::<i32>(), len in any::<i32>(), small_pos in -10_i32..10, small_len in -3_i32..10) {
        let chars: Vec<char> = s.chars().collect();
        let n = i128::try_from(chars.len()).unwrap();
        for (pos, len) in [(pos, len), (small_pos, small_len)] {
            let a = strs(&[Some(&s)]);
            let got = substring(S::Array(&a), int(pos), Some(int(len))).unwrap();
            let start = match pos { 1.. => i128::from(pos) - 1, 0 => 0, _ => n + i128::from(pos) };
            let end = (start + i128::from(len)).clamp(i128::from(i32::MIN), i128::from(i32::MAX));
            let (lo, hi) = (start.max(0), end.min(n));
            let expect: String = if lo < hi {
                chars[usize::try_from(lo).unwrap()..usize::try_from(hi).unwrap()].iter().collect()
            } else {
                String::new()
            };
            prop_assert_eq!(got.value(0), expect.as_str(), "substring({:?}, {}, {})", s, pos, len);
        }
    }
}
