// SPDX-License-Identifier: Apache-2.0
//! `LIKE` and string/binary comparison kernels (a column against a literal
//! pattern or operand, or another column, runs on arrow-string / arrow-ord)
//! against the row-at-a-time oracles: Krishiv's own pattern matcher and
//! `reference::compare_bytes`, over random strings with NULLs, multi-byte
//! characters, newlines and escapes.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers outside #[test] fns fail loudly"
)]

use arrow_array::{Array, BinaryArray, BooleanArray, LargeStringArray, StringArray};
use krishiv_kernels::compare::{
    BytesOperand, CompareKind, Equal, Greater, GreaterEq, Less, LessEq, NotEqual, compare_bytes,
};
use krishiv_kernels::reference;
use krishiv_kernels::string::{LikePattern, StrOperand, like};
use proptest::prelude::*;

// Miri interprets every case, so it runs a handful.
#[cfg(not(miri))]
const LIKE_CASES: u32 = 3000;
#[cfg(miri)]
const LIKE_CASES: u32 = 2;
#[cfg(not(miri))]
const COMPARE_CASES: u32 = 2000;
#[cfg(miri)]
const COMPARE_CASES: u32 = 2;

const ALPHABET: &[&str] = &[
    "a", "b", "é", "😀", "\n", "%", "_", "\\", "$", "A", " ", ".", "*", "(",
];

fn text() -> impl Strategy<Value = String> {
    proptest::collection::vec(proptest::sample::select(ALPHABET), 0..8).prop_map(|v| v.concat())
}

fn values() -> impl Strategy<Value = Vec<Option<String>>> {
    proptest::collection::vec(proptest::option::of(text()), 0..40)
}

fn oracle_like(col: &[Option<String>], pat: &str, esc: char) -> BooleanArray {
    let p = LikePattern::compile(pat, esc, 0).unwrap();
    col.iter()
        .map(|v| v.as_deref().map(|s| p.matches(s)))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(LIKE_CASES))]

    #[test]
    fn like_literal_pattern_matches_the_matcher(
        col in values(), pat in text(), esc in proptest::sample::select(vec!['\\', '$', '%', '_', 'a']),
    ) {
        prop_assume!(LikePattern::compile(&pat, esc, 0).is_ok());
        let want = oracle_like(&col, &pat, esc);
        let small: StringArray = col.iter().map(|v| v.as_deref()).collect();
        let got = like::<i32>(StrOperand::Array(&small), StrOperand::Scalar(Some(&pat)), esc).unwrap();
        prop_assert_eq!(&got, &want, "pattern {:?} escape {:?}", pat, esc);
        let large: LargeStringArray = col.iter().map(|v| v.as_deref()).collect();
        let got = like::<i64>(StrOperand::Array(&large), StrOperand::Scalar(Some(&pat)), esc).unwrap();
        prop_assert_eq!(&got, &want);
    }

    #[test]
    fn like_errors_only_on_a_non_null_row(col in values(), pat in text()) {
        let ok = LikePattern::compile(&pat, '\\', 0).is_ok();
        let arr: StringArray = col.iter().map(|v| v.as_deref()).collect();
        let r = like::<i32>(StrOperand::Array(&arr), StrOperand::Scalar(Some(&pat)), '\\');
        let any_valid = arr.null_count() < arr.len();
        prop_assert_eq!(r.is_ok(), ok || !any_valid);
        if let Err(e) = r {
            prop_assert_eq!(e.row(), (0..arr.len()).find(|&i| arr.is_valid(i)));
        }
    }

    #[test]
    fn compare_literal_matches_the_reference(col in values(), lit in text()) {
        let arr: StringArray = col.iter().map(|v| v.as_deref()).collect();
        let l: Vec<Option<&[u8]>> = col.iter().map(|v| v.as_deref().map(str::as_bytes)).collect();
        let r: Vec<Option<&[u8]>> = vec![Some(lit.as_bytes()); l.len()];
        macro_rules! check {
            ($op:ty, $kind:expr) => {
                prop_assert_eq!(
                    compare_bytes::<_, $op>(BytesOperand::Array(&arr), BytesOperand::Scalar(Some(lit.as_bytes()))).unwrap(),
                    reference::compare_bytes($kind, &l, &r)
                );
            };
        }
        check!(Equal, CompareKind::Equal);
        check!(NotEqual, CompareKind::NotEqual);
        check!(Less, CompareKind::Less);
        check!(LessEq, CompareKind::LessEq);
        check!(Greater, CompareKind::Greater);
        check!(GreaterEq, CompareKind::GreaterEq);
    }
}

#[test]
fn null_literal_and_all_null_column() {
    let arr = StringArray::from(vec![None::<&str>, None]);
    let got = like::<i32>(
        StrOperand::Array(&arr),
        StrOperand::Scalar(Some("a\\")),
        '\\',
    )
    .unwrap();
    assert_eq!(got.null_count(), 2);
    let got =
        compare_bytes::<_, Equal>(BytesOperand::Array(&arr), BytesOperand::Scalar(None)).unwrap();
    assert_eq!(got.null_count(), 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(COMPARE_CASES))]

    #[test]
    fn compare_columns_matches_the_reference(l in values(), seed in proptest::collection::vec(proptest::option::of(text()), 0..40)) {
        let n = l.len().min(seed.len());
        let (l, r) = (&l[..n], &seed[..n]);
        let (la, ra): (StringArray, StringArray) = (
            l.iter().map(|v| v.as_deref()).collect(),
            r.iter().map(|v| v.as_deref()).collect(),
        );
        let lb: Vec<Option<&[u8]>> = l.iter().map(|v| v.as_deref().map(str::as_bytes)).collect();
        let rb: Vec<Option<&[u8]>> = r.iter().map(|v| v.as_deref().map(str::as_bytes)).collect();
        macro_rules! check {
            ($op:ty, $kind:expr) => {
                prop_assert_eq!(
                    compare_bytes::<_, $op>(BytesOperand::Array(&la), BytesOperand::Array(&ra)).unwrap(),
                    reference::compare_bytes($kind, &lb, &rb)
                );
            };
        }
        check!(Equal, CompareKind::Equal);
        check!(NotEqual, CompareKind::NotEqual);
        check!(Less, CompareKind::Less);
        check!(LessEq, CompareKind::LessEq);
        check!(Greater, CompareKind::Greater);
        check!(GreaterEq, CompareKind::GreaterEq);
    }
}

#[test]
fn a_literal_that_is_not_utf8_falls_back_to_the_row_loop() {
    // 0xFF cannot sit in a string array, so arrow-ord is not used; byte order still decides.
    let arr = StringArray::from(vec![Some("a"), None, Some("é")]);
    let got = compare_bytes::<_, Less>(
        BytesOperand::Array(&arr),
        BytesOperand::Scalar(Some(&[0xFF])),
    )
    .unwrap();
    assert_eq!(got, BooleanArray::from(vec![Some(true), None, Some(true)]));
}

#[test]
fn a_pattern_over_arrow_strings_regex_limit_still_matches() {
    // ~20 KB of `_%`: arrow-string's regex exceeds its size limit, the matcher does not.
    let pat = format!("a{}%", "_%".repeat(10_000));
    let arr = StringArray::from(vec![Some("a"), Some("abcdef"), None]);
    let want = oracle_like(&[Some("a".into()), Some("abcdef".into()), None], &pat, '\\');
    let got = like::<i32>(
        StrOperand::Array(&arr),
        StrOperand::Scalar(Some(&pat)),
        '\\',
    )
    .unwrap();
    assert_eq!(got, want);
    let long_underscores = format!("{}x", "_".repeat(50_000));
    let want = oracle_like(
        &[Some("a".into()), Some("abcdef".into()), None],
        &long_underscores,
        '\\',
    );
    let got = like::<i32>(
        StrOperand::Array(&arr),
        StrOperand::Scalar(Some(&long_underscores)),
        '\\',
    )
    .unwrap();
    assert_eq!(got, want);
}

#[test]
fn empty_and_sliced_inputs() {
    let empty = StringArray::from(Vec::<Option<&str>>::new());
    let got = like::<i32>(
        StrOperand::Array(&empty),
        StrOperand::Scalar(Some("a\\")),
        '\\',
    )
    .unwrap();
    assert_eq!(got.len(), 0);
    let arr = StringArray::from(vec![None, Some("ab"), Some("zz"), Some("ac")]);
    let slice = arr.slice(1, 3);
    let slice = slice.as_any().downcast_ref::<StringArray>().unwrap();
    let got = like::<i32>(
        StrOperand::Array(slice),
        StrOperand::Scalar(Some("a%")),
        '\\',
    )
    .unwrap();
    assert_eq!(
        got,
        BooleanArray::from(vec![Some(true), Some(false), Some(true)])
    );
    let got =
        compare_bytes::<_, Less>(BytesOperand::Array(slice), BytesOperand::Scalar(Some(b"b")))
            .unwrap();
    assert_eq!(
        got,
        BooleanArray::from(vec![Some(true), Some(false), Some(true)])
    );
    // The error row is the first valid row of the slice, not of the parent.
    let err = like::<i32>(
        StrOperand::Array(slice),
        StrOperand::Scalar(Some("a\\")),
        '\\',
    )
    .unwrap_err();
    assert_eq!(err.row(), Some(0));
}

#[test]
fn large_string_and_binary_comparisons() {
    let large = LargeStringArray::from(vec![Some("a"), None, Some("é")]);
    let got = compare_bytes::<_, Greater>(
        BytesOperand::Array(&large),
        BytesOperand::Scalar(Some(b"b")),
    )
    .unwrap();
    assert_eq!(got, BooleanArray::from(vec![Some(false), None, Some(true)]));
    let bin = BinaryArray::from(vec![Some(&b"\x01"[..]), Some(&[0xFF][..]), None]);
    let got = compare_bytes::<_, Less>(
        BytesOperand::Array(&bin),
        BytesOperand::Scalar(Some(&[0x80])),
    )
    .unwrap();
    assert_eq!(got, BooleanArray::from(vec![Some(true), Some(false), None]));
    let other = BinaryArray::from(vec![Some(&b"\x01"[..]), Some(&[0x00][..]), Some(&b""[..])]);
    let got =
        compare_bytes::<_, Equal>(BytesOperand::Array(&bin), BytesOperand::Array(&other)).unwrap();
    assert_eq!(got, BooleanArray::from(vec![Some(true), Some(false), None]));
}

#[test]
fn per_row_patterns_keep_the_matcher() {
    let input = StringArray::from(vec![Some("abc"), Some("abc"), None]);
    let pats = StringArray::from(vec![Some("a%"), Some("%x"), Some("a")]);
    let got = like::<i32>(StrOperand::Array(&input), StrOperand::Array(&pats), '\\').unwrap();
    assert_eq!(got, BooleanArray::from(vec![Some(true), Some(false), None]));
}
