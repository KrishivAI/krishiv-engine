// SPDX-License-Identifier: Apache-2.0

use arrow_array::Array as _;
use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use krishiv_common::error::{ErrorClass, KrishivError};

use super::java::{MAX_PATTERN_LEN, MAX_REPETITION, translate};
use super::{extract_all_pieces, split_regex};
use crate::scalar::tests::{ints, run, strs};

/// The strings of a result column.
fn text(a: &ArrayRef) -> Vec<Option<String>> {
    a.as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_owned))
        .collect()
}

/// The booleans of a result column.
fn bools(a: &ArrayRef) -> Vec<Option<bool>> {
    a.as_boolean().iter().collect()
}

/// The ints of a result column.
fn ints_of(a: &ArrayRef) -> Vec<Option<i32>> {
    a.as_primitive::<arrow_array::types::Int32Type>()
        .iter()
        .collect()
}

/// `f(s, re)` for one row.
fn try_bool(f: &str, s: &str, re: &str) -> Result<Option<bool>, KrishivError> {
    let cols = [Some(s), Some(re)].map(|v| strs(&[v]));
    let g = crate::scalar::tests::bind(f, &[&cols[0], &cols[1]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| bools(&a).remove(0))
}

/// `f(s, re)` for one row.
fn try_int(f: &str, s: &str, re: &str) -> Result<Option<i32>, KrishivError> {
    let cols = [Some(s), Some(re)].map(|v| strs(&[v]));
    let g = crate::scalar::tests::bind(f, &[&cols[0], &cols[1]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| ints_of(&a).remove(0))
}

/// `f(s, re)` for one row.
fn try_text(f: &str, s: &str, re: &str) -> Result<Option<String>, KrishivError> {
    let cols = [Some(s), Some(re)].map(|v| strs(&[v]));
    let g = crate::scalar::tests::bind(f, &[&cols[0], &cols[1]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| text(&a).remove(0))
}

/// `f(s, re, idx)` for one row.
fn try_idx(f: &str, s: &str, re: &str, idx: Option<i32>) -> Result<Option<String>, KrishivError> {
    let cols = [strs(&[Some(s)]), strs(&[Some(re)]), ints(&[idx])];
    let g = crate::scalar::tests::bind(f, &[&cols[0], &cols[1], &cols[2]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| text(&a).remove(0))
}

/// `f(s, re, idx)` for one row.
fn try_idx_int(f: &str, s: &str, re: &str, idx: Option<i32>) -> Result<Option<i32>, KrishivError> {
    let cols = [strs(&[Some(s)]), strs(&[Some(re)]), ints(&[idx])];
    let g = crate::scalar::tests::bind(f, &[&cols[0], &cols[1], &cols[2]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| ints_of(&a).remove(0))
}

/// `regexp_extract(s, re)` for one row (the 2-argument form).
fn try_extract2(s: &str, re: &str) -> Result<Option<String>, KrishivError> {
    let cols = [strs(&[Some(s)]), strs(&[Some(re)])];
    let g = crate::scalar::tests::bind("regexp_extract", &[&cols[0], &cols[1]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| text(&a).remove(0))
}

/// `regexp_instr(s, re)` for one row (the 2-argument form).
fn try_instr2(s: &str, re: &str) -> Result<Option<i32>, KrishivError> {
    let cols = [strs(&[Some(s)]), strs(&[Some(re)])];
    let g = crate::scalar::tests::bind("regexp_instr", &[&cols[0], &cols[1]]);
    let args = cols.map(crate::scalar::Arg::Array);
    g.invoke(&args, 1).map(|a| ints_of(&a).remove(0))
}

/// `regexp_replace(s, re, rep)` for one row.
fn try_one(s: &str, re: &str, rep: &str) -> Result<Option<String>, KrishivError> {
    let cols = [Some(s), Some(re), Some(rep)].map(|v| strs(&[v]));
    let f = crate::scalar::tests::bind("regexp_replace", &[&cols[0], &cols[1], &cols[2]]);
    let args = cols.map(crate::scalar::Arg::Array);
    f.invoke(&args, 1).map(|a| text(&a).remove(0))
}

fn one(s: &str, re: &str, rep: &str) -> Option<String> {
    try_one(s, re, rep).unwrap()
}

fn some(s: &str) -> Option<String> {
    Some(s.to_owned())
}

fn fails(s: &str, re: &str, rep: &str) -> KrishivError {
    try_one(s, re, rep).unwrap_err()
}

const HEALTHY: &str = "healthy, wealthy, and wise";

/// Cases of `regexp-functions.sql` (Spark 4.2.0 goldens).
#[test]
fn spark_golden_positions() {
    let at = |position: Option<i32>, re: &str| {
        let s = strs(&[Some(HEALTHY)]);
        let p = ints(&[position]);
        text(&run(
            "regexp_replace",
            &[&s, &strs(&[Some(re)]), &strs(&[Some("something")]), &p],
        ))
        .remove(0)
    };
    assert_eq!(
        at(Some(1), r"\w+thy"),
        some("something, something, and wise")
    );
    assert_eq!(
        at(Some(2), r"\w+thy"),
        some("hsomething, something, and wise")
    );
    assert_eq!(at(Some(8), r"\w+thy"), some("healthy, something, and wise"));
    assert_eq!(
        at(Some(26), r"\w"),
        some("healthy, wealthy, and wissomething")
    );
    assert_eq!(at(Some(27), r"\w"), some(HEALTHY));
    assert_eq!(at(Some(30), r"\w"), some(HEALTHY));
    assert_eq!(at(None, r"\w"), None);
}

/// `DATATYPE_MISMATCH.VALUE_OUT_OF_RANGE` for a position below 1.
#[test]
fn position_below_one_is_out_of_range() {
    for position in [0, -2] {
        let s = strs(&[Some(HEALTHY)]);
        let f =
            crate::scalar::tests::bind("regexp_replace", &[&s, &s, &s, &ints(&[Some(position)])]);
        let args = [&s, &s, &s, &ints(&[Some(position)])]
            .map(|a| crate::scalar::Arg::Array(std::sync::Arc::clone(a)));
        let e = f.invoke(&args, 1).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::DatatypeMismatchValueOutOfRange);
    }
}

/// The function-matrix cases (Spark 4.2.0): NULL arguments, the empty pattern.
#[test]
fn spark_golden_nulls_and_empty_pattern() {
    let s = strs(&[Some("100-200"), None, Some("100-200"), Some("100-200")]);
    let re = strs(&[Some(r"(\d+)"), Some(r"(\d+)"), None, Some(r"(\d+)")]);
    let rep = strs(&[Some("num"), Some("num"), Some("num"), None]);
    assert_eq!(
        text(&run("regexp_replace", &[&s, &re, &rep])),
        [some("num-num"), None, None, None]
    );
    assert_eq!(
        one("100-200", "", "num"),
        some("num1num0num0num-num2num0num0num")
    );
}

/// `$n`, `${name}` and backslash escapes of `Matcher.appendReplacement`.
#[test]
fn replacement_syntax() {
    assert_eq!(
        one("2024-02-29", r"(\d+)-(\d+)-(\d+)", "$3/$2/$1"),
        some("29/02/2024")
    );
    assert_eq!(one("ab", "(?<x>a)", "[${x}]"), some("[a]b"));
    assert_eq!(one("ab", "a", r"\$1"), some("$1b"));
    assert_eq!(one("ab", "a", r"\\"), some("\\b"));
    // The second digit joins the reference only when that group exists.
    assert_eq!(one("a", "(a)", "$10"), some("a0"));
    // A group that did not take part adds nothing.
    assert_eq!(one("b", "(a)?b", "[$1]"), some("[]"));
    // No match: the replacement is never read, so it cannot raise.
    assert_eq!(one("b", "x", "$9"), some("b"));
}

#[test]
fn replacement_errors_are_raised_on_a_match() {
    for rep in ["$", "$x", r"\", "$3", "${y}", "${x"] {
        let e = fails("ab", "(?<x>a)", rep);
        assert_eq!(e.error_class(), ErrorClass::InvalidRegexpReplace, "{rep}");
    }
    // No match, no error.
    assert_eq!(one("bb", "(?<x>a)", "$"), some("bb"));
}

/// `Matcher.find`: an empty match may follow a non-empty one.
#[test]
fn empty_matches() {
    assert_eq!(one("baac", "a*", "X"), some("XbXXcX"));
    assert_eq!(one("", "", "X"), some("X"));
    assert_eq!(one("é😀", "", "-"), some("-é-😀-"));
}

#[test]
fn invalid_pattern_is_invalid_parameter_value() {
    for re in ["(", "[", "a{", "a**", "a+*", "a*??", "\\", "\\G"] {
        let e = try_one("a", re, "x").expect_err(re);
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidParameterValuePattern,
            "{re}"
        );
        assert_eq!(e.sql_state(), Some("22023"), "{re}");
        assert!(
            e.message().contains("`regexp_replace`") && e.message().contains("`regexp`"),
            "{}",
            e.message()
        );
    }
}

/// Java and Rust differ on these.
#[test]
fn java_semantics_where_rust_differs() {
    // `\d`, `\w`, `\s` are ASCII.
    assert_eq!(one("٣a1", r"\d", "N"), some("٣aN"));
    assert_eq!(one("é_a", r"\w", "W"), some("éWW"));
    // `.` stops at every line terminator.
    assert_eq!(one("a\rb\u{2028}c", ".", "x"), some("x\rx\u{2028}x"));
    // `$` matches before a final line terminator, `\z` does not.
    assert_eq!(one("ab\n", "b$", "X"), some("aX\n"));
    assert_eq!(one("ab\r\n", "b$", "X"), some("aX\r\n"));
    assert_eq!(one("ab\n", r"b\z", "X"), some("ab\n"));
    assert_eq!(one("ab", "b$", "X"), some("aX"));
    // `\<` is a literal.
    assert_eq!(one("a<b>", r"\<", "-"), some("a-b>"));
    // `\Q..\E` quotes.
    assert_eq!(one("a.b", r"\Q.\E", "-"), some("a-b"));
    // Back-references and look-around.
    assert_eq!(one("aabc", r"(a)\1", "X"), some("Xbc"));
    assert_eq!(one("ab ac", "a(?=c)", "X"), some("ab Xc"));
    // Inline flags and POSIX classes.
    assert_eq!(one("aA", "(?i)a", "x"), some("xx"));
    assert_eq!(one("a\nb", "(?s)a.b", "x"), some("x"));
    assert_eq!(one("a\nb", "(?m)^b", "x"), some("a\nx"));
    assert_eq!(one("ab1", r"\p{Alpha}", "x"), some("xx1"));
    assert_eq!(one("a1", r"\P{Alpha}", "x"), some("ax"));
    assert_eq!(one("aB", r"[a-z&&[^b]]", "x"), some("xB"));
}

/// The ClickBench referrer-domain expression.
#[test]
fn clickbench_domain() {
    let re = r"^https?://(?:www\.)?([^/]+)/.*$";
    assert_eq!(
        one("https://www.example.com/a/b?c=1", re, "$1"),
        some("example.com")
    );
    assert_eq!(one("not a url", re, "$1"), some("not a url"));
}

#[test]
fn translations_pin_the_syntax() {
    assert_eq!(translate(r"\d+", false).unwrap().text, "[0-9]+");
    assert_eq!(
        translate("a.b", false).unwrap().text,
        r"a[^\n\r\x{85}\x{2028}\x{2029}]b"
    );
    assert_eq!(translate("(?s)a.b", false).unwrap().text, r"a[\s\S]b");
    assert_eq!(translate("(?d)a.b", false).unwrap().text, r"a[^\n]b");
    let t = translate("a$", false).unwrap();
    assert!(t.has_dollar);
    assert_eq!(t.text, r"a\z");
    assert_eq!(
        translate("a$", true).unwrap().text,
        r"a(?=(?:\r\n|[\n\r\x{85}\x{2028}\x{2029}])?\z)"
    );
    assert_eq!(translate("(?d)a$", true).unwrap().text, r"a(?=\n?\z)");
    assert!(translate("a*++", false).is_err());
    assert!(translate("a**", false).is_err());
    assert!(translate("a*??", false).is_err());
    // `(?x)`: whitespace and comments never reach the engine.
    assert_eq!(translate("(?x)a b # note\n c", false).unwrap().text, "abc");
    assert_eq!(translate("(?x)a\\ b", false).unwrap().text, r"a\x{20}b");
    assert_eq!(translate("(?x)[a b]", false).unwrap().text, "[ab]");
    assert_eq!(translate("(?x)a(?-x) b", false).unwrap().text, r"a\x{20}b");
    // `(?i)` spells out ASCII folding; `(?iu)` leaves Unicode folding to the engine.
    assert_eq!(translate("(?i)ab1", false).unwrap().text, "[aA][bB]1");
    assert_eq!(translate("(?i)[a-c_]", false).unwrap().text, "[a-cA-C_]");
    assert_eq!(translate("(?iu)ab", false).unwrap().text, "(?i)ab");
    assert_eq!(translate("(?i:a)b", false).unwrap().text, "(?:[aA])b");
    // Back-references: digits only while they name an open group; a
    // reference to a group that is not open never matches.
    assert_eq!(translate(r"(a)\1", false).unwrap().text, r"(a)(?:\1)");
    assert_eq!(translate(r"(a)\12", false).unwrap().text, r"(a)(?:\1)2");
    assert_eq!(translate(r"a\1", false).unwrap().text, "a(?!)");
    assert_eq!(
        translate(r"(?i)(a)\1", false).unwrap().text,
        r"([aA])(?i:\1)"
    );
    assert_eq!(
        translate(r"(?<n>a)\k<n>", false).unwrap().text,
        r"(?<n>a)(?:\k<n>)"
    );
    assert_eq!(
        translate(r"(?i)(?<n>a)\k<n>", false).unwrap().text,
        r"(?<n>[aA])(?i:\k<n>)"
    );
    assert!(translate("(?<n_>a)", false).is_err());
    assert!(translate("(?<1>a)", false).is_err());
}

/// Spark 4.2.0: `(?x)` ignores whitespace and `#` comments (inside classes
/// too); a comment ends at a line terminator, and `\u0085`, `\u2028` and
/// `\u2029` are then pattern characters.
#[test]
fn oracle_comments_flag() {
    let cases = [
        ("ab", "(?x)a b", "X"),
        ("a#", "(?x)a # note", "X#"),
        ("ab", "(?x)a # (\nb", "X"),
        ("a b", r"(?x)a\ b", "X"),
        ("a b", "(?x)[a b]+", "X X"),
        ("ab", "(?x)a#b\nb", "X"),
        ("a b", r"(?x)\Qa b\E", "X"),
        ("ab", "(?x: a b )", "X"),
        ("ab", "a(?x) b", "X"),
        ("a\tb", r"(?x)a\tb", "X"),
        ("aa", "(?x)a +", "X"),
        ("aa", "(?x)a+ ?", "XX"),
        ("ab", "(?x)(?-x) a", "ab"),
        ("a b", "(?x)(?-x:a b)", "X"),
        ("a b", "(?x)a (?-x) b", "X"),
        ("a#b", "(?x)[a #c\nb]+", "X#X"),
        ("a\u{85}", "(?x)a#c\u{85}", "X"),
        ("a\u{2028}", "(?x)a#c\u{2028}", "X"),
        ("a", "(?x)a#c\u{2028}", "a"),
    ];
    for (s, re, want) in cases {
        assert_eq!(one(s, re, "X"), some(want), "{s:?} {re:?}");
    }
    // Whitespace in a class is dropped, which leaves it empty here.
    for re in ["(?x)a[ ]b", "(?x)[a#]"] {
        assert_eq!(
            try_one("a#", re, "X").unwrap_err().error_class(),
            ErrorClass::InvalidParameterValuePattern,
            "{re}"
        );
    }
}

/// Spark 4.2.0: `(?i)` folds ASCII only unless `(?u)` is set.
#[test]
fn oracle_ascii_case_folding() {
    let cases = [
        ("É é", "(?i)é", "É X"),
        ("K\u{212a} k", "(?i)k", "X\u{212a} X"),
        ("s \u{17f}", "(?i)s", "X \u{17f}"),
        ("K\u{212a} k", "(?iu)k", "XX X"),
        ("É é", "(?iu)É", "X X"),
        ("ÉéKk", "(?i)[a-zé]", "ÉXXX"),
        ("K\u{212a}", "(?i)[a-z]", "X\u{212a}"),
        ("aA", "(?i)[A-Z]+", "X"),
        ("aA", "(?i)[^a]", "aA"),
        ("É", "(?i)[^é]", "X"),
        ("xA", "(?i:a)", "xX"),
        ("xA", r"(?i)\x41", "xX"),
        ("aB", "(?i)a(?-i)b", "aB"),
        ("AB", "(?i:a)b", "AB"),
        ("ab", "(?i)A(?-i:B)", "ab"),
        ("é", "(?i)É", "é"),
        ("É", "(?iu)(?-u)é", "É"),
        ("k", "(?i)[\u{212a}]", "k"),
        ("a-", "(?i)[a-]+", "X"),
        ("m", "(?i)[^b-y]", "m"),
        ("M", "(?i)[b-y]", "X"),
        ("AZ", "(?i)[^b-y]+", "X"),
        ("aA", r"(?i)\Qa\E+", "X"),
        ("É é", r"(?i)\Qé\E", "É X"),
        ("Ǆǅǆ", "(?i)ǆ", "ǄǅX"),
        ("Ǆǅǆ", "(?iu)ǆ", "XXX"),
    ];
    for (s, re, want) in cases {
        assert_eq!(one(s, re, "X"), some(want), "{s:?} {re:?}");
    }
}

/// Spark 4.2.0: back-reference digits and references to missing groups.
#[test]
fn oracle_back_references() {
    let cases = [
        ("a2", r"(a)\12", "a2"),
        ("a", r"a\1", "a"),
        ("a1", r"a\1", "a1"),
        ("aa", r"(a)\1", "X"),
        ("aa", r"\1(a)", "aa"),
        ("abcdefghijj", r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)\10", "X"),
        ("a0", r"(a)\10", "a0"),
        ("aa", r"(a)\2", "aa"),
        ("a", r"(a)\01", "a"),
        ("aA", r"(?i)(a)\1", "X"),
        ("aA", r"(?i)(?<n>a)\k<n>", "X"),
        ("aA", r"(a)(?i)\1", "X"),
        ("aA", r"(?i)(a)(?-i)\1", "aA"),
    ];
    for (s, re, want) in cases {
        assert_eq!(one(s, re, "X"), some(want), "{s:?} {re:?}");
    }
}

/// `regexp_replace(s, re, rep, pos)` for one row.
fn at(s: &str, re: &str, rep: &str, pos: i32) -> Option<String> {
    let cols = [Some(s), Some(re), Some(rep)].map(|v| strs(&[v]));
    let p = ints(&[Some(pos)]);
    text(&run("regexp_replace", &[&cols[0], &cols[1], &cols[2], &p])).remove(0)
}

/// Spark 4.2.0 (local session): `$n`, `\$` and backslash escapes in the
/// replacement, and the error a bad replacement raises.
#[test]
fn oracle_replacement() {
    let cases = [
        ("abc123", r"([a-z]+)(\d+)", "$2-$1", "123-abc"),
        ("abc", "b", r"\$", "a$c"),
        ("abc", "(b)", "$0", "abc"),
        ("abc", "(b)", "$1$1", "abbc"),
        // Through the DataFrame API: Spark's SQL parser substitutes a
        // literal `${n}` (variable substitution) before the function runs.
        ("abc", "(?<n>b)", "${n}", "abc"),
        ("abc", "(?<n>b)", "[${n}]", "a[b]c"),
        ("abc", "(b)", "$10", "ab0c"),
        ("abc", "(b)", r"\\", r"a\c"),
        ("abc", "(c)", r"\n", "abn"),
    ];
    for (s, re, rep, want) in cases {
        assert_eq!(one(s, re, rep), some(want), "{re} {rep}");
    }
    for (re, rep) in [
        ("b", "$"),
        ("b", "$2"),
        ("(b)", "$2"),
        ("(b)", r"\"),
        ("(b)", "$a"),
        ("(b)", "${1}"),
        ("(?<n>b)", "${m}"),
        ("(?<n>b)", "${n"),
        ("(?<n>b)", "${}"),
    ] {
        let e = fails("abc", re, rep);
        assert_eq!(e.error_class(), ErrorClass::InvalidRegexpReplace, "{rep}");
        assert_eq!(e.sql_state(), Some("22023"), "{rep}");
        assert_eq!(
            e.message(),
            format!(
                "Could not perform regexp_replace for source = \"abc\", \
                 pattern = \"{re}\", replacement = \"{rep}\" and position = 1."
            ),
            "{rep}"
        );
    }
}

/// Spark 4.2.0: backslash escapes of the pattern.
#[test]
fn oracle_pattern_escapes() {
    let cases = [
        ("a.b", r"\.", "a-b"),
        (r"a\b", r"\\", "a/b"),
        ("a1b", r"\d", "a#b"),
        ("a b", r"\s", "a#b"),
        ("ab", r"\Qa\E", "#b"),
        ("abc", r"\x62", "a#c"),
        ("abc", r"b", "a#c"),
        ("abc", r"\p{Lower}", "###"),
        ("abc", r"\p{Alpha}+", "#"),
        ("abc", r"\h", "abc"),
        ("a\tb", r"\t", "a#b"),
    ];
    for (s, re, want) in cases {
        let want = want.replace(['-', '/'], "#");
        let got = one(s, re, "#");
        assert_eq!(got, some(&want), "{re}");
    }
}

/// Spark 4.2.0: NULL in any argument gives NULL.
#[test]
fn oracle_null_arguments() {
    let s = strs(&[Some("a"), None, Some("a"), Some("a"), Some("a")]);
    let re = strs(&[Some("a"), Some("a"), None, Some("a"), Some("a")]);
    let rep = strs(&[Some("b"), Some("b"), Some("b"), None, Some("b")]);
    let pos = ints(&[Some(1), Some(1), Some(1), Some(1), None]);
    assert_eq!(
        text(&run("regexp_replace", &[&s, &re, &rep, &pos])),
        [some("b"), None, None, None, None]
    );
}

/// Spark 4.2.0: the error class and message of an invalid pattern.
#[test]
fn oracle_invalid_patterns() {
    for re in ["(", "[a", "*", "(?<n", "\\", "a{2,1}", "(?P<n>a)"] {
        let e = try_one("abc", re, "x").expect_err(re);
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidParameterValuePattern,
            "{re}"
        );
        assert_eq!(
            e.message(),
            format!("The value of parameter(s) `regexp` in `regexp_replace` is invalid: '{re}'."),
            "{re}"
        );
    }
}

/// Spark 4.2.0: the position counts UTF-16 units; the match region starts
/// there, anchors and look-behind do not see what precedes it.
#[test]
fn oracle_position() {
    for (pos, want) in [(1, "bbb"), (2, "abb"), (3, "aab"), (4, "aaa"), (5, "aaa")] {
        assert_eq!(at("aaa", "a", "b", pos), some(want), "{pos}");
    }
    assert_eq!(at("abab", "^a", "X", 3), some("abXb"));
    assert_eq!(at("abab", r"\ba", "X", 3), some("abXb"));
    assert_eq!(at("abab", "(?<=a)b", "X", 2), some("abaX"));
    assert_eq!(at("a😀b", "b", "X", 3), some("a😀X"));
    assert_eq!(at("a😀b", ".", "X", 2), some("aXX"));
    assert_eq!(at("abc", "", "-", 3), some("ab-c-"));
    assert_eq!(at("abc", "", "-", 4), some("abc"));
}

/// Spark 4.2.0: empty matches.
#[test]
fn oracle_empty_matches() {
    let cases = [
        ("abc", "", "-a-b-c-"),
        ("", "", "-"),
        ("abc", "x*", "-a-b-c-"),
        ("abc", "b*", "-a--c-"),
        ("abc", "$", "abc-"),
        ("abc", "^", "-abc"),
    ];
    for (s, re, want) in cases {
        assert_eq!(one(s, re, "-"), some(want), "{re}");
    }
}

/// Spark 4.2.0: `.` excludes `\n`, `\r`, U+0085, U+2028 and U+2029 (and only
/// `\n` under `(?d)`); `(?s)` includes them all.
#[test]
fn oracle_dot_and_line_terminators() {
    for t in ["\n", "\r", "\u{85}", "\u{2028}", "\u{2029}", "\r\n"] {
        let s = format!("a{t}b");
        assert_eq!(one(&s, ".", "X"), some(&format!("X{t}X")), "{t:?}");
        assert_eq!(one(&s, "a.b", "X"), some(&s), "{t:?}");
        assert_eq!(
            one(&s, "(?s).", "X"),
            some(&"X".repeat(2 + t.chars().count()))
        );
    }
    assert_eq!(one("a\nb", "(?d).", "X"), some("X\nX"));
    assert_eq!(one("a\u{85}b", "(?d).", "X"), some("XXX"));
    assert_eq!(one("a\nb", "$", "X"), some("a\nbX"));
    assert_eq!(one("a\nb\n", "$", "X"), some("a\nbX\nX"));
    assert_eq!(one("a\nb", "(?m)^", "X"), some("Xa\nXb"));
    assert_eq!(one("a\nb", "(?m)$", "X"), some("aX\nbX"));
    assert_eq!(one("a\nb", r"\z", "X"), some("a\nbX"));
    assert_eq!(one("a\nb\n", r"\Z", "X"), some("a\nbX\nX"));
    // The ClickBench domain pattern: a line terminator stops `.*`.
    let re = r"^https?://(?:www\.)?([^/]+)/.*$";
    assert_eq!(
        one("https://www.example.com/pa\nth", re, "$1"),
        some("https://www.example.com/pa\nth")
    );
    assert_eq!(
        one("http://example.com", re, "$1"),
        some("http://example.com")
    );
}

/// `(?d)` (UNIX_LINES), checked against Spark 4.2.0: only `\n` is a line
/// terminator for `.`, `^`, `$` and `\Z`.
#[test]
fn unix_lines_flag() {
    let cases = [
        ("a\r\nb", "(?d)a.", "X\nb"),
        ("a\rb", "(?d).", "XXX"),
        ("a\nb\n", "(?d)$", "a\nbX\nX"),
        ("a\rb\r", "(?d)$", "a\rb\rX"),
        ("a\nb", "(?dm)^", "Xa\nXb"),
        ("a\rb", "(?dm)^", "Xa\rb"),
        ("a\nb\n", "(?dm)$", "aX\nbX\nX"),
        ("a\rb", "(?dm)$", "a\rbX"),
        ("a\nb\n", r"(?d)\Z", "a\nbX\nX"),
        ("a\rb\r", r"(?d)\Z", "a\rb\rX"),
        ("a\u{85}b", "(?d)a.b", "X"),
        ("a\nb", "(?-d).", "X\nX"),
    ];
    for (s, re, want) in cases {
        assert_eq!(one(s, re, "X"), some(want), "{s:?} {re}");
    }
}

/// A back-reference sends the pattern to the backtracking engine, where
/// nested quantifiers are exponential (on the JVM this pattern takes about
/// 0.6 s at 24 `a`s and doubles with each more). Spark has no limit; Krishiv
/// stops with a named error instead of hanging the query.
#[test]
fn catastrophic_pattern_hits_the_backtrack_limit() {
    let input = format!("{}!", "a".repeat(40));
    let e = fails(&input, r"^(a*)*\1c", "x");
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivRegexpBacktrackLimitExceeded
    );
    assert_eq!(e.sql_state(), Some("54000"));
    assert!(
        e.message().contains(&super::BACKTRACK_LIMIT.to_string()),
        "{}",
        e.message()
    );
    // The same shape on a short input stays under the limit.
    assert_eq!(one("aaa!", r"^(a*)*\1c", "x"), some("aaa!"));
}

/// `|` is alternation, also inside groups, under inline flags and before an
/// anchor (it was translated as a literal bar).
#[test]
fn alternation_alternates() {
    let cases = [
        ("abc|", "a|b", "XXc|"),
        ("abc|", "(a|b)c", "aX|"),
        ("abc|", "x(?:a|b)|c", "abX|"),
        ("aBc|", "(?i)a|b", "XXc|"),
        ("abc|", r"c\||b", "aXX"),
        ("xa\nb", "(a|b)$", "xa\nX"),
        ("abc|", "[|]", "abcX"),
        ("a|b", r"a\|b", "X"),
    ];
    for (s, re, want) in cases {
        assert_eq!(one(s, re, "X"), some(want), "{s:?} {re}");
    }
    assert_eq!(translate("a|b", false).unwrap().text, "a|b");
}

/// The look-around the translation adds for `$` and the multiline anchors
/// is only a trailing assertion, so a pattern with none of its own stays
/// linear on an input ending in a line terminator: no backtrack limit on
/// these near misses, in the plain and the terminated form.
#[test]
fn translated_anchors_stay_linear_on_a_near_miss() {
    let a = "a".repeat(60);
    for re in ["(a+)+$", "(a*)*$", "(a|aa)+$", "(?m)(a+)+$", "(?m)^(a+)+$"] {
        for tail in ["b\n", "b"] {
            let input = format!("{a}{tail}");
            assert!(try_one(&input, re, "x").is_ok(), "{re} {tail:?}");
        }
    }
    let near_miss = format!("{a}b\n");
    assert_eq!(one(&near_miss, "(a+)+$", "x"), some(&near_miss));
}

/// `rlike` / `regexp` / `regexp_like` over the `regexp-functions.sql`
/// goldens (Spark 4.2.0): Java `find` semantics.
#[test]
fn golden_like() {
    for f in ["regexp", "regexp_like"] {
        assert_eq!(try_bool(f, "1a 2b 14m", r"\d+b"), Ok(Some(true)));
        assert_eq!(try_bool(f, "1a 2b 14m", "[a-z]+b"), Ok(Some(false)));
    }
    // NULL in any argument gives NULL.
    let (s, re) = (strs(&[Some("a"), None]), strs(&[Some("a"), Some("a")]));
    assert_eq!(bools(&run("regexp_like", &[&s, &re])), [Some(true), None]);
    let (s, re) = (strs(&[Some("a"), Some("a")]), strs(&[None, Some("x")]));
    assert_eq!(bools(&run("regexp", &[&s, &re])), [None, Some(false)]);
}

/// `regexp_extract` over the `regexp-functions.sql` goldens.
#[test]
fn golden_extract() {
    let e = |s: &str, re: &str, idx: Option<i32>| try_idx("regexp_extract", s, re, idx);
    // The 2-argument form reads group 1.
    assert!(
        try_extract2("1a 2b 14m", r"\d+")
            .unwrap_err()
            .to_string()
            .contains(
                "The value of parameter(s) `idx` in `regexp_extract` is invalid: \
         Expects group index between 0 and 0, but got 1."
            )
    );
    assert_eq!(e("1a 2b 14m", r"\d+", Some(0)), Ok(some("1")));
    // Out of range, negative, and missing groups.
    for idx in [Some(1), Some(2), Some(-1)] {
        let err = e("1a 2b 14m", r"\d+", idx).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::InvalidParameterValueRegexGroupIndex
        );
        assert_eq!(err.sql_state(), Some("22023"));
    }
    assert_eq!(e("1a 2b 14m", r"(\d+)?", Some(1)), Ok(some("1")));
    assert_eq!(e("a b m", r"(\d+)?", Some(1)), Ok(some("")));
    assert_eq!(try_extract2("1a 2b 14m", r"(\d+)([a-z]+)"), Ok(some("1")));
    assert_eq!(e("1a 2b 14m", r"(\d+)([a-z]+)", Some(0)), Ok(some("1a")));
    assert_eq!(e("1a 2b 14m", r"(\d+)([a-z]+)", Some(1)), Ok(some("1")));
    assert_eq!(e("1a 2b 14m", r"(\d+)([a-z]+)", Some(2)), Ok(some("a")));
    for idx in [Some(3), Some(-1)] {
        let err = e("1a 2b 14m", r"(\d+)([a-z]+)", idx).unwrap_err();
        assert_eq!(
            err.message(),
            format!(
                "The value of parameter(s) `idx` in `regexp_extract` is invalid: \
                 Expects group index between 0 and 2, but got {}.",
                idx.unwrap()
            )
        );
    }
    assert_eq!(e("1a 2b 14m", r"(\d+)?([a-z]+)", Some(1)), Ok(some("1")));
    assert_eq!(e("a b m", r"(\d+)?([a-z]+)", Some(1)), Ok(some("")));
    // An invalid pattern names the function and the parameter.
    let err = try_extract2("1a 2b 14m", "(?l)").unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValuePattern);
    assert_eq!(
        err.message(),
        "The value of parameter(s) `regexp` in `regexp_extract` is invalid: '(?l)'."
    );
    // A non-matching input never raises, even with a bad index.
    assert_eq!(e("abc", r"\d+", Some(5)), Ok(some("")));
    assert_eq!(try_extract2("abc", "x"), Ok(some("")));
    // NULL in any argument gives NULL (and never compiles the pattern).
    let (s, re) = (strs(&[Some("a"), None]), strs(&[Some("1"), Some("(")]));
    assert_eq!(text(&run("regexp_extract", &[&s, &re])), [some(""), None]);
}

/// `regexp_extract_all` over the `regexp-functions.sql` goldens.
#[test]
fn golden_extract_all() {
    let all = |s: &str, re: &str, idx: Option<i32>| {
        extract_all_pieces(re, s, idx, "`regexp_extract_all`")
    };
    let owned = |v: &[&str]| v.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(
        all("1a 2b 14m", r"\d+", Some(0)),
        Ok(owned(&["1", "2", "14"]))
    );
    // The 2-argument form reads group 1.
    assert!(all("1a 2b 14m", r"\d+", None).is_err());
    for idx in [Some(1), Some(2), Some(-1)] {
        let err = all("1a 2b 14m", r"\d+", idx).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::InvalidParameterValueRegexGroupIndex
        );
    }
    assert_eq!(
        all("1a 2b 14m", r"(\d+)?", Some(1)),
        Ok(owned(&["1", "", "", "2", "", "", "14", "", ""]))
    );
    assert_eq!(
        all("a 2b 14m", r"(\d+)?", Some(1)),
        Ok(owned(&["", "", "2", "", "", "14", "", ""]))
    );
    assert_eq!(
        all("1a 2b 14m", r"(\d+)([a-z]+)", None),
        Ok(owned(&["1", "2", "14"]))
    );
    assert_eq!(
        all("1a 2b 14m", r"(\d+)([a-z]+)", Some(0)),
        Ok(owned(&["1a", "2b", "14m"]))
    );
    assert_eq!(
        all("1a 2b 14m", r"(\d+)([a-z]+)", Some(1)),
        Ok(owned(&["1", "2", "14"]))
    );
    assert_eq!(
        all("1a 2b 14m", r"(\d+)([a-z]+)", Some(2)),
        Ok(owned(&["a", "b", "m"]))
    );
    assert!(all("1a 2b 14m", r"(\d+)([a-z]+)", Some(3)).is_err());
    assert_eq!(
        all("1a 2b 14m", r"(\d+)?([a-z]+)", Some(1)),
        Ok(owned(&["1", "2", "14"]))
    );
    assert_eq!(
        all("a 2b 14m", r"(\d+)?([a-z]+)", Some(1)),
        Ok(owned(&["", "2", "14"]))
    );
    let err = all("abc", "], [", Some(1)).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValuePattern);
    assert_eq!(
        err.message(),
        "The value of parameter(s) `regexp` in `regexp_extract_all` is invalid: '], ['."
    );
    // A non-matching input is empty, even with a bad index.
    assert_eq!(all("abc", r"\d+", Some(9)), Ok(Vec::new()));
    assert_eq!(all("abc", "x", None), Ok(Vec::new()));
}

/// `regexp_count` over the `regexp-functions.sql` goldens.
#[test]
fn golden_count() {
    assert_eq!(try_int("regexp_count", "1a 2b 14m", r"\d+"), Ok(Some(3)));
    assert_eq!(try_int("regexp_count", "1a 2b 14m", "mmm"), Ok(Some(0)));
    assert_eq!(try_int("regexp_count", "the fox", "FOX"), Ok(Some(0)));
    assert_eq!(try_int("regexp_count", "the fox", "(?i)FOX"), Ok(Some(1)));
    assert_eq!(
        try_int(
            "regexp_count",
            "passwd7 plain A1234 a1234",
            "(?=[^ ]*[a-z])(?=[^ ]*[0-9])[^ ]+"
        ),
        Ok(Some(2))
    );
    // Empty matches count too, advancing one character at a time.
    assert_eq!(try_int("regexp_count", "abc", ""), Ok(Some(4)));
    assert_eq!(try_int("regexp_count", "abc", "x*"), Ok(Some(4)));
    // NULL in any argument gives NULL.
    let (s, re) = (strs(&[None, Some("abc")]), strs(&[Some("a"), None]));
    assert_eq!(ints_of(&run("regexp_count", &[&s, &re])), [None, None]);
    assert_eq!(
        try_int("regexp_count", "abc", "(")
            .unwrap_err()
            .error_class(),
        ErrorClass::InvalidParameterValuePattern
    );
}

/// `regexp_substr` over the `regexp-functions.sql` goldens.
#[test]
fn golden_substr() {
    assert_eq!(
        try_text("regexp_substr", "1a 2b 14m", r"\d+"),
        Ok(some("1"))
    );
    // An empty match is NULL, as in Spark (even for the empty pattern).
    for re in ["", "x*", r"\d+ "] {
        assert_eq!(try_text("regexp_substr", "1a 2b 14m", re), Ok(None), "{re}");
    }
    assert_eq!(
        try_text("regexp_substr", "1a 2b 14m", r"\d+(a|b|m)"),
        Ok(some("1a"))
    );
    assert_eq!(
        try_text("regexp_substr", "1a 2b 14m", r"\d{2}(a|b|m)"),
        Ok(some("14m"))
    );
    let (s, re) = (
        strs(&[Some("Spark"), None, Some("Spark")]),
        strs(&[None, Some(".*"), Some(".*")]),
    );
    assert_eq!(
        text(&run("regexp_substr", &[&s, &re])),
        [None, None, Some("Spark".to_owned())]
    );
    assert_eq!(
        try_text("regexp_substr", "abc", "(")
            .unwrap_err()
            .error_class(),
        ErrorClass::InvalidParameterValuePattern
    );
}

/// `regexp_instr` over the `regexp-functions.sql` goldens.
#[test]
fn golden_instr() {
    let at = |s: &str, re: &str, idx: Option<i32>| try_idx_int("regexp_instr", s, re, idx);
    // The 2-argument form reads group 0.
    assert_eq!(try_instr2("abc", "b"), Ok(Some(2)));
    assert_eq!(try_instr2("abc", "x"), Ok(Some(0)));
    assert_eq!(try_instr2("ABC", "(?-i)b"), Ok(Some(0)));
    assert_eq!(try_instr2("1a 2b 14m", r"\d{2}(a|b|m)"), Ok(Some(7)));
    assert_eq!(at("1a 2b 14m", r"\d+(a|b|m)", Some(0)), Ok(Some(1)));
    // The position is the whole match's, whatever the validated group.
    assert_eq!(at("1a 2b 14m", r"\d+(a|b|m)", Some(1)), Ok(Some(1)));
    assert_eq!(at("foo123bar", r"(\d+)", Some(1)), Ok(Some(4)));
    assert_eq!(at("1a 2b 14m", r"x(\d+)", Some(1)), Ok(Some(0)));
    // Out-of-range and negative indices, checked once the pattern matched.
    let err = at("1a 2b 14m", r"(\d+)", Some(2)).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::InvalidParameterValueRegexGroupIndex
    );
    assert_eq!(
        err.message(),
        "The value of parameter(s) `idx` in `regexp_instr` is invalid: \
         Expects group index between 0 and 1, but got 2."
    );
    // A non-matching input is 0, even with a bad index.
    assert_eq!(at("abc", r"\d+", Some(7)), Ok(Some(0)));
    // NULL in any argument gives NULL.
    let (s, re) = (strs(&[Some("abc"), None]), strs(&[None, Some("b")]));
    assert_eq!(ints_of(&run("regexp_instr", &[&s, &re])), [None, None]);
    let err = at("abc", ") ?", Some(1)).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValuePattern);
    assert_eq!(
        err.message(),
        "The value of parameter(s) `regexp` in `regexp_instr` is invalid: ') ?'."
    );
}

/// `split` over the `string-functions.sql` goldens (Spark 4.2.0).
#[test]
fn golden_split() {
    /// The pieces of `split(s, re, limit)` in one row.
    fn pieces(s: Option<&str>, re: &str, limit: i32) -> Option<Vec<String>> {
        let input = arrow_array::StringArray::from(vec![s]);
        let out = split_regex(&input, re, &[Some(limit)]).unwrap();
        if out.is_null(0) {
            return None;
        }
        Some(
            out.value(0)
                .as_string::<i32>()
                .iter()
                .map(|v| v.unwrap().to_owned())
                .collect(),
        )
    }
    /// The pieces with a NULL limit: always NULL.
    fn pieces_null_limit(s: Option<&str>, re: &str) -> Option<Vec<String>> {
        let input = arrow_array::StringArray::from(vec![s]);
        let out = split_regex(&input, re, &[None]).unwrap();
        assert!(out.is_null(0));
        None
    }
    let below = ["aa", "cc", "ee", ""].map(str::to_owned);
    assert_eq!(
        pieces(Some("aa1cc2ee3"), "[1-9]+", -1),
        Some(below.to_vec())
    );
    assert_eq!(
        pieces(Some("aa1cc2ee3"), "[1-9]+", 2),
        Some(vec!["aa".to_owned(), "cc2ee3".to_owned()])
    );
    let hello = ["h", "e", "l", "l", "o"].map(str::to_owned);
    assert_eq!(pieces(Some("hello"), "", -1), Some(hello.to_vec()));
    assert_eq!(pieces(Some("hello"), "", 0), Some(hello.to_vec()));
    assert_eq!(pieces(Some("hello"), "", 1), Some(vec!["hello".to_owned()]));
    assert_eq!(
        pieces(Some("hello"), "", 3),
        Some(vec!["h".to_owned(), "e".to_owned(), "llo".to_owned()])
    );
    assert_eq!(pieces(Some("hello"), "", 5), Some(hello.to_vec()));
    assert_eq!(pieces(Some("hello"), "", 100), Some(hello.to_vec()));
    assert_eq!(pieces(Some(""), "", -1), Some(vec!["".to_owned()]));
    assert_eq!(pieces(Some(""), "", 0), Some(vec!["".to_owned()]));
    assert_eq!(pieces(Some(""), "", 1), Some(vec!["".to_owned()]));
    assert_eq!(pieces_null_limit(Some("abc"), "b"), None);
    assert_eq!(pieces(None, "b", -1), None);
    assert_eq!(
        pieces(Some("a,b"), ",", -1),
        Some(vec!["a".to_owned(), "b".to_owned()])
    );
    // `limit <= 0` keeps trailing empty pieces (Spark maps `0` to `-1`).
    assert_eq!(
        pieces(Some("a,b,"), ",", 0),
        Some(vec!["a".to_owned(), "b".to_owned(), "".to_owned()])
    );
    assert_eq!(
        pieces(Some("a,b,"), ",", -1),
        Some(vec!["a".to_owned(), "b".to_owned(), "".to_owned()])
    );
    // A leading empty match splits nothing.
    assert_eq!(
        pieces(Some("axb"), "x*", 0),
        Some(vec![
            "a".to_owned(),
            "".to_owned(),
            "b".to_owned(),
            "".to_owned()
        ])
    );
    // An invalid pattern names `split`.
    let input = arrow_array::StringArray::from(vec![Some("a,b")]);
    let err = split_regex(&input, "(", &[Some(-1)]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValuePattern);
    assert!(err.message().contains("`split`"), "{}", err.message());
}

/// Possessive quantifiers translate to atomic groups.
#[test]
fn possessive_quantifiers() {
    assert_eq!(translate("a*+", false).unwrap().text, "(?>a*)");
    assert_eq!(translate("a++", false).unwrap().text, "(?>a+)");
    assert_eq!(translate("a?+", false).unwrap().text, "(?>a?)");
    assert_eq!(translate("a{2,3}+", false).unwrap().text, "(?>a{2,3})");
    assert_eq!(translate("(ab)*+", false).unwrap().text, "(?>(ab)*)");
    assert_eq!(translate(r"\d++", false).unwrap().text, "(?>[0-9]+)");
    assert_eq!(translate("[ab]++", false).unwrap().text, "(?>[ab]+)");
    assert_eq!(translate("(?i)a*+", false).unwrap().text, "(?>[aA]*)");
    // A possessive match never gives the quantifier back.
    assert_eq!(one("aaa", "a*+", "X"), some("XX"));
    assert_eq!(try_bool("regexp_like", "aaa", "a*+a"), Ok(Some(false)));
    assert_eq!(try_bool("regexp_like", "aaa", "a*a"), Ok(Some(true)));
    // A quantifier right after a possessive wrap dangles, as in Java.
    for re in ["a*++", "a*+?", "a*+*", "a*+{2}", "*a", "a*|*"] {
        assert!(translate(re, false).is_err(), "{re}");
    }
    // Stacked quantifiers dangle, but `?` after one quantifier is lazy.
    for re in ["a**", "a+*", "a?*", "a*??", "a{2}{3}", "a*{2}"] {
        assert!(translate(re, false).is_err(), "{re}");
    }
    assert_eq!(translate("a*?", false).unwrap().text, "a*?");
    assert_eq!(translate("a??", false).unwrap().text, "a??");
    assert_eq!(translate("a{2}?", false).unwrap().text, "a{2}?");
    assert_eq!(try_bool("regexp_like", "aaa", "a*?a"), Ok(Some(true)));
}

/// Patterns that are too large or repeat too much are invalid.
#[test]
fn pattern_bounds() {
    const { assert!(MAX_PATTERN_LEN >= 1024 && MAX_REPETITION >= 1000) };
    let big = "a".repeat(MAX_PATTERN_LEN + 1);
    let err = try_text("regexp_like", "a", &big).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InvalidParameterValuePattern);
    assert_eq!(
        try_bool("regexp_like", "a", &"a".repeat(MAX_PATTERN_LEN)),
        Ok(Some(false))
    );
    // A pattern at the bound translates (matching it is linear, so the
    // binding is not exercised here).
    assert!(translate(&"a".repeat(MAX_PATTERN_LEN), false).is_ok());
    for re in [
        "a{0,1000001}",
        "a{1000001}",
        "a{1000001,}",
        "(a{0,1000001})",
    ] {
        let err = try_text("regexp_like", "aaa", re).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::InvalidParameterValuePattern,
            "{re}"
        );
    }
    assert_eq!(
        try_bool("regexp_like", &"a".repeat(2000), r"a{1000}"),
        Ok(Some(true))
    );
    // The bound counts Java characters, not bytes.
    let big = "é".repeat(MAX_PATTERN_LEN + 1);
    assert_eq!(
        try_text("regexp_like", "é", &big)
            .unwrap_err()
            .error_class(),
        ErrorClass::InvalidParameterValuePattern
    );
}

/// Rejected constructs name `INVALID_PARAMETER_VALUE.PATTERN`.
#[test]
fn rejected_constructs() {
    for re in [
        r"\G",
        "(?U)a",
        r"\p{InGreek}",
        "(?P<n>a)",
        "(?<n_>a)",
        "(?<1>a)",
        "a{2,1}",
        "(?l)",
        "(?d)(?U)",
    ] {
        let err = try_text("regexp_like", "a", re).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::InvalidParameterValuePattern,
            "{re}"
        );
    }
    // ... but every function names itself in the error.
    for f in [
        "regexp",
        "regexp_like",
        "regexp_extract",
        "regexp_count",
        "regexp_instr",
        "regexp_substr",
    ] {
        let cols = [strs(&[Some("a")]), strs(&[Some("(")])];
        let g = crate::scalar::tests::bind(f, &[&cols[0], &cols[1]]);
        let args = cols.map(crate::scalar::Arg::Array);
        let err = g.invoke(&args, 1).unwrap_err();
        assert_eq!(
            err.message(),
            format!("The value of parameter(s) `regexp` in `{f}` is invalid: '('."),
            "{f}"
        );
    }
}

/// A table of Java pattern to match results, taken from the goldens.
#[test]
fn translator_matches_goldens() {
    let yes = |s: &str, re: &str| try_bool("regexp_like", s, re) == Ok(Some(true));
    let no = |s: &str, re: &str| try_bool("regexp_like", s, re) == Ok(Some(false));
    // `regexp-functions.sql`: inline flags and POSIX classes.
    assert!(yes("the fox", "(?i)FOX"));
    assert!(no("the fox", "FOX"));
    assert!(yes("1a 2b 14m", r"\d+b"));
    assert!(no("1a 2b 14m", "[a-z]+b"));
    assert!(yes("ab1", r"\p{Alpha}"));
    assert!(no("1", r"\p{Alpha}"));
    assert!(yes("1", r"\P{Alpha}"));
    assert!(yes(
        "passwd7 plain A1234 a1234",
        "(?=[^ ]*[a-z])(?=[^ ]*[0-9])[^ ]+"
    ));
    assert!(no("ABC", "(?-i)b"));
    assert!(yes("ABC", "(?-i)B"));
    // `string-functions.sql` splits.
    assert!(yes("aa1cc2ee3", "[1-9]+"));
    // Line terminators: `.` stops at each one, `(?s)` matches all.
    for t in ["\n", "\r", "\u{85}", "\u{2028}", "\u{2029}"] {
        assert!(no(&format!("a{t}b"), "a.b"), "{t:?}");
        assert!(yes(&format!("a{t}b"), "(?s)a.b"), "{t:?}");
    }
}

/// Backtracking patterns stop at the limit in every function.
#[test]
fn backtracking_hits_the_limit_everywhere() {
    let input = format!("{}!", "a".repeat(40));
    let class = ErrorClass::KrishivRegexpBacktrackLimitExceeded;
    assert_eq!(
        try_bool("regexp_like", &input, r"^(a*)*\1c$")
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        try_int("regexp_count", &input, r"^(a*)*\1c$")
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        try_text("regexp_substr", &input, r"^(a*)*\1c$")
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        try_idx("regexp_extract", &input, r"^(a*)*\1c$", Some(0))
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        try_idx_int("regexp_instr", &input, r"^(a*)*\1c$", Some(0))
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        extract_all_pieces(r"^(a*)*\1c$", &input, Some(0), "`regexp_extract_all`")
            .unwrap_err()
            .error_class(),
        class
    );
    let one = arrow_array::StringArray::from(vec![Some(input.as_str())]);
    assert_eq!(
        split_regex(&one, r"^(a*)*\1c$", &[Some(-1)])
            .unwrap_err()
            .error_class(),
        class
    );
    // The translated `$`/multiline anchors stay linear on a near miss:
    // no backtrack limit on these, in the plain and the terminated form.
    // (`(a*)*$` still finds the two empty matches before the terminator.)
    let near_miss = format!("{}b\n", "a".repeat(60));
    for re in ["(a+)+$", "(a*)*$", "(a|aa)+$", "(?m)(a+)+$", "(?m)^(a+)+$"] {
        assert!(try_bool("regexp_like", &near_miss, re).is_ok(), "{re}");
        assert!(try_int("regexp_count", &near_miss, re).is_ok(), "{re}");
    }
    assert_eq!(try_int("regexp_count", &near_miss, "(a*)*$"), Ok(Some(2)));
    assert_eq!(try_int("regexp_count", &near_miss, "(a+)+$"), Ok(Some(0)));
}

/// The group index of `regexp_extract_all` and `regexp_count` spans rows.
#[test]
fn batches_share_compiled_patterns() {
    let s = strs(&[Some("1a"), Some("2b"), None, Some("x")]);
    let re = strs(&[
        Some(r"(\d+)([a-z]+)"),
        Some(r"(\d+)([a-z]+)"),
        None,
        Some(r"(\d+)([a-z]+)"),
    ]);
    assert_eq!(
        text(&run("regexp_extract", &[&s, &re])),
        [some("1"), some("2"), None, some("")]
    );
    assert_eq!(
        ints_of(&run("regexp_count", &[&s, &re])),
        [Some(1), Some(1), None, Some(0)]
    );
    assert_eq!(
        bools(&run("regexp_like", &[&s, &re])),
        [Some(true), Some(true), None, Some(false)]
    );
    assert_eq!(
        extract_all_pieces(r"(\d+)", "1a 2b", Some(0), "`regexp_extract_all`"),
        Ok(vec!["1".to_owned(), "2".to_owned()])
    );
}

/// Wrong arities and argument types do not bind.
#[test]
fn regex_bind_errors() {
    use crate::scalar::tests::bind;
    use crate::scalar::{BindArg, bind_scalar};
    use arrow_schema::DataType;
    let opts = crate::scalar::BindOptions::new();
    let utf8 = || BindArg::Column(DataType::Utf8);
    let int = || BindArg::Column(DataType::Int32);
    for (name, args) in [
        ("regexp_extract", vec![utf8()]),
        ("regexp_extract", vec![utf8(), utf8(), utf8(), int()]),
        ("regexp_count", vec![utf8(), int()]),
        ("regexp_count", vec![utf8(), utf8(), utf8()]),
        ("regexp_instr", vec![int(), utf8()]),
        ("regexp_substr", vec![utf8()]),
        ("regexp", vec![utf8()]),
        ("regexp_like", vec![utf8(), utf8(), utf8()]),
        ("regexp_replace", vec![utf8(), utf8(), utf8(), utf8()]),
    ] {
        let err = bind_scalar(name, &args, &opts).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{name}"
        );
    }
    assert!(
        bind("regexp_like", &[&strs(&[Some("a")]), &strs(&[Some("b")])]).return_type()
            == &DataType::Boolean
    );
}

/// Scalar (literal) patterns and indices broadcast over the batch.
#[test]
fn scalar_patterns_and_indices() {
    use std::sync::Arc;

    use crate::scalar::Arg;
    let s = strs(&[Some("1a"), Some("x"), None]);
    let re = strs(&[Some(r"(\d+)")]);
    let f = crate::scalar::tests::bind("regexp_extract", &[&s, &re]);
    let args = [crate::scalar::tests::col(&s), Arg::Scalar(Arc::clone(&re))];
    assert_eq!(
        text(&f.invoke(&args, 3).unwrap()),
        [some("1"), some(""), None]
    );
    let idx = ints(&[Some(1)]);
    let g = crate::scalar::tests::bind("regexp_instr", &[&s, &re, &idx]);
    let args = [
        Arg::Array(Arc::clone(&s)),
        Arg::Scalar(Arc::clone(&re)),
        Arg::Scalar(idx),
    ];
    assert_eq!(
        ints_of(&g.invoke(&args, 3).unwrap()),
        [Some(1), Some(0), None]
    );
}

/// A NULL group index is a NULL row and never validates.
#[test]
fn null_group_index_is_null() {
    let s = strs(&[Some("1a"), Some("1a")]);
    let re = strs(&[Some(r"(\d+)"), Some("(")]);
    let idx = ints(&[None, None]);
    assert_eq!(text(&run("regexp_extract", &[&s, &re, &idx])), [None, None]);
    assert_eq!(
        ints_of(&run("regexp_instr", &[&s, &re, &idx])),
        [None, None]
    );
}

/// `split_regex` needs one limit per row.
#[test]
fn split_limit_mismatch_is_internal() {
    let input = arrow_array::StringArray::from(vec![Some("a,b")]);
    let err = split_regex(&input, ",", &[Some(-1), Some(-1)]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
}
