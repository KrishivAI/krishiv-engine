// SPDX-License-Identifier: Apache-2.0

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use krishiv_common::error::{ErrorClass, KrishivError};

use super::java::translate;
use crate::scalar::tests::{ints, run, strs};

/// The strings of a result column.
fn text(a: &ArrayRef) -> Vec<Option<String>> {
    a.as_string::<i32>()
        .iter()
        .map(|v| v.map(str::to_owned))
        .collect()
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
    for re in ["(", "[", "a{", "a++", "\\", "\\G"] {
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
    assert!(translate("a*+", false).is_err());
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
