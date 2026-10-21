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
    for (rep, java) in [
        ("$", "group index is missing"),
        ("$x", "Illegal group reference"),
        (r"\", "character to be escaped is missing"),
        ("$3", "No group 3"),
        ("${y}", "No group with name {y}"),
        ("${x", "missing trailing '}'"),
    ] {
        let e = fails("ab", "(?<x>a)", rep);
        assert!(e.message().contains(java), "{rep}: {}", e.message());
    }
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
        translate(r"a.b", false).unwrap().text.matches("[^").count(),
        1
    );
    let t = translate("a$", false).unwrap();
    assert!(t.has_dollar);
    assert_eq!(t.text, r"a\z");
    assert!(translate("a$", true).unwrap().text.contains("(?="));
    assert!(translate("a*+", false).is_err());
}
