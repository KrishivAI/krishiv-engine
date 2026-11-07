// SPDX-License-Identifier: Apache-2.0
//! Tests of the attribute-name syntax and its quoting. The cases are what Spark 4.2.0's
//! `UnresolvedAttribute.parseAttributeName` returns for them, and what its `COLUMN_ALREADY_EXISTS`
//! shows for a repeated name (local Spark 4.2.0 run, 2026-11-07).

use crate::error::ErrorClass;

use super::*;

fn parts(name: &str) -> Vec<String> {
    parse_attribute_name(name).expect("parses")
}

#[test]
fn attribute_names_split_at_unquoted_dots() {
    assert_eq!(parts("a"), ["a"]);
    assert_eq!(parts("a.b"), ["a", "b"]);
    assert_eq!(parts("a.`b.c`"), ["a", "b.c"]);
    assert_eq!(parts("`a`.b"), ["a", "b"]);
    assert_eq!(parts("`a``b`"), ["a`b"]);
    assert_eq!(parts("a.`b`.c"), ["a", "b", "c"]);
    assert_eq!(parts("`a`.`b`"), ["a", "b"]);
    assert_eq!(parts("x.*"), ["x", "*"]);
    assert_eq!(parts("``"), [""], "an empty quoted name is one empty part");
    assert_eq!(parts("h\u{e9}llo.w\u{f6}rld"), ["h\u{e9}llo", "w\u{f6}rld"]);
}

#[test]
fn malformed_attribute_names_are_syntax_errors() {
    for bad in ["a.", ".a", "a..b", "`a`b", "a`b", "`a", "`a`.", "a.`b", "."] {
        let e = parse_attribute_name(bad).expect_err(bad);
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidAttributeNameSyntax,
            "{bad}"
        );
        assert_eq!(e.sql_state(), Some("42601"));
        assert!(e.message().contains(bad), "{bad}: {}", e.message());
    }
    // Spark indexes before the start for ".a"; here it is the same error.
    assert!(parse_attribute_name("").is_ok_and(|p| p == [""]));
}

#[test]
fn quoting_follows_to_sql_id() {
    assert_eq!(quote_parts(&["a".into(), "b`c".into()]), "`a`.`b``c`");
    assert_eq!(to_sql_id("a.b").ok().as_deref(), Some("`a`.`b`"));
    assert_eq!(to_sql_id("a b").ok().as_deref(), Some("`a b`"));
    assert_eq!(to_sql_id("``").ok().as_deref(), Some("``"));
    assert_eq!(to_sql_id("").ok().as_deref(), Some("``"));
}

#[test]
fn a_repeated_name_is_quoted_by_parts() {
    for (name, shown) in [
        ("ab", "`ab`"),
        ("a.b", "`a`.`b`"),
        ("a.b.c", "`a`.`b`.`c`"),
        ("A.B", "`A`.`B`"),
        ("a b", "`a b`"),
        ("", "``"),
    ] {
        let e = column_already_exists(name);
        assert_eq!(e.error_class(), ErrorClass::ColumnAlreadyExists, "{name}");
        assert_eq!(e.sql_state(), Some("42711"));
        assert_eq!(
            e.message(),
            format!(
                "The column {shown} already exists. Choose another name or rename the existing column."
            ),
            "{name}"
        );
    }
}

#[test]
fn a_repeated_name_that_is_not_an_attribute_name_is_a_syntax_error() {
    for name in ["a`b", ".a", "a.", "a..b", "a.`b"] {
        let e = column_already_exists(name);
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidAttributeNameSyntax,
            "{name}"
        );
        assert_eq!(e.sql_state(), Some("42601"));
        assert!(
            e.message()
                .starts_with(&format!("Syntax error in the attribute name: {name}.")),
            "{}",
            e.message()
        );
    }
}

#[test]
fn the_smallest_repeated_name_is_reported() {
    let check = |names: &[&str], cs| {
        check_column_name_duplication(names.iter().copied(), cs)
            .err()
            .map(|e| e.message().to_owned())
    };
    // Spark 4.2.0: `SELECT 1 AS z, 2 AS z, 3 AS a, 4 AS a` reports `a`.
    let a = Some(
        "The column `a` already exists. Choose another name or rename the existing column."
            .to_owned(),
    );
    assert_eq!(check(&["z", "z", "a", "a"], false), a);
    assert_eq!(check(&["z", "A", "z", "a"], false), a);
    // ... and a `z` repeat with a bad name still reports `a` (smaller); `a``b` twice is the syntax error.
    assert_eq!(check(&["z`b", "z`b", "a", "a"], false), a);
    assert!(
        check(&["z", "z", "a`b", "a`b"], false).is_some_and(|m| m.contains("attribute name: a`b."))
    );
    // Case-sensitive, `A` and `a` are two names and `A` keeps its case.
    assert_eq!(check(&["A", "a"], true), None);
    assert!(check(&["A", "a", "A"], true).is_some_and(|m| m.contains("`A`")));
    assert_eq!(check(&["a", "b"], false), None);
    assert_eq!(check(&[], false), None);
    // Scala orders strings by UTF-16 code units: U+FF5A sorts after U+1F600 (a surrogate pair).
    assert!(
        check(&["\u{ff5a}", "\u{ff5a}", "\u{1f600}", "\u{1f600}"], true)
            .is_some_and(|m| m.contains('\u{1f600}'))
    );
}

#[test]
fn an_auto_generated_subquery_alias_is_left_out() {
    let parts = |p: &[&str]| quote_parts(&p.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>());
    assert_eq!(parts(&["__auto_generated_subquery_name", "x"]), "`x`");
    assert_eq!(
        parts(&["__auto_generated_subquery_name"]),
        "`__auto_generated_subquery_name`"
    );
    assert_eq!(
        parts(&["a", "__auto_generated_subquery_name", "x"]),
        "`a`.`__auto_generated_subquery_name`.`x`"
    );
}
