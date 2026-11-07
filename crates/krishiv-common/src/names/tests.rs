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
