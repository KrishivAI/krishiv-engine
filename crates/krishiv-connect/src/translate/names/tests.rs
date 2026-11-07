// SPDX-License-Identifier: Apache-2.0
//! Tests of the multipart-identifier syntax (the attribute-name syntax is
//! tested with its parser in `krishiv_common::names`).

use krishiv_common::error::ErrorClass;

use super::*;

fn ident(text: &str) -> Vec<String> {
    parse_multipart_identifier(text).expect("parses")
}

#[test]
fn multipart_identifiers() {
    assert_eq!(ident("t"), ["t"]);
    assert_eq!(ident("db.t"), ["db", "t"]);
    assert_eq!(ident("cat.db.t_1"), ["cat", "db", "t_1"]);
    assert_eq!(ident("`my db`.`ev.ents`"), ["my db", "ev.ents"]);
    assert_eq!(ident("`a``b`"), ["a`b"]);
    assert_eq!(ident("  db . t  "), ["db", "t"]);
    assert_eq!(ident("`h\u{e9}llo`"), ["h\u{e9}llo"]);
    // Digits are identifier characters, but a word that is a numeric
    // literal is not an identifier (Spark 4.2: `spark.read.table("123")`
    // fails to parse, `"123abc"` is a table name).
    for word in [
        "123abc", "_1", "1_", "1e3x", "1e", "1ex", "12Lx", "0x1F", "1eF", "1e3L",
    ] {
        assert_eq!(ident(word), [word], "{word}");
    }
    assert_eq!(ident("a.1b"), ["a", "1b"]);
    assert_eq!(ident("`123`"), ["123"], "quoted, any word is a name");
}

#[test]
fn malformed_multipart_identifiers() {
    for (bad, near) in [
        ("", "end of input"),
        ("   ", "end of input"),
        ("db.", "end of input"),
        (".t", "'.'"),
        ("a..b", "'.'"),
        ("a b", "'b'"),
        ("a-b", "'-'"),
        ("`a", "end of input"),
        ("a.`b", "end of input"),
        ("h\u{e9}llo", "'\u{e9}'"),
        ("a$", "'$'"),
        // Numeric literals, as Spark 4.2's lexer reads them.
        ("123", "'123'"),
        ("2024", "'2024'"),
        ("12L", "'12L'"),
        ("12l", "'12l'"),
        ("12S", "'12S'"),
        ("12Y", "'12Y'"),
        ("1e3", "'1e3'"),
        ("1E3", "'1E3'"),
        ("12BD", "'12BD'"),
        ("12bd", "'12bd'"),
        ("12D", "'12D'"),
        ("12F", "'12F'"),
        ("1e3F", "'1e3F'"),
        ("1e3BD", "'1e3BD'"),
        ("12e3d", "'12e3d'"),
        ("a.123", "'123'"),
        ("x.12d", "'12d'"),
        ("a.12L.b", "'12L'"),
    ] {
        let e = parse_multipart_identifier(bad).expect_err(bad);
        assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError, "{bad:?}");
        assert!(e.message().contains(near), "{bad:?}: {}", e.message());
    }
}
