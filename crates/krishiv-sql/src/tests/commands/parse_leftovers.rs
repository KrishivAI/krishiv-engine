// SPDX-License-Identifier: Apache-2.0
//! Parse checks left over from the `ROW FORMAT` / `CREATE NAMESPACE` review (opus-3, 2026-11-07), each case
//! from a Spark 4.2.0 run (`research/spark/row-format-file-format/probe13.py` in krishiv-internal; the
//! row tags `a1`... are the probe's).

use super::*;
use krishiv_common::error::ErrorClass;

fn err(sql: &str) -> ParseError {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(s) => panic!("{sql} parsed: {s:?}"),
        Err(e) => e,
    }
}

fn ok(sql: &str) {
    if let Err(e) = parse_statement(sql, &ParserOptions::default()) {
        panic!("{sql}: {}", e.error);
    }
}

/// a1-a8: `ALTER NAMESPACE .. UNSET PROPERTIES|DBPROPERTIES (keys)` parses, then
/// `cleanNamespaceProperties` refuses the reserved keys at the whole statement; `IF EXISTS` and an empty
/// list are syntax errors (`comment` is refused when the command runs, in the session).
#[test]
fn alter_namespace_unset_properties() {
    for sql in [
        "ALTER NAMESPACE ns UNSET PROPERTIES ('k')",
        "ALTER DATABASE ns UNSET DBPROPERTIES ('k', 'q')",
        "ALTER NAMESPACE ns UNSET PROPERTIES ('comment')",
        "ALTER NAMESPACE ns UNSET PROPERTIES ('provider')",
    ] {
        ok(sql);
    }
    for (sql, key, msg) in [
        (
            "ALTER NAMESPACE ns UNSET PROPERTIES ('location')",
            "location",
            "please use the LOCATION clause to specify it",
        ),
        (
            "ALTER NAMESPACE ns UNSET PROPERTIES ('owner')",
            "owner",
            "it will be set to the current user",
        ),
    ] {
        let e = err(sql);
        assert_eq!(
            e.error.error_class(),
            ErrorClass::UnsupportedFeatureSetNamespaceProperty,
            "{sql}"
        );
        let params = vec![("property", key.to_owned()), ("msg", msg.to_owned())];
        assert_eq!(e.error.message_parameters(), params, "{sql}");
        assert_eq!(e.span.range(), 0..sql.len(), "{sql}");
    }
    for sql in [
        "ALTER NAMESPACE ns UNSET PROPERTIES IF EXISTS ('k')",
        "ALTER NAMESPACE ns UNSET PROPERTIES ()",
    ] {
        assert_eq!(
            err(sql).error.error_class(),
            ErrorClass::ParseSyntaxError,
            "{sql}"
        );
    }
}
