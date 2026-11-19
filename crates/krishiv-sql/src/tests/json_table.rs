// SPDX-License-Identifier: Apache-2.0
//! `JSON_TABLE` (Spark 4.3, the flat form): what parses, what the lowering refuses, and the shape it lowers to.

use krishiv_common::error::ErrorClass;

use super::super::ast::{JsonTableColumnKind, SetExprKind, TableFactorKind};
use super::super::{ParserOptions, parse_query};

fn lowered(sql: &str) -> Result<krishiv_plan::Plan, super::super::ParseError> {
    let q = parse_query(sql, &ParserOptions::default()).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    crate::lower(&q, krishiv_plan::SourceId(0))
}

fn parse_fails(sql: &str) {
    assert!(
        parse_query(sql, &ParserOptions::default()).is_err(),
        "{sql} parsed"
    );
}

#[test]
fn the_call_parses_into_its_parts() {
    let q = parse_query(
        "SELECT * FROM JSON_TABLE(js, '$[*]' COLUMNS (rn FOR ORDINALITY, a INT PATH '$.a', b STRING, e INT EXISTS PATH '$.z') ERROR ON ERROR) AS t",
        &ParserOptions::default(),
    )
    .unwrap();
    let SetExprKind::Select(select) = &q.body.kind else {
        panic!("not a select: {:?}", q.body);
    };
    let factor = &select.from.as_ref().unwrap().relations[0].primary;
    let TableFactorKind::JsonTable(t) = &factor.kind else {
        panic!("not JSON_TABLE: {:?}", factor.kind);
    };
    assert_eq!(t.path, "$[*]");
    assert!(t.error_on_error);
    let kinds: Vec<_> = t.columns.iter().map(|c| &c.kind).collect();
    assert!(matches!(kinds[0], JsonTableColumnKind::Ordinality));
    assert!(
        matches!(kinds[1], JsonTableColumnKind::Value { path: Some((p, _)), .. } if p == "$.a")
    );
    assert!(matches!(
        kinds[2],
        JsonTableColumnKind::Value { path: None, .. }
    ));
    assert!(matches!(kinds[3], JsonTableColumnKind::Exists { .. }));
    assert!(factor.alias.is_some());
}

#[test]
fn what_spark_rejects_does_not_parse() {
    // Spark: no `NESTED PATH` in the flat form here, no column-level `ON ERROR`, the on-error clause sits inside the
    // parentheses, the row path is a string literal, and `COLUMNS` needs at least one column.
    parse_fails(
        "SELECT * FROM json_table('{}', '$' COLUMNS (NESTED PATH '$.b[*]' COLUMNS (c INT PATH '$')))",
    );
    parse_fails("SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$.a' NULL ON ERROR))");
    parse_fails("SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$.a')) NULL ON ERROR");
    parse_fails("SELECT * FROM json_table('{}', 5 COLUMNS (a INT PATH '$.a'))");
    parse_fails("SELECT * FROM json_table('{}', '$' COLUMNS ())");
    parse_fails("SELECT * FROM json_table('{}', '$', COLUMNS (a INT PATH '$.a'))");
}

#[test]
fn it_lowers_to_a_projection_over_a_posexplode() {
    let plan =
        lowered("SELECT * FROM json_table('[1]', '$[*]' COLUMNS (rn FOR ORDINALITY, a INT))")
            .unwrap();
    let text = plan.root().to_string();
    assert!(text.contains("Generate"), "{text}");
    assert!(text.contains("__json_table_rows"), "{text}");
}

#[test]
fn unsupported_paths_and_duplicate_columns_fail_when_lowered() {
    let class = |sql: &str| lowered(sql).unwrap_err().error.error_class();
    for sql in [
        "SELECT * FROM json_table('{}', '$.*' COLUMNS (a INT PATH '$.a'))",
        "SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$.a[*]'))",
        "SELECT * FROM json_table('{}', '$[*][*]' COLUMNS (a INT PATH '$'))",
        "SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$..a'))",
        "SELECT * FROM json_table('{}', 'lax $' COLUMNS (a INT PATH '$.a'))",
        "SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$.\"a\"'))",
        "SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$[\"a\"]'))",
        "SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH 'a'))",
    ] {
        assert_eq!(
            class(sql),
            ErrorClass::DatatypeMismatchInvalidJsonTablePath,
            "{sql}"
        );
    }
    assert_eq!(
        class("SELECT * FROM json_table('{}', '$' COLUMNS (a INT PATH '$.a', A INT PATH '$.b'))"),
        ErrorClass::InvalidSqlSyntaxDuplicateJsonTableColumn
    );
}

#[test]
fn the_allowed_paths_lower() {
    for path in [
        "$",
        "$.a",
        "$.a.b",
        "$['a']",
        "$[0]",
        "$.a[1].b",
        "$['a b'].c",
    ] {
        let path = path.replace('\'', "''");
        let sql = format!("SELECT * FROM json_table('{{}}', '$' COLUMNS (a INT PATH '{path}'))");
        lowered(&sql).unwrap_or_else(|e| panic!("{path}: {e:?}"));
    }
    for path in ["$", "$[*]", "$.a[*]", "$.a.b[*]", "$[1]"] {
        let sql = format!("SELECT * FROM json_table('{{}}', '{path}' COLUMNS (a INT))");
        lowered(&sql).unwrap_or_else(|e| panic!("{path}: {e:?}"));
    }
}
