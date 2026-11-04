// SPDX-License-Identifier: Apache-2.0
//! The AST helpers other crates walk with: `Expr::for_each_child` (including
//! the expressions of an inline window specification), the span of a script
//! element, and the temporary function name check.
#![expect(clippy::panic, reason = "test code")]

use krishiv_sql::ast::{Expr, Ident, ObjectName, ScriptStatement, Span, Statement};
use krishiv_sql::{
    ParserOptions, check_temporary_function_name, parse_named_expression, parse_statement,
};

/// The source text of each direct child of the expression `sql`.
fn children(sql: &str) -> Vec<String> {
    let item = parse_named_expression(sql, &ParserOptions::default())
        .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    let mut out = Vec::new();
    item.expr
        .for_each_child(&mut |c: &Expr| out.push(sql[c.span.range()].to_string()));
    out
}

#[test]
fn children_are_visited_in_order() {
    assert_eq!(children("a + b * c"), ["a", "b * c"]);
    assert_eq!(children("a"), Vec::<String>::new());
    assert_eq!(children("f(x, y)"), ["x", "y"]);
}

#[test]
fn an_inline_window_specification_is_visited_after_the_arguments() {
    // Arguments, then partitioning, ordering and the frame offsets.
    assert_eq!(
        children(
            "sum(x) OVER (PARTITION BY p, q ORDER BY o, r \
             ROWS BETWEEN 1 PRECEDING AND 2 FOLLOWING)"
        ),
        ["x", "p", "q", "o", "r", "1", "2"]
    );
    // A named window holds no expressions of its own.
    assert_eq!(children("sum(x) OVER w"), ["x"]);
    // A frame without an end bound.
    assert_eq!(
        children("sum(x) OVER (ORDER BY o ROWS 3 PRECEDING)"),
        ["x", "o", "3"]
    );
}

#[test]
fn a_script_element_spans_its_source_text() {
    let sql = "BEGIN DECLARE v INT DEFAULT 1; SELECT v; END";
    let Statement::Script(script) =
        parse_statement(sql, &ParserOptions::default()).unwrap_or_else(|e| panic!("{sql}: {e:?}"))
    else {
        panic!("not a script");
    };
    let spans: Vec<&str> = script
        .body
        .iter()
        .map(|s: &ScriptStatement| &sql[s.span().range()])
        .collect();
    assert_eq!(spans, ["DECLARE v INT DEFAULT 1", "SELECT v"]);
}

#[test]
fn a_temporary_function_name_takes_no_database() {
    let ident = |value: &str| Ident {
        value: value.to_string(),
        quoted: false,
        span: Span::new(0, value.len()),
    };
    let name = |parts: &[&str]| ObjectName {
        parts: parts.iter().map(|p| ident(p)).collect(),
        span: Span::new(0, 4),
    };
    let qualified = name(&["db", "f"]);
    let err = check_temporary_function_name(&qualified, qualified.span).unwrap_err();
    assert!(
        format!("{err:?}").contains("InvalidTempObjQualifier"),
        "{err:?}"
    );
    let bare = name(&["f"]);
    assert!(check_temporary_function_name(&bare, bare.span).is_ok());
    // The session's own qualifier is allowed.
    let session = name(&["session", "f"]);
    assert!(check_temporary_function_name(&session, session.span).is_ok());
}
