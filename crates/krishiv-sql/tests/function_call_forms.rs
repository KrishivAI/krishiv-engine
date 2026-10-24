// SPDX-License-Identifier: Apache-2.0
//! Lowering of the P2-T180 function-call forms: named arguments, qualified
//! builtin names, `ILIKE`, and `LIKE|ILIKE ANY|SOME|ALL`.

use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, Plan, PlanNode, SourceId};
use krishiv_sql::{ParserOptions, parse_and_lower};

fn lower(sql: &str) -> Result<Plan, krishiv_sql::ParseError> {
    parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
}

/// The single select-list expression of `SELECT <expr>`.
fn select_expr(sql: &str) -> Option<Arc<Expr>> {
    let plan = lower(sql).ok()?;
    let PlanNode::Project { exprs, .. } = &plan.root().node else {
        return None;
    };
    let [one] = exprs.as_slice() else {
        return None;
    };
    Some(Arc::clone(one))
}

fn call_of(sql: &str) -> Option<krishiv_plan::FunctionCall> {
    match &select_expr(sql)?.node {
        ExprNode::Function(f) => Some(f.clone()),
        _ => None,
    }
}

#[test]
fn positional_calls_carry_no_names() {
    for sql in ["SELECT abs(-1)", "SELECT lower('a')", "SELECT mask('a')"] {
        let f = call_of(sql).unwrap();
        assert!(f.qualifier.is_empty(), "{sql}");
        assert!(f.arg_names.is_empty(), "{sql}");
    }
}

#[test]
fn named_arguments_carry_their_names() {
    let f = call_of("SELECT mask('a', lowerChar => 'q', str => 'b')").unwrap();
    assert_eq!(f.name, "mask");
    assert_eq!(
        f.arg_names,
        vec![None, Some("lowerChar".to_owned()), Some("str".to_owned())]
    );
    assert_eq!(f.args.len(), 3);
}

#[test]
fn qualified_names_lower_to_qualifier_parts() {
    let f = call_of("SELECT builtin.abs(-1)").unwrap();
    assert_eq!(f.name, "abs");
    assert_eq!(f.qualifier, vec!["builtin".to_owned()]);
    let f = call_of("SELECT system.builtin.abs(-1)").unwrap();
    assert_eq!(f.name, "abs");
    assert_eq!(f.qualifier, vec!["system".to_owned(), "builtin".to_owned()]);
    let f = call_of("SELECT db.f(a)").unwrap();
    assert_eq!(f.name, "f");
    assert_eq!(f.qualifier, vec!["db".to_owned()]);
}

#[test]
fn ilike_lowers_to_the_ilike_operator() {
    let f = call_of("SELECT 'a' ILIKE 'b'").unwrap();
    assert_eq!(f.name, "ilike");
    assert_eq!(f.args.len(), 2);
    let f = call_of("SELECT 'a' ILIKE 'b' ESCAPE '#'").unwrap();
    assert_eq!(f.name, "ilike");
    assert_eq!(f.args.len(), 3);
}

#[test]
fn like_quantified_lowers_to_one_call_per_form() {
    for (sql, name, n) in [
        ("SELECT a LIKE ANY ('x', 'y') FROM t", "like_any", 3),
        ("SELECT a LIKE SOME ('x') FROM t", "like_any", 2),
        ("SELECT a LIKE ALL ('x', 'y') FROM t", "like_all", 3),
        ("SELECT a NOT LIKE ANY ('x') FROM t", "not_like_any", 2),
        ("SELECT a NOT LIKE ALL ('x', 'y') FROM t", "not_like_all", 3),
        ("SELECT a ILIKE ANY ('x') FROM t", "ilike_any", 2),
        ("SELECT a ILIKE ALL ('x', 'y') FROM t", "ilike_all", 3),
        ("SELECT a NOT ILIKE ANY ('x') FROM t", "not_ilike_any", 2),
        (
            "SELECT a NOT ILIKE ALL ('x', 'y') FROM t",
            "not_ilike_all",
            3,
        ),
    ] {
        let f = call_of(sql).unwrap();
        assert_eq!(f.name, name, "{sql}");
        assert_eq!(f.args.len(), n, "{sql}");
        assert!(f.arg_names.is_empty(), "{sql}");
    }
}

#[test]
fn not_outside_quantified_stays_outside() {
    // `NOT a LIKE ANY (...)` is a `NOT` over the quantified call.
    match &select_expr("SELECT NOT a LIKE ANY ('x') FROM t")
        .unwrap()
        .node
    {
        ExprNode::Function(f) => assert_eq!(f.name, "not"),
        other => panic!("not a NOT call: {other:?}"),
    }
}

#[test]
fn empty_like_any_is_still_a_parse_error() {
    let err = lower("SELECT a LIKE ANY () FROM t").unwrap_err();
    assert_eq!(err.error.error_class().name(), "_LEGACY_ERROR_TEMP_0064");
}
