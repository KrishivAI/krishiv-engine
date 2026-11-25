// SPDX-License-Identifier: Apache-2.0
//! Unit tests for the collation resolver.

use super::*;
use crate::AttrId;

fn leaf(c: Collation) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        crate::BoundColumn {
            id: AttrId(1),
            name: "c".to_owned(),
            data_type: DataType::collated_string(c),
            nullable: true,
        },
    ))))
}

#[test]
fn if_and_decode_count_only_their_result_positions() {
    // if(cond, a, b): a and b; decode(e, s1, r1, s2, r2, default): r1, r2, default.
    let ifs: Vec<bool> = (0..3).map(|i| is_result_arg("IF", i, 3)).collect();
    assert_eq!(ifs, [false, true, true]);
    let odd: Vec<bool> = (0..5).map(|i| is_result_arg("decode", i, 5)).collect();
    assert_eq!(odd, [false, false, true, false, true]);
    let even: Vec<bool> = (0..6).map(|i| is_result_arg("decode", i, 6)).collect();
    assert_eq!(even, [false, false, true, false, true, true]);
    assert!(is_result_arg("concat", 0, 2));
}

#[test]
fn a_collated_if_condition_gives_the_result_no_collation() {
    let lcase = Collation::parse("UTF8_LCASE").unwrap();
    let lit = |v: &str| Arc::new(Expr::new(ExprNode::Literal(Literal::String(v.to_owned()))));
    let call = Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
        name: "if".to_owned(),
        qualifier: Vec::new(),
        args: vec![leaf(lcase), lit("a"), lit("b")],
        arg_names: Vec::new(),
        distinct: false,
        filter: None,
        within_group: Vec::new(),
    })));
    let got = strength_of(&call, &mut Memo::with_default(Collation::UTF8_BINARY));
    assert_eq!(got, Some((Collation::UTF8_BINARY, Strength::Default)));
}

#[test]
fn a_decode_takes_its_collation_from_its_results() {
    let lcase = Collation::parse("UTF8_LCASE").unwrap();
    let lit = |v: &str| Arc::new(Expr::new(ExprNode::Literal(Literal::String(v.to_owned()))));
    let decode = |args: Vec<Arc<Expr>>| {
        Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
            name: "decode".to_owned(),
            qualifier: Vec::new(),
            args,
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
        })))
    };
    // Spark 4.2.0: a collated expr is no result; a collated default is.
    let plain = decode(vec![leaf(lcase), lit("a"), lit("x"), lit("y")]);
    assert_eq!(
        strength_of(&plain, &mut Memo::with_default(Collation::UTF8_BINARY)),
        Some((Collation::UTF8_BINARY, Strength::Default))
    );
    let dflt = decode(vec![lit("a"), lit("a"), lit("x"), leaf(lcase)]);
    assert_eq!(
        strength_of(&dflt, &mut Memo::with_default(Collation::UTF8_BINARY)),
        Some((lcase, Strength::Implicit))
    );
}

/// `e = concat(e, e)` sixty times shares one child per level: a walk that expands the
/// sharing would visit 2^60 nodes.
#[test]
fn strength_of_a_shared_expression_dag_is_linear() {
    let lcase = Collation::parse("UTF8_LCASE").unwrap();
    let mut e = leaf(lcase);
    for _ in 0..60 {
        e = Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
            name: "concat".to_owned(),
            qualifier: Vec::new(),
            args: vec![Arc::clone(&e), Arc::clone(&e)],
            arg_names: Vec::new(),
            distinct: false,
            filter: None,
            within_group: Vec::new(),
        })));
    }
    let got = strength_of(&e, &mut Memo::with_default(Collation::UTF8_BINARY));
    assert_eq!(got, Some((lcase, Strength::Implicit)));
}

#[test]
fn collation_and_typeof_end_the_propagation() {
    let lcase = Collation::parse("UTF8_LCASE").unwrap();
    for name in ["collation", "typeof"] {
        let call = Expr::function(name, vec![Expr::clone(&leaf(lcase))]);
        let got = strength_of(&call, &mut Memo::with_default(Collation::UTF8_BINARY));
        assert_eq!(
            got,
            Some((Collation::UTF8_BINARY, Strength::Default)),
            "{name}"
        );
    }
}
