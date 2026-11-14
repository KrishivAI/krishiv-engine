// SPDX-License-Identifier: Apache-2.0
//! Lambda parameters named by keywords (SQL-3) against a Spark 4.2.0 run
//! (`tests/fixtures/spark-4.2.0/lambda-keyword-params.json`,
//! `tests/scripts/gen_lambda_keyword_params.py`): for every keyword and body
//! shape, `SELECT transform(array(1), <lambda>)` parses, and the lowered
//! body holds as many column references to the keyword (the analyzer's
//! lambda variables) as Spark's `AstBuilder` made lambda variables.
#![expect(clippy::panic, reason = "test code")]

use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, PlanNode, SourceId};
use krishiv_sql::{ParserOptions, parse_and_lower};

const FIXTURE: &str =
    include_str!("../../../tests/fixtures/spark-4.2.0/lambda-keyword-params.json");

/// The column references named `name` (first part) under `e`.
fn refs(e: &Expr, name: &str) -> usize {
    let own = match &e.node {
        ExprNode::Column(c) => usize::from(c.name.first().is_some_and(|n| n == name)),
        _ => 0,
    };
    let mut below = 0;
    e.for_each_child(&mut |c: &Arc<Expr>| below += refs(c, name));
    own + below
}

/// The body of the lambda that is `transform`'s second argument.
fn lambda_body(sql: &str) -> Arc<Expr> {
    let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let PlanNode::Project { exprs, .. } = &plan.root().node else {
        panic!("{sql}: no projection");
    };
    let ExprNode::Function(call) = &exprs[0].node else {
        panic!("{sql}: no call");
    };
    let ExprNode::Lambda { body, .. } = &call.args[1].node else {
        panic!("{sql}: no lambda");
    };
    Arc::clone(body)
}

/// `shape` with each standalone `K` replaced by `keyword`.
fn substitute(shape: &str, keyword: &str) -> String {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    let chars: Vec<char> = shape.chars().collect();
    let mut out = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let before = i.checked_sub(1).and_then(|j| chars.get(j).copied());
        if c == 'K' && !word(before) && !word(chars.get(i + 1).copied()) {
            out.push_str(keyword);
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn every_keyword_names_a_lambda_parameter_as_in_spark() {
    let doc: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let shapes: Vec<&str> = doc["shapes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    let mut checked = 0;
    for (keyword, counts) in doc["body_refs"].as_object().unwrap() {
        for (shape, want) in shapes.iter().zip(counts.as_array().unwrap()) {
            let want = want
                .as_u64()
                .unwrap_or_else(|| panic!("{keyword} {shape}: Spark failed with {want}"));
            let lambda = substitute(shape, keyword);
            let sql = format!("SELECT transform(array(1), {lambda})");
            let got = refs(&lambda_body(&sql), keyword);
            assert_eq!(got as u64, want, "{sql}");
            checked += 1;
        }
    }
    assert!(checked > 8000, "{checked} cases");
}
