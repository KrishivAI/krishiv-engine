// SPDX-License-Identifier: Apache-2.0
//! Pretty-printer details: literals, intervals, escaping, redaction.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{
    Builtin, DayTimeField, Expr, ExprNode, FunctionCall, Literal, LogicalPlan, PlanNode,
    ReadSource, SubqueryKind, YearMonthField,
};

fn text(v: Literal) -> String {
    Expr::from(v).to_string()
}

#[test]
fn intervals_print_with_their_qualifiers() {
    use DayTimeField::{Day, Hour, Second};
    use YearMonthField::{Month, Year};
    let ym = |months, start, end| text(Literal::YearMonthInterval { months, start, end });
    assert_eq!(ym(12, Year, Year), "INTERVAL '1' YEAR");
    assert_eq!(ym(12, Month, Month), "INTERVAL '12' MONTH");
    assert_eq!(ym(-14, Year, Month), "INTERVAL '-1-2' YEAR TO MONTH");
    let dt = |micros, start, end| text(Literal::DayTimeInterval { micros, start, end });
    assert_eq!(dt(90 * 86_400_000_000, Day, Day), "INTERVAL '90' DAY");
    let micros = ((26 * 60 + 3) * 60 + 4) * 1_000_000 + 500_000;
    assert_eq!(
        dt(micros, Day, Second),
        "INTERVAL '1 02:03:04.5' DAY TO SECOND"
    );
    assert_eq!(
        dt(micros, Hour, Second),
        "INTERVAL '26:03:04.5' HOUR TO SECOND"
    );
}

#[test]
fn literals_print_as_spark_sql() {
    assert_eq!(text(Literal::Double(f64::NAN)), "CAST('NaN' AS DOUBLE)");
    assert_eq!(
        text(Literal::Float(f32::NEG_INFINITY)),
        "CAST('-Infinity' AS FLOAT)"
    );
    assert_eq!(text(Literal::String("a'b\nc".into())), r"'a\'b\nc'");
    assert_eq!(text(Literal::Date(9374)), "DATE '1995-09-01'");
}

#[test]
fn secret_options_are_redacted() {
    let options: BTreeMap<String, String> = [
        ("fs.s3a.secret.key", "hunter2"),
        ("password", "hunter2"),
        ("header", "true"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let plan = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "csv".to_owned(),
        paths: vec!["/d".to_owned()],
        options,
    }));
    let shown = plan.to_string();
    assert!(!shown.contains("hunter2"), "{shown}");
    assert!(shown.contains("header=true"), "{shown}");
}

#[test]
fn operators_print_in_sql_form() {
    let a = || Expr::column(["a"]);
    let ne = Expr::builtin(
        Builtin::Not,
        vec![Expr::builtin(Builtin::Eq, vec![a(), a()])],
    );
    assert_eq!(ne.to_string(), "(NOT (a == a))");
    let between = Expr::builtin(Builtin::Between, vec![a(), a(), a()]);
    assert_eq!(between.to_string(), "(a BETWEEN a AND a)");
    let when = Expr::builtin(Builtin::When, vec![a(), a(), a()]);
    assert_eq!(when.to_string(), "CASE WHEN a THEN a ELSE a END");
}

#[test]
fn in_subquery_prints_parenthesized() {
    let sub = Expr::subquery(
        SubqueryKind::In(vec![Arc::new(Expr::column(["a"]))]),
        LogicalPlan::table(["u"]),
    );
    assert_eq!(sub.to_string(), "((a) IN (list-query))");
}

/// Shared relations print once and are referred to afterwards; a shared
/// expression's text is cut.
#[test]
fn shared_subtrees_print_once() {
    let text = super::depth::self_joins(30).to_string();
    assert!(text.lines().count() < 100, "{text}");
    assert!(text.contains("(shared #1)"), "{text}");
    assert!(text.contains("Shared #1, printed above"), "{text}");
    let small = super::depth::self_joins(2).to_string();
    assert_eq!(
        small,
        "Join Inner\n\
         :- Join Inner (shared #1)\n\
         :  :- Read t (shared #2)\n\
         :  +- Shared #2, printed above\n\
         +- Shared #1, printed above\n"
    );
    let e = (0..40).fold(Arc::new(Expr::column(["a"])), |e, _| {
        Arc::new(Expr::new(ExprNode::Function(FunctionCall {
            name: Builtin::And.name().to_owned(),
            args: vec![Arc::clone(&e), e],
            distinct: false,
            filter: None,
        })))
    });
    let shown = e.to_string();
    assert!(shown.len() < 70 * 1024 && shown.ends_with("...(truncated)"));
}
