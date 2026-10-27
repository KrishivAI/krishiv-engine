// SPDX-License-Identifier: Apache-2.0
//! Pretty-printer details: literals, intervals, escaping, typed options.

use std::sync::Arc;

use krishiv_common::secret::OptionValue;

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
            within_group: Vec::new(),
            name: Builtin::And.name().to_owned(),
            qualifier: Vec::new(),
            arg_names: Vec::new(),
            args: vec![Arc::clone(&e), e],
            distinct: false,
            filter: None,
        })))
    });
    let shown = e.to_string();
    assert!(shown.len() < 70 * 1024 && shown.ends_with("...(truncated)"));
}

#[test]
fn huge_literals_are_cut_at_the_expression_cap() {
    let big = "x".repeat(200 * 1024);
    let shown = text(Literal::String(big));
    assert!(
        shown.ends_with("...(truncated)"),
        "{}",
        &shown[shown.len() - 40..]
    );
    assert!(shown.len() <= 64 * 1024 + "...(truncated)".len());
    let bytes = text(Literal::Binary(vec![0xAB; 200 * 1024]));
    assert!(bytes.ends_with("...(truncated)"));
    assert!(bytes.len() <= 64 * 1024 + "...(truncated)".len());
}

#[test]
fn options_print_as_typed_values() {
    let options = [
        ("fs.s3a.secret.key", "hunter2"),
        ("password", "hunter2"),
        ("header", "true"),
        (
            "url",
            "jdbc:postgresql://user:hunter2@h/db?password=hunter2&ssl=true",
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), OptionValue::from_user(k, v)))
    .chain([("token".to_owned(), OptionValue::reference("prod", "kafka"))])
    .collect();
    let plan = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "csv".to_owned(),
        paths: vec![OptionValue::from_user("path", "/d")],
        options,
    }));
    assert_eq!(
        plan.to_string(),
        "Read csv [/d] {fs.s3a.secret.key=*********(redacted), header=true, password=*********(redacted), \
         token=secret('prod', 'kafka'), \
         url=jdbc:postgresql://user:*********(redacted)@h/db?password=*********(redacted)&ssl=true}\n"
    );
}

#[test]
fn sql_text_and_literal_urls_never_print() {
    let plan = LogicalPlan::new(PlanNode::Sql {
        query: "select * from t where password = 'hunter2'".to_owned(),
        args: crate::SqlArgs::None,
    });
    assert_eq!(plan.to_string(), "Sql [42 bytes]\n");
    let url = crate::Expr::from(crate::Literal::String("https://u:hunter2@h/".to_owned()));
    let shown = LogicalPlan::table(["t"]).project(vec![url]).to_string();
    assert!(!shown.contains("hunter2"), "{shown}");
    assert!(
        shown.contains("'https://u:*********(redacted)@h/'"),
        "{shown}"
    );
}

#[test]
fn literal_debug_shows_every_kind_and_hides_url_credentials() {
    use crate::{DayTimeField, Literal, YearMonthField};
    let all = [
        Literal::Null(None),
        Literal::Boolean(true),
        Literal::Byte(1),
        Literal::Short(2),
        Literal::Integer(3),
        Literal::Long(4),
        Literal::Float(1.5),
        Literal::Double(2.5),
        Literal::Decimal {
            unscaled: 15,
            data_type: krishiv_types::DecimalType::try_new(2, 1).unwrap(),
        },
        Literal::String("plain".to_owned()),
        Literal::String("https://u:hunter2@h/".to_owned()),
        Literal::Binary(vec![1]),
        Literal::Date(1),
        Literal::Timestamp(2),
        Literal::TimestampNtz(3),
        Literal::YearMonthInterval {
            months: 13,
            start: YearMonthField::Year,
            end: YearMonthField::Month,
        },
        Literal::DayTimeInterval {
            micros: 5,
            start: DayTimeField::Day,
            end: DayTimeField::Second,
        },
    ];
    let shown = format!("{all:?}");
    assert!(!shown.contains("hunter2"), "{shown}");
    for kind in [
        "Null",
        "Decimal",
        "plain",
        "Binary",
        "TimestampNtz",
        "YearMonthInterval",
        "DayTimeInterval",
    ] {
        assert!(shown.contains(kind), "{kind}: {shown}");
    }
}
