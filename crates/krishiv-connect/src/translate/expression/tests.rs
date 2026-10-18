// SPDX-License-Identifier: Apache-2.0
//! Tests of the expression translation.

use krishiv_common::error::ErrorClass;
use krishiv_plan::{Expr, ExprNode, FunctionCall};

use super::super::testing::{attr, attr_with, conf, expr, fails, func, lit, proto_type};
use super::super::{TranslateConf, translate_expression};
use super::*;
use crate::proto::v4_2::data_type as dt;
use crate::proto::v4_2::expression::literal::LiteralType;

fn text(e: &Expression) -> String {
    translate_expression(e, &conf())
        .unwrap_or_else(|e| panic!("{e}"))
        .to_string()
}

fn err(e: &Expression) -> KrishivError {
    fails(translate_expression(e, &conf()))
}

fn function(e: &Expr) -> &FunctionCall {
    match &e.node {
        ExprNode::Function(f) => f,
        other => panic!("not a function: {other:?}"),
    }
}

#[test]
fn attributes_keep_their_plan_id_and_split_their_name() {
    let e = translate_expression(&attr_with("t.`a.b`", Some(7)), &conf()).unwrap();
    let ExprNode::Column(c) = &e.node else {
        panic!("not a column")
    };
    assert_eq!(c.name, ["t", "a.b"]);
    assert_eq!(c.plan_id, Some(PlanId(7)));
    assert_eq!(e.to_string(), "t.`a.b`@plan7");
    // plan_id 0 is a plan id like any other.
    assert_eq!(text(&attr_with("a", Some(0))), "a@plan0");
    assert_eq!(text(&attr("a")), "a");
}

#[test]
fn bad_attribute_names_and_metadata_columns() {
    assert_eq!(
        err(&attr("a..b")).error_class(),
        ErrorClass::InvalidAttributeNameSyntax
    );
    let metadata = expr(ExprType::UnresolvedAttribute(
        expression::UnresolvedAttribute {
            unparsed_identifier: "_metadata".into(),
            plan_id: None,
            is_metadata_column: Some(true),
        },
    ));
    let e = err(&metadata);
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.message().contains("is_metadata_column"),
        "{}",
        e.message()
    );
    // A metadata flag of false is the default.
    let plain = expr(ExprType::UnresolvedAttribute(
        expression::UnresolvedAttribute {
            unparsed_identifier: "a".into(),
            plan_id: None,
            is_metadata_column: Some(false),
        },
    ));
    assert_eq!(text(&plain), "a");
}

#[test]
fn operators_get_their_canonical_names() {
    let a = || attr("a");
    let b = || attr("b");
    // Every spelling the table of `krishiv_plan::function` lists.
    for (sent, canonical) in [
        ("==", "=="),
        ("=", "=="),
        ("<=>", "<=>"),
        ("<", "<"),
        ("<=", "<="),
        (">", ">"),
        (">=", ">="),
        ("and", "and"),
        ("AND", "and"),
        ("Or", "or"),
        ("not", "not"),
        ("+", "+"),
        ("-", "-"),
        ("*", "*"),
        ("/", "/"),
        ("%", "%"),
        ("negative", "negative"),
        ("isNull", "isnull"),
        ("isNotNull", "isnotnull"),
        ("in", "in"),
        ("between", "between"),
        ("like", "like"),
        ("ILIKE", "ilike"),
        ("rlike", "rlike"),
        ("when", "when"),
        ("concat", "concat"),
    ] {
        let e = translate_expression(&func(sent, vec![a(), b()]), &conf()).unwrap();
        assert_eq!(function(&e).name, canonical, "{sent}");
        assert_eq!(function(&e).args.len(), 2);
    }
}

#[test]
fn not_equal_is_not_of_equal() {
    for spelling in ["!=", "<>"] {
        let e = translate_expression(&func(spelling, vec![attr("a"), attr("b")]), &conf()).unwrap();
        assert_eq!(function(&e).name, "not", "{spelling}");
        let [inner] = function(&e).args.as_slice() else {
            panic!("one argument")
        };
        assert_eq!(function(inner).name, "==");
        assert_eq!(e.to_string(), "(NOT (a == b))");
    }
}

#[test]
fn other_functions_keep_their_name_and_distinct_flag() {
    let sum = expr(ExprType::UnresolvedFunction(
        expression::UnresolvedFunction {
            function_name: "Sum".into(),
            arguments: vec![attr("c")],
            is_distinct: true,
            ..expression::UnresolvedFunction::default()
        },
    ));
    let e = translate_expression(&sum, &conf()).unwrap();
    let call = function(&e);
    assert_eq!(
        (call.name.as_str(), call.distinct, call.filter.is_none()),
        ("Sum", true, true)
    );
    assert_eq!(
        text(&func(
            "count",
            vec![expr(ExprType::UnresolvedStar(Default::default()))]
        )),
        "count(*)"
    );
    assert_eq!(text(&func("current_date", vec![])), "current_date()");
    // The distinct flag survives on an operator too.
    let distinct_eq = expr(ExprType::UnresolvedFunction(
        expression::UnresolvedFunction {
            function_name: "=".into(),
            arguments: vec![attr("a"), attr("b")],
            is_distinct: true,
            ..expression::UnresolvedFunction::default()
        },
    ));
    assert!(function(&translate_expression(&distinct_eq, &conf()).unwrap()).distinct);
}

#[test]
fn qualified_and_internal_functions_are_unsupported() {
    let mut udf = expression::UnresolvedFunction {
        function_name: "db.f".into(),
        ..expression::UnresolvedFunction::default()
    };
    udf.is_user_defined_function = true;
    let e = err(&expr(ExprType::UnresolvedFunction(udf)));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.message().contains("is_user_defined_function"),
        "{}",
        e.message()
    );
    let internal = expression::UnresolvedFunction {
        function_name: "product".into(),
        is_internal: Some(true),
        ..expression::UnresolvedFunction::default()
    };
    assert!(
        err(&expr(ExprType::UnresolvedFunction(internal)))
            .message()
            .contains("is_internal")
    );
    let not_internal = expression::UnresolvedFunction {
        function_name: "product".into(),
        is_internal: Some(false),
        ..expression::UnresolvedFunction::default()
    };
    assert_eq!(
        text(&expr(ExprType::UnresolvedFunction(not_internal))),
        "product()"
    );
    let call = expr(ExprType::CallFunction(Default::default()));
    assert!(err(&call).message().contains("CallFunction"));
}

fn alias(names: &[&str], metadata: Option<&str>) -> Expression {
    expr(ExprType::Alias(Box::new(expression::Alias {
        expr: Some(Box::new(attr("a"))),
        name: names.iter().map(|n| (*n).to_owned()).collect(),
        metadata: metadata.map(str::to_owned),
    })))
}

#[test]
fn aliases() {
    assert_eq!(text(&alias(&["x"], None)), "a AS x");
    assert_eq!(
        text(&alias(&["x"], Some(""))),
        "a AS x",
        "empty metadata is none"
    );
    let e = err(&alias(&["x"], Some("{\"k\":1}")));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let e = err(&alias(&["x", "y"], None));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.message().contains("several names"), "{}", e.message());
    let e = err(&alias(&[], None));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanFieldCannotBeEmpty
    );
    assert!(
        e.message().contains("name") && e.message().contains("Expression.Alias"),
        "{}",
        e.message()
    );
    let no_child = expr(ExprType::Alias(Box::new(expression::Alias {
        name: vec!["x".into()],
        ..expression::Alias::default()
    })));
    assert_eq!(
        err(&no_child).error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

fn cast(to: Option<CastToType>, mode: i32) -> Expression {
    expr(ExprType::Cast(Box::new(expression::Cast {
        expr: Some(Box::new(attr("a"))),
        eval_mode: mode,
        cast_to_type: to,
    })))
}

#[test]
fn casts_take_a_type_or_a_ddl_string() {
    let int = CastToType::Type(proto_type(dt::Kind::Integer(dt::Integer::default())));
    assert_eq!(text(&cast(Some(int), 0)), "CAST(a AS INT)");
    for (ddl, shown) in [
        ("int", "INT"),
        ("decimal(5)", "DECIMAL(5,0)"),
        ("TIMESTAMP_NTZ", "TIMESTAMP_NTZ"),
        ("  bigint ", "BIGINT"),
    ] {
        assert_eq!(
            text(&cast(Some(CastToType::TypeStr(ddl.into())), 0)),
            format!("CAST(a AS {shown})"),
            "{ddl}"
        );
    }
    let decimal = CastToType::Type(proto_type(dt::Kind::Decimal(dt::Decimal {
        precision: Some(10),
        scale: Some(2),
        ..dt::Decimal::default()
    })));
    assert_eq!(text(&cast(Some(decimal), 0)), "CAST(a AS DECIMAL(10,2))");
}

#[test]
fn cast_eval_modes() {
    let to = || Some(CastToType::TypeStr("int".into()));
    let mode = |m| {
        let e = translate_expression(&cast(to(), m), &conf()).unwrap();
        match &e.node {
            ExprNode::Cast { mode, .. } => *mode,
            other => panic!("not a cast: {other:?}"),
        }
    };
    assert_eq!(mode(0), CastMode::Default);
    assert_eq!(mode(1), CastMode::Legacy);
    assert_eq!(mode(2), CastMode::Ansi);
    assert_eq!(mode(3), CastMode::Try);
    assert_eq!(
        mode(99),
        CastMode::Default,
        "an unknown mode is the default, as in Spark"
    );
    assert_eq!(text(&cast(to(), 3)), "TRY_CAST(a AS INT)");
}

#[test]
fn cast_errors() {
    assert_eq!(
        err(&cast(None, 0)).error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    let e = err(&cast(Some(CastToType::TypeStr("array<int>".into())), 0));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let e = err(&cast(Some(CastToType::TypeStr("notatype".into())), 0));
    assert_eq!(e.error_class(), ErrorClass::UnsupportedDatatype);
    let e = err(&cast(Some(CastToType::TypeStr("".into())), 0));
    assert_eq!(e.error_class(), ErrorClass::ParseEmptyStatement);
    let e = err(&cast(
        Some(CastToType::Type(proto_type(dt::Kind::Char(
            dt::Char::default(),
        )))),
        0,
    ));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let no_input = expr(ExprType::Cast(Box::new(expression::Cast {
        cast_to_type: Some(CastToType::TypeStr("int".into())),
        ..expression::Cast::default()
    })));
    assert_eq!(
        err(&no_input).error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

fn star(target: Option<&str>, plan_id: Option<i64>) -> Expression {
    expr(ExprType::UnresolvedStar(expression::UnresolvedStar {
        unparsed_target: target.map(str::to_owned),
        plan_id,
    }))
}

#[test]
fn stars() {
    assert_eq!(text(&star(None, None)), "*");
    assert_eq!(text(&star(Some("t.*"), None)), "t.*");
    assert_eq!(text(&star(Some("`a.b`.c.*"), None)), "`a.b`.c.*");
    assert_eq!(text(&star(None, Some(4))), "*@plan4");
    assert_eq!(text(&star(None, Some(0))), "*@plan0");
    let e = err(&star(Some("t"), None));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanUnresolvedStarTargetInvalid
    );
    assert!(
        e.message().contains("'t'") || e.message().contains(" t"),
        "{}",
        e.message()
    );
    assert_eq!(
        err(&star(Some("*"), None)).error_class(),
        ErrorClass::ConnectInvalidPlanUnresolvedStarTargetInvalid
    );
    assert_eq!(
        err(&star(Some("t.*"), Some(1))).error_class(),
        ErrorClass::ConnectInvalidPlanUnresolvedStarWithBothTargetAndPlanId
    );
    assert_eq!(
        err(&star(Some("a..b.*"), None)).error_class(),
        ErrorClass::InvalidAttributeNameSyntax
    );
}

#[test]
fn sql_expression_strings_are_unsupported_for_now() {
    let sql = expr(ExprType::ExpressionString(expression::ExpressionString {
        expression: "a > 1".into(),
    }));
    let e = err(&sql);
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.message().contains("expression_string"), "{}", e.message());
    // The expression text is not echoed into the error.
    assert!(!e.message().contains("a > 1"));
}

#[test]
fn unsupported_expression_kinds_are_named() {
    for (kind, name) in [
        (
            ExprType::UnresolvedRegex(Default::default()),
            "unresolved_regex",
        ),
        (ExprType::SortOrder(Box::default()), "sort_order"),
        (ExprType::Window(Box::default()), "window"),
        (ExprType::LambdaFunction(Box::default()), "lambda_function"),
        (
            ExprType::UnresolvedExtractValue(Box::default()),
            "unresolved_extract_value",
        ),
        (
            ExprType::SubqueryExpression(Default::default()),
            "subquery_expression",
        ),
        (
            ExprType::NamedArgumentExpression(Box::default()),
            "named_argument_expression",
        ),
    ] {
        let e = err(&expr(kind));
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{name}"
        );
        assert!(
            e.message()
                .contains(&format!("spark.connect.Expression.{name}")),
            "{name}: {}",
            e.message()
        );
    }
}

#[test]
fn an_unset_expression_is_a_plan_error() {
    let e = err(&Expression::default());
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    assert!(e.message().contains("EXPR_TYPE_NOT_SET"), "{}", e.message());
    // A literal inside a function is translated, and an unset one fails it.
    let e = err(&func("f", vec![Expression::default()]));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

fn order(direction: i32, nulls: i32) -> expression::SortOrder {
    expression::SortOrder {
        child: Some(Box::new(attr("a"))),
        direction,
        null_ordering: nulls,
    }
}

#[test]
fn sort_orders_default_like_spark() {
    let conf = TranslateConf::default();
    let mut t = Translator::new(&conf);
    let mut show = |d, n| {
        let o = t.sort_order(&order(d, n)).unwrap();
        (o.direction, o.nulls)
    };
    assert_eq!(
        show(1, 1),
        (SortDirection::Ascending, NullOrdering::NullsFirst)
    );
    assert_eq!(
        show(1, 2),
        (SortDirection::Ascending, NullOrdering::NullsLast)
    );
    assert_eq!(
        show(2, 1),
        (SortDirection::Descending, NullOrdering::NullsFirst)
    );
    assert_eq!(
        show(2, 2),
        (SortDirection::Descending, NullOrdering::NullsLast)
    );
    // Spark: only ASCENDING is ascending and only NULLS_FIRST is nulls first.
    assert_eq!(
        show(0, 0),
        (SortDirection::Descending, NullOrdering::NullsLast)
    );
    assert_eq!(
        show(7, 7),
        (SortDirection::Descending, NullOrdering::NullsLast)
    );
    let no_child = expression::SortOrder::default();
    assert!(t.sort_order(&no_child).is_err());
}

#[test]
fn literals_are_translated_in_place() {
    assert_eq!(text(&lit(LiteralType::Integer(5))), "5");
    assert_eq!(text(&lit(LiteralType::String("it's".into()))), "'it\\'s'");
}

#[test]
fn deep_expressions_translate_on_a_small_stack() {
    // 900 nested calls (the IR allows 1000 levels) on a thread with 1 MiB:
    // the translator moves to a big stack after a few levels.
    let deep = nested_not(900);
    on_small_stack(1 << 20, move || {
        let e = translate_expression(&deep, &conf()).expect("translates");
        assert!(e.to_string().starts_with("(NOT (NOT"));
        drop(deep);
    });
}

#[test]
fn expressions_deeper_than_the_ir_are_too_complex() {
    for (depth, ok) in [(999, true), (1000, false)] {
        // `depth` calls and the literal at the bottom: depth + 1 levels.
        let deep = nested_not(depth);
        let result = std::thread::Builder::new()
            .stack_size(32 << 20)
            .spawn(move || {
                let r = translate_expression(&deep, &conf());
                drop(deep);
                r.map(|_| ())
            })
            .unwrap()
            .join()
            .unwrap();
        match (ok, result) {
            (true, Ok(())) => {}
            (false, Err(e)) => assert_eq!(e.error_class(), ErrorClass::FailedToParseTooComplex),
            (ok, r) => panic!("depth {depth}: expected ok={ok}, got {r:?}"),
        }
    }
}

/// `not(not(... not(true)))`, `calls` deep.
fn nested_not(calls: usize) -> Expression {
    let mut e = lit(LiteralType::Boolean(true));
    for _ in 0..calls {
        e = func("not", vec![e]);
    }
    e
}

/// Runs `f` on a thread with `stack` bytes.
fn on_small_stack(stack: usize, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap();
}
