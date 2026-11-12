// SPDX-License-Identifier: Apache-2.0
//! Tests of the session commands.

use std::collections::HashSet;

use futures::executor::block_on;
use krishiv_plan::SourceId;

use super::*;

/// The SQL source ids of the nodes of the stored view `name`.
fn view_sources(session: &Session, name: &str) -> HashSet<SourceId> {
    let entry = session.state.catalog.get(name, false).unwrap();
    let crate::catalog::TempTable::View { plan, .. } = &*entry else {
        panic!("{name} is not a view");
    };
    let mut ids = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    while let Some(plan) = stack.pop() {
        if let Some((source, _)) = plan.origin.sql {
            ids.insert(source);
        }
        plan.for_each_child(&mut |child| stack.push(Arc::clone(child)));
        plan.for_each_expr(&mut |expr| {
            if let krishiv_plan::ExprNode::Subquery { plan, .. } = &expr.node {
                stack.push(Arc::clone(plan));
            }
        });
    }
    ids
}

#[test]
fn views_lower_with_their_own_source_id() {
    // Every created view lowers with a source id of its own, so errors
    // raised while expanding a view point into the view's text, not the
    // outer statement's (`SourceId(0)`).
    let session = Session::new().unwrap();
    block_on(session.sql("CREATE TEMP VIEW v AS SELECT 1 AS a")).unwrap();
    block_on(session.sql("CREATE TEMP VIEW w AS SELECT 2 AS b")).unwrap();
    let (v, w) = (view_sources(&session, "v"), view_sources(&session, "w"));
    assert!(!v.is_empty() && !w.is_empty());
    assert!(!v.contains(&SourceId(0)), "{v:?}");
    assert!(!w.contains(&SourceId(0)), "{w:?}");
    assert!(v.is_disjoint(&w), "{v:?} vs {w:?}");
}

/// A window over `bound` as its frame start, for the expression walks.
fn window_with(bound: Expr) -> krishiv_sql::ast::WindowRef {
    use krishiv_sql::ast::{
        FrameBound, FrameBoundKind, FrameUnits, Span, WindowFrame, WindowRef, WindowSpec,
    };
    WindowRef::Spec(Box::new(WindowSpec {
        partitioning: None,
        partition_by: Vec::new(),
        order_by: Vec::new(),
        frame: Some(WindowFrame {
            units: FrameUnits::Rows,
            start: FrameBound {
                kind: FrameBoundKind::Preceding(bound),
                span: Span::default(),
            },
            end: None,
            span: Span::default(),
        }),
        span: Span::default(),
    }))
}

/// A `f(1)` call over `window`, for the expression walks.
fn function_over(window: krishiv_sql::ast::WindowRef) -> Expr {
    use krishiv_sql::ast::{Function, FunctionArg, Ident, ObjectName, Span};
    let ident = |value: &str| Ident {
        value: value.to_owned(),
        quoted: false,
        span: Span::default(),
    };
    Expr {
        kind: ExprKind::Function(Box::new(Function {
            name: ObjectName {
                parts: vec![ident("f")],
                span: Span::default(),
            },
            quantifier: None,
            args: vec![FunctionArg {
                name: None,
                expr: named_expr("1"),
                span: Span::default(),
            }],
            within_group: Vec::new(),
            filter: None,
            null_treatment: None,
            over: Some(window),
            dynamic_name: None,
            span: Span::default(),
        })),
        span: Span::default(),
    }
}

/// Parses `sql` as a named expression, for the expression walks.
fn named_expr(sql: &str) -> Expr {
    let options = krishiv_sql::ParserOptions::default();
    krishiv_sql::parse_named_expression(sql, &options)
        .unwrap()
        .expr
}

/// Variable names: shapes, quoting and normalization.
#[test]
fn variable_names_quote_and_normalize() {
    use krishiv_sql::ast::{Ident, Span};
    let name = |parts: &[&str]| ObjectName {
        parts: parts
            .iter()
            .map(|p| Ident {
                value: (*p).to_owned(),
                quoted: false,
                span: Span::default(),
            })
            .collect(),
        span: Span::default(),
    };
    assert_eq!(variable_key(&name(&["var1"])), Some("var1".to_owned()));
    assert_eq!(
        variable_key(&name(&["SESSION", "Var1"])),
        Some("var1".to_owned())
    );
    assert_eq!(
        variable_key(&name(&["system", "SESSION", "vAr1"])),
        Some("var1".to_owned())
    );
    assert_eq!(variable_key(&name(&["ses", "var1"])), None);
    assert_eq!(variable_key(&name(&["a", "b", "c", "d"])), None);
    assert_eq!(variable_key(&name(&[])), None);
    assert_eq!(quote_name(&name(&["vAr1"])), "`vAr1`");
    assert_eq!(quote_name(&name(&["s", "v"])), "`s`.`v`");
    assert_eq!(quote_normalized("var1"), "`system`.`session`.`var1`");
}

/// The parts of a plain `SET` assignment shaped like a variable.
#[test]
fn assign_variable_parts_shapes() {
    use krishiv_sql::ast::Span;
    let parts = |key: &str, quoted: bool| {
        let sql = if quoted {
            format!("SET `{key}` = 1")
        } else {
            format!("SET {key} = 1")
        };
        let text = krishiv_sql::redact_sql(&sql);
        let span = Span::new(4, 4 + key.len() + usize::from(quoted) * 2);
        assign_variable_parts(key, span, &text)
    };
    assert_eq!(parts("var1", false), Some(vec!["var1".to_owned()]));
    assert_eq!(
        parts("SESSION.var1", false),
        Some(vec!["SESSION".to_owned(), "var1".to_owned()])
    );
    assert_eq!(
        parts("system.session.var1", false).map(|p| p.len()),
        Some(3)
    );
    assert_eq!(parts("x.var1", false), None);
    assert_eq!(parts("x.session.var1", false), None);
    assert_eq!(parts("", false), None);
    // Backquoted, the same dots are one part.
    assert_eq!(parts("x.var1", true), Some(vec!["x.var1".to_owned()]));
    // A backquoted key is one part, whatever dots it holds.
    assert_eq!(
        parts("session.var1", true),
        Some(vec!["session.var1".to_owned()])
    );
}

/// Duplicate targets report the first repeat, lower-cased and quoted.
#[test]
fn duplicate_targets_reports_first_repeat() {
    assert!(duplicate_targets(&["a".to_owned(), "b".to_owned()]).is_ok());
    let e = duplicate_targets(&["a".to_owned(), "B".to_owned(), "a".to_owned()]).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::DuplicateAssignments);
    assert_eq!(e.message_parameters(), &[("nameList", "`a`".to_owned())]);
}

/// `padded_select` pads the gap so an expression keeps its offset.
#[test]
fn padded_select_aligns_slices() {
    assert_eq!(padded_select("SELECT ", "1", 7, ""), "SELECT 1");
    assert_eq!(
        padded_select("SELECT CAST(", "1", 45, " AS INT)"),
        format!("SELECT CAST({}1 AS INT)", " ".repeat(45 - 12))
    );
    // Before the prefix: no padding, no panic.
    assert_eq!(padded_select("SELECT CAST(", "1", 2, ")"), "SELECT CAST(1)");
}

/// The `DEFAULT` walks: a bare `DEFAULT`, one nested in an
/// expression, and subqueries.
#[test]
fn default_and_subquery_walks() {
    assert!(is_bare_default(&named_expr("DEFAULT")));
    assert!(is_bare_default(&named_expr("(DEFAULT)")));
    assert!(is_bare_default(&named_expr("default")));
    assert!(!is_bare_default(&named_expr("`default`")));
    assert!(!is_bare_default(&named_expr("session.default")));
    assert!(!is_bare_default(&named_expr("'Hello' || DEFAULT")));
    for sql in [
        "DEFAULT",
        "(DEFAULT)",
        "'Hello' || DEFAULT",
        "CASE WHEN TRUE THEN DEFAULT ELSE 1 END",
        "CAST(DEFAULT AS INT)",
        "upper(DEFAULT)",
    ] {
        assert!(contains_default(&named_expr(sql)), "{sql}");
    }
    for sql in ["1", "default2", "session.default", "(SELECT 1)"] {
        assert!(!contains_default(&named_expr(sql)), "{sql}");
    }
    // Through a window: the function, its filter, the ordering and
    // the frame all count.
    assert!(contains_default(&named_expr(
        "sum(x) OVER (ORDER BY DEFAULT)"
    )));
    assert!(contains_default(&function_over(window_with(named_expr(
        "DEFAULT"
    )))));
    assert!(!contains_default(&function_over(window_with(named_expr(
        "1"
    )))));
    for sql in [
        "(SELECT 1)",
        "1 + (SELECT 1)",
        "x IN (SELECT 1)",
        "EXISTS (SELECT 1)",
    ] {
        assert!(contains_subquery(&named_expr(sql)), "{sql}");
    }
    for sql in ["1", "DEFAULT", "'Hello' || DEFAULT"] {
        assert!(!contains_subquery(&named_expr(sql)), "{sql}");
    }
    // A subquery through a window frame counts too.
    assert!(contains_subquery(&function_over(window_with(named_expr(
        "(SELECT 1)"
    )))));
    assert!(!contains_subquery(&function_over(window_with(named_expr(
        "1"
    )))));
}

/// Execution errors from variable evaluation gain Spark's contexts.
#[test]
fn map_variable_error_attaches_contexts() {
    use krishiv_common::error::{CastInvalidInput, CastOverflow};
    use krishiv_sql::ast::Span;
    let text = krishiv_sql::redact_sql("DECLARE OR REPLACE VARIABLE var1 INT DEFAULT 'hello'");
    let command = Span::new(0, text.as_str().len());
    let expr = Span::new(45, 52);
    let at = |span: Span| {
        ErrorQueryContext::sql(
            text.fragment(span.start..span.end),
            span.start,
            span.end.saturating_sub(1),
        )
    };
    // A malformed assignment cast reports the command ...
    let e = map_variable_error(
        CastInvalidInput {
            expression: "'hello'".to_owned(),
            source_type: "\"STRING\"".to_owned(),
            target_type: "\"INT\"".to_owned(),
        }
        .into(),
        &text,
        command,
        expr,
    );
    assert_eq!(e.error_class(), ErrorClass::CastInvalidInput);
    assert_eq!(e.query_contexts(), &[at(command)]);
    // ... while an overflow keeps no context and anything else
    // reports the expression.
    let e = map_variable_error(
        CastOverflow {
            value: "1".to_owned(),
            source_type: "\"INT\"".to_owned(),
            target_type: "\"INT\"".to_owned(),
        }
        .into(),
        &text,
        command,
        expr,
    );
    assert!(e.query_contexts().is_empty());
    let e = map_variable_error(
        VariableNotFound {
            variable_name: "v".to_owned(),
        }
        .into(),
        &text,
        command,
        expr,
    );
    assert_eq!(e.query_contexts(), &[at(expr)]);
    // An error that already has a context keeps it.
    let e = map_variable_error(
        KrishivError::from(VariableNotFound {
            variable_name: "v".to_owned(),
        })
        .with_query_context(at(command)),
        &text,
        command,
        expr,
    );
    assert_eq!(e.query_contexts(), &[at(command)]);
    // `SET`, `DROP` and row forms resolve variables the same way.
    let name = |parts: &[&str]| ObjectName {
        parts: parts
            .iter()
            .map(|p| krishiv_sql::ast::Ident {
                value: (*p).to_owned(),
                quoted: false,
                span: Span::default(),
            })
            .collect(),
        span: Span::new(28, 32),
    };
    let e = unresolved_variable(&name(&["builtin", "var1"]), RESOLVE_SEARCH_PATH, &text);
    assert_eq!(e.error_class(), ErrorClass::UnresolvedVariable);
    assert_eq!(
        e.query_contexts(),
        &[ErrorQueryContext::sql("var1", 28, 31)]
    );
}

/// One-row expressions evaluate; anything else is an internal error.
#[test]
fn eval_variable_value_needs_one_row_one_column() {
    let session = Session::new().unwrap();
    let text = krishiv_common::redact::RedactedSql::opaque("SELECT 1");
    let span = krishiv_sql::ast::Span::new(0, 8);
    let eval = |sql: &str| block_on(session.eval_variable_value(sql, &text, span, span));
    assert_eq!(
        eval("SELECT 7").unwrap(),
        (DataType::Integer, Literal::Integer(7))
    );
    for sql in [
        "SELECT 1 UNION ALL SELECT 2",
        "SELECT 1 LIMIT 0",
        "SELECT 1, 2",
    ] {
        assert_eq!(
            eval(sql).unwrap_err().error_class(),
            ErrorClass::InternalError,
            "{sql}"
        );
    }
}
