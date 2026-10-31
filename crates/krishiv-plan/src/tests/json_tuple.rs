// SPDX-License-Identifier: Apache-2.0
//! `json_tuple` in select lists: the analyzer expands the generator into
//! N output columns over one shared struct computation (Spark's
//! `ExtractGenerator` for the one generator this engine implements).
//!
//! SQL-statement coverage (golden schemas, rows and error classes) runs
//! through sessions in `krishiv-session`; here the plans are hand-built.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_types::DataType;

use super::dsl::{c, f, int, s, star};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog,
    ResolutionConf, TableColumn,
};
use crate::{
    Builtin, Expr, ExprNode, FunctionCall, FunctionKind, Literal, LogicalPlan, Origin, Plan,
    PlanNode, ResolvedExpr,
};

fn ctx() -> AnalysisContext {
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![
            TableColumn {
                name: "j".to_owned(),
                data_type: DataType::String,
                nullable: true,
            },
            TableColumn {
                name: "k".to_owned(),
                data_type: DataType::String,
                nullable: true,
            },
            TableColumn {
                name: "i".to_owned(),
                data_type: DataType::Integer,
                nullable: false,
            },
        ],
        None,
    );
    context(catalog, ResolutionConf::default())
}

fn analyze(exprs: Vec<Expr>) -> Result<AnalyzedPlan, AnalysisError> {
    let plan =
        Plan::try_new(LogicalPlan::table(["t"]).project(exprs)).map_err(|error| AnalysisError {
            error,
            origin: Origin::UNKNOWN,
        })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx()))
}

/// Output names, types and nullability.
fn schema(r: &AnalyzedPlan) -> Vec<(String, String, bool)> {
    r.output
        .iter()
        .map(|c| (c.name.clone(), c.data_type.simple_string(), c.nullable))
        .collect()
}

fn class_and_params(r: &Result<AnalyzedPlan, AnalysisError>) -> (String, Vec<(String, String)>) {
    match r {
        Ok(_) => ("OK".to_owned(), Vec::new()),
        Err(e) => {
            let mut params: Vec<(String, String)> = e
                .error
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect();
            params.sort();
            (e.error.error_class().name().to_owned(), params)
        }
    }
}

fn call(args: Vec<Expr>) -> Expr {
    f("json_tuple", args)
}

fn null() -> Expr {
    Expr::new(ExprNode::Literal(Literal::Null(None)))
}

#[test]
fn output_names_are_c0_to_cn_and_all_nullable_strings() {
    let r = analyze(vec![call(vec![c("j"), s("a")])]).unwrap();
    assert_eq!(schema(&r), [("c0".to_owned(), "string".to_owned(), true)]);
    let r = analyze(vec![call(vec![c("j"), s("a"), s("b"), c("k")])]).unwrap();
    assert_eq!(
        schema(&r),
        [
            ("c0".to_owned(), "string".to_owned(), true),
            ("c1".to_owned(), "string".to_owned(), true),
            ("c2".to_owned(), "string".to_owned(), true),
        ]
    );
}

#[test]
fn mixed_select_lists_keep_their_columns() {
    let r = analyze(vec![c("i"), call(vec![c("j"), s("a")])]).unwrap();
    assert_eq!(
        schema(&r),
        [
            ("i".to_owned(), "int".to_owned(), false),
            ("c0".to_owned(), "string".to_owned(), true),
        ]
    );
    // A star beside the generator expands over the input columns.
    let r = analyze(vec![star(), call(vec![c("j"), s("a"), s("b")])]).unwrap();
    assert_eq!(
        schema(&r),
        [
            ("j".to_owned(), "string".to_owned(), true),
            ("k".to_owned(), "string".to_owned(), true),
            ("i".to_owned(), "int".to_owned(), false),
            ("c0".to_owned(), "string".to_owned(), true),
            ("c1".to_owned(), "string".to_owned(), true),
        ]
    );
}

#[test]
fn non_foldable_field_names_are_allowed() {
    // `if(i < 1, null, 'a')`, as the golden's `if(c1 < 1, null, 'a')`:
    // typed STRING, resolved per row.
    let cond = f(
        "if",
        vec![
            Expr::builtin(Builtin::Lt, vec![c("i"), int(1)]),
            null(),
            s("a"),
        ],
    );
    let r = analyze(vec![call(vec![c("j"), cond])]).unwrap();
    assert_eq!(schema(&r), [("c0".to_owned(), "string".to_owned(), true)]);
}

#[test]
fn single_alias_renames_the_one_output() {
    let one = Expr::new(ExprNode::Alias {
        expr: Arc::new(call(vec![c("j"), s("a")])),
        name: "x".to_owned(),
    });
    let r = analyze(vec![one]).unwrap();
    assert_eq!(schema(&r), [("x".to_owned(), "string".to_owned(), true)]);
}

#[test]
fn single_alias_on_several_outputs_is_udtf_alias_number_mismatch() {
    let aliased = Expr::new(ExprNode::Alias {
        expr: Arc::new(call(vec![c("j"), s("a"), s("b")])),
        name: "x".to_owned(),
    });
    let r = analyze(vec![aliased]);
    let (class, params) = class_and_params(&r);
    assert_eq!(class, "UDTF_ALIAS_NUMBER_MISMATCH");
    assert_eq!(
        params,
        [
            ("aliasesNames".to_owned(), "x".to_owned()),
            ("aliasesSize".to_owned(), "2".to_owned()),
        ]
    );
}

/// Spark 4.2 runs two `json_tuple` generators in one plain select list (a
/// cross product); the engine does not, and says so with its own class
/// rather than a Spark one Spark would not raise.
#[test]
fn two_generators_are_unsupported() {
    let r = analyze(vec![call(vec![c("j"), s("a")]), call(vec![c("j"), s("b")])]);
    let (class, params) = class_and_params(&r);
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
    assert_eq!(params[0].1, "several generators in one SELECT list");
}

#[test]
fn nested_generator_raises_nested_in_expressions() {
    // `if` takes any branch type, so the nested generator survives typing
    // to the generator check (a wrongly-typed parent like `concat` fails
    // typing first, as there is no lazy generator extraction here).
    let nested = f(
        "if",
        vec![
            Expr::builtin(Builtin::Gt, vec![c("i"), int(1)]),
            call(vec![c("j"), s("a")]),
            call(vec![c("j"), s("b")]),
        ],
    );
    let r = analyze(vec![nested]);
    let (class, params) = class_and_params(&r);
    assert_eq!(class, "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS");
    let expression = params.iter().find(|(k, _)| k == "expression").unwrap();
    assert!(expression.1.contains("json_tuple"), "{expression:?}");
    assert_eq!(expression.1.matches("json_tuple").count(), 2);
}

#[test]
fn arity_errors_match_spark() {
    for args in [vec![], vec![c("j")]] {
        let r = analyze(vec![call(args)]);
        let (class, params) = class_and_params(&r);
        assert_eq!(class, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
        assert!(
            params.iter().any(|(k, v)| k == "expectedNum" && v == "> 1"),
            "{params:?}"
        );
        assert!(
            params
                .iter()
                .any(|(k, v)| k == "functionName" && v == "`json_tuple`"),
            "{params:?}"
        );
    }
}

#[test]
fn non_string_arguments_raise_non_string_type() {
    // An integer field, an untyped NULL field, and a non-string JSON text.
    for args in [
        vec![c("j"), c("i")],
        vec![c("j"), null()],
        vec![c("i"), s("a")],
    ] {
        let r = analyze(vec![call(args)]);
        let (class, params) = class_and_params(&r);
        assert_eq!(class, "DATATYPE_MISMATCH.NON_STRING_TYPE");
        assert!(
            params
                .iter()
                .any(|(k, v)| k == "funcName" && v == "`json_tuple`"),
            "{params:?}"
        );
    }
}

/// Counts of struct computations, field reads, projections and leftover
/// generator calls in the resolved plan: one scalar struct computation
/// shared by N field accesses, over two projections, and no generator.
fn shape(r: &AnalyzedPlan) -> (usize, usize, usize, usize) {
    use crate::walk::{Node, visit_distinct};
    use std::ops::ControlFlow;
    let mut computes = 0;
    let mut fields = 0;
    let mut projects = 0;
    let mut generators = 0;
    let _ = visit_distinct(Node::Plan(r.plan.root()), |node| {
        match node {
            Node::Plan(p) => {
                if matches!(&p.node, PlanNode::Project { .. }) {
                    projects += 1;
                }
            }
            Node::Expr(e) => match &e.node {
                ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                    if f.kind == FunctionKind::Scalar
                        && matches!(&f.id, crate::FunctionId::Registry { name, .. } if name == "json_tuple")
                    {
                        computes += 1;
                    }
                    if f.kind == FunctionKind::Generator {
                        generators += 1;
                    }
                }
                ExprNode::Resolved(ResolvedExpr::Field { .. }) => fields += 1,
                _ => {}
            },
        }
        ControlFlow::Continue(())
    });
    (computes, fields, projects, generators)
}

#[test]
fn plan_computes_the_struct_once_and_reads_n_fields() {
    let r = analyze(vec![call(vec![c("j"), s("a"), s("b"), s("c")])]).unwrap();
    assert_eq!(shape(&r), (1, 3, 2, 0));
    // The three field reads share the one computed attribute.
    let PlanNode::Project { exprs, .. } = &r.plan.root().node else {
        panic!("the select list is a projection");
    };
    let bases: HashSet<_> = exprs
        .iter()
        .map(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => match &expr.node {
                ExprNode::Resolved(ResolvedExpr::Field { base, .. }) => match &base.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => c.id,
                    other => panic!("a field reads a column, not {other:?}"),
                },
                other => panic!("a select item aliases a field read, not {other:?}"),
            },
            other => panic!("a select item is aliased, not {other:?}"),
        })
        .collect();
    assert_eq!(bases.len(), 1);
}

/// `SELECT` over an aggregate with a generator stays unsupported: the
/// engine has no row-multiplying `Generate` node, and the one-row rewrite
/// only covers plain select lists.
#[test]
fn generator_with_aggregates_stays_unsupported() {
    let agg = Expr::new(ExprNode::Function(FunctionCall {
        within_group: Vec::new(),
        name: "count".to_owned(),
        qualifier: Vec::new(),
        args: vec![Arc::new(c("i"))],
        arg_names: vec![None],
        distinct: false,
        filter: None,
    }));
    let r = analyze(vec![call(vec![c("j"), s("a")]), agg]);
    assert_eq!(class_and_params(&r).0, "KRISHIV_UNSUPPORTED_FEATURE");
}
