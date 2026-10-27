// SPDX-License-Identifier: Apache-2.0
//! The analyzer over the function-call forms: named arguments,
//! qualified builtin names, `ILIKE`, and `LIKE|ILIKE ANY|SOME|ALL`.

use std::sync::Arc;

use krishiv_types::DataType;

use super::dsl::{c, int, named_call, qualified_call, s};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog,
    ResolutionConf, TableColumn,
};
use crate::{
    Builtin, Expr, ExprNode, FunctionCall, Literal, LogicalPlan, Origin, Plan, ResolvedExpr,
};

fn ctx() -> AnalysisContext {
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![
            TableColumn {
                name: "s".to_owned(),
                data_type: DataType::String,
                nullable: true,
            },
            TableColumn {
                name: "a".to_owned(),
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

fn names(r: &AnalyzedPlan) -> Vec<String> {
    r.output.iter().map(|c| c.name.clone()).collect()
}

fn class_and_params(r: &Result<AnalyzedPlan, AnalysisError>) -> (String, Vec<(String, String)>) {
    match r {
        Ok(_) => ("OK".to_owned(), Vec::new()),
        Err(e) => (
            e.error.error_class().name().to_owned(),
            e.error
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        ),
    }
}

fn named(name: &str, args: &[(Option<&str>, Expr)]) -> Expr {
    named_call(name, args)
}

#[test]
fn named_mask_reorders_and_fills_defaults() {
    let r = analyze(vec![named(
        "mask",
        &[
            (None, s("AbCD123-@$#")),
            (Some("digitChar"), s("d")),
            (Some("upperChar"), s("Q")),
            (Some("lowerChar"), s("q")),
        ],
    )])
    .unwrap();
    assert_eq!(names(&r), ["mask(AbCD123-@$#, Q, q, d, NULL)"]);
    let r = analyze(vec![named(
        "mask",
        &[
            (Some("str"), s("Ab")),
            (Some("upperChar"), s("Q")),
            (Some("lowerChar"), s("q")),
            (Some("digitChar"), s("d")),
            (Some("otherChar"), s("o")),
        ],
    )])
    .unwrap();
    assert_eq!(names(&r), ["mask(Ab, Q, q, d, o)"]);
    // Short positional calls expand with defaults too.
    let r = analyze(vec![named("mask", &[(None, s("Ab"))])]).unwrap();
    assert_eq!(names(&r), ["mask(Ab, X, x, n, NULL)"]);
}

#[test]
fn named_arguments_reject_the_wrong_calls() {
    for (expr, class) in [
        (
            named(
                "encode",
                &[(Some("str"), s("a")), (Some("charset"), s("b"))],
            ),
            "NAMED_PARAMETERS_NOT_SUPPORTED",
        ),
        (
            named("+", &[(Some("a"), int(1))]),
            "NAMED_PARAMETERS_NOT_SUPPORTED",
        ),
        (
            named(
                "mask",
                &[
                    (Some("lowerChar"), s("q")),
                    (None, s("Ab")),
                    (Some("upperChar"), s("Q")),
                ],
            ),
            "UNEXPECTED_POSITIONAL_ARGUMENT",
        ),
        (
            named(
                "mask",
                &[
                    (None, s("Ab")),
                    (Some("digitChar"), s("d")),
                    (Some("digitChar"), s("e")),
                ],
            ),
            "DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.DOUBLE_NAMED_ARGUMENT_REFERENCE",
        ),
        (
            named("mask", &[(None, s("Ab")), (Some("str"), s("Ac"))]),
            "DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.BOTH_POSITIONAL_AND_NAMED",
        ),
        (
            named("mask", &[(Some("cellular"), s("a"))]),
            "UNRECOGNIZED_PARAMETER_NAME",
        ),
        (
            named(
                "mask",
                &[(Some("upperChar"), s("Q")), (Some("lowerChar"), s("q"))],
            ),
            "REQUIRED_PARAMETER_NOT_FOUND",
        ),
    ] {
        let r = analyze(vec![expr]);
        assert_eq!(class_and_params(&r).0, class, "{r:?}");
    }
}

#[test]
fn named_error_parameters_match_spark() {
    let r = analyze(vec![named(
        "mask",
        &[
            (None, s("Ab")),
            (Some("lowerChar"), s("q")),
            (Some("upperChar"), s("Q")),
            (Some("otherChar"), s("o")),
            (Some("digitChar"), s("d")),
            (Some("cellular"), s("x")),
        ],
    )]);
    assert_eq!(
        class_and_params(&r),
        (
            "UNRECOGNIZED_PARAMETER_NAME".to_owned(),
            vec![
                ("routineName".to_owned(), "`mask`".to_owned()),
                ("argumentName".to_owned(), "`cellular`".to_owned()),
                (
                    "proposal".to_owned(),
                    "`str` `upperChar` `otherChar`".to_owned()
                ),
            ]
        )
    );
    let r = analyze(vec![named(
        "mask",
        &[(Some("lowerChar"), s("q")), (None, s("Ab"))],
    )]);
    assert_eq!(
        class_and_params(&r),
        (
            "UNEXPECTED_POSITIONAL_ARGUMENT".to_owned(),
            vec![
                ("routineName".to_owned(), "`mask`".to_owned()),
                ("parameterName".to_owned(), "`lowerChar`".to_owned()),
            ]
        )
    );
}

#[test]
fn unknown_named_calls_are_unresolved_routines() {
    let r = analyze(vec![named("nosuch", &[(Some("a"), int(1))])]);
    let (class, params) = class_and_params(&r);
    assert_eq!(class, "UNRESOLVED_ROUTINE");
    assert_eq!(
        params.iter().find(|(k, _)| k == "routineName"),
        Some(&("routineName".to_owned(), "`nosuch`".to_owned()))
    );
}

#[test]
fn named_arity_counts_required_to_total() {
    // Six positional arguments: more than the five `mask` takes.
    let r = analyze(vec![named(
        "mask",
        &[
            (None, s("a")),
            (None, s("b")),
            (None, s("c")),
            (None, s("d")),
            (None, s("e")),
            (None, s("f")),
        ],
    )]);
    assert_eq!(
        class_and_params(&r),
        (
            "WRONG_NUM_ARGS.WITHOUT_SUGGESTION".to_owned(),
            vec![
                ("functionName".to_owned(), "`mask`".to_owned()),
                ("expectedNum".to_owned(), "[1, 2, 3, 4, 5]".to_owned()),
                ("actualNum".to_owned(), "6".to_owned()),
                (
                    "docroot".to_owned(),
                    "https://spark.apache.org/docs/latest".to_owned()
                ),
            ]
        )
    );
    // A missing parameter past positional arguments cites its index among
    // the parameters after them.
    let r = analyze(vec![named(
        "count_min_sketch",
        &[
            (None, c("a")),
            (None, s("0.5")),
            (Some("confidence"), s("0.5")),
        ],
    )]);
    assert_eq!(
        class_and_params(&r),
        (
            "REQUIRED_PARAMETER_NOT_FOUND".to_owned(),
            vec![
                ("routineName".to_owned(), "`count_min_sketch`".to_owned()),
                ("parameterName".to_owned(), "`seed`".to_owned()),
                ("index".to_owned(), "1".to_owned()),
            ]
        )
    );
}

#[test]
fn count_min_sketch_reorders_then_fails_unsupported() {
    // The signature is known, so named arguments reorder; the function
    // itself is unimplemented.
    let r = analyze(vec![named(
        "count_min_sketch",
        &[
            (Some("seed"), int(1)),
            (Some("column"), c("a")),
            (Some("epsilon"), s("0.5")),
            (Some("confidence"), s("0.5")),
        ],
    )]);
    let (class, _) = class_and_params(&r);
    assert_eq!(class, "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn mask_replacement_checks_quote_the_call() {
    // A multi-character replacement.
    let r = analyze(vec![named(
        "mask",
        &[(None, s("Ab")), (Some("upperChar"), s("QQ"))],
    )]);
    assert_eq!(
        class_and_params(&r).0,
        "DATATYPE_MISMATCH.INPUT_SIZE_NOT_ONE"
    );
    // A non-foldable replacement.
    let r = analyze(vec![named(
        "mask",
        &[(None, s("Ab")), (Some("upperChar"), c("s"))],
    )]);
    assert_eq!(
        class_and_params(&r).0,
        "DATATYPE_MISMATCH.NON_FOLDABLE_INPUT"
    );
    // A replacement that fails while folding (not a string).
    let bad = Expr::new(crate::ExprNode::Cast {
        expr: Arc::new(s("x")),
        to: DataType::Integer,
        mode: crate::CastMode::Default,
    });
    let r = analyze(vec![named(
        "mask",
        &[(None, s("Ab")), (Some("upperChar"), bad)],
    )]);
    let (class, _) = class_and_params(&r);
    assert_ne!(class, "OK", "{r:?}");
}

#[test]
fn ilike_rejects_other_arities() {
    for n in [1, 4] {
        let args = (0..n).map(|_| s("a")).collect::<Vec<_>>();
        let r = analyze(vec![Expr::builtin(Builtin::ILike, args)]);
        assert_eq!(class_and_params(&r).0, "WRONG_NUM_ARGS.WITHOUT_SUGGESTION");
    }
}

#[test]
fn qualified_builtins_resolve() {
    for qualifier in [vec!["builtin"], vec!["system", "builtin"], vec!["BUILTIN"]] {
        let q: Vec<&str> = qualifier;
        let r = analyze(vec![qualified_call(&q, "abs", vec![int(-1)])]).unwrap();
        assert_eq!(names(&r), ["abs(-1)"], "{q:?}");
    }
    let r = analyze(vec![qualified_call(
        &["system", "builtin"],
        "mask",
        vec![s("Ab")],
    )])
    .unwrap();
    assert_eq!(names(&r), ["mask(Ab, X, x, n, NULL)"]);
}

#[test]
fn other_qualifiers_are_unresolved_routines() {
    for qualifier in [vec!["db"], vec!["a", "b"], vec!["system", "session"]] {
        let q: Vec<&str> = qualifier;
        let r = analyze(vec![qualified_call(&q, "abs", vec![int(-1)])]);
        let (class, params) = class_and_params(&r);
        assert_eq!(class, "UNRESOLVED_ROUTINE", "{q:?}");
        assert!(
            params
                .iter()
                .any(|(k, v)| k == "searchPath" && v.contains("system")),
            "{params:?}"
        );
    }
}

#[test]
fn ilike_rewrites_to_lower_like_with_its_name() {
    let r = analyze(vec![Expr::builtin(Builtin::ILike, vec![c("s"), s("A%")])]).unwrap();
    assert_eq!(names(&r), ["ilike(s, A%)"]);
    // The rewritten call evaluates case-insensitively; `rlike` is untouched.
    let r = analyze(vec![Expr::builtin(Builtin::RLike, vec![c("s"), s("A")])]).unwrap();
    assert_eq!(names(&r), ["rlike(s, A)"]);
}

#[test]
fn ilike_with_escape_keeps_the_escape_but_hides_it() {
    let call = Expr::new(ExprNode::Function(FunctionCall {
        within_group: Vec::new(),
        name: "ilike".to_owned(),
        qualifier: Vec::new(),
        args: vec![Arc::new(c("s")), Arc::new(s("a%b")), Arc::new(s("#"))],
        arg_names: Vec::new(),
        distinct: false,
        filter: None,
    }));
    let r = analyze(vec![call]).unwrap();
    assert_eq!(names(&r), ["ilike(s, a%b)"]);
}

/// Spark 4.2's names (checked on Spark): literal patterns build its
/// `LikeAny` family, `likeany(child)` with the child lowered for ILIKE;
/// any other pattern (here `NULL` or a column) builds the chain of single
/// `LIKE`s or `ilike` calls.
#[test]
fn like_quantified_rewrites_to_chains_with_its_name() {
    for (which, literal, chain) in [
        (Builtin::LikeAny, "likeany(s)", "(s LIKE a% OR s LIKE NULL)"),
        (
            Builtin::LikeAll,
            "likeall(s)",
            "(s LIKE a% AND s LIKE NULL)",
        ),
        (
            Builtin::NotLikeAny,
            "notlikeany(s)",
            "((NOT s LIKE a%) OR (NOT s LIKE NULL))",
        ),
        (
            Builtin::NotLikeAll,
            "notlikeall(s)",
            "((NOT s LIKE a%) AND (NOT s LIKE NULL))",
        ),
        (
            Builtin::ILikeAny,
            "likeany(lower(s))",
            "(ilike(s, a%) OR ilike(s, NULL))",
        ),
        (
            Builtin::ILikeAll,
            "likeall(lower(s))",
            "(ilike(s, a%) AND ilike(s, NULL))",
        ),
        (
            Builtin::NotILikeAny,
            "notlikeany(lower(s))",
            "((NOT ilike(s, a%)) OR (NOT ilike(s, NULL)))",
        ),
        (
            Builtin::NotILikeAll,
            "notlikeall(lower(s))",
            "((NOT ilike(s, a%)) AND (NOT ilike(s, NULL)))",
        ),
    ] {
        let r = analyze(vec![Expr::builtin(which, vec![c("s"), s("a%"), s("%b")])]).unwrap();
        assert_eq!(names(&r), [literal]);
        let null = Expr::new(ExprNode::Literal(Literal::Null(None)));
        let r = analyze(vec![Expr::builtin(which, vec![c("s"), s("a%"), null])]).unwrap();
        assert_eq!(names(&r), [chain]);
    }
    // One pattern is still the `LikeAll` form.
    let r = analyze(vec![Expr::builtin(Builtin::LikeAll, vec![c("s"), s("a%")])]).unwrap();
    assert_eq!(names(&r), ["likeall(s)"]);
}

#[test]
fn quantified_chains_are_three_valued() {
    use crate::{PlanNode, ResolvedFunction};
    // The rewrite is an OR/AND of single LIKE calls, so a NULL pattern makes
    // the result NULL rather than false: the resolved form is `or`, not a
    // single call.
    let r = analyze(vec![Expr::builtin(
        Builtin::LikeAny,
        vec![
            c("s"),
            s("a%"),
            Expr::new(ExprNode::Literal(Literal::Null(None))),
        ],
    )])
    .unwrap();
    let PlanNode::Project { exprs, .. } = &r.plan.root().node else {
        panic!("not a projection");
    };
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &exprs[0].node else {
        panic!("not resolved: {:?}", exprs[0].node);
    };
    let ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction { id, .. })) = &expr.node else {
        panic!("not a call: {:?}", expr.node);
    };
    assert_eq!(id, &crate::FunctionId::Builtin(Builtin::Or));
}
