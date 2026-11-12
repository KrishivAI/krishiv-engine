// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// `UNRESOLVED_COLUMN` proposals rank quoted candidates by edit distance
/// to the unquoted name, ties in name order (Spark's
/// `orderSuggestedIdentifiersBySimilarity`, checked against Spark 4.2.0).
#[test]
fn unresolved_column_proposals_rank_like_spark() {
    let renamed = LogicalPlan::table(["wide"]).project(vec![
        c("col1").alias("country"),
        c("col2").alias("city"),
        c("col3").alias("name"),
        c("col4").alias("id"),
        c("col5").alias("power"),
    ]);
    let r = run(renamed.project(vec![c("non_exist")]), &ctx());
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            {
                "objectName": "`non_exist`",
                "proposal": "`id`, `name`, `power`, `city`, `country`",
            }
        ])
    );
    let renamed = LogicalPlan::table(["t"]).project(vec![
        c("a").alias("id"),
        c("b").alias("name"),
        c("c").alias("data"),
    ]);
    let r = run(renamed.project(vec![c("invalid_column")]), &ctx());
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            {
                "objectName": "`invalid_column`",
                "proposal": "`id`, `name`, `data`",
            }
        ])
    );
}

/// A single row is just the `Project`: names, types and nullability come
/// from the row, and no `InlineTable` node remains.
#[test]
fn inline_table_of_one_row_is_a_project() {
    let ctx = ctx();
    let r = run(
        inline_table(vec![vec![s("one"), int(1)]], &["col1", "col2"]),
        &ctx,
    );
    assert_eq!(
        summary(&r),
        serde_json::json!([["col1", false], ["col2", false]])
    );
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    assert!(!has_inline_table(a.plan.root()));
    assert!(
        matches!(a.plan.root().node, PlanNode::Project { .. }),
        "{}",
        a.plan.root()
    );
    assert_eq!(datatypes(&a), [DataType::String, DataType::Integer]);
}

/// Several rows are one `Expand` over `OneRowRelation` with a projection
/// per row, in row order; each column widens to the wider type and is
/// nullable where any row's value is.
#[test]
fn inline_table_rows_widen_and_nullable() {
    let ctx = ctx();
    let r = run(
        inline_table(
            vec![
                vec![s("one"), int(1)],
                vec![s("two"), long(2)],
                vec![s("three"), null()],
            ],
            &["a", "b"],
        ),
        &ctx,
    );
    assert_eq!(summary(&r), serde_json::json!([["a", false], ["b", true]]));
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    assert!(!has_inline_table(a.plan.root()));
    assert_eq!(datatypes(&a), [DataType::String, DataType::Long]);
    let PlanNode::Expand {
        input,
        projections,
        output,
    } = &a.plan.root().node
    else {
        panic!("an expand of row projections:\n{}", a.plan.root());
    };
    assert!(matches!(input.node, PlanNode::OneRowRelation));
    assert_eq!(projections.len(), 3);
    assert_eq!(output.len(), 2);
    for row in projections {
        assert_eq!(row.len(), 2);
    }
}

/// An all-NULL column is `VOID`.
#[test]
fn inline_table_of_nulls_is_void() {
    let ctx = ctx();
    let r = run(
        inline_table(
            vec![vec![s("one"), null()], vec![s("two"), null()]],
            &["a", "b"],
        ),
        &ctx,
    );
    assert_eq!(summary(&r), serde_json::json!([["a", false], ["b", true]]));
    assert_eq!(datatypes(&r.unwrap()), [DataType::String, DataType::Null]);
}

/// A column whose values share no type names it, quoted, in
/// `INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE` (`INT` meets
/// `STRING` at `BIGINT`, so only the second column fails here).
#[test]
fn inline_table_of_incompatible_types_fails() {
    let ctx = ctx();
    let r = run(
        inline_table(
            vec![vec![int(1), int(2)], vec![s("x"), boolean(true)]],
            &["a", "b"],
        ),
        &ctx,
    );
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE",
            {"colName": "`b`"}
        ])
    );
}

/// An aggregate, a nondeterministic call, a window and a subquery cannot
/// be evaluated in an inline table.
#[test]
fn inline_table_of_unevaluable_expressions_fails() {
    let ctx = ctx();
    for (cell, expr) in [
        (f("count", vec![int(1)]), "\"count(1)\""),
        (f("rand", vec![int(5)]), "\"rand(5)\""),
    ] {
        let r = run(inline_table(vec![vec![cell]], &["a"]), &ctx);
        assert_eq!(
            summary(&r),
            serde_json::json!([
                "INVALID_INLINE_TABLE.CANNOT_EVALUATE_EXPRESSION_IN_INLINE_TABLE",
                {"expr": expr}
            ]),
            "{expr}"
        );
    }
    // An unknown routine is `UNRESOLVED_ROUTINE`, as elsewhere.
    let r = run(
        inline_table(vec![vec![f("nosuchfunc", vec![int(1)])]], &["a"]),
        &ctx,
    );
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_ROUTINE",
            {
                "routineName": "`nosuchfunc`",
                "searchPath": "[`system`.`builtin`, `system`.`session`, `spark_catalog`.`default`]"
            }
        ])
    );
    // A window and a subquery cannot be evaluated either.
    let window = Expr::new(ExprNode::Window(crate::WindowExpr {
        function: Arc::new(f("row_number", Vec::new())),
        window: crate::WindowSpecRef::Spec(crate::WindowSpec {
            partition_by: Vec::new(),
            order_by: Vec::new(),
            frame: None,
        }),
    }));
    let r = run(inline_table(vec![vec![window]], &["a"]), &ctx);
    assert_eq!(
        class(&r),
        "INVALID_INLINE_TABLE.CANNOT_EVALUATE_EXPRESSION_IN_INLINE_TABLE"
    );
    let subquery = Expr::new(ExprNode::Subquery {
        kind: crate::SubqueryKind::Scalar,
        plan: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
    });
    let r = run(inline_table(vec![vec![subquery]], &["a"]), &ctx);
    assert_eq!(
        class(&r),
        "INVALID_INLINE_TABLE.CANNOT_EVALUATE_EXPRESSION_IN_INLINE_TABLE"
    );
}

/// A cell that raises when Spark evaluates it while resolving the table is
/// `FAILED_SQL_EXPRESSION_EVALUATION` (ANSI: `CAST('nan' AS INT)` fails;
/// without ANSI it is NULL and the table resolves).
#[test]
fn inline_table_cell_that_fails_to_evaluate() {
    let bad = || {
        Expr::new(ExprNode::Cast {
            expr: Arc::new(s("nan")),
            to: DataType::Integer,
            mode: crate::CastMode::Ansi,
        })
    };
    let mut on = ctx();
    on.conf.ansi = true;
    let r = run(inline_table(vec![vec![int(1)], vec![bad()]], &["a"]), &on);
    assert_eq!(
        class(&r),
        "INVALID_INLINE_TABLE.FAILED_SQL_EXPRESSION_EVALUATION"
    );
    // The cast-named function form fails the same way.
    let named = f("boolean", vec![s("XXX")]);
    let r = run(inline_table(vec![vec![named]], &["a"]), &on);
    assert_eq!(
        class(&r),
        "INVALID_INLINE_TABLE.FAILED_SQL_EXPRESSION_EVALUATION"
    );
    let try_cast = Expr::new(ExprNode::Cast {
        expr: Arc::new(s("nan")),
        to: DataType::Integer,
        mode: crate::CastMode::Try,
    });
    let r = run(inline_table(vec![vec![try_cast]], &["a"]), &on);
    assert_eq!(class(&r), "OK");
}

/// The clock functions evaluate once per query, so they may stay.
#[test]
fn inline_table_of_clock_calls_resolves() {
    let ctx = ctx();
    let r = run(
        inline_table(vec![vec![f("now", Vec::new())]], &["ct"]),
        &ctx,
    );
    assert_eq!(summary(&r), serde_json::json!([["ct", false]]));
    let a = r.unwrap();
    assert_eq!(datatypes(&a), [DataType::Timestamp]);
    assert!(!has_inline_table(a.plan.root()));
}

/// Degenerate hand-built tables (the parser rejects these): no rows is an
/// error.
#[test]
fn inline_table_edge_cases() {
    let ctx = ctx();
    let r = run(
        LogicalPlan::new(PlanNode::InlineTable {
            rows: Vec::new(),
            columns: Vec::new(),
        }),
        &ctx,
    );
    assert_eq!(class(&r), "KRISHIV_UNSUPPORTED_FEATURE");
    // Rewriting an inline table's expressions visits every cell (and an
    // unchanged rewrite shares the plan).
    let p = Arc::new(inline_table(vec![vec![int(1)]], &["a"]));
    let q = Arc::clone(&p)
        .map_expressions(|e| Ok::<_, ()>(Arc::clone(&e)))
        .expect("an infallible rewrite");
    assert!(Arc::ptr_eq(&p, &q));
}

/// A column name resolves against nothing in an inline table: it is
/// `UNRESOLVED_COLUMN`, as in Spark (not `CANNOT_EVALUATE_...`), also
/// inside a larger expression.
#[test]
fn inline_table_of_unknown_column_is_unresolved() {
    let ctx = ctx();
    for cell in [c("x"), add(c("x"), int(1))] {
        let r = run(inline_table(vec![vec![int(0)], vec![cell]], &["a"]), &ctx);
        assert_eq!(
            summary(&r),
            serde_json::json!(["UNRESOLVED_COLUMN.WITHOUT_SUGGESTION", {"objectName": "`x`"}])
        );
    }
}

/// A large inline table analyzes in time linear in its rows: ten thousand
/// rows become one `Expand` with ten thousand projections, not a deep tree.
#[test]
fn inline_table_of_many_rows_resolves() {
    let ctx = ctx();
    let rows: Vec<Vec<Expr>> = (0..10_000).map(|i| vec![int(i), s("v")]).collect();
    let r = run(inline_table(rows, &["a", "b"]), &ctx);
    assert_eq!(summary(&r), serde_json::json!([["a", false], ["b", false]]));
    let a = r.unwrap();
    assert!(a.plan.root().is_resolved());
    assert!(!has_inline_table(a.plan.root()));
    let PlanNode::Expand { projections, .. } = &a.plan.root().node else {
        panic!("one expand:\n{}", a.plan.root());
    };
    assert_eq!(projections.len(), 10_000);
}

/// A hand-built row of the wrong width fails rather than NULL-fills or
/// truncates (the parser rejects these).
#[test]
fn inline_table_of_ragged_rows_fails() {
    let ctx = ctx();
    for rows in [vec![vec![int(1)], vec![]], vec![vec![int(1), int(2)]]] {
        let r = run(inline_table(rows, &["a"]), &ctx);
        assert_eq!(class(&r), "KRISHIV_UNSUPPORTED_FEATURE");
    }
}

/// A DataFrame reference (`df["a"]`) names a column no input has.
#[test]
fn inline_table_of_dataframe_reference_fails() {
    let ctx = ctx();
    let cell = Expr::new(ExprNode::Column(ColumnRef {
        plan_id: Some(PlanId(1)),
        ..ColumnRef::new(vec!["a".to_owned()])
    }));
    let r = run(inline_table(vec![vec![cell]], &["a"]), &ctx);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "CANNOT_RESOLVE_DATAFRAME_COLUMN",
            {"name": "\"a\""}
        ])
    );
}

/// `spark.sql.legacy.ctePrecedencePolicy` and the two `count` flags flow
/// from the session configs, with Spark 4.2's defaults.
#[test]
fn legacy_flag_confs_follow_session_confs() {
    use crate::analyzer::CtePrecedencePolicy;
    let conf = |pairs: &[(&str, &str)]| {
        ResolutionConf::from_conf(|k| pairs.iter().find(|(key, _)| *key == k).map(|(_, v)| *v))
    };
    let d = ResolutionConf::default();
    assert_eq!(d.cte_precedence, CtePrecedencePolicy::Corrected);
    assert!(!d.allow_parameterless_count);
    assert!(!d.allow_star_with_single_table_identifier_in_count);
    assert_eq!(conf(&[]), d);
    let all = conf(&[
        ("spark.sql.legacy.ctePrecedencePolicy", " LEGACY "),
        ("spark.sql.legacy.allowParameterlessCount", "true"),
        (
            "spark.sql.legacy.allowStarWithSingleTableIdentifierInCount",
            "TRUE",
        ),
    ]);
    assert_eq!(all.cte_precedence, CtePrecedencePolicy::Legacy);
    assert!(all.allow_parameterless_count);
    assert!(all.allow_star_with_single_table_identifier_in_count);
    let exc = conf(&[("spark.sql.legacy.ctePrecedencePolicy", "exception")]);
    assert_eq!(exc.cte_precedence, CtePrecedencePolicy::Exception);
    // Unknown policies are rejected by the config registry
    // (`KRISHIV_INVALID_CONFIG_VALUE`); `from_value` reports them as `None`
    // rather than silently falling back, and `from_conf` (which reads an
    // already-validated config in the session path) maps them to the
    // default. The session path never sees one.
    let odd = conf(&[("spark.sql.legacy.ctePrecedencePolicy", "sometimes")]);
    assert_eq!(odd.cte_precedence, CtePrecedencePolicy::Corrected);
    assert_eq!(CtePrecedencePolicy::from_value("sometimes"), None);
    assert_eq!(
        CtePrecedencePolicy::from_value("corrected"),
        Some(CtePrecedencePolicy::Corrected)
    );
    assert_eq!(
        CtePrecedencePolicy::from_value(" LEGACY "),
        Some(CtePrecedencePolicy::Legacy)
    );
    // A non-`true` `allowParameterlessCount` value reads as off (these
    // booleans are not registry entries, so `SET` does not validate them).
    let odd_bool = conf(&[("spark.sql.legacy.allowParameterlessCount", "sometimes")]);
    assert!(!odd_bool.allow_parameterless_count);
    assert_eq!(ResolutionConf::KEYS.len(), 34);
    let charsets = conf(&[
        ("spark.sql.legacy.javaCharsets", "true"),
        ("spark.sql.legacy.codingErrorAction", "TRUE"),
    ]);
    assert!(charsets.legacy_java_charsets && charsets.legacy_coding_error_action);
    assert!(!d.legacy_java_charsets && !d.legacy_coding_error_action);
}

/// `count()` with no arguments: `WRONG_NUM_ARGS.WITH_SUGGESTION`, or 0 with
/// the legacy flag (Spark's golden `count` statements 14 and 16).
#[test]
fn parameterless_count_honours_the_legacy_flag() {
    let plan = || LogicalPlan::table(["t"]).project(vec![f("count", vec![])]);
    let err = run(plan(), &ctx()).unwrap_err();
    assert_eq!(
        summary(&Err(err)),
        serde_json::json!([
            "WRONG_NUM_ARGS.WITH_SUGGESTION",
            {
                "actualNum": "0",
                "expectedNum": " >= 1",
                "functionName": "`count`",
                "legacyConfKey": "\"spark.sql.legacy.allowParameterlessCount\"",
                "legacyConfValue": "\"true\"",
                "legacyNum": "0",
            }
        ])
    );
    let mut legacy = ctx();
    legacy.conf.allow_parameterless_count = true;
    let a = run(plan(), &legacy).unwrap();
    assert_eq!(a.output[0].name, "count()");
    assert_eq!(a.output[0].data_type, DataType::Long);
    assert!(!a.output[0].nullable);
    // `DISTINCT`, `FILTER` and `OVER` ride along: no panic, same flag.
    let distinct = || {
        LogicalPlan::table(["t"]).project(vec![Expr::new(ExprNode::Function(
            crate::FunctionCall {
                within_group: Vec::new(),
                name: "count".to_owned(),
                qualifier: Vec::new(),
                args: Vec::new(),
                arg_names: Vec::new(),
                distinct: true,
                filter: None,
            },
        ))])
    };
    let a = run(distinct(), &legacy).unwrap();
    assert_eq!(a.output[0].name, "count(DISTINCT )");
    let filtered = || {
        LogicalPlan::table(["t"]).project(vec![Expr::new(ExprNode::Function(
            crate::FunctionCall {
                within_group: Vec::new(),
                name: "count".to_owned(),
                qualifier: Vec::new(),
                args: Vec::new(),
                arg_names: Vec::new(),
                distinct: false,
                filter: Some(Arc::new(eq(c("a"), int(1)))),
            },
        ))])
    };
    let a = run(filtered(), &legacy).unwrap();
    assert_eq!(a.output[0].data_type, DataType::Long);
    assert_eq!(a.output[0].name, "count() FILTER (WHERE (a = 1))");
    assert!(class(&run(filtered(), &ctx())).starts_with("WRONG_NUM_ARGS"));
    let over = || {
        LogicalPlan::table(["t"]).project(vec![Expr::new(ExprNode::Window(crate::WindowExpr {
            function: Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
                within_group: Vec::new(),
                name: "count".to_owned(),
                qualifier: Vec::new(),
                args: Vec::new(),
                arg_names: Vec::new(),
                distinct: false,
                filter: None,
            }))),
            window: crate::WindowSpecRef::Spec(crate::WindowSpec {
                partition_by: Vec::new(),
                order_by: Vec::new(),
                frame: None,
            }),
        }))])
    };
    let a = run(over(), &legacy).unwrap();
    assert_eq!(
        a.output[0].name,
        "count() OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)"
    );
    assert!(class(&run(over(), &ctx())).starts_with("WRONG_NUM_ARGS"));
}

/// `count(tbl.*)`: `INVALID_USAGE_OF_STAR_WITH_TABLE_IDENTIFIER_IN_COUNT`,
/// or the table's columns with the legacy flag (Spark's golden `count`
/// statements 18 and 20).
#[test]
fn star_with_table_in_count_honours_the_legacy_flag() {
    let star_t = || {
        Expr::new(ExprNode::Star(Star {
            regex: None,
            replace: Vec::new(),
            position: None,
            qualifier: vec!["t".to_owned()],
            except: Vec::new(),
            plan_id: None,
        }))
    };
    let plan = || LogicalPlan::table(["t"]).project(vec![f("count", vec![star_t()])]);
    let err = run(plan(), &ctx()).unwrap_err();
    assert_eq!(
        summary(&Err(err)),
        serde_json::json!([
            "INVALID_USAGE_OF_STAR_WITH_TABLE_IDENTIFIER_IN_COUNT",
            {"tableName": "`t`"}
        ])
    );
    let mut legacy = ctx();
    legacy.conf.allow_star_with_single_table_identifier_in_count = true;
    let a = run(plan(), &legacy).unwrap();
    assert_eq!(a.output[0].name, "count(a, b, c)");
    // Two qualified stars still expand without the flag (golden `count` 3):
    // `count(tbl.*, tbl.*)` with the flag off is not the single-star error.
    let multi = LogicalPlan::table(["t"]).project(vec![f("count", vec![star_t(), star_t()])]);
    assert_eq!(
        run(multi, &ctx()).unwrap().output[0].name,
        "count(a, b, c, a, b, c)"
    );
    let multi_legacy =
        LogicalPlan::table(["t"]).project(vec![f("count", vec![star_t(), star_t()])]);
    assert_eq!(
        run(multi_legacy, &legacy).unwrap().output[0].name,
        "count(a, b, c, a, b, c)"
    );
    // Multi-part qualifiers match whole identifiers only: `db.t.*` against
    // a scope holding only `t` does not name it (so the star error, not the
    // single-star error, with or without the flag).
    let star_db_t = || {
        Expr::new(ExprNode::Star(Star {
            regex: None,
            replace: Vec::new(),
            position: None,
            qualifier: vec!["db".to_owned(), "t".to_owned()],
            except: Vec::new(),
            plan_id: None,
        }))
    };
    let plan = || LogicalPlan::table(["t"]).project(vec![f("count", vec![star_db_t()])]);
    assert_eq!(class(&run(plan(), &ctx())), "CANNOT_RESOLVE_STAR_EXPAND");
    assert_eq!(class(&run(plan(), &legacy)), "CANNOT_RESOLVE_STAR_EXPAND");
    // An unknown qualifier keeps the star error, with or without the flag.
    let missing = || {
        Expr::new(ExprNode::Star(Star {
            regex: None,
            replace: Vec::new(),
            position: None,
            qualifier: vec!["missing".to_owned()],
            except: Vec::new(),
            plan_id: None,
        }))
    };
    let plan = || LogicalPlan::table(["t"]).project(vec![f("count", vec![missing()])]);
    assert_eq!(class(&run(plan(), &ctx())), "CANNOT_RESOLVE_STAR_EXPAND");
    assert_eq!(class(&run(plan(), &legacy)), "CANNOT_RESOLVE_STAR_EXPAND");
}

/// A legacy `WITH`: nested definition bodies only see their own level's
/// previous siblings, while nested main queries see the enclosing levels
/// first (Spark's golden `cte-legacy` 6, 8 and 19 through inlining).
#[test]
fn legacy_cte_inlining_prefers_outer_definitions() {
    use crate::analyzer::CtePrecedencePolicy;
    use crate::analyzer::legacy::{CtePolicy, eliminate};
    let policy = || CtePolicy {
        precedence: CtePrecedencePolicy::Legacy,
        case_sensitive: false,
    };
    let with = |name: &str, plan: LogicalPlan, input: LogicalPlan| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(input),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(plan),
            }],
        })
    };
    // WITH t AS (SELECT * FROM u), t2 AS (WITH t AS (SELECT * FROM v)
    // SELECT * FROM t) SELECT * FROM t2: the nested query reads the outer t.
    let nested = with(
        "t",
        LogicalPlan::table(["v"]),
        LogicalPlan::table(["t"]).project(vec![star()]),
    );
    let outer = with(
        "t2",
        nested,
        LogicalPlan::table(["t2"]).project(vec![star()]),
    );
    let outer = with("t", LogicalPlan::table(["u"]), outer);
    let gone = eliminate(&Arc::new(outer), policy()).unwrap();
    assert_eq!(single_reads(&gone), ["u"]);
    // WITH t1 AS (SELECT * FROM u), t2 AS (WITH t3 AS (SELECT * FROM t1)
    // SELECT * FROM t3) SELECT * FROM t2: the nested body cannot see t1.
    let nested = with(
        "t3",
        LogicalPlan::table(["t1"]),
        LogicalPlan::table(["t3"]).project(vec![star()]),
    );
    let outer = with(
        "t2",
        nested,
        LogicalPlan::table(["t2"]).project(vec![star()]),
    );
    let outer = with("t1", LogicalPlan::table(["u"]), outer);
    let gone = eliminate(&Arc::new(outer), policy()).unwrap();
    assert_eq!(single_reads(&gone), ["t1"]);
}

/// Only `WITH RECURSIVE` is `RECURSIVE_CTE_IN_LEGACY_MODE` in Spark; a body
/// that names its own definition reads an outer definition substituted later,
/// or else the table of that name.
#[test]
fn legacy_cte_self_reference_reads_the_outer_name_or_the_table() {
    use crate::analyzer::CtePrecedencePolicy;
    use crate::analyzer::legacy::{CtePolicy, eliminate};
    let policy = CtePolicy {
        precedence: CtePrecedencePolicy::Legacy,
        case_sensitive: false,
    };
    let with = |name: &str, plan: LogicalPlan, input: LogicalPlan| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(input),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(plan),
            }],
        })
    };
    // WITH t AS (SELECT * FROM t) SELECT * FROM t: the table `t`.
    let alone = with(
        "t",
        LogicalPlan::table(["t"]),
        LogicalPlan::table(["t"]).project(vec![star()]),
    );
    let gone = eliminate(&Arc::new(alone), policy).unwrap();
    assert_eq!(single_reads(&gone), ["t"]);
    // WITH t AS (SELECT * FROM u) SELECT * FROM (WITH t AS (SELECT * FROM t)
    // SELECT * FROM t): the inner body's `t` is the outer definition.
    let inner = with(
        "t",
        LogicalPlan::table(["t"]),
        LogicalPlan::table(["t"]).project(vec![star()]),
    );
    let outer = with("t", LogicalPlan::table(["u"]), inner.project(vec![star()]));
    let gone = eliminate(&Arc::new(outer), policy).unwrap();
    assert_eq!(single_reads(&gone), ["u"]);
}

/// A definition read k times is one shared node: inlining a chain that
/// doubles at every level stays linear in distinct nodes.
#[test]
fn legacy_cte_inlining_of_a_doubling_chain_is_linear() {
    use crate::analyzer::CtePrecedencePolicy;
    use crate::analyzer::legacy::{CtePolicy, eliminate};
    let policy = CtePolicy {
        precedence: CtePrecedencePolicy::Legacy,
        case_sensitive: false,
    };
    const LEVELS: usize = 30;
    let mut ctes = vec![Cte {
        name: "t0".to_owned(),
        plan: Arc::new(LogicalPlan::table(["base"])),
    }];
    for k in 1..=LEVELS {
        let prev = format!("t{}", k - 1);
        ctes.push(Cte {
            name: format!("t{k}"),
            plan: Arc::new(LogicalPlan::table([prev.clone()]).join(
                LogicalPlan::table([prev]),
                JoinKind::Inner,
                JoinCondition::None,
            )),
        });
    }
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table([format!("t{LEVELS}")])),
        ctes,
    });
    let gone = eliminate(&Arc::new(plan), policy).unwrap();
    assert_eq!(single_reads(&gone), ["base"]);
}

/// `EXCEPTION` checks a plan whose nodes are shared at every level in
/// linear time (the walk keeps the nodes it has finished).
#[test]
fn exception_conflict_check_is_linear_on_shared_plans() {
    use crate::analyzer::CtePrecedencePolicy;
    use crate::analyzer::legacy::{CtePolicy, nested_conflict};
    let policy = CtePolicy {
        precedence: CtePrecedencePolicy::Exception,
        case_sensitive: false,
    };
    let mut plan = Arc::new(LogicalPlan::table(["base"]));
    for _ in 0..30 {
        plan = Arc::new(Arc::clone(&plan).as_ref().clone().join(
            (*plan).clone(),
            JoinKind::Inner,
            JoinCondition::None,
        ));
    }
    assert!(nested_conflict(&plan, policy).unwrap().is_none());
}

/// The cache key of an analysis is the caller's plan, also under `LEGACY`
/// (where the analyzer resolves an inlined copy).
#[test]
fn legacy_analysis_keeps_the_input_key_of_the_callers_plan() {
    use crate::analyzer::CtePrecedencePolicy;
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![star()])),
        ctes: vec![Cte {
            name: "t".to_owned(),
            plan: Arc::new(LogicalPlan::table(["u"])),
        }],
    });
    let expected = crate::Plan::try_new(Arc::new(plan.clone()))
        .unwrap()
        .structural_hash();
    let mut legacy = ctx();
    legacy.conf.cte_precedence = CtePrecedencePolicy::Legacy;
    let a = run(plan, &legacy);
    assert_eq!(a.map(|a| a.input_key).ok(), Some(expected));
}

/// `EXCEPTION`: a nested definition shadowing an outer one fails, with
/// Spark's class and parameters (Spark's golden `cte-nonlegacy` 6); the
/// same shadowing in a `FROM` subquery does not (statement 9).
#[test]
fn exception_cte_conflicts_follow_spark() {
    use crate::analyzer::CtePrecedencePolicy;
    let mut exc = ctx();
    exc.conf.cte_precedence = CtePrecedencePolicy::Exception;
    let with = |name: &str, plan: LogicalPlan, input: LogicalPlan| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(input),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(plan),
            }],
        })
    };
    // WITH t AS (...), t2 AS (WITH t AS (...) SELECT * FROM t) ...
    let nested = with(
        "t",
        LogicalPlan::table(["u"]),
        LogicalPlan::table(["t"]).project(vec![star()]),
    );
    let outer = with(
        "t2",
        nested,
        LogicalPlan::table(["t2"]).project(vec![star()]),
    );
    let outer = with("t", LogicalPlan::table(["u"]), outer);
    let err = run(outer, &exc).unwrap_err();
    assert_eq!(
        summary(&Err(err)),
        serde_json::json!([
            "AMBIGUOUS_ALIAS_IN_NESTED_CTE",
            {
                "config": "\"spark.sql.legacy.ctePrecedencePolicy\"",
                "docroot": "https://spark.apache.org/docs/latest",
                "name": "`t`",
            }
        ])
    );
    // A corrected session resolves the same shape to the inner definition.
    let nested = with(
        "t",
        LogicalPlan::table(["u"]),
        LogicalPlan::table(["t"]).project(vec![star()]),
    );
    let outer = with(
        "t2",
        nested,
        LogicalPlan::table(["t2"]).project(vec![star()]),
    );
    let outer = with("t", LogicalPlan::table(["u"]), outer);
    assert_eq!(class(&run(outer, &ctx())), "OK");
    // WITH t AS (...) SELECT ... FROM (WITH t AS (...) ...): shadowing in a
    // FROM subquery is not a conflict.
    let inner = with(
        "t",
        LogicalPlan::table(["u"]),
        LogicalPlan::table(["t"]).project(vec![star()]),
    );
    let outer = with("t", LogicalPlan::table(["t"]), inner);
    assert_eq!(class(&run(outer, &exc)), "OK");
    // Case-insensitive conflicts report the inner spelling: the shadowing
    // WITH sits in a definition body, as in golden `cte-nonlegacy` 17.
    let inner = with(
        "aBc",
        LogicalPlan::table(["u"]),
        LogicalPlan::table(["aBC"]).project(vec![star()]),
    );
    let mid = with("t", inner, LogicalPlan::table(["t"]).project(vec![star()]));
    let outer = with("abc", LogicalPlan::table(["t"]), mid);
    let err = run(outer, &exc).unwrap_err();
    assert_eq!(
        err.error
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "name")
            .map(|(_, v)| v.as_str()),
        Some("`aBc`")
    );
}

/// Without ANSI a string never meets another type in an inline table; with
/// ANSI it does (`INT` and `STRING` meet at `BIGINT`).
#[test]
fn inline_table_string_promotion_follows_ansi() {
    let rows = || inline_table(vec![vec![int(1)], vec![s("x")]], &["a"]);
    let mut off = ctx();
    off.conf.ansi = false;
    let r = run(rows(), &off);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_INLINE_TABLE.INCOMPATIBLE_TYPES_IN_INLINE_TABLE",
            {"colName": "`a`"}
        ])
    );
    let mut on = ctx();
    on.conf.ansi = true;
    let r = run(rows(), &on);
    assert_eq!(datatypes(&r.unwrap()), [DataType::Long]);
    // The remaining widening is the set operations' either way.
    let r = run(
        inline_table(vec![vec![int(1)], vec![long(2)]], &["a"]),
        &off,
    );
    assert_eq!(datatypes(&r.unwrap()), [DataType::Long]);
}

/// An unused CTE over a missing relation still fails the query with
/// `TABLE_OR_VIEW_NOT_FOUND` (Spark golden `cte` cases 16 and 24): table
/// existence is checked even though the definition is never resolved.
/// Column errors of an unused definition still never fail the query.
#[test]
fn unused_cte_tables_still_fail() {
    let ctx = ctx();
    let missing = |name: &str| LogicalPlan::table([name]).project(vec![c("a")]);
    // An unused definition over a missing table fails.
    let unused_missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(missing("nope")),
        }],
    });
    let e = run(unused_missing, &ctx).unwrap_err();
    assert_eq!(e.error.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(
        params.get("relationName").map(String::as_str),
        Some("`nope`")
    );
    // The main query reading only the valid definition still resolves when
    // the other definition is ambiguous (column errors are skipped), but
    // fails when the other definition reads a missing table.
    let ambiguous = LogicalPlan::table(["t"])
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::None,
        )
        .project(vec![c("a")]);
    assert_eq!(class(&run(ambiguous.clone(), &ctx)), "AMBIGUOUS_REFERENCE");
    let ok = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["good"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(ambiguous),
            },
        ],
    });
    run(ok, &ctx).unwrap();
    let missing_bad = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["good"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(missing("gone")),
            },
        ],
    });
    assert_eq!(class(&run(missing_bad, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
}

/// A forward reference inside a CTE falls through to tables: the later
/// definition is not visible, so reading it is a missing table (Spark
/// golden `cte` case 24, `with r1 as (select * from r2), r2 as ...`).
#[test]
fn cte_forward_references_are_missing_tables() {
    let ctx = ctx();
    let forward = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "r1".to_owned(),
                plan: Arc::new(LogicalPlan::table(["r2"]).project(vec![c("a")])),
            },
            Cte {
                name: "r2".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
        ],
    });
    let e = run(forward, &ctx).unwrap_err();
    assert_eq!(e.error.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("relationName").map(String::as_str), Some("`r2`"));
    // In dependency order the same definitions resolve.
    let ordered = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["b"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "a".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "b".to_owned(),
                plan: Arc::new(LogicalPlan::table(["a"]).project(vec![c("a")])),
            },
        ],
    });
    run(ordered, &ctx).unwrap();
}

/// Unused CTE bodies nested inside an unused definition are checked too:
/// a missing table at any depth fails the query, while an ambiguous
/// column at any depth does not.
#[test]
fn unused_cte_tables_are_checked_at_any_depth() {
    let ctx = ctx();
    let nested_missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "inner".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["missing_deep"]).project(vec![c("a")])),
                }],
            })),
        }],
    });
    assert_eq!(class(&run(nested_missing, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
    // A self-reference inside an unused definition is a missing table, not
    // infinite recursion (Spark golden `cte` cases 4 and 5).
    let self_ref = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "s".to_owned(),
            plan: Arc::new(LogicalPlan::table(["s"]).project(vec![c("a")])),
        }],
    });
    assert_eq!(class(&run(self_ref, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
}

/// CTE name matching is case-insensitive by default: an unused definition
/// shadowing a table with different casing still hides it from table
/// checks, and exact when `caseSensitive` is set.
#[test]
fn unused_cte_shadowing_follows_case_sensitivity() {
    let ctx = ctx();
    // `T` (upper) shadows table `t` case-insensitively: reading `t` in the
    // main query is the CTE, and the unused `T` body over `t` reads the
    // table (which exists), so no table error.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["T"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "T".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    run(shadowed, &ctx).unwrap();
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..ResolutionConf::default()
    };
    let sensitive_ctx = context(catalog(), sensitive);
    // Case-sensitive: `T` does not shadow `t`, so the main query reads the
    // table and the CTE body is unused but over an existing table too.
    run(
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "T".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            }],
        }),
        &sensitive_ctx,
    )
    .unwrap();
    // Case-sensitive: an unused `T` over a missing `T` still fails, since
    // no shadowing applies to its own body.
    let missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "T".to_owned(),
            plan: Arc::new(LogicalPlan::table(["T_missing"]).project(vec![c("a")])),
        }],
    });
    assert_eq!(
        class(&run(missing, &sensitive_ctx)),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

/// `GROUP BY` resolves a backquoted select-list alias (even an implicit one
/// like `` `1` `` for `SELECT 1`): with `groupByAliases` on the key groups
/// by the alias expression, with it off the same key is unresolved (Spark
/// golden `group-by-alias` cases 11 and 32, `having` cases 24 and 25).
/// Hand-built plans use plain names: backquotes are SQL syntax only.
#[test]
fn group_by_resolves_backquoted_select_aliases() {
    let ctx = ctx();
    // Explicit alias (`SELECT a AS k ... GROUP BY k`, backquoted in SQL).
    let aliased = LogicalPlan::table(["t"]).aggregate(vec![c("k")], vec![c("a").alias("k")]);
    run(aliased, &ctx).unwrap();
    // Implicit alias: `SELECT 1 GROUP BY `1`` groups by the literal.
    let implicit = LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    let a = run(implicit, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["1"]
    );
    // With aliases off the same keys are unresolved, with a suggestion when
    // the input has columns and without one over `OneRowRelation`.
    let off = context(
        catalog(),
        ResolutionConf {
            group_by_aliases: false,
            ..ResolutionConf::default()
        },
    );
    let aliased_off = LogicalPlan::table(["t"]).aggregate(vec![c("k")], vec![c("a").alias("k")]);
    assert_eq!(
        class(&run(aliased_off, &off)),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let implicit_off =
        LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    assert_eq!(
        class(&run(implicit_off, &off)),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
    // `SELECT 1 AS `2`, 2 ... GROUP BY `2`` (Spark golden `having` cases 24
    // and 25): the key resolves to the input column, which takes precedence
    // over the select-list aliases (constants need no grouping).
    let backquoted = LogicalPlan::table(["t"])
        .aggregate(vec![c("b")], vec![int(1).alias("b"), int(2).alias("b")]);
    run(backquoted, &ctx).unwrap();
}

/// `GROUP BY` resolves the first matching select-list alias: duplicate
/// aliases do not fail the grouping itself (Spark golden `group-by-alias`
/// case 27 succeeds).
#[test]
fn group_by_picks_the_first_matching_alias() {
    let ctx = ctx();
    // Checked on a Spark 4.2.0 server: `SELECT a AS x, a AS x FROM t GROUP BY x`
    // groups by the first `x` and succeeds; with `b AS x` second the key
    // still resolves, and the query fails only because `b` is not grouped.
    let dup = LogicalPlan::table(["t"])
        .aggregate(vec![c("x")], vec![c("a").alias("x"), c("a").alias("x")]);
    run(dup, &ctx).unwrap();
    let ungrouped = LogicalPlan::table(["t"])
        .aggregate(vec![c("x")], vec![c("a").alias("x"), c("b").alias("x")]);
    assert_eq!(class(&run(ungrouped, &ctx)), "MISSING_AGGREGATION");
    let dup_group = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")]);
    // Resolves to the first alias rather than raising `AMBIGUOUS_REFERENCE`.
    let a = run(dup_group, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "a"]
    );
}

/// `HAVING` and `ORDER BY` over an aggregate read grouping attributes when
/// the select list repeats a name (Spark golden `having` cases 24 and 25,
/// `group-by-alias` case 28's `HAVING`): without leniency the duplicate
/// outputs are `AMBIGUOUS_REFERENCE` where Spark resolves to the grouping
/// column.
#[test]
fn having_and_order_by_prefer_grouping_over_duplicate_outputs() {
    let ctx = ctx();
    // `SELECT 1 AS a, 2 AS a FROM t GROUP BY a HAVING a > 0`: `a` in the
    // grouping and the `HAVING` resolves to the grouping column (input
    // `a`), not to either duplicate output.
    let having = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")])
        .filter(gt(c("a"), int(0)));
    let a = run(having, &ctx).unwrap();
    // The predicate and the grouping key share the input column's id, which
    // neither duplicate output carries.
    let mut predicates = Vec::new();
    let mut groupings = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node {
            match &p.node {
                PlanNode::Filter { predicate, .. } => {
                    predicates.extend(resolved_ids(predicate));
                }
                PlanNode::Aggregate { grouping, .. } => {
                    groupings.extend(grouping_keys(grouping).iter().flat_map(resolved_ids));
                }
                _ => {}
            }
        }
        ControlFlow::Continue(())
    });
    assert_eq!(predicates.len(), 1);
    assert_eq!(groupings, predicates);
    let outputs: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert!(!outputs.contains(&predicates[0]));
    // Without a grouping column of that name, the same shape stays
    // ambiguous: two outputs, nothing below to prefer.
    let ambiguous = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![int(1)], vec![int(1).alias("a"), int(2).alias("a")]);
    let ambiguous = ambiguous.filter(gt(c("a"), int(0)));
    assert_eq!(class(&run(ambiguous, &ctx)), "AMBIGUOUS_REFERENCE");
    // `ORDER BY` over the aggregate behaves the same way.
    let order = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let a = run(order, &ctx).unwrap();
    // The sort key is the grouping column (input `a`), not either duplicate
    // output.
    let mut keys = Vec::new();
    let mut groupings = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node {
            match &p.node {
                PlanNode::Sort { order, .. } => {
                    keys.extend(order.iter().flat_map(|o| resolved_ids(&o.expr)));
                }
                PlanNode::Aggregate { grouping, .. } => {
                    groupings.extend(grouping_keys(grouping).iter().flat_map(resolved_ids));
                }
                _ => {}
            }
        }
        ControlFlow::Continue(())
    });
    assert_eq!(keys.len(), 1);
    assert_eq!(groupings, keys);
    let outputs: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert!(!outputs.contains(&keys[0]));
    let order_ambiguous = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![int(1)], vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    // Spark 4.2.0 reports `UNRESOLVED_COLUMN` (not `AMBIGUOUS_REFERENCE`)
    // for `SELECT 1 AS a, 2 AS a GROUP BY 1 ORDER BY a`, proposing the
    // duplicate outputs.
    let e = run(order_ambiguous, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(params.get("objectName").map(String::as_str), Some("`a`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`, `a`"));
}
