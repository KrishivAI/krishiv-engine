// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// `ORDER BY` over an aggregate with an unknown column stays unresolved:
/// with no grouping attribute of that name either, the lookup error passes
/// through unchanged (Spark 4.2.0 reports `UNRESOLVED_COLUMN`, proposing
/// the aggregate's outputs).
#[test]
fn order_by_unknown_column_over_aggregate_is_unresolved() {
    let ctx = ctx();
    // `SELECT a FROM t GROUP BY a ORDER BY missing`: `missing` is neither
    // an output nor a grouping attribute.
    let order = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![c("a")])
        .sort(vec![crate::SortOrder::asc(c("missing"))]);
    let e = run(order, &ctx).unwrap_err();
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
    assert_eq!(
        params.get("objectName").map(String::as_str),
        Some("`missing`")
    );
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`"));
}

/// `ORDER BY` over duplicate select outputs looks below them (existing
/// leniency): with a single input column of that name it resolves there,
/// otherwise it stays ambiguous rather than unresolved.
#[test]
fn order_by_duplicates_fall_back_then_stay_unresolved() {
    let ctx = ctx();
    // `SELECT 1 AS a, 2 AS a FROM t ORDER BY a`: the input `a` wins over
    // the duplicate outputs.
    let order = LogicalPlan::table(["t"])
        .project(vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    run(order, &ctx).unwrap();
    // No input column of that name: unresolved (Spark 4.2.0:
    // `SELECT 1 AS a, 2 AS a ORDER BY a` is UNRESOLVED_COLUMN, proposing the
    // duplicate outputs).
    let ambiguous = LogicalPlan::new(PlanNode::OneRowRelation)
        .project(vec![int(1).alias("a"), int(2).alias("a")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let e = run(ambiguous, &ctx).unwrap_err();
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

/// `LIMIT` (and `OFFSET`) over a column is `IS_UNFOLDABLE`, not
/// `UNRESOLVED_COLUMN` (Spark golden `limit` case 9): columns resolve
/// against the input and then fail folding, exactly as nondeterministic
/// calls do.
#[test]
fn limit_over_a_column_is_unfoldable() {
    let ctx = ctx();
    for (plan, name) in [
        (LogicalPlan::table(["t"]).limit(gt(c("a"), int(3))), "limit"),
        (LogicalPlan::table(["t"]).limit(c("a")), "limit"),
        (LogicalPlan::table(["t"]).offset(c("b")), "offset"),
    ] {
        let e = run(plan, &ctx).unwrap_err();
        assert_eq!(
            e.error.error_class().name(),
            "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE",
            "{name}"
        );
        let params: std::collections::BTreeMap<_, _> = e
            .error
            .message_parameters()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        assert_eq!(params.get("name").map(String::as_str), Some(name));
    }
    // Foldable non-integers are a type error, foldable negatives and nulls
    // are value errors, in Spark's order (foldable, then type, then value).
    let e = run(LogicalPlan::table(["t"]).limit(s("a")), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.DATA_TYPE"
    );
    let e = run(LogicalPlan::table(["t"]).limit(int(-1)), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    );
    let e = run(
        LogicalPlan::table(["t"]).limit(Expr::new(ExprNode::Literal(Literal::Null(Some(
            krishiv_types::DataType::Integer,
        ))))),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_NULL"
    );
    // A foldable integer limit resolves to a literal.
    let a = run(LogicalPlan::table(["t"]).limit(add(int(1), int(2))), &ctx).unwrap();
    assert!(
        a.plan.root().to_string().contains("Limit"),
        "{}",
        a.plan.root()
    );
}

/// `LIMIT` resolves against the select list: an alias the projection
/// defines is visible (then fails folding when it names a column), and
/// duplicates in the projection stay ambiguous rather than resolving
/// below it.
#[test]
fn limit_resolves_against_the_select_list() {
    let ctx = ctx();
    // `SELECT a AS k FROM t LIMIT k`: `k` resolves to the projection
    // (not `UNRESOLVED_COLUMN`), then fails as unfoldable since it names
    // a column.
    let aliased = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("k")])
        .limit(c("k"));
    assert_eq!(
        class(&run(aliased, &ctx)),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    // Duplicates in the projection are ambiguous, even though the input
    // has a single column of that name.
    let dup = LogicalPlan::table(["t"])
        .project(vec![int(1).alias("a"), int(2).alias("a")])
        .limit(c("a"));
    assert_eq!(class(&run(dup, &ctx)), "AMBIGUOUS_REFERENCE");
    // Checked on a Spark 4.2.0 server: `SELECT a AS x, b AS x FROM t LIMIT x`
    // is `AMBIGUOUS_REFERENCE`; the same attribute selected twice
    // (`SELECT a, a FROM t LIMIT a`) is one attribute, so `LIMIT a` is
    // unfoldable; and a name the select list lacks (`LIMIT b`) is
    // unresolved: the limit reads the select list, not the table below it.
    let twice = LogicalPlan::table(["t"])
        .project(vec![c("a"), c("a")])
        .limit(c("a"));
    assert_eq!(
        class(&run(twice, &ctx)),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    let below = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .limit(c("b"));
    let e = run(below, &ctx).unwrap_err();
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
    assert_eq!(params.get("objectName").map(String::as_str), Some("`b`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`a`"));
}

/// `GROUP BY` ordinals follow `spark.sql.groupByOrdinal`: with it on, an
/// integer literal names the n-th select item; with it off, the same
/// literal is a constant grouping key.
#[test]
fn group_by_ordinals_follow_their_config() {
    let ctx = ctx();
    // `SELECT a, COUNT(b) FROM t GROUP BY 1`: groups by the first item.
    let ordinal =
        LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![c("a"), f("count", vec![c("b")])]);
    let a = run(ordinal, &ctx).unwrap();
    assert_eq!(a.output.len(), 2);
    // Out of range is `GROUP_BY_POS_OUT_OF_RANGE`.
    let out_of_range =
        LogicalPlan::table(["t"]).aggregate(vec![int(5)], vec![c("a"), f("count", vec![c("b")])]);
    assert_eq!(class(&run(out_of_range, &ctx)), "GROUP_BY_POS_OUT_OF_RANGE");
    // With ordinals off, `GROUP BY 1` groups by the constant 1 (one group).
    let off = context(
        catalog(),
        ResolutionConf {
            group_by_ordinal: false,
            ..ResolutionConf::default()
        },
    );
    let constant =
        LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![f("count", vec![star()])]);
    let a = run(constant, &off).unwrap();
    assert_eq!(a.output.len(), 1);
    // Star with an ordinal key is `STAR_GROUP_BY_POS`.
    let star = LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![star()]);
    assert_eq!(class(&run(star, &ctx)), "STAR_GROUP_BY_POS");
}

/// `ORDER BY` ordinals follow `spark.sql.orderByOrdinal`: with it on, an
/// integer literal names the n-th output column; with it off, the same
/// literal sorts by its constant value.
#[test]
fn order_by_ordinals_follow_their_config() {
    let ctx = ctx();
    // `SELECT a, b FROM t ORDER BY 2`: sorts by the second output (`b`).
    let ordinal = LogicalPlan::table(["t"])
        .project(vec![c("a"), c("b")])
        .sort(vec![crate::SortOrder::asc(int(2))]);
    let a = run(ordinal, &ctx).unwrap();
    assert!(
        a.plan.root().to_string().contains("b#"),
        "{}",
        a.plan.root()
    );
    // Out of range is `ORDER_BY_POS_OUT_OF_RANGE`.
    let out_of_range = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .sort(vec![crate::SortOrder::asc(int(5))]);
    assert_eq!(class(&run(out_of_range, &ctx)), "ORDER_BY_POS_OUT_OF_RANGE");
    // With ordinals off, `ORDER BY 2` sorts by the constant 2.
    let off = context(
        catalog(),
        ResolutionConf {
            order_by_ordinal: false,
            ..ResolutionConf::default()
        },
    );
    run(
        LogicalPlan::table(["t"])
            .project(vec![c("a")])
            .sort(vec![crate::SortOrder::asc(int(2))]),
        &off,
    )
    .unwrap();
}

/// `GROUP BY` attribute precedence and `HAVING`/`ORDER BY` over the
/// aggregate: an input column wins over select-list aliases of that name,
/// and later clauses resolve to the grouping column rather than either
/// duplicate output.
#[test]
fn group_by_attributes_take_precedence_over_aliases() {
    let ctx = ctx();
    // `SELECT 1 AS a, 2 AS a FROM t(a) GROUP BY a`: the key is the input
    // column, not either alias.
    let key = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")]);
    // The key is the input column: the grouping expression references an id
    // neither duplicate output carries.
    let a = run(key, &ctx).unwrap();
    let mut groupings = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Aggregate { grouping, .. } = &p.node
        {
            groupings.extend(grouping_keys(grouping).iter().flat_map(resolved_ids));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(groupings.len(), 1);
    let outputs: Vec<_> = a.output.iter().map(|c| c.id).collect();
    assert!(!outputs.contains(&groupings[0]));
    // Without an input column of that name, the first alias wins for the
    // grouping itself (Spark golden `group-by-alias` case 27 succeeds).
    let no_input = LogicalPlan::new(PlanNode::OneRowRelation)
        .aggregate(vec![c("a")], vec![int(1).alias("a"), int(2).alias("a")]);
    run(no_input, &ctx).unwrap();
}

/// CTE shadowing across nesting levels: an inner `WITH` name hides the
/// outer one inside its scope, and reads outside still need the outer
/// definition (which is checked for missing tables even when unused
/// elsewhere).
#[test]
fn cte_shadowing_applies_at_every_level() {
    let ctx = ctx();
    let good = || LogicalPlan::table(["t"]).project(vec![c("a")]);
    let bad = || {
        LogicalPlan::table(["t"])
            .join(
                LogicalPlan::table(["u"]),
                JoinKind::Inner,
                JoinCondition::None,
            )
            .project(vec![c("a")])
    };
    // Inner `x` shadows outer `x`: only the inner (valid) definition is
    // read, so the outer ambiguity never fails the query.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "x".to_owned(),
                plan: Arc::new(good()),
            }],
        })),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    run(shadowed, &ctx).unwrap();
    // A read outside the inner scope still needs the outer definition: its
    // ambiguity fails the query.
    let outer_read = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(
            LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(good()),
                }],
            })
            .join(
                LogicalPlan::table(["x"]).project(vec![c("a")]),
                JoinKind::Inner,
                JoinCondition::None,
            ),
        ),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    assert_eq!(class(&run(outer_read, &ctx)), "AMBIGUOUS_REFERENCE");
}

/// `LIMIT` and `OFFSET` share Spark's checks: foldable, then type, then
/// value, with the same error classes and parameters as the golden
/// `limit.sql.out` shows.
#[test]
fn limit_and_offset_share_sparks_checks() {
    let ctx = ctx();
    // Foldable non-literals pass through as literals for the planner.
    let folded = LogicalPlan::table(["t"]).limit(add(int(1), int(2)));
    let a = run(folded, &ctx).unwrap();
    assert!(
        a.plan.root().to_string().contains("Limit"),
        "{}",
        a.plan.root()
    );
    // Offsets behave like limits (name `offset` in the parameters).
    let e = run(LogicalPlan::table(["t"]).offset(c("a")), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
    assert_eq!(
        e.error
            .message_parameters()
            .iter()
            .find(|(k, _)| *k == "name")
            .map(|(_, v)| v.as_str()),
        Some("offset")
    );
    // A subquery limit is unfoldable (cited at the expression).
    let sub = LogicalPlan::table(["t"]).limit(crate::tests::dsl::scalar(
        LogicalPlan::table(["u"]).project(vec![c("a")]),
    ));
    assert_eq!(
        class(&run(sub, &ctx)),
        "INVALID_LIMIT_LIKE_EXPRESSION.IS_UNFOLDABLE"
    );
}

/// `ORDER BY` over a non-aggregate with duplicate outputs resolves below
/// them (existing leniency), and `GROUP BY` ordinal out-of-range and
/// star-group-by-pos keep their own classes.
#[test]
fn order_by_leniency_and_group_by_star_rules() {
    let ctx = ctx();
    // Duplicate outputs with no input column of that name are unresolved
    // (Spark 4.2.0: `SELECT a AS k, b AS k FROM t ORDER BY k`), proposing
    // the duplicate outputs.
    let order = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("x"), c("b").alias("x")])
        .sort(vec![crate::SortOrder::asc(c("x"))]);
    let e = run(order, &ctx).unwrap_err();
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
    assert_eq!(params.get("objectName").map(String::as_str), Some("`x`"));
    assert_eq!(params.get("proposal").map(String::as_str), Some("`x`, `x`"));
    // Star grouped by all of its columns resolves.
    let star_ok = LogicalPlan::table(["t"]).aggregate(vec![c("a"), c("b"), c("c")], vec![star()]);
    run(star_ok, &ctx).unwrap();
}

/// A CTE name may contain dots when backquoted (Spark golden `cte` case
/// 44, `` WITH `a.b.c` AS ... ``): it is a single identifier, resolved as
/// one CTE rather than a qualified table.
#[test]
fn cte_names_may_contain_dots() {
    let ctx = ctx();
    let dotted = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["a.b.c"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "a.b.c".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    let a = run(dotted, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    // Without the definition, the same dotted name is a missing table.
    assert_eq!(
        class(&run(
            LogicalPlan::table(["a.b.c"]).project(vec![c("a")]),
            &ctx
        )),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

/// An inner `WITH` inside an unused CTE is still checked for missing
/// tables, and shadowing inside it hides what the outer scope would
/// otherwise report.
#[test]
fn unused_cte_inner_with_scopes_shadow() {
    let ctx = ctx();
    // Unused outer over an inner `WITH x ...` that reads `x`: the inner
    // read is the inner CTE, not a table, so no error.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
                }],
            })),
        }],
    });
    run(shadowed, &ctx).unwrap();
    // The same shape over a missing inner table fails.
    let missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::WithCte {
                input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
                ctes: vec![Cte {
                    name: "x".to_owned(),
                    plan: Arc::new(LogicalPlan::table(["missing_inner"]).project(vec![c("a")])),
                }],
            })),
        }],
    });
    assert_eq!(class(&run(missing, &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
}

/// `GROUP BY` with a subquery key and `HAVING`/`ORDER BY` over it resolve
/// without special-casing: scalar subqueries are foldable inputs, not
/// columns, so they never trigger alias or leniency paths.
#[test]
fn group_by_with_subquery_keys_resolves() {
    let ctx = ctx();
    // `SELECT a, COUNT(*) FROM t GROUP BY a, (SELECT ... LIMIT 1)` is
    // covered by golden `group-by-alias` case 15 (passes): the subquery
    // key is not a column, alias, or duplicate output.
    let sub = crate::tests::dsl::scalar(
        LogicalPlan::table(["u"])
            .project(vec![c("a")])
            .limit(int(1)),
    );
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("a"), sub], vec![c("a"), f("count", vec![star()])]);
    run(plan, &ctx).unwrap();
}

/// `ORDER BY` over a `HAVING` filter resolves through the filter to the
/// aggregate below it: duplicate select-list names are looked up below
/// with leniency first, and when that is ambiguous too the aggregate's
/// grouping attributes still get their try. A `HAVING` that grew the
/// aggregate leaves a projection the sort never resolves through, and
/// without a grouping column of the name the duplicates stay
/// `UNRESOLVED_COLUMN` (as over the aggregate directly).
#[test]
fn order_by_through_having_reports_duplicates_as_unresolved() {
    let ctx = ctx();
    // `SELECT 1 AS z, 2 AS z, a FROM t GROUP BY a HAVING a > 0 ORDER BY z`:
    // the `HAVING` needs nothing new (no growth), so the sort retries
    // through the filter against the aggregate, whose grouping has no `z`:
    // unresolved, proposing the duplicate outputs.
    let through = LogicalPlan::table(["t"])
        .aggregate(
            vec![c("a")],
            vec![int(1).alias("z"), int(2).alias("z"), c("a")],
        )
        .filter(gt(c("a"), int(0)))
        .sort(vec![crate::SortOrder::asc(c("z"))]);
    let e = run(through, &ctx).unwrap_err();
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
    assert_eq!(params.get("objectName").map(String::as_str), Some("`z`"));
    assert_eq!(
        params.get("proposal").map(String::as_str),
        Some("`z`, `z`, `a`")
    );
    // `SELECT 1 AS a, 2 AS a FROM t GROUP BY b HAVING b = 'x' ORDER BY a`:
    // the `HAVING` grows the aggregate (a projection the sort never
    // resolves through), and no grouping column is named `a`: unresolved.
    let grown = LogicalPlan::table(["t"])
        .aggregate(vec![c("b")], vec![int(1).alias("a"), int(2).alias("a")])
        .filter(eq(c("b"), s("x")))
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let e = run(grown, &ctx).unwrap_err();
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

/// A `Sql` node inside an unused CTE is still checked for missing tables:
/// the node's reads live in the lowered plan, which the
/// unused-definition walk follows. An unused `Sql` over existing tables
/// passes.
#[test]
fn unused_cte_sql_tables_are_checked() {
    // An unused definition holding SQL text over existing tables passes:
    // the walk follows the lowered plan Bind stored for the `Sql` node.
    let mut good_ctx = ctx();
    good_ctx.sql = Some(Arc::new(Frontend));
    let unused_sql = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::Sql {
                query: "q6".to_owned(),
                args: SqlArgs::None,
            })),
        }],
    });
    run(unused_sql, &good_ctx).unwrap();
    // The same shape over a missing table fails with
    // `TABLE_OR_VIEW_NOT_FOUND` at the lowered read.
    let mut bad_ctx = ctx();
    bad_ctx.sql = Some(Arc::new(Frontend));
    let unused_missing = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        ctes: vec![Cte {
            name: "u".to_owned(),
            plan: Arc::new(LogicalPlan::new(PlanNode::Sql {
                query: "gone".to_owned(),
                args: SqlArgs::None,
            })),
        }],
    });
    let e = run(unused_missing, &bad_ctx).unwrap_err();
    assert_eq!(e.error.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    assert_eq!(
        params.get("relationName").map(String::as_str),
        Some("`gone`")
    );
}

/// `ORDER BY` with several keys converts only the ambiguous one: a
/// resolvable key before it still binds, and the duplicate stays
/// `UNRESOLVED_COLUMN` (Spark 4.2.0 proposes the duplicate outputs, as
/// for a single key). (multi-key case)
#[test]
fn order_by_second_duplicate_key_is_unresolved() {
    let ctx = ctx();
    // `SELECT 1 AS x, 2 AS x, b FROM t ORDER BY b, x`: `b` resolves to
    // the input column while `x` stays unresolved. (multi-key sort)
    let order = LogicalPlan::table(["t"])
        .project(vec![int(1).alias("x"), int(2).alias("x"), c("b")])
        .sort(vec![
            crate::SortOrder::asc(c("b")),
            crate::SortOrder::asc(c("x")),
        ]);
    let e = run(order, &ctx).unwrap_err();
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
    assert_eq!(params.get("objectName").map(String::as_str), Some("`x`"));
}

/// `ORDER BY` through a `HAVING` filter also resolves a grouping
/// expression written out in full: with no growth the sort retries
/// against the aggregate below the filter, where the expression is the
/// aggregate's output (as over the aggregate directly) or is added to
/// it (as over the aggregate directly).
#[test]
fn order_by_through_having_resolves_full_grouping_expression() {
    let ctx = ctx();
    // `SELECT a + 1 AS k FROM t GROUP BY a + 1 HAVING k > 0 ORDER BY a + 1`:
    // the `HAVING` needs nothing new, so the sort retries through the
    // filter against the aggregate, resolving to the `k` output.
    let plan = LogicalPlan::table(["t"])
        .aggregate(
            vec![add(c("a"), int(1))],
            vec![add(c("a"), int(1)).alias("k")],
        )
        .filter(gt(c("k"), int(0)))
        .sort(vec![crate::SortOrder::asc(add(c("a"), int(1)))]);
    let a = run(plan, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["k"]
    );
    // The sort key is the `k` output.
    let mut keys = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            keys.extend(order.iter().flat_map(|o| resolved_ids(&o.expr)));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(keys, vec![a.output[0].id]);
    // `SELECT count(*) AS c FROM t GROUP BY a + 1 HAVING count(*) > 0
    // ORDER BY a + 1`: the grouping expression is not an output, so the
    // retry adds it to the aggregate below the filter and sorts by it. The
    // added column stays internal: the sort restores the select list above
    // what it added, so the query still outputs only `c` (as Spark does).
    let grown = LogicalPlan::table(["t"])
        .aggregate(
            vec![add(c("a"), int(1))],
            vec![f("count", vec![star()]).alias("c")],
        )
        .filter(gt(f("count", vec![star()]), int(0)))
        .sort(vec![crate::SortOrder::asc(add(c("a"), int(1)))]);
    let b = run(grown, &ctx).unwrap();
    assert_eq!(
        b.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["c"]
    );
    let mut grown_keys = Vec::new();
    let mut grown_names = Vec::new();
    let _ = visit_distinct(Node::Plan(b.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            for o in order {
                let _ = visit_distinct(Node::Expr(&o.expr), |n| {
                    if let Node::Expr(x) = n
                        && let ExprNode::Resolved(ResolvedExpr::Column(col)) = &x.node
                    {
                        grown_keys.push(col.id);
                        grown_names.push(col.name.clone());
                    }
                    ControlFlow::Continue(())
                });
            }
        }
        ControlFlow::Continue(())
    });
    // The sort key is the grouping expression the retry added below the
    // filter: one column, named for `a + 1`, and not one of the outputs.
    assert_eq!(grown_keys.len(), 1);
    assert_eq!(grown_names, ["(a + 1)"]);
    assert!(!b.output.iter().any(|o| o.id == grown_keys[0]));
}

/// `ORDER BY` over duplicate outputs above windows still reports them
/// as unresolved: the sort reads the select list only, never resolving
/// through the windows to the aggregate below.
#[test]
fn order_by_over_windows_reports_duplicates_as_unresolved() {
    let ctx = ctx();
    let ranked = Expr::new(ExprNode::Window(crate::WindowExpr {
        function: Arc::new(f("row_number", Vec::new())),
        window: crate::WindowSpecRef::Spec(crate::WindowSpec {
            partition_by: Vec::new(),
            // `row_number` needs an ordered window (Spark rejects `OVER ()`
            // with `WINDOW_FUNCTION_FRAME_NOT_ORDERED`, as does the
            // analyzer); the ordering does not affect the sort below.
            order_by: vec![crate::SortOrder::asc(c("a"))],
            frame: None,
        }),
    }));
    // `SELECT 1 AS x, 2 AS x, a, row_number() OVER (ORDER BY a) FROM t
    // GROUP BY a HAVING a > 0 ORDER BY x`: the `HAVING` needs nothing new, so the
    // sort retries through the filter, but the aggregate holds windows,
    // so the duplicates stay `UNRESOLVED_COLUMN`.
    let plan = LogicalPlan::table(["t"])
        .aggregate(
            vec![c("a")],
            vec![
                int(1).alias("x"),
                int(2).alias("x"),
                c("a"),
                ranked.alias("r"),
            ],
        )
        .filter(gt(c("a"), int(0)))
        .sort(vec![crate::SortOrder::asc(c("x"))]);
    let e = run(plan, &ctx).unwrap_err();
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
    assert_eq!(params.get("objectName").map(String::as_str), Some("`x`"));
}

/// `ORDER BY` passes a sort key's own errors through unchanged: only an
/// `AMBIGUOUS_REFERENCE` over duplicate outputs becomes
/// `UNRESOLVED_COLUMN`; any other failure (here a function's arity) is
/// reported as raised, as Spark 4.2 reports it.
#[test]
fn order_by_passes_sort_key_errors_through() {
    let ctx = ctx();
    // `SELECT a FROM t ORDER BY upper(1, 2)`: the key is not a name error,
    // so the sort reports the function's arity failure.
    let order = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .sort(vec![crate::SortOrder::asc(Expr::function(
            "upper",
            vec![int(1), int(2)],
        ))]);
    let err = run(order, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
    );
    let text = err.error.to_string();
    assert!(
        text.contains("The `upper` requires 1 parameters but the actual number is 2"),
        "{text}"
    );
}

/// `TIME`, `GEOMETRY` and `GEOGRAPHY` have no evaluation yet: a plan that
/// holds one anywhere (a cast, a typed `NULL`, a scanned column) is refused
/// with `KRISHIV_UNSUPPORTED_FEATURE` naming the type (`UNSUPPORTED_TIME_TYPE`
/// for `TIME`), at the expression.
#[test]
fn types_without_evaluation_are_refused() {
    use krishiv_types::Srid;
    let one_row = || LogicalPlan::new(PlanNode::OneRowRelation);
    let time = DataType::Time { precision: 3 };
    let cast = Expr::from(Literal::Null(None)).cast(time.clone());
    let origin = Origin::plan(PlanId(9));
    let e = run(one_row().project(vec![cast.with_origin(origin)]), &ctx())
        .err()
        .unwrap();
    assert_eq!(e.origin, origin);
    // `TIME` is its own class: Spark rejects it while
    // `spark.sql.timeType.enabled` is false (the default).
    assert_eq!(e.error.error_class(), ErrorClass::UnsupportedTimeType);
    let typed_null = Expr::from(Literal::Null(Some(DataType::Geometry { srid: Srid::Any })));
    // Spatial types evaluate: a typed NULL of one resolves.
    assert!(run(one_row().project(vec![typed_null]), &ctx()).is_ok());
    // Nested below a cast to another type, it is still found.
    let nested = Expr::from(Literal::Null(None))
        .cast(time)
        .cast(DataType::String);
    let e = run(one_row().project(vec![nested]), &ctx()).err().unwrap();
    assert_eq!(e.error.error_class(), ErrorClass::UnsupportedTimeType);
    // A scanned column, even when the query does not select it.
    let geo = TableColumn::new(
        "g".to_owned(),
        DataType::Geography {
            srid: Srid::Fixed(4326),
        },
        true,
    );
    let x = TableColumn::new("x".to_owned(), DataType::Integer, true);
    let cat =
        MemoryCatalog::new("spark_catalog").with_table(&["default"], "geo", vec![x, geo], None);
    let geo_ctx = context(cat, ResolutionConf::default());
    assert!(run(LogicalPlan::table(["geo"]).project(vec![c("x")]), &geo_ctx).is_ok());
    // The same shapes over evaluable types resolve.
    let fine = Expr::from(Literal::Null(None)).cast(DataType::Date);
    let analyzed = run(one_row().project(vec![fine]), &ctx()).unwrap();
    assert_eq!(analyzed.output[0].data_type, DataType::Date);
}

/// The refusal looks inside nested types: a `TIME` array element, map value
/// or struct field is `UNSUPPORTED_TIME_TYPE` too (Spark rejects the type
/// where it parses it), and a spatial field is the unevaluable-type error.
#[test]
fn time_and_geometry_inside_nested_types_are_refused() {
    use krishiv_types::{ArrayType, MapType, Srid, StructField, StructType};
    let ctx = ctx();
    let time = DataType::time(3).unwrap();
    let geometry = DataType::Geometry {
        srid: Srid::Fixed(0),
    };
    let cast_null = |to: DataType| {
        let e = Expr::from(Literal::Null(None)).cast(to);
        run(LogicalPlan::table(["t"]).project(vec![e]), &ctx)
            .err()
            .unwrap()
    };
    for nested in [
        DataType::Array(ArrayType::new(time.clone(), true).into()),
        DataType::Map(MapType::new(DataType::String, time.clone(), true).into()),
        DataType::Struct(StructType::new([StructField::new("a", time, true)])),
    ] {
        assert_eq!(
            cast_null(nested).error.error_class(),
            ErrorClass::UnsupportedTimeType
        );
    }
    // A spatial field evaluates.
    let in_struct = DataType::Struct(StructType::new([StructField::new("a", geometry, true)]));
    let typed_null = Expr::new(ExprNode::Literal(Literal::Null(Some(in_struct))));
    assert!(run(LogicalPlan::table(["t"]).project(vec![typed_null]), &ctx).is_ok());
}

/// Spark 4.2 rejects the `TIME` type with `UNSUPPORTED_TIME_TYPE` while
/// `spark.sql.timeType.enabled` is false (its default), before any cast is
/// checked: the same class for a cast of a column and a typed `NULL`.
#[test]
fn casting_to_time_is_unsupported_time_type() {
    let ctx = ctx();
    let time = DataType::time(6).unwrap();
    for e in [
        c("a").cast(time.clone()),
        Expr::new(ExprNode::Literal(Literal::Null(Some(time)))),
    ] {
        let r = run(LogicalPlan::table(["t"]).project(vec![e]), &ctx);
        assert_eq!(
            summary(&r),
            serde_json::json!(["UNSUPPORTED_TIME_TYPE", {}])
        );
    }
}

/// With `spark.sql.timeType.enabled` a `TIME '...'` literal is a `time(6)`
/// named by its normalized value; a time casts to another time and to
/// `int`/`bigint`, and to nothing else.
#[test]
fn time_literals_and_casts_with_the_type_enabled() {
    let on = ResolutionConf {
        time_type: true,
        ..ResolutionConf::default()
    };
    let ctx_on = context(MemoryCatalog::new("spark_catalog"), on);
    let one_row = || LogicalPlan::new(PlanNode::OneRowRelation);
    let literal = Expr::new(ExprNode::TypedLiteral {
        kind: crate::TypedLiteralKind::Time,
        value: "01:02:03 PM".to_owned(),
    });
    let analyzed = run(one_row().project(vec![literal.clone()]), &ctx_on).unwrap();
    assert_eq!(analyzed.output[0].data_type, DataType::time(6).unwrap());
    assert_eq!(analyzed.output[0].name, "TIME '13:02:03'");
    // Off (the default), the literal is refused.
    let r = run(one_row().project(vec![literal]), &ctx());
    assert_eq!(
        summary(&r),
        serde_json::json!(["UNSUPPORTED_TIME_TYPE", {}])
    );
    // Casts: string to time, time to time and to bigint resolve; a double
    // does not.
    let cast = |from: Expr, to: DataType| run(one_row().project(vec![from.cast(to)]), &ctx_on);
    let text = || Expr::from(Literal::String("10:00".to_owned()));
    assert!(cast(text(), DataType::time(3).unwrap()).is_ok());
    let time = || text().cast(DataType::time(6).unwrap());
    assert!(cast(time(), DataType::time(0).unwrap()).is_ok());
    assert!(cast(time(), DataType::Long).is_ok());
    assert!(cast(time(), DataType::Double).is_err());
    assert!(cast(Expr::from(Literal::Double(1.0)), DataType::time(6).unwrap()).is_err());
}

/// Spark's strength reduction for an integral expression compared with a
/// decimal literal (`DecimalPrecisionTypeCoercion.integralAndDecimalLiteral`
/// in Spark 4.2): the comparison narrows the literal to the `BIGINT` bound
/// (`floor`/`ceil`), and the output name shows the rewritten comparison.
/// `select 1 > 0.00001` names its column `(1 > 0)`, as the golden
/// `operators` statement 55 records.
#[test]
fn integral_compared_with_decimal_literal_narrows_to_bigint() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    // (expression, Spark 4.2 output name): every comparison direction, the
    // literal on either side, and negative literals.
    let cases = [
        (gt(int(1), dec("0.00001")), "(1 > 0)"),
        (ge(int(1), dec("0.00001")), "(1 >= 1)"),
        (lt(int(1), dec("0.00001")), "(1 < 1)"),
        (le(int(1), dec("0.00001")), "(1 <= 0)"),
        (gt(dec("0.00001"), int(1)), "(1 > 1)"),
        (ge(dec("0.00001"), int(1)), "(0 >= 1)"),
        (lt(dec("0.00001"), int(1)), "(0 < 1)"),
        (le(dec("0.00001"), int(1)), "(1 <= 1)"),
        (gt(int(1), dec("-0.00001")), "(1 > -1)"),
        (ge(int(1), dec("-0.00001")), "(1 >= 0)"),
        (lt(int(1), dec("-0.00001")), "(1 < 0)"),
        (le(int(1), dec("-0.00001")), "(1 <= -1)"),
        (gt(dec("-0.00001"), int(1)), "(0 > 1)"),
        (ge(dec("-0.00001"), int(1)), "(-1 >= 1)"),
        (lt(dec("-0.00001"), int(1)), "(-1 < 1)"),
        (le(dec("-0.00001"), int(1)), "(0 <= 1)"),
        (gt(long(5), dec("0.5")), "(5 > 0)"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
}

/// A decimal literal outside `BIGINT` range decides an integral comparison
/// outright (`true`/`false`), as Spark's rule does.
#[test]
fn integral_compared_with_huge_decimal_literal_is_decided() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    let cases = [
        (gt(int(1), dec("99999999999999999999.9")), "false"),
        (ge(int(1), dec("99999999999999999999.9")), "false"),
        (lt(int(1), dec("99999999999999999999.9")), "true"),
        (le(int(1), dec("99999999999999999999.9")), "true"),
        (gt(dec("99999999999999999999.9"), int(1)), "true"),
        (ge(dec("99999999999999999999.9"), int(1)), "true"),
        (lt(dec("99999999999999999999.9"), int(1)), "false"),
        (le(dec("99999999999999999999.9"), int(1)), "false"),
        (gt(int(1), dec("-99999999999999999999.9")), "true"),
        (lt(int(1), dec("-99999999999999999999.9")), "false"),
        (gt(dec("-99999999999999999999.9"), int(1)), "false"),
        (lt(dec("-99999999999999999999.9"), int(1)), "true"),
        // The `BIGINT` bounds themselves still narrow instead of deciding.
        (
            gt(int(1), dec("9223372036854775807")),
            "(1 > 9223372036854775807)",
        ),
        (
            lt(int(1), dec("-9223372036854775808")),
            "(1 < -9223372036854775808)",
        ),
        (gt(int(1), dec("9223372036854775807.5")), "false"),
        (lt(int(1), dec("9223372036854775807.5")), "true"),
        (gt(int(1), dec("-9223372036854775808.5")), "true"),
        (lt(int(1), dec("-9223372036854775808.5")), "false"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
}

/// The narrowing fires only for an integral expression against a decimal
/// literal: `=`, two decimals, floating-point and untyped `NULL` operands
/// keep their written form (up to the usual implicit casts, which the name
/// does not show).
#[test]
fn non_integral_or_non_literal_comparisons_keep_their_form() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    let float = Expr::new(ExprNode::Literal(Literal::Float(0.5)));
    let cases = [
        (eq(int(1), dec("0.00001")), "(1 = 0.00001)"),
        (gt(dec("1.5"), dec("0.00001")), "(1.5 > 0.00001)"),
        (gt(int(1), float), "(1 > 0.5)"),
        (gt(int(1), int(2)), "(1 > 2)"),
        (gt(null(), dec("0.00001")), "(NULL > 0.00001)"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
}

/// Columns narrow like literals do: the rewritten comparison, not the
/// written one, names the output.
#[test]
fn integral_columns_compared_with_decimal_literals_narrow() {
    let ctx = ctx();
    let name = |e: Expr| {
        let plan = LogicalPlan::table(["t"]).project(vec![e]);
        run(plan, &ctx).unwrap().output[0].name.clone()
    };
    let cases = [
        (gt(c("a"), dec("0.5")), "(a > 0)"),
        (lt(c("c"), dec("2.5")), "(c < 3)"),
        (ge(c("a"), dec("-0.5")), "(a >= 0)"),
    ];
    for (e, want) in cases {
        assert_eq!(name(e), want, "want {want}");
    }
    // Narrower integral types narrow the same way.
    let plan = LogicalPlan::table(["n"]).project(vec![gt(c("ti"), dec("0.5"))]);
    assert_eq!(run(plan, &ctx).unwrap().output[0].name, "(ti > 0)");
    // The narrowed comparison stays boolean.
    let plan = LogicalPlan::table(["t"]).project(vec![gt(c("a"), dec("0.5"))]);
    assert_eq!(datatypes(&run(plan, &ctx).unwrap()), [DataType::Boolean]);
}
