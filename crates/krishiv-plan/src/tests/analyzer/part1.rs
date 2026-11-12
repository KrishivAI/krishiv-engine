// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

#[test]
fn tpch_plans_resolve_as_spark_does() {
    let g = goldens();
    let ctx = ctx();
    for (i, q) in QUERIES.iter().enumerate() {
        let r = run(q(), &ctx);
        assert_eq!(summary(&r), spark(&g["tpch"][i]), "q{:02}", i + 1);
        let a = r.unwrap();
        assert!(a.plan.root().is_resolved(), "q{:02}", i + 1);
        if i == 14 {
            let text = a.plan.root().to_string();
            assert!(text.contains("CteRef revenue [supplier_no#"), "{text}");
        }
        // Resolving a resolved plan changes nothing.
        let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
        assert!(
            again.plan.structurally_eq(&a.plan),
            "q{:02} is a fixed point",
            i + 1
        );
        assert_eq!(again.output, a.output);
    }
}

#[test]
fn plan_id_references_resolve_as_spark_connect_does() {
    let g = goldens();
    let ctx = ctx();
    let mut raised = BTreeSet::new();
    for (case, (id, plan)) in g["plan_ids"].as_array().unwrap().iter().zip(dataframes(0)) {
        assert_eq!(case["id"], id);
        let r = run(plan, &ctx);
        assert_eq!(summary(&r), spark(case), "{id}");
        raised.insert(class(&r));
    }
    assert!(raised.contains("CANNOT_RESOLVE_DATAFRAME_COLUMN"));
    assert!(raised.contains("AMBIGUOUS_COLUMN_REFERENCE"));
}

/// `TIMESTAMP` and `TIMESTAMP_NTZ` literals are named as Spark's
/// `Literal.sql` prints them (`SELECT TIMESTAMP '2024-01-01 00:00:00'` in
/// Spark 4.2); SQL does not lower timestamp literals yet, so the plan is
/// built by hand.
#[test]
fn timestamp_literals_are_named_as_spark_sql() {
    let lit = |v: Literal| Expr::from(v);
    let r = run(
        LogicalPlan::table(["t"]).project(vec![
            lit(Literal::Timestamp(1_704_067_200_000_000)),
            lit(Literal::TimestampNtz(1_704_067_200_123_456)),
            lit(Literal::Timestamp(-1)),
        ]),
        &ctx(),
    );
    let names: Vec<String> = r.unwrap().output.iter().map(|c| c.name.clone()).collect();
    assert_eq!(
        names,
        [
            "TIMESTAMP '2024-01-01 00:00:00'",
            "TIMESTAMP_NTZ '2024-01-01 00:00:00.123456'",
            "TIMESTAMP '1969-12-31 23:59:59.999999'",
        ]
    );
}

#[test]
fn resolution_errors_and_names_match_spark() {
    let g = goldens();
    let cases = g["corpus"].as_array().unwrap();
    let mut raised = BTreeSet::new();
    for (sql, conf, plan) in corpus() {
        let case = cases.iter().find(|c| c["sql"] == sql).unwrap();
        let ctx = context(catalog(), conf);
        let r = run(plan, &ctx);
        assert_eq!(summary(&r), spark(case), "{sql}");
        raised.insert(class(&r));
    }
    // The analyzer's Spark error classes, each matched against Spark above
    // (the DataFrame ones in `plan_id_references_resolve_as_spark_connect_does`).
    for expected in [
        "AMBIGUOUS_REFERENCE",
        "ASSIGNMENT_ARITY_MISMATCH",
        "CANNOT_RESOLVE_STAR_EXPAND",
        "INVALID_EXTRACT_BASE_FIELD_TYPE",
        "INVALID_SUBQUERY_EXPRESSION.SCALAR_SUBQUERY_RETURN_MORE_THAN_ONE_OUTPUT_COLUMN",
        "INVALID_WHERE_CONDITION",
        "MISSING_AGGREGATION",
        "MISSING_GROUP_BY",
        "NESTED_AGGREGATE_FUNCTION",
        "TABLE_OR_VIEW_NOT_FOUND",
        "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        "UNRESOLVED_ROUTINE",
        "UNRESOLVED_USING_COLUMN_FOR_JOIN",
    ] {
        assert!(raised.contains(expected), "{expected} not raised");
    }
}

/// `mask` quotes the expanded call (with the omitted defaults materialized)
/// in input-type errors, as Spark does (`mask(b, a, x, n, NULL)`).
#[test]
fn mask_errors_quote_the_expanded_call() {
    let r = run(
        LogicalPlan::table(["t"]).project(vec![f("mask", vec![c("b"), c("a")])]),
        &ctx(),
    );
    let e = r.unwrap_err();
    assert_eq!(
        e.error.error_class().name(),
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE"
    );
    let p: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    // `a` is an `int`, not a `STRING`; the error quotes the call with the
    // defaults for `lowerChar`, `digitChar` and `otherChar`.
    assert_eq!(p["sqlExpr"], "\"mask(b, a, x, n, NULL)\"");
    assert_eq!(p["inputSql"], "\"a\"");
}

/// `e = e + e` 30 times (2^30 leaves as a tree): bound, validated and
/// named in time linear in its 31 distinct nodes; the generated name is
/// bounded.
#[test]
fn shared_expressions_stay_linear() {
    let doubled = || {
        (0..30).fold(Arc::new(c("a")), |e, _| {
            Arc::new(Expr::new(ExprNode::Function(crate::FunctionCall {
                within_group: Vec::new(),
                name: "+".to_owned(),
                qualifier: Vec::new(),
                arg_names: Vec::new(),
                args: vec![Arc::clone(&e), e],
                distinct: false,
                filter: None,
            })))
        })
    };
    let grouped = LogicalPlan::table(["t"]).aggregate(
        vec![Arc::unwrap_or_clone(doubled())],
        vec![f("count", vec![star()])],
    );
    let a = timed("grouped by a shared expression", || run(grouped, &ctx())).unwrap();
    assert_eq!(a.output[0].name, "count(1)");
    let named = LogicalPlan::new(PlanNode::Project {
        input: Arc::new(LogicalPlan::table(["t"])),
        exprs: vec![doubled()],
    });
    let a = timed("named after a shared expression", || run(named, &ctx())).unwrap();
    assert!(
        a.output[0].name.len() < 3 * (64 << 10),
        "{}",
        a.output[0].name.len()
    );
}

proptest! {
    /// Renumbered `plan_id`s resolve alike: same output, same key, same
    /// resolved plan.
    #[test]
    fn plan_ids_renumber_without_effect(base in 0_i64..1_000_000) {
        let ctx = ctx();
        for ((id, a), (_, b)) in dataframes(0).into_iter().zip(dataframes(base)) {
            let (a, b) = (Plan::try_new(a).unwrap(), Plan::try_new(b).unwrap());
            prop_assert_eq!(a.structural_hash(), b.structural_hash(), "{}", id);
            let (ra, rb) = (block_on(DefaultAnalyzer.analyze(&a, &ctx)), block_on(DefaultAnalyzer.analyze(&b, &ctx)));
            prop_assert_eq!(summary(&ra), summary(&rb), "{}", id);
            if let (Ok(ra), Ok(rb)) = (ra, rb) {
                prop_assert!(ra.plan.structurally_eq(&rb.plan), "{}", id);
                prop_assert_eq!(ra.input_key, rb.input_key);
            }
        }
    }

    /// Resolution is deterministic: the same input gives byte-identical
    /// resolved plans, whatever hasher seeds the process uses.
    #[test]
    fn resolution_is_deterministic(q in 0_usize..22) {
        let ctx = ctx();
        let a = run(QUERIES[q](), &ctx).unwrap();
        let b = run(QUERIES[q](), &ctx).unwrap();
        prop_assert_eq!(a.plan.canonical_encoding(), b.plan.canonical_encoding());
        prop_assert_eq!(a.input_key, b.input_key);
        prop_assert_eq!(a.context.digest(), b.context.digest());
    }
}

/// `p.join(p)` 30 times (2^31 - 1 relations as a tree) resolves in time
/// and size linear in its 31 distinct relations: each right side is a
/// `Shift` of the shared left side.
#[test]
fn self_joins_stay_linear() {
    let plan = (0..30).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
        let p = Arc::new(p);
        let side = |alias: &str| {
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input: Arc::clone(&p),
                alias: alias.to_owned(),
                columns: Vec::new(),
            })
        };
        side("l")
            .join(
                side("r"),
                JoinKind::Inner,
                JoinCondition::On(Arc::new(eq(c("l.a"), c("r.a")))),
            )
            .project(vec![c("l.a")])
    });
    let a = timed("30 self-joins", || run(plan, &ctx()).unwrap());
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx())).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
    assert!(
        distinct_relations(&a.plan) <= 30 * 5 + 2,
        "{}",
        distinct_relations(&a.plan)
    );
    assert_eq!(a.plan.root().to_string().matches("Shift +").count(), 30);
    // The 31 distinct relations of `p.join(p)` x30, without projections.
    let wide = timed("wide self-joins", || run(self_joins(12), &ctx()).unwrap());
    assert_eq!(wide.output.len(), 3 << 12);
    let ids: BTreeSet<u64> = wide.output.iter().map(|c| c.id.0).collect();
    assert_eq!(ids.len(), 3 << 12, "each instance has its own attributes");
}

/// Self-join instances reserve id spans that double per level: 70 levels
/// would pass 2^64, so the plan is too complex, never an overflow; a
/// crafted `Shift` far out of range is rejected too.
#[test]
fn instance_ids_never_overflow() {
    let deep = (0..70).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
        let p = Arc::new(p);
        let side = |alias: &str| {
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input: Arc::clone(&p),
                alias: alias.to_owned(),
                columns: Vec::new(),
            })
        };
        side("l")
            .join(side("r"), JoinKind::Cross, JoinCondition::None)
            .project(vec![c("l.a")])
    });
    assert_eq!(class(&run(deep, &ctx())), "FAILED_TO_PARSE_TOO_COMPLEX");
    let crafted = LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(LogicalPlan::table(["t"])),
        offset: u64::MAX - 1,
    });
    assert_eq!(class(&run(crafted, &ctx())), "FAILED_TO_PARSE_TOO_COMPLEX");
}

/// `WITH c0 AS (...), c1 AS (SELECT * FROM c0 JOIN c0), ...` 30 deep:
/// each reference is a `CteRef`, so nothing is copied.
#[test]
fn cte_chains_stay_linear() {
    let mut ctes = vec![Cte {
        name: "c0".to_owned(),
        plan: Arc::new(LogicalPlan::table(["u"]).project(vec![c("a")])),
    }];
    for i in 1..30 {
        let prev = format!("c{}", i - 1);
        let body = LogicalPlan::table([prev.as_str()])
            .alias("l")
            .join(
                LogicalPlan::table([prev.as_str()]).alias("r"),
                JoinKind::Inner,
                JoinCondition::On(Arc::new(eq(c("l.a"), c("r.a")))),
            )
            .project(vec![c("l.a")]);
        ctes.push(Cte {
            name: format!("c{i}"),
            plan: Arc::new(body),
        });
    }
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["c29"])),
        ctes,
    });
    let a = timed("CTE chain", || run(plan, &ctx()).unwrap());
    assert!(
        distinct_relations(&a.plan) < 30 * 8,
        "{}",
        distinct_relations(&a.plan)
    );
}

/// `df = df.where(df.exists())` 30 times: each plan is reached both as a
/// child and inside a subquery, 2^30 paths. Resolution reuses a relation
/// without outer references across scopes, so it stays linear.
#[test]
fn subqueries_over_shared_plans_stay_linear() {
    let plan = (0..30).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
        let p = Arc::new(p);
        LogicalPlan::new(PlanNode::Filter {
            input: Arc::clone(&p),
            predicate: Arc::new(Expr::new(ExprNode::Subquery {
                kind: crate::SubqueryKind::Exists,
                plan: p,
            })),
        })
    });
    let a = timed("30 nested EXISTS over one plan", || {
        run(plan, &ctx()).unwrap()
    });
    assert!(
        distinct_relations(&a.plan) <= 30 * 4,
        "{}",
        distinct_relations(&a.plan)
    );
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx())).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
}

/// A `Shift` is the fresh instance: its input resolves as itself even
/// where the input's ids meet an enclosing query's (which makes a direct
/// reference to the input a fresh instance). So the attributes bound over
/// the shift are the ones the retained node reads, `id + offset` of what
/// the input states, and re-analysis is a fixed point.
#[test]
fn shift_input_is_never_re_instanced() {
    let ctx = ctx();
    let p = run(LogicalPlan::table(["t"]).project(vec![c("a")]), &ctx).unwrap();
    let p = Arc::clone(p.plan.root());
    let stated: Vec<_> = p.resolved_output().unwrap();
    // `p WHERE EXISTS (SELECT a FROM <instance of p>)`: in the subquery,
    // `p`'s ids are those of the enclosing query.
    let shift = LogicalPlan::new(PlanNode::Shift {
        input: Arc::clone(&p),
        offset: 100,
    });
    let plan = LogicalPlan::new(PlanNode::Filter {
        input: Arc::clone(&p),
        predicate: Arc::new(Expr::subquery(
            crate::SubqueryKind::Exists,
            shift.project(vec![c("a")]),
        )),
    });
    let a = run(plan, &ctx).unwrap();
    let PlanNode::Filter { predicate, .. } = &a.plan.root().node else {
        panic!("{:?}", a.plan)
    };
    let ExprNode::Subquery { plan: sub, .. } = &predicate.node else {
        panic!("{predicate:?}")
    };
    let PlanNode::Project { input, exprs } = &sub.node else {
        panic!("{sub:?}")
    };
    assert!(matches!(input.node, PlanNode::Shift { offset: 100, .. }));
    let read: Vec<_> = exprs
        .iter()
        .map(|e| match &e.node {
            ExprNode::Resolved(crate::ResolvedExpr::Column(c)) => c.id,
            other => panic!("{other:?}"),
        })
        .collect();
    let shifted: Vec<_> = stated.iter().map(|c| crate::AttrId(c.id.0 + 100)).collect();
    assert_eq!(read, shifted, "bound over the attributes the shift reads");
    let output: Vec<_> = input
        .resolved_output()
        .unwrap()
        .iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(output, shifted);
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
    assert_eq!(again.output, a.output);
}

/// Equal input keys give equal resolved plans, however the input shares
/// its subtrees.
#[test]
fn resolution_ignores_how_the_input_shares() {
    let ctx = ctx();
    for (i, q) in QUERIES.iter().enumerate() {
        let (shared, copied) = (run(q(), &ctx).unwrap(), run(unshared(&q()), &ctx).unwrap());
        assert!(shared.plan.structurally_eq(&copied.plan), "q{:02}", i + 1);
    }
    let t = Arc::new(LogicalPlan::table(["t"]).alias("x"));
    let join = |l: Arc<LogicalPlan>, r: Arc<LogicalPlan>| {
        LogicalPlan::new(PlanNode::Join {
            left: l,
            right: r,
            kind: JoinKind::Cross,
            condition: JoinCondition::None,
        })
    };
    let shared = join(Arc::clone(&t), Arc::clone(&t));
    let copied = join(Arc::new(LogicalPlan::table(["t"]).alias("x")), t);
    let (a, b) = (
        Plan::try_new(shared).unwrap(),
        Plan::try_new(copied).unwrap(),
    );
    assert_eq!(a.structural_hash(), b.structural_hash());
    let (ra, rb) = (
        block_on(DefaultAnalyzer.analyze(&a, &ctx)).unwrap(),
        block_on(DefaultAnalyzer.analyze(&b, &ctx)).unwrap(),
    );
    assert!(ra.plan.structurally_eq(&rb.plan));
    assert_eq!(ra.plan.root().resolved_output(), Some(ra.output.clone()));
}

#[test]
fn the_policy_is_asked_once_per_table_and_fails_closed() {
    let deny = Ok(AccessDecision::Deny {
        reason: "Canary-7f3a9c".to_owned(),
    });
    let (ctx, policy) = with_policy(deny);
    let q21 = QUERIES[20]();
    let e = run(q21.clone(), &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::KrishivInsufficientPermissions
    );
    assert_eq!(
        e.error.to_string(),
        "[KRISHIV_INSUFFICIENT_PERMISSIONS] The principal `tester` lacks the SELECT privilege on \
         table `spark_catalog`.`default`.`lineitem`. SQLSTATE: 42501"
    );
    // The first denial stops the query.
    assert_eq!(policy.calls.lock().unwrap().len(), 1);
    // Four tables (lineitem read three times): four calls.
    let (ctx, policy) = with_policy(Ok(AccessDecision::Allow));
    run(q21.clone(), &ctx).unwrap();
    let calls = policy.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 4, "{calls:?}");
    // A policy error fails closed too, without its text.
    let (ctx, _) = with_policy(Err(krishiv_common::error::KrishivUnsupportedFeature {
        feature: "Canary-7f3a9c".to_owned(),
    }
    .into()));
    let e = run(q21, &ctx).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::KrishivInsufficientPermissions
    );
    assert!(!e.to_string().contains("Canary"));
    // Row filters and column masks are rejected until they can be applied.
    let (ctx, _) = with_policy(Ok(AccessDecision::AllowWith(RowColumnPolicy::default())));
    assert_eq!(
        class(&run(QUERIES[0](), &ctx)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

/// Every relation that reads data reaches the policy, under the catalog's
/// own identifier; a scan must match its table; a denied principal learns
/// nothing about existence; catalog failures carry no provider text.
#[test]
fn no_relation_bypasses_the_policy() {
    let deny = |ident: &str| {
        let mut c = ctx();
        c.policy = Arc::new(DenyOne(format!("table {ident}")));
        c
    };
    let lineitem = "`spark_catalog`.`default`.`lineitem`";
    let files = |paths: Vec<krishiv_common::secret::OptionValue>| {
        LogicalPlan::new(PlanNode::Read(ReadSource::Files {
            format: "parquet".to_owned(),
            paths,
            options: std::collections::BTreeMap::new(),
        }))
    };
    let key = [("l_orderkey", DataType::Long)];
    let shifted = LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(scan(&["LINEITEM"], &key)),
        offset: 10,
    });
    for plan in [
        LogicalPlan::table(["LINEITEM"]),
        LogicalPlan::table(["Default", "LineItem"]),
        scan(&["lineitem"], &key),
        scan(&["LINEITEM"], &key),
        scan(&["spark_catalog", "Default", "LineItem"], &key),
        shifted,
    ] {
        assert_eq!(
            class(&run(plan, &deny(lineitem))),
            "KRISHIV_INSUFFICIENT_PERMISSIONS"
        );
    }
    // A denied name is denied whether it exists or not.
    let nosuch = "`spark_catalog`.`default`.`nosuch`";
    assert_eq!(
        class(&run(LogicalPlan::table(["nosuch"]), &deny(nosuch))),
        "KRISHIV_INSUFFICIENT_PERMISSIONS"
    );
    // Deny-all reaches every kind of relation.
    let (all, _) = with_policy(Ok(AccessDecision::Deny {
        reason: String::new(),
    }));
    let path = || vec![krishiv_common::secret::OptionValue::from_user("path", "/d")];
    for plan in [
        LogicalPlan::table(["t"]),
        LogicalPlan::table(["nosuch"]),
        scan(&["t"], &[("a", DataType::Integer)]),
    ] {
        assert_eq!(class(&run(plan, &all)), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    }
    // Without a file resolver nothing is read by path; with one, every path
    // is asked about first (`tests::files`).
    let file_scan = LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: path(),
            options: std::collections::BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }));
    for plan in [files(path()), file_scan] {
        assert_eq!(class(&run(plan, &all)), "KRISHIV_UNSUPPORTED_FEATURE");
    }
    // Scans that name no table, a missing one, or invent columns fail closed.
    let ctx = ctx();
    assert_eq!(class(&run(scan(&[], &[]), &ctx)), "TABLE_OR_VIEW_NOT_FOUND");
    assert_eq!(
        class(&run(scan(&["nosuch"], &[]), &ctx)),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
    let invented = scan(&["t"], &[("secret_column", DataType::Integer)]);
    assert_eq!(class(&run(invented, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    let retyped = scan(&["t"], &[("a", DataType::Long)]);
    assert_eq!(class(&run(retyped, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
}

/// Every scan of a table is checked, not only the first; a spelling of a
/// name does not dodge a rule on another, existing table or not.
#[test]
fn every_scan_and_spelling_is_checked() {
    let ctx = ctx();
    let valid = scan(&["t"], &[("a", DataType::Integer)]);
    let invented = scan(&["t"], &[("secret_column", DataType::Integer)]);
    let crossed = valid
        .clone()
        .join(invented, JoinKind::Cross, JoinCondition::None);
    assert_eq!(class(&run(crossed, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    let mut versioned = scan(&["t"], &[("secret_column", DataType::Integer)]);
    if let PlanNode::Scan(s) = &mut versioned.node {
        s.version = Some(crate::TableVersion(3));
    }
    let crossed = valid.join(versioned, JoinKind::Cross, JoinCondition::None);
    assert_eq!(class(&run(crossed, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    let deny = |ident: &str| {
        let mut c = ctx.clone();
        c.policy = Arc::new(DenyOne(format!("table {ident}")));
        c
    };
    let nosuch = "`spark_catalog`.`default`.`nosuch`";
    for spelling in [["NoSuch"], ["NOSUCH"]] {
        assert_eq!(
            class(&run(LogicalPlan::table(spelling), &deny(nosuch))),
            "KRISHIV_INSUFFICIENT_PERMISSIONS"
        );
    }
}

/// A shift over a plan that is not resolved is not a resolved plan.
#[test]
fn a_shift_over_an_unresolved_plan_is_rejected() {
    let shifted = LogicalPlan::new(PlanNode::Shift {
        input: Arc::new(LogicalPlan::table(["t"])),
        offset: 10,
    });
    assert_eq!(class(&run(shifted, &ctx())), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn catalog_failures_hide_the_provider_text() {
    let mut ctx = ctx();
    ctx.catalogs = Arc::new(crate::analyzer::testing::OneCatalog(
        Arc::new(Failing),
        vec!["default".to_owned()],
    ));
    let e = run(LogicalPlan::table(["t"]), &ctx).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::KrishivCatalogFailure);
    assert!(!format!("{e} {e:?}").contains("Canary"));
}

#[test]
fn already_resolved_plans_are_checked_too() {
    let resolved = run(QUERIES[5](), &ctx()).unwrap().plan;
    let (ctx, policy) = with_policy(Ok(AccessDecision::Deny {
        reason: String::new(),
    }));
    let e = block_on(DefaultAnalyzer.analyze(&resolved, &ctx)).unwrap_err();
    assert_eq!(
        e.error.error_class(),
        ErrorClass::KrishivInsufficientPermissions
    );
    assert_eq!(policy.calls.lock().unwrap().len(), 1);
}

#[test]
fn pins_securables_and_fingerprint() {
    let a = run(QUERIES[20](), &ctx()).unwrap();
    let pins: Vec<_> = a.pins.iter().map(|(t, v)| (t.to_string(), v.0)).collect();
    assert_eq!(
        pins,
        [("`spark_catalog`.`default`.`lineitem`".to_owned(), 7)]
    );
    // lineitem is read once at the pin; its other two uses (in subqueries)
    // are instances of that scan.
    let text = a.plan.root().to_string();
    assert_eq!(text.matches("@ version 7").count(), 1, "{text}");
    assert_eq!(text.matches("Shift +").count(), 2, "{text}");
    let used: std::collections::BTreeMap<String, Vec<String>> = a
        .securables
        .iter()
        .map(|s| (s.securable.to_string(), s.columns.iter().cloned().collect()))
        .collect();
    assert_eq!(used.len(), 4);
    assert_eq!(
        used["table `spark_catalog`.`default`.`supplier`"],
        ["s_name", "s_nationkey", "s_suppkey"]
    );
    // `EXISTS (SELECT * FROM lineitem ...)` references every column.
    assert_eq!(used["table `spark_catalog`.`default`.`lineitem`"].len(), 16);
    let q6 = run(QUERIES[5](), &ctx()).unwrap();
    assert_eq!(
        q6.securables[0].columns.iter().collect::<Vec<_>>(),
        ["l_discount", "l_extendedprice", "l_quantity", "l_shipdate"]
    );
    let mut other = ctx();
    other.conf.case_sensitive = true;
    let b = run(QUERIES[20](), &other).unwrap();
    assert_ne!(a.context.digest(), b.context.digest());
    assert_eq!(a.context.catalog_epochs.len(), 1);
}

/// Every input of resolution enters the context fingerprint: changing any
/// one changes the digest.
#[test]
fn every_context_input_changes_the_key() {
    use crate::analyzer::BuiltinFunctions;
    use crate::analyzer::testing::OneCatalog;
    use crate::analyzer::{PrincipalKind, SessionVariable, VariableSnapshot, fingerprint};
    let digest = |c: &AnalysisContext, epochs: &[(&str, u64)]| {
        let e = epochs.iter().map(|(n, v)| ((*n).to_owned(), *v)).collect();
        fingerprint(c, e).digest()
    };
    let base = ctx();
    let one = |name: &str, ns: &str| -> Arc<dyn crate::analyzer::CatalogResolver> {
        Arc::new(OneCatalog(
            Arc::new(MemoryCatalog::new(name)),
            vec![ns.to_owned()],
        ))
    };
    let mut variants: Vec<(&str, AnalysisContext)> = Vec::new();
    let mut v = |what, f: &dyn Fn(&mut AnalysisContext)| {
        let mut c = base.clone();
        f(&mut c);
        variants.push((what, c));
    };
    v("profile", &|c| c.profile = SparkProfile::V4_1);
    v("caseSensitive", &|c| c.conf.case_sensitive ^= true);
    v("ansi", &|c| c.conf.ansi ^= true);
    v("legacyTimeParser", &|c| c.conf.legacy_time_parser ^= true);
    v("groupByAliases", &|c| c.conf.group_by_aliases ^= true);
    v("groupByOrdinal", &|c| c.conf.group_by_ordinal ^= true);
    v("orderByOrdinal", &|c| c.conf.order_by_ordinal ^= true);
    v("lateralColumnAlias", &|c| {
        c.conf.lateral_column_alias ^= true;
    });
    v("current catalog", &|c| c.catalogs = one("other", "default"));
    v("current namespace", &|c| {
        c.catalogs = one("spark_catalog", "db2");
    });
    v("catalogs epoch", &|c| {
        let inner = OneCatalog(
            Arc::new(MemoryCatalog::new("spark_catalog")),
            vec!["default".to_owned()],
        );
        c.catalogs = Arc::new(Epoch(inner, 9));
    });
    v("functions", &|c| {
        c.functions = Arc::new(Epoch(BuiltinFunctions::new(), 9));
    });
    v("sql front end", &|c| c.sql = Some(Arc::new(Frontend)));
    v("principal", &|c| {
        c.principal = Arc::new(Principal::new("other", PrincipalKind::User));
    });
    v("principal kind", &|c| {
        c.principal = Arc::new(Principal::new("tester", PrincipalKind::Service));
    });
    v("groups", &|c| {
        let mut p = Principal::new("tester", PrincipalKind::User);
        p.groups = vec!["admins".to_owned()];
        c.principal = Arc::new(p);
    });
    v("tenant", &|c| {
        let mut p = Principal::new("tester", PrincipalKind::User);
        p.tenant = Some("acme".to_owned());
        c.principal = Arc::new(p);
    });
    v("policy", &|c| c.policy = Arc::new(DenyOne(String::new())));
    v("variables", &|c| {
        c.variables = Arc::new(VariableSnapshot {
            variables: [(
                "v".to_owned(),
                SessionVariable {
                    data_type: DataType::Integer,
                    value: Literal::Integer(1).into(),
                },
            )]
            .into_iter()
            .collect(),
            version: 1,
            ..VariableSnapshot::default()
        });
    });
    let key = digest(&base, &[("spark_catalog", 1)]);
    assert_ne!(
        key,
        digest(&base, &[("spark_catalog", 2)]),
        "catalog epochs"
    );
    for (what, c) in &variants {
        assert_ne!(key, digest(c, &[("spark_catalog", 1)]), "{what}");
    }
    assert_eq!(base.catalogs.epoch(), 0);
}

#[test]
fn resolution_reads_only_its_configs() {
    let read = Mutex::new(Vec::new());
    let conf = ResolutionConf::from_conf(|k| {
        if let Ok(mut r) = read.lock() {
            r.push(k.to_owned());
        }
        (k == "spark.sql.caseSensitive").then_some("TRUE")
    });
    assert!(conf.case_sensitive);
    assert_eq!(read.into_inner().unwrap(), ResolutionConf::KEYS);
    assert_eq!(
        ResolutionConf::from_conf(|_| None),
        ResolutionConf::default()
    );
    // The timestamp type and session zone flow through `from_conf`: unset
    // is UTC, fixed offsets and regions resolve, unknown zones are `None`.
    let ntz =
        ResolutionConf::from_conf(|k| (k == "spark.sql.timestampType").then_some("TIMESTAMP_NTZ"));
    assert!(ntz.timestamp_ntz);
    assert_eq!(ntz.session_zone, Some(SessionZone::Fixed(0)));
    let plus =
        ResolutionConf::from_conf(|k| (k == "spark.sql.session.timeZone").then_some("+05:30"));
    assert_eq!(plus.session_zone, Some(SessionZone::Fixed(19_800)));
    assert!(!plus.timestamp_ntz);
    let region = ResolutionConf::from_conf(|k| {
        (k == "spark.sql.session.timeZone").then_some("America/Los_Angeles")
    });
    assert!(matches!(region.session_zone, Some(SessionZone::Region(_))));
    assert_eq!(
        region.session_zone,
        SessionZone::resolve("America/Los_Angeles")
    );
    let bad =
        ResolutionConf::from_conf(|k| (k == "spark.sql.session.timeZone").then_some("Mars/Base"));
    assert_eq!(bad.session_zone, None);
    // Analysis has no configuration source besides `AnalysisContext::conf`
    // (the type holds none): the same context gives the same result.
    let c = ctx();
    assert!(
        run(QUERIES[0](), &c)
            .unwrap()
            .plan
            .structurally_eq(&run(QUERIES[0](), &c).unwrap().plan)
    );
}

#[test]
fn sql_nodes_resolve_through_the_front_end() {
    let sql = |q: &str, args| {
        LogicalPlan::new(PlanNode::Sql {
            query: q.to_owned(),
            args,
        })
    };
    let mut ctx = ctx();
    assert_eq!(
        class(&run(sql("x", SqlArgs::None), &ctx)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    ctx.sql = Some(Arc::new(Frontend));
    let a = run(sql("q6", SqlArgs::None), &ctx).unwrap();
    assert_eq!(a.output[0].name, "revenue");
    assert_eq!(a.query_text.unwrap().as_str(), "*********(redacted)");
    let filtered = sql("q6", SqlArgs::None)
        .filter(Expr::builtin(crate::Builtin::IsNotNull, vec![c("revenue")]));
    assert!(run(filtered, &ctx).unwrap().query_text.is_none());
    assert_eq!(
        class(&run(sql("bad", SqlArgs::None), &ctx)),
        "PARSE_EMPTY_STATEMENT"
    );
    let args = SqlArgs::Positional(vec![Arc::new(int(1))]);
    assert_eq!(
        class(&run(sql("q6", args), &ctx)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
}

#[test]
fn unsupported_inputs_fail_explicitly() {
    let ctx = ctx();
    let files = LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: "parquet".to_owned(),
        paths: vec![krishiv_common::secret::OptionValue::from_user("path", "/d")],
        options: std::collections::BTreeMap::new(),
    }));
    let sha = LogicalPlan::table(["t"]).project(vec![Expr::function("sentences", vec![c("b")])]);
    for plan in [files, sha] {
        assert_eq!(class(&run(plan, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
    }
    // An untyped NULL is the void type; an interval is not supported yet.
    let null = LogicalPlan::table(["t"]).project(vec![Literal::Null(None).into()]);
    let out = run(null, &ctx).unwrap().output;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].data_type, krishiv_types::DataType::Null);
    assert!(out[0].nullable);
    let interval = LogicalPlan::table(["t"]).project(vec![
        Literal::YearMonthInterval {
            months: 1,
            start: crate::YearMonthField::Year,
            end: crate::YearMonthField::Month,
        }
        .into(),
    ]);
    let out = run(interval.clone(), &ctx).unwrap().output;
    assert_eq!(
        out[0].data_type,
        krishiv_types::DataType::DEFAULT_YEAR_MONTH_INTERVAL
    );
    assert!(!out[0].nullable);
    // `spark.sql.legacy.interval.enabled` makes Spark's literals calendar
    // intervals, which is refused rather than answered differently.
    let legacy = context(
        catalog(),
        ResolutionConf {
            legacy_interval: true,
            ..ResolutionConf::default()
        },
    );
    assert_eq!(
        class(&run(interval, &legacy)),
        "KRISHIV_UNSUPPORTED_FEATURE"
    );
    // With ordinals off, an integer key is a constant.
    let off = context(
        catalog(),
        ResolutionConf {
            group_by_ordinal: false,
            ..ResolutionConf::default()
        },
    );
    let constant =
        LogicalPlan::table(["t"]).aggregate(vec![int(1)], vec![f("count", vec![star()])]);
    assert_eq!(class(&run(constant, &off)), "OK");
    // A crafted, already resolved scan of files is checked like a read: with
    // no file resolver, it is not read at all (`tests::files` covers the
    // policy).
    let (deny, _) = with_policy(Ok(AccessDecision::Deny {
        reason: String::new(),
    }));
    let scan = LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: vec![krishiv_common::secret::OptionValue::from_user("path", "/d")],
            options: std::collections::BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }));
    assert_eq!(class(&run(scan, &deny)), "KRISHIV_UNSUPPORTED_FEATURE");
    // A function newer than the session's Spark version does not exist.
    let mut old = ctx.clone();
    old.profile = SparkProfile::V3_5;
    let randstr = LogicalPlan::table(["t"]).project(vec![Expr::function("randstr", vec![int(3)])]);
    assert_eq!(class(&run(randstr.clone(), &old)), "UNRESOLVED_ROUTINE");
    // `randstr` is implemented: it resolves on current versions.
    assert_eq!(class(&run(randstr, &ctx)), "OK");
    // The argument count is checked first, with Spark's `expectedNum` text.
    let upper =
        LogicalPlan::table(["t"]).project(vec![Expr::function("upper", vec![int(1), int(2)])]);
    let err = run(upper, &ctx).unwrap_err();
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

/// The cast-named functions (`float(x)`, `int(x)`, ...): Spark analyzes
/// each call as `CAST(x AS target)`, so the analyzer rewrites the call
/// to a cast during analysis and the plan only ever holds the `Cast`
/// node. The output column is named by the argument, as Spark names it.
#[test]
fn cast_named_functions_rewrite_to_casts() {
    use krishiv_types::DecimalType;

    let ctx = ctx();
    let double = |v: f64| Expr::from(Literal::Double(v));
    let long = |v: i64| Expr::from(Literal::Long(v));
    let boolean = |v: bool| Expr::from(Literal::Boolean(v));
    let decimal = |p: u32, s: u32| DataType::Decimal(DecimalType::try_new(p, s).unwrap());
    // (call, type, output name, nullable).
    let cases: Vec<(Expr, DataType, &str, bool)> = vec![
        (
            f("float", vec![double(15.0)]),
            DataType::Float,
            "15.0",
            false,
        ),
        (f("double", vec![s("1.5")]), DataType::Double, "1.5", true),
        (f("int", vec![s("7")]), DataType::Integer, "7", true),
        (f("boolean", vec![int(1)]), DataType::Boolean, "1", false),
        (f("tinyint", vec![long(1)]), DataType::Byte, "1", false),
        (f("smallint", vec![long(1)]), DataType::Short, "1", false),
        (
            f("bigint", vec![boolean(true)]),
            DataType::Long,
            "true",
            false,
        ),
        (
            f("decimal", vec![s("1.234")]),
            decimal(10, 0),
            "1.234",
            true,
        ),
        (f("string", vec![int(42)]), DataType::String, "42", false),
        (f("binary", vec![s("x")]), DataType::Binary, "x", false),
        (
            f("date", vec![s("2024-01-02")]),
            DataType::Date,
            "2024-01-02",
            true,
        ),
        (
            f("timestamp", vec![s("2024-01-02 12:34:56")]),
            DataType::Timestamp,
            "2024-01-02 12:34:56",
            true,
        ),
        // A cast of a column keeps the column's name.
        (f("int", vec![c("c")]), DataType::Integer, "c", true),
        // Names resolve ASCII case-insensitively.
        (f("FLOAT", vec![int(1)]), DataType::Float, "1", false),
    ];
    for (call, data_type, name, nullable) in cases {
        let plan = LogicalPlan::table(["t"]).project(vec![call]);
        let a = run(plan, &ctx).unwrap();
        assert_eq!(a.output.len(), 1);
        assert_eq!(a.output[0].data_type, data_type);
        assert_eq!(a.output[0].name, name);
        assert_eq!(a.output[0].nullable, nullable);
        // The plan holds a cast, never a function call.
        let text = a.plan.root().to_string();
        assert!(text.contains("CAST("), "{text}");
        assert!(!text.contains("float(") && !text.contains("int("), "{text}");
    }
    // Resolving a resolved plan changes nothing.
    let plan = LogicalPlan::table(["t"]).project(vec![f("int", vec![s("7")])]);
    let a = run(plan, &ctx).unwrap();
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
}

/// The arity error of a cast-named function quotes the call, with
/// Spark's `expectedNum` text.
#[test]
fn cast_named_functions_check_arity() {
    let ctx = ctx();
    for (args, actual) in [(Vec::new(), "0"), (vec![int(1), int(2)], "2")] {
        let plan = LogicalPlan::table(["t"]).project(vec![f("float", args)]);
        let err = run(plan, &ctx).unwrap_err();
        assert_eq!(
            err.error.error_class().name(),
            "WRONG_NUM_ARGS.WITHOUT_SUGGESTION"
        );
        let params = err.error.message_parameters();
        let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
        assert_eq!(get("functionName"), "`float`");
        assert_eq!(get("expectedNum"), "1");
        assert_eq!(get("actualNum"), actual);
    }
}

/// A pair the cast path does not support fails as `CAST(x AS T)` fails,
/// quoting the argument as Spark does.
#[test]
fn cast_named_functions_fail_like_casts() {
    let ctx = ctx();
    let binary = Expr::from(Literal::Binary(vec![1]));
    let plan = LogicalPlan::table(["t"]).project(vec![f("boolean", vec![binary])]);
    let err = run(plan, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION"
    );
    let params = err.error.message_parameters();
    let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
    assert_eq!(get("sqlExpr"), "\"X'01'\"");
    assert_eq!(get("srcType"), "\"BINARY\"");
    assert_eq!(get("targetType"), "\"BOOLEAN\"");
    // A suggestion or a config hint fails as `CAST` does, quoting the
    // argument.
    let plan = LogicalPlan::table(["t"]).project(vec![f("bigint", vec![date("2024-01-02")])]);
    let err = run(plan, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITH_FUNC_SUGGESTION"
    );
    let params = err.error.message_parameters();
    let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
    assert_eq!(get("sqlExpr"), "\"CAST(2024-01-02 AS DATE)\"");
    assert_eq!(get("functionNames"), "`UNIX_DATE`");
    let plan = LogicalPlan::table(["t"]).project(vec![f("binary", vec![int(1)])]);
    let err = run(plan, &ctx).unwrap_err();
    assert_eq!(
        err.error.error_class().name(),
        "DATATYPE_MISMATCH.CAST_WITH_CONF_SUGGESTION"
    );
    let params = err.error.message_parameters();
    let get = |k: &str| params.iter().find(|(k2, _)| *k2 == k).unwrap().1.clone();
    assert_eq!(get("sqlExpr"), "\"1\"");
    assert_eq!(get("srcType"), "\"INT\"");
    assert_eq!(get("targetType"), "\"BINARY\"");
    // Without ANSI the same pair is allowed, as `CAST` allows it.
    let legacy = context(
        catalog(),
        ResolutionConf {
            ansi: false,
            ..ResolutionConf::default()
        },
    );
    let plan = LogicalPlan::table(["t"]).project(vec![f("binary", vec![int(1)])]);
    let a = run(plan, &legacy).unwrap();
    assert_eq!(a.output[0].data_type, DataType::Binary);
    assert_eq!(a.output[0].name, "1");
    // `timestamp_ntz` is not a function.
    let plan = LogicalPlan::table(["t"]).project(vec![f("timestamp_ntz", vec![int(1)])]);
    assert_eq!(class(&run(plan, &ctx)), "UNRESOLVED_ROUTINE");
    // A malformed string analyzes in both modes (it fails at run time):
    // nullability follows the cast, which may yield NULL.
    for ansi in [true, false] {
        let ctx = context(
            catalog(),
            ResolutionConf {
                ansi,
                ..ResolutionConf::default()
            },
        );
        let plan = LogicalPlan::table(["t"]).project(vec![f("int", vec![s("x")])]);
        let a = run(plan, &ctx).unwrap();
        assert_eq!(a.output[0].data_type, DataType::Integer);
        assert_eq!(a.output[0].name, "x");
        assert!(a.output[0].nullable);
    }
}

/// `DISTINCT` and `FILTER` have no cast form: a cast-named function with
/// either is unsupported syntax on the function, as for any other
/// non-aggregate function.
#[test]
fn cast_named_functions_reject_distinct_and_filter() {
    let ctx = ctx();
    let distinct = Expr::new(ExprNode::Function(crate::FunctionCall {
        within_group: Vec::new(),
        name: "float".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("a"))],
        distinct: true,
        filter: None,
    }));
    let r = run(LogicalPlan::table(["t"]).project(vec![distinct]), &ctx);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX",
            {"prettyName": "`cast`", "syntax": "DISTINCT"}
        ])
    );
    let filtered = Expr::new(ExprNode::Function(crate::FunctionCall {
        within_group: Vec::new(),
        name: "float".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("a"))],
        distinct: false,
        filter: Some(Arc::new(Literal::Boolean(true).into())),
    }));
    let r = run(LogicalPlan::table(["t"]).project(vec![filtered]), &ctx);
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX",
            {"prettyName": "`cast`", "syntax": "FILTER CLAUSE"}
        ])
    );
}

/// A column a projection dropped comes back from below it, as Spark's
/// missing references do, down to an aggregate's grouping attributes
/// (which the aggregate then outputs); the result resolves to itself.
#[test]
fn missing_references_reach_grouping_attributes() {
    let ctx = ctx();
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("a")], vec![f("max", vec![c("c")]).alias("m")])
        .project(vec![c("m")])
        .sort(vec![crate::SortOrder::asc(c("a"))]);
    let a = run(plan, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["m"]
    );
    assert_eq!(
        a.plan.root().to_string(),
        "Project [m#3]\n\
         +- Sort [a#0 ASC NULLS FIRST], true\n   \
         +- Project [m#3, a#0]\n      \
         +- Aggregate [a#0], [max(c#2) AS m#3, a#0]\n         \
         +- Scan spark_catalog.default.t [a#0, b#1, c#2]\n"
    );
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
}
