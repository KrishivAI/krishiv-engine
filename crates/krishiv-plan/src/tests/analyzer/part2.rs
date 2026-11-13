// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// In a subquery, a column a projection dropped hides an outer column of
/// its name (Spark resolves missing references before outer ones); a name
/// only the enclosing query has still resolves there.
#[test]
fn missing_references_come_before_outer_ones() {
    let ctx = ctx();
    let inner = |name: &str| {
        LogicalPlan::table(["u"])
            .project(vec![c("a")])
            .filter(eq(c(name), c("a")))
    };
    let query = |name: &str| {
        LogicalPlan::table(["t"])
            .project(vec![c("a").alias("d"), c("b")])
            .filter(Expr::subquery(crate::SubqueryKind::Exists, inner(name)))
    };
    let shown = |name: &str| run(query(name), &ctx).unwrap().plan.root().to_string();
    // `d` is u's own column, dropped below the filter, not the outer alias.
    let own = shown("d");
    assert!(own.contains("Filter (CAST(d#"), "{own}");
    assert!(!own.contains("outer("), "{own}");
    // `b` exists only in the enclosing query: the subquery is correlated,
    // a semi join on it.
    let outer = shown("b");
    assert!(outer.contains("Join LeftSemi, (CAST(b#1 "), "{outer}");
}

/// Session variables resolve after columns and outer references: a name
/// a column has is the column, a qualified `session.v` skips the columns,
/// and anything else stays an unresolved column.
#[test]
fn session_variables_resolve_after_columns() {
    use crate::analyzer::{SessionVariable, VariableSnapshot};
    use std::collections::BTreeMap;
    let catalog = || {
        MemoryCatalog::new("spark_catalog").with_table(
            &["default"],
            "t",
            vec![TableColumn::new("var1".to_owned(), DataType::String, true)],
            None,
        )
    };
    let variables = || {
        let mut ctx = context(catalog(), ResolutionConf::default());
        ctx.variables = Arc::new(VariableSnapshot {
            variables: BTreeMap::from([(
                "var1".to_owned(),
                SessionVariable {
                    data_type: DataType::Integer,
                    value: Literal::Integer(1).into(),
                },
            )]),
            version: 3,
            ..VariableSnapshot::default()
        });
        ctx
    };
    // A name a column has is the column, even when a variable has it too.
    let plan = LogicalPlan::table(["t"]).project(vec![c("var1")]);
    let a = run(plan, &variables()).unwrap();
    assert_eq!(a.output[0].data_type, DataType::String);
    // A qualified `session.v` skips the columns for the variable.
    for name in ["session.var1", "system.session.var1", "SESSION.VAR1"] {
        let plan = LogicalPlan::table(["t"]).project(vec![c(name)]);
        let a = run(plan, &variables()).unwrap();
        assert_eq!(a.output[0].data_type, DataType::Integer, "{name}");
    }
    // Anything else stays an unresolved column, never a variable error.
    // (Outer references before variables are covered end to end, where a
    // correlated `EXISTS` reads the outer column.)
    for name in ["nope", "ses.var1", "b.ses.var1", "var1.var1"] {
        let plan = LogicalPlan::new(PlanNode::OneRowRelation).project(vec![c(name)]);
        assert_eq!(
            class(&run(plan, &variables())),
            "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION",
            "{name}"
        );
    }
}

/// A changed variable version changes the context fingerprint, so a
/// cached resolved plan is never reused after a variable changed.
#[test]
fn variable_versions_change_the_fingerprint() {
    use crate::analyzer::{SessionVariable, VariableSnapshot, fingerprint};
    use std::collections::BTreeMap;
    let digest = |c: &AnalysisContext| fingerprint(c, BTreeMap::new()).digest();
    let base = ctx();
    let key = digest(&base);
    let mut changed = base.clone();
    changed.variables = Arc::new(VariableSnapshot {
        variables: BTreeMap::from([(
            "var1".to_owned(),
            SessionVariable {
                data_type: DataType::Integer,
                value: Literal::Integer(1).into(),
            },
        )]),
        version: base.variables.version + 1,
        ..VariableSnapshot::default()
    });
    assert_ne!(key, digest(&changed));
}

/// A subexpression shared through `Arc` inside a lateral alias item and a
/// sort key over a dropped column is bound and walked once.
#[test]
fn shared_subexpressions_in_rewrites() {
    let ctx = ctx();
    let twice = |name: &str| {
        let x = Arc::new(c(name));
        Expr::new(ExprNode::Function(crate::FunctionCall {
            within_group: Vec::new(),
            name: "+".to_owned(),
            qualifier: Vec::new(),
            arg_names: Vec::new(),
            args: vec![Arc::clone(&x), x],
            distinct: false,
            filter: None,
        }))
    };
    let lateral =
        LogicalPlan::table(["t"]).project(vec![f("max", vec![c("a")]).alias("m"), twice("m")]);
    let a = run(lateral, &ctx).unwrap();
    assert_eq!(
        a.output[1].name,
        "(lateralAliasReference(m) + lateralAliasReference(m))"
    );
    let sorted = LogicalPlan::table(["t"])
        .project(vec![c("a")])
        .sort(vec![crate::SortOrder::asc(twice("c"))]);
    let a = run(sorted, &ctx).unwrap();
    assert!(
        a.plan
            .root()
            .to_string()
            .contains("Sort [(c#2 + c#2) ASC NULLS FIRST]"),
        "{}",
        a.plan.root()
    );
}

/// A registry entry prints its identity, kind and inputs (never its result
/// closure).
#[test]
fn function_entries_print_their_signature() {
    use crate::analyzer::{FunctionLookup, FunctionRegistry as _};
    let FunctionLookup::Found(entry) = crate::analyzer::BuiltinFunctions::new()
        .lookup(&["substring".to_owned()], SparkProfile::default())
    else {
        panic!("substring is registered");
    };
    let text = format!("{entry:?}");
    assert!(
        text.contains("inputs: Expects([OneOf([String, Binary]), Exact(Integer), Exact(Integer)])"),
        "{text}"
    );
    // Built-ins are unqualified; a name only Spark has is known.
    let b = crate::analyzer::BuiltinFunctions::default();
    let qualified = ["builtin".to_owned(), "upper".to_owned()];
    assert!(matches!(
        b.lookup(&qualified, SparkProfile::V4_2),
        FunctionLookup::NotFound
    ));
    let uuid = ["uuid".to_owned()];
    assert!(matches!(
        b.lookup(&uuid, SparkProfile::V4_2),
        FunctionLookup::Found(_)
    ));
    assert_eq!(b.version(), 0);
}

#[test]
fn deep_plans_resolve_on_a_small_stack_and_added_nodes_count() {
    on_small_stack(|| {
        let ctx = ctx();
        // Nested scalar subqueries: the resolver recurses per level.
        let deep =
            (0..MAX_DEPTH / 3 - 2).fold(LogicalPlan::table(["t"]).project(vec![c("a")]), |p, _| {
                LogicalPlan::table(["t"])
                    .project(vec![Expr::subquery(crate::SubqueryKind::Scalar, p)])
            });
        assert_eq!(class(&run(deep, &ctx)), "OK");
        // A column list on an alias adds a projection: one level too many.
        let wide = (0..MAX_DEPTH - 2).fold(LogicalPlan::table(["t"]), |p, _| {
            p.filter(gt(c("a"), int(0)))
        });
        let renamed = LogicalPlan::new(PlanNode::SubqueryAlias {
            input: Arc::new(wide),
            alias: "x".to_owned(),
            columns: vec!["p".to_owned(), "q".to_owned(), "r".to_owned()],
        });
        assert_eq!(class(&run(renamed, &ctx)), "FAILED_TO_PARSE_TOO_COMPLEX");
    });
}

#[test]
fn principals_securables_and_contexts_print_safely() {
    let local = Principal::local_user();
    assert!(local.id().starts_with("user:-/"));
    let mut service = Principal::new("etl", crate::analyzer::PrincipalKind::Service);
    service.tenant = Some("acme".to_owned());
    service.groups = vec!["b".to_owned(), "a".to_owned(), "a".to_owned()];
    assert_eq!(service.id(), "service:`acme`/`etl`[`a`.`b`]");
    let ident = crate::analyzer::TableIdent {
        catalog: "c".to_owned(),
        namespace: vec!["d".to_owned()],
        name: "v`w".to_owned(),
    };
    let url = krishiv_common::secret::OptionValue::from_user("path", "s3a://k:Canary-7f3a9c@b/p");
    let shown: Vec<String> = [
        Securable::View(ident),
        Securable::Function(vec!["f".to_owned()]),
        Securable::Path(url),
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    assert_eq!(
        shown,
        [
            "view `c`.`d`.`v``w`",
            "function `f`",
            "path 's3a://k:*********(redacted)@b/p'"
        ]
    );
    assert!(format!("{:?}", ctx()).starts_with("AnalysisContext"));
    // Errors keep the span of the node that raised them.
    let sql = "SELECT x FROM t";
    let at = Origin::sql(crate::SourceId(0), crate::Span::new(7, 8));
    let plan = LogicalPlan::table(["t"]).project(vec![c("x").with_origin(at)]);
    let e = run(plan, &ctx()).unwrap_err();
    let q = e.query_context(&RedactedSql::opaque(sql)).unwrap();
    assert!(
        matches!(
            q,
            QueryContext::Sql {
                start: 7,
                stop: 7,
                ..
            }
        ),
        "{q:?}"
    );
    // A DataFrame error names the relation that raised it.
    let df = LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(5)));
    let df_error = run(df.project(vec![col("x", 5)]), &ctx()).unwrap_err();
    assert_eq!(
        df_error.query_context(&RedactedSql::opaque(sql)),
        Some(QueryContext::DataFrame { plan_id: PlanId(5) })
    );
    assert_eq!(
        KrishivError::from(e).error_class(),
        ErrorClass::UnresolvedColumnWithSuggestion
    );
}

#[test]
fn one_snapshot_per_table_per_query() {
    let committing = Committing(catalog(), Mutex::new(0));
    let mut ctx = ctx();
    ctx.catalogs = Arc::new(crate::analyzer::testing::OneCatalog(
        Arc::new(committing),
        vec!["default".to_owned()],
    ));
    // `t` and `T` load twice; both read the version pinned first.
    let plan = LogicalPlan::table(["t"])
        .alias("x")
        .join(
            LogicalPlan::table(["T"]).alias("y"),
            JoinKind::Cross,
            JoinCondition::None,
        )
        .project(vec![c("x.a"), c("y.a")]);
    let a = run(plan, &ctx).unwrap();
    assert_eq!(a.pins.values().map(|v| v.0).collect::<Vec<_>>(), [1]);
    assert_eq!(a.plan.root().to_string().matches("@ version 1").count(), 2);
}

#[test]
fn stars_in_calls_and_elsewhere() {
    let ctx = ctx();
    let concat = LogicalPlan::table(["u"]).project(vec![f("concat", vec![star()])]);
    assert_eq!(run(concat, &ctx).unwrap().output[0].name, "concat(a, d)");
    let filtered = LogicalPlan::table(["t"]).filter(star());
    assert_eq!(
        class(&run(filtered, &ctx)),
        "INVALID_USAGE_OF_STAR_OR_REGEX"
    );
    // A USING column two left-side columns could mean.
    let joined = LogicalPlan::table(["t"])
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Cross,
            JoinCondition::None,
        )
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::Using(vec!["a".to_owned()]),
        );
    assert_eq!(class(&run(joined, &ctx)), "AMBIGUOUS_REFERENCE");
    // An already resolved scan of files is checked against the files, which
    // needs a file resolver (see `tests::files`).
    let scan = LogicalPlan::new(PlanNode::Scan(crate::Scan {
        source: ReadSource::Files {
            format: "parquet".to_owned(),
            paths: vec![krishiv_common::secret::OptionValue::from_user("path", "/d")],
            options: std::collections::BTreeMap::new(),
        },
        output: Vec::new(),
        version: None,
    }));
    assert_eq!(class(&run(scan, &ctx)), "KRISHIV_UNSUPPORTED_FEATURE");
}

#[test]
fn types_follow_the_catalog() {
    let a = run(
        LogicalPlan::table(["t"]).project(vec![c("c"), c("b")]),
        &ctx(),
    )
    .unwrap();
    let types: Vec<_> = a.output.iter().map(|c| c.data_type.clone()).collect();
    assert_eq!(types, [DataType::Long, DataType::String]);
}

/// `resolved_output` on hand-written resolved shapes.
#[test]
fn resolved_output_follows_the_contract() {
    let col = |id: u64, name: &str| crate::BoundColumn {
        id: crate::AttrId(id),
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: false,
    };
    let scan = |ids: &[u64]| {
        Arc::new(LogicalPlan::new(PlanNode::Scan(crate::Scan {
            source: ReadSource::Table {
                name: vec!["t".to_owned()],
                options: std::collections::BTreeMap::new(),
            },
            output: ids.iter().map(|i| col(*i, "a")).collect(),
            version: None,
        })))
    };
    let join = |kind| {
        LogicalPlan::new(PlanNode::Join {
            left: scan(&[1]),
            right: scan(&[2]),
            kind,
            condition: JoinCondition::None,
        })
    };
    let nullable = |p: LogicalPlan| -> Vec<bool> {
        p.resolved_output()
            .unwrap()
            .iter()
            .map(|c| c.nullable)
            .collect()
    };
    assert_eq!(nullable(join(JoinKind::LeftOuter)), [false, true]);
    assert_eq!(nullable(join(JoinKind::RightOuter)), [true, false]);
    assert_eq!(nullable(join(JoinKind::FullOuter)), [true, true]);
    assert_eq!(nullable(join(JoinKind::LeftSemi)), [false]);
    let alias = |columns: Vec<String>| {
        LogicalPlan::new(PlanNode::SubqueryAlias {
            input: scan(&[1]),
            alias: "x".to_owned(),
            columns,
        })
    };
    assert_eq!(
        alias(vec!["z".to_owned()]).resolved_output().unwrap()[0].name,
        "z"
    );
    assert!(
        alias(vec!["y".to_owned(), "z".to_owned()])
            .resolved_output()
            .is_none()
    );
    let shift = |offset| {
        LogicalPlan::new(PlanNode::Shift {
            input: scan(&[3]),
            offset,
        })
    };
    assert_eq!(shift(4).resolved_output().unwrap()[0].id, crate::AttrId(7));
    assert!(shift(u64::MAX).resolved_output().is_none());
    let filtered = LogicalPlan::new(PlanNode::Filter {
        input: scan(&[1]),
        predicate: Arc::new(Literal::Boolean(true).into()),
    });
    assert_eq!(filtered.resolved_output().unwrap().len(), 1);
    assert!(LogicalPlan::table(["t"]).resolved_output().is_none());
    assert!(
        LogicalPlan::table(["t"])
            .project(vec![c("a")])
            .resolved_output()
            .is_none()
    );
}

/// `SELECT` without `FROM` reads one row of no columns; the plan resolves
/// to itself, prints as Spark's `OneRowRelation` and has a key of its own.
#[test]
fn one_row_relation_has_no_columns() {
    let ctx = ctx();
    let one = LogicalPlan::new(PlanNode::OneRowRelation);
    let a = run(one.clone(), &ctx).unwrap();
    assert!(a.output.is_empty());
    assert_eq!(a.plan.root().resolved_output(), Some(Vec::new()));
    assert_eq!(a.plan.root().to_string(), "OneRowRelation\n");
    let again = run(a.plan.root().as_ref().clone(), &ctx).unwrap();
    assert!(again.plan.structurally_eq(&a.plan));
    let projected = one.project(vec![int(1)]);
    let p = run(projected, &ctx).unwrap();
    assert_eq!(p.output.len(), 1);
    assert_ne!(p.plan.structural_hash(), a.plan.structural_hash());
}

#[test]
fn try_arithmetic_casts_its_arguments_in_try_mode() {
    let ctx = ctx();
    let project = |name: &str| {
        let call = Expr::function(name, vec![s("x"), int(1)]);
        run(LogicalPlan::table(["t"]).project(vec![call]), &ctx).unwrap()
    };
    let try_add = project("try_add");
    let text = format!("{:?}", try_add.plan);
    assert!(text.contains("mode: Try"), "{text}");
    // `mod` casts as the operator does (ANSI here).
    let text = format!("{:?}", project("mod").plan);
    assert!(
        text.contains("mode: Ansi") && !text.contains("mode: Try"),
        "{text}"
    );
    // A TRY cast of a string may be NULL; `try_*` results are nullable anyway.
    assert!(try_add.output[0].nullable);
}

/// Typed datetime literals resolve in the analyzer with the session zone.
#[test]
fn typed_literals_resolve_with_the_session_zone() {
    use crate::{ExprNode, Literal, LogicalPlan, Plan, PlanNode, ResolvedExpr, TypedLiteralKind};
    let project = |kind: TypedLiteralKind, value: &str, conf: ResolutionConf| {
        let expr = Expr::new(ExprNode::TypedLiteral {
            kind,
            value: value.to_owned(),
        });
        let plan = LogicalPlan::new(crate::PlanNode::Project {
            input: std::sync::Arc::new(LogicalPlan::new(crate::PlanNode::OneRowRelation)),
            exprs: vec![std::sync::Arc::new(expr)],
        });
        let plan = Plan::try_new(plan).unwrap();
        let ctx = context(MemoryCatalog::new("spark_catalog"), conf);
        block_on(DefaultAnalyzer.analyze(&plan, &ctx))
    };
    // The instant a one-column project holds, unwrapping its alias.
    let micros_of = |plan: &LogicalPlan| {
        let PlanNode::Project { exprs, .. } = &plan.node else {
            panic!("not a project: {plan}");
        };
        let inner = match &exprs[0].node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
            ExprNode::Literal(_) => &exprs[0],
            other => panic!("not a literal: {other:?}"),
        };
        match &inner.node {
            ExprNode::Literal(Literal::Timestamp(v)) => *v,
            other => panic!("not a timestamp: {other:?}"),
        }
    };
    let utc = ResolutionConf {
        session_zone: Some(SessionZone::Fixed(0)),
        ..ResolutionConf::default()
    };
    let plus = ResolutionConf {
        session_zone: Some(SessionZone::Fixed(19_800)),
        ..ResolutionConf::default()
    };
    let ntz_conf = ResolutionConf {
        timestamp_ntz: true,
        session_zone: Some(SessionZone::Fixed(0)),
        ..ResolutionConf::default()
    };
    let region = ResolutionConf {
        session_zone: None,
        ..ResolutionConf::default()
    };
    // Plain `TIMESTAMP` is an instant by default, wall-clock under NTZ.
    let a = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", utc).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    assert_eq!(a.output[0].name, "TIMESTAMP '2024-01-01 10:00:00'");
    let a = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", ntz_conf).unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp_ntz");
    assert_eq!(a.output[0].name, "TIMESTAMP_NTZ '2024-01-01 10:00:00'");
    // An explicit zone stays an instant under NTZ and ignores the session.
    let a = project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00+05:30",
        ntz_conf,
    )
    .unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    // The session zone moves wall-clock instants (`+05:30` is 04:30 UTC):
    // Spark 4.2 gives 1_704_103_200_000_000 µs in UTC and
    // 1_704_083_400_000_000 µs in +05:30.
    let utc_plan = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", utc).unwrap();
    let plus_plan = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", plus).unwrap();
    assert_eq!(micros_of(utc_plan.plan.root()), 1_704_103_200_000_000);
    assert_eq!(micros_of(plus_plan.plan.root()), 1_704_083_400_000_000);
    let a = project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00+00:00",
        plus,
    )
    .unwrap();
    assert_eq!(a.output[0].name, "TIMESTAMP '2024-01-01 15:30:00'");
    // `DATE` resolves without a zone.
    let a = project(TypedLiteralKind::Date, "2016-03-12", utc).unwrap();
    assert_eq!(a.output[0].name, "DATE '2016-03-12'");
    assert_eq!(a.output[0].data_type.simple_string(), "date");
    // Malformed text is `INVALID_TYPED_LITERAL`.
    for (kind, value) in [
        (TypedLiteralKind::Timestamp, "2024-13-01 10:00:00"),
        (TypedLiteralKind::TimestampNtz, "10:00"),
        (TypedLiteralKind::Date, "2020-02-30"),
        (TypedLiteralKind::Date, "not-a-date"),
        (TypedLiteralKind::Date, "10:00"),
    ] {
        let e = project(kind, value, utc).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidTypedLiteral,
            "{value}"
        );
    }
    // Spark's special values read the query clock and keep the literal's type.
    for (kind, value, ty) in [
        (TypedLiteralKind::Timestamp, "now", "timestamp"),
        (TypedLiteralKind::TimestampNtz, "today", "timestamp_ntz"),
        (TypedLiteralKind::Date, "tomorrow", "date"),
        (TypedLiteralKind::Timestamp, "epoch", "timestamp"),
    ] {
        let a = project(kind, value, utc).unwrap();
        assert_eq!(a.output[0].data_type.simple_string(), ty, "{value}");
    }
    // A bare time of day is that time on the query clock's date, in the session zone.
    for (kind, value) in [
        (TypedLiteralKind::TimestampLtz, "10:00"),
        (TypedLiteralKind::Timestamp, "T10:00:30.5"),
    ] {
        let ty = project(kind, value, utc).unwrap().output[0]
            .data_type
            .simple_string();
        assert_eq!(ty, "timestamp", "{value}");
    }
    // An unresolvable session zone serves explicit zones but not wall-clock time.
    project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00+05:30",
        region,
    )
    .unwrap();
    let e = project(TypedLiteralKind::Timestamp, "2024-01-01 10:00:00", region).unwrap_err();
    assert_eq!(e.error.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A region id in the value resolves against the bundled database:
    // 10:00 in Los Angeles (UTC-08:00 in January) is 18:00 UTC.
    let a = project(
        TypedLiteralKind::Timestamp,
        "2024-01-01 10:00:00 America/Los_Angeles",
        utc,
    )
    .unwrap();
    assert_eq!(a.output[0].data_type.simple_string(), "timestamp");
    let crate::PlanNode::Project { exprs, .. } = &a.plan.root().node else {
        panic!("a literal projects");
    };
    let Some(expr) = exprs.first() else {
        panic!("a literal projects one column");
    };
    let ExprNode::Resolved(crate::ResolvedExpr::Alias { expr, .. }) = &expr.node else {
        panic!("a literal projects as an alias");
    };
    assert!(matches!(
        expr.node,
        ExprNode::Literal(Literal::Timestamp(1_704_132_000_000_000))
    ));
    // A region session zone reads wall-clock text through the database,
    // with the zone's offset at that instant: January is UTC-08:00, July
    // UTC-07:00 (daylight saving).
    let la = ResolutionConf {
        session_zone: SessionZone::resolve("America/Los_Angeles"),
        ..ResolutionConf::default()
    };
    for (text, micros) in [
        ("2024-01-01 10:00:00", 1_704_132_000_000_000_i64),
        ("2024-07-01 10:00:00", 1_719_853_200_000_000_i64),
    ] {
        let a = project(TypedLiteralKind::Timestamp, text, la).unwrap();
        let crate::PlanNode::Project { exprs, .. } = &a.plan.root().node else {
            panic!("a literal projects");
        };
        let ExprNode::Resolved(crate::ResolvedExpr::Alias { expr, .. }) = &exprs[0].node else {
            panic!("a literal projects as an alias");
        };
        assert!(
            matches!(expr.node, ExprNode::Literal(Literal::Timestamp(v)) if v == micros),
            "{text}"
        );
    }
}

/// A conditional with `FILTER` names the filter once: the shown call takes
/// only the function's own arguments, not the filter child.
#[test]
fn conditional_with_filter_names_the_filter_once() {
    let ctx = ctx();
    let call = Expr::new(ExprNode::Function(crate::FunctionCall {
        within_group: Vec::new(),
        name: "nvl".to_owned(),
        qualifier: Vec::new(),
        arg_names: Vec::new(),
        args: vec![Arc::new(c("a")), Arc::new(int(1))],
        distinct: false,
        filter: Some(Arc::new(eq(c("b"), s("x")))),
    }));
    let a = run(LogicalPlan::table(["t"]).project(vec![call]), &ctx).unwrap();
    assert_eq!(a.output[0].name, "nvl(a, 1) FILTER (WHERE (b = x))");
}

/// `OFFSET rand(1)` is `IS_UNFOLDABLE` (cited as `_nondeterministic`), not
/// `DATA_TYPE` for its `DOUBLE`: Spark checks foldability first (Spark 4.2.0
/// raises the same for `OFFSET rand()`).
#[test]
fn a_nondeterministic_count_is_unfoldable_before_its_type() {
    let mut cx = ctx();
    cx.functions = Arc::new(WithRand(crate::analyzer::BuiltinFunctions::new()));
    let rand = || Expr::function("rand", vec![Expr::from(Literal::Integer(1))]);
    for (plan, name) in [
        (LogicalPlan::table(["t"]).offset(rand()), "offset"),
        (LogicalPlan::table(["t"]).limit(rand()), "limit"),
    ] {
        let e = run(plan, &cx).unwrap_err();
        assert_eq!(
            e.error.error_class(),
            ErrorClass::InvalidLimitLikeExpressionIsUnfoldable,
            "{e:?}"
        );
        let msg = e.error.to_string();
        assert!(
            msg.contains("\"_nondeterministic\"") && msg.contains(name),
            "{msg}"
        );
    }
}

/// An unused CTE is not resolved: its column errors never fail the query.
/// Spark's golden `cte` case 50 keeps an unused `cte3` whose projection
/// over the duplicate-case columns of `cte2` would be ambiguous.
#[test]
fn unused_cte_definitions_are_not_resolved() {
    let ctx = ctx();
    let valid = LogicalPlan::table(["t"]).project(vec![c("a")]);
    let ambiguous = LogicalPlan::table(["t"])
        .join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::None,
        )
        .project(vec![c("a")]);
    assert_eq!(class(&run(ambiguous.clone(), &ctx)), "AMBIGUOUS_REFERENCE");
    // The main query reads only the valid definition; the ambiguous one
    // is never referenced, so the query resolves.
    let unused = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["good"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(valid),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(ambiguous),
            },
        ],
    });
    let a = run(unused, &ctx).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    // Referencing the bad definition fails.
    let used = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["bad"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "good".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
            Cte {
                name: "bad".to_owned(),
                plan: Arc::new(
                    LogicalPlan::table(["t"])
                        .join(
                            LogicalPlan::table(["u"]),
                            JoinKind::Inner,
                            JoinCondition::None,
                        )
                        .project(vec![c("a")]),
                ),
            },
        ],
    });
    assert_eq!(class(&run(used, &ctx)), "AMBIGUOUS_REFERENCE");
}

/// `SELECT 1 GROUP BY `1`` groups by the select-list alias `1`, even
/// though the item has no explicit alias (Spark golden `group-by-alias`
/// case 11). With `groupByAliases` off the same key is unresolved.
#[test]
fn group_by_resolves_implicit_select_aliases() {
    let plan = LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["1"]
    );
    let conf = ResolutionConf {
        group_by_aliases: false,
        ..ResolutionConf::default()
    };
    let no_alias = context(catalog(), conf);
    let plan = LogicalPlan::new(PlanNode::OneRowRelation).aggregate(vec![c("1")], vec![int(1)]);
    assert_eq!(
        class(&run(plan, &no_alias)),
        "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    );
}

/// A grouping expression may not contain an aggregate call, however it was
/// written: directly, or through a select-list alias (`SELECT COUNT(b) AS k
/// ... GROUP BY k`, Spark golden `group-by-alias`). Without the check the
/// alias resolves to the aggregate, whose semantic key is already in the
/// grouping-key set `check_aggregation` skips, so the plan wrongly builds.
#[test]
fn group_by_rejects_aggregate_functions() {
    let aliased = LogicalPlan::table(["t"])
        .aggregate(vec![c("k")], vec![f("count", vec![c("b")]).alias("k")]);
    assert_eq!(class(&run(aliased, &ctx())), "GROUP_BY_AGGREGATE");
    let direct = LogicalPlan::table(["t"]).aggregate(vec![f("count", vec![c("b")])], vec![c("a")]);
    assert_eq!(class(&run(direct, &ctx())), "GROUP_BY_AGGREGATE");
}

/// A `GROUP BY` key naming a select-list alias that reads lateral column
/// aliases is `UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY` (Spark
/// golden `column-resolution-aggregate`: `SELECT k AS lca, lca + 1 AS col
/// ... GROUP BY k, col`). Grouping by a lateral alias itself stays fine.
#[test]
fn group_by_rejects_aliases_reading_lateral_aliases() {
    let items = || vec![c("a").alias("lca"), add(c("lca"), int(1)).alias("col")];
    let lateral = LogicalPlan::table(["t"]).aggregate(vec![c("col")], items());
    assert_eq!(
        class(&run(lateral, &ctx())),
        "UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY"
    );
    let direct = LogicalPlan::table(["t"]).aggregate(vec![c("lca")], items());
    run(direct, &ctx()).unwrap();
}

/// A plan shared under two inner `WITH` scopes with different shadowing
/// names is visited under each scope: visits are keyed by scope contents,
/// not scope length, so a shadowing scope cannot hide a read the other
/// scope reports and drop a definition the input needs.
#[test]
fn cte_refs_visit_shared_plans_under_each_scope() {
    let shared = Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")]));
    let inner = |name: &str| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::clone(&shared),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(LogicalPlan::table(["u"]).project(vec![c("a")])),
            }],
        })
    };
    // The right branch shadows `x`, so `shared` is visited under its scope
    // first (a join pops its right child first): keying visits by scope
    // length alone would skip the left branch's read and drop the outer `x`.
    let input = inner("y").join(inner("x"), JoinKind::Inner, JoinCondition::None);
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(input),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "a"]
    );
}

/// The same shared subquery expression under two inner `WITH` scopes with
/// different shadowing names is visited under each scope too: expression
/// visits are keyed by scope contents as well, or the shadowed visit hides
/// the read the other scope's subquery reports.
#[test]
fn cte_refs_visit_shared_exprs_under_each_scope() {
    let sub: Arc<Expr> = Arc::new(Expr::subquery(
        crate::SubqueryKind::Scalar,
        LogicalPlan::table(["x"]).project(vec![c("a")]),
    ));
    let branch = |name: &str| {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::new(PlanNode::Project {
                input: Arc::new(LogicalPlan::table(["t"])),
                exprs: vec![Arc::clone(&sub)],
            })),
            ctes: vec![Cte {
                name: name.to_owned(),
                plan: Arc::new(LogicalPlan::table(["u"]).project(vec![c("a")])),
            }],
        })
    };
    // As above, the right branch shadows `x` and is visited first.
    let input = branch("y").join(branch("x"), JoinKind::Inner, JoinCondition::None);
    let plan = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(input),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
        }],
    });
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["scalarsubquery()", "scalarsubquery()"]
    );
}

/// An inner `WITH` name shadows the outer one: reads inside the inner scope
/// resolve to the inner definition, so an outer definition nothing else
/// reads is dropped (even when it would not resolve), while a read outside
/// the inner scope still needs the outer definition.
#[test]
fn inner_with_names_shadow_outer_ctes() {
    let bad = || {
        LogicalPlan::table(["t"])
            .join(
                LogicalPlan::table(["u"]),
                JoinKind::Inner,
                JoinCondition::None,
            )
            .project(vec![c("a")])
    };
    assert_eq!(class(&run(bad(), &ctx())), "AMBIGUOUS_REFERENCE");
    let good = || LogicalPlan::table(["t"]).project(vec![c("a")]);
    let inner = || {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "x".to_owned(),
                plan: Arc::new(good()),
            }],
        })
    };
    // Only the inner `x` is read: the outer one is dropped, so its
    // ambiguity never fails the query.
    let shadowed = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(inner()),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    let a = run(shadowed, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    // A read outside the inner scope still needs the outer definition.
    let outer_read = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(inner().join(
            LogicalPlan::table(["x"]).project(vec![c("a")]),
            JoinKind::Inner,
            JoinCondition::None,
        )),
        ctes: vec![Cte {
            name: "x".to_owned(),
            plan: Arc::new(bad()),
        }],
    });
    assert_eq!(class(&run(outer_read, &ctx())), "AMBIGUOUS_REFERENCE");
}

/// A definition's reads see only the outer CTEs and the earlier definitions:
/// a forward reference never marks a later definition, so the read falls
/// through to table binding.
#[test]
fn cte_forward_references_fall_through_to_tables() {
    // `a` reads `b`, which is defined later: `b` is dropped, and resolving
    // `a` reads a missing table.
    let forward = LogicalPlan::new(PlanNode::WithCte {
        input: Arc::new(LogicalPlan::table(["a"]).project(vec![c("a")])),
        ctes: vec![
            Cte {
                name: "a".to_owned(),
                plan: Arc::new(LogicalPlan::table(["b"]).project(vec![c("a")])),
            },
            Cte {
                name: "b".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            },
        ],
    });
    assert_eq!(class(&run(forward, &ctx())), "TABLE_OR_VIEW_NOT_FOUND");
    // The same definitions in dependency order resolve.
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
    let a = run(ordered, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
}

/// CTE-name matching follows `spark.sql.caseSensitive`: case-insensitive by
/// default, exact when set.
#[test]
fn cte_name_matching_follows_case_sensitivity() {
    let plan = || {
        LogicalPlan::new(PlanNode::WithCte {
            input: Arc::new(LogicalPlan::table(["x"]).project(vec![c("a")])),
            ctes: vec![Cte {
                name: "X".to_owned(),
                plan: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a")])),
            }],
        })
    };
    let a = run(plan(), &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a"]
    );
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..ResolutionConf::default()
    };
    assert_eq!(
        class(&run(plan(), &context(catalog(), sensitive))),
        "TABLE_OR_VIEW_NOT_FOUND"
    );
}

/// `GROUP BY ALL` groups by every select item without an aggregate; a lone
/// `all` that resolves to a real column stays that column (Spark's
/// `ResolveReferencesInAggregate`).
#[test]
fn group_by_all_expands_over_non_aggregates() {
    use crate::{ExprNode, Grouping};

    fn grouping_of(plan: &LogicalPlan) -> Vec<String> {
        let mut keys = Vec::new();
        let _ = visit_distinct(Node::Plan(plan), |node| {
            if let Node::Plan(p) = node
                && let PlanNode::Aggregate { grouping, .. } = &p.node
                && let Grouping::GroupBy(ks) = grouping
            {
                keys.extend(ks.iter().map(|k| match &k.node {
                    ExprNode::Resolved(crate::ResolvedExpr::Column(c)) => c.name.clone(),
                    other => format!("{other:?}"),
                }));
            }
            ControlFlow::Continue(())
        });
        keys
    }

    // Only SQL has a `GROUP BY ALL`: the plan carries its span.
    let sql = || Origin::sql(crate::SourceId(0), crate::Span::new(0, 1));
    // `a` groups; `count(*)` does not.
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("all")], vec![c("a"), f("count", vec![star()])])
        .with_origin(sql());
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "count(1)"]
    );
    assert_eq!(grouping_of(a.plan.root()).len(), 1);
    // An item mixing a bare column with an aggregate cannot be inferred.
    let plan = LogicalPlan::table(["t"])
        .aggregate(vec![c("all")], vec![add(c("a"), count_star())])
        .with_origin(sql());
    assert_eq!(class(&run(plan, &ctx())), "UNRESOLVED_ALL_IN_GROUP_BY");
    // A column actually named `all` takes precedence over the expansion.
    let plan = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("all"), c("b")])
        .aggregate(vec![c("all")], vec![c("all"), f("count", vec![star()])])
        .with_origin(sql());
    let a = run(plan, &ctx()).unwrap();
    assert_eq!(grouping_of(a.plan.root()).len(), 1);
    // A DataFrame `groupBy("all")` has no SQL span: an unknown column.
    let plan =
        LogicalPlan::table(["t"]).aggregate(vec![c("all")], vec![c("a"), f("count", vec![star()])]);
    assert_eq!(
        class(&run(plan, &ctx())),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
}

/// `ORDER BY ALL` sorts by every output column; a column actually named
/// `all` takes precedence (Spark's `ResolveReferencesInSort`).
#[test]
fn order_by_all_sorts_by_every_output_column() {
    use crate::SortOrder;

    let plan = LogicalPlan::table(["t"])
        .project(vec![c("a"), c("b")])
        .sort(vec![SortOrder::asc(c("all"))]);
    let a = run(plan, &ctx()).unwrap();
    let mut orders = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            orders.extend(order.iter().map(|o| o.expr.to_string()));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(orders.len(), 2);
    let plan = LogicalPlan::table(["t"])
        .project(vec![c("a").alias("all"), c("b")])
        .sort(vec![SortOrder::asc(c("all"))]);
    let a = run(plan, &ctx()).unwrap();
    let mut orders = Vec::new();
    let _ = visit_distinct(Node::Plan(a.plan.root()), |node| {
        if let Node::Plan(p) = node
            && let PlanNode::Sort { order, .. } = &p.node
        {
            orders.extend(order.iter().map(|o| o.expr.to_string()));
        }
        ControlFlow::Continue(())
    });
    assert_eq!(orders.len(), 1);
    // `SORT BY ALL` / `sortWithinPartitions("all")` is a non-global sort:
    // `all` is an unknown column there.
    let plan = LogicalPlan::new(PlanNode::Sort {
        input: Arc::new(LogicalPlan::table(["t"]).project(vec![c("a"), c("b")])),
        order: vec![SortOrder::asc(c("all"))],
        global: false,
    });
    assert_eq!(
        class(&run(plan, &ctx())),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
}

/// `* EXCEPT` drops the named columns; overlapping entries and nested
/// fields of a non-struct column fail, unknown names resolve as usual.
#[test]
fn star_except_drops_overlaps_and_rejects_nested_fields_of_a_flat_column() {
    use crate::Star;

    let except = |cols: &[&str]| {
        Expr::new(ExprNode::Star(Star {
            regex: None,
            replace: Vec::new(),
            position: None,
            qualifier: Vec::new(),
            except: cols
                .iter()
                .map(|c| c.split('.').map(str::to_owned).collect())
                .collect(),
            plan_id: None,
        }))
    };
    let names = |plan: LogicalPlan| {
        run(plan, &ctx()).map(|a| a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
    };
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["b"])]);
    assert_eq!(names(plan).unwrap(), ["a", "c"]);
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["a", "a"])]);
    assert_eq!(class(&run(plan, &ctx())), "EXCEPT_OVERLAPPING_COLUMNS");
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["a", "a", "a.b"])]);
    assert_eq!(class(&run(plan, &ctx())), "EXCEPT_OVERLAPPING_COLUMNS");
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["a.b"])]);
    assert_eq!(class(&run(plan, &ctx())), "INVALID_EXTRACT_BASE_FIELD_TYPE");
    let plan = LogicalPlan::table(["t"]).project(vec![except(&["z"])]);
    assert_eq!(
        class(&run(plan, &ctx())),
        "UNRESOLVED_COLUMN.WITH_SUGGESTION"
    );
}

/// `NATURAL JOIN` joins on the common column names in left order and
/// projects like `USING`; with no common columns it is a cross join.
#[test]
fn natural_join_uses_common_columns_in_left_order() {
    let names = |plan: LogicalPlan, ctx: &AnalysisContext| {
        run(plan, ctx).map(|a| a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>())
    };
    let join = || {
        LogicalPlan::table(["t"]).join(
            LogicalPlan::table(["u"]),
            JoinKind::Inner,
            JoinCondition::Natural,
        )
    };
    assert_eq!(
        names(join().project(vec![star()]), &ctx()).unwrap(),
        ["a", "b", "c", "d"]
    );
    // Case-insensitive by default: `A` still meets `a`.
    let upper = LogicalPlan::table(["t"]).join(
        LogicalPlan::table(["u"]).project(vec![c("a").alias("A"), c("d")]),
        JoinKind::Inner,
        JoinCondition::Natural,
    );
    assert_eq!(
        names(upper.project(vec![star()]), &ctx()).unwrap(),
        ["a", "b", "c", "d"]
    );
    // Case-sensitive: `a` and `A` share nothing, so the join is a cross.
    let sensitive = ResolutionConf {
        case_sensitive: true,
        ..ResolutionConf::default()
    };
    let upper = LogicalPlan::table(["t"]).join(
        LogicalPlan::table(["u"]).project(vec![c("a").alias("A"), c("d")]),
        JoinKind::Inner,
        JoinCondition::Natural,
    );
    let a = run(upper.project(vec![star()]), &context(catalog(), sensitive)).unwrap();
    assert_eq!(
        a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        ["a", "b", "c", "A", "d"]
    );
}
