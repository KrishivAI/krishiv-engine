// SPDX-License-Identifier: Apache-2.0
//! Tests of the relation translation.

use std::collections::BTreeMap;

use krishiv_common::error::ErrorClass;
use krishiv_common::secret::OptionValue;
use krishiv_plan::PlanNode;

use super::super::testing::{
    attr, boxed, conf, expr, fails, func, lit, read_files, read_table, relation, show, translate,
    with_id,
};
use super::super::{TranslateConf, translate_relation};
use super::*;
use crate::proto::v4_2::expression::{self, ExprType, literal::LiteralType};
use crate::proto::v4_2::{aggregate, join, read, relation::RelType};

fn input() -> Option<Box<Relation>> {
    boxed(read_table("t"))
}

fn code(r: &Relation) -> ErrorClass {
    fails(translate(r)).error_class()
}

fn message(r: &Relation) -> String {
    fails(translate(r)).message().to_owned()
}

fn node(r: &Relation) -> PlanNode {
    root(r).node.clone()
}

fn root(r: &Relation) -> LogicalPlan {
    let plan = translate(r).expect("translates");
    LogicalPlan::clone(plan.root())
}

// ---- reads --------------------------------------------------------------------------------

#[test]
fn named_tables() {
    assert_eq!(show(&read_table("t")), "Read t\n");
    assert_eq!(show(&read_table("cat.db.t")), "Read cat.db.t\n");
    assert_eq!(
        show(&read_table("`my db`.`ev.ents`")),
        "Read `my db`.`ev.ents`\n"
    );
    let e = fails(translate(&read_table("a b")));
    assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError);
    assert_eq!(code(&read_table("")), ErrorClass::ParseSyntaxError);
}

#[test]
fn table_options_are_typed() {
    let r = relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(ReadType::NamedTable(read::NamedTable {
            unparsed_identifier: "t".into(),
            options: BTreeMap::from([
                ("versionAsOf".to_owned(), "3".to_owned()),
                ("password".to_owned(), "CANARY".to_owned()),
            ]),
        })),
    }));
    let PlanNode::Read(ReadSource::Table { options, .. }) = node(&r) else {
        panic!("not a table read")
    };
    assert_eq!(options["versionAsOf"], OptionValue::Plain("3".into()));
    assert!(matches!(options["password"], OptionValue::Secret(_)));
}

#[test]
fn file_reads_and_the_default_format() {
    assert_eq!(
        show(&read_files(Some("parquet"), &["/a.parquet"])),
        "Read parquet [/a.parquet]\n"
    );
    assert_eq!(
        show(&read_files(Some("csv"), &["/a", "/b"])),
        "Read csv [/a, /b]\n"
    );
    assert_eq!(show(&read_files(Some("jdbc"), &[])), "Read jdbc []\n");
    // No format: `spark.sql.sources.default`, parquet unless configured.
    assert_eq!(show(&read_files(None, &["/a"])), "Read parquet [/a]\n");
    let orc = TranslateConf {
        default_data_source: "orc".into(),
    };
    let plan = translate_relation(&read_files(None, &["/a"]), &orc).unwrap();
    assert_eq!(plan.to_string(), "Read orc [/a]\n");
}

fn data_source(source: read::DataSource) -> Relation {
    relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(ReadType::DataSource(source)),
    }))
}

#[test]
fn file_read_options_and_paths_are_classified_once() {
    let r = data_source(read::DataSource {
        format: Some("parquet".into()),
        options: BTreeMap::from([
            ("mergeSchema".to_owned(), "true".to_owned()),
            ("fs.s3a.secret.key".to_owned(), "CANARY-SECRET".to_owned()),
            (
                "url".to_owned(),
                "jdbc:postgresql://u:CANARY-PW@h/db".to_owned(),
            ),
        ]),
        paths: vec!["https://h/p.parquet?X-Amz-Signature=CANARY-SIG&X-Amz-Expires=60".into()],
        ..read::DataSource::default()
    });
    let PlanNode::Read(ReadSource::Files { paths, options, .. }) = node(&r) else {
        panic!("not a file read")
    };
    assert_eq!(options["mergeSchema"], OptionValue::Plain("true".into()));
    assert!(matches!(
        options["fs.s3a.secret.key"],
        OptionValue::Secret(_)
    ));
    assert!(matches!(options["url"], OptionValue::Url(_)));
    assert!(
        matches!(paths[0], OptionValue::Url(_)),
        "a signed URL's signature is secret"
    );
}

#[test]
fn a_streaming_read_a_schema_and_predicates_are_unsupported() {
    let mut streaming = read_files(Some("parquet"), &["/a"]);
    if let Some(RelType::Read(r)) = &mut streaming.rel_type {
        r.is_streaming = true;
    }
    assert!(message(&streaming).contains("is_streaming"));
    let schema = data_source(read::DataSource {
        schema: Some("a INT".into()),
        ..read::DataSource::default()
    });
    assert!(
        message(&schema).contains("Read.DataSource.schema"),
        "{}",
        message(&schema)
    );
    // The client sends an empty schema string when none is given.
    let empty = data_source(read::DataSource {
        format: Some("parquet".into()),
        schema: Some(String::new()),
        ..read::DataSource::default()
    });
    assert_eq!(show(&empty), "Read parquet []\n");
    let predicates = data_source(read::DataSource {
        predicates: vec!["a < 1".into()],
        ..read::DataSource::default()
    });
    assert!(message(&predicates).contains("predicates"));
    let unset = relation(RelType::Read(pb::Read::default()));
    assert_eq!(
        code(&unset),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

// ---- SQL ----------------------------------------------------------------------------------

fn sql(s: pb::Sql) -> Relation {
    relation(RelType::Sql(s))
}

fn sql_args(r: &Relation) -> SqlArgs {
    match node(r) {
        PlanNode::Sql { args, .. } => args,
        other => panic!("not SQL: {other:?}"),
    }
}

#[test]
fn sql_with_and_without_parameters() {
    let plain = sql(pb::Sql {
        query: "SELECT 1".into(),
        ..pb::Sql::default()
    });
    assert_eq!(show(&plain), "Sql [8 bytes]\n");
    assert_eq!(sql_args(&plain), SqlArgs::None);

    let named = pb::Sql {
        query: "SELECT :a".into(),
        named_arguments: BTreeMap::from([("a".to_owned(), lit(LiteralType::Integer(5)))]),
        ..pb::Sql::default()
    };
    let SqlArgs::Named(args) = sql_args(&sql(named)) else {
        panic!("not named")
    };
    assert_eq!(args["a"].to_string(), "5");

    let positional = pb::Sql {
        query: "SELECT ?".into(),
        pos_arguments: vec![lit(LiteralType::String("x".into())), attr("c")],
        ..pb::Sql::default()
    };
    let SqlArgs::Positional(args) = sql_args(&sql(positional)) else {
        panic!("not positional")
    };
    assert_eq!(args.len(), 2);
}

#[test]
#[expect(
    deprecated,
    reason = "the deprecated literal arguments older clients send"
)]
fn sql_parameters_follow_spark_precedence() {
    // Named expressions win over positional ones, which win over the
    // deprecated literal map, which wins over the deprecated literal list.
    let literal = |n| expression::Literal {
        literal_type: Some(LiteralType::Integer(n)),
        data_type: None,
    };
    let all = pb::Sql {
        query: "q".into(),
        args: BTreeMap::from([("old".to_owned(), literal(1))]),
        pos_args: vec![literal(2)],
        named_arguments: BTreeMap::from([("new".to_owned(), lit(LiteralType::Integer(3)))]),
        pos_arguments: vec![lit(LiteralType::Integer(4))],
    };
    let keys = |a: SqlArgs| match a {
        SqlArgs::Named(m) => m.keys().cloned().collect::<Vec<_>>(),
        SqlArgs::Positional(v) => vec![format!("{} positional", v.len())],
        other => panic!("{other:?}"),
    };
    let mut s = all;
    assert_eq!(keys(sql_args(&sql(s.clone()))), ["new"]);
    s.named_arguments.clear();
    assert_eq!(keys(sql_args(&sql(s.clone()))), ["1 positional"]);
    s.pos_arguments.clear();
    assert_eq!(keys(sql_args(&sql(s.clone()))), ["old"]);
    s.args.clear();
    let SqlArgs::Positional(list) = sql_args(&sql(s.clone())) else {
        panic!("not positional")
    };
    assert_eq!(list[0].to_string(), "2");
    s.pos_args.clear();
    assert_eq!(sql_args(&sql(s)), SqlArgs::None);
    // A literal argument that cannot be translated fails the plan.
    let bad = pb::Sql {
        pos_args: vec![expression::Literal::default()],
        ..pb::Sql::default()
    };
    assert_eq!(
        code(&sql(bad)),
        ErrorClass::ConnectInvalidPlanUnsupportedLiteralType
    );
}

// ---- project, filter, limit, alias -----------------------------------------------------------

fn project(exprs: Vec<Expression>) -> Relation {
    relation(RelType::Project(Box::new(pb::Project {
        input: input(),
        expressions: exprs,
    })))
}

#[test]
fn project_filter_limit_and_alias() {
    assert_eq!(
        show(&project(vec![
            attr("a"),
            expr(ExprType::UnresolvedStar(Default::default()))
        ])),
        "Project [a, *]\n+- Read t\n"
    );
    let filter = relation(RelType::Filter(Box::new(pb::Filter {
        input: input(),
        condition: Some(func(">", vec![attr("a"), lit(LiteralType::Integer(1))])),
    })));
    assert_eq!(show(&filter), "Filter (a > 1)\n+- Read t\n");
    let limit = relation(RelType::Limit(Box::new(pb::Limit {
        input: input(),
        limit: 10,
    })));
    assert_eq!(show(&limit), "Limit 10\n+- Read t\n");
    let offset = relation(RelType::Offset(Box::new(pb::Offset {
        input: input(),
        offset: 3,
    })));
    assert_eq!(show(&offset), "Offset 3\n+- Read t\n");
    let alias = relation(RelType::SubqueryAlias(Box::new(pb::SubqueryAlias {
        input: input(),
        alias: "x".into(),
        qualifier: vec![],
    })));
    assert_eq!(show(&alias), "SubqueryAlias x\n+- Read t\n");
}

#[test]
fn range_builds_the_range_leaf() {
    use crate::proto::v4_2 as pb;
    let r = relation(RelType::Range(pb::Range {
        start: Some(1),
        end: 10,
        step: 3,
        num_partitions: Some(2),
    }));
    let got = node(&r);
    let PlanNode::Range {
        start,
        end,
        step,
        num_slices,
        output,
    } = &got
    else {
        panic!("{got:?}");
    };
    assert_eq!((*start, *end, *step, *num_slices), (1, 10, 3, Some(2)));
    assert_eq!(output.name, "id");
    assert!(format!("{got:?}").contains("Range"));
    // Defaults: `start` 0, no partition count.
    let r = relation(RelType::Range(pb::Range {
        start: None,
        end: 5,
        step: 1,
        num_partitions: None,
    }));
    let PlanNode::Range {
        start, num_slices, ..
    } = node(&r)
    else {
        panic!()
    };
    assert_eq!((start, num_slices), (0, None));
    // `step = 0` and a non-positive partition count fail.
    let bad = relation(RelType::Range(pb::Range {
        start: Some(0),
        end: 5,
        step: 0,
        num_partitions: None,
    }));
    assert_eq!(code(&bad), ErrorClass::FailedFunctionCall);
    let bad = relation(RelType::Range(pb::Range {
        start: Some(0),
        end: 5,
        step: 1,
        num_partitions: Some(0),
    }));
    assert_eq!(code(&bad), ErrorClass::KrishivInvalidPartitionNumber);
}

#[test]
fn project_without_input_and_qualified_alias_are_unsupported() {
    let no_input = relation(RelType::Project(Box::default()));
    let e = fails(translate(&no_input));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.message().contains("Project.input"), "{}", e.message());
    let qualified = relation(RelType::SubqueryAlias(Box::new(pb::SubqueryAlias {
        input: input(),
        alias: "x".into(),
        qualifier: vec!["db".into()],
    })));
    assert!(message(&qualified).contains("SubqueryAlias.qualifier"));
}

#[test]
fn missing_inputs_and_conditions_are_plan_errors() {
    let no_input = relation(RelType::Filter(Box::new(pb::Filter {
        input: None,
        condition: Some(attr("a")),
    })));
    assert_eq!(
        code(&no_input),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    assert!(message(&no_input).contains("Filter needs an input"));
    let no_condition = relation(RelType::Filter(Box::new(pb::Filter {
        input: input(),
        condition: None,
    })));
    assert_eq!(
        code(&no_condition),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    let limit = relation(RelType::Limit(Box::default()));
    assert_eq!(code(&limit), ErrorClass::ConnectInvalidPlanAssertionFailure);
    // An empty relation message inside another is an unset relation.
    let empty_child = relation(RelType::Filter(Box::new(pb::Filter {
        input: boxed(Relation::default()),
        condition: Some(attr("a")),
    })));
    assert!(message(&empty_child).contains("REL_TYPE_NOT_SET"));
    assert_eq!(
        code(&Relation::default()),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

// ---- joins -----------------------------------------------------------------------------------

fn join_of(kind: i32, condition: Option<Expression>, using: &[&str]) -> Relation {
    relation(RelType::Join(Box::new(pb::Join {
        left: boxed(read_table("l")),
        right: boxed(read_table("r")),
        join_condition: condition,
        join_type: kind,
        using_columns: using.iter().map(|c| (*c).to_owned()).collect(),
        join_data_type: None,
    })))
}

#[test]
fn every_join_type_maps() {
    let on = || Some(func("==", vec![attr("a"), attr("b")]));
    for (kind, name) in [
        (join::JoinType::Inner, "Inner"),
        (join::JoinType::FullOuter, "FullOuter"),
        (join::JoinType::LeftOuter, "LeftOuter"),
        (join::JoinType::RightOuter, "RightOuter"),
        (join::JoinType::LeftAnti, "LeftAnti"),
        (join::JoinType::LeftSemi, "LeftSemi"),
        (join::JoinType::Cross, "Cross"),
    ] {
        let shown = show(&join_of(kind as i32, on(), &[]));
        assert!(
            shown.starts_with(&format!("Join {name}")),
            "{kind:?}: {shown}"
        );
        assert!(
            shown.contains(":- Read l") && shown.contains("+- Read r"),
            "{shown}"
        );
    }
}

#[test]
fn join_conditions() {
    let on = join_of(1, Some(func("==", vec![attr("a"), attr("b")])), &[]);
    let PlanNode::Join { condition, .. } = node(&on) else {
        panic!("not a join")
    };
    assert!(matches!(condition, JoinCondition::On(_)));
    let using = join_of(1, None, &["id", "k"]);
    let PlanNode::Join { condition, .. } = node(&using) else {
        panic!("not a join")
    };
    assert_eq!(
        condition,
        JoinCondition::Using(vec!["id".into(), "k".into()])
    );
    let none = join_of(7, None, &[]);
    let PlanNode::Join { condition, .. } = node(&none) else {
        panic!("not a join")
    };
    assert_eq!(condition, JoinCondition::None);
}

#[test]
fn join_errors() {
    let both = join_of(1, Some(attr("a")), &["id"]);
    assert_eq!(
        code(&both),
        ErrorClass::ConnectInvalidPlanUsingColumnsOrJoinConditionSetInJoin
    );
    let unspecified = join_of(0, None, &[]);
    assert_eq!(
        code(&unspecified),
        ErrorClass::ConnectInvalidPlanInvalidEnum
    );
    assert!(
        message(&unspecified).contains("JOIN_TYPE_UNSPECIFIED"),
        "{}",
        message(&unspecified)
    );
    let unknown = join_of(99, None, &[]);
    assert!(message(&unknown).contains("UNRECOGNIZED") && message(&unknown).contains("99"));
    let one_side = relation(RelType::Join(Box::new(pb::Join {
        left: boxed(read_table("l")),
        join_type: 1,
        ..pb::Join::default()
    })));
    assert_eq!(
        code(&one_side),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    let mut join_with = join_of(1, None, &[]);
    if let Some(RelType::Join(j)) = &mut join_with.rel_type {
        j.join_data_type = Some(join::JoinDataType::default());
    }
    assert!(message(&join_with).contains("join_data_type"));
}

// ---- sort ------------------------------------------------------------------------------------

fn order(name: &str, direction: i32, nulls: i32) -> expression::SortOrder {
    expression::SortOrder {
        child: Some(Box::new(attr(name))),
        direction,
        null_ordering: nulls,
    }
}

#[test]
fn sorts() {
    let sort = |is_global, order| {
        relation(RelType::Sort(Box::new(pb::Sort {
            input: input(),
            order,
            is_global,
        })))
    };
    let two = vec![order("a", 2, 2), order("b", 1, 1)];
    assert_eq!(
        show(&sort(Some(true), two.clone())),
        "Sort [a DESC NULLS LAST, b ASC NULLS FIRST], true\n+- Read t\n"
    );
    assert!(
        show(&sort(Some(false), two.clone()))
            .starts_with("Sort [a DESC NULLS LAST, b ASC NULLS FIRST], false")
    );
    assert!(
        show(&sort(None, two)).contains("false"),
        "an unset is_global is false, as in Spark"
    );
    let e = fails(translate(&sort(Some(true), vec![])));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    assert!(e.message().contains("'order'"));
}

// ---- aggregates -------------------------------------------------------------------------------

fn aggregate_of(group_type: i32, keys: Vec<Expression>, aggs: Vec<Expression>) -> Relation {
    relation(RelType::Aggregate(Box::new(pb::Aggregate {
        input: input(),
        group_type,
        grouping_expressions: keys,
        aggregate_expressions: aggs,
        ..pb::Aggregate::default()
    })))
}

#[test]
fn aggregates_output_the_keys_then_the_aggregates() {
    let sum = func("sum", vec![attr("c")]);
    let shown = show(&aggregate_of(
        1,
        vec![attr("a"), attr("b")],
        vec![sum.clone()],
    ));
    assert_eq!(shown, "Aggregate [a, b], [a, b, sum(c)]\n+- Read t\n");
    let global = show(&aggregate_of(1, vec![], vec![sum.clone()]));
    assert_eq!(global, "Aggregate [], [sum(c)]\n+- Read t\n");
    assert!(
        show(&aggregate_of(2, vec![attr("a")], vec![sum.clone()]))
            .starts_with("Aggregate ROLLUP [a], [a, sum(c)]")
    );
    assert!(
        show(&aggregate_of(3, vec![attr("a")], vec![sum]))
            .starts_with("Aggregate CUBE [a], [a, sum(c)]")
    );
}

#[test]
fn grouping_sets() {
    let mut r = aggregate_of(5, vec![attr("a"), attr("b")], vec![func("count", vec![])]);
    if let Some(RelType::Aggregate(a)) = &mut r.rel_type {
        a.grouping_sets = vec![
            aggregate::GroupingSets {
                grouping_set: vec![attr("a")],
            },
            aggregate::GroupingSets {
                grouping_set: vec![attr("a"), attr("b")],
            },
            aggregate::GroupingSets::default(),
        ];
    }
    let PlanNode::Aggregate {
        grouping,
        aggregates,
        ..
    } = node(&r)
    else {
        panic!("not an aggregate")
    };
    let Grouping::GroupingSets { exprs, sets } = grouping else {
        panic!("not grouping sets")
    };
    assert_eq!(exprs.len(), 2);
    assert_eq!(sets.iter().map(Vec::len).collect::<Vec<_>>(), [1, 2, 0]);
    assert_eq!(aggregates.len(), 3);
}

#[test]
fn aggregate_errors() {
    let pivot = aggregate_of(4, vec![attr("a")], vec![]);
    let e = fails(translate(&pivot));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.message().contains("pivot"));
    let unspecified = aggregate_of(0, vec![], vec![]);
    assert_eq!(
        code(&unspecified),
        ErrorClass::ConnectInvalidPlanInvalidEnum
    );
    assert!(message(&aggregate_of(42, vec![], vec![])).contains("UNRECOGNIZED"));
    let no_input = relation(RelType::Aggregate(Box::new(pb::Aggregate {
        group_type: 1,
        ..pb::Aggregate::default()
    })));
    assert_eq!(
        code(&no_input),
        ErrorClass::ConnectInvalidPlanAggregateNeedsPlanInput
    );
}

// ---- withColumns -----------------------------------------------------------------------------

fn alias_of(names: &[&str], metadata: Option<&str>) -> expression::Alias {
    expression::Alias {
        expr: Some(Box::new(lit(LiteralType::Integer(1)))),
        name: names.iter().map(|n| (*n).to_owned()).collect(),
        metadata: metadata.map(str::to_owned),
    }
}

fn with_columns(aliases: Vec<expression::Alias>) -> Relation {
    relation(RelType::WithColumns(Box::new(pb::WithColumns {
        input: input(),
        aliases,
    })))
}

#[test]
fn with_columns_names_one_column_per_alias() {
    let r = with_columns(vec![alias_of(&["x"], None), alias_of(&["y"], Some(""))]);
    let PlanNode::WithColumns { columns, .. } = node(&r) else {
        panic!("not WithColumns")
    };
    let names: Vec<_> = columns.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["x", "y"]);
    assert_eq!(show(&r), "WithColumns [x = 1, y = 1]\n+- Read t\n");
}

#[test]
fn with_columns_errors() {
    for names in [&["a", "b"][..], &[][..]] {
        let r = with_columns(vec![alias_of(names, None)]);
        assert_eq!(
            code(&r),
            ErrorClass::ConnectInvalidPlanWithColumnsRequireSingleNamePart
        );
    }
    assert!(message(&with_columns(vec![alias_of(&["a", "b"], None)])).contains("[a, b]"));
    let metadata = with_columns(vec![alias_of(&["x"], Some("{\"k\":1}"))]);
    assert_eq!(code(&metadata), ErrorClass::KrishivUnsupportedFeature);
}

// ---- plan ids, unsupported relations ---------------------------------------------------------

#[test]
fn a_relations_plan_id_becomes_its_origin() {
    let plan_id = |p: &LogicalPlan| p.origin.plan_id;
    let filter = |id| {
        let inner = with_id(read_table("t"), 1);
        let outer = relation(RelType::Filter(Box::new(pb::Filter {
            input: boxed(inner),
            condition: Some(attr("a")),
        })));
        match id {
            Some(id) => with_id(outer, id),
            None => outer,
        }
    };
    let top = root(&filter(Some(2)));
    assert_eq!(plan_id(&top), Some(PlanId(2)));
    let PlanNode::Filter { input, .. } = &top.node else {
        panic!("not a filter")
    };
    assert_eq!(plan_id(input), Some(PlanId(1)));
    // 0 is an id; no common message is none; neither is the origin's SQL part.
    assert_eq!(plan_id(&root(&filter(Some(0)))), Some(PlanId(0)));
    assert_eq!(plan_id(&root(&filter(None))), None);
    assert_eq!(top.origin.sql, None);
    // The id never reaches the structural key.
    let a = translate(&filter(Some(7))).unwrap();
    let b = translate(&filter(Some(8))).unwrap();
    assert!(a.structurally_eq(&b));
    assert_eq!(a.structural_hash(), b.structural_hash());
}

#[test]
fn unsupported_relations_are_named() {
    for (kind, name) in [
        (
            RelType::ShowString(Box::default()),
            "ShowString (spark.connect.Relation.show_string)",
        ),
        (
            RelType::ToDf(Box::default()),
            "ToDF (spark.connect.Relation.to_df)",
        ),
        (
            RelType::SetOp(Box::default()),
            "SetOperation (spark.connect.Relation.set_op)",
        ),
        (RelType::Deduplicate(Box::default()), "Deduplicate"),
        (RelType::LocalRelation(Default::default()), "LocalRelation"),
        (RelType::Drop(Box::default()), "Drop"),
        (RelType::WithRelations(Box::default()), "WithRelations"),
    ] {
        let r = relation(kind);
        let e = fails(translate(&r));
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{name}"
        );
        assert_eq!(e.sql_state(), Some("0A000"));
        assert!(e.message().contains(name), "{name}: {}", e.message());
    }
}

// ---- secrets ---------------------------------------------------------------------------------

#[test]
fn secrets_in_read_options_reach_no_printer_key_or_error() {
    const CANARY: &str = "Canary-91c4e0";
    let r = with_id(
        data_source(read::DataSource {
            format: Some("parquet".into()),
            options: BTreeMap::from([
                ("password".to_owned(), CANARY.to_owned()),
                ("fs.s3a.secret.key".to_owned(), CANARY.to_owned()),
                (
                    "jdbc".to_owned(),
                    format!("jdbc:postgresql://u:{CANARY}@h/db?password={CANARY}"),
                ),
            ]),
            paths: vec![format!(
                "https://h/p?X-Amz-Signature={CANARY}&X-Amz-Expires=60"
            )],
            ..read::DataSource::default()
        }),
        3,
    );
    let plan = translate(&r).unwrap();
    for (what, text) in [
        ("Display", plan.to_string()),
        ("Debug", format!("{plan:?}")),
        ("alternate Debug", format!("{plan:#?}")),
        ("root Debug", format!("{:?}", plan.root())),
    ] {
        assert!(!text.contains(CANARY), "{what} leaks the secret: {text}");
    }
    let key = plan.canonical_encoding();
    assert!(
        !key.windows(CANARY.len()).any(|w| w == CANARY.as_bytes()),
        "the encoding leaks"
    );
    // The decoded proto prints the sizes of its option maps and paths, not
    // their content.
    for text in [format!("{r:?}"), format!("{r:#?}")] {
        assert!(!text.contains(CANARY), "the proto's Debug leaks: {text}");
        assert!(text.contains("<redacted: 3 entries>"), "{text}");
        assert!(text.contains("<redacted: 1 values>"), "{text}");
    }
    let decoded = crate::wire::decode_relation(&prost::Message::encode_to_vec(&r)).unwrap();
    assert!(!format!("{decoded:?}").contains(CANARY));
    // Errors name proto fields, never option values.
    let mut streaming = r.clone();
    if let Some(RelType::Read(read)) = &mut streaming.rel_type {
        read.is_streaming = true;
    }
    let e = fails(translate(&streaming));
    assert!(!e.to_string().contains(CANARY));
    let mut with_schema = r;
    if let Some(RelType::Read(pb::Read {
        read_type: Some(ReadType::DataSource(d)),
        ..
    })) = &mut with_schema.rel_type
    {
        d.schema = Some("a INT".into());
    }
    assert!(!fails(translate(&with_schema)).to_string().contains(CANARY));
    // The printed plan still names the options.
    assert!(
        plan.to_string().contains("password=*********(redacted)"),
        "{plan}"
    );
    let _ = conf();
}
