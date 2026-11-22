// SPDX-License-Identifier: Apache-2.0
//! Tests of the relation translation.

use std::collections::BTreeMap;

use krishiv_common::error::ErrorClass;
use krishiv_common::secret::OptionValue;
use krishiv_plan::{ColumnNamesOp, PlanNode};

use super::super::testing::{
    attr, attr_with, boxed, conf, expr, fails, func, lit, read_files, read_table, relation, show,
    translate, with_id,
};
use super::super::{TranslateConf, translate_relation};
use super::*;
use crate::proto::v4_3::expression::{self, ExprType, literal::LiteralType};
use crate::proto::v4_3::{aggregate, join, read, relation::RelType};

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
        ..TranslateConf::default()
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
fn a_streaming_read_and_predicates_are_unsupported() {
    let mut streaming = read_files(Some("parquet"), &["/a"]);
    if let Some(RelType::Read(r)) = &mut streaming.rel_type {
        r.is_streaming = true;
    }
    assert!(message(&streaming).contains("is_streaming"));
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
    let tail = relation(RelType::Tail(Box::new(pb::Tail {
        input: input(),
        limit: 4,
    })));
    assert_eq!(show(&tail), "Tail 4\n+- Read t\n");
    let alias = relation(RelType::SubqueryAlias(Box::new(pb::SubqueryAlias {
        input: input(),
        alias: "x".into(),
        qualifier: vec![],
    })));
    assert_eq!(show(&alias), "SubqueryAlias x\n+- Read t\n");
}

#[test]
fn range_builds_the_range_leaf() {
    use crate::proto::v4_3 as pb;
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
fn a_project_without_input_reads_one_row() {
    let no_input = relation(RelType::Project(Box::default()));
    match node(&no_input) {
        PlanNode::Project { input, exprs } => {
            assert!(matches!(input.node, PlanNode::OneRowRelation));
            assert!(exprs.is_empty());
        }
        other => panic!("{other:?}"),
    }
}

fn aliased(qualifier: &[&str], alias: &str) -> Relation {
    relation(RelType::SubqueryAlias(Box::new(pb::SubqueryAlias {
        input: input(),
        alias: alias.into(),
        qualifier: qualifier.iter().map(|q| (*q).to_owned()).collect(),
    })))
}

#[test]
fn a_subquery_alias_qualifier_prefixes_the_alias_parts() {
    let sep = krishiv_plan::ALIAS_PART_SEP;
    let PlanNode::SubqueryAlias { alias, .. } = node(&aliased(&["db", "s"], "t")) else {
        panic!("not an alias")
    };
    assert_eq!(alias, format!("db{sep}s{sep}t"));
    let PlanNode::SubqueryAlias { alias, .. } = node(&aliased(&[], "t")) else {
        panic!("not an alias")
    };
    assert_eq!(alias, "t");
    // Printed dotted, as the IR prints a multi-part alias.
    assert_eq!(
        show(&aliased(&["db"], "t")),
        "SubqueryAlias db.t\n+- Read t\n"
    );
}

#[test]
fn an_alias_part_holding_the_separator_is_refused() {
    let sep = krishiv_plan::ALIAS_PART_SEP;
    for r in [
        aliased(&[&format!("a{sep}b")], "t"),
        aliased(&["db"], &format!("t{sep}u")),
        aliased(&[], &format!("t{sep}u")),
    ] {
        let e = fails(translate(&r));
        assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
        assert!(
            e.message().contains("SubqueryAlias.qualifier"),
            "{}",
            e.message()
        );
    }
}

fn read_with_schema(schema: &str) -> Relation {
    relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(ReadType::DataSource(read::DataSource {
            format: Some("csv".into()),
            paths: vec!["/a.csv".into()],
            schema: Some(schema.into()),
            ..read::DataSource::default()
        })),
    }))
}

#[test]
fn a_data_source_schema_is_the_reads_schema() {
    let ddl = node(&read_with_schema("a INT NOT NULL, b STRING"));
    let PlanNode::Read(ReadSource::Files { schema, .. }) = &ddl else {
        panic!("{ddl:?}")
    };
    let schema = schema.as_ref().expect("a schema");
    assert_eq!(schema.fields().len(), 2);
    assert!(!schema.fields()[0].nullable());
    // JSON is the fallback.
    let json = node(&read_with_schema(
        r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":false,"metadata":{}},{"name":"b","type":"string","nullable":true,"metadata":{}}]}"#,
    ));
    let PlanNode::Read(ReadSource::Files {
        schema: from_json, ..
    }) = &json
    else {
        panic!("{json:?}")
    };
    assert_eq!(from_json.as_ref(), Some(schema));
    // A schema that is not a struct (JSON of a bare type) is an invalid plan.
    assert_eq!(
        fails(translate(&read_with_schema("\"integer\"")))
            .error_class()
            .name(),
        "CONNECT_INVALID_PLAN.INVALID_SCHEMA_NON_STRUCT_TYPE"
    );
    // Neither DDL nor JSON: a parse error, and the text is not echoed.
    let e = fails(translate(&read_with_schema("CANARY nonsense nonsense")));
    assert!(!e.message().contains("CANARY"), "{}", e.message());
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

fn set_op_of(
    kind: i32,
    all: Option<bool>,
    by_name: Option<bool>,
    missing: Option<bool>,
) -> Relation {
    relation(RelType::SetOp(Box::new(pb::SetOperation {
        left_input: boxed(read_table("l")),
        right_input: boxed(read_table("r")),
        set_op_type: kind,
        is_all: all,
        by_name,
        allow_missing_columns: missing,
    })))
}

#[test]
fn set_operations_follow_spark() {
    use pb::set_operation::SetOpType as T;
    // UNION ALL is a Union; a UNION that is not ALL (or has no is_all) is a Distinct over it.
    match node(&set_op_of(T::Union as i32, Some(true), None, None)) {
        PlanNode::Union {
            inputs,
            by_name,
            allow_missing_columns,
        } => {
            assert_eq!(inputs.len(), 2);
            assert!(!by_name && !allow_missing_columns);
        }
        other => panic!("{other:?}"),
    }
    for all in [Some(false), None] {
        match node(&set_op_of(T::Union as i32, all, Some(true), Some(true))) {
            PlanNode::Distinct { input } => assert!(matches!(
                &input.node,
                PlanNode::Union {
                    by_name: true,
                    allow_missing_columns: true,
                    ..
                }
            )),
            other => panic!("{other:?}"),
        }
    }
    assert!(matches!(
        node(&set_op_of(T::Except as i32, Some(true), None, None)),
        PlanNode::SetOp {
            op: krishiv_plan::SetOperation::Except,
            all: true,
            ..
        }
    ));
    assert!(matches!(
        node(&set_op_of(T::Intersect as i32, None, None, None)),
        PlanNode::SetOp {
            op: krishiv_plan::SetOperation::Intersect,
            all: false,
            ..
        }
    ));
}

#[test]
fn set_operation_errors_are_spark_classes() {
    use pb::set_operation::SetOpType as T;
    assert_eq!(
        code(&set_op_of(T::Except as i32, None, Some(true), None)),
        ErrorClass::ConnectInvalidPlanExceptDoesNotSupportUnionByName
    );
    assert_eq!(
        code(&set_op_of(T::Intersect as i32, None, Some(true), None)),
        ErrorClass::ConnectInvalidPlanIntersectDoesNotSupportUnionByName
    );
    assert_eq!(
        code(&set_op_of(T::Union as i32, None, None, Some(true))),
        ErrorClass::ConnectInvalidPlanUnionByNameAllowMissingColRequiresByName
    );
    assert_eq!(
        code(&set_op_of(0, None, None, None)),
        ErrorClass::ConnectInvalidPlanInvalidEnum
    );
    assert_eq!(
        code(&set_op_of(99, None, None, None)),
        ErrorClass::ConnectInvalidPlanInvalidEnum
    );
    let one_sided = relation(RelType::SetOp(Box::new(pb::SetOperation {
        left_input: boxed(read_table("l")),
        right_input: None,
        set_op_type: T::Union as i32,
        ..pb::SetOperation::default()
    })));
    assert_eq!(
        code(&one_sided),
        ErrorClass::ConnectInvalidPlanSetOperationMustHaveTwoInputs
    );
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
fn an_aliased_grouping_key_groups_by_the_bare_expression() {
    // `groupBy((a % 2).alias("m"))` (PY-009): the output keeps the alias, the
    // grouping expression is `a`, as `GROUP BY a` is.
    let aliased = expr(ExprType::Alias(Box::new(expression::Alias {
        expr: Some(Box::new(attr("a"))),
        name: vec!["m".to_owned()],
        metadata: None,
    })));
    let r = aggregate_of(1, vec![aliased], vec![func("count", vec![])]);
    let PlanNode::Aggregate {
        grouping,
        aggregates,
        ..
    } = node(&r)
    else {
        panic!("not an aggregate")
    };
    assert_eq!(
        show(&r),
        "Aggregate [a], [a AS m, count()]\n+- Read t\n",
        "{grouping:?} {aggregates:?}"
    );
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

// ---- generators in select lists and withColumn ----------------------------------------------

#[test]
fn a_project_lifts_its_generators_like_the_dataframe_api() {
    let r = project(vec![attr("a"), func("explode", vec![attr("b")])]);
    let PlanNode::Project { input, exprs } = node(&r) else {
        panic!("not Project")
    };
    assert!(matches!(input.node, PlanNode::Generate { .. }), "{input:?}");
    assert_eq!(exprs.len(), 2);
    // Two generators chain, a cross product.
    let two = project(vec![
        func("explode", vec![attr("b")]),
        func("posexplode", vec![attr("b")]),
    ]);
    let PlanNode::Project { input, .. } = node(&two) else {
        panic!("not Project")
    };
    let PlanNode::Generate { input: below, .. } = &input.node else {
        panic!("not Generate")
    };
    assert!(matches!(below.node, PlanNode::Generate { .. }));
    // Beside an aggregate the list is a global aggregation, whose generators the analyzer extracts.
    let agg = project(vec![
        func("explode", vec![attr("b")]),
        func("count", vec![lit(LiteralType::Integer(1))]),
    ]);
    assert!(matches!(node(&agg), PlanNode::Aggregate { .. }));
    // No generator: the plain projection.
    assert!(matches!(
        node(&project(vec![attr("a")])),
        PlanNode::Project { .. }
    ));
}

#[test]
fn with_columns_lifts_a_generator_column() {
    let mut generated = alias_of(&["x"], None);
    generated.expr = Some(Box::new(func("explode", vec![attr("b")])));
    let r = with_columns(vec![generated]);
    let PlanNode::Project { input, .. } = node(&r) else {
        panic!("not Project")
    };
    let PlanNode::WithColumns { input: below, .. } = &input.node else {
        panic!("not WithColumns")
    };
    assert!(matches!(below.node, PlanNode::Generate { .. }));
}

#[test]
fn with_columns_refuses_a_generator_beside_an_alias_reading_a_replaced_column() {
    let mut generated = alias_of(&["x"], None);
    generated.expr = Some(Box::new(func("explode", vec![attr("b")])));
    let mut reader = alias_of(&["y"], None);
    reader.expr = Some(Box::new(attr("x")));
    let r = with_columns(vec![generated, reader]);
    assert_eq!(code(&r), ErrorClass::KrishivUnsupportedFeature);
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
        (RelType::ShowString(Box::default()), "SHOW_STRING(20)"),
        (RelType::Transpose(Box::default()), "TRANSPOSE(42)"),
        (RelType::WithRelations(Box::default()), "WITH_RELATIONS(41)"),
    ] {
        let r = relation(kind);
        let e = fails(translate(&r));
        assert_eq!(
            e.error_class(),
            ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported,
            "{name}"
        );
        assert_eq!(e.sql_state(), Some("56K00"));
        assert!(
            e.message()
                .contains(&format!("spark.connect.Relation is not supported: {name}")),
            "{name}: {}",
            e.message()
        );
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
    let mut with_predicates = r;
    if let Some(RelType::Read(pb::Read {
        read_type: Some(ReadType::DataSource(d)),
        ..
    })) = &mut with_predicates.rel_type
    {
        d.predicates = vec!["a < 1".into()];
    }
    assert!(
        !fails(translate(&with_predicates))
            .to_string()
            .contains(CANARY)
    );
    // The printed plan still names the options.
    assert!(
        plan.to_string().contains("password=*********(redacted)"),
        "{plan}"
    );
    let _ = conf();
}

#[test]
fn collect_metrics_is_the_observe_node() {
    let r = with_id(
        relation(RelType::CollectMetrics(Box::new(pb::CollectMetrics {
            input: input(),
            name: "obs".into(),
            metrics: vec![func("count", vec![lit(LiteralType::Integer(1))])],
        }))),
        9,
    );
    let plan = root(&r);
    let PlanNode::CollectMetrics { name, metrics, .. } = &plan.node else {
        panic!("a CollectMetrics node, got {:?}", plan.node);
    };
    assert_eq!((name.as_str(), metrics.len()), ("obs", 1));
    assert_eq!(plan.origin.plan_id.map(|p| p.0), Some(9));
}

#[test]
fn collect_metrics_without_an_input_is_refused() {
    let r = relation(RelType::CollectMetrics(Box::new(pb::CollectMetrics {
        input: None,
        name: "obs".into(),
        metrics: vec![],
    })));
    // The required-input check's own error, not the unsupported-variant fallback's.
    let e = fails(translate(&r));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    assert_eq!(e.sql_state(), Some("56K00"));
    assert_eq!(
        e.message_parameters(),
        [(
            "message",
            "CollectMetrics needs an input relation".to_owned()
        )]
    );
}

// ---- group 3-5 arms: sample, deduplicate, hint, repartition, unpivot ---------------------------

#[test]
fn sample_is_bernoulli_without_replacement() {
    let sample = |with_replacement: Option<bool>, deterministic_order: bool| {
        relation(RelType::Sample(Box::new(pb::Sample {
            input: input(),
            lower_bound: 0.0,
            upper_bound: 0.5,
            with_replacement,
            seed: Some(7),
            deterministic_order,
        })))
    };
    let PlanNode::Sample { bounds, seed, .. } = node(&sample(None, false)) else {
        panic!("not a sample")
    };
    assert_eq!((bounds.lower, bounds.upper, seed), (0.0, 0.5, Some(7)));
    assert!(matches!(
        node(&sample(Some(false), false)),
        PlanNode::Sample { .. }
    ));
    assert_eq!(
        code(&sample(Some(true), false)),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(
        code(&sample(None, true)),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn sample_bounds_must_span_at_most_the_unit_interval() {
    let sample = |lower: f64, upper: f64| {
        relation(RelType::Sample(Box::new(pb::Sample {
            input: input(),
            lower_bound: lower,
            upper_bound: upper,
            with_replacement: None,
            seed: Some(1),
            deterministic_order: false,
        })))
    };
    assert!(matches!(node(&sample(0.0, 1.0)), PlanNode::Sample { .. }));
    assert!(matches!(
        node(&sample(0.0, 1.0 + 5e-7)),
        PlanNode::Sample { .. }
    ));
    for (lower, upper, shown) in [
        (0.0, 1.5, "1.5"),
        (0.0, -0.1, "-0.1"),
        (0.5, 0.0, "-0.5"),
        (0.0, f64::NAN, "NaN"),
    ] {
        let r = sample(lower, upper);
        assert_eq!(
            code(&r),
            ErrorClass::KrishivSampleFractionOutOfRange,
            "{lower} {upper}"
        );
        assert_eq!(
            message(&r),
            format!(
                "requirement failed: Sampling fraction ({shown}) must be on interval [0, 1] without replacement"
            )
        );
    }
    // The input is translated first: a bad input is reported before a bad fraction.
    let bad_input = relation(RelType::Sample(Box::new(pb::Sample {
        input: None,
        lower_bound: 0.0,
        upper_bound: 1.5,
        with_replacement: None,
        seed: Some(1),
        deterministic_order: false,
    })));
    assert_eq!(
        code(&bad_input),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
}

#[test]
fn deduplicate_over_all_columns_is_distinct() {
    let dedup = |all: Option<bool>, names: &[&str], watermark: Option<bool>| {
        relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
            input: input(),
            column_names: names.iter().map(|n| (*n).to_owned()).collect(),
            all_columns_as_keys: all,
            within_watermark: watermark,
        })))
    };
    assert!(matches!(
        node(&dedup(Some(true), &[], None)),
        PlanNode::Distinct { .. }
    ));
    assert_eq!(
        code(&dedup(None, &[], None)),
        ErrorClass::ConnectInvalidPlanDeduplicateRequiresColumnsOrAll
    );
    assert_eq!(
        code(&dedup(Some(true), &["a"], None)),
        ErrorClass::ConnectInvalidPlanDeduplicateAllColumnsAndSubset
    );
    // No input: Spark's first check.
    let no_input = relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
        input: None,
        column_names: vec![],
        all_columns_as_keys: Some(true),
        within_watermark: None,
    })));
    assert_eq!(
        code(&no_input),
        ErrorClass::ConnectInvalidPlanDeduplicateNeedsInput
    );
    // The column-field checks come before the input is translated: an input that cannot be
    // translated (a sample with replacement) does not hide them.
    let bad_input = relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
        input: boxed(relation(RelType::Sample(Box::new(pb::Sample {
            input: input(),
            lower_bound: 0.0,
            upper_bound: 0.5,
            with_replacement: Some(true),
            seed: None,
            deterministic_order: false,
        })))),
        column_names: vec![],
        all_columns_as_keys: None,
        within_watermark: None,
    })));
    assert_eq!(
        code(&bad_input),
        ErrorClass::ConnectInvalidPlanDeduplicateRequiresColumnsOrAll
    );
    for unsupported in [
        dedup(None, &["a"], None),
        dedup(Some(true), &[], Some(true)),
    ] {
        assert_eq!(code(&unsupported), ErrorClass::KrishivUnsupportedFeature);
    }
}

#[test]
fn hint_and_repartition_by_expression() {
    let hint = relation(RelType::Hint(Box::new(pb::Hint {
        input: input(),
        name: "broadcast".into(),
        parameters: vec![],
    })));
    let PlanNode::Hint { name, params, .. } = node(&hint) else {
        panic!("not a hint")
    };
    assert_eq!((name.as_str(), params.len()), ("broadcast", 0));
    let repartition = |exprs: Vec<Expression>, n: Option<i32>| {
        relation(RelType::RepartitionByExpression(Box::new(
            pb::RepartitionByExpression {
                input: input(),
                partition_exprs: exprs,
                num_partitions: n,
            },
        )))
    };
    let PlanNode::Repartition { exprs, .. } = node(&repartition(vec![attr("a")], None)) else {
        panic!("not a repartition")
    };
    assert_eq!(exprs.len(), 1);
    assert_eq!(
        code(&repartition(vec![attr("a")], Some(4))),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(
        code(&repartition(vec![], None)),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn unpivot_projects_the_ids_and_values_and_keeps_nulls() {
    let unpivot = |values: Option<Vec<Expression>>| {
        relation(RelType::Unpivot(Box::new(pb::Unpivot {
            input: input(),
            ids: vec![attr("id")],
            values: values.map(|values| pb::unpivot::Values { values }),
            variable_column_name: "k".into(),
            value_column_name: "v".into(),
        })))
    };
    let PlanNode::Unpivot {
        input,
        value_columns,
        name_column,
        column_sets,
        include_nulls,
    } = node(&unpivot(Some(vec![attr("a"), attr("b")])))
    else {
        panic!("not an unpivot")
    };
    assert_eq!(
        (value_columns, name_column),
        (vec!["v".to_owned()], "k".to_owned())
    );
    assert_eq!(column_sets.len(), 2);
    assert!(include_nulls);
    let PlanNode::Project { exprs, .. } = &input.node else {
        panic!("ids and values are projected first")
    };
    assert_eq!(exprs.len(), 3);
    assert_eq!(code(&unpivot(None)), ErrorClass::KrishivUnsupportedFeature);
    // An id that is also a value, or a value twice, is refused rather than projected wrongly.
    assert_eq!(
        code(&unpivot(Some(vec![attr("ID"), attr("b")]))),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(
        code(&unpivot(Some(vec![attr("a"), attr("a")]))),
        ErrorClass::KrishivUnsupportedFeature
    );
    // A qualified spelling of the id does not slip past the check.
    assert_eq!(
        code(&unpivot(Some(vec![attr("t.id"), attr("b")]))),
        ErrorClass::KrishivUnsupportedFeature
    );
    // An aliased id is keyed by its output name: `x AS a` collides with the value `a`.
    let aliased = relation(RelType::Unpivot(Box::new(pb::Unpivot {
        input: boxed(read_table("t")),
        ids: vec![expr(ExprType::Alias(Box::new(alias_of(&["a"], None))))],
        values: Some(pb::unpivot::Values {
            values: vec![attr("a")],
        }),
        variable_column_name: "k".into(),
        value_column_name: "v".into(),
    })));
    assert_eq!(code(&aliased), ErrorClass::KrishivUnsupportedFeature);
    // An empty list is the analyzer's to refuse (UNPIVOT_REQUIRES_VALUE_COLUMNS), as in Spark.
    assert!(matches!(
        node(&unpivot(Some(vec![]))),
        PlanNode::Unpivot { .. }
    ));
}

// ---- plan equality with the SQL-parsed plan (T012: `dataframe_equals_sql_plan`) ----------------

/// The translated plan and the plan of `sql`, printed without origins.
fn plans(rel: &Relation, sql: &str) -> (String, String) {
    let translated = translate(rel).expect("translates").to_string();
    let parsed = krishiv_sql::parse_and_lower(
        sql,
        &krishiv_sql::ParserOptions::default(),
        krishiv_plan::SourceId(0),
    )
    .expect("parses")
    .to_string();
    (translated, parsed)
}

/// `SELECT * FROM t` as a Connect relation, so both sides share the SQL plan's `Project [*]`.
fn star_input() -> Option<Box<Relation>> {
    boxed(project(vec![expr(ExprType::UnresolvedStar(
        Default::default(),
    ))]))
}

#[test]
fn dedup_over_all_columns_plans_as_select_distinct() {
    let rel = relation(RelType::Deduplicate(Box::new(pb::Deduplicate {
        input: star_input(),
        column_names: vec![],
        all_columns_as_keys: Some(true),
        within_watermark: None,
    })));
    let (translated, parsed) = plans(&rel, "SELECT DISTINCT * FROM t");
    assert_eq!(translated, parsed);
}

#[test]
fn a_hint_plans_as_the_sql_hint() {
    // PySpark sends the name as the caller typed it; SQL keeps its spelling too.
    let rel = relation(RelType::Hint(Box::new(pb::Hint {
        input: star_input(),
        name: "BROADCAST".into(),
        parameters: vec![],
    })));
    let (translated, parsed) = plans(&rel, "SELECT /*+ BROADCAST */ * FROM t");
    assert_eq!(translated, parsed);
}

#[test]
fn repartition_by_expression_plans_as_distribute_by() {
    let rel = relation(RelType::RepartitionByExpression(Box::new(
        pb::RepartitionByExpression {
            input: star_input(),
            partition_exprs: vec![attr("a")],
            num_partitions: None,
        },
    )));
    let (translated, parsed) = plans(&rel, "SELECT * FROM t DISTRIBUTE BY a");
    assert_eq!(translated, parsed);
}

#[test]
fn repartition_and_coalesce_plan_as_their_hints() {
    let rel = |shuffle: Option<bool>| {
        relation(RelType::Repartition(Box::new(pb::Repartition {
            input: star_input(),
            num_partitions: 4,
            shuffle,
        })))
    };
    // `shuffle` is `optional bool`, default false: Spark reads it with `getShuffle`, so unset is `coalesce`.
    let (translated, parsed) = plans(&rel(None), "SELECT /*+ COALESCE(4) */ * FROM t");
    assert_eq!(translated, parsed);
    let (translated, parsed) = plans(&rel(Some(true)), "SELECT /*+ REPARTITION(4) */ * FROM t");
    assert_eq!(translated, parsed);
    let (translated, parsed) = plans(&rel(Some(false)), "SELECT /*+ COALESCE(4) */ * FROM t");
    assert_eq!(translated, parsed);
}

#[test]
fn na_replace_flattens_the_pairs_for_the_analyzer() {
    let long = |v: i64| pb::expression::Literal {
        literal_type: Some(pb::expression::literal::LiteralType::Long(v)),
        data_type: None,
    };
    let rel = relation(RelType::Replace(Box::new(pb::NaReplace {
        input: input(),
        cols: vec!["a".into()],
        replacements: vec![pb::na_replace::Replacement {
            old_value: Some(long(1)),
            new_value: Some(long(10)),
        }],
    })));
    let PlanNode::ColumnNames { op, .. } = node(&rel) else {
        panic!("not a ColumnNames node")
    };
    let ColumnNamesOp::Replace { cols, pairs } = op else {
        panic!("not a Replace op")
    };
    assert!(cols == ["a"] && pairs.len() == 2);
    let pairs: Vec<String> = pairs.iter().map(|e| format!("{e:?}")).collect();
    assert!(
        pairs[0].contains("Long(1)") && !pairs[0].contains("Long(10)"),
        "{pairs:?}"
    );
    assert!(pairs[1].contains("Long(10)"), "{pairs:?}");
}

#[test]
fn na_fill_and_na_drop_translate_and_check_their_values() {
    let long = |v: i64| pb::expression::Literal {
        literal_type: Some(pb::expression::literal::LiteralType::Long(v)),
        data_type: None,
    };
    let fill = |cols: Vec<&str>, values: Vec<pb::expression::Literal>| {
        relation(RelType::FillNa(Box::new(pb::NaFill {
            input: input(),
            cols: cols.into_iter().map(String::from).collect(),
            values,
        })))
    };
    let PlanNode::ColumnNames { op, .. } = node(&fill(vec!["a"], vec![long(1)])) else {
        panic!("not a ColumnNames node")
    };
    assert!(matches!(op, ColumnNamesOp::FillNa { .. }));
    assert_eq!(
        code(&fill(vec!["a"], vec![])),
        ErrorClass::ConnectInvalidPlanNaFillValuesEmpty
    );
    assert_eq!(
        code(&fill(vec!["a"], vec![long(1), long(2)])),
        ErrorClass::ConnectInvalidPlanNaFillValuesLengthMismatch
    );
    let drop = relation(RelType::DropNa(Box::new(pb::NaDrop {
        input: input(),
        cols: vec!["a".into()],
        min_non_nulls: Some(2),
    })));
    let PlanNode::ColumnNames { op, .. } = node(&drop) else {
        panic!("not a ColumnNames node")
    };
    assert_eq!(
        op,
        ColumnNamesOp::DropNa {
            subset: vec!["a".into()],
            min_non_nulls: Some(2)
        }
    );
}

#[test]
fn cov_and_corr_plan_as_one_row_aggregates_named_for_the_statistic() {
    for (rel, expected) in [
        (
            relation(RelType::Cov(Box::new(pb::StatCov {
                input: input(),
                col1: "a".into(),
                col2: "b".into(),
            }))),
            "cov",
        ),
        (
            relation(RelType::Corr(Box::new(pb::StatCorr {
                input: input(),
                col1: "a".into(),
                col2: "b".into(),
                method: Some("pearson".into()),
            }))),
            "corr",
        ),
    ] {
        let PlanNode::Aggregate {
            grouping,
            aggregates,
            ..
        } = node(&rel)
        else {
            panic!("not an Aggregate node")
        };
        assert!(matches!(grouping, Grouping::GroupBy(k) if k.is_empty()));
        assert!(matches!(&aggregates[0].node, ExprNode::Alias { name, .. } if name == expected));
    }
}

#[test]
fn cov_reads_its_columns_as_attribute_references() {
    let rel = relation(RelType::Cov(Box::new(pb::StatCov {
        input: input(),
        col1: "t.x".into(),
        col2: "`a.b`".into(),
    })));
    let PlanNode::Aggregate { aggregates, .. } = node(&rel) else {
        panic!("not an Aggregate node")
    };
    // The two argument columns of covar_samp, under the alias and the zero fill: `t.x` is the
    // qualified reference ["t", "x"], the quoted name the single part "a.b".
    let mut seen: Vec<Vec<String>> = Vec::new();
    fn columns(e: &krishiv_plan::Expr, out: &mut Vec<Vec<String>>) {
        if let krishiv_plan::ExprNode::Column(c) = &e.node {
            out.push(c.name.clone());
        }
        e.for_each_child(&mut |child| columns(child, out));
    }
    columns(&aggregates[0], &mut seen);
    assert_eq!(
        seen,
        vec![vec!["t".to_owned(), "x".to_owned()], vec!["a.b".to_owned()]]
    );
}

#[test]
fn corr_refuses_a_method_other_than_pearson() {
    let rel = relation(RelType::Corr(Box::new(pb::StatCorr {
        input: input(),
        col1: "a".into(),
        col2: "b".into(),
        method: Some("spearman".into()),
    })));
    assert_eq!(code(&rel), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn sample_by_plans_a_filter_drawing_rand_for_every_row() {
    let stratum = |n: i32| pb::expression::Literal {
        literal_type: Some(LiteralType::Integer(n)),
        data_type: None,
    };
    let sample_by = |fraction: f64| {
        relation(RelType::SampleBy(Box::new(pb::StatSampleBy {
            input: input(),
            col: Some(attr("k")),
            fractions: vec![pb::stat_sample_by::Fraction {
                stratum: Some(stratum(1)),
                fraction,
            }],
            seed: Some(7),
        })))
    };
    let rel = sample_by(0.5);
    let PlanNode::Filter { predicate, .. } = node(&rel) else {
        panic!("not a Filter node")
    };
    let text = format!("{predicate:?}");
    // rand(7) < CASE WHEN k = 1 THEN 0.5 ELSE 0.0 END, in this order.
    let in_order = [
        "name: \"<\"",
        "name: \"rand\"",
        "Literal(Long(7))",
        "name: \"when\"",
        "name: \"==\"",
        "name: [\"k\"]",
        "Literal(Integer(1))",
        "Literal(Double(0.5))",
        "Literal(Double(0.0))",
    ];
    let mut at = 0;
    for part in in_order {
        let found = text[at..]
            .find(part)
            .unwrap_or_else(|| panic!("{part} after {at}: {text}"));
        at += found + part.len();
    }
    assert_eq!(
        code(&sample_by(1.5)),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
}

#[test]
fn unpivot_plans_as_unpivot_include_nulls() {
    let rel = relation(RelType::Unpivot(Box::new(pb::Unpivot {
        input: input(),
        ids: vec![attr("id")],
        values: Some(pb::unpivot::Values {
            values: vec![attr("a"), attr("b")],
        }),
        variable_column_name: "k".into(),
        value_column_name: "v".into(),
    })));
    let (translated, parsed) = plans(
        &rel,
        "SELECT * FROM (SELECT id, a, b FROM t) UNPIVOT INCLUDE NULLS (v FOR k IN (a, b))",
    );
    // The SQL form wraps the node in `Project [*]` and a subquery alias; the node and the
    // projection under it are the same.
    let line = |text: &str, prefix: &str| {
        text.lines()
            .find(|l| l.trim_start_matches(['+', '-', ' ']).starts_with(prefix))
            .map(|l| l.trim_start_matches(['+', '-', ' ']).to_owned())
    };
    for prefix in ["UnpivotIncludeNulls", "Project [id, a, b]", "Read t"] {
        assert!(
            line(&translated, prefix).is_some(),
            "{prefix}\n{translated}"
        );
        assert_eq!(line(&translated, prefix), line(&parsed, prefix), "{prefix}");
    }
}

// ---- column-name arms: to_df, with_columns_renamed, drop ----------------------------------------

fn column_names_op(rel: &Relation) -> ColumnNamesOp {
    let PlanNode::ColumnNames { op, .. } = node(rel) else {
        panic!("not a ColumnNames node")
    };
    op
}

#[test]
fn to_df_keeps_the_names_in_order() {
    let rel = relation(RelType::ToDf(Box::new(pb::ToDf {
        input: input(),
        column_names: vec!["x".into(), "y".into()],
    })));
    assert_eq!(
        column_names_op(&rel),
        ColumnNamesOp::ToDf(vec!["x".into(), "y".into()])
    );
}

#[test]
fn with_columns_renamed_reads_the_list_then_the_old_map() {
    let listed = relation(RelType::WithColumnsRenamed(Box::new(
        pb::WithColumnsRenamed {
            input: input(),
            rename_columns_map: [("ignored".to_owned(), "x".to_owned())].into(),
            renames: vec![pb::with_columns_renamed::Rename {
                col_name: "a".into(),
                new_col_name: "b".into(),
            }],
        },
    )));
    assert_eq!(
        column_names_op(&listed),
        ColumnNamesOp::Rename(vec![("a".into(), "b".into())])
    );
    let mapped = relation(RelType::WithColumnsRenamed(Box::new(
        pb::WithColumnsRenamed {
            input: input(),
            rename_columns_map: [("a".to_owned(), "b".to_owned())].into(),
            renames: vec![],
        },
    )));
    assert_eq!(
        column_names_op(&mapped),
        ColumnNamesOp::Rename(vec![("a".into(), "b".into())])
    );
}

#[test]
fn drop_takes_names_and_plain_columns_and_refuses_other_expressions() {
    let rel = relation(RelType::Drop(Box::new(pb::Drop {
        input: input(),
        columns: vec![attr("a")],
        column_names: vec!["b".into()],
    })));
    assert_eq!(
        column_names_op(&rel),
        ColumnNamesOp::Drop {
            names: vec!["b".into()],
            columns: vec!["a".into()],
        }
    );
    let drop_of = |e: Expression| {
        relation(RelType::Drop(Box::new(pb::Drop {
            input: input(),
            columns: vec![e],
            column_names: vec![],
        })))
    };
    // A literal, a qualified column and a column tied to a DataFrame drop by attribute in Spark.
    for e in [
        lit(LiteralType::Integer(1)),
        attr("t.a"),
        attr_with("a", Some(7)),
    ] {
        assert_eq!(code(&drop_of(e)), ErrorClass::KrishivUnsupportedFeature);
    }
}

#[test]
fn a_rename_map_whose_result_depends_on_entry_order_is_refused() {
    let map = |entries: &[(&str, &str)]| {
        relation(RelType::WithColumnsRenamed(Box::new(
            pb::WithColumnsRenamed {
                input: input(),
                rename_columns_map: entries
                    .iter()
                    .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
                    .collect(),
                renames: vec![],
            },
        )))
    };
    // A chain, a new name equal to another key ignoring case, and two keys equal ignoring case.
    for entries in [
        &[("b", "c"), ("a", "b")][..],
        &[("a", "B"), ("b", "x")][..],
        &[("A", "x"), ("a", "y")][..],
    ] {
        assert_eq!(
            code(&map(entries)),
            ErrorClass::KrishivUnsupportedFeature,
            "{entries:?}"
        );
    }
    // A case-only rename and a no-op depend on nothing.
    for entries in [&[("name", "Name")][..], &[("a", "a")][..]] {
        assert!(
            matches!(column_names_op(&map(entries)), ColumnNamesOp::Rename(_)),
            "{entries:?}"
        );
    }
    assert_eq!(
        column_names_op(&map(&[("a", "x"), ("b", "y")])),
        ColumnNamesOp::Rename(vec![("a".into(), "x".into()), ("b".into(), "y".into())])
    );
}
