// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relation arms: shapes, the host's part and the refusals.

use krishiv_common::error::ErrorClass;
use krishiv_plan::{PlanNode, PythonEvalType, PythonUdfRef};
use krishiv_types::DataType;

use super::super::testing::{attr, boxed, conf, fails, proto_type, read_table, relation};
use super::super::{
    PythonUdf, Result, UdfCallName, UdfHost, translate_relation, translate_relation_with,
};
use crate::proto::v4_3::{self as pb, data_type::Kind, relation::RelType};

/// A host that names every relation UDF `f` at version 7, every inline call `session.inline<n>`, and
/// records what it was sent.
#[derive(Default)]
struct Host {
    seen: Vec<PythonUdf>,
    inline: Vec<PythonUdf>,
}

impl UdfHost for Host {
    fn inline_python_udf(&mut self, udf: &PythonUdf) -> Result<UdfCallName> {
        self.inline.push(udf.clone());
        Ok(UdfCallName {
            qualifier: vec!["session".to_owned()],
            name: format!("inline{}", self.inline.len()),
        })
    }

    fn relation_python_udf(&mut self, udf: &PythonUdf) -> Result<PythonUdfRef> {
        self.seen.push(udf.clone());
        Ok(PythonUdfRef {
            name: "f".to_owned(),
            version: 7,
            eval_type: udf.plan_eval_type,
            thread_safe: true,
            deterministic: udf.deterministic,
        })
    }

    fn inline_python_udtf(&mut self, _: &super::super::PythonUdtf) -> Result<String> {
        Err(super::super::errors::unsupported("UDTFs in this test host"))
    }
}

/// `struct<x: bigint, y: string not null>`.
fn schema() -> pb::DataType {
    let field = |name: &str, kind: Kind, nullable: bool| pb::data_type::StructField {
        name: name.to_owned(),
        data_type: Some(proto_type(kind)),
        nullable,
        metadata: None,
    };
    proto_type(Kind::Struct(pb::data_type::Struct {
        fields: vec![
            field("x", Kind::Long(pb::data_type::Long::default()), true),
            field("y", Kind::String(pb::data_type::String::default()), false),
        ],
        type_variation_reference: 0,
    }))
}

fn func(eval_type: i32, output: pb::DataType) -> pb::CommonInlineUserDefinedFunction {
    pb::CommonInlineUserDefinedFunction {
        function_name: "my_func".to_owned(),
        deterministic: true,
        arguments: vec![attr("k"), attr("v")],
        is_distinct: false,
        function: Some(
            pb::common_inline_user_defined_function::Function::PythonUdf(pb::PythonUdf {
                output_type: Some(output),
                eval_type,
                command: vec![1, 2, 3],
                python_ver: "3.12".to_owned(),
                additional_includes: Vec::new(),
            }),
        ),
    }
}

fn map(eval_type: i32) -> pb::MapPartitions {
    pb::MapPartitions {
        input: boxed(read_table("t")),
        func: Some(func(eval_type, schema())),
        is_barrier: None,
        profile_id: None,
    }
}

fn group(eval_type: i32) -> pb::GroupMap {
    pb::GroupMap {
        input: boxed(read_table("t")),
        grouping_expressions: vec![attr("k")],
        func: Some(func(eval_type, schema())),
        ..pb::GroupMap::default()
    }
}

fn cogroup(eval_type: i32) -> pb::CoGroupMap {
    pb::CoGroupMap {
        input: boxed(read_table("t")),
        input_grouping_expressions: vec![attr("k")],
        other: boxed(read_table("u")),
        other_grouping_expressions: vec![attr("k2")],
        func: Some(func(eval_type, schema())),
        ..pb::CoGroupMap::default()
    }
}

fn with_host(r: &pb::Relation) -> (Result<krishiv_plan::Plan>, Host) {
    let mut host = Host::default();
    let plan = translate_relation_with(r, &conf(), &mut host);
    (plan, host)
}

fn ok(r: &pb::Relation) -> (krishiv_plan::Plan, Host) {
    let (plan, host) = with_host(r);
    (plan.unwrap(), host)
}

fn refused(r: &pb::Relation) -> krishiv_common::error::KrishivError {
    fails(with_host(r).0)
}

fn declared(output: &[krishiv_plan::BoundColumn]) -> Vec<(&str, DataType, bool)> {
    output
        .iter()
        .map(|c| (c.name.as_str(), c.data_type.clone(), c.nullable))
        .collect()
}

fn expected() -> Vec<(&'static str, DataType, bool)> {
    vec![("x", DataType::Long, true), ("y", DataType::String, false)]
}

#[test]
fn map_in_pandas_and_map_in_arrow_become_map_partitions() {
    for (code, t) in [
        (205, PythonEvalType::MapPandasIter),
        (207, PythonEvalType::MapArrowIter),
    ] {
        let (plan, host) = ok(&relation(RelType::MapPartitions(Box::new(map(code)))));
        let PlanNode::MapPartitions {
            input,
            udf,
            output,
            barrier,
        } = &plan.root().node
        else {
            panic!("{plan}")
        };
        assert_eq!(input.to_string(), "Read t\n");
        assert_eq!((udf.name.as_str(), udf.version, udf.eval_type), ("f", 7, t));
        assert_eq!(declared(output), expected());
        assert!(!barrier);
        let [sent] = host.seen.as_slice() else {
            panic!("{:?}", host.seen)
        };
        assert_eq!(
            (sent.name.as_str(), sent.eval_type, sent.plan_eval_type),
            ("my_func", code, t)
        );
        assert_eq!(
            (sent.command.as_slice(), sent.python_ver.as_str()),
            (&[1_u8, 2, 3][..], "3.12")
        );
        // The command never prints.
        assert!(format!("{sent:?}").contains("command_len: 3"));
    }
}

#[test]
fn a_barrier_map_or_a_resource_profile_is_refused() {
    let mut m = map(205);
    m.is_barrier = Some(true);
    let e = refused(&relation(RelType::MapPartitions(Box::new(m))));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        e.message().contains("MapPartitions.is_barrier"),
        "{}",
        e.message()
    );
    // `is_barrier = false` is the default.
    let mut m = map(205);
    m.is_barrier = Some(false);
    assert!(
        with_host(&relation(RelType::MapPartitions(Box::new(m))))
            .0
            .is_ok()
    );
    let mut m = map(207);
    m.profile_id = Some(1);
    let e = refused(&relation(RelType::MapPartitions(Box::new(m))));
    assert!(
        e.message().contains("MapPartitions.profile_id"),
        "{}",
        e.message()
    );
}

#[test]
fn an_eval_type_of_another_kind_is_sparks_function_eval_type_not_supported() {
    for code in [100, 101, 201, 206, 211, 300, 999] {
        let e = refused(&relation(RelType::MapPartitions(Box::new(map(code)))));
        assert_eq!(
            e.error_class(),
            ErrorClass::ConnectInvalidPlanFunctionEvalTypeNotSupported,
            "{code}"
        );
        assert!(e.message().contains(&code.to_string()), "{}", e.message());
    }
    for code in [205, 202, 206, 208] {
        let e = refused(&relation(RelType::GroupMap(Box::new(group(code)))));
        assert_eq!(
            e.error_class(),
            ErrorClass::ConnectInvalidPlanFunctionEvalTypeNotSupported,
            "{code}"
        );
    }
    for code in [201, 205, 209, 212] {
        let e = refused(&relation(RelType::CoGroupMap(Box::new(cogroup(code)))));
        assert_eq!(
            e.error_class(),
            ErrorClass::ConnectInvalidPlanFunctionEvalTypeNotSupported,
            "{code}"
        );
    }
}

#[test]
fn grouped_maps_of_every_python_kind_become_group_map() {
    for (code, t) in [
        (201, PythonEvalType::GroupedMapPandas),
        (216, PythonEvalType::GroupedMapPandasIter),
        (209, PythonEvalType::GroupedMapArrow),
        (215, PythonEvalType::GroupedMapArrowIter),
    ] {
        let (plan, _) = ok(&relation(RelType::GroupMap(Box::new(group(code)))));
        let PlanNode::GroupMap {
            input,
            keys,
            udf,
            output,
        } = &plan.root().node
        else {
            panic!("{plan}")
        };
        assert_eq!(input.to_string(), "Read t\n");
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].to_string(), "k");
        assert_eq!(udf.eval_type, t);
        assert_eq!(declared(output), expected());
    }
    // No grouping expressions: one group.
    let mut g = group(201);
    g.grouping_expressions.clear();
    let (plan, _) = ok(&relation(RelType::GroupMap(Box::new(g))));
    assert!(matches!(&plan.root().node, PlanNode::GroupMap { keys, .. } if keys.is_empty()));
}

/// Sets one field of a grouped map.
type Setter = Box<dyn Fn(&mut pb::GroupMap)>;

#[test]
fn transform_with_state_and_the_stateful_fields_are_refused() {
    for code in [211, 212, 213, 214] {
        let e = refused(&relation(RelType::GroupMap(Box::new(group(code)))));
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{code}"
        );
        assert!(
            e.message().contains("transformWithState"),
            "{}",
            e.message()
        );
    }
    let cases: Vec<(&str, Setter)> = vec![
        (
            "sorting_expressions",
            Box::new(|g: &mut pb::GroupMap| g.sorting_expressions.push(attr("v"))),
        ),
        (
            "initial_input",
            Box::new(|g: &mut pb::GroupMap| g.initial_input = boxed(read_table("s"))),
        ),
        (
            "initial_grouping_expressions",
            Box::new(|g: &mut pb::GroupMap| g.initial_grouping_expressions.push(attr("k"))),
        ),
        (
            "is_map_groups_with_state",
            Box::new(|g: &mut pb::GroupMap| g.is_map_groups_with_state = Some(false)),
        ),
        (
            "output_mode",
            Box::new(|g: &mut pb::GroupMap| g.output_mode = Some("append".to_owned())),
        ),
        (
            "timeout_conf",
            Box::new(|g: &mut pb::GroupMap| g.timeout_conf = Some("NoTimeout".to_owned())),
        ),
        (
            "state_schema",
            Box::new(|g: &mut pb::GroupMap| g.state_schema = Some(schema())),
        ),
        (
            "transform_with_state_info",
            Box::new(|g: &mut pb::GroupMap| {
                g.transform_with_state_info = Some(pb::TransformWithStateInfo::default());
            }),
        ),
    ];
    for (field, set) in cases {
        let mut g = group(201);
        set(&mut g);
        let e = refused(&relation(RelType::GroupMap(Box::new(g))));
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{field}"
        );
        assert!(
            e.message().contains(&format!("GroupMap.{field}")),
            "{field}: {}",
            e.message()
        );
    }
}

#[test]
fn cogrouped_maps_become_cogroup_map() {
    for (code, t) in [
        (206, PythonEvalType::CogroupedMapPandas),
        (210, PythonEvalType::CogroupedMapArrow),
    ] {
        let (plan, host) = ok(&relation(RelType::CoGroupMap(Box::new(cogroup(code)))));
        let PlanNode::CoGroupMap {
            left,
            right,
            left_keys,
            right_keys,
            udf,
            output,
        } = &plan.root().node
        else {
            panic!("{plan}")
        };
        assert_eq!(
            (left.to_string().as_str(), right.to_string().as_str()),
            ("Read t\n", "Read u\n")
        );
        assert_eq!(
            (left_keys[0].to_string(), right_keys[0].to_string()),
            ("k".to_owned(), "k2".to_owned())
        );
        assert_eq!(udf.eval_type, t);
        assert_eq!(declared(output), expected());
        assert_eq!(host.seen.len(), 1);
    }
}

#[test]
fn a_cogroup_refuses_sorting_and_unequal_key_counts() {
    for side in [0, 1] {
        let mut g = cogroup(206);
        if side == 0 {
            g.input_sorting_expressions.push(attr("k"));
        } else {
            g.other_sorting_expressions.push(attr("k2"));
        }
        let e = refused(&relation(RelType::CoGroupMap(Box::new(g))));
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(
            e.message().contains("sorting_expressions"),
            "{}",
            e.message()
        );
    }
    let mut g = cogroup(206);
    g.other_grouping_expressions.push(attr("v"));
    let e = refused(&relation(RelType::CoGroupMap(Box::new(g))));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.message().contains("key counts"), "{}", e.message());
}

#[test]
fn the_function_must_be_a_python_udf_returning_a_struct() {
    // A Scala function: Krishiv runs no JVM code.
    let mut m = map(205);
    if let Some(f) = &mut m.func {
        f.function = Some(
            pb::common_inline_user_defined_function::Function::ScalarScalaUdf(
                pb::ScalarScalaUdf::default(),
            ),
        );
    }
    let e = refused(&relation(RelType::MapPartitions(Box::new(m))));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported
    );
    // No function case.
    let mut m = map(205);
    if let Some(f) = &mut m.func {
        f.function = None;
    }
    let e = refused(&relation(RelType::MapPartitions(Box::new(m))));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    // No function at all.
    let mut g = group(201);
    g.func = None;
    let e = refused(&relation(RelType::GroupMap(Box::new(g))));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    // A scalar return type.
    let mut m = map(205);
    m.func = Some(func(
        205,
        proto_type(Kind::Long(pb::data_type::Long::default())),
    ));
    let e = refused(&relation(RelType::MapPartitions(Box::new(m))));
    assert_eq!(
        e.error_class(),
        ErrorClass::ConnectInvalidPlanAssertionFailure
    );
    // Additional includes (archives are PR-26c).
    let mut m = map(205);
    if let Some(pb::common_inline_user_defined_function::Function::PythonUdf(u)) =
        m.func.as_mut().and_then(|f| f.function.as_mut())
    {
        u.additional_includes.push("x.zip".to_owned());
    }
    let e = refused(&relation(RelType::MapPartitions(Box::new(m))));
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn without_a_host_the_relations_are_refused() {
    for r in [
        relation(RelType::MapPartitions(Box::new(map(205)))),
        relation(RelType::GroupMap(Box::new(group(201)))),
        relation(RelType::CoGroupMap(Box::new(cogroup(206)))),
    ] {
        let e = fails(translate_relation(&r, &conf()));
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(
            e.message().contains("Spark Connect server"),
            "{}",
            e.message()
        );
    }
}

#[test]
fn the_determinism_and_name_reach_the_host() {
    let mut g = group(201);
    if let Some(f) = &mut g.func {
        f.deterministic = false;
    }
    let (plan, host) = ok(&relation(RelType::GroupMap(Box::new(g))));
    assert!(!host.seen[0].deterministic);
    // The node carries the host's reference; the analyzer refuses it (krishiv-plan's tests).
    assert!(matches!(&plan.root().node, PlanNode::GroupMap { udf, .. } if !udf.deterministic));
}

#[test]
fn a_plan_mixing_an_inline_udf_and_a_map_translates_with_one_host() {
    // `df.select(my_udf(df.k).alias("k")).mapInPandas(f, schema)`: both UDFs go to the same host.
    let call = pb::Expression {
        common: None,
        expr_type: Some(pb::expression::ExprType::CommonInlineUserDefinedFunction(
            func(101, proto_type(Kind::Long(pb::data_type::Long::default()))),
        )),
    };
    let project = relation(RelType::Project(Box::new(pb::Project {
        input: boxed(read_table("t")),
        expressions: vec![call],
    })));
    let mut m = map(205);
    m.input = boxed(project);
    let (plan, host) = ok(&relation(RelType::MapPartitions(Box::new(m))));
    assert_eq!((host.inline.len(), host.seen.len()), (1, 1));
    assert_eq!(host.inline[0].eval_type, 101);
    assert_eq!(host.seen[0].eval_type, 205);
    let PlanNode::MapPartitions { input, udf, .. } = &plan.root().node else {
        panic!("{plan}")
    };
    assert_eq!(udf.version, 7);
    assert!(input.to_string().contains("inline1"), "{input}");
}
