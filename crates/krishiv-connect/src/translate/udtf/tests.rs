// SPDX-License-Identifier: Apache-2.0
//! Tests of the Python UDTF translation: `register_table_function` (command 10) and
//! `common_inline_user_defined_table_function` (relation 38).

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::{ExprNode, PlanNode, PythonEvalType, PythonUdfRef};
use krishiv_types::DataType;

use super::*;
use crate::proto::v4_3::command::CommandType;
use crate::proto::v4_3::data_type::{Kind, Long};
use crate::proto::v4_3::expression::ExprType;
use crate::proto::v4_3::expression::literal::LiteralType;
use crate::proto::v4_3::relation::RelType;
use crate::translate::testing::{attr, conf, expr, fails, lit, proto_type, relation};
use crate::translate::{
    Command, PythonUdf, UdfCallName, UdfHost, translate_command, translate_relation,
    translate_relation_with,
};

/// A host that names every inline UDTF `udtf<n>` and records what it was sent.
#[derive(Default)]
struct Host(Vec<PythonUdtf>);

impl UdfHost for Host {
    fn inline_python_udf(&mut self, _: &PythonUdf) -> Result<UdfCallName> {
        Err(super::super::errors::unsupported(
            "scalar UDFs in this test host",
        ))
    }

    fn relation_python_udf(&mut self, _: &PythonUdf) -> Result<PythonUdfRef> {
        Err(super::super::errors::unsupported(
            "relations in this test host",
        ))
    }

    fn inline_python_udtf(&mut self, udtf: &PythonUdtf) -> Result<String> {
        self.0.push(udtf.clone());
        Ok(format!("udtf{}", self.0.len()))
    }
}

/// `struct<a: bigint>`.
fn row_type() -> pb::DataType {
    proto_type(Kind::Struct(pb::data_type::Struct {
        fields: vec![pb::data_type::StructField {
            name: "a".to_owned(),
            data_type: Some(proto_type(Kind::Long(Long::default()))),
            nullable: true,
            metadata: None,
        }],
        type_variation_reference: 0,
    }))
}

fn python(eval_type: i32, return_type: Option<pb::DataType>) -> pb::PythonUdtf {
    pb::PythonUdtf {
        return_type,
        eval_type,
        command: vec![9, 8, 7],
        python_ver: "3.12".to_owned(),
    }
}

fn udtf(
    function: Option<pb::common_inline_user_defined_table_function::Function>,
    arguments: Vec<pb::Expression>,
) -> pb::CommonInlineUserDefinedTableFunction {
    pb::CommonInlineUserDefinedTableFunction {
        function_name: "Repeat".to_owned(),
        deterministic: false,
        arguments,
        function,
    }
}

fn typed(eval_type: i32) -> Option<pb::common_inline_user_defined_table_function::Function> {
    Some(Function::PythonUdtf(python(eval_type, Some(row_type()))))
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

fn register(f: pb::CommonInlineUserDefinedTableFunction) -> pb::Command {
    pb::Command {
        command_type: Some(CommandType::RegisterTableFunction(Box::new(f))),
    }
}

fn call(f: pb::CommonInlineUserDefinedTableFunction) -> pb::Relation {
    relation(RelType::CommonInlineUserDefinedTableFunction(f))
}

fn long(v: i64) -> pb::Expression {
    lit(LiteralType::Long(v))
}

#[test]
fn register_table_function_keeps_the_eval_type_the_client_sent() {
    for (code, plan) in [
        (300, PythonEvalType::TableUdf),
        (301, PythonEvalType::ArrowTableUdf),
        (302, PythonEvalType::ArrowUdtf),
    ] {
        let Ok(Command::RegisterPythonUdtf(u)) =
            translate_command(&register(udtf(typed(code), Vec::new())))
        else {
            panic!("{code}: a registration");
        };
        assert_eq!(u.name, "Repeat");
        assert!(!u.deterministic);
        assert_eq!((u.eval_type, u.plan_eval_type), (code, plan));
        assert_eq!(
            u.return_type.as_ref().map(DataType::sql),
            Some("STRUCT<a: BIGINT>".to_owned())
        );
        assert_eq!(u.command, [9, 8, 7]);
        assert_eq!(u.python_ver, "3.12");
        // The payload never reaches a log.
        assert!(!format!("{u:?}").contains("[9, 8, 7]"));
    }
}

#[test]
fn a_return_type_string_is_parsed_as_the_planner_parses_it() {
    let unparsed = proto_type(Kind::Unparsed(pb::data_type::Unparsed {
        data_type_string: "a int, i int".to_owned(),
    }));
    let u = python_udtf(&udtf(
        Some(Function::PythonUdtf(python(300, Some(unparsed)))),
        Vec::new(),
    ))
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        u.return_type.as_ref().map(DataType::sql),
        Some("STRUCT<a: INT, i: INT>".to_owned())
    );
}

#[test]
fn a_missing_return_type_is_a_udtf_with_analyze() {
    let f = udtf(Some(Function::PythonUdtf(python(301, None))), Vec::new());
    let u = python_udtf(&f).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(u.return_type, None);
    assert_eq!(u.eval_type, 301);
}

#[test]
fn other_eval_types_non_struct_results_and_unset_functions_are_refused() {
    for code in [0, 100, 101, 205, 299, 303, 1000] {
        let e = fails(python_udtf(&udtf(typed(code), Vec::new())));
        assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature, "{code}");
        assert!(e.to_string().contains(&format!("eval type {code}")), "{e}");
    }
    // Spark's `createPythonUserDefinedTableFunction`: `invalidSchemaTypeNonStruct`.
    let scalar = Some(Function::PythonUdtf(python(
        300,
        Some(proto_type(Kind::Long(Long::default()))),
    )));
    let e = fails(python_udtf(&udtf(scalar, Vec::new())));
    assert_eq!(class(&e), ErrorClass::InvalidSchemaTypeNonStruct);
    assert!(e.to_string().contains("\"BIGINT\""), "{e}");
    let e = fails(python_udtf(&udtf(None, Vec::new())));
    assert_eq!(
        class(&e),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    assert!(e.to_string().contains("FUNCTION_NOT_SET"), "{e}");
}

#[test]
fn an_inline_udtf_is_the_table_function_the_host_names_over_its_arguments() {
    let c = conf();
    let mut host = Host::default();
    let plan = translate_relation_with(
        &call(udtf(typed(302), vec![long(7), attr("x")])),
        &c,
        &mut host,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let PlanNode::TableFunction { name, args } = &plan.root().node else {
        panic!("a table function: {plan}");
    };
    assert_eq!(name, "udtf1");
    assert_eq!(args.len(), 2);
    assert_eq!(args[1].to_string(), "x");
    let [u] = host.0.as_slice() else {
        panic!("one UDTF: {:?}", host.0);
    };
    assert_eq!(
        (u.eval_type, u.plan_eval_type),
        (302, PythonEvalType::ArrowUdtf)
    );
    assert_eq!(u.name, "Repeat");
}

#[test]
fn a_named_argument_is_an_alias_as_sql_lowers_it() {
    let c = conf();
    let mut host = Host::default();
    let named = expr(ExprType::NamedArgumentExpression(Box::new(
        pb::NamedArgumentExpression {
            key: "n".to_owned(),
            value: Some(Box::new(long(3))),
        },
    )));
    let plan = translate_relation_with(&call(udtf(typed(300), vec![named])), &c, &mut host)
        .unwrap_or_else(|e| panic!("{e}"));
    let PlanNode::TableFunction { args, .. } = &plan.root().node else {
        panic!("a table function: {plan}");
    };
    let ExprNode::Alias { name, .. } = &args[0].node else {
        panic!("a named argument: {:?}", args[0]);
    };
    assert_eq!(name, "n");
    // A plain alias is not a parameter name: refused rather than read as one.
    let aliased = expr(ExprType::Alias(Box::new(pb::expression::Alias {
        expr: Some(Box::new(long(3))),
        name: vec!["n".to_owned()],
        metadata: None,
    })));
    let e = fails(translate_relation_with(
        &call(udtf(typed(300), vec![aliased])),
        &c,
        &mut host,
    ));
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn a_table_argument_is_refused() {
    let c = conf();
    let mut host = Host::default();
    let table = expr(ExprType::SubqueryExpression(pb::SubqueryExpression {
        plan_id: 1,
        subquery_type: pb::subquery_expression::SubqueryType::TableArg as i32,
        table_arg_options: None,
        in_subquery_values: Vec::new(),
    }));
    let e = fails(translate_relation_with(
        &call(udtf(typed(300), vec![table])),
        &c,
        &mut host,
    ));
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("TABLE arguments"), "{e}");
    assert!(host.0.is_empty(), "nothing reached the host");
}

#[test]
fn without_a_host_an_inline_udtf_is_refused() {
    let e = fails(translate_relation(
        &call(udtf(typed(300), Vec::new())),
        &conf(),
    ));
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
}
