// SPDX-License-Identifier: Apache-2.0
//! Tests of the Python UDF translation.

use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::ExprNode;

use super::*;
use crate::proto::v4_3::data_type::{Kind, Long, String as StringKind};
use crate::proto::v4_3::expression::ExprType;
use crate::translate::Translator;
use crate::translate::testing::{attr, conf, expr, fails, proto_type};

/// A host that names every call `session.udf<n>` and keeps what it was sent.
#[derive(Default)]
struct Recording(Vec<PythonUdf>);

impl UdfHost for Recording {
    fn inline_python_udf(&mut self, udf: &PythonUdf) -> Result<UdfCallName> {
        self.0.push(udf.clone());
        Ok(UdfCallName {
            qualifier: vec!["session".to_owned()],
            name: format!("udf{}", self.0.len()),
        })
    }

    fn relation_python_udf(&mut self, udf: &PythonUdf) -> Result<krishiv_plan::PythonUdfRef> {
        self.0.push(udf.clone());
        Err(unsupported("relations in this test host"))
    }
}

fn python(eval_type: i32) -> pb::PythonUdf {
    pb::PythonUdf {
        output_type: Some(proto_type(Kind::Long(Long::default()))),
        eval_type,
        command: vec![1, 2, 3],
        python_ver: "3.12".to_owned(),
        additional_includes: Vec::new(),
    }
}

fn inline(
    function: Option<Function>,
    args: Vec<pb::Expression>,
) -> pb::CommonInlineUserDefinedFunction {
    pb::CommonInlineUserDefinedFunction {
        function_name: "plus_one".to_owned(),
        deterministic: true,
        arguments: args,
        function,
        is_distinct: false,
    }
}

fn call(f: pb::CommonInlineUserDefinedFunction) -> pb::Expression {
    expr(ExprType::CommonInlineUserDefinedFunction(f))
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

#[test]
fn an_inline_call_is_the_function_the_host_names_over_its_arguments() {
    let c = conf();
    let mut host = Recording::default();
    let e = Translator::with_udfs(&c, &mut host)
        .expression(&call(inline(
            Some(Function::PythonUdf(python(101))),
            vec![attr("id")],
        )))
        .unwrap_or_else(|e| panic!("{e}"));
    let ExprNode::Function(f) = &e.node else {
        panic!("a call: {e:?}");
    };
    assert_eq!(f.qualifier, ["session"]);
    assert_eq!(f.name, "udf1");
    assert_eq!(f.args.len(), 1);
    assert_eq!(f.args[0].to_string(), "id");
    let [udf] = host.0.as_slice() else {
        panic!("one UDF: {:?}", host.0);
    };
    assert_eq!(udf.name, "plus_one");
    assert!(udf.deterministic);
    assert_eq!(udf.eval_type, 101);
    assert_eq!(udf.plan_eval_type, PythonEvalType::ArrowBatched);
    assert_eq!(udf.return_type, DataType::Long);
    assert_eq!(udf.command, [1, 2, 3]);
    assert_eq!(udf.python_ver, "3.12");
    // The payload never reaches a log.
    assert!(!format!("{udf:?}").contains("[1, 2, 3]"));
}

#[test]
fn without_a_host_an_inline_udf_is_refused() {
    let e = fails(crate::translate::translate_expression(
        &call(inline(Some(Function::PythonUdf(python(101))), Vec::new())),
        &conf(),
    ));
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
}

#[test]
fn the_scalar_eval_types_are_supported_and_100_runs_arrow_batched() {
    for (code, plan) in [
        (100, PythonEvalType::ArrowBatched),
        (101, PythonEvalType::ArrowBatched),
        (200, PythonEvalType::ScalarPandas),
        (250, PythonEvalType::ScalarArrow),
    ] {
        let udf = python_udf(&inline(Some(Function::PythonUdf(python(code))), Vec::new()))
            .unwrap_or_else(|e| panic!("{code}: {e}"));
        assert_eq!((udf.eval_type, udf.plan_eval_type), (code, plan));
    }
    for code in [0, 99, 202, 204, 205, 251, 300, 1000] {
        let e = fails(python_udf(&inline(
            Some(Function::PythonUdf(python(code))),
            Vec::new(),
        )));
        assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature, "{code}");
        assert!(e.to_string().contains(&format!("eval type {code}")), "{e}");
    }
}

#[test]
fn jvm_functions_unset_functions_and_includes_are_refused() {
    let e = fails(python_udf(&inline(
        Some(Function::JavaUdf(pb::JavaUdf::default())),
        Vec::new(),
    )));
    assert_eq!(
        class(&e),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported
    );
    assert!(e.to_string().contains("JAVA_UDF(6)"), "{e}");
    let e = fails(python_udf(&inline(
        Some(Function::ScalarScalaUdf(pb::ScalarScalaUdf::default())),
        Vec::new(),
    )));
    assert!(e.to_string().contains("SCALAR_SCALA_UDF(5)"), "{e}");
    let e = fails(python_udf(&inline(None, Vec::new())));
    assert_eq!(
        class(&e),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
    let mut with_includes = python(101);
    with_includes.additional_includes = vec!["x.zip".to_owned()];
    let e = fails(python_udf(&inline(
        Some(Function::PythonUdf(with_includes)),
        Vec::new(),
    )));
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("additional_includes"), "{e}");
    // No output type is an unset kind, as Spark's `transformDataType` of the default message.
    let mut untyped = python(101);
    untyped.output_type = None;
    let e = fails(python_udf(&inline(
        Some(Function::PythonUdf(untyped)),
        Vec::new(),
    )));
    assert_eq!(
        class(&e),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSet
    );
}

#[test]
fn register_function_is_a_command_the_server_runs() {
    let mut f = inline(Some(Function::PythonUdf(python(100))), Vec::new());
    f.function_name = "Shout".to_owned();
    if let Some(Function::PythonUdf(p)) = &mut f.function {
        p.output_type = Some(proto_type(Kind::String(StringKind::default())));
    }
    let command = pb::Command {
        command_type: Some(CommandType::RegisterFunction(Box::new(f))),
    };
    let Ok(Command::RegisterPythonUdf(udf)) = translate_command(&command) else {
        panic!("a registration");
    };
    assert_eq!(udf.name, "Shout");
    assert_eq!(udf.eval_type, 100);
    assert_eq!(udf.return_type, DataType::String);
    // Every other command is still refused.
    let other = pb::Command {
        command_type: Some(CommandType::SqlCommand(Box::default())),
    };
    let e = fails(translate_command(&other));
    assert_eq!(
        class(&e),
        ErrorClass::ConnectInvalidPlanInvalidOneOfFieldNotSupported
    );
}

#[test]
fn a_persist_extension_translates_to_its_command_and_other_extensions_do_not() {
    use prost::Message as _;
    let command = |type_url: &str, value: Vec<u8>| pb::Command {
        command_type: Some(CommandType::Extension(Box::new(prost_types::Any {
            type_url: type_url.to_owned(),
            value,
        }))),
    };
    let ok = command(
        "type.googleapis.com/krishiv.v1.PersistPythonFunction",
        crate::extension::PersistPythonFunction {
            name: "f".to_owned(),
            drop: true,
        }
        .encode_to_vec(),
    );
    assert_eq!(
        translate_command(&ok).unwrap(),
        Command::PersistPythonFunction {
            name: "f".to_owned(),
            drop: true
        }
    );
    let echo = command("type.googleapis.com/krishiv.v1.Echo", Vec::new());
    assert!(translate_command(&echo).is_err());
}

#[test]
fn from_stored_builds_what_register_function_would_and_refuses_a_bad_row() {
    let udf =
        PythonUdf::from_stored("f".into(), true, 101, "bigint", vec![1], "3.12".into()).unwrap();
    assert_eq!(udf.plan_eval_type, PythonEvalType::ArrowBatched);
    assert_eq!(udf.return_type, krishiv_types::DataType::Long);
    let eval =
        PythonUdf::from_stored("f".into(), true, 999, "bigint", vec![], String::new()).unwrap_err();
    assert_eq!(
        eval.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
    // A return type that does not parse is the DDL parser's error, not the eval type's.
    let ty = PythonUdf::from_stored("f".into(), true, 101, "not a type (", vec![], String::new())
        .unwrap_err();
    assert_ne!(
        ty.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
}
