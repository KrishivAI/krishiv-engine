// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.v1` extension slots through the translator (P3-T015).

use krishiv_common::error::ErrorClass;
use krishiv_connect::extension::{Echo, TYPE_URL_PREFIX};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::translate::{TranslateConf, translate_command, translate_relation};
use krishiv_plan::{ExprNode, Literal, PlanNode};
use prost::Message;

fn echo_any() -> prost_types::Any {
    prost_types::Any {
        type_url: format!("{TYPE_URL_PREFIX}Echo"),
        value: Echo {
            message: "hi".to_owned(),
        }
        .encode_to_vec(),
    }
}

fn relation(any: prost_types::Any) -> pb::Relation {
    pb::Relation {
        common: None,
        rel_type: Some(pb::relation::RelType::Extension(any)),
    }
}

#[test]
fn an_echo_relation_is_a_one_row_project() {
    let plan = translate_relation(&relation(echo_any()), &TranslateConf::default()).expect("plan");
    let PlanNode::Project { input, exprs } = &plan.root().node else {
        panic!("a project: {plan:?}");
    };
    assert!(matches!(input.node, PlanNode::OneRowRelation));
    assert_eq!(exprs.len(), 1);
    let ExprNode::Alias { expr, name } = &exprs[0].node else {
        panic!("an alias: {:?}", exprs[0]);
    };
    assert_eq!(name, "message");
    assert!(matches!(&expr.node, ExprNode::Literal(Literal::String(s)) if s == "hi"));
}

#[test]
fn an_unknown_extension_is_unsupported_not_a_decode_failure() {
    let any = prost_types::Any {
        type_url: format!("{TYPE_URL_PREFIX}Teleport"),
        value: vec![0xFF; 8],
    };
    let err = translate_relation(&relation(any), &TranslateConf::default()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("capability `teleport`"), "{err}");
}

#[test]
fn a_command_extension_names_why() {
    let command = pb::Command {
        command_type: Some(pb::command::CommandType::Extension(Box::new(echo_any()))),
    };
    let err = translate_command(&command).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("in a command"), "{err}");
}
