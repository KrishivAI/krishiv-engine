// SPDX-License-Identifier: Apache-2.0
//! Builders for the tests of the translation.

use krishiv_common::error::KrishivError;
use krishiv_plan::Plan;

use super::{Result, TranslateConf, translate_relation};
use crate::proto::v4_2::{
    DataType, Expression, Relation, RelationCommon,
    data_type::Kind,
    expression::{self, ExprType, Literal, literal::LiteralType},
    read::{DataSource, NamedTable, ReadType},
    relation::RelType,
    {self as pb},
};

pub(super) fn conf() -> TranslateConf {
    TranslateConf::default()
}

pub(super) fn relation(kind: RelType) -> Relation {
    Relation {
        common: None,
        rel_type: Some(kind),
    }
}

/// `r` with `plan_id` in its common message.
pub(super) fn with_id(mut r: Relation, id: i64) -> Relation {
    r.common = Some(RelationCommon {
        plan_id: Some(id),
        ..RelationCommon::default()
    });
    r
}

pub(super) fn read_table(name: &str) -> Relation {
    relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(ReadType::NamedTable(NamedTable {
            unparsed_identifier: name.to_owned(),
            options: Default::default(),
        })),
    }))
}

pub(super) fn read_files(format: Option<&str>, paths: &[&str]) -> Relation {
    relation(RelType::Read(pb::Read {
        is_streaming: false,
        read_type: Some(ReadType::DataSource(DataSource {
            format: format.map(str::to_owned),
            paths: paths.iter().map(|p| (*p).to_owned()).collect(),
            ..DataSource::default()
        })),
    }))
}

pub(super) fn expr(kind: ExprType) -> Expression {
    Expression {
        common: None,
        expr_type: Some(kind),
    }
}

pub(super) fn attr(name: &str) -> Expression {
    attr_with(name, None)
}

pub(super) fn attr_with(name: &str, plan_id: Option<i64>) -> Expression {
    expr(ExprType::UnresolvedAttribute(
        expression::UnresolvedAttribute {
            unparsed_identifier: name.to_owned(),
            plan_id,
            is_metadata_column: None,
        },
    ))
}

pub(super) fn lit(value: LiteralType) -> Expression {
    expr(ExprType::Literal(Literal {
        literal_type: Some(value),
        data_type: None,
    }))
}

pub(super) fn func(name: &str, args: Vec<Expression>) -> Expression {
    expr(ExprType::UnresolvedFunction(
        expression::UnresolvedFunction {
            function_name: name.to_owned(),
            arguments: args,
            is_distinct: false,
            is_user_defined_function: false,
            is_internal: None,
        },
    ))
}

pub(super) fn proto_type(kind: Kind) -> DataType {
    DataType { kind: Some(kind) }
}

pub(super) fn boxed<T>(value: T) -> Option<Box<T>> {
    Some(Box::new(value))
}

/// Translates `r` and prints the plan.
pub(super) fn show(r: &Relation) -> String {
    translate(r).expect("translates").to_string()
}

pub(super) fn translate(r: &Relation) -> Result<Plan> {
    translate_relation(r, &conf())
}

/// The error of a translation that must fail.
pub(super) fn fails<T>(result: Result<T>) -> KrishivError {
    match result {
        Ok(_) => panic!("the translation should have failed"),
        Err(e) => e,
    }
}
