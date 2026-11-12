// SPDX-License-Identifier: Apache-2.0
//! Tests for the function registry: the signatures, arities, result types and Spark catalog agreement of the definitions. The helpers are here; the tests are in `tests/`.

use std::sync::Arc;

use krishiv_common::error::{ErrorClass, InvalidOptionsNonMapFunction, KrishivError};
use krishiv_types::{ArrayType, DataType, DecimalType};

use super::{Aggregate, Execution, FunctionDef, Lookup, builtins, is_grammar_only};
use crate::scalar::scalar_names;
use crate::signature::{
    ArgType, CallTypes, Constant, FunctionKind, InputType, Inputs, Operator, SparkProfile,
};

fn def(name: &str) -> &'static FunctionDef {
    match builtins().lookup(name, SparkProfile::default()) {
        Lookup::Found(d) => d,
        other => panic!("{name}: {other:?}"),
    }
}

fn result(
    name: &str,
    args: &[ArgType],
    consts: &[Option<Constant>],
) -> Result<ArgType, KrishivError> {
    (def(name).result)(&CallTypes::new(args, consts))
}

fn ty(t: DataType) -> ArgType {
    ArgType::new(t, false)
}

fn null_ty() -> ArgType {
    ArgType {
        data_type: None,
        nullable: true,
    }
}

fn dec(p: u32, s: u32) -> DataType {
    DataType::Decimal(DecimalType::try_new(p, s).unwrap())
}

/// The input of the time fields: either timestamp, or a time of any
/// precision.
fn timestamp_or_time() -> InputType {
    InputType::OneOf(
        [DataType::Timestamp, DataType::TimestampNtz]
            .into_iter()
            .chain((0..=6).map(|precision| DataType::Time { precision }))
            .collect(),
    )
}

fn nullable(t: DataType) -> ArgType {
    ArgType::new(t, true)
}

/// A nullable struct of one `INT` field, for `min`/`max` result checks.
fn nullable_struct() -> ArgType {
    use krishiv_types::{StructField, StructType};

    ArgType::new(
        DataType::Struct(StructType::new(vec![StructField::new(
            "f",
            DataType::Integer,
            true,
        )])),
        true,
    )
}

/// The analyzer's `uniform` result type and the executor's bind type agree
/// on every pair of bound kinds: both layers map their own type to the
/// shared `uniform_result` rule, and the test pins the two mappers against
/// each other so one side cannot drift without the other.
/// `t` as the executor hands it to a function, and as functions return it.
fn wide(t: &DataType) -> arrow_schema::DataType {
    krishiv_types::arrow::client_arrow_type(&t.to_arrow())
}

mod text;

mod signatures1;
mod signatures2;
mod signatures3;
