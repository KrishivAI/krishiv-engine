// SPDX-License-Identifier: Apache-2.0
//! The errors of the translation. Every one is a registry class: Spark's own
//! `CONNECT_INVALID_PLAN.*` for a plan that is invalid, and
//! `KRISHIV_UNSUPPORTED_FEATURE` for a valid plan Krishiv cannot run yet. They
//! name proto messages and fields, never values: a decoded plan can hold
//! credentials (read options, literals, SQL text).

use krishiv_common::error::{
    ConnectInvalidPlanAssertionFailure, ConnectInvalidPlanFieldCannotBeEmpty,
    ConnectInvalidPlanIncompatibleLiteralDataType, ConnectInvalidPlanInvalidEnum,
    ConnectInvalidPlanInvalidOneOfFieldNotSet, ConnectInvalidPlanUnsupportedLiteralType,
    KrishivError, KrishivUnsupportedFeature,
};

use crate::proto::Variant;

/// A valid plan that uses something Krishiv does not support yet.
pub(crate) fn unsupported(feature: impl Into<String>) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.into(),
    }
    .into()
}

/// A `oneof` variant (`kind` is `relation` or `expression`) that is not
/// translated: `parent` is the message that holds it.
pub(crate) fn unsupported_variant(kind: &str, parent: &str, variant: Variant) -> KrishivError {
    unsupported(format!(
        "Spark Connect {kind} {} ({parent}.{})",
        variant.short_type(),
        variant.field
    ))
}

/// A field of a supported message that is set to something not supported:
/// `what` is a plural noun phrase ("streaming reads").
pub(crate) fn unsupported_field(message: &str, field: &str, what: &str) -> KrishivError {
    unsupported(format!("{what} (spark.connect.{message}.{field})"))
}

/// A `oneof` with no variant set: Spark's `INVALID_ONE_OF_FIELD_NOT_SET`.
/// `name` is the Java name of the unset case (`REL_TYPE_NOT_SET`).
pub(crate) fn not_set(full_name: &str, name: &str) -> KrishivError {
    ConnectInvalidPlanInvalidOneOfFieldNotSet {
        full_name: full_name.to_owned(),
        name: name.to_owned(),
    }
    .into()
}

pub(crate) fn relation_not_set() -> KrishivError {
    not_set("spark.connect.Relation", "REL_TYPE_NOT_SET")
}

pub(crate) fn expression_not_set() -> KrishivError {
    not_set("spark.connect.Expression", "EXPR_TYPE_NOT_SET")
}

/// An enum value that is unspecified or unknown: Spark's `INVALID_ENUM`.
pub(crate) fn invalid_enum(full_name: &str, name: &str, number: i32) -> KrishivError {
    ConnectInvalidPlanInvalidEnum {
        full_name: full_name.to_owned(),
        name: name.to_owned(),
        number: number.to_string(),
    }
    .into()
}

/// Spark's `assertPlan`: a plan that breaks a rule of the protocol.
pub(crate) fn assertion(message: impl Into<String>) -> KrishivError {
    ConnectInvalidPlanAssertionFailure {
        message: message.into(),
    }
    .into()
}

/// A literal whose declared data type does not fit its value.
pub(crate) fn incompatible_literal(data_type: &str, literal: &str) -> KrishivError {
    ConnectInvalidPlanIncompatibleLiteralDataType {
        data_type_kind_case: data_type.to_uppercase(),
        literal_type_case: literal.to_uppercase(),
    }
    .into()
}

pub(crate) fn unsupported_literal_type(type_info: &str) -> KrishivError {
    ConnectInvalidPlanUnsupportedLiteralType {
        type_info: type_info.to_owned(),
    }
    .into()
}

/// A required repeated or string field that is empty: Spark's
/// `FIELD_CANNOT_BE_EMPTY`.
pub(crate) fn field_empty(field: &str, full_name: &str) -> KrishivError {
    ConnectInvalidPlanFieldCannotBeEmpty {
        field_name: field.to_owned(),
        full_name: full_name.to_owned(),
    }
    .into()
}
