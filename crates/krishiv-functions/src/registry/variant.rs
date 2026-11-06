// SPDX-License-Identifier: Apache-2.0
//! The variant function signatures: `parse_json`, `try_parse_json`,
//! `is_variant_null`, `variant_get` and `try_variant_get` (Spark 4.2's
//! `ParseJson`, `IsVariantNull` and `VariantGet`, `variantExpressions.scala`
//! at v4.2.0).
//!
//! `parse_json` and the path take a string, the variant arguments a
//! `VARIANT`, with nothing cast implicitly. The type of `variant_get(v, path,
//! type)` is the DDL type in the third argument, which must be a non-null
//! string literal (`INVALID_SCHEMA.NON_STRING_LITERAL`, as `from_json`'s
//! schema) naming a type a variant casts to (`VariantGet.checkDataType`:
//! numbers, booleans, strings, binaries, datetimes (`TIME` too) and variants, arrays of
//! them, maps with string keys and structs); without it the result is a
//! `VARIANT`. The result is always nullable.

use krishiv_common::error::{
    DatatypeMismatchCastWithoutSuggestion, InvalidSchemaNonStringLiteral, KrishivError,
};
use krishiv_types::{DataType, parse_data_type};

use super::defs::{Spec, scalar};
use super::hof::call_sql;
use crate::signature::{ArgType, CallTypes, Constant, InputType, Inputs};

/// Spark's `VariantGet.checkDataType`: whether a variant casts to `t`.
#[must_use]
pub fn variant_castable(t: &DataType, allow_structs_and_maps: bool) -> bool {
    match t {
        DataType::Boolean
        | DataType::Byte
        | DataType::Short
        | DataType::Integer
        | DataType::Long
        | DataType::Float
        | DataType::Double
        | DataType::Decimal(_)
        | DataType::String
        | DataType::CollatedString(_)
        | DataType::Binary
        | DataType::Date
        | DataType::Timestamp
        | DataType::TimestampNtz
        | DataType::Time { .. }
        | DataType::Variant => true,
        DataType::Array(a) => variant_castable(a.element_type(), allow_structs_and_maps),
        DataType::Map(m) if allow_structs_and_maps => {
            matches!(m.key_type(), DataType::String | DataType::CollatedString(_))
                && variant_castable(m.value_type(), allow_structs_and_maps)
        }
        DataType::Struct(s) if allow_structs_and_maps => s
            .fields()
            .iter()
            .all(|f| variant_castable(f.data_type(), allow_structs_and_maps)),
        _ => false,
    }
}

fn parse_json(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let nullable = c.args().first().is_none_or(|a| a.nullable);
    Ok(ArgType::new(DataType::Variant, nullable))
}

fn try_parse_json(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Variant, true))
}

fn is_variant_null(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Boolean, false))
}

/// The target type of a `variant_get` call: its third argument as a DDL type.
///
/// # Errors
/// `INVALID_SCHEMA.NON_STRING_LITERAL` for a third argument that is not a
/// non-null string literal, the DDL parser's errors, and
/// `DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION` for a type a variant does not
/// cast to.
pub(super) fn target_type(name: &str, c: &CallTypes<'_>) -> Result<DataType, KrishivError> {
    if c.args().len() < 3 {
        return Ok(DataType::Variant);
    }
    let t = match c.constant(2) {
        Some(Constant::String(s)) => parse_data_type(s)?,
        Some(Constant::Error(e)) => return Err(e.clone()),
        _ => {
            return Err(InvalidSchemaNonStringLiteral {
                input_schema: format!("\"{}\"", c.sql(2).unwrap_or_default()),
            }
            .into());
        }
    };
    if !variant_castable(&t, true) {
        return Err(DatatypeMismatchCastWithoutSuggestion {
            sql_expr: call_sql(name, c),
            src_type: "\"VARIANT\"".to_owned(),
            target_type: format!("\"{}\"", t.sql()),
        }
        .into());
    }
    Ok(t)
}

/// `VariantGet.dataType`: the target `asNullable`.
fn variant_get(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        target_type("variant_get", c)?.as_nullable(),
        true,
    ))
}

fn try_variant_get(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        target_type("try_variant_get", c)?.as_nullable(),
        true,
    ))
}

pub(super) fn specs() -> Vec<Spec> {
    // `ExpectsInputTypes` on `StringTypeWithCollation`: no implicit cast.
    let string = Inputs::Expects(vec![InputType::Strict(DataType::String)]);
    let variant = Inputs::Expects(vec![InputType::Strict(DataType::Variant)]);
    let get = Inputs::Expects(vec![
        InputType::Strict(DataType::Variant),
        InputType::Strict(DataType::String),
        InputType::Any,
    ]);
    let mut out = scalar(&["parse_json"], &string, parse_json);
    out.extend(scalar(&["try_parse_json"], &string, try_parse_json));
    out.extend(scalar(&["is_variant_null"], &variant, is_variant_null));
    out.extend(scalar(&["variant_get"], &get, variant_get));
    out.extend(scalar(&["try_variant_get"], &get, try_variant_get));
    out
}
