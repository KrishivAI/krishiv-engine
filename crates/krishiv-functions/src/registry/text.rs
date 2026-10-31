// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of the remaining scalar string functions:
//! character functions (`ascii`, `chr`, `left`, `right`, `repeat`, `space`,
//! `reverse`, `initcap`, `translate`, `locate` and its aliases,
//! `substring_index`, `split_part`, `find_in_set`, `overlay`, `levenshtein`,
//! `soundex`, `luhn_check`, `quote`, `elt`, `concat_ws`, the
//! lengths) and the encodings (`base64`, `encode`, `decode`, `to_binary`,
//! the UTF-8 validators, the URL codecs).
//!
//! Ground truth: Spark 4.2's `stringExpressions.scala`, `maskExpressions.scala`,
//! `urlExpressions.scala`, the `string-functions.sql`, `try-string-functions.sql`
//! and `mask-functions.sql` goldens and the `string`/`url` matrix families.
//! A function that takes an array in Spark (`reverse`, `concat_ws`) takes
//! strings only here; the analyzer has no array type yet.

use krishiv_common::error::{
    DatatypeMismatchInvalidArgValue, DatatypeMismatchNonFoldableInput, KrishivError,
    KrishivUnsupportedFeature,
};
use krishiv_types::DataType;

use super::defs::{Spec, exact, first_type, int, is_binary, scalar, string, typed};
use crate::signature::{ArgType, CallTypes, Constant, InputBuilder, InputType, Inputs};

fn nullable_string(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, true))
}

fn nullable_of(c: &CallTypes<'_>, i: usize) -> bool {
    c.args().get(i).is_some_and(|a| a.nullable)
}

/// `locate`/`position`: NULL when the substring or the string is, never for
/// the start (a NULL start gives 0).
fn locate_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(
        DataType::Integer,
        nullable_of(c, 0) || nullable_of(c, 1),
    ))
}

/// `concat_ws`: NULL only when the separator is.
fn concat_ws_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, nullable_of(c, 0)))
}

fn binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    typed(DataType::Binary, c)
}

fn nullable_binary(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Binary, true))
}

fn nullable_boolean(_: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::Boolean, true))
}

/// `overlay`: both texts are binary or both are strings.
fn overlay_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let t = if is_binary(c, 0) && is_binary(c, 1) {
        DataType::Binary
    } else {
        DataType::String
    };
    vec![
        exact(t.clone()),
        exact(t),
        exact(DataType::Integer),
        exact(DataType::Integer),
    ]
}

/// `elt(n, str...)`: Spark reads the arguments after the index as strings
/// when any is not a string or binary.
fn elt_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    let all_binary = c.args().len() > 1 && (1..c.args().len()).all(|i| is_binary(c, i));
    let t = if all_binary {
        DataType::Binary
    } else {
        DataType::String
    };
    vec![exact(DataType::Integer), exact(t)]
}

/// `elt` is NULL for an index out of range, whatever the arguments.
fn elt_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let t = c.data_type(1).cloned().unwrap_or(DataType::String);
    Ok(ArgType::new(t, true))
}

/// `decode(bin, charset)`; the `decode(expr, search, result, ...)` form is
/// Spark's `CASE WHEN`, which the analyzer builds from other nodes.
fn decode_inputs(c: &CallTypes<'_>) -> Vec<InputType> {
    if c.args().len() == 2 {
        vec![exact(DataType::Binary), exact(DataType::String)]
    } else {
        vec![InputType::Any]
    }
}

fn decode_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    if c.args().len() != 2 {
        // The analyzer rewrites the CASE form before typing the call.
        return Err(KrishivUnsupportedFeature {
            feature: "decode with search and result arguments".to_owned(),
        }
        .into());
    }
    string(c)
}

/// `format_string(format, args...)`: the arguments are not coerced.
fn format_inputs(_: &CallTypes<'_>) -> Vec<InputType> {
    vec![exact(DataType::String), InputType::Any]
}

/// NULL only when the format is.
fn format_result(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    Ok(ArgType::new(DataType::String, nullable_of(c, 0)))
}

/// A function call as Spark's errors quote it: `name(arg, ...)`.
fn call_sql(c: &CallTypes<'_>, name: &str) -> String {
    let parts: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    format!("\"{name}({})\"", parts.join(", "))
}

/// The formats `to_binary` knows, as Spark lists them.
const BINARY_FORMATS: &str = "'hex', 'utf-8', 'utf8', or 'base64'";

/// `to_binary(str[, fmt])`: a format that is not foldable is refused, and
/// so is a foldable one that names no format (`try_to_binary` gives NULL
/// there instead).
fn to_binary_check(c: &CallTypes<'_>, try_: bool) -> Result<ArgType, KrishivError> {
    if c.args().len() == 2 {
        let sql_expr = || call_sql(c, "to_binary");
        let text = match c.constant(1) {
            Some(Constant::Null) => None,
            Some(Constant::String(s)) => Some(s.clone()),
            Some(Constant::Integral(n)) => Some(n.to_string()),
            _ => {
                return Err(DatatypeMismatchNonFoldableInput {
                    sql_expr: sql_expr(),
                    input_name: "`fmt`".to_owned(),
                    input_type: "\"STRING\"".to_owned(),
                    input_expr: format!("\"{}\"", c.sql(1).unwrap_or_default()),
                }
                .into());
            }
        };
        if let Some(fmt) = text {
            let lower = fmt.to_lowercase();
            if !try_ && !matches!(lower.as_str(), "hex" | "utf-8" | "utf8" | "base64") {
                return Err(DatatypeMismatchInvalidArgValue {
                    sql_expr: sql_expr(),
                    input_name: "fmt".to_owned(),
                    require_type: "case-insensitive \"STRING\"".to_owned(),
                    valid_values: BINARY_FORMATS.to_owned(),
                    input_value: format!("'{lower}'"),
                }
                .into());
            }
        }
    }
    nullable_binary(c)
}

fn to_binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    to_binary_check(c, false)
}

fn try_to_binary(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    to_binary_check(c, true)
}

pub(super) fn specs() -> Vec<Spec> {
    let text = exact(DataType::String);
    let text_or_bytes = InputType::OneOf(vec![DataType::String, DataType::Binary]);
    let one_text = Inputs::Expects(vec![text.clone()]);
    let int_arg = exact(DataType::Integer);
    let mut out = scalar(&["ascii"], &one_text, int);
    out.extend(scalar(
        &["bit_length", "octet_length"],
        &Inputs::Expects(vec![text_or_bytes.clone()]),
        int,
    ));
    out.extend(scalar(
        &["chr", "char"],
        &Inputs::Expects(vec![exact(DataType::Long)]),
        string,
    ));
    out.extend(scalar(
        &["left"],
        &Inputs::Expects(vec![text_or_bytes.clone(), int_arg.clone()]),
        first_type,
    ));
    out.extend(scalar(
        &["right"],
        &Inputs::Expects(vec![text.clone(), int_arg.clone()]),
        nullable_string,
    ));
    out.extend(scalar(
        &["repeat"],
        &Inputs::Expects(vec![text.clone(), int_arg.clone()]),
        string,
    ));
    out.extend(scalar(
        &["space"],
        &Inputs::Expects(vec![int_arg.clone()]),
        string,
    ));
    out.extend(scalar(&["initcap"], &one_text, string));
    out.extend(scalar(
        &["soundex"],
        &Inputs::Expects(vec![InputType::Strict(DataType::String)]),
        string,
    ));
    out.extend(scalar(
        &["reverse"],
        &Inputs::Expects(vec![text_or_bytes.clone()]),
        first_type,
    ));
    out.extend(scalar(&["quote"], &one_text, nullable_string));
    out.extend(scalar(
        &["translate"],
        &Inputs::Expects(vec![text.clone(), text.clone(), text.clone()]),
        string,
    ));
    out.extend(scalar(
        &["instr", "find_in_set"],
        &Inputs::Expects(vec![text.clone(), text.clone()]),
        int,
    ));
    out.extend(scalar(
        &["locate", "position"],
        &Inputs::Expects(vec![text.clone(), text.clone(), int_arg.clone()]),
        locate_result,
    ));
    out.extend(scalar(
        &["substring_index"],
        &Inputs::Expects(vec![text.clone(), text.clone(), int_arg.clone()]),
        string,
    ));
    out.extend(scalar(
        &["split_part"],
        &Inputs::Expects(vec![text.clone(), text.clone(), int_arg.clone()]),
        nullable_string,
    ));
    out.extend(scalar(
        &["overlay"],
        &Inputs::Built(InputBuilder(overlay_inputs)),
        first_type,
    ));
    out.extend(scalar(
        &["levenshtein"],
        &Inputs::Expects(vec![text.clone(), text.clone(), int_arg]),
        int,
    ));
    out.extend(scalar(&["luhn_check"], &one_text, nullable_boolean));
    out.extend(scalar(
        &["elt"],
        &Inputs::Built(InputBuilder(elt_inputs)),
        elt_result,
    ));
    out.extend(scalar(&["concat_ws"], &one_text, concat_ws_result));
    out.extend(scalar(
        &["base64"],
        &Inputs::Expects(vec![exact(DataType::Binary)]),
        string,
    ));
    out.extend(scalar(&["unbase64"], &one_text, binary));
    out.extend(scalar(
        &["encode"],
        &Inputs::Expects(vec![text.clone(), text.clone()]),
        nullable_binary,
    ));
    out.extend(scalar(
        &["decode"],
        &Inputs::Built(InputBuilder(decode_inputs)),
        decode_result,
    ));
    out.extend(scalar(
        &["format_string", "printf"],
        &Inputs::Built(InputBuilder(format_inputs)),
        format_result,
    ));
    out.extend(scalar(
        &["to_binary"],
        &Inputs::Expects(vec![text.clone(), text.clone()]),
        to_binary,
    ));
    out.extend(scalar(
        &["try_to_binary"],
        &Inputs::Expects(vec![text.clone(), text]),
        try_to_binary,
    ));
    let validated = Inputs::Expects(vec![text_or_bytes]);
    out.extend(scalar(&["is_valid_utf8"], &validated, nullable_boolean));
    out.extend(scalar(
        &["make_valid_utf8", "validate_utf8"],
        &validated,
        nullable_string,
    ));
    out.extend(scalar(&["try_validate_utf8"], &validated, nullable_string));
    out.extend(scalar(
        &["url_encode", "url_decode", "try_url_decode"],
        &one_text,
        nullable_string,
    ));
    out
}
