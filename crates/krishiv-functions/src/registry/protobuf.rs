// SPDX-License-Identifier: Apache-2.0
//! The analysis signatures of `from_protobuf` and `to_protobuf` (Spark 4.0+, `toFromProtobufSqlFunctions.scala`).
//!
//! `(data, messageName[, descFilePathOrOptions][, options])`: with three arguments a literal string or binary third argument is the
//! descriptor (a path or the bytes of a `FileDescriptorSet`) and a map is the options; anything else is no descriptor, which makes
//! `messageName` a Java class name (`CANNOT_LOAD_PROTOBUF_CLASS`). The message name must be a foldable string, the descriptor a
//! foldable string, binary or NULL, the options a foldable `MAP<STRING, STRING>` (`DATATYPE_MISMATCH.TYPE_CHECK_FAILURE_WITH_HINT`
//! with Spark's text). `from_protobuf`'s type is the Spark type of the message; `to_protobuf`'s is `BINARY` and fails here when the
//! struct does not fit the message. Spark raises `INTERNAL_ERROR` for a first argument of `from_protobuf` that is not binary
//! (goldens `errors_args`); here it is the type mismatch.

use krishiv_common::error::{
    DatatypeMismatchNonStructType, DatatypeMismatchTypeCheckFailureWithHint,
    DatatypeMismatchUnexpectedInputType, KrishivError,
};
use krishiv_io::protobuf::{Call, DescriptorArg};
use krishiv_types::DataType;

use super::defs::{Spec, scalar};
use crate::signature::{ArgType, CallTypes, Constant, Inputs};

/// `from_protobuf` and `to_protobuf`.
pub(super) fn specs() -> Vec<Spec> {
    let mut out = scalar(&["from_protobuf"], &Inputs::Any, from_protobuf);
    out.extend(scalar(&["to_protobuf"], &Inputs::Any, to_protobuf));
    out
}

fn call_sql(c: &CallTypes<'_>, name: &str) -> String {
    // Spark quotes the four-argument form, NULLs filling what was not given.
    let mut args: Vec<String> = (0..c.args().len())
        .map(|i| c.sql(i).unwrap_or_default())
        .collect();
    while args.len() < 4 {
        args.push("NULL".to_owned());
    }
    format!("\"{name}({})\"", args.join(", "))
}

fn mismatch(c: &CallTypes<'_>, name: &str, msg: String) -> KrishivError {
    DatatypeMismatchTypeCheckFailureWithHint {
        sql_expr: call_sql(c, name),
        msg,
        hint: String::new(),
    }
    .into()
}

/// The three constant arguments as Spark's constructors normalise them.
struct Parts {
    message: String,
    /// The descriptor, or the error reading its file: raised after every argument check, where Spark reads the file (its
    /// `replacement`, after `checkInputDataTypes`).
    descriptor: Result<DescriptorArg, KrishivError>,
    options: Vec<(String, String)>,
}

fn is_string_or_binary(t: Option<&DataType>) -> bool {
    matches!(t, Some(DataType::String | DataType::Binary))
}

fn map_of_strings(t: Option<&DataType>) -> bool {
    match t {
        Some(DataType::Map(m)) => {
            matches!(m.key_type(), DataType::String | DataType::Null)
                && matches!(m.value_type(), DataType::String | DataType::Null)
        }
        _ => false,
    }
}

/// Spark's argument checks, in its order, and the values of the constants.
fn parts(
    c: &CallTypes<'_>,
    name: &str,
    upper: &str,
    direction: &str,
) -> Result<Parts, KrishivError> {
    let n = c.args().len();
    // The three-argument form: a literal string or binary is the descriptor, a map the options, anything else neither.
    let (desc_index, options_index) = match n {
        2 => (None, None),
        3 => {
            if map_of_strings(c.data_type(2)) || matches!(c.data_type(2), Some(DataType::Map(_))) {
                (None, Some(2))
            } else if is_string_or_binary(c.data_type(2))
                && (c.is_literal(2) || c.is_plain_literal(2))
            {
                (Some(2), None)
            } else {
                (None, None)
            }
        }
        _ => (Some(2), Some(3)),
    };
    let message = match (c.data_type(1), c.constant(1)) {
        (Some(DataType::String), Some(Constant::String(s))) => Ok(s.clone()),
        (Some(DataType::String), Some(Constant::Error(e))) => return Err(e.clone()),
        _ => Err(mismatch(
            c,
            name,
            format!(
                "The second argument of the {upper} SQL function must be a constant string representing the Protobuf message name"
            ),
        )),
    };
    let descriptor = match desc_index {
        None => Ok(Ok(DescriptorArg::None)),
        Some(i) => match (c.data_type(i), c.constant(i)) {
            (Some(DataType::String), Some(Constant::String(s))) => {
                Ok(Ok(DescriptorArg::Path(s.clone())))
            }
            // A literal path the analyzer read under the access policy: the file's bytes, or why it could not be read.
            (Some(DataType::String), Some(Constant::Binary(b))) => {
                Ok(Ok(DescriptorArg::File(b.clone())))
            }
            (Some(DataType::String), Some(Constant::Error(e))) => Ok(Err(e.clone())),
            (Some(DataType::Binary), Some(Constant::Binary(b))) => {
                Ok(Ok(DescriptorArg::Bytes(b.clone())))
            }
            (Some(DataType::String | DataType::Binary) | None, Some(Constant::Null))
            | (None, None)
                if c.foldable(i) =>
            {
                Ok(Ok(DescriptorArg::None))
            }
            _ => Err(mismatch(
                c,
                name,
                format!(
                    "The third argument of the {upper} SQL function must be a constant string or binary data representing the Protobuf descriptor file path"
                ),
            )),
        },
    };
    let options = match options_index {
        None => Ok(Vec::new()),
        Some(i) => {
            let ok_type = map_of_strings(c.data_type(i))
                || matches!(c.data_type(i), None | Some(DataType::Null));
            match c.constant(i) {
                Some(Constant::StringMap(entries)) if ok_type => Ok(entries
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone().unwrap_or_default()))
                    .collect()),
                Some(Constant::Null) if ok_type => Ok(Vec::new()),
                _ => Err(mismatch(
                    c,
                    name,
                    format!(
                        "The {} argument of the {upper} SQL function must be a constant map of strings to strings containing the options to use for converting the value {direction} Protobuf format",
                        if n == 3 { "third" } else { "fourth" }
                    ),
                )),
            }
        }
    };
    Ok(Parts {
        message: message?,
        descriptor: descriptor?,
        options: options?,
    })
}

/// `from_protobuf(binary, messageName[, descFilePath][, options])`.
fn from_protobuf(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let p = parts(c, "from_protobuf", "FROM_PROTOBUF", "from")?;
    let call = Call::new(&p.message, p.descriptor?, p.options, false)?;
    // Spark never types the first argument (an INTERNAL_ERROR there), after it has read the descriptor file.
    if !matches!(c.data_type(0), Some(DataType::Binary)) {
        let input = c.data_type(0).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchUnexpectedInputType {
            sql_expr: call_sql(c, "from_protobuf"),
            param_index: "first".to_owned(),
            required_type: "\"BINARY\"".to_owned(),
            input_sql: format!("\"{}\"", c.sql(0).unwrap_or_default()),
            input_type: format!("\"{}\"", input.sql()),
        }
        .into());
    }
    let decoder = call.decoder()?;
    Ok(ArgType::new(decoder.spark_type().clone(), true))
}

/// `to_protobuf(struct, messageName[, descFilePath][, options])`: `BINARY`.
fn to_protobuf(c: &CallTypes<'_>) -> Result<ArgType, KrishivError> {
    let Some(DataType::Struct(_)) = c.data_type(0) else {
        let input = c.data_type(0).cloned().unwrap_or(DataType::Null);
        return Err(DatatypeMismatchNonStructType {
            sql_expr: call_sql(c, "to_protobuf"),
            input_name: "data".to_owned(),
            input_type: format!("\"{}\"", input.sql()),
        }
        .into());
    };
    let p = parts(c, "to_protobuf", "TO_PROTOBUF", "to")?;
    let call = Call::new(&p.message, p.descriptor?, p.options, true)?;
    if let Some(input) = c.data_type(0) {
        call.encoder(input)?;
    }
    Ok(ArgType::new(DataType::Binary, true))
}
