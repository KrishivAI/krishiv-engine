// SPDX-License-Identifier: Apache-2.0
//! The checks Spark's `AstBuilder` makes while it builds a plan from a
//! parse tree: they raise `ParseException`s for statements the grammar
//! accepts. The parser keeps the facts they need in the tree, and these
//! checks run on the finished tree, so the parser never raises them.
//!
//! Data types (`visitPrimitiveDataType`, `visitComplexDataType`):
//!
//! | Type | Error class |
//! |---|---|
//! | unknown name, a quoted name, or arguments a type does not take | `UNSUPPORTED_DATATYPE` |
//! | `CHAR` / `VARCHAR` without a length | `DATATYPE_MISSING_SIZE` |
//! | bare `ARRAY` / `MAP` / `STRUCT` | `INCOMPLETE_TYPE_DEFINITION.*` |
//! | decimal bounds | `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION`, `_LEGACY_ERROR_TEMP_1228` |
//! | `TIME(p)` with `p > 6` | `UNSUPPORTED_TIME_PRECISION` |

use krishiv_common::error::{
    DatatypeMissingSize, IncompleteTypeDefinitionArray, IncompleteTypeDefinitionMap,
    IncompleteTypeDefinitionStruct, KrishivError, UnsupportedDatatype, UnsupportedTimePrecision,
};

use crate::ast::{DataTypeKind, DataTypeSpec, Span};
use crate::error::ParseError;

/// The largest `TIME` precision.
const MAX_TIME_PRECISION: u32 = 6;

/// Type names that take no arguments.
const PLAIN_TYPES: &[&str] = &[
    "boolean",
    "tinyint",
    "byte",
    "smallint",
    "short",
    "int",
    "integer",
    "bigint",
    "long",
    "float",
    "real",
    "double",
    "date",
    "timestamp",
    "timestamp_ntz",
    "timestamp_ltz",
    "string",
    "binary",
    "void",
    "variant",
    "interval",
];

fn at(error: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: error.into(),
        span,
    }
}

/// Checks a data type as `AstBuilder` does; the first error in tree order.
///
/// # Errors
/// The error classes in the module table, at the offending type.
pub(crate) fn data_type(t: &DataTypeSpec) -> Result<(), ParseError> {
    match &t.kind {
        DataTypeKind::Array(None) => Err(at(
            IncompleteTypeDefinitionArray {
                element_type: "<INT>".to_owned(),
            },
            t.span,
        )),
        DataTypeKind::Map(None) => Err(at(IncompleteTypeDefinitionMap {}, t.span)),
        DataTypeKind::Struct(None) => Err(at(IncompleteTypeDefinitionStruct {}, t.span)),
        DataTypeKind::Array(Some(element)) => data_type(element),
        DataTypeKind::Map(Some(kv)) => {
            data_type(&kv.0)?;
            data_type(&kv.1)
        }
        DataTypeKind::Struct(Some(fields)) => {
            fields.iter().try_for_each(|f| data_type(&f.data_type))
        }
        DataTypeKind::Quoted { name, args } => Err(unsupported(name, args, t.span)),
        DataTypeKind::Primitive { name, args } => primitive(name, args, t.span),
    }
}

fn primitive(name: &str, args: &[String], span: Span) -> Result<(), ParseError> {
    // Multi-word names (`interval day to second`, `timestamp with local
    // time zone`) never carry arguments after their last word.
    let head = name.split(' ').next().unwrap_or(name);
    match head {
        "decimal" | "dec" | "numeric" if args.len() <= 2 => {
            let ddl = if args.is_empty() {
                "decimal".to_owned()
            } else {
                format!("decimal({})", args.join(","))
            };
            match krishiv_types::parse_data_type(&ddl) {
                Ok(_) => Ok(()),
                Err(e) => Err(at(e, span)),
            }
        }
        "char" | "character" | "varchar" if args.is_empty() => Err(at(
            DatatypeMissingSize {
                r#type: format!("\"{}\"", head.to_uppercase()),
            },
            span,
        )),
        // The spatial reference of a geometry type is checked when the type
        // is resolved.
        "char" | "character" | "varchar" | "geometry" | "geography" if args.len() == 1 => Ok(()),
        "time" if args.is_empty() => Ok(()),
        "time" if args.len() == 1 => {
            let fits = args[0]
                .parse::<u32>()
                .is_ok_and(|p| p <= MAX_TIME_PRECISION);
            if fits {
                Ok(())
            } else {
                Err(at(
                    UnsupportedTimePrecision {
                        precision: args[0].clone(),
                    },
                    span,
                ))
            }
        }
        _ if args.is_empty() && (name != head || PLAIN_TYPES.contains(&head)) => Ok(()),
        _ => Err(unsupported(name, args, span)),
    }
}

/// `UNSUPPORTED_DATATYPE` for `name(args)`, upper-cased as Spark shows it.
fn unsupported(name: &str, args: &[String], span: Span) -> ParseError {
    let mut type_name = name.to_uppercase();
    if !args.is_empty() {
        type_name = format!("{type_name}({})", args.join(","));
    }
    at(
        UnsupportedDatatype {
            type_name: format!("\"{type_name}\""),
        },
        span,
    )
}
