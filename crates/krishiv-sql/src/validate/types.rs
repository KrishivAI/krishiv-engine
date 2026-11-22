// SPDX-License-Identifier: Apache-2.0
//! Data types (`visitPrimitiveDataType`, `visitComplexDataType`):
//!
//! | Type | Error class |
//! |---|---|
//! | unknown name, a quoted name, or arguments a type does not take | `UNSUPPORTED_DATATYPE` |
//! | `CHAR` / `VARCHAR` without a length | `DATATYPE_MISSING_SIZE` |
//! | bare `ARRAY` / `MAP` / `STRUCT` | `INCOMPLETE_TYPE_DEFINITION.*` |
//! | decimal bounds | `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION`, `_LEGACY_ERROR_TEMP_1228` |
//! | `TIME(p)` with `p > 6` (`p > 9` under Spark 4.3) | `UNSUPPORTED_TIME_PRECISION` |
//! | Spark 4.3: `TIMESTAMP[_NTZ|_LTZ](p)`, `p` not 6 or 7..=9 | `INVALID_TIMESTAMP_PRECISION` |
//! | Spark 4.3: `TIMESTAMP[_NTZ|_LTZ](7..=9)` (the nanosecond types, off as in Spark) | `FEATURE_NOT_ENABLED` |

use krishiv_common::error::{
    DatatypeMissingSize, FeatureNotEnabled, IncompleteTypeDefinitionArray,
    IncompleteTypeDefinitionMap, IncompleteTypeDefinitionStruct, InvalidTimestampPrecision,
    KrishivError, KrishivUnsupportedFeature, SparkRelease, UnsupportedDatatype,
    UnsupportedTimePrecision,
};

use crate::ParserOptions;
use crate::ast::{DataTypeKind, DataTypeSpec, Span};
use crate::error::ParseError;

/// The largest `TIME` precision before Spark 4.3.
const MAX_TIME_PRECISION: u32 = 6;

/// The largest `TIME` precision from Spark 4.3 (`TimeType.NANOS_PRECISION`).
const MAX_TIME_PRECISION_4_3: u32 = 9;

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
        relation: None,
    }
}

/// Checks a data type as `AstBuilder` does; the first error in tree order.
/// `opts` gives the profile's grammar ([`ParserOptions::spark_4_3`]).
///
/// # Errors
/// The error classes in the module table, at the offending type.
pub(crate) fn data_type(t: &DataTypeSpec, opts: ParserOptions) -> Result<(), ParseError> {
    let data_type = |t: &DataTypeSpec| data_type(t, opts);
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
        DataTypeKind::Primitive { name, args } => primitive(name, args, t.span, opts),
    }
}

fn primitive(
    name: &str,
    args: &[String],
    span: Span,
    opts: ParserOptions,
) -> Result<(), ParseError> {
    let spark_4_3 = opts.spark_4_3;
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
        // Spark reads the length as a Java `int`: a longer one is an unclassified `NumberFormatException`
        // ("For input string"), refused here as unsupported.
        "char" | "character" | "varchar"
            if args.len() == 1 && args[0].trim().parse::<i32>().is_err() =>
        {
            Err(at(
                krishiv_common::error::KrishivUnsupportedFeature {
                    feature: format!(
                        "{} length {} (the largest is 2147483647)",
                        head.to_uppercase(),
                        args[0].trim()
                    ),
                },
                span,
            ))
        }
        "char" | "character" | "varchar" | "geometry" | "geography" if args.len() == 1 => Ok(()),
        "time" if args.is_empty() => Ok(()),
        "time" if args.len() == 1 => {
            let max = if spark_4_3 {
                MAX_TIME_PRECISION_4_3
            } else {
                MAX_TIME_PRECISION
            };
            let fits = args[0].parse::<u32>().is_ok_and(|p| p <= max);
            if fits {
                Ok(())
            } else {
                let e = KrishivError::from(UnsupportedTimePrecision {
                    precision: args[0].clone(),
                });
                Err(at(
                    if spark_4_3 {
                        e.in_release(SparkRelease::V4_3)
                    } else {
                        e
                    },
                    span,
                ))
            }
        }
        "timestamp" | "timestamp_ntz" | "timestamp_ltz" if spark_4_3 && args.len() == 1 => {
            timestamp_precision(name, &args[0], span, opts)
        }
        // Spark 4.2's grammar has no precision before `WITH [LOCAL] TIME ZONE` or `WITHOUT TIME
        // ZONE`: the parser stops at that word. (Spark adds `: missing ')'` for some shapes; the hint
        // is not reproduced.)
        "timestamp" if !spark_4_3 && !args.is_empty() && name.ends_with(" time zone") => {
            let word = name.split(' ').nth(1).unwrap_or_default().to_uppercase();
            Err(crate::error::syntax_error(
                format!("'{word}'"),
                String::new(),
                span,
            ))
        }
        _ if args.is_empty() && (name != head || PLAIN_TYPES.contains(&head)) => Ok(()),
        _ => Err(unsupported(name, args, span)),
    }
}

/// Spark 4.3's `TIMESTAMP[_NTZ|_LTZ](p)` (`AstBuilder.visitPrimitiveDataType`): 6 is the
/// microsecond type; 7..=9 are the nanosecond types, which Krishiv has not built yet and Spark
/// 4.3.0 keeps behind `spark.sql.timestampNanosTypes.enabled` (internal, off): `FEATURE_NOT_ENABLED`,
/// as Spark's default; any other precision is `INVALID_TIMESTAMP_PRECISION`, naming
/// `TIMESTAMP_NTZ` for `_NTZ` and `WITHOUT TIME ZONE`, `TIMESTAMP_LTZ` for `_LTZ` and `WITH LOCAL TIME
/// ZONE`, and for a bare `TIMESTAMP` the type `spark.sql.timestampType` names. With the key on, Spark
/// would build the type; Krishiv has none yet (`KRISHIV_UNSUPPORTED_FEATURE`, an explicit divergence).
fn timestamp_precision(
    name: &str,
    arg: &str,
    span: Span,
    opts: ParserOptions,
) -> Result<(), ParseError> {
    let ntz = match name {
        "timestamp_ntz" => true,
        "timestamp" => opts.timestamp_ntz,
        n => n.ends_with("without time zone"),
    };
    match arg.parse::<u32>() {
        // Spark 4.3 reads `TIMESTAMP(6)` as the session's timestamp type; Krishiv's bare
        // `TIMESTAMP` cast target does not follow `spark.sql.timestampType` yet
        // (known-differences.md), so the precision form is refused rather than typed wrong.
        Ok(6) if name == "timestamp" && opts.timestamp_ntz => Err(at(
            KrishivUnsupportedFeature {
                feature: "TIMESTAMP(6) with spark.sql.timestampType=TIMESTAMP_NTZ".to_owned(),
            },
            span,
        )),
        Ok(6) => Ok(()),
        Ok(7..=9) if opts.timestamp_nanos_types => Err(at(
            KrishivUnsupportedFeature {
                feature: "nanosecond-precision timestamp types (Krishiv 0.3.1)".to_owned(),
            },
            span,
        )),
        Ok(7..=9) => Err(at(
            FeatureNotEnabled {
                feature_name: "Nanosecond-precision timestamp types".to_owned(),
                config_key: "spark.sql.timestampNanosTypes.enabled".to_owned(),
                config_value: "true".to_owned(),
            },
            span,
        )),
        _ => Err(at(
            InvalidTimestampPrecision {
                precision: arg.to_owned(),
                r#type: if ntz {
                    "TIMESTAMP_NTZ"
                } else {
                    "TIMESTAMP_LTZ"
                }
                .to_owned(),
            },
            span,
        )),
    }
}

/// `UNSUPPORTED_DATATYPE` for `name(args)`, upper-cased as Spark shows it.
fn unsupported(name: &str, args: &[String], span: Span) -> ParseError {
    let mut type_name = name.to_uppercase();
    if !args.is_empty() {
        // Spark shows an integer argument as read (`TIMESTAMP_NTZ(06)` is `TIMESTAMP_NTZ(6)`).
        let shown: Vec<String> = args
            .iter()
            .map(|a| {
                a.trim()
                    .parse::<i32>()
                    .map_or_else(|_| a.clone(), |v| v.to_string())
            })
            .collect();
        type_name = format!("{type_name}({})", shown.join(","));
    }
    at(
        UnsupportedDatatype {
            type_name: format!("\"{type_name}\""),
        },
        span,
    )
}
