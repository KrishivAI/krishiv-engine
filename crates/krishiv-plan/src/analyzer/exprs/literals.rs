// SPDX-License-Identifier: Apache-2.0
//! Literal types, typed literals (`DATE '...'` and friends) and the cast errors Spark raises for them.

use super::*;

pub(in crate::analyzer) fn literal_type(v: &Literal) -> Option<DataType> {
    Some(match v {
        Literal::Null(t) => return t.clone().filter(|t| *t != DataType::Null),
        Literal::Boolean(_) => DataType::Boolean,
        Literal::Byte(_) => DataType::Byte,
        Literal::Short(_) => DataType::Short,
        Literal::Integer(_) => DataType::Integer,
        Literal::Long(_) => DataType::Long,
        Literal::Float(_) => DataType::Float,
        Literal::Double(_) => DataType::Double,
        Literal::Decimal { data_type, .. } => DataType::Decimal(*data_type),
        Literal::String(_) => DataType::String,
        Literal::Map(_) => DataType::Map(
            krishiv_types::MapType::new(DataType::String, DataType::String, false).into(),
        ),
        Literal::Binary(_) => DataType::Binary,
        Literal::Date(_) => DataType::Date,
        Literal::Timestamp(_) => DataType::Timestamp,
        Literal::TimestampNtz(_) => DataType::TimestampNtz,
        Literal::YearMonthInterval { start, end, .. } => DataType::YearMonthInterval {
            start: *start,
            end: *end,
        },
        Literal::DayTimeInterval { start, end, .. } => DataType::DayTimeInterval {
            start: *start,
            end: *end,
        },
        Literal::CalendarInterval { .. } => DataType::CalendarInterval,
    })
}

/// A typed datetime literal as Spark's `AstBuilder` builds it: the same
/// `CAST('...' AS TIMESTAMP)` kernels execution uses, evaluated here with
/// the session's resolved time zone and timestamp type. Plain `TIMESTAMP`
/// means `TIMESTAMP_NTZ` when `spark.sql.timestampType = TIMESTAMP_NTZ`
/// and the value names no zone, else an instant. Values needing the query
/// clock (`now`, `today`, a bare time of day) or the time-zone database
/// (a region id in the value, or a region session zone for a value
/// without a zone) are `KRISHIV_UNSUPPORTED_FEATURE`, as the kernels
/// report them.
///
/// # Errors
/// `INVALID_TYPED_LITERAL` for text the type does not read, at the
/// literal's origin; `KRISHIV_UNSUPPORTED_FEATURE` for clock- or
/// time-zone-dependent values.
pub(super) fn typed_literal(
    kind: TypedLiteralKind,
    value: &str,
    conf: &super::super::ResolutionConf,
    origin: Origin,
) -> Result<Literal, AnalysisError> {
    use krishiv_common::error::{ErrorClass, InvalidTypedLiteral, KrishivUnsupportedFeature};
    use krishiv_kernels::cast::{
        CastMode as KernelMode, CastOptions, FixedOffset, TimestampText, cast, timestamp_has_zone,
        timestamp_text,
    };
    let invalid = |value_type: &str| {
        AnalysisError::new(
            InvalidTypedLiteral {
                value: format!("'{value}'"),
                value_type: format!("\"{value_type}\""),
            },
            origin,
        )
    };
    let unsupported = |feature: &str| {
        AnalysisError::new(
            KrishivUnsupportedFeature {
                feature: feature.to_owned(),
            },
            origin,
        )
    };
    let type_name = match kind {
        TypedLiteralKind::Date => "DATE",
        TypedLiteralKind::Timestamp => "TIMESTAMP",
        TypedLiteralKind::TimestampNtz => "TIMESTAMP_NTZ",
        TypedLiteralKind::TimestampLtz => "TIMESTAMP_LTZ",
        // A time literal is resolved by the caller, as a cast.
        TypedLiteralKind::Time => return Err(unsupported("a TIME literal here")),
    };
    // Clock-dependent spellings evaluate with the query clock, which the
    // analyzer has not got.
    if is_special(value) {
        return Err(unsupported(
            "a timestamp literal that needs the query clock",
        ));
    }
    let shape = timestamp_text(value);
    // A `DATE` is checked by its own cast below: the timestamp grammar
    // rejects dates whose instant is past the microsecond range
    // (`DATE '999999-03-18'`, valid in Spark), so only a bare time of day
    // is read from the timestamp shape for it.
    if shape == TimestampText::Malformed && kind != TypedLiteralKind::Date {
        return Err(invalid(type_name));
    }
    // A bare time of day is valid only for an instant type, and needs the
    // clock there too.
    if shape == TimestampText::TimeOnly {
        let without_zone = kind == TypedLiteralKind::Date
            || kind == TypedLiteralKind::TimestampNtz
            || (kind == TypedLiteralKind::Timestamp && conf.timestamp_ntz);
        if without_zone {
            return Err(invalid(type_name));
        }
        return Err(unsupported(
            "a timestamp literal that needs the query clock",
        ));
    }
    // One string cast through the kernels, as a microsecond instant or
    // NTZ wall-clock value.
    let cast_one = |value: &str, to: &arrow_schema::DataType, zone: &dyn TimeZoneRules| {
        use arrow_array::StringArray;
        use arrow_array::cast::AsArray as _;
        use arrow_array::types::TimestampMicrosecondType;
        let input = StringArray::from(vec![value]);
        let opts = CastOptions {
            mode: KernelMode::Ansi,
            time_zone: zone,
        };
        let out = cast(&input, to, &opts).map_err(KrishivError::from)?;
        if out.is_null(0) {
            // ANSI casts raise rather than null malformed input; a NULL
            // here is unreachable.
            return Err(KrishivError::from(KrishivUnsupportedFeature {
                feature: "a timestamp literal".to_owned(),
            }));
        }
        Ok(out.as_primitive::<TimestampMicrosecondType>().value(0))
    };
    let ltz_arrow = || {
        arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into()))
    };
    let ntz_arrow = || arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None);
    let map_error = |e: KrishivError| {
        if e.error_class() == ErrorClass::KrishivUnsupportedFeature {
            AnalysisError::new(e, origin)
        } else {
            invalid(type_name)
        }
    };
    match kind {
        TypedLiteralKind::Time => Err(unsupported("a TIME literal here")),
        TypedLiteralKind::Date => {
            use arrow_array::cast::AsArray as _;
            use arrow_array::types::Date32Type;
            let input = arrow_array::StringArray::from(vec![value]);
            let opts = CastOptions {
                mode: KernelMode::Ansi,
                time_zone: &FixedOffset(0),
            };
            let out = cast(&input, &arrow_schema::DataType::Date32, &opts)
                .map_err(|e| map_error(KrishivError::from(e)))?;
            if out.is_null(0) {
                return Err(invalid(type_name));
            }
            Ok(Literal::Date(out.as_primitive::<Date32Type>().value(0)))
        }
        TypedLiteralKind::TimestampNtz => {
            let micros = cast_one(value, &ntz_arrow(), &FixedOffset(0)).map_err(map_error)?;
            Ok(Literal::TimestampNtz(micros))
        }
        TypedLiteralKind::TimestampLtz => {
            // Session-independent values (an explicit zone) evaluate under
            // any zone; only session-dependent ones read the session zone.
            if timestamp_has_zone(value) {
                let micros = cast_one(value, &ltz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::Timestamp(micros))
            } else {
                match conf.session_zone {
                    Some(zone) => {
                        let micros = cast_one(value, &ltz_arrow(), &zone).map_err(map_error)?;
                        Ok(Literal::Timestamp(micros))
                    }
                    None => Err(unsupported(
                        "a timestamp literal in an unresolvable session time zone",
                    )),
                }
            }
        }
        TypedLiteralKind::Timestamp if conf.timestamp_ntz => {
            // Under `TIMESTAMP_NTZ`, a value naming a zone is still an
            // instant; without one it is wall-clock time.
            if timestamp_has_zone(value) {
                let micros = cast_one(value, &ltz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::Timestamp(micros))
            } else {
                let micros = cast_one(value, &ntz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::TimestampNtz(micros))
            }
        }
        TypedLiteralKind::Timestamp => {
            if timestamp_has_zone(value) {
                let micros = cast_one(value, &ltz_arrow(), &FixedOffset(0)).map_err(map_error)?;
                Ok(Literal::Timestamp(micros))
            } else {
                match conf.session_zone {
                    Some(zone) => {
                        let micros = cast_one(value, &ltz_arrow(), &zone).map_err(map_error)?;
                        Ok(Literal::Timestamp(micros))
                    }
                    None => Err(unsupported(
                        "a timestamp literal in an unresolvable session time zone",
                    )),
                }
            }
        }
    }
}

/// Whether `value` is one of Spark's special date/timestamp spellings
/// (`now`, `today`, ...), which depend on the session's clock and zone.
pub(super) fn is_special(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    ["epoch", "now", "today", "yesterday", "tomorrow"].contains(&v.as_str())
}

/// The type of a value as typing sees it: `VOID` is an untyped `NULL`
/// (`data_type: None`) everywhere but in plans and schemas.
pub(in crate::analyzer) fn arg_type(data_type: DataType, nullable: bool) -> ArgType {
    ArgType {
        data_type: (data_type != DataType::Null).then_some(data_type),
        nullable,
    }
}

/// Spark's error for a cast of `arg` to `to` it refuses.
pub(super) fn bad_cast(
    bad: BadCast,
    origin: Origin,
    arg: &Bound,
    to: &DataType,
    mode: CastMode,
    zone: Option<SessionZone>,
) -> AnalysisError {
    let shown = ExprNode::Cast {
        expr: arg.shown(),
        to: to.clone(),
        mode,
    };
    let sql_expr = format!("\"{}\"", pretty(&Expr::new(shown), zone));
    let error = cast_mismatch(
        bad,
        sql_expr,
        quoted_type(arg.ty.data_type.as_ref()),
        quoted_type(Some(to)),
    );
    AnalysisError::new(error, origin)
}

/// The error for a cast of `arg` to `to` that `bad` says is refused;
/// `sql_expr` is the quoted expression Spark shows in it.
pub(in crate::analyzer) fn cast_error(
    bad: BadCast,
    sql_expr: String,
    arg: &Bound,
    to: &DataType,
) -> KrishivError {
    cast_mismatch(
        bad,
        sql_expr,
        quoted_type(arg.ty.data_type.as_ref()),
        quoted_type(Some(to)),
    )
}

/// Why a cast fails: the error of [`BadCast`], quoting `sql_expr` (the
/// expression Spark's message names) with the source and target types.
/// A `CAST` quotes the cast; a cast-named function (`float(x)`, ...)
/// quotes its argument, as Spark does.
pub(super) fn cast_mismatch(
    bad: BadCast,
    sql_expr: String,
    src_type: String,
    target_type: String,
) -> KrishivError {
    match bad {
        BadCast::Without => DatatypeMismatchCastWithoutSuggestion {
            sql_expr,
            src_type,
            target_type,
        }
        .into(),
        BadCast::Func(f) => DatatypeMismatchCastWithFuncSuggestion {
            sql_expr,
            src_type,
            target_type,
            function_names: f.to_owned(),
        }
        .into(),
        BadCast::Conf => DatatypeMismatchCastWithConfSuggestion {
            sql_expr,
            src_type,
            target_type,
            config: "\"spark.sql.ansi.enabled\"".to_owned(),
            config_val: "'false'".to_owned(),
        }
        .into(),
    }
}
