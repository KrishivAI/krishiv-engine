// SPDX-License-Identifier: Apache-2.0
//! Literal types, typed literals (`DATE '...'` and friends) and the cast errors Spark raises for them.

use super::*;
use krishiv_kernels::cast::{TimestampText, timestamp_text};

pub(in crate::analyzer) fn literal_type(v: &Literal) -> Option<DataType> {
    v.data_type()
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
/// The type a typed datetime literal denotes.
#[derive(Clone, Copy)]
pub(super) enum SpecialKind {
    /// `DATE`.
    Date,
    /// `TIMESTAMP` / `TIMESTAMP_LTZ`: an instant.
    Instant,
    /// `TIMESTAMP_NTZ` (or `TIMESTAMP` under `spark.sql.timestampType`).
    Local,
}

pub(super) fn special_kind(kind: TypedLiteralKind, timestamp_ntz: bool) -> SpecialKind {
    match kind {
        TypedLiteralKind::Date => SpecialKind::Date,
        TypedLiteralKind::TimestampNtz => SpecialKind::Local,
        TypedLiteralKind::Timestamp if timestamp_ntz => SpecialKind::Local,
        _ => SpecialKind::Instant,
    }
}

/// The clock expression Spark's special value `now`, `today`, `tomorrow` or
/// `yesterday` stands for, of the literal's type.
pub(super) fn special_clock(special: &str, kind: SpecialKind) -> Expr {
    let day = |offset: &str| match offset {
        "tomorrow" => Expr::function(
            "date_add",
            vec![
                Expr::function("current_date", Vec::new()),
                Expr::literal(Literal::Integer(1)),
            ],
        ),
        "yesterday" => Expr::function(
            "date_sub",
            vec![
                Expr::function("current_date", Vec::new()),
                Expr::literal(Literal::Integer(1)),
            ],
        ),
        _ => Expr::function("current_date", Vec::new()),
    };
    let cast = |expr: Expr, to: DataType| {
        Expr::new(ExprNode::Cast {
            expr: Arc::new(expr),
            to,
            mode: CastMode::Ansi,
        })
    };
    match (kind, special) {
        (SpecialKind::Date, d) => day(d),
        (SpecialKind::Instant, "now") => Expr::function("current_timestamp", Vec::new()),
        (SpecialKind::Local, "now") => Expr::function("localtimestamp", Vec::new()),
        (SpecialKind::Instant, d) => cast(day(d), DataType::Timestamp),
        (SpecialKind::Local, d) => cast(day(d), DataType::TimestampNtz),
    }
}

/// A bare time of day (`10:00`, `T10:00:30.5`, no zone) of an instant type: Spark reads it as that time
/// today, in the session zone. It binds as the cast of the query-start date and the time, so a
/// `TIMESTAMP '11:00'` is `CAST(CONCAT(CAST(current_date() AS STRING), 'T11:00') AS TIMESTAMP)`.
/// `None` for any other value (a zone in the value moves the date to that zone's today, which the
/// session-zone clock cannot give; those stay `KRISHIV_UNSUPPORTED_FEATURE`).
pub(super) fn time_only_today(
    kind: TypedLiteralKind,
    value: &str,
    timestamp_ntz: bool,
) -> Option<Expr> {
    let instant = match kind {
        TypedLiteralKind::TimestampLtz => true,
        TypedLiteralKind::Timestamp => !timestamp_ntz,
        _ => false,
    };
    let time = value.trim();
    let time = time.strip_prefix('T').unwrap_or(time);
    let mut parts = time.splitn(3, ':');
    let digits = |p: Option<&str>, max: usize| {
        p.is_some_and(|p| (1..=max).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
    };
    let seconds_ok = |p: Option<&str>| {
        p.is_none_or(|p| {
            let (whole, fraction) = p.split_once('.').map_or((p, None), |(w, f)| (w, Some(f)));
            digits(Some(whole), 2)
                && fraction.is_none_or(|f| {
                    (1..=9).contains(&f.len()) && f.bytes().all(|b| b.is_ascii_digit())
                })
        })
    };
    let (hours, minutes, seconds) = (parts.next(), parts.next(), parts.next());
    if !instant
        || !digits(hours, 2)
        || !digits(minutes, 2)
        || !seconds_ok(seconds)
        || timestamp_text(value) != TimestampText::TimeOnly
    {
        return None;
    }
    let today = Expr::new(ExprNode::Cast {
        expr: Arc::new(Expr::function("current_date", Vec::new())),
        to: DataType::String,
        mode: CastMode::Ansi,
    });
    let text = Expr::function(
        "concat",
        vec![today, Expr::literal(Literal::String(format!("T{time}")))],
    );
    Some(Expr::new(ExprNode::Cast {
        expr: Arc::new(text),
        to: DataType::Timestamp,
        mode: CastMode::Ansi,
    }))
}

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
