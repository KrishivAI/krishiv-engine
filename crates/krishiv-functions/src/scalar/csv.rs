// SPDX-License-Identifier: Apache-2.0
//! `from_csv` and `to_csv`: CSV records into structs and back.
//!
//! `from_csv(csvStr, schema[, options])` parses one CSV record per row into
//! the struct its foldable DDL schema names; `to_csv(struct[, options])`
//! renders a struct to one CSV line. Options are parsed once at bind; plans
//! stay data. Malformed rows follow `mode` (`PERMISSIVE` with
//! `columnNameOfCorruptRecord`, `DROPMALFORMED`, `FAILFAST`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, Field, Fields};
use krishiv_common::error::{ErrorClass, InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_kernels::cast::TimeZoneRules;
use krishiv_kernels::csv::{
    FieldKind, FromCsvOptions, ParseMode, Syntax, ToCsvOptions, Unescaped,
    from_csv as kernel_from_csv, struct_to_csv,
};
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose};
use krishiv_kernels::text::{spark_string_to_timestamp, spark_string_to_timestamp_ntz};

use super::args::{Inputs, internal, is_string, to_utf8, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["from_csv", "to_csv"];

/// The shape of an argument list for error messages.
fn shape(args: &[BindArg]) -> String {
    args.iter()
        .map(|a| match a {
            BindArg::Column { ty, .. } => match ty {
                DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => 'S',
                DataType::Struct(_) => 'T',
                _ => '?',
            },
            BindArg::Literal(a) => match a.data_type() {
                DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => 's',
                DataType::Struct(_) => 't',
                DataType::Map(_, _) => 'm',
                DataType::Null => 'n',
                _ => '?',
            },
        })
        .collect()
}

/// Reads a `STRING` literal's value.
pub(super) fn literal_string(arg: &BindArg) -> Result<String, KrishivError> {
    let BindArg::Literal(a) = arg else {
        return Err(internal(
            "a literal schema where a column was bound".to_owned(),
        ));
    };
    let s = to_utf8(a.as_ref())?;
    if s.is_null(0) {
        return Err(internal("a null literal schema".to_owned()));
    }
    Ok(s.value(0).to_owned())
}

/// Parses a DDL schema string into struct fields (Spark's `DataType.fromDDL`).
pub(super) fn struct_fields(ddl: &str) -> Result<Fields, KrishivError> {
    let parsed: krishiv_types::DataType = ddl.parse().or_else(|_: KrishivError| {
        let wrapped = format!("STRUCT<{ddl}>");
        wrapped.parse()
    })?;
    let krishiv_types::DataType::Struct(s) = parsed else {
        // The registry already rejected non-struct schemas; the binding
        // sees only structs.
        return Err(krishiv_common::error::InvalidSchemaNonStructType {
            input_schema: format!("\"{ddl}\""),
            data_type: format!("\"{}\"", parsed.sql()),
        }
        .into());
    };
    let fields: Vec<Field> = s
        .fields()
        .iter()
        .map(|f| {
            let arrow = f.data_type().to_arrow();
            Field::new(f.name().to_owned(), arrow, f.nullable())
        })
        .collect();
    Ok(Fields::from(fields))
}

/// One `sep`/`delimiter` value with Spark's backslash escapes.
fn parse_sep(raw: &str) -> Result<String, KrishivError> {
    if raw.is_empty() {
        return Err(unsupported("from_csv with an empty separator".to_owned()));
    }
    let mut out = String::new();
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{8}'),
            Some('f') => out.push('\u{c}'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                let c = u32::from_str_radix(&hex, 16)
                    .ok()
                    .filter(|_| hex.len() == 4)
                    .and_then(char::from_u32)
                    .ok_or_else(|| {
                        unsupported("from_csv with a bad \\u escape in sep".to_owned())
                    })?;
                out.push(c);
            }
            _ => {
                return Err(unsupported(
                    "from_csv with an unsupported escape in sep".to_owned(),
                ));
            }
        }
    }
    Ok(out)
}

/// A single-character option (`quote`, `escape`): `None` when empty.
fn one_char(value: &str, name: &str, func: &str) -> Result<Option<char>, KrishivError> {
    let mut chars = value.chars();
    match (chars.next(), chars.next()) {
        (None, _) => Ok(None),
        (Some(c), None) => Ok(Some(c)),
        _ => Err(unsupported(format!(
            "{func} with option {name} of more than one character"
        ))),
    }
}

/// Reads `MAP<STRING, STRING>` literal options into pairs.
pub(super) fn map_pairs(arg: &BindArg) -> Result<Vec<(String, String)>, KrishivError> {
    let BindArg::Literal(a) = arg else {
        return Err(internal("CSV options that are not a literal".to_owned()));
    };
    let map = a.as_map();
    if map.len() != 1 {
        return Err(internal(format!("CSV options of {} rows", map.len())));
    }
    let keys = to_utf8(map.keys().as_ref())?;
    let values = to_utf8(map.values().as_ref())?;
    let mut out = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        if keys.is_null(i) || values.is_null(i) {
            continue;
        }
        out.push((keys.value(i).to_owned(), values.value(i).to_owned()));
    }
    Ok(out)
}

/// Binds the CSV function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let result = match shape(args).as_str() {
        s if listed == "from_csv" && matches!(s, "Ss" | "SS" | "Ssm" | "SSm" | "Ss?" | "SS?") => {
            bind_from_csv(args, opts)
        }
        s if listed == "to_csv" && matches!(s, "T" | "Tm" | "t" | "tm") => bind_to_csv(args, opts),
        _ => Err(bad_types(listed, args)),
    };
    // Handle shapes the table above misses (e.g. a `Utf8View` column is
    // `S`, but a literal struct or map has its own letters).
    let result = match result {
        Err(e) if args.len() == 2 && listed == "from_csv" && is_string(&args[0].data_type()) => {
            bind_from_csv_fallback(args, opts, e)
        }
        Err(e) if args.len() == 3 && listed == "from_csv" && is_string(&args[0].data_type()) => {
            bind_from_csv_fallback(args, opts, e)
        }
        Err(e) if (args.len() == 1 || args.len() == 2) && listed == "to_csv" => {
            bind_to_csv_fallback(args, opts, e)
        }
        other => other,
    };
    Some((listed, result))
}

/// `from_csv` when the shape table misses (e.g. a `LargeUtf8` column): check
/// by kind rather than by letter.
fn bind_from_csv_fallback(
    args: &[BindArg],
    opts: &BindOptions,
    first_err: KrishivError,
) -> Result<Binding, KrishivError> {
    if !is_string(&args[0].data_type()) {
        return Err(first_err);
    }
    bind_from_csv(args, opts)
}

/// `to_csv` shape fallback: the first argument must be a struct.
fn bind_to_csv_fallback(
    args: &[BindArg],
    opts: &BindOptions,
    first_err: KrishivError,
) -> Result<Binding, KrishivError> {
    if !matches!(args[0].data_type(), DataType::Struct(_)) {
        return Err(first_err);
    }
    bind_to_csv(args, opts)
}

/// Parses shared CSV options from `pairs` for `func`.
struct Parsed {
    /// Tokenizer syntax.
    syntax: Syntax,
    /// Parse mode.
    mode: ParseMode,
    /// Corrupt-record column name.
    corrupt_column: String,
    /// NULL text.
    null_value: String,
    /// NaN text.
    nan: String,
    /// +Inf text.
    pos_inf: String,
    /// -Inf text.
    neg_inf: String,
    /// Date pattern.
    date_pattern: Option<DateTimePattern>,
    /// Timestamp pattern.
    timestamp_pattern: Option<DateTimePattern>,
    /// Timestamp_ntz pattern.
    ntz_pattern: Option<DateTimePattern>,
    /// Fallback to lenient grammar.
    fallback: bool,
    /// Quote every value (`to_csv` write).
    quote_all: bool,
    /// Quote values holding a quote.
    escape_quotes: bool,
    /// Custom line separator.
    line_sep: Option<String>,
}

/// A `dateFormat` / `timestampFormat` / `timestampNTZFormat` option, compiled
/// for the operation: `to_csv` formats, `from_csv` parses. Spark's pattern
/// errors pass through; only a pattern letter this engine lacks becomes an
/// unsupported-feature error naming the option.
pub(super) fn datetime_option(
    pattern: &str,
    func: &str,
    option: &str,
) -> Result<DateTimePattern, KrishivError> {
    let purpose = if func == "to_csv" {
        Purpose::Format
    } else {
        Purpose::Parse
    };
    datetime_option_as(pattern, func, option, purpose)
}

/// [`datetime_option`] for an explicit `purpose` (the XML functions).
pub(super) fn datetime_option_as(
    pattern: &str,
    func: &str,
    option: &str,
    purpose: Purpose,
) -> Result<DateTimePattern, KrishivError> {
    DateTimePattern::new(pattern, purpose).map_err(|e| {
        if e.error_class() == ErrorClass::KrishivUnsupportedFeature {
            KrishivUnsupportedFeature {
                feature: format!("{func} with option {option}"),
            }
            .into()
        } else {
            e
        }
    })
}

impl Parsed {
    /// Defaults for `func`.
    fn defaults() -> Self {
        Self {
            syntax: Syntax::default(),
            mode: ParseMode::Permissive,
            corrupt_column: "_corrupt_record".to_owned(),
            null_value: String::new(),
            nan: "NaN".to_owned(),
            pos_inf: "Inf".to_owned(),
            neg_inf: "-Inf".to_owned(),
            date_pattern: None,
            timestamp_pattern: None,
            ntz_pattern: None,
            fallback: false,
            quote_all: false,
            escape_quotes: true,
            line_sep: None,
        }
    }

    /// Applies `pairs` for `func` (analysis already checked the map holds
    /// strings).
    fn apply(&mut self, pairs: &[(String, String)], func: &str) -> Result<(), KrishivError> {
        for (k, v) in pairs {
            // Option names are case-insensitive in Spark.
            let lower = k.to_ascii_lowercase();
            match lower.as_str() {
                "sep" | "delimiter" => {
                    if lower.as_str() == "delimiter"
                        && pairs.iter().any(|(k2, _)| k2.eq_ignore_ascii_case("sep"))
                    {
                        // `sep` wins when both are given (as Spark's reader does).
                        continue;
                    }
                    self.syntax.sep = parse_sep(v)?;
                }
                "quote" => self.syntax.quote = one_char(v, "quote", func)?,
                "escape" => self.syntax.escape = one_char(v, "escape", func)?,
                "chartoescapequoteescaping" => {
                    self.syntax.escape_escape = one_char(v, "charToEscapeQuoteEscaping", func)?;
                }
                "nullvalue" => self.null_value = v.clone(),
                "nanvalue" => self.nan = v.clone(),
                "positiveinf" => self.pos_inf = v.clone(),
                "negativeinf" => self.neg_inf = v.clone(),
                "mode" => {
                    self.mode = match v.to_ascii_uppercase().as_str() {
                        "PERMISSIVE" => ParseMode::Permissive,
                        // Spark's `from_csv` and `to_csv` accept PERMISSIVE and
                        // FAILFAST only.
                        "DROPMALFORMED" => {
                            return Err(krishiv_common::error::ParseModeUnsupported {
                                func_name: format!("`{func}`"),
                                mode: "DROPMALFORMED".to_owned(),
                            }
                            .into());
                        }
                        "FAILFAST" => ParseMode::FailFast,
                        _ => {
                            return Err(unsupported(format!("{func} with option mode={v}")));
                        }
                    };
                }
                "columnnameofcorruptrecord" => self.corrupt_column = v.clone(),
                "dateformat" => {
                    self.date_pattern = Some(datetime_option(v, func, "dateFormat")?);
                }
                "timestampformat" => {
                    self.timestamp_pattern = Some(datetime_option(v, func, "timestampFormat")?);
                }
                "timestampntzformat" => {
                    self.ntz_pattern = Some(datetime_option(v, func, "timestampNTZFormat")?);
                }
                "enabledatetimeparsingfallback" => {
                    self.fallback = v.eq_ignore_ascii_case("true");
                }
                "timeformat" => {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!("{func} with option timeFormat"),
                    }
                    .into());
                }
                "header" => {
                    // Ignored for `from_csv` (Spark ignores it); `to_csv`
                    // never writes one. Any value goes.
                }
                "emptyvalue" => self.syntax.empty_value = v.clone(),
                "ignoreleadingwhitespace" => {
                    self.syntax.ltrim = v.eq_ignore_ascii_case("true");
                }
                "ignoretrailingwhitespace" => {
                    self.syntax.rtrim = v.eq_ignore_ascii_case("true");
                }
                "quoteall" => self.quote_all = v.eq_ignore_ascii_case("true"),
                "escapequotes" => self.escape_quotes = !v.eq_ignore_ascii_case("false"),
                "linesep" => {
                    if v != "\n" {
                        self.line_sep = Some(v.clone());
                    }
                }
                "unescapedquotehandling" => {
                    self.syntax.unescaped = match v.to_ascii_uppercase().as_str() {
                        "STOP_AT_CLOSING_QUOTE" => Unescaped::StopAtClosingQuote,
                        "BACK_TO_DELIMITER" => Unescaped::BackToDelimiter,
                        "STOP_AT_DELIMITER" => Unescaped::StopAtDelimiter,
                        "SKIP_VALUE" => Unescaped::SkipValue,
                        "RAISE_ERROR" => Unescaped::RaiseError,
                        _ => {
                            return Err(unsupported(format!(
                                "{func} with option unescapedQuoteHandling={v}"
                            )));
                        }
                    };
                }
                "maxcolumns" => {
                    if let Ok(n) = v.parse::<usize>() {
                        self.syntax.max_columns = n.max(1);
                    }
                }
                "maxcharspercolumn" => {
                    if let Ok(n) = v.parse::<i64>() {
                        self.syntax.max_chars = (n >= 0).then(|| usize::try_from(n).ok()).flatten();
                    }
                }
                _ => {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!("{func} with option {k}"),
                    }
                    .into());
                }
            }
        }
        // `sep` wins over `delimiter` when both are given.
        Ok(())
    }
}

/// Binds `from_csv(csvStr, schema[, options])`.
fn bind_from_csv(args: &[BindArg], opts: &BindOptions) -> Result<Binding, KrishivError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(bad_types("from_csv", args));
    }
    if !is_string(&args[0].data_type()) {
        return Err(bad_types("from_csv", args));
    }
    let ddl = literal_string(&args[1])?;
    let fields = struct_fields(&ddl)?;
    let kinds: Vec<FieldKind> = fields
        .iter()
        .map(|f| {
            FieldKind::of(f.data_type()).map_err(|_| KrishivUnsupportedFeature {
                feature: "from_csv with a nested field type".to_owned(),
            })
        })
        .collect::<Result<_, _>>()?;
    let mut parsed = Parsed::defaults();
    if args.len() == 3 {
        // Options are a `MAP<STRING, STRING>` literal (analysis checked the
        // types); anything else was refused during analysis.
        if !matches!(args[2].data_type(), DataType::Map(_, _)) {
            return Err(bad_types("from_csv", args));
        }
        let pairs = map_pairs(&args[2])?;
        parsed.apply(&pairs, "from_csv")?;
    }
    let corrupt = fields
        .iter()
        .position(|f| f.name() == &parsed.corrupt_column);
    // A corrupt-record column must be a string.
    if let Some(i) = corrupt
        && !matches!(
            fields[i].data_type(),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
        )
    {
        return Err(bad_types("from_csv", args));
    }
    let return_type = DataType::Struct(fields.clone());
    let zone: Arc<dyn TimeZoneRules> = Arc::clone(&opts.time_zone);
    let date_pattern = parsed.date_pattern.clone();
    let timestamp_pattern = parsed.timestamp_pattern.clone();
    let ntz_pattern = parsed.ntz_pattern.clone();
    let from_opts = FromCsvOptions {
        syntax: parsed.syntax,
        mode: parsed.mode,
        null_value: parsed.null_value,
        nan: parsed.nan,
        pos_inf: parsed.pos_inf,
        neg_inf: parsed.neg_inf,
        date_pattern,
        timestamp_pattern,
        ntz_pattern,
        fallback: parsed.fallback,
    };
    // A time-only timestamp (`12:30:00`) is today's, in the session zone.
    let today = krishiv_kernels::datetime::current_date(opts.query_start_micros, zone.as_ref())
        .map_err(KrishivError::from)?;
    Ok(Binding::new(return_type, move |x: &Inputs<'_>| {
        let csv = x.strings(0)?;
        let zone_ref: &dyn TimeZoneRules = zone.as_ref();
        let parse_timestamp = |s: &str| {
            spark_string_to_timestamp(s, zone_ref, today)
                .map_err(|e| krishiv_kernels::error::KernelError::with_row(None, e))
        };
        let parse_ntz = |s: &str| -> Result<Option<i64>, krishiv_kernels::error::KernelError> {
            Ok(spark_string_to_timestamp_ntz(s))
        };
        let out = kernel_from_csv(
            &csv,
            &fields,
            &kinds,
            corrupt,
            &from_opts,
            zone_ref,
            &parse_timestamp,
            &parse_ntz,
        )
        .map_err(|e| e.error().clone())?;
        Ok(Arc::new(out) as ArrayRef)
    }))
}

/// Binds `to_csv(struct[, options])`.
fn bind_to_csv(args: &[BindArg], opts: &BindOptions) -> Result<Binding, KrishivError> {
    if args.len() > 2 {
        return Err(bad_types("to_csv", args));
    }
    let DataType::Struct(fields) = args[0].data_type() else {
        return Err(bad_types("to_csv", args));
    };
    let kinds: Vec<FieldKind> = fields
        .iter()
        .map(|f| {
            FieldKind::of_write(f.data_type()).map_err(|_| {
                krishiv_common::error::UnsupportedDataTypeForDatasource {
                    format: "CSV".to_owned(),
                    column_name: format!("`{}`", f.name()),
                    column_type: format!("\"{}\"", f.data_type()),
                }
            })
        })
        .collect::<Result<_, _>>()?;
    let mut parsed = Parsed::defaults();
    // Writing defaults (Spark's CSV writer): an empty string is written as
    // two quotes, and the whitespace around a value is dropped.
    parsed.syntax.empty_value = "\"\"".to_owned();
    parsed.syntax.ltrim = true;
    parsed.syntax.rtrim = true;
    if args.len() == 2 {
        if !matches!(args[1].data_type(), DataType::Map(_, _)) {
            return Err(bad_types("to_csv", args));
        }
        let pairs = map_pairs(&args[1])?;
        parsed.apply(&pairs, "to_csv")?;
    }
    let zone: Arc<dyn TimeZoneRules> = Arc::clone(&opts.time_zone);
    let date_pattern = parsed
        .date_pattern
        .clone()
        .or_else(|| DateTimePattern::new("yyyy-MM-dd", Purpose::Format).ok());
    let timestamp_pattern = parsed
        .timestamp_pattern
        .clone()
        .or_else(|| DateTimePattern::new("yyyy-MM-dd'T'HH:mm:ss.SSSXXX", Purpose::Format).ok());
    let ntz_pattern = parsed
        .ntz_pattern
        .clone()
        .or_else(|| DateTimePattern::new("yyyy-MM-dd'T'HH:mm:ss.SSS", Purpose::Format).ok());
    let to_opts = ToCsvOptions {
        syntax: parsed.syntax,
        quote_all: parsed.quote_all,
        escape_quotes: parsed.escape_quotes,
        line_sep: parsed.line_sep,
        null_value: parsed.null_value,
        date_pattern: date_pattern.clone(),
        timestamp_pattern: timestamp_pattern.clone(),
        ntz_pattern: ntz_pattern.clone(),
    };
    Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let col = x.plain_column(0)?;
        let col = col.as_struct();
        let zone_ref: &dyn TimeZoneRules = zone.as_ref();
        let missing = || {
            krishiv_kernels::error::KernelError::with_row(
                None,
                KrishivError::from(InternalError {
                    message: "a pattern for writing".to_owned(),
                }),
            )
        };
        let format_date = |d: i32| match &date_pattern {
            Some(p) => p
                .format_date(d)
                .map_err(|e| krishiv_kernels::error::KernelError::with_row(None, e)),
            None => Err(missing()),
        };
        let format_timestamp = |m: i64| match &timestamp_pattern {
            Some(p) => p
                .format_timestamp(
                    m,
                    krishiv_kernels::datetime_pattern::Zone {
                        id: "UTC",
                        rules: zone_ref,
                    },
                )
                .map_err(|e| krishiv_kernels::error::KernelError::with_row(None, e)),
            None => Err(missing()),
        };
        let format_ntz = |m: i64| match &ntz_pattern {
            Some(p) => p
                .format_local(m)
                .map_err(|e| krishiv_kernels::error::KernelError::with_row(None, e)),
            None => Err(missing()),
        };
        let out = struct_to_csv(
            col,
            &kinds,
            &to_opts,
            &format_date,
            &format_timestamp,
            &format_ntz,
        )
        .map_err(|e| e.error().clone())?;
        Ok(Arc::new(out) as ArrayRef)
    }))
}
