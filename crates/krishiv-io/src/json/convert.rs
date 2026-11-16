// SPDX-License-Identifier: Apache-2.0
//! Scalar values of JSON records as typed columns: Spark's `JacksonParser`
//! value conversions for the scalar types, over the CSV column builders
//! (structs, arrays and maps are `super::nested`).

use arrow_schema::Field;
use krishiv_common::error::KrishivError;
use krishiv_kernels::text::{
    java_parse_decimal, java_parse_double, java_parse_float, java_parse_integral,
};

use super::decode::JVal;
use crate::csv::convert::{Conv, Kind, Val, interval_of_text};
use crate::error::unsupported;
use crate::text::{Datetimes, spark_type_name};
use krishiv_common::error::{
    CannotParseJsonField, EmptyJsonFieldValue, MalformedRecordInParsingCannotParseStringAsDatatype,
};

/// The Spark type of a scalar column, as the JSON reader handles it:
/// exactly the CSV kinds (scalars and binary).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the column's type.
pub(crate) fn kind_of(f: &Field) -> Result<Kind, KrishivError> {
    Kind::of_field(f).map_err(|_| {
        unsupported(format!(
            "the JSON data source with a column of type {}",
            spark_type_name(f.data_type())
        ))
    })
}

/// Converts the fields of records.
pub(crate) struct JsonConverter<'a> {
    pub dt: &'a Datetimes,
    /// `allowNonNumericNumbers`: `"NaN"`, `"Infinity"` and the like read as
    /// floating point values.
    pub non_numeric: bool,
}

/// Jackson's token type of a value, for conversion errors.
fn token_type(v: &JVal) -> &'static str {
    match v {
        JVal::Null => "VALUE_NULL",
        JVal::Bool(true) => "VALUE_TRUE",
        JVal::Bool(false) => "VALUE_FALSE",
        JVal::Num(_, false) => "VALUE_NUMBER_INT",
        JVal::Num(_, true) => "VALUE_NUMBER_FLOAT",
        JVal::Str(_) => "VALUE_STRING",
        JVal::Object(..) => "START_OBJECT",
        JVal::Array(..) => "START_ARRAY",
    }
}

impl JsonConverter<'_> {
    /// `v` (absent when the field is missing) as a value of `kind` for the
    /// field `name` of Spark type `target`, in a record whose text is
    /// `record`. Spark's `JacksonParser.makeConverter`: each type reads only
    /// its own JSON tokens (a boolean only `true`/`false`, an integer only an
    /// integral number, a float a number or a non-numeric string, a date,
    /// timestamp, binary or interval only a string), and a string column
    /// reads any value, a non-string one as its exact text.
    ///
    /// # Errors
    /// [`Conv::Malformed`] for a value of the right token that is not a value
    /// of the type; [`Conv::Fatal`] with `CANNOT_PARSE_JSON_FIELD` for the
    /// wrong token (the decoder makes the record malformed), or for a
    /// conversion that cannot be done at all (a region time zone).
    pub(crate) fn convert(
        &self,
        kind: &Kind,
        name: &str,
        target: &str,
        v: Option<&JVal>,
        record: &str,
    ) -> Result<Val, Conv> {
        let Some(v) = v else {
            return Ok(Val::Null);
        };
        let bad = || Conv::Malformed;
        Ok(match (kind, v) {
            (_, JVal::Null) | (Kind::Null, _) => Val::Null,
            (Kind::Str, JVal::Str(s)) => Val::Text(s.clone()),
            (Kind::Str, JVal::Bool(b)) => Val::Text(b.to_string()),
            (Kind::Str, JVal::Num(t, _)) => Val::Text(t.clone()),
            (Kind::Str, JVal::Object(_, span) | JVal::Array(_, span)) => {
                Val::Text(span.text(record).to_owned())
            }
            (Kind::Bin, JVal::Str(s)) => Val::Bytes(decode_base64(s).ok_or_else(bad)?),
            (Kind::Bool, JVal::Bool(b)) => Val::Bool(*b),
            (Kind::Int(bits), JVal::Num(t, false)) => {
                Val::I64(java_parse_integral(t, *bits).ok_or_else(bad)?)
            }
            (Kind::F32, JVal::Num(t, _)) => Val::F32(java_parse_float(t).ok_or_else(bad)?),
            (Kind::F64, JVal::Num(t, _)) => Val::F64(java_parse_double(t).ok_or_else(bad)?),
            (Kind::F32, JVal::Str(s)) if !s.is_empty() => {
                let v = self
                    .non_numeric_value(s)
                    .ok_or_else(|| not_a_float(record, name, s, target))?;
                // NaN or an infinity: the same value as a float.
                Val::F32(if v.is_nan() {
                    f32::NAN
                } else if v > 0.0 {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                })
            }
            (Kind::F64, JVal::Str(s)) if !s.is_empty() => Val::F64(
                self.non_numeric_value(s)
                    .ok_or_else(|| not_a_float(record, name, s, target))?,
            ),
            (Kind::Decimal(p, sc), JVal::Num(t, _)) => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                Val::I128(java_parse_decimal(t, *p, sc).ok_or_else(bad)?)
            }
            (Kind::Decimal(p, sc), JVal::Str(s)) if !s.is_empty() => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                Val::I128(java_parse_decimal(s, *p, sc).ok_or_else(bad)?)
            }
            (Kind::Date, JVal::Str(s)) if !s.is_empty() => {
                Val::I64(i64::from(self.dt.parse_date(s)?.ok_or_else(bad)?))
            }
            (Kind::Timestamp, JVal::Str(s)) if !s.is_empty() => {
                Val::I64(self.dt.parse_timestamp(s)?.ok_or_else(bad)?)
            }
            // An integral number is seconds since the epoch (Spark's
            // `getLongValue * 1000000L` wraps on overflow).
            (Kind::Timestamp, JVal::Num(t, false)) => Val::I64(
                java_parse_integral(t, 64)
                    .ok_or_else(bad)?
                    .wrapping_mul(1_000_000),
            ),
            (Kind::TimestampNtz, JVal::Str(s)) if !s.is_empty() => {
                Val::I64(self.dt.parse_ntz(s)?.ok_or_else(bad)?)
            }
            (Kind::YearMonth(..) | Kind::DayTime(..), JVal::Str(s)) => interval_of_text(kind, s)?,
            // An empty string is not a value of a non-string type.
            (_, JVal::Str(s)) if s.is_empty() => {
                let e: KrishivError = EmptyJsonFieldValue {
                    data_type: format!("\"{}\"", target.to_uppercase()),
                }
                .into();
                return Err(Conv::Fatal(e));
            }
            _ => {
                let e: KrishivError = CannotParseJsonField {
                    field_name: name.to_owned(),
                    field_value: scalar_text(v, record),
                    json_type: token_type(v).to_owned(),
                    data_type: target.to_owned(),
                }
                .into();
                return Err(Conv::Fatal(e));
            }
        })
    }

    /// A non-numeric floating point string (`allowNonNumericNumbers`).
    fn non_numeric_value(&self, s: &str) -> Option<f64> {
        if !self.non_numeric {
            return None;
        }
        match s {
            "NaN" => Some(f64::NAN),
            "+INF" | "+Infinity" | "Infinity" => Some(f64::INFINITY),
            "-INF" | "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        }
    }
}

/// A string that is not a float for a float or double column (Spark's
/// `StringAsDataTypeException`, raised as is in FAILFAST).
fn not_a_float(record: &str, name: &str, s: &str, target: &str) -> Conv {
    Conv::Fatal(
        MalformedRecordInParsingCannotParseStringAsDatatype {
            bad_record: record.to_owned(),
            fail_fast_mode: "FAILFAST".to_owned(),
            field_value: format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
            field_name: format!("`{}`", name.replace('`', "``")),
            target_type: format!("\"{}\"", target.to_uppercase()),
            input_type: "StringType".to_owned(),
        }
        .into(),
    )
}

/// A value as text, for conversion errors.
fn scalar_text(v: &JVal, record: &str) -> String {
    match v {
        JVal::Null => "null".to_owned(),
        JVal::Bool(b) => b.to_string(),
        JVal::Num(t, _) => t.clone(),
        JVal::Str(s) => s.clone(),
        JVal::Object(_, span) | JVal::Array(_, span) => span.text(record).to_owned(),
    }
}

/// The standard base64 alphabet.
///
/// Hand-rolled per ADR 0028 (prefer small own code over a new dependency):
/// standard base64 with padding, exactly as Spark writes binary JSON. The
/// edge cases a vetted implementation buys (padding only at the end, at most
/// two pad chars, length a multiple of four, no white space: JSON strings
/// carry none) are asserted in `super::tests::base64_edge_cases`.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `bytes` in standard base64 with padding, as Spark writes binary JSON.
pub(crate) fn encode_base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let mut n = 0_u32;
        for b in chunk {
            n = (n << 8) | u32::from(*b);
        }
        n <<= 8 * (3 - chunk.len());
        let chars = (chunk.len() + 1).min(4);
        for i in 0..4 {
            if i < chars {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Standard base64 with padding, as Spark reads binary JSON; `None` for
/// anything else (white space included: JSON strings carry none).
pub(crate) fn decode_base64(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(4) {
        return None;
    }
    let chunks: Vec<&[u8]> = s.as_bytes().chunks(4).collect();
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        let mut n = 0_u32;
        let mut pad = 0;
        for b in *chunk {
            let v = match b {
                b'A'..=b'Z' => u32::from(b - b'A'),
                b'a'..=b'z' => u32::from(b - b'a') + 26,
                b'0'..=b'9' => u32::from(b - b'0') + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => {
                    pad += 1;
                    0
                }
                _ => return None,
            };
            // Padding only ends the input.
            if pad > 0 && *b != b'=' {
                return None;
            }
            n = (n << 6) | v;
        }
        if pad > 2 {
            return None;
        }
        // Padding only in the final quantum.
        if pad > 0 && !last {
            return None;
        }
        let bytes = n.to_be_bytes();
        out.extend_from_slice(&bytes[1..4 - pad]);
    }
    Some(out)
}
