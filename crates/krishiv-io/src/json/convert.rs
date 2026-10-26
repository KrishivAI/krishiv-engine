// SPDX-License-Identifier: Apache-2.0
//! Values of a flat JSON object as typed columns: Spark's `JacksonParser`
//! value conversions for the flat types, over the CSV column builders.

use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::text::{
    java_parse_decimal, java_parse_double, java_parse_float, java_parse_integral,
};

use super::decode::JVal;
use crate::csv::convert::{Conv, Kind, Val};
use crate::error::unsupported;
use crate::text::{Datetimes, spark_type_name};
use krishiv_common::error::CannotParseJsonField;

/// The Spark type of a column, as the JSON reader handles it: exactly the
/// CSV kinds (flat scalars and binary); anything else is not flat.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the column's type.
pub(crate) fn kind_of(dt: &DataType) -> Result<Kind, KrishivError> {
    Kind::of(dt).map_err(|_| {
        unsupported(format!(
            "the JSON data source with a column of type {} (only flat types are supported)",
            spark_type_name(dt)
        ))
    })
}

/// Converts the fields of records.
pub(crate) struct JsonConverter<'a> {
    pub primitives_as_string: bool,
    pub dt: &'a Datetimes,
}

/// The JSON token type of a value, for conversion errors.
fn token_type(v: &JVal) -> &'static str {
    match v {
        JVal::Null => "NULL",
        JVal::Bool(_) => "BOOLEAN",
        JVal::Num(_, _) => "NUMBER",
        JVal::Str(_) => "STRING",
        JVal::Nested(_) => "OBJECT",
    }
}

impl JsonConverter<'_> {
    /// `v` (absent when the field is missing) as a value of `kind` for the
    /// field `name` of Spark type `target`.
    ///
    /// # Errors
    /// [`Conv::Malformed`] for a value that is not of the type (including a
    /// nested value flat schemas cannot hold: it is a malformed record, not
    /// a failed scan);
    /// [`Conv::Fatal`] for a conversion that cannot be done at all (a
    /// region time zone).
    pub(crate) fn convert(
        &self,
        kind: &Kind,
        name: &str,
        target: &str,
        v: Option<&JVal>,
    ) -> Result<Val, Conv> {
        let Some(v) = v else {
            return Ok(Val::Null);
        };
        if let (Kind::Str, JVal::Nested(raw)) = (kind, v) {
            return Ok(Val::Text(raw.clone()));
        }
        if self.primitives_as_string {
            return match v {
                JVal::Null => Ok(Val::Null),
                JVal::Bool(b) => Ok(Val::Text(b.to_string())),
                JVal::Num(t, _) => Ok(Val::Text(t.clone())),
                JVal::Str(s) => Ok(Val::Text(s.clone())),
                JVal::Nested(_) => Err(Self::nested(name)),
            };
        }
        let bad = || Conv::Malformed;
        Ok(match (kind, v) {
            (_, JVal::Nested(_)) => return Err(Self::nested(name)),
            (_, JVal::Null) | (Kind::Null, _) => Val::Null,
            (Kind::Str, JVal::Str(s)) => Val::Text(s.clone()),
            (Kind::Str, JVal::Bool(b)) => Val::Text(b.to_string()),
            (Kind::Str, JVal::Num(t, _)) => Val::Text(t.clone()),
            (Kind::Bin, JVal::Str(s)) => Val::Bytes(decode_base64(s).ok_or_else(bad)?),
            (Kind::Bool, JVal::Bool(b)) => Val::Bool(*b),
            // Like the CSV converter: only `true`/`false` (any case) read;
            // anything else is a malformed record (Spark's JacksonParser
            // rejects non-boolean strings for BooleanType).
            (Kind::Bool, JVal::Str(s)) => {
                if s.eq_ignore_ascii_case("true") {
                    Val::Bool(true)
                } else if s.eq_ignore_ascii_case("false") {
                    Val::Bool(false)
                } else {
                    return Err(Conv::Malformed);
                }
            }
            // Spark's JacksonParser reads `VALUE_NUMBER_*` for BooleanType
            // as `!= 0` (`getLongValue`/`getDoubleValue`); `NaN != 0` is
            // true in Java, so `NaN` reads `true`.
            (Kind::Bool, JVal::Num(t, _)) => {
                Val::Bool(java_parse_double(t).ok_or_else(bad)? != 0.0)
            }
            (Kind::Int(_), JVal::Bool(b)) => Val::I64(i64::from(*b)),
            (Kind::Int(bits), JVal::Num(t, false)) => {
                Val::I64(java_parse_integral(t, *bits).ok_or_else(bad)?)
            }
            (Kind::F32, JVal::Bool(b)) => Val::F32(if *b { 1.0 } else { 0.0 }),
            (Kind::F32, JVal::Num(t, _)) => Val::F32(java_parse_float(t).ok_or_else(bad)?),
            (Kind::F64, JVal::Bool(b)) => Val::F64(if *b { 1.0 } else { 0.0 }),
            (Kind::F64, JVal::Num(t, _)) => Val::F64(java_parse_double(t).ok_or_else(bad)?),
            (Kind::Decimal(p, sc), JVal::Bool(b)) => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                let v = java_parse_decimal(if *b { "1" } else { "0" }, *p, sc).ok_or_else(bad)?;
                Val::I128(v)
            }
            (Kind::Decimal(p, sc), JVal::Num(t, _)) => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                Val::I128(java_parse_decimal(t, *p, sc).ok_or_else(bad)?)
            }
            (Kind::Int(_), JVal::Str(s)) => {
                Val::I64(java_parse_integral(s, int_bits(kind)).ok_or_else(bad)?)
            }
            (Kind::F32, JVal::Str(s)) => Val::F32(java_parse_float(s).ok_or_else(bad)?),
            (Kind::F64, JVal::Str(s)) => Val::F64(java_parse_double(s).ok_or_else(bad)?),
            (Kind::Decimal(p, sc), JVal::Str(s)) => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                Val::I128(java_parse_decimal(s, *p, sc).ok_or_else(bad)?)
            }
            (Kind::Date, JVal::Str(s)) => {
                Val::I64(i64::from(self.dt.parse_date(s)?.ok_or_else(bad)?))
            }
            (Kind::Timestamp, JVal::Str(s)) => {
                Val::I64(self.dt.parse_timestamp(s)?.ok_or_else(bad)?)
            }
            (Kind::TimestampNtz, JVal::Str(s)) => Val::I64(self.dt.parse_ntz(s)?.ok_or_else(bad)?),
            _ => {
                let e: KrishivError = CannotParseJsonField {
                    field_name: name.to_owned(),
                    field_value: scalar_text(v),
                    json_type: token_type(v).to_owned(),
                    data_type: target.to_owned(),
                }
                .into();
                return Err(Conv::Fatal(e));
            }
        })
    }

    /// A nested value where a flat column was requested: a malformed
    /// record (handled by the parse mode), never a failed scan. Only a
    /// nested *schema* is unsupported at open.
    fn nested(_name: &str) -> Conv {
        Conv::Malformed
    }
}

/// The width of an integer kind.
fn int_bits(kind: &Kind) -> u32 {
    match kind {
        Kind::Int(bits) => *bits,
        _ => 64,
    }
}

/// A scalar as text, for conversion errors.
fn scalar_text(v: &JVal) -> String {
    match v {
        JVal::Null => "null".to_owned(),
        JVal::Bool(b) => b.to_string(),
        JVal::Num(t, _) => t.clone(),
        JVal::Str(s) => s.clone(),
        JVal::Nested(raw) => raw.clone(),
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
