// SPDX-License-Identifier: Apache-2.0
//! Encodings between strings and binary: `base64`, `unbase64`, `encode`,
//! `decode` (Spark's default charsets), `to_binary`/`try_to_binary`, the
//! UTF-8 validators and the URL codecs, on `krishiv_kernels::string`.
//!
//! `base64` always writes lines of 76 symbols separated by CRLF, as Spark
//! does by default (`spark.sql.chunkBase64String.enabled`; the switch is not
//! read, chunking is always on); the legacy charset switches
//! (`spark.sql.legacy.javaCharsets`, `spark.sql.legacy.codingErrorAction`)
//! are not read, so `encode`/`decode` always take Spark's default charsets
//! and report unmappable input.
//!
//! The validators take `string` or `binary`: a string is always valid
//! UTF-8 (Arrow's invariant), so only `binary` input can fail.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};
use arrow_schema::DataType;
use krishiv_kernels::string;

use super::args::Str;
use super::string::{shape, utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "base64",
    "unbase64",
    "encode",
    "decode",
    "to_binary",
    "try_to_binary",
    "is_valid_utf8",
    "make_valid_utf8",
    "validate_utf8",
    "try_validate_utf8",
    "url_encode",
    "url_decode",
    "try_url_decode",
];

/// Binds the encoding function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let shape = shape(args);
    let binding = match (listed, shape.as_str()) {
        ("base64", "B") => Binding::new(DataType::Utf8, |x| {
            Ok(utf8(string::base64(&x.binaries(0)?, true)?))
        }),
        ("unbase64", "S") => Binding::new(DataType::Binary, |x| {
            Ok(Arc::new(string::unbase64(&x.strings(0)?, false)?) as ArrayRef)
        }),
        ("encode", "SS") => Binding::new(DataType::Binary, |x| {
            let (s, charset) = (x.str(0)?, x.str(1)?);
            let out = string::encode(s.operand(), charset.operand())?;
            Ok(Arc::new(out) as ArrayRef)
        }),
        ("decode", "BS") => Binding::new(DataType::Utf8, |x| {
            let (b, charset) = (x.bin(0)?, x.str(1)?);
            Ok(utf8(string::decode(b.operand(), charset.operand())?))
        }),
        ("to_binary" | "try_to_binary", "S" | "SS") => {
            let (with_format, try_) = (shape.len() == 2, listed == "try_to_binary");
            Binding::new(DataType::Binary, move |x| {
                let s = x.str(0)?;
                let format = with_format.then(|| x.str(1)).transpose()?;
                let out = string::to_binary(s.operand(), format.as_ref().map(Str::operand), try_)?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("is_valid_utf8", "S") => Binding::new(DataType::Boolean, |x| {
            let a = x.strings(0)?;
            let valid: BooleanArray = a.iter().map(|v| v.map(|_| true)).collect();
            Ok(Arc::new(valid) as ArrayRef)
        }),
        ("is_valid_utf8", "B") => Binding::new(DataType::Boolean, |x| {
            Ok(Arc::new(string::is_valid_utf8(&x.binaries(0)?)) as ArrayRef)
        }),
        ("make_valid_utf8" | "validate_utf8" | "try_validate_utf8", "S") => {
            Binding::new(DataType::Utf8, |x| Ok(utf8(x.strings(0)?)))
        }
        ("make_valid_utf8", "B") => Binding::new(DataType::Utf8, |x| {
            Ok(utf8(string::make_valid_utf8(&x.binaries(0)?)?))
        }),
        ("validate_utf8" | "try_validate_utf8", "B") => {
            let try_ = listed == "try_validate_utf8";
            Binding::new(DataType::Utf8, move |x| {
                Ok(utf8(string::validate_utf8(&x.binaries(0)?, try_)?))
            })
        }
        ("url_encode", "S") => Binding::new(DataType::Utf8, |x| {
            Ok(utf8(string::url_encode(&x.strings(0)?)?))
        }),
        ("url_decode" | "try_url_decode", "S") => {
            let try_ = listed == "try_url_decode";
            Binding::new(DataType::Utf8, move |x| {
                Ok(utf8(string::url_decode(&x.strings(0)?, try_)?))
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
