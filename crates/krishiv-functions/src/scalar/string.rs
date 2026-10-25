// SPDX-License-Identifier: Apache-2.0
//! String functions (P2-T023a kernels): `upper`, `lower`, `length`, the
//! `trim` family, `startswith`/`endswith`/`contains`, `replace`,
//! `lpad`/`rpad` and `substring`. `binary` arguments take the byte-wise
//! overloads where Spark has them (`length`, the matches, the pads,
//! `substring`).

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use krishiv_kernels::string::{self, MatchKind, PadSide, TrimSide};

use super::args::{Bin, Inputs, Prim, Str, is_binary, is_string};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "upper",
    "ucase",
    "lower",
    "lcase",
    "length",
    "char_length",
    "character_length",
    "trim",
    "ltrim",
    "rtrim",
    "btrim",
    "startswith",
    "endswith",
    "contains",
    "replace",
    "mask",
    "lpad",
    "rpad",
    "substring",
    "substr",
];

/// The shape of an argument list: `S` string, `B` binary, `I` an integral
/// up to `int`.
fn shape(args: &[BindArg]) -> String {
    args.iter()
        .map(|a| {
            let t = a.data_type();
            if is_string(&t) {
                'S'
            } else if is_binary(&t) {
                'B'
            } else if matches!(t, DataType::Int8 | DataType::Int16 | DataType::Int32) {
                'I'
            } else {
                '?'
            }
        })
        .collect()
}

fn utf8(a: arrow_array::StringArray) -> ArrayRef {
    Arc::new(a)
}

/// Binds the string function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let shape = shape(args);
    let ctx = opts.kernel;
    let text = DataType::Utf8;
    let binding = match (listed, shape.as_str()) {
        ("upper" | "ucase" | "lower" | "lcase", "S") => {
            let upper = listed.starts_with('u');
            Binding::new(text, move |x: &Inputs<'_>| {
                let a = x.strings(0)?;
                Ok(utf8(if upper {
                    string::upper(&a, &ctx)?
                } else {
                    string::lower(&a, &ctx)?
                }))
            })
        }
        ("length" | "char_length" | "character_length", "S") => {
            Binding::new(DataType::Int32, |x: &Inputs<'_>| {
                Ok(Arc::new(string::char_length(&x.strings(0)?)) as ArrayRef)
            })
        }
        ("length" | "char_length" | "character_length", "B") => {
            Binding::new(DataType::Int32, |x: &Inputs<'_>| {
                Ok(Arc::new(string::byte_length(&x.binaries(0)?)) as ArrayRef)
            })
        }
        ("trim" | "ltrim" | "rtrim" | "btrim", "S" | "SS") => {
            let side = match listed {
                "ltrim" => TrimSide::Leading,
                "rtrim" => TrimSide::Trailing,
                _ => TrimSide::Both,
            };
            // `trim(chars, str)`, `ltrim`, `rtrim` take the characters
            // first; `btrim(str, chars)` last.
            let (src, chars) = match (listed, shape.len()) {
                (_, 1) => (0, None),
                ("btrim", _) => (0, Some(1)),
                _ => (1, Some(0)),
            };
            Binding::new(text, move |x: &Inputs<'_>| {
                let s = x.str(src)?;
                let c = chars.map(|i| x.str(i)).transpose()?;
                Ok(utf8(string::trim(
                    side,
                    s.operand(),
                    c.as_ref().map(Str::operand),
                )?))
            })
        }
        ("startswith" | "endswith" | "contains", "SS" | "BB") => {
            let kind = match listed {
                "startswith" => MatchKind::StartsWith,
                "endswith" => MatchKind::EndsWith,
                _ => MatchKind::Contains,
            };
            let bytes = shape == "BB";
            Binding::new(DataType::Boolean, move |x: &Inputs<'_>| {
                let out = if bytes {
                    let (l, r) = (x.bin(0)?, x.bin(1)?);
                    string::match_binary(kind, l.operand(), r.operand())?
                } else {
                    let (l, r) = (x.str(0)?, x.str(1)?);
                    string::match_str(kind, l.operand(), r.operand())?
                };
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("replace", "SS" | "SSS") => {
            let with = shape.len() == 3;
            Binding::new(text, move |x: &Inputs<'_>| {
                let (s, search) = (x.str(0)?, x.str(1)?);
                let replacement = if with {
                    x.str(2)?
                } else {
                    Str::Scalar(Some(String::new()))
                };
                Ok(utf8(string::replace(
                    s.operand(),
                    search.operand(),
                    replacement.operand(),
                )?))
            })
        }
        ("mask", "SSSSS") => Binding::new(text, |x: &Inputs<'_>| {
            let (s, u, l, d, o) = (x.str(0)?, x.str(1)?, x.str(2)?, x.str(3)?, x.str(4)?);
            Ok(utf8(string::mask(
                s.operand(),
                u.operand(),
                l.operand(),
                d.operand(),
                o.operand(),
            )?))
        }),
        ("lpad" | "rpad", "SI" | "SIS" | "BI" | "BIB") => {
            let side = if listed == "lpad" {
                PadSide::Left
            } else {
                PadSide::Right
            };
            let (bytes, fill) = (shape.starts_with('B'), shape.len() == 3);
            let ty = if bytes { DataType::Binary } else { text };
            Binding::new(ty, move |x: &Inputs<'_>| {
                let len = x.int(1)?;
                if bytes {
                    let (s, f) = (x.bin(0)?, fill.then(|| x.bin(2)).transpose()?);
                    let out = string::pad_binary(
                        side,
                        s.operand(),
                        len.operand(),
                        f.as_ref().map(Bin::operand),
                    )?;
                    return Ok(Arc::new(out) as ArrayRef);
                }
                let (s, f) = (x.str(0)?, fill.then(|| x.str(2)).transpose()?);
                Ok(utf8(string::pad(
                    side,
                    s.operand(),
                    len.operand(),
                    f.as_ref().map(Str::operand),
                )?))
            })
        }
        ("substring" | "substr", "SI" | "SII" | "BI" | "BII") => {
            let (bytes, with_len) = (shape.starts_with('B'), shape.len() == 3);
            let ty = if bytes { DataType::Binary } else { text };
            Binding::new(ty, move |x: &Inputs<'_>| {
                let pos = x.int(1)?;
                let len = with_len.then(|| x.int(2)).transpose()?;
                let len = len.as_ref().map(Prim::operand);
                if bytes {
                    let s = x.bin(0)?;
                    let out = string::substring_binary(s.operand(), pos.operand(), len)?;
                    return Ok(Arc::new(out) as ArrayRef);
                }
                let s = x.str(0)?;
                Ok(utf8(string::substring(s.operand(), pos.operand(), len)?))
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
