// SPDX-License-Identifier: Apache-2.0
//! The character-level string functions (`ascii`, `chr`, `left`, `right`,
//! `repeat`, `space`, `reverse`, `initcap`, `translate`, `locate` and its
//! aliases, `substring_index`, `split_part`, `find_in_set`, `overlay`,
//! `levenshtein`, `soundex`, `luhn_check`, `quote`, `elt`,
//! `concat_ws` and the lengths in bits and bytes) on the kernels of
//! `krishiv_kernels::string`.

use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, Int32Array, StringArray};
use arrow_schema::DataType;
use krishiv_kernels::KernelError;
use krishiv_kernels::Operand;
use krishiv_kernels::string;

use super::args::{Bin, Prim, Str};
use super::string::{shape, utf8};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "ascii",
    "bit_length",
    "octet_length",
    "chr",
    "char",
    "left",
    "right",
    "repeat",
    "space",
    "reverse",
    "initcap",
    "soundex",
    "quote",
    "translate",
    "instr",
    "find_in_set",
    "locate",
    "position",
    "substring_index",
    "split_part",
    "overlay",
    "levenshtein",
    "luhn_check",
    "elt",
    "concat_ws",
];

/// Binds the character function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let shape = shape(args);
    let ansi = opts.ansi;
    let int = DataType::Int32;
    let one = Operand::Scalar(Some(1));
    let binding = match (listed, shape.as_str()) {
        ("ascii", "S") => Binding::new(int, |x| {
            Ok(Arc::new(string::ascii(&x.strings(0)?)) as ArrayRef)
        }),
        ("octet_length", "S") => Binding::new(int, |x| {
            Ok(Arc::new(string::byte_length(&x.strings(0)?)) as ArrayRef)
        }),
        ("octet_length", "B") => Binding::new(int, |x| {
            Ok(Arc::new(string::byte_length(&x.binaries(0)?)) as ArrayRef)
        }),
        ("bit_length", "S") => Binding::new(int, |x| {
            Ok(Arc::new(string::bit_length(&x.strings(0)?)) as ArrayRef)
        }),
        ("bit_length", "B") => Binding::new(int, |x| {
            Ok(Arc::new(string::bit_length(&x.binaries(0)?)) as ArrayRef)
        }),
        ("chr" | "char", "L") => Binding::new(DataType::Utf8, |x| {
            Ok(utf8(string::chr(&x.array::<Int64Type>(0)?)?))
        }),
        ("left", "SI") => Binding::new(DataType::Utf8, move |x| {
            let (s, n) = (x.str(0)?, x.int(1)?);
            Ok(utf8(string::substring(
                s.operand(),
                one,
                Some(n.operand()),
            )?))
        }),
        ("left", "BI") => Binding::new(DataType::Binary, move |x| {
            let (s, n) = (x.bin(0)?, x.int(1)?);
            let out = string::substring_binary(s.operand(), one, Some(n.operand()))?;
            Ok(Arc::new(out) as ArrayRef)
        }),
        ("right", "SI") => Binding::new(DataType::Utf8, |x| {
            let (s, n) = (x.str(0)?, x.int(1)?);
            Ok(utf8(string::right(s.operand(), n.operand())?))
        }),
        ("repeat", "SI") => Binding::new(DataType::Utf8, |x| {
            let (s, n) = (x.str(0)?, x.int(1)?);
            Ok(utf8(string::repeat(s.operand(), n.operand())?))
        }),
        ("space", "I") => Binding::new(DataType::Utf8, |x| {
            Ok(utf8(string::space(x.int(0)?.operand())?))
        }),
        ("reverse", "B") => Binding::new(DataType::Binary, |x| {
            Ok(Arc::new(string::reverse_binary(&x.binaries(0)?)?) as ArrayRef)
        }),
        ("reverse" | "initcap" | "soundex" | "quote", "S") => {
            let kernel: fn(&StringArray) -> Result<StringArray, KernelError> = match listed {
                "reverse" => string::reverse,
                "initcap" => string::initcap,
                "soundex" => string::soundex,
                _ => string::quote,
            };
            Binding::new(DataType::Utf8, move |x| Ok(utf8(kernel(&x.strings(0)?)?)))
        }
        ("translate", "SSS") => Binding::new(DataType::Utf8, |x| {
            let (s, from, to) = (x.str(0)?, x.str(1)?, x.str(2)?);
            Ok(utf8(string::translate(
                s.operand(),
                from.operand(),
                to.operand(),
            )?))
        }),
        // `instr(str, substr)` is `locate(substr, str)`.
        ("instr", "SS") => Binding::new(int, move |x| {
            let (s, sub) = (x.str(0)?, x.str(1)?);
            let out = string::locate(sub.operand(), s.operand(), one)?;
            Ok(Arc::new(out) as ArrayRef)
        }),
        ("locate" | "position", "SS" | "SSI") => {
            let with_start = shape.len() == 3;
            Binding::new(int, move |x| {
                let (sub, s) = (x.str(0)?, x.str(1)?);
                let start = with_start.then(|| x.int(2)).transpose()?;
                let start = start.as_ref().map_or(one, Prim::operand);
                let out = string::locate(sub.operand(), s.operand(), start)?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("substring_index", "SSI") => Binding::new(DataType::Utf8, |x| {
            let (s, d, n) = (x.str(0)?, x.str(1)?, x.int(2)?);
            Ok(utf8(string::substring_index(
                s.operand(),
                d.operand(),
                n.operand(),
            )?))
        }),
        ("split_part", "SSI") => Binding::new(DataType::Utf8, |x| {
            let (s, d, n) = (x.str(0)?, x.str(1)?, x.int(2)?);
            Ok(utf8(string::split_part(
                s.operand(),
                d.operand(),
                n.operand(),
            )?))
        }),
        ("find_in_set", "SS") => Binding::new(int, |x| {
            let (w, s) = (x.str(0)?, x.str(1)?);
            Ok(Arc::new(string::find_in_set(w.operand(), s.operand())?) as ArrayRef)
        }),
        ("overlay", "SSI" | "SSII") => {
            let with_len = shape.len() == 4;
            Binding::new(DataType::Utf8, move |x| {
                let (s, r, pos) = (x.str(0)?, x.str(1)?, x.int(2)?);
                let len = with_len.then(|| x.int(3)).transpose()?;
                let len = len.as_ref().map(Prim::operand);
                let out = string::overlay(s.operand(), r.operand(), pos.operand(), len)?;
                Ok(utf8(out))
            })
        }
        ("overlay", "BBI" | "BBII") => {
            let with_len = shape.len() == 4;
            Binding::new(DataType::Binary, move |x| {
                let (s, r, pos) = (x.bin(0)?, x.bin(1)?, x.int(2)?);
                let len = with_len.then(|| x.int(3)).transpose()?;
                let len = len.as_ref().map(Prim::operand);
                let out = string::overlay_binary(s.operand(), r.operand(), pos.operand(), len)?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("levenshtein", "SS" | "SSI") => {
            let with_limit = shape.len() == 3;
            // A NULL literal threshold makes the call NULL (Spark's optimizer
            // propagates it); a NULL read from a column counts as 0.
            let null_limit = matches!(args.get(2), Some(BindArg::Literal(a)) if a.is_null(0));
            Binding::new(int, move |x| {
                if null_limit {
                    return Ok(Arc::new(Int32Array::new_null(x.rows())) as ArrayRef);
                }
                let (l, r) = (x.str(0)?, x.str(1)?);
                let limit = with_limit.then(|| x.int(2)).transpose()?;
                let limit = limit.as_ref().map(Prim::operand);
                let out = string::levenshtein(l.operand(), r.operand(), limit)?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("luhn_check", "S") => Binding::new(DataType::Boolean, |x| {
            Ok(Arc::new(string::luhn_check(&x.strings(0)?)) as ArrayRef)
        }),
        ("elt", s) if s.len() > 1 && s.starts_with('I') && s[1..].chars().all(|c| c == 'S') => {
            let count = shape.len();
            Binding::new(DataType::Utf8, move |x| {
                let index = x.int(0)?;
                let parts: Vec<Str> = (1..count).map(|i| x.str(i)).collect::<Result<_, _>>()?;
                let operands: Vec<_> = parts.iter().map(Str::operand).collect();
                Ok(utf8(string::elt(index.operand(), &operands, ansi)?))
            })
        }
        ("elt", s) if s.len() > 1 && s.starts_with('I') && s[1..].chars().all(|c| c == 'B') => {
            let count = shape.len();
            Binding::new(DataType::Binary, move |x| {
                let index = x.int(0)?;
                let parts: Vec<Bin> = (1..count).map(|i| x.bin(i)).collect::<Result<_, _>>()?;
                let operands: Vec<_> = parts.iter().map(Bin::operand).collect();
                let out = string::elt_binary(index.operand(), &operands, ansi)?;
                Ok(Arc::new(out) as ArrayRef)
            })
        }
        ("concat_ws", s) if !s.is_empty() && s.chars().all(|c| c == 'S') => {
            let count = shape.len();
            Binding::new(DataType::Utf8, move |x| {
                let sep = x.str(0)?;
                let parts: Vec<Str> = (1..count).map(|i| x.str(i)).collect::<Result<_, _>>()?;
                let operands: Vec<_> = parts.iter().map(Str::operand).collect();
                Ok(utf8(string::concat_ws(sep.operand(), &operands)?))
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}
