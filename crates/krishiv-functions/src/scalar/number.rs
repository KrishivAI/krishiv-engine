// SPDX-License-Identifier: Apache-2.0
//! Number formatting and conversion: `hex`/`unhex` (bytes as upper-case
//! hex and back; invalid input is NULL), `bin` (a long in binary),
//! `conv` (between bases 2 and 36, Hive's `NumberConverter` rules),
//! `format_number` (MySQL-style, or a Java `DecimalFormat` pattern),
//! `to_number`/`try_to_number` (an Oracle pattern to a decimal) and
//! `to_char`/`to_varchar` (a decimal by an Oracle pattern).
//!
//! The analyzer casts every value to what the function takes (numbers for
//! `hex`/`bin`, strings for the rest), so bindings see one shape each.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, BinaryArray, Int64Array, StringArray};
use arrow_schema::DataType;
use krishiv_common::error::KrishivError;
use krishiv_kernels::number_format::convert;
use krishiv_kernels::number_format::decimal_format::{self, Value as DecimalValue};
use krishiv_kernels::number_format::{bin_long, hex_bytes, hex_long, parse_format, unhex_text};

use super::args::{Inputs, internal, is_binary, is_string, to_utf8, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &[
    "hex",
    "unhex",
    "bin",
    "factorial",
    "conv",
    "format_number",
    "to_number",
    "try_to_number",
    "to_char",
    "to_varchar",
];

/// A `StringArray` of owned values.
fn string_array(owned: &[Option<String>]) -> ArrayRef {
    let refs: Vec<Option<&str>> = owned.iter().map(|o| o.as_deref()).collect();
    Arc::new(StringArray::from(refs)) as ArrayRef
}

/// A null-preserving `StringArray` kernel over one string column.
fn strings(
    x: &Inputs<'_>,
    i: usize,
    f: impl Fn(&str) -> Option<String>,
) -> Result<ArrayRef, KrishivError> {
    let a = x.strings(i)?;
    let owned: Vec<Option<String>> = (0..a.len())
        .map(|r| if a.is_null(r) { None } else { f(a.value(r)) })
        .collect();
    Ok(string_array(&owned))
}

/// `hex(x)`: a long in minimal upper-case hex (`0` for zero), or the hex
/// of binary/string bytes.
fn hex(t: &DataType) -> Option<Binding> {
    Some(match t {
        DataType::Int64 => Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
            let a = x.column(0)?.as_primitive::<Int64Type>().clone();
            let owned: Vec<Option<String>> = (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(hex_long(a.value(i)))
                    }
                })
                .collect();
            Ok(string_array(&owned))
        }),
        t if is_binary(t) => Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
            let a = x.binaries(0)?;
            let owned: Vec<Option<String>> = (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(hex_bytes(a.value(i)))
                    }
                })
                .collect();
            Ok(string_array(&owned))
        }),
        t if is_string(t) => Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
            strings(x, 0, |s| Some(hex_bytes(s.as_bytes())))
        }),
        _ => return None,
    })
}

/// `unhex(x)`: hex text to bytes (an odd length pads a leading zero
/// nibble); anything else is NULL.
fn unhex() -> Binding {
    Binding::new(DataType::Binary, move |x: &Inputs<'_>| {
        let a = x.strings(0)?;
        let parts: Vec<Option<Vec<u8>>> = (0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    unhex_text(a.value(i))
                }
            })
            .collect();
        let refs: Vec<Option<&[u8]>> = parts.iter().map(|p| p.as_deref()).collect();
        Ok(Arc::new(BinaryArray::from(refs)) as ArrayRef)
    })
}

/// `bin(x)`: a long in binary (`0` for zero, 64 digits for negatives).
fn bin() -> Binding {
    Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let a = x.column(0)?.as_primitive::<Int64Type>().clone();
        let owned: Vec<Option<String>> = (0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    None
                } else {
                    Some(bin_long(a.value(i)))
                }
            })
            .collect();
        Ok(string_array(&owned))
    })
}

/// `factorial(n)`: `n!` for `0..=20`, NULL otherwise (a long holds 20!).
fn factorial() -> Binding {
    Binding::new(DataType::Int64, move |x: &Inputs<'_>| {
        let a = x.column(0)?.as_primitive::<Int32Type>().clone();
        let out: Int64Array = (0..a.len())
            .map(|i| {
                if a.is_null(i) {
                    return None;
                }
                let n = a.value(i);
                (0..=20)
                    .contains(&n)
                    .then(|| (2..=i64::from(n)).product::<i64>())
            })
            .collect();
        Ok(Arc::new(out) as ArrayRef)
    })
}

/// `conv(num, from_base, to_base)`: Hive's `NumberConverter.convert`.
/// Invalid bases and empty input are NULL; overflow raises in ANSI mode
/// and formats `2^64 - 1` otherwise.
fn conv(ansi: bool) -> Binding {
    Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let (num, from, to) = (
            x.strings(0)?,
            x.array::<Int32Type>(1)?,
            x.array::<Int32Type>(2)?,
        );
        Ok(Arc::new(convert::conv(&num, &from, &to, ansi)?) as ArrayRef)
    })
}

/// The default `format_number(x, d)` pattern for an int `d`.
fn default_pattern(d: i32) -> String {
    let mut pattern = "#,###,###,###,###,###,##0".to_owned();
    if d > 0 {
        pattern.push('.');
        pattern.extend(std::iter::repeat_n('0', usize::try_from(d).unwrap_or(0)));
    }
    pattern
}

/// `format_number(x, d)`: MySQL-style for an int `d` (negative is NULL),
/// else a Java `DecimalFormat` pattern (empty is the default).
fn format_number() -> Binding {
    Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        use arrow_array::cast::AsArray;
        let (values, d) = (x.column(0)?, x.column(1)?);
        let rows = x.rows();
        let at = |len: usize, i: usize| if len == 1 { 0 } else { i };
        let mut out = Vec::with_capacity(rows);
        for i in 0..rows {
            if values.is_null(i) {
                out.push(None);
                continue;
            }
            let pattern = match d.data_type() {
                DataType::Int32 => {
                    let a = d.as_primitive::<Int32Type>();
                    let v = a.value(at(a.len(), i));
                    if a.is_null(at(a.len(), i)) || v < 0 {
                        out.push(None);
                        continue;
                    }
                    default_pattern(v)
                }
                t if is_string(t) => {
                    let a = x.strings(1)?;
                    let v = a.value(at(a.len(), i));
                    if a.is_null(at(a.len(), i)) {
                        out.push(None);
                        continue;
                    }
                    if v.is_empty() {
                        default_pattern(0)
                    } else {
                        v.to_owned()
                    }
                }
                _ => {
                    return Err(internal(
                        "format_number with a pattern that is not an int or string".to_owned(),
                    ));
                }
            };
            let value = match values.data_type() {
                DataType::Int8 => DecimalValue::Int(i64::from(
                    values
                        .as_primitive::<arrow_array::types::Int8Type>()
                        .value(at(values.len(), i)),
                )),
                DataType::Int16 => DecimalValue::Int(i64::from(
                    values
                        .as_primitive::<arrow_array::types::Int16Type>()
                        .value(at(values.len(), i)),
                )),
                DataType::Int32 => DecimalValue::Int(i64::from(
                    values
                        .as_primitive::<Int32Type>()
                        .value(at(values.len(), i)),
                )),
                DataType::Int64 => DecimalValue::Int(
                    values
                        .as_primitive::<Int64Type>()
                        .value(at(values.len(), i)),
                ),
                DataType::Float32 => DecimalValue::Float(f64::from(
                    values
                        .as_primitive::<arrow_array::types::Float32Type>()
                        .value(at(values.len(), i)),
                )),
                DataType::Float64 => DecimalValue::Float(
                    values
                        .as_primitive::<arrow_array::types::Float64Type>()
                        .value(at(values.len(), i)),
                ),
                DataType::Decimal128(_, s) => DecimalValue::Decimal {
                    unscaled: values
                        .as_primitive::<Decimal128Type>()
                        .value(at(values.len(), i)),
                    scale: u32::try_from((*s).max(0)).unwrap_or(0),
                },
                DataType::Null => {
                    out.push(None);
                    continue;
                }
                t => {
                    return Err(internal(format!("format_number of {t}")));
                }
            };
            out.push(Some(
                decimal_format::format(&pattern, value)
                    .map_err(decimal_format::PatternError::into_error)?,
            ));
        }
        Ok(string_array(&out))
    })
}

/// A decimal value as plain text (`78.12`, `-0.5`), as Java's
/// `toPlainString`.
fn decimal_plain(unscaled: i128, scale: u32) -> String {
    let (neg, abs) = if unscaled < 0 {
        (true, unscaled.unsigned_abs())
    } else {
        (false, unscaled.cast_unsigned())
    };
    let digits = abs.to_string();
    let text = if scale == 0 {
        digits
    } else if digits.len() > scale as usize {
        let (i, f) = digits.split_at(digits.len() - scale as usize);
        format!("{i}.{f}")
    } else {
        format!("0.{:0>width$}", digits, width = scale as usize)
    };
    format!("{}{text}", if neg { "-" } else { "" })
}

/// The literal format argument, if it is one (`None` for a NULL format).
fn literal_format(arg: &BindArg) -> Result<Option<String>, KrishivError> {
    let BindArg::Literal(a) = arg else {
        return Err(unsupported(
            "a number format that is not a literal".to_owned(),
        ));
    };
    let a = to_utf8(a.as_ref())?;
    if !a.is_valid(0) {
        return Ok(None);
    }
    Ok(Some(a.value(0).to_owned()))
}

/// `to_number(x, format)` / `try_to_number`: the Oracle pattern to a
/// decimal (a mismatch raises, or is NULL for `try_to_number`). The
/// analyzer only lets literal formats through, so the pattern parses once
/// here; its type is the binding's return type.
fn to_number(name: &'static str, args: &[BindArg], try_: bool) -> Result<Binding, KrishivError> {
    use krishiv_types::DecimalType;
    let format = literal_format(&args[1])?;
    let (precision, scale) = match format.as_deref() {
        None => (
            DecimalType::USER_DEFAULT.precision(),
            DecimalType::USER_DEFAULT.scale(),
        ),
        Some(pattern) => {
            let number = parse_format(pattern);
            number.validate(pattern)?;
            let (p, s) = number.decimal_type();
            DecimalType::try_new(p, s).map(|d| (d.precision(), d.scale()))?
        }
    };
    let out = DataType::Decimal128(precision, i8::try_from(scale).unwrap_or(i8::MAX));
    let scale_i8 = i8::try_from(scale).unwrap_or(i8::MAX);
    Ok(Binding::new(out, move |x: &Inputs<'_>| {
        let values = x.strings(0)?;
        let number = match format.as_deref() {
            None => None,
            Some(pattern) => {
                let number = parse_format(pattern);
                number.validate(pattern)?;
                Some(number)
            }
        };
        let mut unscaled = Vec::with_capacity(x.rows());
        let Some(number) = number.as_ref() else {
            // A NULL format: every row is NULL.
            unscaled.resize(x.rows(), None);
            let Ok(array) = arrow_array::Decimal128Array::from(unscaled)
                .with_precision_and_scale(precision, scale_i8)
            else {
                return Err(internal(format!("{name} decimal type")));
            };
            return Ok(Arc::new(array) as ArrayRef);
        };
        for i in 0..x.rows() {
            let row = if values.len() == 1 { 0 } else { i };
            if values.is_null(row) {
                unscaled.push(None);
                continue;
            }
            let input = values.value(row);
            match number.parse_input(input) {
                Ok(parsed) => match parsed.unscaled(u32::from(scale)) {
                    Ok(v) => unscaled.push(Some(v)),
                    Err(_) if try_ => unscaled.push(None),
                    Err(f) => {
                        return Err(KrishivError::from(f.mismatch(
                            Some(i),
                            "\"STRING\"",
                            input,
                            values.value(row),
                        )));
                    }
                },
                Err(_) if try_ => unscaled.push(None),
                Err(f) => {
                    return Err(KrishivError::from(f.mismatch(
                        Some(i),
                        "\"STRING\"",
                        input,
                        // The format, as written.
                        format.as_deref().unwrap_or_default(),
                    )));
                }
            }
        }
        let Ok(array) = arrow_array::Decimal128Array::from(unscaled)
            .with_precision_and_scale(precision, scale_i8)
        else {
            return Err(internal(format!("{name} decimal type")));
        };
        Ok(Arc::new(array) as ArrayRef)
    }))
}

/// `to_char(x, format)` / `to_varchar`: a decimal by an Oracle pattern (a
/// mismatch raises `INVALID_FORMAT.MISMATCH_INPUT`).
fn to_char(args: &[BindArg]) -> Result<Binding, KrishivError> {
    let format = literal_format(&args[1])?;
    let (precision, scale) = match &args[0].data_type() {
        DataType::Decimal128(p, s) => (*p, *s),
        t => {
            return Err(internal(format!("to_char of {t}")));
        }
    };
    let input_type = format!("\"DECIMAL({precision},{scale})\"");
    Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let (values, formats) = (x.column(0)?, x.strings(1)?);
        let number = match format.as_deref() {
            None => None,
            Some(pattern) => {
                let number = parse_format(pattern);
                number.validate(pattern)?;
                Some(number)
            }
        };
        let mut out = Vec::with_capacity(values.len());
        let Some(number) = number.as_ref() else {
            // A NULL format: every row is NULL.
            out.resize(values.len(), None);
            return Ok(string_array(&out));
        };
        for i in 0..values.len() {
            if values.is_null(i) || formats.is_null(i) {
                out.push(None);
                continue;
            }
            let unscaled = values.as_primitive::<Decimal128Type>().value(i);
            let plain = decimal_plain(unscaled, u32::try_from(scale.max(0)).unwrap_or(0));
            match number.format_decimal(&plain) {
                Ok(text) => out.push(Some(text)),
                Err(f) => {
                    return Err(KrishivError::from(f.mismatch(
                        Some(i),
                        &input_type,
                        &plain,
                        formats.value(i),
                    )));
                }
            }
        }
        Ok(string_array(&out))
    }))
}

/// Binds the number function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let types: Vec<DataType> = args.iter().map(BindArg::data_type).collect();
    let binding = match (listed, types.as_slice()) {
        ("hex", [DataType::Int64]) => match hex(&types[0]) {
            Some(b) => Ok(b),
            None => return Some((listed, Err(bad_types(listed, args)))),
        },
        ("hex", [t]) if is_string(t) || is_binary(t) => match hex(&types[0]) {
            Some(b) => Ok(b),
            None => return Some((listed, Err(bad_types(listed, args)))),
        },
        ("unhex", [t]) if is_string(t) => Ok(unhex()),
        ("bin", [DataType::Int64]) => Ok(bin()),
        ("factorial", [DataType::Int32]) => Ok(factorial()),
        ("conv", [n, DataType::Int32, DataType::Int32]) if is_string(n) => Ok(conv(opts.ansi)),
        ("format_number", [_, DataType::Int32]) => Ok(format_number()),
        ("format_number", [_, t]) if is_string(t) => Ok(format_number()),
        ("to_number", [n, f]) if is_string(n) && is_string(f) => to_number(listed, args, false),
        ("try_to_number", [n, f]) if is_string(n) && is_string(f) => to_number(listed, args, true),
        ("to_char" | "to_varchar", [DataType::Decimal128(..), f]) if is_string(f) => to_char(args),
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, binding))
}
