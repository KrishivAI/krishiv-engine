// SPDX-License-Identifier: Apache-2.0
//! `format_string` and `printf`: Java's `java.util.Formatter` over string,
//! boolean, integral, floating-point and decimal arguments.
//!
//! The conversions are `b B s S c C d o x X e E f g G n %`, with argument
//! indexes (`%2$s`, `%<s`), the flags `- # + space 0 , (`, a width and a
//! precision. `h`, `a` and the date/time conversions (`t`) are not
//! implemented. Rounding is Java's `HALF_UP` on the argument's shortest
//! decimal representation. What Java raises as a `java.util.*Exception`
//! (Spark gives it no error class) is `KRISHIV_INVALID_FORMAT_STRING`; an
//! argument index of 0 is `INVALID_PARAMETER_VALUE.ZERO_INDEX`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use krishiv_common::error::{
    InvalidParameterValueZeroIndex, KrishivError, KrishivInvalidFormatString, UnableToAcquireMemory,
};
use krishiv_kernels::text::{java_decimal_string, java_double_string, java_float_string32};

use super::args::{Inputs, is_string, unsupported};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["format_string", "printf"];

/// One argument value.
#[derive(Debug, Clone)]
enum Val {
    Null,
    Str(String),
    Bool(bool),
    /// An integral value and its width in bits (the width matters for the
    /// two's complement of `%x` and `%o`).
    Int(i64, u32),
    F32(f32),
    F64(f64),
    /// `unscaled × 10^-scale`.
    Dec(i128, i8),
}

impl Val {
    /// Java's class name, as `IllegalFormatConversionException` quotes it.
    fn class(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Str(_) => "java.lang.String",
            Self::Bool(_) => "java.lang.Boolean",
            Self::Int(_, 8) => "java.lang.Byte",
            Self::Int(_, 16) => "java.lang.Short",
            Self::Int(_, 32) => "java.lang.Integer",
            Self::Int(..) => "java.lang.Long",
            Self::F32(_) => "java.lang.Float",
            Self::F64(_) => "java.lang.Double",
            // Spark hands its own `Decimal` to the formatter, which only `%s` takes.
            Self::Dec(..) => "org.apache.spark.sql.types.Decimal",
        }
    }
}

/// The widest width or precision, and the longest result of one row, taken
/// before anything is allocated (Java would build strings up to 2 GiB).
const MAX_FIELD: usize = 1 << 24;
const MAX_ROW: usize = 1 << 28;

fn too_big(requested: usize) -> KrishivError {
    UnableToAcquireMemory {
        requested_bytes: requested.to_string(),
        received_bytes: "0".to_owned(),
    }
    .into()
}

fn bad(reason: impl Into<String>) -> KrishivError {
    KrishivInvalidFormatString {
        reason: reason.into(),
    }
    .into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Index {
    Ordinary,
    Explicit(usize),
    Previous,
}

#[derive(Debug, Default, Clone, Copy)]
struct Flags {
    left: bool,
    alt: bool,
    plus: bool,
    space: bool,
    zero: bool,
    group: bool,
    paren: bool,
}

/// A parsed `%` specifier.
#[derive(Debug)]
struct Spec {
    index: Index,
    flags: Flags,
    width: Option<usize>,
    precision: Option<usize>,
    conv: char,
    upper: bool,
    /// The text as written, for error messages.
    text: String,
}

fn digits(chars: &[char], at: &mut usize) -> Option<usize> {
    let start = *at;
    let mut v: usize = 0;
    while let Some(d) = chars.get(*at).and_then(|c| c.to_digit(10)) {
        v = v.saturating_mul(10).saturating_add(d as usize);
        *at += 1;
    }
    (*at > start).then_some(v)
}

/// Parses the specifier after the `%` at `chars[*at - 1]`.
/// Sets a flag; a flag given twice is Java's `DuplicateFormatFlags`.
fn dup(flag: &mut bool) -> Result<(), KrishivError> {
    if std::mem::replace(flag, true) {
        return Err(bad("a flag is given twice"));
    }
    Ok(())
}

fn parse_spec(chars: &[char], at: &mut usize) -> Result<Spec, KrishivError> {
    let start = *at;
    let mut index = Index::Ordinary;
    let mut probe = *at;
    if let Some(n) = digits(chars, &mut probe)
        && chars.get(probe) == Some(&'$')
    {
        if n == 0 {
            return Err(InvalidParameterValueZeroIndex {
                function_name: "`format_string`".to_owned(),
                parameter: "`strfmt`".to_owned(),
            }
            .into());
        }
        index = Index::Explicit(n);
        *at = probe + 1;
    }
    let mut flags = Flags::default();
    loop {
        match chars.get(*at) {
            Some('-') => dup(&mut flags.left)?,
            Some('#') => dup(&mut flags.alt)?,
            Some('+') => dup(&mut flags.plus)?,
            Some(' ') => dup(&mut flags.space)?,
            Some('0') => dup(&mut flags.zero)?,
            Some(',') => dup(&mut flags.group)?,
            Some('(') => dup(&mut flags.paren)?,
            Some('<') => index = Index::Previous,
            _ => break,
        }
        *at += 1;
    }
    let width = digits(chars, &mut *at);
    let precision = if chars.get(*at) == Some(&'.') {
        *at += 1;
        Some(digits(chars, &mut *at).ok_or_else(|| bad("an unknown conversion '.'"))?)
    } else {
        None
    };
    let Some(&conv) = chars.get(*at) else {
        return Err(bad("an unknown conversion '%'"));
    };
    *at += 1;
    let text: String = std::iter::once('%')
        .chain(chars[start..*at].iter().copied())
        .collect();
    let upper = conv.is_ascii_uppercase();
    let lower = conv.to_ascii_lowercase();
    // Java has an upper-case form of only some conversions.
    let known = match lower {
        'b' | 's' | 'c' | 'x' | 'e' | 'g' => true,
        'd' | 'o' | 'f' | 'n' | '%' => !upper,
        'h' | 'a' | 't' => {
            return Err(unsupported(format!("format_string with {text}")));
        }
        _ => false,
    };
    if !known {
        return Err(bad(format!("an unknown conversion '{conv}' in '{text}'")));
    }
    for n in [width, precision].into_iter().flatten() {
        if n > MAX_FIELD {
            return Err(too_big(n));
        }
    }
    if (flags.left || flags.zero) && width.is_none() && lower != 'n' {
        return Err(bad(format!("{text} needs a width")));
    }
    if flags.left && flags.zero {
        return Err(bad(format!("illegal flags in {text}")));
    }
    if flags.plus && flags.space {
        return Err(bad(format!("illegal flags in {text}")));
    }
    let spec = Spec {
        index,
        flags,
        width,
        precision,
        conv: lower,
        upper,
        text,
    };
    check_flags(&spec)?;
    Ok(spec)
}

/// Java's per-conversion rules for flags and precision, checked whatever
/// the argument is (a NULL argument does not skip them).
fn check_flags(spec: &Spec) -> Result<(), KrishivError> {
    let f = spec.flags;
    let first = |names: &[(bool, char)]| names.iter().find(|(on, _)| *on).map(|(_, c)| *c);
    let others = [
        (f.alt, '#'),
        (f.plus, '+'),
        (f.space, ' '),
        (f.zero, '0'),
        (f.group, ','),
        (f.paren, '('),
    ];
    let banned = match spec.conv {
        'n' => {
            if spec.width.is_some() || spec.precision.is_some() {
                return Err(bad(format!("{} takes no width or precision", spec.text)));
            }
            first(&others).or(f.left.then_some('-'))
        }
        '%' | 'c' | 's' | 'b' => {
            if spec.precision.is_some() && matches!(spec.conv, '%' | 'c') {
                return Err(bad(format!("precision does not fit {}", spec.text)));
            }
            first(&others)
        }
        'd' | 'o' | 'x' => {
            if spec.precision.is_some() {
                return Err(bad(format!("precision does not fit {}", spec.text)));
            }
            if spec.conv == 'd' {
                f.alt.then_some('#')
            } else {
                first(&[
                    (f.plus, '+'),
                    (f.space, ' '),
                    (f.group, ','),
                    (f.paren, '('),
                ])
            }
        }
        'e' => f.group.then_some(','),
        'g' => f.alt.then_some('#'),
        _ => None,
    };
    banned.map_or(Ok(()), |c| Err(flags_mismatch(spec, c)))
}

/// The length Java pads by: UTF-16 code units.
fn java_len(s: &str) -> usize {
    s.encode_utf16().count()
}

fn pad(s: String, spec: &Spec) -> String {
    let Some(w) = spec.width else { return s };
    let len = java_len(&s);
    if len >= w {
        return s;
    }
    let fill = " ".repeat(w - len);
    if spec.flags.left {
        s + &fill
    } else {
        fill + &s
    }
}

fn mismatch(spec: &Spec, v: &Val) -> KrishivError {
    bad(format!("{} != {} in {}", spec.conv, v.class(), spec.text))
}

fn flags_mismatch(spec: &Spec, flag: char) -> KrishivError {
    bad(format!("flag '{flag}' does not fit {}", spec.text))
}

/// Inserts `,` between groups of three digits.
fn group(int: &str) -> String {
    let n = int.len();
    let mut out = String::with_capacity(n + n / 3);
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (n - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Sign, grouping and zero padding around the digits of a number.
fn finish_number(neg: bool, mut body: String, spec: &Spec) -> String {
    let f = spec.flags;
    let (mut prefix, mut suffix) = (String::new(), String::new());
    if neg {
        if f.paren {
            prefix.push('(');
            suffix.push(')');
        } else {
            prefix.push('-');
        }
    } else if f.plus {
        prefix.push('+');
    } else if f.space {
        prefix.push(' ');
    }
    if f.zero
        && let Some(w) = spec.width
    {
        let used = java_len(&prefix) + java_len(&body) + java_len(&suffix);
        if used < w {
            body = "0".repeat(w - used) + &body;
        }
    }
    pad(format!("{prefix}{body}{suffix}"), spec)
}

// ---------------------------------------------------------------------------
// Decimal digit strings.
// ---------------------------------------------------------------------------

/// A non-negative decimal `0.DIGITS × 10^dp`, `digits` without leading zeros
/// (empty for zero).
#[derive(Debug, Clone)]
struct Dec {
    digits: Vec<u8>,
    dp: i32,
}

impl Dec {
    fn from_f64(v: f64) -> Self {
        let a = v.abs();
        if a == 0.0 {
            return Self {
                digits: Vec::new(),
                dp: 0,
            };
        }
        let text = format!("{a:e}");
        let (mantissa, exp) = text.split_once('e').unwrap_or((&text, "0"));
        let digits = mantissa
            .bytes()
            .filter(u8::is_ascii_digit)
            .map(|b| b - b'0')
            .collect();
        Self {
            digits,
            dp: exp.parse::<i32>().unwrap_or(0) + 1,
        }
    }

    /// Keeps `keep` digits, rounding half up.
    fn round(&mut self, keep: i32) {
        let Ok(k) = usize::try_from(keep) else {
            // Everything is below the kept position.
            self.digits.clear();
            self.dp = 0;
            return;
        };
        if k >= self.digits.len() {
            return;
        }
        let up = self.digits[k] >= 5;
        self.digits.truncate(k);
        if up {
            let mut i = k;
            loop {
                if i == 0 {
                    self.digits.insert(0, 1);
                    self.dp += 1;
                    break;
                }
                i -= 1;
                if self.digits[i] == 9 {
                    self.digits[i] = 0;
                } else {
                    self.digits[i] += 1;
                    break;
                }
            }
        }
        while self.digits.last() == Some(&0) {
            self.digits.pop();
        }
        if self.digits.is_empty() {
            self.dp = 0;
        }
    }

    fn digit(&self, pos: i32) -> char {
        usize::try_from(pos)
            .ok()
            .and_then(|p| self.digits.get(p))
            .map_or('0', |d| char::from(b'0' + d))
    }

    /// Fixed notation with `prec` fraction digits (already rounded).
    fn fixed(&self, prec: usize, alt: bool) -> (String, String) {
        let int = if self.dp <= 0 {
            "0".to_owned()
        } else {
            (0..self.dp).map(|i| self.digit(i)).collect()
        };
        let frac: String = (0..i32::try_from(prec).unwrap_or(i32::MAX))
            .map(|j| self.digit(self.dp + j))
            .collect();
        let _ = alt;
        (int, frac)
    }
}

fn nan_or_infinity(v: f64, spec: &Spec) -> Option<String> {
    if spec.upper {
        return nan_or_infinity_lower(v, spec).map(|t| t.to_uppercase());
    }
    nan_or_infinity_lower(v, spec)
}

fn nan_or_infinity_lower(v: f64, spec: &Spec) -> Option<String> {
    if v.is_nan() {
        return Some(pad("NaN".to_owned(), spec));
    }
    if v.is_infinite() {
        let neg = v < 0.0;
        let body = "Infinity".to_owned();
        // Java never zero pads a non-finite value.
        let plain = Spec {
            flags: Flags {
                zero: false,
                ..spec.flags
            },
            ..clone_spec(spec)
        };
        return Some(finish_number(neg, body, &plain));
    }
    None
}

fn clone_spec(s: &Spec) -> Spec {
    Spec {
        index: s.index,
        flags: s.flags,
        width: s.width,
        precision: s.precision,
        conv: s.conv,
        upper: s.upper,
        text: s.text.clone(),
    }
}

/// A floating-point argument as a non-negative decimal and its
/// sign, or the text of a non-finite value.
fn as_decimal(v: &Val, spec: &Spec) -> Result<Result<(bool, Dec), String>, KrishivError> {
    Ok(match v {
        Val::F64(x) => match nan_or_infinity(*x, spec) {
            Some(t) => Err(t),
            None => Ok((x.is_sign_negative(), Dec::from_f64(*x))),
        },
        Val::F32(x) => {
            let x = f64::from(*x);
            match nan_or_infinity(x, spec) {
                Some(t) => Err(t),
                None => Ok((x.is_sign_negative(), Dec::from_f64(x))),
            }
        }
        other => return Err(mismatch(spec, other)),
    })
}

fn format_fixed(v: &Val, spec: &Spec) -> Result<String, KrishivError> {
    let (neg, mut d) = match as_decimal(v, spec)? {
        Ok(x) => x,
        Err(text) => return Ok(text),
    };
    let prec = spec.precision.unwrap_or(6);
    d.round(d.dp + i32::try_from(prec).unwrap_or(i32::MAX));
    let (int, frac) = d.fixed(prec, spec.flags.alt);
    let int = if spec.flags.group { group(&int) } else { int };
    let mut body = int;
    if prec > 0 || spec.flags.alt {
        body.push('.');
    }
    body.push_str(&frac);
    Ok(finish_number(neg, body, spec))
}

fn sci_body(d: &Dec, prec: usize, upper: bool, alt: bool) -> String {
    let mut body = String::new();
    body.push(d.digit(0));
    if prec > 0 || alt {
        body.push('.');
    }
    for j in 1..=i32::try_from(prec).unwrap_or(i32::MAX) {
        body.push(d.digit(j));
    }
    let exp = if d.digits.is_empty() { 0 } else { d.dp - 1 };
    body.push(if upper { 'E' } else { 'e' });
    body.push(if exp < 0 { '-' } else { '+' });
    body.push_str(&format!("{:02}", exp.abs()));
    body
}

fn format_sci(v: &Val, spec: &Spec) -> Result<String, KrishivError> {
    let (neg, mut d) = match as_decimal(v, spec)? {
        Ok(x) => x,
        Err(text) => return Ok(text),
    };
    let prec = spec.precision.unwrap_or(6);
    d.round(i32::try_from(prec).unwrap_or(i32::MAX).saturating_add(1));
    let body = sci_body(&d, prec, spec.upper, spec.flags.alt);
    Ok(finish_number(neg, body, spec))
}

fn format_general(v: &Val, spec: &Spec) -> Result<String, KrishivError> {
    let (neg, mut d) = match as_decimal(v, spec)? {
        Ok(x) => x,
        Err(text) => return Ok(text),
    };
    let prec = match spec.precision {
        None => 6,
        Some(0) => 1,
        Some(p) => p,
    };
    let p = i32::try_from(prec).unwrap_or(i32::MAX);
    d.round(p);
    let exp = if d.digits.is_empty() { 0 } else { d.dp - 1 };
    let body = if d.digits.is_empty() || (-4..p).contains(&exp) {
        let frac = usize::try_from((p - (exp + 1)).max(0)).unwrap_or(0);
        let frac = if d.digits.is_empty() { prec - 1 } else { frac };
        let (int, fr) = d.fixed(frac, false);
        let int = if spec.flags.group { group(&int) } else { int };
        if frac > 0 { format!("{int}.{fr}") } else { int }
    } else {
        sci_body(&d, prec - 1, spec.upper, false)
    };
    Ok(finish_number(neg, body, spec))
}

fn format_integer(v: &Val, spec: &Spec) -> Result<String, KrishivError> {
    let Val::Int(n, bits) = v else {
        return Err(mismatch(spec, v));
    };
    let f = spec.flags;
    match spec.conv {
        'd' => {
            let mut body = n.unsigned_abs().to_string();
            if f.group {
                body = group(&body);
            }
            Ok(finish_number(*n < 0, body, spec))
        }
        _ => {
            let mask = if *bits >= 64 {
                u64::MAX
            } else {
                (1u64 << *bits) - 1
            };
            let bits_value = n.cast_unsigned() & mask;
            let (mut digits, prefix) = if spec.conv == 'o' {
                (format!("{bits_value:o}"), "0")
            } else {
                (format!("{bits_value:x}"), "0x")
            };
            let prefix = if f.alt { prefix } else { "" };
            if spec.upper {
                digits = digits.to_uppercase();
            }
            let prefix = if spec.upper {
                prefix.to_uppercase()
            } else {
                prefix.to_owned()
            };
            if f.zero
                && let Some(w) = spec.width
            {
                let used = prefix.len() + digits.len();
                if used < w {
                    digits = "0".repeat(w - used) + &digits;
                }
            }
            Ok(pad(format!("{prefix}{digits}"), spec))
        }
    }
}

fn text_of(v: &Val) -> String {
    match v {
        Val::Null => "null".to_owned(),
        Val::Str(s) => s.clone(),
        Val::Bool(b) => b.to_string(),
        Val::Int(n, _) => n.to_string(),
        Val::F32(x) => java_float_string32(*x),
        Val::F64(x) => java_double_string(*x),
        Val::Dec(u, s) => java_decimal_string(*u, *s),
    }
}

fn truncate(mut s: String, precision: Option<usize>) -> String {
    if let Some(p) = precision
        && let Some((at, _)) = s.char_indices().nth(p)
    {
        s.truncate(at);
    }
    s
}

fn format_one(v: &Val, spec: &Spec) -> Result<String, KrishivError> {
    if matches!(v, Val::Null) && !matches!(spec.conv, 'b' | 's') {
        // Java prints `null` for any other conversion.
        let t = truncate("null".to_owned(), spec.precision);
        return Ok(pad(if spec.upper { t.to_uppercase() } else { t }, spec));
    }
    match spec.conv {
        's' | 'b' => {
            let text = if spec.conv == 'b' {
                match v {
                    Val::Null => "false".to_owned(),
                    Val::Bool(b) => b.to_string(),
                    _ => "true".to_owned(),
                }
            } else {
                text_of(v)
            };
            let t = truncate(text, spec.precision);
            Ok(pad(if spec.upper { t.to_uppercase() } else { t }, spec))
        }
        'c' => {
            let Val::Int(n, 8 | 16 | 32) = v else {
                return Err(mismatch(spec, v));
            };
            let c = u32::try_from(*n)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| bad(format!("{n} is not a code point in {}", spec.text)))?;
            let t = c.to_string();
            Ok(pad(if spec.upper { t.to_uppercase() } else { t }, spec))
        }
        'd' | 'o' | 'x' => format_integer(v, spec),
        'f' => format_fixed(v, spec),
        'e' => format_sci(v, spec),
        _ => format_general(v, spec),
    }
}

/// `format` over `args`.
fn format(fmt: &str, args: &[Val]) -> Result<String, KrishivError> {
    let chars: Vec<char> = fmt.chars().collect();
    let mut out = String::with_capacity(fmt.len() + 16);
    let (mut at, mut next, mut last) = (0usize, 0usize, None::<usize>);
    while at < chars.len() {
        let c = chars[at];
        at += 1;
        if c != '%' {
            out.push(c);
            continue;
        }
        let spec = parse_spec(&chars, &mut at)?;
        match spec.conv {
            'n' => {
                out.push('\n');
                continue;
            }
            '%' => {
                out.push_str(&pad("%".to_owned(), &spec));
                continue;
            }
            _ => {}
        }
        let idx = match spec.index {
            Index::Explicit(n) => n - 1,
            Index::Previous => last.ok_or_else(|| bad(format!("no argument for {}", spec.text)))?,
            Index::Ordinary => {
                next += 1;
                next - 1
            }
        };
        let v = args
            .get(idx)
            .ok_or_else(|| bad(format!("no argument for {}", spec.text)))?;
        last = Some(idx);
        out.push_str(&format_one(v, &spec)?);
        if out.len() > MAX_ROW {
            return Err(too_big(out.len()));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Binding.
// ---------------------------------------------------------------------------

fn supported(t: &DataType) -> bool {
    is_string(t)
        || matches!(
            t,
            DataType::Null
                | DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
                | DataType::Decimal128(..)
        )
}

fn value_at(col: &ArrayRef, i: usize) -> Val {
    if col.is_null(i) {
        return Val::Null;
    }
    match col.data_type() {
        DataType::Utf8 => Val::Str(col.as_string::<i32>().value(i).to_owned()),
        DataType::Boolean => Val::Bool(col.as_boolean().value(i)),
        DataType::Int8 => Val::Int(i64::from(col.as_primitive::<Int8Type>().value(i)), 8),
        DataType::Int16 => Val::Int(i64::from(col.as_primitive::<Int16Type>().value(i)), 16),
        DataType::Int32 => Val::Int(i64::from(col.as_primitive::<Int32Type>().value(i)), 32),
        DataType::Int64 => Val::Int(col.as_primitive::<Int64Type>().value(i), 64),
        DataType::Float32 => Val::F32(col.as_primitive::<Float32Type>().value(i)),
        DataType::Float64 => Val::F64(col.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(_, s) => Val::Dec(
            col.as_primitive::<arrow_array::types::Decimal128Type>()
                .value(i),
            *s,
        ),
        _ => Val::Null,
    }
}

fn bind_format(args: &[BindArg]) -> Result<Binding, KrishivError> {
    if args.is_empty() || !is_string(&args[0].data_type()) {
        return Err(bad_types("format_string", args));
    }
    if !args[1..].iter().all(|a| supported(&a.data_type())) {
        return Err(bad_types("format_string", args));
    }
    let count = args.len();
    Ok(Binding::new(DataType::Utf8, move |x: &Inputs<'_>| {
        let cols: Vec<ArrayRef> = (0..count)
            .map(|i| x.plain_column(i))
            .collect::<Result<_, _>>()?;
        let fmt = cols[0].as_string::<i32>();
        let mut out = arrow_array::builder::StringBuilder::new();
        let mut total = 0usize;
        let mut vals = Vec::with_capacity(count - 1);
        for i in 0..x.rows() {
            if fmt.is_null(i) {
                out.append_null();
                continue;
            }
            vals.clear();
            vals.extend(cols[1..].iter().map(|c| value_at(c, i)));
            let text = format(fmt.value(i), &vals)?;
            // The builder's offsets are i32: report instead of panicking.
            total = total.saturating_add(text.len());
            if total > i32::MAX as usize {
                return Err(UnableToAcquireMemory {
                    requested_bytes: total.to_string(),
                    received_bytes: "0".to_owned(),
                }
                .into());
            }
            out.append_value(text);
        }
        Ok(Arc::new(out.finish()) as ArrayRef)
    }))
}

pub(super) fn bind(name: &str, args: &[BindArg], _opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    Some((listed, bind_format(args)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(fmt: &str, args: &[Val]) -> String {
        format(fmt, args).unwrap_or_else(|e| format!("ERR {e}"))
    }

    fn s(x: &str) -> Val {
        Val::Str(x.to_owned())
    }

    fn i(n: i64) -> Val {
        Val::Int(n, 32)
    }

    #[test]
    fn strings_and_indexes() {
        assert_eq!(f("Hello %s", &[s("World")]), "Hello World");
        assert_eq!(f("Hello %%%%", &[]), "Hello %%");
        assert_eq!(f("%1$s %3$s", &[i(1), i(2), i(3)]), "1 3");
        assert_eq!(
            f("Hello %s %1$s %s", &[s("World"), s("Hello again")]),
            "Hello World World Hello again"
        );
        assert_eq!(f("%s %<s", &[s("a")]), "a a");
        assert_eq!(f(">>%10s<<", &[s("Hello")]), ">>     Hello<<");
        assert_eq!(f(">>%-10s<<", &[s("Hello")]), ">>Hello     <<");
        assert_eq!(f(">>%10s<<", &[Val::Null]), ">>      null<<");
        assert_eq!(
            f("%.2s|%S|%b|%B", &[s("abc"), s("up"), Val::Null, i(1)]),
            "ab|UP|false|TRUE"
        );
        assert_eq!(f("a%nb", &[]), "a\nb");
    }

    #[test]
    fn integers() {
        assert_eq!(f("Hello %x", &[i(20)]), "Hello 14");
        assert_eq!(
            f("%x %X %o %#x", &[i(-1), Val::Int(255, 64), i(8), i(255)]),
            "ffffffff FF 10 0xff"
        );
        assert_eq!(f("%x", &[Val::Int(-1, 8)]), "ff");
        assert_eq!(
            f(
                "%,d|%+d|% d|%(d|%05d",
                &[i(1234567), i(5), i(5), i(-5), i(-42)]
            ),
            "1,234,567|+5| 5|(5)|-0042"
        );
        assert_eq!(f("%c%C", &[i(65), i(98)]), "AB");
    }

    #[test]
    fn floating_point() {
        assert_eq!(f("%f", &[Val::F64(1.5)]), "1.500000");
        assert_eq!(f("%.2f", &[Val::F64(1.005)]), "1.01");
        assert_eq!(f("%.1f", &[Val::F64(0.25)]), "0.3");
        assert_eq!(f("%.0f", &[Val::F64(2.5)]), "3");
        assert_eq!(f("%,.2f", &[Val::F64(1234567.891)]), "1,234,567.89");
        assert_eq!(f("%08.2f", &[Val::F64(-3.15159)]), "-0003.15");
        assert_eq!(f("%e", &[Val::F64(12345.6789)]), "1.234568e+04");
        assert_eq!(f("%.2E", &[Val::F64(0.000123)]), "1.23E-04");
        assert_eq!(f("%g", &[Val::F64(123.456)]), "123.456");
        assert_eq!(f("%g", &[Val::F64(0.00001234)]), "1.23400e-05");
        assert_eq!(
            f("%f %f", &[Val::F64(f64::NAN), Val::F64(f64::NEG_INFINITY)]),
            "NaN -Infinity"
        );
        assert_eq!(f("%s %s", &[Val::F64(1.0), Val::F32(0.1)]), "1.0 0.1");
    }

    #[test]
    fn errors() {
        for (fmt, args) in [
            ("Hello %s", vec![]),
            ("%1$s %4$s", vec![i(1), i(2), i(3)]),
            ("%d", vec![s("x")]),
            ("%f", vec![i(1)]),
            ("%I", vec![s("x")]),
            ("%-s", vec![s("x")]),
            ("%1$", vec![i(1)]),
            ("%.2d", vec![i(1)]),
            ("%#d", vec![i(1)]),
            ("%+x", vec![i(1)]),
            ("%-05d", vec![i(1)]),
            ("%<s", vec![i(1)]),
            ("%c", vec![i(-1)]),
            ("%tY", vec![i(1)]),
            ("%.1f", vec![Val::Dec(15, 1)]),
            ("%D", vec![i(1)]),
            ("%F", vec![Val::F64(1.0)]),
            ("%N", vec![]),
            ("%--5d", vec![i(1)]),
            ("%5n", vec![]),
            ("%.2%", vec![]),
            ("%#s", vec![s("x")]),
            ("%,e", vec![Val::F64(1.0)]),
            ("%#g", vec![Val::F64(1.0)]),
            ("%2c", vec![Val::Int(65, 64)]),
            ("%+c", vec![i(65)]),
            ("%.2d", vec![Val::Null]),
            ("%#d", vec![Val::Null]),
            ("%,x", vec![Val::Null]),
            ("%99999999999999999999s", vec![s("x")]),
            ("%.2147483647f", vec![Val::F64(1.0)]),
            ("%2147483647s", vec![s("x")]),
        ] {
            assert!(f(fmt, &args).starts_with("ERR"), "{fmt}");
        }
        assert!(format("%0$s", &[s("x")]).is_err());
        assert_eq!(
            f("%E %G", &[Val::F64(f64::NAN), Val::F64(f64::INFINITY)]),
            "NAN INFINITY"
        );
        // Many wide fields stay under the row cap.
        let wide = "%16777216s".repeat(20);
        assert!(format(&wide, &[s("x")]).is_err());
    }
}
