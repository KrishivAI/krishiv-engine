// SPDX-License-Identifier: Apache-2.0
//! Encodings between strings and binary: `base64`/`unbase64`, `encode`/
//! `decode` (charsets), `to_binary`, the UTF-8 validators and the URL
//! codecs.
//!
//! A `STRING` here is always valid UTF-8 (Arrow's invariant), so the
//! validators only have work to do on `BINARY` input; there the bytes are
//! judged as Spark's `UTF8String` does (the Unicode "maximal subpart"
//! rules, which `std::str` implements).
//!
//! Spark's defaults apply unless told otherwise: `encode`/`decode` take
//! Spark's seven default charsets, and `encode_with`/`decode_with` the two
//! legacy switches through [`CharsetOptions`] (`spark.sql.legacy.javaCharsets`,
//! `spark.sql.legacy.codingErrorAction`); `base64` always
//! writes MIME chunking (Spark's default of
//! `spark.sql.chunkBase64String.enabled`).

use arrow_array::{Array, BooleanArray, GenericBinaryArray, GenericStringArray, OffsetSizeTrait};
use krishiv_common::error::{
    CannotDecodeUrl, ConversionInvalidInput, InvalidParameterValueCharset,
    KrishivInvalidBase64Input, KrishivUnsupportedFeature, MalformedCharacterCoding,
};

use super::charsets::{SINGLE_BYTE, UNMAPPED};
use super::{
    BinOperand, BinSink, Slot, StrOperand, StrSink, all_valid, bool_result, check_offset, reserve,
    row_count,
};
use crate::cast::text::java_digit;
use crate::error::KernelError;
use crate::number_format::unhex_text;

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The value of a base64 symbol.
fn base64_value(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b.wrapping_sub(b'A')),
        b'a'..=b'z' => Some(b.wrapping_sub(b'a').wrapping_add(26)),
        b'0'..=b'9' => Some(b.wrapping_sub(b'0').wrapping_add(52)),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// `base64(bin)`: RFC 4648 with padding. With `chunked` (Spark's default,
/// `spark.sql.chunkBase64String.enabled`) a CRLF follows every 76 symbols,
/// as the MIME encoder writes it.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn base64<O: OffsetSizeTrait>(
    a: &GenericBinaryArray<O>,
    chunked: bool,
) -> Result<GenericStringArray<O>, KernelError> {
    let input = BinOperand::Array(a);
    let slots: [&dyn Slot; 1] = [&input];
    let mut sink = StrSink::<O>::new(a.len(), 0);
    let mut scratch = String::new();
    for i in 0..a.len() {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let bytes = a.value(i);
        let symbols = bytes.len().div_ceil(3).saturating_mul(4);
        let breaks = if chunked {
            (symbols.saturating_sub(1) / 76).saturating_mul(2)
        } else {
            0
        };
        let need = symbols.saturating_add(breaks);
        check_offset::<O>(sink.bytes.saturating_add(need))?;
        scratch.clear();
        scratch
            .try_reserve(need)
            .map_err(|_| KernelError::out_of_memory(need))?;
        let mut on_line = 0_usize;
        for group in bytes.chunks(3) {
            let n = group
                .iter()
                .zip([16_u32, 8, 0])
                .fold(0_u32, |n, (b, shift)| n | (u32::from(*b) << shift));
            // A group of `g` bytes has `g + 1` symbols; the rest is padding.
            let real = group.len().saturating_add(1);
            for (k, shift) in [18_u32, 12, 6, 0].into_iter().enumerate() {
                if chunked && on_line == 76 {
                    scratch.push_str("\r\n");
                    on_line = 0;
                }
                let symbol = if k < real {
                    let index = usize::try_from((n >> shift) & 63).unwrap_or(0);
                    BASE64.get(index).map_or('=', |b| char::from(*b))
                } else {
                    '='
                };
                scratch.push(symbol);
                on_line = on_line.saturating_add(1);
            }
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// Whether Spark's `UnBase64.isValidBase64` accepts `s`: groups of four
/// symbols, a last group of two to four with optional padding that makes it
/// four, whitespace anywhere before the padding.
fn is_strict_base64(s: &str) -> bool {
    let (mut symbols, mut pad) = (0_usize, 0_usize);
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '+' | '/' => {
                if pad != 0 {
                    return false;
                }
                symbols = symbols.saturating_add(1);
            }
            '=' => {
                pad = pad.saturating_add(1);
                if pad > 2 || symbols % 4 < 2 {
                    return false;
                }
            }
            c if is_java_whitespace(c) => {
                if pad != 0 {
                    return false;
                }
            }
            _ => return false,
        }
    }
    if pad > 0 {
        symbols.saturating_add(pad) % 4 == 0
    } else {
        symbols % 4 != 1
    }
}

/// `Character.isWhitespace`: Unicode spaces except the no-break ones, plus
/// the ASCII separators.
fn is_java_whitespace(c: char) -> bool {
    match c {
        '\u{00A0}' | '\u{2007}' | '\u{202F}' => false,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | '\u{1C}'..='\u{1F}' => true,
        c => c.is_whitespace(),
    }
}

/// Java's `Base64.getMimeDecoder().decode`: symbols outside the alphabet
/// are skipped; padding must follow a group of two or three symbols (two
/// need both `=`, directly after each other); nothing may follow the
/// padding but skipped characters. `None` for the input Java rejects.
fn mime_decode(s: &str, out: &mut Vec<u8>) -> Result<(), &'static str> {
    let mut chars = s.chars().map(|c| u8::try_from(c).ok());
    let (mut bits, mut have) = (0_u32, 0_u8);
    while let Some(c) = chars.next() {
        let Some(b) = c else { continue };
        if b == b'=' {
            if have == 0 || (have == 2 && chars.next() != Some(Some(b'='))) {
                return Err("Input byte array has wrong 4-byte ending unit");
            }
            break;
        }
        let Some(v) = base64_value(b) else { continue };
        bits = (bits << 6) | u32::from(v);
        have = have.saturating_add(1);
        if have == 4 {
            out.extend_from_slice(&bits.to_be_bytes()[1..]);
            (bits, have) = (0, 0);
        }
    }
    match have {
        2 => out.push(u8::try_from(bits >> 4).unwrap_or(0)),
        3 => {
            let b = bits >> 2;
            out.push(u8::try_from(b >> 8).unwrap_or(0));
            out.push(u8::try_from(b & 0xFF).unwrap_or(0));
        }
        1 => return Err("Last unit does not have enough valid bits"),
        _ => {}
    }
    if chars.any(|c| c.and_then(base64_value).is_some()) {
        return Err("Input byte array has incorrect ending byte");
    }
    Ok(())
}

/// `unbase64(str)`: Java's MIME decoding. With `strict` (what `to_binary`
/// asks for) the string must first be valid base64 as Spark's `isValidBase64` reads it (groups of
/// four symbols, a last group of two to four with padding that completes it,
/// whitespace before the padding); failing that is `CONVERSION_INVALID_INPUT`.
///
/// # Errors
/// `KRISHIV_INVALID_BASE64_INPUT` for input the MIME decoder rejects (Spark
/// raises `IllegalArgumentException`); `CONVERSION_INVALID_INPUT` for a
/// strict failure; `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit.
pub fn unbase64<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
    strict: bool,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let mut sink = BinSink::<O>::new(a.len(), 0);
    let mut scratch = Vec::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        let s = a.value(i);
        if strict && !is_strict_base64(s) {
            return Err(conversion_invalid(i, s, "BASE64"));
        }
        scratch.clear();
        reserve(
            &mut scratch,
            (s.len() / 4).saturating_mul(3).saturating_add(3),
        )?;
        mime_decode(s, &mut scratch).map_err(|reason| {
            KernelError::with_row(
                Some(i),
                KrishivInvalidBase64Input {
                    reason: reason.to_owned(),
                }
                .into(),
            )
        })?;
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `CONVERSION_INVALID_INPUT` of `to_binary` (the hint names `try_to_binary`).
fn conversion_invalid(row: usize, s: &str, format: &str) -> KernelError {
    KernelError::with_row(
        Some(row),
        ConversionInvalidInput {
            str: format!("'{s}'"),
            fmt: format!("'{format}'"),
            target_type: "\"BINARY\"".to_owned(),
            suggestion: "`try_to_binary`".to_owned(),
        }
        .into(),
    )
}

/// The conversion `to_binary` applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryFormat {
    Hex,
    Utf8,
    Base64,
}

impl BinaryFormat {
    /// The format named by `fmt` (case-insensitive).
    fn parse(fmt: &str) -> Option<Self> {
        if fmt.eq_ignore_ascii_case("hex") {
            Some(Self::Hex)
        } else if fmt.eq_ignore_ascii_case("utf-8") || fmt.eq_ignore_ascii_case("utf8") {
            Some(Self::Utf8)
        } else if fmt.eq_ignore_ascii_case("base64") {
            Some(Self::Base64)
        } else {
            None
        }
    }
}

/// `to_binary(str[, fmt])`: `hex` (the default), `utf-8`/`utf8` or `base64`.
/// With `try_`, input that does not convert and an unknown format give
/// NULL; without it the first is `CONVERSION_INVALID_INPUT` and the second
/// is rejected by the analyzer, so it cannot reach this kernel.
///
/// # Errors
/// As above; `INTERNAL_ERROR` for an unknown format without `try_` or
/// operands of different lengths; `UNABLE_TO_ACQUIRE_MEMORY` if the result
/// does not fit the offset type.
pub fn to_binary<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    fmt: Option<StrOperand<'_, O>>,
    try_: bool,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let default = StrOperand::<O>::Scalar(Some("hex"));
    let fmt = fmt.unwrap_or(default);
    let slots: [&dyn Slot; 2] = [&input, &fmt];
    let rows = row_count(&slots)?;
    let mut sink = BinSink::<O>::new(rows, 0);
    let mut scratch = Vec::new();
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let s = input.at(i);
        let Some(format) = BinaryFormat::parse(fmt.at(i)) else {
            if try_ {
                sink.push_null();
                continue;
            }
            return Err(KernelError::internal(format!(
                "to_binary format {:?} passed the analyzer",
                fmt.at(i)
            )));
        };
        scratch.clear();
        let converted = match format {
            BinaryFormat::Utf8 => {
                scratch.extend_from_slice(s.as_bytes());
                Ok(())
            }
            BinaryFormat::Hex => unhex_text(s)
                .map(|b| scratch = b)
                .ok_or_else(|| conversion_invalid(i, s, "HEX")),
            BinaryFormat::Base64 => {
                if is_strict_base64(s) {
                    mime_decode(s, &mut scratch).map_err(|_| conversion_invalid(i, s, "BASE64"))
                } else {
                    Err(conversion_invalid(i, s, "BASE64"))
                }
            }
        };
        match converted {
            Ok(()) => sink.push(&scratch)?,
            Err(_) if try_ => sink.push_null(),
            Err(e) => return Err(e),
        }
    }
    Ok(sink.finish())
}

// ---- encode, decode ----

/// The charsets `encode`/`decode` take (Spark's default set, matched
/// case-insensitively).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Charset {
    Ascii,
    Latin1,
    Utf8,
    Utf16Be,
    Utf16Le,
    /// Java's `UTF-16`: a big-endian byte order mark on output, a BOM
    /// honoured (default big-endian) on input.
    Utf16,
    /// Java's `UTF-32`: big-endian without a BOM on output, a BOM honoured
    /// on input.
    Utf32,
    /// Java's `UTF-32BE` / `UTF-32LE`: no byte order mark either way.
    Utf32Be,
    Utf32Le,
    /// A single-byte table of [`SINGLE_BYTE`] (legacy Java charsets only).
    Single(usize),
}

/// What the legacy switches change in `encode`/`decode`; the default is
/// Spark's own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CharsetOptions {
    /// `spark.sql.legacy.javaCharsets`: any charset Java knows is accepted
    /// (here: the default seven, `windows-1252` and the ISO-8859 family,
    /// under Java's names and common aliases).
    pub java_charsets: bool,
    /// `spark.sql.legacy.codingErrorAction`: a character the charset cannot
    /// hold encodes as `?` and malformed input decodes to U+FFFD, instead
    /// of `MALFORMED_CHARACTER_CODING`.
    pub replace_malformed: bool,
}

/// The names `encode`/`decode` accept, sorted as Spark's error lists them.
const CHARSET_NAMES: [(&str, Charset); 7] = [
    ("iso-8859-1", Charset::Latin1),
    ("us-ascii", Charset::Ascii),
    ("utf-16", Charset::Utf16),
    ("utf-16be", Charset::Utf16Be),
    ("utf-16le", Charset::Utf16Le),
    ("utf-32", Charset::Utf32),
    ("utf-8", Charset::Utf8),
];

/// `INVALID_PARAMETER_VALUE.CHARSET`.
fn invalid_charset(row: usize, function: &str, name: &str) -> KernelError {
    let names: Vec<&str> = CHARSET_NAMES.iter().map(|(n, _)| *n).collect();
    KernelError::with_row(
        Some(row),
        InvalidParameterValueCharset {
            parameter: "`charset`".to_owned(),
            function_name: format!("`{function}`"),
            charsets: names.join(", "),
            charset: name.to_owned(),
        }
        .into(),
    )
}

fn charset_of(
    row: usize,
    function: &str,
    name: &str,
    opts: CharsetOptions,
) -> Result<Charset, KernelError> {
    if let Some((_, cs)) = CHARSET_NAMES
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
    {
        return Ok(*cs);
    }
    if opts.java_charsets {
        match legacy_charset(name) {
            Ok(Some(cs)) => return Ok(cs),
            Ok(None) => {}
            Err(()) => {
                return Err(KernelError::with_row(
                    Some(row),
                    KrishivUnsupportedFeature {
                        feature: format!("the charset {name}"),
                    }
                    .into(),
                ));
            }
        }
    }
    Err(invalid_charset(row, function, name))
}

/// Java's aliases of the charsets above (lower-case), read from a Spark 4.2 run.
const ALIASES: [(&str, &str); 130] = [
    ("646", "us-ascii"),
    ("8859_1", "iso-8859-1"),
    ("8859_13", "iso-8859-13"),
    ("8859_15", "iso-8859-15"),
    ("8859_2", "iso-8859-2"),
    ("8859_3", "iso-8859-3"),
    ("8859_4", "iso-8859-4"),
    ("8859_5", "iso-8859-5"),
    ("8859_6", "iso-8859-6"),
    ("8859_7", "iso-8859-7"),
    ("8859_8", "iso-8859-8"),
    ("8859_9", "iso-8859-9"),
    ("ansi_x3.4-1968", "us-ascii"),
    ("ansi_x3.4-1986", "us-ascii"),
    ("arabic", "iso-8859-6"),
    ("ascii", "us-ascii"),
    ("ascii7", "us-ascii"),
    ("cp1252", "windows-1252"),
    ("cp367", "us-ascii"),
    ("cp5348", "windows-1252"),
    ("cp819", "iso-8859-1"),
    ("cp912", "iso-8859-2"),
    ("cp920", "iso-8859-9"),
    ("cp921", "iso-8859-13"),
    ("csascii", "us-ascii"),
    ("csisolatin1", "iso-8859-1"),
    ("csisolatin2", "iso-8859-2"),
    ("csisolatin5", "iso-8859-9"),
    ("csisolatinarabic", "iso-8859-6"),
    ("csisolatincyrillic", "iso-8859-5"),
    ("csisolatingreek", "iso-8859-7"),
    ("csisolatinhebrew", "iso-8859-8"),
    ("cyrillic", "iso-8859-5"),
    ("ecma-114", "iso-8859-6"),
    ("ecma-118", "iso-8859-7"),
    ("greek", "iso-8859-7"),
    ("greek8", "iso-8859-7"),
    ("hebrew", "iso-8859-8"),
    ("ibm-1252", "windows-1252"),
    ("ibm-912", "iso-8859-2"),
    ("ibm-920", "iso-8859-9"),
    ("ibm-921", "iso-8859-13"),
    ("ibm1252", "windows-1252"),
    ("ibm367", "us-ascii"),
    ("ibm819", "iso-8859-1"),
    ("ibm912", "iso-8859-2"),
    ("ibm920", "iso-8859-9"),
    ("ibm921", "iso-8859-13"),
    ("iso-ir-101", "iso-8859-2"),
    ("iso-ir-109", "iso-8859-3"),
    ("iso-ir-110", "iso-8859-4"),
    ("iso-ir-126", "iso-8859-7"),
    ("iso-ir-127", "iso-8859-6"),
    ("iso-ir-138", "iso-8859-8"),
    ("iso-ir-144", "iso-8859-5"),
    ("iso-ir-148", "iso-8859-9"),
    ("iso-ir-6", "us-ascii"),
    ("iso646-us", "us-ascii"),
    ("iso8859-1", "iso-8859-1"),
    ("iso8859-13", "iso-8859-13"),
    ("iso8859-15", "iso-8859-15"),
    ("iso8859-2", "iso-8859-2"),
    ("iso8859-3", "iso-8859-3"),
    ("iso8859-4", "iso-8859-4"),
    ("iso8859-5", "iso-8859-5"),
    ("iso8859-6", "iso-8859-6"),
    ("iso8859-7", "iso-8859-7"),
    ("iso8859-8", "iso-8859-8"),
    ("iso8859-9", "iso-8859-9"),
    ("iso8859_1", "iso-8859-1"),
    ("iso8859_11", "iso-8859-11"),
    ("iso8859_13", "iso-8859-13"),
    ("iso8859_15", "iso-8859-15"),
    ("iso8859_16", "iso-8859-16"),
    ("iso8859_2", "iso-8859-2"),
    ("iso8859_3", "iso-8859-3"),
    ("iso8859_4", "iso-8859-4"),
    ("iso8859_5", "iso-8859-5"),
    ("iso8859_6", "iso-8859-6"),
    ("iso8859_7", "iso-8859-7"),
    ("iso8859_8", "iso-8859-8"),
    ("iso8859_9", "iso-8859-9"),
    ("iso_646.irv:1991", "us-ascii"),
    ("iso_8859-1", "iso-8859-1"),
    ("iso_8859-13", "iso-8859-13"),
    ("iso_8859-15", "iso-8859-15"),
    ("iso_8859-16", "iso-8859-16"),
    ("iso_8859-1:1987", "iso-8859-1"),
    ("iso_8859-2", "iso-8859-2"),
    ("iso_8859-2:1987", "iso-8859-2"),
    ("iso_8859-3", "iso-8859-3"),
    ("iso_8859-4", "iso-8859-4"),
    ("iso_8859-5", "iso-8859-5"),
    ("iso_8859-6", "iso-8859-6"),
    ("iso_8859-6:1987", "iso-8859-6"),
    ("iso_8859-7", "iso-8859-7"),
    ("iso_8859-7:1987", "iso-8859-7"),
    ("iso_8859-8", "iso-8859-8"),
    ("iso_8859-9", "iso-8859-9"),
    ("iso_8859_1", "iso-8859-1"),
    ("l1", "iso-8859-1"),
    ("l2", "iso-8859-2"),
    ("l3", "iso-8859-3"),
    ("l4", "iso-8859-4"),
    ("l5", "iso-8859-9"),
    ("l9", "iso-8859-15"),
    ("latin0", "iso-8859-15"),
    ("latin1", "iso-8859-1"),
    ("latin2", "iso-8859-2"),
    ("latin3", "iso-8859-3"),
    ("latin4", "iso-8859-4"),
    ("latin5", "iso-8859-9"),
    ("latin9", "iso-8859-15"),
    ("unicode", "utf-16"),
    ("unicode-1-1-utf-8", "utf-8"),
    ("unicodebig", "utf-16"),
    ("unicodebigunmarked", "utf-16be"),
    ("unicodelittleunmarked", "utf-16le"),
    ("us", "us-ascii"),
    ("utf16", "utf-16"),
    ("utf8", "utf-8"),
    ("utf_16", "utf-16"),
    ("utf_16be", "utf-16be"),
    ("utf_16le", "utf-16le"),
    ("utf_32", "utf-32be"),
    ("utf_32be", "utf-32be"),
    ("utf_32le", "utf-32le"),
    ("x-iso-8859-11", "iso-8859-11"),
    ("x-utf-32be", "utf-32be"),
    ("x-utf-32le", "utf-32le"),
];

/// Java charsets Krishiv has no table for: named, so they fail as unsupported
/// rather than as an unknown charset.
const UNSUPPORTED: [&str; 18] = [
    "big5",
    "euc-jp",
    "euc-kr",
    "gb18030",
    "gb2312",
    "gbk",
    "koi8-r",
    "shift_jis",
    "tis-620",
    "unicodelittle",
    "utf-32be-bom",
    "windows-1250",
    "windows-1251",
    "windows-1253",
    "windows-1254",
    "windows-1255",
    "windows-1256",
    "windows-1257",
];

/// A charset under `spark.sql.legacy.javaCharsets`: Java's canonical name or
/// one of its aliases, case-insensitively.
fn legacy_charset(name: &str) -> Result<Option<Charset>, ()> {
    let lower = name.to_ascii_lowercase();
    let canonical = ALIASES
        .iter()
        .find(|(alias, _)| *alias == lower)
        .map_or(lower.as_str(), |(_, canonical)| *canonical);
    if let Some((_, cs)) = CHARSET_NAMES.iter().find(|(n, _)| *n == canonical) {
        return Ok(Some(*cs));
    }
    match canonical {
        "utf-32be" => return Ok(Some(Charset::Utf32Be)),
        "utf-32le" => return Ok(Some(Charset::Utf32Le)),
        _ => {}
    }
    if let Some(table) = SINGLE_BYTE.iter().position(|t| t.name == canonical) {
        return Ok(Some(Charset::Single(table)));
    }
    if UNSUPPORTED.contains(&canonical) {
        return Err(());
    }
    Ok(None)
}

/// `MALFORMED_CHARACTER_CODING`.
fn malformed(row: usize, function: &str, name: &str) -> KernelError {
    KernelError::with_row(
        Some(row),
        MalformedCharacterCoding {
            function: format!("`{function}`"),
            charset: name.to_owned(),
        }
        .into(),
    )
}

/// `encode(str, charset)`: the bytes of `str` in `charset`.
///
/// # Errors
/// `INVALID_PARAMETER_VALUE.CHARSET` for a charset outside Spark's default
/// set; `MALFORMED_CHARACTER_CODING` for a character the charset cannot
/// hold; `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn encode<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    charset: StrOperand<'_, O>,
) -> Result<GenericBinaryArray<O>, KernelError> {
    encode_with(input, charset, CharsetOptions::default())
}

/// [`encode`] under the legacy switches in `opts`.
///
/// # Errors
/// As [`encode`]; `MALFORMED_CHARACTER_CODING` only without
/// `opts.replace_malformed`.
pub fn encode_with<O: OffsetSizeTrait>(
    input: StrOperand<'_, O>,
    charset: StrOperand<'_, O>,
    opts: CharsetOptions,
) -> Result<GenericBinaryArray<O>, KernelError> {
    let slots: [&dyn Slot; 2] = [&input, &charset];
    let rows = row_count(&slots)?;
    let mut sink = BinSink::<O>::new(rows, 0);
    let mut scratch = Vec::new();
    let mut last: Option<(&str, Charset)> = None;
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (s, name) = (input.at(i), charset.at(i));
        let cs = match last {
            Some((previous, cs)) if previous == name => cs,
            _ => {
                let cs = charset_of(i, "encode", name, opts)?;
                last = Some((name, cs));
                cs
            }
        };
        scratch.clear();
        if !s.is_empty() {
            // UTF-32 writes 4 bytes per character (plus a 2-byte BOM for
            // UTF-16), so `4 * len + 2` bounds every charset.
            reserve(&mut scratch, s.len().saturating_mul(4).saturating_add(2))?;
            encode_into(cs, s, &mut scratch, opts.replace_malformed)
                .map_err(|()| malformed(i, "encode", name))?;
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `s` in `cs`; `Err` for a character the charset cannot hold, unless
/// `replace`, which writes `?` for it.
fn encode_into(cs: Charset, s: &str, out: &mut Vec<u8>, replace: bool) -> Result<(), ()> {
    let byte = |b: Option<u8>| match (b, replace) {
        (Some(b), _) => Ok(b),
        (None, true) => Ok(b'?'),
        (None, false) => Err(()),
    };
    match cs {
        Charset::Utf8 => out.extend_from_slice(s.as_bytes()),
        Charset::Ascii => {
            for c in s.chars() {
                out.push(byte(u8::try_from(u32::from(c)).ok().filter(u8::is_ascii))?);
            }
        }
        Charset::Latin1 => {
            for c in s.chars() {
                out.push(byte(u8::try_from(u32::from(c)).ok())?);
            }
        }
        Charset::Single(table) => {
            let high = &SINGLE_BYTE[table].high;
            for c in s.chars() {
                let found = if c.is_ascii() {
                    u8::try_from(u32::from(c)).ok()
                } else {
                    u16::try_from(u32::from(c))
                        .ok()
                        .and_then(|u| high.iter().position(|h| *h == u && u != UNMAPPED))
                        .and_then(|at| u8::try_from(at.saturating_add(0x80)).ok())
                };
                out.push(byte(found)?);
            }
        }
        Charset::Utf16Be | Charset::Utf16Le | Charset::Utf16 => {
            if cs == Charset::Utf16 {
                out.extend_from_slice(&[0xFE, 0xFF]);
            }
            for unit in s.encode_utf16() {
                out.extend_from_slice(&if cs == Charset::Utf16Le {
                    unit.to_le_bytes()
                } else {
                    unit.to_be_bytes()
                });
            }
        }
        Charset::Utf32 | Charset::Utf32Be => {
            for c in s.chars() {
                out.extend_from_slice(&u32::from(c).to_be_bytes());
            }
        }
        Charset::Utf32Le => {
            for c in s.chars() {
                out.extend_from_slice(&u32::from(c).to_le_bytes());
            }
        }
    }
    Ok(())
}

/// `decode(bin, charset)`: the text of the bytes in `charset`.
///
/// # Errors
/// `INVALID_PARAMETER_VALUE.CHARSET` for a charset outside Spark's default
/// set; `MALFORMED_CHARACTER_CODING` for bytes the charset does not
/// decode; `INTERNAL_ERROR` for operands of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn decode<O: OffsetSizeTrait>(
    input: BinOperand<'_, O>,
    charset: StrOperand<'_, O>,
) -> Result<GenericStringArray<O>, KernelError> {
    decode_with(input, charset, CharsetOptions::default())
}

/// [`decode`] under the legacy switches in `opts`.
///
/// # Errors
/// As [`decode`]; `MALFORMED_CHARACTER_CODING` only without
/// `opts.replace_malformed`.
pub fn decode_with<O: OffsetSizeTrait>(
    input: BinOperand<'_, O>,
    charset: StrOperand<'_, O>,
    opts: CharsetOptions,
) -> Result<GenericStringArray<O>, KernelError> {
    let slots: [&dyn Slot; 2] = [&input, &charset];
    let rows = row_count(&slots)?;
    let mut sink = StrSink::<O>::new(rows, 0);
    let mut scratch = String::new();
    let mut last: Option<(&str, Charset)> = None;
    for i in 0..rows {
        if !all_valid(&slots, i) {
            sink.push_null();
            continue;
        }
        let (bytes, name) = (input.at(i), charset.at(i));
        let cs = match last {
            Some((previous, cs)) if previous == name => cs,
            _ => {
                let cs = charset_of(i, "decode", name, opts)?;
                last = Some((name, cs));
                cs
            }
        };
        scratch.clear();
        // UTF-16/32 expand at most: 2 bytes decode to up to 3 UTF-8 bytes.
        scratch
            .try_reserve(bytes.len().saturating_mul(3).saturating_add(3))
            .map_err(|_| KernelError::out_of_memory(bytes.len().saturating_mul(3)))?;
        decode_into(cs, bytes, &mut scratch, opts.replace_malformed)
            .map_err(|()| malformed(i, "decode", name))?;
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// `bytes` as UTF-8 with each malformed unit replaced by U+FFFD: Java's
/// maximal subparts, except that an encoded surrogate (`ED A0..BF 80..BF`)
/// is one unit.
fn utf8_replacing(mut bytes: &[u8], out: &mut String) {
    while let Err(e) = std::str::from_utf8(bytes) {
        let (valid, bad) = bytes.split_at(e.valid_up_to());
        if let Ok(text) = std::str::from_utf8(valid) {
            out.push_str(text);
        }
        out.push('\u{FFFD}');
        let skip = match bad {
            [0xED, 0xA0..=0xBF, 0x80..=0xBF, ..] => 3,
            _ => e.error_len().unwrap_or(bad.len()),
        };
        bytes = bad.get(skip..).unwrap_or_default();
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        out.push_str(text);
    }
}

/// The text of `bytes` in `cs`; `Err` for bytes that are not valid in it,
/// unless `replace`, which writes U+FFFD for each malformed unit (as Java's
/// replacing decoder does).
fn decode_into(cs: Charset, bytes: &[u8], out: &mut String, replace: bool) -> Result<(), ()> {
    const BAD: char = '\u{FFFD}';
    match cs {
        Charset::Utf8 if replace => utf8_replacing(bytes, out),
        Charset::Utf8 => out.push_str(std::str::from_utf8(bytes).map_err(|_| ())?),
        Charset::Ascii => {
            for b in bytes {
                match (b.is_ascii(), replace) {
                    (true, _) => out.push(char::from(*b)),
                    (false, true) => out.push(BAD),
                    (false, false) => return Err(()),
                }
            }
        }
        Charset::Latin1 => out.extend(bytes.iter().map(|b| char::from(*b))),
        Charset::Single(table) => {
            let high = &SINGLE_BYTE[table].high;
            for b in bytes {
                if b.is_ascii() {
                    out.push(char::from(*b));
                } else {
                    match (high[usize::from(*b & 0x7F)], replace) {
                        (UNMAPPED, true) => out.push(BAD),
                        (UNMAPPED, false) => return Err(()),
                        (code, _) => out.push(char::from_u32(u32::from(code)).ok_or(())?),
                    }
                }
            }
        }
        Charset::Utf16Be | Charset::Utf16Le | Charset::Utf16 => {
            let (mut little, mut body) = (cs == Charset::Utf16Le, bytes);
            if cs == Charset::Utf16 {
                match body {
                    [0xFE, 0xFF, rest @ ..] => body = rest,
                    [0xFF, 0xFE, rest @ ..] => (little, body) = (true, rest),
                    _ => {}
                }
            }
            if body.len() % 2 != 0 && !replace {
                return Err(());
            }
            let mut units = body.as_chunks::<2>().0.iter().map(|pair| {
                if little {
                    u16::from_le_bytes(*pair)
                } else {
                    u16::from_be_bytes(*pair)
                }
            });
            let mut tail_taken = false;
            while let Some(unit) = units.next() {
                match unit {
                    0xD800..=0xDBFF => {
                        // A high surrogate needs a low one next; Java's
                        // replacing decoder drops both units otherwise.
                        match units.next() {
                            Some(low @ 0xDC00..=0xDFFF) => {
                                let pair = char::decode_utf16([unit, low]).next();
                                out.push(pair.and_then(Result::ok).ok_or(())?);
                            }
                            None if replace => {
                                // An odd trailing byte belongs to the same
                                // malformed unit.
                                out.push(BAD);
                                tail_taken = true;
                            }
                            _ if replace => out.push(BAD),
                            _ => return Err(()),
                        }
                    }
                    0xDC00..=0xDFFF if replace => out.push(BAD),
                    0xDC00..=0xDFFF => return Err(()),
                    _ => out.push(char::from_u32(u32::from(unit)).ok_or(())?),
                }
            }
            if body.len() % 2 != 0 && !tail_taken {
                out.push(BAD);
            }
        }
        Charset::Utf32 | Charset::Utf32Be | Charset::Utf32Le => {
            let (mut little, mut body) = (cs == Charset::Utf32Le, bytes);
            if cs == Charset::Utf32 {
                match body {
                    [0, 0, 0xFE, 0xFF, rest @ ..] => body = rest,
                    [0xFF, 0xFE, 0, 0, rest @ ..] => (little, body) = (true, rest),
                    _ => {}
                }
            }
            if body.len() % 4 != 0 && !replace {
                return Err(());
            }
            for quad in body.as_chunks::<4>().0 {
                let code = if little {
                    u32::from_le_bytes(*quad)
                } else {
                    u32::from_be_bytes(*quad)
                };
                match char::from_u32(code) {
                    Some(c) => out.push(c),
                    // A surrogate code point survives Java's decoder and
                    // reaches Spark's UTF-8 conversion as `?`.
                    None if (0xD800..=0xDFFF).contains(&code) => out.push('?'),
                    None if replace => out.push(BAD),
                    None => return Err(()),
                }
            }
            if body.len() % 4 != 0 {
                out.push(BAD);
            }
        }
    }
    Ok(())
}

// ---- UTF-8 validation of binary ----

/// `is_valid_utf8(bin)`.
#[must_use]
pub fn is_valid_utf8<O: OffsetSizeTrait>(a: &GenericBinaryArray<O>) -> BooleanArray {
    let input = BinOperand::Array(a);
    bool_result(a.len(), &[&input], |i| {
        std::str::from_utf8(a.value(i)).is_ok()
    })
}

/// `make_valid_utf8(bin)`: each maximal invalid subpart replaced by U+FFFD.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the offset type.
pub fn make_valid_utf8<O: OffsetSizeTrait>(
    a: &GenericBinaryArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    let mut sink = StrSink::<O>::new(a.len(), a.values().len());
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
        } else {
            sink.push(&String::from_utf8_lossy(a.value(i)))?;
        }
    }
    Ok(sink.finish())
}

/// `validate_utf8(bin)` and, with `try_`, `try_validate_utf8(bin)`: the text,
/// or `INVALID_UTF8_STRING` (NULL with `try_`) for bytes that are not UTF-8.
///
/// # Errors
/// As above; `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit the
/// offset type.
pub fn validate_utf8<O: OffsetSizeTrait>(
    a: &GenericBinaryArray<O>,
    try_: bool,
) -> Result<GenericStringArray<O>, KernelError> {
    let mut sink = StrSink::<O>::new(a.len(), a.values().len());
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        match std::str::from_utf8(a.value(i)) {
            Ok(s) => sink.push(s)?,
            Err(_) if try_ => sink.push_null(),
            Err(_) => return Err(KernelError::invalid_utf8(i, a.value(i))),
        }
    }
    Ok(sink.finish())
}

// ---- url_encode, url_decode ----

/// `url_encode(str)`: Java's `URLEncoder` (`application/x-www-form-urlencoded`,
/// UTF-8): letters, digits and `.-*_` stay, a space is `+`, every other
/// byte `%XX`.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the result does not fit memory or the
/// offset type.
pub fn url_encode<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
) -> Result<GenericStringArray<O>, KernelError> {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut sink = StrSink::<O>::new(a.len(), a.values().len());
    let mut scratch = String::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        let s = a.value(i);
        // Every byte becomes `%XX` (3 bytes) worst case.
        check_offset::<O>(sink.bytes.saturating_add(s.len().saturating_mul(3)))?;
        scratch.clear();
        scratch
            .try_reserve(s.len().saturating_mul(3))
            .map_err(|_| KernelError::out_of_memory(s.len().saturating_mul(3)))?;
        for &b in s.as_bytes() {
            match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'.' | b'-' | b'*' | b'_' => {
                    scratch.push(char::from(b));
                }
                b' ' => scratch.push('+'),
                _ => {
                    scratch.push('%');
                    for nibble in [b >> 4, b & 15] {
                        let digit = HEX.get(usize::from(nibble)).copied().unwrap_or(b'0');
                        scratch.push(char::from(digit));
                    }
                }
            }
        }
        sink.push(&scratch)?;
    }
    Ok(sink.finish())
}

/// The byte `%` + `hi` + `lo` stands for in Java's `URLDecoder`, which reads
/// the two characters with `Integer.parseInt(.., 16)`: a sign is accepted
/// (`%+f` is 15, `%-0` is 0) but a negative value is an error.
fn url_escape_byte(hi: char, lo: char) -> Option<u8> {
    let digit = |c: char| match c {
        '0'..='9' | 'a'..='f' | 'A'..='F' => c.to_digit(16),
        c => java_digit(c).map(u32::from),
    };
    let value = match hi {
        '+' => digit(lo)?,
        '-' => (digit(lo)? == 0).then_some(0)?,
        h => digit(h)?.checked_mul(16)?.checked_add(digit(lo)?)?,
    };
    u8::try_from(value).ok()
}

/// `url_decode(str)` and, with `try_`, `try_url_decode(str)`: Java's
/// `URLDecoder` in UTF-8: `+` is a space, `%XX` runs are decoded as UTF-8
/// with each malformed part replaced by U+FFFD. A malformed or incomplete
/// escape is `CANNOT_DECODE_URL` (NULL with `try_`).
///
/// # Errors
/// `CANNOT_DECODE_URL` as above; `UNABLE_TO_ACQUIRE_MEMORY` if the result
/// does not fit memory or the offset type.
pub fn url_decode<O: OffsetSizeTrait>(
    a: &GenericStringArray<O>,
    try_: bool,
) -> Result<GenericStringArray<O>, KernelError> {
    let mut sink = StrSink::<O>::new(a.len(), a.values().len());
    let mut scratch = String::new();
    let mut run: Vec<u8> = Vec::new();
    for i in 0..a.len() {
        if a.is_null(i) {
            sink.push_null();
            continue;
        }
        let s = a.value(i);
        check_offset::<O>(sink.bytes.saturating_add(s.len()))?;
        scratch.clear();
        scratch
            .try_reserve(s.len())
            .map_err(|_| KernelError::out_of_memory(s.len()))?;
        run.clear();
        reserve(&mut run, s.len())?;
        let mut chars = s.chars().peekable();
        let mut ok = true;
        while let Some(c) = chars.next() {
            if c != '%' {
                scratch.push(if c == '+' { ' ' } else { c });
                continue;
            }
            run.clear();
            let mut next = Some(c);
            while next == Some('%') {
                let (hi, lo) = (chars.next(), chars.next());
                let byte = hi.zip(lo).and_then(|(h, l)| url_escape_byte(h, l));
                let Some(byte) = byte else {
                    ok = false;
                    break;
                };
                run.push(byte);
                next = chars.next_if(|c| *c == '%');
            }
            if !ok {
                break;
            }
            scratch.push_str(&String::from_utf8_lossy(&run));
        }
        if ok {
            sink.push(&scratch)?;
        } else if try_ {
            sink.push_null();
        } else {
            return Err(KernelError::with_row(
                Some(i),
                CannotDecodeUrl {
                    url: a.value(i).to_owned(),
                }
                .into(),
            ));
        }
    }
    Ok(sink.finish())
}
