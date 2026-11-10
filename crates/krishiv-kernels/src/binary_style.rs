// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.binaryOutputStyle`: how `to_csv` and `to_xml` print a `BINARY` value.
//!
//! Ground truth: Spark 4.2.0's `ToStringBase.getBinaryFormatter`, pinned with a real run
//! (`select to_csv(named_struct('info', X'4561736F6E'))` under each style). With the config unset,
//! Spark prints `[45 61 73 6F 6E]`, which is `HEX_DISCRETE`; `cast(binary as string)` ignores the
//! setting in every style.

use std::fmt::Write as _;

/// A value of `spark.sql.binaryOutputStyle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinaryStyle {
    /// The bytes as UTF-8 text, each malformed unit replaced by U+FFFD as Java's decoder does
    /// (an encoded surrogate is one unit).
    Utf8,
    /// Signed byte values, `[69, 97]`.
    Basic,
    /// Base64 without padding.
    Base64,
    /// Upper-case hex, `4561`.
    Hex,
    /// Upper-case hex pairs, `[45 61]`: what an unset config prints.
    #[default]
    HexDiscrete,
}

impl BinaryStyle {
    /// The style a config value names (case-insensitive); `None` for anything else.
    #[must_use]
    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_uppercase().as_str() {
            "UTF-8" => Some(Self::Utf8),
            "BASIC" => Some(Self::Basic),
            "BASE64" => Some(Self::Base64),
            "HEX" => Some(Self::Hex),
            "HEX_DISCRETE" => Some(Self::HexDiscrete),
            _ => None,
        }
    }

    /// The config value that names the style.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Utf8 => "UTF-8",
            Self::Basic => "BASIC",
            Self::Base64 => "BASE64",
            Self::Hex => "HEX",
            Self::HexDiscrete => "HEX_DISCRETE",
        }
    }

    /// Appends `bytes` as the style prints them.
    pub fn write(self, bytes: &[u8], out: &mut String) {
        match self {
            Self::Utf8 => crate::string::codec::utf8_replacing(bytes, out),
            Self::Basic => {
                out.push('[');
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}", b.cast_signed());
                }
                out.push(']');
            }
            Self::Base64 => base64_unpadded(bytes, out),
            Self::Hex => {
                for b in bytes {
                    let _ = write!(out, "{b:02X}");
                }
            }
            Self::HexDiscrete => {
                out.push('[');
                for (i, b) in bytes.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    let _ = write!(out, "{b:02X}");
                }
                out.push(']');
            }
        }
    }

    /// The bytes `text` stands for when Spark reads a `BINARY` value back: `from_xml` runs
    /// `ToStringBase.getBinaryParser` (Spark 4.1+) on the element text. `None` is the parser
    /// throwing, a miss (the field is NULL under `PERMISSIVE`, an error under `FAILFAST`).
    ///
    /// The decoders are lenient where Java's are: optional brackets, `Byte.parseByte` signs,
    /// `split(",")` with `trim`, odd-length hex, unpadded base64 and unchecked trailing bits.
    /// Source-derived from Spark v4.2.0 / v4.1.1 (no Spark run yet; see the PR 670 brief).
    #[must_use]
    pub fn parse(self, text: &str) -> Option<Vec<u8>> {
        // Scala `stripPrefix("[").stripSuffix("]")`: each removed independently, at most once.
        let unbracketed = || {
            let t = text.strip_prefix('[').unwrap_or(text);
            t.strip_suffix(']').unwrap_or(t)
        };
        match self {
            Self::Utf8 => Some(text.as_bytes().to_vec()),
            Self::Basic => java_split_comma(unbracketed())
                .into_iter()
                .map(parse_java_byte)
                .collect(),
            Self::Base64 => base64_decode(text),
            Self::Hex => hex_decode(text),
            Self::HexDiscrete => {
                let inner = unbracketed();
                if inner.is_empty() {
                    return Some(Vec::new());
                }
                inner.split(' ').map(hex_pair).collect()
            }
        }
    }

    /// The bytes as the style prints them.
    #[must_use]
    pub fn render(self, bytes: &[u8]) -> String {
        let mut out = String::new();
        self.write(bytes, &mut out);
        out
    }
}

/// Java's `String.split(",")`: no match gives the whole text (even empty), otherwise trailing
/// empty strings are dropped.
fn java_split_comma(text: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = text.split(',').collect();
    if parts.len() > 1 {
        while parts.last().is_some_and(|p| p.is_empty()) {
            parts.pop();
        }
    }
    parts
}

/// `s.trim.toByte`: Java `trim` strips chars up to U+0020; `Byte.parseByte` takes an optional
/// sign and ASCII digits, range -128..=127 (Java also takes other Unicode decimal digits: not
/// matched here, a miss).
fn parse_java_byte(s: &str) -> Option<u8> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    t.strip_prefix('+')
        .unwrap_or(t)
        .parse::<i8>()
        .ok()
        .map(i8::cast_unsigned)
}

/// `Hex.unhex`: hex digits of either case; an odd length reads the first digit as a byte alone.
fn hex_decode(text: &str) -> Option<Vec<u8>> {
    let b = text.as_bytes();
    let digit = |c: u8| {
        char::from(c)
            .to_digit(16)
            .and_then(|d| u8::try_from(d).ok())
    };
    let mut out = Vec::with_capacity(b.len().div_ceil(2));
    let mut rest = b;
    if !b.len().is_multiple_of(2) {
        out.push(digit(*b.first()?)?);
        rest = b.get(1..)?;
    }
    for pair in rest.chunks(2) {
        let hi = digit(*pair.first()?)?;
        let lo = digit(*pair.get(1)?)?;
        out.push(hi.wrapping_shl(4) | lo);
    }
    Some(out)
}

/// One byte of exactly two hex digits (`from_str_radix` would take a leading `+` or one digit).
fn hex_pair(p: &str) -> Option<u8> {
    if p.len() != 2 || !p.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u8::from_str_radix(p, 16).ok()
}

/// The low byte of `v`.
fn low(v: u32) -> Option<u8> {
    u8::try_from(v & 0xff).ok()
}

/// `java.util.Base64.getDecoder().decode`: standard alphabet, padding optional, no whitespace,
/// nothing after the padding, unused trailing bits not checked.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [255_u8; 256];
    for (i, c) in ALPHABET.iter().enumerate() {
        if let Some(slot) = table.get_mut(usize::from(*c)) {
            *slot = u8::try_from(i).ok()?;
        }
    }
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len().div_ceil(4).saturating_mul(3));
    let mut acc = 0_u32;
    let mut n = 0_u32; // characters in the current group of four
    let mut i = 0;
    while let Some(&c) = b.get(i) {
        i = i.saturating_add(1);
        if c == b'=' {
            match n {
                // A lone `=`, or a dangling single character.
                0 | 1 => return None,
                // `xx=` needs a second `=`.
                2 => {
                    if b.get(i) != Some(&b'=') {
                        return None;
                    }
                    i = i.saturating_add(1);
                }
                _ => {}
            }
            // Padding ends the input: anything after it is an error.
            if i != b.len() {
                return None;
            }
            break;
        }
        let v = u32::from(*table.get(usize::from(c))?);
        if v == 255 {
            return None;
        }
        acc = (acc << 6) | v;
        n = n.saturating_add(1);
        if n == 4 {
            out.extend_from_slice(&[low(acc >> 16)?, low(acc >> 8)?, low(acc)?]);
            acc = 0;
            n = 0;
        }
    }
    match n {
        0 => {}
        2 => out.push(low(acc >> 4)?),
        3 => out.extend_from_slice(&[low(acc >> 10)?, low(acc >> 2)?]),
        _ => return None,
    }
    Some(out)
}

/// Standard-alphabet base64 without `=` padding (Java's `Base64.getEncoder().withoutPadding()`).
fn base64_unpadded(bytes: &[u8], out: &mut String) {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for chunk in bytes.chunks(3) {
        let (b0, b1, b2) = (
            u32::from(chunk.first().copied().unwrap_or(0)),
            u32::from(chunk.get(1).copied().unwrap_or(0)),
            u32::from(chunk.get(2).copied().unwrap_or(0)),
        );
        let n = (b0 << 16) | (b1 << 8) | b2;
        // 1, 2 or 3 input bytes give 2, 3 or 4 characters; the rest would be `=` padding.
        let chars = match chunk.len() {
            1 => 2,
            2 => 3,
            _ => 4,
        };
        for shift in [18_u32, 12, 6, 0].into_iter().take(chars) {
            let idx = usize::try_from((n >> shift) & 0x3f).unwrap_or(0);
            out.push(ALPHABET.get(idx).map_or('A', |c| char::from(*c)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reads_back_what_render_prints() {
        for style in [
            BinaryStyle::Utf8,
            BinaryStyle::Basic,
            BinaryStyle::Base64,
            BinaryStyle::Hex,
            BinaryStyle::HexDiscrete,
        ] {
            for bytes in [&b"Eason"[..], b"", b"a", b"ab"] {
                // BASIC prints an empty value `[]`, which Spark reads as a miss.
                if style == BinaryStyle::Basic && bytes.is_empty() {
                    assert_eq!(style.parse(&style.render(bytes)), None);
                    continue;
                }
                assert_eq!(
                    style.parse(&style.render(bytes)).as_deref(),
                    Some(bytes),
                    "{style:?}"
                );
            }
        }
        assert_eq!(BinaryStyle::Hex.parse("4G"), None);
        assert_eq!(BinaryStyle::Base64.parse("A"), None);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: if cfg!(miri) { 2 } else { 256 },
            failure_persistence: None,
            ..proptest::prelude::ProptestConfig::default()
        })]

        /// `parse(render(b)) == b` for every style but BASIC with no bytes (`[]` is a miss).
        #[test]
        fn parse_inverts_render(bytes in proptest::collection::vec(0_u8..=255, 1..40)) {
            for style in [BinaryStyle::Basic, BinaryStyle::Base64, BinaryStyle::Hex, BinaryStyle::HexDiscrete] {
                proptest::prop_assert_eq!(style.parse(&style.render(&bytes)), Some(bytes.clone()));
            }
            // UTF-8 reads back only valid UTF-8 faithfully.
            if let Ok(t) = std::str::from_utf8(&bytes) {
                proptest::prop_assert_eq!(BinaryStyle::Utf8.parse(&BinaryStyle::Utf8.render(&bytes)), Some(t.as_bytes().to_vec()));
            }
        }
    }

    /// Spark 4.1+/4.2 `ToStringBase.getBinaryParser` per style, SOURCE-DERIVED from the
    /// v4.2.0/v4.1.1 sources (brief pr670-binary-readback-semantics.md section 3); no Spark run yet.
    /// `None` is a miss.
    #[test]
    fn parse_is_sparks_lenient_decoder_source_derived() {
        use BinaryStyle::{Base64, Basic, Hex, HexDiscrete, Utf8};
        let cases: &[(BinaryStyle, &str, Option<&[u8]>)] = &[
            (HexDiscrete, "Eason", None),
            (HexDiscrete, "[01 FF]", Some(&[1, 255])),
            (HexDiscrete, "[01 ff]", Some(&[1, 255])),
            (HexDiscrete, "01 FF", Some(&[1, 255])),
            (HexDiscrete, "[]", Some(&[])),
            (HexDiscrete, "01FF", None),
            (HexDiscrete, "[4]", None),
            (Hex, "Eason", None),
            (Hex, "[01 FF]", None),
            (Hex, "01 FF", None),
            (Hex, "[]", None),
            (Hex, "01FF", Some(&[1, 255])),
            (Hex, "01ff", Some(&[1, 255])),
            (Hex, "BAF", Some(&[0x0B, 0xAF])),
            (Hex, "é", None),
            (Basic, "Eason", None),
            (Basic, "[01 FF]", None),
            (Basic, "[]", None),
            (Basic, "01FF", None),
            (Basic, "[4]", Some(&[4])),
            (Basic, "[1,2]", Some(&[1, 2])),
            (Basic, "[1, 2 ]", Some(&[1, 2])),
            (Basic, "1, 2", Some(&[1, 2])),
            (Basic, "[+5]", Some(&[5])),
            (Basic, "[-1]", Some(&[255])),
            (Basic, "[128]", None),
            (Basic, "[255]", None),
            (Basic, "[-129]", None),
            (Basic, "[1,]", Some(&[1])),
            (Basic, "[,1]", None),
            (Basic, "[1,,2]", None),
            (Basic, "[+]", None),
            (Base64, "Eason", None),
            (Base64, "[01 FF]", None),
            (Base64, "[]", None),
            (Base64, "01FF", Some(&[0xD3, 0x51, 0x45])),
            (Base64, "BAF", Some(&[0x04, 0x01])),
            (Base64, "AQ==", Some(&[1])),
            (Base64, "AQ", Some(&[1])),
            (Base64, "AB", Some(&[0])),
            (Base64, "A", None),
            (Base64, "YQ=", None),
            (Base64, "YQ=====", None),
            (Base64, "Y!", None),
            (Base64, "AQ==AQ==", None),
            (Base64, "YQ==", Some(b"a")),
            (Base64, "YWI=", Some(b"ab")),
            (Utf8, "Eason", Some(b"Eason")),
            (Utf8, "[01 FF]", Some(b"[01 FF]")),
        ];
        for (style, text, want) in cases {
            assert_eq!(style.parse(text).as_deref(), *want, "{style:?} {text:?}");
        }
    }

    /// A byte of 0x80 or more under each style, as Spark 4.2.0 prints `X'00FF'`.
    #[test]
    fn high_bytes_read_back() {
        for style in [
            BinaryStyle::Basic,
            BinaryStyle::Base64,
            BinaryStyle::Hex,
            BinaryStyle::HexDiscrete,
        ] {
            let bytes = [0x00, 0x7F, 0x80, 0xFF];
            assert_eq!(
                style.parse(&style.render(&bytes)),
                Some(bytes.to_vec()),
                "{style:?}"
            );
        }
    }

    /// Spark 4.2.0 `to_csv(named_struct('info', X'4561736F6E'))` and `X'00FF'` under each style.
    #[test]
    fn styles_match_spark() {
        let eason = b"Eason";
        assert_eq!(BinaryStyle::Utf8.render(eason), "Eason");
        assert_eq!(BinaryStyle::Basic.render(eason), "[69, 97, 115, 111, 110]");
        assert_eq!(BinaryStyle::Base64.render(eason), "RWFzb24");
        assert_eq!(BinaryStyle::Hex.render(eason), "4561736F6E");
        assert_eq!(BinaryStyle::HexDiscrete.render(eason), "[45 61 73 6F 6E]");
        let edge = [0x00, 0xff];
        assert_eq!(BinaryStyle::Basic.render(&edge), "[0, -1]");
        assert_eq!(BinaryStyle::Base64.render(&edge), "AP8");
        assert_eq!(BinaryStyle::Hex.render(&edge), "00FF");
        assert_eq!(BinaryStyle::HexDiscrete.render(&edge), "[00 FF]");
        assert_eq!(BinaryStyle::default(), BinaryStyle::HexDiscrete);
    }

    /// Spark 4.2.0, UTF-8 style: `to_csv(named_struct('a', X'EDA080'))` is one U+FFFD and
    /// `X'FF41'` is U+FFFD then `A`.
    #[test]
    fn malformed_utf8_is_replaced_as_java_does() {
        assert_eq!(BinaryStyle::Utf8.render(&[0xED, 0xA0, 0x80]), "\u{FFFD}");
        assert_eq!(BinaryStyle::Utf8.render(&[0xFF, 0x41]), "\u{FFFD}A");
        assert_eq!(BinaryStyle::Utf8.render(&[0xC3]), "\u{FFFD}");
    }

    #[test]
    fn base64_lengths_and_empty() {
        assert_eq!(BinaryStyle::Base64.render(b""), "");
        assert_eq!(BinaryStyle::Base64.render(b"a"), "YQ");
        assert_eq!(BinaryStyle::Base64.render(b"ab"), "YWI");
        assert_eq!(BinaryStyle::Base64.render(b"abc"), "YWJj");
        assert_eq!(BinaryStyle::HexDiscrete.render(b""), "[]");
        assert_eq!(BinaryStyle::Basic.render(b""), "[]");
    }

    #[test]
    fn names_round_trip() {
        for s in [
            BinaryStyle::Utf8,
            BinaryStyle::Basic,
            BinaryStyle::Base64,
            BinaryStyle::Hex,
            BinaryStyle::HexDiscrete,
        ] {
            assert_eq!(BinaryStyle::from_name(s.name()), Some(s));
        }
        assert_eq!(
            BinaryStyle::from_name(" hex_discrete "),
            Some(BinaryStyle::HexDiscrete)
        );
        assert_eq!(BinaryStyle::from_name("rot13"), None);
    }
}
