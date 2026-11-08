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

    /// The bytes as the style prints them.
    #[must_use]
    pub fn render(self, bytes: &[u8]) -> String {
        let mut out = String::new();
        self.write(bytes, &mut out);
        out
    }
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
