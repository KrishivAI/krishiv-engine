// SPDX-License-Identifier: Apache-2.0
//! `Authorization: Basic` on the Arrow Flight SQL surface (P3-T101 note §3.1, owner item 5,
//! 2026-11-20): the user name and password of RFC 7617, where the password is the caller's bearer
//! token and the user name must be its principal's name. The transport decodes the header
//! (`basic_header`) only for `Surface::FlightSql`, and passes the result to
//! [`super::Credentials::with_basic`]; every other surface keeps refusing it as malformed.

use std::fmt;

use krishiv_common::secret::SecretString;

/// The longest `Basic` value decoded (a password is a bearer token, capped as one is).
const MAX_BASIC_BYTES: usize = 32 * 1024;

/// What an `Authorization` header value is, read as `Basic`.
#[derive(Debug, PartialEq, Eq)]
pub enum BasicHeader {
    /// Another scheme (`Bearer`, ...), or no scheme at all.
    NotBasic,
    /// `Basic`, but not base64 of UTF-8 `user:password` with a non-empty password.
    Malformed,
    /// The user name and password.
    Credentials(BasicCredentials),
}

/// A decoded `Basic` credential. The password is wiped when dropped and never printed.
#[derive(PartialEq, Eq)]
pub struct BasicCredentials {
    user: String,
    password: SecretString,
}

impl BasicCredentials {
    /// The user name (may be empty).
    #[must_use]
    pub fn user(&self) -> &str {
        &self.user
    }

    /// The password: the bearer token it stands for.
    #[must_use]
    pub fn password(&self) -> &str {
        self.password.expose_secret()
    }
}

impl fmt::Debug for BasicCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasicCredentials")
            .field("user", &self.user)
            .field("password", &krishiv_common::secret::REDACTED)
            .finish()
    }
}

/// `value` read as `Basic <base64(user ":" password)>` (RFC 7617; the scheme compared ignoring
/// ASCII case, RFC 9110 §11.1): standard base64, padded or not (the Go ADBC driver sends
/// `RawStdEncoding`). The user name ends at the first `:`; a value without one, with an empty
/// password, or that is not UTF-8 is [`BasicHeader::Malformed`].
#[must_use]
pub fn basic_header(value: &str) -> BasicHeader {
    let Some((scheme, rest)) = value.split_once(' ') else {
        return BasicHeader::NotBasic;
    };
    if !scheme.eq_ignore_ascii_case("Basic") {
        return BasicHeader::NotBasic;
    }
    let encoded = rest.trim_matches(' ');
    if encoded.is_empty() || encoded.len() > MAX_BASIC_BYTES {
        return BasicHeader::Malformed;
    }
    let Some(mut bytes) = decode(encoded) else {
        return BasicHeader::Malformed;
    };
    let parsed = std::str::from_utf8(&bytes).ok().and_then(|text| {
        let (user, password) = text.split_once(':')?;
        (!password.is_empty()).then(|| BasicCredentials {
            user: user.to_owned(),
            password: SecretString::new(password),
        })
    });
    krishiv_common::secret::wipe_bytes(&mut bytes);
    parsed.map_or(BasicHeader::Malformed, BasicHeader::Credentials)
}

/// Standard base64 (RFC 4648 §4), with its `=` padding or without; `None` for a character outside
/// the alphabet, misplaced padding, a length of 1 mod 4, or non-zero trailing bits.
fn decode(s: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        Some(u32::from(v))
    }
    let bytes = s.as_bytes();
    let last = bytes.iter().rposition(|&c| c != b'=')?;
    let unpadded = &bytes[..=last];
    let padding = bytes.len() - unpadded.len();
    // Padding, when present, completes the last group exactly.
    if padding > 2 || (padding > 0 && !bytes.len().is_multiple_of(4)) || unpadded.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(unpadded.len() / 4 * 3 + 2);
    for chunk in unpadded.chunks(4) {
        let mut acc: u32 = 0;
        for &c in chunk {
            acc = (acc << 6) | value(c)?;
        }
        let bits = chunk.len() * 6;
        let bytes_out = bits / 8;
        let spare = bits - bytes_out * 8;
        if acc & ((1 << spare) - 1) != 0 {
            return None;
        }
        let acc = acc >> spare;
        for i in (0..bytes_out).rev() {
            out.push(((acc >> (8 * i)) & 0xff) as u8);
        }
    }
    if padding > 0 && unpadded.len() % 4 + padding != 4 {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn creds(value: &str) -> (String, String) {
        match basic_header(value) {
            BasicHeader::Credentials(c) => (c.user().to_owned(), c.password().to_owned()),
            _ => panic!("{value}: not credentials"),
        }
    }

    #[test]
    fn padded_and_unpadded_decode_alike() {
        // base64("ops:secret-token") = b3BzOnNlY3JldC10b2tlbg== (two padding characters).
        assert_eq!(
            creds("Basic b3BzOnNlY3JldC10b2tlbg=="),
            ("ops".to_owned(), "secret-token".to_owned())
        );
        assert_eq!(
            creds("basic b3BzOnNlY3JldC10b2tlbg"),
            ("ops".to_owned(), "secret-token".to_owned())
        );
        // A password holding `:` keeps it; an empty user name is kept (the check refuses it).
        assert_eq!(creds("Basic YTpiOmM="), ("a".to_owned(), "b:c".to_owned()));
        assert_eq!(creds("Basic OnRvaw=="), (String::new(), "tok".to_owned()));
        // `+` and `/`: base64("u:>>>???") = dTo+Pj4/Pz8=.
        assert_eq!(
            creds("Basic dTo+Pj4/Pz8="),
            ("u".to_owned(), ">>>???".to_owned())
        );
    }

    #[test]
    fn other_schemes_and_bad_values() {
        assert_eq!(basic_header("Bearer abc"), BasicHeader::NotBasic);
        assert_eq!(basic_header("Basic"), BasicHeader::NotBasic);
        for bad in [
            "Basic ",
            "Basic !!!!",
            "Basic b3BzOnNlY3JldC10b2tlbg===", // three `=`
            "Basic b3BzOnNlY3JldC10b2tlbg=",   // padding that does not complete the group
            "Basic b3Bz=OnNl",                 // `=` inside
            "Basic b3BzOnNlY3JldC10b2tlbh==",  // non-zero trailing bits
            "Basic b",                         // 1 mod 4
            "Basic b3Bz",                      // "ops": no `:`
            "Basic b3BzOg==",                  // "ops:": empty password
            "Basic /w==",                      // 0xff: not UTF-8
            "Basic -_-_",                      // base64url, not base64
        ] {
            assert_eq!(basic_header(bad), BasicHeader::Malformed, "{bad}");
        }
        let long = format!("Basic {}", "A".repeat(MAX_BASIC_BYTES + 4));
        assert_eq!(basic_header(&long), BasicHeader::Malformed);
    }

    #[test]
    fn debug_never_shows_the_password() {
        let BasicHeader::Credentials(c) = basic_header("Basic b3BzOnNlY3JldC10b2tlbg==") else {
            panic!("credentials");
        };
        let shown = format!("{c:?}");
        assert!(
            shown.contains("ops") && !shown.contains("secret-token"),
            "Debug shows the user name and never the password"
        );
    }
}
