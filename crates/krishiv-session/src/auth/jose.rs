// SPDX-License-Identifier: Apache-2.0
//! JWS compact tokens and JWK sets, parsed but not verified (design §3.3, steps 1, 2, 4 and 6).
//!
//! Everything here is a pure function of bytes: no clock, no network, no cryptography. The
//! signature check and the key cache are in `oidc.rs`. The fuzz target `jwt_parse` includes
//! this file by path, so it uses only `std`, `serde` and `serde_json`.
//!
//! - Tokens are capped at [`MAX_TOKEN_BYTES`]; base64url is the unpadded, canonical alphabet of
//!   RFC 7515 §2 (padding, other characters and non-zero trailing bits are refused).
//! - A JSON object that names a member twice is refused (RFC 7515 §5.2 step 4, RFC 7519 §4):
//!   otherwise the header's `alg` or a claim could mean two things.
//! - `alg` is one of RS256, RS384, RS512, ES256 and ES384. `none` and every HMAC algorithm are
//!   not parsed at all, so no configuration can accept them. `crit` is refused. The header's
//!   key-locating members (`jku`, `jwk`, `x5u`, `x5c`) are never used: keys come only from the
//!   configured JWK set.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;

/// The largest token parsed (design §3.3 step 1).
pub(crate) const MAX_TOKEN_BYTES: usize = 16 * 1024;
/// The largest JWK set or discovery document read.
pub(crate) const MAX_DOCUMENT_BYTES: usize = 1024 * 1024;
/// The most keys a JWK set may hold.
pub(crate) const MAX_JWKS_KEYS: usize = 64;
/// The longest principal name taken from a claim.
pub(crate) const MAX_PRINCIPAL_BYTES: usize = 256;
/// The most group values taken from the role claim.
pub(crate) const MAX_GROUPS: usize = 256;

/// A JWS signature algorithm Krishiv verifies (RFC 7518 §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Alg {
    /// RSASSA-PKCS1-v1_5 with SHA-256.
    Rs256,
    /// RSASSA-PKCS1-v1_5 with SHA-384.
    Rs384,
    /// RSASSA-PKCS1-v1_5 with SHA-512.
    Rs512,
    /// ECDSA on P-256 with SHA-256.
    Es256,
    /// ECDSA on P-384 with SHA-384.
    Es384,
}

impl Alg {
    /// Every algorithm.
    pub const ALL: [Self; 5] = [
        Self::Rs256,
        Self::Rs384,
        Self::Rs512,
        Self::Es256,
        Self::Es384,
    ];

    /// The JOSE name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rs256 => "RS256",
            Self::Rs384 => "RS384",
            Self::Rs512 => "RS512",
            Self::Es256 => "ES256",
            Self::Es384 => "ES384",
        }
    }

    /// The algorithm named exactly `s` (JOSE names are case-sensitive).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.as_str() == s)
    }

    /// The curve an ECDSA algorithm needs; `None` for RSA.
    #[must_use]
    pub(crate) fn curve(self) -> Option<Curve> {
        match self {
            Self::Es256 => Some(Curve::P256),
            Self::Es384 => Some(Curve::P384),
            Self::Rs256 | Self::Rs384 | Self::Rs512 => None,
        }
    }
}

/// An elliptic curve of an EC key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum Curve {
    /// NIST P-256.
    P256,
    /// NIST P-384.
    P384,
}

impl Curve {
    /// The byte length of one coordinate.
    #[must_use]
    pub(crate) fn coordinate_len(self) -> usize {
        match self {
            Self::P256 => 32,
            Self::P384 => 48,
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "P-256" => Some(Self::P256),
            "P-384" => Some(Self::P384),
            _ => None,
        }
    }
}

/// Why a token or document was refused. Never carries token content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum JoseError {
    /// Refused, with a fixed reason for tests and logs.
    Invalid(&'static str),
    /// The token's `exp` has passed.
    Expired,
}

impl fmt::Display for JoseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(why) => f.write_str(why),
            Self::Expired => f.write_str("expired"),
        }
    }
}

/// A JSON object whose members are all distinct.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Object(pub BTreeMap<String, Value>);

impl<'de> serde::Deserialize<'de> for Object {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Object;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Object, A::Error> {
                let mut out = BTreeMap::new();
                while let Some(k) = map.next_key::<String>()? {
                    let v: Value = map.next_value()?;
                    if out.insert(k, v).is_some() {
                        return Err(de::Error::custom("duplicate member"));
                    }
                }
                Ok(Object(out))
            }
        }
        d.deserialize_map(V)
    }
}

impl Object {
    fn parse(bytes: &[u8]) -> Result<Self, JoseError> {
        serde_json::from_slice(bytes).map_err(|_| JoseError::Invalid("not a JSON object"))
    }

    /// The string member `name`; `Ok(None)` when absent, an error when not a string.
    fn opt_str(&self, name: &str) -> Result<Option<&str>, JoseError> {
        match self.0.get(name) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(JoseError::Invalid("member is not a string")),
        }
    }
}

/// A token's protected header.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct Header {
    /// The signature algorithm.
    pub(crate) alg: Alg,
    /// The key id, if given.
    pub(crate) kid: Option<String>,
}

/// A JWS compact token, split and decoded; its signature is not checked yet.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct Jws<'a> {
    /// The protected header.
    pub(crate) header: Header,
    /// The claims set.
    pub(crate) claims: Object,
    /// `base64url(header) '.' base64url(payload)`: the bytes the signature covers.
    pub(crate) signing_input: &'a [u8],
    /// The decoded signature.
    pub(crate) signature: Vec<u8>,
}

/// Whether `token` has the shape of a JWS compact token: three base64url parts whose first
/// decodes to a JSON object with an `alg` (design §3.1: such a bearer goes to OIDC, any other
/// to the static tokens). Says nothing about whether it is valid.
#[must_use]
pub(crate) fn looks_like_jws(token: &str) -> bool {
    let mut parts = token.split('.');
    let (Some(h), Some(p), Some(s), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if token.len() > MAX_TOKEN_BYTES || p.is_empty() || s.is_empty() {
        return false;
    }
    b64url_decode(h)
        .ok()
        .and_then(|b| Object::parse(&b).ok())
        .is_some_and(|o| matches!(o.0.get("alg"), Some(Value::String(_))))
}

/// Splits and decodes `token` (design §3.3 steps 1 and 2).
///
/// # Errors
/// [`JoseError::Invalid`] for a token too long, not three base64url parts, a header or payload
/// that is not a JSON object with distinct members, an `alg` Krishiv does not verify, a `crit`
/// header, or a `typ` other than `JWT` or `at+jwt` (RFC 9068).
pub(crate) fn parse_jws(token: &str) -> Result<Jws<'_>, JoseError> {
    if token.len() > MAX_TOKEN_BYTES {
        return Err(JoseError::Invalid("token too long"));
    }
    let mut parts = token.splitn(4, '.');
    let (Some(h), Some(p), Some(s), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(JoseError::Invalid("not three parts"));
    };
    let header = Object::parse(&b64url_decode(h)?)?;
    let alg = match header.opt_str("alg")? {
        Some(a) => Alg::parse(a).ok_or(JoseError::Invalid("algorithm not supported"))?,
        None => return Err(JoseError::Invalid("no algorithm")),
    };
    if header.0.contains_key("crit") {
        return Err(JoseError::Invalid("critical header extensions"));
    }
    if let Some(typ) = header.opt_str("typ")? {
        let typ = typ.strip_prefix("application/").unwrap_or(typ);
        if !typ.eq_ignore_ascii_case("JWT") && !typ.eq_ignore_ascii_case("at+jwt") {
            return Err(JoseError::Invalid("not a JWT"));
        }
    }
    let kid = header.opt_str("kid")?.map(str::to_owned);
    let claims = Object::parse(&b64url_decode(p)?)?;
    let signature = b64url_decode(s)?;
    if signature.is_empty() {
        return Err(JoseError::Invalid("empty signature"));
    }
    let signing_input = &token.as_bytes()[..h.len() + 1 + p.len()];
    Ok(Jws {
        header: Header { alg, kid },
        claims,
        signing_input,
        signature,
    })
}

/// Decodes unpadded base64url (RFC 7515 §2), refusing anything non-canonical.
///
/// # Errors
/// [`JoseError::Invalid`] for padding, a character outside the alphabet, a length of 1 mod 4,
/// or non-zero trailing bits.
pub(crate) fn b64url_decode(s: &str) -> Result<Vec<u8>, JoseError> {
    const BAD: JoseError = JoseError::Invalid("not base64url");
    fn value(c: u8) -> Option<u32> {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => return None,
        };
        Some(u32::from(v))
    }
    let bytes = s.as_bytes();
    if bytes.len() % 4 == 1 {
        return Err(BAD);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3 + 2);
    for chunk in bytes.chunks(4) {
        let mut acc: u32 = 0;
        for &c in chunk {
            acc = (acc << 6) | value(c).ok_or(BAD)?;
        }
        // The chunk's bits, left aligned to 24: 4 chars = 3 bytes, 3 = 2, 2 = 1.
        let bits = chunk.len() * 6;
        let bytes_out = bits / 8;
        let spare = bits - bytes_out * 8;
        if acc & ((1 << spare) - 1) != 0 {
            return Err(BAD);
        }
        let acc = acc >> spare;
        for i in (0..bytes_out).rev() {
            out.push(((acc >> (8 * i)) & 0xff) as u8);
        }
    }
    Ok(out)
}

/// What the claims must hold (design §3.3 step 6; ADR 0046 §6).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClaimRules<'a> {
    /// The exact `iss`.
    pub(crate) issuer: &'a str,
    /// A value `aud` must contain.
    pub(crate) audience: &'a str,
    /// The claim naming the principal (`sub` by default).
    pub(crate) principal_claim: &'a str,
    /// The claim holding the groups (`groups` by default): a string or an array of strings.
    pub(crate) role_claim: &'a str,
    /// Allowed clock difference, in seconds.
    pub(crate) skew_secs: u64,
}

/// Who a valid token names.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct Identity {
    /// The principal claim's value.
    pub(crate) principal: String,
    /// The role claim's values.
    pub(crate) groups: Vec<String>,
}

/// Checks the claims of a token whose signature was verified, at `now` (seconds since the Unix
/// epoch).
///
/// # Errors
/// [`JoseError::Expired`] once `exp` (plus the skew) has passed; [`JoseError::Invalid`] for a
/// missing or non-numeric `exp`, an `nbf` still ahead, another `iss`, an `aud` without the
/// audience, a missing, empty, too long or control-character principal, or a role claim that is
/// not a string or an array of at most [`MAX_GROUPS`] strings.
pub(crate) fn check_claims(
    claims: &Object,
    rules: &ClaimRules<'_>,
    now: u64,
) -> Result<Identity, JoseError> {
    // Seconds as f64: a NumericDate may be fractional (RFC 7519 §2); any time this century is
    // exact in an f64.
    #[expect(
        clippy::cast_precision_loss,
        reason = "seconds since 1970 fit an f64 exactly"
    )]
    let (now, skew) = (now as f64, rules.skew_secs as f64);
    let date = |name: &str| match claims.0.get(name) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .filter(|d| d.is_finite())
            .map(Some)
            .ok_or(JoseError::Invalid("date claim is not a number")),
    };
    let exp = date("exp")?.ok_or(JoseError::Invalid("no exp"))?;
    if now >= exp + skew {
        return Err(JoseError::Expired);
    }
    if date("nbf")?.is_some_and(|nbf| now + skew < nbf) {
        return Err(JoseError::Invalid("not valid yet"));
    }
    if claims.opt_str("iss")? != Some(rules.issuer) {
        return Err(JoseError::Invalid("issuer"));
    }
    let audience_ok = match claims.0.get("aud") {
        Some(Value::String(a)) => a == rules.audience,
        Some(Value::Array(list)) => list.iter().any(|a| a.as_str() == Some(rules.audience)),
        _ => false,
    };
    if !audience_ok {
        return Err(JoseError::Invalid("audience"));
    }
    let principal = claims
        .opt_str(rules.principal_claim)?
        .filter(|p| is_name(p))
        .ok_or(JoseError::Invalid("principal claim"))?
        .to_owned();
    let groups = match claims.0.get(rules.role_claim) {
        None => Vec::new(),
        Some(Value::String(g)) => vec![g.clone()],
        Some(Value::Array(list)) if list.len() <= MAX_GROUPS => list
            .iter()
            .map(|g| g.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()
            .ok_or(JoseError::Invalid("role claim"))?,
        Some(_) => return Err(JoseError::Invalid("role claim")),
    };
    Ok(Identity { principal, groups })
}

/// A usable principal name: non-empty, at most [`MAX_PRINCIPAL_BYTES`], no control or format
/// characters (it is logged and shown in `system.queries`).
fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_PRINCIPAL_BYTES
        && !s.chars().any(|c| c.is_control() || is_format_char(c))
}

/// Invisible format characters that make two names look alike or reorder text in logs and
/// tables: the same set the certificate name rule refuses (`der.rs` `is_format`; a test checks
/// that both agree).
pub(crate) fn is_format_char(c: char) -> bool {
    matches!(
        c,
        '\u{00ad}'
            | '\u{061c}'
            | '\u{180e}'
            | '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206f}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
    )
}

/// The public part of a JSON Web Key (RFC 7517) Krishiv can verify with.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum KeyMaterial {
    /// An RSA key: modulus and exponent, big-endian without leading zeros.
    Rsa {
        /// The modulus.
        n: Vec<u8>,
        /// The public exponent.
        e: Vec<u8>,
    },
    /// An EC key: the uncompressed point `04 || x || y`.
    Ec {
        /// The curve.
        curve: Curve,
        /// The point.
        point: Vec<u8>,
    },
}

/// One signing key of a JWK set.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct Jwk {
    /// `kid`, if given.
    pub(crate) kid: Option<String>,
    /// `alg`, if given: the key may then only verify that algorithm.
    pub(crate) alg: Option<Alg>,
    /// The key.
    pub(crate) material: KeyMaterial,
}

impl Jwk {
    /// Whether this key may verify a token signed with `alg` (design §3.3 step 4): the key type
    /// and curve fit the algorithm, and the key's own `alg`, if any, is the same.
    #[must_use]
    pub(crate) fn fits(&self, alg: Alg) -> bool {
        if self.alg.is_some_and(|a| a != alg) {
            return false;
        }
        match (&self.material, alg.curve()) {
            (KeyMaterial::Rsa { .. }, None) => true,
            (KeyMaterial::Ec { curve, .. }, Some(c)) => *curve == c,
            _ => false,
        }
    }
}

/// The `keys` member of a JWK set document, each key with distinct members.
struct JwksDocument(Vec<Object>);

impl<'de> serde::Deserialize<'de> for JwksDocument {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = JwksDocument;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JWK set")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<JwksDocument, A::Error> {
                let mut keys = None;
                while let Some(k) = map.next_key::<String>()? {
                    if k == "keys" {
                        if keys.is_some() {
                            return Err(de::Error::custom("duplicate member"));
                        }
                        keys = Some(map.next_value::<Vec<Object>>()?);
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                keys.map(JwksDocument)
                    .ok_or_else(|| de::Error::missing_field("keys"))
            }
        }
        d.deserialize_map(V)
    }
}

/// Parses a JWK set document, keeping the signature keys Krishiv can use.
///
/// Keys of another type (`oct`, `OKP`), another curve, `use` other than `sig`, `key_ops`
/// without `verify`, an `alg` Krishiv does not verify, or with malformed numbers are skipped,
/// so one odd key does not take the others down.
///
/// # Errors
/// [`JoseError::Invalid`] for a document larger than [`MAX_DOCUMENT_BYTES`], not a JSON object
/// with a `keys` array of objects, or with more than [`MAX_JWKS_KEYS`] keys.
pub(crate) fn parse_jwks(bytes: &[u8]) -> Result<Vec<Jwk>, JoseError> {
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(JoseError::Invalid("JWK set too large"));
    }
    let JwksDocument(keys) =
        serde_json::from_slice(bytes).map_err(|_| JoseError::Invalid("not a JWK set"))?;
    if keys.len() > MAX_JWKS_KEYS {
        return Err(JoseError::Invalid("too many keys"));
    }
    Ok(keys.iter().filter_map(|k| jwk(k).ok().flatten()).collect())
}

/// One key; `Ok(None)` for a key Krishiv does not use.
fn jwk(k: &Object) -> Result<Option<Jwk>, JoseError> {
    if k.opt_str("use")?.is_some_and(|u| u != "sig") {
        return Ok(None);
    }
    if let Some(ops) = k.0.get("key_ops") {
        let verify = ops
            .as_array()
            .is_some_and(|ops| ops.iter().any(|o| o.as_str() == Some("verify")));
        if !verify {
            return Ok(None);
        }
    }
    let alg = match k.opt_str("alg")? {
        None => None,
        Some(a) => match Alg::parse(a) {
            Some(a) => Some(a),
            None => return Ok(None),
        },
    };
    let field = |name: &str| -> Result<Vec<u8>, JoseError> {
        b64url_decode(
            k.opt_str(name)?
                .ok_or(JoseError::Invalid("missing key member"))?,
        )
    };
    let material = match k.opt_str("kty")? {
        Some("RSA") => {
            let n = strip_zeros(field("n")?);
            let e = strip_zeros(field("e")?);
            if n.is_empty() || e.is_empty() {
                return Ok(None);
            }
            KeyMaterial::Rsa { n, e }
        }
        Some("EC") => {
            let Some(curve) = k.opt_str("crv")?.and_then(Curve::parse) else {
                return Ok(None);
            };
            let (x, y) = (field("x")?, field("y")?);
            let len = curve.coordinate_len();
            if x.len() != len || y.len() != len {
                return Ok(None);
            }
            let mut point = Vec::with_capacity(1 + 2 * len);
            point.push(0x04);
            point.extend_from_slice(&x);
            point.extend_from_slice(&y);
            KeyMaterial::Ec { curve, point }
        }
        _ => return Ok(None),
    };
    let jwk = Jwk {
        kid: k.opt_str("kid")?.map(str::to_owned),
        alg,
        material,
    };
    // A key whose own `alg` does not fit its type is unusable.
    Ok(jwk.alg.is_none_or(|a| jwk.fits(a)).then_some(jwk))
}

fn strip_zeros(mut v: Vec<u8>) -> Vec<u8> {
    let first = v.iter().position(|b| *b != 0).unwrap_or(v.len());
    v.drain(..first);
    v
}

/// The `jwks_uri` of an OpenID Provider configuration document (OpenID Connect Discovery 1.0
/// §3), after checking that its `issuer` is `issuer` exactly (§4.3).
///
/// # Errors
/// [`JoseError::Invalid`] for a document too large, not a JSON object, with another issuer or
/// without a string `jwks_uri`.
pub(crate) fn discovered_jwks_uri(bytes: &[u8], issuer: &str) -> Result<String, JoseError> {
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(JoseError::Invalid("discovery document too large"));
    }
    let doc = Object::parse(bytes)?;
    if doc.opt_str("issuer")? != Some(issuer) {
        return Err(JoseError::Invalid("discovery issuer"));
    }
    doc.opt_str("jwks_uri")?
        .map(str::to_owned)
        .ok_or(JoseError::Invalid("no jwks_uri"))
}
