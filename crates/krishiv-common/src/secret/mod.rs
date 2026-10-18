// SPDX-License-Identifier: Apache-2.0
//! Typed secrets for plan and connector options (ADR 0038).
//!
//! An option value is classified once, where a front end builds it
//! ([`OptionValue::from_user`]), into plain text, a secret, or a URL whose
//! credentials are lifted into secret parts. Redaction is then a property
//! of the types: no `Display` or `Debug` of these types prints a secret, so
//! no sink (EXPLAIN, logs, UI, errors, `Debug`) has to guess. The plaintext
//! is reachable only through an explicit `expose_secret` call, made by the
//! connector that opens the source.
//!
//! This module lives in `krishiv-common` (layer L0) because the plan IR
//! (L3), the connector SDK (L1, connector option schemas and the
//! `SecretProvider` of ADR 0036 R12), connectors, logging and the UI all
//! share it.
//!
//! # Classification (fail-safe default)
//!
//! Until connector option schemas mark sensitive options (P4, ADR 0036),
//! a key is sensitive when, split into words at punctuation and
//! lower-to-upper case changes and lowercased:
//!
//! - a word is one of `pwd`, `passwd`, `sas`, `sig`, `signature`, `jwt`; or
//! - the words joined contain `password`, `passphrase`, `secret`, `token`,
//!   `credential`, `accesskey`, `accountkey`, `privatekey`, `apikey`,
//!   `keystorekey`, `sessionkey`, `jaasconfig` or `authorization`.
//!
//! So `password`, `fs.s3a.access.key`, `sasl.jaas.config`, `apiKey` and
//! `fs.azure.account.key.x` are secrets. The list errs towards hiding: an
//! option it hides by mistake prints as `*********(redacted)` but works unchanged.
//!
//! # URLs
//!
//! A value of a non-sensitive key that is a URL (`scheme://...`, or an
//! opaque `jdbc:` URL with `user/password@`) is split structurally, never
//! hidden whole. Secret parts are:
//!
//! - the password of the userinfo (`user:password@host`; Oracle's opaque
//!   `jdbc:oracle:thin:user/password@host`); the user name stays visible.
//!   The userinfo is looked for only before the first parameter separator
//!   (`?`, `&`, `;`, `#`), and the last `@` there ends it, so a password may
//!   hold `@`, `:` or `/` but an `@` in a parameter value
//!   (`jdbc:sqlserver://h;password=p@ss`) is never taken for one. A path
//!   holding `@` after a `:` (`https://h:8080/a@b`) is hidden by mistake,
//!   the fail-safe direction;
//! - the value of every parameter (after `?`, `&`, `;` or `#`; a value in
//!   braces, as SQL Server's `password={a;b}}c}`, runs to its closing
//!   brace, `}}` escaping one) whose name,
//!   percent-decoded and lowercased, is a sensitive key (above) or one of
//!   `pwd`, `passwd`, `access_token`, `refresh_token`, `id_token`,
//!   `client_secret`, `api_key`, `apikey`, `key`, `sig`, `signature`,
//!   `x-amz-signature`, `x-amz-credential`, `x-amz-security-token`,
//!   `x-goog-signature`, `x-goog-credential`. Azure SAS fields other than
//!   `sig` (`sv`, `se`, `sp`, ...) and AWS/GCS fields other than the
//!   signature, credential and session token are not secret.
//!
//! So `jdbc:postgresql://u:pw@h/db?password=pw&ssl=true` prints as
//! `jdbc:postgresql://u:*********(redacted)@h/db?password=*********(redacted)&ssl=true`.
//!
//! # Memory
//!
//! A [`SecretString`] is zeroed on drop, over its whole allocation. The
//! zeroing is safe code (this crate forbids `unsafe`, ADR 0029): the bytes
//! are overwritten and then passed to [`std::hint::black_box`] behind a
//! compiler fence, so the compiler must assume they are read and cannot
//! drop the writes as dead stores. Like `zeroize`, this is best effort: it
//! does not reach copies the allocator or the caller made earlier.

mod classify;
#[cfg(test)]
mod tests;

use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;
use std::sync::atomic::{Ordering, compiler_fence};

pub use classify::is_sensitive_key;

/// What every secret prints as: Spark's redaction marker, so EXPLAIN and
/// error text read as Spark's where users see them.
pub const REDACTED: &str = "*********(redacted)";
pub(crate) use classify::{semicolon_authority, url_secret_ranges};

/// A secret string: no `Display`, `Debug` prints `*********(redacted)`, zeroed on drop,
/// compared in constant time (for equal lengths).
#[derive(Clone)]
pub struct SecretString(String);

impl SecretString {
    /// Takes ownership of `value` without copying it.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The plaintext. Only a connector opening its source calls this.
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }

    /// A 128-bit keyed fingerprint of the secret: SipHash-1-3 (std's
    /// `RandomState`) under one random key per process, run twice with a
    /// domain byte. Equal secrets have equal fingerprints within a process
    /// and unrelated ones across processes, and the plaintext cannot be
    /// recovered from it without the key. It identifies, it does not
    /// authenticate.
    #[must_use]
    pub fn fingerprint(&self) -> [u8; 16] {
        static KEY: OnceLock<RandomState> = OnceLock::new();
        let key = KEY.get_or_init(RandomState::new);
        let mut out = [0_u8; 16];
        for (domain, chunk) in (0_u8..).zip(out.as_chunks_mut::<8>().0) {
            let mut h = key.build_hasher();
            h.write_u8(domain);
            h.write_usize(self.0.len());
            h.write(self.0.as_bytes());
            *chunk = h.finish().to_le_bytes();
        }
        out
    }

    /// The string, handed over without zeroing (it moves, so no copy is
    /// left behind).
    fn into_inner(mut self) -> String {
        std::mem::take(&mut self.0)
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

impl PartialEq for SecretString {
    fn eq(&self, other: &Self) -> bool {
        ct_eq(self.0.as_bytes(), other.0.as_bytes())
    }
}

impl Eq for SecretString {}

impl Drop for SecretString {
    fn drop(&mut self) {
        wipe(&mut self.0);
    }
}

/// Overwrites `value`, over its whole allocation, with zeros (as a
/// [`SecretString`] is on drop) and leaves it empty: for plaintext that
/// cannot be typed where it is decoded, such as the option values of a
/// decoded Spark Connect plan, once their typed copy exists.
pub fn wipe(value: &mut String) {
    let mut bytes = std::mem::take(value).into_bytes();
    zero(&mut bytes);
}

/// [`wipe`] for bytes.
pub fn wipe_bytes(value: &mut Vec<u8>) {
    let mut bytes = std::mem::take(value);
    zero(&mut bytes);
}

/// Overwrites the whole allocation of `bytes` (length and spare capacity)
/// with zeros, so that the writes cannot be optimised away (see the module
/// docs).
fn zero(bytes: &mut Vec<u8>) {
    let capacity = bytes.capacity();
    bytes.fill(0);
    // Within capacity: writes the spare bytes, never reallocates.
    bytes.resize(capacity, 0);
    std::hint::black_box(bytes.as_mut_slice());
    compiler_fence(Ordering::SeqCst);
}

/// Equality whose time depends only on the lengths, not on where the
/// inputs differ.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let diff = a.iter().zip(b).fold(0_u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

/// A secret option value. Not `non_exhaustive`: every consumer (the plan
/// encoding, connectors) must handle a new kind explicitly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretValue {
    /// `secret('scope', 'key')`: resolved by the `SecretProvider` only when
    /// a connector opens the source (ADR 0036 R12), never stored resolved.
    Reference {
        /// The provider scope.
        scope: String,
        /// The key within the scope.
        key: String,
    },
    /// A secret given inline, as Spark's `.option("password", ...)`.
    Inline(SecretString),
}

impl fmt::Display for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reference { scope, key } => {
                write!(f, "secret('{}', '{}')", quoted(scope), quoted(key))
            }
            Self::Inline(_) => f.write_str(REDACTED),
        }
    }
}

/// `s` with `\` and `'` escaped, for a SQL string literal.
fn quoted(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

/// One part of a [`SecretUrl`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UrlPart {
    /// Printable text.
    Text(String),
    /// A credential: a userinfo password or a parameter value.
    Secret(SecretString),
}

/// A URL with credentials lifted out (see the module docs). `Display`
/// prints it with `*********(redacted)` in their place; [`Self::expose_secret`] rebuilds
/// the full URL.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretUrl {
    parts: Vec<UrlPart>,
}

impl SecretUrl {
    /// Splits `url` at its credentials; `None` when it has none.
    #[must_use]
    pub fn parse(url: &str) -> Option<Self> {
        let ranges = url_secret_ranges(url);
        if ranges.is_empty() {
            return None;
        }
        let mut parts = Vec::with_capacity(2 * ranges.len() + 1);
        let mut at = 0;
        for r in ranges {
            if r.start > at {
                parts.push(UrlPart::Text(url[at..r.start].to_owned()));
            }
            parts.push(UrlPart::Secret(SecretString::new(&url[r.clone()])));
            at = r.end;
        }
        if at < url.len() {
            parts.push(UrlPart::Text(url[at..].to_owned()));
        }
        Some(Self { parts })
    }

    /// The parts, in order.
    #[must_use]
    pub fn parts(&self) -> &[UrlPart] {
        &self.parts
    }

    /// The full URL. Only a connector opening its source calls this.
    #[must_use]
    pub fn expose_secret(&self) -> SecretString {
        let len = self.parts.iter().map(|p| part_str(p).len()).sum();
        // Exact capacity: the buffer never reallocates, so no unzeroed
        // copy is left behind.
        let mut out = String::with_capacity(len);
        self.parts.iter().for_each(|p| out.push_str(part_str(p)));
        SecretString(out)
    }
}

fn part_str(p: &UrlPart) -> &str {
    match p {
        UrlPart::Text(s) => s,
        UrlPart::Secret(s) => s.expose_secret(),
    }
}

impl fmt::Display for SecretUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for p in &self.parts {
            match p {
                UrlPart::Text(s) => f.write_str(s)?,
                UrlPart::Secret(_) => f.write_str(REDACTED)?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for SecretUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretUrl({:?})", self.to_string())
    }
}

/// A data source option value (ADR 0038). Build user-supplied values with
/// [`Self::from_user`]; `Display` and `Debug` never print a secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionValue {
    /// A value that is not secret.
    Plain(String),
    /// A secret value.
    Secret(SecretValue),
    /// A URL with credentials in it.
    Url(SecretUrl),
}

impl OptionValue {
    /// Classifies a user-supplied option (see the module docs): the one
    /// constructor front ends (SQL, Spark Connect, DataFrame APIs) use.
    #[must_use]
    pub fn from_user(key: &str, value: impl Into<String>) -> Self {
        let value = SecretString::new(value);
        if is_sensitive_key(key) {
            return Self::Secret(SecretValue::Inline(value));
        }
        // `value` is zeroed when dropped if it held a URL with credentials.
        match SecretUrl::parse(value.expose_secret()) {
            Some(url) => Self::Url(url),
            None => Self::Plain(value.into_inner()),
        }
    }

    /// A `secret('scope', 'key')` reference.
    #[must_use]
    pub fn reference(scope: impl Into<String>, key: impl Into<String>) -> Self {
        Self::Secret(SecretValue::Reference {
            scope: scope.into(),
            key: key.into(),
        })
    }

    /// The value, when it is not secret.
    #[must_use]
    pub fn as_plain(&self) -> Option<&str> {
        match self {
            Self::Plain(s) => Some(s),
            Self::Secret(_) | Self::Url(_) => None,
        }
    }
}

impl fmt::Display for OptionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plain(s) => f.write_str(s),
            Self::Secret(s) => s.fmt(f),
            Self::Url(u) => u.fmt(f),
        }
    }
}
