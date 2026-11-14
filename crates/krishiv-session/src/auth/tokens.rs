// SPDX-License-Identifier: Apache-2.0
//! Static tokens (design §3.2): one token, one principal.
//!
//! Each secret is resolved once at startup, hashed with SHA-256 and dropped (a
//! [`SecretString`] zeroes itself). A presented token is hashed once and compared with every
//! stored digest in constant time, without an early exit: a map keyed by the digest would leak
//! timing on the digest and gains nothing at tens of tokens.

use krishiv_common::error::KrishivError;
use krishiv_common::principal::Principal;
use krishiv_common::secret::SecretString;
use sha2::{Digest, Sha256};
use std::time::SystemTime;

use super::Role;
use super::config::{SPARK_TOKEN_KEY, SPARK_TOKEN_NAME, TokenSpec};
use super::secret_ref::invalid;

/// The shortest secret accepted, in bytes (design §3.2; threat model T-A4).
pub(crate) const MIN_SECRET_BYTES: usize = 32;

/// The longest secret accepted, in bytes: the longest bearer token the server hashes
/// (design §3.3's JWT cap), so every configured secret can be presented.
pub(crate) const MAX_SECRET_BYTES: usize = 16 * 1024;

/// One stored token.
#[derive(Debug, Clone)]
struct Entry {
    digest: [u8; 32],
    principal: Principal,
    role: Role,
    expires: Option<SystemTime>,
}

/// The configured static tokens, as digests.
#[derive(Debug, Clone, Default)]
pub struct StaticTokens {
    entries: Vec<Entry>,
}

impl StaticTokens {
    /// Resolves and hashes every token of `specs`.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` when a secret cannot be resolved, is shorter than 32
    /// bytes or longer than 16 KiB, holds anything but visible ASCII, or is the same as another token's secret, or when two tokens give one principal
    /// different roles.
    pub fn build(specs: &[TokenSpec]) -> Result<Self, KrishivError> {
        Self::build_at(specs, SystemTime::now())
    }

    /// [`Self::build`] at `now`: a token already expired then refuses startup.
    ///
    /// # Errors
    /// As [`Self::build`].
    pub(crate) fn build_at(specs: &[TokenSpec], now: SystemTime) -> Result<Self, KrishivError> {
        let mut entries: Vec<(Entry, &str)> = Vec::with_capacity(specs.len());
        for spec in specs {
            let key = secret_key(&spec.name);
            if spec.expires.is_some_and(|e| now >= e) {
                return Err(invalid(
                    &format!("krishiv.auth.principals.{}.expires", spec.principal),
                    "",
                    &format!("token `{}` has expired", spec.name),
                ));
            }
            let secret: SecretString = spec.secret.resolve(&key)?;
            let len = secret.expose_secret().len();
            if len < MIN_SECRET_BYTES {
                return Err(invalid(
                    &key,
                    krishiv_common::secret::REDACTED,
                    &format!("a token must be at least {MIN_SECRET_BYTES} bytes"),
                ));
            }
            if len > MAX_SECRET_BYTES {
                return Err(invalid(
                    &key,
                    krishiv_common::secret::REDACTED,
                    &format!("a token must be at most {MAX_SECRET_BYTES} bytes"),
                ));
            }
            if !secret.expose_secret().bytes().all(|b| b.is_ascii_graphic()) {
                // `Authorization: Bearer <token>` cannot carry it as written: header parsing drops
                // leading and trailing spaces, and controls or non-ASCII bytes are not header text.
                return Err(invalid(
                    &key,
                    krishiv_common::secret::REDACTED,
                    "a token must be visible ASCII characters only (no spaces or controls)",
                ));
            }
            let digest = digest(secret.expose_secret());
            drop(secret);
            if let Some((_, other)) = entries.iter().find(|(e, _)| e.digest == digest) {
                return Err(invalid(
                    &key,
                    krishiv_common::secret::REDACTED,
                    &format!("the same secret as token `{other}`; each token names one principal"),
                ));
            }
            let principal = Principal::new(spec.principal.clone(), spec.kind);
            // A principal has exactly one role (ADR 0046 §2): two tokens may name one principal
            // (rotation), but only with the same role.
            if let Some((other, other_name)) = entries
                .iter()
                .find(|(e, _)| e.principal.name == principal.name && e.role != spec.role)
            {
                return Err(invalid(
                    &format!("krishiv.auth.tokens.{}.role", spec.name),
                    spec.role.as_str(),
                    &format!(
                        "principal `{}` already has role {} through token `{other_name}`; a \
                         principal has exactly one role",
                        principal.name, other.role
                    ),
                ));
            }
            entries.push((
                Entry {
                    digest,
                    principal,
                    role: spec.role,
                    expires: spec.expires,
                },
                &spec.name,
            ));
        }
        Ok(Self {
            entries: entries.into_iter().map(|(e, _)| e).collect(),
        })
    }

    /// How many tokens are configured.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no token is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The principal and role of `presented`, if it is a configured token. Every entry is
    /// compared, whether or not an earlier one matched.
    #[must_use]
    pub fn lookup(&self, presented: &str) -> Option<(&Principal, Role)> {
        let (found, _) = self.find(presented);
        let e = self.entries.get(found?)?;
        Some((&e.principal, e.role))
    }

    /// [`Self::lookup`], and whether the token has expired at `now`.
    #[must_use]
    pub(crate) fn lookup_at(
        &self,
        presented: &str,
        now: SystemTime,
    ) -> Option<(&Principal, Role, bool)> {
        let (found, _) = self.find(presented);
        let e = self.entries.get(found?)?;
        Some((&e.principal, e.role, e.expires.is_some_and(|x| now >= x)))
    }

    /// The index of the entry `presented` matches, and how many digests were compared to find
    /// it (always all of them).
    fn find(&self, presented: &str) -> (Option<usize>, usize) {
        let d = digest(presented);
        let mut found = None;
        let mut compared = 0;
        for (i, e) in self.entries.iter().enumerate() {
            let same = ct_eq(&d, &e.digest);
            compared += 1;
            if same && found.is_none() {
                found = Some(i);
            }
        }
        (found, compared)
    }

    /// How many digests a lookup of `presented` compared (the structural constant-time test).
    #[cfg(test)]
    pub(crate) fn comparisons_for(&self, presented: &str) -> usize {
        self.find(presented).1
    }
}

/// The settings key naming the secret of token `name`.
fn secret_key(name: &str) -> String {
    if name == SPARK_TOKEN_NAME {
        SPARK_TOKEN_KEY.to_owned()
    } else {
        format!("krishiv.auth.tokens.{name}.secret")
    }
}

/// SHA-256 of `s`.
fn digest(s: &str) -> [u8; 32] {
    Sha256::digest(s.as_bytes()).into()
}

/// Equality of two digests in time independent of where they differ.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let diff = a.iter().zip(b).fold(0_u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}
