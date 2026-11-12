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

use super::Role;
use super::config::{SPARK_TOKEN_KEY, SPARK_TOKEN_NAME, TokenSpec};
use super::secret_ref::invalid;

/// The shortest secret accepted, in bytes (design §3.2; threat model T-A4).
pub(crate) const MIN_SECRET_BYTES: usize = 32;

/// One stored token.
#[derive(Debug, Clone)]
struct Entry {
    digest: [u8; 32],
    principal: Principal,
    role: Role,
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
    /// bytes, or is the same as another token's secret, or when two tokens give one principal
    /// different roles.
    pub fn build(specs: &[TokenSpec]) -> Result<Self, KrishivError> {
        let mut entries: Vec<(Entry, &str)> = Vec::with_capacity(specs.len());
        for spec in specs {
            let key = secret_key(&spec.name);
            let secret: SecretString = spec.secret.resolve(&key)?;
            if secret.expose_secret().len() < MIN_SECRET_BYTES {
                return Err(invalid(
                    &key,
                    krishiv_common::secret::REDACTED,
                    &format!("a token must be at least {MIN_SECRET_BYTES} bytes"),
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
        let d = digest(presented);
        let mut found = None;
        for (i, e) in self.entries.iter().enumerate() {
            let same = ct_eq(&d, &e.digest);
            if same && found.is_none() {
                found = Some(i);
            }
        }
        let e = self.entries.get(found?)?;
        Some((&e.principal, e.role))
    }

    /// How many digests one lookup compares (the structural constant-time test).
    #[cfg(test)]
    pub(crate) fn comparisons_per_lookup(&self) -> usize {
        self.entries.len()
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
