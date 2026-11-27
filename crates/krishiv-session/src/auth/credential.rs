// SPDX-License-Identifier: Apache-2.0
//! The inbound credential a session keeps for delegated catalogs (design P3-T067 §3.3): the raw
//! bearer token of an OIDC/JWT session, or the T061 signed assertion of an impersonated one, with
//! the claims a token exchange depends on. Static tokens, client certificates and plain act-as
//! metadata keep none: they prove no user to a third party.
//!
//! The token is a [`SecretString`] from the moment authentication accepts it; `Debug` prints the
//! kind, the audiences and the expiry, never the token. A signed assertion is marked as one: its
//! audience is Krishiv by construction, so it is a subject token for an exchange and never
//! forwarded.

use std::fmt;
use std::time::SystemTime;

use krishiv_common::secret::SecretString;

pub use krishiv_jose::ExchangeClaims;

/// What kind of user token a session presented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CredentialKind {
    /// An OIDC/JWT bearer token the configured issuer signed.
    Oidc,
    /// A T061 signed act-as assertion (`x-krishiv-act-as-token`): exchangeable, never forwardable.
    SignedAssertion,
}

impl CredentialKind {
    /// How error messages name it: `oidc` or `a signed assertion`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oidc => "oidc",
            Self::SignedAssertion => "a signed assertion",
        }
    }
}

/// The canonical encoding the exchange cache keys on: every value length-prefixed, groups
/// sorted, an absent claim and an empty one encoded differently.
fn canonical(claims: &ExchangeClaims) -> String {
    fn list(out: &mut String, tag: &str, values: Option<&[String]>, sort: bool) {
        out.push_str(tag);
        match values {
            None => out.push('-'),
            Some(v) => {
                let mut v: Vec<&str> = v.iter().map(String::as_str).collect();
                if sort {
                    v.sort_unstable();
                }
                out.push_str(&format!("{}[", v.len()));
                for s in v {
                    out.push_str(&format!("{}:{s}", s.len()));
                }
                out.push(']');
            }
        }
    }
    let mut out = String::new();
    list(&mut out, "aud", Some(&claims.audiences), true);
    let scope: Option<Vec<String>> = claims.scope.as_ref().map(|s| vec![s.clone()]);
    list(&mut out, "scope", scope.as_deref(), false);
    list(&mut out, "scp", claims.scp.as_deref(), true);
    list(&mut out, "groups", claims.groups.as_deref(), true);
    out
}

/// The user token one RPC authenticated with, kept by the session for its delegated catalogs.
#[derive(Clone, PartialEq, Eq)]
pub struct InboundCredential {
    kind: CredentialKind,
    token: SecretString,
    claims: ExchangeClaims,
    expires_at: SystemTime,
}

impl fmt::Debug for InboundCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InboundCredential")
            .field("kind", &self.kind)
            .field("token", &self.token)
            .field("audiences", &self.claims.audiences)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

impl InboundCredential {
    /// A credential of `kind`: the raw `token` with its validated `claims`, valid until
    /// `expires_at`. Only an authenticator that verified the token builds one for a session.
    #[must_use]
    pub fn new(
        kind: CredentialKind,
        token: SecretString,
        claims: ExchangeClaims,
        expires_at: SystemTime,
    ) -> Self {
        Self {
            kind,
            token,
            claims,
            expires_at,
        }
    }

    /// The same from a JWT's `exp` (seconds since the Unix epoch).
    #[cfg(feature = "oidc")]
    pub(crate) fn from_exp(
        kind: CredentialKind,
        token: &str,
        claims: ExchangeClaims,
        exp: u64,
    ) -> Self {
        Self::new(
            kind,
            SecretString::new(token),
            claims,
            std::time::UNIX_EPOCH
                .checked_add(std::time::Duration::from_secs(exp))
                .unwrap_or(std::time::UNIX_EPOCH),
        )
    }

    /// OIDC token or signed assertion.
    #[must_use]
    pub fn kind(&self) -> CredentialKind {
        self.kind
    }

    /// The raw token. Only a delegated catalog's token source reads it.
    #[must_use]
    pub fn token(&self) -> &SecretString {
        &self.token
    }

    /// The validated claims.
    #[must_use]
    pub fn claims(&self) -> &ExchangeClaims {
        &self.claims
    }

    /// When it expires (`exp`).
    #[must_use]
    pub fn expires_at(&self) -> SystemTime {
        self.expires_at
    }

    /// A keyed 128-bit fingerprint of the exchange claims ([`ExchangeClaims`]): equal claims,
    /// equal fingerprints within the process; the claims cannot be read back from it.
    #[must_use]
    pub fn claims_fingerprint(&self) -> [u8; 16] {
        SecretString::new(canonical(&self.claims)).fingerprint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(claims: ExchangeClaims) -> InboundCredential {
        InboundCredential::new(
            CredentialKind::Oidc,
            SecretString::new("planted-subject-token"),
            claims,
            SystemTime::UNIX_EPOCH,
        )
    }

    #[test]
    fn debug_never_shows_the_token() {
        let text = format!("{:?}", with(ExchangeClaims::default()));
        assert!(!text.contains("planted-subject-token"), "{text}");
        assert!(text.contains("redacted"), "{text}");
    }

    #[test]
    fn the_claims_fingerprint_sorts_groups_and_tells_absent_from_empty() {
        let groups = |g: Option<Vec<&str>>| {
            let mut c = ExchangeClaims::default();
            c.groups = g.map(|g| g.into_iter().map(str::to_owned).collect());
            c
        };
        let fp = |c| with(c).claims_fingerprint();
        assert_eq!(
            fp(groups(Some(vec!["a", "b"]))),
            fp(groups(Some(vec!["b", "a"])))
        );
        assert_ne!(fp(groups(None)), fp(groups(Some(vec![]))));
        assert_ne!(
            fp(groups(Some(vec!["ab"]))),
            fp(groups(Some(vec!["a", "b"])))
        );
        let scope = |s: Option<&str>| {
            let mut c = ExchangeClaims::default();
            c.scope = s.map(str::to_owned);
            c
        };
        assert_ne!(fp(scope(None)), fp(scope(Some(""))));
        assert_ne!(fp(scope(Some("read write"))), fp(scope(Some("read"))));
        // A different token with the same claims has the same fingerprint.
        let mut other = with(scope(Some("read")));
        other.token = SecretString::new("another-token");
        assert_eq!(other.claims_fingerprint(), fp(scope(Some("read"))));
    }
}
