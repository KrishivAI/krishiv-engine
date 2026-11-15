// SPDX-License-Identifier: Apache-2.0
//! Trusted impersonation on the Connect listener (P3-T061 I2/I3, design
//! `design/P3-T061-impersonation-jwt-hardening.md` §4, amendment A12): the act-as metadata of
//! one RPC, read from its headers before the body is decoded, and handed to the server's
//! [`ServerAuth`].
//!
//! - **Only explicit metadata asks** (security review requirement 1): `x-krishiv-act-as-user`
//!   names the end user, `x-krishiv-act-as-groups` (comma separated) and
//!   `x-krishiv-act-as-tenant` go with it, each value URL-encoded; `x-krishiv-act-as-token`
//!   carries a signed assertion instead. A request with none of them acts for its caller.
//!   `UserContext.user_id` never asks: it stays the unverified client claim (ADR 0046 §5).
//! - **Never ignored** (§4.2 rule 1): metadata that cannot be read (twice the same header, not
//!   text, a bad `%` escape, groups or a tenant without a user) is refused like any other
//!   refusal (`KRISHIV_AUTH_IMPERSONATION_DENIED`, 42501, one audit record), and so is
//!   act-as on a server without authentication or without an impersonating authenticator.
//! - **Per RPC** (requirement 4): the layer runs on every RPC, so a service whose `expires`
//!   passed, or an assertion past its lifetime, stops acting at its next call.
//! - The peer is the connection's own TCP peer (`PeerInfo`, requirement 2), never a header.
//! - **Plain metadata beside an assertion** must name the same user, groups and tenant, or the
//!   request is refused (requirement 5); only the [`ActAs`] and [`Authenticated`] the server
//!   returns are used afterwards (requirement 6), so the end user's role is the capped one.

use std::net::SocketAddr;

use krishiv_common::error::KrishivError;
use krishiv_session::auth::{ActAs, Authenticated, Denial, ServerAuth, Surface};
use tonic::codegen::http::HeaderMap;

/// The end user's name (plain act-as).
pub const ACT_AS_USER: &str = "x-krishiv-act-as-user";
/// The end user's groups, comma separated (plain act-as).
pub const ACT_AS_GROUPS: &str = "x-krishiv-act-as-groups";
/// The end user's tenant (plain act-as).
pub const ACT_AS_TENANT: &str = "x-krishiv-act-as-tenant";
/// A signed act-as assertion (`krishiv.auth.impersonation.signedAssertion`).
pub const ACT_AS_TOKEN: &str = "x-krishiv-act-as-token";

const HEADERS: [&str; 4] = [ACT_AS_USER, ACT_AS_GROUPS, ACT_AS_TENANT, ACT_AS_TOKEN];

/// Whether `headers` carry any act-as metadata.
pub(super) fn asks(headers: &HeaderMap) -> bool {
    HEADERS.iter().any(|h| headers.contains_key(*h))
}

/// The act-as metadata of one request, decoded.
#[derive(Debug, Default, PartialEq, Eq)]
struct Metadata {
    user: Option<String>,
    groups: Option<Vec<String>>,
    tenant: Option<String>,
    token: Option<String>,
}

impl Metadata {
    /// Reads `headers`; `None` when any act-as header cannot be read.
    fn read(headers: &HeaderMap) -> Option<Self> {
        let groups = match one(headers, ACT_AS_GROUPS)? {
            None => None,
            // `x-krishiv-act-as-groups=` with nothing after it: no groups.
            Some("") => Some(Vec::new()),
            Some(text) => Some(
                text.split(',')
                    .map(percent_decode)
                    .collect::<Option<Vec<_>>>()?,
            ),
        };
        Some(Self {
            user: decoded(headers, ACT_AS_USER)?,
            groups,
            tenant: decoded(headers, ACT_AS_TENANT)?,
            // A compact JWS is base64url and dots: no escapes to undo.
            token: one(headers, ACT_AS_TOKEN)?.map(str::to_owned),
        })
    }

    /// Whether the plain part names something other than `act_as`.
    fn disagrees_with(&self, act_as: &ActAs) -> bool {
        self.user.as_ref().is_some_and(|u| *u != act_as.user)
            || self.groups.as_ref().is_some_and(|g| *g != act_as.groups)
            || self.tenant.is_some() && self.tenant != act_as.tenant
    }
}

/// The value of header `name`: `Some(None)` when absent, `None` when it is present twice or is
/// not visible ASCII.
fn one<'h>(headers: &'h HeaderMap, name: &str) -> Option<Option<&'h str>> {
    let mut values = headers.get_all(name).iter();
    match (values.next(), values.next()) {
        (None, _) => Some(None),
        (Some(value), None) => value.to_str().ok().map(Some),
        (Some(_), Some(_)) => None,
    }
}

/// [`one`], URL-decoded.
fn decoded(headers: &HeaderMap, name: &str) -> Option<Option<String>> {
    match one(headers, name)? {
        None => Some(None),
        Some(text) => percent_decode(text).map(Some),
    }
}

/// `%XX` escapes undone; `None` for a broken escape or bytes that are not UTF-8. `+` stays a
/// plus sign (these are not form values).
fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        if b == b'%' {
            let hex = bytes.get(i + 1..i + 3)?;
            let hex = std::str::from_utf8(hex).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(b);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Who the request acts for: `caller` itself without act-as metadata, else the end user
/// `server` allows (module docs).
///
/// # Errors
/// `KRISHIV_AUTH_IMPERSONATION_DENIED` (42501) for every refusal, the reason in the audit
/// record only.
pub(super) async fn resolve(
    server: &ServerAuth,
    headers: &HeaderMap,
    peer: SocketAddr,
    caller: Authenticated,
) -> Result<Authenticated, KrishivError> {
    if !asks(headers) {
        return Ok(caller);
    }
    let Some(metadata) = Metadata::read(headers) else {
        return Err(server.refuse_impersonation(
            &caller,
            Surface::Connect,
            peer,
            Denial::Malformed,
        ));
    };
    if let Some(token) = &metadata.token {
        let (act_as, end_user) = server
            .impersonate_signed(&caller, Surface::Connect, peer, token)
            .await?;
        if metadata.disagrees_with(&act_as) {
            return Err(server.refuse_impersonation(
                &caller,
                Surface::Connect,
                peer,
                Denial::MetadataMismatch,
            ));
        }
        return Ok(end_user);
    }
    // Groups or a tenant name no one without a user.
    let Some(user) = metadata.user else {
        return Err(server.refuse_impersonation(
            &caller,
            Surface::Connect,
            peer,
            Denial::Malformed,
        ));
    };
    let mut act_as = ActAs::new(user).with_groups(metadata.groups.unwrap_or_default());
    if let Some(tenant) = metadata.tenant {
        act_as = act_as.with_tenant(tenant);
    }
    server.impersonate(&caller, Surface::Connect, peer, &act_as)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::codegen::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(*k, HeaderValue::from_static(v));
        }
        h
    }

    #[test]
    fn percent_escapes_are_undone() {
        assert_eq!(percent_decode("a%2Cb%20c+d").as_deref(), Some("a,b c+d"));
        assert_eq!(percent_decode("%C3%A9").as_deref(), Some("é"));
        assert_eq!(percent_decode("%2"), None);
        assert_eq!(percent_decode("%zz"), None);
        assert_eq!(percent_decode("%FF"), None);
    }

    #[test]
    fn metadata_reads_user_groups_and_tenant() {
        let h = headers(&[
            (ACT_AS_USER, "alice"),
            (ACT_AS_GROUPS, "eng,data%2Cteam"),
            (ACT_AS_TENANT, "acme"),
        ]);
        assert!(asks(&h));
        assert_eq!(
            Metadata::read(&h),
            Some(Metadata {
                user: Some("alice".to_owned()),
                groups: Some(vec!["eng".to_owned(), "data,team".to_owned()]),
                tenant: Some("acme".to_owned()),
                token: None,
            })
        );
    }

    #[test]
    fn a_repeated_or_broken_header_is_unreadable() {
        let twice = headers(&[(ACT_AS_USER, "alice"), (ACT_AS_USER, "bob")]);
        assert_eq!(Metadata::read(&twice), None);
        let broken = headers(&[(ACT_AS_USER, "al%ice")]);
        assert_eq!(Metadata::read(&broken), None);
    }

    #[test]
    fn no_metadata_asks_nothing() {
        assert!(!asks(&headers(&[("authorization", "Bearer x")])));
    }

    #[test]
    fn plain_metadata_must_match_the_assertion() {
        let act_as = ActAs::new("alice").with_groups(vec!["eng".to_owned()]);
        let same = Metadata {
            user: Some("alice".to_owned()),
            groups: Some(vec!["eng".to_owned()]),
            ..Metadata::default()
        };
        assert!(!same.disagrees_with(&act_as));
        assert!(!Metadata::default().disagrees_with(&act_as));
        let other_user = Metadata {
            user: Some("bob".to_owned()),
            ..Metadata::default()
        };
        assert!(other_user.disagrees_with(&act_as));
        let other_groups = Metadata {
            groups: Some(vec!["admins".to_owned()]),
            ..Metadata::default()
        };
        assert!(other_groups.disagrees_with(&act_as));
        let a_tenant = Metadata {
            tenant: Some("acme".to_owned()),
            ..Metadata::default()
        };
        assert!(a_tenant.disagrees_with(&act_as));
    }
}
