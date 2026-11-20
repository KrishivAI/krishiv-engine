// SPDX-License-Identifier: Apache-2.0
//! Flight SQL session cookies (design P3-T101 §3.2). Flight SQL commands carry no session id,
//! so the session rides in a cookie, `arrow_flight_session_id`, which the server mints and the
//! client sends back (JDBC by default, ADBC with `adbc.flight.sql.rpc.with_cookie_middleware`).
//! This module reads and writes the cookie only; whether the id names a live session of the
//! caller is the session manager's answer, and a refused or malformed id gets a new session
//! and a new cookie (the caller's job).

use tonic::metadata::{MetadataMap, MetadataValue};

/// The cookie's name (the Arrow Flight SQL session convention).
pub const SESSION_COOKIE: &str = "arrow_flight_session_id";

/// The longest cookie value read. A server-minted id is a 36-character UUID; anything longer
/// is not ours and is answered as malformed without being copied.
pub const MAX_COOKIE_VALUE: usize = 64;

/// What a request said about its session cookie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionCookie {
    /// No `arrow_flight_session_id` cookie: a cookie-less client (one fetch per ticket).
    Absent,
    /// The cookie's value, for the session manager to look up.
    Present(String),
    /// A cookie this server cannot have set (empty, too long, non-token characters, or sent
    /// twice with different values): the client keeps cookies, so it gets a new session and
    /// a new cookie, as for an id the manager refuses.
    Malformed,
}

/// The session cookie of a request: every `cookie` header (HTTP/2 may split cookies over several
/// headers, RFC 9113 §8.2.3), each split on `;`.
#[must_use]
pub fn session_cookie(metadata: &MetadataMap) -> SessionCookie {
    let mut found: Option<&str> = None;
    for header in metadata.get_all("cookie") {
        let Ok(text) = header.to_str() else {
            continue;
        };
        for pair in text.split(';') {
            let Some((name, value)) = pair.trim().split_once('=') else {
                continue;
            };
            if name.trim() != SESSION_COOKIE {
                continue;
            }
            let value = value.trim();
            if !well_formed(value) {
                return SessionCookie::Malformed;
            }
            match found {
                Some(seen) if seen != value => return SessionCookie::Malformed,
                _ => found = Some(value),
            }
        }
    }
    match found {
        None => SessionCookie::Absent,
        Some(v) => SessionCookie::Present(v.to_owned()),
    }
}

/// A cookie value this server could have set: 1 to [`MAX_COOKIE_VALUE`] ASCII letters, digits
/// or `-`.
fn well_formed(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_COOKIE_VALUE
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

/// Adds `set-cookie: arrow_flight_session_id=<id>` to a response's metadata. The id is a
/// server-minted session id; `false` (and nothing added) when it holds a character a cookie
/// cannot carry, which a minted UUID never does.
#[must_use]
pub fn set_session_cookie(metadata: &mut MetadataMap, session_id: &str) -> bool {
    if !well_formed(session_id) {
        return false;
    }
    match MetadataValue::try_from(format!("{SESSION_COOKIE}={session_id}")) {
        Ok(value) => {
            metadata.append("set-cookie", value);
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: &str = "0b1f6a52-3c4d-4e5f-8a9b-0c1d2e3f4a5b";

    fn with_cookies(headers: &[&str]) -> MetadataMap {
        let mut m = MetadataMap::new();
        for h in headers {
            m.append("cookie", MetadataValue::try_from(*h).expect("ascii"));
        }
        m
    }

    #[test]
    fn absent_without_cookie_or_with_other_cookies() {
        assert_eq!(session_cookie(&MetadataMap::new()), SessionCookie::Absent);
        let m = with_cookies(&["a=1; b=2", "theme=dark"]);
        assert_eq!(session_cookie(&m), SessionCookie::Absent);
    }

    #[test]
    fn present_among_others_and_across_headers() {
        let m = with_cookies(&["a=1", &format!("b=2; {SESSION_COOKIE}={ID}; c=3")]);
        assert_eq!(session_cookie(&m), SessionCookie::Present(ID.to_owned()));
        // The same value sent twice (a client resending its jar) is still one session.
        let m = with_cookies(&[
            &format!("{SESSION_COOKIE}={ID}"),
            &format!("{SESSION_COOKIE}={ID}"),
        ]);
        assert_eq!(session_cookie(&m), SessionCookie::Present(ID.to_owned()));
    }

    #[test]
    fn spaces_around_the_pair_are_ignored() {
        let m = with_cookies(&[&format!("  {SESSION_COOKIE} = {ID} ;x=y")]);
        assert_eq!(session_cookie(&m), SessionCookie::Present(ID.to_owned()));
    }

    #[test]
    fn malformed_values() {
        for bad in [
            format!("{SESSION_COOKIE}="),
            format!("{SESSION_COOKIE}=\"{ID}\""),
            format!("{SESSION_COOKIE}=a,b"),
            format!("{SESSION_COOKIE}={}", "a".repeat(MAX_COOKIE_VALUE + 1)),
        ] {
            assert_eq!(
                session_cookie(&with_cookies(&[&bad])),
                SessionCookie::Malformed,
                "{bad}"
            );
        }
        let two = with_cookies(&[&format!("{SESSION_COOKIE}={ID}; {SESSION_COOKIE}=other-id")]);
        assert_eq!(session_cookie(&two), SessionCookie::Malformed);
    }

    #[test]
    fn longest_value_is_read() {
        let long = "a".repeat(MAX_COOKIE_VALUE);
        let m = with_cookies(&[&format!("{SESSION_COOKIE}={long}")]);
        assert_eq!(session_cookie(&m), SessionCookie::Present(long));
    }

    #[test]
    fn set_cookie_round_trips() {
        let mut out = MetadataMap::new();
        assert!(set_session_cookie(&mut out, ID));
        let sent = out
            .get("set-cookie")
            .and_then(|v| v.to_str().ok())
            .expect("set-cookie");
        assert_eq!(sent, format!("{SESSION_COOKIE}={ID}"));
        // What the client sends back is what was set.
        let back = with_cookies(&[sent]);
        assert_eq!(session_cookie(&back), SessionCookie::Present(ID.to_owned()));
    }

    #[test]
    fn set_cookie_refuses_unsafe_ids() {
        let mut out = MetadataMap::new();
        assert!(!set_session_cookie(&mut out, "a; Domain=evil"));
        assert!(!set_session_cookie(&mut out, ""));
        assert!(out.get("set-cookie").is_none());
    }
}
