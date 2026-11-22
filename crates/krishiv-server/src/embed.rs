// SPDX-License-Identifier: Apache-2.0
//! Embedding the web UI in another page (P3-T071): the origins that may frame it, the signed read-only link that
//! shows one query's profile without a token, and the theme an embedding console may pass.
//!
//! Threat model (P3 security notes, T-W3): a signed link has a bounded life and one scope (one query, read-only);
//! it is an HMAC-SHA256 over the query id, the session when the link names one, and the expiry, with a key made at
//! start-up, so it ends with the process and cannot be forged from another query's link. Only the embeddable views are ever framable, and only by the
//! origins the operator lists (`frame-ancestors`; the default is `'none'`).

use std::time::Duration;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// The query parameter that turns the UI into the embedded view.
pub const EMBED_PARAM: &str = "embed";

/// Most bytes of a theme (a signed link carries it in the address).
pub const MAX_THEME_BYTES: usize = 2048;

/// The settings that make a view embeddable.
#[derive(Clone)]
pub struct EmbedPolicy {
    origins: Vec<String>,
    hide_attribution: bool,
    ttl: Duration,
    key: [u8; 32],
}

impl std::fmt::Debug for EmbedPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedPolicy")
            .field("origins", &self.origins)
            .field("hide_attribution", &self.hide_attribution)
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl Default for EmbedPolicy {
    fn default() -> Self {
        Self::new(Vec::new(), false, Duration::from_secs(300))
    }
}

impl EmbedPolicy {
    /// A policy for `origins` (already checked by [`parse_origins`]) with a fresh signing key. If the operating
    /// system has no random source the key stays zero and every link is refused by [`EmbedPolicy::sign`].
    #[must_use]
    pub fn new(origins: Vec<String>, hide_attribution: bool, ttl: Duration) -> Self {
        let mut key = [0u8; 32];
        if getrandom::fill(&mut key).is_err() {
            key = [0u8; 32];
        }
        Self {
            origins,
            hide_attribution,
            ttl,
            key,
        }
    }

    /// The origins that may frame an embeddable view.
    #[must_use]
    pub fn origins(&self) -> &[String] {
        &self.origins
    }

    /// Whether the "Powered by Krishiv" mark is dropped.
    #[must_use]
    pub fn hide_attribution(&self) -> bool {
        self.hide_attribution
    }

    /// The longest life of a link.
    #[must_use]
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// The `Content-Security-Policy` `frame-ancestors` source list of an embeddable view.
    #[must_use]
    pub fn frame_ancestors(&self) -> String {
        if self.origins.is_empty() {
            "'none'".to_owned()
        } else {
            self.origins.join(" ")
        }
    }

    /// The signature of a link for `query_id` that ends at `expires` (seconds since the epoch). `None` when the
    /// server has no random key.
    #[must_use]
    pub fn sign(&self, query_id: &str, expires: u64) -> Option<String> {
        self.sign_in(query_id, None, expires)
    }

    /// [`EmbedPolicy::sign`] for the query of one session: two sessions may run a query under one id, and a link
    /// signed with `session` opens that session's query only. `None` signs for the id alone.
    #[must_use]
    pub fn sign_in(&self, query_id: &str, session: Option<&str>, expires: u64) -> Option<String> {
        if self.key == [0u8; 32] {
            return None;
        }
        Some(hex(&hmac_sha256(
            &self.key,
            &message(query_id, session, expires),
        )))
    }

    /// Whether `sig` is the signature of a link for `query_id` ending at `expires`, and `expires` is after `now`
    /// and no more than the policy's life away (a link cannot outlive the life this process issues links for).
    #[must_use]
    pub fn verify(&self, query_id: &str, expires: u64, sig: &str, now: u64) -> bool {
        self.verify_in(query_id, None, expires, sig, now)
    }

    /// [`EmbedPolicy::verify`] for a link that names `session` (or none): a link signed for one session is not
    /// one for another, nor for the id alone, and a link for the id alone gains no session.
    #[must_use]
    pub fn verify_in(
        &self,
        query_id: &str,
        session: Option<&str>,
        expires: u64,
        sig: &str,
        now: u64,
    ) -> bool {
        if expires <= now || expires > now.saturating_add(self.ttl.as_secs()) {
            return false;
        }
        let Some(want) = self.sign_in(query_id, session, expires) else {
            return false;
        };
        constant_time_eq(want.as_bytes(), sig.as_bytes())
    }
}

/// The signed text. A link for an id alone keeps the first form. One that names a session carries both lengths,
/// so no id and session pair reads as another pair (either may hold a line break: both come from an address).
fn message(query_id: &str, session: Option<&str>, expires: u64) -> Vec<u8> {
    match session {
        None => format!("krishiv-embed-v1\n{query_id}\n{expires}"),
        Some(session) => format!(
            "krishiv-embed-v2\n{}\n{}\n{query_id}\n{session}\n{expires}",
            query_id.len(),
            session.len()
        ),
    }
    .into_bytes()
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// HMAC-SHA256 (RFC 2104).
fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(k.map(|b| b ^ 0x36));
    inner.update(msg);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(k.map(|b| b ^ 0x5c));
    outer.update(inner);
    outer.finalize().into()
}

/// Parses `krishiv.ui.allowedOrigins`: comma-separated `scheme://host[:port]`, scheme `http` or `https`, no path,
/// wildcard, quote or space. Empty entries are skipped.
///
/// # Errors
/// A message naming the first entry that is not an origin.
pub fn parse_origins(list: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for raw in list.split(',') {
        let o = raw.trim().to_ascii_lowercase();
        let o = o.as_str();
        if o.is_empty() {
            continue;
        }
        if !is_origin(o) {
            return Err(format!(
                "krishiv.ui.allowedOrigins: `{o}` is not an origin (scheme://host[:port])"
            ));
        }
        // The browser reports an origin without its default port, so the theme check compares without it.
        let o = o
            .strip_suffix(":443")
            .filter(|_| o.starts_with("https://"))
            .or_else(|| o.strip_suffix(":80").filter(|_| o.starts_with("http://")))
            .unwrap_or(o);
        if !out.iter().any(|x| x == o) {
            out.push(o.to_owned());
        }
    }
    Ok(out)
}

/// Whether `o` is `http://` or `https://` plus a host name or bracketed IPv6 address and an optional port.
#[must_use]
pub fn is_origin(o: &str) -> bool {
    let Some(rest) = o
        .strip_prefix("https://")
        .or_else(|| o.strip_prefix("http://"))
    else {
        return false;
    };
    let (host, port) = if rest.starts_with('[') {
        match rest.find(']') {
            Some(end) => (&rest[..=end], &rest[end + 1..]),
            None => return false,
        }
    } else {
        match rest.find(':') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        }
    };
    let host_ok = if host.starts_with('[') {
        host.len() > 2
            && host.contains(':')
            && host[1..host.len() - 1]
                .bytes()
                .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.')
    } else {
        !host.is_empty()
            && !host.starts_with(['.', '-'])
            && !host.contains("..")
            && host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
    };
    let port_ok = port.is_empty()
        || port.strip_prefix(':').is_some_and(|p| {
            p.len() <= 5
                && p.bytes().all(|b| b.is_ascii_digit())
                && p.parse::<u32>().is_ok_and(|n| (1..=65535).contains(&n))
        });
    host_ok && port_ok
}

/// The theme tokens an embedding console may set: colours of the brand `--k-*` custom properties.
pub const THEME_COLOURS: &[&str] = &[
    "accent",
    "bg",
    "bg-subtle",
    "border",
    "border-strong",
    "code-bg",
    "error",
    "error-bg",
    "focus",
    "info",
    "info-bg",
    "muted",
    "primary",
    "primary-hover",
    "on-primary",
    "success",
    "success-bg",
    "surface",
    "surface-2",
    "text",
    "text-2",
    "warning",
    "warning-bg",
];

/// The font tokens (names only).
pub const THEME_FONTS: &[&str] = &["font-display", "font-mono", "font-text"];

/// The radius tokens.
pub const THEME_RADII: &[&str] = &["radius-sm", "radius-md", "radius-lg", "radius-xl"];

/// A theme that passed [`validate_theme`]: only these keys and shapes reach the page.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Theme {
    /// `light`, `dark` or `auto`.
    pub mode: Option<String>,
    /// Custom property (without `--k-`) to a checked value.
    pub tokens: Vec<(String, String)>,
}

impl Theme {
    /// The theme as the JSON object the page applies.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        if let Some(mode) = &self.mode {
            m.insert("mode".to_owned(), Value::String(mode.clone()));
        }
        let mut t = Map::new();
        for (k, v) in &self.tokens {
            t.insert(k.clone(), Value::String(v.clone()));
        }
        m.insert("tokens".to_owned(), Value::Object(t));
        Value::Object(m)
    }
}

/// Checks a theme: `{"mode": "light"|"dark"|"auto", "tokens": {name: value}}` where a name is one of the colour,
/// font or radius tokens above and a value is a colour (`#rgb[a]`, `#rrggbb[aa]`), a font name (letters, digits,
/// space and dash, up to 64) or a length (`0` to `64` px or rem). No URLs, no CSS text, no unknown key.
///
/// # Errors
/// A message naming the first thing refused.
pub fn validate_theme(v: &Value) -> Result<Theme, String> {
    let Value::Object(top) = v else {
        return Err("the theme is not an object".to_owned());
    };
    let mut theme = Theme::default();
    for (k, val) in top {
        match k.as_str() {
            "mode" => match val.as_str() {
                Some(m @ ("light" | "dark" | "auto")) => theme.mode = Some(m.to_owned()),
                _ => return Err("mode is light, dark or auto".to_owned()),
            },
            "tokens" => {
                let Value::Object(tokens) = val else {
                    return Err("tokens is not an object".to_owned());
                };
                for (name, value) in tokens {
                    let Some(s) = value.as_str() else {
                        return Err(format!("token {name}: the value is not a string"));
                    };
                    let ok = if THEME_COLOURS.contains(&name.as_str()) {
                        is_colour(s)
                    } else if THEME_FONTS.contains(&name.as_str()) {
                        is_font_name(s)
                    } else if THEME_RADII.contains(&name.as_str()) {
                        is_length(s)
                    } else {
                        return Err(format!("token {name}: not a theme token"));
                    };
                    if !ok {
                        return Err(format!("token {name}: the value is not allowed"));
                    }
                    theme.tokens.push((name.clone(), s.to_owned()));
                }
            }
            other => return Err(format!("{other}: not a theme field")),
        }
    }
    Ok(theme)
}

fn is_colour(s: &str) -> bool {
    s.strip_prefix('#').is_some_and(|h| {
        matches!(h.len(), 3 | 4 | 6 | 8) && h.bytes().all(|b| b.is_ascii_hexdigit())
    })
}

fn is_font_name(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b' ' || b == b'-')
        && !s.starts_with(' ')
}

fn is_length(s: &str) -> bool {
    let num = s
        .strip_suffix("px")
        .or_else(|| s.strip_suffix("rem"))
        .unwrap_or("");
    let mut parts = num.splitn(2, '.');
    let int = parts.next().unwrap_or("");
    let frac = parts.next();
    !int.is_empty()
        && int.len() <= 2
        && int.bytes().all(|b| b.is_ascii_digit())
        && frac.is_none_or(|f| (1..=3).contains(&f.len()) && f.bytes().all(|b| b.is_ascii_digit()))
        && num.parse::<f64>().is_ok_and(|n| (0.0..=64.0).contains(&n))
}

/// Percent-decodes a query-string value (`+` is a space); `None` for a bad escape or non-UTF-8 bytes.
#[must_use]
pub fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let h = s.get(i + 1..i + 3)?;
                if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
                    return None;
                }
                out.push(u8::from_str_radix(h, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// Percent-encodes `s` for a query-string value.
#[must_use]
pub fn percent_encode(s: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy() -> EmbedPolicy {
        EmbedPolicy::new(
            vec!["https://console.example.com".to_owned()],
            false,
            Duration::from_secs(600),
        )
    }

    #[test]
    fn hmac_matches_the_rfc_4231_vectors() {
        let mac = hmac_sha256(&[0x0b; 20], b"Hi There");
        assert_eq!(
            hex(&mac),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        // A key longer than the block (case 6).
        let mac = hmac_sha256(
            &[0xaa; 131],
            b"Test Using Larger Than Block-Size Key - Hash Key First",
        );
        assert_eq!(
            hex(&mac),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn a_link_verifies_for_its_query_only_and_until_it_expires() {
        let p = policy();
        let sig = p.sign("q1", 1_000).expect("key");
        assert!(p.verify("q1", 1_000, &sig, 500));
        assert!(!p.verify("q2", 1_000, &sig, 500), "another query");
        assert!(!p.verify("q1", 1_001, &sig, 500), "another expiry");
        assert!(!p.verify("q1", 1_000, &sig, 1_000), "expired");
        assert!(!p.verify("q1", 1_000, &sig, 2_000), "long expired");
        assert!(!p.verify("q1", 1_000, "", 500), "no signature");
        assert!(!p.verify("q1", 1_000, &sig[..sig.len() - 1], 500), "short");
    }

    #[test]
    fn a_link_for_one_session_is_not_one_for_another_or_for_the_id_alone() {
        let p = policy();
        let sig = p.sign_in("q1", Some("s1"), 1_000).expect("key");
        assert!(p.verify_in("q1", Some("s1"), 1_000, &sig, 500));
        assert!(!p.verify_in("q1", Some("s2"), 1_000, &sig, 500), "another");
        assert!(
            !p.verify_in("q1", None, 1_000, &sig, 500),
            "session dropped"
        );
        assert!(!p.verify("q1", 1_000, &sig, 500), "session dropped");
        assert!(
            !p.verify_in("q1", Some("s1"), 1_000, &sig, 1_000),
            "expired"
        );
        let bare = p.sign("q1", 1_000).expect("key");
        assert_eq!(p.sign_in("q1", None, 1_000).as_deref(), Some(&*bare));
        assert!(!p.verify_in("q1", Some("s1"), 1_000, &bare, 500), "added");
        assert!(!p.verify_in("q1", Some(""), 1_000, &bare, 500), "empty");
    }

    #[test]
    fn an_id_and_a_session_do_not_read_as_another_pair() {
        let p = policy();
        // Without the lengths both pairs would sign the same lines.
        let sig = p.sign_in("a\nb", Some("c"), 1_000).expect("key");
        assert!(!p.verify_in("a", Some("b\nc"), 1_000, &sig, 500));
        // Nor does a session-less id that spells the second form's lines.
        let sig = p.sign_in("q", Some("s"), 1_000).expect("key");
        assert!(!p.verify("1\n1\nq\ns", 1_000, &sig, 500));
    }

    #[test]
    fn an_expiry_beyond_the_life_is_refused_even_when_signed() {
        let p = policy();
        let sig = p.sign("q1", 10_000).expect("key");
        assert!(!p.verify("q1", 10_000, &sig, 100), "ttl is 600 s");
        assert!(p.verify("q1", 10_000, &sig, 9_500));
    }

    #[test]
    fn another_server_cannot_sign_for_this_one() {
        let a = policy();
        let b = policy();
        let sig = a.sign("q1", 1_000).expect("key");
        assert!(!b.verify("q1", 1_000, &sig, 500));
    }

    #[test]
    fn frame_ancestors_default_to_none() {
        assert_eq!(EmbedPolicy::default().frame_ancestors(), "'none'");
        assert_eq!(policy().frame_ancestors(), "https://console.example.com");
    }

    #[test]
    fn odd_origins_are_refused_and_default_ports_are_dropped() {
        for bad in [
            "https://[.]",
            "https://a..b",
            "https://a.example:0",
            "https://a.example:99999",
        ] {
            assert!(!is_origin(bad), "{bad}");
        }
        assert_eq!(
            parse_origins("https://c.example:443, http://d.example:80,http://e.example:443,https://f.example:8443")
                .unwrap(),
            vec![
                "https://c.example",
                "http://d.example",
                "http://e.example:443",
                "https://f.example:8443"
            ]
        );
    }

    #[test]
    fn origins_are_checked() {
        assert_eq!(
            parse_origins("https://a.example.com, HTTP://B.example:8080,,https://a.example.com")
                .unwrap(),
            vec!["https://a.example.com", "http://b.example:8080"]
        );
        assert_eq!(parse_origins("https://[::1]:3000").unwrap().len(), 1);
        for bad in [
            "*",
            "https://*.example.com",
            "https://a.example.com/path",
            "a.example.com",
            "ftp://a.example.com",
            "https://a.example.com;script-src",
            "https://a b",
            "https://",
            "https://a.example.com:",
            "https://a.example.com:123456",
            "https://-a",
            "https://'self'",
        ] {
            assert!(parse_origins(bad).is_err(), "{bad}");
        }
        assert_eq!(parse_origins("").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn a_theme_holds_colours_fonts_and_radii_only() {
        let t = validate_theme(&json!({
            "mode": "dark",
            "tokens": {"accent": "#ff8800", "bg": "#0a0b", "font-text": "Inter", "radius-md": "6px"}
        }))
        .unwrap();
        assert_eq!(t.mode.as_deref(), Some("dark"));
        assert_eq!(t.tokens.len(), 4);
        for bad in [
            json!({"mode": "sepia"}),
            json!({"logo": "x"}),
            json!({"tokens": {"accent": "url(https://evil/x)"}}),
            json!({"tokens": {"accent": "red"}}),
            json!({"tokens": {"accent": "#12"}}),
            json!({"tokens": {"font-text": "Inter; } body{display:none"}}),
            json!({"tokens": {"font-text": "x\"y"}}),
            json!({"tokens": {"radius-md": "999px"}}),
            json!({"tokens": {"radius-md": "1e1px"}}),
            json!({"tokens": {"radius-md": "6"}}),
            json!({"tokens": {"space-1": "4px"}}),
            json!({"tokens": {"accent": 5}}),
            json!({"tokens": []}),
            json!("dark"),
        ] {
            assert!(validate_theme(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn percent_coding_round_trips() {
        let s = "{\"a\":\"b c\"}&x=é";
        assert_eq!(percent_decode(&percent_encode(s)).as_deref(), Some(s));
        assert_eq!(percent_decode("%zz"), None);
        assert_eq!(percent_decode("%+f"), None, "a sign is not a hex digit");
        assert_eq!(percent_decode("%e9"), None, "not UTF-8");
        assert_eq!(percent_decode("a+b").as_deref(), Some("a b"));
    }
}
