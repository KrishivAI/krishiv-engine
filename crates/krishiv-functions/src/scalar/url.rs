// SPDX-License-Identifier: Apache-2.0
//! URL functions: `parse_url` and `try_parse_url` (Spark 4.2's
//! `urlExpressions`). `url_encode`, `url_decode` and `try_url_decode` are
//! the string-codec family's.
//!
//! `parse_url(url, part[, key])` follows `java.net.URI`: the parts `HOST`,
//! `PATH`, `QUERY`, `REF`, `PROTOCOL`, `FILE`, `AUTHORITY` and `USERINFO`
//! are the URI's (raw) components, matched case-sensitively, anything else
//! is NULL. A `key` is only read for `QUERY` (any other part with a key is
//! NULL); it searches the raw query with Spark's `(&|^)key=([^&]*)`
//! pattern, where `key` is a Java regular expression (so `a.b` also matches
//! `aXb=`, and `(a)(b)` reads group 2), returning group 2 of the first
//! match. An invalid key pattern fails the query, as Spark's untyped
//! `PatternSyntaxException` does. A URL the URI parser
//! rejects is `INVALID_URL` under ANSI and NULL otherwise; `try_parse_url`
//! is NULL whatever the mode. A NULL argument makes the row NULL.
//!
//! The key pattern is compiled through the same Java-to-`fancy_regex`
//! translation the `regexp` functions use, with the same backtracking cap:
//! one compile per distinct key (a literal key compiles once per call).
//! A pattern that does not build is `INTERNAL_ERROR` (Spark's
//! `PatternSyntaxException` has no error class); a backtracking cap hit is
//! `KRISHIV_REGEXP_BACKTRACK_LIMIT_EXCEEDED`, as for `regexp`.
//!
//! The URI splitter below is hand-written against OpenJDK's `java.net.URI`
//! (no regex in the splitter itself, no new dependency): strict character
//! tables per component (`%` must open two hex digits; visible non-ASCII
//! of U+0081 and above except spaces and controls is allowed), first-`@`
//! userinfo, registry-based authorities (kept, with a NULL host) versus
//! server-based ones, strict IPv6 literals (brackets kept in the host),
//! and ports that must be digits (an empty port reads as absent).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use fancy_regex::{Regex, RegexBuilder};
use krishiv_common::error::{InvalidUrl, KrishivError, KrishivRegexpBacktrackLimitExceeded};

use super::args::{Inputs, internal, is_string};
use super::regex::{BACKTRACK_LIMIT, translate};
use super::{BindArg, BindOptions, Binding, FamilyResult, bad_types, listed};

/// Every name this family binds.
pub(super) const NAMES: &[&str] = &["parse_url", "try_parse_url"];

/// The shape of an argument list: `S` string, anything else `?`.
fn shape(args: &[BindArg]) -> String {
    args.iter()
        .map(|a| if is_string(&a.data_type()) { 'S' } else { '?' })
        .collect()
}

/// Binds the URL function `name`.
pub(super) fn bind(name: &str, args: &[BindArg], opts: &BindOptions) -> FamilyResult {
    let listed = listed(NAMES, name)?;
    let shape = shape(args);
    let text = DataType::Utf8;
    let binding = match (listed, shape.as_str()) {
        ("parse_url" | "try_parse_url", "SS" | "SSS") => {
            // `parse_url` raises on an invalid URL under ANSI;
            // `try_parse_url` is NULL whatever the mode.
            let fail = opts.ansi && !listed.starts_with("try_");
            let keyed = shape.len() == 3;
            Binding::new(text, move |x: &Inputs<'_>| {
                let (urls, parts) = (x.strings(0)?, x.strings(1)?);
                let keys = keyed.then(|| x.strings(2)).transpose()?;
                let mut out = Vec::with_capacity(urls.len());
                // The compiled key pattern of the last distinct key: a
                // literal key compiles once per call.
                let mut keys_cache: Option<KeyPattern> = None;
                for i in 0..urls.len() {
                    // A NULL key (like a NULL anything else) makes NULL.
                    let mut key = "";
                    let mut null_key = false;
                    if let Some(k) = &keys {
                        if k.is_null(i) {
                            null_key = true;
                        } else {
                            key = k.value(i);
                        }
                    }
                    if urls.is_null(i) || parts.is_null(i) || null_key {
                        out.push(None);
                        continue;
                    }
                    let (url, part) = (urls.value(i), parts.value(i));
                    // A key is only read for `QUERY`; any other part with
                    // a key is NULL without parsing the URL (as Spark's
                    // evaluator checks the part first).
                    if keyed && part != "QUERY" {
                        out.push(None);
                        continue;
                    }
                    match split_uri(url) {
                        Err(()) if fail => {
                            return Err(InvalidUrl {
                                url: url.to_owned(),
                            }
                            .into());
                        }
                        Err(()) => out.push(None),
                        Ok(uri) => {
                            out.push(if keyed {
                                // No query component: NULL, the key is not read.
                                match uri.query {
                                    Some(q) => key_value(q, key, &mut keys_cache)?,
                                    None => None,
                                }
                            } else {
                                extract_part(&uri, part)
                            });
                        }
                    }
                }
                Ok(Arc::new(arrow_array::StringArray::from_iter(
                    out.iter().map(|o| o.as_deref()),
                )) as ArrayRef)
            })
        }
        _ => return Some((listed, Err(bad_types(listed, args)))),
    };
    Some((listed, Ok(binding)))
}

/// A URI split the way `java.net.URI` splits it: `None` is an undefined
/// component (an opaque URI has no path, query or fragment; a hierarchical
/// one always has a path, possibly empty).
#[derive(Debug, Clone, Copy)]
struct Uri<'a> {
    scheme: Option<&'a str>,
    authority: Option<&'a str>,
    userinfo: Option<&'a str>,
    host: Option<&'a str>,
    path: Option<&'a str>,
    query: Option<&'a str>,
    fragment: Option<&'a str>,
}

/// Whether `c` is ASCII alphabetic.
fn is_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

/// Whether `c` may continue a scheme name (`alpha *( alpha | digit |
/// "+" | "-" | "." )`).
fn is_scheme_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'+' | b'-' | b'.')
}

/// A visible non-ASCII character `java.net.URI` lets through unescaped:
/// U+0081 and above, neither a space nor a control.
fn is_allowed_other(c: char) -> bool {
    u32::from(c) > 128 && !c.is_whitespace() && !c.is_control()
}

/// The ASCII members of one of `java.net.URI`'s character masks.
fn in_mask(c: u8, mask: &[u8]) -> bool {
    mask.contains(&c)
}

/// The ASCII members of one of `java.net.URI`'s character masks, written
/// out (`in_mask`).
///
/// Unreserved (`A-Za-z0-9-_.!~*'()`), reserved (`;/?:@&=+$,[]`) and each
/// component's set: opaque parts, queries and fragments take both;
/// paths add `:@&=+$,` and `;/`; userinfo adds `;:&=+$,`; registry names
/// add `$,;:@&=+`; server-based authorities the userinfo characters with
/// `.:@[]` (with a literal `%` when the authority holds `]`, for an IPv6
/// zone id, whose `%` is not an escape).
const URIC: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'();/?:@&=+$,[]";
/// Unreserved, `:@&=+$,` and `;/`: paths.
const PATH: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'():@&=+$,;/";
/// Unreserved and `;:&=+$,`: userinfo.
const USERINFO: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'();:&=+$,";
/// Unreserved and `$,;:@&=+`: registry names.
const REG_NAME: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'()$,;:@&=+";
/// Userinfo characters with `.:@[]`: server-based authorities.
const SERVER: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'();:&=+$,.:@[]";
/// As [`SERVER`], with a literal `%` (an authority holding `]` may hold an
/// IPv6 zone id, whose `%` is not an escape).
const SERVER_PERCENT: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.!~*'();:&=+$,.:@[]%";

/// Scans `s` from `pos` while characters match `ascii` (byte members), a
/// valid `%XX` escape, or an allowed non-ASCII character. Returns the first
/// position past the run, or `Err(())` for a malformed escape pair (as
/// `java.net.URI`'s escape scan throws).
fn scan_while(s: &str, mut pos: usize, ascii: &[u8]) -> Result<usize, ()> {
    let bytes = s.as_bytes();
    while pos < bytes.len() {
        let b = bytes[pos];
        if b < 128 {
            if in_mask(b, ascii) {
                pos += 1;
                continue;
            }
            if b == b'%' {
                let ok = [bytes.get(pos + 1), bytes.get(pos + 2)]
                    .into_iter()
                    .flatten()
                    .all(u8::is_ascii_hexdigit);
                if pos + 3 <= bytes.len() && ok {
                    pos += 3;
                    continue;
                }
                return Err(());
            }
            return Ok(pos);
        }
        let c = s[pos..].chars().next().unwrap_or('\0');
        if is_allowed_other(c) {
            pos += c.len_utf8();
        } else {
            return Ok(pos);
        }
    }
    Ok(pos)
}

/// `scan_while`, then `Err(())` unless it consumed `s[pos..end)`.
fn check_chars(s: &str, pos: usize, end: usize, ascii: &[u8]) -> Result<(), ()> {
    if scan_while(s, pos, ascii)? == end {
        Ok(())
    } else {
        Err(())
    }
}

/// The first index of a character of `stop` at or after `pos`.
fn scan_to(s: &str, pos: usize, stop: &[u8]) -> usize {
    s.as_bytes()
        .iter()
        .skip(pos)
        .position(|b| stop.contains(b))
        .map_or(s.len(), |p| pos + p)
}

/// Splits `s` the way `new java.net.URI(s)` does.
///
/// # Errors
/// `Err(())` where the URI parser throws `URISyntaxException`: a bad
/// scheme, an empty opaque part, an illegal character or escape, a
/// malformed authority.
fn split_uri(s: &str) -> Result<Uri<'_>, ()> {
    let n = s.len();
    let p = scan_to(s, 0, b"/?#:");
    if p < n && s.as_bytes()[p] == b':' {
        if p == 0 || !is_alpha(s.as_bytes()[0]) {
            return Err(());
        }
        if !s.as_bytes()[1..p].iter().all(|b| is_scheme_char(*b)) {
            return Err(());
        }
        let scheme = &s[..p];
        let p = p + 1;
        if p < n && s.as_bytes()[p] == b'/' {
            hierarchical(s, Some(scheme), p)
        } else {
            let q = scan_to(s, p, b"#");
            if q <= p {
                return Err(());
            }
            check_chars(s, p, q, URIC)?;
            fragment_of(s, q).map(|fragment| Uri {
                scheme: Some(scheme),
                authority: None,
                userinfo: None,
                host: None,
                path: None,
                query: None,
                fragment,
            })
        }
    } else {
        hierarchical(s, None, 0)
    }
}

/// The fragment at `p` (`p` at `#` or the end).
fn fragment_of(s: &str, p: usize) -> Result<Option<&str>, ()> {
    if p < s.len() && s.as_bytes()[p] == b'#' {
        check_chars(s, p + 1, s.len(), URIC)?;
        Ok(Some(&s[p + 1..]))
    } else {
        Ok(None)
    }
}

/// `[//authority]<path>[?<query>][#<fragment>]` from `p`.
fn hierarchical<'a>(s: &'a str, scheme: Option<&'a str>, mut p: usize) -> Result<Uri<'a>, ()> {
    let n = s.len();
    let mut authority = None;
    let mut userinfo = None;
    let mut host = None;
    if s[p..].starts_with("//") {
        p += 2;
        let q = scan_to(s, p, b"/?#");
        if q > p {
            (authority, userinfo, host) = parse_authority(s, p, q)?;
            p = q;
        } else if q >= n {
            // `//` with nothing after it.
            return Err(());
        }
    }
    let q = scan_to(s, p, b"?#");
    check_chars(s, p, q, PATH)?;
    let path = &s[p..q];
    p = q;
    let mut query = None;
    if p < n && s.as_bytes()[p] == b'?' {
        p += 1;
        let q = scan_to(s, p, b"#");
        check_chars(s, p, q, URIC)?;
        query = Some(&s[p..q]);
        p = q;
    }
    Ok(Uri {
        scheme,
        authority,
        userinfo,
        host,
        path: Some(path),
        query,
        fragment: fragment_of(s, p)?,
    })
}

/// A parsed authority: the whole range, the userinfo and the host.
type Authority<'a> = (Option<&'a str>, Option<&'a str>, Option<&'a str>);

/// A parsed server authority: the userinfo and host ranges.
type ServerParts = (Option<(usize, usize)>, Option<(usize, usize)>);

/// The authority `s[start..end]`: the whole range, the userinfo (before
/// the first `@`) and the host, or `Err(())`.
///
/// A registry-based authority (no server parse) is kept with a NULL
/// userinfo and host; a server parse that fails where a registry parses
/// falls back to one. Anything else is malformed.
fn parse_authority(s: &str, start: usize, end: usize) -> Result<Authority<'_>, ()> {
    let a = &s[start..end];
    let mask = if a.contains(']') {
        SERVER_PERCENT
    } else {
        SERVER
    };
    let server_ok = scan_while(a, 0, mask)? == a.len();
    let reg_ok = scan_while(a, 0, REG_NAME)? == a.len();
    // A registry name is a subset of the server characters, so a
    // registry-only authority cannot reach here; anything the server
    // parse refuses but the registry takes falls through to it below.
    if server_ok && let Ok((userinfo, host)) = parse_server(s, start, end) {
        return Ok((
            Some(a),
            userinfo.map(|u| &s[u.0..u.1]),
            host.map(|h| &s[h.0..h.1]),
        ));
    }
    if reg_ok {
        return Ok((Some(a), None, None));
    }
    Err(())
}

/// `[<userinfo>@]<host>[:<port>]` in `s[start..end]`: the userinfo and host
/// ranges. A failed server parse is `Err(())`, whatever its cause.
fn parse_server(s: &str, start: usize, end: usize) -> Result<ServerParts, ()> {
    let bytes = s.as_bytes();
    let mut p = start;
    let mut userinfo = None;
    if let Some(i) = s[p..end].find('@') {
        let i = p + i;
        check_chars(s, p, i, USERINFO)?;
        userinfo = Some((p, i));
        p = i + 1;
    }
    let host: Option<(usize, usize)>;
    if p < end && bytes[p] == b'[' {
        p += 1;
        let Some(q) = s[p..end].find(']') else {
            return Err(());
        };
        let q = p + q;
        if let Some(r) = s[p..q].find('%') {
            let r = p + r;
            parse_ipv6_ref(&s[p..r])?;
            if r + 1 == q {
                return Err(());
            }
            check_scope_id(&s[r + 1..q])?;
        } else {
            parse_ipv6_ref(&s[p..q])?;
        }
        host = Some((p - 1, q + 1));
        p = q + 1;
    } else if let Some(hend) = scan_ipv4_end(s, p, end)
        && (hend == end || bytes[hend] == b':')
    {
        host = Some((p, hend));
        p = hend;
    } else {
        let hend = parse_hostname(s, p, end)?;
        host = Some((p, hend));
        p = hend;
    }
    if p < end && bytes[p] == b':' {
        p += 1;
        if p < end {
            if !bytes[p..end].iter().all(u8::is_ascii_digit) {
                return Err(());
            }
            s[p..end].parse::<i32>().map_err(|_| ())?;
        }
        p = end;
    }
    // Anything trailing the host (junk after `]`, a second `@`) is not a
    // port.
    if p < end {
        return Err(());
    }
    Ok((userinfo, host))
}

/// A hostname in `s[p..end]`: domain labels of alphanumerics and dashes
/// (never ending in a dash), the last of several starting with a letter.
/// Returns the end position.
fn parse_hostname(s: &str, start: usize, end: usize) -> Result<usize, ()> {
    let bytes = s.as_bytes();
    let mut p = start;
    let mut last = None;
    loop {
        let q = scan_ascii(s, p, end, |b| b.is_ascii_alphanumeric());
        if q == p {
            break;
        }
        last = Some(p);
        p = q;
        let q = scan_ascii(s, p, end, |b| b.is_ascii_alphanumeric() || b == b'-');
        if q > p {
            if bytes[q - 1] == b'-' {
                return Err(());
            }
            p = q;
        }
        if p < end && bytes[p] == b'.' {
            p += 1;
        } else {
            break;
        }
        if p == end {
            break;
        }
    }
    if p < end && bytes[p] != b':' {
        return Err(());
    }
    let Some(l) = last else {
        return Err(());
    };
    if l > start && !is_alpha(bytes[l]) {
        return Err(());
    }
    Ok(p)
}

/// Scans ASCII bytes satisfying `f` from `p` to `end`.
fn scan_ascii(s: &str, mut p: usize, end: usize, f: fn(u8) -> bool) -> usize {
    let bytes = s.as_bytes();
    while p < end && f(bytes[p]) {
        p += 1;
    }
    p
}

/// An IPv4 address in `s[p..end]`, or `None`.
fn scan_ipv4_end(s: &str, p: usize, end: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let m = scan_ascii(s, p, end, |b| b.is_ascii_digit() || b == b'.');
    if m == p {
        return None;
    }
    let mut q = p;
    for i in 0..4 {
        q = scan_byte(s, q, m)?;
        if i < 3 {
            if q < m && bytes[q] == b'.' {
                q += 1;
            } else {
                return None;
            }
        }
    }
    (q == m).then_some(q)
}

/// A byte's worth of decimal digits in `s[p..m]`, or `None`.
fn scan_byte(s: &str, p: usize, m: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let q = scan_ascii(s, p, m, |b| b.is_ascii_digit());
    if q == p {
        return None;
    }
    let mut i = p;
    while i < q && bytes[i] == b'0' {
        i += 1;
    }
    if q - i < 3 {
        return Some(q);
    }
    if q - i > 3 {
        return None;
    }
    let v: u32 = s[p..q].bytes().fold(0, |v, b| v * 10 + u32::from(b - b'0'));
    (v <= 255).then_some(q)
}

/// An IPv6 literal's content (between the brackets, before any `%`
/// scope): strict groups, `::` compression and an IPv4 tail.
fn parse_ipv6_ref(s: &str) -> Result<(), ()> {
    let mut bytes = 0_usize;
    let mut compressed = false;
    let mut p = 0;
    match scan_hex_seq(s, 0, &mut bytes)? {
        Some(q) => {
            p = q;
            if s[p..].starts_with("::") {
                compressed = true;
                p = scan_hex_post(s, p + 2, &mut bytes)?;
            } else if s[p..].starts_with(':') {
                p = take_ipv4(s, p + 1)?;
                bytes += 4;
            }
        }
        None => {
            if s[p..].starts_with("::") {
                compressed = true;
                p = scan_hex_post(s, p + 2, &mut bytes)?;
            }
        }
    }
    if p < s.len() || bytes > 16 || !compressed && bytes < 16 || compressed && bytes == 16 {
        return Err(());
    }
    Ok(())
}

/// A hex group run from `start`: groups of 1-4 hex digits joined by single
/// colons, stopping before `::` or an IPv4 tail. `None` when no group
/// starts there.
fn scan_hex_seq(s: &str, start: usize, bytes: &mut usize) -> Result<Option<usize>, ()> {
    let mut p = start;
    let mut q = scan_ascii(s, p, s.len(), |b| b.is_ascii_hexdigit());
    if q == p {
        return Ok(None);
    }
    if s[q..].starts_with('.') {
        return Ok(None);
    }
    if q - p > 4 {
        return Err(());
    }
    *bytes += 2;
    p = q;
    while p < s.len() {
        if s.as_bytes()[p] != b':' {
            break;
        }
        if s[p + 1..].starts_with(':') {
            break;
        }
        p += 1;
        q = scan_ascii(s, p, s.len(), |b| b.is_ascii_hexdigit());
        if q == p {
            return Err(());
        }
        if s[q..].starts_with('.') {
            p -= 1;
            break;
        }
        if q - p > 4 {
            return Err(());
        }
        *bytes += 2;
        p = q;
    }
    Ok(Some(p))
}

/// After `::`: a hex run with an optional IPv4 tail, or an IPv4 address.
fn scan_hex_post(s: &str, start: usize, bytes: &mut usize) -> Result<usize, ()> {
    let mut p = start;
    if p == s.len() {
        return Ok(p);
    }
    match scan_hex_seq(s, p, bytes)? {
        Some(q) => {
            p = q;
            if s[p..].starts_with(':') {
                p = take_ipv4(s, p + 1)?;
                *bytes += 4;
            }
        }
        None => {
            p = take_ipv4(s, p)?;
            *bytes += 4;
        }
    }
    Ok(p)
}

/// An IPv4 address covering exactly `s[p..]`.
fn take_ipv4(s: &str, p: usize) -> Result<usize, ()> {
    let m = scan_ascii(s, p, s.len(), |b| b.is_ascii_digit() || b == b'.');
    if m == p || m != s.len() {
        return Err(());
    }
    let mut q = p;
    for i in 0..4 {
        q = scan_byte(s, q, m).ok_or(())?;
        if i < 3 {
            if q < m && s.as_bytes()[q] == b'.' {
                q += 1;
            } else {
                return Err(());
            }
        }
    }
    if q != m {
        return Err(());
    }
    Ok(q)
}

/// A `%` scope id: ASCII alphanumerics, `_` and `.` (no escapes).
fn check_scope_id(s: &str) -> Result<(), ()> {
    if s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
    {
        Ok(())
    } else {
        Err(())
    }
}

/// The URI's `part`: the (raw) component, or `None` when undefined or when
/// `part` names nothing.
fn extract_part(uri: &Uri<'_>, part: &str) -> Option<String> {
    if part == "FILE" {
        return match (uri.path, uri.query) {
            (Some(path), Some(query)) => Some(format!("{path}?{query}")),
            (Some(path), None) => Some(path.to_owned()),
            _ => None,
        };
    }
    match part {
        "HOST" => uri.host,
        "PATH" => uri.path,
        "QUERY" => uri.query,
        "REF" => uri.fragment,
        "PROTOCOL" => uri.scheme,
        "AUTHORITY" => uri.authority,
        "USERINFO" => uri.userinfo,
        _ => None,
    }
    .map(str::to_owned)
}

/// Spark's `(&|^)key=([^&]*)` search of the raw `query`, where `key` is a
/// Java regular expression: group 2 of the first match (`None` when the
/// pattern does not match or group 2 takes no part, as Spark's
/// `m.group(2)` reads null then).
///
/// # Errors
/// `INTERNAL_ERROR` for a key that is not a valid Java pattern (Spark
/// itself throws an untyped `PatternSyntaxException`, which has no error
/// class); `KRISHIV_REGEXP_BACKTRACK_LIMIT_EXCEEDED` past the
/// backtracking cap, as for `regexp`.
fn key_value(
    query: &str,
    key: &str,
    cache: &mut Option<KeyPattern>,
) -> Result<Option<String>, KrishivError> {
    if cache.as_ref().is_none_or(|c| c.key != key) {
        *cache = Some(KeyPattern::new(key)?);
    }
    let compiled = cache
        .as_ref()
        .ok_or_else(|| internal("no key pattern".to_owned()))?;
    let found = compiled
        .for_query(query)
        .captures_from_pos(query, 0)
        .map_err(|e| compiled.backtrack(e))?;
    Ok(found
        .and_then(|caps| caps.get(2))
        .map(|m| m.as_str().to_owned()))
}

/// The compiled `(&|^)key=([^&]*)` search for one `key`.
struct KeyPattern {
    /// The key as written, for the error message.
    key: String,
    /// Matches `$` (in the key) at the very end of the query only.
    plain: Regex,
    /// Matches `$` before a final line terminator as well; only used for
    /// queries that end with one (as the `regexp` engine does).
    terminated: Option<Regex>,
}

impl KeyPattern {
    /// Compiles `key` the way Spark's `ParseUrlEvaluator.getPattern` does.
    fn new(key: &str) -> Result<Self, KrishivError> {
        let source = format!("(&|^){key}=([^&]*)");
        let invalid = || {
            internal(format!(
                "parse_url: the key is not a valid pattern: {key:?}"
            ))
        };
        let translated = translate(&source, false).map_err(|_| invalid())?;
        let build = |text: &str| {
            RegexBuilder::new(text)
                .backtrack_limit(BACKTRACK_LIMIT)
                .build()
                .map_err(|_| invalid())
        };
        Ok(Self {
            key: key.to_owned(),
            terminated: translated
                .has_dollar
                .then(|| translate(&source, true).map_err(|_| invalid()))
                .transpose()?
                .map(|t| build(&t.text))
                .transpose()?,
            plain: build(&translated.text)?,
        })
    }

    /// The regex for a query ending (or not) in a line terminator.
    fn for_query(&self, query: &str) -> &Regex {
        match &self.terminated {
            Some(t) if query.ends_with(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']) => t,
            _ => &self.plain,
        }
    }

    /// [`BACKTRACK_LIMIT`] exceeded while matching the key.
    fn backtrack(&self, e: fancy_regex::Error) -> KrishivError {
        match e {
            fancy_regex::Error::RuntimeError(fancy_regex::RuntimeError::BacktrackLimitExceeded) => {
                KrishivRegexpBacktrackLimitExceeded {
                    pattern: format!("'(&|^){}=([^&]*)'", self.key),
                    function_name: "`parse_url`".to_owned(),
                    limit: BACKTRACK_LIMIT.to_string(),
                }
                .into()
            }
            e => internal(format!("parse_url key {:?}: {e}", self.key)),
        }
    }
}
