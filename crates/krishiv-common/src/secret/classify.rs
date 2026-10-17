// SPDX-License-Identifier: Apache-2.0
//! Sensitive keys and the credentials inside URLs (lists in the parent
//! module's docs).

use std::ops::Range;

/// Words that make a key sensitive on their own.
const WORDS: &[&str] = &["pwd", "passwd", "sas", "sig", "signature", "jwt"];

/// Stems that make a key sensitive anywhere in its joined words.
const STEMS: &[&str] = &[
    "password",
    "passphrase",
    "secret",
    "token",
    "credential",
    "accesskey",
    "accountkey",
    "privatekey",
    "apikey",
    "keystorekey",
    "sessionkey",
    "jaasconfig",
    "authorization",
];

/// URL parameter names whose values are credentials, besides sensitive
/// keys.
const PARAMS: &[&str] = &[
    "pwd",
    "passwd",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "api_key",
    "apikey",
    "key",
    "sig",
    "signature",
    "x-amz-signature",
    "x-amz-credential",
    "x-amz-security-token",
    "x-goog-signature",
    "x-goog-credential",
];

/// Whether an option key looks sensitive (the fail-safe default list in the
/// module docs). Case-insensitive.
#[must_use]
pub fn is_sensitive_key(key: &str) -> bool {
    let mut joined = String::with_capacity(key.len());
    let mut word = String::new();
    let mut hit = false;
    let mut after_lower = false;
    for c in key.chars() {
        let alnum = c.is_alphanumeric();
        if !alnum || (after_lower && c.is_uppercase()) {
            hit |= WORDS.contains(&word.as_str());
            word.clear();
        }
        if alnum {
            word.extend(c.to_lowercase());
            joined.extend(c.to_lowercase());
        }
        after_lower = c.is_lowercase() || c.is_numeric();
    }
    hit || WORDS.contains(&word.as_str()) || STEMS.iter().any(|s| joined.contains(s))
}

/// Characters that start a parameter.
const SEPARATORS: [char; 4] = ['?', '&', ';', '#'];

/// The byte ranges of the credentials in `url`, sorted and disjoint;
/// empty when `url` is not a URL or holds none. Linear in `url.len()`.
///
/// Two scans, merged, so that an ambiguous URL is over-redacted, never
/// leaked:
/// - the userinfo runs to the last `@` before the query or fragment (`?`
///   or `#`, and `;` for SQL Server URLs), so a password may hold raw `/`
///   (base64 secret keys do) as well as raw
///   `@`, `:`, `;` or `&`; without a `:` the whole userinfo is hidden (it may
///   be a token, as in `https://<token>@host`);
/// - `;`/`?`/`&` parameters are scanned from the start of the URL, so a
///   credential parameter whose value holds `@`
///   (`jdbc:sqlserver://h;password=p@ss`) is hidden even when the userinfo
///   scan above misreads where the userinfo ends.
pub(crate) fn url_secret_ranges(url: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let params_from = if let Some(i) = url.find("://").filter(|&i| is_scheme(&url[..i])) {
        // `scheme://user:password@host...`
        let start = i + 3;
        // The userinfo search stops at the query or fragment, not at `/`:
        // secret keys may hold `/`. A path holding `@` is then over-redacted,
        // never leaked. SQL Server URLs (`jdbc:sqlserver://h:1433;...`, jTDS)
        // carry credentials only as `;` parameters, so `;` ends it too.
        let scheme = url[..i].to_ascii_lowercase();
        let semicolon_params = scheme.ends_with("sqlserver") || scheme.contains("jtds");
        let auth_end = url[start..]
            .find(|c: char| matches!(c, '?' | '#') || (semicolon_params && c == ';'))
            .map_or(url.len(), |e| start + e);
        if let Some(at) = url[start..auth_end].rfind('@') {
            let userinfo = start..start + at;
            match url[userinfo.clone()].find(':') {
                Some(colon) => push(&mut out, start + colon + 1..userinfo.end),
                None => push(&mut out, userinfo),
            }
        }
        start
    } else if url
        .get(..5)
        .is_some_and(|p| p.eq_ignore_ascii_case("jdbc:"))
    {
        // Opaque: `jdbc:<subprotocol>:...:user/password@host`, e.g.
        // `jdbc:oracle:thin:scott/tiger@//h:1521/svc`. The user starts after
        // the subprotocol names (`oracle:`, `thin:`).
        let head = url.find(['?', '#']).unwrap_or(url.len());
        if let Some(at) = url[..head].rfind('@') {
            let mut user = 5;
            while let Some(colon) = url[user..at].find(':') {
                let name = &url[user..user + colon];
                if name.is_empty()
                    || !name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                {
                    break;
                }
                user += colon + 1;
            }
            if let Some(slash) = url[user..at].find('/') {
                push(&mut out, user + slash + 1..at);
            }
        }
        5
    } else {
        return out;
    };
    params(url, params_from, &mut out);
    merged(out)
}

/// `ranges` sorted, with overlapping or touching ranges joined.
fn merged(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_unstable_by_key(|r| r.start);
    let mut out: Vec<Range<usize>> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

/// The secret parameter values after `from`. A value ends at the next
/// separator, except that a value starting with `{` runs to its closing `}`
/// (`}}` escapes one), as SQL Server JDBC URLs quote `;` in values.
fn params(url: &str, from: usize, out: &mut Vec<Range<usize>>) {
    let mut at = from;
    while let Some(off) = url[at..].find(SEPARATORS) {
        let start = at + off + 1;
        let next = url[start..].find(SEPARATORS).map(|e| start + e);
        let eq = url[start..next.unwrap_or(url.len())]
            .find('=')
            .map(|e| start + e);
        let Some(eq) = eq else {
            at = next.unwrap_or(url.len());
            continue;
        };
        let value = eq + 1;
        let end = if url[value..].starts_with('{') {
            braced_end(url, value + 1)
        } else {
            next.unwrap_or(url.len())
        };
        if is_secret_param(&url[start..eq]) {
            push(out, value..end);
        }
        at = end;
    }
}

/// The end of a `{...}` value whose content starts at `from`: just past the
/// `}` that is not doubled, or the end of `url`.
fn braced_end(url: &str, from: usize) -> usize {
    let bytes = url.as_bytes();
    let mut i = from;
    while let Some(&b) = bytes.get(i) {
        if b == b'}' {
            if bytes.get(i + 1) == Some(&b'}') {
                i += 2;
                continue;
            }
            return i + 1;
        }
        i += 1;
    }
    url.len()
}

fn push(out: &mut Vec<Range<usize>>, r: Range<usize>) {
    if !r.is_empty() {
        out.push(r);
    }
}

/// `jdbc:postgresql`, `https`, `s3a`: a letter, then letters, digits and
/// `+-.:`.
fn is_scheme(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || "+-.:".contains(c))
}

fn is_secret_param(name: &str) -> bool {
    let name = percent_decoded(name).to_lowercase();
    PARAMS.contains(&name.as_str()) || is_sensitive_key(&name)
}

/// `name` with `%XX` escapes decoded (invalid escapes kept as they are).
fn percent_decoded(name: &str) -> String {
    let bytes = name.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while let Some(&b) = bytes.get(i) {
        let escaped = (b == b'%')
            .then(|| bytes.get(i + 1..i + 3))
            .flatten()
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u8::from_str_radix(h, 16).ok());
        match escaped {
            Some(v) => {
                out.push(v);
                i += 3;
            }
            None => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
