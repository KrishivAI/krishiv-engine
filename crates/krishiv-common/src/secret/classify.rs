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

/// The byte ranges of the credentials in `url`, in order and disjoint;
/// empty when `url` is not a URL or holds none.
pub(crate) fn url_secret_ranges(url: &str) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let params_from = if let Some(i) = url.find("://").filter(|&i| is_scheme(&url[..i])) {
        let start = i + 3;
        let end = url[start..]
            .find(['/', '?', '#'])
            .map_or(url.len(), |e| start + e);
        match url[start..end].rfind('@') {
            Some(at) => {
                if let Some(colon) = url[start..start + at].find(':') {
                    push(&mut out, start + colon + 1..start + at);
                }
                start + at + 1
            }
            None => start,
        }
    } else if url
        .get(..5)
        .is_some_and(|p| p.eq_ignore_ascii_case("jdbc:"))
    {
        // Opaque, e.g. `jdbc:oracle:thin:user/password@host:1521:sid`.
        match url.find('@') {
            Some(at) => {
                let user = url[..at].rfind(':').map_or(0, |c| c + 1);
                if let Some(slash) = url[user..at].find('/') {
                    push(&mut out, user + slash + 1..at);
                }
                at + 1
            }
            None => 5,
        }
    } else {
        return out;
    };
    const SEPARATORS: [char; 4] = ['?', '&', ';', '#'];
    let mut at = params_from;
    while let Some(off) = url[at..].find(SEPARATORS) {
        let start = at + off + 1;
        let end = url[start..]
            .find(SEPARATORS)
            .map_or(url.len(), |e| start + e);
        if let Some(eq) = url[start..end].find('=')
            && is_secret_param(&url[start..start + eq])
        {
            push(&mut out, start + eq + 1..end);
        }
        at = end;
    }
    out
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
