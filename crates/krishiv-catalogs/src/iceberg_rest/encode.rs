// SPDX-License-Identifier: Apache-2.0
//! The text forms the Iceberg REST protocol puts on the wire, matched to the Java client at `apache-iceberg-1.12.0`
//! (`RESTUtil`, `ResourcePaths`): a path segment is percent-encoded per RFC 3986 (only the unreserved characters
//! `A-Z a-z 0-9 - . _ ~` stay as they are; `RESTUtil.encodePathSegment`), a multi-level namespace in a path is its
//! encoded levels joined by the catalog's separator as the server advertised it (`%1F` by default;
//! `encodeNamespaceAsPathSegment`), and the same namespace in the `parent` query parameter is its raw levels joined by
//! the decoded separator (`namespaceToQueryParam`). A form body is `application/x-www-form-urlencoded` as Java's
//! `URLEncoder` writes it (`RESTUtil.encodeFormData`).
//!
//! Also here: a pre-scan of a JSON body's nesting depth, so a response nested deeper than the limit is refused
//! before a parser recurses into it (design P3-lakehouse §5.6).

use std::fmt::Write as _;

/// The namespace separator when the server's `/v1/config` names none: the unit separator, URL-encoded.
pub(crate) const DEFAULT_NAMESPACE_SEPARATOR: &str = "%1F";

fn is_unreserved(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~')
}

/// One path segment, percent-encoded per RFC 3986: every byte of the UTF-8 form except the unreserved characters.
#[must_use]
pub(crate) fn path_segment(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        if is_unreserved(b) {
            out.push(char::from(b));
        } else {
            // Writing to a `String` cannot fail.
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// A namespace as one path segment: each level encoded, joined by `separator` exactly as the server advertised it
/// (already URL-encoded, e.g. `%1F` or `.`).
#[must_use]
pub(crate) fn namespace_path(levels: &[String], separator: &str) -> String {
    levels
        .iter()
        .map(|l| path_segment(l))
        .collect::<Vec<_>>()
        .join(separator)
}

/// A namespace as the `parent` query value: the raw levels joined by the decoded separator. The query encoding
/// itself is the URL builder's.
///
/// # Errors
/// The separator does not percent-decode to UTF-8 text.
pub(crate) fn namespace_query(levels: &[String], separator: &str) -> Result<String, String> {
    let sep = percent_decode(separator)?;
    Ok(levels.join(&sep))
}

/// Decodes `%XX` escapes (no `+` rule: a separator is a path-style token).
///
/// # Errors
/// A malformed escape, or bytes that are not UTF-8.
pub(crate) fn percent_decode(text: &str) -> Result<String, String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes
                .get(i + 1..i + 3)
                .and_then(|h| std::str::from_utf8(h).ok())
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .ok_or_else(|| "a malformed percent escape".to_owned())?;
            out.push(hex);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| "an escape that is not UTF-8".to_owned())
}

/// One form value as Java's `URLEncoder` writes it: alphanumerics and `.-*_` as they are, a space as `+`, every
/// other byte of the UTF-8 form as `%XX`.
#[must_use]
pub(crate) fn form_value(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        match b {
            b' ' => out.push('+'),
            b if b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'*' | b'_') => {
                out.push(char::from(b));
            }
            b => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

/// A form body from `(key, value)` pairs, in order.
#[must_use]
pub(crate) fn form_body(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{}={}", form_value(k), form_value(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// Whether a separator the server advertised can go into a path as it is: non-empty, no `/`, `?` or `#`, and only
/// characters a path segment may carry (unreserved, sub-delimiters, `:`, `@` and `%` escapes that decode).
#[must_use]
pub(crate) fn separator_is_safe(separator: &str) -> bool {
    !separator.is_empty()
        && separator.bytes().all(|b| {
            is_unreserved(b)
                || matches!(
                    b,
                    b'%' | b'!'
                        | b'$'
                        | b'&'
                        | b'\''
                        | b'('
                        | b')'
                        | b'*'
                        | b'+'
                        | b','
                        | b';'
                        | b'='
                        | b':'
                        | b'@'
                )
        })
        && percent_decode(separator).is_ok_and(|s| !s.is_empty())
}

/// Whether a `prefix` the server advertised can go into the path as it is (Java joins it in raw): path characters
/// only, segments separated by single `/`, no `.` or `..` segment, no leading or trailing `/`. So a prefix can
/// never leave `/v1/` or reach another host.
#[must_use]
pub(crate) fn prefix_is_safe(prefix: &str) -> bool {
    !prefix.is_empty()
        && prefix.split('/').all(|seg| {
            !seg.is_empty() && seg != "." && seg != ".." && separator_is_safe(seg) && {
                percent_decode(seg).is_ok_and(|d| d != "." && d != ".." && !d.contains(['/', '\\']))
            }
        })
}

/// Whether a JSON text nests objects and arrays no deeper than `max_depth`. Strings (with their escapes) are
/// skipped, so brackets inside them do not count. Does not validate the JSON: the parser does that after.
#[must_use]
pub(crate) fn json_depth_within(bytes: &[u8], max_depth: usize) -> bool {
    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for &b in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > max_depth {
                    return false;
                }
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn path_segments_keep_only_unreserved_characters() {
        assert_eq!(path_segment("abc-._~XYZ09"), "abc-._~XYZ09");
        assert_eq!(path_segment("a b+c/d"), "a%20b%2Bc%2Fd");
        assert_eq!(path_segment("é"), "%C3%A9");
        assert_eq!(path_segment("%1F"), "%251F");
    }

    #[test]
    fn namespaces_join_encoded_levels_with_the_advertised_separator() {
        let ns = levels(&["accounting", "tax paid"]);
        assert_eq!(namespace_path(&ns, "%1F"), "accounting%1Ftax%20paid");
        assert_eq!(namespace_path(&ns, "."), "accounting.tax%20paid");
        assert_eq!(
            namespace_query(&ns, "%1F").unwrap(),
            "accounting\u{1f}tax paid"
        );
        assert_eq!(namespace_query(&ns, "%2E").unwrap(), "accounting.tax paid");
        assert!(namespace_query(&ns, "%zz").is_err());
    }

    #[test]
    fn form_values_follow_java_url_encoder() {
        assert_eq!(form_value("client_credentials"), "client_credentials");
        assert_eq!(form_value("a b*c~d"), "a+b*c%7Ed");
        assert_eq!(form_value("PRINCIPAL_ROLE:ALL"), "PRINCIPAL_ROLE%3AALL");
        assert_eq!(
            form_body(&[("grant_type", "client_credentials"), ("scope", "catalog")]),
            "grant_type=client_credentials&scope=catalog"
        );
    }

    #[test]
    fn unsafe_separators_and_prefixes_are_refused() {
        for ok in ["%1F", ".", "%2E", ":"] {
            assert!(separator_is_safe(ok), "{ok}");
        }
        for bad in ["", "/", "%2", "a?b", "#", "%FF"] {
            assert!(!separator_is_safe(bad), "{bad}");
        }
        for ok in ["ws", "ws/item.Lakehouse", "cat%20one", "a/b/c"] {
            assert!(prefix_is_safe(ok), "{ok}");
        }
        for bad in [
            "", "/a", "a/", "a//b", "..", "a/../b", "%2E%2E", "a%2Fb", "a?x", "a#x", "http://x",
            "a b",
        ] {
            assert!(!prefix_is_safe(bad), "{bad}");
        }
    }

    #[test]
    fn depth_counts_brackets_outside_strings_only() {
        assert!(json_depth_within(br#"{"a":[1,{"b":2}]}"#, 3));
        assert!(!json_depth_within(br#"{"a":[1,{"b":2}]}"#, 2));
        assert!(json_depth_within(br#"{"a":"[[[[{{{{\"]]]"}"#, 1));
        let deep = format!("{}{}", "[".repeat(65), "]".repeat(65));
        assert!(!json_depth_within(deep.as_bytes(), 64));
        let ok = format!("{}{}", "[".repeat(64), "]".repeat(64));
        assert!(json_depth_within(ok.as_bytes(), 64));
    }
}
