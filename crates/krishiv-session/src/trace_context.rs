// SPDX-License-Identifier: Apache-2.0
//! Inbound W3C trace context (P3-T068): the `traceparent` and `baggage` header parsers.

/// The remote span a W3C `traceparent` names: the trace, the caller's span
/// (the parent of the spans the engine starts) and the trace flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RemoteParent {
    pub(crate) trace_id: String,
    pub(crate) span_id: String,
    pub(crate) flags: u8,
}

/// [`RemoteParent`] of `header`, `None` for a malformed one (version `ff`, an all-zero id or wrong lengths).
pub(crate) fn parse_remote_parent(header: &str) -> Option<RemoteParent> {
    let hex = |s: &str, n: usize| {
        s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    // At most five pieces are looked at: a header of a million dashes costs nothing more.
    let parts: Vec<&str> = header.trim().splitn(5, '-').collect();
    let [version, trace, parent, flags, ..] = parts.as_slice() else {
        return None;
    };
    let zero = |s: &str| s.bytes().all(|b| b == b'0');
    // Version 00 has exactly four fields; later versions may add more.
    let shape = hex(version, 2) && *version != "ff" && (*version != "00" || parts.len() == 4);
    if !(shape
        && hex(trace, 32)
        && !zero(trace)
        && hex(parent, 16)
        && !zero(parent)
        && hex(flags, 2))
    {
        return None;
    }
    Some(RemoteParent {
        trace_id: (*trace).to_owned(),
        span_id: (*parent).to_owned(),
        flags: u8::from_str_radix(flags, 16).ok()?,
    })
}

/// Most `baggage` entries and bytes a session keeps from an inbound header.
/// The header is untrusted input: what is over either bound is dropped.
pub(crate) const MAX_BAGGAGE_ENTRIES: usize = 32;
pub(crate) const MAX_BAGGAGE_BYTES: usize = 4096;

/// A W3C `baggage` key: a non-empty RFC 7230 token.
fn is_baggage_key(k: &str) -> bool {
    !k.is_empty()
        && k.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// A W3C `baggage-octet`: printable ASCII except space, `"`, `,`, `;` and `\`
/// (anything else must be percent-encoded).
fn is_baggage_octet(b: u8) -> bool {
    matches!(b, 0x21 | 0x23..=0x2B | 0x2D..=0x3A | 0x3C..=0x5B | 0x5D..=0x7E)
}

/// The `key=value` entries of a W3C `baggage` header (properties after `;` are
/// dropped), at most [`MAX_BAGGAGE_ENTRIES`] within [`MAX_BAGGAGE_BYTES`]
/// bytes of header; an entry outside the W3C grammar (no `=`, a key that is not
/// a token, a value with a character that must be percent-encoded) is skipped.
pub(crate) fn parse_baggage(header: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut budget = MAX_BAGGAGE_BYTES;
    for entry in header.split(',') {
        if out.len() >= MAX_BAGGAGE_ENTRIES || entry.len() > budget {
            break;
        }
        budget -= entry.len() + 1.min(budget - entry.len());
        let pair = entry.split(';').next().unwrap_or_default();
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        if !is_baggage_key(k) || !v.bytes().all(is_baggage_octet) {
            continue;
        }
        out.push((k.to_owned(), v.to_owned()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baggage_keeps_only_well_formed_members() {
        assert_eq!(
            parse_baggage("a=b,bad key=v,k=\"v\",n=\u{e9},c=d%20e;prop=1"),
            [
                ("a".to_owned(), "b".to_owned()),
                ("c".to_owned(), "d%20e".to_owned())
            ]
        );
    }

    #[test]
    fn a_header_of_dashes_is_refused() {
        assert_eq!(parse_remote_parent(&"-".repeat(1_000_000)), None);
    }
}
