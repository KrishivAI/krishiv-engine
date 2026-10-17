// SPDX-License-Identifier: Apache-2.0
//! Defence-in-depth redaction of free text (ADR 0038 #3).
//!
//! Options are typed ([`crate::secret`]) and never go through here. This
//! is for text that cannot be typed: query text with literals, and error
//! messages from external systems. Every sink that shows such text (EXPLAIN,
//! profiles, logs and tracing, the UI, query history, debug bundles,
//! `system.*` tables) calls [`redact`].
//!
//! It replaces with `*********(redacted)`:
//! - URL credentials, as [`crate::secret::SecretUrl`] splits them, in every
//!   `scheme://...` or `jdbc:...` run of non-space text;
//! - the value after `key=`, `key: ` or `"key": ` when the key is
//!   sensitive ([`crate::secret::is_sensitive_key`]), quoted values whole;
//! - the token after `Bearer` or `Basic`.
//!
//! It is a pattern match: it may hide too much and cannot find a secret
//! that nothing marks.

use std::borrow::Cow;
use std::ops::Range;

use crate::secret::{is_sensitive_key, url_secret_ranges};

pub use crate::secret::REDACTED;

/// `text` with the secrets it can find replaced by [`REDACTED`]; borrowed
/// when there are none.
#[must_use]
pub fn redact(text: &str) -> Cow<'_, str> {
    let mut found = Vec::new();
    urls(text, &mut found);
    pairs(text, &mut found);
    schemes(text, &mut found);
    if found.is_empty() {
        return Cow::Borrowed(text);
    }
    found.sort_by_key(|r| r.start);
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for r in found {
        if r.end <= at {
            continue;
        }
        let start = r.start.max(at);
        out.push_str(&text[at..start]);
        if start == r.start {
            out.push_str(REDACTED);
        }
        at = r.end;
    }
    out.push_str(&text[at..]);
    Cow::Owned(out)
}

/// Characters that end an unquoted value or a URL.
fn ends_value(b: u8) -> bool {
    b.is_ascii_whitespace() || b"\"'`<>,;&)]}".contains(&b)
}

fn ends_url(b: u8) -> bool {
    b.is_ascii_whitespace() || b"\"'`<>".contains(&b)
}

fn is_key_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"_.-".contains(&b)
}

/// The end of the run starting at `from` that `stop` does not end.
fn run_end(text: &str, from: usize, stop: fn(u8) -> bool) -> usize {
    text.as_bytes()
        .get(from..)
        .and_then(|rest| rest.iter().position(|&b| stop(b)))
        .map_or(text.len(), |n| from + n)
}

/// URL credentials. Each run of non-space text is parsed once, from its
/// first URL start, so the work is linear in `text` (a run such as
/// `jdbc:jdbc:...` or `x://x://...` is not rescanned per match).
fn urls(text: &str, out: &mut Vec<Range<usize>>) {
    let bytes = text.as_bytes();
    let mut run_start = 0;
    while run_start < text.len() {
        let run_end = run_end(text, run_start, ends_url);
        let run = &text[run_start..run_end];
        let scheme = run.find("://").map(|i| {
            bytes[run_start..run_start + i]
                .iter()
                .rposition(|&b| !(b.is_ascii_alphanumeric() || b"+-.:".contains(&b)))
                .map_or(0, |p| p + 1)
        });
        let jdbc = run.find("jdbc:");
        let first = match (scheme, jdbc) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if let Some(first) = first {
            let start = run_start + first;
            out.extend(
                url_secret_ranges(&text[start..run_end])
                    .into_iter()
                    .map(|r| start + r.start..start + r.end),
            );
        }
        run_start = run_end + 1;
    }
}

/// `key=value` pairs with sensitive keys. A separator inside a value
/// already redacted is skipped, so every byte is scanned a bounded number
/// of times.
fn pairs(text: &str, out: &mut Vec<Range<usize>>) {
    let bytes = text.as_bytes();
    let mut covered = 0;
    for (sep, &b) in bytes.iter().enumerate() {
        if (b != b'=' && b != b':') || sep < covered {
            continue;
        }
        if b == b':' && bytes.get(sep + 1..sep + 3) == Some(b"//") {
            continue;
        }
        // The key: before the separator, past spaces and a closing quote.
        let mut end = sep;
        while end > 0 && bytes[end - 1] == b' ' {
            end -= 1;
        }
        if end > 0 && b"\"'".contains(&bytes[end - 1]) {
            end -= 1;
        }
        let start = bytes[..end]
            .iter()
            .rposition(|&c| !is_key_char(c))
            .map_or(0, |p| p + 1);
        if start == end || !is_sensitive_key(&text[start..end]) {
            continue;
        }
        // The value: after spaces; quoted values end at the closing quote.
        let mut v = sep + 1;
        while bytes.get(v) == Some(&b' ') {
            v += 1;
        }
        let range = match bytes.get(v) {
            Some(&q @ (b'"' | b'\'')) => {
                let end = bytes[v + 1..]
                    .iter()
                    .position(|&c| c == q)
                    .map_or(text.len(), |n| v + 1 + n);
                v + 1..end
            }
            _ => {
                let mut end = run_end(text, v, ends_value);
                let word = &text[v..end];
                if word.eq_ignore_ascii_case("bearer") || word.eq_ignore_ascii_case("basic") {
                    let token = end + text[end..].len() - text[end..].trim_start().len();
                    end = run_end(text, token, ends_value);
                }
                v..end
            }
        };
        covered = range.end;
        if !range.is_empty() && !text[range.start..].starts_with(REDACTED) {
            out.push(range);
        }
    }
}

/// The token after an HTTP authorization scheme (`Bearer x`, `Basic x`).
fn schemes(text: &str, out: &mut Vec<Range<usize>>) {
    let lower = text.to_ascii_lowercase();
    for word in ["bearer ", "basic "] {
        let mut covered = 0;
        for (i, _) in lower.match_indices(word) {
            if i < covered {
                continue;
            }
            let mut v = i + word.len();
            while text.as_bytes().get(v) == Some(&b' ') {
                v += 1;
            }
            let end = run_end(text, v, ends_value);
            covered = end;
            if v < end && !text[v..].starts_with(REDACTED) {
                out.push(v..end);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::redact;

    const CANARY: &str = "Canary-7f3a9c";

    #[test]
    fn finds_marked_secrets() {
        let cases = [
            (
                "connect to jdbc:postgresql://bob:Canary-7f3a9c@db/x?ssl=true failed",
                "connect to jdbc:postgresql://bob:*********(redacted)@db/x?ssl=true failed",
            ),
            (
                "GET https://b.s3.amazonaws.com/k?X-Amz-Signature=Canary-7f3a9c&X-Amz-Expires=60",
                "GET https://b.s3.amazonaws.com/k?X-Amz-Signature=*********(redacted)&X-Amz-Expires=60",
            ),
            (
                "password=Canary-7f3a9c, user=bob",
                "password=*********(redacted), user=bob",
            ),
            (
                r#"{"client_secret": "Canary-7f3a9c x", "id": 1}"#,
                r#"{"client_secret": "*********(redacted)", "id": 1}"#,
            ),
            (
                "Authorization: Bearer Canary-7f3a9c",
                "Authorization: *********(redacted)",
            ),
            (
                "got bearer Canary-7f3a9c back",
                "got bearer *********(redacted) back",
            ),
            (
                "fs.s3a.secret.key : 'Canary-7f3a9c' used",
                "fs.s3a.secret.key : '*********(redacted)' used",
            ),
            (
                "jdbc:sqlserver://h:1433;password=p@ssCanary-7f3a9c;user=u",
                "jdbc:sqlserver://h:1433;password=*********(redacted);user=u",
            ),
            (
                "jdbc:oracle:thin:scott/Canary-7f3a9c@h:1521:orcl",
                "jdbc:oracle:thin:scott/*********(redacted)@h:1521:orcl",
            ),
        ];
        for (text, want) in cases {
            let got = redact(text);
            assert_eq!(got, want);
            assert!(!got.contains(CANARY));
        }
    }

    #[test]
    fn leaves_other_text_alone() {
        for text in [
            "select * from t where a = 1",
            "password authentication failed for user \"bob\"",
            "at 12:30 see https://spark.apache.org/docs?x=1",
            "",
            "ünïcödé = ok",
        ] {
            assert!(
                matches!(redact(text), std::borrow::Cow::Borrowed(_)),
                "{text}"
            );
        }
    }

    /// Adversarial inputs without spaces take time linear in their size:
    /// quadrupling the input at most quadruples the work (plus noise), where
    /// a quadratic scan would multiply it by 16.
    #[test]
    fn work_is_linear_on_adversarial_runs() {
        for unit in ["jdbc:", "x://", "token=", "a:b@", "?sig=", "bearer "] {
            let time = |n: usize| {
                let text = unit.repeat(n / unit.len());
                (0..3)
                    .map(|_| {
                        let start = std::time::Instant::now();
                        std::hint::black_box(redact(&text));
                        start.elapsed()
                    })
                    .min()
                    .unwrap_or_default()
            };
            let (small, large) = (time(256 << 10), time(1 << 20));
            assert!(
                large < small * 10 + std::time::Duration::from_millis(5),
                "{unit}: {small:?} -> {large:?}"
            );
        }
    }

    #[test]
    fn is_idempotent() {
        let once = redact("token=abc https://u:p@h/?sig=s");
        assert_eq!(
            once,
            "token=*********(redacted) https://u:*********(redacted)@h/?sig=*********(redacted)"
        );
        assert_eq!(redact(&once), once);
    }
}
