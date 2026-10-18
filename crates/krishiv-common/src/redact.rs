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
//!
//! Query text is not redacted by pattern: the SQL front end marks the
//! secret ranges from lexer tokens and builds a [`RedactedSql`], the only
//! form of query text that sinks accept (ADR 0038 update); only comments,
//! which are free text, go through [`redact`].

use std::borrow::Cow;
use std::ops::Range;

use crate::secret::{is_sensitive_key, semicolon_authority, url_secret_ranges};

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

/// What to do with a byte range of query text, for [`RedactedSql`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mark {
    /// A secret: replaced by [`REDACTED`].
    Secret(Range<usize>),
    /// Free text (a comment): passed through [`redact`].
    FreeText(Range<usize>),
}

/// Query text with its secrets replaced, built by a front end that knows
/// where they are (`krishiv_sql::redact_sql`). `Display`, `Debug` and
/// [`Self::as_str`] give the redacted text; [`Self::fragment`] the
/// redacted text of a byte range of the original, for error contexts.
#[derive(Clone, PartialEq, Eq)]
pub struct RedactedSql {
    text: String,
    /// Each replaced region: (original range, redacted range), in order.
    regions: Vec<(Range<usize>, Range<usize>)>,
}

impl RedactedSql {
    /// `sql` with every [`Mark`] applied. Marks out of range, inside an
    /// earlier mark or not on character boundaries are widened or skipped
    /// so that no part of a marked range survives.
    ///
    /// Only the SQL front end's token redactor (`krishiv_sql::redact_sql`)
    /// calls this: the marks must be every secret of the text, which only
    /// the lexer knows. `cargo xtask source-guard` rejects any other caller.
    #[doc(hidden)]
    #[must_use]
    pub fn from_marks_unchecked(sql: &str, mut marks: Vec<Mark>) -> Self {
        let range = |m: &Mark| match m {
            Mark::Secret(r) | Mark::FreeText(r) => r.clone(),
        };
        marks.sort_by_key(|m| range(m).start);
        let mut text = String::with_capacity(sql.len());
        let mut regions = Vec::new();
        let mut at = 0;
        for mark in marks {
            let r = range(&mark);
            let (start, end) = (floor(sql, r.start.max(at)), ceil(sql, r.end));
            if start >= end {
                continue;
            }
            let replaced = match mark {
                Mark::Secret(_) => Cow::Borrowed(REDACTED),
                Mark::FreeText(_) => redact(&sql[start..end]),
            };
            if let Cow::Borrowed(same) = replaced
                && same == &sql[start..end]
            {
                continue;
            }
            text.push_str(&sql[at..start]);
            let new_start = text.len();
            text.push_str(&replaced);
            regions.push((start..end, new_start..text.len()));
            at = end;
        }
        text.push_str(&sql[at..]);
        Self { text, regions }
    }

    /// `sql` hidden whole: for text no lexer has read (it reveals only that
    /// there was text).
    #[must_use]
    pub fn opaque(sql: &str) -> Self {
        Self::from_marks_unchecked(sql, vec![Mark::Secret(0..sql.len())])
    }

    /// The redacted text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The redacted text of the original byte range `range` (a span of the
    /// query): a secret it touches shows as [`REDACTED`].
    #[must_use]
    pub fn fragment(&self, range: Range<usize>) -> &str {
        let start = self.position(range.start, false);
        let end = self.position(range.end, true).max(start);
        self.text.get(start..end).unwrap_or_default()
    }

    /// Where original byte `pos` lands; inside a region, at its start (or
    /// end, for the end of a range).
    fn position(&self, pos: usize, end: bool) -> usize {
        let mut shift = (0_usize, 0_usize);
        for (old, new) in &self.regions {
            if pos <= old.start {
                break;
            }
            if pos < old.end {
                return if end { new.end } else { new.start };
            }
            shift = (old.end, new.end);
        }
        (pos - shift.0 + shift.1).min(self.text.len())
    }
}

/// The char boundary at or before `i`.
fn floor(s: &str, i: usize) -> usize {
    (0..=i.min(s.len()))
        .rev()
        .find(|&j| s.is_char_boundary(j))
        .unwrap_or(0)
}

/// The char boundary at or after `i`.
fn ceil(s: &str, i: usize) -> usize {
    (i.min(s.len())..=s.len())
        .find(|&j| s.is_char_boundary(j))
        .unwrap_or(s.len())
}

impl std::fmt::Display for RedactedSql {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::fmt::Debug for RedactedSql {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RedactedSql({:?})", self.text)
    }
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

/// URL credentials. A run of non-space text can hold several URLs (a
/// comma-separated list of paths or hosts), so the run is split at every URL
/// start (`scheme://` or `jdbc:`) and each segment is parsed on its own. The
/// segments are disjoint and each scheme is scanned back only to the
/// previous segment, so the work stays linear in `text` (`jdbc:jdbc:...` or
/// `x://x://...` become short segments).
fn urls(text: &str, out: &mut Vec<Range<usize>>) {
    let mut run_start = 0;
    while run_start < text.len() {
        let run_end = run_end(text, run_start, ends_url);
        for seg in url_segments(&text[run_start..run_end]) {
            let (start, end) = (run_start + seg.start, run_start + seg.end);
            out.extend(
                url_secret_ranges(&text[start..end])
                    .into_iter()
                    .map(|r| start + r.start..start + r.end),
            );
        }
        run_start = run_end + 1;
    }
}

/// The URL segments of one run: from each URL start to the next one (or
/// the end of the run), in order. A scheme-like `x://` or `jdbc:` inside an
/// earlier URL's userinfo (a password may hold `://`) is not a new start.
/// Linear: the userinfo ends come from two precomputed tables.
fn url_segments(run: &str) -> Vec<Range<usize>> {
    let bytes = run.as_bytes();
    // (start, where the part after the scheme begins), in order.
    let mut starts: Vec<(usize, usize)> = Vec::new();
    let mut floor = 0;
    for (i, _) in run.match_indices("://") {
        // The scheme runs back from `://` over scheme characters, but never
        // past the previous match, which bounds the backward scans.
        let start = bytes[floor..i]
            .iter()
            .rposition(|&b| !(b.is_ascii_alphanumeric() || b"+-.:".contains(&b)))
            .map_or(floor, |p| floor + p + 1);
        if start < i {
            starts.push((start, i + 3));
        }
        floor = i + 3;
    }
    starts.extend(run.match_indices("jdbc:").map(|(i, _)| (i, i + 5)));
    // At one start keep the `scheme://` entry (the larger `after`): it is
    // the one `url_secret_ranges` parses.
    starts.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    starts.dedup_by_key(|s| s.0);
    // next_qf[p] / next_qfs[p]: the first `?` or `#` (and `;`) at or after
    // p; last_at[p]: the last `@` before p. The userinfo boundary matches
    // `url_secret_ranges`, including `;` for SQL Server and jTDS URLs.
    let mut next_qf = vec![run.len(); run.len() + 1];
    let mut next_qfs = vec![run.len(); run.len() + 1];
    for p in (0..run.len()).rev() {
        next_qf[p] = if matches!(bytes[p], b'?' | b'#') {
            p
        } else {
            next_qf[p + 1]
        };
        next_qfs[p] = if matches!(bytes[p], b'?' | b'#' | b';') {
            p
        } else {
            next_qfs[p + 1]
        };
    }
    let mut last_at = vec![None; run.len() + 1];
    for p in 0..run.len() {
        last_at[p + 1] = if bytes[p] == b'@' {
            Some(p)
        } else {
            last_at[p]
        };
    }
    let mut segments: Vec<Range<usize>> = Vec::new();
    let mut skip_until = 0;
    for (start, after) in starts {
        if start < skip_until {
            continue; // inside the previous URL's userinfo
        }
        if let Some(prev) = segments.last_mut() {
            prev.end = start;
        }
        segments.push(start..run.len());
        let semicolon = after >= start + 3
            && run[start..after].ends_with("://")
            && semicolon_authority(&run[start..after - 3]);
        let boundary = if semicolon {
            next_qfs[after]
        } else {
            next_qf[after]
        };
        if let Some(at) = last_at[boundary].filter(|&at| at >= after) {
            skip_until = at + 1;
        }
    }
    segments
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
            Some(&q @ (b'"' | b'\'')) => v + 1..quoted_end(bytes, v + 1, q),
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

/// The end of a quoted value whose content starts at `from`: the closing
/// `q`, skipping `\\`-escaped characters and doubled quotes (`''` in SQL),
/// or the end of the text when it is unterminated.
fn quoted_end(bytes: &[u8], from: usize, q: u8) -> usize {
    let mut i = from;
    while let Some(&b) = bytes.get(i) {
        if b == b'\\' {
            i += 2;
        } else if b == q {
            if bytes.get(i + 1) == Some(&q) {
                i += 2;
            } else {
                return i;
            }
        } else {
            i += 1;
        }
    }
    bytes.len()
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
    use super::{Mark, RedactedSql, redact};

    #[test]
    fn marks_replace_secrets_and_fragments_follow() {
        let sql = "SELECT 'Canary-7f3a9c' /* token=Canary-7f3a9c */ FROM t é";
        let r = RedactedSql::from_marks_unchecked(
            sql,
            vec![
                Mark::FreeText(23..48),
                Mark::Secret(8..21),
                Mark::Secret(9..12),
                Mark::FreeText(0..6),
            ],
        );
        let want = "SELECT '*********(redacted)' /* token=*********(redacted) */ FROM t é";
        assert_eq!(r.as_str(), want);
        assert_eq!(r.to_string(), want);
        assert!(!format!("{r:?}").contains(CANARY));
        assert_eq!(r.fragment(0..6), "SELECT");
        assert_eq!(r.fragment(7..22), "'*********(redacted)'");
        assert_eq!(r.fragment(10..12), REDACTED_MARK);
        assert_eq!(r.fragment(49..55), "FROM t");
        assert_eq!(r.fragment(56..99), "é");
        let plain = RedactedSql::from_marks_unchecked("SELECT 1", Vec::new());
        assert_eq!(plain.fragment(7..8), "1");
        assert_eq!(RedactedSql::opaque("SELECT 1").as_str(), REDACTED_MARK);
    }

    const REDACTED_MARK: &str = super::REDACTED;

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
            // Every URL of a comma-separated list, not just the first.
            (
                "https://h1/?a=1,https://u2:Canary-7f3a9c@h2/",
                "https://h1/?a=1,https://u2:*********(redacted)@h2/",
            ),
            // Raw `/`, `;` and `&` inside a password.
            (
                "s3a://AKIA:ab/c+d@Canary-7f3a9c@bucket/k",
                "s3a://AKIA:*********(redacted)@bucket/k",
            ),
            (
                "postgresql://u:pa;Canary-7f3a9c@h/db",
                "postgresql://u:*********(redacted)@h/db",
            ),
            (
                "postgresql://u:pa&Canary-7f3a9c@h/db?ssl=1",
                "postgresql://u:*********(redacted)@h/db?ssl=1",
            ),
            // A userinfo without `:` may be a token.
            (
                "https://Canary-7f3a9c@github.com/o/r",
                "https://*********(redacted)@github.com/o/r",
            ),
            // A SQL Server URL ends its userinfo at `;`, so a later URL in
            // the same run is still found.
            (
                "jdbc:sqlserver://h:1433;user=a,https://u:Canary-7f3a9c@h2/",
                "jdbc:sqlserver://h:1433;user=a,https://u:*********(redacted)@h2/",
            ),
            // A password holding `://` is not split into two URLs.
            (
                "postgresql://u:abc://Canary-7f3a9c@host/db",
                "postgresql://u:*********(redacted)@host/db",
            ),
            (
                "jdbc:postgresql://u:x,jdbc:Canary-7f3a9c@h/db",
                "jdbc:postgresql://u:*********(redacted)@h/db",
            ),
            // Doubled and backslash-escaped quotes stay inside the value.
            (
                "password = 'abc''Canary-7f3a9c' and x = 1",
                "password = '*********(redacted)' and x = 1",
            ),
            (
                r#"token: "a\"Canary-7f3a9c" rest"#,
                r#"token: "*********(redacted)" rest"#,
            ),
            // Opaque JDBC with a raw `;` in the password.
            (
                "jdbc:oracle:thin:scott/ti;Canary-7f3a9c@h:1521:orcl",
                "jdbc:oracle:thin:scott/*********(redacted)@h:1521:orcl",
            ),
            // jTDS: credentials only as `;` parameters.
            (
                "jdbc:jtds:sqlserver://h:1433/db;user=u;password=p@Canary-7f3a9c",
                "jdbc:jtds:sqlserver://h:1433/db;user=u;password=*********(redacted)",
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
