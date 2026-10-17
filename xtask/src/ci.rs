// SPDX-License-Identifier: Apache-2.0
//! CI policy logic (P2-T145, P2-T111): PR size cap, path filters, DCO.
//!
//! Everything here is pure (strings in, verdicts out) so that it can be
//! tested without a git checkout; `main` does the I/O.

/// A PR may change at most this many counted lines (additions + deletions).
pub const MAX_CHANGED_LINES: u64 = 1500;

/// Files that do not count towards the PR size cap: goldens, fixtures,
/// vendored and generated files (P2-T145). The allowlist is deliberately
/// short and lives here so that widening it is itself a reviewed change.
const EXCLUDED_NAMES: &[&str] = &[
    "Cargo.lock",
    "budgets.json",
    "pnpm-lock.yaml",
    "package-lock.json",
    "uv.lock",
    "LICENSE",
];
const EXCLUDED_PREFIXES: &[&str] = &[
    "tests/spark-golden/",
    "tests/fixtures/",
    "vendor/",
    "third_party/",
    "crates/krishiv-ui-assets/assets/",
    ".github/ci-fixtures/",
];
/// Suffixes of machine-written files. Kept specific: a loose suffix such as
/// `.out` or `.map` would also exclude hand-written files.
const EXCLUDED_SUFFIXES: &[&str] = &[".lock", ".snap", ".pb.rs", ".min.js", ".js.map"];
/// Data directories directly under a crate (`crates/<name>/<dir>/`). They are
/// matched at that position only, so a `generated/` or `fixtures/` directory
/// inside `src/` still counts.
const CRATE_DATA_DIRS: &[&str] = &["vendored/", "tests/fixtures/", "tests/goldens/"];

/// Binary files have no line count in `git diff --numstat`; each started
/// block of this many bytes counts as one changed line.
pub const BINARY_BYTES_PER_LINE: u64 = 64;

/// Whether `path` is on the size-cap allowlist.
#[must_use]
pub fn is_size_excluded(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    EXCLUDED_NAMES.contains(&name)
        || EXCLUDED_PREFIXES.iter().any(|p| path.starts_with(p))
        || EXCLUDED_SUFFIXES.iter().any(|s| path.ends_with(s))
        || path
            .strip_prefix("crates/")
            .and_then(|rest| rest.split_once('/'))
            .is_some_and(|(_, inner)| CRATE_DATA_DIRS.iter().any(|d| inner.starts_with(d)))
}

/// Result of measuring a diff against the cap.
#[derive(Debug, PartialEq, Eq)]
pub struct SizeReport {
    /// Changed lines that count towards the cap.
    pub counted: u64,
    /// Changed lines in allowlisted files.
    pub excluded: u64,
    /// Binary files outside the allowlist; their size is added to `counted`
    /// with [`SizeReport::add_binary`].
    pub binaries: Vec<String>,
}

impl SizeReport {
    /// Whether the diff is within [`MAX_CHANGED_LINES`].
    #[must_use]
    pub fn within_cap(&self) -> bool {
        self.counted <= MAX_CHANGED_LINES
    }

    /// Counts a binary file of `bytes` bytes as
    /// `ceil(bytes / BINARY_BYTES_PER_LINE)` changed lines.
    pub fn add_binary(&mut self, bytes: u64) {
        self.counted += bytes.div_ceil(BINARY_BYTES_PER_LINE);
    }
}

/// Measures `git diff --numstat -z` output. The NUL-separated form keeps
/// paths verbatim (no quoting, no `old => new` notation to parse); a rename
/// is `added\tdeleted\t` followed by the old and new paths. Binary files
/// (`-`) outside the allowlist are listed in [`SizeReport::binaries`] for the
/// caller to size.
///
/// # Errors
/// Returns an error for a malformed record.
pub fn measure_numstat(raw: &str) -> Result<SizeReport, String> {
    let mut report = SizeReport {
        counted: 0,
        excluded: 0,
        binaries: Vec::new(),
    };
    let mut fields = raw.split('\0');
    while let Some(record) = fields.next() {
        if record.trim().is_empty() {
            continue;
        }
        let mut parts = record.splitn(3, '\t');
        let (Some(added), Some(deleted), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            return Err(format!("malformed numstat record: {record:?}"));
        };
        let path = if path.is_empty() {
            // Rename or copy: the old path, then the new one.
            let _old = fields.next();
            fields
                .next()
                .filter(|p| !p.is_empty())
                .ok_or_else(|| format!("rename record without paths: {record:?}"))?
        } else {
            path
        };
        let count = |s: &str| -> Result<u64, String> {
            if s == "-" {
                Ok(0)
            } else {
                s.parse().map_err(|_| format!("bad count in {record:?}"))
            }
        };
        let lines = count(added)? + count(deleted)?;
        if is_size_excluded(path) {
            report.excluded += lines;
        } else if added == "-" && deleted == "-" {
            report.binaries.push(path.to_owned());
        } else {
            report.counted += lines;
        }
    }
    Ok(report)
}

/// Whether a PR touching `paths` needs the Rust jobs. Docs-, bench- and
/// plan-only PRs skip them (P2-T145). CI config, agent config and hooks are
/// policy-tested by xtask, so they always run the Rust jobs.
#[must_use]
pub fn rust_relevant<'a>(paths: impl IntoIterator<Item = &'a str>) -> bool {
    paths.into_iter().any(|p| {
        let p = p.trim();
        if p.is_empty() {
            return false;
        }
        if [".github/", ".claude/", ".githooks/"]
            .iter()
            .any(|d| p.starts_with(d))
        {
            return true;
        }
        // Markdown under `crates/` or `xtask/` can be compiled into rustdoc
        // (`include_str!`), so only Markdown elsewhere is docs-only.
        let skipped = ["docs/", "bench/", ".superpowers/"]
            .iter()
            .any(|d| p.starts_with(d))
            || (p.ends_with(".md") && !p.starts_with("crates/") && !p.starts_with("xtask/"));
        !skipped
    })
}

/// Bot identities exempt from the sign-off requirement (exact email suffixes).
const BOT_EMAILS: &[&str] = &[
    "renovate[bot]@users.noreply.github.com",
    "github-actions[bot]@users.noreply.github.com",
];

/// GitHub's web-flow committer, used when a bot's change is applied via the API.
const WEB_FLOW_EMAIL: &str = "noreply@github.com";

fn is_bot(email: &str) -> bool {
    BOT_EMAILS
        .iter()
        .any(|b| email == *b || email.ends_with(&format!("+{b}")))
}

/// Commits that lack a `Signed-off-by: Name <email>` trailer (ADR 0033).
///
/// `log` is `git log --format=%H%x1f%ae%x1f%ce%x1f%B%x1e`. A commit is exempt
/// only when its author is an allowlisted bot (Renovate, github-actions) and
/// its committer is the same kind of bot or GitHub's web-flow.
#[must_use]
pub fn commits_missing_signoff(log: &str) -> Vec<String> {
    log.split('\u{1e}')
        .map(|r| r.trim_start_matches('\n'))
        .filter(|r| !r.trim().is_empty())
        .filter_map(|record| {
            let mut fields = record.splitn(4, '\u{1f}');
            let (sha, author, committer, body) = (
                fields.next()?,
                fields.next()?,
                fields.next()?,
                fields.next()?,
            );
            let signed = body.lines().any(is_signoff_line);
            let exempt = is_bot(author) && (is_bot(committer) || committer == WEB_FLOW_EMAIL);
            (!signed && !exempt).then(|| sha.trim().to_owned())
        })
        .collect()
}

fn is_signoff_line(line: &str) -> bool {
    line.strip_prefix("Signed-off-by: ")
        .is_some_and(|rest| rest.contains(" <") && rest.ends_with('>') && !rest.starts_with('<'))
}
