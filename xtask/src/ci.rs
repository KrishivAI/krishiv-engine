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
];
const EXCLUDED_PREFIXES: &[&str] = &[
    "tests/spark-golden/",
    "tests/fixtures/",
    "vendor/",
    "third_party/",
    "crates/krishiv-ui-assets/assets/",
    ".github/ci-fixtures/",
];
const EXCLUDED_SUFFIXES: &[&str] = &[
    ".lock", ".out", ".snap", ".golden", ".pb.rs", ".min.js", ".map",
];
const EXCLUDED_CONTAINS: &[&str] = &["/generated/", "/fixtures/", "/goldens/", "/vendored/"];

/// Whether `path` is on the size-cap allowlist.
#[must_use]
pub fn is_size_excluded(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    EXCLUDED_NAMES.contains(&name)
        || EXCLUDED_PREFIXES.iter().any(|p| path.starts_with(p))
        || EXCLUDED_SUFFIXES.iter().any(|s| path.ends_with(s))
        || EXCLUDED_CONTAINS.iter().any(|c| path.contains(c))
}

/// Result of measuring a diff against the cap.
#[derive(Debug, PartialEq, Eq)]
pub struct SizeReport {
    /// Changed lines that count towards the cap.
    pub counted: u64,
    /// Changed lines in allowlisted files.
    pub excluded: u64,
}

impl SizeReport {
    /// Whether the diff is within [`MAX_CHANGED_LINES`].
    #[must_use]
    pub fn within_cap(&self) -> bool {
        self.counted <= MAX_CHANGED_LINES
    }
}

/// Resolves git's rename notation (`a/{old => new}/b`, `old => new`) to the
/// new path.
fn resolve_rename(path: &str) -> String {
    if let (Some(open), Some(close)) = (path.find('{'), path.find('}'))
        && open < close
        && let Some((_, new)) = path[open + 1..close].split_once(" => ")
    {
        let joined = format!("{}{}{}", &path[..open], new, &path[close + 1..]);
        return joined.replace("//", "/");
    }
    match path.split_once(" => ") {
        Some((_, new)) => new.to_owned(),
        None => path.to_owned(),
    }
}

/// Measures `git diff --numstat` output. Binary files (`-`) count as zero.
///
/// # Errors
/// Returns an error for a line that is not `added<TAB>deleted<TAB>path`.
pub fn measure_numstat(numstat: &str) -> Result<SizeReport, String> {
    let mut report = SizeReport {
        counted: 0,
        excluded: 0,
    };
    for line in numstat.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.splitn(3, '\t');
        let (Some(added), Some(deleted), Some(path)) = (parts.next(), parts.next(), parts.next())
        else {
            return Err(format!("malformed numstat line: {line:?}"));
        };
        let count = |s: &str| -> Result<u64, String> {
            if s == "-" {
                Ok(0)
            } else {
                s.parse().map_err(|_| format!("bad count in {line:?}"))
            }
        };
        let lines = count(added)? + count(deleted)?;
        if is_size_excluded(&resolve_rename(path)) {
            report.excluded += lines;
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
        let skipped = ["docs/", "bench/", ".superpowers/"]
            .iter()
            .any(|d| p.starts_with(d))
            || p.ends_with(".md");
        !skipped
    })
}

/// Commits that lack a `Signed-off-by: Name <email>` trailer (ADR 0033).
///
/// `log` is `git log --format=%H%x1f%ae%x1f%B%x1e`. Bot authors
/// (`[bot]` in the email, e.g. Renovate) are exempt.
#[must_use]
pub fn commits_missing_signoff(log: &str) -> Vec<String> {
    log.split('\u{1e}')
        .map(|r| r.trim_start_matches('\n'))
        .filter(|r| !r.trim().is_empty())
        .filter_map(|record| {
            let mut fields = record.splitn(3, '\u{1f}');
            let (sha, email, body) = (fields.next()?, fields.next()?, fields.next()?);
            let signed = body.lines().any(is_signoff_line);
            (!signed && !email.contains("[bot]")).then(|| sha.trim().to_owned())
        })
        .collect()
}

fn is_signoff_line(line: &str) -> bool {
    line.strip_prefix("Signed-off-by: ")
        .is_some_and(|rest| rest.contains(" <") && rest.ends_with('>') && !rest.starts_with('<'))
}
