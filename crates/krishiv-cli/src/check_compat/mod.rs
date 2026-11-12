// SPDX-License-Identifier: Apache-2.0
//! `krishiv check-compat` (P3-T029): reads `.py`, `.sql` and `.ipynb` files
//! and reports the Spark functions they call that Krishiv does not (fully)
//! support, from the generated function matrix (`functions.json`, built from
//! the SQL reference pages by `tests/scripts/gen_compat_matrix.py`).
//! Nothing is executed. Exit codes: 0 no findings, 1 findings, 2 read errors.

mod scan;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use clap::{Args, ValueEnum};

pub(crate) use scan::{Finding, scan_file};

/// Output format of the report.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ReportFormat {
    /// One `file:line: name: status` line per finding.
    Text,
    /// A JSON object `{schema_version, files_scanned, findings, errors}` (the stable form).
    Json,
}

/// Arguments of `krishiv check-compat`.
#[derive(Args)]
pub(crate) struct CheckArgs {
    /// Files or directories to scan (`.py`, `.sql`, `.ipynb`).
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// Report format.
    #[arg(long, value_enum, default_value = "text")]
    format: ReportFormat,
}

/// Status of a function in the matrix, or `Unknown` for a name it does not list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// Fully implemented.
    Supported,
    /// Implemented with differences.
    Partial,
    /// Listed by Spark, not implemented yet.
    NotYet,
    /// Not in the matrix at all.
    Unknown,
}

impl Status {
    /// The status word of the report (`unknown` is spelled out in the text form).
    pub(crate) fn label(self) -> &'static str {
        match self {
            Status::Supported => "supported",
            Status::Partial => "partial",
            Status::NotYet => "not yet",
            Status::Unknown => "unknown",
        }
    }
}

/// Parses the matrix JSON; an unrecognised status is an error, not `not yet`.
pub(crate) fn parse_matrix(raw: &str) -> Result<BTreeMap<String, Status>, String> {
    let raw: BTreeMap<String, String> = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    raw.into_iter()
        .map(|(k, v)| {
            let s = match v.as_str() {
                "supported" => Status::Supported,
                "partial" => Status::Partial,
                "not yet" => Status::NotYet,
                other => return Err(format!("{k}: unknown status {other:?}")),
            };
            Ok((k, s))
        })
        .collect()
}

/// The function matrix: lower-case name to status. The embedded file is checked by a
/// unit test; a parse failure here would make every name `unknown`.
pub(crate) fn matrix() -> &'static BTreeMap<String, Status> {
    static M: OnceLock<BTreeMap<String, Status>> = OnceLock::new();
    M.get_or_init(|| parse_matrix(include_str!("functions.json")).unwrap_or_default())
}

/// Workarounds for "not yet" functions that have a real one: lower-case name to a
/// sentence. Every function the text names must be supported (a unit test checks).
pub(crate) fn alternatives() -> &'static BTreeMap<String, String> {
    static A: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    A.get_or_init(|| serde_json::from_str(include_str!("alternatives.json")).unwrap_or_default())
}

/// Directories never descended into: VCS, virtual environments, installed packages, build output.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".ipynb_checkpoints",
    "target",
];

/// Collects the scannable files under `path`, sorted. A file named on the command
/// line with another extension is an error; read errors are collected, not fatal.
fn collect(path: &Path, top: bool, out: &mut Vec<PathBuf>, errors: &mut Vec<(String, String)>) {
    let mut err = |e: String| errors.push((path.display().to_string(), e));
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => return err(e.to_string()),
    };
    if meta.is_dir() {
        let rd = match std::fs::read_dir(path) {
            Ok(rd) => rd,
            Err(e) => return err(e.to_string()),
        };
        let mut entries = Vec::new();
        for e in rd {
            match e {
                Ok(e) => entries.push(e.path()),
                Err(e) => err(e.to_string()),
            }
        }
        entries.sort();
        for e in entries {
            // A directory symlink can point at an ancestor and loop: never followed.
            let linked_dir = e.is_dir() && e.is_symlink();
            let skipped = linked_dir
                || e.is_dir()
                    && e.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| SKIP_DIRS.contains(&n));
            if !skipped {
                collect(&e, false, out, errors);
            }
        }
    } else if matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("py" | "sql" | "ipynb")
    ) {
        out.push(path.to_path_buf());
    } else if top {
        err("not a .py, .sql or .ipynb file".to_string());
    }
}

/// Version of the JSON report; bumped only for a breaking change.
const SCHEMA_VERSION: u32 = 1;

/// Runs the command; returns the exit code.
pub(crate) fn run(args: &CheckArgs) -> i32 {
    let mut files = Vec::new();
    let mut errors: Vec<(String, String)> = Vec::new();
    for p in &args.paths {
        collect(p, true, &mut files, &mut errors);
    }
    let mut findings = Vec::new();
    for f in &files {
        match scan_file(f) {
            Ok(mut v) => findings.append(&mut v),
            Err(e) => errors.push((f.display().to_string(), e.to_string())),
        }
    }
    for (path, e) in &errors {
        let _ = writeln!(std::io::stderr(), "check-compat: {path}: {e}");
    }
    let report = match args.format {
        ReportFormat::Text => findings.iter().fold(String::new(), |mut acc, f| {
            acc.push_str(&f.text());
            acc.push('\n');
            acc
        }),
        ReportFormat::Json => {
            let report = serde_json::json!({
                "schema_version": SCHEMA_VERSION,
                "files_scanned": files.len(),
                "findings": findings.iter().map(Finding::json).collect::<Vec<_>>(),
                "errors": errors
                    .iter()
                    .map(|(p, m)| serde_json::json!({"path": p, "message": m}))
                    .collect::<Vec<_>>(),
            });
            format!("{report}\n")
        }
    };
    // A report that did not get written is a failure, never a clean run.
    let mut out = std::io::stdout().lock();
    if let Err(e) = out.write_all(report.as_bytes()).and_then(|()| out.flush()) {
        let _ = writeln!(
            std::io::stderr(),
            "check-compat: cannot write the report: {e}"
        );
        return 2;
    }
    if errors.is_empty() {
        i32::from(!findings.is_empty())
    } else {
        2
    }
}
