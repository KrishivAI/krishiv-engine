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
    /// A JSON array of findings (the documented, stable form).
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

/// Status of a function in the matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// Fully implemented.
    Supported,
    /// Implemented with differences.
    Partial,
    /// Listed by Spark, not implemented yet.
    NotYet,
}

/// The function matrix: lower-case name to status.
pub(crate) fn matrix() -> &'static BTreeMap<String, Status> {
    static M: OnceLock<BTreeMap<String, Status>> = OnceLock::new();
    M.get_or_init(|| {
        let raw: BTreeMap<String, String> =
            serde_json::from_str(include_str!("functions.json")).unwrap_or_default();
        raw.into_iter()
            .map(|(k, v)| {
                let s = match v.as_str() {
                    "supported" => Status::Supported,
                    "partial" => Status::Partial,
                    _ => Status::NotYet,
                };
                (k, s)
            })
            .collect()
    })
}

/// Collects the scannable files under `path`, sorted.
fn collect(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for e in entries {
            collect(&e, out)?;
        }
    } else if matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("py" | "sql" | "ipynb")
    ) {
        out.push(path.to_path_buf());
    }
    Ok(())
}

/// Runs the command; returns the exit code.
pub(crate) fn run(args: &CheckArgs) -> i32 {
    let mut files = Vec::new();
    let mut errors = 0;
    for p in &args.paths {
        if let Err(e) = collect(p, &mut files) {
            let _ = writeln!(std::io::stderr(), "check-compat: {}: {e}", p.display());
            errors += 1;
        }
    }
    let mut findings = Vec::new();
    for f in &files {
        match scan_file(f) {
            Ok(mut v) => findings.append(&mut v),
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "check-compat: {}: {e}", f.display());
                errors += 1;
            }
        }
    }
    let mut out = std::io::stdout().lock();
    match args.format {
        ReportFormat::Text => {
            for f in &findings {
                let _ = writeln!(out, "{}", f.text());
            }
        }
        ReportFormat::Json => {
            let v: Vec<_> = findings.iter().map(Finding::json).collect();
            let _ = writeln!(out, "{}", serde_json::Value::Array(v));
        }
    }
    if errors > 0 {
        2
    } else {
        i32::from(!findings.is_empty())
    }
}
