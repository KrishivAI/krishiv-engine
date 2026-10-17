// SPDX-License-Identifier: Apache-2.0
//! The golden-file report: three separate numbers, per-file denominators and the exclusion
//! manifest, as JSON ([`REPORT_FORMAT_VERSION`]) and as a short text summary.

use std::fmt::Write as _;

use serde_json::{Value, json};

use super::manifest::ExclusionManifest;
use super::run::{FileResult, RunResult, Tally};

/// Report JSON format version.
pub const REPORT_FORMAT_VERSION: u64 = 1;

fn tally(t: Tally) -> Value {
    json!({"passed": t.passed, "total": t.total})
}

fn file_json(f: &FileResult) -> Value {
    json!({
        "file": f.name,
        "statements": f.statements,
        "parse": tally(f.parse),
        "negative": tally(f.negative),
        "result": tally(f.result),
        "unclassified_errors": f.unclassified_errors,
    })
}

/// Sum of the per-file numbers: `(parse, negative, result)`.
#[must_use]
pub fn totals(run: &RunResult) -> (Tally, Tally, Tally) {
    let (mut p, mut n, mut r) = (Tally::default(), Tally::default(), Tally::default());
    for f in &run.files {
        p.add(f.parse);
        n.add(f.negative);
        r.add(f.result);
    }
    (p, n, r)
}

/// Build the report JSON for `run`, published together with `manifest`.
#[must_use]
pub fn to_json(run: &RunResult, tag: &str, manifest: &ExclusionManifest) -> Value {
    let (parse, negative, result) = totals(run);
    let statements: u64 = run.files.iter().map(|f| f.statements).sum();
    let entries: Vec<Value> = manifest
        .entries
        .iter()
        .map(|e| {
            let statements = run
                .excluded
                .iter()
                .find(|s| s.name == e.file)
                .map_or(0, |s| s.statements);
            json!({
                "file": e.file,
                "reason": e.reason.key(),
                "note": e.note,
                "statements": statements,
                "added_after_freeze": e.added_after_freeze,
            })
        })
        .collect();
    json!({
        "format_version": REPORT_FORMAT_VERSION,
        "spark_tag": tag,
        "engine": run.engine,
        "totals": {
            "files_run": run.files.len(),
            "files_excluded": run.excluded.len(),
            "files_without_expected_output": run.without_expected.len(),
            "statements": statements,
            "parse": tally(parse),
            "negative": tally(negative),
            "result": tally(result),
        },
        "files": run.files.iter().map(file_json).collect::<Vec<_>>(),
        "without_expected_output": run.without_expected,
        "exclusions": {
            "frozen": manifest.frozen,
            "excluded_since_freeze": manifest.entries.iter().filter(|e| e.added_after_freeze).count(),
            "entries": entries,
        },
        "unregistered_error_classes": run.unregistered_classes,
    })
}

/// The report as pretty-printed JSON with a trailing newline.
pub fn to_json_string(report: &Value) -> Result<String, serde_json::Error> {
    let mut s = serde_json::to_string_pretty(report)?;
    s.push('\n');
    Ok(s)
}

/// A short human-readable summary of the three numbers.
#[must_use]
pub fn summary(run: &RunResult) -> String {
    let (parse, negative, result) = totals(run);
    let mut s = format!(
        "golden run: engine `{}`, {} files run, {} excluded\n",
        run.engine,
        run.files.len(),
        run.excluded.len()
    );
    for (label, t) in [
        ("parse", parse),
        ("negative (error class)", negative),
        ("result", result),
    ] {
        let _ = writeln!(s, "  {label:<24}{} / {}", t.passed, t.total);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::manifest::ExclusionManifest;
    use crate::golden::run::SkippedFile;

    #[test]
    fn report_json_schema_snapshot() -> Result<(), Box<dyn std::error::Error>> {
        let manifest = ExclusionManifest::parse(
            r#"{"format_version":1,"frozen":true,"exclusions":[
              {"file":"udf/x","reason":"jvm_udf","note":"scala udf"},
              {"file":"y","reason":"jdk_specific","added_after_freeze":true}]}"#,
        )?;
        let run = RunResult {
            engine: "reject-all".into(),
            files: vec![FileResult {
                name: "a".into(),
                statements: 4,
                parse: Tally {
                    passed: 1,
                    total: 3,
                },
                negative: Tally {
                    passed: 0,
                    total: 1,
                },
                result: Tally {
                    passed: 1,
                    total: 2,
                },
                unclassified_errors: 1,
            }],
            excluded: vec![SkippedFile {
                name: "udf/x".into(),
                statements: 7,
            }],
            without_expected: vec!["z".into()],
            unregistered_classes: ["NOT_A_CLASS".to_owned()].into(),
        };
        insta::assert_snapshot!(to_json_string(&to_json(&run, "v4.2.0", &manifest))?);
        Ok(())
    }
}
