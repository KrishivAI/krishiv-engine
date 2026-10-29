// SPDX-License-Identifier: Apache-2.0
//! The golden-file report: three separate numbers, per-file denominators and the exclusion
//! manifest, as JSON ([`REPORT_FORMAT_VERSION`]) and as a short text summary.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Value, json};

use super::manifest::ExclusionManifest;
use super::run::{ClassTally, FileResult, RunResult, Tally};

/// Report JSON format version.
pub const REPORT_FORMAT_VERSION: u64 = 1;

fn tally(t: Tally) -> Value {
    json!({"passed": t.passed, "total": t.total})
}

fn file_json(f: &FileResult) -> Value {
    json!({
        "file": f.name,
        "statements": f.statements,
        "runs": f.runs,
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

/// A parse-phase class passes the negative gate when this share of its statements match.
pub const CLASS_GATE_PERCENT: u64 = 95;

/// The negative number per expected Spark error class, summed over the files.
#[must_use]
pub fn negative_by_class(run: &RunResult) -> BTreeMap<String, ClassTally> {
    let mut out: BTreeMap<String, ClassTally> = BTreeMap::new();
    for f in &run.files {
        for (class, c) in &f.classes {
            let e = out.entry(class.clone()).or_default();
            e.parse_phase = c.parse_phase;
            e.tally.add(c.tally);
        }
    }
    out
}

/// The negative gates of the parser: `(PARSE_SYNTAX_ERROR matched, its total, parse-phase classes
/// at or above [`CLASS_GATE_PERCENT`], parse-phase classes)`.
#[must_use]
pub fn parse_phase_gates(classes: &BTreeMap<String, ClassTally>) -> (Tally, u64, u64) {
    let syntax = classes
        .get("PARSE_SYNTAX_ERROR")
        .map_or_else(Tally::default, |c| c.tally);
    let parse_classes: Vec<&ClassTally> = classes.values().filter(|c| c.parse_phase).collect();
    let met = parse_classes
        .iter()
        .filter(|c| c.tally.passed * 100 >= CLASS_GATE_PERCENT * c.tally.total)
        .count() as u64;
    (syntax, met, parse_classes.len() as u64)
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
        "jdk": run.jdk.version(),
        "jdk_variant_files": run.variant_files,
        "engine": run.engine,
        "totals": {
            "files_run": run.files.len(),
            "files_excluded": run.excluded.len(),
            "files_without_expected_output": run.without_expected.len(),
            "files_failed": run.failed_files.len(),
            "unclassified_errors": run.files.iter().map(|f| f.unclassified_errors).sum::<u64>(),
            "statements": statements,
            "parse": tally(parse),
            "negative": tally(negative),
            "result": tally(result),
        },
        "negative_by_class": negative_by_class(run)
            .into_iter()
            .map(|(class, c)| {
                (
                    class,
                    json!({"passed": c.tally.passed, "total": c.tally.total, "parse_phase": c.parse_phase}),
                )
            })
            .collect::<serde_json::Map<_, _>>(),
        "files": run.files.iter().map(file_json).collect::<Vec<_>>(),
        "without_expected_output": run.without_expected,
        "failed_files": run.failed_files.iter().map(|f| json!({"file": f.name, "error": f.error})).collect::<Vec<_>>(),
        "exclusions": {
            "frozen": manifest.frozen,
            "excluded_since_freeze": manifest.entries.iter().filter(|e| e.added_after_freeze).count(),
            "entries": entries,
        },
        "unregistered_error_classes": run.unregistered_classes,
    })
}

/// Every counted statement that did not pass, per file: index, the numbers it failed and why.
#[must_use]
pub fn cases_json(run: &RunResult) -> Value {
    let mut files = serde_json::Map::new();
    for f in &run.files {
        let failed: Vec<Value> = f
            .cases
            .iter()
            .filter(|c| !c.passed())
            .map(|c| {
                let failed: Vec<&str> = [
                    ("parse", c.parse),
                    ("negative", c.negative),
                    ("result", c.result),
                ]
                .into_iter()
                .filter(|(_, v)| *v == Some(false))
                .map(|(k, _)| k)
                .collect();
                json!({"index": c.index, "failed": failed, "detail": c.detail})
            })
            .collect();
        if !failed.is_empty() {
            files.insert(f.name.clone(), Value::Array(failed));
        }
    }
    Value::Object(files)
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
    let classes = negative_by_class(run);
    let (syntax, met, parse_classes) = parse_phase_gates(&classes);
    let _ = writeln!(
        s,
        "  PARSE_SYNTAX_ERROR      {} / {}  (gate: all)",
        syntax.passed, syntax.total
    );
    let _ = writeln!(
        s,
        "  parse-phase classes     {met} / {parse_classes} at >= {CLASS_GATE_PERCENT}%  (gate: >= {CLASS_GATE_PERCENT}% of the classes)"
    );
    let mut worst: Vec<(&String, &ClassTally)> = classes
        .iter()
        .filter(|(_, c)| c.tally.passed < c.tally.total)
        .collect();
    worst.sort_by_key(|(name, c)| {
        (
            std::cmp::Reverse(c.tally.total - c.tally.passed),
            (*name).clone(),
        )
    });
    for (name, c) in worst.into_iter().take(10) {
        let _ = writeln!(
            s,
            "    unmatched {name}: {} of {}{}",
            c.tally.total - c.tally.passed,
            c.tally.total,
            if c.parse_phase { " (parse phase)" } else { "" }
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden::Jdk;
    use crate::golden::manifest::ExclusionManifest;
    use crate::golden::run::{FailedFile, SkippedFile};

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
                classes: BTreeMap::from([(
                    "PARSE_SYNTAX_ERROR".to_owned(),
                    ClassTally {
                        tally: Tally {
                            passed: 0,
                            total: 1,
                        },
                        parse_phase: true,
                    },
                )]),
                runs: 2,
                cases: Vec::new(),
                timeouts: Vec::new(),
            }],
            excluded: vec![SkippedFile {
                name: "udf/x".into(),
                statements: 7,
            }],
            without_expected: vec!["z".into()],
            jdk: Jdk::Java21,
            variant_files: vec!["v".into()],
            failed_files: vec![FailedFile {
                name: "w".into(),
                error: "bad".into(),
            }],
            unregistered_classes: ["NOT_A_CLASS".to_owned()].into(),
        };
        insta::assert_snapshot!(to_json_string(&to_json(&run, "v4.2.0", &manifest))?);
        Ok(())
    }

    fn class(passed: u64, total: u64, parse_phase: bool) -> ClassTally {
        ClassTally {
            tally: Tally { passed, total },
            parse_phase,
        }
    }

    #[test]
    fn negative_by_class_sums_files_and_gates_the_parse_phase() {
        let run = RunResult {
            files: vec![
                FileResult {
                    classes: BTreeMap::from([
                        ("PARSE_SYNTAX_ERROR".to_owned(), class(2, 2, true)),
                        ("UNRESOLVED_COLUMN".to_owned(), class(1, 3, false)),
                        ("INVALID_LIMIT".to_owned(), class(19, 20, true)),
                    ]),
                    ..FileResult::default()
                },
                FileResult {
                    classes: BTreeMap::from([
                        ("PARSE_SYNTAX_ERROR".to_owned(), class(1, 2, true)),
                        ("INVALID_LIMIT".to_owned(), class(0, 1, true)),
                    ]),
                    ..FileResult::default()
                },
            ],
            ..RunResult::default()
        };
        let by = negative_by_class(&run);
        assert_eq!(
            by["PARSE_SYNTAX_ERROR"].tally,
            Tally {
                passed: 3,
                total: 4
            }
        );
        assert_eq!(
            by["INVALID_LIMIT"].tally,
            Tally {
                passed: 19,
                total: 21
            }
        );
        assert!(!by["UNRESOLVED_COLUMN"].parse_phase);
        // PARSE_SYNTAX_ERROR 3/4; parse-phase classes: syntax 75% (no), limit 90% (no): 0 of 2.
        let (syntax, met, classes) = parse_phase_gates(&by);
        assert_eq!(
            syntax,
            Tally {
                passed: 3,
                total: 4
            }
        );
        assert_eq!((met, classes), (0, 2));
        let text = summary(&run);
        assert!(text.contains("PARSE_SYNTAX_ERROR      3 / 4"));
        assert!(text.contains("parse-phase classes     0 / 2 at >= 95%"));
        assert!(text.contains("unmatched UNRESOLVED_COLUMN: 2 of 3"));
        assert!(text.contains("unmatched PARSE_SYNTAX_ERROR: 1 of 4 (parse phase)"));
        let json = to_json(&run, "v4.2.0", &ExclusionManifest::default());
        assert_eq!(
            json["negative_by_class"]["INVALID_LIMIT"],
            serde_json::json!({"passed": 19, "total": 21, "parse_phase": true})
        );
        // A run with no parse-phase class has nothing to gate.
        assert_eq!(
            parse_phase_gates(&BTreeMap::new()),
            (Tally::default(), 0, 0)
        );
    }

    #[test]
    fn cases_json_lists_only_the_failures() {
        use crate::golden::run::Case;
        let run = RunResult {
            files: vec![
                FileResult {
                    name: "a".into(),
                    cases: vec![
                        Case {
                            index: 0,
                            parse: Some(true),
                            result: Some(true),
                            ..Case::default()
                        },
                        Case {
                            index: 3,
                            parse: Some(false),
                            negative: Some(false),
                            detail: "parse failed; no error".into(),
                            ..Case::default()
                        },
                    ],
                    ..FileResult::default()
                },
                FileResult {
                    name: "b".into(),
                    ..FileResult::default()
                },
            ],
            ..RunResult::default()
        };
        assert_eq!(
            cases_json(&run),
            serde_json::json!({"a": [{"index": 3, "failed": ["parse", "negative"], "detail": "parse failed; no error"}]})
        );
    }
}
