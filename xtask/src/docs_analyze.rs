// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask docs-analyze [--check] [--base REF]`: the `krishiv.analyze.v1` JSON Schema
//! (`docs/reference/analyze/krishiv.analyze.v1.schema.json`), written from the hand-written description in
//! `krishiv_plan::analyzer::lineage::schema` (P3-T063 N1b; no schema crate, ADR 0028).
//!
//! `--check` fails when the file differs from the description, and when it is not additive over the file on
//! `REF` (default `origin/main`): every member, enum value and `const` of the base file must still be there,
//! `required` must not grow (a member added within v1 is optional) and no object may be closed or narrowed
//! (`additionalProperties`), nor gain any other constraint keyword. A missing `REF` is an error; only a `REF`
//! that exists and lacks the file is trivially additive.

use std::io::Write as _;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// The schema file, relative to the repository root.
pub const OUT: &str = "docs/reference/analyze/krishiv.analyze.v1.schema.json";

/// Writes the schema file, or checks it (`check`) against the description and against `base`'s file.
///
/// # Errors
/// A stale file, a non-additive change, or an I/O error, as text.
pub fn run(root: &Path, check: bool, base: &str) -> Result<(), String> {
    let want = krishiv_plan::analyzer::lineage::schema::json_schema();
    let path = root.join(OUT);
    if !check {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        return std::fs::write(&path, want).map_err(|e| format!("{}: {e}", path.display()));
    }
    if std::fs::read_to_string(&path).ok().as_deref() != Some(want.as_str()) {
        return Err(format!("{OUT} is stale; run `cargo xtask docs-analyze`"));
    }
    let Some(old) = base_file(root, base)? else {
        let _ = writeln!(
            std::io::stdout(),
            "{OUT}: no file on {base}, so the schema is trivially additive"
        );
        return Ok(());
    };
    let old: Value = serde_json::from_str(&old).map_err(|e| format!("{OUT} on {base}: {e}"))?;
    let new: Value = serde_json::from_str(&want).map_err(|e| format!("{OUT}: {e}"))?;
    let mut problems = Vec::new();
    additive(&old, &new, "$", &mut problems);
    if problems.is_empty() {
        let _ = writeln!(std::io::stdout(), "{OUT}: additive over {base}");
        Ok(())
    } else {
        Err(format!(
            "{OUT} is not additive over {base} (a removal or a change of meaning is krishiv.analyze.v2):\n  {}",
            problems.join("\n  ")
        ))
    }
}

/// The schema file as `base` has it: `None` when the ref exists and lacks the file.
///
/// # Errors
/// When git cannot run, `base` is not a commit (a shallow checkout without the base branch), or the file
/// cannot be read there.
fn base_file(root: &Path, base: &str) -> Result<Option<String>, String> {
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .map_err(|e| format!("git: {e}"))
    };
    if !git(&[
        "rev-parse",
        "--verify",
        "--quiet",
        &format!("{base}^{{commit}}"),
    ])?
    .status
    .success()
    {
        return Err(format!(
            "{base} is not a commit here: fetch it (a shallow checkout lacks it) or pass --base REF"
        ));
    }
    let spec = format!("{base}:{OUT}");
    if !git(&["cat-file", "-e", &spec])?.status.success() {
        return Ok(None);
    }
    let out = git(&["show", &spec])?;
    if !out.status.success() {
        return Err(format!(
            "git show {spec}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(Some(String::from_utf8_lossy(&out.stdout).into_owned()))
}

/// Schema keywords a newer file may add without narrowing what validates.
const NEUTRAL_KEYWORDS: [&str; 5] = ["description", "title", "$comment", "examples", "properties"];

/// Collects into `problems`, by JSON path, how `new` is not additive over `old` (both JSON Schemas): a
/// member, enum value or `const` removed, a meaning changed, a `required` entry added, an object closed
/// or narrowed (`additionalProperties`), or any other constraint keyword added. Descriptions and titles
/// may change.
pub fn additive(old: &Value, new: &Value, path: &str, problems: &mut Vec<String>) {
    schema(old, new, path, problems);
}

/// `old` and `new` at a schema position: their keys are keywords.
fn schema(old: &Value, new: &Value, path: &str, problems: &mut Vec<String>) {
    let (Value::Object(o), Value::Object(n)) = (old, new) else {
        value(old, new, path, problems);
        return;
    };
    for (k, ov) in o {
        if matches!(
            k.as_str(),
            "description" | "title" | "$comment" | "examples"
        ) {
            continue;
        }
        let at = format!("{path}.{k}");
        let Some(nv) = n.get(k) else {
            problems.push(format!("{at} removed"));
            continue;
        };
        match k.as_str() {
            // A set that may grow: every old value stays.
            "enum" => {
                let have = nv.as_array().cloned().unwrap_or_default();
                for v in ov.as_array().into_iter().flatten() {
                    if !have.contains(v) {
                        problems.push(format!("{at}: {v} removed"));
                    }
                }
            }
            // A set that may not change within v1: an added entry rejects documents v1 accepted.
            "required" => {
                let had = ov.as_array().cloned().unwrap_or_default();
                let have = nv.as_array().cloned().unwrap_or_default();
                for v in &had {
                    if !have.contains(v) {
                        problems.push(format!("{at}: {v} removed"));
                    }
                }
                for v in &have {
                    if !had.contains(v) {
                        problems.push(format!(
                            "{at}: {v} added (a member added within v1 is optional)"
                        ));
                    }
                }
            }
            // Keys are member names, each a schema; a new member is additive.
            "properties" | "$defs" => {
                if let (Value::Object(op), Value::Object(np)) = (ov, nv) {
                    for (name, os) in op {
                        let at = format!("{at}.{name}");
                        match np.get(name) {
                            Some(ns) => schema(os, ns, &at, problems),
                            None => problems.push(format!("{at} removed")),
                        }
                    }
                } else {
                    value(ov, nv, &at, problems);
                }
            }
            // A list of schemas; a new alternative is additive.
            "oneOf" | "anyOf" => {
                if let (Value::Array(oa), Value::Array(na)) = (ov, nv) {
                    for (i, os) in oa.iter().enumerate() {
                        let at = format!("{at}[{i}]");
                        match na.get(i) {
                            Some(ns) => schema(os, ns, &at, problems),
                            None => problems.push(format!("{at} removed")),
                        }
                    }
                } else {
                    value(ov, nv, &at, problems);
                }
            }
            "items" | "additionalProperties" => schema(ov, nv, &at, problems),
            _ => value(ov, nv, &at, problems),
        }
    }
    for (k, nv) in n {
        if o.contains_key(k) || NEUTRAL_KEYWORDS.contains(&k.as_str()) {
            continue;
        }
        let at = format!("{path}.{k}");
        if k == "additionalProperties" {
            if *nv != Value::Bool(true) {
                problems.push(format!("{at}: {nv} added (an object of v1 stays open)"));
            }
        } else {
            problems.push(format!("{at}: {nv} added (a new constraint)"));
        }
    }
}

/// `old` and `new` as plain values: equal, or a change of meaning.
fn value(old: &Value, new: &Value, path: &str, problems: &mut Vec<String>) {
    if old != new {
        problems.push(format!("{path}: {old} became {new}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problems(old: &str, new: &str) -> Vec<String> {
        let mut p = Vec::new();
        additive(
            &serde_json::from_str(old).unwrap_or_default(),
            &serde_json::from_str(new).unwrap_or_default(),
            "$",
            &mut p,
        );
        p.sort();
        p
    }

    #[test]
    fn additions_pass_and_removals_fail() {
        let base = r#"{"required":["a"],"properties":{"a":{"enum":["x","y"],"description":"d"}}}"#;
        assert!(problems(base, base).is_empty());
        // A new member, an enum value, a new description: additive.
        let grown = r#"{"required":["a"],"properties":{"a":{"enum":["x","y","z"],"description":"e"},
            "b":{"type":"string"}}}"#;
        assert!(problems(base, grown).is_empty());
        // An enum value, a member, a required entry removed; a type changed.
        assert_eq!(
            problems(
                base,
                r#"{"required":["a"],"properties":{"a":{"enum":["x"]}}}"#
            ),
            [r#"$.properties.a.enum: "y" removed"#]
        );
        assert_eq!(
            problems(base, r#"{"required":[],"properties":{}}"#),
            [r#"$.properties.a removed"#, r#"$.required: "a" removed"#]
        );
        assert_eq!(
            problems(r#"{"type":"string"}"#, r#"{"type":"integer"}"#),
            [r#"$.type: "string" became "integer""#]
        );
    }

    #[test]
    fn narrowing_fails_and_widening_passes() {
        let base = r#"{"type":"object","required":["a"],"properties":{"a":{"type":"string"}}}"#;
        // A new optional member: additive.
        assert!(
            problems(
                base,
                r#"{"type":"object","required":["a"],"properties":{"a":{"type":"string"},"b":{"type":"string"}}}"#
            )
            .is_empty()
        );
        // An added required entry, a closed object, a new constraint: not additive.
        assert_eq!(
            problems(
                base,
                r#"{"type":"object","required":["a","b"],"properties":{"a":{"type":"string"},"b":{"type":"string"}}}"#
            ),
            [r#"$.required: "b" added (a member added within v1 is optional)"#]
        );
        assert_eq!(
            problems(
                base,
                r#"{"type":"object","required":["a"],"additionalProperties":false,"properties":{"a":{"type":"string"}}}"#
            ),
            ["$.additionalProperties: false added (an object of v1 stays open)"]
        );
        assert!(
            problems(
                base,
                r#"{"type":"object","required":["a"],"additionalProperties":true,"properties":{"a":{"type":"string"}}}"#
            )
            .is_empty()
        );
        assert_eq!(
            problems(
                base,
                r#"{"type":"object","required":["a"],"properties":{"a":{"type":"string","maxLength":3}}}"#
            ),
            [r#"$.properties.a.maxLength: 3 added (a new constraint)"#]
        );
        // An open object narrowed: `true` became `false`.
        assert_eq!(
            problems(
                r#"{"additionalProperties":true}"#,
                r#"{"additionalProperties":false}"#
            ),
            ["$.additionalProperties: true became false"]
        );
        // A member named like a keyword is still a member (review n4).
        let named = r#"{"properties":{"description":{"type":"string"}}}"#;
        assert_eq!(
            problems(named, r#"{"properties":{}}"#),
            ["$.properties.description removed"]
        );
    }

    #[test]
    fn a_missing_base_ref_is_an_error() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let err = base_file(&root, "refs/heads/no-such-branch-for-docs-analyze").unwrap_err();
        assert!(err.contains("is not a commit"), "{err}");
    }
}
