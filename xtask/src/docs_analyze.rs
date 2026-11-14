// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask docs-analyze [--check] [--base REF]`: the `krishiv.analyze.v1` JSON Schema
//! (`docs/reference/analyze/krishiv.analyze.v1.schema.json`), written from the hand-written description in
//! `krishiv_plan::analyzer::lineage::schema` (P3-T063 N1b; no schema crate, ADR 0028).
//!
//! `--check` fails when the file differs from the description, and when it is not additive over the file on
//! `REF` (default `origin/main`): every member, `required` entry, enum value and `const` of the base file must
//! still be there. A base without the file (or without the ref) is trivially additive.

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
    let Some(old) = base_file(root, base) else {
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

/// The schema file as `base` has it, if it does.
fn base_file(root: &Path, base: &str) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["show", &format!("{base}:{OUT}")])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Collects into `problems` what `old` has and `new` lost, by JSON path. Descriptions and titles may change.
pub fn additive(old: &Value, new: &Value, path: &str, problems: &mut Vec<String>) {
    match (old, new) {
        (Value::Object(o), Value::Object(n)) => {
            for (k, ov) in o {
                if matches!(k.as_str(), "description" | "title") {
                    continue;
                }
                let at = format!("{path}.{k}");
                match (k.as_str(), n.get(k)) {
                    (_, None) => problems.push(format!("{at} removed")),
                    // A set: every old entry stays (an enum may grow; a required member stays required).
                    ("enum" | "required", Some(nv)) => {
                        let have = nv.as_array().cloned().unwrap_or_default();
                        for v in ov.as_array().into_iter().flatten() {
                            if !have.contains(v) {
                                problems.push(format!("{at}: {v} removed"));
                            }
                        }
                    }
                    (_, Some(nv)) => additive(ov, nv, &at, problems),
                }
            }
        }
        (Value::Array(o), Value::Array(n)) => {
            for (i, ov) in o.iter().enumerate() {
                match n.get(i) {
                    Some(nv) => additive(ov, nv, &format!("{path}[{i}]"), problems),
                    None => problems.push(format!("{path}[{i}] removed")),
                }
            }
        }
        (o, n) if o == n => {}
        (o, n) => problems.push(format!("{path}: {o} became {n}")),
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
    fn the_committed_file_is_current() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let have = std::fs::read_to_string(root.join(OUT)).unwrap_or_default();
        assert_eq!(have, krishiv_plan::analyzer::lineage::schema::json_schema());
    }
}
