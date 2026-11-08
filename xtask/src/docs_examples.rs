// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask docs-examples [--check]`: runs every SQL example of the reference pages on the
//! engine at this commit (`krishiv-testkit`'s `doc-examples` binary) and records the answers in
//! `docs/reference/sql/functions/_examples.json`, which `docs-sql` prints under each example. With `--check`
//! the recorded answers must equal a fresh run, so a behaviour change that is not in the docs
//! fails. An example whose answer differs between two runs (`rand()`, `uuid()`, `now()`) is
//! recorded as volatile and shown without a result.

use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Map, Value, json};

use crate::docs_sql::{CAPTURED, examples};

fn run_once(root: &Path, items: &[(String, String)]) -> Result<Vec<Value>, String> {
    let input = Value::Array(
        items
            .iter()
            .map(|(id, sql)| json!({"id": id, "sql": sql}))
            .collect(),
    );
    let mut child = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--locked",
            "-p",
            "krishiv-testkit",
            "--bin",
            "doc-examples",
        ])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cargo run doc-examples: {e}"))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(input.to_string().as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err("doc-examples failed".to_owned());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("doc-examples output: {e}"))
}

/// Functions whose answer depends on the clock, session or host: never recorded.
fn is_clock(id: &str) -> bool {
    let name = id.split('#').next().unwrap_or_default();
    matches!(
        name,
        "current_date"
            | "curdate"
            | "current_timestamp"
            | "now"
            | "localtimestamp"
            | "current_timezone"
            | "current_time"
            | "current_user"
            | "user"
            | "session_user"
            | "current_database"
            | "current_catalog"
            | "current_schema"
            | "version"
            | "uuid"
            | "rand"
            | "random"
            | "randn"
            | "unix_timestamp"
            | "monotonically_increasing_id"
            | "spark_partition_id"
            | "input_file_name"
            | "shuffle"
    )
}

/// The engine's answers, two runs compared.
fn capture(
    mut runner: impl FnMut(&[(String, String)]) -> Result<Vec<Value>, String>,
) -> Result<Value, String> {
    let items = examples()?;
    let (a, b) = (runner(&items)?, runner(&items)?);
    let mut map = Map::new();
    for (x, y) in a.into_iter().zip(b) {
        let id = x["id"].as_str().unwrap_or_default().to_owned();
        let mut v = x.clone();
        if x != y || is_clock(&id) {
            v = json!({"volatile": true});
        } else if let Some(o) = v.as_object_mut() {
            o.remove("id");
            o.remove("message");
        }
        map.insert(id, v);
    }
    Ok(Value::Object(map))
}

/// Records the answers, or with `check` compares them with the recorded ones.
///
/// # Errors
/// On I/O errors, a failed run, or with `check` when the recorded answers are stale.
pub fn run(root: &Path, check: bool) -> Result<(), String> {
    run_with(root, check, |items| run_once(root, items))
}

fn run_with(
    root: &Path,
    check: bool,
    runner: impl FnMut(&[(String, String)]) -> Result<Vec<Value>, String>,
) -> Result<(), String> {
    let fresh = capture(runner)?;
    let text = serde_json::to_string_pretty(&fresh).map_err(|e| e.to_string())? + "\n";
    let path = root.join(CAPTURED);
    if check {
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        if have != text {
            return Err(format!(
                "{CAPTURED} is stale: an example's result changed; run `cargo xtask docs-examples` then `cargo xtask docs-sql`"
            ));
        }
        return Ok(());
    }
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    writeln!(
        std::io::stdout(),
        "recorded {} example answers in {CAPTURED}",
        fresh.as_object().map_or(0, Map::len)
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An answer per example: `ok` with the id as the row, and `rand` varying between runs.
    fn fake(
        counter: &mut u32,
    ) -> impl FnMut(&[(String, String)]) -> Result<Vec<Value>, String> + '_ {
        move |items| {
            *counter += 1;
            let n = *counter;
            Ok(items
                .iter()
                .map(|(id, _)| {
                    let row = if id.starts_with("rand#") {
                        n.to_string()
                    } else {
                        id.clone()
                    };
                    json!({"id": id, "ok": true, "rows": [row], "message": "m"})
                })
                .collect())
        }
    }

    #[test]
    fn clock_functions_are_recognised_by_page_name() {
        assert!(is_clock("now#1"));
        assert!(is_clock("uuid"));
        assert!(!is_clock("abs#1"));
    }

    #[test]
    fn a_varying_or_clock_answer_is_volatile_and_the_rest_drops_id_and_message() {
        let mut n = 0;
        let v = capture(fake(&mut n)).unwrap_or_default();
        assert_eq!(v["rand#1"], json!({"volatile": true}));
        assert_eq!(v["now#1"], json!({"volatile": true}));
        assert_eq!(v["abs#1"], json!({"ok": true, "rows": ["abs#1"]}));
    }

    #[test]
    fn record_then_check_then_stale() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let out = dir.path().join(CAPTURED);
        std::fs::create_dir_all(out.parent().unwrap_or(dir.path())).unwrap_or_default();
        let mut n = 0;
        assert!(
            run_with(dir.path(), true, fake(&mut n)).is_err(),
            "nothing recorded yet"
        );
        assert!(run_with(dir.path(), false, fake(&mut n)).is_ok());
        assert!(out.is_file());
        assert!(run_with(dir.path(), true, fake(&mut n)).is_ok());
        std::fs::write(&out, "{}\n").unwrap_or_default();
        let err = run_with(dir.path(), true, fake(&mut n))
            .err()
            .unwrap_or_default();
        assert!(err.contains("stale"), "{err}");
    }

    #[test]
    fn a_failed_run_is_an_error() {
        let r = run_with(Path::new("."), true, |_| Err("boom".to_owned()));
        assert_eq!(r.err().as_deref(), Some("boom"));
    }
}
