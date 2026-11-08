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
fn capture(root: &Path) -> Result<Value, String> {
    let items = examples()?;
    let (a, b) = (run_once(root, &items)?, run_once(root, &items)?);
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
    let fresh = capture(root)?;
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
