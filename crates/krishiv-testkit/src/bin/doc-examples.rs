// SPDX-License-Identifier: Apache-2.0
//! Runs the SQL examples of the reference pages on `krishiv-session` and prints what the engine
//! answered (`cargo xtask docs-examples` drives it).
//!
//! Reads a JSON array on stdin: `[{"id": "...", "setup": ["..."], "sql": "..."}]`. Every example
//! gets a fresh ANSI-on session; the `setup` statements run first and must succeed. Writes a JSON
//! array on stdout, one object per example in the same order: `{"id", "ok": true, "rows": [...]}`
//! with each row as the tab-joined cells Spark prints, or `{"id", "ok": false, "class": "...",
//! "message": "..."}` for an error.
#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::process::ExitCode;

use krishiv_testkit::golden::run::Engine;
use krishiv_testkit::golden::session_engine::SessionEngine;
use krishiv_testkit::golden::setup::{SessionSetup, TestKind};
use serde_json::{Value, json};

fn run(input: &str) -> Result<Value, String> {
    let examples: Vec<Value> = serde_json::from_str(input).map_err(|e| format!("stdin: {e}"))?;
    let mut engine = SessionEngine::new().map_err(|e| e.message)?;
    let mut out = Vec::with_capacity(examples.len());
    for ex in &examples {
        let id = ex["id"].as_str().ok_or("an example has no id")?;
        let sql = ex["sql"].as_str().ok_or("an example has no sql")?;
        engine
            .reset(&SessionSetup::plain(TestKind::Regular))
            .map_err(|e| e.message)?;
        for s in ex["setup"].as_array().into_iter().flatten() {
            let s = s.as_str().ok_or("setup entries are strings")?;
            engine
                .execute(s)
                .map_err(|e| format!("{id}: setup `{s}` failed: {}", e.message))?;
        }
        out.push(match engine.execute(sql) {
            Ok(r) => json!({"id": id, "ok": true, "rows": r.rows.iter().map(|c| c.join("\t")).collect::<Vec<_>>()}),
            Err(e) => json!({"id": id, "ok": false, "class": e.class, "message": e.message}),
        });
    }
    Ok(Value::Array(out))
}

fn main() -> ExitCode {
    let mut input = String::new();
    let result = std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())
        .and_then(|_| run(&input));
    match result {
        Ok(v) => {
            drop(writeln!(std::io::stdout().lock(), "{v}"));
            ExitCode::SUCCESS
        }
        Err(e) => {
            drop(writeln!(std::io::stderr().lock(), "doc-examples: {e}"));
            ExitCode::from(2)
        }
    }
}
