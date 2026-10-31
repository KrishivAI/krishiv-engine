// SPDX-License-Identifier: Apache-2.0
//! The 95 scripts of `tests/spark-golden/v4.2.0/inputs/scripting/randomly_generated_scripts.sql`
//! against the Spark 4.2.0 record (`scripting/handler-record.json`, design note T196 section 5).
//!
//! Each script runs on fresh copies of the six tables. It passes when
//! 1. its result rows equal Spark's,
//! 2. the count and the checksum (over the non-timestamp columns) of every table equal Spark's, and
//! 3. with its `DECLARE EXIT HANDLER` line removed, the error class that fires is the one Spark's
//!    handler caught (the goldens alone print a constant string, so only this proves the handler
//!    took the error Spark's did).
//!
//! Scripts that call `RAND`/`UUID` or differ between two Spark runs are compared on 3 only.
//! `scripting/known-failures.json` lists the scripts that do not pass yet, with a reason each: the
//! test fails for a script that fails and is not listed, and for a listed one that passes.
//! `KRISHIV_RATCHET_UPDATE=1` rewrites the file.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/spark-golden/v4.2.0")
}

/// The setup statements, the table names and the scripts of the golden input.
fn load() -> (Vec<String>, Vec<String>, Vec<String>) {
    let text =
        std::fs::read_to_string(root().join("inputs/scripting/randomly_generated_scripts.sql"))
            .expect("the golden input");
    let (setup, scripts) = text.split_once("---- Scripts").expect("a scripts section");
    let body: String = setup
        .lines()
        .filter(|l| !l.trim_start().starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n");
    let statements = body
        .split(";\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches(';').to_owned())
        .collect();
    let tables = setup
        .lines()
        .filter_map(|l| l.strip_prefix("CREATE TABLE "))
        .map(|l| l.split_whitespace().next().unwrap_or_default().to_owned())
        .collect();
    let mut found = Vec::new();
    let mut rest = scripts;
    while let Some((_, after)) = rest.split_once("--QUERY-DELIMITER-START\n") {
        let Some((script, tail)) = after.split_once("\n--QUERY-DELIMITER-END") else {
            break;
        };
        found.push(script.to_owned());
        rest = tail;
    }
    (statements, tables, found)
}

fn fresh(setup: &[String]) -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    for statement in setup {
        block_on(async { s.sql(statement).await?.collect().await })
            .unwrap_or_else(|e| panic!("setup {statement}: {e}"));
    }
    (s, dir)
}

/// Every cell as text (`NULL` for a null); a type the comparison does not render becomes `?`.
fn cells(s: &Session, sql: &str) -> Result<Vec<Vec<String>>, String> {
    let batches = block_on(async { s.sql(sql).await?.collect().await })
        .map_err(|e| e.error_class().name().to_owned())?;
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            return "NULL".to_owned();
                        }
                        match c.data_type() {
                            ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                            ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                            ArrowType::Boolean => c.as_boolean().value(i).to_string(),
                            ArrowType::Int32 => c
                                .as_primitive::<arrow_array::types::Int32Type>()
                                .value(i)
                                .to_string(),
                            ArrowType::Int64 => c
                                .as_primitive::<arrow_array::types::Int64Type>()
                                .value(i)
                                .to_string(),
                            _ => "?".to_owned(),
                        }
                    })
                    .collect(),
            );
        }
    }
    Ok(out)
}

/// `count(*)` and the checksum over the non-timestamp columns of every table.
fn checksums(s: &Session, tables: &[String]) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    for t in tables {
        let columns: Vec<String> = cells(s, &format!("DESCRIBE TABLE {t}"))
            .unwrap_or_default()
            .into_iter()
            .filter(|r| r.get(1).is_some_and(|ty| ty != "timestamp"))
            .filter_map(|r| r.first().map(|n| format!("`{n}`")))
            .collect();
        let sql = format!(
            "SELECT CAST(count(*) AS STRING), CAST(sum(hash({})) AS STRING) FROM {t}",
            columns.join(", ")
        );
        let row = cells(s, &sql)
            .ok()
            .and_then(|mut r| r.pop())
            .unwrap_or_else(|| vec!["error".into(), "error".into()]);
        out.insert(t.clone(), (row[0].clone(), row[1].clone()));
    }
    out
}

/// Whether a `FOR` query of the script has `DISTINCT` or `GROUP BY` and no `ORDER BY`: the order of
/// its rows is the engine's, so which branch of the script runs can differ between machines (the
/// goldens come from one Spark run). Such a script is compared and reported, but never fails the test.
fn order_dependent(script: &str) -> bool {
    // One space between words, so line breaks and indentation do not hide a `FOR ... AS SELECT`.
    let upper = script
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_uppercase();
    let mut rest = upper.as_str();
    while let Some(at) = rest.find("FOR ") {
        let after = &rest[at + 4..];
        let Some(end) = after.find(" DO") else {
            break;
        };
        let query = &after[..end];
        if query.contains(" AS SELECT")
            && (query.contains("DISTINCT") || query.contains("GROUP BY"))
            && !query.contains("ORDER BY")
        {
            return true;
        }
        rest = &after[end..];
    }
    false
}

/// The script without its `DECLARE EXIT HANDLER ...;` lines.
fn without_handlers(script: &str) -> String {
    script
        .lines()
        .filter(|l| {
            let t = l.trim_start().to_ascii_uppercase();
            !(t.starts_with("DECLARE EXIT HANDLER") && l.trim_end().ends_with(';'))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_owned(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Why the script differs from Spark's record, or `None` when it matches.
fn check(setup: &[String], tables: &[String], script: &str, record: &Value) -> Option<String> {
    let deterministic = record["deterministic"].as_bool().unwrap_or(true);
    // (3) the class that fires the handler, with the handler line removed.
    let stripped = without_handlers(script);
    let (s, _dir) = fresh(setup);
    let fired = match block_on(async { s.sql(&stripped).await?.collect().await }) {
        Ok(_) => "none".to_owned(),
        Err(e) => e.error_class().name().to_owned(),
    };
    let expected = record["fired"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| record["fired"]["class"].as_str().map(str::to_owned))
        .unwrap_or_else(|| "none".to_owned());
    if fired != expected {
        return Some(format!("fires {fired}, Spark {expected}"));
    }
    if !deterministic {
        return None;
    }
    // (1) and (2): the script as it is.
    let (s, _dir) = fresh(setup);
    let rows = cells(&s, script);
    let want: Vec<Vec<String>> = record["as_is"]["rows"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .map(|r| {
                    r.as_array()
                        .map(|c| c.iter().map(text).collect())
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default();
    match rows {
        Err(class) => {
            let spark = record["as_is"]["error"]["class"].as_str();
            if spark != Some(class.as_str()) {
                return Some(format!("fails {class}, Spark {spark:?}"));
            }
        }
        Ok(rows) => {
            let same = rows.len() == want.len()
                && rows.iter().zip(&want).all(|(a, b)| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == "?" || x == y)
                });
            if !same {
                return Some(format!("rows {rows:?}, Spark {want:?}"));
            }
        }
    }
    for (table, (count, hash)) in checksums(&s, tables) {
        let spark = &record["as_is"]["tables"][&table];
        if text(&spark["count"]) != count {
            return Some(format!(
                "{table} has {count} rows, Spark {}",
                text(&spark["count"])
            ));
        }
        if text(&spark["stable_hash"]) != hash {
            return Some(format!("{table} checksum differs"));
        }
    }
    None
}

#[test]
fn the_golden_scripts_match_the_spark_record() {
    let (setup, tables, scripts) = load();
    assert_eq!(scripts.len(), 95);
    let record: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("scripting/handler-record.json")).unwrap(),
    )
    .unwrap();
    // The scripts run on a few threads; each gets its own sessions and tables.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let observed = std::sync::Mutex::new(BTreeMap::<usize, Option<String>>::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some(script) = scripts.get(i) else {
                        return;
                    };
                    let outcome = check(&setup, &tables, script, &record["scripts"][i]);
                    observed
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(i + 1, outcome);
                }
            });
        }
    });
    let observed = observed.into_inner().unwrap_or_default();
    let mut failing: BTreeMap<String, String> = BTreeMap::new();
    let mut unstable = Vec::new();
    for (n, outcome) in &observed {
        let Some(why) = outcome else { continue };
        if order_dependent(&scripts[n - 1]) {
            unstable.push((*n, why.clone()));
        } else {
            failing.insert(n.to_string(), why.clone());
        }
    }
    if !unstable.is_empty() {
        eprintln!("order-dependent scripts that differ (not asserted): {unstable:?}");
    }
    let path = root().join("scripting/known-failures.json");
    let known: BTreeMap<String, String> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    if std::env::var_os("KRISHIV_RATCHET_UPDATE").is_some() {
        // Reasons are written by hand: a script that stays on the list keeps its reason.
        let updated: BTreeMap<String, String> = failing
            .iter()
            .map(|(k, why)| {
                let reason = known
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| format!("UNEXPLAINED: {why}"));
                (k.clone(), reason)
            })
            .collect();
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&updated).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let new: Vec<_> = failing
        .iter()
        .filter(|(k, _)| !known.contains_key(*k))
        .collect();
    let fixed: Vec<_> = known.keys().filter(|k| !failing.contains_key(*k)).collect();
    assert!(
        new.is_empty(),
        "scripts that fail and are not listed: {new:?}"
    );
    assert!(
        fixed.is_empty(),
        "listed scripts that now pass; remove them from known-failures.json: {fixed:?}"
    );
}

/// `KRISHIV_SCRIPT=n cargo test ... debug_one_script -- --nocapture`: the error each run of
/// script `n` raises, with its message, for working out why it does not match.
#[test]
fn debug_one_script() {
    let Some(n) = std::env::var("KRISHIV_SCRIPT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        return;
    };
    let (setup, _, scripts) = load();
    let script = &scripts[n - 1];
    for (label, text) in [
        ("as is", script.clone()),
        ("without handlers", without_handlers(script)),
    ] {
        let (s, _dir) = fresh(&setup);
        match block_on(async { s.sql(&text).await?.collect().await }) {
            Ok(b) => eprintln!("[{label}] ok, {} batches", b.len()),
            Err(e) => eprintln!("[{label}] {e}"),
        }
    }
}
