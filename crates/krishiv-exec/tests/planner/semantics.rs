// SPDX-License-Identifier: Apache-2.0
//! Expressions, joins, aggregates and subqueries end to end over small
//! Parquet tables (`tests/fixtures/planner`), against Spark 4.2.0's rows or
//! error classes with ANSI mode on and off
//! (`tests/scripts/gen_planner_goldens.py`).

use std::path::Path;

use super::tpch::{answers, parquet_dir, rows};

fn check(ansi: bool) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/planner");
    let db = parquet_dir(&dir, &["t", "u"], ansi, &rt);
    let mode = if ansi { "ansi" } else { "legacy" };
    let want = answers(&dir.join(format!("spark-{mode}.json")));
    let queries: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("queries.json")).unwrap()).unwrap();
    let mut failed = Vec::new();
    for (name, sql) in queries.as_object().unwrap() {
        let sql = sql.as_str().unwrap();
        let expected = &want["queries"][name];
        let got = db.plan_sql(sql).and_then(|p| super::run(&p));
        match (got, expected.get("rows"), expected.get("error")) {
            (Ok(batches), Some(rows_want), _) => {
                let rows_want: Vec<String> = rows_want
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r.as_str().unwrap().to_owned())
                    .collect();
                let got = rows(&batches);
                if got != rows_want {
                    failed.push(format!("{name}: got {got:?}\n  want {rows_want:?}"));
                }
            }
            (Err(e), _, Some(class)) if e.error_class().name() == class.as_str().unwrap() => {}
            (got, _, _) => failed.push(format!(
                "{name}: got {:?}, want {expected}",
                got.map(|b| rows(&b))
            )),
        }
    }
    assert!(failed.is_empty(), "{mode}:\n{}", failed.join("\n"));
}

#[test]
fn matches_spark_with_ansi() {
    check(true);
}

#[test]
fn matches_spark_without_ansi() {
    check(false);
}
