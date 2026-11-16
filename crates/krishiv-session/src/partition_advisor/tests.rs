// SPDX-License-Identifier: Apache-2.0
//! The partition advisor over a real history: a planted workload, the `EXPLAIN` line and the system table.

use std::sync::Arc;

use futures::executor::block_on;

use super::*;
use crate::{EngineRuntime, Session};

/// The scan nodes of the newest stored profile that has one, for a failure message.
fn scans_dump(dir: &std::path::Path) -> String {
    fn walk(n: &krishiv_exec::profile::ProfileNode, out: &mut String) {
        if n.name == "Scan" {
            out.push_str(&format!("{} facts={:?}\n", n.detail, n.facts));
        }
        n.children.iter().for_each(|c| walk(c, out));
    }
    let mut out = String::new();
    for s in crate::profile_history::read_history(dir)
        .unwrap()
        .iter()
        .rev()
        .take(2)
    {
        if let Ok(p) = krishiv_exec::profile::QueryProfile::from_json(&s.profile) {
            walk(&p.plan, &mut out);
        }
    }
    out
}

fn history_session(dir: &std::path::Path) -> (Arc<EngineRuntime>, Session) {
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", dir.to_str().unwrap())
        .build()
        .unwrap();
    let s = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    (rt, s)
}

fn run(s: &Session, sql: &str) {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.collect()).unwrap();
}

/// A Parquet table `ev(id, d, c)` of 100,000 rows: `d` has 365 values, `c` 3 and `id` is unique; queried 100
/// times by `d`, 20 by `c` and 50 by `id`.
fn planted() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    Arc<EngineRuntime>,
    Session,
) {
    let hist = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let (rt, s) = history_session(hist.path());
    let path = data.path().join("ev");
    let df =
        block_on(s.sql("SELECT id, id % 365 AS d, id % 3 AS c FROM range(0, 100000)")).unwrap();
    block_on(df.write().parquet(path.to_str().unwrap())).unwrap();
    run(
        &s,
        &format!(
            "CREATE TABLE ev (id BIGINT, d BIGINT, c BIGINT) USING parquet LOCATION '{}'",
            path.display()
        ),
    );
    for i in 0..100 {
        run(
            &s,
            &format!("SELECT count(*) FROM ev WHERE d = {}", i % 365),
        );
    }
    for i in 0..20 {
        run(&s, &format!("SELECT count(*) FROM ev WHERE c = {}", i % 3));
    }
    for i in 0..50 {
        run(&s, &format!("SELECT count(*) FROM ev WHERE id = {i}"));
    }
    rt.profile_history.as_ref().unwrap().flush();
    (hist, data, rt, s)
}

#[test]
fn advisor_finds_planted_partition_key() {
    let (hist, _data, _rt, _s) = planted();
    let all = advise_history(hist.path());
    let ev = all
        .iter()
        .find(|t| t.table.ends_with("ev"))
        .unwrap_or_else(|| panic!("no advice for ev: {all:#?}"));
    assert!(
        !ev.advice.is_empty(),
        "{all:#?}\n{}",
        scans_dump(hist.path())
    );
    assert_eq!(ev.advice[0].column, "d", "{ev:#?}");
    assert_eq!(ev.advice[1].column, "c", "{ev:#?}");
    assert!(ev.advice[0].pruned_fraction > ev.advice[1].pruned_fraction);
    // The history masks literals: the advice never needed them.
    assert!(ev.queries >= 170);
}

#[test]
fn advisor_rejects_unique_key() {
    let (hist, _data, _rt, _s) = planted();
    let all = advise_history(hist.path());
    let ev = all.iter().find(|t| t.table.ends_with("ev")).unwrap();
    assert!(
        !ev.advice.is_empty(),
        "the unique key test needs advice to reject from: {all:#?}"
    );
    assert!(
        ev.advice.iter().all(|a| a.column != "id"),
        "a unique key is never advised: {ev:#?}"
    );
}

#[test]
fn explain_and_the_system_table_show_the_advice() {
    let (_hist, _data, _rt, s) = planted();
    let df = block_on(s.sql("EXPLAIN SELECT count(*) FROM ev WHERE d = 1")).unwrap();
    let batches = block_on(df.collect()).unwrap();
    let text = format!("{batches:?}");
    assert!(text.contains("partition advice"), "{text}");
    let df = block_on(s.sql(
        "SELECT table_name, column_name, rank FROM system.partition_advice ORDER BY table_name, rank",
    ))
    .unwrap();
    let batches = block_on(df.collect()).unwrap();
    let rows: usize = batches.iter().map(arrow_array::RecordBatch::num_rows).sum();
    assert!(rows >= 2, "{batches:?}");
}
