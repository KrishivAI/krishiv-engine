// SPDX-License-Identifier: Apache-2.0
//! How often one query broadcasts the same subtree twice over TPC-DS, the
//! opportunity for Spark's exchange reuse (design P2-T052b, owner question 2).

use std::collections::HashMap;
use std::path::Path;

use krishiv_exec::physical::{Partitioning, PhysNode};

/// Replaces each id in a node's debug text by the order it first appears in,
/// so that two copies of a subtree planned apart print the same. An id is a
/// number after a name ending in `Id(` or after `id: `; literals stay.
fn normalized(text: &str) -> String {
    let mut ids: HashMap<&str, usize> = HashMap::new();
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(|c: char| c.is_ascii_digit()) {
        let (before, tail) = rest.split_at(start);
        let end = tail
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(tail.len());
        let (number, after) = tail.split_at(end);
        out.push_str(before);
        if before.ends_with("Id(") || before.ends_with("id: ") {
            let next = ids.len();
            out.push_str(&ids.entry(number).or_insert(next).to_string());
        } else {
            out.push_str(number);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

fn broadcasts(node: &PhysNode, out: &mut Vec<String>) {
    if let PhysNode::Exchange(e) = node
        && matches!(e.partitioning, Partitioning::Broadcast)
    {
        out.push(normalized(&format!("{:?}", e.input)));
    }
    for c in node.children() {
        broadcasts(c, out);
    }
}

#[test]
fn ids_are_normalized_and_literals_are_not() {
    assert_eq!(
        normalized("ExchangeId(7) id: 9 Int(7) ExchangeId(7)"),
        normalized("ExchangeId(3) id: 4 Int(7) ExchangeId(3)"),
    );
    assert_ne!(normalized("Int(7)"), normalized("Int(8)"));
}

/// Counts them over the planned TPC-DS queries (`KRISHIV_REUSE_DIR`, one
/// `<table>.parquet` per table). Run with
/// `cargo test -p krishiv-exec --test planner broadcast_reuse_count -- --ignored --nocapture`.
#[test]
#[ignore = "needs the TPC-DS data set"]
fn broadcast_reuse_count() {
    let dir = std::env::var("KRISHIV_REUSE_DIR").expect("KRISHIV_REUSE_DIR");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "parquet").then(|| p.file_stem()?.to_str().map(str::to_owned))?
        })
        .collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    let t = super::tpch::parquet_dir(Path::new(&dir), &names, true, &rt);
    let queries = super::tpch::root().join("bench/queries/tpcds");
    let mut files: Vec<_> = std::fs::read_dir(&queries)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "sql"))
        .collect();
    files.sort();
    let (mut planned, mut with_repeat, mut total, mut repeats, mut failed) = (0, 0, 0, 0, 0);
    for f in files {
        let q = f.file_stem().unwrap().to_string_lossy().into_owned();
        let Ok(p) = t.plan_sql(&std::fs::read_to_string(&f).unwrap()) else {
            failed += 1;
            continue;
        };
        planned += 1;
        let mut found = Vec::new();
        for root in p.plan.roots() {
            broadcasts(root, &mut found);
        }
        total += found.len();
        let n = found.len();
        found.sort();
        found.dedup();
        if found.len() < n {
            with_repeat += 1;
            repeats += n - found.len();
            println!("{q}: {} of {n} broadcasts repeat one", n - found.len());
        }
    }
    println!(
        "planned {planned}, failed {failed}: {total} broadcasts, {repeats} repeats, in {with_repeat} queries"
    );
}
