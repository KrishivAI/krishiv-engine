// SPDX-License-Identifier: Apache-2.0
//! The salted directory's load (`krishiv.exec.join.directoryLoad`; perf-2,
//! the S5 q03 regression at SF1000): under `auto` a build whose pool grants
//! the extra bytes at once takes the roomy directory (load ≤ 1/2), a build
//! under memory pressure keeps the compact one (load ≤ 4/5), and every
//! setting returns the same rows, in memory and spilled. `EXPLAIN ANALYZE`
//! names the kind (`hash_index`).

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_exec::{ExecConfig, JoinDirectoryLoad, JoinSpec, JoinType, Metric, QueryContext};

use super::util::*;

const COMPACT: &str = "row-salted+chain";
const ROOMY: &str = "row-salted-roomy+chain";

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

fn data(keys: impl Iterator<Item = i64>) -> RecordBatch {
    let k: Vec<i64> = keys.collect();
    let v: Vec<i64> = k.iter().map(|k| k * 3 + 1).collect();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int64Array::from(k)), Arc::new(Int64Array::from(v))],
    )
    .unwrap()
}

/// 200,000 distinct build keys: 25,000 per level-0 partition, far past the
/// flat index's size, so every table is salted.
const BUILD: i64 = 200_000;

/// The build's `hash_index` text (empty when no table was built in memory).
fn hash_index(q: &QueryContext) -> String {
    q.metrics()
        .operators
        .values()
        .filter(|e| e.name == "HashJoinBuild")
        .flat_map(|e| e.extra.iter())
        .find_map(|(n, v)| match (n.as_ref(), v) {
            ("hash_index", Metric::Text(t)) => Some(t.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// The inner join of every third key in `0..2 * BUILD` (half of them miss)
/// with the build, under `limit` bytes; returns the rows, the build's index
/// kinds and its spilled partitions.
async fn run_join(
    load: JoinDirectoryLoad,
    limit: usize,
    spills: bool,
) -> (Vec<String>, String, u64) {
    try_join(load, limit, spills).await.unwrap()
}

/// [`run_join`], or the query's error (the pool released either way).
async fn try_join(
    load: JoinDirectoryLoad,
    limit: usize,
    spills: bool,
) -> krishiv_exec::Result<(Vec<String>, String, u64)> {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        join_row_layout: true,
        join_directory_load: load,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let spec = JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: schema(),
        build_schema: schema(),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: spills.then(|| scope.clone()),
    };
    let plan = join(
        spec,
        scan(&schema(), batches(&data(0..BUILD), 4096)),
        scan(&schema(), batches(&data((0..2 * BUILD).step_by(3)), 4096)),
    );
    let out = run(&runner_for(2, limit), plan, &q).await;
    assert_released(&root);
    let out = out?;
    let kind = hash_index(&q);
    // `roomy_partitions` counts the roomy tables: none, all 8 level-0
    // partitions, or some of them when `hash_index` lists both kinds.
    let roomy = metric(&q, "HashJoinBuild", "roomy_partitions");
    match kind.as_str() {
        ROOMY => assert_eq!(roomy, 8, "{kind}"),
        k if k.contains(ROOMY) => assert!((1..8).contains(&roomy), "{kind}: {roomy}"),
        _ => assert_eq!(roomy, 0, "{kind}"),
    }
    Ok((
        render(&out),
        kind,
        metric(&q, "HashJoinBuild", "spilled_partitions"),
    ))
}

/// With memory to spare, `auto` and `roomy` build roomy directories and
/// `compact` compact ones, with or without a spill scope (a build without
/// one asks for the queued partitions' needs too); the rows are the same.
#[tokio::test]
async fn ample_memory_takes_the_roomy_directory() {
    let (expect, kind, spilled) = run_join(JoinDirectoryLoad::Compact, 256 * MIB, true).await;
    assert_eq!(kind, COMPACT);
    assert_eq!(spilled, 0);
    assert_eq!(expect.len(), 66_667);
    for spills in [true, false] {
        for load in [JoinDirectoryLoad::Auto, JoinDirectoryLoad::Roomy] {
            let (got, kind, spilled) = run_join(load, 256 * MIB, spills).await;
            assert_eq!(kind, ROOMY, "{load:?}, spill scope {spills}");
            assert_eq!(spilled, 0);
            assert!(got == expect, "{load:?}: other rows");
        }
    }
}

/// At the smallest budget (in 1 MiB steps) where the compact build stays in
/// memory, `auto` builds the same tables in memory and cannot give every
/// partition the roomy directory (it would need 3 MiB more); under a budget
/// that spills, every setting returns the in-memory rows.
#[tokio::test(start_paused = true)]
async fn a_tight_pool_keeps_the_compact_directory() {
    let (expect, _, _) = run_join(JoinDirectoryLoad::Compact, 256 * MIB, true).await;
    let mut fits = None;
    for mib in 8..48 {
        let (_, kind, spilled) = run_join(JoinDirectoryLoad::Compact, mib * MIB, true).await;
        if spilled == 0 && kind == COMPACT {
            fits = Some(mib * MIB);
            break;
        }
    }
    let limit = fits.unwrap_or_else(|| panic!("no budget under 48 MiB holds the build"));
    let (got, kind, spilled) = run_join(JoinDirectoryLoad::Auto, limit, true).await;
    assert_eq!(spilled, 0, "auto reserves the compact need first");
    assert!(
        kind.split("; ").any(|k| k == COMPACT),
        "auto at {limit} bytes: {kind}"
    );
    assert!(got == expect, "auto at {limit} bytes: other rows");
    for load in [
        JoinDirectoryLoad::Auto,
        JoinDirectoryLoad::Compact,
        JoinDirectoryLoad::Roomy,
    ] {
        let (got, _, spilled) = run_join(load, 2 * MIB, true).await;
        assert!(spilled > 0, "{load:?}: the build spills");
        assert!(got == expect, "{load:?}: spilled rows differ");
    }
}

/// Without a spill scope a build cannot spill, so a partition the pool
/// refuses waits and, with nothing else running, fails the query. At the
/// smallest budget (in 1 MiB steps) where the compact build succeeds, `auto`
/// succeeds too with the same rows: a partition takes the roomy directory
/// only if the pool also holds what the partitions still queued will ask
/// for, so it never leaves a later one short (review F2).
#[tokio::test(start_paused = true)]
async fn without_a_spill_scope_roomy_never_starves_a_later_partition() {
    let (expect, _, _) = run_join(JoinDirectoryLoad::Compact, 256 * MIB, false).await;
    let mut fits = None;
    for mib in 8..48 {
        if let Ok((got, kind, _)) = try_join(JoinDirectoryLoad::Compact, mib * MIB, false).await {
            assert!(got == expect, "compact at {mib} MiB: other rows");
            assert_eq!(kind, COMPACT);
            fits = Some(mib * MIB);
            break;
        }
    }
    let limit = fits.unwrap_or_else(|| panic!("no budget under 48 MiB holds the build"));
    let (got, kind, _) = try_join(JoinDirectoryLoad::Auto, limit, false)
        .await
        .unwrap_or_else(|e| panic!("auto at {limit} bytes failed: {e}"));
    assert!(got == expect, "auto at {limit} bytes: other rows");
    assert!(
        kind.split("; ").any(|k| k == COMPACT),
        "auto at {limit} bytes: {kind}"
    );
}
