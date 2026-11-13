// SPDX-License-Identifier: Apache-2.0
//! A hash join's build goes back once every probe driver is done with it,
//! before the sink above the probe finalizes (design `P2-q16-build-release`).

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use krishiv_exec::testing::{BlockingGate, Break, Buffer, Limit, Pipe};
use krishiv_exec::{ExecConfig, ExecNode, JoinType};
use krishiv_memory::{MemoryPool, SpillScope};

use super::util::*;
use super::{Row, case, data, r, schema};

/// A join of `build` with `probe` under the pool's `limit`, spilling into
/// `spill`, on `drivers` drivers, below a sink whose `finalize` waits at a
/// gate: what the pool holds while it waits, what the build's in-memory
/// partitions had reserved, and the rows.
async fn reserved_at_finalize(
    join_type: JoinType,
    build: &[Row],
    probe: &[Row],
    drivers: usize,
    limit: usize,
    spill: Option<SpillScope>,
) -> (usize, u64, Vec<String>) {
    let (held, build_bytes, _, out) =
        held_at_finalize(join_type, build, probe, drivers, limit, spill).await;
    (held, build_bytes, out)
}

/// As [`reserved_at_finalize`], also returning the bytes the sink itself
/// had reserved for the rows it buffered (the batches it replays).
async fn held_at_finalize(
    join_type: JoinType,
    build: &[Row],
    probe: &[Row],
    drivers: usize,
    limit: usize,
    spill: Option<SpillScope>,
) -> (usize, u64, u64, Vec<String>) {
    const HANG: Duration = Duration::from_secs(60);
    let root = root(limit);
    let q = query(&root, 1, ExecConfig::default());
    let gate = BlockingGate::new();
    let sink = Buffer::with_finalize_gate(Arc::clone(&gate));
    let plan = join(
        case(join_type, 1, false, false).spec(spill),
        scan(&schema(), batches(&data(build), 1024)),
        scan(&schema(), batches(&data(probe), 256)),
    );
    let plan: Arc<dyn ExecNode> = Arc::new(Break(plan, sink, false));
    let mut stream = runner(drivers).execute(plan, Arc::clone(&q)).await.unwrap();
    let reached = tokio::time::timeout(HANG, async {
        while !gate.reached() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    let held = root.reserved();
    gate.open(); // before asserting, so a failure never leaves it shut
    assert!(reached.is_ok(), "the sink above the probe finalized");
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b.unwrap());
    }
    drop(stream);
    q.finished().await;
    assert_released(&root);
    let build_bytes = metric(&q, "HashJoinBuild", "steady_reserved_bytes");
    let sink_bytes = out
        .iter()
        .map(|b| b.data().get_array_memory_size() as u64)
        .sum();
    (held, build_bytes, sink_bytes, render(&out))
}

fn rows(n: i64) -> Vec<Row> {
    (0..n)
        .map(|i| r(Some(i), Some("a key of some length"), i))
        .collect()
}

/// While the sink above the probe finalizes, the pool holds the sink's
/// buffered rows but not the build's tables: the last probe driver to
/// finish released them.
#[tokio::test]
async fn the_build_goes_back_before_the_sink_above_the_probe_finalizes() {
    let build = rows(40_000);
    let probe: Vec<Row> = (0..2_000).map(|i| r(Some(i * 7), Some("p"), i)).collect();
    for drivers in [1, 4] {
        let (held, build_bytes, out) =
            reserved_at_finalize(JoinType::Inner, &build, &probe, drivers, 256 * MIB, None).await;
        assert_eq!(out.len(), 2_000, "{drivers} drivers: the join's rows");
        assert!(
            build_bytes > MIB as u64,
            "{drivers} drivers: a build of some size"
        );
        assert!(
            (held as u64) < build_bytes,
            "{drivers} drivers: the pool held {held} bytes at the sink's finalize, the build \
             {build_bytes}"
        );
    }
}

/// A join that emits its build rows after the probe (right outer) emits
/// them all before its tables go back.
#[tokio::test]
async fn a_right_outer_join_emits_its_build_rows_before_the_release() {
    let build = rows(40_000);
    let probe: Vec<Row> = (0..500).map(|i| r(Some(i * 3), Some("p"), i)).collect();
    for drivers in [1, 4] {
        let (held, build_bytes, sink_bytes, out) = held_at_finalize(
            JoinType::RightOuter,
            &build,
            &probe,
            drivers,
            256 * MIB,
            None,
        )
        .await;
        // Every build row once: 500 matched, the rest with NULL probe columns.
        assert_eq!(out.len(), 40_000, "{drivers} drivers: every build row");
        // The sink holds the emitted rows, not the tables beside them.
        // Measured: released, about 14 KiB beside the sink's rows; with the
        // release skipped, more than the whole build.
        let beside_sink = (held as u64).saturating_sub(sink_bytes);
        assert!(
            beside_sink < build_bytes / 2,
            "{drivers} drivers: held {beside_sink} beside the sink's {sink_bytes} bytes, \
             build {build_bytes}"
        );
    }
}

/// A build that spilled some partitions: the grace phase joins them after
/// the in-memory ones, and the release waits for every driver's grace work
/// (a regression guard: it passes with or without the release).
#[tokio::test(start_paused = true)]
async fn a_spilled_build_joins_every_row_before_the_release() {
    let build = rows(40_000);
    let probe: Vec<Row> = (0..40_000).map(|i| r(Some(i), Some("p"), i)).collect();
    let (_manager, scope) = spill();
    let limit = MIB;
    let root = root(limit);
    let q = query(&root, 1, ExecConfig::default());
    let plan = join(
        case(JoinType::Inner, 1, false, false).spec(Some(scope)),
        scan(&schema(), batches(&data(&build), 1024)),
        scan(&schema(), batches(&data(&probe), 256)),
    );
    let plan: Arc<dyn ExecNode> = Arc::new(Break(plan, Buffer::new(true, None), false));
    let out = run(&runner_for(2, limit), plan, &q).await.unwrap();
    assert_eq!(
        out.iter()
            .map(krishiv_types::batch::Batch::num_rows)
            .sum::<usize>(),
        40_000
    );
    assert!(
        metric(&q, "HashJoinBuild", "spilled_partitions") > 0,
        "the build spilled"
    );
    assert_released(&root);
}

/// A limit above the probe ends drivers before their flush: the drivers that
/// left early release nothing, and the query still ends with its pool empty
/// (a regression guard: it passes with or without the release).
#[tokio::test]
async fn a_limit_above_the_probe_ends_the_join_cleanly() {
    let build = rows(40_000);
    for drivers in [1, 4] {
        let root = root(256 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = join(
            case(JoinType::Inner, 1, false, false).spec(None),
            scan(&schema(), batches(&data(&build), 1024)),
            scan(&schema(), batches(&data(&build), 512)),
        );
        let plan = Arc::new(Pipe(plan, Limit::new(10)));
        let out = run(&runner(drivers), plan, &q).await.unwrap();
        assert_eq!(
            out.iter()
                .map(krishiv_types::batch::Batch::num_rows)
                .sum::<usize>(),
            10
        );
        assert_released(&root);
    }
}
