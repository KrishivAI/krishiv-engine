// SPDX-License-Identifier: Apache-2.0
//! Hybrid hash join (`krishiv.exec.join.hybridResident`): a build that spills
//! keeps the level-0 partitions that fit in memory, so only the probe rows
//! of the partitions on disk go through the grace phase.

use std::sync::Arc;

use krishiv_exec::{ExecConfig, JoinType};

use super::util::*;
use super::{Row, case, data, r, schema};

/// Rows with `keys` distinct keys, spread evenly.
fn rows(n: usize, keys: i64) -> Vec<Row> {
    (0..n)
        .map(|i| {
            let i = i64::try_from(i).unwrap();
            r(
                Some(i % keys),
                Some(["a", "b"][usize::from(i % 3 == 0)]),
                i % 50,
            )
        })
        .collect()
}

/// The join of `build` and `probe` under `limit` with 64 level-0
/// partitions, with or without resident partitions.
async fn run_hybrid(
    join_type: JoinType,
    build: &[Row],
    probe: &[Row],
    limit: usize,
    hybrid: bool,
) -> (Vec<String>, Arc<krishiv_exec::QueryContext>) {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 1024,
        join_row_layout: true,
        join_radix_bits: 6,
        join_hybrid_resident: hybrid,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let c = case(join_type, 1, false, false);
    let plan = join(
        c.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), 1024)),
        scan(&schema(), batches(&data(probe), 1024)),
    );
    let out = run(&runner_for(2, limit), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    (render(&out), q)
}

const LIMIT: usize = 3 << 20;
const LIMIT_KEYS: usize = 1 << 20;

/// A build somewhat past the budget keeps part of its partitions: fewer
/// spill than with every partition sent to disk, fewer probe rows are
/// deferred, and the answer is the in-memory join's, for each join type
/// that defers probe rows or emits build rows by their marks.
#[tokio::test(start_paused = true)]
async fn a_spilling_build_keeps_the_partitions_that_fit() {
    let build = rows(30_000, 20_000);
    let probe = rows(30_000, 25_000);
    for jt in [
        JoinType::Inner,
        JoinType::LeftOuter,
        JoinType::RightOuter,
        JoinType::FullOuter,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
    ] {
        let (expect, mem) = run_hybrid(jt, &build, &probe, 256 << 20, true).await;
        assert_eq!(metric(&mem, "HashJoinBuild", "spilled_partitions"), 0);
        // A key-only build (semi, anti) is smaller.
        let limit = if matches!(jt, JoinType::LeftSemi | JoinType::LeftAnti) {
            LIMIT_KEYS
        } else {
            LIMIT
        };
        let (all, a) = run_hybrid(jt, &build, &probe, limit, false).await;
        let (got, b) = run_hybrid(jt, &build, &probe, limit, true).await;
        assert!(all == expect, "{jt:?}: grace result differs");
        assert!(got == expect, "{jt:?}: hybrid result differs");
        let spilled_all = metric(&a, "HashJoinBuild", "spilled_partitions");
        let spilled = metric(&b, "HashJoinBuild", "spilled_partitions");
        assert_eq!(spilled_all, 64, "{jt:?}: every partition spills when off");
        assert!(
            spilled > 0 && spilled < 64,
            "{jt:?}: {spilled} of 64 spilled"
        );
        let deferred_all = metric(&a, "HashJoinProbe", "probe_spilled_rows");
        let deferred = metric(&b, "HashJoinProbe", "probe_spilled_rows");
        assert!(
            deferred < deferred_all,
            "{jt:?}: {deferred} vs {deferred_all}"
        );
    }
}
