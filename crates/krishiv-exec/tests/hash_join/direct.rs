// SPDX-License-Identifier: Apache-2.0
//! The direct-index probe of tiny builds (perf lever J1,
//! `krishiv.exec.join.directIndex`): every join type returns the oracle's
//! rows with it on and off, over negative, sparse, duplicate and NULL keys
//! and probe keys outside the build's span; it runs only where it applies.

use super::*;

/// Runs `case` (one `k1` key, row layout) with the direct index on or off;
/// also returns the probe's `direct_probe_rows`.
async fn run_direct(case: &Case, probe: &[Row], build: &[Row], on: bool) -> (Vec<String>, u64) {
    let root = root(64 * MIB);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 5,
        join_row_layout: case.row_layout,
        join_direct_index: on,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), 3)),
        scan(&schema(), batches(&data(probe), 4)),
    );
    let out = run(&runner_for(2, 64 * MIB), plan, &q).await.unwrap();
    assert_released(&root);
    (
        render(&out),
        metric(&q, "HashJoinProbe", "direct_probe_rows"),
    )
}

fn keyed(keys: &[Option<i64>]) -> Vec<Row> {
    keys.iter()
        .enumerate()
        .map(|(i, &k)| r(k, Some("x"), i64::try_from(i).unwrap() % 20))
        .collect()
}

/// Build keys from -7 to 9 with gaps, duplicates and NULLs; probe keys inside
/// the span, in its gaps, outside it on both sides, at the type's limits and
/// NULL.
fn sides() -> (Vec<Row>, Vec<Row>) {
    let build = keyed(&[
        Some(-7),
        Some(0),
        Some(3),
        Some(3),
        None,
        Some(9),
        Some(3),
        Some(-7),
        None,
        Some(5),
    ]);
    let probe = keyed(&[
        Some(3),
        Some(-7),
        Some(4),
        Some(-8),
        Some(10),
        Some(i64::MIN),
        Some(i64::MAX),
        None,
        Some(9),
        Some(0),
        Some(5),
        Some(3),
        Some(1),
        None,
        Some(-7),
    ]);
    (probe, build)
}

#[tokio::test]
async fn every_join_type_matches_the_oracle_with_the_direct_index() {
    let (probe, build) = sides();
    for join_type in TYPES {
        for filter in [false, true] {
            let filter = filter && join_type != JoinType::NullAwareAnti;
            let mut c = case(join_type, 1, false, filter);
            c.row_layout = true;
            let want = c.oracle(&probe, &build);
            let (on, direct) = run_direct(&c, &probe, &build, true).await;
            let (off, none) = run_direct(&c, &probe, &build, false).await;
            assert_eq!(on, want, "{join_type:?} filter {filter}");
            assert_eq!(off, want, "{join_type:?} filter {filter}");
            assert_eq!(none, 0, "{join_type:?}: switched off");
            // A null-aware anti join keeps the hashed path (it reads whether
            // the build holds a NULL key).
            if join_type == JoinType::NullAwareAnti {
                assert_eq!(direct, 0, "{join_type:?}");
            } else {
                assert_eq!(direct, probe.len() as u64, "{join_type:?}");
            }
        }
    }
}

/// Where the index does not apply, the hashed path answers: a key span wider
/// than four slots a row, two keys, null-safe keys, arrow-row keys.
#[tokio::test]
async fn builds_the_direct_index_does_not_cover_keep_the_hashed_probe() {
    let (probe, _) = sides();
    let sparse = keyed(&[Some(0), Some(1_000_000), Some(3), None]);
    let (_, build) = sides();
    let mut wide = case(JoinType::Inner, 1, false, false);
    wide.row_layout = true;
    let mut two = case(JoinType::LeftOuter, 2, false, false);
    two.row_layout = true;
    let mut safe = case(JoinType::Inner, 1, true, false);
    safe.row_layout = true;
    let arrow = case(JoinType::LeftSemi, 1, false, false);
    for (c, build) in [
        (&wide, &sparse),
        (&two, &build),
        (&safe, &build),
        (&arrow, &build),
    ] {
        let (got, direct) = run_direct(c, &probe, build, true).await;
        assert_eq!(got, c.oracle(&probe, build), "{c:?}");
        assert_eq!(direct, 0, "{c:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]
    /// Random one-key joins over small key spans (so the index applies):
    /// the oracle's answer either way.
    #[test]
    fn random_direct_joins_match_the_oracle(
        t in 0usize..TYPES.len(),
        filter in any::<bool>(),
        probe in prop::collection::vec(row(12, 4), 0..60),
        build in prop::collection::vec(row(12, 4), 0..60),
    ) {
        let join_type = TYPES[t];
        let mut c = case(join_type, 1, false, filter && join_type != JoinType::NullAwareAnti);
        c.row_layout = true;
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (got, _) = rt.block_on(run_direct(&c, &probe, &build, true));
        prop_assert_eq!(got, c.oracle(&probe, &build));
    }
}
