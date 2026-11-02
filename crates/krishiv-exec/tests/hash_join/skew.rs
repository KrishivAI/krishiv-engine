// SPDX-License-Identifier: Apache-2.0
//! Skew (design P2-T052b §2.3, §6.5): a build whose keys follow Zipf α 1.2,
//! in memory and under a budget that spills everything but the hot key, and
//! the skew-split table against split executions.

use std::time::{Duration, Instant};

use krishiv_exec::{SkewSplit, skew_split_sides};

use super::*;

/// `rows` keys under Zipf α 1.2 over `0..keys`, in a fixed scattered order:
/// each rank gets its share of the rows (largest shares first, the remainder
/// to the top ranks), so no random source and the same data on every run.
fn zipf_keys(rows: usize, keys: usize) -> Vec<i64> {
    let weights: Vec<f64> = (1..=keys).map(|r| (r as f64).powf(-1.2)).collect();
    let total: f64 = weights.iter().sum();
    let mut out = Vec::with_capacity(rows);
    for (k, w) in weights.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a share of `rows`, rounded down: the remainder goes to the top ranks below"
        )]
        let n = (w / total * rows as f64) as usize;
        out.extend(std::iter::repeat_n(i64::try_from(k).unwrap(), n));
    }
    let mut rank = 0;
    while out.len() < rows {
        out.push(i64::try_from(rank % keys).unwrap());
        rank += 1;
    }
    scatter(out)
}

/// `rows` keys spread evenly over `0..keys`.
fn uniform_keys(rows: usize, keys: usize) -> Vec<i64> {
    scatter(
        (0..rows)
            .map(|i| i64::try_from(i % keys).unwrap())
            .collect(),
    )
}

/// Moves slot `i` to `i * STEP mod n`, a permutation for a prime `STEP` that
/// does not divide `n`.
fn scatter(keys: Vec<i64>) -> Vec<i64> {
    const STEP: usize = 2_147_483_647;
    let n = keys.len();
    assert_ne!(n % STEP, 0);
    let mut out = vec![0; n];
    for (i, k) in keys.into_iter().enumerate() {
        out[(i % n * (STEP % n)) % n] = k;
    }
    out
}

/// The result's row count and two order-independent sums: the build rows'
/// `v`, and the probe key against the build key of each pair.
#[derive(Debug, PartialEq, Eq)]
struct Checksum {
    rows: usize,
    build_v: i64,
    keys: i64,
}

fn checksum(out: &[Batch]) -> Checksum {
    let mut c = Checksum {
        rows: 0,
        build_v: 0,
        keys: 0,
    };
    for b in out {
        let d = b.data();
        let probe_key = d.column(0).as_primitive::<Int64Type>();
        let build_key = d.column(3).as_primitive::<Int64Type>();
        let build_v = d.column(5).as_primitive::<Int64Type>();
        c.rows += d.num_rows();
        for i in 0..d.num_rows() {
            assert_eq!(probe_key.value(i), build_key.value(i));
            c.build_v += build_v.value(i);
            c.keys += probe_key.value(i);
        }
    }
    c
}

/// What the join must return: the probe holds every key once, so each build
/// row meets exactly one probe row.
fn expected(build_keys: &[i64]) -> Checksum {
    let n = i64::try_from(build_keys.len()).unwrap();
    Checksum {
        rows: build_keys.len(),
        build_v: n * (n - 1) / 2,
        keys: build_keys.iter().sum(),
    }
}

struct Run {
    sums: Checksum,
    wall: Duration,
    spilled: u64,
    share: Option<f64>,
    table_ns: u64,
}

async fn inner_join(build_keys: &[i64], probe_rows: usize, limit: usize) -> Run {
    let build: Vec<Row> = build_keys
        .iter()
        .enumerate()
        .map(|(i, &k)| r(Some(k), Some("a"), i64::try_from(i).unwrap()))
        .collect();
    let probe: Vec<Row> = (0..probe_rows)
        .map(|k| {
            let k = i64::try_from(k).unwrap();
            r(Some(k), Some("a"), k)
        })
        .collect();
    let build = batches(&data(&build), 4096);
    let probe = batches(&data(&probe), 4096);
    let root = root(limit);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let plan = join(
        case(JoinType::Inner, 1, false, false).spec(Some(scope.clone())),
        scan(&schema(), build),
        scan(&schema(), probe),
    );
    let started = Instant::now();
    let out = run(&runner_for(4, limit), plan, &q).await.unwrap();
    let wall = started.elapsed();
    let sums = checksum(&out);
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0);
    let metrics = q.metrics();
    Run {
        sums,
        wall,
        spilled: metric(&q, "HashJoinBuild", "spilled_partitions"),
        share: metrics.hash_probe_share(),
        table_ns: metric(&q, "HashJoinProbe", "probe_table_ns"),
    }
}

/// The CI-sized run: in memory, and under a budget the hot key fits in while
/// the whole build does not.
#[tokio::test(start_paused = true)]
async fn zipf_build_keys_match_the_uniform_run_in_memory_and_spilling() {
    const ROWS: usize = 200_000;
    const KEYS: usize = 20_000;
    let zipf = zipf_keys(ROWS, KEYS);
    let hot = zipf.iter().filter(|&&k| k == 0).count();
    assert!(hot > ROWS / 8, "the top key holds {hot} rows");
    let uniform = uniform_keys(ROWS, KEYS);

    let even = inner_join(&uniform, ROWS, 1 << 30).await;
    assert_eq!(even.sums, expected(&uniform));
    assert_eq!(even.spilled, 0);

    let skewed = inner_join(&zipf, ROWS, 1 << 30).await;
    assert_eq!(skewed.sums, expected(&zipf));
    assert_eq!(skewed.spilled, 0);
    // Every build row matches once and every value of `v` appears once, so
    // the two runs agree on all but the keys' sum.
    assert_eq!(skewed.sums.rows, even.sums.rows);
    assert_eq!(skewed.sums.build_v, even.sums.build_v);

    let spilling = inner_join(&zipf, ROWS, 12 << 20).await;
    assert_eq!(spilling.sums, expected(&zipf));
    assert!(spilling.spilled > 0, "the small budget must spill");
}

/// The table work is timed, and reaches the query's share.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_probe_reports_its_table_time_and_share() {
    let keys = uniform_keys(50_000, 5_000);
    let run = inner_join(&keys, 50_000, 1 << 30).await;
    assert!(run.table_ns > 0);
    let share = run.share.expect("a query that probed has a share");
    assert!(share > 0.0 && share.is_finite(), "{share}");
}

/// The full-size run of design §6.5: 10M build rows over 10⁶ keys, 10M unique
/// probe keys. Prints the skewed/uniform wall the design predicts at 1.5-2.5x.
/// `KRISHIV_SKEW_MIB` sets the spilling run's budget. The default, 256 MiB,
/// fails with `KRISHIV_JOIN_PARTITION_TOO_LARGE` until the hash join can fall
/// back to sort-merge (T056): loading a partition holds its chunks and their
/// concatenated copy at once, so the 1.9M-row hot key needs more than that.
/// Run by hand: `cargo nextest run -p krishiv-exec --release --run-ignored only zipf_full_size`.
#[tokio::test(start_paused = true)]
#[ignore = "10M rows: run by hand for the design's measurement"]
async fn zipf_full_size() {
    const ROWS: usize = 10_000_000;
    const KEYS: usize = 1_000_000;
    let zipf = zipf_keys(ROWS, KEYS);
    let uniform = uniform_keys(ROWS, KEYS);
    let even = inner_join(&uniform, ROWS, 1 << 36).await;
    assert_eq!(even.sums, expected(&uniform));
    let skewed = inner_join(&zipf, ROWS, 1 << 36).await;
    assert_eq!(skewed.sums, expected(&zipf));
    println!(
        "uniform {:?} (share {:?}), skewed {:?} (share {:?}), ratio {:.2}",
        even.wall,
        even.share,
        skewed.wall,
        skewed.share,
        skewed.wall.as_secs_f64() / even.wall.as_secs_f64(),
    );
    let mib = std::env::var("KRISHIV_SKEW_MIB").map_or(256, |m| m.parse().unwrap());
    let spilling = inner_join(&zipf, ROWS, mib << 20).await;
    assert_eq!(spilling.sums, expected(&zipf));
    assert!(spilling.spilled > 0);
    println!("under {mib} MiB: {:?}", spilling.wall);
}

#[test]
fn the_split_table_follows_the_design() {
    let both = SkewSplit {
        probe: true,
        build: true,
    };
    let probe = SkewSplit {
        probe: true,
        build: false,
    };
    let build = SkewSplit {
        probe: false,
        build: true,
    };
    let neither = SkewSplit {
        probe: false,
        build: false,
    };
    for (t, want) in [
        (JoinType::Inner, both),
        (JoinType::LeftOuter, probe),
        (JoinType::LeftSemi, probe),
        (JoinType::LeftAnti, probe),
        (JoinType::Existence, probe),
        (JoinType::RightOuter, build),
        (JoinType::RightSemi, build),
        (JoinType::RightAnti, build),
        (JoinType::FullOuter, neither),
        (JoinType::NullAwareAnti, neither),
    ] {
        assert_eq!(skew_split_sides(t, false), want, "{t:?}");
    }
    assert_eq!(skew_split_sides(JoinType::Existence, true), neither);
}

/// Duplicates on both sides, NULL keys, keys only one side holds.
fn split_inputs() -> (Vec<Row>, Vec<Row>) {
    let side = |rows: i64, keys: i64, offset: i64| -> Vec<Row> {
        (0..rows)
            .map(|i| {
                let k = (i % 5 != 4).then_some((i + offset) % keys);
                r(k, Some("a"), i % 11)
            })
            .collect()
    };
    (side(120, 17, 0), side(90, 13, 6))
}

/// Runs the split side as three instantiations, each with the other side
/// whole, for every legal pair: the union equals the unsplit run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn split_executions_equal_the_unsplit_join() {
    const LIMIT: usize = 1 << 30;
    let (probe, build) = split_inputs();
    for t in [
        JoinType::Inner,
        JoinType::LeftOuter,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::Existence,
        JoinType::RightOuter,
        JoinType::RightSemi,
        JoinType::RightAnti,
        JoinType::FullOuter,
        JoinType::NullAwareAnti,
    ] {
        for filter in [false, true] {
            if t == JoinType::NullAwareAnti && filter {
                continue;
            }
            let c = case(t, 1, false, filter);
            let (mut whole, _) = run_case(&c, &probe, &build, 16, 2, LIMIT).await;
            whole.sort();
            let sides = skew_split_sides(t, false);
            if sides.probe {
                let mut got = Vec::new();
                for part in probe.chunks(probe.len().div_ceil(3)) {
                    got.extend(run_case(&c, part, &build, 16, 2, LIMIT).await.0);
                }
                got.sort();
                assert_eq!(got, whole, "{t:?} filter={filter}, probe split");
            }
            if sides.build {
                let mut got = Vec::new();
                for part in build.chunks(build.len().div_ceil(3)) {
                    got.extend(run_case(&c, &probe, part, 16, 2, LIMIT).await.0);
                }
                got.sort();
                assert_eq!(got, whole, "{t:?} filter={filter}, build split");
            }
        }
    }
}

/// What the table forbids is forbidden for a reason: splitting either side
/// of a full outer join pads rows that another split matched.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn splitting_a_full_outer_join_changes_its_result() {
    const LIMIT: usize = 1 << 30;
    let (probe, build) = split_inputs();
    let c = case(JoinType::FullOuter, 1, false, false);
    let (mut whole, _) = run_case(&c, &probe, &build, 16, 2, LIMIT).await;
    whole.sort();
    let mut got = Vec::new();
    for part in probe.chunks(probe.len().div_ceil(3)) {
        got.extend(run_case(&c, part, &build, 16, 2, LIMIT).await.0);
    }
    got.sort();
    assert_ne!(got, whole);
}
