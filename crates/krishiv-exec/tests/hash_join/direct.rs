// SPDX-License-Identifier: Apache-2.0
//! The direct-index probe of tiny builds (perf lever J1,
//! `krishiv.exec.join.directIndex`): every join type returns the oracle's
//! rows with it on and off, over negative, sparse, duplicate and NULL keys
//! and probe keys outside the build's span, with `Int64` and `Int32` keys
//! (the TPC-DS surrogate keys), sliced key arrays with NULLs and a probe
//! filter fused into the join; it runs only where it applies, and a build
//! past the payload cap (perf lever P3) decodes its columns per pair.

use arrow_array::Int32Array;
use krishiv_exec::{FnPredicate, HashJoin, HashJoinNode};

use super::*;

/// How [`run_direct`] runs a case.
#[derive(Clone, Copy, Debug, Default)]
struct Run {
    /// The direct index on.
    on: bool,
    /// `k1` as `Int32` on both sides.
    int32: bool,
    /// [`keep`] fused into the probe (a selection of the probe batch).
    fused: bool,
}

/// `k1` as `Int32`, the other columns as [`schema`].
fn schema32() -> SchemaRef {
    let mut fields: Vec<Field> = schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields[0] = Field::new("k1", DataType::Int32, true);
    Arc::new(Schema::new(fields))
}

/// [`data`] with `k1` as `Int32` (every key must fit).
fn data32(rows: &[Row]) -> RecordBatch {
    let wide = data(rows);
    let mut cols = wide.columns().to_vec();
    cols[0] = Arc::new(Int32Array::from(
        rows.iter()
            .map(|r| r.k1.map(|k| i32::try_from(k).unwrap()))
            .collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema32(), cols).unwrap()
}

/// The fused probe filter: `v % 3 != 0`.
fn keep(b: &RecordBatch) -> Result<BooleanArray> {
    let v = b.column(2).as_primitive::<Int64Type>();
    Ok(v.iter().map(|v| v.map(|v| v % 3 != 0)).collect())
}

/// The oracle of `case` for `run`: over the probe rows [`keep`] keeps when
/// the filter is fused.
fn want(case: &Case, probe: &[Row], build: &[Row], run: Run) -> Vec<String> {
    if !run.fused {
        return case.oracle(probe, build);
    }
    let kept: Vec<Row> = probe.iter().filter(|r| r.v % 3 != 0).cloned().collect();
    case.oracle(&kept, build)
}

/// Runs `case` (one `k1` key) as `run` says; also returns the probe's
/// `direct_probe_rows` (absent, so 0, when the index did not run).
async fn run_direct(case: &Case, probe: &[Row], build: &[Row], run_as: Run) -> (Vec<String>, u64) {
    let root = root(64 * MIB);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: 5,
        join_row_layout: case.row_layout,
        join_direct_index: run_as.on,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let (schema, data): (SchemaRef, fn(&[Row]) -> RecordBatch) = if run_as.int32 {
        (schema32(), data32)
    } else {
        (schema(), data)
    };
    let mut spec = case.spec(Some(scope.clone()));
    spec.probe_schema = Arc::clone(&schema);
    spec.build_schema = Arc::clone(&schema);
    let mut hash_join = HashJoin::try_new(spec).unwrap();
    if run_as.fused {
        hash_join = hash_join
            .with_probe_filter(Arc::new(FnPredicate(keep)))
            .unwrap();
    }
    let plan = Arc::new(HashJoinNode {
        join: hash_join,
        build: scan(&schema, batches(&data(build), 3)),
        probe: scan(&schema, batches(&data(probe), 4)),
    });
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
            let (on, direct) = run_direct(
                &c,
                &probe,
                &build,
                Run {
                    on: true,
                    ..Run::default()
                },
            )
            .await;
            let (off, none) = run_direct(&c, &probe, &build, Run::default()).await;
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
        let (got, direct) = run_direct(
            c,
            &probe,
            build,
            Run {
                on: true,
                ..Run::default()
            },
        )
        .await;
        assert_eq!(got, c.oracle(&probe, build), "{c:?}");
        assert_eq!(direct, 0, "{c:?}");
    }
}

/// `Int32` keys (the TPC-DS surrogate keys) through the same oracle:
/// negative and duplicate build keys, NULLs on both sides, probe keys at
/// `i32::MIN` and `i32::MAX`; the key arrays are slices (batches cut at
/// offsets) with null buffers.
#[tokio::test]
async fn int32_keys_match_the_oracle_with_the_direct_index() {
    let build = keyed(&[
        Some(-7),
        Some(0),
        Some(3),
        None,
        Some(3),
        Some(9),
        Some(-7),
        Some(5),
        None,
        Some(3),
    ]);
    let probe = keyed(&[
        Some(3),
        Some(i64::from(i32::MIN)),
        Some(-7),
        None,
        Some(i64::from(i32::MAX)),
        Some(4),
        Some(-8),
        Some(10),
        Some(9),
        None,
        Some(0),
        Some(5),
        Some(-7),
        Some(3),
    ]);
    for join_type in TYPES {
        for filter in [false, true] {
            let filter = filter && join_type != JoinType::NullAwareAnti;
            let mut c = case(join_type, 1, false, filter);
            c.row_layout = true;
            let run_as = Run {
                on: true,
                int32: true,
                fused: false,
            };
            let (on, direct) = run_direct(&c, &probe, &build, run_as).await;
            let (off, _) = run_direct(
                &c,
                &probe,
                &build,
                Run {
                    on: false,
                    ..run_as
                },
            )
            .await;
            let want = want(&c, &probe, &build, run_as);
            assert_eq!(on, want, "{join_type:?} filter {filter}");
            assert_eq!(off, want, "{join_type:?} filter {filter}");
            if join_type == JoinType::NullAwareAnti {
                assert_eq!(direct, 0, "{join_type:?}");
            } else {
                assert_eq!(direct, probe.len() as u64, "{join_type:?}");
            }
        }
    }
}

/// A probe filter fused into the join hands the probe a selection of each
/// batch: the direct index reads the selected rows' keys, `Int64` and
/// `Int32`, and joins what the join of the filtered probe joins.
#[tokio::test]
async fn a_fused_probe_selection_matches_the_oracle_with_the_direct_index() {
    let (probe, build) = sides();
    for int32 in [false, true] {
        // `i64::MIN`/`MAX` probe keys do not fit an `Int32` column.
        let probe: Vec<Row> = if int32 {
            probe
                .iter()
                .filter(|r| r.k1.is_none_or(|k| i32::try_from(k).is_ok()))
                .cloned()
                .collect()
        } else {
            probe.clone()
        };
        for join_type in [
            JoinType::Inner,
            JoinType::LeftOuter,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
        ] {
            let mut c = case(join_type, 1, false, false);
            c.row_layout = true;
            let run_as = Run {
                on: true,
                int32,
                fused: true,
            };
            let (on, direct) = run_direct(&c, &probe, &build, run_as).await;
            let (off, _) = run_direct(
                &c,
                &probe,
                &build,
                Run {
                    on: false,
                    ..run_as
                },
            )
            .await;
            let want = want(&c, &probe, &build, run_as);
            assert_eq!(on, want, "{join_type:?} int32 {int32}");
            assert_eq!(off, want, "{join_type:?} int32 {int32}");
            assert!(direct > 0, "{join_type:?} int32 {int32}: the index ran");
        }
    }
}

/// A long payload string: 3,000 build rows of it hold more than the
/// 256 KiB a direct build may hold for its payload to be decoded once.
const LONG: &str = "a payload string long enough that three thousand rows of it are \
                    well past the decoded payload cap of a direct build index";

/// A direct build too large to decode its payload once (perf lever P3)
/// keeps the direct index and decodes its build columns per pair; a small
/// one takes them from the payload decoded once. Both match the oracle.
#[tokio::test]
async fn the_payload_cap_keeps_the_direct_index_either_way() {
    let big: Vec<Row> = (0..3_000_i64)
        .map(|k| r(Some(k), Some(LONG), k % 7))
        .collect();
    let small: Vec<Row> = (0..40_i64).map(|k| r(Some(k), Some(LONG), k)).collect();
    let probe = keyed(
        &(0..60)
            .map(|i| Some((i * 53) % 3_100 - 20))
            .chain([None])
            .collect::<Vec<_>>(),
    );
    for join_type in [JoinType::Inner, JoinType::LeftOuter] {
        let mut c = case(join_type, 1, false, false);
        c.row_layout = true;
        for build in [&big, &small] {
            let (got, direct) = run_direct(
                &c,
                &probe,
                build,
                Run {
                    on: true,
                    ..Run::default()
                },
            )
            .await;
            assert_eq!(
                got,
                c.oracle(&probe, build),
                "{join_type:?} {}",
                build.len()
            );
            assert_eq!(direct, probe.len() as u64, "{join_type:?} {}", build.len());
        }
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
        let (got, _) = rt.block_on(run_direct(&c, &probe, &build, Run { on: true, ..Run::default() }));
        prop_assert_eq!(got, c.oracle(&probe, &build));
    }
}
