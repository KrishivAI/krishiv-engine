// SPDX-License-Identifier: Apache-2.0
//! The direct-index probe of tiny builds (perf lever J1,
//! `krishiv.exec.join.directIndex`): every join type returns the oracle's
//! rows with it on and off, over negative, sparse, duplicate and NULL keys
//! and probe keys outside the build's span, with `Int64` and `Int32` keys
//! (the TPC-DS surrogate keys), sliced key arrays with NULLs and a probe
//! filter fused into the join; it runs only where it applies, and a build
//! past the payload cap (perf lever P3) decodes its columns per pair while
//! a smaller one decodes them once (`direct_payload_rows`), `Utf8` and
//! `Utf8View` payloads, with a residual filter and the build-preserving
//! join types.

use arrow_array::{Int32Array, StringViewArray};
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
    /// `k2` as `Utf8View` on both sides (not with `int32`).
    view: bool,
}

/// `k2` as `Utf8View`, the other columns as [`schema`].
fn schema_view() -> SchemaRef {
    let mut fields: Vec<Field> = schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields[1] = Field::new("k2", DataType::Utf8View, true);
    Arc::new(Schema::new(fields))
}

/// [`data`] with `k2` as `Utf8View`.
fn data_view(rows: &[Row]) -> RecordBatch {
    let wide = data(rows);
    let mut cols = wide.columns().to_vec();
    cols[1] = Arc::new(rows.iter().map(|r| r.k2).collect::<StringViewArray>());
    RecordBatch::try_new(schema_view(), cols).unwrap()
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
    let (rows, direct, _, _, _) = run_counted(case, probe, build, run_as).await;
    (rows, direct)
}

/// [`run_direct`] with the build rows whose payload was decoded once
/// (`direct_payload_rows`) and the rows the one-pass inner probe emitted
/// (`direct_inner_rows`, perf lever C), and the probe rows output as their
/// own batch (`probe_passthrough_rows`).
async fn run_counted(
    case: &Case,
    probe: &[Row],
    build: &[Row],
    run_as: Run,
) -> (Vec<String>, u64, u64, u64, u64) {
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
    } else if run_as.view {
        (schema_view(), data_view)
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
        metric(&q, "HashJoinProbe", "direct_payload_rows"),
        metric(&q, "HashJoinProbe", "direct_inner_rows"),
        metric(&q, "HashJoinProbe", "probe_passthrough_rows"),
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
                view: false,
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
                view: false,
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
            // The index reads exactly the rows the fused filter keeps.
            let kept = probe.iter().filter(|r| r.v % 3 != 0).count() as u64;
            assert_eq!(direct, kept, "{join_type:?} int32 {int32}: the index ran");
        }
    }
}

/// A long payload string: 3,000 build rows of it hold more than the
/// 256 KiB a direct build may hold for its payload to be decoded once.
const LONG: &str = "a payload string long enough that three thousand rows of it are \
                    well past the decoded payload cap of a direct build index";

/// A direct build too large to decode its payload once (perf lever P3)
/// keeps the direct index and decodes its build columns per pair; a small
/// one takes them from the payload decoded once (`direct_payload_rows`
/// counts its rows). Both match the oracle, with `Utf8` and `Utf8View`
/// payloads, a residual filter and the build-preserving join types.
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
    let cases = [
        (JoinType::Inner, false),
        (JoinType::LeftOuter, false),
        (JoinType::Inner, true),
        (JoinType::RightOuter, true),
        (JoinType::FullOuter, true),
    ];
    for view in [false, true] {
        for (join_type, filter) in cases {
            let mut c = case(join_type, 1, false, filter);
            c.row_layout = true;
            for build in [&big, &small] {
                let what = format!("{join_type:?} filter {filter} view {view} {}", build.len());
                let (got, direct, payload, inner, _) = run_counted(
                    &c,
                    &probe,
                    build,
                    Run {
                        on: true,
                        view,
                        ..Run::default()
                    },
                )
                .await;
                assert_eq!(got, c.oracle(&probe, build), "{what}");
                assert_eq!(direct, probe.len() as u64, "{what}");
                let want = if build.len() == small.len() {
                    small.len() as u64
                } else {
                    0
                };
                assert_eq!(payload, want, "{what}: rows decoded once");
                // Only the unique, decoded build of an inner join without a
                // filter takes the one-pass probe, for each output row.
                let one_pass = want > 0 && join_type == JoinType::Inner && !filter;
                let rows = got.len() as u64;
                assert_eq!(inner, if one_pass { rows } else { 0 }, "{what}: one pass");
                assert!(!one_pass || rows > 0, "{what}: the probe matches");
            }
        }
    }
}

/// The one-pass inner probe (perf lever C) runs only over unique build
/// keys: a decoded build with a duplicate key walks its chains, and both
/// match the oracle, with and without a fused probe selection, `Int32` keys
/// and `Utf8View` payloads.
#[tokio::test]
async fn the_one_pass_inner_probe_needs_unique_build_keys() {
    let unique: Vec<Row> = (-5..30_i64)
        .map(|k| r(Some(k), Some(LONG), k))
        .chain([r(None, Some("n"), 1)])
        .collect();
    let mut dup = unique.clone();
    dup.push(r(Some(7), Some("again"), 2));
    let probe = keyed(
        &(0..80)
            .map(|i| Some((i * 7) % 45 - 10))
            .chain([None, Some(1_000)])
            .collect::<Vec<_>>(),
    );
    let mut c = case(JoinType::Inner, 1, false, false);
    c.row_layout = true;
    for fused in [false, true] {
        for (int32, view) in [(false, false), (true, false), (false, true)] {
            for build in [&unique, &dup] {
                let run_as = Run {
                    on: true,
                    int32,
                    fused,
                    view,
                };
                let what = format!("{run_as:?} dup {}", build.len() > unique.len());
                let (got, direct, payload, inner, _) = run_counted(&c, &probe, build, run_as).await;
                assert_eq!(got, want(&c, &probe, build, run_as), "{what}");
                assert!(direct > 0, "{what}: direct");
                // An inner join keeps no NULL-key build row.
                let keyed = build.iter().filter(|r| r.k1.is_some()).count();
                assert_eq!(payload, keyed as u64, "{what}: decoded once");
                let rows = got.len() as u64;
                let expect = if build.len() == unique.len() { rows } else { 0 };
                assert!(rows > 0, "{what}: the probe matches");
                assert_eq!(inner, expect, "{what}: one pass");
            }
        }
    }
}

/// An inner join whose probe batches (4 rows each) match one build row per
/// row passes them through with no gather, on the one-pass direct probe and
/// on the general one; a batch with a probe row of two matches or none, or
/// under a fused selection, is gathered. Answers match the oracle either way.
#[tokio::test]
async fn whole_matched_probe_batches_pass_through() {
    let unique: Vec<Row> = (-5..30_i64).map(|k| r(Some(k), Some("b"), k)).collect();
    let mut dup = unique.clone();
    dup.push(r(Some(7), Some("again"), 2));
    // 12 rows, three batches; the third holds key 7.
    let probe = keyed(&[0, 1, 2, 3, 4, 5, 6, 8, 9, 7, 10, 11].map(Some));
    let mut missing = probe.clone();
    missing[1].k1 = Some(1_000);
    let mut c = case(JoinType::Inner, 1, false, false);
    c.row_layout = true;
    for on in [true, false] {
        for fused in [false, true] {
            for (build, probe, expect) in [
                (&unique, &probe, 12),
                (&dup, &probe, 8),
                (&unique, &missing, 8),
            ] {
                let run_as = Run {
                    on,
                    fused,
                    ..Run::default()
                };
                let what = format!("{run_as:?} build {} probe {:?}", build.len(), probe[1].k1);
                let (got, _, _, _, passed) = run_counted(&c, probe, build, run_as).await;
                assert_eq!(got, want(&c, probe, build, run_as), "{what}");
                let expect = if fused { 0 } else { expect };
                assert_eq!(passed, expect, "{what}: passed through");
            }
        }
    }
    // The counter belongs to the inner pairs' output: a residual filter (its
    // mask may drop pairs) and the semi and anti tails count nothing.
    for (join_type, filter) in [
        (JoinType::Inner, true),
        (JoinType::LeftSemi, false),
        (JoinType::LeftSemi, true),
        (JoinType::LeftAnti, false),
    ] {
        let mut c = case(join_type, 1, false, filter);
        c.row_layout = true;
        for on in [true, false] {
            let run_as = Run {
                on,
                ..Run::default()
            };
            let what = format!("{join_type:?} filter {filter} {run_as:?}");
            let (got, _, _, _, passed) = run_counted(&c, &probe, &unique, run_as).await;
            assert_eq!(got, want(&c, &probe, &unique, run_as), "{what}");
            assert_eq!(passed, 0, "{what}: passed through");
        }
    }
}

/// A weighted probe batch or build side keeps the general path (each pair
/// multiplies the weights there); unweighted ones take the one pass, whose
/// output rows weigh one.
#[tokio::test]
async fn weighted_inputs_keep_the_general_inner_probe() {
    let probe = [
        r(Some(1), None, 1),
        r(Some(2), None, 2),
        r(Some(9), None, 3),
    ];
    let build = [r(Some(1), None, 3), r(Some(2), None, 4)];
    let plain = |rows: &[Row]| vec![Batch::new(data(rows))];
    let weighted = |rows: &[Row], w: Weights| vec![Batch::with_weights(data(rows), w).unwrap()];
    let cases = [
        (
            plain(&probe),
            plain(&build),
            vec!["1|∅|1|1|∅|3", "2|∅|2|2|∅|4"],
            true,
        ),
        (
            weighted(&probe, Weights::Column(Int64Array::from(vec![-2, 5, 7]))),
            plain(&build),
            vec!["1|∅|1|1|∅|3|*-2", "2|∅|2|2|∅|4|*5"],
            false,
        ),
        (
            plain(&probe),
            weighted(&build, Weights::Const(3)),
            vec!["1|∅|1|1|∅|3|*3", "2|∅|2|2|∅|4|*3"],
            false,
        ),
    ];
    for (probe_side, build_side, expect, one_pass) in cases {
        let root = root(64 * MIB);
        let config = ExecConfig {
            join_row_layout: true,
            join_direct_index: true,
            ..ExecConfig::default()
        };
        let q = query(&root, 1, config);
        let mut c = case(JoinType::Inner, 1, false, false);
        c.row_layout = true;
        let plan = join(
            c.spec(None),
            scan(&schema(), build_side),
            scan(&schema(), probe_side),
        );
        let out = run(&runner(2), plan, &q).await.unwrap();
        assert_eq!(render(&out), expect, "one pass {one_pass}");
        assert!(metric(&q, "HashJoinProbe", "direct_probe_rows") > 0);
        let inner = metric(&q, "HashJoinProbe", "direct_inner_rows");
        assert_eq!(inner, if one_pass { 2 } else { 0 }, "one pass {one_pass}");
        assert_released(&root);
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]
    /// Random one-key joins over small key spans (so the index applies):
    /// the oracle's answer, every probe row through the index when the build
    /// qualifies (a non-NULL key, not a null-aware anti join) and none
    /// otherwise.
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
        let (got, direct) = rt.block_on(run_direct(&c, &probe, &build, Run { on: true, ..Run::default() }));
        prop_assert_eq!(got, c.oracle(&probe, &build));
        let qualifies = join_type != JoinType::NullAwareAnti && build.iter().any(|r| r.k1.is_some());
        prop_assert_eq!(direct, if qualifies { probe.len() as u64 } else { 0 });
    }
}
