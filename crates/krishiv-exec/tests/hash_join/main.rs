// SPDX-License-Identifier: Apache-2.0
//! Hash join (P2-T052a): every join type against a nested-loop oracle,
//! Spark's NULL and `NOT IN` semantics, weights, grace spill under tiny
//! memory budgets (results equal to the in-memory path, recursion, the
//! depth limit), deterministic replay, cancellation, and TPC-H join shapes
//! ([`tpch`]). Every test checks that the query's memory is released.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

mod tpch;
mod util;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_exec::{ExecConfig, JoinSpec, JoinType, Predicate, QueryRunner, Result, TokioExecutor};
use krishiv_memory::SpillScope;
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;
use util::*;

/// One test row: keys `k1`, `k2` and a value `v`.
#[derive(Clone, Debug)]
struct Row {
    k1: Option<i64>,
    k2: Option<&'static str>,
    v: i64,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, true),
        Field::new("k2", DataType::Utf8, true),
        Field::new("v", DataType::Int64, false),
    ]))
}

fn data(rows: &[Row]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.k1).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.k2).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.v).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// `probe.v < build.v`, NULL when `probe.v + build.v` is a multiple of 7.
#[derive(Debug)]
struct VLess;

impl VLess {
    fn eval(p: i64, b: i64) -> bool {
        (p + b) % 7 != 0 && p < b
    }
}

impl Predicate for VLess {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        let p = batch.column(2).as_primitive::<Int64Type>();
        let b = batch.column(5).as_primitive::<Int64Type>();
        Ok(p.iter()
            .zip(b.iter())
            .map(|(p, b)| {
                let (p, b) = (p.unwrap_or(0), b.unwrap_or(0));
                ((p + b) % 7 != 0).then_some(p < b)
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
struct Case {
    join_type: JoinType,
    keys: usize,
    null_safe: Vec<bool>,
    filter: bool,
}

impl Case {
    fn spec(&self, spill: Option<SpillScope>) -> JoinSpec {
        JoinSpec {
            join_type: self.join_type,
            probe_schema: schema(),
            build_schema: schema(),
            probe_keys: (0..self.keys).collect(),
            build_keys: (0..self.keys).collect(),
            null_safe: self.null_safe.clone(),
            filter: self.filter.then(|| Arc::new(VLess) as Arc<dyn Predicate>),
            spill,
        }
    }

    fn key_eq(&self, p: &Row, b: &Row) -> bool {
        fn eq<T: PartialEq>(safe: bool, x: &Option<T>, y: &Option<T>) -> bool {
            if safe { x == y } else { x.is_some() && x == y }
        }
        (0..self.keys).all(|k| match k {
            0 => eq(self.null_safe[0], &p.k1, &b.k1),
            _ => eq(self.null_safe[1], &p.k2, &b.k2),
        })
    }

    /// The nested-loop reference join, rendered like [`render`].
    fn oracle(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        let fmt = |r: &Row| {
            let k1 = r.k1.map_or("∅".to_owned(), |v| v.to_string());
            format!("{k1}|{}|{}", r.k2.unwrap_or("∅"), r.v)
        };
        let mut out = Vec::new();
        for p in probe {
            let matches: Vec<&Row> = build
                .iter()
                .filter(|b| self.key_eq(p, b) && (!self.filter || VLess::eval(p.v, b.v)))
                .collect();
            match self.join_type {
                JoinType::Inner | JoinType::LeftOuter => {
                    out.extend(matches.iter().map(|b| format!("{}|{}", fmt(p), fmt(b))));
                    if matches.is_empty() && self.join_type == JoinType::LeftOuter {
                        out.push(format!("{}|∅|∅|∅", fmt(p)));
                    }
                }
                JoinType::LeftSemi if !matches.is_empty() => out.push(fmt(p)),
                JoinType::LeftAnti if matches.is_empty() => out.push(fmt(p)),
                JoinType::NullAwareAnti => {
                    let keep = build.is_empty()
                        || (!build.iter().any(|b| b.k1.is_none())
                            && p.k1.is_some()
                            && matches.is_empty());
                    if keep {
                        out.push(fmt(p));
                    }
                }
                _ => {}
            }
        }
        out.sort();
        out
    }
}

/// Runs `case` over `probe` ⋈ `build` with `batch_rows`-row input batches.
async fn run_case(
    case: &Case,
    probe: &[Row],
    build: &[Row],
    batch_rows: usize,
    threads: usize,
    limit: usize,
) -> (Vec<String>, Arc<krishiv_exec::QueryContext>) {
    let root = root(limit);
    let (_manager, scope) = spill();
    // Tiny output batches exercise the output bound, on small inputs.
    let config = ExecConfig {
        batch_size: if probe.len() < 1000 { 5 } else { 1024 },
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), batch_rows)),
        scan(&schema(), batches(&data(probe), batch_rows)),
    );
    let out = run(&runner(threads), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    (render(&out), q)
}

fn r(k1: Option<i64>, k2: Option<&'static str>, v: i64) -> Row {
    Row { k1, k2, v }
}

fn case(join_type: JoinType, keys: usize, null_safe: bool, filter: bool) -> Case {
    Case {
        join_type,
        keys,
        null_safe: vec![null_safe; keys],
        filter,
    }
}

const TYPES: [JoinType; 5] = [
    JoinType::Inner,
    JoinType::LeftOuter,
    JoinType::LeftSemi,
    JoinType::LeftAnti,
    JoinType::NullAwareAnti,
];

/// NULL keys, duplicates on both sides, empty and all-NULL inputs, for
/// every join type, with `=` and `<=>` keys and with the residual filter.
#[tokio::test]
async fn join_types_match_the_oracle_on_edge_cases() {
    let probe = vec![
        r(Some(1), Some("a"), 1),
        r(Some(1), Some("a"), 9),
        r(Some(2), None, 3),
        r(None, Some("a"), 4),
        r(None, None, 5),
        r(Some(7), Some("x"), 6),
    ];
    let build = vec![
        r(Some(1), Some("a"), 8),
        r(Some(1), Some("a"), 2),
        r(Some(1), Some("b"), 3),
        r(Some(2), None, 12),
        r(None, None, 10),
        r(None, Some("a"), 11),
    ];
    let all_null = vec![r(None, None, 1), r(None, None, 2)];
    let inputs: [(&[Row], &[Row]); 5] = [
        (&probe, &build),
        (&probe, &[]),
        (&[], &build),
        (&probe, &all_null),
        (&all_null, &build),
    ];
    for jt in TYPES {
        for keys in [1, 2] {
            for safe in [false, true] {
                for filter in [false, true] {
                    if jt == JoinType::NullAwareAnti && (keys > 1 || safe || filter) {
                        continue;
                    }
                    let c = case(jt, keys, safe, filter);
                    for (p, b) in inputs {
                        let (got, _) = run_case(&c, p, b, 2, 2, 64 * MIB).await;
                        assert_eq!(
                            got,
                            c.oracle(p, b),
                            "{c:?} probe {} build {}",
                            p.len(),
                            b.len()
                        );
                    }
                }
            }
        }
    }
}

/// Spark's `NOT IN`: an empty subquery keeps every row (NULL too); a NULL in
/// the subquery keeps none; otherwise a NULL probe value is dropped.
#[tokio::test]
async fn not_in_follows_spark_null_semantics() {
    let c = case(JoinType::NullAwareAnti, 1, false, false);
    let probe = [r(Some(1), None, 0), r(Some(2), None, 0), r(None, None, 0)];
    let (got, _) = run_case(&c, &probe, &[], 1, 1, 64 * MIB).await;
    assert_eq!(got, ["1|∅|0", "2|∅|0", "∅|∅|0"]);
    let (got, _) = run_case(
        &c,
        &probe,
        &[r(Some(1), None, 0), r(None, None, 0)],
        1,
        1,
        64 * MIB,
    )
    .await;
    assert!(got.is_empty());
    let (got, _) = run_case(&c, &probe, &[r(Some(1), None, 0)], 1, 1, 64 * MIB).await;
    assert_eq!(got, ["2|∅|0"]);
}

const LONG_A: &str = "a-key-long-enough-to-fill-memory-quickly-0123456789";
const LONG_B: &str = "b-key-long-enough-to-fill-memory-quickly-0123456789";

/// A row with keys in `0..keys` and NULL keys one time in `nulls + 1`.
fn row(keys: i64, nulls: u32) -> impl Strategy<Value = Row> {
    let k1 = prop_oneof![1 => Just(None), nulls => (0i64..keys).prop_map(Some)];
    // Long strings, so the spilling variant needs few rows to fill memory.
    let k2 = prop_oneof![1 => Just(None), 4 => prop::sample::select(vec![LONG_A, LONG_B, ""]).prop_map(Some)];
    (k1, k2, 0i64..20).prop_map(|(k1, k2, v)| Row { k1, k2, v })
}

fn any_case() -> impl Strategy<Value = Case> {
    (
        0usize..5,
        1usize..3,
        prop::collection::vec(any::<bool>(), 2),
        any::<bool>(),
    )
        .prop_map(|(t, keys, safe, filter)| {
            let join_type = TYPES[t];
            let naaj = join_type == JoinType::NullAwareAnti;
            Case {
                join_type,
                keys: if naaj { 1 } else { keys },
                null_safe: if naaj {
                    vec![false]
                } else {
                    safe[..keys].to_vec()
                },
                filter: filter && !naaj,
            }
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]
    /// Random data with NULLs, one or two key columns, `=` / `<=>` keys and
    /// the residual filter: always the oracle's answer.
    #[test]
    fn random_joins_match_the_nested_loop_oracle(
        c in any_case(),
        probe in prop::collection::vec(row(6, 4), 0..80),
        build in prop::collection::vec(row(6, 4), 0..80),
        batch_rows in 1usize..17,
        threads in 1usize..4,
    ) {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (got, _) = rt.block_on(run_case(&c, &probe, &build, batch_rows, threads, 64 * MIB));
        prop_assert_eq!(got, c.oracle(&probe, &build));
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, max_shrink_iters: 16, ..ProptestConfig::default() })]
    /// Larger random inputs under a budget that forces grace spill
    /// (asserted): the same answer as the in-memory path.
    #[test]
    fn random_spilling_joins_equal_the_in_memory_path(
        c in any_case(),
        probe in prop::collection::vec(row(20_000, 200), 10_000..15_000),
        build in prop::collection::vec(row(20_000, 200), 15_000..20_000),
        batch_rows in 64usize..512,
        threads in 1usize..4,
    ) {
        // Two keys (one long string) make the build several times the
        // budget; null-aware anti (one key) is covered by the grace test.
        let mut c = c;
        if c.join_type == JoinType::NullAwareAnti {
            c.join_type = JoinType::LeftAnti;
        }
        c.keys = 2;
        c.null_safe.resize(2, false);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (expect, _) = rt.block_on(run_case(&c, &probe, &build, batch_rows, threads, 256 * MIB));
        let (got, q) = rt.block_on(run_case(&c, &probe, &build, batch_rows, threads, MIB));
        prop_assert!(got == expect, "grace result differs");
        prop_assert!(metric(&q, "HashJoinBuild", "spilled_partitions") > 0);
    }
}

/// Inner joins multiply weights; outer, semi and anti joins carry the
/// probe weight; a weighted build side is rejected outside inner joins.
#[tokio::test]
async fn weights_multiply_in_inner_joins_and_pass_through_otherwise() {
    let weighted = |rows: &[Row], w: Vec<i64>| {
        vec![Batch::with_weights(data(rows), Weights::Column(Int64Array::from(w))).unwrap()]
    };
    let probe = [r(Some(1), None, 1), r(Some(2), None, 2)];
    let build = [r(Some(1), None, 3), r(Some(1), None, 4)];
    for (jt, expect) in [
        (JoinType::Inner, vec!["1|∅|1|1|∅|3|*-6", "1|∅|1|1|∅|4|*-6"]),
        (
            JoinType::LeftOuter,
            vec!["1|∅|1|1|∅|3|*-2", "1|∅|1|1|∅|4|*-2", "2|∅|2|∅|∅|∅|*5"],
        ),
        (JoinType::LeftSemi, vec!["1|∅|1|*-2"]),
        (JoinType::LeftAnti, vec!["2|∅|2|*5"]),
    ] {
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let build_w = if jt == JoinType::Inner {
            vec![3, 3]
        } else {
            vec![1, 1]
        };
        let build_batches = if jt == JoinType::Inner {
            weighted(&build, build_w)
        } else {
            vec![Batch::new(data(&build))]
        };
        let plan = join(
            case(jt, 1, false, false).spec(None),
            scan(&schema(), build_batches),
            scan(&schema(), weighted(&probe, vec![-2, 5])),
        );
        let out = run(&runner(2), plan, &q).await.unwrap();
        assert_eq!(render(&out), expect, "{jt:?}");
        assert_released(&root);
    }
    let root = root(64 * MIB);
    let q = query(&root, 2, ExecConfig::default());
    let plan = join(
        case(JoinType::LeftSemi, 1, false, false).spec(None),
        scan(
            &schema(),
            vec![Batch::with_weights(data(&build), Weights::Const(2)).unwrap()],
        ),
        scan(&schema(), vec![Batch::new(data(&probe))]),
    );
    let e = run(&runner(1), plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert_released(&root);
}

fn skewed(rows: usize, keys: i64) -> Vec<Row> {
    (0..rows)
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

/// A tiny budget forces build partitions, probe rows and (one level down)
/// repartitioned partitions to disk; the answer equals the in-memory one
/// and `EXPLAIN ANALYZE` shows the spill.
#[tokio::test]
async fn grace_spill_with_recursion_equals_the_in_memory_join() {
    let build = skewed(30_000, 20_000);
    let probe = skewed(40_000, 25_000);
    for jt in TYPES {
        let keys = if jt == JoinType::NullAwareAnti { 1 } else { 2 };
        let c = case(jt, keys, false, jt == JoinType::LeftOuter);
        let (expect, mem) = run_case(&c, &probe, &build, 1024, 3, 256 * MIB).await;
        assert_eq!(metric(&mem, "HashJoinBuild", "spilled_partitions"), 0);
        let (got, q) = run_case(&c, &probe, &build, 1024, 3, 1536 << 10).await;
        assert_eq!(got.len(), expect.len(), "{jt:?}");
        assert!(got == expect, "{jt:?}: grace result differs");
        assert!(
            metric(&q, "HashJoinBuild", "spilled_partitions") > 0,
            "{jt:?}"
        );
        assert!(metric(&q, "HashJoinBuild", "spill_file_bytes") > 0);
        assert!(metric(&q, "HashJoinProbe", "probe_spilled_rows") > 0);
        assert!(metric(&q, "HashJoinProbe", "grace_partitions") > 0);
        assert_eq!(metric(&q, "HashJoinBuild", "build_rows"), 30_000);
        assert_eq!(metric(&q, "HashJoinProbe", "probe_rows"), 40_000);
    }
}

/// Keys that cannot be split: a partition still too large at the depth
/// limit fails with a typed error, releasing memory and files.
#[tokio::test]
async fn a_partition_skewed_on_one_key_fails_at_the_depth_limit() {
    let build: Vec<Row> = (0..60_000).map(|i| r(Some(7), Some("a"), i)).collect();
    let probe = vec![r(Some(7), Some("a"), 1); 10];
    let root = root(1 << 20);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let plan = join(
        case(JoinType::Inner, 1, false, false).spec(Some(scope.clone())),
        scan(&schema(), batches(&data(&build), 2048)),
        scan(&schema(), batches(&data(&probe), 2048)),
    );
    let e = run(&runner(2), plan, &q).await.unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivJoinPartitionTooLarge,
        "{e}"
    );
    assert_eq!(metric(&q, "HashJoinProbe", "repartitions"), 6);
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0);
}

/// Deterministic mode: the same seed replays the same batches (order
/// included) and the same spill metrics.
#[tokio::test(start_paused = true)]
async fn deterministic_mode_replays_spilling_joins() {
    let build = skewed(20_000, 8_000);
    let probe = skewed(20_000, 9_000);
    let once = || async {
        let runner = QueryRunner::new(TokioExecutor::deterministic(3).unwrap());
        let root = root(2 * MIB);
        let (_manager, scope) = spill();
        let config = ExecConfig {
            seed: Some(42),
            ..ExecConfig::default()
        };
        let q = query(&root, 1, config);
        let plan = join(
            case(JoinType::LeftOuter, 1, true, true).spec(Some(scope)),
            scan(&schema(), batches(&data(&build), 500)),
            scan(&schema(), batches(&data(&probe), 500)),
        );
        let out = run(&runner, plan, &q).await.unwrap();
        assert_released(&root);
        let seq: Vec<String> = out
            .iter()
            .flat_map(|b| render(std::slice::from_ref(b)))
            .collect();
        let spilled = metric(&q, "HashJoinBuild", "spilled_partitions");
        (
            seq,
            spilled,
            metric(&q, "HashJoinProbe", "probe_spilled_rows"),
        )
    };
    let a = once().await;
    assert!(a.1 > 0, "the run spills");
    assert_eq!(a, once().await);
}

/// Cancelling while the grace phase runs releases memory and files.
#[tokio::test]
async fn cancelling_a_spilling_join_releases_everything() {
    let build = skewed(30_000, 20_000);
    let probe = skewed(40_000, 25_000);
    let root = root(1536 << 10);
    let (_manager, scope) = spill();
    let q = query(&root, 9, ExecConfig::default());
    let plan = join(
        case(JoinType::Inner, 1, false, false).spec(Some(scope.clone())),
        scan(&schema(), batches(&data(&build), 1024)),
        scan(&schema(), batches(&data(&probe), 1024)),
    );
    let runner = runner(2);
    let mut stream = krishiv_exec::Scheduler::submit(&runner, plan, Arc::clone(&q))
        .await
        .unwrap();
    let first = futures::StreamExt::next(&mut stream).await;
    assert!(first.is_some_and(|b| b.is_ok()));
    q.cancel();
    drop(stream);
    q.finished().await;
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0);
}

/// Without a spill scope, a join that does not fit fails through the
/// memory arbitrator instead of spilling.
#[tokio::test]
async fn without_a_spill_scope_memory_pressure_is_an_error() {
    let build = skewed(30_000, 20_000);
    let root = root(256 << 10);
    let q = query(&root, 1, ExecConfig::default());
    let plan = join(
        case(JoinType::Inner, 1, false, false).spec(None),
        scan(&schema(), batches(&data(&build), 1024)),
        scan(&schema(), batches(&data(&build), 1024)),
    );
    let e = run(&runner(1), plan, &q).await.unwrap_err();
    assert_ne!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert_released(&root);
}

/// A `LIMIT` above a spilling join stops its flush; a `LIMIT` below it
/// (on the probe side) still lets the spilled partitions be joined.
#[tokio::test]
async fn limits_above_and_below_a_spilling_join() {
    use krishiv_exec::testing::{Limit, Pipe};
    let unique: Vec<Row> = (0..40_000).map(|i| r(Some(i), Some("k"), i % 50)).collect();
    for below in [false, true] {
        let root = root(1536 << 10);
        let (_manager, scope) = spill();
        let q = query(&root, 1, ExecConfig::default());
        let mut probe = scan(&schema(), batches(&data(&unique), 1024));
        if below {
            probe = Arc::new(Pipe(probe, Limit::new(5_000)));
        }
        let mut plan = join(
            case(JoinType::Inner, 1, false, false).spec(Some(scope.clone())),
            scan(&schema(), batches(&data(&unique), 1024)),
            probe,
        );
        if !below {
            plan = Arc::new(Pipe(plan, Limit::new(5_000)));
        }
        let out = run(&runner(3), plan, &q).await.unwrap();
        assert_eq!(
            out.iter().map(Batch::num_rows).sum::<usize>(),
            5_000,
            "below: {below}"
        );
        assert!(metric(&q, "HashJoinBuild", "spilled_partitions") > 0);
        assert_released(&root);
        assert_eq!(spill_files(&scope), 0);
    }
}

/// Probe rows whose build partition is empty are never deferred, spilled
/// build or not: they match nothing and stream.
#[tokio::test]
async fn empty_build_partitions_do_not_defer_probe_rows() {
    let probe = skewed(20_000, 20_000);
    for build in [vec![], vec![r(Some(1), Some("a"), 1)]] {
        let c = case(JoinType::LeftAnti, 1, false, false);
        let (got, q) = run_case(&c, &probe, &build, 1024, 2, 64 * MIB).await;
        assert_eq!(got, c.oracle(&probe, &build));
        assert_eq!(metric(&q, "HashJoinProbe", "probe_spilled_rows"), 0);
    }
}
