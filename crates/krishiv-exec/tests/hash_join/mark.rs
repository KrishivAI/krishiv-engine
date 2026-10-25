// SPDX-License-Identifier: Apache-2.0
//! Mark (existence) joins and single-match left outer joins (P2-T115):
//! the full mark truth table with and without a residual condition, both
//! key formats, spill forced; single-match joins with zero, one and two
//! matches; weighted input rejected like the other join types.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_exec::{
    ExecConfig, HashJoin, JoinSpec, JoinType, NestedLoopJoin, NestedLoopSpec, Predicate, Result,
};
use krishiv_memory::SpillScope;
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;

use super::util::*;

/// One row: a join key `k` and a value `v` the residual reads.
#[derive(Clone, Copy, Debug)]
struct Row {
    k: Option<i64>,
    v: Option<i64>,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn data(rows: &[Row]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.k).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.v).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn r(k: Option<i64>, v: Option<i64>) -> Row {
    Row { k, v }
}

/// `probe.v < build.v`, NULL when either side is NULL or their sum is a
/// multiple of 7.
#[derive(Debug)]
struct VLess;

impl Predicate for VLess {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        let p = batch.column(1).as_primitive::<Int64Type>();
        let b = batch.column(3).as_primitive::<Int64Type>();
        Ok(p.iter()
            .zip(b.iter())
            .map(|(p, b)| match (p, b) {
                (Some(p), Some(b)) => ((p + b) % 7 != 0).then_some(p < b),
                _ => None,
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
struct Case {
    join_type: JoinType,
    null_safe: bool,
    filter: bool,
    single_match: bool,
    row_layout: bool,
}

impl Case {
    fn spec(&self, spill: Option<SpillScope>) -> JoinSpec {
        JoinSpec {
            join_type: self.join_type,
            probe_schema: schema(),
            build_schema: schema(),
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![self.null_safe],
            filter: self.filter.then(|| Arc::new(VLess) as Arc<dyn Predicate>),
            single_match: self.single_match,
            spill,
        }
    }

    fn key_eq(&self, x: &Option<i64>, y: &Option<i64>) -> bool {
        if self.null_safe {
            x == y
        } else {
            x.is_some() && x == y
        }
    }

    fn residual(&self, p: &Row, b: &Row) -> bool {
        !self.filter || matches!((p.v, b.v), (Some(p), Some(b)) if (p + b) % 7 != 0 && p < b)
    }

    /// The mark of `p` against `build`: `None` renders as NULL.
    fn mark(&self, p: &Row, build: &[Row]) -> Option<bool> {
        debug_assert!(matches!(
            self.join_type,
            JoinType::MarkExists | JoinType::MarkIn
        ));
        let matched = build
            .iter()
            .any(|b| self.key_eq(&p.k, &b.k) && self.residual(p, b));
        match self.join_type {
            JoinType::MarkExists => Some(matched),
            _ => {
                if matched {
                    Some(true)
                } else if build.is_empty() {
                    Some(false)
                } else if p.k.is_none() || build.iter().any(|b| b.k.is_none()) {
                    None
                } else {
                    Some(false)
                }
            }
        }
    }

    /// The nested-loop reference mark join, rendered like [`render`].
    fn oracle_mark(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        let cell = |x: &Option<i64>| x.map_or("∅".to_owned(), |v| v.to_string());
        let mut out: Vec<String> = probe
            .iter()
            .map(|p| {
                let m = self
                    .mark(p, build)
                    .map_or("∅".to_owned(), |v| v.to_string());
                format!("{}|{}|{m}", cell(&p.k), cell(&p.v))
            })
            .collect();
        out.sort();
        out
    }

    /// The nested-loop reference single-match join (at most one match per
    /// probe row, as the tests arrange), rendered like [`render`].
    fn oracle_single(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        debug_assert!(
            self.join_type == JoinType::LeftOuter && self.single_match,
            "{:?}",
            self.join_type
        );
        let cell = |x: &Option<i64>| x.map_or("∅".to_owned(), |v| v.to_string());
        let mut out: Vec<String> = probe
            .iter()
            .map(|p| {
                let hits: Vec<&Row> = build
                    .iter()
                    .filter(|b| self.key_eq(&p.k, &b.k) && self.residual(p, b))
                    .collect();
                assert!(hits.len() <= 1, "the oracle holds one match at most");
                match hits.as_slice() {
                    [b] => format!(
                        "{}|{}|{}|{}",
                        cell(&p.k),
                        cell(&p.v),
                        cell(&b.k),
                        cell(&b.v)
                    ),
                    _ => format!("{}|{}|∅|∅", cell(&p.k), cell(&p.v)),
                }
            })
            .collect();
        out.sort();
        out
    }

    /// The reference join of `case`: the mark oracle for mark joins, the
    /// single-match oracle for single-match joins.
    fn oracle(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        if self.single_match {
            self.oracle_single(probe, build)
        } else {
            self.oracle_mark(probe, build)
        }
    }
}

/// Runs `case` over `probe` ⋈ `build`.
async fn run_case(
    case: &Case,
    probe: &[Row],
    build: &[Row],
    batch_rows: usize,
    threads: usize,
    limit: usize,
) -> (Result<Vec<String>>, Arc<krishiv_exec::QueryContext>) {
    let root = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: if probe.len() < 1000 { 5 } else { 1024 },
        join_row_layout: case.row_layout,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), batch_rows)),
        scan(&schema(), batches(&data(probe), batch_rows)),
    );
    let out = run(&runner_for(threads, limit), plan, &q)
        .await
        .map(|b| render(&b));
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    (out, q)
}

/// Every combination of match/no match, NULL or non-NULL probe key, NULL
/// key in the build or not, and an empty build, with and without the
/// residual condition, in both key formats.
#[tokio::test]
async fn mark_joins_match_the_oracle_on_edge_cases() {
    let probe = vec![
        r(Some(1), Some(1)),
        r(Some(1), Some(90)),
        r(Some(2), Some(3)),
        r(Some(3), None),
        r(None, Some(4)),
        r(None, None),
    ];
    // A NULL value exercises a NULL residual on an otherwise matching pair
    // (`Some(3)` meets `None`); the sums that are multiples of 7 do too.
    let builds: &[&[Row]] = &[
        &[r(Some(1), Some(8)), r(Some(2), Some(4))],
        &[r(Some(1), Some(8)), r(None, Some(5))],
        &[r(Some(1), None)],
        &[r(Some(9), Some(9))],
        &[r(None, Some(5))],
        &[],
    ];
    for join_type in [JoinType::MarkExists, JoinType::MarkIn] {
        for safe in [false, true] {
            for filter in [false, true] {
                for row_layout in [false, true] {
                    let case = Case {
                        join_type,
                        null_safe: safe && join_type == JoinType::MarkExists,
                        filter,
                        single_match: false,
                        row_layout,
                    };
                    for build in builds {
                        let (got, _) = run_case(&case, &probe, build, 2, 2, 64 * MIB).await;
                        assert_eq!(
                            got.unwrap(),
                            case.oracle(&probe, build),
                            "{case:?} build {build:?}"
                        );
                    }
                }
            }
        }
    }
}

/// A NULL-safe `MarkExists` key matches NULL to NULL.
#[tokio::test]
async fn null_safe_mark_exists_matches_null_keys() {
    let case = Case {
        join_type: JoinType::MarkExists,
        null_safe: true,
        filter: false,
        single_match: false,
        row_layout: false,
    };
    let probe = [r(None, Some(1)), r(Some(1), Some(1))];
    let build = [r(None, Some(2))];
    let (got, _) = run_case(&case, &probe, &build, 1, 1, 64 * MIB).await;
    assert_eq!(got.unwrap(), case.oracle(&probe, &build));
}

/// The specs the executor refuses: a null-aware mark join over several
/// keys or null-safe keys, a mark join by nested loops, and single match
/// on a join that is not left outer.
#[test]
fn invalid_mark_and_single_match_specs_are_refused() {
    let schema = schema();
    let mark_in = |keys: usize, null_safe: bool, filter: bool| JoinSpec {
        join_type: JoinType::MarkIn,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0; keys],
        build_keys: vec![0; keys],
        null_safe: vec![null_safe; keys],
        filter: filter.then(|| Arc::new(VLess) as Arc<dyn Predicate>),
        single_match: false,
        spill: None,
    };
    let e = HashJoin::try_new(mark_in(2, false, false)).err().unwrap();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("null-aware mark join"), "{e}");
    let e = HashJoin::try_new(mark_in(1, true, false)).err().unwrap();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A residual condition is allowed on a null-aware mark join.
    assert!(HashJoin::try_new(mark_in(1, false, true)).is_ok());
    // `MarkExists` holds any number of keys, null-safe or not, with a
    // residual condition.
    let spec = JoinSpec {
        join_type: JoinType::MarkExists,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0, 0],
        build_keys: vec![0, 0],
        null_safe: vec![true, false],
        filter: Some(Arc::new(VLess) as Arc<dyn Predicate>),
        single_match: false,
        spill: None,
    };
    assert!(HashJoin::try_new(spec).is_ok());
    // No nested-loop path for mark joins.
    for join_type in [JoinType::MarkExists, JoinType::MarkIn] {
        let e = NestedLoopJoin::try_new(NestedLoopSpec {
            join_type,
            probe_schema: Arc::clone(&schema),
            build_schema: Arc::clone(&schema),
            filter: None,
        })
        .err()
        .unwrap();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
        assert!(e.to_string().contains("nested loops"), "{e}");
    }
    // Single match is a flag on the left outer join only.
    for join_type in [
        JoinType::Inner,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::NullAwareAnti,
        JoinType::MarkExists,
        JoinType::MarkIn,
    ] {
        let e = HashJoin::try_new(JoinSpec {
            join_type,
            probe_schema: Arc::clone(&schema),
            build_schema: Arc::clone(&schema),
            probe_keys: vec![0],
            build_keys: vec![0],
            null_safe: vec![false],
            filter: None,
            single_match: true,
            spill: None,
        })
        .err()
        .unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{join_type:?}");
    }
}

/// Zero or one match: a single-match join emits what a left outer join
/// does.
#[tokio::test]
async fn single_match_with_zero_or_one_match_is_a_left_outer_join() {
    for filter in [false, true] {
        for row_layout in [false, true] {
            let case = Case {
                join_type: JoinType::LeftOuter,
                null_safe: false,
                filter,
                single_match: true,
                row_layout,
            };
            let probe = [r(Some(1), Some(1)), r(Some(2), Some(3)), r(None, Some(4))];
            let build = [r(Some(1), Some(8))];
            let (got, _) = run_case(&case, &probe, &build, 2, 2, 64 * MIB).await;
            assert_eq!(got.unwrap(), case.oracle(&probe, &build), "{case:?}");
        }
    }
}

/// Two matches fail the query with `SCALAR_SUBQUERY_TOO_MANY_ROWS`
/// (SQLSTATE 21000), with and without a residual condition; one match
/// filtered down from two is fine.
#[tokio::test]
async fn single_match_with_two_matches_fails() {
    // In memory, without and with a residual condition.
    for filter in [false, true] {
        let case = Case {
            join_type: JoinType::LeftOuter,
            null_safe: false,
            filter,
            single_match: true,
            row_layout: false,
        };
        // `Some(1)` meets both build rows; with the filter both pairs hold
        // (`1 < 8`, `1 < 9`, and neither sum is a multiple of 7).
        let probe = [r(Some(1), Some(1))];
        let build = [r(Some(1), Some(8)), r(Some(1), Some(9))];
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = join(
            case.spec(None),
            scan(&schema(), vec![Batch::new(data(&build))]),
            scan(&schema(), vec![Batch::new(data(&probe))]),
        );
        let e = run(&runner(2), plan, &q).await.unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::ScalarSubqueryTooManyRows,
            "{e}"
        );
        assert_eq!(e.sql_state(), Some("21000"));
        assert!(
            e.to_string()
                .contains("More than one row returned by a subquery used as an expression"),
            "{e}"
        );
        assert_released(&root);
    }
    // Two key matches of which the residual keeps one: no error.
    let case = Case {
        join_type: JoinType::LeftOuter,
        null_safe: false,
        filter: true,
        single_match: true,
        row_layout: false,
    };
    // `1 < 0` fails, `1 < 8` holds.
    let probe = [r(Some(1), Some(1))];
    let build = [r(Some(1), Some(0)), r(Some(1), Some(8))];
    let (got, _) = run_case(&case, &probe, &build, 1, 1, 64 * MIB).await;
    assert_eq!(got.unwrap(), case.oracle(&probe, &build));
    // A NULL residual on the second pair is not a match either.
    let probe = [r(Some(1), Some(1))];
    let build = [r(Some(1), Some(8)), r(Some(1), None)];
    let (got, _) = run_case(&case, &probe, &build, 1, 1, 64 * MIB).await;
    assert_eq!(got.unwrap(), case.oracle(&probe, &build));
}

/// Two matches with spill forced fail the same way, releasing memory and
/// deleting spill files.
#[tokio::test(start_paused = true)]
async fn single_match_with_two_matches_fails_when_spilling() {
    let build: Vec<Row> = (0..20_000)
        .map(|i| r(Some(i % 9_000), Some(i % 50)))
        .chain([r(Some(7), Some(1)), r(Some(7), Some(2))])
        .collect();
    let probe = [r(Some(7), Some(0))];
    let case = Case {
        join_type: JoinType::LeftOuter,
        null_safe: false,
        filter: false,
        single_match: true,
        row_layout: false,
    };
    let root = root(1 << 20);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let plan = join(
        case.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(&build), 512)),
        scan(&schema(), batches(&data(&probe), 512)),
    );
    let e = run(&runner_for(2, 1 << 20), plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::ScalarSubqueryTooManyRows);
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0);
}

/// A tiny budget forces the mark join through grace spill (partitions,
/// deferred probe rows, repartitioning): the answer equals the in-memory
/// one, in the other key format.
#[tokio::test(start_paused = true)]
async fn spilling_mark_joins_equal_the_in_memory_path() {
    fn skewed(rows: usize, keys: i64, null_every: i64) -> Vec<Row> {
        (0..rows)
            .map(|i| {
                let i = i64::try_from(i).unwrap();
                let k = (i % null_every != 0).then_some(i % keys);
                r(k, Some(i % 50))
            })
            .collect()
    }
    let build = skewed(30_000, 20_000, 17);
    let mut probe = skewed(40_000, 25_000, 13);
    probe.extend((0..50).map(|i| r(None, Some(i))));
    for join_type in [JoinType::MarkExists, JoinType::MarkIn] {
        for filter in [false, true] {
            for row_layout in [false, true] {
                let case = Case {
                    join_type,
                    null_safe: false,
                    filter,
                    single_match: false,
                    row_layout,
                };
                let other = Case {
                    row_layout: !row_layout,
                    ..case.clone()
                };
                let (expect, _) = run_case(&other, &probe, &build, 1024, 3, 256 * MIB).await;
                let expect = expect.unwrap();
                let (got, q) = run_case(&case, &probe, &build, 1024, 3, 768 << 10).await;
                assert!(got.unwrap() == expect, "{case:?}: grace result differs");
                assert!(
                    metric(&q, "HashJoinBuild", "spilled_partitions") > 0,
                    "{case:?}"
                );
                assert!(metric(&q, "HashJoinProbe", "grace_partitions") > 0);
                // `build_rows` counts the stored rows: the NULL-key rows
                // are dropped from the table (but still count for the
                // mark's NULL semantics).
                let stored = build.iter().filter(|r| r.k.is_some()).count() as u64;
                assert_eq!(metric(&q, "HashJoinBuild", "build_rows"), stored);
                assert_eq!(metric(&q, "HashJoinProbe", "probe_rows"), 40_050);
            }
        }
    }
}

/// A weighted build side is rejected for mark and single-match joins, as
/// for the other non-inner join types; probe weights pass through.
#[tokio::test]
async fn weights_pass_through_or_are_rejected_like_the_other_joins() {
    let probe = [r(Some(1), Some(1)), r(Some(2), Some(2))];
    let build = [r(Some(1), Some(3))];
    for join_type in [JoinType::MarkExists, JoinType::MarkIn, JoinType::LeftOuter] {
        let single_match = join_type == JoinType::LeftOuter;
        // A weighted build side is rejected.
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = join(
            Case {
                join_type,
                null_safe: false,
                filter: false,
                single_match,
                row_layout: false,
            }
            .spec(None),
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
    // Probe weights pass through the mark column and the single-match
    // pairs alike.
    for (join_type, single_match, expect) in [
        (
            JoinType::MarkExists,
            false,
            vec!["1|1|true|*2", "2|2|false|*5"],
        ),
        (JoinType::MarkIn, false, vec!["1|1|true|*2", "2|2|false|*5"]),
        (JoinType::LeftOuter, true, vec!["1|1|1|3|*2", "2|2|∅|∅|*5"]),
    ] {
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = join(
            Case {
                join_type,
                null_safe: false,
                filter: false,
                single_match,
                row_layout: false,
            }
            .spec(None),
            scan(&schema(), vec![Batch::new(data(&build))]),
            scan(
                &schema(),
                vec![
                    Batch::with_weights(
                        data(&probe),
                        Weights::Column(Int64Array::from(vec![2, 5])),
                    )
                    .unwrap(),
                ],
            ),
        );
        let out = run(&runner(1), plan, &q).await.unwrap();
        assert_eq!(render(&out), expect, "{join_type:?}");
        assert_released(&root);
    }
}

fn row(keys: i64, nulls: u32) -> impl Strategy<Value = Row> {
    let k = prop_oneof![1 => Just(None), nulls => (0i64..keys).prop_map(Some)];
    (k, prop::option::of(0i64..20)).prop_map(|(k, v)| Row { k, v })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]
    /// Random data with NULLs, with and without the residual condition, in
    /// both key formats: always the oracle's answer.
    #[test]
    fn random_marks_match_the_oracle(
        join_type in prop_oneof![Just(JoinType::MarkExists), Just(JoinType::MarkIn)],
        safe in any::<bool>(),
        filter in any::<bool>(),
        row_layout in any::<bool>(),
        probe in prop::collection::vec(row(6, 4), 0..40),
        build in prop::collection::vec(row(6, 4), 0..40),
        batch_rows in 1usize..17,
        threads in 1usize..4,
    ) {
        let case = Case {
            join_type,
            null_safe: safe && join_type == JoinType::MarkExists,
            filter,
            single_match: false,
            row_layout,
        };
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (got, _) = rt.block_on(run_case(&case, &probe, &build, batch_rows, threads, 64 * MIB));
        prop_assert_eq!(got.unwrap(), case.oracle(&probe, &build));
    }
}
