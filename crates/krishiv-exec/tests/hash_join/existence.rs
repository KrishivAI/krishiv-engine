// SPDX-License-Identifier: Apache-2.0
//! Existence joins (Spark's `ExistenceJoin`) and single-match left outer
//! joins (P2-T115).
//!
//! The reference model is Spark's definition, written independently of the
//! executor: a probe row's `exists` is TRUE iff some build row makes the
//! whole join condition (key equalities AND the residual) TRUE under SQL
//! three-valued logic, else FALSE; it is never NULL. The cases replayed
//! from Spark 4.2.0 (results recorded on Spark 4.2.0, ANSI on and off
//! alike, and from `tests/spark-golden`) run each statement through the
//! join shape Spark plans for it (its `EXPLAIN`).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::ErrorClass;
use krishiv_exec::{
    ExecConfig, ExecNode, HashJoin, JoinSpec, JoinType, NestedLoopJoin, NestedLoopJoinNode,
    NestedLoopSpec, Predicate, Result,
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

/// Spark's null-aware `NOT IN` condition `(p = b) OR isnull(p = b)` over
/// pair columns `p` and `b` (probe columns, then build columns): TRUE when
/// either side is NULL or they are equal, else FALSE.
#[derive(Debug)]
struct NullAwareEq {
    p: usize,
    b: usize,
}

impl Predicate for NullAwareEq {
    fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
        let p = batch.column(self.p).as_primitive::<Int64Type>();
        let b = batch.column(self.b).as_primitive::<Int64Type>();
        Ok(p.iter()
            .zip(b.iter())
            .map(|(p, b)| Some(p.is_none() || b.is_none() || p == b))
            .collect())
    }
}

/// SQL `x AND y` over three-valued booleans.
fn and3(x: Option<bool>, y: Option<bool>) -> Option<bool> {
    match (x, y) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
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
    fn existence(null_safe: bool, filter: bool, row_layout: bool) -> Self {
        Self {
            join_type: JoinType::Existence,
            null_safe,
            filter,
            single_match: false,
            row_layout,
        }
    }

    fn single(filter: bool, row_layout: bool) -> Self {
        Self {
            join_type: JoinType::LeftOuter,
            null_safe: false,
            filter,
            single_match: true,
            row_layout,
        }
    }

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

    /// The join condition of a pair, in SQL three-valued logic: `p.k = b.k`
    /// (or `p.k <=> b.k`) AND (with the residual) `VLess`.
    fn condition(&self, p: &Row, b: &Row) -> Option<bool> {
        let key = match (p.k, b.k) {
            (Some(x), Some(y)) => Some(x == y),
            (x, y) if self.null_safe => Some(x == y),
            _ => None,
        };
        let residual = if self.filter {
            match (p.v, b.v) {
                (Some(x), Some(y)) => ((x + y) % 7 != 0).then_some(x < y),
                _ => None,
            }
        } else {
            Some(true)
        };
        and3(key, residual)
    }

    /// Spark's `exists` of `p`: some build row makes the condition TRUE.
    fn exists(&self, p: &Row, build: &[Row]) -> bool {
        build.iter().any(|b| self.condition(p, b) == Some(true))
    }

    /// The reference existence join, rendered like [`render`].
    fn reference_existence(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        let cell = |x: &Option<i64>| x.map_or("∅".to_owned(), |v| v.to_string());
        let mut out: Vec<String> = probe
            .iter()
            .map(|p| format!("{}|{}|{}", cell(&p.k), cell(&p.v), self.exists(p, build)))
            .collect();
        out.sort();
        out
    }

    /// The reference single-match join (at most one match per probe row,
    /// as the tests arrange), rendered like [`render`].
    fn reference_single(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        let cell = |x: &Option<i64>| x.map_or("∅".to_owned(), |v| v.to_string());
        let mut out: Vec<String> = probe
            .iter()
            .map(|p| {
                let hits: Vec<&Row> = build
                    .iter()
                    .filter(|b| self.condition(p, b) == Some(true))
                    .collect();
                assert!(hits.len() <= 1, "the reference holds one match at most");
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

    fn reference(&self, probe: &[Row], build: &[Row]) -> Vec<String> {
        if self.single_match {
            self.reference_single(probe, build)
        } else {
            self.reference_existence(probe, build)
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
/// residual condition, `=` and `<=>` keys, in both key formats.
#[tokio::test]
async fn existence_matches_the_reference_on_edge_cases() {
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
    for null_safe in [false, true] {
        for filter in [false, true] {
            for row_layout in [false, true] {
                let case = Case::existence(null_safe, filter, row_layout);
                for build in builds {
                    let (got, _) = run_case(&case, &probe, build, 2, 2, 64 * MIB).await;
                    assert_eq!(
                        got.unwrap(),
                        case.reference(&probe, build),
                        "{case:?} build {build:?}"
                    );
                }
            }
        }
    }
}

/// `exists` is never NULL: a NULL `=` key on either side, a NULL key in
/// the build and a NULL residual all read FALSE, never NULL (Spark's
/// `ExistenceJoin`; the T115 design note's three-valued mark is not what
/// Spark returns).
#[tokio::test]
async fn exists_is_never_null() {
    let probe = [r(Some(2), Some(1)), r(None, Some(1)), r(Some(3), None)];
    let build = [r(Some(1), Some(1)), r(None, Some(1)), r(Some(3), Some(5))];
    for filter in [false, true] {
        let case = Case::existence(false, filter, false);
        let (got, _) = run_case(&case, &probe, &build, 1, 1, 64 * MIB).await;
        let got = got.unwrap();
        assert!(got.iter().all(|row| !row.ends_with('∅')), "{got:?}");
        let expect = if filter {
            ["2|1|false", "3|∅|false", "∅|1|false"]
        } else {
            ["2|1|false", "3|∅|true", "∅|1|false"]
        };
        assert_eq!(got, expect, "{case:?}");
    }
}

/// The specs the executor refuses: single match on a join that is not
/// left outer. Existence joins run by hash over any keys (null-safe or
/// not, with a residual) and by nested loops.
#[test]
fn existence_and_single_match_specs() {
    let schema = schema();
    let spec = |join_type, single_match, keys: usize| JoinSpec {
        join_type,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        probe_keys: vec![0; keys],
        build_keys: vec![0; keys],
        null_safe: vec![true; keys],
        filter: Some(Arc::new(VLess) as Arc<dyn Predicate>),
        single_match,
        spill: None,
    };
    assert!(HashJoin::try_new(spec(JoinType::Existence, false, 2)).is_ok());
    let nested = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: JoinType::Existence,
        probe_schema: Arc::clone(&schema),
        build_schema: Arc::clone(&schema),
        filter: None,
    })
    .unwrap();
    assert_eq!(
        nested.schema().field(2),
        &Field::new("exists", DataType::Boolean, false)
    );
    for join_type in [
        JoinType::Inner,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::NullAwareAnti,
        JoinType::Existence,
    ] {
        let mut s = spec(join_type, true, 1);
        s.filter = None;
        s.null_safe = vec![false];
        let e = HashJoin::try_new(s).err().unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{join_type:?}");
    }
}

/// Zero or one match: a single-match join emits what a left outer join
/// does.
#[tokio::test]
async fn single_match_with_zero_or_one_match_is_a_left_outer_join() {
    for filter in [false, true] {
        for row_layout in [false, true] {
            let case = Case::single(filter, row_layout);
            let probe = [r(Some(1), Some(1)), r(Some(2), Some(3)), r(None, Some(4))];
            let build = [r(Some(1), Some(8))];
            let (got, _) = run_case(&case, &probe, &build, 2, 2, 64 * MIB).await;
            assert_eq!(got.unwrap(), case.reference(&probe, &build), "{case:?}");
        }
    }
}

/// Two matches fail the query with `SCALAR_SUBQUERY_TOO_MANY_ROWS`
/// (SQLSTATE 21000), with and without a residual condition; one match
/// filtered down from two is fine.
#[tokio::test]
async fn single_match_with_two_matches_fails() {
    for filter in [false, true] {
        let case = Case::single(filter, false);
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
    let case = Case::single(true, false);
    // `1 < 0` fails, `1 < 8` holds.
    let probe = [r(Some(1), Some(1))];
    let build = [r(Some(1), Some(0)), r(Some(1), Some(8))];
    let (got, _) = run_case(&case, &probe, &build, 1, 1, 64 * MIB).await;
    assert_eq!(got.unwrap(), case.reference(&probe, &build));
    // A NULL residual on the second pair is not a match either.
    let build = [r(Some(1), Some(8)), r(Some(1), None)];
    let (got, _) = run_case(&case, &probe, &build, 1, 1, 64 * MIB).await;
    assert_eq!(got.unwrap(), case.reference(&probe, &build));
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
    let case = Case::single(false, false);
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

fn skewed(rows: usize, keys: i64, null_every: i64) -> Vec<Row> {
    (0..rows)
        .map(|i| {
            let i = i64::try_from(i).unwrap();
            let k = (i % null_every != 0).then_some(i % keys);
            r(k, Some(i % 50))
        })
        .collect()
}

/// A tiny budget forces the existence join through grace spill
/// (partitions, deferred probe rows, repartitioning): the answer is the
/// reference's, in both key formats.
#[tokio::test(start_paused = true)]
async fn spilling_existence_joins_match_the_reference() {
    let build = skewed(30_000, 20_000, 17);
    let mut probe = skewed(40_000, 25_000, 13);
    probe.extend((0..50).map(|i| r(None, Some(i))));
    for filter in [false, true] {
        for row_layout in [false, true] {
            let case = Case::existence(false, filter, row_layout);
            let expect = case.reference(&probe, &build);
            let (got, q) = run_case(&case, &probe, &build, 1024, 3, 768 << 10).await;
            assert!(got.unwrap() == expect, "{case:?}: grace result differs");
            assert!(
                metric(&q, "HashJoinBuild", "spilled_partitions") > 0,
                "{case:?}"
            );
            assert!(metric(&q, "HashJoinProbe", "grace_partitions") > 0);
            // NULL `=` keys never match: those build rows are not stored.
            let stored = build.iter().filter(|r| r.k.is_some()).count() as u64;
            assert_eq!(metric(&q, "HashJoinBuild", "build_rows"), stored);
            assert_eq!(metric(&q, "HashJoinProbe", "probe_rows"), 40_050);
        }
    }
}

/// A weighted build side is rejected for existence and single-match joins,
/// as for the other non-inner join types; probe weights pass through.
#[tokio::test]
async fn weights_pass_through_or_are_rejected_like_the_other_joins() {
    let probe = [r(Some(1), Some(1)), r(Some(2), Some(2))];
    let build = [r(Some(1), Some(3))];
    let cases = [
        Case::existence(false, false, false),
        Case::single(false, false),
    ];
    for case in &cases {
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = join(
            case.spec(None),
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
    for (case, expect) in cases.iter().zip([
        vec!["1|1|true|*2", "2|2|false|*5"],
        vec!["1|1|1|3|*2", "2|2|∅|∅|*5"],
    ]) {
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = join(
            case.spec(None),
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
        assert_eq!(render(&out), expect, "{case:?}");
        assert_released(&root);
    }
}

fn row(keys: i64, nulls: u32) -> impl Strategy<Value = Row> {
    let k = prop_oneof![1 => Just(None), nulls => (0i64..keys).prop_map(Some)];
    (k, prop::option::of(0i64..20)).prop_map(|(k, v)| Row { k, v })
}

/// The partition of a key under a co-partitioning by `k` into `n` parts.
/// NULL keys of the probe and of the build go to different partitions: an
/// existence join needs no build-wide fact, so where they go is irrelevant.
fn part(k: Option<i64>, n: i64, probe: bool) -> i64 {
    match k {
        Some(k) => k.rem_euclid(n),
        None if probe => 0,
        None => n - 1,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]
    /// Random data with NULLs, with and without the residual condition,
    /// `=` and `<=>` keys, in both key formats: always the reference.
    #[test]
    fn random_existence_matches_the_reference(
        null_safe in any::<bool>(),
        filter in any::<bool>(),
        row_layout in any::<bool>(),
        probe in prop::collection::vec(row(6, 4), 0..40),
        build in prop::collection::vec(row(6, 4), 0..40),
        batch_rows in 1usize..17,
        threads in 1usize..4,
    ) {
        let case = Case::existence(null_safe, filter, row_layout);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (got, _) = rt.block_on(run_case(&case, &probe, &build, batch_rows, threads, 64 * MIB));
        prop_assert_eq!(got.unwrap(), case.reference(&probe, &build));
    }

    /// A partitioned existence join (probe and build co-partitioned by the
    /// key, each partition joined on its own, as tasks of a partitioned
    /// join are) returns what one join over all rows returns.
    #[test]
    fn partitioned_existence_equals_one_join(
        filter in any::<bool>(),
        parts in 1i64..5,
        probe in prop::collection::vec(row(8, 3), 0..40),
        build in prop::collection::vec(row(8, 3), 0..40),
    ) {
        let case = Case::existence(false, filter, false);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let mut got = Vec::new();
        for p in 0..parts {
            let probe: Vec<Row> =
                probe.iter().filter(|r| part(r.k, parts, true) == p).copied().collect();
            let build: Vec<Row> =
                build.iter().filter(|r| part(r.k, parts, false) == p).copied().collect();
            let (out, _) = rt.block_on(run_case(&case, &probe, &build, 4, 2, 64 * MIB));
            got.extend(out.unwrap());
        }
        got.sort();
        prop_assert_eq!(got, case.reference(&probe, &build));
    }
}

// ---------------------------------------------------------------------------
// Spark 4.2.0 replays.
// ---------------------------------------------------------------------------

/// Rows of nullable Int64 columns named `names`.
fn table(names: &[&str], rows: &[&[Option<i64>]]) -> (SchemaRef, RecordBatch) {
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|n| Field::new(*n, DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    let columns = (0..names.len())
        .map(|c| {
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r[c]).collect::<Vec<_>>(),
            )) as _
        })
        .collect();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(&schema),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
    .unwrap();
    (schema, batch)
}

/// How Spark plans the statement (its `EXPLAIN`).
enum Shape {
    /// `BroadcastHashJoin [probe keys], [build keys], ExistenceJoin`, with
    /// an optional `(p = b) OR isnull(p = b)` residual over pair columns.
    Hash {
        probe_keys: Vec<usize>,
        build_keys: Vec<usize>,
        null_aware: Option<(usize, usize)>,
    },
    /// `BroadcastNestedLoopJoin ExistenceJoin, (p = b) OR isnull(p = b)`.
    NestedNullAware(usize, usize),
}

/// The `exists` column of `probe` joined with `build` in `shape`, in probe
/// row order (sorted by the first probe column, an id).
async fn exists_of(
    shape: &Shape,
    probe: &(SchemaRef, RecordBatch),
    build: &(SchemaRef, RecordBatch),
) -> Vec<(i64, bool)> {
    let root = root(64 * MIB);
    let q = query(&root, 1, ExecConfig::default());
    let probe_scan = scan(&probe.0, vec![Batch::new(probe.1.clone())]);
    let build_scan = scan(&build.0, vec![Batch::new(build.1.clone())]);
    let plan: Arc<dyn ExecNode> = match shape {
        Shape::Hash {
            probe_keys,
            build_keys,
            null_aware,
        } => join(
            JoinSpec {
                join_type: JoinType::Existence,
                probe_schema: Arc::clone(&probe.0),
                build_schema: Arc::clone(&build.0),
                probe_keys: probe_keys.clone(),
                build_keys: build_keys.clone(),
                null_safe: vec![false; probe_keys.len()],
                filter: null_aware.map(|(p, b)| Arc::new(NullAwareEq { p, b }) as _),
                single_match: false,
                spill: None,
            },
            build_scan,
            probe_scan,
        ),
        Shape::NestedNullAware(p, b) => Arc::new(NestedLoopJoinNode {
            join: NestedLoopJoin::try_new(NestedLoopSpec {
                join_type: JoinType::Existence,
                probe_schema: Arc::clone(&probe.0),
                build_schema: Arc::clone(&build.0),
                filter: Some(Arc::new(NullAwareEq { p: *p, b: *b })),
            })
            .unwrap(),
            build: build_scan,
            probe: probe_scan,
        }),
    };
    let out = run(&runner(2), plan, &q).await.unwrap();
    assert_released(&root);
    let mut rows = Vec::new();
    for b in &out {
        let b = b.data();
        let id = b.column(0).as_primitive::<Int64Type>();
        let exists = b.column(b.num_columns() - 1).as_boolean();
        assert_eq!(exists.null_count(), 0, "exists is never NULL");
        for i in 0..b.num_rows() {
            rows.push((id.value(i), exists.value(i)));
        }
    }
    rows.sort_unstable();
    rows
}

/// The ids and their expected values, `NOT` applied where the statement
/// negates the existence join (`NOT IN`).
fn expect(values: &[bool]) -> Vec<(i64, bool)> {
    (1..).zip(values.iter().copied()).collect()
}

fn negated(rows: Vec<(i64, bool)>) -> Vec<(i64, bool)> {
    rows.into_iter().map(|(id, e)| (id, !e)).collect()
}

/// Results recorded on Spark 4.2.0 (2026-10-25, `spark.sql.ansi.enabled`
/// true and false alike) over
/// `t(id, k) = (1,1), (2,2), (3,NULL), (4,4)`, `un(k) = (1), (NULL)`,
/// `nn(k) = (1), (2)`, `e(k)` empty and
/// `c(g, k) = (1,1), (2,NULL), (4,9), (3,5)`, each run in the shape of
/// Spark's plan. The same values come out in the SELECT list, under `OR`
/// in WHERE, in `CASE` and in a join condition: Spark plans every one as an
/// `ExistenceJoin`, whose `exists` is never NULL.
#[tokio::test]
async fn spark_in_and_not_in_outside_a_filter_conjunct() {
    let none = None;
    let t = table(
        &["id", "k"],
        &[
            &[Some(1), Some(1)],
            &[Some(2), Some(2)],
            &[Some(3), none],
            &[Some(4), Some(4)],
        ],
    );
    let un = table(&["k"], &[&[Some(1)], &[none]]);
    let nn = table(&["k"], &[&[Some(1)], &[Some(2)]]);
    let e = table(&["k"], &[]);
    let c = table(
        &["g", "k"],
        &[
            &[Some(1), Some(1)],
            &[Some(2), none],
            &[Some(4), Some(9)],
            &[Some(3), Some(5)],
        ],
    );
    // `k IN (SELECT k FROM s)`: BroadcastHashJoin [k], [k], ExistenceJoin.
    let in_shape = Shape::Hash {
        probe_keys: vec![1],
        build_keys: vec![0],
        null_aware: None,
    };
    // `k NOT IN (SELECT k FROM s)`: NOT of BroadcastNestedLoopJoin
    // ExistenceJoin, (k = s.k) OR isnull(k = s.k).
    let not_in_shape = Shape::NestedNullAware(1, 2);
    for (s, want_in, want_not_in) in [
        // SELECT id, k IN (SELECT k FROM un): 1 true, 2 false, 3 false,
        // 4 false; NOT IN: all false.
        (&un, [true, false, false, false], [false; 4]),
        // nn: IN 1 true, 2 true, 3 false, 4 false; NOT IN only 4 true.
        (&nn, [true, true, false, false], [false, false, false, true]),
        // e: IN all false; NOT IN all true (NULL probe key too).
        (&e, [false; 4], [true; 4]),
    ] {
        assert_eq!(exists_of(&in_shape, &t, s).await, expect(&want_in));
        assert_eq!(
            negated(exists_of(&not_in_shape, &t, s).await),
            expect(&want_not_in)
        );
    }
    // `k IN (SELECT c.k FROM c WHERE c.g = t.id)`:
    // BroadcastHashJoin [k, id], [k, g], ExistenceJoin. 1 true, else false.
    let corr_in = Shape::Hash {
        probe_keys: vec![1, 0],
        build_keys: vec![1, 0],
        null_aware: None,
    };
    assert_eq!(
        exists_of(&corr_in, &t, &c).await,
        expect(&[true, false, false, false])
    );
    // `k NOT IN (SELECT c.k FROM c WHERE c.g = t.id)`: NOT of
    // BroadcastHashJoin [id], [g], ExistenceJoin,
    // (k = c.k) OR isnull(k = c.k). Only 4 true: 2 meets a NULL `c.k`, 3
    // has a NULL `k`.
    let corr_not_in = Shape::Hash {
        probe_keys: vec![0],
        build_keys: vec![0],
        null_aware: Some((1, 3)),
    };
    assert_eq!(
        negated(exists_of(&corr_not_in, &t, &c).await),
        expect(&[false, false, false, true])
    );
    // `EXISTS (SELECT 1 FROM c WHERE c.g = t.id AND c.k = t.k)`: 1 true.
    assert_eq!(
        exists_of(&corr_in, &t, &c).await,
        expect(&[true, false, false, false])
    );
}

/// `tests/spark-golden/v4.2.0/results/subquery/in-subquery/in-null-semantics.sql.out`
/// (`spark.sql.legacy.nullInEmptyListBehavior = false`): over
/// `t(c) = (1), (NULL)` and an empty `t_empty(e)`,
/// `c in (select e from t_empty)` is false, false and
/// `c not in (select e from t_empty)` is true, true.
#[tokio::test]
async fn spark_golden_in_null_semantics() {
    let t = table(&["c"], &[&[Some(1)], &[None]]);
    let t_empty = table(&["e"], &[]);
    // Both rows expect the same value, so their order does not matter.
    let in_shape = Shape::Hash {
        probe_keys: vec![0],
        build_keys: vec![0],
        null_aware: None,
    };
    let rows = exists_of(&in_shape, &t, &t_empty).await;
    assert_eq!(rows.iter().map(|r| r.1).collect::<Vec<_>>(), [false, false]);
    let rows = negated(exists_of(&Shape::NestedNullAware(0, 1), &t, &t_empty).await);
    assert_eq!(rows.iter().map(|r| r.1).collect::<Vec<_>>(), [true, true]);
}

/// `tests/spark-golden/v4.2.0/results/subquery/exists-subquery/exists-outside-filter.sql.out`:
/// `SELECT emp_name, EXISTS (SELECT 1 FROM dept WHERE emp.dept_id =
/// dept.dept_id) FROM emp`, emp 1 to emp 8 by id.
#[tokio::test]
async fn spark_golden_exists_outside_filter() {
    let emp = table(
        &["id", "dept_id"],
        &[
            &[Some(100), Some(10)],
            &[Some(100), Some(10)],
            &[Some(200), Some(10)],
            &[Some(300), Some(20)],
            &[Some(400), Some(30)],
            &[Some(500), None],
            &[Some(600), Some(100)],
            &[Some(700), Some(100)],
            &[Some(800), Some(70)],
        ],
    );
    let dept = table(
        &["dept_id"],
        &[
            &[Some(10)],
            &[Some(20)],
            &[Some(30)],
            &[Some(40)],
            &[Some(50)],
            &[Some(70)],
        ],
    );
    let shape = Shape::Hash {
        probe_keys: vec![1],
        build_keys: vec![0],
        null_aware: None,
    };
    let rows = exists_of(&shape, &emp, &dept).await;
    // emp 1, emp 1, emp 2, emp 3, emp 4: true; emp 5 (NULL dept), emp 6,
    // emp 7: false; emp 8: true.
    assert_eq!(
        rows,
        [
            (100, true),
            (100, true),
            (200, true),
            (300, true),
            (400, true),
            (500, false),
            (600, false),
            (700, false),
            (800, true),
        ]
    );
}

/// The nested-loop existence join over random data equals the reference
/// with the key equality moved into the condition.
#[test]
fn nested_loop_existence_matches_the_reference() {
    #[derive(Debug)]
    struct KeyAndV;
    impl Predicate for KeyAndV {
        fn evaluate(&self, batch: &RecordBatch) -> Result<BooleanArray> {
            let pk = batch.column(0).as_primitive::<Int64Type>();
            let bk = batch.column(2).as_primitive::<Int64Type>();
            let v = VLess.evaluate(batch)?;
            Ok((0..batch.num_rows())
                .map(|i| {
                    let key =
                        (pk.is_valid(i) && bk.is_valid(i)).then(|| pk.value(i) == bk.value(i));
                    let v = v.is_valid(i).then(|| v.value(i));
                    and3(key, v)
                })
                .collect())
        }
    }
    let case = Case::existence(false, true, false);
    let probe = skewed(300, 40, 7);
    let build = skewed(200, 50, 5);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let got = rt.block_on(async {
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let plan = Arc::new(NestedLoopJoinNode {
            join: NestedLoopJoin::try_new(NestedLoopSpec {
                join_type: JoinType::Existence,
                probe_schema: schema(),
                build_schema: schema(),
                filter: Some(Arc::new(KeyAndV)),
            })
            .unwrap(),
            build: scan(&schema(), batches(&data(&build), 64)),
            probe: scan(&schema(), batches(&data(&probe), 64)),
        });
        let out = run(&runner(2), plan, &q).await.unwrap();
        assert_released(&root);
        render(&out)
    });
    assert_eq!(got, case.reference(&probe, &build));
}
