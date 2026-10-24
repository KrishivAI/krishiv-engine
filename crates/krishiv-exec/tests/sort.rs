// SPDX-License-Identifier: Apache-2.0
//! The sort breakers end to end (P2-T054 acceptance): Spark's ordering
//! (NULLS FIRST/LAST × ASC/DESC × NaN/−0.0 × multi-key) against goldens
//! from Spark 4.2.0, a reference-sort property test in memory and spilling,
//! an external sort under an 8:1 data-to-memory ratio, the top-N
//! reservation probe, cancellation and deterministic replay; every query
//! releases its memory.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test row counts and ids are small"
)]

use std::cmp::Ordering as Cmp;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float64Type, Int32Type, Int64Type};
use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::testing::{Break, Breaker, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, ExecutorConfig, MemoryScan, QueryContext, QueryId, QueryRunner, Result,
    Sort, SortKey, Source, TokioClock, TokioExecutor, TopN, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;

const KIB: usize = 1 << 10;

fn root() -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 4 << 10,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
        },
        Arc::new(TokioClock),
    )
}

/// A query whose pool is capped at `limit` bytes, failing on a stall (a
/// driver slice over the threshold: generous for unoptimised builds unless
/// the test sets its own).
fn query(root: &TreePool, mut config: ExecConfig, limit: Option<usize>) -> Arc<QueryContext> {
    if !config.fail_on_stall {
        config.fail_on_stall = true;
        config.stall_threshold = Duration::from_millis(500);
    }
    let pool = root.child("q", limit);
    QueryContext::new(QueryId(1), WorkloadClass::Interactive, config, pool)
}

fn batch_rows(n: usize) -> ExecConfig {
    ExecConfig {
        batch_size: n,
        ..ExecConfig::default()
    }
}

fn deterministic(slots: usize) -> QueryRunner {
    QueryRunner::new(TokioExecutor::deterministic(slots).unwrap())
}

fn threaded() -> QueryRunner {
    let config = ExecutorConfig {
        compute_threads: 2,
        io_threads: 1,
        blocking_threads: 2,
    };
    QueryRunner::new(TokioExecutor::new(&config).unwrap())
}

/// A fresh spill directory under the target dir (not `/tmp`).
fn spill() -> (SpillManager, SpillScope) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("sort-{n}"));
    let manager = SpillManager::open(&SpillConfig {
        dir,
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    (manager, scope)
}

async fn run(
    runner: &QueryRunner,
    plan: Arc<dyn ExecNode>,
    q: &Arc<QueryContext>,
) -> Result<Vec<Batch>> {
    let mut stream = runner.execute(plan, Arc::clone(q)).await?;
    let mut out = Vec::new();
    let mut error = None;
    while let Some(b) = stream.next().await {
        match b {
            Ok(b) => out.push(b),
            Err(e) => error = Some(e),
        }
    }
    drop(stream);
    q.finished().await;
    error.map_or(Ok(out), Err)
}

fn assert_released(root: &TreePool) {
    assert_eq!(root.reserved(), 0, "the query's memory was not released");
}

/// Scan `input` into `breaker`; the pipeline above keeps its order.
fn plan<B: Breaker + 'static>(input: Vec<Batch>, breaker: &Arc<B>) -> Arc<dyn ExecNode> {
    let schema = Source::schema(&**breaker);
    let scan = Arc::new(Scan(Arc::new(MemoryScan::try_new(schema, input).unwrap())));
    Arc::new(Break(scan, Arc::clone(breaker), true))
}

fn column<T: ArrowPrimitiveType>(batches: &[Batch], c: usize) -> Vec<T::Native> {
    let values = |b: &Batch| b.data().column(c).as_primitive::<T>().values().to_vec();
    batches.iter().flat_map(values).collect()
}

/// `batch`'s rows cut into batches of `rows` (slices: the sort compacts
/// them).
fn split(batch: &Batch, rows: usize) -> Vec<Batch> {
    let n = batch.num_rows();
    let piece = |o| batch.slice(o, rows.min(n - o)).unwrap();
    (0..n).step_by(rows.max(1)).map(piece).collect()
}

// ---------------------------------------------------------------- Spark goldens

type Case = (String, Vec<SortKey>, Vec<i32>);

/// The fixture's table (id, d, f, s, i) and its cases, each with the
/// `id ASC` tie-break the goldens end with.
fn golden() -> (Batch, Vec<Case>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/sort-ordering.json"
    );
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(doc["spark_version"], "4.2.0");
    let rows = doc["rows"].as_array().unwrap();
    let text = |c: usize| rows.iter().map(move |r| r[c].as_str().unwrap());
    let float = |t: &str| match t {
        "NULL" => None,
        "-NaN" => Some(-f64::NAN),
        t => Some(t.parse::<f64>().unwrap()),
    };
    let int = |t: &str| (t != "NULL").then(|| t.parse::<i32>().unwrap());
    let id: Int32Array = rows
        .iter()
        .map(|r| r[0].as_i64().map(|v| v as i32))
        .collect();
    let d: Float64Array = text(1).map(float).collect();
    let f: Float32Array = text(2)
        .map(|t| match t {
            "-NaN" => Some(-f32::NAN),
            t => float(t).map(|_| t.parse::<f32>().unwrap()),
        })
        .collect();
    let s: StringArray = text(3)
        .map(|t| (t != "NULL").then(|| t.trim_matches('\'')))
        .collect();
    let i: Int32Array = text(4).map(int).collect();
    assert!(
        d.values()
            .iter()
            .any(|v| v.is_nan() && v.is_sign_negative())
    );
    let columns: [(_, ArrayRef); 5] = [
        ("id", Arc::new(id)),
        ("d", Arc::new(d)),
        ("f", Arc::new(f)),
        ("s", Arc::new(s)),
        ("i", Arc::new(i)),
    ];
    let batch = Batch::new(RecordBatch::try_from_iter(columns).unwrap());
    let key = |k: &serde_json::Value| {
        let names = ["id", "d", "f", "s", "i"];
        let c = names.iter().position(|n| k["column"] == *n).unwrap();
        let key = match k["descending"].as_bool() {
            Some(true) => SortKey::desc(c),
            _ => SortKey::asc(c),
        };
        k["nulls_first"]
            .as_bool()
            .map_or(key, |n| key.nulls_first(n))
    };
    let cases = doc["cases"].as_array().unwrap().iter().map(|c| {
        let mut keys: Vec<SortKey> = c["keys"].as_array().unwrap().iter().map(key).collect();
        keys.push(SortKey::asc(0));
        let ids = c["ids"].as_array().unwrap().iter();
        let ids = ids.map(|v| v.as_i64().unwrap() as i32).collect();
        (c["sql"].as_str().unwrap().to_owned(), keys, ids)
    });
    (batch, cases.collect())
}

/// Acceptance: NULLS FIRST/LAST × ASC/DESC × NaN/−0.0 × multi-key orders
/// match Spark 4.2.0, in memory (one batch, and one row per batch), and
/// the output keeps the original float values (-0.0, -NaN).
#[tokio::test(start_paused = true)]
async fn sort_matches_spark_ordering_goldens() {
    let (batch, cases) = golden();
    let runner = deterministic(1);
    let original = batch.data().column(1).as_primitive::<Float64Type>().clone();
    for (sql, keys, expected) in &cases {
        for per_batch in [batch.num_rows(), 1] {
            let root = root();
            let q = query(&root, batch_rows(4), None);
            let sort = Arc::new(Sort::new(batch.schema(), keys, None).unwrap());
            let out = run(&runner, plan(split(&batch, per_batch), &sort), &q).await;
            let out = out.unwrap();
            let ids = column::<Int32Type>(&out, 0);
            assert_eq!(&ids, expected, "{sql} ({per_batch} rows per batch)");
            for (id, d) in ids.iter().zip(column::<Float64Type>(&out, 1)) {
                let row = usize::try_from(id - 1).unwrap();
                assert_eq!(original.value(row).to_bits(), d.to_bits());
            }
            assert_released(&root);
        }
    }
}

/// The same goldens through the external path: every row repeated, small
/// batches, a memory cap that forces spills and a fan-in of 2 (several
/// merge passes); ties within a row's copies keep their input order.
#[tokio::test(start_paused = true)]
async fn external_sort_matches_spark_ordering_goldens() {
    let (batch, cases) = golden();
    let copies = 100;
    // A padding column makes the copies outgrow the memory cap.
    let pad = StringArray::from_iter_values((0..batch.num_rows()).map(|_| "p".repeat(300)));
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("pad", DataType::Utf8, false)));
    let mut columns = batch.data().columns().to_vec();
    columns.push(Arc::new(pad));
    let padded = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let batch = Batch::new(padded);
    let input = Batch::concat(&batch.schema(), &vec![batch.clone(); copies]).unwrap();
    let runner = deterministic(1);
    for i in [0, 4, 9, 14, 20, 24, 25, 26, 27, 28] {
        let (sql, keys, expected) = &cases[i];
        let root = root();
        let q = query(&root, batch_rows(8), Some(192 * KIB));
        let (_manager, scope) = spill();
        let sort = Sort::new(batch.schema(), keys, Some(scope)).unwrap();
        let sort = Arc::new(sort.with_max_fan_in(2));
        let out = run(&runner, plan(split(&input, 8), &sort), &q).await;
        let want = expected
            .iter()
            .flat_map(|id| std::iter::repeat_n(*id, copies));
        assert_eq!(
            column::<Int32Type>(&out.unwrap(), 0),
            want.collect::<Vec<_>>(),
            "{sql}"
        );
        let m = sort.metrics();
        assert!(
            m.runs_spilled > 2 && m.merge_passes > 0 && m.final_fan_in == 2,
            "{m:?}"
        );
        assert_released(&root);
    }
}

/// Top-N gives Spark's prefix, for several k and offsets.
#[tokio::test(start_paused = true)]
async fn topn_matches_spark_ordering_goldens() {
    let (batch, cases) = golden();
    let runner = deterministic(2);
    for (sql, keys, expected) in &cases {
        for (offset, limit, per_batch) in [(0, 15, 15), (0, 4, 1), (3, 5, 2), (14, 9, 4), (0, 0, 3)]
        {
            let root = root();
            let q = query(&root, batch_rows(2), None);
            let topn = Arc::new(TopN::new(batch.schema(), keys, offset, limit).unwrap());
            let out = run(&runner, plan(split(&batch, per_batch), &topn), &q).await;
            let want: Vec<i32> = expected.iter().skip(offset).take(limit).copied().collect();
            let got = column::<Int32Type>(&out.unwrap(), 0);
            assert_eq!(got, want, "{sql} OFFSET {offset} LIMIT {limit}");
            assert!(topn.max_retained_rows() <= (offset + limit) as u64);
            assert_released(&root);
        }
    }
}

// ------------------------------------------------------- reference property test

/// Spark's comparison of one nullable key (floats: NaN largest and equal,
/// -0.0 = 0.0).
fn spark_cmp<T: PartialOrd>(a: Option<T>, b: Option<T>, k: SortKey, nan: fn(&T) -> bool) -> Cmp {
    let nulls = if k.nulls_first {
        Cmp::Less
    } else {
        Cmp::Greater
    };
    match (a, b) {
        (None, None) => Cmp::Equal,
        (None, Some(_)) => nulls,
        (Some(_), None) => nulls.reverse(),
        (Some(x), Some(y)) => {
            let o = match (nan(&x), nan(&y)) {
                (true, true) => Cmp::Equal,
                (true, false) => Cmp::Greater,
                (false, true) => Cmp::Less,
                (false, false) => x.partial_cmp(&y).unwrap(),
            };
            if k.descending { o.reverse() } else { o }
        }
    }
}

type Row = (Option<i64>, Option<f64>, Option<String>);

/// `(pos, a, d, s, pad)`: the padding makes the external case spill.
fn rows_batch(rows: &[Row]) -> Batch {
    let pos = Int32Array::from_iter_values(0..rows.len() as i32);
    let a: Int64Array = rows.iter().map(|r| r.0).collect();
    let d: Float64Array = rows.iter().map(|r| r.1).collect();
    let s: StringArray = rows.iter().map(|r| r.2.clone()).collect();
    let pad = StringArray::from_iter_values(rows.iter().map(|_| "x".repeat(400)));
    let columns: [(_, ArrayRef); 5] = [
        ("pos", Arc::new(pos)),
        ("a", Arc::new(a)),
        ("d", Arc::new(d)),
        ("s", Arc::new(s)),
        ("pad", Arc::new(pad)),
    ];
    Batch::new(RecordBatch::try_from_iter(columns).unwrap())
}

/// The stable reference order of `rows` by `keys` (columns 1..=3).
fn reference(rows: &[Row], keys: &[SortKey]) -> Vec<i32> {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&x, &y| {
        let (rx, ry) = (&rows[x], &rows[y]);
        let cmp = |&k: &SortKey| match k.column {
            1 => spark_cmp(rx.0, ry.0, k, |_| false),
            2 => spark_cmp(rx.1, ry.1, k, |v| v.is_nan()),
            _ => spark_cmp(rx.2.as_deref(), ry.2.as_deref(), k, |_| false),
        };
        keys.iter()
            .map(cmp)
            .find(|o| o.is_ne())
            .unwrap_or(Cmp::Equal)
    });
    idx.into_iter().map(|i| i as i32).collect()
}

fn row_strategy() -> impl Strategy<Value = Row> {
    let float = prop_oneof![
        Just(f64::NAN),
        Just(-f64::NAN),
        Just(0.0),
        Just(-0.0),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        (-3i32..3).prop_map(f64::from),
    ];
    (
        proptest::option::weighted(0.8, -4i64..4),
        proptest::option::weighted(0.8, float),
        proptest::option::weighted(0.8, "[ab\u{e9}]{0,3}"),
    )
}

fn key_strategy() -> impl Strategy<Value = Vec<SortKey>> {
    let key = (1usize..4, any::<bool>(), any::<bool>()).prop_map(|(c, desc, nulls_first)| {
        let k = if desc {
            SortKey::desc(c)
        } else {
            SortKey::asc(c)
        };
        k.nulls_first(nulls_first)
    });
    proptest::collection::vec(key, 1..4)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// The sort (in memory, and spilling with fan-in 2) equals a stable
    /// reference sort with Spark's comparisons, for any batch split.
    #[test]
    fn sort_equals_a_stable_reference(
        rows in proptest::collection::vec(row_strategy(), 0..300),
        keys in key_strategy(),
        per_batch in 1usize..40,
        external in any::<bool>(),
    ) {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let out = rt.block_on(async {
            let root = root();
            let (limit, spill) = if external { (Some(160 * KIB), Some(spill())) } else { (None, None) };
            let q = query(&root, batch_rows(7), limit);
            let batch = rows_batch(&rows);
            let scope = spill.as_ref().map(|s| s.1.clone());
            let sort = Arc::new(Sort::new(batch.schema(), &keys, scope).unwrap().with_max_fan_in(2));
            let out = run(&deterministic(1), plan(split(&batch, per_batch), &sort), &q).await;
            assert_released(&root);
            column::<Int32Type>(&out.unwrap(), 0)
        });
        prop_assert_eq!(out, reference(&rows, &keys));
    }
}

// ------------------------------------------------------------ larger inputs

/// A pseudo-random key with many duplicates (`rows / 4` distinct).
fn key(rows: u64, i: u64) -> i64 {
    let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) % (rows / 4).max(1)) as i64
}

/// `rows` rows `(k, i, pad)` in batches of 4096: `k = key(rows, i)`, `i`
/// the row number, `pad` a `pad`-byte string.
fn table(rows: u64, pad: usize) -> Vec<Batch> {
    let batch = |start: u64| {
        let ids = start..(start + 4096).min(rows);
        let k = Int64Array::from_iter_values(ids.clone().map(|i| key(rows, i)));
        let i = Int64Array::from_iter_values(ids.clone().map(|i| i as i64));
        let pad = StringArray::from_iter_values(ids.map(|i| format!("{i:0>pad$}")));
        let columns: [(_, ArrayRef); 3] = [
            ("k", Arc::new(k)),
            ("i", Arc::new(i)),
            ("pad", Arc::new(pad)),
        ];
        Batch::new(RecordBatch::try_from_iter(columns).unwrap())
    };
    (0..rows).step_by(4096).map(batch).collect()
}

/// Checks sorting `rows` rows of [`table`] by `k`: every row once, intact,
/// in key order; with `stable`, ties in input order.
fn check_sorted(rows: u64, out: &[Batch], stable: bool) {
    let (k, i) = (column::<Int64Type>(out, 0), column::<Int64Type>(out, 1));
    assert_eq!(k.len() as u64, rows);
    let mut seen = vec![false; k.len()];
    for r in 0..k.len() {
        let row = i[r] as usize;
        assert!(!std::mem::replace(&mut seen[row], true), "row {row} twice");
        assert_eq!(k[r], key(rows, row as u64), "row {row} kept its key");
        if r > 0 {
            assert!(k[r - 1] <= k[r], "key order at {r}");
            assert!(
                !stable || k[r - 1] < k[r] || i[r - 1] < i[r],
                "tie order at {r}"
            );
        }
    }
}

/// Acceptance (scaled): sorting 8× the memory budget spills, merges in
/// several passes and is correct, and leaves no memory and no spill file
/// behind. `KRISHIV_SORT_SPILL_MIB` / `KRISHIV_SORT_BUDGET_MIB` set the
/// data size and the budget (the plan's run: 2048 under 256).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_sort_spills_under_an_eighth_of_its_data() {
    let mib = |var, default| std::env::var(var).map_or(default, |v| v.parse().unwrap());
    let (data, budget): (usize, usize) = (
        mib("KRISHIV_SORT_SPILL_MIB", 32),
        mib("KRISHIV_SORT_BUDGET_MIB", 4),
    );
    // About 100 bytes of padding and 20 of ints and offsets per row.
    let rows = ((data << 20) / 120) as u64;
    let root = root();
    let config = ExecConfig {
        morsel_rows: 64 * 1024,
        fail_on_stall: true,
        stall_threshold: Duration::from_secs(1),
        ..ExecConfig::default()
    };
    let q = query(&root, config, Some(budget << 20));
    let (manager, scope) = spill();
    let sort = Sort::new(table(1, 100)[0].schema(), &[SortKey::asc(0)], Some(scope)).unwrap();
    let sort = Arc::new(sort.with_max_fan_in(4));
    let started = std::time::Instant::now();
    let out = run(&threaded(), plan(table(rows, 100), &sort), &q)
        .await
        .unwrap();
    let took = started.elapsed();
    check_sorted(rows, &out, false);
    let m = sort.metrics();
    assert_eq!(m.rows_sorted, rows);
    assert!(
        m.runs_spilled >= 8 && m.merge_passes > 0 && m.final_fan_in <= 4,
        "{m:?}"
    );
    let qm = q.metrics();
    let limit = u64::try_from(budget << 20).unwrap();
    assert_eq!(qm.memory.limit, Some(limit));
    assert!(
        qm.memory.peak_reserved > 0 && qm.memory.peak_reserved <= limit,
        "{:?}",
        qm.memory
    );
    let sink = qm
        .operators
        .into_values()
        .find(|e| e.name == "Sort")
        .unwrap();
    assert!(sink.metrics.spills > 0, "the drivers spilled on request");
    // Every run and every merge pass's output is a file of the sort's.
    assert!(
        sink.metrics.spill_files >= m.runs_spilled && sink.metrics.spill_write_bytes > 0,
        "{:?} {m:?}",
        sink.metrics
    );
    assert_released(&root);
    assert_eq!(manager.used_bytes(), 0, "every spill file is gone");
    // Reported by the full-size local run (`--no-capture`).
    eprintln!("external sort: {data} MiB under {budget} MiB in {took:?}: {m:?}");
}

/// Without keys every row ties: the sort gathers its input into one
/// stream, in input order with one driver, through spills and merge
/// passes (a window over all rows sits on it).
#[tokio::test(start_paused = true)]
async fn a_sort_without_keys_gathers_its_input_in_order() {
    let pool = root();
    let q = query(&pool, ExecConfig::default(), Some(2 << 20));
    let (_manager, scope) = spill();
    let sort = Sort::new(table(1, 30)[0].schema(), &[], Some(scope)).unwrap();
    let sort = Arc::new(sort.with_max_fan_in(2));
    let out = run(&deterministic(1), plan(table(60_000, 30), &sort), &q)
        .await
        .unwrap();
    let ids = column::<Int64Type>(&out, 1);
    assert_eq!(ids, (0..60_000).collect::<Vec<i64>>());
    assert!(sort.metrics().merge_passes > 0, "{:?}", sort.metrics());
    drop(out);
    assert_released(&pool);
}

/// One driver keeps ties in input order through spills and merge passes,
/// and deterministic mode replays the same order with several drivers.
#[tokio::test(start_paused = true)]
async fn stable_within_a_driver_and_deterministic_across_drivers() {
    let pool = root();
    let q = query(&pool, ExecConfig::default(), Some(2 << 20));
    let (_manager, scope) = spill();
    let sort = Sort::new(table(1, 30)[0].schema(), &[SortKey::asc(0)], Some(scope)).unwrap();
    let sort = Arc::new(sort.with_max_fan_in(2));
    let out = run(&deterministic(1), plan(table(60_000, 30), &sort), &q).await;
    check_sorted(60_000, &out.unwrap(), true);
    assert!(sort.metrics().merge_passes > 0, "{:?}", sort.metrics());
    assert_released(&pool);

    let mut outputs = Vec::new();
    for _ in 0..2 {
        let root = root();
        let config = ExecConfig {
            batch_size: 256,
            morsel_rows: 1024,
            seed: Some(7),
            ..ExecConfig::default()
        };
        let q = query(&root, config, Some(2 << 20));
        let (_manager, scope) = spill();
        let schema = table(1, 60)[0].schema();
        let sort = Arc::new(Sort::new(schema, &[SortKey::desc(0)], Some(scope)).unwrap());
        let out = run(&deterministic(4), plan(table(40_000, 60), &sort), &q).await;
        assert!(sort.metrics().runs_spilled > 0);
        outputs.push(column::<Int64Type>(&out.unwrap(), 1));
        assert_released(&root);
    }
    assert_eq!(outputs[0].len(), 40_000);
    assert_eq!(outputs[0], outputs[1]);
}

/// Acceptance: top-N keeps at most k rows (the reservation probe: retained
/// rows within k, the sink's peak reservation within two k-row states per
/// driver) while scanning far more.
#[tokio::test(start_paused = true)]
async fn topn_keeps_at_most_k_rows() {
    let (k, rows) = (100, 200_000);
    let root = root();
    let q = query(&root, ExecConfig::default(), None);
    let keys = [SortKey::desc(0), SortKey::asc(1)];
    let topn = Arc::new(TopN::new(table(1, 20)[0].schema(), &keys, 0, k).unwrap());
    let out = run(&deterministic(2), plan(table(rows, 20), &topn), &q)
        .await
        .unwrap();
    let got = column::<Int64Type>(&out, 0)
        .into_iter()
        .zip(column::<Int64Type>(&out, 1));
    let mut want: Vec<(i64, i64)> = (0..rows).map(|i| (key(rows, i), i as i64)).collect();
    want.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    want.truncate(k);
    assert_eq!(got.collect::<Vec<_>>(), want);
    assert!(topn.max_retained_rows() <= k as u64);
    // A k-row state: ~k * 56 bytes of data plus key rows and buffer slack.
    let state = k * 56 + 4 * KIB;
    let metrics = q.metrics().operators.into_values();
    let sink = metrics.into_iter().find(|e| e.name == "TopN").unwrap();
    let peak = sink.metrics.peak_reserved;
    assert!(peak > 0 && peak <= (2 * 2 * state) as u64, "{peak}");
    assert_released(&root);
}

// ------------------------------------------------------- contracts and errors

fn small(weights: Weights) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int32, false),
    ]));
    let k = Int64Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);
    let v = Int32Array::from(vec![0, 1, 2, 3, 4]);
    let data = RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).unwrap();
    Batch::with_weights(data, weights).unwrap()
}

/// Empty input, empty batches, all-null keys, LIMIT 0, weighted top-N and
/// invalid sorts.
#[tokio::test(start_paused = true)]
async fn edge_cases_and_errors() {
    let b = small(Weights::Const(1));
    let nulls = RecordBatch::try_new(
        b.schema(),
        vec![
            Arc::new(Int64Array::from(vec![None::<i64>; 3])),
            Arc::new(Int32Array::from(vec![7, 8, 9])),
        ],
    );
    let nulls = Batch::new(nulls.unwrap());
    let inputs = [vec![], vec![b.slice(0, 0).unwrap()], vec![nulls]];
    for (input, want) in inputs.into_iter().zip([vec![], vec![], vec![7, 8, 9]]) {
        let root = root();
        let q = query(&root, ExecConfig::default(), None);
        let sort = Arc::new(Sort::new(b.schema(), &[SortKey::desc(0)], None).unwrap());
        let out = run(&deterministic(1), plan(input.clone(), &sort), &q).await;
        assert_eq!(column::<Int32Type>(&out.unwrap(), 1), want);
        let topn = Arc::new(TopN::new(b.schema(), &[SortKey::desc(0)], 0, 2).unwrap());
        let out = run(&deterministic(1), plan(input, &topn), &q).await;
        assert_eq!(
            column::<Int32Type>(&out.unwrap(), 1),
            want[..want.len().min(2)]
        );
        assert_released(&root);
    }

    let root = root();
    let q = query(&root, ExecConfig::default(), None);
    let topn = Arc::new(TopN::new(b.schema(), &[SortKey::asc(0)], 0, 3).unwrap());
    let weighted = small(Weights::Const(2));
    let e = run(&deterministic(1), plan(vec![weighted], &topn), &q).await;
    assert_eq!(
        e.unwrap_err().error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_released(&root);

    let class = |e: KrishivError| e.error_class();
    let s = b.schema();
    // A sort without keys gathers (see below); a TopN without keys is not
    // planned (a limit without an order is a `Limit`).
    assert_eq!(
        class(TopN::new(Arc::clone(&s), &[], 0, 3).unwrap_err()),
        ErrorClass::InternalError
    );
    let out_of_range = Sort::new(Arc::clone(&s), &[SortKey::asc(9)], None);
    assert_eq!(class(out_of_range.unwrap_err()), ErrorClass::InternalError);
    for (offset, limit) in [(1, TopN::MAX_ROWS), (usize::MAX, 2)] {
        let e = TopN::new(Arc::clone(&s), &[SortKey::asc(0)], offset, limit).unwrap_err();
        assert_eq!(class(e), ErrorClass::InternalError);
    }
    let nested = Arc::new(Schema::new(vec![
        Field::new_list("l", Field::new_list_field(DataType::Int32, true), true),
        Field::new("h", DataType::Float16, true),
        Field::new_dictionary("x", DataType::Int32, DataType::Float64, true),
        Field::new_dictionary("y", DataType::Int32, DataType::Utf8, true),
    ]));
    for c in 0..3 {
        let e = Sort::new(Arc::clone(&nested), &[SortKey::desc(c)], None).unwrap_err();
        assert_eq!(
            class(e),
            ErrorClass::KrishivUnsupportedFeature,
            "column {c}"
        );
    }
    assert!(Sort::new(nested, &[SortKey::asc(3)], None).is_ok());
}

/// Cancelling a spilling sort stops it, releases every byte and deletes
/// the spill files.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_a_spilling_sort_releases_everything() {
    let root = root();
    let q = query(&root, ExecConfig::default(), Some(2 << 20));
    let (manager, scope) = spill();
    let sort = Sort::new(table(1, 40)[0].schema(), &[SortKey::asc(0)], Some(scope)).unwrap();
    let sort = Arc::new(sort.with_max_fan_in(2));
    let runner = threaded();
    let mut stream = runner
        .execute(plan(table(400_000, 40), &sort), Arc::clone(&q))
        .await
        .unwrap();
    while sort.metrics().runs_spilled < 2 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    q.cancel();
    q.finished().await;
    while let Some(b) = stream.next().await {
        if let Err(e) = b {
            assert_eq!(e.error_class(), ErrorClass::OperationCanceled);
        }
    }
    assert_released(&root);
    assert_eq!(manager.used_bytes(), 0, "the spill files are gone");
}

/// A sort's in-memory result is spillable until read. Its reader feeds a
/// breaker that cannot spill; the memory that breaker needs comes from
/// writing the rest of the merge to disk, and the reader goes on from the
/// file. Before, the buffers stayed in memory until the merge ended and
/// these limits failed.
#[tokio::test(start_paused = true)]
async fn an_unread_result_spills_the_rest_of_its_merge() {
    let rows = 20_000;
    for limit in (57..=67).step_by(2).map(|i| i * 32 * KIB) {
        let root = root();
        let q = query(&root, ExecConfig::default(), Some(limit));
        let (manager, scope) = spill();
        let sort = Sort::new(table(1, 16)[0].schema(), &[SortKey::asc(0)], Some(scope)).unwrap();
        let sort = Arc::new(sort);
        let reader = krishiv_exec::testing::Buffer::new(false, None);
        let p = Arc::new(Break(plan(table(rows, 16), &sort), reader, true));
        let out = run(&deterministic(1), p, &q)
            .await
            .unwrap_or_else(|e| panic!("{limit} B: {e}"));
        check_sorted(rows, &out, true);
        let m = sort.metrics();
        // Sorted in memory; the one run is the rest of the merge.
        assert_eq!(m.final_fan_in, 1, "{limit} B: {m:?}");
        assert!(
            m.runs_spilled == 1 && m.spilled_bytes > 0,
            "{limit} B: {m:?}"
        );
        assert_released(&root);
        assert_eq!(manager.used_bytes(), 0);
    }
}

// ------------------------------------------------------------ hash buckets

/// Checks a bucketed sort of `rows` rows of [`table`] by `k, i` (buckets by
/// `k`), read by several drivers into one buffer: every row once, each
/// key's rows in `i` order (the drivers' batches interleave, so a key's
/// rows need not be adjacent here; the window tests check that).
fn check_grouped(rows: u64, out: &[Batch]) {
    let (k, i) = (column::<Int64Type>(out, 0), column::<Int64Type>(out, 1));
    assert_eq!(k.len() as u64, rows);
    let mut seen = vec![false; k.len()];
    let mut last = std::collections::HashMap::new();
    for r in 0..k.len() {
        let row = i[r] as usize;
        assert!(!std::mem::replace(&mut seen[row], true), "row {row} twice");
        assert_eq!(k[r], key(rows, row as u64), "row {row} kept its key");
        if let Some(prev) = last.insert(k[r], i[r]) {
            assert!(prev < i[r], "order within key {} at {r}", k[r]);
        }
    }
}

/// A sort by `k, i` bucketed by `k` over `spill` (with fan-in 2).
fn bucketed_sort(pad: usize, scope: Option<SpillScope>) -> Arc<Sort> {
    let schema = table(1, pad)[0].schema();
    let sort = Sort::new(schema, &[SortKey::asc(0), SortKey::asc(1)], scope)
        .unwrap()
        .with_max_fan_in(2)
        .with_hash_buckets(&[0])
        .unwrap();
    assert!(sort.bucketed());
    Arc::new(sort)
}

/// Bucketed sorts keep each key's rows together and in order, in memory
/// and spilled, read by several drivers; one bucket per driver count of
/// one, or with `krishiv.exec.sort.hashBuckets` off, keeps the total order.
#[tokio::test(start_paused = true)]
async fn hash_buckets_keep_each_key_together_and_in_order() {
    let rows = 30_000;
    for (limit, drivers, on) in [
        (None, 4, true),
        (Some(2 << 20), 4, true),
        (None, 1, true),
        (Some(2 << 20), 4, false),
    ] {
        let root = root();
        let config = ExecConfig {
            batch_size: 512,
            morsel_rows: 2048,
            sort_hash_buckets: on,
            ..ExecConfig::default()
        };
        let q = query(&root, config, limit);
        let (manager, scope) = spill();
        let sort = bucketed_sort(30, Some(scope));
        let reader = Arc::new(Break(
            plan_unordered(table(rows, 30), &sort),
            krishiv_exec::testing::Buffer::new(false, None),
            false,
        ));
        let out = run(&deterministic(drivers), reader, &q)
            .await
            .unwrap_or_else(|e| panic!("{limit:?} {drivers} {on}: {e}"));
        let m = sort.metrics();
        let want = if on && drivers > 1 { 16 } else { 1 };
        assert_eq!(m.buckets, want, "{limit:?} {drivers}: {m:?}");
        // `EXPLAIN ANALYZE` shows a bucketed sort's buckets.
        let sink = q
            .metrics()
            .operators
            .into_values()
            .find(|e| e.name == "Sort")
            .unwrap();
        assert_eq!(
            sink.extra("buckets"),
            Some(&krishiv_exec::Metric::Max(want)),
        );
        if want == 1 {
            check_sorted(rows, &out, true);
        } else {
            check_grouped(rows, &out);
        }
        assert_eq!(m.runs_spilled > 0, limit.is_some(), "{limit:?}: {m:?}");
        drop(out);
        assert_released(&root);
        assert_eq!(manager.used_bytes(), 0, "every spill file is gone");
    }
}

/// Readers of a spilled bucketed result merge buckets in parallel while
/// the pool has memory to spare, and fall back to fewer readers when it
/// has not: a non-spillable reader downstream holding nearly the whole
/// limit gets the memory one reader of an unbucketed sort leaves it.
#[tokio::test(start_paused = true)]
async fn bucket_readers_fit_the_memory_of_one() {
    let rows = 30_000;
    // (downstream buffer, its memory spillable): none, the output streams
    // out of the pool.
    for downstream in [None, Some(false)] {
        let root = root();
        let config = ExecConfig {
            batch_size: 512,
            morsel_rows: 2048,
            ..ExecConfig::default()
        };
        let q = query(&root, config, Some(2 << 20));
        let (manager, scope) = spill();
        let sort = bucketed_sort(30, Some(scope));
        let p = plan_unordered(table(rows, 30), &sort);
        let p: Arc<dyn ExecNode> = match downstream {
            None => p,
            Some(spillable) => Arc::new(Break(
                p,
                krishiv_exec::testing::Buffer::new(spillable, None),
                false,
            )),
        };
        let out = run(&deterministic(4), p, &q)
            .await
            .unwrap_or_else(|e| panic!("{downstream:?}: {e}"));
        check_grouped(rows, &out);
        let m = sort.metrics();
        assert!(m.runs_spilled > 0, "{m:?}");
        if downstream.is_none() {
            assert!(m.merge_readers > 1, "parallel with memory to spare: {m:?}");
        } else {
            assert!(m.merge_readers >= 1, "{m:?}");
        }
        drop(out);
        assert_released(&root);
        assert_eq!(manager.used_bytes(), 0);
    }
}

/// The same on threads: readers claiming, giving back and waiting race for
/// real, and every bucket is still read once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bucket_readers_race_on_threads() {
    let rows = 30_000;
    let runner = QueryRunner::new(
        TokioExecutor::new(&ExecutorConfig {
            compute_threads: 4,
            io_threads: 1,
            blocking_threads: 4,
        })
        .unwrap(),
    );
    for spillable in [None, Some(false), None, Some(false)] {
        let root = root();
        let config = ExecConfig {
            batch_size: 512,
            morsel_rows: 2048,
            ..ExecConfig::default()
        };
        let q = query(&root, config, Some(2 << 20));
        let (manager, scope) = spill();
        let sort = bucketed_sort(30, Some(scope));
        let p = plan_unordered(table(rows, 30), &sort);
        let p: Arc<dyn ExecNode> = match spillable {
            None => p,
            Some(s) => Arc::new(Break(p, krishiv_exec::testing::Buffer::new(s, None), false)),
        };
        let out = run(&runner, p, &q)
            .await
            .unwrap_or_else(|e| panic!("{spillable:?}: {e}"));
        check_grouped(rows, &out);
        assert!(sort.metrics().runs_spilled > 0);
        drop(out);
        assert_released(&root);
        assert_eq!(manager.used_bytes(), 0);
    }
}

/// Scan `input` into `breaker`, read by as many drivers as it allows.
fn plan_unordered<B: Breaker + 'static>(input: Vec<Batch>, breaker: &Arc<B>) -> Arc<dyn ExecNode> {
    let schema = Source::schema(&**breaker);
    let scan = Arc::new(Scan(Arc::new(MemoryScan::try_new(schema, input).unwrap())));
    Arc::new(Break(scan, Arc::clone(breaker), false))
}

/// An in-memory bucketed result spills the rest of every bucket's merge
/// when a reader downstream holds memory, and the readers continue from
/// the runs.
#[tokio::test(start_paused = true)]
async fn an_unread_bucketed_result_spills_each_bucket() {
    let rows = 20_000;
    let mut spilled = 0;
    for limit in (57..=73).step_by(4).map(|i| i * 32 * KIB) {
        let root = root();
        let config = ExecConfig {
            batch_size: 256,
            ..ExecConfig::default()
        };
        let q = query(&root, config, Some(limit));
        let (manager, scope) = spill();
        let sort = bucketed_sort(16, Some(scope));
        let reader = krishiv_exec::testing::Buffer::new(false, None);
        let p = Arc::new(Break(plan_unordered(table(rows, 16), &sort), reader, false));
        let out = run(&deterministic(4), p, &q)
            .await
            .unwrap_or_else(|e| panic!("{limit} B: {e}"));
        check_grouped(rows, &out);
        spilled += sort.metrics().runs_spilled;
        drop(out);
        assert_released(&root);
        assert_eq!(manager.used_bytes(), 0);
    }
    assert!(spilled > 0, "some limit spilled the result");
}

#[test]
fn hash_buckets_need_keys() {
    let schema = table(1, 4)[0].schema();
    let sort = Sort::new(schema, &[SortKey::asc(0)], None).unwrap();
    let err = sort.with_hash_buckets(&[]).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(err.to_string().contains("without keys"), "{err}");
}
