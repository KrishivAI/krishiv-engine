// SPDX-License-Identifier: Apache-2.0
//! Throughput of the stateless operators over a prebuilt in-memory table, on
//! one deterministic worker (simulation-friendly: no thread parking in the
//! instruction counts).
//!
//! - `operators/filter_1m_rows/{50_percent,all_rows}`: scan → filter → count.
//!   The predicate is a stand-in closure; the cost measured is the mask
//!   handling, Arrow's filter kernel and the driver. `all_rows` keeps every
//!   row, so the filter shares buffers instead of copying: its extra cost over
//!   `project_1m_rows/columns` is the mask itself.
//! - `operators/filter_1m_rows/selectivity/{1,100,900}_per_mille`: the same on
//!   `v`, keeping that share of the rows. The keep test hashes `v` (the row
//!   number), so the kept rows are spread at random through every batch, as
//!   the SQL cases it twins (`u < limit`, a random `u`) are: every batch is
//!   mixed (at least 90 % of the batches, checked at startup), never the
//!   all-true or all-false fast paths.
//! - `operators/filter_count_1m_rows/selectivity/{...}_per_mille`: the same
//!   keyed on the repeating column `w` (hashed too), so the mask is spread
//!   through every batch rather than a prefix of it.
//! - `operators/filter_string_1m_rows/predicate/{prefix,contains,equals}`: a
//!   `Utf8View` column (the engine's string type, after a row-number column)
//!   tested with the engine's compiled `LIKE` pattern (as `Like::eval` runs it) for a prefix and a contained
//!   substring, and for equality with one value.
//! - `operators/project_1m_rows/{columns,expr}`: scan → project → count.
//!   `columns` reorders two columns (zero-copy); `expr` adds a computed one.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::testing::{Break, Count, Pipe, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Filter, FnPredicate, MemoryScan, Operator, Predicate, Project,
    ProjectItem, QueryContext, QueryId, QueryRunner, Result, ScalarExpr, Source, TokioExecutor,
    WorkloadClass,
};
use krishiv_kernels::string::LikePattern;
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;

const BATCHES: usize = 256;
const BATCH_ROWS: usize = 4096;
const ROWS: u64 = (BATCHES * BATCH_ROWS) as u64;

/// A fixed scramble of `x` (SplitMix64's finaliser): a keep test of
/// `mix(x) % 1000 < per_mille` keeps that share of rows spread at random,
/// so every batch has a mixed mask and none takes the all-true or all-false
/// fast path.
fn mix(x: i64) -> u64 {
    let mut z = x.cast_unsigned().wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Whether row number `v` is kept at `per_mille` (column `v`, unique values).
fn keeps_v(v: i64, per_mille: i64) -> bool {
    ((mix(v) % 1000) as i64) < per_mille
}

/// Whether `w` (the repeating column, `0..BATCH_ROWS` in every batch) is
/// kept at `per_mille`: `w * 2531 mod 4096` is a permutation of the
/// column's values (2531 is odd), and the `per_mille` share of them below the
/// cut are kept, so every batch keeps exactly `BATCH_ROWS * per_mille / 1000`
/// rows, spread through it (a hash of so few distinct values would keep an
/// uncontrolled number).
fn keeps_w(w: i64, per_mille: i64) -> bool {
    let cut = BATCH_ROWS as i64 * per_mille / 1000;
    w.wrapping_mul(2531).rem_euclid(BATCH_ROWS as i64) < cut
}

/// Panics unless at least 90 % of the batches of `table`'s column `col` are
/// mixed (neither all kept nor all dropped) at `per_mille`, with `keeps` as the keep
/// test: a selectivity bench over a column that fails this measures the fast paths, not the
/// selectivity. (At 1 per mille a batch keeps about 4 rows, so a few random
/// batches keep none; a prefix mask over the row number, the first version
/// of these benches, had 1 mixed batch in 256 at 900 per mille.)
#[expect(
    clippy::panic,
    reason = "a broken fixture must fail the bench loudly, not measure the wrong thing"
)]
fn assert_mixed(table: &[Batch], col: usize, keeps: fn(i64, i64) -> bool, per_mille: i64) {
    let mixed = table
        .iter()
        .filter(|b| {
            let kept = b
                .data()
                .column(col)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .filter(|x| keeps(**x, per_mille))
                .count();
            kept != 0 && kept != BATCH_ROWS
        })
        .count();
    if mixed * 10 < table.len() * 9 {
        panic!(
            "only {mixed} of {} batches are mixed at {per_mille} per mille",
            table.len()
        );
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, false),
        Field::new("w", DataType::Int64, false),
    ]))
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn table() -> Vec<Batch> {
    (0..BATCHES)
        .map(|k| {
            let v =
                Int64Array::from_iter_values((0..BATCH_ROWS).map(|i| (k * BATCH_ROWS + i) as i64));
            let w = Int64Array::from_iter_values((0..BATCH_ROWS).map(|i| i as i64));
            Batch::new(
                RecordBatch::try_new(schema(), vec![Arc::new(v), Arc::new(w)]).expect("batch"),
            )
        })
        .collect()
}

fn string_schema() -> SchemaRef {
    // `Count` sums the first column, which must be an `Int64`.
    Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("s", DataType::Utf8View, false),
    ]))
}

/// `ROWS` strings `item-<n>`, `n` the row number mod 1000 (every value
/// repeats, 12 bytes at most).
#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn string_table() -> Vec<Batch> {
    (0..BATCHES)
        .map(|k| {
            let n = Int64Array::from_iter_values((0..BATCH_ROWS).map(|i| i as i64));
            let s = StringViewArray::from_iter_values(
                (0..BATCH_ROWS).map(|i| format!("item-{}", (k * BATCH_ROWS + i) % 1000)),
            );
            let cols: Vec<ArrayRef> = vec![Arc::new(n), Arc::new(s)];
            Batch::new(RecordBatch::try_new(string_schema(), cols).expect("batch"))
        })
        .collect()
}

#[derive(Debug)]
struct Double;

impl ScalarExpr for Double {
    fn data_type(&self) -> DataType {
        DataType::Int64
    }

    fn nullable(&self) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let v = batch.column(0).as_primitive::<Int64Type>();
        Ok(Arc::new(Int64Array::from_iter_values(
            v.values().iter().map(|x| x.wrapping_mul(2)),
        )))
    }
}

fn plan<O: Operator + 'static>(table: &[Batch], op: &Arc<O>) -> Arc<dyn ExecNode> {
    plan_of(schema(), table, op)
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn plan_of<O: Operator + 'static>(
    schema: SchemaRef,
    table: &[Batch],
    op: &Arc<O>,
) -> Arc<dyn ExecNode> {
    let scan: Arc<dyn Source> =
        Arc::new(MemoryScan::try_new(schema, table.to_vec()).expect("scan"));
    Arc::new(Break(
        Arc::new(Pipe(
            Arc::new(Scan(scan)),
            Arc::clone(op) as Arc<dyn Operator>,
        )),
        Count::new(),
        false,
    ))
}

#[expect(
    clippy::expect_used,
    reason = "a failing query must fail the bench loudly, not skip it"
)]
fn run(runner: &QueryRunner, caller: &tokio::runtime::Runtime, plan: Arc<dyn ExecNode>) -> usize {
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("bench", None),
    );
    caller.block_on(async {
        let mut stream = runner
            .execute(plan, Arc::clone(&query))
            .await
            .expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        query.finished().await;
        rows
    })
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn bench(c: &mut Criterion) {
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("caller runtime");
    let executor = {
        let _in_runtime = caller.enter();
        TokioExecutor::deterministic(1).expect("executor")
    };
    let runner = QueryRunner::new(executor);
    let table = table();

    let half = |b: &RecordBatch| -> Result<BooleanArray> {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .map(|v| v & 1 == 0)
            .collect())
    };
    let all = |b: &RecordBatch| -> Result<BooleanArray> {
        Ok(BooleanArray::from(vec![true; b.num_rows()]))
    };
    let mut g = c.benchmark_group("operators");
    g.throughput(Throughput::Elements(ROWS));
    g.bench_function("filter_1m_rows/50_percent", |b| {
        let f = Filter::new(schema(), Arc::new(FnPredicate(half)));
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &f))));
    });
    g.bench_function("filter_1m_rows/all_rows", |b| {
        let f = Filter::new(schema(), Arc::new(FnPredicate(all)));
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &f))));
    });
    // `v` runs 0..ROWS over the table; `w` repeats 0..BATCH_ROWS per batch.
    for per_mille in [1_i64, 100, 900] {
        assert_mixed(&table, 0, keeps_v, per_mille);
        assert_mixed(&table, 1, keeps_w, per_mille);
        let by_v = move |b: &RecordBatch| -> Result<BooleanArray> {
            Ok(b.column(0)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .map(|v| keeps_v(*v, per_mille))
                .collect())
        };
        g.bench_function(
            format!("filter_1m_rows/selectivity/{per_mille}_per_mille"),
            |b| {
                let f = Filter::new(schema(), Arc::new(FnPredicate(by_v)));
                b.iter(|| black_box(run(&runner, &caller, plan(&table, &f))));
            },
        );
        let by_w = move |b: &RecordBatch| -> Result<BooleanArray> {
            Ok(b.column(1)
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .map(|w| keeps_w(*w, per_mille))
                .collect())
        };
        g.bench_function(
            format!("filter_count_1m_rows/selectivity/{per_mille}_per_mille"),
            |b| {
                let f = Filter::new(schema(), Arc::new(FnPredicate(by_w)));
                b.iter(|| black_box(run(&runner, &caller, plan(&table, &f))));
            },
        );
    }
    let strings = string_table();
    // `Like::eval` in the executor compiles the pattern once and matches the
    // view column's values in place; the predicates do the same.
    let like = |pattern: &'static str| {
        move |b: &RecordBatch| -> Result<BooleanArray> {
            let compiled = LikePattern::compile(pattern, '\\', 0).map_err(|e| e.error().clone())?;
            Ok(b.column(1)
                .as_string_view()
                .iter()
                .map(|s| s.is_some_and(|s| compiled.matches(s)))
                .collect())
        }
    };
    let equals = |b: &RecordBatch| -> Result<BooleanArray> {
        Ok(b.column(1)
            .as_string_view()
            .iter()
            .map(|s| s == Some("item-123"))
            .collect())
    };
    for (name, pred) in [
        (
            "prefix",
            Arc::new(FnPredicate(like("item-12%"))) as Arc<dyn Predicate>,
        ),
        ("contains", Arc::new(FnPredicate(like("%-12%")))),
        ("equals", Arc::new(FnPredicate(equals))),
    ] {
        g.bench_function(format!("filter_string_1m_rows/predicate/{name}"), |b| {
            let f = Filter::new(string_schema(), Arc::clone(&pred));
            b.iter(|| {
                black_box(run(
                    &runner,
                    &caller,
                    plan_of(string_schema(), &strings, &f),
                ))
            });
        });
    }
    g.bench_function("project_1m_rows/columns", |b| {
        let p = Project::try_new(
            &schema(),
            vec![ProjectItem::column(1), ProjectItem::column(0)],
        )
        .expect("project");
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &p))));
    });
    g.bench_function("project_1m_rows/expr", |b| {
        let p = Project::try_new(
            &schema(),
            vec![
                ProjectItem::column(0),
                ProjectItem::expr("d", Arc::new(Double)),
            ],
        )
        .expect("project");
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &p))));
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
