// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// One driver whose table spills several times still picks `first`/`last`
/// in input order: spill runs merge in the order their rows arrived, on
/// both layouts (the row tables also through partial duplicates).
#[tokio::test]
async fn first_and_last_keep_input_order_through_spills() {
    use AggregateFunction::{First, Last};
    let (n, keys) = (60_000i64, 20_000i64);
    let (s, b) = grouped_input(n, keys);
    let (_m, scope) = spill_dir();
    let aggs = vec![
        agg(&First { ignore_nulls: true }, 1, &s),
        agg(&Last { ignore_nulls: true }, 1, &s),
        agg(
            &First {
                ignore_nulls: false,
            },
            1,
            &s,
        ),
    ];
    let mut sp = spec(&s, vec![0], aggs);
    sp.spill = Some(scope);
    // No seed: one driver claims the batches in order.
    let o = Opts {
        slots: 1,
        limit: 1 << 20,
        seed: None,
        ..Opts::default()
    };
    let (out, metrics) = exec_each(|| node(scan(&s, b.clone()), sp.clone()), o).await;
    // Both layouts spilled several runs.
    for m in &metrics {
        assert!(extra(m, "HashAggregate", "spill_runs") > 1);
    }
    let expected: Vec<String> = {
        let mut v: Vec<String> = (0..keys)
            .map(|k| {
                let valid: Vec<i64> = (k..n)
                    .step_by(usize::try_from(keys).unwrap())
                    .filter(|i| i % 7 != 0)
                    .collect();
                let first = if k % 7 == 0 {
                    "NULL".to_owned()
                } else {
                    k.to_string()
                };
                format!("{k}|{}|{}|{first}", valid[0], valid[valid.len() - 1])
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(out, expected);
    let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Fixed { limit: 300 }).unwrap();
    let on = exec_with(AggregateNode::new(scan(&s, b), agg), o, true)
        .await
        .unwrap();
    assert_eq!(rows(&on.0), expected);
    assert!(extra(&on.1, "HashAggregate", "spill_runs") > 1);
}

/// The row tables' final phase: several partition tasks in parallel over
/// tables with partial duplicates; when the merge cannot get memory (here
/// another consumer holds the pool), everything left goes to spill runs
/// and the disk phase returns the same rows, with nothing leaked.
#[tokio::test]
async fn row_final_phase_parallel_and_spill_fallback() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    for hog in [false, true] {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], grouped_aggs(&s));
        sp.spill = Some(scope);
        let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Fixed { limit: 5_000 }).unwrap();
        let root = pool(1 << 30);
        let config = ExecConfig {
            agg_row_layout: true,
            ..ExecConfig::default()
        };
        let limit = 256 << 20;
        let q = QueryContext::new(
            QueryId(4),
            WorkloadClass::Batch,
            config,
            root.child("q", Some(limit)),
        );
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(4).unwrap();
        let ctx = |id, name| {
            let e = Arc::clone(&executor);
            TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
        };
        let sink_ctx = ctx(0, "agg");
        let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
        for part in b.chunks(70) {
            let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
            for cols in part {
                let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
                assert!(matches!(
                    agg.sink(batch, &mut l, &g).unwrap(),
                    crate::SinkStatus::NeedMoreInput
                ));
            }
            agg.combine(l, &g).unwrap();
        }
        // An unspillable consumer takes all but about 1 MiB: tasks start,
        // stall partway, and cannot go on.
        let hog_ctx = ctx(1, "hog");
        let mut held = hog_ctx.consumer().reservation();
        if hog {
            let free = limit - root.reserved();
            held.try_grow(free - (1 << 20)).unwrap();
        }
        agg.finalize(&g, &sink_ctx).await.unwrap();
        drop(held);
        let src = ctx(2, "src").with_dependency(Arc::clone(&g));
        let sg = Source::global_state(&*agg, &src).unwrap();
        let parallel = agg.max_parallelism(&sg);
        let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
        let mut out = Vec::new();
        while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
            out.push(b);
        }
        assert_eq!(rows(&out), expected, "hog {hog}");
        if hog {
            assert!(agg.metrics().spill_runs > 0);
            assert!(parallel > 1, "{parallel}");
        } else {
            assert_eq!(agg.metrics().spill_runs, 0);
            assert!(parallel > 1, "{parallel}");
        }
        assert_eq!(agg.metrics().groups, 90_000);
        drop((sl, sg, src, g, sink_ctx));
        assert_eq!(root.reserved(), 0);
    }
}

/// Every key type with a row encoding through the row tables (fixed
/// indexes that abandon, several drivers, up to three keys at a time)
/// returns the columnar tables' rows, NULL keys included: fixed widths, and
/// strings and binaries of every carrier, empty, inline, 12 and 13 bytes
/// and long (heap), binaries that are not UTF-8.
#[tokio::test]
async fn row_layout_key_types_equal_default() {
    use arrow_array::{
        BooleanArray, Decimal64Array, Float32Array, TimestampMicrosecondArray, UInt16Array,
    };
    let n = 5_000usize;
    let i =
        |m: usize| (0..n).map(move |r| (r % 11 != 0).then(|| i64::try_from(r * 7919 % m).unwrap()));
    let mut cols: Vec<(&str, ArrayRef)> = vec![
        (
            "d128",
            Arc::new(
                Decimal128Array::from_iter(i(300).map(|v| v.map(|v| i128::from(v) - 150)))
                    .with_precision_and_scale(20, 3)
                    .unwrap(),
            ),
        ),
        (
            "d64",
            Arc::new(
                Decimal64Array::from_iter(i(70))
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ),
        (
            "date",
            Arc::new(Date32Array::from_iter(
                i(400).map(|v| v.map(|v| i32::try_from(v - 200).unwrap())),
            )),
        ),
        (
            "ts",
            Arc::new(
                TimestampMicrosecondArray::from_iter(i(90).map(|v| v.map(|v| v * 1_000)))
                    .with_timezone("UTC"),
            ),
        ),
        (
            "f32",
            Arc::new(Float32Array::from_iter(
                i(9).map(|v| v.map(|v| f32::from(i16::try_from(v).unwrap()) - 4.0)),
            )),
        ),
        (
            "u16",
            Arc::new(UInt16Array::from_iter(
                i(50).map(|v| v.map(|v| u16::try_from(v).unwrap())),
            )),
        ),
        (
            "b",
            Arc::new(BooleanArray::from_iter(i(2).map(|v| v.map(|v| v == 1)))),
        ),
    ];
    // Strings and binaries of every view shape: empty, inline (up to 12
    // bytes), exactly 12 and 13 bytes, and long; binaries not UTF-8.
    let text = |m: usize| {
        i(m).map(|v| {
            v.map(|v| match v % 5 {
                0 => String::new(),
                1 => format!("{v}"),
                2 => format!("{v:0>12}"),
                3 => format!("{v:0>13}"),
                _ => format!("a-long-string-key-{v:0>30}"),
            })
        })
    };
    let bytes = |m: usize| {
        text(m).map(|v| {
            v.map(|v| {
                let mut b = vec![0xFF_u8, 0x00];
                b.extend(v.into_bytes());
                b
            })
        })
    };
    let strs = |m| text(m).collect::<Vec<_>>();
    let bins = |m| bytes(m).collect::<Vec<_>>();
    cols.extend([
        (
            "utf8",
            Arc::new(arrow_array::StringArray::from(strs(200))) as ArrayRef,
        ),
        (
            "large_utf8",
            Arc::new(arrow_array::LargeStringArray::from(strs(60))),
        ),
        ("utf8_view", Arc::new(StringViewArray::from(strs(500)))),
        (
            "binary",
            Arc::new(arrow_array::BinaryArray::from_iter(bins(80))),
        ),
        (
            "binary_view",
            Arc::new(arrow_array::BinaryViewArray::from_iter(bins(300))),
        ),
        (
            "large_binary",
            Arc::new(arrow_array::LargeBinaryArray::from_iter(bins(40))),
        ),
        ("x", Arc::new(Int64Array::from_iter(i(1_000)))),
    ]);
    let s: SchemaRef = Arc::new(Schema::new(
        cols.iter()
            .map(|(n, c)| Field::new(*n, c.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, c)| c).collect();
    let batches: Vec<Vec<ArrayRef>> = (0..n)
        .step_by(700)
        .map(|at| {
            arrays
                .iter()
                .map(|a| a.slice(at, 700.min(n - at)))
                .collect()
        })
        .collect();
    let x = s.fields().len() - 1;
    let aggs = || {
        vec![
            AggregateExpr::count_star("c"),
            agg(&Sum, x, &s),
            agg(&Min, 0, &s),
            agg(&Max, 2, &s),
        ]
    };
    for keys in [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        vec![0, 3],
        vec![2, 6],
        vec![7],
        vec![8],
        vec![9],
        vec![10],
        vec![11],
        vec![12],
        vec![9, 0],
        vec![7, 6, 11],
        vec![8, 12, 2],
    ] {
        let sp = spec(&s, keys.clone(), aggs());
        let o = Opts {
            both_layouts: false,
            ..Opts::default()
        };
        let want = rows(
            &exec(node(scan(&s, batches.clone()), sp.clone()), o)
                .await
                .unwrap()
                .0,
        );
        assert!(!want.is_empty());
        for mode in [
            rowtable::Mode::Grow,
            rowtable::Mode::Fixed { limit: 7 },
            rowtable::Mode::Append,
        ] {
            let agg = HashAggregate::with_row_mode(sp.clone(), mode).unwrap();
            let (got, m) = exec_with(AggregateNode::new(scan(&s, batches.clone()), agg), o, true)
                .await
                .unwrap();
            assert_eq!(rows(&got), want, "keys {keys:?} {mode:?}");
            // The row tables ran (no fallback to the columnar tables).
            let layout = m
                .operators
                .values()
                .flat_map(|e| e.extra.iter())
                .any(|(n, v)| {
                    *n == "table_layout" && *v == crate::metrics::Metric::Text("rows".to_owned())
                });
            assert!(layout, "keys {keys:?}: not on the row tables");
        }
    }
}

/// A spec whose key field is non-nullable over batches that hold NULL keys
/// (their own schema is nullable): both layouts fail the same way, when
/// the output batch is checked against the declared schema (the row
/// tables give every key a validity bit, so they group the NULLs first,
/// as the columnar tables do).
#[tokio::test]
async fn row_layout_nullable_batches_under_a_non_nullable_spec() {
    let nullable = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let strict: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("x", DataType::Int64, true),
    ]));
    let b = vec![
        vec![
            i64s(&[Some(1), None, Some(1), None]),
            i64s(&[Some(1), Some(2), Some(3), Some(4)]),
        ],
        vec![i64s(&[None, Some(2)]), i64s(&[Some(5), Some(6)])],
    ];
    let plan = node(
        scan(&nullable, b),
        spec(
            &strict,
            vec![0],
            vec![AggregateExpr::count_star("c"), agg(&Sum, 1, &strict)],
        ),
    );
    // `exec` runs both layouts and requires the same error.
    let e = exec(plan, Opts::default()).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(
        e.to_string()
            .contains("declared as non-nullable but contains null"),
        "{e}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: tight_memory_cases(),
        rng_seed: proptest::test_runner::RngSeed::Fixed(20_261_020),
        ..ProptestConfig::default()
    })]
    /// At 768 KiB the row tables finish wherever the columnar tables do,
    /// with partial duplicates from tiny fixed indexes (one row per input
    /// row at `limit` 1): the final phase frees each driver partition once
    /// merged and writes finished partitions to the output file instead of
    /// spilling them, and a batch's scatter uses the idle spill headroom.
    /// This fixed seed found the final-phase gap (2x memory) and then the
    /// scatter's transient bytes.
    #[test]
    fn row_layout_tight_memory(
        data in proptest::collection::vec(
            (
                proptest::option::of(any::<i32>()),
                proptest::option::of(any::<u8>()),
                proptest::option::of(any::<bool>()),
                proptest::option::of(-1_000_000i64..1_000_000),
                proptest::option::of(-10_000_000i64..10_000_000),
                proptest::option::of(any::<u16>()),
            ),
            0..4000,
        ),
        keys in 1i32..3000,
        nkeys in 1usize..=6,
        batch in 1usize..900,
        slots in 1usize..4,
        mode in 0u8..3,
        limit in 1usize..64,
        two in any::<bool>(),
        // As in `row_layout_equals_default`, so its saved seeds (one of them
        // is this gap, found at 768 KiB) replay here first.
        _tight in any::<bool>(),
    ) {
        row_case(&data, keys, nkeys, batch, slots, mode, limit, two, Some(768 << 10))?;
    }
}

/// A spill request during the row tables' parallel merge writes the
/// finished tasks' groups to the output file (final, never merged again)
/// and the merge goes on; the rest of the result follows them to the file.
/// The rows are unchanged and nothing leaks.
#[tokio::test]
async fn row_merge_spills_on_request() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Fixed { limit: 5_000 }).unwrap();
    let root = pool(1 << 30);
    let config = ExecConfig {
        agg_row_layout: true,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(5),
        WorkloadClass::Batch,
        config,
        root.child("q", None),
    );
    // One worker: the merge runs one task at a time.
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
    for part in b.chunks(70) {
        let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
            agg.sink(batch, &mut l, &g).unwrap();
        }
        agg.combine(l, &g).unwrap();
    }
    let request = async {
        while !super::super::row_merge_running(&g) {
            tokio::task::yield_now().await;
        }
        // Asked, not freed yet: the merge frees between its tasks.
        assert_eq!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap(), 0);
    };
    let (done, ()) = tokio::join!(agg.finalize(&g, &sink_ctx), request);
    done.unwrap();
    let m = agg.metrics();
    assert_eq!((m.spill_runs, m.groups), (0, 90_000));
    assert!(m.spill_files > 0);
    assert!(sink_metric(&agg, "row_final_flushes") > 0);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 0);
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        out.push(b);
    }
    assert_eq!(rows(&out), expected);
    drop((sl, sg, src, g, sink_ctx));
    assert_eq!(root.reserved(), 0);
}

/// Both table layouts count their index reallocations, and, in a detailed
/// profile only, their probe lengths.
#[tokio::test]
async fn resizes_and_probe_lengths_on_both_layouts() {
    let (s, b) = grouped_input(200_000, 90_000);
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    for rows in [false, true] {
        let o = Opts {
            both_layouts: false,
            detailed: true,
            ..Opts::default()
        };
        let (_, m) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, rows)
            .await
            .unwrap();
        assert!(
            extra(&m, "HashAggregate", "hash_resizes") > 0,
            "rows {rows}"
        );
        let lookups = extra(&m, "HashAggregate", "probe_lookups");
        assert!(lookups > 0, "rows {rows}");
        assert!(
            extra(&m, "HashAggregate", "probe_key_compares") > 0,
            "rows {rows}"
        );
        assert!(
            extra(&m, "HashAggregate", "max_probe_length") >= 1,
            "rows {rows}"
        );
        let o = Opts {
            both_layouts: false,
            ..Opts::default()
        };
        let (_, m) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, rows)
            .await
            .unwrap();
        assert!(
            extra(&m, "HashAggregate", "hash_resizes") > 0,
            "rows {rows}"
        );
        assert_eq!(
            extra(&m, "HashAggregate", "probe_lookups"),
            0,
            "rows {rows}"
        );
    }
}

/// `EXPLAIN ANALYZE` counters of the row tables: abandons of full fixed
/// indexes, final-phase tasks, the groups and the layout; the default
/// tables report none of the row counters.
#[tokio::test]
async fn row_layout_metrics() {
    let (s, b) = grouped_input(200_000, 90_000);
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let agg =
        HashAggregate::with_row_mode(sp.clone(), rowtable::Mode::Fixed { limit: 5_000 }).unwrap();
    let (_, m) = exec_with(AggregateNode::new(scan(&s, b.clone()), agg), o, true)
        .await
        .unwrap();
    assert!(extra(&m, "HashAggregate", "row_abandons") > 0);
    assert!(extra(&m, "HashAggregate", "row_final_tasks") > 1);
    assert_eq!(extra(&m, "HashAggregate", "row_final_spills"), 0);
    assert_eq!(extra(&m, "HashAggregate", "row_final_flushes"), 0);
    assert_eq!(extra(&m, "HashAggregate", "groups"), 90_000);
    assert_eq!(extra(&m, "HashAggregate", "hash_entries"), 90_000);
    let layout = m
        .operators
        .values()
        .flat_map(|e| e.extra.iter())
        .any(|(n, v)| {
            *n == "table_layout" && *v == crate::metrics::Metric::Text("rows".to_owned())
        });
    assert!(layout);
    let (_, m) = exec_with(node(scan(&s, b), sp), o, false).await.unwrap();
    let names: Vec<_> = m
        .operators
        .values()
        .flat_map(|e| e.extra.iter())
        .map(|(n, _)| n.to_string())
        .collect();
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("row_") || n == "table_layout"),
        "{names:?}"
    );
}

/// When the final phase runs short of memory it first writes the finished
/// tasks' groups to the output file; when it runs short again with no
/// task finished, the rest spills and the disk phase appends to that same
/// file. Every group comes out once, with its values.
#[tokio::test]
async fn row_merge_flushes_then_spills() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    // After the first task, no task gets memory: flush it, then spill.
    let agg =
        HashAggregate::with_final_stall(sp, rowtable::Mode::Fixed { limit: 5_000 }, 1).unwrap();
    let got = by_hand(&agg, &s, &b, 3).await;
    assert_eq!(got.rows.len(), 90_000);
    assert_eq!(got.rows, expected);
    assert!(sink_metric(&agg, "row_final_tasks") > 1);
    assert_eq!(sink_metric(&agg, "row_final_flushes"), 1);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 1);
    let m = agg.metrics();
    assert!(m.spill_runs > 0);
    assert_eq!(m.groups, 90_000);
}

/// The final phase frees each driver partition once merged: with every key
/// distinct (the merged tables are as large as the inputs), the peak stays
/// well below holding the inputs and the merged tables together.
#[tokio::test]
async fn row_merge_frees_inputs_as_it_merges() {
    let (s, b) = grouped_input(120_000, 120_000);
    let expected = run_rows(single(&s, &b)).await;
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Append).unwrap();
    let got = by_hand(&agg, &s, &b, 3).await;
    assert_eq!(got.rows, expected);
    assert!(sink_metric(&agg, "row_final_tasks") > 2);
    assert!(
        got.peak < got.combined * 3 / 2,
        "peak {} vs {} combined",
        got.peak,
        got.combined
    );
}

/// An in-memory result is spillable until read. Its reader feeds a
/// breaker that cannot spill; the memory that breaker needs comes from
/// spilling the result's unread chunks. Before, the result held its memory
/// unspillable until the query ended: a limit at which the aggregate fits
/// failed or passed depending on how much of the result had to coexist
/// with the reader's state (1.8 MB failed while 1.6 MB, where the aggregate
/// spilled while building, passed). Now every limit in the range completes,
/// with each layout and with one or several readers.
#[tokio::test]
async fn an_unread_result_spills_for_its_reader() {
    let (_d, scope) = spill_dir();
    let groups = 20_000;
    let free = Opts {
        slots: 1,
        ..Opts::default()
    };
    let (want, _) = exec_with(unread_plan(groups, None), free, false)
        .await
        .unwrap();
    let want = rows(&want);
    assert_eq!(want.len(), 20_000);
    for slots in [1, 3] {
        for limit in [10, 12, 14, 16, 17, 18, 20].map(|tenth| tenth * 100_000) {
            for layout in [false, true] {
                let o = Opts {
                    slots,
                    limit,
                    ..Opts::default()
                };
                let plan = unread_plan(groups, Some(scope.clone()));
                let r = exec_with(plan, o, layout).await;
                let (got, m) = r.unwrap_or_else(|e| panic!("{slots} slots, {limit} B: {e}"));
                assert_eq!(rows(&got), want, "{slots} slots, {limit} B");
                if slots == 1 && limit == 1_700_000 && !layout {
                    // The aggregate finished in memory; only its result
                    // was written to disk, by its own consumer.
                    assert_eq!(extra(&m, "HashAggregate", "spill_runs"), 0);
                    let written: u64 = m
                        .operators
                        .values()
                        .filter(|e| e.name == "HashAggregate")
                        .map(|e| e.metrics.spill_write_bytes)
                        .sum();
                    assert!(written > 0, "{m:?}");
                }
            }
        }
    }
}

/// A query cancelled while its aggregate's result is being spilled ends
/// only once the spill has let go of the result's memory and its files.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_during_a_result_spill_releases_everything() {
    let (spill, scope) = spill_dir();
    let gate = crate::testing::BlockingGate::new();
    let plan = unread_plan_with(20_000, Some(scope), Some(Arc::clone(&gate)));
    // Threads of their own: the spill blocks at the gate on the blocking
    // pool while the query is cancelled.
    let executor = TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 1,
        io_threads: 1,
        blocking_threads: 1,
        io_blocking_threads: 2,
    })
    .unwrap();
    let runner = QueryRunner::new(executor);
    let root = pool(1_700_000);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            seed: Some(7),
            agg_row_layout: false,
            ..ExecConfig::default()
        },
        root.child("q", Some(1_700_000)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let drained = tokio::spawn(async move {
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.map_or(0, |b| b.num_rows());
        }
        rows
    });
    while !gate.reached() {
        tokio::task::yield_now().await;
    }
    q.cancel();
    // What the pool holds the moment the query reports it has ended.
    let (fq, froot) = (Arc::clone(&q), Arc::clone(&root));
    let at_end = tokio::spawn(async move {
        fq.finished().await;
        froot.reserved()
    });
    // The gate stays shut while the query winds down: the query must not
    // end while the spill job still holds the result's memory. (The wait
    // is bounded by count; the query ends here only if it does not wait
    // for the spill.)
    for _ in 0..200 {
        if at_end.is_finished() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    gate.open();
    // Nothing the spill held outlives the query.
    assert_eq!(at_end.await.unwrap(), 0, "memory left when the query ended");
    assert_eq!(root.reserved(), 0, "memory left after the query");
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
    assert!(drained.await.unwrap() < 20_000);
    assert_eq!(
        q.error().map(|e| e.error_class()),
        Some(ErrorClass::OperationCanceled)
    );
}

/// A spill of an aggregate's result that fails (its spill directory is
/// full) fails the query with that error; nothing is left behind.
#[tokio::test]
async fn a_failed_result_spill_fails_the_query() {
    let (spill, scope) = spill_dir_capped(4096);
    let o = Opts {
        slots: 1,
        limit: 1_700_000,
        ..Opts::default()
    };
    let r = exec_with(unread_plan(20_000, Some(scope)), o, false).await;
    let e = r
        .err()
        .unwrap_or_else(|| unreachable!("the result could not spill"));
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillCapExceeded, "{e}");
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
}

/// A thread-local table widens to its radix partitions once large, and the
/// final phase widens the tables that stayed narrow before it merges, so
/// a driver with many groups and one with few merge to the same result.
/// When the final phase cannot reserve that widening, every driver table
/// spills and the disk phase merges them.
#[tokio::test]
async fn narrow_and_wide_tables_merge() {
    let (s, mut b) = grouped_input(1_000, 1_000);
    let (_, small) = grouped_input(40, 10);
    b.extend(small);
    let expected = run_rows(single(&s, &b)).await;
    let mode = rowtable::Mode::Fixed { limit: 5_000 };
    // One driver per batch: 1,000 groups (widens at 64) and 10 (narrow).
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let agg = HashAggregate::with_row_mode(sp, mode).unwrap();
    let got = by_hand(&agg, &s, &b, 2).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 0);
    // The widening of the narrow table is refused: spill and merge on disk.
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg = HashAggregate::with_final_stall(sp, mode, 0).unwrap();
    let got = by_hand(&agg, &s, &b, 2).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 1);
    assert!(agg.metrics().spill_runs >= 2);
}

/// Without a spill directory, a narrow table that cannot reserve its
/// widening stays narrow (waiting for that memory could wait forever: at
/// a tight limit nothing else frees it), and one task with one partition
/// merges every partition of every table instead of one task per range.
#[tokio::test]
async fn a_refused_widening_merges_in_one_task() {
    let (s, mut b) = grouped_input(80_000, 70_000);
    let (_, small) = grouped_input(40, 10);
    b.extend(small);
    let expected = run_rows(single(&s, &b)).await;
    // A fixed index (radix partitions at one thread too) that never fills.
    let mode = rowtable::Mode::Fixed { limit: 100_000 };
    // Widened: the 70,000 groups merge in several tasks.
    let agg = HashAggregate::with_row_mode(spec(&s, vec![0], grouped_aggs(&s)), mode).unwrap();
    let got = by_hand(&agg, &s, &b, 81).await;
    assert_eq!(got.rows, expected);
    assert!(sink_metric(&agg, "row_final_tasks") > 1);
    // Refused (and no spill directory): one task. (A stall after
    // `usize::MAX` tasks refuses only the widening.)
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let agg = HashAggregate::with_final_stall(sp, mode, usize::MAX).unwrap();
    let got = by_hand(&agg, &s, &b, 81).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_tasks"), 1);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 0);
}

/// Grouped and global moment aggregates, on both table layouts, against a
/// single-pass reference.
#[tokio::test]
async fn moments_grouped_and_global() {
    let (s, b) = moment_input(8_000, 13);
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let grouped = || node(scan(&s, b.clone()), spec(&s, vec![0], moment_aggs()));
    check_both_layouts(grouped, o, true, &moment_oracle(&b, 13, true, false)).await;
    let global = || node(scan(&s, b.clone()), spec(&s, vec![], moment_aggs()));
    check_both_layouts(global, o, false, &moment_oracle(&b, 13, false, false)).await;
    // A global aggregate of no rows is one row of NULLs; a grouped one has none.
    let empty = |keys: Vec<usize>| node(scan(&s, vec![]), spec(&s, keys, moment_aggs()));
    let (out, _) = exec(empty(vec![]), o).await.unwrap();
    assert_eq!(rows(&out), ["NULL|NULL|NULL|NULL|NULL|NULL|NULL|NULL|NULL"]);
    assert!(rows(&exec(empty(vec![0]), o).await.unwrap().0).is_empty());
}

/// `spark.sql.legacy.statisticalAggregate` reaches the accumulators of
/// both layouts and of the final phase of a two-phase aggregation: groups
/// of one row give NULL, or NaN with the flag.
#[tokio::test]
async fn moments_legacy_flag_reaches_every_accumulator() {
    // Every group has one row, so the sample statistics divide by zero.
    let (s, b) = moment_input(40, 40);
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    for legacy in [false, true] {
        let mut sp = spec(&s, vec![0], moment_aggs());
        sp.legacy_statistical = legacy;
        let want = moment_oracle(&b, 40, true, legacy);
        // The oracle itself shows the flag: some groups' cells are NaN or NULL.
        let nan_cells = want
            .values()
            .flatten()
            .filter(|v| v.is_some_and(f64::is_nan))
            .count();
        assert_eq!(nan_cells > 0, legacy);
        let single = || node(scan(&s, b.clone()), sp.clone());
        check_both_layouts(single, o, true, &want).await;
        let phased = || {
            two_phase_of(
                scan(&s, b.clone()),
                sp.clone(),
                BypassPolicy::default(),
                None,
            )
            .0
        };
        check_both_layouts(phased, o, true, &want).await;
    }
}

/// `corr` of several rows over a constant column: `DIVIDE_BY_ZERO` with ANSI
/// and NULL without, on both layouts and across a partial and a final
/// phase, whatever the legacy flag says; the other groups are unaffected
/// when none is constant.
#[tokio::test]
async fn corr_of_a_constant_column_follows_ansi() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Float64),
        ("y", DataType::Float64),
    ]);
    let batch = |k: &[i64], x: &[f64], y: &[f64]| {
        let some = |v: &[f64]| f64s(v.iter().map(|&v| Some(v)).collect());
        let k: Vec<_> = k.iter().map(|&k| Some(k)).collect();
        vec![i64s(&k), some(x), some(y)]
    };
    let corr = || vec![moment_aggs().remove(6)];
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let varying = batch(
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 1.0, 2.0, 4.0],
        &[3.0, 1.0, 2.0, 5.0, 6.0],
    );
    let constant = batch(
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 7.0, 7.0, 7.0],
        &[3.0, 1.0, 2.0, 5.0, 6.0],
    );
    for legacy in [false, true] {
        for ansi in [false, true] {
            let mut sp = spec(&s, vec![0], corr());
            sp.ansi = ansi;
            sp.legacy_statistical = legacy;
            let plans: [Plan<'_>; 2] = [
                Box::new(|b| node(scan(&s, b.to_vec()), sp.clone())),
                Box::new(|b| {
                    two_phase_of(
                        scan(&s, b.to_vec()),
                        sp.clone(),
                        BypassPolicy::default(),
                        None,
                    )
                    .0
                }),
            ];
            for plan in &plans {
                for rows_layout in [false, true] {
                    let ok = exec_with(plan(std::slice::from_ref(&varying)), o, rows_layout)
                        .await
                        .unwrap()
                        .0;
                    assert!(moment_rows(&ok, true).values().all(|v| v[0].is_some()));
                    let got =
                        exec_with(plan(std::slice::from_ref(&constant)), o, rows_layout).await;
                    if ansi {
                        let e = got.unwrap_err();
                        assert_eq!(e.error_class(), ErrorClass::DivideByZero, "{e}");
                    } else {
                        let m = moment_rows(&got.unwrap().0, true);
                        assert_eq!(m["2"], [None], "legacy={legacy} layout={rows_layout}");
                        assert!(m["1"][0].is_some());
                    }
                }
            }
        }
    }
}

/// Partial → final over partial states equals one phase, also when the
/// partial phase flushes and bypasses.
#[tokio::test]
async fn moments_two_phase_equals_single_phase() {
    let (s, b) = moment_input(60_000, 700);
    let want = moment_oracle(&b, 700, true, false);
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let tiny = BypassPolicy {
        max_table_bytes: 64 << 10,
        min_rows: u64::MAX,
        ..BypassPolicy::default()
    };
    let policies = [BypassPolicy::default(), tiny];
    for policy in policies {
        let make = || {
            two_phase_of(
                scan(&s, b.clone()),
                spec(&s, vec![0], moment_aggs()),
                policy,
                None,
            )
            .0
        };
        check_both_layouts(make, o, true, &want).await;
    }
    // Unique keys: the partial phase passes rows through, one state per row.
    let (s, b) = moment_input(40_000, 1_000_000);
    let want = moment_oracle(&b, 40_000, true, false);
    let bypass = BypassPolicy {
        min_rows: 10_000,
        ..BypassPolicy::default()
    };
    let (plan, partial) = two_phase_of(
        scan(&s, b.clone()),
        spec(&s, vec![0], moment_aggs()),
        bypass,
        None,
    );
    let (out, _) = exec_with(plan, o, false).await.unwrap();
    assert!(partial.bypassed_drivers() > 0);
    assert_moments_close(&moment_rows(&out, true), &want, "bypassed");
}

/// Under a small budget the moment states spill with the others, merge from
/// the runs and give the in-memory result.
#[tokio::test]
async fn moments_spill_and_merge() {
    let (s, b) = moment_input(30_000, 25_000);
    // Three of the aggregates (one of each state width): the budget below
    // holds about a third of their groups.
    let aggs: Vec<AggregateExpr> = [0, 4, 6]
        .into_iter()
        .map(|i| moment_aggs().remove(i))
        .collect();
    let keep = |m: Moments| -> Moments {
        m.into_iter()
            .map(|(k, v)| (k, [0, 4, 6].iter().map(|&i| v[i]).collect()))
            .collect()
    };
    let want = keep(moment_oracle(&b, 25_000, true, false));
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], aggs);
    sp.spill = Some(scope.clone());
    let o = Opts {
        limit: 2 << 20,
        both_layouts: false,
        ..Opts::default()
    };
    for rows_layout in [false, true] {
        let (out, m) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, rows_layout)
            .await
            .unwrap();
        assert!(
            extra(&m, "HashAggregate", "spill_runs") > 0,
            "rows {rows_layout}"
        );
        assert_eq!(extra(&m, "HashAggregate", "groups"), 25_000);
        assert_moments_close(&moment_rows(&out, true), &want, "spilled");
    }
    let (plan, _) = two_phase_of(
        scan(&s, b.clone()),
        sp,
        BypassPolicy::default(),
        Some(scope),
    );
    let (out, _) = exec_with(plan, o, false).await.unwrap();
    assert_moments_close(&moment_rows(&out, true), &want, "spilled two-phase");
}

/// `DISTINCT` pairs: `corr(DISTINCT x, y)` over the distinct non-NULL
/// pairs, next to a plain `count(*)`.
#[tokio::test]
async fn moments_over_distinct_arguments() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Float64),
        ("y", DataType::Float64),
    ]);
    let k = i64s(&[Some(1); 6]);
    let x = f64s(vec![
        Some(1.0),
        Some(1.0),
        Some(2.0),
        Some(3.0),
        None,
        Some(3.0),
    ]);
    let y = f64s(vec![
        Some(2.0),
        Some(2.0),
        Some(4.5),
        Some(5.0),
        Some(1.0),
        Some(5.0),
    ]);
    let input = || {
        scan(
            &s,
            vec![vec![Arc::clone(&k), Arc::clone(&x), Arc::clone(&y)]],
        )
    };
    let mut corr = moment_aggs().remove(6);
    assert_eq!(corr.function, AggregateFunction::Corr);
    corr.name = "c".to_owned();
    let flags = [true, false];
    let plan = distinct_plan(
        input(),
        spec(
            &s,
            vec![],
            vec![corr.clone(), AggregateExpr::count_star("n")],
        ),
        &flags,
    )
    .unwrap();
    // The distinct pairs (1,2), (2,4.5), (3,5), with a correlation of its own.
    let pairs = vec![Some(1.0), Some(2.0), Some(3.0)];
    let want = moment_oracle(
        &[vec![
            i64s(&[Some(1); 3]),
            f64s(pairs),
            f64s(vec![Some(2.0), Some(4.5), Some(5.0)]),
        ]],
        1,
        false,
        false,
    );
    let got = moment_rows(&exec(plan, Opts::default()).await.unwrap().0, false);
    let corr_at = |m: &Moments, i: usize| m[""][i];
    let want_corr = corr_at(&want, 6).unwrap();
    assert!((got[""][0].unwrap() - want_corr).abs() < 1e-12, "{got:?}");
    assert_eq!(got[""][1], Some(6.0));
}

/// Batches of 4,096 rows take the batch-local key map (`RowTable::probe_local`, E2 step 2): every
/// index mode, one and three drivers, one and two phases, a tiny fixed index that clears between rows
/// of one key; fixed-width and inline string keys, NULL keys, and 1,000 keys (past `LOCAL_MAX`: the
/// map gives up and backs off). The rows equal the columnar tables'.
#[test]
fn few_keys_per_batch_take_the_local_map_and_equal_default() {
    // (keys, every `nulls`th key NULL, strings inline (else 4 in 7 long)).
    for (keys, nulls, inline) in [(10, 0, false), (10, 11, true), (1000, 0, true)] {
        let data: Vec<RowCase> = (0..8192_i32)
            .map(|i| {
                let x = i64::from(i.wrapping_mul(7919) % 1000);
                let null = nulls > 0 && i % nulls == 0;
                let s = if inline { 3 * (i % 7) } else { i % 7 };
                (
                    (!null).then(|| i.wrapping_mul(2_654_435_761_u32 as i32)),
                    u8::try_from(i % 5).ok(),
                    Some(i % 2 == 0),
                    Some(x),
                    Some(x * 3),
                    (!null).then(|| u16::try_from(s).unwrap_or(0)),
                )
            })
            .collect();
        for nkeys in 1..=6 {
            for mode in 0..3 {
                for slots in [1, 3] {
                    for two in [false, true] {
                        row_case(&data, keys, nkeys, 4096, slots, mode, 3, two, None).unwrap_or_else(|e| {
                            panic!("keys {keys} nulls {nulls} nkeys {nkeys} mode {mode} slots {slots} two {two}: {e}")
                        });
                    }
                }
            }
        }
    }
}
