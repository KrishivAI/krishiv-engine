// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every function over random partitions, orderings with ties and NULLs,
    /// and batch splits (partitions span batches) gives the naive value.
    #[test]
    fn window_matches_a_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        partitioned in any::<bool>(),
        kt in prop_oneof![Just(Keys::Int), Just(Keys::Float), Just(Keys::Decimal), Just(Keys::Text)],
    ) {
        let functions = functions();
        let window = plan(functions.clone(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt.block_on(run_keys(kt, &rows, split, &window, 1 << 30)).unwrap();
        let got = outputs(&out, functions.len());
        let mut sorted = rows;
        sorted.sort_by_key(|r| {
            let p = if partitioned { kt.key(r.0) } else { (false, Vec::new()) };
            (p, kt.key(r.1), r.2)
        });
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(kt, &sorted, f, *frame, partitioned))
            .collect();
        let want: Vec<String> = sorted
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let cells: Vec<String> = columns.iter().map(|c| c[i].clone()).collect();
                format!("{}: {}", r.2, cells.join(", "))
            })
            .collect();
        prop_assert_eq!(got, want);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Frames with offsets and frames to the partition's end give the
    /// naive value over each row's frame, with ANSI on (integer sums in
    /// order) and off (integer sums through the segment tree).
    #[test]
    fn frames_match_a_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        partitioned in any::<bool>(),
        ansi in any::<bool>(),
    ) {
        let functions = frame_functions();
        let window = plan(functions.clone(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt
            .block_on(run_keys_ansi(Keys::Int, &rows, split, &window, 1 << 30, ansi))
            .unwrap();
        let got = outputs(&out, functions.len());
        let mut sorted = rows;
        sorted.sort_by_key(|r| {
            let p = if partitioned { Keys::Int.key(r.0) } else { (false, Vec::new()) };
            (p, Keys::Int.key(r.1), r.2)
        });
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(Keys::Int, &sorted, f, *frame, partitioned))
            .collect();
        let want: Vec<String> = sorted
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let cells: Vec<String> = columns.iter().map(|c| c[i].clone()).collect();
                format!("{}: {}", r.2, cells.join(", "))
            })
            .collect();
        prop_assert_eq!(got, want);
    }
}

#[test]
fn ntile_closed_form_is_sparks_algorithm() {
    for rows in 0..40 {
        for k in 1..45 {
            let got: Vec<i64> = (0..rows)
                .map(|r| i64::from(super::super::eval::ntile(r, rows, k).unwrap()))
                .collect();
            assert_eq!(
                got,
                ntile_spark(rows, i64::from(k)),
                "{rows} rows, {k} buckets"
            );
        }
    }
    assert_eq!(super::super::eval::ntile(0, 1, 0), None);
}

/// Spark keeps a window's decimal sum exact and checks it when evaluated:
/// `M, 1, -5` leaves `decimal(38, 0)` after the second row and comes back
/// (Spark 4.2.0: ANSI off `[M, NULL, M - 4]`; ANSI on the second row raises
/// `NUMERIC_VALUE_OUT_OF_RANGE`; over the whole partition `M - 4` either
/// way).
#[tokio::test]
async fn decimal_frames_stay_exact_until_evaluated() {
    let m = 10_i128.pow(38) - 1;
    let (_, input) = decimal_rows(&[m, 1, -5], 38);
    let (_root, ctx) = task(1 << 30);
    let values = |ansi: bool, frame: WindowFrame| {
        let op = decimal_window(AggregateFunction::Sum, frame, 38, ansi);
        drive(&op, &ctx, std::slice::from_ref(&input)).map(|out| column_text(&out, 2))
    };
    let running = WindowFrame::running(FrameUnits::Rows);
    let whole = WindowFrame::whole(FrameUnits::Rows);
    let ms = m.to_string();
    let m4 = (m - 4).to_string();
    assert_eq!(
        values(false, running).unwrap(),
        [ms.as_str(), "null", m4.as_str()]
    );
    let e = values(true, running).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::NumericValueOutOfRangeWithSuggestion
    );
    assert!(
        e.to_string()
            .contains("100000000000000000000000000000000000000"),
        "{e}"
    );
    assert_eq!(
        values(true, whole).unwrap(),
        [m4.as_str(), m4.as_str(), m4.as_str()]
    );
    // Beyond `i128` the exact sum still comes back: M + M + M - M - M.
    let (_, wide) = decimal_rows(&[m, m, m, -m, -m], 38);
    let op = decimal_window(AggregateFunction::Sum, whole, 38, true);
    let out = drive(&op, &ctx, &[wide]).unwrap();
    assert_eq!(column_text(&out, 2)[0], ms);
}

/// Decimal `avg` of an exact sum: ANSI off NULL past the result's
/// precision, ANSI on Spark's error with the scale-39 quotient.
#[tokio::test]
async fn decimal_averages_round_as_spark() {
    let m = 10_i128.pow(38) - 1;
    let (_, input) = decimal_rows(&[m, 1, -5], 38);
    let (_root, ctx) = task(1 << 30);
    let running = WindowFrame::running(FrameUnits::Rows);
    let op = decimal_window(AggregateFunction::Avg, running, 38, false);
    let out = drive(&op, &ctx, std::slice::from_ref(&input)).unwrap();
    assert_eq!(column_text(&out, 2), ["null"; 3]);
    let op = decimal_window(AggregateFunction::Avg, running, 38, true);
    let e = drive(&op, &ctx, &[input]).unwrap_err();
    assert!(
        e.to_string().contains(
            "99999999999999999999999999999999999999.000000000000000000000000000000000000000"
        ),
        "{e}"
    );
    // Within range: (1 + 2) / 2 = 1.5000 in decimal(14, 4).
    let (_, small) = decimal_rows(&[1, 2], 10);
    let op = decimal_window(
        AggregateFunction::Avg,
        WindowFrame::whole(FrameUnits::Rows),
        10,
        true,
    );
    let out = drive(&op, &ctx, &[small]).unwrap();
    assert_eq!(column_text(&out, 2), ["1.5000", "1.5000"]);
}

/// `FILTER (WHERE ...)`: the aggregate reads the rows where the filter is
/// true (not NULL); `first` with `IGNORE NULLS` skips NULL values.
#[tokio::test]
async fn filters_and_ignore_nulls() {
    let s = Arc::new(Schema::new(vec![
        Field::new("o", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("f", DataType::Boolean, true),
    ]));
    let b = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![None, Some(10), Some(20), None])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(true),
                Some(false),
            ])),
        ],
    )
    .unwrap();
    let f = |function, filter, frame| WindowFunctionPlan {
        function: WindowFunction::Aggregate(AggregateExpr {
            function,
            args: vec![1],
            input: AggInput::Raw,
            mask: None,
            arg_type: DataType::Int64,
            name: "x".to_owned(),
            element: None,
        }),
        frame,
        filter,
        name: "x".to_owned(),
    };
    let running = WindowFrame::running(FrameUnits::Rows);
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions: vec![
            f(AggregateFunction::Sum, Some(2), running),
            f(
                AggregateFunction::Count,
                Some(2),
                WindowFrame::whole(FrameUnits::Rows),
            ),
            f(
                AggregateFunction::First { ignore_nulls: true },
                None,
                running,
            ),
            f(
                AggregateFunction::Last {
                    ignore_nulls: false,
                },
                None,
                running,
            ),
        ],
    };
    let op = Window::new(Arc::clone(&s), plan, true, None).unwrap();
    let (_root, ctx) = task(1 << 30);
    let out = drive(&op, &ctx, &[Batch::new(b)]).unwrap();
    assert_eq!(column_text(&out, 3), ["null", "null", "20", "20"]);
    assert_eq!(column_text(&out, 4), ["1", "1", "1", "1"]);
    assert_eq!(column_text(&out, 5), ["null", "10", "10", "10"]);
    assert_eq!(column_text(&out, 6), ["null", "10", "20", "null"]);
    let m = op.extra_metrics();
    assert!(
        m.iter()
            .any(|(k, v)| *k == "partitions" && *v == crate::Metric::Count(1)),
        "{m:?}"
    );
}

#[tokio::test]
async fn memory_is_reserved_before_a_batch_is_taken() {
    let rows: Vec<Row> = (0..1000)
        .map(|i| (Some(i % 7), Some(i), i, Some(i)))
        .collect();
    let window = plan(functions(), true);
    let op = Window::new(schema(), window.clone(), true, None).unwrap();
    let (root, ctx) = task(64 << 10);
    let mut local = op.local_state(&ctx).unwrap();
    match op.execute(&batch(&rows), &mut local).unwrap() {
        OperatorOutput::NeedMemory { bytes } => assert!(bytes > 64 << 10, "{bytes}"),
        other => panic!("expected NeedMemory, got {other:?}"),
    }
    // Nothing was taken: the operator finishes empty.
    assert!(op.finish(&mut local).unwrap().is_none());
    drop((local, ctx));
    assert_eq!(root.reserved(), 0);
    // Through the runner, a pool that cannot hold the partition fails the
    // query with the memory error, and frees everything.
    let e = run(&rows, 1000, &window, 96 << 10).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
}

#[tokio::test]
async fn weights_other_than_one_are_refused() {
    let op = Window::new(schema(), plan(functions(), true), true, None).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    let rows: Vec<Row> = vec![(Some(1), Some(1), 0, Some(1))];
    let weighted = Batch::with_weights(batch(&rows).data().clone(), Weights::Const(2)).unwrap();
    let e = op.execute(&weighted, &mut local).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("weights other than 1"), "{e}");
}

#[tokio::test]
async fn empty_input_gives_no_rows() {
    let op = Window::new(schema(), plan(functions(), false), true, None).unwrap();
    let (_root, ctx) = task(1 << 30);
    let out = drive(&op, &ctx, &[batch(&[])]).unwrap();
    assert!(out.is_empty());
    assert_eq!(op.schema().fields().len(), 5 + functions().len());
}

#[test]
fn invalid_plans_are_rejected() {
    let window = |f: WindowFunctionPlan, partition_by: Vec<usize>| WindowPlan {
        partition_by,
        order_by: vec![SortKey::asc(1)],
        functions: vec![f],
    };
    let f = |function, frame, filter| WindowFunctionPlan {
        function,
        frame,
        filter,
        name: "x".to_owned(),
    };
    let rows = WindowFrame::running(FrameUnits::Rows);
    let class = |p: WindowPlan| {
        Window::new(schema(), p, true, None)
            .unwrap_err()
            .error_class()
    };
    let internal = ErrorClass::InternalError;
    assert_eq!(
        class(window(f(WindowFunction::RowNumber, rows, None), vec![9])),
        internal
    );
    assert_eq!(
        class(window(f(WindowFunction::Ntile(0), rows, None), vec![])),
        internal
    );
    assert_eq!(
        class(window(f(WindowFunction::RowNumber, rows, Some(0)), vec![])),
        internal
    );
    assert_eq!(
        class(window(
            f(agg(AggregateFunction::Sum, Some(9)), rows, None),
            vec![]
        )),
        internal
    );
    // RANGE frames with value bounds are a later part of P2-T055.
    let following = WindowFrame {
        units: FrameUnits::Range,
        lower: FrameBound::Offset(-1),
        upper: FrameBound::CurrentRow,
    };
    assert_eq!(
        class(window(
            f(agg(AggregateFunction::Sum, Some(3)), following, None),
            vec![]
        )),
        ErrorClass::KrishivUnsupportedFeature
    );
    let op = Window::new(schema(), plan(functions(), true), true, None).unwrap();
    assert!(format!("{op:?}").contains("Window"));
}

/// What a unit's evaluation reserves matches what it holds: at least its
/// outputs and row maps, and at most twice that (the rest is what it
/// holds while it runs: states, group ids, gather indices).
#[test]
fn reserved_cost_matches_what_evaluation_holds() {
    let numeric: Vec<(WindowFunction, WindowFrame)> = functions()
        .into_iter()
        .filter(|(f, _)| !matches!(f, WindowFunction::Aggregate(a) if a.args == [4]))
        .collect();
    for (rows, parts) in [(10_000, 100), (10_000, 1), (4_096, 4_096)] {
        for f in &numeric {
            let window = plan(vec![f.clone()], true);
            let (b, part, peer) = sorted_unit(rows, parts);
            let n = b.num_rows();
            let slice = super::super::eval::Slice {
                batch: b.data(),
                offset: 0,
                len: n,
                part_start: &part,
                peer_start: &peer,
            };
            let out =
                super::super::eval::evaluate(&window, &[slice], true, &super::super::utc_zone())
                    .unwrap();
            let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
            // Outputs, row maps (`part_of`, `peer_of`) and the partition and
            // peer-group vectors.
            let held: usize = out
                .iter()
                .map(|c| Array::get_array_memory_size(c.as_ref()))
                .sum::<usize>()
                + 8 * n
                + 20 * count(&part)
                + 16 * count(&peer);
            let cost = super::super::eval::cost(&window, n, count(&part), count(&peer), 0).unwrap();
            assert!(cost >= held, "{f:?}: reserved {cost} < held {held}");
            assert!(cost <= 2 * held, "{f:?}: reserved {cost} for {held} held");
        }
    }
}

/// A partitioned window runs within a limit that holds a few batches and
/// their exact evaluation cost (the earlier flat per-row estimate needed
/// several times more and failed here).
#[tokio::test]
async fn partitioned_windows_run_at_a_tight_limit() {
    let window = plan(
        vec![
            (WindowFunction::Rank, WindowFrame::running(FrameUnits::Rows)),
            (
                agg(AggregateFunction::Sum, Some(3)),
                WindowFrame::running(FrameUnits::Range),
            ),
        ],
        true,
    );
    let rows: Vec<Row> = (0..40_000)
        .map(|i| (Some(i / 100), Some(i / 3), i, Some(i % 11)))
        .collect();
    let per_batch = 4_000;
    let batches: Vec<Batch> = rows.chunks(per_batch).map(batch).collect();
    // A batch: 40 partitions, 1334 peer groups.
    let exact = batches[0].data().get_array_memory_size()
        + 2 * per_batch
        + super::super::eval::cost(&window, per_batch, 40, 1_334, 0).unwrap();
    let limit = 4 * exact;
    // The flat estimate this replaces reserved 2 + 16 + 64 + 16 bytes per
    // row besides the input.
    assert!(batches.len() * per_batch * 98 > limit);
    let op = Window::new(schema(), window, true, None).unwrap();
    let (root, ctx) = task(limit);
    let out = drive(&op, &ctx, &batches).unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 40_000);
    drop((out, ctx));
    assert_eq!(root.reserved(), 0);
}

/// String results are views of the input's buffers: a running `max` over
/// one partition of 1 KB strings reserves and holds about the input, not
/// a copy of the maximum per row.
#[tokio::test]
async fn string_results_share_the_input_buffers() {
    let rows = 10_000;
    let s = Arc::new(Schema::new(vec![
        Field::new("o", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let values: Vec<String> = (0..rows).map(|i| format!("{i:0>1024}")).collect();
    let input = arrow_array::StringArray::from_iter_values(&values);
    let data_ptr = input.values().as_ptr();
    let b = RecordBatch::try_new(
        Arc::clone(&s),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows)),
            Arc::new(input),
        ],
    )
    .unwrap();
    let input_bytes = b.get_array_memory_size();
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions: vec![WindowFunctionPlan {
            function: WindowFunction::Aggregate(AggregateExpr {
                function: AggregateFunction::Max,
                args: vec![1],
                input: AggInput::Raw,
                mask: None,
                arg_type: DataType::Utf8View,
                name: "m".to_owned(),
                element: None,
            }),
            frame: WindowFrame::running(FrameUnits::Rows),
            filter: None,
            name: "m".to_owned(),
        }],
    };
    let op = Window::new(Arc::clone(&s), plan, true, None).unwrap();
    // A copy per row would need 10 MB more than the input.
    let (root, ctx) = task(input_bytes + input_bytes / 2);
    let out = drive(&op, &ctx, &[Batch::new(b)]).unwrap();
    let m = out[0].data().column(2).as_string_view();
    assert_eq!(m.value(9_999), values[9_999]);
    assert!(m.data_buffers().iter().any(|d| d.as_ptr() == data_ptr));
    drop((out, ctx));
    assert_eq!(root.reserved(), 0);
}

#[test]
fn windows_require_their_partitioning() {
    assert_eq!(
        plan(functions(), true).required_distribution(),
        RequiredDistribution::Clustered(vec![0])
    );
    assert_eq!(
        plan(functions(), false).required_distribution(),
        RequiredDistribution::Single
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// With spills at any batch, and partitions read back in memory or
    /// evaluated from disk, every function gives what it gives without a
    /// spill, in the same order.
    #[test]
    fn spilled_windows_match_in_memory_ones(
        rows in big_rows_strategy(),
        split in 1_usize..12,
        every in 0_usize..4,
        slack in prop_oneof![Just(None), (0_usize..8192).prop_map(Some)],
        partitioned in any::<bool>(),
    ) {
        let window = plan(functions(), partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let spills = Spills { every, slack };
        let (spilled, _) = rt.block_on(compare_spilling(&rows, split, &window, spills));
        prop_assert!(every == 0 || rows.len() < split.max(1) * every || spilled > 0);
    }
}

/// A partition that does not fit in memory is copied to a file of its own
/// and evaluated from there, with every function, partitioned or not.
#[tokio::test]
async fn a_partition_larger_than_memory_is_evaluated_from_disk() {
    for partitioned in [false, true] {
        let rows: Vec<Row> = (0..6_000)
            .map(|i| {
                // One large partition between two small ones.
                let p = if i < 100 {
                    0
                } else if i < 5_900 {
                    1
                } else {
                    2
                };
                (
                    Some(p),
                    (i % 7 != 0).then_some(i / 5),
                    i,
                    (i % 13 != 0).then_some(i % 17),
                )
            })
            .collect();
        let window = plan(functions(), partitioned);
        // Room for evaluating a batch from disk, not for the partition.
        let spills = Spills {
            every: 1,
            slack: Some(96 << 10),
        };
        let (spilled, external) = compare_spilling(&rows, 250, &window, spills).await;
        assert!(spilled > 0);
        assert!(external >= 1, "partitioned {partitioned}: {external}");
    }
}

/// The ROWS and RANGE aggregates of strings, decimals and filtered rows
/// from disk equal their in-memory values.
#[tokio::test]
async fn exact_and_string_frames_from_disk() {
    use AggregateFunction::{Avg, Max, Min, Sum};
    let (s, b) = decimal_rows(
        &(0..3_000)
            .map(|i| i128::from(i % 97) * 1_000_003)
            .collect::<Vec<_>>(),
        20,
    );
    let frames = [
        WindowFrame::running(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Range),
        WindowFrame::whole(FrameUnits::Rows),
    ];
    let mut functions = Vec::new();
    for (k, frame) in frames.iter().enumerate() {
        for function in [Sum, Avg, Min, Max] {
            let name = format!("d{k}{function:?}");
            functions.push(WindowFunctionPlan {
                function: WindowFunction::Aggregate(AggregateExpr {
                    function: function.clone(),
                    args: vec![1],
                    input: AggInput::Raw,
                    mask: None,
                    arg_type: DataType::Decimal128(20, 0),
                    name: "d".to_owned(),
                    element: None,
                }),
                frame: *frame,
                filter: None,
                name,
            });
        }
    }
    let plan = WindowPlan {
        partition_by: Vec::new(),
        order_by: vec![SortKey::asc(0)],
        functions,
    };
    let batches: Vec<Batch> = (0..b.num_rows())
        .step_by(100)
        .map(|i| Batch::new(b.data().slice(i, 100.min(b.num_rows() - i))))
        .collect();
    let plain = Window::new(Arc::clone(&s), plan.clone(), true, None).unwrap();
    let (_r, ctx) = task(1 << 30);
    let want = drive(&plain, &ctx, &batches).unwrap();
    let (_manager, scope) = spill_scope();
    let op = Window::new(s, plan.clone(), true, Some(scope)).unwrap();
    let limit = 1 << 30;
    let (root, ctx) = task(limit);
    let spills = Spills {
        every: 1,
        slack: Some(0),
    };
    let got = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
        .await
        .unwrap();
    assert!(
        op.counters
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    );
    for c in 2..2 + plan.functions.len() {
        assert_eq!(column_text(&got, c), column_text(&want, c), "column {c}");
    }
}

/// The flush phase reads each spilled batch with the memory its headroom lends as credit. Here the pool is taken whole
/// after the input, and spilled batches alternate between small and large: the flush phase must read them all and
/// give the in-memory result.
#[tokio::test]
async fn spilled_batches_of_alternating_sizes_read_back_from_a_full_pool() {
    for large in [300, 1_500] {
        alternating_sizes_from_a_full_pool(large, functions(), true).await;
    }
}

/// Offsets and `nth_value` keep the batches they read past their read: the flush phase's headroom covers what
/// their offset spans, so a pool taken whole does not refuse them.
#[tokio::test]
async fn offsets_and_nth_value_read_back_from_a_full_pool() {
    let rows = WindowFrame::running(FrameUnits::Rows);
    let whole = WindowFrame::whole(FrameUnits::Rows);
    let functions = vec![(lag_lead(-2, Some(3), false), whole), (nth(3, true), rows)];
    // With larger batches or more functions, the two batches each keeps are more than the buffered partition lends:
    // the query's error then comes from the flush phase's headroom, not from a charge with no fallback.
    alternating_sizes_from_a_full_pool(300, functions, false).await;
}

/// A spill whose write fails loses the rows it took: every later call
/// fails with that error instead of returning a short result. The write
/// fails with an I/O error (the scope's directory is gone), not a
/// cancellation, so the stored error's class differs from the placeholder
/// the spill holds while the blocking write runs.
#[tokio::test]
async fn a_failed_spill_fails_every_later_call() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope.clone())).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    assert!(matches!(
        op.execute(&batch(&rows), &mut local).unwrap(),
        OperatorOutput::NeedMoreInput(_)
    ));
    std::fs::remove_dir_all(scope.dir()).unwrap();
    let e = op.spill(&mut local, 1, &ctx).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillIoError, "{e}");
    let class = e.error_class();
    let again = op.execute(&batch(&rows[..1]), &mut local).unwrap_err();
    assert_eq!(again.error_class(), class, "{again}");
    assert_eq!(op.finish(&mut local).unwrap_err().error_class(), class);
    assert_eq!(
        op.end_input(&mut local, &ctx)
            .await
            .unwrap_err()
            .error_class(),
        class
    );
    assert_eq!(
        op.flush(&mut local, &ctx).await.unwrap_err().error_class(),
        class
    );
}

/// A window's spill write runs under its consumer's `spilling` guard, so
/// another driver's round that cannot spill waits for the rows it frees
/// (the memory crate's `a_round_waits_for_another_consumers_spill_write`).
#[tokio::test]
async fn a_spill_write_holds_the_spilling_guard() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope.clone())).unwrap();
    let (root, ctx) = task(1 << 30);
    let consumer = Arc::clone(ctx.consumer().consumer());
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    scope.set_test_writer_hook(Arc::new({
        let seen = Arc::clone(&seen);
        move || seen.lock().unwrap().push(consumer.is_spilling())
    }));
    let mut local = op.local_state(&ctx).unwrap();
    assert!(matches!(
        op.execute(&batch(&rows), &mut local).unwrap(),
        OperatorOutput::NeedMoreInput(_)
    ));
    assert!(op.spill(&mut local, 1, &ctx).await.unwrap() > 0);
    assert_eq!(*seen.lock().unwrap(), [true]);
    assert!(!ctx.consumer().consumer().is_spilling());
    drop((local, op, ctx));
    assert_eq!(root.reserved(), 0);
}

/// A spill dropped mid-write (its query cancelled) leaves the window
/// failed with `OPERATION_CANCELED`; a refused scratch reservation spills
/// nothing and loses nothing.
#[tokio::test]
async fn a_dropped_spill_leaves_the_window_cancelled() {
    use futures::FutureExt;
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope)).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows), &mut local).unwrap();
    // Polled once: it waits for its blocking write, and is dropped.
    assert!(op.spill(&mut local, 1, &ctx).now_or_never().is_none());
    let e = op.finish(&mut local).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::OperationCanceled, "{e}");

    // A pool without room for the writer's scratch: the spill declines.
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope)).unwrap();
    let limit = 256 << 10;
    let (root, ctx) = task(limit);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows[..10]), &mut local).unwrap();
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    take_all_but(&mut hog, limit - root.reserved(), 0);
    assert_eq!(op.spill(&mut local, 1, &ctx).await.unwrap(), 0);
    drop(hog);
    let out: usize = std::iter::from_fn(|| op.finish(&mut local).unwrap())
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(out, 10);
}

/// A failed read in the flush phase fails it and every later step. The
/// read fails with an I/O error (the spill files are gone), not a
/// cancellation, so the stored error's class differs from the placeholder
/// the flush holds while the blocking read runs.
#[tokio::test]
async fn a_failed_read_back_fails_the_flush_phase() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope.clone())).unwrap();
    let (_root, ctx) = task(1 << 30);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows), &mut local).unwrap();
    assert!(op.spill(&mut local, 1, &ctx).await.unwrap() > 0);
    assert!(op.finish(&mut local).unwrap().is_none());
    op.end_input(&mut local, &ctx).await.unwrap();
    std::fs::remove_dir_all(scope.dir()).unwrap();
    let e = op.flush(&mut local, &ctx).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillIoError, "{e}");
    let again = op.flush(&mut local, &ctx).await.unwrap_err();
    assert_eq!(again.error_class(), e.error_class());
}

/// One partition many times the memory limit, through the sort and the
/// window with spilling: every row's values are checked against closed
/// forms as the output streams, and the pool is empty at the end.
#[tokio::test]
async fn one_large_partition_spills_through_the_runner() {
    use AggregateFunction::{Count, Sum};
    let (n, limit) = large_case();
    let rows = WindowFrame::running(FrameUnits::Rows);
    let range = WindowFrame::running(FrameUnits::Range);
    let window = plan(
        vec![
            (WindowFunction::RowNumber, rows),
            (WindowFunction::Rank, rows),
            (agg(Sum, Some(3)), rows),
            (agg(Count, None), range),
            (agg(Sum, Some(3)), WindowFrame::whole(FrameUnits::Rows)),
            // An exact percentile over the whole partition, folded from disk.
            (
                agg(
                    AggregateFunction::Percentile {
                        percentages: vec![0.5],
                        array: false,
                        reverse: false,
                    },
                    Some(3),
                ),
                WindowFrame::whole(FrameUnits::Rows),
            ),
            // Read through `disk.rs`: an offset function, a sliding frame, a
            // frame from the start, a frame over key values, `nth_value`.
            (lag_lead(-1, None, false), rows),
            (
                agg(Sum, Some(3)),
                between(
                    FrameUnits::Rows,
                    FrameBound::Offset(-1),
                    FrameBound::Offset(1),
                ),
            ),
            (
                agg(Count, None),
                between(
                    FrameUnits::Rows,
                    FrameBound::UnboundedPreceding,
                    FrameBound::Offset(-1),
                ),
            ),
            (
                agg(Count, None),
                between(
                    FrameUnits::Range,
                    FrameBound::RangeInt(-3),
                    FrameBound::RangeInt(3),
                ),
            ),
            (nth(2, false), range),
        ],
        true,
    );
    let s = schema();
    // Generated in batches, unsorted (descending ids), into the sort.
    let per = 8_192;
    let source = Generated::new(Arc::clone(&s), (n + per - 1) / per, {
        move |k: i64| {
            let hi = n - k * per;
            let lo = (hi - per).max(0);
            let rows: Vec<Row> = (lo..hi)
                .rev()
                .map(|i| (Some(0), Some(i / 3), i, Some(i % 11)))
                .collect();
            batch(&rows)
        }
    });
    let (_manager, scope) = spill_scope();
    let keys = [SortKey::asc(0), SortKey::asc(1), SortKey::asc(2)];
    let sort = Arc::new(Sort::new(Arc::clone(&s), &keys, Some(scope.clone())).unwrap());
    let op = Arc::new(Window::new(Arc::clone(&s), window, true, Some(scope)).unwrap());
    let node: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input: Arc::new(BreakerNode {
            input: Arc::new(SourceNode(source)),
            sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
            source: sort,
            ordered: true,
        }),
        op: Arc::clone(&op) as Arc<dyn crate::Operator>,
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(2).unwrap());
    let root = pool(limit);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(limit)),
    );
    let total: i64 = (0..n).map(|i| i % 11).sum();
    let median = {
        let mut v: Vec<i64> = (0..n).map(|i| i % 11).collect();
        v.sort_unstable();
        // percentile(x, 0.5): the mean of the two middle values.
        let (lo, hi) = ((v.len() - 1) / 2, v.len() / 2);
        (v[lo] + v[hi]) as f64 / 2.0
    };
    let mut stream = runner.execute(node, Arc::clone(&q)).await.unwrap();
    let mut next = 0_i64;
    let mut running = 0_i64;
    while let Some(b) = stream.next().await {
        let b = b.unwrap();
        let d = b.data();
        let col = |c: usize| d.column(c).as_primitive::<Int64Type>().clone();
        let int = |c: usize| d.column(c).as_primitive::<Int32Type>().clone();
        let (ids, rn, rank, sum, count, whole) = (col(2), int(5), int(6), col(7), col(8), col(9));
        let med = d
            .column(10)
            .as_primitive::<arrow_array::types::Float64Type>()
            .clone();
        for r in 0..d.num_rows() {
            let i = ids.value(r);
            assert_eq!(i, next, "rows in order");
            running += i % 11;
            assert_eq!(i64::from(rn.value(r)), i + 1);
            assert_eq!(i64::from(rank.value(r)), i / 3 * 3 + 1);
            assert_eq!(sum.value(r), running);
            assert_eq!(count.value(r), (i / 3 * 3 + 3).min(n));
            assert_eq!(whole.value(r), total);
            assert!((med.value(r) - median).abs() < 1e-9);
            let (lag, slide, before, near, second) = (col(11), col(12), col(13), col(14), col(15));
            if i == 0 {
                assert!(lag.is_null(r));
            } else {
                assert_eq!(lag.value(r), (i - 1) % 11);
            }
            let around: i64 = ((i - 1).max(0)..=(i + 1).min(n - 1)).map(|x| x % 11).sum();
            assert_eq!(slide.value(r), around);
            assert_eq!(before.value(r), i);
            let k = i / 3;
            assert_eq!(near.value(r), ((k + 3) * 3 + 3).min(n) - (k - 3).max(0) * 3);
            assert_eq!(second.value(r), 1);
            next += 1;
        }
    }
    assert_eq!(next, n);
    drop((stream, q));
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    assert!(
        op.counters
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );
}

/// The flush phase's headroom refused: the window is failed, and a later
/// flush fails too instead of finding no rows.
#[tokio::test(start_paused = true)]
async fn a_refused_flush_headroom_fails_the_window() {
    let rows: Vec<Row> = (0..400)
        .map(|i| (Some(i / 300), Some(i), i, Some(i)))
        .collect();
    let (_manager, scope) = spill_scope();
    let op = Window::new(schema(), plan(functions(), true), true, Some(scope)).unwrap();
    let limit = 1 << 20;
    let (root, ctx) = task(limit);
    let mut local = op.local_state(&ctx).unwrap();
    op.execute(&batch(&rows), &mut local).unwrap();
    assert!(op.spill(&mut local, 1, &ctx).await.unwrap() > 0);
    assert!(op.finish(&mut local).unwrap().is_none());
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    take_all_but(&mut hog, limit - root.reserved(), 0);
    let e = op.end_input(&mut local, &ctx).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
    let again = op.flush(&mut local, &ctx).await.unwrap_err();
    assert_eq!(again.error_class(), ErrorClass::UnableToAcquireMemory);
    drop((hog, local));
}

/// The reservation for frames with offsets covers their outputs and row
/// maps (it also holds the frames, the tree and a chunk of scratch).
#[test]
fn frame_costs_cover_their_outputs() {
    for (rows, parts) in [(10_000, 100), (4_096, 4_096)] {
        for f in frame_functions() {
            let window = plan(vec![f.clone()], true);
            let (b, part, peer) = sorted_unit(rows, parts);
            let n = b.num_rows();
            let slice = super::super::eval::Slice {
                batch: b.data(),
                offset: 0,
                len: n,
                part_start: &part,
                peer_start: &peer,
            };
            let out =
                super::super::eval::evaluate(&window, &[slice], false, &super::super::utc_zone())
                    .unwrap();
            let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
            let held: usize = out
                .iter()
                .map(|c| Array::get_array_memory_size(c.as_ref()))
                .sum::<usize>()
                + 8 * n * 2;
            let cost = super::super::eval::cost(&window, n, count(&part), count(&peer), 0).unwrap();
            assert!(cost >= held, "{f:?}: reserved {cost} < held {held}");
        }
    }
}
