// SPDX-License-Identifier: Apache-2.0
//! Tests, part of this module's test suite (split by size; the helpers are in the parent).

use super::*;

/// `lag`/`lead`/`nth_value` reserve their peak: the input copied into
/// views (strings), the default column's views too, the counting rows, the
/// picks, the gather's locations and indices, the frames (`nth_value`) and
/// the result, all alive while the result is gathered.
#[test]
fn offset_function_costs_cover_their_scratch() {
    let rows = |lower, upper| WindowFrame {
        units: FrameUnits::Rows,
        lower,
        upper,
    };
    let lag_s = WindowFunction::Offset {
        input: 4,
        offset: -1,
        default: Some(4),
        ignore_nulls: false,
        data_type: DataType::Utf8View,
    };
    let nth_s = WindowFunction::NthValue {
        input: 4,
        n: 2,
        ignore_nulls: true,
        data_type: DataType::Utf8View,
    };
    let cases = [
        (
            lag_s,
            rows(FrameBound::Offset(-1), FrameBound::Offset(-1)),
            true,
            0,
        ),
        (nth_s, WindowFrame::whole(FrameUnits::Rows), false, 8),
        (
            nth(3, false),
            WindowFrame::whole(FrameUnits::Rows),
            false,
            8,
        ),
    ];
    for (function, frame, with_default, frames) in cases {
        let window = plan(vec![(function.clone(), frame)], true);
        let (b, part, peer) = sorted_unit(10_000, 100);
        let n = b.num_rows();
        let slice = super::super::eval::Slice {
            batch: b.data(),
            offset: 0,
            len: n,
            part_start: &part,
            peer_start: &peer,
        };
        let out = super::super::eval::evaluate(&window, &[slice], false, &super::super::utc_zone())
            .unwrap();
        // What an array holds of its own: views share the input's value
        // buffers.
        let own = |a: &ArrayRef| {
            let shared: usize = match a.data_type() {
                DataType::Utf8View => a
                    .as_string_view()
                    .data_buffers()
                    .iter()
                    .map(arrow_buffer::Buffer::capacity)
                    .sum(),
                _ => 0,
            };
            Array::get_array_memory_size(a.as_ref()) - shared
        };
        let output: usize = out.iter().map(own).sum();
        let views = match &function {
            WindowFunction::Offset { .. } | WindowFunction::NthValue { input: 4, .. } => {
                let v = super::super::eval::to_view(b.data().column(4)).unwrap();
                own(&v) * (1 + usize::from(with_default))
            }
            _ => 0,
        };
        // Counting rows, picks, gather locations and indices, frames.
        let scratch = n * (8 + 16 + 32 + frames);
        let held = output + views + scratch;
        let count = |v: &[bool]| v.iter().filter(|&&x| x).count();
        let cost = super::super::eval::cost(&window, n, count(&part), count(&peer), 0).unwrap();
        assert!(cost >= held, "{function:?}: reserved {cost} < held {held}");
    }
}

/// A segment tree over more rows than a chunk is built a chunk of parents
/// at a time and still gives each frame's value.
#[test]
fn trees_larger_than_a_chunk_match_the_reference() {
    use AggregateFunction::{Count, Max, Min, Sum};
    let frame = WindowFrame {
        units: FrameUnits::Rows,
        lower: FrameBound::Offset(-3),
        upper: FrameBound::Offset(2),
    };
    let functions: Vec<(WindowFunction, WindowFrame)> = [Sum, Count, Min, Max]
        .into_iter()
        .map(|f| (agg(f, Some(3)), frame))
        .collect();
    let n = i64::try_from(2 * super::super::frames::CHUNK + 777).unwrap();
    let data: Vec<Row> = (0..n)
        .map(|i| (Some(0), Some(i), i, (i % 13 != 0).then_some(i % 97 - 40)))
        .collect();
    let window = plan(functions, true);
    let b = batch(&data);
    let part: Vec<bool> = (0..data.len()).map(|i| i == 0).collect();
    let peer = vec![true; data.len()];
    let slice = super::super::eval::Slice {
        batch: b.data(),
        offset: 0,
        len: data.len(),
        part_start: &part,
        peer_start: &peer,
    };
    let out =
        super::super::eval::evaluate(&window, &[slice], false, &super::super::utc_zone()).unwrap();
    let got: Vec<String> = data
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let cells: Vec<String> = out.iter().map(|c| cell(c, i)).collect();
            format!("{}: {}", r.2, cells.join(", "))
        })
        .collect();
    // Each row's frame of non-NULL values, directly (one partition).
    let want: Vec<String> = (0..data.len())
        .map(|i| {
            let lo = i.saturating_sub(3);
            let hi = (i + 3).min(data.len());
            let vals: Vec<i64> = data[lo..hi].iter().filter_map(|r| r.3).collect();
            let show = |v: Option<i64>| v.map_or_else(|| "null".to_owned(), |v| v.to_string());
            let sum = (!vals.is_empty()).then(|| vals.iter().sum::<i64>());
            let count = i64::try_from(vals.len()).unwrap();
            format!(
                "{}: {}, {}, {}, {}",
                data[i].2,
                show(sum),
                count,
                show(vals.iter().min().copied()),
                show(vals.iter().max().copied())
            )
        })
        .collect();
    assert_eq!(got, want);
}

/// A running `collect_list` snapshots a growing copy per row: over a
/// partition larger than memory it is refused too.
#[tokio::test]
async fn a_growing_running_state_over_a_partition_on_disk_is_refused() {
    refused_on_disk(
        agg(AggregateFunction::CollectList, Some(3)),
        WindowFrame::running(FrameUnits::Rows),
    )
    .await;
}

/// The window's metrics count its input once: rows read back in the flush
/// phase (in memory or copied to a partition's own file) are not counted
/// again.
#[tokio::test]
async fn spilled_rows_are_counted_once() {
    let rows: Vec<Row> = (0..6_000)
        .map(|i| {
            let p = if i < 100 {
                0
            } else if i < 5_900 {
                1
            } else {
                2 + i % 5
            };
            (Some(p), Some(i), i, Some(i % 17))
        })
        .collect();
    let window = plan(functions(), true);
    let (_dir, scope) = spill_scope();
    let op = Window::new(schema(), window, true, Some(scope)).unwrap();
    let limit = 1 << 30;
    let (root, ctx) = task(limit);
    let mut sorted = rows.clone();
    sorted.sort_by_key(|r| (r.0, r.1));
    let batches: Vec<Batch> = sorted.chunks(250).map(batch).collect();
    let spills = Spills {
        every: 1,
        slack: Some(96 << 10),
    };
    let out = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
        .await
        .unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 6_000);
    assert!(
        op.counters
            .external
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );
    let count = |name: &str| {
        op.extra_metrics()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m)
    };
    assert!(
        matches!(count("rows"), Some(crate::metrics::Metric::Count(6_000))),
        "{:?}",
        count("rows")
    );
    // Partitions 0, 1, then 2..=6.
    assert!(
        matches!(count("partitions"), Some(crate::metrics::Metric::Count(7))),
        "{:?}",
        count("partitions")
    );
}

/// A batch whose first rows wait in the output keeps its bytes reserved
/// for them when the rest of it spills: they are not the spill's to free
/// (nor to fund its writer's scratch) until those rows are emitted.
#[tokio::test]
async fn bytes_of_rows_waiting_in_the_output_stay_reserved_through_a_spill() {
    // `a`: 8 rows of partition 0. `b`: 2 more rows of partition 0, then a
    // large partition 1, so `b`'s first rows are evaluated while the rest
    // of it stays buffered.
    let a: Vec<Row> = (0..8).map(|i| (Some(0), Some(i), i, Some(i))).collect();
    let b: Vec<Row> = (8..20_000)
        .map(|i| (Some(i64::from(i >= 10)), Some(i), i, Some(i)))
        .collect();
    let (_dir, scope) = spill_scope();
    let window = plan(
        vec![(
            WindowFunction::RowNumber,
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    let op = Window::new(schema(), window, true, Some(scope)).unwrap();
    let limit = 64 << 20;
    let config = ExecConfig {
        batch_size: 4,
        ..ExecConfig::default()
    };
    let (root, ctx) = task_with(limit, config);
    let mut local = op.local_state(&ctx).unwrap();
    let mut out = 0;
    let b = batch(&b);
    for input in [batch(&a), b.clone()] {
        match op.execute(&input, &mut local).unwrap() {
            OperatorOutput::NeedMoreInput(o) => out += o.map_or(0, |o| o.num_rows()),
            other => panic!("{other:?}"),
        }
    }
    // `a`'s rows went out; `b`'s first two wait in the output.
    assert_eq!(out, 8);
    // The buffered rows are all of `b`, whose bytes its waiting rows hold:
    // the spill writes them but frees nothing yet.
    assert_eq!(op.spill(&mut local, 1, &ctx).await.unwrap(), 0);
    let held = b.data().get_array_memory_size();
    assert!(root.reserved() >= held, "{} < {held}", root.reserved());
    let waiting = op.finish(&mut local).unwrap().unwrap();
    assert_eq!(waiting.num_rows(), 2);
    assert!(root.reserved() < held, "released with its last rows");
    out += waiting.num_rows();
    assert!(op.finish(&mut local).unwrap().is_none());
    op.end_input(&mut local, &ctx).await.unwrap();
    loop {
        match op.flush(&mut local, &ctx).await.unwrap() {
            crate::operator::FlushOutput::Batch(b) => out += b.num_rows(),
            crate::operator::FlushOutput::Done => break,
            _ => {}
        }
    }
    assert_eq!(out, 20_000);
    drop(local);
    ctx.consumer().consumer().release_credit();
    assert_eq!(root.reserved(), 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Every function over a bucketed sort (partitions split among four
    /// drivers) gives the naive value of every row.
    #[test]
    fn bucketed_windows_match_the_naive_reference(
        rows in rows_strategy(),
        split in 1_usize..9,
        kt in prop_oneof![Just(Keys::Int), Just(Keys::Float), Just(Keys::Decimal), Just(Keys::Text)],
    ) {
        let functions = functions();
        let window = plan(functions.clone(), true);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let out = rt.block_on(run_bucketed(kt, &rows, split, &window)).unwrap();
        let mut got = outputs(&out, functions.len());
        got.sort();
        let mut sorted = rows;
        sorted.sort_by_key(|r| (kt.key(r.0), kt.key(r.1), r.2));
        let columns: Vec<Vec<String>> = functions
            .iter()
            .map(|(f, frame)| reference(kt, &sorted, f, *frame, true))
            .collect();
        let mut want: Vec<String> = sorted
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let cells: Vec<String> = columns.iter().map(|c| c[i].clone()).collect();
                format!("{}: {}", r.2, cells.join(", "))
            })
            .collect();
        want.sort();
        prop_assert_eq!(got, want);
    }
}

/// A growing state's snapshots are priced by position in the partition, so
/// a skewed input costs its real quadratic share and the price is additive
/// across batches.
#[test]
fn snapshot_slots_follow_partition_sizes_across_batches() {
    use super::super::eval::snapshot_slots;
    // Partitions of 3 and 1 rows: 1+2+3 and 1.
    assert_eq!(snapshot_slots(&[true, false, false, true], 0), (7, 1));
    // One partition of 6 rows, whole or split into batches of 4 and 2.
    let whole = snapshot_slots(&[true, false, false, false, false, false], 0);
    let (a, open) = snapshot_slots(&[true, false, false, false], 0);
    let (b, open) = snapshot_slots(&[false, false], open);
    assert_eq!(whole, (21, 6));
    assert_eq!((a + b, open), (21, 6));
    // Skew is priced exactly: partitions of 100 and 2 rows.
    let mut flags = vec![false; 102];
    flags[0] = true;
    flags[100] = true;
    assert_eq!(snapshot_slots(&flags, 0).0, 5_050 + 3);
    assert_eq!(snapshot_slots(&[], 7), (0, 7));
}

/// The window cost of a running `collect_list` grows with the snapshot
/// elements; a whole-partition frame and a non-growing aggregate ignore them.
#[test]
fn window_cost_counts_snapshot_elements_of_growing_aggregates() {
    let running = WindowFrame::running(FrameUnits::Rows);
    let growing = plan(
        vec![(agg(AggregateFunction::CollectList, Some(3)), running)],
        true,
    );
    let summing = plan(vec![(agg(AggregateFunction::Sum, Some(3)), running)], true);
    let base = super::super::eval::cost(&growing, 10, 1, 10, 0).unwrap();
    assert_eq!(
        super::super::eval::cost(&growing, 10, 1, 10, 880).unwrap(),
        base + 880
    );
    let sum_base = super::super::eval::cost(&summing, 10, 1, 10, 0).unwrap();
    assert_eq!(
        super::super::eval::cost(&summing, 10, 1, 10, 880).unwrap(),
        sum_base
    );
}

/// A running `collect_list` charges its snapshots by the width of its
/// argument: long strings cost more per element than the 16-byte floor, a
/// narrow integer column does not.
#[test]
fn snapshot_width_follows_the_argument_column() {
    let rows: Vec<Row> = (0..50)
        .map(|i| (Some(0), Some(i), i, Some(i % 7)))
        .collect();
    let b = batch(&rows);
    let narrow = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(2)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    assert_eq!(super::super::eval::snapshot_width(&narrow, b.data()), 16);
    let mut columns = b.data().columns().to_vec();
    let long = "x".repeat(200);
    columns[4] = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..50).map(|_| long.as_str()),
    ));
    let wide = RecordBatch::try_new(b.data().schema(), columns).unwrap();
    let list = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(4)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    assert!(super::super::eval::snapshot_width(&list, &wide) >= 200);
    // Distinct keys are kept besides the values: twice the width.
    let set = plan(
        vec![(
            agg(AggregateFunction::CollectSet, Some(4)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    assert!(super::super::eval::snapshot_width(&set, &wide) >= 400);
    // A whole-partition frame snapshots nothing.
    let whole = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(4)),
            WindowFrame::whole(FrameUnits::Rows),
        )],
        true,
    );
    assert_eq!(super::super::eval::snapshot_width(&whole, &wide), 16);
}

/// One large partition under a running `collect_list` needs a quadratic
/// reservation: under a tight limit the window fails with a memory error
/// before the snapshots are made, and releases what it reserved.
#[tokio::test]
async fn a_running_collect_list_over_a_big_partition_fails_under_a_tight_limit() {
    let window = plan(
        vec![(
            agg(AggregateFunction::CollectList, Some(2)),
            WindowFrame::running(FrameUnits::Rows),
        )],
        true,
    );
    let rows: Vec<Row> = (0..3_000)
        .map(|i| (Some(0), Some(i), i, Some(i % 7)))
        .collect();
    let batches: Vec<Batch> = rows.chunks(500).map(batch).collect();
    let op = Window::new(schema(), window, true, None).unwrap();
    let limit = 16 << 20;
    let (root, ctx) = task(limit);
    let e = drive(&op, &ctx, &batches).unwrap_err();
    // The window asks for memory (a spill) before it makes the snapshots;
    // the driver reports that request.
    assert!(e.to_string().contains("NeedMemory"), "{e}");
    drop(ctx);
    assert_eq!(root.reserved(), 0);
}

/// A partition larger than memory gives what it gives in memory, for every
/// frame of aggregates and `nth_value`.
#[tokio::test]
async fn frames_of_a_partition_on_disk_match_memory() {
    let rows = skewed_rows(3_000);
    let spills = Spills {
        every: 1,
        slack: Some(4 << 20),
    };
    for frame in disk_frames() {
        let window = plan(over(frame), true);
        let (spilled, external) = compare_spilling(&rows, 250, &window, spills).await;
        assert!(spilled > 0, "{frame:?}");
        assert!(external >= 1, "{frame:?}");
    }
}

/// `lag`/`lead` of integers and strings from disk: any offset (past the
/// partition too), defaults, `IGNORE NULLS`.
#[tokio::test]
async fn offsets_of_a_partition_on_disk_match_memory() {
    let frame = WindowFrame::whole(FrameUnits::Rows);
    let rows = skewed_rows(6_000);
    let spills = Spills {
        every: 1,
        slack: Some(1 << 20),
    };
    for offset in [-100_000, -300, -2, -1, 0, 1, 7, 300, 100_000] {
        let mut functions = Vec::new();
        for default in [None, Some(2)] {
            for ignore_nulls in [false, true] {
                functions.push((lag_lead(offset, default, ignore_nulls), frame));
            }
        }
        for ignore_nulls in [false, true] {
            functions.push((
                WindowFunction::Offset {
                    input: 4,
                    offset,
                    default: Some(4),
                    ignore_nulls,
                    data_type: DataType::Utf8View,
                },
                frame,
            ));
        }
        let window = plan(functions, true);
        let (_, external) = compare_spilling(&rows, 250, &window, spills).await;
        assert!(external >= 1, "lead by {offset}");
    }
}

/// Frames that do not fit fold their rows from the file, with the same
/// values: a frame to the partition's end and one wider than memory.
#[tokio::test]
async fn frames_wider_than_memory_fold_from_the_file() {
    use AggregateFunction::{Count, First, Max, Min, Sum};
    use FrameBound::{CurrentRow, Offset, UnboundedFollowing};
    use FrameUnits::Rows;
    // Few enough for the quadratic fold.
    let rows = skewed_rows(1_500);
    for frame in [
        between(Rows, CurrentRow, UnboundedFollowing),
        between(Rows, Offset(-1_000), Offset(1_000)),
    ] {
        let functions = vec![
            (agg(Sum, Some(3)), frame),
            (agg(Count, None), frame),
            (agg(Min, Some(3)), frame),
            (agg(Max, Some(4)), frame),
            (agg(First { ignore_nulls: true }, Some(4)), frame),
        ];
        let window = plan(functions, true);
        let spills = Spills {
            every: 1,
            slack: Some(96 << 10),
        };
        let counters = compared_spilling(&rows, 50, &window, spills).await;
        assert!(counters.external.load(std::sync::atomic::Ordering::Relaxed) >= 1);
        assert!(
            counters
                .file_folds
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "{frame:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Frames and offset functions from disk equal the in-memory ones over
    /// random rows, split anywhere.
    #[test]
    fn disk_frames_match_memory_on_random_rows(
        rows in big_rows_strategy(),
        split in 1_usize..12,
        which in 0_usize..16,
        offset in -9_i64..9,
        ignore_nulls in any::<bool>(),
        partitioned in any::<bool>(),
    ) {
        let frame = disk_frames()[which];
        let mut functions = over(frame);
        functions.push((lag_lead(offset, Some(2), ignore_nulls), frame));
        let window = plan(functions, partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let spills = Spills { every: 1, slack: Some(256 << 10) };
        rt.block_on(compare_spilling(&rows, split, &window, spills));
    }
}

/// A charge the pool refuses takes back the batches other functions' frames
/// hold; without any to take, it fails. The frame then folds from the file.
#[tokio::test]
async fn a_refused_charge_takes_batches_back_from_a_frame() {
    use super::super::disk::{Framed, Shed};
    let wide = between(
        FrameUnits::Rows,
        FrameBound::Offset(-1000),
        FrameBound::Offset(1000),
    );
    let window = plan(vec![(agg(AggregateFunction::Sum, Some(3)), wide)], false);
    let rows: Vec<Row> = (0..1_000).map(|i| (None, Some(i), i, Some(i))).collect();
    let (_dir, scope) = spill_scope();
    let op = Window::new(schema(), window, true, Some(scope.clone())).unwrap();
    let limit = 1 << 20;
    let (root, ctx) = task(limit);
    let consumer = Arc::clone(ctx.consumer().consumer());
    let mut credit = ctx.consumer().reservation();
    credit.try_grow(1 << 16).unwrap();
    credit.shrink_to_credit(1 << 16);
    let scratch = ctx
        .consumer()
        .reservation()
        .releasing_to_credit()
        .for_spill_scratch();
    let mut writer = scope.writer(schema(), scratch).unwrap();
    for b in rows.chunks(50).map(batch) {
        writer.write(&b).unwrap();
    }
    let file = writer.finish().unwrap();
    let f = &op.env.plan.functions[0];
    let held = Framed::open(&op.env, f, &file, &consumer).unwrap();
    let mut framed = vec![held];
    let Some(frame) = framed[0].as_mut() else {
        panic!("a frame reads the file");
    };
    let before = root.reserved();
    assert!(frame.hold_rows(0, 1_000).unwrap(), "the pool has room");
    let held = root.reserved() - before;

    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    take_all_but(&mut hog, limit, 0);
    // From the pool, as the frames' charges are, not from the readers' credit.
    let mut charge = krishiv_memory::Reservation::new(Arc::clone(consumer.pool()), consumer.id());
    // Half of what the frame holds, above any part of a quantum left over.
    let want = held / 2;
    assert!(
        Shed::none().grow(&mut charge, want).is_err(),
        "nothing to take"
    );
    Shed::all(&mut framed).grow(&mut charge, want).unwrap();
    assert_eq!(charge.size(), want);
    let Some(frame) = framed[0].as_mut() else {
        panic!("a frame reads the file");
    };
    assert!(!frame.hold_rows(0, 1_000).unwrap(), "folds from the file");
    drop((charge, hog, framed));
    ctx.consumer().consumer().release_credit();
    drop(credit);
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
}

/// One refused charge costs no more frames than it needs: the second frame
/// keeps its batches when the first gives back enough.
#[tokio::test]
async fn a_refused_charge_takes_batches_from_as_few_frames_as_it_needs() {
    use super::super::disk::{Framed, Shed};
    let wide = between(
        FrameUnits::Rows,
        FrameBound::Offset(-1000),
        FrameBound::Offset(1000),
    );
    let sum = (agg(AggregateFunction::Sum, Some(3)), wide);
    let window = plan(vec![sum.clone(), sum], false);
    let rows: Vec<Row> = (0..1_000).map(|i| (None, Some(i), i, Some(i))).collect();
    let (_dir, scope) = spill_scope();
    let op = Window::new(schema(), window, true, Some(scope.clone())).unwrap();
    let limit = 1 << 20;
    let (_root, ctx) = task(limit);
    let consumer = Arc::clone(ctx.consumer().consumer());
    let mut credit = ctx.consumer().reservation();
    credit.try_grow(1 << 16).unwrap();
    credit.shrink_to_credit(1 << 16);
    let scratch = ctx
        .consumer()
        .reservation()
        .releasing_to_credit()
        .for_spill_scratch();
    let mut writer = scope.writer(schema(), scratch).unwrap();
    for b in rows.chunks(50).map(batch) {
        writer.write(&b).unwrap();
    }
    let file = writer.finish().unwrap();
    let mut framed: Vec<Option<Framed>> = op
        .env
        .plan
        .functions
        .iter()
        .map(|f| Framed::open(&op.env, f, &file, &consumer).unwrap())
        .collect();
    for frame in framed.iter_mut().flatten() {
        assert!(frame.hold_rows(0, 1_000).unwrap(), "the pool has room");
    }
    let pool = Arc::clone(ctx.query().pool());
    let mut hog = krishiv_memory::Reservation::new(Arc::clone(&pool), pool.register("hog", None));
    take_all_but(&mut hog, limit, 0);
    let mut charge = krishiv_memory::Reservation::new(Arc::clone(consumer.pool()), consumer.id());
    Shed::all(&mut framed).grow(&mut charge, 32 << 10).unwrap();
    let mut holds = framed
        .iter_mut()
        .flatten()
        .map(|frame| frame.hold_rows(0, 1_000).unwrap());
    assert_eq!(holds.next(), Some(false), "the first gave its batches up");
    assert_eq!(holds.next(), Some(true), "the second kept them");
    drop(holds);
    drop((charge, hog, framed));
    ctx.consumer().consumer().release_credit();
}

/// An exact percentile's value map is charged as it grows: distinct values
/// far beyond the memory left are refused with the memory error (the pool
/// empty after), and few distinct values over the same rows complete.
#[tokio::test]
async fn an_exact_percentile_on_disk_is_charged() {
    let percentile = agg(
        AggregateFunction::Percentile {
            percentages: vec![0.5],
            array: false,
            reverse: false,
        },
        Some(3),
    );
    for frame in [
        WindowFrame::whole(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Rows),
        WindowFrame::running(FrameUnits::Range),
    ] {
        let window = plan(vec![(percentile.clone(), frame)], true);
        let spills = Spills {
            every: 1,
            slack: Some(256 << 10),
        };
        let few: Vec<Row> = (0..20_000)
            .map(|i| (Some(0), Some(i), i, Some(i % 11)))
            .collect();
        compare_spilling(&few, 500, &window, spills).await;

        let distinct: Vec<Row> = (0..200_000)
            .map(|i| (Some(0), Some(i), i, Some(i)))
            .collect();
        let (_dir, scope) = spill_scope();
        let op = Window::new(schema(), window, true, Some(scope)).unwrap();
        let limit = 1 << 30;
        let (root, ctx) = task(limit);
        let batches: Vec<Batch> = distinct.chunks(500).map(batch).collect();
        let e = drive_spilling(&op, &ctx, limit, &root, &batches, spills)
            .await
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory, "{e}");
        drop(ctx);
        assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    }
}

/// `RANGE` frames from disk over a descending key with NULLs last, as in
/// memory: both edges move the other way through the keys.
#[tokio::test]
async fn range_frames_on_disk_over_a_descending_key_match_memory() {
    use FrameBound::{CurrentRow, RangeInt, UnboundedFollowing, UnboundedPreceding};
    let rows = skewed_rows(3_000);
    let spills = Spills {
        every: 1,
        slack: Some(4 << 20),
    };
    for frame in [
        between(FrameUnits::Range, RangeInt(-3), RangeInt(2)),
        between(FrameUnits::Range, CurrentRow, RangeInt(4)),
        between(FrameUnits::Range, UnboundedPreceding, RangeInt(1)),
        between(FrameUnits::Range, RangeInt(-2), UnboundedFollowing),
    ] {
        for (descending, nulls_first) in [(true, false), (true, true), (false, false)] {
            let mut window = plan(over(frame), true);
            window.order_by = vec![SortKey {
                column: 1,
                descending,
                nulls_first,
            }];
            let (_, external) = compare_spilling(&rows, 250, &window, spills).await;
            assert!(external >= 1, "{frame:?} {descending} {nulls_first}");
        }
    }
}

/// Float and decimal `RANGE` keys and `FILTER` from disk, as in memory.
#[tokio::test]
async fn float_and_decimal_keys_and_filters_on_disk_match_memory() {
    use FrameBound::{Offset, RangeFloat, RangeInt, UnboundedFollowing};
    use arrow_array::{BooleanArray, Decimal128Array, Float64Array};
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("f", DataType::Float64, true),
        Field::new("d", DataType::Decimal128(12, 2), true),
        Field::new("v", DataType::Int64, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let n = 3_000_i64;
    let sorted_batches = || -> Vec<Batch> {
        // Ascending in both keys, NULLs first, NaN largest for the float.
        let ids: Vec<i64> = (0..n).collect();
        ids.chunks(250)
            .map(|c| {
                let f = Float64Array::from_iter(c.iter().map(|&i| match i {
                    i if i < 20 => None,
                    i if i >= n - 20 => Some(f64::NAN),
                    #[expect(clippy::cast_precision_loss, reason = "small test keys")]
                    i => Some((i / 4) as f64 / 2.0),
                }));
                let d = Decimal128Array::from_iter(
                    c.iter()
                        .map(|&i| (i >= 20).then_some(i128::from(i / 4) * 25)),
                )
                .with_data_type(DataType::Decimal128(12, 2));
                let v = Int64Array::from_iter(c.iter().map(|&i| (i % 13 != 0).then_some(i % 17)));
                let b =
                    BooleanArray::from_iter(c.iter().map(|&i| (i % 11 != 0).then_some(i % 3 != 0)));
                let columns: Vec<ArrayRef> =
                    vec![Arc::new(f), Arc::new(d), Arc::new(v), Arc::new(b)];
                Batch::new(RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
            })
            .collect()
    };
    let sum = || agg(AggregateFunction::Sum, Some(2));
    let filtered = |frame| WindowFunctionPlan {
        function: sum(),
        frame,
        filter: Some(3),
        name: "w".to_owned(),
    };
    let float = between(
        FrameUnits::Range,
        RangeFloat((-1.5_f64).to_bits()),
        RangeFloat(1.0_f64.to_bits()),
    );
    let cases = [
        (0, float),
        (1, between(FrameUnits::Range, RangeInt(-50), RangeInt(75))),
        (1, between(FrameUnits::Rows, Offset(-2), Offset(3))),
        (1, between(FrameUnits::Rows, Offset(-1), UnboundedFollowing)),
        (
            1,
            between(FrameUnits::Rows, FrameBound::UnboundedPreceding, Offset(-1)),
        ),
    ];
    for (column, frame) in cases {
        let window = WindowPlan {
            partition_by: Vec::new(),
            order_by: vec![SortKey::asc(column)],
            functions: vec![filtered(frame)],
        };
        let concat = |batches: &[Batch]| {
            let data: Vec<RecordBatch> = batches.iter().map(|b| b.data().clone()).collect();
            arrow_select::concat::concat_batches(&data[0].schema(), &data).unwrap()
        };
        let input = sorted_batches();
        let plain = Window::new(Arc::clone(&schema), window.clone(), true, None).unwrap();
        let (_r, ctx) = task(1 << 30);
        let want = concat(&drive(&plain, &ctx, &input).unwrap());
        let (_manager, scope) = spill_scope();
        let op = Window::new(Arc::clone(&schema), window, true, Some(scope)).unwrap();
        let limit = 1 << 30;
        let (root, ctx) = task(limit);
        let spills = Spills {
            every: 1,
            slack: Some(256 << 10),
        };
        let got = drive_spilling(&op, &ctx, limit, &root, &input, spills)
            .await
            .unwrap();
        assert_eq!(concat(&got), want, "{frame:?}");
        assert!(
            op.counters
                .external
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "{frame:?}"
        );
        drop((got, ctx));
        assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    }
}

/// A row `FILTER` drops adds nothing, so it can't overflow an ANSI `sum`,
/// from disk as in memory. Frames from the partition's start fold (`Prefix`)
/// and frames that don't fit fold from the file.
#[tokio::test]
async fn rows_a_filter_drops_do_not_overflow_on_disk() {
    use FrameBound::{Offset, UnboundedPreceding};
    use arrow_array::BooleanArray;
    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
        Field::new("b", DataType::Boolean, true),
    ]));
    let n = 3_000_i64;
    let ids: Vec<i64> = (0..n).collect();
    let input: Vec<Batch> = ids
        .chunks(250)
        .map(|c| {
            let k = Int64Array::from_iter_values(c.iter().copied());
            let v = Int64Array::from_iter_values(c.iter().map(|_| i64::MAX));
            let b = BooleanArray::from_iter(c.iter().map(|_| Some(false)));
            let columns: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(v), Arc::new(b)];
            Batch::new(RecordBatch::try_new(Arc::clone(&schema), columns).unwrap())
        })
        .collect();
    for frame in [
        between(FrameUnits::Rows, UnboundedPreceding, Offset(-1)),
        between(FrameUnits::Rows, Offset(-2), Offset(3)),
    ] {
        let window = WindowPlan {
            partition_by: Vec::new(),
            order_by: vec![SortKey::asc(0)],
            functions: vec![WindowFunctionPlan {
                function: agg(AggregateFunction::Sum, Some(1)),
                frame,
                filter: Some(2),
                name: "w".to_owned(),
            }],
        };
        let (_manager, scope) = spill_scope();
        let op = Window::new(Arc::clone(&schema), window, true, Some(scope)).unwrap();
        let limit = 1 << 30;
        let (root, ctx) = task(limit);
        let spills = Spills {
            every: 1,
            slack: Some(256 << 10),
        };
        let got = drive_spilling(&op, &ctx, limit, &root, &input, spills)
            .await
            .unwrap();
        let rows: usize = got.iter().map(Batch::num_rows).sum();
        assert_eq!(rows, 3_000, "{frame:?}");
        for b in &got {
            let w = b.data().columns().last().unwrap();
            assert_eq!(w.null_count(), b.num_rows(), "{frame:?}");
        }
        assert!(
            op.counters
                .external
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "{frame:?}"
        );
        drop((got, ctx));
        assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    }
}

#[test]
fn a_refused_frame_is_asked_for_once_and_the_same_frame_refused_again_fails() {
    use super::super::spill::{Refusal, refusal};
    let refused = |bytes: usize| -> krishiv_common::error::KrishivError {
        (&krishiv_memory::MemoryError::Exhausted {
            requested: bytes,
            available: 0,
        })
            .into()
    };
    let mut asked = None;
    // The first refusal of the frame after 3 consumed asks for the bytes it named.
    assert!(matches!(
        refusal(&mut asked, 3, refused(73_144)),
        Refusal::Ask(73_144)
    ));
    // The same frame refused again fails.
    assert!(matches!(
        refusal(&mut asked, 3, refused(73_144)),
        Refusal::Fail(_)
    ));
    // After a failure the state is clear: a refusal asks again.
    assert!(matches!(
        refusal(&mut asked, 3, refused(8)),
        Refusal::Ask(8)
    ));
    // A footer passed (progress 4) and the next file's first frame refused: a different frame, so it asks.
    assert!(matches!(
        refusal(&mut asked, 4, refused(512)),
        Refusal::Ask(512)
    ));
    // Any other error fails at once.
    let other = internal_for_test();
    assert!(matches!(refusal(&mut asked, 5, other), Refusal::Fail(_)));
}

#[test]
fn a_refused_frame_asks_for_at_least_the_base() {
    use super::super::spill::need_memory;
    use crate::operator::FlushOutput;
    assert!(matches!(
        need_memory(10, 4096),
        FlushOutput::NeedMemory { bytes: 4096 }
    ));
    assert!(matches!(
        need_memory(73_144, 4096),
        FlushOutput::NeedMemory { bytes: 73_144 }
    ));
}
