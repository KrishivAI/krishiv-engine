// SPDX-License-Identifier: Apache-2.0
//! `count(DISTINCT v)` counted at the merge (`aggregate/distinct.rs`)
//! against the two aggregations in series (`distinct_plan_with(.., false)`):
//! the same rows for NULL keys and values, empty and one-key inputs, many
//! keys, float keys with `-0.0`/`0.0`/NaN, a forced parallel merge, and a
//! tiny-memory spill. `exec_with` checks the pool is empty after each run.

use super::*;
use crate::aggregate::distinct::DistinctCount;
use crate::aggregate::distinct_plan_with;
use arrow_array::Float64Array;

/// Generated `(k, v, w)` batches: `n` rows, `keys` keys, every 7th key and
/// every 5th value NULL, `w` a small measure.
fn gen_input(n: i64, keys: i64, per_batch: usize) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("v", DataType::Int64),
        ("w", DataType::Int64),
    ]);
    let rows: Vec<i64> = (0..n).collect();
    let b = rows
        .chunks(per_batch)
        .map(|c| {
            let k: Vec<Option<i64>> = c
                .iter()
                .map(|&i| {
                    let k = (i * 7919) % keys.max(1);
                    (k % 7 != 3).then_some(k)
                })
                .collect();
            let v: Vec<Option<i64>> = c
                .iter()
                .map(|&i| ((i % 5) != 0).then_some((i * 31) % 97))
                .collect();
            let w: Vec<Option<i64>> = c.iter().map(|&i| Some(i % 11)).collect();
            vec![i64s(&k), i64s(&v), i64s(&w)]
        })
        .collect();
    (s, b)
}

/// `[sum(w), count(*), avg(w),] count(DISTINCT v) group by k` with the
/// count at `at` (`plain`: the plain aggregates too).
fn shape(s: &SchemaRef, plain: bool, at: usize) -> (AggregateSpec, Vec<bool>) {
    let mut aggs = Vec::new();
    if plain {
        aggs.push(agg(&Sum, 2, s));
        aggs.push(AggregateExpr::count_star("c"));
        aggs.push(agg(&Avg, 2, s));
    }
    let at = at.min(aggs.len());
    aggs.insert(at, agg(&Count, 1, s));
    let mut flags = vec![false; aggs.len()];
    flags[at] = true;
    (spec(s, vec![0], aggs), flags)
}

async fn both(
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
    sp: &AggregateSpec,
    flags: &[bool],
    o: Opts,
) -> (Vec<String>, Vec<String>) {
    let old = || distinct_plan_with(scan(s, b.to_vec()), sp.clone(), flags, false).unwrap();
    // The counted plan itself, whatever the slots (`distinct_plan_with`
    // keeps the old plan on one).
    let new = || {
        crate::aggregate::distinct::DistinctCountNode::new(
            scan(s, b.to_vec()),
            DistinctCount::new(sp, flags).unwrap(),
        ) as Arc<dyn ExecNode>
    };
    let (want, _) = exec_each(old, o).await;
    let (got, _) = exec_each(new, o).await;
    (got, want)
}

#[tokio::test]
async fn counts_at_the_merge_as_the_second_aggregation_does() {
    for (n, keys, batch) in [
        (0, 10, 64),
        (500, 1, 64),
        (3_000, 50, 128),
        (20_000, 4_000, 512),
    ] {
        let (s, b) = gen_input(n, keys, batch);
        for (plain, at) in [(true, 3), (true, 0), (true, 1), (false, 0)] {
            let (sp, flags) = shape(&s, plain, at);
            let (got, want) = both(&s, &b, &sp, &flags, Opts::default()).await;
            assert_eq!(got, want, "n {n}, keys {keys}, plain {plain}, at {at}");
            if n == 0 {
                assert!(got.is_empty());
            }
        }
    }
}

#[tokio::test]
async fn one_thread_and_many_drivers_agree() {
    let (s, b) = gen_input(20_000, 2_000, 256);
    let (sp, flags) = shape(&s, true, 3);
    for slots in [1, 4] {
        let o = Opts {
            slots,
            seed: None,
            ..Opts::default()
        };
        let (got, want) = both(&s, &b, &sp, &flags, o).await;
        assert_eq!(got, want, "{slots} slots");
    }
}

/// `distinct_plan_with(.., true)` counts at the merge only when a pipeline
/// runs more than one driver: one slot runs the two aggregations, four
/// run `DistinctCount`; the rows are the same.
#[tokio::test]
async fn one_slot_keeps_the_two_aggregations() {
    let (s, b) = gen_input(20_000, 2_000, 256);
    let (sp, flags) = shape(&s, true, 3);
    let mut seen = Vec::new();
    for slots in [1, 4] {
        let o = Opts {
            slots,
            seed: None,
            ..Opts::default()
        };
        let plan = distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, true).unwrap();
        let (out, m) = exec_with(plan, o, true).await.unwrap();
        let names: Vec<&str> = m.operators.values().map(|e| e.name.as_ref()).collect();
        let counted = names.iter().filter(|n| **n == "DistinctCount").count();
        let series = names.iter().filter(|n| **n == "HashAggregate").count();
        seen.push((slots, counted, series));
        let (want, _) = exec_each(
            || distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, false).unwrap(),
            o,
        )
        .await;
        assert_eq!(rows(&out), want, "{slots} slots");
    }
    // A breaker is named once as a sink and once as a source.
    assert_eq!(seen, vec![(1, 0, 4), (4, 2, 0)]);
}

#[tokio::test]
async fn float_keys_group_signed_zeros_and_nans() {
    let s = schema(vec![("k", DataType::Float64), ("v", DataType::Int64)]);
    let k: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(0.0),
        Some(-0.0),
        Some(f64::NAN),
        Some(-f64::NAN),
        None,
        Some(1.5),
        Some(-0.0),
        Some(f64::NAN),
    ]));
    let v = i64s(&[
        Some(1),
        Some(2),
        Some(1),
        Some(1),
        None,
        Some(3),
        Some(2),
        Some(9),
    ]);
    let b = vec![vec![k, v]];
    for plain in [false, true] {
        let mut aggs = vec![agg(&Count, 1, &s)];
        if plain {
            aggs.push(AggregateExpr::count_star("c"));
        }
        let mut flags = vec![false; aggs.len()];
        flags[0] = true;
        let sp = spec(&s, vec![0], aggs);
        let (got, want) = both(&s, &b, &sp, &flags, Opts::default()).await;
        assert_eq!(got, want, "plain {plain}");
        assert_eq!(got.len(), 4, "0.0, NaN, NULL, 1.5: {got:?}");
    }
}

/// Small tables widen early and merge in several parallel tasks, each
/// counting its partitions in the merge hook.
#[tokio::test]
async fn parallel_merge_tasks_count_their_partitions() {
    let (s, b) = gen_input(30_000, 3_000, 256);
    for plain in [false, true] {
        let (sp, flags) = shape(&s, plain, 0);
        let old = distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, false).unwrap();
        let want = rows(&exec_with(old, Opts::default(), true).await.unwrap().0);
        let op = DistinctCount::with_pairs(&sp, &flags, |p| {
            HashAggregate::with_row_mode(p, rowtable::Mode::Grow)
        })
        .unwrap();
        let plan = crate::aggregate::distinct::DistinctCountNode::new(scan(&s, b.clone()), op);
        let o = Opts {
            slots: 4,
            ..Opts::default()
        };
        let before =
            crate::aggregate::distinct::HOOK_PARTS.load(std::sync::atomic::Ordering::Relaxed);
        let got = rows(&exec_with(plan, o, true).await.unwrap().0);
        assert_eq!(got, want, "plain {plain}");
        let after =
            crate::aggregate::distinct::HOOK_PARTS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after > before + 1,
            "the merge tasks counted {} partitions",
            after - before
        );
    }
}

#[tokio::test]
async fn string_keys() {
    let s = schema(vec![("k", DataType::Utf8View), ("v", DataType::Int64)]);
    let b: Vec<Vec<ArrayRef>> = (0..8)
        .map(|j| {
            let k: ArrayRef = Arc::new(StringViewArray::from_iter((0..500).map(|i| {
                let x = (i * 13 + j * 7) % 300;
                (x % 11 != 0).then(|| format!("a fairly long search phrase {x:04}"))
            })));
            let v = i64s(
                &(0..500)
                    .map(|i| Some((i * 17 + j) % 40))
                    .collect::<Vec<_>>(),
            );
            vec![k, v]
        })
        .collect();
    let (sp, flags) = {
        let aggs = vec![agg(&Count, 1, &s)];
        (spec(&s, vec![0], aggs), vec![true])
    };
    let (got, want) = both(&s, &b, &sp, &flags, Opts::default()).await;
    assert_eq!(got, want);
}

/// A pool far smaller than the tables: both tables spill, the pairs reach
/// the output on disk and are counted in `finalize`; the rows are those of
/// the plan with memory to spare.
#[tokio::test]
async fn tiny_memory_spills_and_counts_the_same() {
    let (s, b) = gen_input(60_000, 20_000, 1_024);
    for plain in [true, false] {
        let (sp, flags) = shape(&s, plain, 1);
        let (want, _) = exec_each(
            || distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, false).unwrap(),
            Opts::default(),
        )
        .await;
        let (_m, scope) = spill_dir();
        let mut spilled = sp.clone();
        spilled.spill = Some(scope);
        let make = || {
            let op = DistinctCount::new(&spilled, &flags).unwrap();
            let pairs = Arc::clone(op.pairs());
            let plan = crate::aggregate::distinct::DistinctCountNode::new(scan(&s, b.clone()), op)
                as Arc<dyn ExecNode>;
            (plan, pairs)
        };
        let o = Opts {
            limit: 3 << 20,
            ..Opts::default()
        };
        for layout in [false, true] {
            let (plan, pairs) = make();
            let got = rows(&exec_with(plan, o, layout).await.unwrap().0);
            assert_eq!(got, want, "plain {plain}, layout {layout}");
            assert!(
                pairs.metrics().spill_runs > 0,
                "plain {plain}, layout {layout}"
            );
        }
    }
}

/// The new plan is picked only for a single `count(DISTINCT)` under a
/// `GROUP BY`: other specs keep the two aggregations.
#[test]
fn covers_only_a_grouped_single_distinct_count() {
    use crate::aggregate::distinct::covers;
    let (s, _) = gen_input(0, 1, 1);
    let (sp, flags) = shape(&s, true, 3);
    assert!(covers(&sp, &flags));
    let global = spec(&s, vec![], vec![agg(&Count, 1, &s)]);
    assert!(!covers(&global, &[true]));
    let sum = spec(&s, vec![0], vec![agg(&Sum, 1, &s)]);
    assert!(!covers(&sum, &[true]));
    let two = spec(&s, vec![0], vec![agg(&Count, 1, &s), agg(&Count, 1, &s)]);
    assert!(!covers(&two, &[true, true]));
}

/// The counter's reservations refused part-way through the merge tasks
/// (after `n` grants, while counting a partition or while merging one into
/// the shared shards, or its chunk scratch): the partition refused is left
/// to `finalize`, whole, and the rows equal the old plan's; the pool is
/// empty after each run.
#[tokio::test]
async fn refused_count_memory_mid_merge_counts_the_same() {
    use crate::aggregate::distinct::{
        DistinctCountNode, PENDING_MERGES, REFUSE_AFTER, SCRATCH_REFUSALS,
    };
    use std::sync::atomic::Ordering::Relaxed;
    let (s, b) = gen_input(30_000, 3_000, 256);
    let (sp, flags) = shape(&s, true, 1);
    let old = distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, false).unwrap();
    let want = rows(&exec_with(old, Opts::default(), true).await.unwrap().0);
    let o = Opts {
        slots: 4,
        ..Opts::default()
    };
    for grants in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55] {
        REFUSE_AFTER.store(grants, Relaxed);
        let op = DistinctCount::with_pairs(&sp, &flags, |p| {
            HashAggregate::with_row_mode(p, rowtable::Mode::Grow)
        })
        .unwrap();
        let plan = DistinctCountNode::new(scan(&s, b.clone()), op);
        let got = rows(&exec_with(plan, o, true).await.unwrap().0);
        REFUSE_AFTER.store(usize::MAX, Relaxed);
        assert_eq!(got, want, "refused after {grants} grants");
    }
    assert!(PENDING_MERGES.load(Relaxed) > 0, "no merge was refused");
    assert!(
        SCRATCH_REFUSALS.load(Relaxed) > 0,
        "no chunk scratch was refused"
    );
}

/// The rows of `batches` in the order they came.
fn in_order(batches: &[Batch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let d = b.data();
            (0..d.num_rows()).map(move |r| {
                let cells: Vec<String> = d.columns().iter().map(|c| cell(c, r)).collect();
                cells.join("|")
            })
        })
        .collect()
}

/// Under `krishiv.deterministicOrder` the rows come in the same order on
/// every run and equal today's plan's order, with and without plain
/// aggregates, whatever the slots: the counted plan reads its counter's
/// shards in the order parallel tasks filled them, so the two aggregations
/// run there.
#[tokio::test]
async fn deterministic_order_keeps_the_two_aggregations_order() {
    let (s, b) = gen_input(20_000, 2_000, 256);
    for plain in [false, true] {
        let (sp, flags) = shape(&s, plain, 0);
        for slots in [1, 4] {
            let o = Opts {
                slots,
                deterministic: true,
                ..Opts::default()
            };
            let series =
                distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, false).unwrap();
            let want = in_order(&exec_with(series, o, true).await.unwrap().0);
            for run in 0..3 {
                let plan =
                    distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, true).unwrap();
                let (out, m) = exec_with(plan, o, true).await.unwrap();
                assert!(
                    m.operators.values().all(|e| e.name != "DistinctCount"),
                    "plain {plain}, {slots} slots: counted at the merge"
                );
                assert_eq!(
                    in_order(&out),
                    want,
                    "plain {plain}, {slots} slots, run {run}"
                );
            }
        }
    }
}

/// Once `finalize` has counted the pairs that reached the output, their
/// memory is gone: without plain aggregates the query holds only the
/// counter's kept bytes (one table without duplicates on one slot, the
/// merge tasks' leftovers on four), within the reservation quanta. Each
/// output piece leaves the result when its last chunk is claimed
/// (`sink.rs` `InMemory`), so nothing of the pairs stays reserved.
#[tokio::test]
async fn finalize_releases_the_counted_pairs() {
    use crate::aggregate::distinct::{DistinctCountNode, FINAL_HELD, FINAL_POOL};
    use std::sync::atomic::Ordering::Relaxed;
    let (s, b) = gen_input(40_000, 4_000, 512);
    let (sp, flags) = shape(&s, false, 0);
    for slots in [1, 4] {
        let o = Opts {
            slots,
            ..Opts::default()
        };
        for layout in [false, true] {
            let plan = DistinctCountNode::new(
                scan(&s, b.clone()),
                DistinctCount::new(&sp, &flags).unwrap(),
            );
            exec_with(plan, o, layout).await.unwrap();
            let (pool, held) = (FINAL_POOL.load(Relaxed), FINAL_HELD.load(Relaxed));
            assert!(held > 0, "{slots} slots, layout {layout}");
            // Within the consumers' 16 KiB reservation quanta; the pairs
            // (about 40,000 of them) would be hundreds of KiB.
            assert!(
                pool.saturating_sub(held) < 64 << 10,
                "{slots} slots, layout {layout}: pool {pool}, counter {held}"
            );
        }
    }
}

/// `count(DISTINCT v, w)`: a pair counts only when every argument is
/// non-NULL, so a NULL in only one of them drops it.
#[tokio::test]
async fn several_arguments_with_a_null_in_one() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("v", DataType::Int64),
        ("w", DataType::Int64),
    ]);
    let n = 3_000_i64;
    let k = i64s(&(0..n).map(|i| Some(i % 37)).collect::<Vec<_>>());
    let v = i64s(
        &(0..n)
            .map(|i| (i % 3 != 0).then_some(i % 5))
            .collect::<Vec<_>>(),
    );
    let w = i64s(
        &(0..n)
            .map(|i| (i % 4 != 0).then_some(i % 7))
            .collect::<Vec<_>>(),
    );
    let b = [vec![k, v, w]];
    let mut count = agg(&Count, 1, &s);
    count.args = vec![1, 2];
    for plain in [false, true] {
        let mut aggs = vec![count.clone()];
        if plain {
            aggs.push(agg(&Sum, 2, &s));
        }
        let mut flags = vec![false; aggs.len()];
        flags[0] = true;
        let sp = spec(&s, vec![0], aggs);
        for slots in [1, 4] {
            let o = Opts {
                slots,
                ..Opts::default()
            };
            let (got, want) = both(&s, &b, &sp, &flags, o).await;
            assert_eq!(got, want, "plain {plain}, {slots} slots");
            assert_eq!(got.len(), 37);
        }
    }
}

/// Float distinct values: `0.0` and `-0.0` are one value, every NaN is one
/// value, NULL does not count.
#[tokio::test]
async fn float_values_count_signed_zeros_and_nans_once() {
    let s = schema(vec![("k", DataType::Int64), ("v", DataType::Float64)]);
    let vals = [
        Some(0.0),
        Some(-0.0),
        Some(f64::NAN),
        Some(-f64::NAN),
        None,
        Some(1.5),
    ];
    let n = 600_usize;
    let k = i64s(&(0..n).map(|i| Some((i % 3) as i64)).collect::<Vec<_>>());
    let v: ArrayRef = Arc::new(Float64Array::from(
        (0..n)
            .map(|i| vals[(i / 3) % vals.len()])
            .collect::<Vec<_>>(),
    ));
    let b = vec![vec![k, v]];
    for plain in [false, true] {
        let mut aggs = vec![agg(&Count, 1, &s)];
        if plain {
            aggs.push(AggregateExpr::count_star("c"));
        }
        let mut flags = vec![false; aggs.len()];
        flags[0] = true;
        let sp = spec(&s, vec![0], aggs);
        let (got, want) = both(&s, &b, &sp, &flags, Opts::default()).await;
        assert_eq!(got, want, "plain {plain}");
        // 0.0, NaN and 1.5 for every key.
        assert!(
            got.iter().all(|r| r.split('|').nth(1) == Some("3")),
            "{got:?}"
        );
    }
}

/// The pairs' final phase runs short after its first task finished (and
/// counted its partitions): the finished tasks' tables are flushed to the
/// output file, the rest spills; what reaches the output is counted in
/// `finalize`. Rows equal the old plan's; the pool is empty after.
#[tokio::test]
async fn a_flushed_final_phase_counts_the_same() {
    let (s, b) = gen_input(60_000, 20_000, 512);
    for plain in [false, true] {
        let (sp, flags) = shape(&s, plain, 0);
        let (want, _) = exec_each(
            || distinct_plan_with(scan(&s, b.clone()), sp.clone(), &flags, false).unwrap(),
            Opts::default(),
        )
        .await;
        let (_m, scope) = spill_dir();
        let mut spilled = sp.clone();
        spilled.spill = Some(scope);
        let op = DistinctCount::with_pairs(&spilled, &flags, |p| {
            HashAggregate::with_final_stall(p, rowtable::Mode::Fixed { limit: 5_000 }, 1)
        })
        .unwrap();
        let pairs = Arc::clone(op.pairs());
        let plan = crate::aggregate::distinct::DistinctCountNode::new(scan(&s, b.clone()), op);
        // One slot: the merge runs its tasks one at a time, so the second
        // finds the first finished (its driver table, past the fixed
        // index's limit, has duplicates and goes through the merge tasks).
        let o = Opts {
            slots: 1,
            ..Opts::default()
        };
        let hooked =
            crate::aggregate::distinct::HOOK_PARTS.load(std::sync::atomic::Ordering::Relaxed);
        let got = rows(&exec_with(plan, o, true).await.unwrap().0);
        assert_eq!(got, want, "plain {plain}");
        assert_eq!(sink_metric(&pairs, "row_final_flushes"), 1, "plain {plain}");
        assert_eq!(sink_metric(&pairs, "row_final_spills"), 1, "plain {plain}");
        assert!(
            crate::aggregate::distinct::HOOK_PARTS.load(std::sync::atomic::Ordering::Relaxed)
                > hooked,
            "plain {plain}: the first task counted nothing"
        );
    }
}

/// Cancelled while a merge task counts its partitions, or while `finalize`
/// counts what reached the output: the query fails with the cancellation
/// and the pool is empty (`exec_with`).
#[tokio::test]
async fn cancelled_while_counting_leaves_the_pool_empty() {
    use crate::aggregate::distinct::{CANCEL_IN_COUNT, CANCEL_IN_FINALIZE, DistinctCountNode};
    use std::sync::atomic::Ordering::Relaxed;
    let (s, b) = gen_input(40_000, 4_000, 256);
    for plain in [false, true] {
        let (sp, flags) = shape(&s, plain, 0);
        for (hook, slots) in [(&CANCEL_IN_FINALIZE, 1), (&CANCEL_IN_COUNT, 4)] {
            hook.store(true, Relaxed);
            let op = DistinctCount::with_pairs(&sp, &flags, |p| {
                HashAggregate::with_row_mode(p, rowtable::Mode::Grow)
            })
            .unwrap();
            let plan = DistinctCountNode::new(scan(&s, b.clone()), op);
            let o = Opts {
                slots,
                ..Opts::default()
            };
            let out = exec_with(plan, o, true).await;
            hook.store(false, Relaxed);
            let e = out
                .err()
                .unwrap_or_else(|| panic!("plain {plain}, {slots} slots: not cancelled"));
            assert!(
                e.to_string().to_lowercase().contains("cancel"),
                "plain {plain}, {slots} slots: {e}"
            );
        }
    }
}
