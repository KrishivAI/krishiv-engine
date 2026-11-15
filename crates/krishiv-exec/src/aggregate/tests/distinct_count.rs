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
    let new = || distinct_plan_with(scan(s, b.to_vec()), sp.clone(), flags, true).unwrap();
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
