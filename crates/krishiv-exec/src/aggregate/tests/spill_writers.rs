// SPDX-License-Identifier: Apache-2.0
//! Spill writers: a run's partition files written by several threads (X2b,
//! `sink/parts.rs`), the final phase's task runs written at once (X2b',
//! `sink/parallel.rs`), and the timing of run writes.

use super::*;

/// `spill_global` during `finalize` (an arbitration request after the last
/// driver combined) moves the merge to disk; the result is unchanged.
#[tokio::test]
async fn spill_global_during_finalize() {
    let (s, b) = grouped_input(30_000, 20_000);
    let expected = run_rows(single(&s, &b)).await;
    for rows_layout in [false, true] {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], grouped_aggs(&s));
        sp.spill = Some(scope);
        let agg = HashAggregate::new(sp).unwrap();
        let root = pool(1 << 30);
        let config = ExecConfig {
            agg_row_layout: rows_layout,
            ..ExecConfig::default()
        };
        let q = QueryContext::new(
            QueryId(9),
            WorkloadClass::Batch,
            config,
            root.child("q", None),
        );
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
        let ctx = |id, name| {
            let e = Arc::clone(&executor);
            TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
        };
        let sink_ctx = ctx(0, "agg");
        let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
        for part in b.chunks(10) {
            let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
            for cols in part {
                let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
                agg.sink(batch, &mut l, &g).unwrap();
            }
            agg.combine(l, &g).unwrap();
        }
        assert!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap() > 0);
        assert_eq!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap(), 0);
        // A run's write is timed (X2b' step 0).
        let m = Sink::extra_metrics(&*agg);
        let get = |name: &str| m.iter().find_map(|(k, v)| (k == name).then_some(v.clone()));
        assert!(
            matches!(get("spill_write_ns"), Some(crate::metrics::Metric::Count(n)) if n > 0),
            "{m:?}"
        );
        assert!(
            matches!(get("spill_write_max_ns"), Some(crate::metrics::Metric::Max(n)) if n > 0),
            "{m:?}"
        );
        // With room in the pool, the run's partition files were written by
        // several threads (`sink/parts.rs`); the result below is unchanged.
        if std::thread::available_parallelism().map_or(1, std::num::NonZero::get) > 1 {
            let writers = Sink::extra_metrics(&*agg)
                .into_iter()
                .find_map(|(k, v)| (k == "spill_writers").then_some(v));
            assert!(
                matches!(writers, Some(crate::metrics::Metric::Max(n)) if n > 1),
                "{writers:?}"
            );
        }
        agg.finalize(&g, &sink_ctx).await.unwrap();
        assert_eq!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap(), 0);
        let src = ctx(1, "src").with_dependency(Arc::clone(&g));
        let sg = Source::global_state(&*agg, &src).unwrap();
        assert!(agg.max_parallelism(&sg) > 1, "a file per disk task");
        let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
        let mut out = Vec::new();
        while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
            out.push(b);
        }
        assert_eq!(rows(&out), expected);
        assert!(agg.metrics().spill_runs >= 2);
        drop((sl, sg, src, g, sink_ctx));
        assert_eq!(root.reserved(), 0);
    }
}

/// Spill files under `dir`, at any depth (not the manager's lock file).
fn files_under(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .filter_map(std::result::Result::ok)
            .map(|e| {
                let path = e.path();
                if path.is_dir() {
                    files_under(&path)
                } else {
                    usize::from(path.extension().is_some_and(|x| x == "spill"))
                }
            })
            .sum()
    })
}

/// A run's extra writers that hand back their partition (review M3 of
/// X2b: the pool refused their scratch) leave it to the calling thread,
/// which writes it after the others: one file per non-empty partition (no
/// partition written twice, no file left behind), the same result, and
/// nothing reserved after the query. Every extra writer hands back its
/// first partition, or only two of them do and the others write theirs.
#[tokio::test]
async fn handed_back_partitions_are_written_by_the_calling_thread() {
    if std::thread::available_parallelism().map_or(1, std::num::NonZero::get) < 2 {
        return;
    }
    let (s, b) = grouped_input(30_000, 20_000);
    let expected = run_rows(single(&s, &b)).await;
    let mut id = 4_200;
    for handbacks_asked in [usize::MAX, 2] {
        for rows_layout in [false, true] {
            id += 1;
            let query = QueryId(id);
            super::super::sink::HANDBACKS
                .lock()
                .unwrap()
                .push((query, handbacks_asked));
            let (m, scope) = spill_dir();
            let mut sp = spec(&s, vec![0], grouped_aggs(&s));
            sp.spill = Some(scope);
            let agg = HashAggregate::new(sp).unwrap();
            let root = pool(1 << 30);
            let config = ExecConfig {
                agg_row_layout: rows_layout,
                ..ExecConfig::default()
            };
            let q = QueryContext::new(query, WorkloadClass::Batch, config, root.child("q", None));
            let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
            let ctx = |id, name| {
                let e = Arc::clone(&executor);
                TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
            };
            let sink_ctx = ctx(0, "agg");
            let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
            for part in b.chunks(10) {
                let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
                for cols in part {
                    let batch =
                        Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
                    agg.sink(batch, &mut l, &g).unwrap();
                }
                agg.combine(l, &g).unwrap();
            }
            assert!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap() > 0);
            let metric = |name: &str| {
                Sink::extra_metrics(&*agg)
                    .into_iter()
                    .find_map(|(k, v)| (k == name).then_some(v))
            };
            let crate::metrics::Metric::Max(writers) = metric("spill_writers").unwrap() else {
                panic!("spill_writers is a maximum");
            };
            assert!(writers > 1, "extra writers started: {writers}");
            let handbacks = match metric("spill_handbacks") {
                Some(crate::metrics::Metric::Count(n)) => n,
                other => panic!(
                    "asked {handbacks_asked}, rows {rows_layout}: spill_handbacks {other:?}, {writers} writers, {:?}",
                    agg.metrics()
                ),
            };
            // All: each extra writer of each run hands back one partition
            // (the spill wrote a run per combined table); two: exactly two.
            let runs = agg.metrics().spill_runs;
            if handbacks_asked == 2 {
                assert_eq!(handbacks, 2, "{writers} writers, {runs} runs");
            } else {
                assert!(
                    handbacks >= writers - 1 && handbacks <= runs * (writers - 1),
                    "{handbacks} handbacks, {writers} writers, {runs} runs"
                );
            }
            let written = agg.metrics().spill_files;
            assert_eq!(
                files_under(m._dir.path()),
                usize::try_from(written).unwrap(),
                "one file per partition written, none left over"
            );
            agg.finalize(&g, &sink_ctx).await.unwrap();
            let src = ctx(1, "src").with_dependency(Arc::clone(&g));
            let sg = Source::global_state(&*agg, &src).unwrap();
            let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
            let mut out = Vec::new();
            while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
                out.push(b);
            }
            assert_eq!(
                rows(&out),
                expected,
                "asked {handbacks_asked}, rows {rows_layout}"
            );
            drop((sl, sg, src, g, sink_ctx));
            assert_eq!(root.reserved(), 0);
        }
    }
}

/// X2b' (design P2-X2 §11.3): when the final phase spills with tasks left,
/// their runs are written by several threads, each task's run whole, in
/// the tasks' order; a writer the pool refuses scratch puts its task back,
/// with the files it finished, for the next writer. Extra writers write
/// runs; the result, `collect_list`'s order included, is the in-memory
/// final phase's, with or without hand-backs; hand-backs write no file
/// twice (as many files as without); nothing stays reserved.
#[tokio::test]
async fn final_phase_runs_are_written_at_once() {
    // Extra writers take places among the process's (`LIVE_HELPERS`): the
    // test assumes nextest's process per test, where no other spill holds them.
    if std::thread::available_parallelism().map_or(1, std::num::NonZero::get) < 2 {
        return;
    }
    let (s, b) = grouped_input(200_000, 90_000);
    let mode = rowtable::Mode::Fixed { limit: 5_000 };
    let mut aggs = grouped_aggs(&s);
    aggs.push(agg(&AggregateFunction::CollectList, 1, &s));
    let in_memory = HashAggregate::with_row_mode(spec(&s, vec![0], aggs.clone()), mode).unwrap();
    let expected = by_hand(&in_memory, &s, &b, 3).await.rows;
    assert_eq!(sink_metric(&in_memory, "row_final_spills"), 0);
    let mut files = None;
    for handbacks in [0, 1, usize::MAX] {
        // `by_hand` runs query 6.
        {
            let mut asked = super::super::sink::HANDBACKS.lock().unwrap();
            asked.retain(|(q, _)| *q != QueryId(6));
            if handbacks > 0 {
                asked.push((QueryId(6), handbacks));
            }
        }
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], aggs.clone());
        sp.spill = Some(scope);
        // No task gets memory: everything spills at once.
        let agg = HashAggregate::with_final_stall(sp, mode, 0).unwrap();
        let got = by_hand(&agg, &s, &b, 3).await;
        assert_eq!(got.rows, expected, "handbacks {handbacks}");
        assert_eq!(sink_metric(&agg, "row_final_spills"), 1);
        let tasks = sink_metric(&agg, "row_final_tasks");
        assert!(tasks > 2, "{tasks} tasks");
        assert!(
            sink_metric(&agg, "spill_writers") > 1,
            "handbacks {handbacks}"
        );
        let helper_runs = sink_metric(&agg, "spill_helper_runs");
        let back = sink_metric(&agg, "spill_handbacks");
        match handbacks {
            0 => {
                assert_eq!(back, 0);
                assert!(helper_runs > 0, "an extra writer wrote a run");
            }
            1 => assert_eq!(back, 1),
            // Every extra writer hands back its first task: the calling
            // thread writes them all.
            _ => {
                assert!(back >= 1, "{back}");
                assert_eq!(helper_runs, 0);
            }
        }
        // Files: hand-backs neither drop nor repeat any. (Not bytes: the
        // disk phase's output files batch differently with the memory
        // left, which depends on how many writers ran.)
        let written = agg.metrics().spill_files;
        assert_eq!(
            *files.get_or_insert(written),
            written,
            "handbacks {handbacks}"
        );
    }
    super::super::sink::HANDBACKS
        .lock()
        .unwrap()
        .retain(|(q, _)| *q != QueryId(6));
}
