// SPDX-License-Identifier: Apache-2.0
//! Window group limit tests: the limited plan (a sort pruning its batches,
//! then [`WindowGroupLimit`]) keeps exactly the rows the window's filter
//! keeps, ties included; pruning and its switch-off; weights; plan checks.

use super::*;
use crate::operators::SortMetrics;

const FUNCTIONS: [RankLike; 3] = [RankLike::RowNumber, RankLike::Rank, RankLike::DenseRank];

fn window_function(f: RankLike) -> WindowFunction {
    match f {
        RankLike::RowNumber => WindowFunction::RowNumber,
        RankLike::Rank => WindowFunction::Rank,
        RankLike::DenseRank => WindowFunction::DenseRank,
    }
}

fn group_limit(f: RankLike, k: u64, partitioned: bool) -> GroupLimitPlan {
    GroupLimitPlan {
        partition_by: if partitioned { vec![0] } else { Vec::new() },
        order_by: vec![SortKey::asc(1)],
        function: f,
        limit: k,
    }
}

/// Runs `rows` (in batches of `split`) through a sort by `p, o`, the group
/// limit when `limited`, and the window computing `f`; returns the output
/// and the sort's counters. Checks the query pool is empty afterwards.
async fn run_limited(
    rows: &[Row],
    split: usize,
    limit: &GroupLimitPlan,
    limited: bool,
) -> crate::Result<(Vec<Batch>, SortMetrics)> {
    let s = schema();
    let batches: Vec<Batch> = rows.chunks(split.max(1)).map(batch).collect();
    let window = plan(
        vec![(
            window_function(limit.function),
            WindowFrame::running(FrameUnits::Rows),
        )],
        !limit.partition_by.is_empty(),
    );
    let mut keys = limit.partition_keys();
    keys.extend(limit.order_by.iter().copied());
    let mut sort = Sort::new(Arc::clone(&s), &keys, None)?;
    if limited {
        sort = sort.with_group_limit(&keys, limit)?;
    }
    let sort = Arc::new(sort);
    let mut input: Arc<dyn ExecNode> = Arc::new(BreakerNode {
        input: Arc::new(SourceNode(Batches::new(Arc::clone(&s), batches))),
        sink: Arc::clone(&sort) as Arc<dyn crate::Sink>,
        source: Arc::clone(&sort) as Arc<dyn crate::Source>,
        ordered: true,
    });
    if limited {
        input = Arc::new(OperatorNode {
            input,
            op: Arc::new(WindowGroupLimit::new(Arc::clone(&s), limit.clone())?),
        });
    }
    let plan: Arc<dyn ExecNode> = Arc::new(OperatorNode {
        input,
        op: Arc::new(Window::new(Arc::clone(&s), window, true, None)?),
    });
    let runner = QueryRunner::new(TokioExecutor::deterministic(2)?);
    let root = pool(1 << 30);
    let config = ExecConfig {
        batch_size: 4,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(1 << 30)),
    );
    let result = async {
        let mut stream = runner.execute(plan, Arc::clone(&q)).await?;
        let mut out = Vec::new();
        while let Some(b) = stream.next().await {
            out.push(b?);
        }
        Ok(out)
    }
    .await;
    drop(q);
    assert_eq!(root.reserved(), 0, "the query pool is empty at the end");
    result.map(|out| (out, sort.metrics()))
}

/// The rows ranked at most `k`, sorted: `p, o, rank` for `row_number`
/// (which tied row gets a number is arbitrary), `id, rank` otherwise.
fn kept(out: &[Batch], f: RankLike, k: u64) -> Vec<String> {
    let mut rows = Vec::new();
    for b in out {
        let d = b.data();
        let rank = d.column(d.num_columns() - 1).as_primitive::<Int32Type>();
        for i in 0..d.num_rows() {
            let r = u64::try_from(rank.value(i)).unwrap();
            if r > k {
                continue;
            }
            rows.push(match f {
                RankLike::RowNumber => {
                    format!("{} {} {r}", cell(d.column(0), i), cell(d.column(1), i))
                }
                _ => format!("{} {r}", cell(d.column(2), i)),
            });
        }
    }
    rows.sort();
    rows
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Over random partitions, ties, NULL keys and batch splits, the
    /// limited plan's rows are exactly the rows the filter `f <= k` keeps
    /// of the full window, and the limit drops every other row.
    #[test]
    fn the_limit_keeps_exactly_the_filtered_rows(
        rows in rows_strategy(),
        split in 1_usize..9,
        partitioned in any::<bool>(),
        f in 0_usize..3,
        k in 1_u64..4,
    ) {
        let limit = group_limit(FUNCTIONS[f], k, partitioned);
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (full, _) = rt.block_on(run_limited(&rows, split, &limit, false)).unwrap();
        let (limited, _) = rt.block_on(run_limited(&rows, split, &limit, true)).unwrap();
        let want = kept(&full, limit.function, k);
        let total: usize = limited.iter().map(Batch::num_rows).sum();
        prop_assert_eq!(total, want.len());
        prop_assert_eq!(kept(&limited, limit.function, k), want);
    }
}

/// Ties: `rank` keeps every row of a tied group within the limit and skips
/// after it, `dense_rank` counts distinct values, `row_number` keeps `k`.
#[tokio::test]
async fn ties_follow_each_function() {
    // One partition, order keys 1, 1, 2, 2, 2, 3 over two batches.
    let rows: Vec<Row> = [1, 1, 2, 2, 2, 3]
        .iter()
        .enumerate()
        .map(|(i, o)| (Some(0), Some(*o), i64::try_from(i).unwrap(), None))
        .collect();
    let ids = |out: &[Batch]| {
        let mut v: Vec<i64> = out
            .iter()
            .flat_map(|b| {
                b.data()
                    .column(2)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        v.sort_unstable();
        v
    };
    for (f, k, want) in [
        (RankLike::Rank, 2, vec![0, 1]),
        (RankLike::Rank, 3, vec![0, 1, 2, 3, 4]),
        (RankLike::DenseRank, 2, vec![0, 1, 2, 3, 4]),
        (RankLike::DenseRank, 3, vec![0, 1, 2, 3, 4, 5]),
    ] {
        let (out, _) = run_limited(&rows, 3, &group_limit(f, k, true), true)
            .await
            .unwrap();
        assert_eq!(ids(&out), want, "{f:?} <= {k}");
    }
    let (out, _) = run_limited(&rows, 3, &group_limit(RankLike::RowNumber, 3, true), true)
        .await
        .unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 3);
}

/// The sort drops rows past the limit within each batch while that drops
/// a tenth of them or more, and stops looking once its first batches lose
/// less.
#[tokio::test]
async fn the_sort_prunes_while_it_pays() {
    // 40 batches of 4 rows of one partition: 3 of every 4 go.
    let dense: Vec<Row> = (0..160).map(|i| (Some(0), Some(i % 7), i, None)).collect();
    let limit = group_limit(RankLike::RowNumber, 1, true);
    let (out, m) = run_limited(&dense, 4, &limit, true).await.unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 1);
    assert_eq!(m.rows_pruned, 120, "{m:?}");
    // Every row its own partition for the first batches: nothing to drop,
    // so pruning stops before the batches that would lose rows.
    let mut sparse: Vec<Row> = (0..80).map(|i| (Some(i), Some(0), i, None)).collect();
    sparse.extend((80..160).map(|i| (Some(1000), Some(i % 7), i, None)));
    let (out, m) = run_limited(&sparse, 4, &limit, true).await.unwrap();
    assert_eq!(out.iter().map(Batch::num_rows).sum::<usize>(), 81);
    assert_eq!(m.rows_pruned, 0, "{m:?}");
}

#[tokio::test]
async fn weighted_input_is_refused() {
    let (_root, ctx) = task(1 << 30);
    let op = WindowGroupLimit::new(schema(), group_limit(RankLike::Rank, 1, true)).unwrap();
    let mut local = op.local_state(&ctx).unwrap();
    let b = batch(&[(Some(0), Some(0), 0, None)]);
    let weighted =
        Batch::with_weights(b.data().clone(), Weights::Column(Int64Array::from(vec![2]))).unwrap();
    let err = op.execute(&weighted, &mut local).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("window group limit"), "{err}");
}

#[test]
fn invalid_limits_are_rejected() {
    let zero = WindowGroupLimit::new(schema(), group_limit(RankLike::Rank, 0, true)).unwrap_err();
    assert_eq!(zero.error_class(), ErrorClass::InternalError);
    assert!(zero.to_string().contains("of 0 rows"), "{zero}");
    let mut out_of_range = group_limit(RankLike::Rank, 1, true);
    out_of_range.partition_by = vec![9];
    let err = WindowGroupLimit::new(schema(), out_of_range).unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");
    // A sort by other keys than the limit's partitioning and ordering.
    let limit = group_limit(RankLike::Rank, 1, true);
    let err = Sort::new(schema(), &[SortKey::asc(1)], None)
        .unwrap()
        .with_group_limit(&[SortKey::asc(1)], &limit)
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    assert!(err.to_string().contains("other keys"), "{err}");
}

#[tokio::test]
async fn the_operator_reports_its_rows() {
    let (_root, ctx) = task(1 << 30);
    let op = WindowGroupLimit::new(schema(), group_limit(RankLike::RowNumber, 1, false)).unwrap();
    assert_eq!(op.name(), "WindowGroupLimit");
    assert_eq!(op.plan().limit, 1);
    let mut local = op.local_state(&ctx).unwrap();
    let rows: Vec<Row> = (0..3).map(|i| (None, Some(i), i, None)).collect();
    for r in rows.chunks(1) {
        op.execute(&batch(r), &mut local).unwrap();
    }
    assert!(matches!(
        op.execute(&batch(&[]), &mut local).unwrap(),
        OperatorOutput::NeedMoreInput(None)
    ));
    let metrics = op.extra_metrics();
    let count = |name: &str| {
        metrics
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m.clone())
    };
    assert_eq!(count("rows_in"), Some(crate::metrics::Metric::Count(3)));
    assert_eq!(count("rows_kept"), Some(crate::metrics::Metric::Count(1)));
}
