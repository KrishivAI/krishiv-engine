// SPDX-License-Identifier: Apache-2.0
//! Sort-merge join (P2-T056): every join type against the nested-loop oracle
//! the hash join answers to, float and NULL keys, hot keys spilling under a
//! tiny budget, single-match errors, and memory and spill files released.

use std::sync::Arc;

use arrow_array::{Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use krishiv_common::error::ErrorClass;
use krishiv_exec::{ExecConfig, JoinSpec, JoinType, SortMergeJoin, SortMergeJoinNode};
use krishiv_memory::TreePool;
use proptest::prelude::*;

use super::util::*;
use super::{Case, Row, TYPES, any_case, case, data, r, row, schema};

fn merge_join(
    spec: JoinSpec,
    build: Arc<dyn krishiv_exec::ExecNode>,
    probe: Arc<dyn krishiv_exec::ExecNode>,
) -> Arc<dyn krishiv_exec::ExecNode> {
    let join = Arc::new(SortMergeJoin::try_new(spec).unwrap());
    Arc::new(SortMergeJoinNode { join, probe, build })
}

async fn run_merge(
    c: &Case,
    probe: &[Row],
    build: &[Row],
    batch_rows: usize,
    threads: usize,
    limit: usize,
    group_bytes: usize,
) -> (Vec<String>, Arc<krishiv_exec::QueryContext>) {
    let root: Arc<TreePool> = root(limit);
    let (_manager, scope) = spill();
    let config = ExecConfig {
        batch_size: if probe.len() < 1000 { 5 } else { 1024 },
        join_merge_buffer_bytes: group_bytes,
        ..ExecConfig::default()
    };
    let q = query(&root, 1, config);
    let plan = merge_join(
        c.spec(Some(scope.clone())),
        scan(&schema(), batches(&data(build), batch_rows)),
        scan(&schema(), batches(&data(probe), batch_rows)),
    );
    let out = run(&runner_for(threads, limit), plan, &q).await.unwrap();
    assert_released(&root);
    assert_eq!(spill_files(&scope), 0, "spill files are deleted");
    (render(&out), q)
}

#[tokio::test]
async fn join_types_match_the_oracle_on_edge_cases() {
    let probe = vec![
        r(Some(1), Some("a"), 1),
        r(Some(1), Some("a"), 9),
        r(Some(2), None, 3),
        r(None, Some("a"), 4),
        r(None, None, 5),
        r(Some(7), Some("x"), 6),
    ];
    let build = vec![
        r(Some(1), Some("a"), 8),
        r(Some(1), Some("a"), 2),
        r(Some(1), Some("b"), 3),
        r(Some(2), None, 12),
        r(None, None, 10),
        r(None, Some("a"), 11),
    ];
    let all_null = vec![r(None, None, 1), r(None, None, 2)];
    let inputs: [(&[Row], &[Row]); 5] = [
        (&probe, &build),
        (&probe, &[]),
        (&[], &build),
        (&probe, &all_null),
        (&all_null, &build),
    ];
    for jt in TYPES {
        if jt == JoinType::NullAwareAnti {
            continue;
        }
        for keys in [1, 2] {
            for safe in [false, true] {
                for filter in [false, true] {
                    let c = case(jt, keys, safe, filter);
                    for (p, b) in inputs {
                        let (got, _) = run_merge(&c, p, b, 2, 2, 64 * MIB, 0).await;
                        assert_eq!(got, c.oracle(p, b), "{c:?} {} {}", p.len(), b.len());
                    }
                }
            }
        }
    }
}

/// `NOT IN` stays on the hash join: its answer reads the whole build side.
#[test]
fn null_aware_anti_is_refused() {
    let c = case(JoinType::NullAwareAnti, 1, false, false);
    assert!(SortMergeJoin::try_new(c.spec(None)).is_err());
}

fn merge_case() -> impl Strategy<Value = Case> {
    any_case().prop_map(|mut c| {
        if c.join_type == JoinType::NullAwareAnti {
            c.join_type = JoinType::LeftAnti;
        }
        c
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: std::env::var("KRISHIV_TEST_SMJ_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(10_000),
        ..ProptestConfig::default()
    })]
    /// Duplicates on both sides, many-to-many, NULL keys, `=` and `<=>`,
    /// residual conditions, one and two keys, all join types.
    #[test]
    fn random_joins_match_the_nested_loop_oracle(
        c in merge_case(),
        probe in prop::collection::vec(row(6, 4), 0..60),
        build in prop::collection::vec(row(6, 4), 0..60),
        batch_rows in 1usize..17,
        threads in 1usize..4,
    ) {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (got, _) = rt.block_on(run_merge(&c, &probe, &build, batch_rows, threads, 64 * MIB, 0));
        prop_assert_eq!(got, c.oracle(&probe, &build));
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, max_shrink_iters: 0, ..ProptestConfig::default() })]
    /// Larger inputs under a budget that makes both sorts and the groups
    /// spill: the hash join's answer in memory.
    #[test]
    fn random_spilling_joins_equal_the_hash_join(
        c in merge_case(),
        probe in prop::collection::vec(row(40, 200), 3_000..5_000),
        build in prop::collection::vec(row(40, 200), 3_000..5_000),
        batch_rows in 64usize..512,
        threads in 1usize..4,
    ) {
        let mut c = c;
        c.keys = 1;
        c.null_safe.truncate(1);
        // Products of hot keys are too large to compare with a residual-free
        // inner join; the residual keeps them small.
        c.filter = true;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .unwrap();
        let (expect, _) = rt.block_on(super::run_case(&c, &probe, &build, batch_rows, threads, 1 << 30));
        let (got, q) = rt.block_on(run_merge(&c, &probe, &build, batch_rows, threads, 4 * MIB, 1 << 10));
        prop_assert!(got == expect, "the merge join's result differs");
        prop_assert!(metric(&q, "SortMergeJoin", "groups_spilled") > 0);
    }
}

/// One key's group far over the group bound: it spills and every join type
/// still answers as the oracle does.
#[tokio::test(start_paused = true)]
async fn a_hot_key_spills_its_group() {
    let hot: Vec<Row> = (0..4_000)
        .map(|v| r(Some(1), Some("hot"), v % 40))
        .collect();
    let probe = vec![
        r(Some(1), Some("hot"), 3),
        r(Some(2), None, 4),
        r(None, None, 5),
    ];
    for jt in TYPES {
        if jt == JoinType::NullAwareAnti {
            continue;
        }
        for filter in [false, true] {
            let c = case(jt, 1, false, filter);
            let (got, q) = run_merge(&c, &probe, &hot, 256, 2, 2 * MIB, 4 << 10).await;
            assert_eq!(got, c.oracle(&probe, &hot), "{c:?}");
            let probed = !matches!(jt, JoinType::LeftAnti | JoinType::LeftSemi) || filter;
            if probed {
                assert!(metric(&q, "SortMergeJoin", "groups_spilled") > 0, "{c:?}");
            }
        }
    }
}

/// `-0.0` equals `0.0` and NaN equals NaN as join keys, as in Spark.
#[tokio::test]
async fn float_keys_normalize() {
    let s = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Float64, true),
        Field::new("v", DataType::Int64, false),
    ]));
    let side = |k: Vec<Option<f64>>, v: Vec<i64>| {
        RecordBatch::try_new(
            Arc::clone(&s),
            vec![
                Arc::new(Float64Array::from(k)),
                Arc::new(Int64Array::from(v)),
            ],
        )
        .unwrap()
    };
    let probe = side(
        vec![Some(0.0), Some(f64::NAN), None, Some(1.5)],
        vec![1, 2, 3, 4],
    );
    let build = side(
        vec![Some(-0.0), Some(-f64::NAN), None, Some(2.5)],
        vec![5, 6, 7, 8],
    );
    let root = root(64 * MIB);
    let q = query(&root, 1, ExecConfig::default());
    let spec = JoinSpec {
        join_type: JoinType::Inner,
        probe_schema: Arc::clone(&s),
        build_schema: Arc::clone(&s),
        probe_keys: vec![0],
        build_keys: vec![0],
        null_safe: vec![false],
        filter: None,
        single_match: false,
        spill: None,
    };
    let plan = merge_join(
        spec,
        scan(&s, batches(&build, 2)),
        scan(&s, batches(&probe, 2)),
    );
    let out = run(&runner(2), plan, &q).await.unwrap();
    let mut pairs: Vec<(i64, i64)> = out
        .iter()
        .flat_map(|b| {
            let col = |c: usize| {
                arrow_array::cast::AsArray::as_primitive::<arrow_array::types::Int64Type>(
                    b.data().column(c),
                )
                .values()
                .to_vec()
            };
            col(1).into_iter().zip(col(3)).collect::<Vec<_>>()
        })
        .collect();
    pairs.sort_unstable();
    assert_eq!(pairs, [(1, 5), (2, 6)]);
    assert_released(&root);
}

/// A correlated scalar subquery's join fails on a second match, as the hash
/// join does, with and without a residual.
#[tokio::test]
async fn single_match_fails_on_a_second_match() {
    for filter in [false, true] {
        let c = case(JoinType::LeftOuter, 1, false, filter);
        let spec = JoinSpec {
            single_match: true,
            ..c.spec(None)
        };
        let root = root(64 * MIB);
        let q = query(&root, 1, ExecConfig::default());
        let build = [r(Some(1), None, 4), r(Some(1), None, 5)];
        let probe = [r(Some(1), None, 1)];
        let plan = merge_join(
            spec,
            scan(&schema(), batches(&data(&build), 2)),
            scan(&schema(), batches(&data(&probe), 2)),
        );
        let err = run(&runner(2), plan, &q).await.unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::ScalarSubqueryTooManyRows);
        assert_released(&root);
    }
}
