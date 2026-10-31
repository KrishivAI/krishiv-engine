// SPDX-License-Identifier: Apache-2.0
//! The hash join build as a runtime filter producer (design P2-T163): what
//! it publishes for `=` and `<=>` keys, that a spilled build still covers
//! every key it stored, that a failed build disables its filters, and the
//! join types that cannot publish.

use std::sync::Arc;

use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use krishiv_common::error::ErrorClass;
use krishiv_exec::testing::Pipe;
use krishiv_exec::{
    ExecConfig, Filter, FnPredicate, HashJoin, HashJoinNode, JoinRuntimeFilter, JoinType,
};
use krishiv_io::{
    ColumnPath, FilterId, FilterSnapshot, KeyFilter, KeyFilterOptions, RuntimeFilter,
};

use super::util::*;
use super::{Case, case, data, r, schema};

fn filter(id: u64, key: usize, options: KeyFilterOptions) -> JoinRuntimeFilter {
    let types = [arrow_schema::DataType::Int64, arrow_schema::DataType::Utf8];
    JoinRuntimeFilter {
        key,
        filter: Arc::new(RuntimeFilter::new(
            FilterId(id),
            ColumnPath::new(["k1", "k2"][key]),
            types[key].clone(),
        )),
        options,
    }
}

fn published(f: &JoinRuntimeFilter) -> Arc<KeyFilter> {
    match f.filter.published().snapshot.clone() {
        FilterSnapshot::Final(k) => k,
        other => unreachable!("not final: {other:?}"),
    }
}

/// Runs `case` with `filters` over `build` (the probe is empty: only the
/// build's publication matters here).
async fn run_build(
    case: &Case,
    build: &[super::Row],
    filters: Vec<JoinRuntimeFilter>,
    limit: usize,
) -> krishiv_exec::Result<()> {
    let root = root(limit);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let join = HashJoin::try_new(case.spec(Some(scope)))?.with_runtime_filters(filters)?;
    let plan = Arc::new(HashJoinNode {
        join,
        build: scan(&schema(), batches(&data(build), 7)),
        probe: scan(&schema(), Vec::new()),
    });
    let out = run(&runner_for(2, limit), plan, &q).await.map(drop);
    assert_released(&root);
    out
}

/// A scan's pipeline waits for the build only for the filters that ask for it.
#[test]
fn a_build_reports_only_the_filters_a_scan_waits_for() {
    let (_manager, scope) = spill();
    let waits = KeyFilterOptions {
        scan_waits: true,
        ..KeyFilterOptions::default()
    };
    let join = HashJoin::try_new(case(JoinType::Inner, 2, false, false).spec(Some(scope)))
        .unwrap()
        .with_runtime_filters(vec![
            filter(1, 0, waits),
            filter(2, 1, KeyFilterOptions::default()),
        ])
        .unwrap();
    assert_eq!(join.build_sink().published_filters(), [FilterId(1)]);
}

#[tokio::test]
async fn builds_publish_the_keys_they_store() {
    let build = vec![
        r(Some(3), Some("c"), 1),
        r(Some(1), None, 2),
        r(None, Some("a"), 3),
        r(Some(2), Some("b"), 4),
    ];
    // `=` keys: a row with a NULL in either key is not stored.
    let fs = vec![
        filter(1, 0, KeyFilterOptions::default()),
        filter(2, 1, KeyFilterOptions::default()),
    ];
    run_build(
        &case(JoinType::Inner, 2, false, false),
        &build,
        fs.clone(),
        SPILL_FREE,
    )
    .await
    .unwrap();
    let k1 = published(&fs[0]);
    assert_eq!(k1.value_count(), Some(2), "keys 3 and 2");
    assert!(!k1.nulls_pass());
    let probe = Int64Array::from(vec![Some(1), Some(2), Some(3), None]);
    let pass: Vec<bool> = k1
        .evaluate(&probe)
        .unwrap()
        .iter()
        .map(|b| b == Some(true))
        .collect();
    assert_eq!(pass, [false, true, true, false]);
    let k2 = published(&fs[1]);
    let strings = StringArray::from(vec!["a", "b", "c"]);
    let pass: Vec<bool> = k2
        .evaluate(&strings)
        .unwrap()
        .iter()
        .map(|b| b == Some(true))
        .collect();
    assert_eq!(pass, [false, true, true]);
    // `<=>` keys: NULL is stored, and passes.
    let fs = vec![filter(3, 0, KeyFilterOptions::default())];
    run_build(
        &case(JoinType::LeftSemi, 2, true, false),
        &build,
        fs.clone(),
        SPILL_FREE,
    )
    .await
    .unwrap();
    let k = published(&fs[0]);
    assert!(k.nulls_pass() && k.value_count() == Some(3));
    // An empty build publishes an empty filter.
    let fs = vec![filter(4, 0, KeyFilterOptions::default())];
    run_build(
        &case(JoinType::Inner, 1, false, false),
        &[],
        fs.clone(),
        SPILL_FREE,
    )
    .await
    .unwrap();
    assert!(published(&fs[0]).is_empty());
}

/// A build that spills (and repartitions) still publishes every key it
/// stored: partials see rows before they go to disk. Under memory
/// pressure the Bloom filter's hashes may be dropped, never the range.
#[tokio::test]
async fn spilled_builds_cover_every_key() {
    let build: Vec<super::Row> = (0..200_000).map(|i| r(Some(i * 3), Some("k"), i)).collect();
    let bloom = KeyFilterOptions {
        max_values: 0,
        ..KeyFilterOptions::default()
    };
    for limit in [SPILL_FREE, 2 * MIB] {
        let fs = vec![filter(1, 0, bloom)];
        run_build(
            &case(JoinType::Inner, 1, false, false),
            &build,
            fs.clone(),
            limit,
        )
        .await
        .unwrap();
        let k = published(&fs[0]);
        assert!(k.has_range());
        // A `NeedMemory` retry sinks a batch again: its keys count once.
        assert_eq!(k.key_count(), 200_000, "limit {limit}");
        if limit == SPILL_FREE {
            assert!(k.bloom_bytes().is_some());
        }
        let keys = Int64Array::from_iter_values((0..200_000).map(|i| i * 3));
        let pass = k.evaluate(&keys).unwrap();
        assert_eq!(pass.true_count(), 200_000, "limit {limit}");
    }
}

#[tokio::test]
async fn failed_builds_disable_their_filters() {
    let root = root(SPILL_FREE);
    let q = query(&root, 1, ExecConfig::default());
    let fs = vec![filter(1, 0, KeyFilterOptions::default())];
    let join = HashJoin::try_new(case(JoinType::Inner, 1, false, false).spec(None))
        .unwrap()
        .with_runtime_filters(fs.clone())
        .unwrap();
    let failing = FnPredicate(|_: &RecordBatch| -> krishiv_exec::Result<BooleanArray> {
        Err(krishiv_common::error::InternalError {
            message: "build input failed".to_owned(),
        }
        .into())
    });
    let build = Arc::new(Pipe(
        scan(&schema(), batches(&data(&[r(Some(1), None, 1)]), 1)),
        Filter::new(schema(), Arc::new(failing)),
    ));
    let plan = Arc::new(HashJoinNode {
        join,
        build,
        probe: scan(&schema(), Vec::new()),
    });
    let e = run(&runner(2), plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(matches!(
        fs[0].filter.published().snapshot,
        FilterSnapshot::Disabled(_)
    ));
    assert_released(&root);
}

#[test]
fn only_inner_and_semi_joins_publish() {
    let f = || vec![filter(1, 0, KeyFilterOptions::default())];
    for t in [
        JoinType::LeftOuter,
        JoinType::LeftAnti,
        JoinType::NullAwareAnti,
    ] {
        let join = HashJoin::try_new(case(t, 1, false, false).spec(None)).unwrap();
        assert!(join.with_runtime_filters(f()).is_err(), "{t:?}");
    }
    let join = HashJoin::try_new(case(JoinType::Inner, 1, false, false).spec(None)).unwrap();
    let mut wrong = f();
    wrong[0].key = 1;
    assert!(
        join.clone().with_runtime_filters(wrong).is_err(),
        "Utf8 key, Int64 filter"
    );
    let mut out = f();
    out[0].key = 5;
    assert!(join.clone().with_runtime_filters(out).is_err());
    assert!(
        join.clone().with_runtime_filters(f()).is_err(),
        "shared join"
    );
    drop(join);
    let join = HashJoin::try_new(case(JoinType::LeftSemi, 1, false, false).spec(None))
        .unwrap()
        .with_runtime_filters(f())
        .unwrap();
    assert_eq!(join.runtime_filters().len(), 1);
}
