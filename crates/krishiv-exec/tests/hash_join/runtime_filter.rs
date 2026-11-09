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

/// A large build's shared Bloom array (design P2b-R1): reserved before the
/// build stores a row, kept when the build spills (the form is the Bloom,
/// not the range: the funded array is never dropped for memory), every
/// stored key passes, and its reservation goes with the published filter,
/// released when the filter drops. A build that does not spill publishes
/// the range instead (§2.7, `bloom_unspilled`).
#[tokio::test]
async fn a_spilling_build_keeps_its_shared_bloom() {
    let n = 100_000u32;
    let build: Vec<super::Row> = (0..i64::from(n))
        .map(|i| r(Some(i * 3), Some("k"), i))
        .collect();
    let planned = krishiv_io::runtime_filter::RuntimeBloom::bytes_for(u64::from(n), 0.01);
    let shared = KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom_max_items: 1000,
        bloom_bytes: Some(u32::try_from(planned).unwrap()),
        ..KeyFilterOptions::default()
    };
    assert!(shared.shared_bloom_bytes().is_some());
    for (limit, spills) in [(SPILL_FREE, false), (2 * MIB, true)] {
        let fs = vec![filter(1, 0, shared)];
        let root = root(limit);
        let (_manager, scope) = spill();
        let q = query(&root, 1, ExecConfig::default());
        let case = case(JoinType::Inner, 1, false, false);
        let join = HashJoin::try_new(case.spec(Some(scope)))
            .unwrap()
            .with_runtime_filters(fs.clone())
            .unwrap();
        let plan = Arc::new(HashJoinNode {
            join,
            build: scan(&schema(), batches(&data(&build), 7)),
            probe: scan(&schema(), Vec::new()),
        });
        run(&runner_for(2, limit), plan, &q).await.unwrap();
        assert_eq!(
            metric(&q, "HashJoinBuild", "spilled_partitions") > 0,
            spills,
            "limit {limit}"
        );
        // Published only when the build spilled (design P2b-R1 §2.7).
        assert_eq!(
            metric(&q, "HashJoinBuild", "shared_bloom_bytes"),
            if spills { planned as u64 } else { 0 },
            "limit {limit}"
        );
        assert_eq!(
            metric(&q, "HashJoinBuild", "bloom_unspilled"),
            u64::from(!spills),
            "limit {limit}"
        );
        let k = published(&fs[0]);
        assert_eq!(k.key_count(), u64::from(n), "limit {limit}");
        let keys = Int64Array::from_iter_values((0..i64::from(n)).map(|i| i * 3));
        assert_eq!(k.evaluate(&keys).unwrap().true_count(), n as usize);
        if spills {
            assert_eq!(k.bloom_bytes(), Some(planned), "limit {limit}");
            // A Bloom, not only a superset: the non-keys inside the keys'
            // range (`3i + 1`) mostly fail it (planned at 1%).
            let others = Int64Array::from_iter_values((0..i64::from(n)).map(|i| i * 3 + 1));
            let passed = k.evaluate(&others).unwrap().true_count();
            assert!(passed < n as usize / 20, "{passed} of {n} non-keys pass");
            assert_eq!(k.held_bytes(), planned, "limit {limit}");
            // The filter holds the array's reservation until it goes.
            assert!(
                krishiv_memory::MemoryPool::reserved(&*root) >= planned,
                "limit {limit}"
            );
        } else {
            assert!(k.bloom_bytes().is_none() && k.has_range());
            assert_eq!(k.held_bytes(), 0, "limit {limit}");
        }
        drop((k, fs, q));
        assert_released(&root);
    }
}

/// A pool whose share for the array (1/16 of its limit) is too small for
/// the build's keys publishes the range, counted, never fails the query.
#[tokio::test]
async fn an_undersized_shared_bloom_publishes_the_range() {
    let n = 100_000u32;
    let build: Vec<super::Row> = (0..i64::from(n))
        .map(|i| r(Some(i * 3), Some("k"), i))
        .collect();
    let shared = KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom_max_items: 1000,
        bloom_bytes: Some(512 << 20),
        ..KeyFilterOptions::default()
    };
    let fs = vec![filter(1, 0, shared)];
    let limit = MIB;
    let root = root(limit);
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let join = HashJoin::try_new(case(JoinType::Inner, 1, false, false).spec(Some(scope)))
        .unwrap()
        .with_runtime_filters(fs.clone())
        .unwrap();
    let plan = Arc::new(HashJoinNode {
        join,
        build: scan(&schema(), batches(&data(&build), 7)),
        probe: scan(&schema(), Vec::new()),
    });
    run(&runner_for(2, limit), plan, &q).await.unwrap();
    assert_eq!(metric(&q, "HashJoinBuild", "bloom_undersized"), 1);
    let k = published(&fs[0]);
    assert!(k.bloom_bytes().is_none() && k.has_range());
    assert_eq!(k.held_bytes(), 0);
    drop((k, fs, q));
    assert_released(&root);
}

/// A pool too full for the shared array (held by a consumer that cannot
/// spill): the request never fails the query (design P2b-R1 M2), the array
/// is abandoned and the filter is the range, every stored key passes, and
/// the pool is empty at the end.
#[tokio::test]
async fn an_unfundable_shared_bloom_never_fails_the_query() {
    use krishiv_memory::{MemoryPool, Reservation};
    let n = 3_000u32;
    let build: Vec<super::Row> = (0..i64::from(n))
        .map(|i| r(Some(i * 3), Some("k"), i))
        .collect();
    let shared = KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom_max_items: 100,
        bloom_bytes: Some(512 << 20),
        bloom_max_bytes: 512 << 20,
        ..KeyFilterOptions::default()
    };
    let fs = vec![filter(1, 0, shared)];
    // The array's share is 1 MiB (1/16); leave the build less than that.
    let limit = 16 * MIB;
    let root = root(limit);
    let hog_id = root.register("hog", None);
    let mut hog = Reservation::new(Arc::clone(&root) as Arc<dyn MemoryPool>, hog_id);
    hog.try_grow(limit - 900 * 1024).unwrap();
    let (_manager, scope) = spill();
    let q = query(&root, 1, ExecConfig::default());
    let join = HashJoin::try_new(case(JoinType::Inner, 1, false, false).spec(Some(scope)))
        .unwrap()
        .with_runtime_filters(fs.clone())
        .unwrap();
    let plan = Arc::new(HashJoinNode {
        join,
        build: scan(&schema(), batches(&data(&build), 7)),
        probe: scan(&schema(), Vec::new()),
    });
    run(&runner_for(2, limit), plan, &q).await.unwrap();
    assert_eq!(metric(&q, "HashJoinBuild", "bloom_unfunded"), 1);
    assert_eq!(metric(&q, "HashJoinBuild", "shared_bloom_bytes"), 0);
    let k = published(&fs[0]);
    assert!(k.bloom_bytes().is_none() && k.has_range());
    assert_eq!(k.held_bytes(), 0);
    let keys = Int64Array::from_iter_values((0..i64::from(n)).map(|i| i * 3));
    assert_eq!(k.evaluate(&keys).unwrap().true_count(), n as usize);
    drop((k, fs, q, hog));
    assert_released(&root);
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
