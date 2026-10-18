// SPDX-License-Identifier: Apache-2.0
//! The nested-loop join against a brute-force reference: every join type,
//! empty and multi-batch build sides, NULL conditions, and probe batches
//! large enough to be split into several chunks.
#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "test helpers sit outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::ErrorClass;
use krishiv_exec::testing::Scan;
use krishiv_exec::{
    ExecConfig, ExecutorConfig, FnPredicate, JoinType, MemoryScan, NestedLoopJoin,
    NestedLoopJoinNode, NestedLoopSpec, Predicate, QueryContext, QueryId, QueryRunner,
    ScanPushdown as _, ScanPushdowns, Scheduler as _, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool as _, TreePool};
use krishiv_types::batch::{Batch, Weights};

fn schema(name: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)]))
}

/// Batches of `values` (`None` is NULL), `per` rows each.
fn batches(name: &str, values: &[Option<i64>], per: usize) -> Vec<Batch> {
    values
        .chunks(per.max(1))
        .map(|c| {
            let a = Int64Array::from(c.to_vec());
            Batch::new(RecordBatch::try_new(schema(name), vec![Arc::new(a)]).unwrap())
        })
        .collect()
}

/// `p < b`, NULL when either is NULL.
fn less() -> Arc<dyn Predicate> {
    Arc::new(FnPredicate(|b: &RecordBatch| {
        let (p, q) = (
            b.column(0).as_primitive::<Int64Type>(),
            b.column(1).as_primitive::<Int64Type>(),
        );
        Ok(p.iter()
            .zip(q.iter())
            .map(|(x, y)| Some(x? < y?))
            .collect::<BooleanArray>())
    }))
}

fn run(
    join_type: JoinType,
    filter: Option<Arc<dyn Predicate>>,
    probe: Vec<Batch>,
    build: Vec<Batch>,
) -> Result<Vec<String>, krishiv_common::error::KrishivError> {
    let source = |name: &str, b: Vec<Batch>| {
        MemoryScan::try_new(schema(name), b)
            .unwrap()
            .build(ScanPushdowns::default())
            .unwrap()
    };
    let join = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type,
        probe_schema: schema("p"),
        build_schema: schema("b"),
        filter,
    })?;
    let plan = Arc::new(NestedLoopJoinNode {
        join,
        build: Arc::new(Scan(source("b", build))),
        probe: Arc::new(Scan(source("p", probe))),
    });
    let executor = TokioExecutor::new(&ExecutorConfig::default())?;
    let runner = QueryRunner::new(executor);
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        root.child("q", None),
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let out = rt.block_on(async {
        let mut stream = runner.submit(plan, Arc::clone(&query)).await?;
        let mut rows = Vec::new();
        while let Some(b) = stream.next().await {
            let b = b?.into_parts().0;
            for r in 0..b.num_rows() {
                let cells: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| {
                        let c = c.as_primitive::<Int64Type>();
                        if c.is_null(r) {
                            "N".to_owned()
                        } else {
                            c.value(r).to_string()
                        }
                    })
                    .collect();
                rows.push(cells.join(","));
            }
        }
        query.finished().await;
        Ok::<_, krishiv_common::error::KrishivError>(rows)
    });
    assert_eq!(root.reserved(), 0, "memory is released");
    let mut out = out?;
    out.sort();
    Ok(out)
}

/// The reference: pairs by brute force.
fn expected(
    join_type: JoinType,
    cond: bool,
    probe: &[Option<i64>],
    build: &[Option<i64>],
) -> Vec<String> {
    let s = |v: Option<i64>| v.map_or_else(|| "N".to_owned(), |v| v.to_string());
    let hit =
        |p: Option<i64>, b: Option<i64>| !cond || matches!((p, b), (Some(p), Some(b)) if p < b);
    let mut out = Vec::new();
    for &p in probe {
        let matches: Vec<Option<i64>> = build.iter().copied().filter(|&b| hit(p, b)).collect();
        match join_type {
            JoinType::Inner => out.extend(matches.iter().map(|&b| format!("{},{}", s(p), s(b)))),
            JoinType::LeftOuter if matches.is_empty() => out.push(format!("{},N", s(p))),
            JoinType::LeftOuter => {
                out.extend(matches.iter().map(|&b| format!("{},{}", s(p), s(b))));
            }
            JoinType::LeftSemi if !matches.is_empty() => out.push(s(p)),
            JoinType::LeftAnti if matches.is_empty() => out.push(s(p)),
            _ => {}
        }
    }
    out.sort();
    out
}

#[test]
fn every_join_type_matches_the_reference() {
    let probe: Vec<Option<i64>> = (0..3000).map(|i| (i % 7 != 3).then_some(i % 50)).collect();
    let build: Vec<Option<i64>> = (0..40).map(|i| (i % 9 != 4).then_some(i)).collect();
    for join_type in [
        JoinType::Inner,
        JoinType::LeftOuter,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
    ] {
        for (cond, per_build) in [(true, 7), (true, 40), (false, 13)] {
            let got = run(
                join_type,
                cond.then(less),
                batches("p", &probe, 1024),
                batches("b", &build, per_build),
            )
            .unwrap();
            assert_eq!(
                got,
                expected(join_type, cond, &probe, &build),
                "{join_type:?} cond={cond} per_build={per_build}"
            );
        }
        // An empty build side.
        let got = run(
            join_type,
            Some(less()),
            batches("p", &probe[..10], 4),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            got,
            expected(join_type, true, &probe[..10], &[]),
            "{join_type:?} empty"
        );
    }
}

#[test]
fn null_aware_and_weighted_inputs_are_refused() {
    let e = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: JoinType::NullAwareAnti,
        probe_schema: schema("p"),
        build_schema: schema("b"),
        filter: None,
    })
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    let weighted = |name: &str| {
        let b = &batches(name, &[Some(1), Some(2)], 2)[0];
        vec![Batch::with_weights(b.data().clone(), Weights::Const(2)).unwrap()]
    };
    for (probe, build) in [
        (weighted("p"), batches("b", &[Some(5)], 1)),
        (batches("p", &[Some(1)], 1), weighted("b")),
    ] {
        let e = run(JoinType::Inner, None, probe, build).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }
    let j = NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: JoinType::LeftOuter,
        probe_schema: schema("p"),
        build_schema: schema("b"),
        filter: None,
    })
    .unwrap();
    assert!(j.schema().field(1).is_nullable());
    assert!(format!("{j:?}").contains("LeftOuter"));
}
