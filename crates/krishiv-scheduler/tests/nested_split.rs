// SPDX-License-Identifier: Apache-2.0
//! Nested-loop joins that return build rows, split across tasks (design
//! P2-T052b, change log 2026-11-06): right outer, full outer, right semi
//! and right anti on 4 tasks of [`StageRunner`] return the rows of one
//! worker ([`QueryRunner`]), with each build row matched by a probe row of
//! only one task.

#![expect(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "test helpers sit outside #[test] fns"
)]

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::StreamExt as _;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{
    ExchangeId, ExchangePlan, ExprEnv, MarksId, NestedLoopPlan, Partitioning, PhysNode,
    PhysicalPlan, PlanSchemas, RangeNode, Residual, UnmatchedBuildPlan, WorkerContext,
    plan_schemas,
};
use krishiv_exec::planner::{ExplainNode, MemoryTables};
use krishiv_exec::{
    ExecConfig, ExecutorConfig, JoinType, QueryContext, QueryId, QueryRunner, Scheduler,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, TreePool};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr,
    ResolvedFunction,
};
use krishiv_scheduler::{StageRunner, fragment, split_nested_loops, stage_tasks};
use krishiv_types::DataType;

const JOINS: [JoinType; 4] = [
    JoinType::RightOuter,
    JoinType::FullOuter,
    JoinType::RightSemi,
    JoinType::RightAnti,
];

fn env() -> ExprEnv {
    ExprEnv {
        ansi: true,
        legacy_statistical_aggregate: false,
        zone: Some(0),
        zone_name: None,
        query_start_micros: 0,
        user: Arc::from("u"),
        map_key_dedup_last_win: false,
        legacy_java_charsets: false,
        legacy_coding_error_action: false,
        legacy_time_parser: false,
    }
}

fn range(end: i64, output: u64) -> PhysNode {
    PhysNode::Range(RangeNode {
        start: 0,
        end,
        step: 1,
        num_slices: None,
        output: AttrId(output),
    })
}

fn exchange(id: u64, input: PhysNode, partitioning: Partitioning) -> PhysNode {
    PhysNode::Exchange(ExchangePlan {
        schema: None,
        id: ExchangeId(id),
        input: Box::new(input),
        partitioning,
        filters: Vec::new(),
    })
}

fn col(id: u64) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        BoundColumn {
            id: AttrId(id),
            name: format!("c{id}"),
            data_type: DataType::Long,
            nullable: false,
        },
    ))))
}

fn call(op: Builtin, args: Vec<Arc<Expr>>, data_type: DataType) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(op),
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn range_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, false)]))
}

/// `probe JOIN build ON probe.id = build.id + 12`: probe `0..20` behind a
/// round-robin exchange (so 4 tasks each hold some probe rows), build
/// `0..10` broadcast. Build rows 0-7 match one probe row each, of one task;
/// 8 and 9 match none.
fn join_plan(join_type: JoinType) -> PhysicalPlan {
    join_of(join_type, 20, 10)
}

/// [`join_plan`] over a probe `0..probe` and a build `0..build`.
fn join_of(join_type: JoinType, probe: i64, build: i64) -> PhysicalPlan {
    let shifted = call(
        Builtin::Add,
        vec![col(2), Arc::new(Expr::literal(Literal::Long(12)))],
        DataType::Long,
    );
    let root = PhysNode::NestedLoopJoin(NestedLoopPlan {
        probe_schema: None,
        probe: Box::new(exchange(1, range(probe, 1), Partitioning::RoundRobin)),
        build: Box::new(exchange(2, range(build, 2), Partitioning::Broadcast)),
        join_type,
        residual: Some(Residual {
            layout: vec![AttrId(1), AttrId(2)],
            conjuncts: vec![call(Builtin::Eq, vec![col(1), shifted], DataType::Boolean)],
        }),
        marks: None,
    });
    let schema = UnmatchedBuildPlan {
        marks: MarksId(0),
        join_type,
        exchange: ExchangeId(2),
        probe_schema: range_schema(),
        build_schema: range_schema(),
    }
    .schema()
    .unwrap();
    PhysicalPlan::new(root, schema, ExplainNode::leaf("Join", ""), env())
}

fn worker() -> WorkerContext {
    WorkerContext::new(Arc::new(MemoryTables::new()) as _)
}

/// `plan`'s rows through `scheduler`, as sorted text.
fn run(
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    pool: &Arc<dyn MemoryPool>,
    id: u64,
) -> Result<Vec<String>, KrishivError> {
    run_on(scheduler, plan, &worker(), pool, id)
}

/// [`run`] on `worker`.
fn run_on(
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    pool: &Arc<dyn MemoryPool>,
    id: u64,
) -> Result<Vec<String>, KrishivError> {
    let query = QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        ExecConfig::default(),
        pool.child(&format!("q{id}"), None),
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = rt.block_on(async {
        let mut stream = scheduler
            .submit(plan.clone(), worker.clone(), Arc::clone(&query))
            .await?;
        let mut out = Vec::new();
        while let Some(b) = stream.next().await {
            out.push(b?.into_parts().0);
        }
        query.finished().await;
        Ok::<_, KrishivError>(out)
    })?;
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| format!("{:?}", c.slice(r, 1)))
                .collect();
            rows.push(cells.join(" | "));
        }
    }
    rows.sort();
    Ok(rows)
}

fn split(plan: &PhysicalPlan) -> (PhysicalPlan, PlanSchemas) {
    let mut schemas = plan_schemas(plan, &worker()).unwrap();
    let split = split_nested_loops(plan, &mut schemas).unwrap();
    (split, schemas)
}

#[test]
fn the_split_runs_the_join_on_every_probe_task_and_its_build_rows_once() {
    for join_type in JOINS {
        let (plan, schemas) = split(&join_plan(join_type));
        let stages = fragment(&plan, &schemas.exchanges).unwrap();
        let tasks = stage_tasks(&stages, 4, None).unwrap();
        // The local half: several tasks, publishing a bitmap.
        let local = stages
            .iter()
            .find(|s| !krishiv_scheduler::published_marks(&s.plan).is_empty())
            .unwrap();
        assert!(!local.single, "{join_type:?}");
        assert_eq!(tasks[local.id].len(), 4, "{join_type:?}");
        // The build rows: one task, after the local half, reading the
        // broadcast the join reads.
        let rest = stages
            .iter()
            .find(|s| {
                let mut stack = vec![s.plan.root()];
                std::iter::from_fn(|| {
                    let n = stack.pop()?;
                    stack.extend(n.children());
                    Some(matches!(n, PhysNode::UnmatchedBuild(_)))
                })
                .any(|b| b)
            })
            .unwrap();
        assert!(rest.single && rest.id > local.id, "{join_type:?}");
        assert_eq!(tasks[rest.id].len(), 1);
        let broadcast = ExchangeId(2);
        assert!(rest.inputs.iter().any(|i| i.exchange == broadcast));
        assert!(local.inputs.iter().any(|i| i.exchange == broadcast));
        // The broadcast is cut once.
        assert_eq!(
            stages
                .iter()
                .filter(|s| s.output.as_ref().is_some_and(|(e, _)| *e == broadcast))
                .count(),
            1
        );
    }
}

#[test]
fn split_joins_on_four_tasks_equal_one_worker() {
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    let stages = StageRunner::new(Arc::clone(&executor) as _, 4).unwrap();
    let one = StageRunner::new(Arc::clone(&executor) as _, 1).unwrap();
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let mut id = 0;
    for join_type in JOINS {
        let plan = join_plan(join_type);
        id += 1;
        let want = run(&plain, &plan, &root, id).unwrap();
        let expected = match join_type {
            // 8 matched pairs, 2 unmatched build rows (and, full outer, 12
            // unmatched probe rows).
            JoinType::RightOuter => 10,
            JoinType::FullOuter => 22,
            JoinType::RightSemi => 8,
            _ => 2,
        };
        assert_eq!(want.len(), expected, "{join_type:?}: {want:?}");
        for runner in [&stages, &one] {
            id += 1;
            let got = run(runner, &plan, &root, id).unwrap();
            assert_eq!(got, want, "{join_type:?}");
        }
    }
    assert_eq!(root.reserved(), 0);
}

#[test]
fn a_spilled_build_publishes_its_bitmap_after_the_last_chunk() {
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    // Two tasks each hold and spill the whole build. Four under a 4 MiB
    // pool oversubscribe it: each spilled nested loop sizes its chunks
    // from the pool's limit (a follow-up, not the split's).
    let stages = StageRunner::new(Arc::clone(&executor) as _, 2).unwrap();
    let big: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let small: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
        limit: Some(4 << 20),
        ..MemoryConfig::default()
    });
    let dir = tempfile::tempdir().unwrap();
    let spill = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let spilling = worker().with_spill(spill.scope().unwrap());
    let mut id = 100;
    for join_type in [JoinType::FullOuter, JoinType::RightSemi] {
        // A 1,000,000-row build (8 MB of keys) does not fit the 4 MiB pool.
        let plan = join_of(join_type, 16, 1_000_000);
        id += 1;
        let want = run(&plain, &plan, &big, id).unwrap();
        assert_eq!(
            want.len(),
            if join_type == JoinType::RightSemi {
                4
            } else {
                1_000_012
            }
        );
        id += 1;
        assert!(
            run(&plain, &plan, &small, id).is_err(),
            "the build fits the small pool: nothing spills"
        );
        id += 1;
        let got = run_on(&stages, &plan, &spilling, &small, id).unwrap();
        assert_eq!(got, want, "{join_type:?}");
    }
    assert_eq!(small.reserved(), 0);
    assert_eq!(big.reserved(), 0);
}

#[test]
fn a_join_whose_rows_round_robin_cannot_route_stays_whole() {
    let plan = join_plan(JoinType::FullOuter);
    let mut schemas = plan_schemas(&plan, &worker()).unwrap();
    // As if the probe carried an array column: no routing hash for it.
    let list = ArrowType::List(Arc::new(Field::new("item", ArrowType::Int64, true)));
    schemas.nested_loop_probes.insert(
        ExchangeId(2),
        Arc::new(Schema::new(vec![Field::new("l", list, true)])),
    );
    let before = schemas.exchanges.clone();
    let same = split_nested_loops(&plan, &mut schemas).unwrap();
    assert_eq!(same, plan);
    assert_eq!(schemas.exchanges, before);
    // A join that returns no build rows is never split.
    let mut inner = plan_schemas(&join_plan(JoinType::FullOuter), &worker()).unwrap();
    let left = {
        let PhysNode::NestedLoopJoin(mut j) = plan.root().clone() else {
            unreachable!()
        };
        j.join_type = JoinType::LeftOuter;
        PhysicalPlan::new(
            PhysNode::NestedLoopJoin(j),
            SchemaRef::clone(plan.schema()),
            ExplainNode::leaf("Join", ""),
            env(),
        )
    };
    assert_eq!(split_nested_loops(&left, &mut inner).unwrap(), left);
}

#[test]
fn a_task_without_probe_rows_still_publishes_its_bitmap() {
    // Two probe rows over four tasks: two tasks read none, and no build row
    // matches.
    let executor = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
    let plain = QueryRunner::new(Arc::clone(&executor) as _);
    let stages = StageRunner::new(Arc::clone(&executor) as _, 4).unwrap();
    let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let mut id = 200;
    for join_type in JOINS {
        let plan = join_of(join_type, 2, 10);
        id += 1;
        let want = run(&plain, &plan, &root, id).unwrap();
        let expected = match join_type {
            JoinType::RightOuter | JoinType::RightAnti => 10,
            JoinType::FullOuter => 12,
            _ => 0,
        };
        assert_eq!(want.len(), expected, "{join_type:?}");
        id += 1;
        assert_eq!(
            run(&stages, &plan, &root, id).unwrap(),
            want,
            "{join_type:?}"
        );
    }
    assert_eq!(root.reserved(), 0);
}
