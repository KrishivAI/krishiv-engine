// SPDX-License-Identifier: Apache-2.0
//! Nested-loop joins that return build rows, split across tasks on two
//! simulated workers (design P2-T052b, change log 2026-11-06): right outer,
//! full outer, right semi and right anti return one worker's rows, each
//! build row matched by a probe row of one task only, so on one worker of
//! the two; and a killed task of the join's local stage reruns with the
//! same rows (its bitmap is the committed attempt's).

#![expect(clippy::unwrap_used, clippy::panic, reason = "test code")]

use std::sync::{Arc, Mutex};

use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_exec::physical::{
    ExchangeId, ExchangePlan, ExprEnv, MarksId, NestedLoopPlan, Partitioning, PhysNode,
    PhysicalPlan, RangeNode, Residual, UnmatchedBuildPlan, WorkerContext,
};
use krishiv_exec::planner::{ExplainNode, MemoryTables};
use krishiv_exec::{
    ExecConfig, Executor, JoinType, QueryContext, QueryId, QueryRunner, Scheduler, TokioClock,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr,
    ResolvedFunction,
};
use krishiv_sim::{ClusterConfig, Coordinator, Fault, HostSetup, SimConfig, simulate};
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

/// `probe JOIN build ON probe.id = build.id + 12` over probe `0..20` (behind
/// a round-robin exchange, so each of the 4 tasks holds some probe rows)
/// and a broadcast build `0..10`: build rows 0-7 match one probe row each.
fn join_plan(join_type: JoinType) -> PhysicalPlan {
    let shifted = call(
        Builtin::Add,
        vec![col(2), Arc::new(Expr::literal(Literal::Long(12)))],
        DataType::Long,
    );
    let root = PhysNode::NestedLoopJoin(NestedLoopPlan {
        probe: Box::new(exchange(1, range(20, 1), Partitioning::RoundRobin)),
        build: Box::new(exchange(2, range(10, 2), Partitioning::Broadcast)),
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

/// Hosts without tables: the plans read ranges only.
struct Empty;

impl HostSetup for Empty {
    fn worker(&self) -> BoxFuture<'static, Result<WorkerContext, KrishivError>> {
        Box::pin(async { Ok(WorkerContext::new(Arc::new(MemoryTables::new()) as _)) })
    }
}

/// `plan`'s rows through `scheduler`, as sorted text.
async fn rows(
    seed: u64,
    scheduler: &dyn Scheduler,
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    pool: &Arc<dyn MemoryPool>,
    id: u64,
) -> Result<Vec<String>, KrishivError> {
    let query = QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        ExecConfig {
            seed: Some(seed),
            ..ExecConfig::default()
        },
        pool.child(&format!("q{id}"), None),
    );
    let mut stream = scheduler
        .submit(plan.clone(), worker.clone(), Arc::clone(&query))
        .await?;
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        let (batch, _) = b?.into_parts();
        for r in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|c| format!("{:?}", c.slice(r, 1)))
                .collect();
            out.push(cells.join(" | "));
        }
    }
    drop(stream);
    query.finished().await;
    out.sort();
    Ok(out)
}

/// Per join type, the cluster's rows and one worker's; and the trace.
type Results = Vec<(JoinType, Vec<String>, Vec<String>)>;

fn run(seed: u64, fault: Option<Fault>) -> (Results, Vec<String>) {
    let config = ClusterConfig {
        sim: SimConfig::with_seed(seed),
        workers: 2,
        partitions: 4,
        slots: 2,
        fault,
    };
    let results = Arc::new(Mutex::new(Vec::new()));
    let out = Arc::clone(&results);
    let trace = simulate(
        config,
        Arc::new(Empty),
        move |coordinator: Coordinator, worker: WorkerContext| async move {
            let executor: Arc<dyn Executor> = TokioExecutor::deterministic(config.slots)?;
            let plain = QueryRunner::new(executor);
            let pool: Arc<dyn MemoryPool> =
                TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
            for (i, join_type) in JOINS.into_iter().enumerate() {
                let plan = join_plan(join_type);
                let id = 2 * i as u64;
                let want = rows(seed, &plain, &plan, &worker, &pool, id).await?;
                let got = rows(seed, &coordinator, &plan, &worker, &pool, id + 1).await?;
                out.lock().unwrap().push((join_type, got, want));
            }
            Ok(())
        },
    )
    .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    let results = std::mem::take(&mut *results.lock().unwrap());
    (results, trace)
}

fn assert_equal(results: &Results) {
    assert_eq!(results.len(), JOINS.len());
    for (join_type, got, want) in results {
        let expected = match join_type {
            JoinType::RightOuter => 10,
            JoinType::FullOuter => 22,
            JoinType::RightSemi => 8,
            _ => 2,
        };
        assert_eq!(want.len(), expected, "{join_type:?}");
        assert_eq!(got, want, "{join_type:?}");
    }
}

#[test]
fn split_joins_on_two_workers_equal_one_worker() {
    let (results, trace) = run(5, None);
    assert_equal(&results);
    for w in ["worker-0", "worker-1"] {
        assert!(trace.iter().any(|e| e.contains(w)), "{w}: {trace:?}");
    }
}

#[test]
fn a_killed_local_task_reruns_with_the_same_rows() {
    // Map partition 1 runs only in the join's local stage (every other map
    // stage is one task).
    let (results, trace) = run(
        5,
        Some(Fault {
            map: 1,
            attempt: 0,
            after: 0,
        }),
    );
    assert_equal(&results);
    assert!(
        trace
            .iter()
            .any(|e| e.contains("task 1 attempt 0") && e.contains("failed")),
        "{trace:?}"
    );
    assert!(
        trace
            .iter()
            .any(|e| e.contains("task 1 attempt 1") && e.ends_with("done")),
        "{trace:?}"
    );
}
