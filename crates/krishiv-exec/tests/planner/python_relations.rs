// SPDX-License-Identifier: Apache-2.0
//! The Python UDF relation nodes planned and run over a fake runtime whose
//! tasks echo or count the frames they get (design P3-T085 §3.1, §4): rows,
//! groups, cogroup sides, one row per aggregate group, output not aligned to
//! input, the refusals, and a large skewed group through the spilling sort.

use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::future::BoxFuture;
use futures::task::AtomicWaker;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_exec::python_relation::{Frame, PreparedRelationUdf, RelationCallSpec, UdfTask};
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionKind, GeneratorKind, Grouping, LogicalPlan, Plan,
    PlanNode, PythonEvalType, PythonUdfRef, ResolvedExpr, ResolvedFunction,
};
use krishiv_types::DataType;

use super::{Env, Runnable, cell, env, run};

/// What a fake task does, by the UDF's name.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// Each batch back once.
    Echo,
    /// Each batch back twice.
    Twice,
    /// Each batch back eight times.
    Burst,
    /// One `BIGINT` per group (per partition without groups): its rows.
    Count,
}

#[derive(Debug, Default)]
struct Runtime {
    /// Every frame: `B<side>:<rows>`, `G`, `E`.
    log: Arc<std::sync::Mutex<Vec<String>>>,
    /// Each task's own frames.
    per_task: Arc<std::sync::Mutex<Vec<Log>>>,
    /// Per `Batch` frame of a grouped kind: `<side>:<key values>:<rows>`
    /// (`-` for the key of a zero-row frame).
    keys: Log,
    /// Sends a slow task made wait (`slow_*` UDFs).
    waits: Arc<std::sync::atomic::AtomicUsize>,
}

type Log = Arc<std::sync::Mutex<Vec<String>>>;

impl UdfRuntime for Runtime {
    fn prepare(&self, spec: &UdfCallSpec) -> Result<Arc<dyn PreparedUdf>, KrishivError> {
        Err(krishiv_common::error::InternalError {
            message: format!("no scalar {}", spec.name),
        }
        .into())
    }

    fn prepare_relation(
        &self,
        spec: &RelationCallSpec,
    ) -> Result<Arc<dyn PreparedRelationUdf>, KrishivError> {
        let slow = spec.name.starts_with("slow_");
        let mode = match spec.name.trim_start_matches("slow_") {
            "echo" => Mode::Echo,
            "twice" => Mode::Twice,
            "burst" => Mode::Burst,
            _ => Mode::Count,
        };
        assert_eq!(spec.input_names.len(), spec.input_types.len(), "{spec:?}");
        for (names, types) in spec.input_names.iter().zip(&spec.input_types) {
            assert_eq!(names.len(), types.len(), "{spec:?}");
            assert!(names.len() >= spec.key_count, "{spec:?}");
        }
        assert_eq!(
            spec.arity + spec.key_count,
            spec.input_types[0].len(),
            "{spec:?}"
        );
        Ok(Arc::new(Prepared {
            mode,
            output: Arc::clone(&spec.output_arrow),
            log: Arc::clone(&self.log),
            per_task: Arc::clone(&self.per_task),
            keys: Arc::clone(&self.keys),
            waits: slow.then(|| Arc::clone(&self.waits)),
            names: spec.input_names.clone(),
            key_count: spec.key_count,
        }))
    }
}

/// The fake's prepared site.
struct Prepared {
    mode: Mode,
    output: arrow_schema::SchemaRef,
    log: Arc<std::sync::Mutex<Vec<String>>>,
    per_task: Arc<std::sync::Mutex<Vec<Log>>>,
    keys: Log,
    waits: Option<Arc<std::sync::atomic::AtomicUsize>>,
    names: Vec<Vec<String>>,
    key_count: usize,
}

impl PreparedRelationUdf for Prepared {
    fn open(&self) -> Result<Box<dyn UdfTask>, KrishivError> {
        let own: Log = Arc::default();
        self.per_task.lock().unwrap().push(Arc::clone(&own));
        Ok(Box::new(Task {
            own,
            mode: self.mode,
            output: Arc::clone(&self.output),
            log: Arc::clone(&self.log),
            keys: Arc::clone(&self.keys),
            waits: self.waits.clone(),
            names: self.names.clone(),
            key_count: self.key_count,
            state: std::sync::Mutex::new(State::default()),
            ready: AtomicWaker::new(),
        }))
    }
}

#[derive(Default)]
struct State {
    out: VecDeque<RecordBatch>,
    rows: i64,
    groups: usize,
    ended: bool,
}

struct Task {
    own: Log,
    mode: Mode,
    output: arrow_schema::SchemaRef,
    log: Arc<std::sync::Mutex<Vec<String>>>,
    keys: Log,
    waits: Option<Arc<std::sync::atomic::AtomicUsize>>,
    names: Vec<Vec<String>>,
    key_count: usize,
    state: std::sync::Mutex<State>,
    ready: AtomicWaker,
}

impl Task {
    /// Checks a frame against the spec: its columns are the side's, led by
    /// the keys, whose values are the same on every row; logs them.
    fn check(&self, side: u8, batch: &RecordBatch) {
        let names: Vec<&str> = batch
            .schema_ref()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, self.names[usize::from(side)], "frame columns");
        if self.key_count == 0 {
            return;
        }
        let key = |r: usize| {
            (0..self.key_count)
                .map(|c| cell(batch.column(c).as_ref(), r))
                .collect::<Vec<_>>()
                .join(",")
        };
        let first = if batch.num_rows() == 0 {
            "-".to_owned()
        } else {
            key(0)
        };
        for r in 1..batch.num_rows() {
            assert_eq!(key(r), first, "one key per group");
        }
        self.keys
            .lock()
            .unwrap()
            .push(format!("{side}:{first}:{}", batch.num_rows()));
    }

    fn push(&self, entry: String) {
        self.own.lock().unwrap().push(entry.clone());
        self.log.lock().unwrap().push(entry);
    }

    fn count(&self, n: i64) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(&self.output),
            vec![Arc::new(Int64Array::from(vec![n])) as ArrayRef],
        )
        .unwrap()
    }
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, Result<(), KrishivError>> {
        Box::pin(async move {
            if let Some(waits) = &self.waits {
                // A slow task: each frame waits one poll, so the operator's
                // window fills and the driver blocks until it is woken.
                let mut first = true;
                poll_fn(|cx| {
                    if std::mem::take(&mut first) {
                        waits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    } else {
                        Poll::Ready(())
                    }
                })
                .await;
            }
            let mut s = self.state.lock().unwrap();
            match frame {
                Frame::Batch { side, batch } => {
                    self.check(side, &batch);
                    self.push(format!("B{side}:{}", batch.num_rows()));
                    s.rows += i64::try_from(batch.num_rows()).unwrap();
                    let back =
                        RecordBatch::try_new(Arc::clone(&self.output), batch.columns().to_vec());
                    match self.mode {
                        Mode::Echo => s.out.push_back(back.unwrap()),
                        Mode::Twice | Mode::Burst => {
                            let back = back.unwrap();
                            let n = if self.mode == Mode::Twice { 2 } else { 8 };
                            for _ in 0..n {
                                s.out.push_back(back.clone());
                            }
                        }
                        Mode::Count => {}
                    }
                }
                Frame::EndGroup => {
                    self.push("G".to_owned());
                    s.groups += 1;
                    if self.mode == Mode::Count {
                        let n = std::mem::take(&mut s.rows);
                        let b = self.count(n);
                        s.out.push_back(b);
                    }
                }
                Frame::End => {
                    self.push("E".to_owned());
                    if self.mode == Mode::Count && s.groups == 0 {
                        let n = s.rows;
                        let b = self.count(n);
                        s.out.push_back(b);
                    }
                    s.ended = true;
                }
            }
            drop(s);
            self.ready.wake();
            Ok(())
        })
    }

    fn next(&self) -> BoxFuture<'_, Result<Option<RecordBatch>, KrishivError>> {
        Box::pin(poll_fn(|cx| {
            self.ready.register(cx.waker());
            let mut s = self.state.lock().unwrap();
            match s.out.pop_front() {
                Some(b) => Poll::Ready(Ok(Some(b))),
                None if s.ended => Poll::Ready(Ok(None)),
                None => Poll::Pending,
            }
        }))
    }
}

fn udf(name: &str, eval_type: PythonEvalType) -> PythonUdfRef {
    PythonUdfRef {
        name: name.to_owned(),
        version: 1,
        eval_type,
        thread_safe: true,
    }
}

fn col(c: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
        c.clone(),
    ))))
}

fn bound(id: u64, name: &str, data_type: DataType) -> BoundColumn {
    BoundColumn {
        id: AttrId(id),
        name: name.to_owned(),
        data_type,
        nullable: true,
    }
}

/// The analyzed `sql` and its columns.
fn input(env: &Env, sql: &str) -> (Arc<LogicalPlan>, Vec<BoundColumn>) {
    let plan = env.analyze(sql);
    let root = Arc::clone(plan.root());
    let cols = root.resolved_output().unwrap();
    (root, cols)
}

/// `n BIGINT` (id 900).
fn counted() -> Vec<BoundColumn> {
    vec![bound(900, "n", DataType::Long)]
}

/// `node` planned and run with `runtime` (none: no runtime).
fn run_with(
    env: &Env,
    node: LogicalPlan,
    runtime: Option<Arc<Runtime>>,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let plan = krishiv_optimizer::optimize(&Plan::try_new(node)?)?;
    let physical = krishiv_exec::planner::PhysicalPlanner::plan(
        &krishiv_exec::planner::DefaultPlanner,
        &plan,
        &env.cx(),
    )?;
    let mut runnable: Runnable = env.runnable(physical);
    if let Some(rt) = runtime {
        runnable = runnable.with_worker(|w| w.with_python_udfs(rt));
    }
    run(&runnable)
}

fn log_of(rt: &Arc<Runtime>) -> Vec<String> {
    rt.log.lock().unwrap().clone()
}

/// Every cell of `batches` as `a|b` lines, sorted.
fn lines(batches: &[RecordBatch]) -> Vec<String> {
    let mut rows = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

#[test]
fn a_map_keeps_every_row() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::MapPartitions {
        input,
        udf: udf("echo", PythonEvalType::MapArrowIter),
        output: vec![
            bound(900, "x", cols[0].data_type.clone()),
            bound(901, "y", cols[1].data_type.clone()),
        ],
        barrier: false,
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    let rows = lines(&out);
    assert_eq!(rows.len(), 10, "{rows:?}");
    assert!(rows.contains(&"NULL|30".to_owned()), "{rows:?}");
    assert!(log_of(&rt).iter().any(|e| e == "E"));
}

#[test]
fn output_rows_are_not_aligned_to_input_rows() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::MapPartitions {
        input,
        udf: udf("twice", PythonEvalType::MapPandasIter),
        output: vec![
            bound(900, "x", cols[0].data_type.clone()),
            bound(901, "y", cols[1].data_type.clone()),
        ],
        barrier: false,
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    assert_eq!(lines(&out).len(), 20);
}

#[test]
fn a_grouped_map_sees_each_key_once_with_all_its_rows() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input,
        keys: vec![col(&cols[0])],
        udf: udf("count", PythonEvalType::GroupedMapPandas),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    // `a` is 0..9 with 3 and 7 NULL: eight keys of one row, NULL of two.
    let rows = lines(&out);
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert_eq!(rows.iter().filter(|r| *r == "1").count(), 8, "{rows:?}");
    assert!(rows.contains(&"2".to_owned()), "{rows:?}");
    let log = log_of(&rt);
    assert_eq!(log.iter().filter(|e| *e == "G").count(), 9, "{log:?}");
}

/// A slow task through the real driver: the window fills, the driver
/// pauses on `Blocked` and is woken, and every row arrives once.
#[test]
fn a_slow_task_blocks_the_driver_and_loses_nothing() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input,
        keys: vec![col(&cols[0])],
        udf: udf("slow_count", PythonEvalType::GroupedMapPandas),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    let rows = lines(&out);
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert_eq!(rows.iter().filter(|r| *r == "1").count(), 8, "{rows:?}");
    let frames = log_of(&rt).len();
    assert_eq!(
        rt.waits.load(std::sync::atomic::Ordering::SeqCst),
        frames,
        "every frame waited"
    );
}

#[test]
fn a_grouped_map_without_keys_is_one_group() {
    let env = env();
    let (input, _) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input,
        keys: Vec::new(),
        udf: udf("count", PythonEvalType::GroupedMapArrow),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    assert_eq!(lines(&out), ["10"]);
}

#[test]
fn a_cogroup_gives_each_key_both_sides_and_a_one_sided_key_its_side() {
    let env = env();
    let (left, lc) = input(&env, "SELECT a, b FROM t");
    let (right, rc) = input(&env, "SELECT a AS c, s FROM t WHERE a < 2");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::CoGroupMap {
        left,
        right,
        left_keys: vec![col(&lc[0])],
        right_keys: vec![col(&rc[0])],
        udf: udf("count", PythonEvalType::CogroupedMapArrow),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    // Keys 0 and 1 have a row on each side; the others the left one only.
    let rows = lines(&out);
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert_eq!(rows.iter().filter(|r| *r == "2").count(), 3, "{rows:?}");
    let log = log_of(&rt);
    let count = |e: &str| log.iter().filter(|x| *x == e).count();
    assert_eq!(count("B1:1"), 2, "{log:?}");
    // The seven keys on the left only get a zero-row right side.
    assert_eq!(count("B1:0"), 7, "{log:?}");
    assert_eq!(count("B0:0"), 0, "{log:?}");
    // Within each task, every group is a left frame, a right frame, then
    // `EndGroup`.
    for own in rt.per_task.lock().unwrap().iter() {
        let own = own.lock().unwrap();
        for group in own
            .split(|e| e == "G")
            .filter(|g| *g != ["E"] && !g.is_empty())
        {
            assert!(group[0].starts_with("B0"), "{own:?}");
            assert!(group.last().unwrap().starts_with("B1"), "{own:?}");
        }
    }
    // The frames lead with the key: the right side's keys are 0 and 1.
    let keys = rt.keys.lock().unwrap().clone();
    let mut right: Vec<&String> = keys
        .iter()
        .filter(|k| k.starts_with("1:") && !k.ends_with(":0"))
        .collect();
    right.sort();
    assert_eq!(right, ["1:0:1", "1:1:1"], "{keys:?}");
}

#[test]
fn a_computed_grouping_key_leads_every_frame() {
    let env = env();
    // `groupBy(df.a % 2)`: the key is computed, not an input column.
    let analyzed = env.analyze("SELECT a, b, a % 2 AS m FROM t");
    let root = analyzed.root();
    let PlanNode::Project { input, exprs } = &root.node else {
        panic!("{root}")
    };
    let key = match &exprs[2].node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => Arc::clone(expr),
        other => panic!("{other:?}"),
    };
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input: Arc::clone(input),
        keys: vec![key],
        udf: udf("count", PythonEvalType::GroupedMapPandas),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    // `a` is 0..9 with 3 and 7 NULL: five even, three odd, two NULL.
    assert_eq!(lines(&out), ["2", "3", "5"]);
    let mut keys = rt.keys.lock().unwrap().clone();
    keys.sort();
    assert_eq!(keys, ["0:0:5", "0:1:3", "0:NULL:2"]);
}

/// A grouped-aggregate Python UDF call over `arg`.
fn udaf(arg: &BoundColumn) -> Arc<Expr> {
    udaf_named("count", arg)
}

fn udaf_named(name: &str, arg: &BoundColumn) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: udf(name, PythonEvalType::GroupedAggArrow).function_id(),
            kind: FunctionKind::Aggregate,
            args: vec![col(arg)],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Long,
            nullable: true,
            deterministic: true,
        },
    ))))
}

fn alias(e: Arc<Expr>, id: u64, name: &str) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
        expr: e,
        output: bound(id, name, DataType::Long),
    })))
}

#[test]
fn a_python_aggregate_yields_one_row_per_group() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(vec![col(&cols[0])]),
        aggregates: vec![col(&cols[0]), alias(udaf(&cols[1]), 950, "n")],
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    let rows = lines(&out);
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert!(rows.contains(&"NULL|2".to_owned()), "{rows:?}");
    assert!(rows.contains(&"5|1".to_owned()), "{rows:?}");
}

/// Two Python aggregates, one slow: each key's values pair up.
#[test]
fn two_python_aggregates_pair_their_values_per_key() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(vec![col(&cols[0])]),
        aggregates: vec![
            col(&cols[0]),
            alias(udaf(&cols[1]), 950, "n"),
            alias(udaf_named("slow_count", &cols[0]), 951, "m"),
        ],
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    let rows = lines(&out);
    assert_eq!(rows.len(), 9, "{rows:?}");
    for row in &rows {
        let cells: Vec<&str> = row.split('|').collect();
        assert_eq!(cells[1], cells[2], "{rows:?}");
    }
    assert!(rows.contains(&"NULL|2|2".to_owned()), "{rows:?}");
    assert!(rows.contains(&"5|1|1".to_owned()), "{rows:?}");
}

#[test]
fn a_python_aggregate_beside_a_builtin_is_refused() {
    let env = env();
    let analyzed = env.analyze("SELECT a, sum(b) AS s FROM t GROUP BY a");
    let root = analyzed.root();
    let PlanNode::Aggregate {
        input,
        grouping,
        aggregates,
    } = &root.node
    else {
        panic!("{root}")
    };
    let cols = input.resolved_output().unwrap();
    let mut aggregates = aggregates.clone();
    aggregates.push(alias(udaf(&cols[1]), 950, "n"));
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(input),
        grouping: grouping.clone(),
        aggregates,
    });
    // Raised at physical planning, as Spark does (`SparkStrategies`): the
    // plan itself is valid and optimizes.
    let plan = krishiv_optimizer::optimize(&Plan::try_new(node).unwrap()).unwrap();
    let err = krishiv_exec::planner::PhysicalPlanner::plan(
        &krishiv_exec::planner::DefaultPlanner,
        &plan,
        &env.cx(),
    )
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::InvalidPandasUdfPlacement,
        "{err}"
    );
    assert!(err.to_string().contains("`count`"), "{err}");
}

/// A Python UDAF call with `DISTINCT`, `FILTER (WHERE ...)` or `WITHIN
/// GROUP` would run over every row of its group: refused at planning.
#[test]
fn a_python_aggregate_with_distinct_filter_or_within_group_is_refused() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let changed = |change: &dyn Fn(&mut ResolvedFunction)| {
        let ExprNode::Resolved(ResolvedExpr::Function(f)) = &udaf(&cols[1]).node else {
            unreachable!()
        };
        let mut f = f.clone();
        change(&mut f);
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))))
    };
    let cases: [(&str, Arc<Expr>); 3] = [
        ("DISTINCT", changed(&|f| f.distinct = true)),
        ("FILTER", changed(&|f| f.filter = Some(col(&cols[0])))),
        (
            "WITHIN GROUP",
            changed(&|f| f.within_group = vec![krishiv_plan::SortOrder::asc(col(&cols[0]))]),
        ),
    ];
    for (what, call) in cases {
        let node = LogicalPlan::new(PlanNode::Aggregate {
            input: Arc::clone(&input),
            grouping: Grouping::GroupBy(vec![col(&cols[0])]),
            aggregates: vec![col(&cols[0]), alias(call, 950, "n")],
        });
        let err = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{what}: {err}"
        );
        assert!(err.to_string().contains(what), "{what}: {err}");
    }
}

#[test]
fn barrier_mode_is_refused_at_planning() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::MapPartitions {
        input,
        udf: udf("echo", PythonEvalType::MapArrowIter),
        output: vec![bound(900, "x", cols[0].data_type.clone())],
        barrier: true,
    });
    let err = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(err.to_string().contains("isBarrier"), "{err}");
}

#[test]
fn python_udtfs_and_window_aggregates_are_refused_until_pr_26c() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let call = |eval_type, kind| {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: udf("f", eval_type).function_id(),
                kind,
                args: vec![col(&cols[1])],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Long,
                nullable: true,
                deterministic: true,
            },
        ))))
    };
    let lateral = LogicalPlan::new(PlanNode::Generate {
        input: Arc::clone(&input),
        kind: GeneratorKind::PythonUdtf,
        outer: false,
        arg: call(PythonEvalType::TableUdf, FunctionKind::Generator),
        output: counted(),
        names: Vec::new(),
        qualifier: None,
    });
    let table = LogicalPlan::new(PlanNode::PythonTableFunction {
        args: Vec::new(),
        table_arg: None,
        udf: udf("f", PythonEvalType::ArrowUdtf),
        output: counted(),
    });
    let window = LogicalPlan::new(PlanNode::Window {
        input,
        exprs: vec![alias(
            call(PythonEvalType::WindowAggArrow, FunctionKind::Aggregate),
            950,
            "w",
        )],
        partition_by: vec![col(&cols[0])],
        order_by: Vec::new(),
    });
    for node in [lateral, table, window] {
        let text = node.to_string();
        let err = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{text}: {err}"
        );
        assert!(err.to_string().contains("PR-26c"), "{text}: {err}");
    }
}

#[test]
fn a_scalar_python_udf_in_a_window_is_not_named_a_window_aggregate() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let scalar = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: udf("f", PythonEvalType::ScalarArrow).function_id(),
            kind: FunctionKind::Scalar,
            args: vec![col(&cols[1])],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Long,
            nullable: true,
            deterministic: true,
        },
    ))));
    let window = LogicalPlan::new(PlanNode::Window {
        input,
        exprs: vec![alias(scalar, 950, "w")],
        partition_by: vec![col(&cols[0])],
        order_by: Vec::new(),
    });
    let err = run_with(&env, window, Some(Arc::new(Runtime::default()))).unwrap_err();
    assert!(!err.to_string().contains("PR-26c"), "{err}");
}

#[test]
fn a_worker_without_a_runtime_refuses_the_plan() {
    let env = env();
    let (input, _) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input,
        keys: Vec::new(),
        udf: udf("count", PythonEvalType::GroupedMapArrow),
        output: counted(),
    });
    let err = run_with(&env, node, None).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUdfNotAvailable,
        "{err}"
    );
}

/// A table `big (k INT, v BIGINT)` of `rows` rows keyed by `key`, `node`
/// over its scan planned and run under a 4 MiB limit with a spill
/// directory: the results, and the spill files written.
fn run_big(
    rows: i32,
    key: impl Fn(i32) -> i32,
    node: impl FnOnce(Arc<LogicalPlan>, &BoundColumn) -> LogicalPlan,
) -> (Vec<RecordBatch>, u64) {
    use krishiv_exec::planner::{MemoryTable, MemoryTables};
    use krishiv_memory::{MemoryConfig, SpillConfig, SpillManager, TreePool};
    use krishiv_types::batch::Batch;

    let schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowType::Int32, false),
        Field::new("v", ArrowType::Int64, false),
    ]));
    let batches: Vec<Batch> = (0..rows)
        .step_by(8192)
        .map(|start| {
            let end = (start + 8192).min(rows);
            let k: Int32Array = (start..end).map(&key).collect();
            let v: Int64Array = (start..end).map(i64::from).collect();
            Batch::new(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(k), Arc::new(v)]).unwrap(),
            )
        })
        .collect();
    let tables = Arc::new(MemoryTables::new().with_table(
        &["spark_catalog", "default", "big"],
        Arc::new(MemoryTable::new(Arc::clone(&schema), batches)),
    ));
    let k = bound(1, "k", DataType::Integer);
    let v = bound(2, "v", DataType::Long);
    let scan = LogicalPlan::new(PlanNode::Scan(krishiv_plan::Scan {
        source: krishiv_plan::ReadSource::Table {
            name: vec![
                "spark_catalog".to_owned(),
                "default".to_owned(),
                "big".to_owned(),
            ],
            options: std::collections::BTreeMap::new(),
        },
        output: vec![k.clone(), v],
        version: None,
    }));
    let plan = Plan::try_new(node(Arc::new(scan), &k)).unwrap();
    let cx = krishiv_exec::planner::PlanningContext::new(
        Arc::clone(&tables) as _,
        Arc::new(krishiv_common::config::Config::new()),
    );
    let physical = krishiv_exec::planner::PhysicalPlanner::plan(
        &krishiv_exec::planner::DefaultPlanner,
        &plan,
        &cx,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let scope = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap()
    .scope()
    .unwrap();
    let runnable = Runnable::new(physical, tables as _)
        .with_spill(scope)
        .with_worker(|w| w.with_python_udfs(Arc::new(Runtime::default())));
    let root = TreePool::new(&MemoryConfig {
        limit: Some(4 << 20),
        ..MemoryConfig::default()
    });
    let (out, query) = super::run_in(&runnable, &root);
    let out = out.unwrap();
    let files: u64 = query
        .metrics()
        .operators
        .values()
        .map(|o| o.metrics.spill_files)
        .sum();
    drop(query);
    assert_eq!(krishiv_memory::MemoryPool::reserved(&*root), 0);
    (out, files)
}

#[test]
fn a_skewed_group_runs_through_the_spilling_sort_in_bounded_memory() {
    const ROWS: i32 = 300_000;
    // One key of 270k rows (90%), the rest in keys of 100 rows; the
    // brief's 1M/900k shape shrunk to stay near 10 s in debug builds.
    let (out, files) = run_big(
        ROWS,
        |i| if i < 270_000 { 0 } else { 1 + i / 100 },
        |scan, k| {
            LogicalPlan::new(PlanNode::GroupMap {
                input: scan,
                keys: vec![col(k)],
                udf: udf("count", PythonEvalType::GroupedMapArrow),
                output: counted(),
            })
        },
    );
    let mut counts: Vec<i64> = out
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    counts.sort_unstable();
    assert_eq!(counts.len(), 301);
    assert_eq!(counts.last(), Some(&270_000));
    assert_eq!(counts.iter().sum::<i64>(), i64::from(ROWS));
    assert!(files > 0, "the sort did not spill");
}

/// A task yielding eight times its input under a 4 MiB limit: its results
/// are reserved as they arrive, so the store spills them and the task waits
/// for memory instead of piling up results nobody accounts for.
#[test]
fn a_task_yielding_far_more_than_its_input_runs_in_bounded_memory() {
    const ROWS: i32 = 100_000;
    let (out, files) = run_big(
        ROWS,
        |i| i % 7,
        |scan, k| {
            LogicalPlan::new(PlanNode::MapPartitions {
                input: scan,
                udf: udf("burst", PythonEvalType::MapArrowIter),
                output: vec![
                    bound(900, "k", k.data_type.clone()),
                    bound(901, "v", DataType::Long),
                ],
                barrier: false,
            })
        },
    );
    let rows: usize = out.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 8 * ROWS as usize);
    assert!(files > 0, "the results did not spill");
}
