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
use krishiv_exec::python_relation::{
    Frame, PreparedRelationUdf, RelationCallSpec, RelationSlot, UdfTask,
};
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
    /// A scalar iterator: each batch's first column plus one, as it comes.
    PlusOne,
    /// A window aggregate: per row, its frame's rows (`end - start`, the
    /// first two columns) when the frames lead with the bounds, else the
    /// partition's rows; all at the partition's end.
    FrameRows,
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
        if spec.name == "plus1" && spec.arg_types == [DataType::Long] {
            return Ok(Arc::new(PlusOne));
        }
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
            "iterplus" => Mode::PlusOne,
            "framerows" => Mode::FrameRows,
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

/// The scalar fake `plus1(BIGINT) -> BIGINT`.
struct PlusOne;

impl PreparedUdf for PlusOne {
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, Result<ArrayRef, KrishivError>> {
        let column = args.column(0);
        let values = column.as_any().downcast_ref::<Int64Array>().unwrap();
        let out: Int64Array = values.iter().map(|v| v.map(|v| v + 1)).collect();
        Box::pin(async move { Ok(Arc::new(out) as ArrayRef) })
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
    fn open(&self, _slot: Option<RelationSlot>) -> Result<Box<dyn UdfTask>, KrishivError> {
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
    /// `FrameRows`: the open partition's values.
    held: Vec<i64>,
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
        // Equal under the key's collation: the tests' collated keys are
        // `UTF8_LCASE`, and a group's rows keep their own values.
        for r in 1..batch.num_rows() {
            assert_eq!(
                key(r).to_lowercase(),
                first.to_lowercase(),
                "one key per group"
            );
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
                        Mode::PlusOne => {
                            let a = batch
                                .column(0)
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap();
                            let out: Int64Array = a.iter().map(|v| v.map(|v| v + 1)).collect();
                            s.out.push_back(
                                RecordBatch::try_new(
                                    Arc::clone(&self.output),
                                    vec![Arc::new(out) as ArrayRef],
                                )
                                .unwrap(),
                            );
                        }
                        Mode::FrameRows => {
                            if batch.num_columns() >= 3 {
                                let start = batch.column(0).as_any().downcast_ref::<Int32Array>();
                                let end = batch.column(1).as_any().downcast_ref::<Int32Array>();
                                let (start, end) = (start.unwrap(), end.unwrap());
                                for r in 0..batch.num_rows() {
                                    s.held.push(i64::from(end.value(r) - start.value(r)));
                                }
                            } else {
                                let n = batch.num_rows();
                                s.held.extend(std::iter::repeat_n(-1, n));
                            }
                        }
                    }
                }
                Frame::EndGroup => {
                    self.push("G".to_owned());
                    s.groups += 1;
                    if self.mode == Mode::FrameRows {
                        let mut held = std::mem::take(&mut s.held);
                        let n = i64::try_from(held.len()).unwrap();
                        for v in &mut held {
                            if *v < 0 {
                                *v = n;
                            }
                        }
                        let b = RecordBatch::try_new(
                            Arc::clone(&self.output),
                            vec![Arc::new(Int64Array::from(held)) as ArrayRef],
                        )
                        .unwrap();
                        s.out.push_back(b);
                    }
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
        deterministic: true,
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

/// `groupBy(k)` of a `UTF8_LCASE` key: `K` and `k` are one group (Spark's
/// `GroupedIterator` compares keys by the collation), each row keeping its
/// own value.
#[test]
fn a_grouped_map_groups_a_collated_key_by_its_collation() {
    let env = env();
    let (input, cols) = input(
        &env,
        "SELECT CASE WHEN b < 50 THEN 'K' ELSE 'k' END COLLATE UTF8_LCASE AS k, b FROM t",
    );
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input,
        keys: vec![col(&cols[0])],
        udf: udf("count", PythonEvalType::GroupedMapPandas),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    assert_eq!(lines(&out), ["10"]);
    // The frames carry the rows' own key values, not the collation key.
    let keys = rt.keys.lock().unwrap().clone();
    assert!(
        keys.iter()
            .all(|k| k.starts_with("0:K:") || k.starts_with("0:k:")),
        "{keys:?}"
    );
}

/// A cogroup whose sides store a string in different Arrow types (`u.s` is
/// `Utf8`, `t.s` and the other side's NULLs `Utf8View`), as a data column
/// and as the key.
#[test]
fn a_cogroup_runs_sides_that_store_a_string_differently() {
    let env = super::union::carriers();
    for (key, expected) in [(0, vec!["1", "2"]), (1, vec!["1", "1", "1"])] {
        for swapped in [false, true] {
            let mut sides = [
                input(&env, "SELECT a, s FROM u"),
                input(&env, "SELECT a, s FROM t WHERE a = 1"),
            ];
            if swapped {
                sides.reverse();
            }
            let [(left, lc), (right, rc)] = sides;
            let node = LogicalPlan::new(PlanNode::CoGroupMap {
                left,
                right,
                left_keys: vec![col(&lc[key])],
                right_keys: vec![col(&rc[key])],
                udf: udf("count", PythonEvalType::CogroupedMapArrow),
                output: counted(),
            });
            let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
            assert_eq!(lines(&out), expected, "key {key}, swapped {swapped}");
        }
    }
}

/// A cogroup on `UTF8_LCASE` keys: the left side's `K` and the right side's
/// `k` are one key (Spark's `CoGroupedIterator` compares them by the
/// collation), so the UDF gets one call with both sides.
#[test]
fn a_cogroup_matches_collated_keys_by_their_collation() {
    let env = env();
    let (left, lc) = input(
        &env,
        "SELECT 'K' COLLATE UTF8_LCASE AS k, b FROM t WHERE b < 30",
    );
    let (right, rc) = input(
        &env,
        "SELECT 'k' COLLATE UTF8_LCASE AS c, s FROM t WHERE b >= 80",
    );
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
    assert_eq!(lines(&out), ["5"]);
    let log = log_of(&rt);
    assert_eq!(log.iter().filter(|e| *e == "G").count(), 1, "{log:?}");
    let mut keys = rt.keys.lock().unwrap().clone();
    keys.sort();
    assert_eq!(keys, ["0:K:3", "1:k:2"], "{keys:?}");
}

/// `groupBy(plus1(b)).applyInPandas`: the scalar call in the key runs in a
/// `PythonEval` below the node (Spark evaluates it in a `Project` under
/// `FlatMapGroupsInPandas`, which `ExtractPythonUDFs` at v4.2.0 turns into
/// an `ArrowEvalPython`); the frames lead with its values.
#[test]
fn a_scalar_python_call_in_a_grouped_maps_key_runs_below_it() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::GroupMap {
        input,
        keys: vec![plus1(col(&cols[1]))],
        udf: udf("count", PythonEvalType::GroupedMapPandas),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    // `b` is 0, 10, ..., 90: ten keys of one row each.
    assert_eq!(lines(&out), vec!["1"; 10]);
    let keys = rt.keys.lock().unwrap().clone();
    assert!(keys.contains(&"0:1:1".to_owned()), "{keys:?}");
    assert!(keys.contains(&"0:91:1".to_owned()), "{keys:?}");
}

/// A cogroup keyed by `plus1(b)` on both sides: each side's call runs
/// below its branch, and equal results meet.
#[test]
fn a_scalar_python_call_in_a_cogroups_keys_runs_below_it() {
    let env = env();
    let (left, lc) = input(&env, "SELECT a, b FROM t");
    let (right, rc) = input(&env, "SELECT b AS c, s FROM t WHERE b < 20");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::CoGroupMap {
        left,
        right,
        left_keys: vec![plus1(col(&lc[1]))],
        right_keys: vec![plus1(col(&rc[0]))],
        udf: udf("count", PythonEvalType::CogroupedMapArrow),
        output: counted(),
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    // Keys 1 and 11 have a row on each side; the other eight the left one.
    let rows = lines(&out);
    assert_eq!(rows.len(), 10, "{rows:?}");
    assert_eq!(rows.iter().filter(|r| *r == "2").count(), 2, "{rows:?}");
    let keys = rt.keys.lock().unwrap().clone();
    let mut right: Vec<&String> = keys
        .iter()
        .filter(|k| k.starts_with("1:") && !k.ends_with(":0"))
        .collect();
    right.sort();
    assert_eq!(right, ["1:11:1", "1:1:1"], "{keys:?}");
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

/// `SELECT k, udaf(b) ... GROUP BY k` of a `UTF8_LCASE` key: `K` and `k`
/// are one group (Spark's `ArrowAggregatePythonExec` cuts groups with a
/// `GroupedIterator` comparing keys by the collation, v4.2.0), and the key
/// shown is a value of the group's own (its first row's), not the
/// comparison key. The collation rewrite leaves the aggregate to the Python
/// aggregate's planner, so no `any_value(k)` sits beside the UDAF.
#[test]
fn a_python_aggregate_groups_a_collated_key_by_its_collation() {
    let env = env();
    let (input, cols) = input(
        &env,
        "SELECT CASE WHEN b < 50 THEN 'K' ELSE 'k' END COLLATE UTF8_LCASE AS k, b FROM t",
    );
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(vec![col(&cols[0])]),
        aggregates: vec![col(&cols[0]), alias(udaf(&cols[1]), 950, "n")],
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    let rows = lines(&out);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0] == "K|10" || rows[0] == "k|10", "{rows:?}");
    let key_type = out[0].schema().field(0).data_type().clone();
    assert!(
        matches!(key_type, ArrowType::Utf8 | ArrowType::Utf8View),
        "the key column is the key's own, not its comparison key: {key_type}"
    );
}

/// A scalar Python UDF call `plus1(arg)`.
fn plus1(arg: Arc<Expr>) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: udf("plus1", PythonEvalType::ScalarArrow).function_id(),
            kind: FunctionKind::Scalar,
            args: vec![arg],
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type: DataType::Long,
            nullable: true,
            deterministic: true,
        },
    ))))
}

/// `udaf(arg)` with `arg` any expression.
fn udaf_of(arg: Arc<Expr>) -> Arc<Expr> {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) =
        &udaf_named("count", &bound(0, "x", DataType::Long)).node
    else {
        unreachable!()
    };
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            args: vec![arg],
            ..f.clone()
        },
    ))))
}

/// `udaf(plus1(b))`: the scalar call in the aggregate's argument runs
/// below it (Spark's `ExtractPythonUDFs` at v4.2.0 evaluates it in an
/// `ArrowEvalPython` under the aggregate).
#[test]
fn a_scalar_python_call_in_a_python_aggregates_argument_runs_below_it() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(vec![col(&cols[0])]),
        aggregates: vec![
            col(&cols[0]),
            alias(udaf_of(plus1(col(&cols[1]))), 950, "n"),
        ],
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    let rows = lines(&out);
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert!(rows.contains(&"NULL|2".to_owned()), "{rows:?}");
}

/// `GROUP BY plus1(b)`: the scalar call in the key runs below the aggregate
/// (Spark's `ExtractGroupingPythonUDFFromAggregate` at v4.2.0), and the
/// output reads the key's column.
#[test]
fn a_scalar_python_call_in_a_python_aggregates_key_runs_below_it() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let key = plus1(col(&cols[1]));
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(vec![Arc::clone(&key)]),
        aggregates: vec![alias(key, 949, "k"), alias(udaf(&cols[0]), 950, "n")],
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    let rows = lines(&out);
    // `b` is 0, 10, ..., 90: ten keys of one row each.
    assert_eq!(rows.len(), 10, "{rows:?}");
    assert!(rows.contains(&"1|1".to_owned()), "{rows:?}");
    assert!(rows.contains(&"91|1".to_owned()), "{rows:?}");
}

/// `plus1(udaf(b))`: the scalar call over the aggregate's result runs above
/// it (Spark's `ExtractPythonUDFFromAggregate` at v4.2.0).
#[test]
fn a_scalar_python_call_over_a_python_aggregate_runs_above_it() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(vec![col(&cols[0])]),
        aggregates: vec![col(&cols[0]), alias(plus1(udaf(&cols[1])), 950, "n")],
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    let rows = lines(&out);
    // Eight keys of one row (1 + 1), NULL of two (2 + 1).
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert!(rows.contains(&"NULL|3".to_owned()), "{rows:?}");
    assert!(rows.contains(&"5|2".to_owned()), "{rows:?}");
}

/// A keyless Python aggregate over no rows yields no row: Spark's
/// `ArrowAggregatePythonExec` (ArrowAggregatePythonExec.scala L156-157 at
/// v4.2.0: `mapPartitionsInternal { iter => if (iter.isEmpty) iter else
/// ...`) returns an empty partition as it is, without calling the UDF,
/// unlike a built-in global aggregate's one row. Pinned from the source
/// only: a Spark-run golden awaits a lane with Spark.
#[test]
fn a_keyless_python_aggregate_over_no_rows_yields_no_row() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t WHERE b < 0");
    let rt = Arc::new(Runtime::default());
    let node = LogicalPlan::new(PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(Vec::new()),
        aggregates: vec![alias(udaf(&cols[1]), 950, "n")],
    });
    let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
    assert!(lines(&out).is_empty(), "{:?}", lines(&out));
    assert!(log_of(&rt).is_empty(), "{:?}", log_of(&rt));
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

/// A grouped or cogrouped map over a nondeterministic UDF is refused with
/// `INVALID_NON_DETERMINISTIC_EXPRESSIONS`: Spark's `CheckAnalysis` allows
/// a nondeterministic expression on a few operators only, of the Python
/// ones `MapInPandas` and `MapInArrow` (CheckAnalysis.scala L1032-1052 at
/// v4.2.0), and a node's `functionExpr` is a `PythonUDF` whose
/// `deterministic` is `udfDeterministic`. A map stays allowed.
#[test]
fn a_nondeterministic_grouped_or_cogrouped_map_udf_is_refused() {
    let env = env();
    let random = |name: &str, eval_type| PythonUdfRef {
        deterministic: false,
        ..udf(name, eval_type)
    };
    let refused = |node: LogicalPlan| {
        let plan = krishiv_optimizer::optimize(&Plan::try_new(node).unwrap()).unwrap();
        krishiv_exec::planner::PhysicalPlanner::plan(
            &krishiv_exec::planner::DefaultPlanner,
            &plan,
            &env.cx(),
        )
        .unwrap_err()
    };
    let (input_, cols) = input(&env, "SELECT a, b FROM t");
    let err = refused(LogicalPlan::new(PlanNode::GroupMap {
        input: input_,
        keys: vec![col(&cols[0])],
        udf: random("count", PythonEvalType::GroupedMapPandas),
        output: counted(),
    }));
    assert_eq!(
        err.error_class(),
        ErrorClass::InvalidNonDeterministicExpressions,
        "{err}"
    );
    assert!(
        err.to_string().contains(r#""a", "count(a, b)", "n""#),
        "{err}"
    );
    let (left, lc) = input(&env, "SELECT a, b FROM t");
    let (right, rc) = input(&env, "SELECT a AS c, s FROM t WHERE a < 2");
    let err = refused(LogicalPlan::new(PlanNode::CoGroupMap {
        left,
        right,
        left_keys: vec![col(&lc[0])],
        right_keys: vec![col(&rc[0])],
        udf: random("count", PythonEvalType::CogroupedMapArrow),
        output: counted(),
    }));
    assert_eq!(
        err.error_class(),
        ErrorClass::InvalidNonDeterministicExpressions,
        "{err}"
    );
    assert!(
        err.to_string().contains(r#""count(a, b, c, s)", "n""#),
        "{err}"
    );
    let (input_, cols) = input(&env, "SELECT a, b FROM t");
    let node = LogicalPlan::new(PlanNode::MapPartitions {
        input: input_,
        udf: random("echo", PythonEvalType::MapArrowIter),
        output: vec![
            bound(900, "x", cols[0].data_type.clone()),
            bound(901, "y", cols[1].data_type.clone()),
        ],
        barrier: false,
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap();
    assert_eq!(lines(&out).len(), 10);
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
fn python_udtfs_are_refused_until_pr_26c() {
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
    // A `PythonTableFunction` is planned (PR-26c-4, `krishiv-session/tests/python_udtf.rs`); the analyzer
    // builds no `Generate` of a UDTF, so one stays refused.
    let text = lateral.to_string();
    let err = run_with(&env, lateral, Some(Arc::new(Runtime::default()))).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{text}: {err}"
    );
    assert!(err.to_string().contains("PR-26c"), "{text}: {err}");
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

/// A scalar iterator Python UDF in a projection (Spark's 204, 251) runs as a
/// task over the node's rows: each row gets its own value, in order.
#[test]
fn a_scalar_iterator_udf_yields_one_value_per_row() {
    let env = env();
    let (input, cols) = input(&env, "SELECT a, b FROM t");
    for eval_type in [
        PythonEvalType::ScalarArrowIter,
        PythonEvalType::ScalarPandasIter,
    ] {
        let call = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: udf("iterplus", eval_type).function_id(),
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
        let node = LogicalPlan::new(PlanNode::Project {
            input: Arc::clone(&input),
            exprs: vec![col(&cols[1]), alias(call, 950, "p")],
        });
        let rt = Arc::new(Runtime::default());
        let out = run_with(&env, node, Some(Arc::clone(&rt))).unwrap();
        let rows = lines(&out);
        assert_eq!(rows.len(), 10, "{rows:?}");
        for r in &rows {
            let (b, p) = r.split_once('|').unwrap();
            if b == "NULL" {
                assert_eq!(p, "NULL", "{rows:?}");
            } else {
                assert_eq!(
                    p.parse::<i64>().unwrap(),
                    b.parse::<i64>().unwrap() + 1,
                    "{rows:?}"
                );
            }
        }
        // No group markers: batches, then the end.
        let log = log_of(&rt);
        assert!(log.iter().all(|l| l != "G"), "{log:?}");
        assert_eq!(log.last().map(String::as_str), Some("E"), "{log:?}");
    }
}

/// The `Window` node below the first expression of `p` (its first one).
fn window_of(p: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
    if matches!(p.node, PlanNode::Window { .. }) {
        return Some(Arc::clone(p));
    }
    let mut found = None;
    p.for_each_child(&mut |c| {
        if found.is_none() {
            found = window_of(c);
        }
    });
    found
}

/// `count(*) OVER (spec)` analyzed, and the same window with a Python
/// window UDF of `eval_type` (named `framerows`: it yields each row's frame
/// size from the frame bounds it gets) beside it: the two columns agree on
/// every row, so the bounds the engine sends are Spark's frame rows.
fn frame_rows_match_count(spec: &str, eval_type: PythonEvalType) {
    let env = env();
    let plan = env.analyze(&format!("SELECT a, b, count(*) OVER ({spec}) AS c FROM t"));
    let window = window_of(plan.root()).unwrap_or_else(|| panic!("no window: {spec}"));
    let PlanNode::Window {
        input,
        exprs,
        partition_by,
        order_by,
    } = &window.node
    else {
        unreachable!()
    };
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &exprs[0].node else {
        panic!("{spec}: {}", exprs[0])
    };
    let ExprNode::Window(w) = &expr.node else {
        panic!("{spec}: {expr}")
    };
    let cols = input.resolved_output().unwrap();
    let b = cols.iter().find(|c| c.name == "b").unwrap();
    let python = Arc::new(Expr::new(ExprNode::Window(krishiv_plan::WindowExpr {
        function: Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: udf("framerows", eval_type).function_id(),
                kind: FunctionKind::Aggregate,
                args: vec![col(b)],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Long,
                nullable: true,
                deterministic: true,
            },
        )))),
        window: w.window.clone(),
    })));
    let node = LogicalPlan::new(PlanNode::Window {
        input: Arc::clone(input),
        exprs: vec![Arc::clone(&exprs[0]), alias(python, 960, "w")],
        partition_by: partition_by.clone(),
        order_by: order_by.clone(),
    });
    let out = run_with(&env, node, Some(Arc::new(Runtime::default())))
        .unwrap_or_else(|e| panic!("{spec}: {e}"));
    let rows = lines(&out);
    assert_eq!(rows.len(), 10, "{spec}: {rows:?}");
    for r in &rows {
        let cells: Vec<&str> = r.split('|').collect();
        let n = cells.len();
        assert_eq!(cells[n - 2], cells[n - 1], "{spec}: {rows:?}");
    }
}

#[test]
fn a_python_window_udf_gets_each_row_s_frame_bounds() {
    for spec in [
        "PARTITION BY a ORDER BY b ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
        "PARTITION BY a ORDER BY b ROWS BETWEEN 3 PRECEDING AND 2 PRECEDING",
        "PARTITION BY a ORDER BY b ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING",
        "PARTITION BY a ORDER BY b",
        "ORDER BY b RANGE BETWEEN 2 PRECEDING AND CURRENT ROW",
        "PARTITION BY a",
        "",
    ] {
        for t in [
            PythonEvalType::GroupedAggArrow,
            PythonEvalType::GroupedAggPandas,
            PythonEvalType::WindowAggArrow,
        ] {
            frame_rows_match_count(spec, t);
        }
    }
}

/// The iterator forms of a grouped aggregate (217, 254) over a window are
/// refused at planning, as a Python UDAF call with `FILTER` is.
#[test]
fn an_iterator_grouped_aggregate_over_a_window_is_refused() {
    let env = env();
    let plan = env.analyze("SELECT a, b, count(*) OVER (PARTITION BY a) AS c FROM t");
    let window = window_of(plan.root()).unwrap();
    let PlanNode::Window {
        input,
        exprs,
        partition_by,
        order_by,
    } = &window.node
    else {
        unreachable!()
    };
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &exprs[0].node else {
        unreachable!()
    };
    let ExprNode::Window(w) = &expr.node else {
        unreachable!()
    };
    let cols = input.resolved_output().unwrap();
    let python = Arc::new(Expr::new(ExprNode::Window(krishiv_plan::WindowExpr {
        function: udaf_of(col(&cols[1])),
        window: w.window.clone(),
    })));
    let python = match &python.node {
        ExprNode::Window(w) => {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &w.function.node else {
                unreachable!()
            };
            Arc::new(Expr::new(ExprNode::Window(krishiv_plan::WindowExpr {
                function: Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                    ResolvedFunction {
                        id: udf("count", PythonEvalType::GroupedAggArrowIter).function_id(),
                        ..f.clone()
                    },
                )))),
                window: w.window.clone(),
            })))
        }
        _ => unreachable!(),
    };
    let node = LogicalPlan::new(PlanNode::Window {
        input: Arc::clone(input),
        exprs: vec![alias(python, 960, "w")],
        partition_by: partition_by.clone(),
        order_by: order_by.clone(),
    });
    let err = run_with(&env, node, Some(Arc::new(Runtime::default()))).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
    assert!(err.to_string().contains("254 over a window"), "{err}");
}
