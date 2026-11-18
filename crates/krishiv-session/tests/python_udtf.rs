// SPDX-License-Identifier: Apache-2.0
//! Python table functions (UDTFs) from SQL with a Rust fake runtime (PR-26c-4): a registered UDTF resolves in
//! `FROM` and on the right of `LATERAL`, runs as a relation task per partition fed its argument columns (a
//! `TABLE` argument as one struct column, partitioned and sorted as asked), a lateral call's rows lead with
//! their input row, and `analyze` runs once per analysis, its result kept in the plan and handed to the tasks.
//! The pool runtime with real Python workers replays Spark's goldens in `krishiv-udf-pool/tests/udtf_goldens.rs`.
#![expect(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "test code: a poisoned lock in the fake runtime fails the test"
)]

use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use futures::StreamExt as _;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_exec::python_relation::{
    Frame, PreparedRelationUdf, RelationCallSpec, RelationSlot, UdfTask,
};
use krishiv_exec::python_udtf::{
    UdtfAnalysis, UdtfAnalyzeSpec, UdtfCall, UdtfOrdering, UdtfSelect,
};
use krishiv_plan::PythonEvalType;
use krishiv_session::{PythonUdtfDef, Session};
use krishiv_types::{DataType, StructField, StructType};

/// What the fake saw.
#[derive(Debug, Default)]
struct Seen {
    /// Each `analyze` call: the argument types and whether each was constant.
    analyzed: Vec<Vec<(DataType, bool, bool)>>,
    /// Each prepared call site's UDTF call data.
    calls: Vec<UdtfCall>,
    /// A `slow` analyze: told when it starts; set when the runtime's future is dropped (the call abandoned).
    slow_started: Option<std::sync::mpsc::Sender<()>>,
    slow_dropped: Arc<std::sync::atomic::AtomicBool>,
    /// Each task's input rows.
    tasks: Vec<usize>,
    /// Each task's input, all its frames in one batch.
    inputs: Vec<RecordBatch>,
}

#[derive(Debug, Default)]
struct Fake {
    seen: Arc<Mutex<Seen>>,
}

impl UdfRuntime for Fake {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        Err(KrishivError::from(
            krishiv_common::error::KrishivUdfNotAvailable {
                name: spec.name.clone(),
                version: spec.version.to_string(),
            },
        ))
    }

    fn prepare_relation(
        &self,
        spec: &RelationCallSpec,
    ) -> krishiv_exec::Result<Arc<dyn PreparedRelationUdf>> {
        let call = spec.udtf.clone().unwrap();
        self.seen.lock().unwrap().calls.push(call.clone());
        Ok(Arc::new(Prepared {
            name: spec.name.clone(),
            call,
            output: Arc::clone(&spec.output_arrow),
            seen: Arc::clone(&self.seen),
        }))
    }

    /// `describe`: one `INT` column named `c<number of arguments>`, a single partition, ordered by `id`
    /// descending when the call has a `TABLE` argument.
    fn analyze_udtf(
        &self,
        spec: &UdtfAnalyzeSpec,
    ) -> BoxFuture<'static, krishiv_exec::Result<UdtfAnalysis>> {
        self.seen.lock().unwrap().analyzed.push(
            spec.args
                .iter()
                .map(|a| (a.data_type.clone(), a.value.is_some(), a.is_table))
                .collect(),
        );
        if spec.name == "slow" {
            // Never answers; its drop is the call abandoned.
            struct Dropped(Arc<std::sync::atomic::AtomicBool>);
            impl Drop for Dropped {
                fn drop(&mut self) {
                    self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let mut seen = self.seen.lock().unwrap();
            let guard = Dropped(Arc::clone(&seen.slow_dropped));
            if let Some(tx) = seen.slow_started.take() {
                tx.send(()).unwrap();
            }
            return Box::pin(async move {
                let _guard = guard;
                std::future::pending::<krishiv_exec::Result<UdtfAnalysis>>().await
            });
        }
        let table = spec.args.iter().any(|a| a.is_table);
        let schema = DataType::Struct(StructType::new([StructField::new(
            format!("c{}", spec.args.len()),
            DataType::Integer,
            true,
        )]));
        if let Some(scripted) = scripted_analysis(&spec.name, schema.clone()) {
            return Box::pin(std::future::ready(Ok(scripted)));
        }
        let analysis = UdtfAnalysis {
            schema,
            pickled: Arc::from(b"pickled".to_vec()),
            with_single_partition: table,
            partition_by: Vec::new(),
            order_by: if table {
                vec![UdtfOrdering {
                    name: "id".to_owned(),
                    ascending: false,
                    nulls_first: None,
                }]
            } else {
                Vec::new()
            },
            select: Vec::new(),
        };
        Box::pin(std::future::ready(Ok(analysis)))
    }
}

/// Scripted `analyze` results, by the UDTF's name (`PythonUDTFAnalyzeResult.applyToTableArgument`, v4.2.0): the
/// metadata a `TABLE` argument cannot hold, which Spark refuses, and a full request it applies.
fn scripted_analysis(name: &str, schema: DataType) -> Option<UdtfAnalysis> {
    let base = UdtfAnalysis {
        schema,
        pickled: Arc::from(b"pickled".to_vec()),
        with_single_partition: false,
        partition_by: Vec::new(),
        order_by: Vec::new(),
        select: Vec::new(),
    };
    let order = |name: &str| UdtfOrdering {
        name: name.to_owned(),
        ascending: true,
        nulls_first: None,
    };
    Some(match name {
        // One partition and partition keys at once.
        "single_and_keys" => UdtfAnalysis {
            with_single_partition: true,
            partition_by: vec!["g".to_owned()],
            ..base
        },
        // Its own partitioning, on a call that already partitions (`PARTITION BY` in the query).
        "own_partitioning" => UdtfAnalysis {
            partition_by: vec!["g".to_owned()],
            ..base
        },
        // An alias inside an ordering string.
        "aliased_order" => UdtfAnalysis {
            with_single_partition: true,
            order_by: vec![order("id AS k")],
            ..base
        },
        // Partitioned by `g`, ordered by `id` descending with NULLs first, selecting `id` bare and `g` as `grp`.
        "keyed" => UdtfAnalysis {
            partition_by: vec!["g".to_owned()],
            order_by: vec![UdtfOrdering {
                name: "id".to_owned(),
                ascending: false,
                nulls_first: Some(true),
            }],
            select: vec![
                UdtfSelect {
                    name: "id".to_owned(),
                    alias: String::new(),
                },
                UdtfSelect {
                    name: "g".to_owned(),
                    alias: "grp".to_owned(),
                },
            ],
            ..base
        },
        // A result schema that is no struct.
        "not_a_struct" => UdtfAnalysis {
            schema: DataType::Integer,
            ..base
        },
        // A selected expression that is not a column, without an alias.
        "unaliased_select" => UdtfAnalysis {
            select: vec![UdtfSelect {
                name: "id + 1".to_owned(),
                alias: String::new(),
            }],
            ..base
        },
        _ => return None,
    })
}

struct Prepared {
    name: String,
    call: UdtfCall,
    output: SchemaRef,
    seen: Arc<Mutex<Seen>>,
}

impl PreparedRelationUdf for Prepared {
    fn open(&self, _slot: Option<RelationSlot>) -> krishiv_exec::Result<Box<dyn UdfTask>> {
        let (tx, rx) = mpsc::unbounded();
        Ok(Box::new(Task {
            name: self.name.clone(),
            call: self.call.clone(),
            output: Arc::clone(&self.output),
            seen: Arc::clone(&self.seen),
            frames: Mutex::new(Vec::new()),
            tx: Mutex::new(Some(tx)),
            rx: futures::lock::Mutex::new(rx),
        }))
    }
}

struct Task {
    name: String,
    call: UdtfCall,
    output: SchemaRef,
    seen: Arc<Mutex<Seen>>,
    frames: Mutex<Vec<RecordBatch>>,
    tx: Mutex<Option<mpsc::UnboundedSender<RecordBatch>>>,
    rx: futures::lock::Mutex<mpsc::UnboundedReceiver<RecordBatch>>,
}

fn ints(c: &ArrayRef) -> Vec<Option<i32>> {
    c.as_primitive::<Int32Type>().iter().collect()
}

impl Task {
    /// The results of every frame: `repeat(a, n)` gives `(a, i)` per `i < n`; `rows(TABLE)` gives, at each
    /// change of the partition fields and at the end, `(rows of the partition, first id)`; `describe(..)` gives
    /// one row, the pickled result's length.
    fn results(&self, input: &RecordBatch) -> RecordBatch {
        let carried = self.call.passthrough.len();
        let n = input.num_rows();
        let mut take: Vec<u64> = Vec::new();
        let mut cols: Vec<Vec<Option<i32>>> =
            vec![Vec::new(); self.output.fields().len() - carried];
        match self.name.as_str() {
            "repeat" => {
                let (a, k) = (ints(input.column(0)), ints(input.column(1)));
                for r in 0..n {
                    for i in 0..k[r].unwrap_or(0) {
                        take.push(r as u64);
                        cols[0].push(a[r]);
                        cols[1].push(Some(i));
                    }
                }
            }
            "rows" => {
                let table = input.column(self.call.table_arg_offsets[0]).as_struct();
                let ids = ints(table.column(0));
                let keys: Vec<Vec<String>> = (0..n)
                    .map(|r| {
                        self.call
                            .partition_child_indexes
                            .iter()
                            .map(|&i| format!("{:?}", table.column(i).slice(r, 1)))
                            .collect()
                    })
                    .collect();
                let mut start = 0;
                for r in 1..=n {
                    if r == n || keys[r] != keys[start] {
                        cols[0].push(Some(i32::try_from(r - start).unwrap()));
                        cols[1].push(ids[start]);
                        start = r;
                    }
                }
            }
            _ => {
                let len = self.call.analyze_result.as_ref().map_or(0, |p| p.len());
                cols[0].push(Some(i32::try_from(len).unwrap()));
            }
        }
        let take = UInt64Array::from(take);
        let mut arrays: Vec<ArrayRef> = (0..carried)
            .map(|j| {
                arrow_select::take::take(input.column(self.call.passthrough[j]), &take, None)
                    .unwrap()
            })
            .collect();
        arrays.extend(
            cols.into_iter()
                .map(|c| Arc::new(Int32Array::from(c)) as ArrayRef),
        );
        RecordBatch::try_new(Arc::clone(&self.output), arrays).unwrap()
    }
}

impl UdfTask for Task {
    fn send(&self, frame: Frame) -> BoxFuture<'_, krishiv_exec::Result<()>> {
        match frame {
            Frame::Batch { batch, .. } => self.frames.lock().unwrap().push(batch),
            Frame::EndGroup => {}
            Frame::End => {
                let frames = std::mem::take(&mut *self.frames.lock().unwrap());
                let schema = frames.first().map(RecordBatch::schema);
                let input = match schema {
                    Some(s) => arrow_select::concat::concat_batches(&s, &frames).unwrap(),
                    None => RecordBatch::new_empty(Arc::new(arrow_schema::Schema::empty())),
                };
                {
                    let mut seen = self.seen.lock().unwrap();
                    seen.tasks.push(input.num_rows());
                    seen.inputs.push(input.clone());
                }
                if let Some(tx) = self.tx.lock().unwrap().take()
                    && (input.num_rows() > 0 || input.num_columns() == 0)
                {
                    tx.unbounded_send(self.results(&input)).unwrap();
                }
            }
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn next(&self) -> BoxFuture<'_, krishiv_exec::Result<Option<RecordBatch>>> {
        Box::pin(async move { Ok(self.rx.lock().await.next().await) })
    }
}

fn session() -> (Session, Arc<Mutex<Seen>>) {
    let session = Session::new().unwrap();
    let fake = Fake::default();
    let seen = Arc::clone(&fake.seen);
    session.set_python_udf_runtime(Arc::new(fake));
    (session, seen)
}

fn udtf(name: &str, eval_type: PythonEvalType, columns: &[&str]) -> PythonUdtfDef {
    PythonUdtfDef {
        name: name.to_owned(),
        eval_type,
        return_type: (!columns.is_empty()).then(|| {
            DataType::Struct(StructType::new(
                columns
                    .iter()
                    .map(|c| StructField::new((*c).to_owned(), DataType::Integer, true)),
            ))
        }),
        deterministic: true,
    }
}

/// The rows as text, sorted.
async fn rows(session: &Session, sql: &str) -> Result<Vec<String>, KrishivError> {
    let batches = session.sql(sql).await?.collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        "null".to_owned()
                    } else {
                        arrow_cast::display::array_value_to_string(c, r).unwrap()
                    }
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    Ok(out)
}

const T: &str = "CREATE OR REPLACE TEMP VIEW t AS SELECT * FROM VALUES (2, 'a'), (1, 'a'), (4, 'b'), (3, 'b'), (5, 'c') AS t(id, g)";

#[tokio::test]
async fn a_udtf_in_from_gets_its_arguments_over_one_row() {
    let (session, seen) = session();
    let entry = session
        .register_python_udtf(udtf("repeat", PythonEvalType::TableUdf, &["a", "i"]))
        .unwrap();
    assert_eq!(entry.version, 1);
    assert_eq!(
        rows(&session, "SELECT * FROM repeat(3, 2)").await.unwrap(),
        ["3|0", "3|1"]
    );
    let call = seen.lock().unwrap().calls[0].clone();
    assert_eq!(call.args.len(), 2);
    assert!(call.passthrough.is_empty() && call.table_arg_offsets.is_empty());
}

#[tokio::test]
async fn a_lateral_udtf_carries_each_input_row() {
    let (session, seen) = session();
    session
        .register_python_udtf(udtf("repeat", PythonEvalType::ArrowTableUdf, &["a", "i"]))
        .unwrap();
    session.sql(T).await.unwrap();
    let want = ["4|4|0", "5|5|0", "5|5|1"];
    for sql in [
        "SELECT t.id, f.* FROM t, LATERAL repeat(t.id, t.id - 3) AS f",
        "SELECT t.id, f.* FROM t LEFT JOIN LATERAL repeat(t.id, t.id - 3) AS f",
    ] {
        assert_eq!(rows(&session, sql).await.unwrap(), want, "{sql}");
    }
    let call = seen.lock().unwrap().calls[0].clone();
    // The left side pruned to what the query reads: `t.id` only, after the two arguments.
    assert_eq!(call.passthrough, [2], "{call:?}");
}

#[tokio::test]
async fn an_arrow_udtf_is_refused_as_a_lateral_target() {
    let (session, _) = session();
    session
        .register_python_udtf(udtf("repeat", PythonEvalType::ArrowUdtf, &["a", "i"]))
        .unwrap();
    session.sql(T).await.unwrap();
    let err = rows(
        &session,
        "SELECT t.id, f.* FROM t, LATERAL repeat(t.id, 1) AS f",
    )
    .await
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::LateralJoinWithArrowUdtfUnsupported,
        "{err}"
    );
}

#[tokio::test]
async fn a_table_argument_is_partitioned_and_sorted_as_written() {
    let (session, seen) = session();
    session
        .register_python_udtf(udtf("rows", PythonEvalType::TableUdf, &["n", "first"]))
        .unwrap();
    session.sql(T).await.unwrap();
    assert_eq!(
        rows(
            &session,
            "SELECT * FROM rows(TABLE(t) PARTITION BY g ORDER BY id DESC)"
        )
        .await
        .unwrap(),
        ["1|5", "2|2", "2|4"]
    );
    let call = seen.lock().unwrap().calls[0].clone();
    assert_eq!(call.table_arg_offsets, [0]);
    assert_eq!(call.partition_child_indexes, [2], "{call:?}");
    assert_eq!(
        rows(
            &session,
            "SELECT * FROM rows(TABLE(t) WITH SINGLE PARTITION)"
        )
        .await
        .unwrap(),
        ["5|2"]
    );
    let err = rows(&session, "SELECT * FROM range(TABLE(t))")
        .await
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
    let err = rows(&session, "SELECT * FROM rows(TABLE(t), TABLE(t))")
        .await
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::TableValuedFunctionTooManyTableArguments,
        "{err}"
    );
}

#[tokio::test]
async fn analyze_runs_at_analysis_and_its_result_reaches_the_tasks() {
    let (session, seen) = session();
    session
        .register_python_udtf(udtf("describe", PythonEvalType::TableUdf, &[]))
        .unwrap();
    session.sql(T).await.unwrap();
    let df = session
        .sql("SELECT * FROM describe(1, 'x', t.id) AS d, t")
        .await;
    // A column of another relation is no constant here: `t` is not in scope of the call.
    assert!(df.is_err());
    let df = session
        .sql("SELECT * FROM describe(1, CAST(NULL AS INT))")
        .await
        .unwrap();
    assert_eq!(df.schema().await.unwrap().fields()[0].name(), "c2");
    let batches = df.collect().await.unwrap();
    assert_eq!(batches[0].column(0).as_primitive::<Int32Type>().value(0), 7);
    {
        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.analyzed.last().unwrap(),
            &vec![
                (DataType::Integer, true, false),
                (DataType::Integer, true, false)
            ]
        );
        assert_eq!(
            seen.calls.last().unwrap().analyze_result.as_deref(),
            Some(&b"pickled"[..])
        );
    }
    // A table argument: analyze asked for one partition ordered by `id` descending.
    assert_eq!(
        rows(&session, "SELECT * FROM describe(TABLE(t))")
            .await
            .unwrap(),
        ["7"]
    );
    let seen = seen.lock().unwrap();
    assert!(seen.analyzed.last().unwrap()[0].2, "{:?}", seen.analyzed);
}

#[tokio::test]
async fn a_udtf_re_registration_is_a_new_version_and_a_fresh_analysis() {
    let (session, seen) = session();
    session
        .register_python_udtf(udtf("describe", PythonEvalType::TableUdf, &[]))
        .unwrap();
    rows(&session, "SELECT * FROM describe(1)").await.unwrap();
    let before = seen.lock().unwrap().analyzed.len();
    let again = session
        .register_python_udtf(udtf("describe", PythonEvalType::TableUdf, &[]))
        .unwrap();
    assert_eq!(again.version, 2);
    rows(&session, "SELECT * FROM describe(1)").await.unwrap();
    assert_eq!(seen.lock().unwrap().analyzed.len(), before + 1);
    let err = session
        .register_python_udtf(udtf("bad", PythonEvalType::ScalarArrow, &["a"]))
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError, "{err}");
}

/// A built-in table function resolves before a UDTF of the same name, as Spark 4.2.0's
/// `spark.sql.functionResolution.sessionOrder` (`second`: builtin, then session) resolves it (review R1-1).
#[tokio::test]
async fn a_built_in_table_function_resolves_before_a_udtf_of_its_name() {
    let (session, seen) = session();
    for name in ["range", "explode"] {
        session
            .register_python_udtf(udtf(name, PythonEvalType::TableUdf, &["a", "i"]))
            .unwrap();
    }
    assert_eq!(
        rows(&session, "SELECT * FROM range(2)").await.unwrap(),
        ["0", "1"]
    );
    assert_eq!(
        rows(&session, "SELECT * FROM explode(array(1, 2))")
            .await
            .unwrap(),
        ["1", "2"]
    );
    assert!(seen.lock().unwrap().calls.is_empty());
}

/// `spark.sql.execution.pythonUDTF.arrow.enabled` has its release's default: `false` before 4.2, `true` at
/// 4.2 (`SQLConf.scala` at v3.5.6, v4.0.0, v4.1.0, v4.2.0), and a set value wins (review R1-2).
#[tokio::test]
async fn the_udtf_arrow_default_follows_the_spark_profile() {
    use krishiv_plan::analyzer::SparkProfile;
    let key = "spark.sql.execution.pythonUDTF.arrow.enabled";
    for (profile, want) in [
        (SparkProfile::V3_5, "false"),
        (SparkProfile::V4_0, "false"),
        (SparkProfile::V4_1, "false"),
        (SparkProfile::V4_2, "true"),
    ] {
        let s = Session::builder().profile(profile).build().unwrap();
        assert_eq!(s.conf().get(key).as_deref(), Some(want), "{profile:?}");
        // Krishiv's own key is the same setting as Spark's alias.
        assert_eq!(
            s.conf()
                .get("krishiv.sql.execution.pythonUDTF.arrow.enabled")
                .as_deref(),
            Some(want),
            "{profile:?}"
        );
        let shown = rows(&s, &format!("SET {key}")).await.unwrap();
        assert_eq!(shown, [format!("{key}|{want}")], "{profile:?}");
        // `SET -v` lists every entry (under its canonical name) with its release's default; `SET` lists only
        // keys that were set.
        let canonical = "krishiv.sql.execution.pythonUDTF.arrow.enabled";
        let verbose = rows(&s, "SET -v").await.unwrap();
        let line = verbose
            .iter()
            .find(|r| r.starts_with(&format!("{canonical}|")))
            .unwrap_or_else(|| panic!("{canonical} is in SET -v"));
        assert!(
            line.starts_with(&format!("{canonical}|{want}|")),
            "{profile:?}: {line}"
        );
        let listed = rows(&s, "SET").await.unwrap();
        assert!(
            listed.iter().all(|r| !r.contains("pythonUDTF")),
            "{profile:?}: {listed:?}"
        );
        assert!(
            s.conf()
                .entries()
                .iter()
                .all(|(k, _)| !k.contains("pythonUDTF")),
            "{profile:?}"
        );
        s.conf().set(key, "true").unwrap();
        assert_eq!(s.conf().get(key).as_deref(), Some("true"), "{profile:?}");
    }
}

/// A slow `analyze` neither holds the statement past its cancel nor keeps running: `interrupt_all` ends the
/// statement with `OPERATION_CANCELED` and the runtime's call is dropped (review R1-3).
#[test]
fn cancelling_a_statement_abandons_its_analyze() {
    let (session, seen) = session();
    let (tx, started) = std::sync::mpsc::channel();
    seen.lock().unwrap().slow_started = Some(tx);
    let dropped = Arc::clone(&seen.lock().unwrap().slow_dropped);
    session
        .register_python_udtf(udtf("slow", PythonEvalType::TableUdf, &[]))
        .unwrap();
    let (done, ended) = std::sync::mpsc::channel();
    let s = session.clone();
    std::thread::spawn(move || {
        let out = futures::executor::block_on(s.sql("SELECT * FROM slow(1)"));
        let _ = done.send(out.map(drop).map_err(|e| e.error_class()));
    });
    started
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("analyze started");
    session.interrupt_all();
    let out = ended
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the statement ended");
    assert_eq!(out, Err(ErrorClass::OperationCanceled));
    assert!(
        dropped.load(std::sync::atomic::Ordering::SeqCst),
        "the analyze call was not dropped"
    );
}

/// Runs `f` on its own thread, failing the test when it does not finish within 30 s (a statement blocking the
/// only thread of a current-thread runtime hangs; review R2-1).
fn within_30s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, ended) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = done.send(f());
    });
    ended
        .recv_timeout(std::time::Duration::from_secs(30))
        // A timeout: the analysis blocked the current-thread runtime.
        .unwrap()
}

fn current_thread() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// On a session whose executor is a current-thread runtime (the deterministic simulation's), the wait for
/// `analyze` would block that runtime's only thread (nothing of it could cancel the wait, and a policy needing
/// the runtime would hang), so the statement is refused with `KRISHIV_UNSUPPORTED_FEATURE` and runs no
/// `analyze`; a UDTF with a declared result needs none and runs (review R2-1).
#[test]
fn analyze_on_a_current_thread_runtime_is_refused_not_hung() {
    let (refused, analyzed, declared) = within_30s(|| {
        current_thread().block_on(async {
            let executor = krishiv_exec::TokioExecutor::deterministic(1).unwrap();
            let session = Session::builder().executor(executor).build().unwrap();
            let fake = Fake::default();
            let seen = Arc::clone(&fake.seen);
            session.set_python_udf_runtime(Arc::new(fake));
            session
                .register_python_udtf(udtf("describe", PythonEvalType::TableUdf, &[]))
                .unwrap();
            session
                .register_python_udtf(udtf("known", PythonEvalType::TableUdf, &["a"]))
                .unwrap();
            let refused = session
                .sql("SELECT * FROM describe(1)")
                .await
                .map(drop)
                .map_err(|e| e.error_class());
            let analyzed = seen.lock().unwrap().analyzed.len();
            let declared = rows(&session, "SELECT * FROM known(1)").await.is_ok();
            (refused, analyzed, declared)
        })
    });
    assert_eq!(refused, Err(ErrorClass::KrishivUnsupportedFeature));
    assert_eq!(analyzed, 0);
    assert!(declared, "a UDTF with a declared result needs no analyze");
}

/// An inner `JOIN LATERAL f(..) ON cond` filters the call's rows by the condition, as Spark 4.2.0 runs it; a
/// `LEFT JOIN LATERAL f(..) ON cond` is refused (review R1-4).
#[tokio::test]
async fn an_inner_lateral_join_with_a_condition_filters_its_rows() {
    let (session, _) = session();
    session
        .register_python_udtf(udtf("repeat", PythonEvalType::TableUdf, &["a", "i"]))
        .unwrap();
    session.sql(T).await.unwrap();
    assert_eq!(
        rows(
            &session,
            "SELECT t.id, f.* FROM t JOIN LATERAL repeat(t.id, t.id - 3) AS f ON f.i > 0"
        )
        .await
        .unwrap(),
        ["5|5|1"]
    );
    let err = rows(
        &session,
        "SELECT t.id, f.* FROM t LEFT JOIN LATERAL repeat(t.id, 1) AS f ON f.i > 0",
    )
    .await
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{err}"
    );
}

/// A frame's plan keeps `analyze`'s result: collecting it again does not run `analyze` again (review R1-9).
#[tokio::test]
async fn a_planned_frame_does_not_run_analyze_again() {
    let (session, seen) = session();
    session
        .register_python_udtf(udtf("describe", PythonEvalType::TableUdf, &[]))
        .unwrap();
    let df = session.sql("SELECT * FROM describe(1)").await.unwrap();
    let before = seen.lock().unwrap().analyzed.len();
    df.clone().collect().await.unwrap();
    df.collect().await.unwrap();
    assert_eq!(seen.lock().unwrap().analyzed.len(), before);
}

/// Allows everything but `Usage` on `System("udf")` (running client Python, ADR 0046 §3b).
struct NoPython;

impl krishiv_session::AccessPolicy for NoPython {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(0)
    }

    fn check<'a>(
        &'a self,
        _: &'a krishiv_session::Principal,
        securable: &'a krishiv_session::Securable,
        privilege: krishiv_session::Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<'a, Result<krishiv_session::AccessDecision, KrishivError>>
    {
        let python = privilege == krishiv_session::Privilege::Usage
            && *securable == krishiv_session::Securable::System("udf".to_owned());
        Box::pin(std::future::ready(Ok(if python {
            krishiv_session::AccessDecision::Deny {
                reason: "no Python".to_owned(),
            }
        } else {
            krishiv_session::AccessDecision::Allow
        })))
    }
}

/// An analysis that runs `analyze` asks `Usage` on `System("udf")` every time, also for a registered UDTF
/// called from SQL: a principal without it is refused and no `analyze` runs (ADR 0046 §3b; S-2).
#[tokio::test]
async fn analyze_needs_usage_on_python_every_time() {
    let session = Session::builder()
        .principal(krishiv_session::Principal::new(
            "ana",
            krishiv_session::PrincipalKind::User,
        ))
        .access_policy(Arc::new(NoPython) as Arc<dyn krishiv_session::AccessPolicy>)
        .build()
        .unwrap();
    let fake = Fake::default();
    let seen = Arc::clone(&fake.seen);
    session.set_python_udf_runtime(Arc::new(fake));
    session
        .register_python_udtf(udtf("describe", PythonEvalType::TableUdf, &[]))
        .unwrap();
    let err = rows(&session, "SELECT * FROM describe(1)")
        .await
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivInsufficientPermissions,
        "{err}"
    );
    assert!(seen.lock().unwrap().analyzed.is_empty());
}

/// An inline UDTF (Spark Connect's `CommonInlineUserDefinedTableFunction`) runs without an entry in the
/// session's library: SQL does not find it by name afterwards (S-1).
#[tokio::test]
async fn an_inline_udtf_runs_without_a_registration() {
    let (session, seen) = session();
    let version = krishiv_session::ANONYMOUS_VERSION_BIT | 1;
    let def = udtf("repeat", PythonEvalType::TableUdf, &["a", "i"]);
    let df = session
        .python_udtf_frame(
            &def,
            version,
            vec![
                krishiv_session::functions::lit(3),
                krishiv_session::functions::lit(2),
            ],
        )
        .unwrap();
    let batches = df.collect().await.unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    assert_eq!(seen.lock().unwrap().calls.len(), 1);
    let err = rows(&session, "SELECT * FROM repeat(1, 1)")
        .await
        .unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::UnresolvableTableValuedFunction,
        "{err}"
    );
    assert!(session.python_udtf_frame(&def, 1, Vec::new()).is_err());
}

/// Each metadata `analyze` may not ask of a `TABLE` argument is refused at analysis with Spark 4.2.0's class
/// (`PythonUDTFAnalyzeResult.applyToTableArgument`), and no task runs.
#[tokio::test]
async fn analyze_metadata_a_table_argument_cannot_hold_is_refused() {
    let (session, seen) = session();
    session.sql(T).await.unwrap();
    for (name, sql, class) in [
        (
            "single_and_keys",
            "SELECT * FROM single_and_keys(TABLE(t))",
            ErrorClass::TableValuedFunctionRequiredMetadataInvalid,
        ),
        (
            "own_partitioning",
            "SELECT * FROM own_partitioning(TABLE(t) PARTITION BY g)",
            ErrorClass::TableValuedFunctionRequiredMetadataIncompatibleWithCall,
        ),
        (
            "aliased_order",
            "SELECT * FROM aliased_order(TABLE(t))",
            ErrorClass::UdtfInvalidAliasInRequestedOrderingStringFromAnalyzeMethod,
        ),
        (
            "unaliased_select",
            "SELECT * FROM unaliased_select(TABLE(t))",
            ErrorClass::UdtfInvalidRequestedSelectedExpressionFromAnalyzeMethodRequiresAlias,
        ),
    ] {
        session
            .register_python_udtf(udtf(name, PythonEvalType::TableUdf, &[]))
            .unwrap();
        let err = rows(&session, sql).await.unwrap_err();
        assert_eq!(err.error_class(), class, "{name}: {err}");
    }
    assert!(
        seen.lock().unwrap().calls.is_empty(),
        "a refused analysis runs no task"
    );
}

/// The partitioning, ordering and selection `analyze` asks of a `TABLE` argument reach the tasks: the rows of
/// one `g` together, `id` descending within it, and the argument's struct holding `id` and `g` as `grp`
/// (`PythonUDTFAnalyzeResult.applyToTableArgument`, v4.2.0). A result that is no struct is refused.
#[tokio::test]
async fn analyze_partitioning_ordering_and_selection_reach_the_tasks() {
    let (session, seen) = session();
    session.sql(T).await.unwrap();
    session
        .register_python_udtf(udtf("keyed", PythonEvalType::TableUdf, &[]))
        .unwrap();
    rows(&session, "SELECT * FROM keyed(TABLE(t))")
        .await
        .unwrap();
    let (call, inputs) = {
        let seen = seen.lock().unwrap();
        (seen.calls.last().unwrap().clone(), seen.inputs.clone())
    };
    assert_eq!(call.table_arg_offsets, [0], "{call:?}");
    assert!(!call.partition_child_indexes.is_empty(), "{call:?}");
    let mut groups: Vec<(String, Vec<Option<i32>>)> = Vec::new();
    for input in inputs.iter().filter(|b| b.num_rows() > 0) {
        let table = input.column(0).as_struct();
        let names: Vec<&str> = table.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(&names[..2], ["id", "grp"], "{names:?}");
        let ids = ints(table.column(0));
        let grp = table.column(1);
        for (r, id) in ids.into_iter().enumerate() {
            let g = arrow_cast::display::array_value_to_string(grp, r).unwrap();
            match groups.last_mut() {
                Some((last, ids_of)) if *last == g => ids_of.push(id),
                _ => groups.push((g, vec![id])),
            }
        }
    }
    groups.sort();
    assert_eq!(
        groups,
        [
            ("a".to_owned(), vec![Some(2), Some(1)]),
            ("b".to_owned(), vec![Some(4), Some(3)]),
            ("c".to_owned(), vec![Some(5)]),
        ]
    );
    session
        .register_python_udtf(udtf("not_a_struct", PythonEvalType::TableUdf, &[]))
        .unwrap();
    let err = rows(&session, "SELECT * FROM not_a_struct(1)")
        .await
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError, "{err}");
}
