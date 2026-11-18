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
use krishiv_exec::python_udtf::{UdtfAnalysis, UdtfAnalyzeSpec, UdtfCall, UdtfOrdering};
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
    /// Each task's input rows.
    tasks: Vec<usize>,
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
        let table = spec.args.iter().any(|a| a.is_table);
        let schema = DataType::Struct(StructType::new([StructField::new(
            format!("c{}", spec.args.len()),
            DataType::Integer,
            true,
        )]));
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
                self.seen.lock().unwrap().tasks.push(input.num_rows());
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
