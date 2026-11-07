// SPDX-License-Identifier: Apache-2.0
//! Python UDFs end to end with a Rust fake runtime (P2-T059 unit 1): a
//! registered UDF resolves by name, is extracted into a `PythonEval` node,
//! and its calls reach the session's runtime by name and version, in slices
//! of at most `spark.sql.execution.arrow.maxRecordsPerBatch` rows, with the
//! results kept in row order. The embedded Python runtime is unit 2.
#![expect(
    clippy::unwrap_used,
    clippy::unwrap_in_result,
    reason = "test code: a poisoned lock in the fake runtime fails the test"
)]

use std::sync::{Arc, Mutex};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use futures::future::BoxFuture;
use krishiv_common::error::{ErrorClass, InternalError};
use krishiv_exec::python_eval::{PreparedUdf, UdfCallSpec, UdfRuntime};
use krishiv_plan::PythonEvalType;
use krishiv_session::{PythonUdfDef, Session};
use krishiv_types::DataType;

/// What the fake saw: each prepared `(name, version)` and each call's rows.
#[derive(Debug, Default)]
struct Seen {
    prepared: Vec<(String, u32)>,
    calls: Vec<usize>,
}

/// `x + 1` over one `bigint` argument (a NULL stays NULL); fails a call
/// whose first value is `fail_on`.
#[derive(Debug)]
struct PlusOne {
    seen: Arc<Mutex<Seen>>,
    fail_on: Option<i64>,
}

impl UdfRuntime for PlusOne {
    fn prepare(&self, spec: &UdfCallSpec) -> krishiv_exec::Result<Arc<dyn PreparedUdf>> {
        self.seen
            .lock()
            .unwrap()
            .prepared
            .push((spec.name.clone(), spec.version));
        Ok(Arc::new(Prepared {
            seen: Arc::clone(&self.seen),
            fail_on: self.fail_on,
        }))
    }
}

struct Prepared {
    seen: Arc<Mutex<Seen>>,
    fail_on: Option<i64>,
}

impl PreparedUdf for Prepared {
    fn call(&self, args: RecordBatch) -> BoxFuture<'static, krishiv_exec::Result<ArrayRef>> {
        self.seen.lock().unwrap().calls.push(args.num_rows());
        let x = args
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        let fail = self
            .fail_on
            .is_some_and(|v| !x.is_empty() && x.value(0) == v);
        Box::pin(async move {
            if fail {
                return Err(InternalError {
                    message: "the UDF raised".to_owned(),
                }
                .into());
            }
            let out: Int64Array = x.iter().map(|v| v.map(|v| v + 1)).collect();
            Ok(Arc::new(out) as ArrayRef)
        })
    }
}

fn plus_one_def(name: &str) -> PythonUdfDef {
    PythonUdfDef {
        name: name.to_owned(),
        eval_type: PythonEvalType::ArrowBatched,
        return_type: DataType::Long,
        deterministic: true,
        thread_safe: true,
    }
}

fn session(fail_on: Option<i64>) -> (Session, Arc<Mutex<Seen>>) {
    let session = Session::new().unwrap();
    let seen = Arc::new(Mutex::new(Seen::default()));
    session.set_python_udf_runtime(Arc::new(PlusOne {
        seen: Arc::clone(&seen),
        fail_on,
    }));
    (session, seen)
}

/// The query's `bigint` column values, in row order.
async fn longs(session: &Session, sql: &str) -> Vec<Option<i64>> {
    let batches = session.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            c.iter().collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn a_projection_calls_the_udf_by_name_and_version() {
    let (s, seen) = session(None);
    let entry = s.register_python_udf(plus_one_def("py_plus"));
    assert_eq!(entry.version, 1);
    let got = longs(&s, "SELECT py_plus(id) FROM range(5) ORDER BY 1").await;
    assert_eq!(got, (1..=5).map(Some).collect::<Vec<_>>());
    let seen = seen.lock().unwrap();
    assert!(
        seen.prepared
            .iter()
            .all(|p| p == &("py_plus".to_owned(), 1)),
        "{:?}",
        seen.prepared
    );
    assert_eq!(seen.calls.iter().sum::<usize>(), 5);
}

#[tokio::test]
async fn a_filter_on_the_udf_keeps_the_matching_rows() {
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(
        &s,
        "SELECT id FROM range(10) WHERE py_plus(id) > 7 ORDER BY id",
    )
    .await;
    assert_eq!(got, vec![Some(7), Some(8), Some(9)]);
}

#[tokio::test]
async fn calls_are_sliced_to_max_records_per_batch_and_rows_stay_in_order() {
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    s.conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "3")
        .unwrap();
    // No ORDER BY: one source, so the rows must come back in input order.
    let got = longs(&s, "SELECT py_plus(id) FROM range(0, 10, 1, 1)").await;
    assert_eq!(got, (1..=10).map(Some).collect::<Vec<_>>());
    let calls = seen.lock().unwrap().calls.clone();
    assert!(calls.iter().all(|&n| n <= 3), "{calls:?}");
    assert_eq!(calls.iter().sum::<usize>(), 10);
}

#[tokio::test]
async fn a_re_registration_is_called_at_its_new_version() {
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let second = s.register_python_udf(plus_one_def("py_plus"));
    assert_eq!(second.version, 2);
    let _ = longs(&s, "SELECT py_plus(id) FROM range(2)").await;
    let seen = seen.lock().unwrap();
    assert!(
        seen.prepared.iter().all(|p| p.1 == 2),
        "{:?}",
        seen.prepared
    );
}

#[tokio::test]
async fn an_error_in_one_call_fails_the_query() {
    let (s, _) = session(Some(0));
    s.register_python_udf(plus_one_def("py_plus"));
    let df = s
        .sql("SELECT py_plus(id) FROM range(0, 10, 1, 1)")
        .await
        .unwrap();
    assert!(df.collect().await.is_err());
}

#[tokio::test]
async fn without_a_runtime_the_query_fails_with_udf_not_available() {
    let s = Session::new().unwrap();
    s.register_python_udf(plus_one_def("py_plus"));
    let err = match s.sql("SELECT py_plus(id) FROM range(3)").await {
        Ok(df) => df.collect().await.unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(
        err.error_class(),
        ErrorClass::KrishivUdfNotAvailable,
        "{err}"
    );
}

#[tokio::test]
async fn another_session_does_not_see_the_udf() {
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let other = Session::new().unwrap();
    assert!(other.sql("SELECT py_plus(1)").await.is_err());
}
