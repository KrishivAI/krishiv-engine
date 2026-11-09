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

use arrow_array::cast::AsArray;
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
async fn calls_get_exactly_max_records_rows_per_partition_whatever_the_input_batches() {
    // Spark cuts a partition into batches of exactly `maxRecordsPerBatch`
    // rows and sends the rest last (research/spark-notes/python-udf-batching.md;
    // a pandas UDF sees `len(series)`).
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    s.conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "10")
        .unwrap();
    let got = longs(&s, "SELECT py_plus(id) FROM range(0, 25, 1, 1)").await;
    assert_eq!(got, (1..=25).map(Some).collect::<Vec<_>>());
    assert_eq!(seen.lock().unwrap().calls, vec![10, 10, 5]);
    // Across the engine's 4,096-row input batches too.
    seen.lock().unwrap().calls.clear();
    s.conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "3000")
        .unwrap();
    let got = longs(&s, "SELECT py_plus(id) FROM range(0, 10000, 1, 1)").await;
    assert_eq!(got.len(), 10_000);
    assert_eq!(seen.lock().unwrap().calls, vec![3000, 3000, 3000, 1000]);
}

#[tokio::test]
async fn a_collected_range_keeps_its_order_through_calls_on_many_slices() {
    // Spark collects a range's partitions in order, and a Python UDF keeps
    // each partition's order: the rows come back in range order, as a plain
    // collected `range` does here (several producing drivers once appended
    // their slices in completion order).
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    s.conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .unwrap();
    for _ in 0..3 {
        let got = longs(&s, "SELECT py_plus(id) FROM range(0, 50000, 1, 12)").await;
        assert_eq!(got, (1..=50_000).map(Some).collect::<Vec<_>>());
    }
}

#[tokio::test]
async fn a_limit_over_calls_on_a_range_takes_its_first_rows() {
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    s.conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .unwrap();
    let got = longs(&s, "SELECT py_plus(id) FROM range(0, 50000, 1, 12) LIMIT 5").await;
    assert_eq!(got, (1..=5).map(Some).collect::<Vec<_>>());
}

#[tokio::test]
async fn an_aggregate_over_calls_on_a_range_reads_their_results_in_parallel() {
    // Nothing above the calls needs the range's order, so their results are
    // read by several drivers (design P2-T059 §6.3: one reader made the
    // aggregate a serial stage that capped the pool's scaling).
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    s.conf()
        .set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
        .unwrap();
    let sql = "SELECT sum(py_plus(id)) FROM range(0, 100000, 1, 12)";
    assert_eq!(longs(&s, sql).await, vec![Some(5_000_050_000)]);
    let text: String = s
        .sql(&format!("EXPLAIN ANALYZE {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
        .iter()
        .flat_map(|b| {
            let c = b.column(0);
            (0..c.len())
                .map(|i| match c.data_type() {
                    arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                    _ => c.as_string::<i32>().value(i).to_owned(),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let reader = text
        .lines()
        .find(|l| l.contains("SharedSource"))
        .unwrap_or_else(|| panic!("no shared source in {text}"));
    assert!(!reader.contains(" 1 driver)"), "{reader}");
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

/// Rows the fake was called on, over all calls.
fn rows_called(seen: &Arc<Mutex<Seen>>) -> usize {
    seen.lock().unwrap().calls.iter().sum()
}

#[tokio::test]
async fn a_call_inside_case_runs_on_every_row_as_in_spark() {
    // Spark's ExtractPythonUDFs evaluates the call below the projection:
    // the branch not taken still calls the UDF.
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(
        &s,
        "SELECT CASE WHEN id < 3 THEN py_plus(id) ELSE 0L END FROM range(0, 10, 1, 1)",
    )
    .await;
    assert_eq!(got[..3], [Some(1), Some(2), Some(3)]);
    assert_eq!(rows_called(&seen), 10);
}

#[tokio::test]
async fn a_call_over_a_call_stacks_two_nodes() {
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(&s, "SELECT py_plus(py_plus(id)) FROM range(0, 4, 1, 1)").await;
    assert_eq!(got, vec![Some(2), Some(3), Some(4), Some(5)]);
    assert_eq!(rows_called(&seen), 8, "each level runs over every row");
}

#[tokio::test]
async fn equal_calls_are_evaluated_once() {
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let batches = s
        .sql("SELECT py_plus(id) AS a, py_plus(id) AS b FROM range(0, 5, 1, 1)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
    assert_eq!(rows_called(&seen), 5);
}

#[tokio::test]
async fn a_call_in_a_grouping_key_is_evaluated_below_the_aggregate() {
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(
        &s,
        "SELECT count(*) FROM range(10) GROUP BY py_plus(id) % 2 ORDER BY 1",
    )
    .await;
    assert_eq!(got, vec![Some(5), Some(5)]);
}

#[tokio::test]
async fn a_call_that_is_a_grouping_key_can_be_selected() {
    // Python's `groupBy(udf(x)).agg(...)`: the key is the call and so is an output.
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(
        &s,
        "SELECT py_plus(id) AS k, count(*) FROM range(3) GROUP BY py_plus(id) ORDER BY 1",
    )
    .await;
    assert_eq!(got, vec![Some(1), Some(2), Some(3)]);
}

/// Two-argument `x + y` registered as `py_add` (both `bigint`).
fn add_def() -> PythonUdfDef {
    PythonUdfDef {
        name: "py_add".to_owned(),
        eval_type: PythonEvalType::ArrowBatched,
        return_type: DataType::Long,
        deterministic: true,
        thread_safe: true,
    }
}

#[tokio::test]
async fn a_call_reading_one_side_of_an_outer_join_is_evaluated_below_it() {
    // Spark 4.2.0 accepts `f(a.id) = b.id` for every join type
    // (research/spark-notes/python-udf-errors.md §2).
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(
        &s,
        "SELECT b.id FROM range(3) a LEFT JOIN range(3) b ON py_plus(a.id) = b.id ORDER BY a.id",
    )
    .await;
    assert_eq!(got, vec![Some(1), Some(2), None]);
}

#[tokio::test]
async fn a_call_over_a_call_on_each_side_of_an_outer_join_is_evaluated_below_it() {
    // The outer call reads the inner call's column, in neither side's layout
    // (review H1); each side's nest is still one-sided, so Spark accepts it.
    let (s, _) = session(None);
    s.register_python_udf(plus_one_def("py_plus"));
    let got = longs(
        &s,
        "SELECT b.id FROM range(3) a LEFT JOIN range(5) b \
         ON py_plus(py_plus(a.id)) = py_plus(py_plus(b.id)) - 1 ORDER BY a.id",
    )
    .await;
    assert_eq!(got, vec![Some(1), Some(2), Some(3)]);
}

#[tokio::test]
async fn a_call_reading_both_sides_of_an_outer_join_is_refused_with_its_join_type() {
    let (s, _) = session(None);
    s.register_python_udf(add_def());
    for (join, name) in [("LEFT JOIN", "LEFT OUTER"), ("LEFT ANTI JOIN", "LEFT ANTI")] {
        let sql =
            format!("SELECT a.id FROM range(3) a {join} range(3) b ON py_add(a.id, b.id) = 2");
        let err = match s.sql(&sql).await {
            Ok(df) => df.collect().await.unwrap_err(),
            Err(e) => e,
        };
        assert_eq!(
            err.error_class(),
            ErrorClass::UnsupportedFeaturePythonUdfInOnClause,
            "{join}: {err}"
        );
        assert!(err.to_string().contains(name), "{join}: {err}");
    }
}

/// An unregistered UDF (a DataFrame-side `udf(...)` call, design unit 2c (a)):
/// its version has the top bit set.
const ANON: u32 = 0x8000_0001;

#[tokio::test]
async fn an_unregistered_call_resolves_by_its_encoded_name_and_shows_its_own() {
    let (s, seen) = session(None);
    let call = krishiv_session::functions::python_udf_call(
        &plus_one_def("<lambda>"),
        ANON,
        [krishiv_session::functions::col("id")],
    )
    .unwrap();
    let df = s.range(0, 3, 1, None).select([call]);
    let schema = df.schema().await.unwrap();
    assert_eq!(schema.field(0).name(), "<lambda>(id)");
    let batches = df.collect().await.unwrap();
    let got: Vec<Option<i64>> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            c.iter().collect::<Vec<_>>()
        })
        .collect();
    let mut got = got;
    got.sort_unstable();
    assert_eq!(got, vec![Some(1), Some(2), Some(3)]);
    let seen = seen.lock().unwrap();
    assert!(
        seen.prepared
            .iter()
            .all(|p| p == &("<lambda>".to_owned(), ANON)),
        "{:?}",
        seen.prepared
    );
}

#[tokio::test]
async fn an_unregistered_call_prints_its_display_name_in_every_plan_and_error() {
    // Review 2c I2: the encoded name (NUL-separated fields) never shows. Spark
    // builds the `PythonUDF` at once, so even its parsed plan shows the name.
    let (s, _) = session(None);
    let call = |arg: &str| {
        krishiv_session::functions::python_udf_call(
            &plus_one_def("<lambda>"),
            ANON,
            [krishiv_session::functions::col(arg)],
        )
        .unwrap()
    };
    let text = s
        .range(0, 3, 1, None)
        .select([call("id")])
        .explain(krishiv_session::ExplainMode::Extended)
        .await
        .unwrap();
    assert!(!text.contains('\0'), "{text:?}");
    assert!(text.contains("<lambda>(id)"), "{text}");
    let parsed = text
        .split("== Analyzed Logical Plan ==")
        .next()
        .unwrap_or_default();
    assert!(parsed.contains("<lambda>"), "{parsed}");
    // An error over an unresolved argument, raised before the call resolves.
    let err = s
        .range(0, 3, 1, None)
        .select([call("missing")])
        .collect()
        .await
        .unwrap_err();
    assert!(!err.to_string().contains('\0'), "{err:?}");
    assert_eq!(
        err.error_class(),
        ErrorClass::UnresolvedColumnWithSuggestion,
        "{err}"
    );
}

#[tokio::test]
async fn an_unregistered_call_is_not_reachable_by_its_display_name_from_sql() {
    let (s, _) = session(None);
    let err = s
        .sql("SELECT `<lambda>`(id) FROM range(1)")
        .await
        .unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::UnresolvedRoutine, "{err}");
}

#[tokio::test]
async fn a_registered_udf_is_called_and_shown_by_its_registered_name() {
    // Spark's `PythonUDF` carries the registered name and prints it as made
    // (PySpark 4.2.0 `sql/udf.py:621-631`: `stringLengthString(test)`).
    let (s, seen) = session(None);
    s.register_python_udf(plus_one_def("PyPlus"));
    let df = s.sql("SELECT pyplus(id) FROM range(2)").await.unwrap();
    assert_eq!(df.schema().await.unwrap().field(0).name(), "PyPlus(id)");
    df.collect().await.unwrap();
    let seen = seen.lock().unwrap();
    assert!(
        seen.prepared.iter().all(|p| p == &("PyPlus".to_owned(), 1)),
        "{:?}",
        seen.prepared
    );
}
