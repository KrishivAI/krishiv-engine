// SPDX-License-Identifier: Apache-2.0
//! A registered Python UDF in an observed metric's argument (P3-T084): the observing operator
//! does not run `PythonEval` stages, so it is refused with a clear error, not `INTERNAL_ERROR`.
//! The refusal comes when the query is instantiated (its submit fails), and it still completes
//! the observation with that error: `get` does not wait forever.

use std::time::Duration;

use krishiv_common::error::KrishivError;
use krishiv_plan::PythonEvalType;
use krishiv_session::functions::{call_udf, col, sum};
use krishiv_session::{Observation, PythonUdfDef, Session};
use krishiv_types::DataType;

#[tokio::test]
async fn a_python_udf_in_a_metric_is_unsupported_not_internal() {
    let s = Session::new().unwrap();
    s.register_python_udf(PythonUdfDef {
        name: "py_plus".to_owned(),
        eval_type: PythonEvalType::ArrowBatched,
        return_type: DataType::Long,
        deterministic: true,
        thread_safe: true,
    });
    let obs = Observation::new();
    let df = s
        .sql("SELECT id FROM range(5)")
        .await
        .unwrap()
        .observe(&obs, [sum(call_udf("py_plus", [col("id")])).alias("s")])
        .unwrap();
    let err = match df.count().await {
        Ok(n) => panic!("expected an error, counted {n}"),
        Err(e) => e,
    };
    assert_refused(&err);
    assert_completed_with_the_error(&obs);
}

#[tokio::test]
async fn a_python_udf_over_a_metric_result_is_unsupported_not_internal() {
    let s = Session::new().unwrap();
    s.register_python_udf(PythonUdfDef {
        name: "py_plus".to_owned(),
        eval_type: PythonEvalType::ArrowBatched,
        return_type: DataType::Long,
        deterministic: true,
        thread_safe: true,
    });
    let obs = Observation::new();
    let df = s
        .sql("SELECT id FROM range(5)")
        .await
        .unwrap()
        .observe(&obs, [call_udf("py_plus", [sum(col("id"))]).alias("s")])
        .unwrap();
    let err = match df.count().await {
        Ok(n) => panic!("expected an error, counted {n}"),
        Err(e) => e,
    };
    assert_refused(&err);
    assert_completed_with_the_error(&obs);
}

/// The refusal: `KRISHIV_UNSUPPORTED_FEATURE` (SQLSTATE 0A000) naming the feature.
fn assert_refused(err: &KrishivError) {
    assert_eq!(
        err.error_class().name(),
        "KRISHIV_UNSUPPORTED_FEATURE",
        "{err}"
    );
    assert_eq!(err.sql_state(), Some("0A000"), "{err}");
    assert!(
        err.to_string()
            .contains("a Python UDF in an observed metric (Dataset.observe)"),
        "{err}"
    );
}

/// The failed action completed the observation with its error (Spark completes it whether the
/// action succeeded or not); without that the timeout would give `Ok(None)`.
fn assert_completed_with_the_error(obs: &Observation) {
    let got = obs.get_timeout(Duration::from_secs(5));
    assert!(
        got.is_err(),
        "the observation is not completed with the error: {got:?}"
    );
    if let Err(e) = got {
        assert_refused(&e);
    }
}
