// SPDX-License-Identifier: Apache-2.0
//! A registered Python UDF in an observed metric's argument (P3-T084): the observing operator
//! does not run `PythonEval` stages, so it is refused with a clear error, not `INTERNAL_ERROR`.

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
    let text = err.to_string();
    assert!(text.contains("Python UDF"), "{text}");
    assert_ne!(
        format!("{:?}", err.error_class()),
        "InternalError",
        "{text}"
    );
}
