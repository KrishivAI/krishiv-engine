// SPDX-License-Identifier: Apache-2.0
//! Python table functions' `analyze` on the pool (P3-T014 PR-26c-3 part B; design P3-T085 §3.3 "UDTF
//! analyze"): Spark runs a UDTF's static `analyze` at analysis time in a Python worker
//! (`UserDefinedPythonFunction.scala`, `pyspark/sql/worker/analyze_udtf.py` at v4.2.0). Here it runs on a
//! worker of the session's owner, in the sandbox like any call, on a pool call thread (never on the caller):
//! wire `analyze` over the loaded payload, answered by `analyzed` with the result `udtf.py` `analyze_udtf`
//! builds. A failure in Python is `TABLE_VALUED_FUNCTION_FAILED_TO_ANALYZE_IN_PYTHON`.
//!
//! The UDTF's tasks (eval types 300, 301, 302) run as relation tasks (`relation.rs`, `udtf.py`
//! `make_udtf_runner`); `@arrow_udtf` (302) has no `analyze` (pyspark refuses one when the UDTF is made,
//! `INVALID_ARROW_UDTF_WITH_ANALYZE`), so only a 300 or 301 payload is analyzed here.

use std::sync::{Arc, Mutex};

use futures::channel::oneshot;
use futures::future::BoxFuture;
use krishiv_common::error::{
    InternalError, KrishivError, TableValuedFunctionFailedToAnalyzeInPython,
};
use krishiv_exec::python_udtf::{UdtfAnalysis, UdtfAnalyzeSpec, UdtfOrdering, UdtfSelect};
use krishiv_types::DataType;
use serde_json::{Value, json};

use crate::artifacts::ArtifactSet;
use crate::pool::{Owner, Pool};
use crate::runtime::{CallState, CancelOnDrop, Registered, check_eval_type, with_worker};

/// Whether a payload of eval type `registered` has an `analyze` the pool runs: a Python UDTF (300) or an
/// Arrow-optimized one (301).
fn analyzable(registered: i32, _called: i32) -> bool {
    matches!(registered, 300 | 301)
}

/// Runs `registered`'s `analyze` for `spec` on a worker of `owner` (see the module doc).
pub(crate) fn analyze(
    pool: &Pool,
    owner: &Owner,
    registered: Result<Registered, KrishivError>,
    artifacts: Arc<ArtifactSet>,
    spec: &UdtfAnalyzeSpec,
) -> BoxFuture<'static, Result<UdtfAnalysis, KrishivError>> {
    let ready = |r| -> BoxFuture<'static, Result<UdtfAnalysis, KrishivError>> {
        Box::pin(std::future::ready(r))
    };
    let registered = match registered.and_then(|r| {
        let code = r.payload.eval_type;
        check_eval_type(
            &spec.name,
            spec.version,
            &r,
            code,
            "a table function's analyze",
            analyzable,
        )
        .map(|()| r)
    }) {
        Ok(r) => r,
        Err(e) => return ready(Err(e)),
    };
    let args = Value::Array(
        spec.args
            .iter()
            .map(|a| {
                let mut arg = json!({
                    "type": a.data_type.json(),
                    "is_table": a.is_table,
                    "name": a.name,
                });
                if let (Some(v), Some(map)) = (&a.value, arg.as_object_mut()) {
                    map.insert("value".to_owned(), v.clone());
                }
                arg
            })
            .collect(),
    );
    let name = spec.name.clone();
    let (done, result) = oneshot::channel();
    let call = Arc::new(Mutex::new(CallState::default()));
    let cancel = CancelOnDrop(Arc::clone(&call));
    let (thread_pool, thread_owner) = (pool.clone(), owner.clone());
    let submitted = pool.submit(
        owner,
        Box::new(move || {
            // A cancel before the worker starts or while it runs ends the call (`with_worker`).
            let result = with_worker(
                &thread_pool,
                &thread_owner,
                &registered,
                &artifacts,
                &call,
                |worker| worker.analyze(registered.id, &name, &args),
            )
            .and_then(|answer| analysis(&answer));
            let _ = done.send(result);
        }),
    );
    if let Err(e) = submitted {
        return ready(Err(e));
    }
    Box::pin(async move {
        let _cancel = cancel;
        result.await.unwrap_or_else(|_| {
            Err(InternalError {
                message: "a Python UDTF's analyze ended without a result".to_owned(),
            }
            .into())
        })
    })
}

/// The worker's `analyzed` result as a [`UdtfAnalysis`]. The answer is the worker's (untrusted): one it did
/// not form as `udtf.py` does fails the analysis.
fn analysis(answer: &Value) -> Result<UdtfAnalysis, KrishivError> {
    let bad = |what: &str| -> KrishivError {
        TableValuedFunctionFailedToAnalyzeInPython {
            msg: format!("the Python worker's analyze result is unreadable: {what}"),
        }
        .into()
    };
    let text = |v: &Value, what: &str| v.as_str().map(str::to_owned).ok_or_else(|| bad(what));
    let schema = DataType::from_json(&text(&answer["schema"], "schema")?)
        .map_err(|e| bad(&format!("its schema: {e}")))?;
    if !matches!(schema, DataType::Struct(_)) {
        return Err(bad("its schema is not a struct"));
    }
    let pickled = unhex(&text(&answer["pickled"], "pickled")?).ok_or_else(|| bad("pickled"))?;
    let list = |key: &str| answer[key].as_array().cloned().ok_or_else(|| bad(key));
    let partition_by = list("partition_by")?
        .iter()
        .map(|v| text(v, "partition_by"))
        .collect::<Result<Vec<_>, _>>()?;
    let order_by = list("order_by")?
        .iter()
        .map(|v| {
            Ok(UdtfOrdering {
                name: text(&v["name"], "order_by")?,
                ascending: v["ascending"].as_bool().ok_or_else(|| bad("order_by"))?,
                nulls_first: v["nulls_first"].as_bool(),
            })
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    let select = list("select")?
        .iter()
        .map(|v| {
            Ok(UdtfSelect {
                name: text(&v["name"], "select")?,
                alias: text(&v["alias"], "select")?,
            })
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    Ok(UdtfAnalysis {
        schema,
        pickled: Arc::from(pickled),
        with_single_partition: answer["with_single_partition"]
            .as_bool()
            .ok_or_else(|| bad("with_single_partition"))?,
        partition_by,
        order_by,
        select,
    })
}

/// The bytes of lower-case or upper-case hex `text`.
fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            text.get(i..i + 2)
                .and_then(|b| u8::from_str_radix(b, 16).ok())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::hex;

    #[test]
    fn hex_round_trips() {
        let bytes = [0u8, 1, 0x7f, 0x80, 0xff];
        assert_eq!(unhex(&hex(&bytes)), Some(bytes.to_vec()));
        assert_eq!(unhex("abc"), None);
        assert_eq!(unhex("zz"), None);
    }

    #[test]
    fn an_analysis_is_read_and_a_bad_one_refused() {
        let good = json!({
            "schema": r#"{"type":"struct","fields":[{"name":"a","type":"long","nullable":true,"metadata":{}}]}"#,
            "pickled": "0102",
            "with_single_partition": true,
            "partition_by": ["x"],
            "order_by": [{"name": "y", "ascending": false, "nulls_first": null}],
            "select": [{"name": "z", "alias": ""}],
        });
        let a = analysis(&good).unwrap();
        assert!(matches!(a.schema, DataType::Struct(_)));
        assert_eq!(&*a.pickled, &[1, 2]);
        assert!(a.with_single_partition);
        assert_eq!(a.partition_by, ["x"]);
        assert_eq!(
            a.order_by,
            [UdtfOrdering {
                name: "y".to_owned(),
                ascending: false,
                nulls_first: None
            }]
        );
        let mut bad = good.clone();
        bad["schema"] = json!("\"long\"");
        let e = analysis(&bad).unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::TableValuedFunctionFailedToAnalyzeInPython
        );
        let mut bad = good;
        bad["pickled"] = json!("x");
        assert!(analysis(&bad).is_err());
    }
}
