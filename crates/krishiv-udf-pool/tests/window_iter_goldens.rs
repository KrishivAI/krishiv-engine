// SPDX-License-Identifier: Apache-2.0
//! PySpark 4.2.0's answers for Python window aggregates (eval types 203, 253) and scalar iterators (204, 251),
//! replayed through a [`Session`] whose Python UDFs run on the worker pool.
//!
//! The cases and their answers are `python/tests/pyspark_udf/window_iter/{cases.py,goldens.json}`, a real local
//! PySpark 4.2.0 run. The embedded Python facade does not run these kinds yet (its in-process runtime is PR-48),
//! so `test_window_iter_goldens.py` lists them all as known gaps; this test runs the same UDF bodies and the same
//! window specs as SQL over the pool runtime (the one Spark Connect uses) and compares each case with the golden:
//! the schema and rows, or `PYTHON_EXCEPTION` carrying the worker's `[CLASS] message` line.
//!
//! As `tests/relations.rs`: the interpreter comes from `KRISHIV_TEST_UDF_PYTHON`, and without it the test passes
//! after a note unless `KRISHIV_UDF_POOL_REQUIRED` is set.
#![cfg(target_os = "linux")]
#![expect(
    clippy::print_stderr,
    clippy::expect_used,
    clippy::panic,
    reason = "the skip note when no interpreter is configured; test helpers sit outside #[test] fns"
)]

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::ErrorClass;
use krishiv_plan::function::PythonEvalType;
use krishiv_session::{PythonUdfDef, Session};
use krishiv_types::DataType;
use krishiv_udf_pool::{Launcher, Owner, Pool, PoolConfig, PythonPayload, SessionUdfs};
use serde_json::Value;

/// The child entry: run as the trampoline when started with `--` and spec arguments.
#[test]
fn worker_child() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let Some(split) = args.iter().position(|a| a == "--") else {
        return;
    };
    let spec: Vec<OsString> = args.get(split + 1..).unwrap_or_default().to_vec();
    if spec.is_empty() {
        return;
    }
    let code = krishiv_udf_sandbox::trampoline_main(spec);
    let failed = code == ExitCode::from(krishiv_udf_sandbox::EXIT_SANDBOX_FAILED);
    std::process::exit(if failed { 125 } else { 1 });
}

fn python() -> Option<PathBuf> {
    match std::env::var_os("KRISHIV_TEST_UDF_PYTHON") {
        Some(p) => Some(PathBuf::from(p)),
        None => {
            assert!(
                std::env::var_os("KRISHIV_UDF_POOL_REQUIRED").is_none(),
                "KRISHIV_UDF_POOL_REQUIRED is set but KRISHIV_TEST_UDF_PYTHON is not"
            );
            eprintln!("note: KRISHIV_TEST_UDF_PYTHON is not set; the golden replay did not run");
            None
        }
    }
}

fn pool(python: &Path) -> Pool {
    let mut config = PoolConfig::default();
    config.executables = vec![python.display().to_string()];
    config.launcher = Launcher {
        program: std::env::current_exe().expect("test binary"),
        args: ["worker_child", "--exact", "--nocapture", "--"]
            .map(OsString::from)
            .to_vec(),
    };
    let pool = Pool::new(config).expect("pool");
    assert!(pool.refusal().is_none(), "{:?}", pool.refusal());
    pool
}

/// Spark's `command` for `body` (a Python function `f`) returning `return_type`, pickled by the interpreter.
fn command(python: &Path, body: &str, return_type: &str, eval_type: i32) -> PythonPayload {
    let code = format!(
        "import sys\nfrom pyspark import cloudpickle\nfrom pyspark.sql.types import *\n{body}\n\
         sys.stdout.buffer.write(cloudpickle.dumps((f, {return_type})))"
    );
    let out = Command::new(python)
        .args(["-c", &code])
        .output()
        .expect("run the interpreter");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let version = Command::new(python)
        .args([
            "-c",
            "import sys; print('%d.%d' % sys.version_info[:2], end='')",
        ])
        .output()
        .expect("version");
    PythonPayload {
        command: Arc::from(out.stdout),
        eval_type,
        python_ver: String::from_utf8_lossy(&version.stdout).into_owned(),
    }
}

/// The UDFs of `cases.py` (`_window_udfs`, `_iter_udfs`): name, eval type, return type and body (`f`).
/// Kept by hand in step with those bodies (`cases.py` says so where they are defined).
const UDFS: &[(&str, PythonEvalType, DataType, &str)] = &[
    (
        "pandas_sum",
        PythonEvalType::GroupedAggPandas,
        DataType::Double,
        "def f(s):\n    return float(s.sum())",
    ),
    (
        "pandas_mean",
        PythonEvalType::GroupedAggPandas,
        DataType::Double,
        "def f(s):\n    return float(s.mean())",
    ),
    (
        "pandas_count",
        PythonEvalType::GroupedAggPandas,
        DataType::Long,
        "def f(s):\n    return len(s)",
    ),
    (
        "arrow_sum",
        PythonEvalType::GroupedAggArrow,
        DataType::Double,
        "import pyarrow.compute as pc\ndef f(a):\n    return pc.sum(a).as_py()",
    ),
    (
        "arrow_count",
        PythonEvalType::GroupedAggArrow,
        DataType::Long,
        "def f(a):\n    return len(a)",
    ),
    (
        "pandas_plus_one",
        PythonEvalType::ScalarPandasIter,
        DataType::Long,
        "def f(it):\n    for s in it:\n        yield s + 1",
    ),
    (
        "pandas_rebatched",
        PythonEvalType::ScalarPandasIter,
        DataType::Long,
        "import pandas as pd\ndef f(it):\n    whole = pd.concat(list(it), ignore_index=True)\n    \
         yield whole.iloc[:3] * 2\n    yield whole.iloc[3:] * 2",
    ),
    (
        "pandas_twice",
        PythonEvalType::ScalarPandasIter,
        DataType::Long,
        "def f(it):\n    for s in it:\n        yield s\n        yield s",
    ),
    (
        "pandas_first_only",
        PythonEvalType::ScalarPandasIter,
        DataType::Long,
        "def f(it):\n    yield next(it)",
    ),
    (
        "pandas_short",
        PythonEvalType::ScalarPandasIter,
        DataType::Long,
        "def f(it):\n    for s in it:\n        yield s.iloc[:1]",
    ),
    (
        "pandas_not_iterator",
        PythonEvalType::ScalarPandasIter,
        DataType::Long,
        "import pandas as pd\ndef f(it):\n    return pd.Series([1, 2])",
    ),
    (
        "arrow_plus_one",
        PythonEvalType::ScalarArrowIter,
        DataType::Long,
        "import pyarrow.compute as pc\ndef f(it):\n    for a in it:\n        yield pc.add(a, 1)",
    ),
    (
        "arrow_twice",
        PythonEvalType::ScalarArrowIter,
        DataType::Long,
        "def f(it):\n    for a in it:\n        yield a\n        yield a",
    ),
    (
        "arrow_first_only",
        PythonEvalType::ScalarArrowIter,
        DataType::Long,
        "def f(it):\n    yield next(it)",
    ),
    (
        "arrow_short",
        PythonEvalType::ScalarArrowIter,
        DataType::Long,
        "def f(it):\n    for a in it:\n        yield a.slice(0, 1)",
    ),
];

/// `cases.py`'s window specs (`_window_specs`) as SQL.
fn window(spec: &str) -> &'static str {
    match spec {
        "partition" => "PARTITION BY g",
        "ordered_default" => "PARTITION BY g ORDER BY o",
        "rows_unbounded" => {
            "PARTITION BY g ORDER BY o, id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
        }
        "rows_sliding" => "PARTITION BY g ORDER BY o, id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING",
        "range_back" => "PARTITION BY g ORDER BY o RANGE BETWEEN 1 PRECEDING AND CURRENT ROW",
        "rows_to_end" => {
            "PARTITION BY g ORDER BY o, id ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
        }
        "rows_ahead" => "PARTITION BY g ORDER BY o, id ROWS BETWEEN 1 FOLLOWING AND 2 FOLLOWING",
        other => panic!("no window spec {other}"),
    }
}

/// `cases.py`'s `WINDOW_ROWS` and `ITER_ROWS`.
const WINDOW_INPUT: &str = "(VALUES (1L, 'a', 1L, 1.0D), (2L, 'a', 2L, 2.0D), (3L, 'a', 2L, 4.0D), \
    (4L, 'a', 3L, CAST(NULL AS DOUBLE)), (5L, 'a', 5L, 8.0D), (6L, 'b', 1L, 16.0D), (7L, 'b', 4L, 32.0D), \
    (8L, 'c', 1L, CAST(NULL AS DOUBLE)) AS t(id, g, o, v))";
const ITER_INPUT: &str = "(VALUES (1L, 10L), (2L, 20L), (3L, CAST(NULL AS BIGINT)), (4L, 40L), (5L, 50L) \
    AS t(id, x))";

/// The SQL of golden case `case` (`cases.py` `_cases`).
fn sql(case: &str) -> String {
    let parts: Vec<&str> = case.split('/').collect();
    match parts.as_slice() {
        ["window", kind, "mixed_bounds"] => {
            let agg = if *kind == "pandas" { "sum" } else { "count" };
            format!(
                "SELECT id, {kind}_{agg}(v) OVER ({}) AS whole, {kind}_{agg}(v) OVER ({}) AS sliding, \
                 {kind}_count(v) OVER ({}) AS unbounded, {kind}_count(v) OVER ({}) AS ahead \
                 FROM {WINDOW_INPUT} ORDER BY id",
                window("partition"),
                window("rows_sliding"),
                window("rows_unbounded"),
                window("rows_ahead"),
            )
        }
        ["window", kind, spec] => {
            let udfs: &[&str] = if *kind == "pandas" {
                &["sum", "mean", "count"]
            } else {
                &["sum", "count"]
            };
            let columns: Vec<String> = udfs
                .iter()
                .map(|u| format!("{kind}_{u}(v) OVER ({}) AS {u}", window(spec)))
                .collect();
            format!(
                "SELECT id, {} FROM {WINDOW_INPUT} ORDER BY id",
                columns.join(", ")
            )
        }
        ["iter", udf] => format!("SELECT id, {udf}(x) AS r FROM {ITER_INPUT} ORDER BY id"),
        _ => panic!("no case {case}"),
    }
}

fn simple_type(t: &ArrowType) -> &'static str {
    match t {
        ArrowType::Int64 => "bigint",
        ArrowType::Float64 => "double",
        ArrowType::Int32 => "int",
        _ => "?",
    }
}

/// The golden form of `batches`: `{"schema": struct<...>, "rows": [[...]]}` (NaN and infinities as text).
fn answer(batches: &[RecordBatch]) -> Value {
    let schema = batches.first().map_or_else(String::new, |b| {
        let fields: Vec<String> = b
            .schema_ref()
            .fields()
            .iter()
            .map(|f| format!("{}:{}", f.name(), simple_type(f.data_type())))
            .collect();
        format!("struct<{}>", fields.join(","))
    });
    let mut rows = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let row: Vec<Value> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(r) {
                        Value::Null
                    } else if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                        Value::from(a.value(r))
                    } else if let Some(a) = c.as_any().downcast_ref::<Float64Array>() {
                        let v = a.value(r);
                        if v.is_finite() {
                            Value::from(v)
                        } else {
                            Value::from(format!("{v:?}").to_lowercase())
                        }
                    } else if let Some(a) = c.as_any().downcast_ref::<StringViewArray>() {
                        Value::from(a.value(r))
                    } else {
                        Value::from(format!("{:?}", c.data_type()))
                    }
                })
                .collect();
            rows.push(Value::Array(row));
        }
    }
    serde_json::json!({ "schema": schema, "rows": rows })
}

/// What differs between Krishiv's answer to `case` and Spark's `want`, if anything.
async fn difference(session: &Session, case: &str, want: &Value) -> Option<String> {
    let run = async {
        let df = session.sql(&sql(case)).await?;
        df.collect().await
    };
    match run.await {
        Ok(batches) => {
            let got = answer(&batches);
            if want.get("error").is_some() {
                Some(format!("returned {got}, Spark raised {}", want["error"]))
            } else if &got == want {
                None
            } else {
                Some(format!("returned {got}, Spark {want}"))
            }
        }
        Err(e) => {
            let Some(err) = want.get("error") else {
                return Some(format!("raised {e}"));
            };
            if err["condition"] != "PYTHON_EXCEPTION"
                || e.error_class() != ErrorClass::PythonException
            {
                return Some(format!("raised {e}, Spark {err}"));
            }
            let worker = &err["worker"];
            let line = format!(
                "[{}] {}",
                worker["error_class"].as_str().unwrap_or_default(),
                worker["message"].as_str().unwrap_or_default()
            );
            (!e.to_string().contains(&line)).then(|| format!("no {line:?} in: {e}"))
        }
    }
}

/// A session whose Python UDFs (`UDFS`) run on a new pool, with `conf` set.
fn session_on_pool(
    py: &Path,
    conf: &serde_json::Map<String, Value>,
) -> (Session, Arc<SessionUdfs>) {
    let pool = pool(py);
    let session = Session::new().expect("session");
    for (k, v) in conf {
        session
            .conf()
            .set(k, v.as_str().expect("conf value"))
            .expect("set conf");
    }
    let udfs = Arc::new(SessionUdfs::new(
        pool,
        Owner {
            principal: "alice".to_owned(),
            session: "s1".to_owned(),
            tenant: None,
            capped: true,
        },
    ));
    for (name, eval_type, return_type, body) in UDFS {
        let entry = session.register_python_udf(PythonUdfDef {
            name: (*name).to_owned(),
            eval_type: *eval_type,
            return_type: return_type.clone(),
            deterministic: true,
            thread_safe: true,
        });
        let returns = if *return_type == DataType::Double {
            "DoubleType()"
        } else {
            "LongType()"
        };
        udfs.register(
            *name,
            entry.version,
            command(py, body, returns, eval_type.code()),
        );
    }
    session.set_python_udf_runtime(Arc::clone(&udfs) as Arc<_>);
    (session, udfs)
}

/// Every case of `goldens.json` answers as PySpark 4.2.0 did, run on the pool.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn window_aggregates_and_scalar_iterators_match_pyspark_4_2_0() {
    let Some(py) = python() else { return };
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/tests/pyspark_udf/window_iter/goldens.json");
    let goldens: Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("goldens")).expect("json");
    assert_eq!(goldens["spark"], "4.2.0");
    let (session, udfs) = session_on_pool(&py, goldens["conf"].as_object().expect("conf"));
    let cases = goldens["cases"].as_object().expect("cases");
    assert_eq!(cases.len(), 26);
    let mut differ = BTreeMap::new();
    for (case, want) in cases {
        if let Some(diff) = difference(&session, case, want).await {
            differ.insert(case.clone(), diff);
        }
    }
    udfs.release();
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The rows of `sql` as JSON (`answer`), or its error's class and text.
async fn query(session: &Session, sql: &str) -> Result<Value, (ErrorClass, String)> {
    let run = async {
        let df = session.sql(sql).await?;
        df.collect().await
    };
    run.await
        .map(|batches| answer(&batches)["rows"].clone())
        .map_err(|e| (e.error_class(), e.to_string()))
}

/// A grouped-aggregate Python UDF (pandas 202, Arrow 252) is an aggregate in SQL `GROUP BY` and in a global
/// aggregate (review R1-6). The shapes Spark 4.2.0 refuses are refused with its classes:
/// - a Python UDAF beside a built-in aggregate in one aggregation: `INVALID_PANDAS_UDF_PLACEMENT`
///   (`SparkStrategies.scala` `Aggregation`, `QueryCompilationErrors.invalidPandasUDFPlacementError`);
/// - `DISTINCT` or `FILTER` on a Python UDAF: `INVALID_SQL_SYNTAX.FUNCTION_WITH_UNSUPPORTED_SYNTAX`
///   (`FunctionResolution.scala` `checkUnsupportedAggregateClause`), with `syntax` `DISTINCT` or
///   `FILTER CLAUSE` and the function named `pythonudaf` (the UDAF's `prettyName`: no alias tag, so its
///   class name).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grouped_aggregate_udfs_in_sql_group_by() {
    let Some(py) = python() else { return };
    let (session, udfs) = session_on_pool(&py, &serde_json::Map::new());
    let ok = [
        (
            format!("SELECT g, pandas_sum(v) AS s FROM {WINDOW_INPUT} GROUP BY g ORDER BY g"),
            serde_json::json!([["a", 15.0], ["b", 48.0], ["c", 0.0]]),
        ),
        (
            format!("SELECT g, arrow_count(v) AS n FROM {WINDOW_INPUT} GROUP BY g ORDER BY g"),
            serde_json::json!([["a", 5], ["b", 2], ["c", 1]]),
        ),
        (
            format!(
                "SELECT g, pandas_count(v) + 1 AS n, pandas_mean(o) AS m FROM {WINDOW_INPUT} \
                 GROUP BY g ORDER BY g"
            ),
            serde_json::json!([["a", 6, 2.6], ["b", 3, 2.5], ["c", 2, 1.0]]),
        ),
        (
            format!("SELECT arrow_sum(v) AS s FROM {WINDOW_INPUT}"),
            serde_json::json!([[63.0]]),
        ),
    ];
    for (sql, want) in ok {
        let got = query(&session, &sql)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e:?}"));
        assert_eq!(got, want, "{sql}");
    }
    let refused = [
        (
            format!("SELECT g, pandas_sum(v), sum(v) FROM {WINDOW_INPUT} GROUP BY g"),
            ErrorClass::InvalidPandasUdfPlacement,
            "`pandas_sum`",
        ),
        (
            format!("SELECT arrow_count(v), count(*) FROM {WINDOW_INPUT}"),
            ErrorClass::InvalidPandasUdfPlacement,
            "`arrow_count`",
        ),
        (
            format!("SELECT g, pandas_sum(DISTINCT v) FROM {WINDOW_INPUT} GROUP BY g"),
            ErrorClass::InvalidSqlSyntaxFunctionWithUnsupportedSyntax,
            "`pythonudaf` does not support DISTINCT",
        ),
        (
            format!("SELECT arrow_sum(v) FILTER (WHERE v > 1) FROM {WINDOW_INPUT}"),
            ErrorClass::InvalidSqlSyntaxFunctionWithUnsupportedSyntax,
            "`pythonudaf` does not support FILTER CLAUSE",
        ),
    ];
    for (sql, class, text) in refused {
        let Err((got, message)) = query(&session, &sql).await else {
            panic!("{sql} ran; Spark refuses it")
        };
        assert_eq!(got, class, "{sql}: {message}");
        assert!(message.contains(text), "{sql}: {message}");
    }
    udfs.release();
}
