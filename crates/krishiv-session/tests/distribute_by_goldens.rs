// SPDX-License-Identifier: Apache-2.0
//! When `DISTRIBUTE BY` / `CLUSTER BY` evaluate their expressions, against
//! `tests/fixtures/spark-4.2.0/distribute-by-goldens.json` (a real Spark
//! 4.2.0 run, generator `tests/scripts/gen_distribute_by_goldens.py`): the
//! rows, or the error class with the fragment and span of its SQL context,
//! under the default settings, with ANSI off and with AQE off.
//!
//! Spark keeps the repartition under `count(*)`, `SELECT 1`, `LIMIT 1`,
//! aggregates and joins and evaluates its expressions on every row; filters
//! move below it; an outer repartition directly over an inner one drops the
//! inner one; empty input and `LIMIT 0` evaluate nothing.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// Cases Krishiv does not match: the query, whether ANSI is off for it,
/// and why. A fixed one fails the test until it is removed.
const DIFFERENT: &[(&str, bool, &str)] = &[
    // Spark's `ConstantFolding` evaluates a foldable expression when it
    // optimizes, so it raises with no row to evaluate; Krishiv's folding
    // keeps a raising call in the plan (`krishiv-optimizer` invariants),
    // which raises only on a row, as for `SELECT CAST('a' AS INT) FROM t
    // WHERE false`.
    (
        "SELECT x FROM t WHERE false DISTRIBUTE BY CAST('a' AS INT)",
        false,
        "folding does not raise",
    ),
    // Spark's shuffle evaluates every row before the limit reads one; on
    // one worker the exchange runs inside each driver's pipeline, so a limit
    // above stops it after the rows it needs (a distributed run's map
    // stage reads every row, as Spark's: `docs/known-differences.md`).
    (
        "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) LIMIT 1",
        false,
        "a limit stops the repartition early",
    ),
    // Spark moves every deterministic conjunct of a filter down
    // (`splitConjunctivePredicates(..).partition(_.deterministic)`, any
    // position); Krishiv moves only the deterministic prefix, so `x <> 'x'`
    // after `rand() < 2` stays above the repartition and the cast sees the
    // row. A general pushdown difference (over a sort too), recorded in
    // `docs/known-differences.md`; not this node's.
    (
        "SELECT * FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) WHERE rand() < 2 AND x <> 'x'",
        false,
        "only the deterministic prefix of a filter moves",
    ),
    // Implicit string casts with ANSI off (`a.x = u.k`, string and int) are
    // `KRISHIV_UNSUPPORTED_FEATURE` with or without the repartition.
    (
        "SELECT count(*) FROM (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) a JOIN u ON a.x = u.k",
        true,
        "implicit string casts with ANSI off",
    ),
    (
        "SELECT u.k FROM u JOIN (SELECT x FROM t DISTRIBUTE BY CAST(x AS INT)) a ON a.x = u.k",
        true,
        "implicit string casts with ANSI off",
    ),
];

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/distribute-by-goldens.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn cell(c: &dyn Array, row: usize) -> Value {
    if c.is_null(row) {
        return Value::Null;
    }
    Value::String(match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
        other => format!("<{other}>"),
    })
}

/// What `sql` gives: `{"rows": [...]}` sorted as the generator sorts them,
/// or `{"error": {class, fragment, start, stop}}`.
fn outcome(s: &Session, sql: &str) -> Value {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(batches) => {
            let mut rows: Vec<Vec<Value>> = batches
                .iter()
                .flat_map(|b| {
                    (0..b.num_rows())
                        .map(|r| b.columns().iter().map(|c| cell(c.as_ref(), r)).collect())
                        .collect::<Vec<_>>()
                })
                .collect();
            rows.sort_by_key(|r| {
                r.iter()
                    .map(|v| v.as_str().unwrap_or_default().to_owned())
                    .collect::<Vec<_>>()
            });
            serde_json::json!({ "rows": rows })
        }
        Err(e) => {
            let ctx = e.query_contexts().first();
            serde_json::json!({ "error": {
                "class": e.error_class().name(),
                "fragment": ctx.map(|c| c.fragment.clone()),
                "start": ctx.map(|c| c.start),
                "stop": ctx.map(|c| c.stop),
            }})
        }
    }
}

/// The fixture's case in the shape [`outcome`] gives.
fn expected(case: &Value) -> Value {
    if let Some(rows) = case.get("rows") {
        return serde_json::json!({ "rows": rows });
    }
    let e = &case["error"];
    let ctx = e.get("context");
    serde_json::json!({ "error": {
        "class": e["class"],
        "fragment": ctx.map(|c| c["fragment"].clone()),
        "start": ctx.map(|c| c["start"].clone()),
        "stop": ctx.map(|c| c["stop"].clone()),
    }})
}

fn session(fixture: &Value, conf: &serde_json::Map<String, Value>) -> Session {
    let s = Session::new().unwrap();
    let mut setup: Vec<String> = fixture["views"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, q)| format!("CREATE TEMPORARY VIEW {name} AS {}", q.as_str().unwrap()))
        .collect();
    // AQE is Spark's alone: the cases with it off only show that the
    // results do not depend on it.
    setup.extend(
        conf.iter()
            .filter(|(k, _)| *k != "spark.sql.adaptive.enabled")
            .map(|(k, v)| format!("SET {k}={}", v.as_str().unwrap())),
    );
    for sql in setup {
        block_on(async { s.sql(&sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    s
}

#[test]
fn distribute_by_evaluates_as_spark() {
    let fixture = fixture();
    assert_eq!(fixture["spark_version"], "4.2.0");
    let mut report = String::new();
    let mut checked = 0;
    let mut sessions: Vec<(Value, Session)> = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        if case.get("explain").is_some() {
            continue;
        }
        let sql = case["sql"].as_str().unwrap();
        let conf = case["conf"].as_object().unwrap();
        if !sessions.iter().any(|(c, _)| c == &case["conf"]) {
            sessions.push((case["conf"].clone(), session(&fixture, conf)));
        }
        let s = &sessions.iter().find(|(c, _)| c == &case["conf"]).unwrap().1;
        let got = outcome(s, sql);
        let mut want = expected(case);
        // An error raised where no span is known has no context: compare its class.
        // Spark 4.2.0 also raises some with none (a DISTRIBUTE BY in a correlated subquery loses
        // its expressions' origin), where Krishiv may give the expression's span.
        let mut got = got;
        if got["error"].is_object() && got["error"]["fragment"].is_null() {
            for k in ["fragment", "start", "stop"] {
                want["error"][k] = Value::Null;
            }
        } else if want["error"].is_object() && want["error"]["fragment"].is_null() {
            for k in ["fragment", "start", "stop"] {
                got["error"][k] = Value::Null;
            }
        }
        let ansi_off = conf.get("spark.sql.ansi.enabled").and_then(Value::as_str) == Some("false");
        let differs = DIFFERENT
            .iter()
            .any(|(q, off, _)| *q == sql && *off == ansi_off);
        if differs {
            if got == want {
                report.push_str(&format!("now matches, remove from DIFFERENT: {sql}\n"));
            }
        } else if got != want {
            report.push_str(&format!("{sql} {conf:?}\n  want {want}\n  got  {got}\n"));
        }
        checked += 1;
    }
    assert!(checked >= 100, "{checked} cases");
    assert!(report.is_empty(), "{report}");
}
