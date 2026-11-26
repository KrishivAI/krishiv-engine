// SPDX-License-Identifier: Apache-2.0
//! TIME(p <= 9), `LOCALTIME` and the nanosecond timestamp refusals (P3 R3)
//! against real Spark 4.2.0 and 4.3.0 runs: `tests/fixtures/spark-<v>/time-precision-goldens.json`,
//! written by `tests/scripts/gen_time_precision_goldens.py`. Each case runs on the session profile of
//! its release, in UTC. A case that runs is compared by its column names and by every value cast to
//! STRING on the engine (so TIME values keep their nanoseconds); a case that reads a clock only by
//! its `typed` columns (the `typeof(..)` values, which do not depend on the clock); both also by their
//! output types and nullability (`Session::analyze`). A case that fails is compared by its message's
//! first line, exactly, and, when Spark gave an error class, by the class, SQLSTATE and parameters.
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_in_result,
    reason = "a golden runner: a malformed fixture or a mismatch fails the test"
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;
use serde_json::Value;

/// Result rows, each value as text (NULL as `None`).
type Rows = Vec<Vec<Option<String>>>;

/// Cases not matched yet, by `(release, id, how)`: each must still differ, and Krishiv's own outcome
/// must be exactly `how` (`Mismatch::got`); a fixed gap, or one that differs in another way, fails
/// the test.
const KNOWN_GAPS: &[(&str, &str, &str)] = &[
    // The nanosecond timestamp types are Krishiv 0.3.1 (scope note P3-R3): with
    // `spark.sql.timestampNanosTypes.enabled` on, Spark 4.3 builds them; Krishiv refuses by name.
    (
        "4.3.0",
        "ts_ntz9_nanos_on",
        "[KRISHIV_UNSUPPORTED_FEATURE] The feature nanosecond-precision timestamp types (Krishiv 0.3.1) is not supported by Krishiv yet.",
    ),
    (
        "4.3.0",
        "ts_ltz7_nanos_on",
        "[KRISHIV_UNSUPPORTED_FEATURE] The feature nanosecond-precision timestamp types (Krishiv 0.3.1) is not supported by Krishiv yet.",
    ),
    // `date_format` refuses a TIME before the implicit cast Spark tries (4.2: the class differs).
    (
        "4.2.0",
        "date_format",
        "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve \"date_format(TIME '12:00:00.123456789', HH:mm:ss.SSSSSSSSS)\" due to data type mismatch: The first parameter requires the (\"STRING\" or \"DATE\" or \"TIMESTAMP\" or \"TIMESTAMP_NTZ\") type, however \"TIME '12:00:00.123456'\" has the type \"TIME(6)\".",
    ),
    (
        "4.3.0",
        "date_format",
        "[KRISHIV_UNSUPPORTED_FEATURE] The feature date_format of a TIME value (Spark 4.3) is not supported by Krishiv yet.",
    ),
    // A bare TIMESTAMP does not follow `spark.sql.timestampType` yet.
    (
        "4.3.0",
        "ts6_ntz_default",
        "[KRISHIV_UNSUPPORTED_FEATURE] The feature TIMESTAMP(6) with spark.sql.timestampType=TIMESTAMP_NTZ is not supported by Krishiv yet.",
    ),
    // Spark 4.2 adds `: missing ')'` to the syntax error for some shapes; the hint is not reproduced.
    (
        "4.2.0",
        "ts3_with_local",
        "[PARSE_SYNTAX_ERROR] Syntax error at or near 'WITH'.",
    ),
    (
        "4.2.0",
        "ts9_without",
        "[PARSE_SYNTAX_ERROR] Syntax error at or near 'WITHOUT'.",
    ),
    // SQL-41: `instr` and `locate` name their collated literal as Spark since #986; the `=` of a column
    // given a collation still loses its `collate` in the name. The case stays in the fixture (the
    // generator records every case of the run), so it is pinned here until that is fixed.
    (
        "4.2.0",
        "eq_collated",
        "names [\"(collate(name, UTF8_LCASE) = APPLE)\"]",
    ),
    (
        "4.3.0",
        "eq_collated",
        "names [\"(collate(name, UTF8_LCASE) = APPLE)\"]",
    ),
    // A foldable precision Krishiv's folder cannot evaluate: Spark evaluates it (6), Krishiv refuses
    // it by name.
    (
        "4.3.0",
        "current_timestamp_dayofweek",
        "[KRISHIV_UNSUPPORTED_FEATURE] The feature `current_timestamp(dayofweek(DATE '1970-01-02'))` with a precision Krishiv cannot fold is not supported by Krishiv yet.",
    ),
    // `instr` and `locate` of a collated column give a NOT NULL result in Spark; Krishiv's is nullable.
    ("4.2.0", "instr_collated", "schema [(\"int\", true)]"),
    (
        "4.2.0",
        "instr_literal_collated",
        "schema [(\"int\", true)]",
    ),
    ("4.2.0", "locate_collated", "schema [(\"int\", true)]"),
    ("4.3.0", "instr_collated", "schema [(\"int\", true)]"),
    (
        "4.3.0",
        "instr_literal_collated",
        "schema [(\"int\", true)]",
    ),
    ("4.3.0", "locate_collated", "schema [(\"int\", true)]"),
    // Spark 4.2 parses a precision past `Int` with `Integer.parseInt` and fails outside its error
    // classes (`NumberFormatException: For input string`); Krishiv raises its own class.
    (
        "4.2.0",
        "time_huge",
        "[UNSUPPORTED_TIME_PRECISION] The seconds precision 99999999999 of the TIME data type is out of the supported range [0, 6].",
    ),
    (
        "4.2.0",
        "ts_ntz_huge",
        "[UNSUPPORTED_DATATYPE] Unsupported data type \"TIMESTAMP_NTZ(99999999999)\".",
    ),
];

fn fixture(release: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(format!("spark-{release}/time-precision-goldens.json"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn session(profile: SparkProfile, conf: &serde_json::Map<String, Value>) -> Session {
    let s = Session::builder()
        .profile(profile)
        .build()
        .expect("session");
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).expect("zone");
    for (k, v) in conf {
        let v = v.as_str().expect("conf value");
        block_on(s.sql(&format!("SET {k} = {v}"))).expect("conf");
    }
    s
}

/// The column names, and the values cast to STRING (rows of `Option<String>`).
fn run(s: &Session, sql: &str) -> Result<(Vec<String>, Rows), KrishivError> {
    let df = block_on(s.sql(sql))?;
    let batches = block_on(df.collect())?;
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => return Ok((vec![], vec![])),
    };
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let aliases: Vec<String> = (0..names.len()).map(|i| format!("c{i}")).collect();
    let casts: Vec<String> = aliases
        .iter()
        .map(|a| format!("CAST({a} AS STRING)"))
        .collect();
    let wrapped = format!(
        "SELECT {} FROM ({sql}) AS t({})",
        casts.join(", "),
        aliases.join(", ")
    );
    let batches = block_on(block_on(s.sql(&wrapped))?.collect())?;
    let mut rows = vec![];
    for b in &batches {
        for r in 0..b.num_rows() {
            rows.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        let c = arrow_cast::cast(c, &arrow_schema::DataType::Utf8).expect("utf8");
                        let c = c.as_string::<i32>();
                        (!c.is_null(r)).then(|| c.value(r).to_owned())
                    })
                    .collect(),
            );
        }
    }
    Ok((names, rows))
}

/// How a case differs: `why` compares with Spark, `got` is Krishiv's own outcome alone (its error as
/// `[CLASS] message`, or the names, column or values it gave), which `KNOWN_GAPS` pins exactly.
struct Mismatch {
    why: String,
    got: String,
}

fn mismatch(why: String, got: String) -> Option<Mismatch> {
    Some(Mismatch { why, got })
}

fn shown(e: &KrishivError) -> String {
    format!("[{}] {}", e.error_class().name(), e.message())
}

/// How case `c` differs on `profile`, or `None` when it matches.
fn check(profile: SparkProfile, c: &Value) -> Option<Mismatch> {
    let sql = c["sql"].as_str().expect("sql");
    let clock = c["clock"].as_bool().unwrap_or(false);
    let conf = c["conf"].as_object().expect("conf");
    let s = session(profile, conf);
    let got = run(&s, sql);
    if c.get("error").is_some() || c.get("message").is_some() {
        let err = match got {
            Ok(v) => {
                return mismatch(
                    format!("ran ({v:?}), Spark fails: {}", c["message"]),
                    format!("ran {v:?}"),
                );
            }
            Err(e) => e,
        };
        let want = c["message"].as_str().unwrap_or_default();
        let got = shown(&err);
        // The message's first line without a SQLSTATE suffix, as the generator records Spark's.
        let line = got.lines().next().unwrap_or_default();
        let line = line.split(" SQLSTATE:").next().unwrap_or_default().trim();
        if line != want {
            return mismatch(format!("error `{got}`, Spark `{want}`"), got);
        }
        if let Some(class) = c["error"].as_str() {
            if err.error_class().name() != class {
                return mismatch(format!("class of `{got}`, Spark {class}"), got);
            }
            let want_state = c["sql_state"].as_str();
            if err.sql_state() != want_state {
                return mismatch(
                    format!("SQLSTATE {:?}, Spark {want_state:?}", err.sql_state()),
                    got,
                );
            }
            let want_params: BTreeMap<String, String> = c["parameters"]
                .as_object()
                .expect("parameters")
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().expect("parameter").to_owned()))
                .collect();
            let params: BTreeMap<String, String> = err
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect();
            if params != want_params {
                return mismatch(
                    format!("parameters {params:?}, Spark {want_params:?}"),
                    format!("{got} parameters {params:?}"),
                );
            }
        }
        return None;
    }
    let (names, rows) = match got {
        Ok(v) => v,
        Err(e) => {
            let got = shown(&e);
            return mismatch(format!("fails `{got}`"), got);
        }
    };
    let want_names: Vec<String> = c["names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().expect("name").to_owned())
        .collect();
    if names != want_names {
        return mismatch(
            format!("names {names:?}, Spark {want_names:?}"),
            format!("names {names:?}"),
        );
    }
    if clock {
        for t in c["typed"].as_array().into_iter().flatten() {
            let i = usize::try_from(t[0].as_u64().expect("typed index")).expect("index");
            let want = t[1].as_str().map(str::to_owned);
            let got = rows.first().and_then(|r| r.get(i)).cloned().flatten();
            if got != want {
                return mismatch(
                    format!("column {i}: {got:?}, Spark {want:?}"),
                    format!("column {i}: {got:?}"),
                );
            }
        }
    } else {
        let want: Rows = c["values"]
            .as_array()
            .expect("values")
            .iter()
            .map(|r| {
                r.as_array()
                    .expect("row")
                    .iter()
                    .map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .collect();
        if rows != want {
            return mismatch(
                format!("values {rows:?}, Spark {want:?}"),
                format!("values {rows:?}"),
            );
        }
    }
    // The statement's own output types and nullability, before the STRING casts of `run`.
    // Compared last, so a known schema gap still has its values checked.
    let want_schema: Vec<(String, bool)> = c["schema"]
        .as_array()
        .expect("schema")
        .iter()
        .map(|f| {
            (
                f[0].as_str().expect("type").to_owned(),
                f[1].as_bool().expect("nullable"),
            )
        })
        .collect();
    let schema: Vec<(String, bool)> = match block_on(s.analyze(sql)) {
        Ok(a) => a
            .output_schema
            .iter()
            .map(|f| (f.data_type.simple_string(), f.nullable))
            .collect(),
        Err(e) => {
            let got = shown(&e);
            return mismatch(format!("analysis fails `{got}`"), got);
        }
    };
    if schema != want_schema {
        return mismatch(
            format!("schema {schema:?}, Spark {want_schema:?}"),
            format!("schema {schema:?}"),
        );
    }
    None
}

fn release(release: &str, profile: SparkProfile) {
    let doc = fixture(release);
    assert_eq!(
        doc["format_version"], 2,
        "Spark {release}: fixture format_version"
    );
    let cases = doc["cases"].as_array().expect("cases");
    assert!(
        !cases.is_empty(),
        "Spark {release}: the fixture has no cases"
    );
    let mut failures = vec![];
    for c in cases {
        let id = c["id"].as_str().expect("id");
        let gap = KNOWN_GAPS
            .iter()
            .find(|(r, g, _)| *r == release && *g == id)
            .map(|(_, _, how)| *how);
        match (check(profile, c), gap) {
            (Some(m), None) => failures.push(format!("{id}: {}", m.why)),
            (None, Some(_)) => failures.push(format!("{id}: listed in KNOWN_GAPS but matches now")),
            (Some(m), Some(how)) if m.got != how => {
                failures.push(format!(
                    "{id}: a known gap that differs in another way: {} (Krishiv: {:?})",
                    m.why, m.got
                ));
            }
            _ => {}
        }
    }
    assert!(
        failures.is_empty(),
        "Spark {release}:\n{}",
        failures.join("\n")
    );
}

#[test]
fn spark_4_3_time_precision_goldens() {
    release("4.3.0", SparkProfile::V4_3);
}

#[test]
fn spark_4_2_time_precision_goldens() {
    release("4.2.0", SparkProfile::V4_2);
}
