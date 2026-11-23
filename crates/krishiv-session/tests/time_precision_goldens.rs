// SPDX-License-Identifier: Apache-2.0
//! TIME(p <= 9), `LOCALTIME` and the nanosecond timestamp refusals (P3 R3)
//! against real Spark 4.2.0 and 4.3.0 runs: `tests/fixtures/spark-<v>/time-precision-goldens.json`,
//! written by `tests/scripts/gen_time_precision_goldens.py`. Each case runs on the session profile of
//! its release, in UTC. A case that runs is compared by its column names and by every value cast to
//! STRING on the engine (so TIME values keep their nanoseconds); a case that reads a clock only by
//! its `typed` columns (the `typeof(..)` values, which do not depend on the clock). A case that fails
//! is compared by its error condition and message.
#![forbid(unsafe_code)]
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_in_result,
    reason = "a golden runner: a malformed fixture or a mismatch fails the test"
)]

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

/// Cases not matched yet, by `(release, id, how)`: each must still differ, in the way `how` (a part
/// of the mismatch report) says; a fixed gap, or one that differs in another way, fails the test.
const KNOWN_GAPS: &[(&str, &str, &str)] = &[
    // `min`/`max` have no `Time64` accumulator yet.
    ("4.2.0", "max_distinct", "Max of Time64(ns)"),
    ("4.3.0", "max_distinct", "Max of Time64(ns)"),
    ("4.2.0", "max_only", "Max of Time64(ns)"),
    ("4.3.0", "max_only", "Max of Time64(ns)"),
    // The nanosecond timestamp types are Krishiv 0.3.1 (scope note P3-R3): with
    // `spark.sql.timestampNanosTypes.enabled` on, Spark 4.3 builds them; Krishiv refuses by name.
    (
        "4.3.0",
        "ts_ntz9_nanos_on",
        "nanosecond-precision timestamp types",
    ),
    (
        "4.3.0",
        "ts_ltz7_nanos_on",
        "nanosecond-precision timestamp types",
    ),
    // `date_format` refuses a TIME before the implicit cast Spark tries (4.2: the class differs).
    (
        "4.2.0",
        "date_format",
        "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE",
    ),
    ("4.3.0", "date_format", "date_format of a TIME value"),
    // No cast from TIME to TIMESTAMP_NTZ yet.
    (
        "4.3.0",
        "to_timestamp_ntz",
        "a cast from TIME to TIMESTAMP_NTZ",
    ),
    // A bare TIMESTAMP does not follow `spark.sql.timestampType` yet.
    (
        "4.3.0",
        "ts6_ntz_default",
        "TIMESTAMP(6) with spark.sql.timestampType",
    ),
    // Spark 4.2 adds `: missing ')'` to the syntax error for some shapes; the hint is not reproduced.
    ("4.2.0", "ts3_with_local", "missing ')'"),
    ("4.2.0", "ts9_without", "missing ')'"),
    // SQL-41: a literal given a collation by the implicit cast loses its `collate` in these rewrites' names.
    // The fix is row SQL-41's, not R3's; the cases stay in the fixture (the generator records every
    // case of the run), so they are pinned here until SQL-41 lands and removes these six lines.
    ("4.2.0", "instr_collated", "names ["),
    ("4.2.0", "locate_collated", "names ["),
    ("4.2.0", "eq_collated", "names ["),
    ("4.3.0", "instr_collated", "names ["),
    ("4.3.0", "locate_collated", "names ["),
    ("4.3.0", "eq_collated", "names ["),
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

/// Why case `c` differs on `profile`, or `None` when it matches.
fn check(profile: SparkProfile, c: &Value) -> Option<String> {
    let sql = c["sql"].as_str().expect("sql");
    let clock = c["clock"].as_bool().unwrap_or(false);
    let conf = c["conf"].as_object().expect("conf");
    let s = session(profile, conf);
    let got = run(&s, sql);
    if c.get("error").is_some() || c.get("message").is_some() {
        let err = match got {
            Ok(v) => return Some(format!("ran ({v:?}), Spark fails: {}", c["message"])),
            Err(e) => e,
        };
        let Some(class) = c["error"].as_str() else {
            // Spark failed outside its error classes (a JVM exception): any error matches.
            return None;
        };
        let want = c["message"].as_str().unwrap_or_default();
        let got = format!("[{}] {}", err.error_class().name(), err.message());
        if err.error_class().name() != class || !got.starts_with(want.trim_end_matches('.')) {
            return Some(format!("error `{got}`, Spark `{want}`"));
        }
        return None;
    }
    let (names, rows) = match got {
        Ok(v) => v,
        Err(e) => {
            return Some(format!(
                "fails `[{}] {}`",
                e.error_class().name(),
                e.message()
            ));
        }
    };
    let want_names: Vec<String> = c["names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().expect("name").to_owned())
        .collect();
    if names != want_names {
        return Some(format!("names {names:?}, Spark {want_names:?}"));
    }
    if clock {
        for t in c["typed"].as_array().into_iter().flatten() {
            let i = usize::try_from(t[0].as_u64().expect("typed index")).expect("index");
            let want = t[1].as_str().map(str::to_owned);
            let got = rows.first().and_then(|r| r.get(i)).cloned().flatten();
            if got != want {
                return Some(format!("column {i}: {got:?}, Spark {want:?}"));
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
            return Some(format!("values {rows:?}, Spark {want:?}"));
        }
    }
    None
}

fn release(release: &str, profile: SparkProfile) {
    let doc = fixture(release);
    let mut failures = vec![];
    for c in doc["cases"].as_array().expect("cases") {
        let id = c["id"].as_str().expect("id");
        let gap = KNOWN_GAPS
            .iter()
            .find(|(r, g, _)| *r == release && *g == id)
            .map(|(_, _, how)| *how);
        match (check(profile, c), gap) {
            (Some(why), None) => failures.push(format!("{id}: {why}")),
            (None, Some(_)) => failures.push(format!("{id}: listed in KNOWN_GAPS but matches now")),
            (Some(why), Some(how)) if !why.contains(how) => {
                failures.push(format!(
                    "{id}: a known gap that differs in another way: {why}"
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
