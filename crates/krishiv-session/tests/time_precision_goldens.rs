// SPDX-License-Identifier: Apache-2.0
//! TIME(p <= 9), `LOCALTIME`, the nanosecond timestamp refusals and the collated literal names (P3 R3)
//! against real Spark 4.2.0 and 4.3.0 runs: `tests/fixtures/spark-<v>/time-precision-goldens.json`,
//! written by `tests/scripts/gen_time_precision_goldens.py`. Each case runs on the session profile of
//! its release, in UTC. A case that runs is compared by its column names and, unless it reads a clock,
//! by every value cast to STRING on the engine (so TIME values keep their nanoseconds); a case that
//! fails by its error condition and message.
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

/// Cases not matched yet, by `(release, id)`: each must still differ (a fixed gap fails the test).
const KNOWN_GAPS: &[(&str, &str)] = &[
    // Spark gates TIME per expression (a cast to it, `TimeExpression`s); Krishiv's gate is plan-wide, so a
    // literal or `localtime` with `spark.sql.timeType.enabled` off is refused.
    ("4.2.0", "lit_types"),
    ("4.2.0", "lit_name9"),
    ("4.2.0", "lit_trailing_zeros"),
    ("4.2.0", "lit_to_string"),
    ("4.3.0", "lit_types"),
    ("4.3.0", "lit_name9"),
    ("4.3.0", "lit_trailing_zeros"),
    ("4.3.0", "lit_to_string"),
    ("4.3.0", "localtime_key_off"),
    // Errors render the Spark 4.2.1 message template; release-keyed rendering is not built yet ([0, 9] in 4.3).
    ("4.3.0", "time10"),
    ("4.3.0", "time10_key_off"),
    ("4.3.0", "time_huge"),
    // `min`/`max` have no `Time64` accumulator yet.
    ("4.2.0", "max_distinct"),
    ("4.3.0", "max_distinct"),
    // `to_time` with a pattern of nine `S` keeps six digits.
    ("4.2.0", "to_time"),
    ("4.3.0", "to_time"),
    // `date_format` refuses a TIME before the implicit cast Spark tries (the class differs).
    ("4.2.0", "date_format"),
    ("4.3.0", "date_format"),
    // No cast from TIME to TIMESTAMP_NTZ yet.
    ("4.3.0", "to_timestamp_ntz"),
    // A bare TIMESTAMP does not follow `spark.sql.timestampType` yet.
    ("4.3.0", "ts6_ntz_default"),
    // Spark 4.2 adds `: missing ')'` to the syntax error for some shapes; the hint is not reproduced.
    ("4.2.0", "ts3_with_local"),
    ("4.2.0", "ts9_without"),
    // SQL-41: a literal given a collation by the implicit cast loses its `collate` in these rewrites' names.
    ("4.2.0", "instr_collated"),
    ("4.2.0", "locate_collated"),
    ("4.2.0", "eq_collated"),
    ("4.3.0", "instr_collated"),
    ("4.3.0", "locate_collated"),
    ("4.3.0", "eq_collated"),
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
fn run(s: &Session, sql: &str, clock: bool) -> Result<(Vec<String>, Rows), KrishivError> {
    let df = block_on(s.sql(sql))?;
    let batches = block_on(df.collect())?;
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => return Ok((vec![], vec![])),
    };
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    if clock {
        return Ok((names, vec![]));
    }
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
    let got = run(&s, sql, clock);
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
    if !clock {
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
        let gap = KNOWN_GAPS.contains(&(release, id));
        match (check(profile, c), gap) {
            (Some(why), false) => failures.push(format!("{id}: {why}")),
            (None, true) => failures.push(format!("{id}: listed in KNOWN_GAPS but matches now")),
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
