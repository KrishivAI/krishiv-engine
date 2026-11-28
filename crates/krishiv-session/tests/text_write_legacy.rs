// SPDX-License-Identifier: Apache-2.0
//! CSV and JSON writes of a `TIMESTAMP`, a `DATE` and a `TIMESTAMP_NTZ` under
//! the default parser policy and `spark.sql.legacy.timeParserPolicy = LEGACY`,
//! in UTC and in `America/Los_Angeles`, against the Spark 4.2.1 and 4.3.0 runs
//! of `tests/scripts/probe_time_text_profiles.py` (the `write` section of
//! `tests/fixtures/spark-<v>/time-text-profiles.json`). LEGACY writes dates
//! and timestamps with `FastDateFormat` (`JSONOptions`/`CSVOptions`
//! `timestampFormatInWrite` defaults to `yyyy-MM-dd'T'HH:mm:ss.SSSXXX`): the
//! hybrid calendar (`1582-10-14` is written `1582-10-15`, weekdays before the
//! cutover are Julian), `java.util.TimeZone` offsets (no LMT) and the legacy
//! letters (`u` is the day of the week). `TIMESTAMP_NTZ` keeps the `java.time`
//! pattern under either policy.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// The lines of every `part-` file under `dir`, in file-name order.
fn written_lines(dir: &std::path::Path) -> Vec<String> {
    let mut parts: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("part-"))
        })
        .collect();
    parts.sort();
    parts
        .iter()
        .flat_map(|p| {
            std::fs::read_to_string(p)
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Whether Spark's recorded refusal `class` is a Java (or Py4J) exception
/// rather than an error class. The cases are a `dateFormat` with the letter
/// `b`: `java.lang.IllegalArgumentException: Unknown pattern letter: b`
/// (`java.time`), `Illegal pattern component: b` under LEGACY
/// (`FastDateFormat`), raised by the write task (`Py4JJavaError`) or, for the
/// expressions, as `IllegalArgumentException` (both seen in the 4.2.1 and 4.3.0
/// runs). Krishiv refuses those with an error class, so any refusal matches.
fn is_java_exception(class: &str) -> bool {
    class.bytes().any(|b| b.is_ascii_lowercase())
}

fn lit(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

#[test]
fn csv_and_json_writes_follow_the_parser_policy_as_spark_runs() {
    let dir = tempfile::tempdir().unwrap();
    let mut checked = 0;
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let sections = run["write"].as_object().unwrap();
        assert_eq!(sections.len(), 4, "{version}: two policies in two zones");
        for (setting, cases) in sections {
            let s = Session::builder().profile(profile).build().unwrap();
            let (legacy, zone) = match setting.as_str() {
                "default" => (false, "UTC"),
                "legacy" => (true, "UTC"),
                "default la" => (false, "America/Los_Angeles"),
                "legacy la" => (true, "America/Los_Angeles"),
                other => panic!("unknown setting {other}"),
            };
            block_on(s.sql(&format!("SET spark.sql.session.timeZone = {zone}"))).unwrap();
            if legacy {
                block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
            }
            for (i, case) in cases.as_array().unwrap().iter().enumerate() {
                let (ts, d) = (
                    case["timestamp"].as_str().unwrap(),
                    case["date"].as_str().unwrap(),
                );
                let opts = case["options"].as_object().unwrap();
                for (format, want) in case["lines"].as_object().unwrap() {
                    let at = format!("{format} {version} {setting} {ts} {d} {opts:?}");
                    let df = block_on(s.sql(&format!(
                        "SELECT TIMESTAMP {} AS ts, DATE {} AS d, TIMESTAMP_NTZ {} AS ntz",
                        lit(ts),
                        lit(d),
                        lit(ts)
                    )))
                    .unwrap_or_else(|e| panic!("{at}: {e}"));
                    let out = dir.path().join(format!(
                        "{version}-{}-{i}.{format}",
                        setting.replace(' ', "-")
                    ));
                    let mut w = df.write().format(format);
                    for (k, v) in opts {
                        w = w.option(k, v.as_str().unwrap());
                    }
                    let got = block_on(w.save(out.to_str().unwrap()));
                    let want: Vec<&str> = want
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap())
                        .collect();
                    match (want.first().and_then(|w| w.strip_prefix("ERROR ")), got) {
                        (Some(class), Err(e)) => {
                            if !is_java_exception(class) {
                                let name = e.error_class().name();
                                assert!(
                                    class == name || class.starts_with(&format!("{name}.")),
                                    "{at}: {class} vs {e}"
                                );
                            }
                        }
                        (Some(class), Ok(())) => panic!("{at}: wrote, Spark refused ({class})"),
                        (None, Err(e)) => panic!("{at}: {e}"),
                        (None, Ok(())) => assert_eq!(written_lines(&out), want, "{at}"),
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 400, "{checked} writes checked");
}

/// The `write_expr` section: the same rows through `to_json`, `to_csv` and
/// `to_xml` with each write option set as the options map. The expressions
/// take the file writers' default `timestampFormat`
/// (`FileSourceOptions.commonTimestampFormat`): `[.SSS][XXXXX]`, so
/// `1880-01-01` in Los Angeles ends `-07:52:58`, and `[XXX]` (`-07:52`) with
/// `spark.sql.files.supportSecondOffsetFormat = false`.
#[test]
fn text_expressions_write_as_the_file_writers_do() {
    let mut checked = 0;
    let mut misses: Vec<String> = Vec::new();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let sections = run["write_expr"].as_object().unwrap();
        assert_eq!(
            sections.len(),
            5,
            "{version}: the write settings and minute offsets"
        );
        for (setting, cases) in sections {
            let s = Session::builder().profile(profile).build().unwrap();
            let (legacy, zone, seconds) = match setting.as_str() {
                "default" => (false, "UTC", true),
                "legacy" => (true, "UTC", true),
                "default la" => (false, "America/Los_Angeles", true),
                "legacy la" => (true, "America/Los_Angeles", true),
                "default la minutes" => (false, "America/Los_Angeles", false),
                other => panic!("unknown setting {other}"),
            };
            block_on(s.sql(&format!("SET spark.sql.session.timeZone = {zone}"))).unwrap();
            if legacy {
                block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
            }
            if !seconds {
                block_on(s.sql("SET spark.sql.files.supportSecondOffsetFormat = false")).unwrap();
            }
            for case in cases.as_array().unwrap() {
                let (ts, d) = (
                    case["timestamp"].as_str().unwrap(),
                    case["date"].as_str().unwrap(),
                );
                let opts = case["options"].as_object().unwrap();
                let map = if opts.is_empty() {
                    String::new()
                } else {
                    let pairs: Vec<String> = opts
                        .iter()
                        .flat_map(|(k, v)| [lit(k), lit(v.as_str().unwrap())])
                        .collect();
                    format!(", map({})", pairs.join(", "))
                };
                let row = format!(
                    "named_struct('ts', TIMESTAMP {}, 'd', DATE {}, 'ntz', TIMESTAMP_NTZ {})",
                    lit(ts),
                    lit(d),
                    lit(ts)
                );
                for (func, want) in case["exprs"].as_object().unwrap() {
                    let sql = format!("SELECT {func}({row}{map}) AS s");
                    let at = format!("{version} {setting}: {sql}");
                    let got = block_on(async { s.sql(&sql).await?.collect().await }).map(|b| {
                        let b = b.iter().find(|b| b.num_rows() > 0).unwrap();
                        let c = b.column(0);
                        (!c.is_null(0)).then(|| {
                            arrow_array::cast::AsArray::as_string_view(c.as_ref())
                                .value(0)
                                .to_owned()
                        })
                    });
                    match (want.get("error").and_then(|e| e.as_str()), got) {
                        (Some(class), Err(e)) => {
                            let class = class.strip_prefix("ERROR ").unwrap_or(class);
                            if !is_java_exception(class) {
                                let name = e.error_class().name();
                                if !(class == name || class.starts_with(&format!("{name}."))) {
                                    misses.push(format!("{at}: {class} vs {e}"));
                                }
                            }
                        }
                        (Some(class), Ok(v)) => {
                            misses.push(format!("{at}: {v:?}, Spark refused ({class})"));
                        }
                        (None, Err(e)) => misses.push(format!("{at}: {e}")),
                        (None, Ok(v)) => {
                            if v.as_deref() != want["value"].as_str() {
                                misses.push(format!("{at}: {v:?}, Spark {}", want["value"]));
                            }
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 900, "{checked} expressions checked");
    assert!(
        misses.is_empty(),
        "{} of {checked}:\n{}",
        misses.len(),
        misses.join("\n")
    );
}
