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
                            // Spark raises a plain Java exception (no error
                            // class) for a bad `dateFormat`; any refusal matches it.
                            if !class.starts_with("Py4J") {
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
