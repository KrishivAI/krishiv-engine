// SPDX-License-Identifier: Apache-2.0
//! TIME columns of CSV and JSON files (lane review of `1fbc1fdc9`, finding 1): a
//! `12:00:00.123456789` cell read into `time(p)` under each profile.
//!
//! Spark (lane probe `probe_time_files.py`, local[1]): 4.2.0 reads `time`, `time(3)` and `time(6)`
//! as `12:00:00.123456` and refuses `time(9)` (`UNSUPPORTED_TIME_PRECISION`); 4.3.0 reads every
//! precision as `12:00:00.123456789`. Krishiv's file readers have no TIME column yet, so every case
//! is `KRISHIV_UNSUPPORTED_FEATURE` (docs/concepts/known-differences.md), never a wrong value.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn read(profile: SparkProfile, format: &str, ty: &str) -> Result<Option<String>, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let (name, body) = match format {
        "csv" => ("t.csv", "12:00:00.123456789\n"),
        _ => ("t.json", "{\"t\":\"12:00:00.123456789\"}\n"),
    };
    std::fs::write(dir.path().join(name), body).map_err(|e| e.to_string())?;
    let s = Session::builder()
        .profile(profile)
        .build()
        .map_err(|e| e.to_string())?;
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).map_err(|e| e.to_string())?;
    let path = dir.path().join(name);
    block_on(s.sql(&format!(
        "CREATE TEMPORARY VIEW v(t {ty}) USING {format} OPTIONS (path '{}')",
        path.display()
    )))
    .map_err(|e| e.to_string())?;
    let b = block_on(async {
        s.sql("SELECT CAST(t AS STRING) FROM v")
            .await?
            .collect()
            .await
    })
    .map_err(|e| e.to_string())?;
    let c = &b[0].columns()[0];
    Ok((!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned()))
}

#[test]
fn a_time_column_of_a_csv_or_json_file_is_refused_not_misread() {
    for profile in [SparkProfile::V4_2, SparkProfile::V4_3] {
        for format in ["csv", "json"] {
            for ty in ["time(9)", "time(6)", "time(3)"] {
                if profile == SparkProfile::V4_2 && ty == "time(9)" {
                    // Spark 4.2's precision bound comes first.
                    let e = read(profile, format, ty).unwrap_err();
                    assert!(
                        e.contains("UNSUPPORTED_TIME_PRECISION"),
                        "{format} {ty}: {e}"
                    );
                    continue;
                }
                let e = read(profile, format, ty).unwrap_err();
                assert!(
                    e.contains("KRISHIV_UNSUPPORTED_FEATURE"),
                    "{profile:?} {format} {ty}: {e}"
                );
            }
        }
    }
}
