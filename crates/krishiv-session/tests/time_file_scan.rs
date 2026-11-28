// SPDX-License-Identifier: Apache-2.0
//! TIME columns of CSV and JSON files (lane review of `1fbc1fdc9`, finding 1): a
//! `12:00:00.123456789` cell read into `time(p)` under each profile.
//!
//! Spark (lane probe `probe_time_files.py`, local[1]): 4.2.0 reads `time`, `time(3)` and `time(6)`
//! as `12:00:00.123456` and refuses `time(9)` (`UNSUPPORTED_TIME_PRECISION`); 4.3.0 reads every
//! precision as `12:00:00.123456789`. Krishiv gives the same values (`krishiv.io.text.timeNanos`, set
//! from the profile, keeps nine digits or six; `time_file_reads.rs` has the rest).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// A directory holding one `format` file with the cell.
fn file(format: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let (name, body) = match format {
        "csv" => ("t.csv", "12:00:00.123456789\n"),
        _ => ("t.json", "{\"t\":\"12:00:00.123456789\"}\n"),
    };
    let path = dir.path().join(name);
    std::fs::write(&path, body).expect("write the file");
    (dir, path)
}

fn read(profile: SparkProfile, format: &str, ty: &str) -> Result<Option<String>, KrishivError> {
    let (_dir, path) = file(format);
    let s = Session::builder().profile(profile).build()?;
    block_on(s.sql("SET spark.sql.timeType.enabled = true"))?;
    block_on(s.sql(&format!(
        "CREATE TEMPORARY VIEW v(t {ty}) USING {format} OPTIONS (path '{}')",
        path.display()
    )))?;
    let b = block_on(async {
        s.sql("SELECT CAST(t AS STRING) FROM v")
            .await?
            .collect()
            .await
    })?;
    let c = &b[0].columns()[0];
    Ok((!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned()))
}

#[test]
fn a_time_column_of_a_csv_or_json_file_reads_as_spark_does() {
    for profile in [SparkProfile::V4_2, SparkProfile::V4_3] {
        for format in ["csv", "json"] {
            for ty in ["time(9)", "time(6)", "time(3)"] {
                if profile == SparkProfile::V4_2 && ty == "time(9)" {
                    // Spark 4.2's precision bound comes first.
                    let e = read(profile, format, ty).unwrap_err();
                    assert_eq!(
                        e.error_class(),
                        ErrorClass::UnsupportedTimePrecision,
                        "{format} {ty}: {e}"
                    );
                    continue;
                }
                let want = if profile == SparkProfile::V4_2 {
                    "12:00:00.123456"
                } else {
                    "12:00:00.123456789"
                };
                let got = read(profile, format, ty).unwrap();
                assert_eq!(got.as_deref(), Some(want), "{profile:?} {format} {ty}");
            }
        }
    }
}
