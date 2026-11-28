// SPDX-License-Identifier: Apache-2.0
//! Zero-byte files beside data files, read as each Spark line reads them.
//!
//! Observed with Spark 3.5.9, 4.0.5, 4.1.4, 4.2.0, 4.2.1 and 4.3.0: no line scans a zero-byte file. Up to
//! 4.2 Parquet schema inference still opens one (the first file, or every file under `mergeSchema`) and
//! fails with `FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER`; 4.3 leaves it out of inference too.
#![cfg(test)]

use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// A directory with the files `a.<format>` and `b.<format>`: one row (`1`, `x`) where `good` says so,
/// zero bytes otherwise.
fn directory(format: &str, good: [bool; 2]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    for (name, good) in ["a", "b"].into_iter().zip(good) {
        let target = dir.path().join(format!("{name}.{format}"));
        if !good {
            std::fs::write(target, b"").expect("write the empty file");
        } else if format == "parquet" {
            write_parquet(&target);
        } else {
            let body = match format {
                "csv" => "1,x\n",
                _ => "{\"id\":1,\"s\":\"x\"}\n",
            };
            std::fs::write(target, body).expect("write the file");
        }
    }
    dir
}

/// One Parquet file holding the row, written by the engine.
fn write_parquet(target: &Path) {
    let out = tempfile::tempdir().expect("temp dir");
    let out = out.path().join("t");
    let s = Session::builder().build().expect("session");
    block_on(async {
        s.sql(&format!(
            "INSERT OVERWRITE DIRECTORY '{}' USING parquet SELECT CAST(1 AS BIGINT) AS id, 'x' AS s",
            out.display()
        ))
        .await?
        .collect()
        .await
    })
    .expect("write the file");
    let part = std::fs::read_dir(&out)
        .expect("list the output")
        .map(|e| e.expect("entry").path())
        .find(|p| p.extension().is_some_and(|e| e == "parquet"))
        .expect("a part file");
    std::fs::copy(part, target).expect("copy the file");
}

/// The `id` values of the directory read with `options`, or the error's class name.
fn ids(
    profile: SparkProfile,
    format: &str,
    dir: &Path,
    columns: &str,
    options: &str,
) -> Result<Vec<i64>, KrishivError> {
    let s = Session::builder().profile(profile).build()?;
    block_on(async {
        let from = if columns.is_empty() && options.is_empty() {
            // Queried directly: a temporary view needs at least one column.
            format!("{format}.`{}`", dir.display())
        } else {
            s.sql(&format!(
                "CREATE TEMPORARY VIEW v{columns} USING {format} OPTIONS (path '{}'{options})",
                dir.display()
            ))
            .await?
            .collect()
            .await?;
            "v".to_owned()
        };
        let batches = s
            .sql(&format!("SELECT CAST(count(*) AS BIGINT) FROM {from}"))
            .await?
            .collect()
            .await?;
        let n = batches[0].columns()[0].as_primitive::<Int64Type>().value(0);
        Ok(vec![n])
    })
}

fn outcome(r: Result<Vec<i64>, KrishivError>) -> String {
    match r {
        Ok(rows) => format!("{} rows", rows[0]),
        Err(e) => e.error_class().name().to_owned(),
    }
}

const FOOTER: &str = "FAILED_READ_FILE.CANNOT_READ_FILE_FOOTER";

#[test]
fn a_zero_byte_parquet_file_is_read_as_spark_reads_it() {
    let mut wrong = Vec::new();
    for profile in SparkProfile::ALL {
        let old = profile < SparkProfile::V4_3;
        let cases: [(&str, [bool; 2], &str, &str, &str); 8] = [
            ("good first", [true, false], "", "", "1 rows"),
            (
                "good first, schema",
                [true, false],
                "(id bigint, s string)",
                "",
                "1 rows",
            ),
            (
                "good first, merge",
                [true, false],
                "",
                ", mergeSchema 'true'",
                if old { FOOTER } else { "1 rows" },
            ),
            (
                "empty first",
                [false, true],
                "",
                "",
                if old { FOOTER } else { "1 rows" },
            ),
            (
                "empty first, schema",
                [false, true],
                "(id bigint, s string)",
                "",
                "1 rows",
            ),
            (
                "empty first, merge",
                [false, true],
                "",
                ", mergeSchema 'true'",
                if old { FOOTER } else { "1 rows" },
            ),
            (
                "only empty",
                [false, false],
                "",
                "",
                if old {
                    FOOTER
                } else {
                    "UNABLE_TO_INFER_SCHEMA"
                },
            ),
            (
                "only empty, schema",
                [false, false],
                "(id bigint, s string)",
                "",
                "0 rows",
            ),
        ];
        for (name, good, columns, options, want) in cases {
            let dir = directory("parquet", good);
            let got = outcome(ids(profile, "parquet", dir.path(), columns, options));
            if got != want {
                wrong.push(format!("{profile:?} {name}: {got}, Spark: {want}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_zero_byte_text_source_file_adds_no_row() {
    let mut wrong = Vec::new();
    for profile in SparkProfile::ALL {
        for format in ["csv", "json"] {
            for (name, good, want) in [
                ("good first", [true, false], "1 rows"),
                ("empty first", [false, true], "1 rows"),
                ("only empty", [false, false], "0 rows"),
            ] {
                let dir = directory(format, good);
                let got = outcome(ids(profile, format, dir.path(), "", ""));
                if got != want {
                    wrong.push(format!("{profile:?} {format} {name}: {got}, Spark: {want}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
