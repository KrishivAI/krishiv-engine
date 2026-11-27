// SPDX-License-Identifier: Apache-2.0
//! Nanosecond timestamps in Parquet (Spark 4.3 `TIMESTAMP_NTZ(p)` / `TIMESTAMP_LTZ(p)`, `p` 7..=9), against
//! Spark 4.3.0 and 4.2.1 runs over `tests/fixtures/parquet/nanos` (`tests/scripts/gen_parquet_nanos_fixtures.py`):
//! a write is `INT64 TIMESTAMP(NANOS)` with the precision in the footer's Spark schema; an inferred schema
//! with such a column needs `spark.sql.timestampNanosTypes.enabled` (else `FEATURE_NOT_ENABLED` for a file
//! whose Spark schema types one, `PARQUET_TYPE_ILLEGAL` for any other), unless
//! `spark.sql.legacy.parquet.nanosAsLong` reads a foreign file's as `bigint`.
#![expect(clippy::unwrap_used, clippy::expect_used, reason = "a test")]
use std::path::PathBuf;

use krishiv_session::Session;

fn fixture(name: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/parquet/nanos")
        .join(name)
        .display()
        .to_string()
}

fn session(nanos: bool) -> Session {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    s.conf()
        .set(
            "spark.sql.timestampNanosTypes.enabled",
            if nanos { "true" } else { "false" },
        )
        .unwrap();
    s
}

/// The rows of `q` (cells joined by `|`, rows by ` ; `), or `ERR <message>`.
fn run(s: &Session, q: &str) -> String {
    match futures::executor::block_on(async { s.sql(q).await?.collect().await }) {
        Err(e) => format!("ERR {e}"),
        Ok(batches) => {
            let mut out = Vec::new();
            for b in &batches {
                for r in 0..b.num_rows() {
                    let cells: Vec<String> = b
                        .columns()
                        .iter()
                        .map(|c| arrow_cast::display::array_value_to_string(c, r).expect("display"))
                        .collect();
                    out.push(cells.join("|"));
                }
            }
            out.join(" ; ")
        }
    }
}

const NOT_ENABLED: &str = "ERR [FEATURE_NOT_ENABLED] The feature Nanosecond-precision timestamp types is not \
                           enabled. Consider setting the config spark.sql.timestampNanosTypes.enabled to true to \
                           enable this capability. SQLSTATE: 56038";
const V: &str = "2020-01-01 00:00:00.123456789";

#[test]
fn reads_nanosecond_timestamps_with_their_precision() {
    let s = session(true);
    let pa = fixture("pyarrow.parquet");
    assert_eq!(
        run(
            &s,
            &format!(
                "SELECT typeof(utc), typeof(local), CAST(utc AS STRING), CAST(local AS STRING) FROM parquet.`{pa}`"
            )
        ),
        format!("timestamp_ltz(9)|timestamp_ntz(9)|{V}|{V}")
    );
    let flat = fixture("spark-flat.parquet");
    assert_eq!(
        run(
            &s,
            &format!(
                "SELECT typeof(n9), typeof(l9), typeof(n7), CAST(n9 AS STRING), CAST(n7 AS STRING) FROM parquet.`{flat}`"
            )
        ),
        format!(
            "timestamp_ntz(9)|timestamp_ltz(9)|timestamp_ntz(7)|{V}|2020-01-01 00:00:00.1234567"
        )
    );
    let nested = fixture("spark-nested.parquet");
    assert_eq!(
        run(
            &s,
            &format!("SELECT typeof(a), typeof(s), CAST(s.x AS STRING) FROM parquet.`{nested}`")
        ),
        "array<timestamp_ntz(9)>|struct<x:timestamp_ltz(7)>|2020-01-01 00:00:00.1234567"
    );
}

#[test]
fn an_inferred_nanosecond_column_needs_the_flag() {
    let s = session(false);
    assert_eq!(
        run(
            &s,
            &format!("SELECT * FROM parquet.`{}`", fixture("pyarrow.parquet"))
        ),
        "ERR [PARQUET_TYPE_ILLEGAL] Illegal Parquet type: INT64 (TIMESTAMP(NANOS,true)). SQLSTATE: 42846"
    );
    for f in ["spark-flat.parquet", "spark-nested.parquet"] {
        assert_eq!(
            run(&s, &format!("SELECT * FROM parquet.`{}`", fixture(f))),
            NOT_ENABLED,
            "{f}"
        );
    }
}

#[test]
fn nanos_as_long_reads_a_foreign_files_nanoseconds_as_bigint() {
    for nanos in [false, true] {
        let s = session(nanos);
        s.conf()
            .set("spark.sql.legacy.parquet.nanosAsLong", "true")
            .unwrap();
        assert_eq!(
            run(
                &s,
                &format!(
                    "SELECT typeof(utc), utc, local FROM parquet.`{}`",
                    fixture("pyarrow.parquet")
                )
            ),
            "bigint|1577836800123456789|1577836800123456789",
            "nanos {nanos}"
        );
        // A Spark 4.3 file's own schema wins over the setting.
        let flat = fixture("spark-flat.parquet");
        let expected = if nanos {
            "timestamp_ntz(7)".to_owned()
        } else {
            NOT_ENABLED.to_owned()
        };
        assert_eq!(
            run(&s, &format!("SELECT typeof(n7) FROM parquet.`{flat}`")),
            expected
        );
    }
}

#[test]
fn a_user_schema_reads_nanoseconds_at_its_precision_or_refuses_microseconds() {
    for nanos in [false, true] {
        let s = session(nanos);
        let pa = fixture("pyarrow.parquet");
        run(
            &s,
            &format!(
                "CREATE OR REPLACE TEMPORARY VIEW us (utc TIMESTAMP, local TIMESTAMP_NTZ) USING parquet OPTIONS (path '{pa}')"
            ),
        );
        let e = run(&s, "SELECT CAST(utc AS STRING) FROM us");
        assert!(
            e.starts_with("ERR [FAILED_READ_FILE.PARQUET_COLUMN_DATA_TYPE_MISMATCH]")
                && e.contains("Parquet column [utc]. Expected Spark type timestamp"),
            "{e}"
        );
    }
    let s = session(true);
    let pa = fixture("pyarrow.parquet");
    run(
        &s,
        &format!(
            "CREATE OR REPLACE TEMPORARY VIEW u7 (utc TIMESTAMP_LTZ(7), local TIMESTAMP_NTZ(7)) USING parquet OPTIONS (path '{pa}')"
        ),
    );
    assert_eq!(
        run(
            &s,
            "SELECT typeof(utc), CAST(utc AS STRING), CAST(local AS STRING) FROM u7"
        ),
        "timestamp_ltz(7)|2020-01-01 00:00:00.1234567|2020-01-01 00:00:00.1234567"
    );
}

#[test]
fn a_write_keeps_the_precision_in_the_footer() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().display();
    let on = session(true);
    assert_eq!(
        run(
            &on,
            &format!(
                "INSERT OVERWRITE DIRECTORY '{p}/flat' USING parquet SELECT CAST('{V}' AS TIMESTAMP_NTZ(9)) AS n9, \
                           CAST('{V}' AS TIMESTAMP_LTZ(9)) AS l9, CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_NTZ(7)) AS n7, \
                           array(CAST('{V}' AS TIMESTAMP_NTZ(8))) AS a, named_struct('x', CAST('{V}' AS TIMESTAMP_LTZ(7))) AS s"
            )
        ),
        ""
    );
    assert_eq!(
        run(
            &on,
            &format!(
                "SELECT typeof(n9), typeof(l9), typeof(n7), typeof(a), typeof(s), CAST(l9 AS STRING), CAST(s.x AS STRING) FROM parquet.`{p}/flat`"
            )
        ),
        format!(
            "timestamp_ntz(9)|timestamp_ltz(9)|timestamp_ntz(7)|array<timestamp_ntz(8)>|struct<x:timestamp_ltz(7)>|{V}|2020-01-01 00:00:00.1234567"
        )
    );
    // The footer's Spark schema names the types, as Spark 4.3's does: refused without the flag.
    assert_eq!(
        run(
            &session(false),
            &format!("SELECT * FROM parquet.`{p}/flat`")
        ),
        NOT_ENABLED
    );
}

/// A Spark 4.2 session has no nanosecond types, whatever `spark.sql.timestampNanosTypes.enabled` says
/// (Spark 4.2.1 run): a foreign file's `INT64 TIMESTAMP(NANOS)` is `PARQUET_TYPE_ILLEGAL`, or `bigint`
/// with `nanosAsLong`; a Spark 4.3 file's footer schema fails to parse there and is not used, so its
/// columns read as their Parquet types say.
#[test]
fn a_spark_4_2_session_reads_nanoseconds_as_4_2_does() {
    let illegal = |adjusted: bool| {
        format!(
            "ERR [PARQUET_TYPE_ILLEGAL] Illegal Parquet type: INT64 (TIMESTAMP(NANOS,{adjusted})). SQLSTATE: 42846"
        )
    };
    for flag in ["true", "false"] {
        let s = Session::builder()
            .profile(krishiv_plan::analyzer::SparkProfile::V4_2)
            .build()
            .unwrap();
        s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
        s.conf()
            .set("spark.sql.timestampNanosTypes.enabled", flag)
            .unwrap();
        assert_eq!(
            run(
                &s,
                &format!("SELECT * FROM parquet.`{}`", fixture("pyarrow.parquet"))
            ),
            illegal(true),
            "flag {flag}"
        );
        assert_eq!(
            run(
                &s,
                &format!("SELECT * FROM parquet.`{}`", fixture("spark-flat.parquet"))
            ),
            illegal(false),
            "flag {flag}"
        );
        s.conf()
            .set("spark.sql.legacy.parquet.nanosAsLong", "true")
            .unwrap();
        assert_eq!(
            run(
                &s,
                &format!(
                    "SELECT typeof(utc), utc, typeof(local) FROM parquet.`{}`",
                    fixture("pyarrow.parquet")
                )
            ),
            "bigint|1577836800123456789|bigint",
            "flag {flag}"
        );
        assert_eq!(
            run(
                &s,
                &format!(
                    "SELECT typeof(n7), n7, n9 FROM parquet.`{}`",
                    fixture("spark-flat.parquet")
                )
            ),
            // The stored nanoseconds, decoded with the footer schema dropped.
            "bigint|1577836800123456700|1577836800123456789",
            "flag {flag}"
        );
    }
}

/// On a profile before 4.3 a footer schema that names a nanosecond type is not used at all (Spark 4.2
/// fails to parse the whole string): a collated string in the same file reads as a plain `string`.
#[test]
fn a_dropped_footer_schema_drops_its_collations_too() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t");
    let w = session(true);
    let q = format!(
        "INSERT OVERWRITE DIRECTORY '{}' USING parquet \
         SELECT 'a' COLLATE UTF8_LCASE AS s, CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)) AS t",
        path.display()
    );
    assert_eq!(run(&w, &q), "", "{q}");
    let read = format!(
        "SELECT typeof(s), s = 'A', typeof(t) FROM parquet.`{}`",
        path.display()
    );
    assert_eq!(
        run(&w, &read),
        "string collate UTF8_LCASE|true|timestamp_ntz(9)"
    );
    let s = Session::builder()
        .profile(krishiv_plan::analyzer::SparkProfile::V4_2)
        .build()
        .unwrap();
    s.conf()
        .set("spark.sql.legacy.parquet.nanosAsLong", "true")
        .unwrap();
    assert_eq!(run(&s, &read), "string|false|bigint");
}

/// The pre-4.3 pin of the Parquet nanosecond options reaches only Parquet: under
/// `krishiv.io.strictOptions` a 4.2 session still reads CSV and JSON files (another format does not
/// know the two names and would refuse them).
#[test]
fn the_pre_4_3_pin_leaves_other_formats_alone() {
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "1\n").unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(&json, "{\"a\":1}\n").unwrap();
    let s = Session::builder()
        .profile(krishiv_plan::analyzer::SparkProfile::V4_2)
        .build()
        .unwrap();
    s.conf().set("krishiv.io.strictOptions", "true").unwrap();
    assert_eq!(
        run(&s, &format!("SELECT * FROM csv.`{}`", csv.display())),
        "1"
    );
    assert_eq!(
        run(&s, &format!("SELECT * FROM json.`{}`", json.display())),
        "1"
    );
    assert_eq!(
        run(
            &s,
            &format!(
                "SELECT typeof(utc) FROM parquet.`{}`",
                fixture("pyarrow.parquet")
            )
        ),
        "ERR [PARQUET_TYPE_ILLEGAL] Illegal Parquet type: INT64 (TIMESTAMP(NANOS,true)). SQLSTATE: 42846"
    );
}
