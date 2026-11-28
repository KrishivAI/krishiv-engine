// SPDX-License-Identifier: Apache-2.0
//! `unix_seconds`, `unix_millis` and `unix_micros` of a `TIMESTAMP_NTZ`: refused before Spark 4.3, its wall
//! clock read in UTC from 4.3 whatever the session zone. Expected values and messages from Spark 4.2.0, 4.2.1
//! and 4.3.0 runs (`spark-sql -f`, session zones UTC and America/Los_Angeles; 3.5.7 refuses it too).
#![expect(clippy::unwrap_used, clippy::expect_used, reason = "a test")]

use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn session(profile: SparkProfile, zone: &str) -> Session {
    let s = Session::builder().profile(profile).build().unwrap();
    s.conf().set("spark.sql.session.timeZone", zone).unwrap();
    s
}

fn run(s: &Session, q: &str) -> Result<String, String> {
    let batches = futures::executor::block_on(async { s.sql(q).await?.collect().await })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| arrow_cast::display::array_value_to_string(c, r).expect("display"))
                .collect();
            out.push(cells.join(" "));
        }
    }
    Ok(out.join(" | "))
}

const PARTS: &str = "SELECT unix_micros(TIMESTAMP_NTZ'2020-01-02 03:04:05.123456'), \
     unix_millis(TIMESTAMP_NTZ'2020-01-02 03:04:05.123456'), \
     unix_seconds(TIMESTAMP_NTZ'2020-01-02 03:04:05.123456')";

#[test]
fn spark_43_reads_a_timestamp_ntz_as_its_wall_clock_in_utc() {
    for zone in ["UTC", "America/Los_Angeles"] {
        let s = session(SparkProfile::V4_3, zone);
        assert_eq!(
            run(&s, PARTS).unwrap(),
            "1577934245123456 1577934245123 1577934245",
            "{zone}"
        );
        assert_eq!(
            run(
                &s,
                "SELECT typeof(unix_micros(TIMESTAMP_NTZ'2020-01-02 03:04:05')), \
                 unix_micros(CAST(NULL AS TIMESTAMP_NTZ)), unix_seconds(TIMESTAMP_NTZ'1969-12-31 23:59:59.5')"
            )
            .unwrap(),
            "bigint  -1",
            "{zone}: NULL stays NULL; seconds floor"
        );
    }
}

#[test]
fn spark_43_names_every_timestamp_kind_for_another_type() {
    let s = session(SparkProfile::V4_3, "UTC");
    for q in [
        "SELECT unix_micros(DATE'2020-01-02')",
        "SELECT unix_micros('2020-01-02 03:04:05')",
        "SELECT unix_seconds(DATE'2020-01-02')",
    ] {
        let e = run(&s, q).expect_err(q);
        assert!(
            e.contains("DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE")
                && e.contains(
                    r#"requires the ("(TIMESTAMP OR TIMESTAMP WITHOUT TIME ZONE)" or "(TIMESTAMP_LTZ(P) OR TIMESTAMP_NTZ(P) WITH P IN [7, 9])") type"#
                ),
            "{q}: {e}"
        );
    }
}

#[test]
fn before_spark_43_a_timestamp_ntz_is_refused() {
    for profile in [SparkProfile::V4_2, SparkProfile::V3_5] {
        let s = session(profile, "UTC");
        let e = run(&s, PARTS).expect_err("refused");
        assert!(
            e.contains("DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE")
                && e.contains(r#"requires the "TIMESTAMP" type"#),
            "{profile:?}: {e}"
        );
        assert_eq!(
            run(
                &s,
                "SELECT unix_micros(TIMESTAMP'2020-01-02 03:04:05.123456')"
            )
            .unwrap(),
            "1577934245123456",
            "{profile:?}: a TIMESTAMP still reads"
        );
    }
}
