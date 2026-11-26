// SPDX-License-Identifier: Apache-2.0
//! The nanosecond timestamp types through the SQL grammar and the analyzer
//! (`spark.sql.timestampNanosTypes.enabled`, Spark 4.3).
#![expect(clippy::unwrap_used, clippy::expect_used, reason = "a test")]
use krishiv_session::Session;

fn session(on: bool) -> Session {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    s.conf()
        .set(
            "spark.sql.timestampNanosTypes.enabled",
            if on { "true" } else { "false" },
        )
        .unwrap();
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
    Ok(out.join("|"))
}

const N9: &str = "CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))";

fn one(q: &str) -> String {
    run(&session(true), q).unwrap_or_else(|e| format!("ERR {e}"))
}

#[test]
fn types_print_with_their_precision() {
    assert_eq!(one(&format!("SELECT typeof({N9})")), "timestamp_ntz(9)");
    assert_eq!(
        one("SELECT typeof(CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_LTZ(7)))"),
        "timestamp_ltz(7)"
    );
    assert_eq!(
        one(&format!("SELECT CAST({N9} AS STRING)")),
        "2020-01-01 00:00:00.123456789"
    );
}

#[test]
fn casts_between_nanos_micros_and_dates_resolve() {
    assert_eq!(one(&format!("SELECT CAST({N9} AS DATE)")), "2020-01-01");
    assert!(
        !one(&format!(
            "SELECT CAST(CAST({N9} AS TIMESTAMP_NTZ) AS STRING)"
        ))
        .starts_with("ERR")
    );
    assert!(!one(&format!("SELECT CAST(CAST({N9} AS TIMESTAMP) AS STRING)")).starts_with("ERR"));
    assert!(!one("SELECT CAST(DATE'2020-01-01' AS TIMESTAMP_NTZ(8))").starts_with("ERR"));
}

#[test]
fn nanos_types_meet_at_the_wider_precision() {
    let seven = "CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_NTZ(7))";
    let micro = "TIMESTAMP_NTZ'2020-01-01 00:00:00.123456'";
    assert_eq!(one(&format!("SELECT {seven} < {N9}")), "true");
    assert_eq!(one(&format!("SELECT {micro} < {seven}")), "true");
    assert_eq!(
        one(&format!("SELECT typeof(greatest({seven}, {N9}))")),
        "timestamp_ntz(9)"
    );
    assert_eq!(
        one(&format!("SELECT typeof(coalesce({micro}, {N9}))")),
        "timestamp_ntz(9)"
    );
    assert_eq!(
        one(&format!(
            "SELECT typeof(coalesce(CAST({N9} AS TIMESTAMP_LTZ(9)), {seven}))"
        )),
        "timestamp_ltz(9)"
    );
}

#[test]
fn precisions_outside_the_range_are_refused_and_the_key_gates_the_types() {
    assert!(one("SELECT CAST(1 AS TIMESTAMP_NTZ(3))").contains("INVALID_TIMESTAMP_PRECISION"));
    assert!(one("SELECT current_timestamp(10)").contains("INVALID_TIMESTAMP_PRECISION"));
    let off = run(&session(false), "SELECT CAST(1 AS TIMESTAMP_NTZ(9))").unwrap_err();
    assert!(off.contains("FEATURE_NOT_ENABLED"), "{off}");
}
