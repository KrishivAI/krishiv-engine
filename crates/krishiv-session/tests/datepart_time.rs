// SPDX-License-Identifier: Apache-2.0
//! `datepart` reads the TIME precision like `date_part`.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn one(s: &Session, sql: &str) -> Result<String, KrishivError> {
    let df = block_on(s.sql(sql))?;
    let batches = block_on(df.collect())?;
    let c = batches[0].column(0);
    Ok(match c.data_type() {
        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(0).to_owned(),
        arrow_schema::DataType::Utf8View => c.as_string_view().value(0).to_owned(),
        other => panic!("unexpected result type {other}"),
    })
}

fn seconds(profile: SparkProfile) -> (String, String) {
    let s = Session::builder().profile(profile).build().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let a = one(
        &s,
        "SELECT CAST(datepart('SECOND', CAST('09:08:01.987654' AS TIME(1))) AS STRING)",
    )
    .unwrap();
    let b = one(
        &s,
        "SELECT CAST(date_part('SECOND', CAST('09:08:01.987654' AS TIME(1))) AS STRING)",
    )
    .unwrap();
    (a, b)
}

#[test]
fn datepart_reads_the_time_precision_like_date_part() {
    // Spark 4.2.0: TIME(1) keeps one fractional digit, so SECOND is 1.9 as DECIMAL(8,6).
    let (a, b) = seconds(SparkProfile::V4_2);
    assert_eq!(a, "1.900000");
    assert_eq!(a, b);
    // Spark 4.3.0: SECOND of a TIME(p) is a DECIMAL(p + 2, p) (golden
    // `extract_second_low_precision`).
    let (a, b) = seconds(SparkProfile::V4_3);
    assert_eq!(a, "1.9");
    assert_eq!(a, b);
}
