// SPDX-License-Identifier: Apache-2.0
//! The vector functions follow the session's Spark profile at execution: Spark 4.3's infinity norm
//! propagates a NaN element where 4.2's skips it.
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn one(profile: SparkProfile, sql: &str) -> String {
    let s = Session::builder().profile(profile).build().unwrap();
    block_on(async {
        let batches = s.sql(sql).await.unwrap().collect().await.unwrap();
        batches[0].column(0).as_string_view().value(0).to_owned()
    })
}

#[test]
fn the_infinity_norm_follows_the_sessions_profile() {
    let sql = "SELECT CAST(vector_norm(array(CAST('NaN' AS FLOAT), CAST(1.0 AS FLOAT)), \
               CAST('inf' AS FLOAT)) AS STRING)";
    assert_eq!(one(SparkProfile::V4_3, sql), "NaN");
    assert_eq!(one(SparkProfile::V4_2, sql), "1.0");
}
