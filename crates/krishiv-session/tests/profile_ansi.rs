// SPDX-License-Identifier: Apache-2.0
//! The ANSI default follows the session's Spark profile (issue S4M-3, ADR 0003 update 2026-11-27): on the 3.5
//! profile `spark.sql.ansi.enabled` is `false` unless set, so `1/0` and a failed cast are NULL as in Spark 3.5.9
//! (`SQLConf.ANSI_ENABLED` at v3.5.10: off unless `SPARK_ANSI_SQL_MODE=true`); from 4.0 they raise.
#![forbid(unsafe_code)]

use arrow_array::Array;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Whether the one cell `sql` returns is NULL.
fn is_null(s: &Session, sql: &str) -> R<bool> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let batch = batches.iter().find(|b| b.num_rows() > 0).ok_or("no row")?;
        Ok(batch.column(0).is_null(0))
    })
}

fn error_class(s: &Session, sql: &str) -> Option<String> {
    block_on(async {
        match s.sql(sql).await {
            Err(e) => Some(e.error_class().name().to_owned()),
            Ok(df) => df
                .collect()
                .await
                .err()
                .map(|e| e.error_class().name().to_owned()),
        }
    })
}

#[test]
fn a_3_5_session_returns_null_where_4_x_raises() -> R {
    let old = Session::builder().profile(SparkProfile::V3_5).build()?;
    assert!(is_null(&old, "SELECT 1/0")?);
    assert!(is_null(&old, "SELECT cast('abc' AS int)")?);
    assert_eq!(
        old.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );

    let new = Session::builder().profile(SparkProfile::V4_0).build()?;
    assert_eq!(
        error_class(&new, "SELECT 1/0").as_deref(),
        Some("DIVIDE_BY_ZERO")
    );
    assert_eq!(
        error_class(&new, "SELECT cast('abc' AS int)").as_deref(),
        Some("CAST_INVALID_INPUT")
    );
    Ok(())
}

/// A set value wins over the profile's default, both ways.
#[test]
fn a_set_value_wins_over_the_profile_default() -> R {
    let old = Session::builder()
        .profile(SparkProfile::V3_5)
        .config("spark.sql.ansi.enabled", "true")
        .build()?;
    assert_eq!(
        error_class(&old, "SELECT 1/0").as_deref(),
        Some("DIVIDE_BY_ZERO")
    );
    let new = Session::builder().profile(SparkProfile::V4_3).build()?;
    block_on(new.sql("SET spark.sql.ansi.enabled = false"))?;
    assert!(is_null(&new, "SELECT 1/0")?);
    Ok(())
}
