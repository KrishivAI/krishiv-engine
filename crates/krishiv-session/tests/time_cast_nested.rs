// SPDX-License-Identifier: Apache-2.0
//! TIME inside nested values, long-fraction literals and `listagg` over a
//! TIME cast, against what Spark 4.2.0 gave (local-mode run, TIME enabled).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn text(sql: &str) -> String {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    let c = &b[0].columns()[0];
    if c.is_null(0) {
        "NULL".to_owned()
    } else {
        c.as_string_view().value(0).to_owned()
    }
}

#[test]
fn nested_time_casts_drop_digits_below_the_precision() {
    for (sql, want) in [
        (
            "CAST(array(TIME '12:34:56.789123') AS array<time(3)>)",
            "[12:34:56.789]",
        ),
        (
            "CAST(named_struct('a', TIME '12:34:56.789123') AS struct<a: time(3)>)",
            "{12:34:56.789}",
        ),
    ] {
        assert_eq!(text(sql), want, "{sql}");
    }
}

#[test]
fn a_time_literal_with_a_long_fraction_truncates_to_microseconds() {
    for sql in [
        "TIME '12:34:56.12345678901'",
        "TIME '12:34:56.1234567890123'",
    ] {
        assert_eq!(text(sql), "12:34:56.123456", "{sql}");
    }
}

/// Spark accepts `listagg(DISTINCT CAST(t AS STRING)) ... ORDER BY t` for a
/// TIME (`01:00:0002:00:00`). The analyzer must too; ordering by a TIME key
/// is not built yet, which is a different error.
#[test]
fn listagg_distinct_over_a_time_cast_passes_the_analyzer() {
    let sql = "SELECT listagg(DISTINCT CAST(t AS STRING)) WITHIN GROUP (ORDER BY t) \
               FROM VALUES (TIME '01:00:00'), (TIME '02:00:00') AS v(t)";
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
}

/// Spark types a TIME cast to `tinyint`/`smallint` as nullable (the seconds
/// may overflow, NULL with ANSI off) and `int`/`bigint`/`time(p)` as not; a
/// map key that may become NULL is refused.
#[test]
fn narrowing_time_casts_are_nullable() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    for (to, nullable) in [
        ("tinyint", true),
        ("smallint", true),
        ("int", false),
        ("bigint", false),
        ("time(3)", false),
    ] {
        let df = block_on(s.sql(&format!(
            "SELECT CAST(t AS {to}) c FROM (SELECT TIME '12:00:00' t)"
        )))
        .unwrap();
        let schema = block_on(df.schema()).unwrap();
        assert_eq!(schema.field(0).is_nullable(), nullable, "{to}");
    }
    let e =
        block_on(s.sql("SELECT CAST(map(TIME '23:59:59', 1) AS map<tinyint, int>)")).unwrap_err();
    assert_ne!(
        e.error_class(),
        krishiv_common::error::ErrorClass::InternalError,
        "{e}"
    );
}
