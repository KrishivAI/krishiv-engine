// SPDX-License-Identifier: Apache-2.0
//! A cast to `time(p)` drops the digits below `p`, for a constant (folded
//! by the analyzer) as for a column (cast by the executor). Values from the
//! Spark 4.2.0 `time.sql` golden.
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
    assert!(!c.is_null(0), "{sql}");
    c.as_string_view().value(0).to_owned()
}

#[test]
fn a_constant_cast_to_time_p_truncates() {
    for (p, want) in [
        (0, "1.000000"),
        (1, "1.900000"),
        (2, "1.980000"),
        (3, "1.987000"),
        (4, "1.987600"),
        (5, "1.987650"),
        (6, "1.987654"),
    ] {
        let sql = format!("extract(SECOND FROM cast('09:08:01.987654' as time({p})))");
        assert_eq!(text(&sql), want, "{sql}");
    }
    assert_eq!(
        text("cast(TIME '09:08:01.987654' as time(2))"),
        "09:08:01.98"
    );
}

#[test]
fn a_column_cast_to_time_p_truncates_the_same() {
    assert_eq!(
        text("(SELECT CAST(t AS time(1)) FROM (SELECT TIME '09:08:01.987654' t))"),
        "09:08:01.9"
    );
}
