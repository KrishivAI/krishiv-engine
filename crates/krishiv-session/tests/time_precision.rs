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
fn a_constant_time_cast_to_time_p_truncates() {
    // Spark: `time_to_micros(cast(TIME '09:08:01.987654' as time(1)))` is
    // 32881900000.
    for (p, want) in [
        (0, "09:08:01"),
        (1, "09:08:01.9"),
        (2, "09:08:01.98"),
        (3, "09:08:01.987"),
        (4, "09:08:01.9876"),
        (5, "09:08:01.98765"),
        (6, "09:08:01.987654"),
    ] {
        let sql = format!("cast(TIME '09:08:01.987654' as time({p}))");
        assert_eq!(text(&sql), want, "{sql}");
    }
}

#[test]
fn a_string_cast_to_time_p_keeps_the_digits() {
    // Spark 4.2.0: `cast(cast('09:08:01.987654' as time(1)) as string)` is
    // `09:08:01.987654` (the design's owner question 5).
    assert_eq!(
        text("cast(cast('09:08:01.987654' as time(1)) as string)"),
        "09:08:01.987654"
    );
    assert_eq!(
        text("(SELECT CAST(s AS time(1)) FROM (SELECT '09:08:01.987654' s))"),
        "09:08:01.987654"
    );
}

#[test]
fn a_column_cast_to_time_p_truncates_the_same() {
    assert_eq!(
        text("(SELECT CAST(t AS time(1)) FROM (SELECT TIME '09:08:01.987654' t))"),
        "09:08:01.9"
    );
}

#[test]
fn nested_time_constants_truncate() {
    assert_eq!(
        text("cast(array(TIME '09:08:01.987654') as array<time(2)>)"),
        "[09:08:01.98]"
    );
    assert_eq!(
        text("cast(named_struct('a', TIME '09:08:01.987654') as struct<a: time(3)>)"),
        "{09:08:01.987}"
    );
}

#[test]
fn null_and_invalid_inputs_keep_their_modes() {
    assert_eq!(
        text("coalesce(CAST(CAST(NULL AS STRING) AS time(2)), TIME '01:02:03')"),
        "01:02:03"
    );
    assert_eq!(
        text("coalesce(try_cast('nope' AS time(2)), TIME '01:02:03')"),
        "01:02:03"
    );
}

#[test]
fn an_invalid_string_is_an_error_under_ansi() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = true")).unwrap();
    let e = block_on(s.sql("SELECT CAST('nope' AS time(2))"))
        .and_then(|df| block_on(df.collect()))
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::CastInvalidInput
    );
}

/// `extract(SECOND ...)` cuts a `time(p)` to `p` digits even when the value
/// holds more (a string cast keeps them): golden `time.sql` 47-52.
#[test]
fn extract_second_follows_the_declared_precision() {
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
        text("date_part('SECOND', cast('09:08:01.987654' as time(2)))"),
        "1.980000"
    );
    assert_eq!(
        text("(SELECT extract(SECOND FROM CAST(s AS time(3))) FROM (SELECT '09:08:01.987654' s))"),
        "1.987000"
    );
    assert_eq!(
        text("isnull(extract(SECOND FROM CAST(CAST(NULL AS STRING) AS time(3))))"),
        "true"
    );
    // Other fields are not touched.
    assert_eq!(
        text("extract(MINUTE FROM cast('09:08:01.987654' as time(1)))"),
        "8"
    );
}

/// A cast to the type a value already has is a no-op (Spark 4.2.0:
/// `time_to_micros(cast(cast('09:08:01.987654' as time(1)) as time(1)))` is
/// 32881987654), while a cast to another precision truncates.
#[test]
fn a_cast_to_the_same_time_type_is_a_no_op() {
    assert_eq!(
        text("cast(cast(cast('09:08:01.987654' as time(1)) as time(1)) as string)"),
        "09:08:01.987654"
    );
    assert_eq!(
        text("cast(cast(cast('09:08:01.987654' as time(1)) as time(2)) as string)"),
        "09:08:01.98"
    );
}

/// A `time(3)` column read from rows (not a folded constant) is cut by
/// `extract`, and the output column is named and typed as Spark names it
/// (golden: `extract(SECOND FROM CAST(.. AS TIME(1))):decimal(8,6)`).
#[test]
fn extract_second_on_a_column_and_the_output_schema() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let sql = "SELECT extract(SECOND FROM CAST(s AS time(3))) FROM \
               VALUES ('09:08:01.987654'), ('00:00:00.123456') AS v(s)";
    let df = block_on(s.sql(sql)).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema.field(0).name(),
        "extract(SECOND FROM CAST(s AS TIME(3)))"
    );
    assert_eq!(
        schema.field(0).data_type(),
        &arrow_schema::DataType::Decimal128(8, 6)
    );
    let batches = block_on(df.collect()).unwrap();
    let mut got = Vec::new();
    for b in &batches {
        let col = b
            .column(0)
            .as_primitive::<arrow_array::types::Decimal128Type>();
        got.extend((0..col.len()).map(|i| col.value(i)));
    }
    got.sort_unstable();
    assert_eq!(got, vec![123_000, 1_987_000]);
}
