// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.timestampType = TIMESTAMP_NTZ`: `to_timestamp` and `make_timestamp`
//! produce `TIMESTAMP_NTZ` and keep their names (Spark 4.2.0); a bare `TIMESTAMP` type is
//! `TIMESTAMP_NTZ`, resolved while parsing (`DataTypeAstBuilder.visitPrimitiveDataType`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The column name, Arrow type and first value (as text) of a one-cell query.
fn cell(s: &Session, sql: &str) -> (String, String, String) {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let b = &batches[0];
    let schema = b.schema();
    let f = &schema.fields()[0];
    let c = b.column(0);
    let v = if c.is_null(0) {
        "NULL".to_owned()
    } else {
        match c.data_type() {
            arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, tz) => {
                let micros = c
                    .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                    .value(0);
                format!("{micros} {}", tz.is_some())
            }
            other => format!("<{other}>"),
        }
    };
    (f.name().clone(), format!("{}", f.data_type()), v)
}

fn ntz() -> Session {
    Session::builder()
        .config("spark.sql.timestampType", "TIMESTAMP_NTZ")
        .build()
        .unwrap()
}

#[test]
fn to_timestamp_is_ntz_and_keeps_its_name() {
    let s = ntz();
    // 2016-12-31 00:12:00 as microseconds since the epoch, no zone.
    let want = "1483143120000000 false";
    for sql in [
        "select to_timestamp('2016-12-31 00:12:00')",
        "select to_timestamp('2016-12-31 00:12:00', 'yyyy-MM-dd HH:mm:ss')",
    ] {
        let (name, _, v) = cell(&s, sql);
        assert!(name.starts_with("to_timestamp("), "{name}");
        assert_eq!(v, want, "{sql}");
    }
    let (_, _, v) = cell(&s, "select to_timestamp(date'2016-12-31')");
    assert_eq!(v, "1483142400000000 false");
}

#[test]
fn make_timestamp_is_ntz_and_ignores_its_zone() {
    let s = ntz();
    let plain = cell(&s, "select make_timestamp(2021, 7, 11, 6, 30, 45.678)");
    assert!(plain.0.starts_with("make_timestamp("), "{}", plain.0);
    assert!(plain.2.ends_with("false"), "{}", plain.2);
    let zoned = cell(
        &s,
        "select make_timestamp(2021, 7, 11, 6, 30, 45.678, 'CET')",
    );
    assert_eq!(zoned.2, plain.2);
    assert!(zoned.0.ends_with(", CET)"), "{}", zoned.0);
    let tried = cell(&s, "select try_make_timestamp(2021, 7, 11, 6, 30, 45.678)");
    assert_eq!(tried.2, plain.2);
}

#[test]
fn the_default_session_still_makes_zoned_timestamps() {
    let s = Session::new().unwrap();
    let (_, _, v) = cell(&s, "select to_timestamp('2016-12-31 00:12:00')");
    assert!(v.ends_with("true"), "{v}");
}

/// The `typeof` of `expr` in a one-row query.
fn type_of(s: &Session, expr: &str) -> String {
    let sql = format!("select typeof({expr})");
    let df = block_on(s.sql(&sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let c = batches[0].column(0);
    match c.data_type() {
        arrow_schema::DataType::Utf8View => c.as_string_view().value(0).to_owned(),
        arrow_schema::DataType::LargeUtf8 => c.as_string::<i64>().value(0).to_owned(),
        _ => c.as_string::<i32>().value(0).to_owned(),
    }
}

#[test]
fn a_bare_timestamp_type_is_the_session_timestamp_type() {
    let s = ntz();
    assert_eq!(type_of(&s, "cast(null as timestamp)"), "timestamp_ntz");
    assert_eq!(
        type_of(&s, "cast('2016-12-31 00:12:00' as timestamp)"),
        "timestamp_ntz"
    );
    assert_eq!(
        type_of(&s, "cast(null as array<struct<a: timestamp>>)"),
        "array<struct<a:timestamp_ntz>>"
    );
    // The explicit forms keep their own type.
    assert_eq!(type_of(&s, "cast(null as timestamp_ltz)"), "timestamp");
    assert_eq!(
        type_of(&s, "cast(null as timestamp with local time zone)"),
        "timestamp"
    );
    let (_, _, v) = cell(&s, "select cast('2016-12-31 00:12:00' as timestamp)");
    assert_eq!(v, "1483143120000000 false");
    // The default session keeps the zoned type.
    assert_eq!(
        type_of(&Session::new().unwrap(), "cast(null as timestamp)"),
        "timestamp"
    );
}

#[test]
fn a_table_column_keeps_the_type_its_create_resolved() {
    let dir = tempfile::tempdir().unwrap();
    let s = ntz();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    block_on(s.sql("create table t (c timestamp) using parquet")).unwrap();
    // Resolved at CREATE: a later change of the setting leaves the stored column alone.
    block_on(s.sql("set spark.sql.timestampType = TIMESTAMP_LTZ")).unwrap();
    let df = block_on(s.sql("select c from t")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    assert_eq!(
        schema.fields()[0].data_type(),
        &arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
    );
}
