// SPDX-License-Identifier: Apache-2.0
//! `GetXdbcTypeInfo` (design P3-T101 §5): one row per Spark type a result column can have, in
//! the XDBC (JDBC/ODBC) terms of FlightSql.proto, built once with arrow-flight's
//! `XdbcTypeInfoDataBuilder`. The command reads no session state, so it runs without one, as
//! `GetSqlInfo` does.
//!
//! Spark has no Flight SQL service, so nothing here is Spark-observable; the rows describe the
//! types as Krishiv sends them (Spark's Arrow shapes, `krishiv_connect_server::execute::arrow`):
//! - `type_name` is Spark's SQL name (`INT`, `STRING`, `TIMESTAMP_NTZ`, ...), as `CAST` and DDL
//!   take it; `local_type_name` is the same.
//! - `TIMESTAMP` is a time zone-aware instant (the column carries the session time zone), so its
//!   datetime subcode is `TIMESTAMP WITH TIME ZONE`; `TIMESTAMP_NTZ` is the plain `TIMESTAMP`.
//! - `FLOAT` is four bytes, JDBC's `REAL`; `DOUBLE` is JDBC's `DOUBLE`.
//! - The complex types (`ARRAY`, `MAP`, `STRUCT`), `VARIANT`, the geospatial types and `VOID`
//!   have no XDBC type code in FlightSql.proto's `XdbcDataType`, so they are not listed: a row
//!   with `XDBC_UNKNOWN_TYPE` would tell a driver nothing it can map.

use arrow_flight::sql::metadata::{XdbcTypeInfo, XdbcTypeInfoData, XdbcTypeInfoDataBuilder};
use arrow_flight::sql::{Nullable, Searchable, XdbcDataType, XdbcDatetimeSubcode};
use krishiv_common::error::{InternalError, KrishivError};

/// Spark's widest `DECIMAL` precision (and scale).
const MAX_DECIMAL_PRECISION: i32 = 38;
/// Digits of the fraction of a Spark timestamp (microseconds).
const TIMESTAMP_SCALE: i32 = 6;
/// `yyyy-MM-dd HH:mm:ss.SSSSSS`.
const TIMESTAMP_COLUMN_SIZE: i32 = 26;
/// The column size of the unbounded types (`STRING`, `BINARY`), as JDBC drivers report it.
const UNBOUNDED: i32 = i32::MAX;

/// A row with the fields every Spark type shares: nullable, searchable except by `LIKE`, not
/// case sensitive, no literal quoting.
fn base(type_name: &str, data_type: XdbcDataType) -> XdbcTypeInfo {
    XdbcTypeInfo {
        type_name: type_name.to_owned(),
        data_type,
        sql_data_type: data_type,
        local_type_name: Some(type_name.to_owned()),
        nullable: Nullable::NullabilityNullable,
        searchable: Searchable::Basic,
        auto_increment: Some(false),
        ..XdbcTypeInfo::default()
    }
}

/// A signed exact or approximate number with `digits` of precision in base 10.
fn number(type_name: &str, data_type: XdbcDataType, digits: i32) -> XdbcTypeInfo {
    XdbcTypeInfo {
        column_size: Some(digits),
        unsigned_attribute: Some(false),
        num_prec_radix: Some(10),
        minimum_scale: Some(0),
        maximum_scale: Some(0),
        ..base(type_name, data_type)
    }
}

/// A datetime type (`sql_data_type` `XDBC_DATETIME` with its subcode, as ODBC's
/// `SQL_DESC_CONCISE_TYPE` / `SQL_DESC_DATETIME_INTERVAL_CODE`).
fn datetime(
    type_name: &str,
    data_type: XdbcDataType,
    subcode: XdbcDatetimeSubcode,
    column_size: i32,
    scale: i32,
) -> XdbcTypeInfo {
    XdbcTypeInfo {
        column_size: Some(column_size),
        literal_prefix: Some(format!("{type_name} '")),
        literal_suffix: Some("'".to_owned()),
        minimum_scale: Some(0),
        maximum_scale: Some(scale),
        sql_data_type: XdbcDataType::XdbcDatetime,
        datetime_subcode: Some(subcode),
        ..base(type_name, data_type)
    }
}

/// The interval type `INTERVAL <fields>` (`XDBC_INTERVAL` with its subcode).
fn interval(fields: &str, subcode: XdbcDatetimeSubcode, scale: i32) -> XdbcTypeInfo {
    XdbcTypeInfo {
        literal_prefix: Some("INTERVAL '".to_owned()),
        literal_suffix: Some(format!("' {fields}")),
        minimum_scale: Some(0),
        maximum_scale: Some(scale),
        datetime_subcode: Some(subcode),
        ..base(&format!("INTERVAL {fields}"), XdbcDataType::XdbcInterval)
    }
}

/// The rows, one per Spark type a result column can have (module docs).
pub(crate) fn xdbc_type_rows() -> Vec<XdbcTypeInfo> {
    use XdbcDataType as X;
    use XdbcDatetimeSubcode as S;
    [
        XdbcTypeInfo {
            column_size: Some(1),
            ..base("BOOLEAN", X::XdbcBit)
        },
        number("TINYINT", X::XdbcTinyint, 3),
        number("SMALLINT", X::XdbcSmallint, 5),
        number("INT", X::XdbcInteger, 10),
        number("BIGINT", X::XdbcBigint, 19),
        XdbcTypeInfo {
            minimum_scale: None,
            maximum_scale: None,
            ..number("FLOAT", X::XdbcReal, 7)
        },
        XdbcTypeInfo {
            minimum_scale: None,
            maximum_scale: None,
            ..number("DOUBLE", X::XdbcDouble, 15)
        },
        XdbcTypeInfo {
            create_params: Some(vec!["precision".to_owned(), "scale".to_owned()]),
            maximum_scale: Some(MAX_DECIMAL_PRECISION),
            ..number("DECIMAL", X::XdbcDecimal, MAX_DECIMAL_PRECISION)
        },
        XdbcTypeInfo {
            column_size: Some(UNBOUNDED),
            literal_prefix: Some("'".to_owned()),
            literal_suffix: Some("'".to_owned()),
            case_sensitive: true,
            searchable: Searchable::Full,
            ..base("STRING", X::XdbcVarchar)
        },
        XdbcTypeInfo {
            column_size: Some(UNBOUNDED),
            literal_prefix: Some("X'".to_owned()),
            literal_suffix: Some("'".to_owned()),
            ..base("BINARY", X::XdbcVarbinary)
        },
        // ODBC's date subcode is 1, which FlightSql.proto names `XDBC_SUBCODE_YEAR` (the enum
        // shares its values between the datetime and the interval codes).
        datetime("DATE", X::XdbcDate, S::XdbcSubcodeYear, 10, 0),
        datetime(
            "TIMESTAMP",
            X::XdbcTimestamp,
            S::XdbcSubcodeTimestampWithTimezone,
            TIMESTAMP_COLUMN_SIZE,
            TIMESTAMP_SCALE,
        ),
        datetime(
            "TIMESTAMP_NTZ",
            X::XdbcTimestamp,
            S::XdbcSubcodeTimestamp,
            TIMESTAMP_COLUMN_SIZE,
            TIMESTAMP_SCALE,
        ),
    ]
    .into_iter()
    .chain(INTERVALS.iter().map(|(fields, subcode)| {
        // Seconds carry microseconds; an interval ending in another field has no fraction.
        let scale = if fields.ends_with("SECOND") {
            TIMESTAMP_SCALE
        } else {
            0
        };
        interval(fields, *subcode, scale)
    }))
    .collect()
}

/// Every field range of Spark's `YearMonthIntervalType` and `DayTimeIntervalType` (a result
/// column can have any of them, `INTERVAL '1' HOUR` is `INTERVAL HOUR`), each with its XDBC
/// interval subcode.
const INTERVALS: [(&str, XdbcDatetimeSubcode); 13] = [
    ("YEAR", XdbcDatetimeSubcode::XdbcSubcodeIntervalYear),
    ("MONTH", XdbcDatetimeSubcode::XdbcSubcodeIntervalMonth),
    (
        "YEAR TO MONTH",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalYearToMonth,
    ),
    ("DAY", XdbcDatetimeSubcode::XdbcSubcodeIntervalDay),
    (
        "DAY TO HOUR",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalDayToHour,
    ),
    (
        "DAY TO MINUTE",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalDayToMinute,
    ),
    (
        "DAY TO SECOND",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalDayToSecond,
    ),
    ("HOUR", XdbcDatetimeSubcode::XdbcSubcodeIntervalHour),
    (
        "HOUR TO MINUTE",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalHourToMinute,
    ),
    (
        "HOUR TO SECOND",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalHourToSecond,
    ),
    ("MINUTE", XdbcDatetimeSubcode::XdbcSubcodeIntervalMinute),
    (
        "MINUTE TO SECOND",
        XdbcDatetimeSubcode::XdbcSubcodeIntervalMinuteToSecond,
    ),
    ("SECOND", XdbcDatetimeSubcode::XdbcSubcodeIntervalSecond),
];

/// The `GetXdbcTypeInfo` table (built once, at server start).
///
/// # Errors
/// `INTERNAL_ERROR` when arrow-flight cannot build the table.
pub(crate) fn xdbc_type_info_data() -> Result<XdbcTypeInfoData, KrishivError> {
    let mut builder = XdbcTypeInfoDataBuilder::new();
    for row in xdbc_type_rows() {
        builder.append(row);
    }
    builder.build().map_err(|e| {
        InternalError {
            message: format!("Flight SQL GetXdbcTypeInfo table: {e}"),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;

    use super::*;

    #[test]
    fn every_row_is_built_and_names_are_unique() {
        let data = xdbc_type_info_data().expect("table builds");
        let all = data.record_batch(None).expect("all rows");
        assert_eq!(all.num_rows(), xdbc_type_rows().len());
        let names = all.column(0).as_string::<i32>();
        let mut seen: Vec<&str> = names.iter().flatten().collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), all.num_rows());
    }

    #[test]
    fn a_data_type_filter_selects_its_rows() {
        let data = xdbc_type_info_data().expect("table builds");
        let timestamps = data
            .record_batch(XdbcDataType::XdbcTimestamp as i32)
            .expect("filtered");
        let names: Vec<&str> = timestamps
            .column(0)
            .as_string::<i32>()
            .iter()
            .flatten()
            .collect();
        assert_eq!(names, ["TIMESTAMP", "TIMESTAMP_NTZ"]);
        let codes = timestamps.column(1).as_primitive::<Int32Type>();
        assert!(
            codes
                .iter()
                .all(|c| c == Some(XdbcDataType::XdbcTimestamp as i32))
        );
        assert_eq!(
            data.record_batch(XdbcDataType::XdbcWchar as i32)
                .expect("filtered")
                .num_rows(),
            0
        );
    }

    #[test]
    fn interval_literals_name_their_fields() {
        let rows = xdbc_type_rows();
        let day = rows
            .iter()
            .find(|r| r.type_name == "INTERVAL DAY TO SECOND")
            .expect("row");
        assert_eq!(day.literal_prefix.as_deref(), Some("INTERVAL '"));
        assert_eq!(day.literal_suffix.as_deref(), Some("' DAY TO SECOND"));
        assert_eq!(day.maximum_scale, Some(TIMESTAMP_SCALE));
        let hour = rows
            .iter()
            .find(|r| r.type_name == "INTERVAL HOUR")
            .expect("single-field intervals are listed");
        assert_eq!(
            hour.datetime_subcode,
            Some(XdbcDatetimeSubcode::XdbcSubcodeIntervalHour)
        );
        assert_eq!(hour.maximum_scale, Some(0));
        let listed = rows
            .iter()
            .filter(|r| r.data_type == XdbcDataType::XdbcInterval)
            .count();
        assert_eq!(listed, 13);
    }
}
