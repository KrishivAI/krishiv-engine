// SPDX-License-Identifier: Apache-2.0
//! Nanosecond timestamps (Spark 4.3 `TIMESTAMP_NTZ(9)` / `TIMESTAMP_LTZ(9)`, Arrow `Timestamp(ns)`) through
//! the executor: a table of such columns is sorted, grouped, joined, aggregated, hashed, nested, windowed
//! and printed with every nanosecond. The SQL spellings of the types (`timestamp_ntz(9)`) are the
//! grammar's; these queries read Arrow columns, so they need none. Values are Spark's (a `TimestampNanosVal`
//! hash is `hashInt(nanosWithinMicro, hashLong(epochMicros, seed))`).
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "a test"
)]
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use krishiv_session::Session;

/// `t(k, c timestamp_ntz(9), l timestamp_ltz(9))`: two equal values, one a nanosecond apart, one before the
/// epoch with nine digits, and a NULL; the session zone is UTC.
fn session() -> Session {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("c", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new(
            "l",
            DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("UTC"))),
            true,
        ),
    ]));
    let v = vec![
        Some(1_577_836_800_000_000_001_i64),
        Some(1_577_836_800_000_000_999),
        Some(1_577_836_800_000_000_001),
        Some(-315_619_200_000_000_000 + 123_456_789),
        None,
    ];
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_iter_values(0..5)),
        Arc::new(TimestampNanosecondArray::from(v.clone())),
        Arc::new(TimestampNanosecondArray::from(v).with_timezone("UTC")),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), cols).unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

/// The rows of `q`, each column cast to text by the engine (`CAST(.. AS STRING)` is the caller's), cells
/// joined by spaces and rows by ` | `; NULL is empty.
fn rows(s: &Session, q: &str) -> String {
    let batches = futures::executor::block_on(async { s.sql(q).await?.collect().await })
        .unwrap_or_else(|e| panic!("{q}: {e}"));
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
    out.join(" | ")
}

const A: &str = "2020-01-01 00:00:00.000000001";
const B: &str = "2020-01-01 00:00:00.000000999";
const OLD: &str = "1960-01-01 00:00:00.123456789";

#[test]
fn values_print_with_every_nanosecond() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(c AS STRING), CAST(l AS STRING) FROM t ORDER BY k"
        ),
        format!("{A} {A} | {B} {B} | {A} {A} | {OLD} {OLD} |  ")
    );
}

#[test]
fn sort_group_distinct_and_set_operations_see_nanoseconds() {
    let s = session();
    let text = |q: &str| rows(&s, &q.replace("c ", "CAST(c AS STRING) "));
    assert_eq!(
        rows(&s, "SELECT CAST(c AS STRING) FROM t ORDER BY c"),
        format!(" | {OLD} | {A} | {A} | {B}")
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(c AS STRING), count(*) FROM t GROUP BY c ORDER BY c"
        ),
        format!(" 1 | {OLD} 1 | {A} 2 | {B} 1")
    );
    assert_eq!(rows(&s, "SELECT count(DISTINCT c) FROM t"), "3");
    assert_eq!(
        text("SELECT c FROM (SELECT c FROM t INTERSECT SELECT c FROM t WHERE k <> 1) ORDER BY 1"),
        format!(" | {OLD} | {A}")
    );
    assert_eq!(
        rows(
            &s,
            "SELECT a.k, b.k FROM t a JOIN t b ON a.c = b.c ORDER BY a.k, b.k"
        ),
        "0 0 | 0 2 | 1 1 | 2 0 | 2 2 | 3 3"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT k FROM t WHERE c IN (SELECT c FROM t WHERE k = 1) ORDER BY k"
        ),
        "1"
    );
}

#[test]
fn aggregates_and_windows_keep_nanoseconds() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(max(c) AS STRING), CAST(min(c) AS STRING), CAST(max(l) AS STRING) FROM t"
        ),
        format!("{B} {OLD} {B}")
    );
    assert_eq!(
        rows(&s, "SELECT k FROM t WHERE c = (SELECT max(c) FROM t)"),
        "1"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT max_by(k, c), min_by(k, c), CAST(min_by(c, k) AS STRING) FROM t"
        ),
        format!("1 3 {A}")
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(sort_array(collect_set(c)) AS STRING) FROM t"
        ),
        format!("[{OLD}, {A}, {B}]")
    );
    assert_eq!(
        rows(
            &s,
            "SELECT k, row_number() OVER (ORDER BY c) FROM t ORDER BY k"
        ),
        "0 3 | 1 5 | 2 4 | 3 2 | 4 1"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(greatest(c, c) AS STRING), CAST(least(l, l) AS STRING) FROM t WHERE k = 1"
        ),
        format!("{B} {B}")
    );
}

#[test]
fn nested_values_and_map_keys_keep_nanoseconds() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(map(c, k) AS STRING), CAST(named_struct('f', c) AS STRING) FROM t WHERE k < 2 ORDER BY k"
        ),
        format!("{{{A} -> 0}} {{{A}}} | {{{B} -> 1}} {{{B}}}")
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(element_at(array(c, c), 2) AS STRING) FROM t WHERE k = 3"
        ),
        OLD.to_owned()
    );
}

/// `hash` and `xxhash64` of a nanosecond timestamp chain its microseconds (a long) and the
/// nanoseconds within the microsecond (an int), as Spark's `TimestampNanosVal`.
#[test]
fn hashes_follow_timestamp_nanos_val() {
    let s = session();
    // The kernel test `hash::nanos_tests` pins these to the long-then-int chain.
    assert_eq!(
        rows(
            &s,
            "SELECT hash(c), xxhash64(c) FROM t WHERE k IN (0, 4) ORDER BY k"
        ),
        "1924051557 8613156613325252680 | 42 42"
    );
}

/// The functions that read a nanosecond timestamp's microseconds (Spark 4.3 goldens
/// `timestamp-ntz-nanos.sql`: `hour(.. 13:24:35.123456789)` is 13, `extract(SECOND ..)` is the
/// `decimal(11,9)` `35.123456789`, `date_trunc('SECOND', ..)` keeps the type, `unix_*` drop the rest: `unix_seconds` floors, `unix_timestamp`
/// divides toward zero, as for microseconds).
#[test]
fn datetime_functions_read_nanosecond_timestamps() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT hour(c), minute(c), second(c), hour(l) FROM t WHERE k = 3"
        ),
        "0 0 0 0"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(extract(SECOND FROM c) AS STRING), typeof(date_part('SECOND', l)) FROM t WHERE k IN (0, 3) ORDER BY k"
        ),
        "0.000000001 decimal(11,9) | 0.123456789 decimal(11,9)"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(date_trunc('SECOND', c) AS STRING), CAST(date_trunc('MICROSECOND', l) AS STRING), date_trunc('NANOSECOND', c) IS NULL FROM t WHERE k = 3"
        ),
        "1960-01-01 00:00:00 1960-01-01 00:00:00.123456 true"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT unix_seconds(l), unix_millis(l), unix_micros(l), unix_timestamp(c) FROM t WHERE k = 3"
        ),
        "-315619200 -315619199877 -315619199876544 -315619199"
    );
}

/// `ts ± interval` keeps the start's nanoseconds; `ts - ts` and `ts - date` are the difference of the
/// microseconds (Spark 4.3: `.123456789 - .000000111` is `1 00:00:00.123456000`).
#[test]
fn interval_arithmetic_keeps_or_drops_nanoseconds_as_spark() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(c + INTERVAL '2 00:03:00.000456' DAY TO SECOND AS STRING), CAST(c - INTERVAL '1 00:04:00.000321' DAY TO SECOND AS STRING) FROM t WHERE k = 3"
        ),
        "1960-01-03 00:03:00.123912789 1959-12-30 23:56:00.123135789"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(c + INTERVAL '1' MONTH AS STRING), CAST(INTERVAL '1' YEAR + c AS STRING), CAST(c - INTERVAL '1-2' YEAR TO MONTH AS STRING), CAST(l + make_interval(0, 1, 0, 2, 0, 0, 0) AS STRING) FROM t WHERE k = 3"
        ),
        "1960-02-01 00:00:00.123456789 1961-01-01 00:00:00.123456789 1958-11-01 00:00:00.123456789 1960-02-03 00:00:00.123456789"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(a.c - b.c AS STRING), CAST(b.l - a.l AS STRING), CAST(b.c - DATE '1960-01-01' AS STRING) FROM t a, t b WHERE a.k = 1 AND b.k = 3"
        ),
        "INTERVAL '21914 23:59:59.876544' DAY TO SECOND INTERVAL '-21914 23:59:59.876544' DAY TO SECOND INTERVAL '0 00:00:00.123456' DAY TO SECOND"
    );
}

/// The error class and message of `q`.
fn error(s: &Session, q: &str) -> String {
    let e = futures::executor::block_on(async { s.sql(q).await?.collect().await })
        .err()
        .unwrap_or_else(|| panic!("{q}: no error"));
    e.to_string()
}

/// Spark 4.3's `unix_nanos` (a `DECIMAL(21, 0)`) and `timestamp_nanos` (a `TIMESTAMP_LTZ(9)`, a decimal count
/// floored), values and refusals from a Spark 4.3.0 run (`timestamp-ltz-nanos.sql` and a probe).
#[test]
fn unix_nanos_and_timestamp_nanos_are_spark_43s() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT unix_nanos(c), unix_nanos(l), typeof(unix_nanos(c)) FROM t WHERE k IN (0, 3, 4) ORDER BY k"
        ),
        "1577836800000000001 1577836800000000001 decimal(21,0) | -315619199876543211 -315619199876543211 decimal(21,0) |   decimal(21,0)"
    );
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(timestamp_nanos(1230219000123456789) AS STRING), CAST(timestamp_nanos(-1) AS STRING), CAST(timestamp_nanos(1.9BD) AS STRING), CAST(timestamp_nanos(-1.1BD) AS STRING), CAST(timestamp_nanos(CAST(7 AS TINYINT)) AS STRING)"
        ),
        "2008-12-25 15:30:00.123456789 1969-12-31 23:59:59.999999999 1970-01-01 00:00:00.000000001 1969-12-31 23:59:59.999999998 1970-01-01 00:00:00.000000007"
    );
    // `typeof(timestamp_nanos(NULL))` is `timestamp_ltz(9)` in Spark; the type's printed name is the types
    // half of the row (`DataType::sql`), not checked here.
    assert_eq!(
        rows(
            &s,
            "SELECT CAST(timestamp_nanos(NULL) AS STRING), CAST(timestamp_nanos(CAST(NULL AS BIGINT)) AS STRING), unix_nanos(NULL), unix_nanos(timestamp_nanos(9223372036854775807))"
        ),
        "   9223372036854775807"
    );
    let overflow = error(&s, "SELECT timestamp_nanos(10000000000000000000000000BD)");
    assert!(
        overflow.contains("DATETIME_OVERFLOW")
            && overflow.contains(
                "create a TIMESTAMP_LTZ(9) from 10000000000000000000000000 nanoseconds since the epoch"
            ),
        "{overflow}"
    );
    let double = error(&s, "SELECT timestamp_nanos(1.0D)");
    assert!(
        double.contains("DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE")
            && double.contains(r#"requires the ("INTEGRAL" or "DECIMAL") type"#),
        "{double}"
    );
    for q in [
        "SELECT unix_nanos(TIMESTAMP '2020-01-01 00:00:00')",
        "SELECT unix_nanos('2020-01-01 00:00:00')",
        "SELECT unix_nanos(DATE '2020-01-01')",
    ] {
        let e = error(&s, q);
        assert!(
            e.contains("DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE")
                && e.contains(
                    r#"requires the "(TIMESTAMP_LTZ(P) OR TIMESTAMP_NTZ(P) WITH P IN [7, 9])" type"#
                ),
            "{q}: {e}"
        );
    }
}

/// `date_format` (and `to_char`/`to_varchar`, its aliases) prints the nanoseconds of a nanosecond timestamp
/// with up to nine `S`; fewer `S` cut them (Spark 4.3 `timestamp-ntz-nanos.sql`).
#[test]
fn date_format_prints_nanoseconds() {
    let s = session();
    assert_eq!(
        rows(
            &s,
            "SELECT date_format(c, 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS'), to_char(l, 'HH:mm:ss.SSSSSSSS'), to_varchar(c, 'ss.SSSSSSS'), date_format(c, 'SSS') FROM t WHERE k IN (1, 3, 4) ORDER BY k"
        ),
        "2020-01-01 00:00:00.000000999 00:00:00.00000099 00.0000009 000 | 1960-01-01 00:00:00.123456789 00:00:00.12345678 00.1234567 123 |    "
    );
    assert_eq!(
        rows(
            &s,
            "SELECT date_format(c, f) FROM (SELECT c, 'ss.SSSSSSSSS' AS f FROM t WHERE k = 3)"
        ),
        "00.123456789"
    );
}

/// Spark 4.3's `current_timestamp(p)` / `now(p)` / `localtimestamp(p)` for `p` in 7..=9 with
/// `spark.sql.timestampNanosTypes.enabled`: `CurrentTimestampNanos` / `LocalTimestampNanos`, named
/// `current_timestamp()` / `localtimestamp()` whatever the spelling, one value per query; 3 and 10 are
/// `INVALID_TIMESTAMP_PRECISION` (Spark 4.3.0 `timestamp-ltz-nanos.sql`, `timestamp-ntz-nanos.sql`).
#[test]
fn clocks_at_nanosecond_precision_are_spark_43s() {
    let s = session();
    s.conf()
        .set("spark.sql.timestampNanosTypes.enabled", "true")
        .unwrap();
    let q = "SELECT current_timestamp(9), now(8), localtimestamp(7 + 2), current_timestamp(6), now(6), \
             current_timestamp(9) = current_timestamp(9), now(9) = current_timestamp(9), \
             localtimestamp(9) = localtimestamp(9), \
             CAST(current_timestamp(9) AS TIMESTAMP) = current_timestamp()";
    let batches = futures::executor::block_on(async { s.sql(q).await?.collect().await }).unwrap();
    let schema = batches[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        [
            "current_timestamp()",
            "current_timestamp()",
            "localtimestamp()",
            "current_timestamp()",
            "now()",
            "(current_timestamp() = current_timestamp())",
            "(current_timestamp() = current_timestamp())",
            "(localtimestamp() = localtimestamp())",
            "(CAST(current_timestamp() AS TIMESTAMP) = current_timestamp())",
        ]
    );
    let utc = Some(Arc::from("UTC"));
    assert_eq!(
        schema.field(0).data_type(),
        &DataType::Timestamp(TimeUnit::Nanosecond, utc.clone())
    );
    assert_eq!(
        schema.field(2).data_type(),
        &DataType::Timestamp(TimeUnit::Nanosecond, None)
    );
    assert_eq!(
        schema.field(3).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, utc)
    );
    assert!(!schema.field(0).is_nullable());
    assert_eq!(
        rows(
            &s,
            "SELECT current_timestamp(9) = current_timestamp(9), now(9) = current_timestamp(9), \
             localtimestamp(9) = localtimestamp(9), \
             CAST(current_timestamp(9) AS TIMESTAMP) = current_timestamp()"
        ),
        "true true true true"
    );
    for (q, p, ty) in [
        ("SELECT current_timestamp(3)", "3", "TIMESTAMP_LTZ"),
        ("SELECT now(10)", "10", "TIMESTAMP_LTZ"),
        ("SELECT localtimestamp(3)", "3", "TIMESTAMP_NTZ"),
        ("SELECT localtimestamp(10)", "10", "TIMESTAMP_NTZ"),
    ] {
        let e = error(&s, q);
        assert!(e.contains("INVALID_TIMESTAMP_PRECISION"), "{q}: {e}");
        assert!(e.contains(p) && e.contains(ty), "{q}: {e}");
    }
}

#[test]
fn a_sort_and_an_aggregate_that_spill_keep_nanoseconds() {
    // 300,000 rows of 60,000 values a nanosecond apart under a 6 MiB limit: the aggregate and the sort
    // spill (Arrow IPC spill files of `Timestamp(ns)` columns) and read every nanosecond back.
    const ROWS: i64 = 300_000;
    const KEYS: i64 = 60_000;
    const BASE: i64 = 1_577_836_800_000_000_000;
    let dir = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config(
            "krishiv.spill.dir",
            dir.path().join("spill").to_str().unwrap(),
        )
        .config("krishiv.memory.limit", "6m")
        .config("spark.sql.session.timeZone", "UTC")
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "c",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
    )]));
    let values: Vec<i64> = (0..ROWS).map(|i| BASE + (i * 7_919) % KEYS).collect();
    let batches = values
        .chunks(8192)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(TimestampNanosecondArray::from(c.to_vec())) as ArrayRef],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("f", schema, batches).unwrap();
    let q = "SELECT /* nanos_spill */ CAST(c AS STRING), count(*) FROM f GROUP BY c ORDER BY c";
    let got = rows(&s, q);
    let lines: Vec<&str> = got.split(" | ").collect();
    assert_eq!(lines.len(), usize::try_from(KEYS).unwrap());
    assert_eq!(lines[0], "2020-01-01 00:00:00 5");
    assert_eq!(lines[1], "2020-01-01 00:00:00.000000001 5");
    assert_eq!(lines[59_999], "2020-01-01 00:00:00.000059999 5");
    let spilled = rows(
        &s,
        "SELECT spill_bytes > 0 FROM system.queries WHERE query_text LIKE '%nanos_spill%' \
         AND query_text NOT LIKE '%system.queries%'",
    );
    assert_eq!(spilled, "true");
}
