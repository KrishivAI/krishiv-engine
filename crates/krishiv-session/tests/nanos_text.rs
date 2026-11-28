// SPDX-License-Identifier: Apache-2.0
//! Nanosecond timestamps (Spark 4.3 `TIMESTAMP_NTZ(p)` / `TIMESTAMP_LTZ(p)`, `p` 7..=9) in JSON and CSV
//! files. Every expected text and value is Spark 4.3.0's output for the same statement
//! (`spark.sql.timestampNanosTypes.enabled=true`, session zone UTC unless named), from the
//! `JacksonGenerator` / `UnivocityGenerator` `formatNanos` and `JacksonParser` / `UnivocityParser`
//! `parseNanos` paths: a write floors the value to its precision and prints as many fraction digits as
//! the pattern has (the default patterns print milliseconds); a read keeps up to `p` digits, with no
//! fallback to the default grammar once a pattern fails, and JSON takes only a string.
#![expect(clippy::unwrap_used, clippy::expect_used, reason = "a test")]
use std::path::Path;

use krishiv_session::Session;

fn session(zone: &str) -> Session {
    let s = Session::new().unwrap();
    s.conf().set("spark.sql.session.timeZone", zone).unwrap();
    s.conf()
        .set("spark.sql.timestampNanosTypes.enabled", "true")
        .unwrap();
    s
}

/// The rows of `q` (cells joined by `|`, rows by ` ; `), or `ERR <message>`.
fn run(s: &Session, q: &str) -> String {
    match futures::executor::block_on(async { s.sql(q).await?.collect().await }) {
        Err(e) => format!("ERR {e}"),
        Ok(batches) => {
            let mut out = Vec::new();
            for b in &batches {
                for r in 0..b.num_rows() {
                    let cells: Vec<String> = b
                        .columns()
                        .iter()
                        .map(|c| arrow_cast::display::array_value_to_string(c, r).expect("display"))
                        .collect();
                    out.push(cells.join("|"));
                }
            }
            out.join(" ; ")
        }
    }
}

/// The lines of the data files in `dir`, in file-name order.
fn lines(dir: &Path) -> Vec<String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("part"))
        .collect();
    files.sort();
    files
        .iter()
        .flat_map(|f| {
            std::fs::read_to_string(f)
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Three rows: nine digits, a value just below and above a second before the epoch with a NULL, and
/// values whose digits past the microsecond are zero or only the seventh.
const ROWS: &str = "SELECT * FROM VALUES \
    (CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)), \
     CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9)), \
     CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_LTZ(7))), \
    (CAST('1969-12-31 23:59:59.999999999' AS TIMESTAMP_NTZ(9)), \
     CAST('1969-12-31 23:59:59.000000001' AS TIMESTAMP_LTZ(9)), NULL), \
    (CAST('2020-01-01 00:00:00' AS TIMESTAMP_NTZ(9)), \
     CAST('2020-01-01 00:00:00.5' AS TIMESTAMP_LTZ(9)), \
     CAST('2020-01-01 00:00:00.0000001' AS TIMESTAMP_LTZ(7))) AS t(n9, l9, l7)";

const NINE: &str = "'timestampFormat' 'yyyy-MM-dd HH:mm:ss.SSSSSSSSSXXX', \
                    'timestampNTZFormat' 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS'";
const SIX: &str = "'timestampFormat' 'yyyy-MM-dd HH:mm:ss.SSSSSS', \
                   'timestampNTZFormat' 'yyyy-MM-dd HH:mm:ss.SSSSSS'";
const OPTIONAL: &str = "'timestampFormat' 'yyyy-MM-dd''T''HH:mm:ss[.SSSSSSSSS]', \
                        'timestampNTZFormat' 'yyyy-MM-dd''T''HH:mm:ss[.SSSSSSSSS]'";

/// Writes [`ROWS`] as `format` with `options` (none when empty) into `dir/name`; the files' lines.
fn write(s: &Session, dir: &Path, name: &str, format: &str, options: &str) -> Vec<String> {
    let path = dir.join(name);
    let opts = if options.is_empty() {
        String::new()
    } else {
        format!("OPTIONS ({options})")
    };
    let q = format!(
        "INSERT OVERWRITE DIRECTORY '{}' USING {format} {opts} {ROWS}",
        path.display()
    );
    assert_eq!(run(s, &q), "", "{q}");
    lines(&path)
}

#[test]
fn json_writes_follow_the_timestamp_patterns() {
    let s = session("UTC");
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        write(&s, dir.path(), "default", "json", ""),
        [
            r#"{"n9":"2020-01-01T00:00:00.123","l9":"2020-01-01T00:00:00.123Z","l7":"2020-01-01T00:00:00.123Z"}"#,
            r#"{"n9":"1969-12-31T23:59:59.999","l9":"1969-12-31T23:59:59.000Z"}"#,
            r#"{"n9":"2020-01-01T00:00:00.000","l9":"2020-01-01T00:00:00.500Z","l7":"2020-01-01T00:00:00.000Z"}"#,
        ]
    );
    assert_eq!(
        write(&s, dir.path(), "nine", "json", NINE),
        [
            r#"{"n9":"2020-01-01 00:00:00.123456789","l9":"2020-01-01 00:00:00.123456789Z","l7":"2020-01-01 00:00:00.123456700Z"}"#,
            r#"{"n9":"1969-12-31 23:59:59.999999999","l9":"1969-12-31 23:59:59.000000001Z"}"#,
            r#"{"n9":"2020-01-01 00:00:00.000000000","l9":"2020-01-01 00:00:00.500000000Z","l7":"2020-01-01 00:00:00.000000100Z"}"#,
        ]
    );
    assert_eq!(
        write(&s, dir.path(), "six", "json", SIX),
        [
            r#"{"n9":"2020-01-01 00:00:00.123456","l9":"2020-01-01 00:00:00.123456","l7":"2020-01-01 00:00:00.123456"}"#,
            r#"{"n9":"1969-12-31 23:59:59.999999","l9":"1969-12-31 23:59:59.000000"}"#,
            r#"{"n9":"2020-01-01 00:00:00.000000","l9":"2020-01-01 00:00:00.500000","l7":"2020-01-01 00:00:00.000000"}"#,
        ]
    );
    assert_eq!(
        write(&s, dir.path(), "optional", "json", OPTIONAL),
        [
            r#"{"n9":"2020-01-01T00:00:00.123456789","l9":"2020-01-01T00:00:00.123456789","l7":"2020-01-01T00:00:00.123456700"}"#,
            r#"{"n9":"1969-12-31T23:59:59.999999999","l9":"1969-12-31T23:59:59.000000001"}"#,
            r#"{"n9":"2020-01-01T00:00:00.000000000","l9":"2020-01-01T00:00:00.500000000","l7":"2020-01-01T00:00:00.000000100"}"#,
        ]
    );
}

#[test]
fn json_writes_an_instant_in_the_session_zone() {
    let s = session("America/Los_Angeles");
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        write(&s, dir.path(), "la", "json", ""),
        [
            r#"{"n9":"2020-01-01T00:00:00.123","l9":"2020-01-01T00:00:00.123-08:00","l7":"2020-01-01T00:00:00.123-08:00"}"#,
            r#"{"n9":"1969-12-31T23:59:59.999","l9":"1969-12-31T23:59:59.000-08:00"}"#,
            r#"{"n9":"2020-01-01T00:00:00.000","l9":"2020-01-01T00:00:00.500-08:00","l7":"2020-01-01T00:00:00.000-08:00"}"#,
        ]
    );
}

#[test]
fn csv_writes_follow_the_timestamp_patterns() {
    let s = session("UTC");
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        write(&s, dir.path(), "default", "csv", ""),
        [
            "2020-01-01T00:00:00.123,2020-01-01T00:00:00.123Z,2020-01-01T00:00:00.123Z",
            "1969-12-31T23:59:59.999,1969-12-31T23:59:59.000Z,",
            "2020-01-01T00:00:00.000,2020-01-01T00:00:00.500Z,2020-01-01T00:00:00.000Z",
        ]
    );
    assert_eq!(
        write(&s, dir.path(), "nine", "csv", NINE),
        [
            "2020-01-01 00:00:00.123456789,2020-01-01 00:00:00.123456789Z,2020-01-01 00:00:00.123456700Z",
            "1969-12-31 23:59:59.999999999,1969-12-31 23:59:59.000000001Z,",
            "2020-01-01 00:00:00.000000000,2020-01-01 00:00:00.500000000Z,2020-01-01 00:00:00.000000100Z",
        ]
    );
    assert_eq!(
        write(&s, dir.path(), "six", "csv", SIX),
        [
            "2020-01-01 00:00:00.123456,2020-01-01 00:00:00.123456,2020-01-01 00:00:00.123456",
            "1969-12-31 23:59:59.999999,1969-12-31 23:59:59.000000,",
            "2020-01-01 00:00:00.000000,2020-01-01 00:00:00.500000,2020-01-01 00:00:00.000000",
        ]
    );
    assert_eq!(
        write(&s, dir.path(), "optional", "csv", OPTIONAL),
        [
            "2020-01-01T00:00:00.123456789,2020-01-01T00:00:00.123456789,2020-01-01T00:00:00.123456700",
            "1969-12-31T23:59:59.999999999,1969-12-31T23:59:59.000000001,",
            "2020-01-01T00:00:00.000000000,2020-01-01T00:00:00.500000000,2020-01-01T00:00:00.000000100",
        ]
    );
}

/// `CAST(c AS STRING)` of the three columns read back from `dir` as `format` with `options` (a NULL
/// shows empty; Spark shows `null`).
fn read_back(s: &Session, dir: &Path, format: &str, options: &str) -> String {
    let opts = if options.is_empty() {
        String::new()
    } else {
        format!(", {options}")
    };
    let view = format!(
        "CREATE OR REPLACE TEMPORARY VIEW v (n9 TIMESTAMP_NTZ(9), l9 TIMESTAMP_LTZ(9), l7 TIMESTAMP_LTZ(7)) \
         USING {format} OPTIONS (path '{}'{opts})",
        dir.display()
    );
    assert_eq!(run(s, &view), "", "{view}");
    run(
        s,
        "SELECT typeof(n9), typeof(l7), CAST(n9 AS STRING), CAST(l9 AS STRING), CAST(l7 AS STRING) FROM v",
    )
}

#[test]
fn reads_keep_the_digits_the_text_has_up_to_the_precision() {
    let s = session("UTC");
    let dir = tempfile::tempdir().unwrap();
    for format in ["json", "csv"] {
        write(&s, dir.path(), &format!("{format}-default"), format, "");
        write(&s, dir.path(), &format!("{format}-nine"), format, NINE);
        assert_eq!(
            read_back(
                &s,
                &dir.path().join(format!("{format}-default")),
                format,
                ""
            ),
            "timestamp_ntz(9)|timestamp_ltz(7)|2020-01-01 00:00:00.123|2020-01-01 00:00:00.123|2020-01-01 00:00:00.123 ; \
             timestamp_ntz(9)|timestamp_ltz(7)|1969-12-31 23:59:59.999|1969-12-31 23:59:59| ; \
             timestamp_ntz(9)|timestamp_ltz(7)|2020-01-01 00:00:00|2020-01-01 00:00:00.5|2020-01-01 00:00:00",
            "{format}"
        );
        assert_eq!(
            read_back(&s, &dir.path().join(format!("{format}-nine")), format, NINE),
            "timestamp_ntz(9)|timestamp_ltz(7)|2020-01-01 00:00:00.123456789|2020-01-01 00:00:00.123456789|2020-01-01 00:00:00.1234567 ; \
             timestamp_ntz(9)|timestamp_ltz(7)|1969-12-31 23:59:59.999999999|1969-12-31 23:59:59.000000001| ; \
             timestamp_ntz(9)|timestamp_ltz(7)|2020-01-01 00:00:00|2020-01-01 00:00:00.5|2020-01-01 00:00:00.0000001",
            "{format}"
        );
    }
}

#[test]
fn the_json_and_csv_functions_follow_the_same_rules() {
    let s = session("UTC");
    assert_eq!(
        run(
            &s,
            "SELECT to_json(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)))), \
             to_csv(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9))))"
        ),
        r#"{"a":"2020-01-01T00:00:00.123"}|2020-01-01T00:00:00.123Z"#
    );
    assert_eq!(
        run(
            &s,
            "SELECT to_json(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))), \
             map('timestampNTZFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS'))"
        ),
        r#"{"a":"2020-01-01 00:00:00.123456789"}"#
    );
    assert_eq!(
        run(
            &s,
            r#"SELECT CAST(from_json('{"a":"2020-01-01T00:00:00.123456789"}', 'a timestamp_ntz(9)').a AS STRING)"#
        ),
        "2020-01-01 00:00:00.123456789"
    );
}

#[test]
fn function_reads_floor_to_the_precision_and_take_only_matching_strings() {
    let s = session("UTC");
    assert_eq!(
        run(
            &s,
            r#"SELECT CAST(from_json('{"a":"2020-01-01T00:00:00.123456789"}', 'a timestamp_ntz(7)').a AS STRING), CAST(from_json('{"a":"2020-01-01T00:00:00.123456789Z"}', 'a timestamp_ltz(8)').a AS STRING)"#
        ),
        "2020-01-01 00:00:00.1234567|2020-01-01 00:00:00.12345678"
    );
    // A number is no nanosecond timestamp, and a pattern that does not match has no fallback: the
    // field is NULL, the struct is not.
    for q in [
        r#"from_json('{"a":1}', 'a timestamp_ntz(9)')"#,
        r#"from_json('{"a":1}', 'a timestamp_ltz(9)')"#,
        r#"from_json('{"a":"2020-01-01T00:00:00"}', 'a timestamp_ltz(9)', map('timestampFormat', 'yyyy-MM-dd HH:mm:ss'))"#,
    ] {
        assert_eq!(
            run(&s, &format!("SELECT {q} IS NULL, {q}.a IS NULL")),
            "false|true",
            "{q}"
        );
    }
    assert_eq!(
        run(
            &s,
            "SELECT CAST(from_csv('2020-01-01 00:00:00.123456789,2020-01-01T00:00:00.987654321Z', \
             'a timestamp_ntz(9), b timestamp_ltz(7)') AS STRING)"
        ),
        "{2020-01-01 00:00:00.123456789, 2020-01-01 00:00:00.9876543}"
    );
    assert_eq!(
        run(
            &s,
            "SELECT to_csv(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))), \
             map('timestampNTZFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSSSSS')), \
             to_json(named_struct('a', array(CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9)))), \
             map('timestampFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSSSSSXXX'))"
        ),
        r#"2020-01-01 00:00:00.123456789|{"a":["2020-01-01 00:00:00.123456789Z"]}"#
    );
}

/// `spark.sql.legacy.timeParserPolicy=LEGACY` (Spark 4.3.0 run): the legacy formatter has no nanosecond
/// methods, so a `TIMESTAMP_LTZ(p)` read is a bad record (a NULL field under PERMISSIVE, the rest of the
/// record kept; FAILFAST raises) and a write raises `TIMESTAMP_NANOS_WITH_LEGACY_TIME_PARSER`.
/// `TIMESTAMP_NTZ(p)` keeps the `java.time` formatter in JSON and CSV; `date_format` of either raises.
#[test]
fn the_legacy_time_parser_refuses_nanosecond_ltz_values() {
    let s = session("UTC");
    s.conf()
        .set("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .unwrap();
    let text = "'{\"a\":\"2020-01-01 00:00:00.123456789\"}'";
    for (q, want) in [
        (
            format!("SELECT cast(from_json({text}, 'a timestamp_ltz(9)', map('timestampFormat','yyyy-MM-dd HH:mm:ss.SSSSSSSSS')).a as string)"),
            "",
        ),
        (format!("SELECT cast(from_json({text}, 'a timestamp_ltz(9)').a as string)"), ""),
        (
            "SELECT to_json(from_json('{\"a\":\"2020-01-01 00:00:00.1\", \"b\": 1}', 'a timestamp_ltz(9), b int'))".to_owned(),
            "{\"b\":1}",
        ),
        (format!("SELECT cast(from_json({text}, 'a timestamp_ntz(9)').a as string)"), "2020-01-01 00:00:00.123456789"),
        ("SELECT cast(from_csv('2020-01-01 00:00:00.123456789', 'a timestamp_ltz(9)').a as string)".to_owned(), ""),
        (
            "SELECT to_json(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))))".to_owned(),
            "{\"a\":\"2020-01-01T00:00:00.123\"}",
        ),
        ("SELECT cast(CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9)) as string)".to_owned(), "2020-01-01 00:00:00.123456789"),
    ] {
        assert_eq!(run(&s, &q), want, "{q}");
    }
    let failfast = run(
        &s,
        &format!("SELECT from_json({text}, 'a timestamp_ltz(9)', map('mode','FAILFAST'))"),
    );
    assert!(
        failfast.contains("MALFORMED_RECORD_IN_PARSING"),
        "{failfast}"
    );
    for q in [
        "SELECT to_json(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9))))",
        "SELECT to_csv(named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9))))",
        "SELECT date_format(CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9)), 'yyyy-MM-dd HH:mm:ss.SSS')",
        "SELECT date_format(CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)), 'yyyy-MM-dd HH:mm:ss.SSS')",
    ] {
        let got = run(&s, q);
        assert!(
            got.contains("UNSUPPORTED_FEATURE.TIMESTAMP_NANOS_WITH_LEGACY_TIME_PARSER"),
            "{q}: {got}"
        );
    }
    // The microsecond types keep the legacy formatter.
    let micros = run(
        &s,
        "SELECT to_json(named_struct('a', CAST('2020-01-01 00:00:00.123456' AS TIMESTAMP)))",
    );
    assert!(
        micros.starts_with("{\"a\":\"2020-01-01T00:00:00.123"),
        "{micros}"
    );
}

/// A bare time of day (Spark 4.3.0 run, `parseTimestampToInstant`): today's date in the zone, with all
/// nine fraction digits, through the default grammar of `from_json`, `from_csv`, a cast and a JSON file.
#[test]
fn a_bare_time_of_day_keeps_nine_digits() {
    let s = session("UTC");
    let today = run(&s, "SELECT cast(current_date() as string)");
    let want = format!("{today} 12:34:56.123456789");
    for q in [
        "SELECT cast(from_json('{\"a\":\"T12:34:56.123456789\"}', 'a timestamp_ltz(9)').a as string)",
        "SELECT cast(from_json('{\"a\":\"12:34:56.123456789\"}', 'a timestamp_ltz(9)').a as string)",
        "SELECT cast(from_csv('T12:34:56.123456789', 'a timestamp_ltz(9)').a as string)",
        "SELECT cast(cast('T12:34:56.123456789' as timestamp_ltz(9)) as string)",
    ] {
        assert_eq!(run(&s, q), want, "{q}");
    }
}

/// The file readers' default grammar reads a trailing zone as it is (Spark 4.3.0: `parseNanos` has no
/// legacy clean-up of `GMT`; session zone `America/Los_Angeles`): ` GMT` is UTC, `Etc/GMT+1` an hour behind.
#[test]
fn file_reads_take_a_trailing_zone_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("a.json");
    std::fs::write(
        &json,
        "{\"a\":\"2020-01-01 10:00:00.123456789 GMT\",\"c\":\"2020-01-01 10:00:00.123456789 Etc/GMT+1\"}\n",
    )
    .unwrap();
    let csv = dir.path().join("a.csv");
    std::fs::write(&csv, "2020-01-01 10:00:00.123456789 GMT\n").unwrap();
    let s = session("America/Los_Angeles");
    for (view, format, path, cols) in [
        ("j", "json", &json, "a TIMESTAMP_LTZ(9), c TIMESTAMP_LTZ(9)"),
        ("c", "csv", &csv, "a TIMESTAMP_LTZ(9)"),
    ] {
        let q = format!(
            "CREATE OR REPLACE TEMPORARY VIEW {view} ({cols}) USING {format} OPTIONS (path '{}')",
            path.display()
        );
        assert_eq!(run(&s, &q), "", "{q}");
    }
    assert_eq!(
        run(&s, "SELECT CAST(a AS STRING), CAST(c AS STRING) FROM j"),
        "2020-01-01 02:00:00.123456789|2020-01-01 03:00:00.123456789"
    );
    assert_eq!(
        run(&s, "SELECT CAST(a AS STRING) FROM c"),
        "2020-01-01 02:00:00.123456789"
    );
    assert_eq!(
        run(
            &s,
            "SELECT CAST(from_json('{\"a\":\"2020-01-01 10:00:00.123456789 GMT\"}', 'a timestamp_ltz(9)').a AS STRING)"
        ),
        "2020-01-01 02:00:00.123456789"
    );
}
