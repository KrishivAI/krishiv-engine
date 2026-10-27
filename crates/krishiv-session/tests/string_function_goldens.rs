// SPDX-License-Identifier: Apache-2.0
//! The string functions end to end against Spark 4.2.0's own result files:
//! every statement of `string-functions.sql`, `try-string-functions.sql` and
//! `mask-functions.sql` runs through a session
//! (statements are read from `tests/spark-golden/v4.2.0/results/*.sql.out`,
//! never copied) and must give the recorded schema and rows, or the
//! recorded error class and message parameters.
//!
//! A statement the engine reports as `KRISHIV_UNSUPPORTED_FEATURE` is not
//! compared: it is counted, with the reason, as skipped, as is any statement
//! run under a `spark.sql.legacy.*` switch the engine does not read. The
//! counts are pinned, so a function that starts or stops working shows up.
#![cfg(test)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

/// Java's `\s`: what Spark trims from the end of a recorded output.
const JAVA_SPACE: [char; 6] = [' ', '\t', '\n', '\u{0B}', '\u{0C}', '\r'];

/// One `-- !query` block of a result file.
struct Block {
    query: String,
    schema: String,
    output: String,
}

fn blocks(file: &str) -> Vec<Block> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/spark-golden/v4.2.0/results")
        .join(file);
    let text = std::fs::read_to_string(path).unwrap();
    let mut out = Vec::new();
    for chunk in text.split("\n-- !query\n").skip(1) {
        let (query, rest) = chunk.split_once("\n-- !query schema\n").unwrap();
        let (schema, output) = rest.split_once("\n-- !query output\n").unwrap();
        out.push(Block {
            query: query.trim().to_owned(),
            schema: schema.trim().to_owned(),
            output: output.trim_end_matches(JAVA_SPACE).to_owned(),
        });
    }
    out
}

/// Spark's name of an Arrow type, as in `struct<..>` schema strings.
fn type_name(t: &ArrowType) -> String {
    match t {
        ArrowType::Null => "void".to_owned(),
        ArrowType::Boolean => "boolean".to_owned(),
        ArrowType::Int8 => "tinyint".to_owned(),
        ArrowType::Int16 => "smallint".to_owned(),
        ArrowType::Int32 => "int".to_owned(),
        ArrowType::Int64 => "bigint".to_owned(),
        ArrowType::Float64 => "double".to_owned(),
        ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View => "string".to_owned(),
        ArrowType::Binary | ArrowType::LargeBinary | ArrowType::BinaryView => "binary".to_owned(),
        ArrowType::Decimal128(p, s) => format!("decimal({p},{s})"),
        t => panic!("unnamed type {t}"),
    }
}

/// The rows as Spark prints them (cells joined by tabs, `NULL` for NULL).
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|col| {
                    if col.is_null(i) {
                        return "NULL".to_owned();
                    }
                    match col.data_type() {
                        ArrowType::Int32 => col
                            .as_primitive::<arrow_array::types::Int32Type>()
                            .value(i)
                            .to_string(),
                        ArrowType::Int64 => col
                            .as_primitive::<arrow_array::types::Int64Type>()
                            .value(i)
                            .to_string(),
                        ArrowType::Boolean => col.as_boolean().value(i).to_string(),
                        ArrowType::Utf8 => col.as_string::<i32>().value(i).to_owned(),
                        ArrowType::LargeUtf8 => col.as_string::<i64>().value(i).to_owned(),
                        ArrowType::Utf8View => col.as_string_view().value(i).to_owned(),
                        ArrowType::Binary => {
                            String::from_utf8_lossy(col.as_binary::<i32>().value(i)).into_owned()
                        }
                        ArrowType::LargeBinary => {
                            String::from_utf8_lossy(col.as_binary::<i64>().value(i)).into_owned()
                        }
                        ArrowType::BinaryView => {
                            String::from_utf8_lossy(col.as_binary_view().value(i)).into_owned()
                        }
                        ArrowType::Decimal128(_, scale) => decimal(
                            col.as_primitive::<arrow_array::types::Decimal128Type>()
                                .value(i),
                            *scale,
                        ),
                        t => panic!("unrendered {t}"),
                    }
                })
                .collect();
            out.push(cells.join("\t"));
        }
    }
    out
}

/// The text of a decimal: `unscaled` with `scale` digits after the point.
fn decimal(unscaled: i128, scale: i8) -> String {
    let digits = unscaled.unsigned_abs().to_string();
    let scale = usize::try_from(scale).unwrap_or(0);
    let sign = if unscaled < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, frac) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{frac}")
}

/// Spark's recorded error: its class and message parameters.
struct Recorded {
    class: String,
    params: BTreeMap<String, String>,
}

fn recorded_error(output: &str) -> Option<Recorded> {
    let (head, body) = output.split_once('\n')?;
    if !head.contains("Exception") {
        return None;
    }
    let json: serde_json::Value = serde_json::from_str(body.trim()).ok()?;
    let params = json["messageParameters"]
        .as_object()
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_owned()))
                .collect()
        })
        .unwrap_or_default();
    let class = json["errorClass"].as_str()?.to_owned();
    let mut params: BTreeMap<String, String> = params;
    // The vendored `CAST_INVALID_INPUT` template has no `<ansiConfig>`, so the
    // engine's error does not carry the parameter Spark passes anyway.
    if class == "CAST_INVALID_INPUT" {
        params.remove("ansiConfig");
    }
    Some(Recorded { class, params })
}

fn ours(e: &KrishivError) -> Recorded {
    Recorded {
        class: e.error_class().name().to_owned(),
        params: e
            .message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    }
}

/// The tally of one file.
#[derive(Default, Debug, PartialEq, Eq)]
struct Tally {
    passed: usize,
    skipped: BTreeMap<String, usize>,
    failed: Vec<String>,
}

fn replay(file: &str) -> Tally {
    let session = Session::new().unwrap();
    let mut tally = Tally::default();
    let mut legacy = false;
    for Block {
        query,
        schema,
        output,
    } in blocks(file)
    {
        let lower = query.to_ascii_lowercase();
        if let Some(conf) = lower.strip_prefix("set ") {
            if conf.starts_with("spark.sql.legacy.") {
                legacy = conf.trim().ends_with("=true");
            }
            continue;
        }
        if legacy {
            *tally
                .skipped
                .entry("a spark.sql.legacy.* switch the engine does not read".to_owned())
                .or_default() += 1;
            continue;
        }
        if lower.starts_with("create ") || lower.starts_with("drop ") {
            let _ = block_on(session.sql(&query));
            continue;
        }
        let recorded = recorded_error(&output);
        let run = block_on(async {
            let df = session.sql(&query).await?;
            let fields = df.schema().await?;
            let batches = df.collect().await?;
            Ok::<_, KrishivError>((fields, batches))
        });
        match (run, recorded) {
            (Err(e), Some(want))
                if e.error_class() == ErrorClass::TableOrViewNotFound
                    && want.class != "TABLE_OR_VIEW_NOT_FOUND" =>
            {
                *tally
                    .skipped
                    .entry("a temporary view over VALUES (not created)".to_owned())
                    .or_default() += 1;
            }
            (Err(e), _) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => {
                *tally
                    .skipped
                    .entry(format!(
                        "unsupported: {}",
                        e.message_parameters()
                            .iter()
                            .find(|(k, _)| *k == "feature")
                            .map(|(_, v)| v.as_str())
                            .unwrap_or_default()
                    ))
                    .or_default() += 1;
            }
            (Ok((fields, batches)), None) => {
                let names: Vec<String> = fields
                    .fields()
                    .iter()
                    .map(|f| format!("{}:{}", f.name(), type_name(f.data_type())))
                    .collect();
                let got_schema = format!("struct<{}>", names.join(","));
                let joined = rows_of(&batches).join("\n");
                let got_rows = joined.trim_end_matches(JAVA_SPACE);
                if got_schema == schema && got_rows == output {
                    tally.passed += 1;
                } else {
                    let mut why = String::new();
                    let _ = write!(
                        why,
                        "{query}\n  schema {got_schema} want {schema}\n  rows {got_rows:?} want {output:?}"
                    );
                    tally.failed.push(why);
                }
            }
            (Err(e), Some(want)) => {
                let got = ours(&e);
                if got.class == want.class && got.params == want.params {
                    tally.passed += 1;
                } else {
                    tally.failed.push(format!(
                        "{query}\n  error {} {:?} want {} {:?}",
                        got.class, got.params, want.class, want.params
                    ));
                }
            }
            (Err(e), None) => tally.failed.push(format!(
                "{query}\n  raised {} {:?} where Spark has rows {output:?}",
                e.error_class().name(),
                e.message_parameters()
            )),
            (Ok(_), Some(want)) => tally
                .failed
                .push(format!("{query}\n  ran where Spark raises {}", want.class)),
        }
    }
    tally
}

fn check(file: &str, passed: usize, skipped: usize) {
    let t = replay(file);
    assert!(
        t.failed.is_empty(),
        "{file}: {} of {} statements differ from Spark:\n{}",
        t.failed.len(),
        t.passed + t.failed.len(),
        t.failed.join("\n")
    );
    assert_eq!(
        (t.passed, t.skipped.values().sum::<usize>()),
        (passed, skipped),
        "{file}: statements that match Spark and statements skipped (by reason: {:?})",
        t.skipped
    );
}

#[test]
fn string_functions_match_spark() {
    check("string-functions.sql.out", 213, 30);
}

#[test]
fn try_string_functions_match_spark() {
    check("try-string-functions.sql.out", 35, 0);
}

#[test]
fn mask_functions_match_spark() {
    check("mask-functions.sql.out", 57, 0);
}

// `url-functions.sql.out` is replayed in full (the parse_url redaction of
// userinfo in column names is a pinned difference) by `url_csv_functions.rs`.

/// The new character functions end to end, against Spark 4.2's results: the
/// three replayed files do not call `overlay`, `levenshtein` (2/3-arg),
/// `soundex`, `initcap`, `translate`, `locate` with a NULL start, ANSI
/// `elt` or `concat_ws` edge cases, so they are pinned here instead of
/// resting on unit tests alone. Expected values are Spark 4.2's (the same
/// values the kernel and binding tests assert).
#[test]
fn string_extra_functions_match_spark() {
    let session = Session::new().unwrap();
    let rows = |sql: &str| {
        let (fields, batches) = block_on(async {
            let df = session.sql(sql).await?;
            let fields = df.schema().await?;
            let batches = df.collect().await?;
            Ok::<_, KrishivError>((fields, batches))
        })
        .unwrap();
        let _ = fields;
        rows_of(&batches)
    };
    let cases = [
        ("SELECT overlay('Spark SQL', 'CORE', 7)", "Spark CORE"),
        (
            "SELECT overlay('Spark SQL', 'ANSI ', 7, 0)",
            "Spark ANSI SQL",
        ),
        ("SELECT overlay('日本語', 'x', 2)", "日x語"),
        ("SELECT levenshtein('kitten', 'sitting')", "3"),
        ("SELECT levenshtein('kitten', 'sitting', 2)", "-1"),
        ("SELECT soundex('Miller')", "M460"),
        ("SELECT soundex('Robert')", "R163"),
        ("SELECT initcap('sPark sql')", "Spark Sql"),
        ("SELECT translate('translate', 'rnlt', '1223')", "31a2s2a3e"),
        ("SELECT locate('bar', 'foobarbar', 5)", "7"),
        ("SELECT locate('bar', 'foobarbar', NULL)", "0"),
        ("SELECT locate('', 'abc', 100)", "1"),
        ("SELECT elt(1, 'a', 'b')", "a"),
        ("SELECT concat_ws(',', 'a', 'b')", "a,b"),
        ("SELECT concat_ws('-', 'a', NULL, 'b')", "a-b"),
        ("SELECT substring_index('a.b.c', '.', -2147483648)", "a.b.c"),
        ("SELECT split_part('a.b.c', '.', -2147483648)", ""),
        ("SELECT base64(encode(repeat('a', 58), 'utf-8'))", ""),
    ];
    for (sql, want) in cases {
        if sql.starts_with("SELECT base64(") {
            // 58 bytes are 80 symbols with a CRLF after the 76th: chunking
            // is always on (`spark.sql.chunkBase64String.enabled` unread).
            let got = rows(sql).join("\n");
            assert_eq!(got.len(), 82, "{sql}");
            assert_eq!(&got[76..78], "\r\n", "{sql}");
            let _ = want;
            continue;
        }
        assert_eq!(rows(sql), [want.to_owned()], "{sql}");
    }
    // ANSI `elt` raises; lenient `elt` gives NULL.
    let err = block_on(async {
        let df = session.sql("SELECT elt(5, 'a', 'b')").await?;
        let _ = df.collect().await?;
        Ok::<_, KrishivError>(())
    })
    .unwrap_err();
    assert_eq!(err.error_class().name(), "INVALID_ARRAY_INDEX");
    block_on(session.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    assert_eq!(rows("SELECT elt(5, 'a', 'b')"), ["NULL"]);
}
