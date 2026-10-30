// SPDX-License-Identifier: Apache-2.0
//! URL and CSV scalar functions end to end: the literal statements of
//! Spark 4.2's `url-functions.sql` run with Spark's rows, output names and
//! error classes, as do the `schema_of_csv` statements of
//! `csv-functions.sql` that use no other unimplemented function. The
//! `from_csv`/`to_csv` statements live in `csv_from_to.rs`; the
//! `schema_of_csv` options statement is covered by
//! `schema_of_csv_reads_map_options` (values from the Spark 4.2.0 oracle).
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

fn legacy() -> Session {
    Session::builder()
        .config("spark.sql.ansi.enabled", "false")
        .build()
        .unwrap()
}

/// The rows of a `SELECT`, rendered as Spark renders them (columns
/// joined by tabs).
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for col in b.columns() {
                if col.is_null(i) {
                    row.push("NULL".to_owned());
                    continue;
                }
                row.push(match col.data_type() {
                    ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
                        match col.data_type() {
                            ArrowType::Utf8View => col.as_string_view().value(i).to_owned(),
                            ArrowType::LargeUtf8 => col.as_string::<i64>().value(i).to_owned(),
                            _ => col.as_string::<i32>().value(i).to_owned(),
                        }
                    }
                    t => panic!("unrendered {t}"),
                });
            }
            out.push(row.join("\t"));
        }
    }
    out
}

/// Runs `sql`, expecting rows.
fn run(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    rows_of(&block_on(df.collect()).unwrap())
}

/// The output names of `sql`.
fn names(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Runs `sql`, expecting an analysis or execution error class.
fn class_of(s: &Session, sql: &str) -> ErrorClass {
    error_of(s, sql).error_class()
}

fn error_of(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Ok(df) => match block_on(df.collect()) {
            Ok(b) => panic!("{sql} ran: {b:?}"),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

#[test]
fn url_goldens() {
    let s = session();
    // `parse_url`: every part of the golden URL.
    let url = "'http://userinfo@spark.apache.org/path?query=1#Ref'";
    for (part, want) in [
        ("HOST", "spark.apache.org"),
        ("PATH", "/path"),
        ("QUERY", "query=1"),
        ("REF", "Ref"),
        ("PROTOCOL", "http"),
        ("FILE", "/path?query=1"),
        ("AUTHORITY", "userinfo@spark.apache.org"),
        ("USERINFO", "userinfo"),
    ] {
        assert_eq!(
            run(&s, &format!("select parse_url({url}, '{part}')")),
            [want]
        );
    }
    assert_eq!(
        names(
            &s,
            "select parse_url('http://userinfo@spark.apache.org/path?query=1#Ref', 'HOST')"
        ),
        // The engine redacts URL credentials in names (ADR 0038), even a
        // bare userinfo with no password; Spark's golden keeps it.
        ["parse_url(http://*********(redacted)@spark.apache.org/path?query=1#Ref, HOST)"]
    );
    // `url_encode`.
    assert_eq!(
        run(&s, "select url_encode('https://spark.apache.org')"),
        ["https%3A%2F%2Fspark.apache.org"]
    );
    // The codecs cast any argument: an `int` encodes as its text
    // (checked against Spark 4.2.0).
    assert_eq!(run(&s, "select url_encode(123)"), ["123"]);
    assert_eq!(run(&s, "select url_decode(123)"), ["123"]);
    // `url_encode` (the golden writes `\;`, which this engine's lexer
    // reads as `;` while Spark keeps `\;`; `\\;` is a backslash in both).
    assert_eq!(
        run(
            &s,
            r"select url_encode('inva lid://user:pass@host/file\\;param?query\\;p2')"
        ),
        ["inva+lid%3A%2F%2Fuser%3Apass%40host%2Ffile%5C%3Bparam%3Fquery%5C%3Bp2"]
    );
    assert_eq!(run(&s, "select url_encode(null)"), ["NULL"]);
    assert_eq!(
        names(&s, "select url_encode('https://spark.apache.org')"),
        ["url_encode(https://spark.apache.org)"]
    );
    // `url_decode`.
    assert_eq!(
        run(&s, "select url_decode('https%3A%2F%2Fspark.apache.org')"),
        ["https://spark.apache.org"]
    );
    assert_eq!(
        class_of(&s, "select url_decode('http%3A%2F%2spark.apache.org')"),
        ErrorClass::CannotDecodeUrl
    );
    assert_eq!(
        error_of(&s, "select url_decode('http%3A%2F%2spark.apache.org')").to_string(),
        "[CANNOT_DECODE_URL] The provided URL cannot be decoded: \
         http%3A%2F%2spark.apache.org. Please ensure that the URL is properly \
         formatted and try again. SQLSTATE: 22546"
    );
    assert_eq!(
        run(
            &s,
            r"select url_decode('inva lid://user:pass@host/file\\;param?query\\;p2')"
        ),
        [r"inva lid://user:pass@host/file\;param?query\;p2"]
    );
    assert_eq!(run(&s, "select url_decode(null)"), ["NULL"]);
    // `try_url_decode` is NULL wherever decoding fails.
    assert_eq!(
        run(
            &s,
            "select try_url_decode('https%3A%2F%2Fspark.apache.org')"
        ),
        ["https://spark.apache.org"]
    );
    assert_eq!(
        run(&s, "select try_url_decode('http%3A%2F%2spark.apache.org')"),
        ["NULL"]
    );
    assert_eq!(
        run(
            &s,
            r"select try_url_decode('inva lid://user:pass@host/file\\;param?query\\;p2')"
        ),
        [r"inva lid://user:pass@host/file\;param?query\;p2"]
    );
    assert_eq!(run(&s, "select try_url_decode(null)"), ["NULL"]);
    assert_eq!(
        names(&s, "select try_url_decode('http%3A%2F%2spark.apache.org')"),
        ["try_url_decode(http%3A%2F%2spark.apache.org)"]
    );
}

#[test]
fn parse_url_keys_and_invalid_urls() {
    let s = session();
    // `parse_url(url, 'QUERY', key)`.
    assert_eq!(
        run(
            &s,
            "select parse_url('http://spark.apache.org/path?query=1', 'QUERY', 'query')"
        ),
        ["1"]
    );
    assert_eq!(
        run(
            &s,
            "select parse_url('http://spark.apache.org/path?query=1', 'QUERY', 'missing')"
        ),
        ["NULL"]
    );
    assert_eq!(
        run(
            &s,
            "select parse_url('http://spark.apache.org/path?query=1', 'HOST', 'query')"
        ),
        ["NULL"]
    );
    // An invalid URL raises under ANSI and is NULL without it;
    // `try_parse_url` is NULL in both modes.
    assert_eq!(
        class_of(&s, "select parse_url(' \\t a\u{3000}\n ', 'HOST')"),
        ErrorClass::InvalidUrl
    );
    assert_eq!(
        error_of(&s, "select parse_url(' \\t a\u{3000}\n ', 'HOST')").to_string(),
        "[INVALID_URL] The url is invalid:  \t a\u{3000}\n . Use `try_parse_url` to tolerate \
         invalid URL and return NULL instead. SQLSTATE: 22P02"
    );
    let l = legacy();
    assert_eq!(
        run(&l, "select parse_url(' \\t a\u{3000}\n ', 'HOST')"),
        ["NULL"]
    );
    assert_eq!(
        run(&s, "select try_parse_url(' \\t a\u{3000}\n ', 'HOST')"),
        ["NULL"]
    );
    assert_eq!(
        run(&l, "select try_parse_url(' \\t a\u{3000}\n ', 'HOST')"),
        ["NULL"]
    );
    assert_eq!(
        run(
            &s,
            "select try_parse_url('inva lid://spark.apache.org/path?query=1', 'QUERY')"
        ),
        ["NULL"]
    );
    // The key is a Java regular expression, not a literal: `a.b` matches
    // `aXb=` first (checked against Spark 4.2.0).
    assert_eq!(
        run(
            &s,
            "select parse_url('http://h/?aXb=1&a.b=2', 'QUERY', 'a.b')"
        ),
        ["1"]
    );
    assert_eq!(
        run(&s, "select parse_url('http://h/?aXb=1', 'QUERY', 'a.b')"),
        ["1"]
    );
    // An invalid key pattern fails the query in both modes (Spark itself
    // throws `PatternSyntaxException`, which has no error class, also for
    // `try_parse_url`), as an untyped internal error.
    assert_eq!(
        class_of(&s, "select parse_url('http://h/?a=1', 'QUERY', '[')"),
        ErrorClass::InternalError
    );
    assert_eq!(
        class_of(&s, "select try_parse_url('http://h/?a=1', 'QUERY', '[')"),
        ErrorClass::InternalError
    );
    // A mistyped URL is refused, not cast.
    assert_eq!(
        class_of(&s, "select parse_url(1, 'HOST')"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    // An unknown part is NULL, not an error (checked against Spark 4.2.0).
    assert_eq!(run(&s, "select parse_url('http://h/', 'FOO')"), ["NULL"]);
    assert_eq!(
        run(&s, "select try_parse_url('http://h/', 'FOO')"),
        ["NULL"]
    );
    // Wrong argument counts name what Spark expects.
    assert_eq!(
        class_of(&s, "select parse_url('http://h/')"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    assert_eq!(
        class_of(&s, "select url_encode('a', 'b')"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    assert_eq!(
        class_of(&s, "select schema_of_csv()"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    // What encodes, decodes.
    assert_eq!(
        run(&s, "select url_decode(url_encode('a b+c~d'))"),
        ["a b+c~d"]
    );
}

#[test]
fn parse_url_over_a_table() {
    let s = session();
    let schema = Arc::new(Schema::new(vec![Field::new("u", ArrowType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![
            Some("http://user:pass@h:8080/p?q=1#frag"),
            Some("http://[::1]:8080/p"),
            Some("mailto:user@example.com"),
            Some("not-a-url"),
            None,
        ]))],
    )
    .unwrap();
    s.register_batches("urls", schema, vec![batch]).unwrap();
    assert_eq!(
        run(
            &s,
            "SELECT parse_url(u, 'HOST'), parse_url(u, 'PATH') FROM urls"
        ),
        [
            "h\t/p",
            "[::1]\t/p",
            "NULL\tNULL",
            "NULL\tnot-a-url",
            "NULL\tNULL",
        ]
    );
    assert_eq!(
        run(&s, "SELECT parse_url(u, 'QUERY', 'q') FROM urls"),
        ["1", "NULL", "NULL", "NULL", "NULL"]
    );
}

#[test]
fn schema_of_csv_goldens() {
    let s = session();
    // The DDL of literals.
    assert_eq!(
        run(&s, "select schema_of_csv('1,abc')"),
        ["STRUCT<_c0: INT, _c1: STRING>"]
    );
    assert_eq!(
        names(&s, "select schema_of_csv('1,abc')"),
        ["schema_of_csv(1,abc)"]
    );
    assert_eq!(
        run(&s, "select schema_of_csv('14:30:45')"),
        ["STRUCT<_c0: TIMESTAMP>"]
    );
    assert_eq!(
        run(&s, "select schema_of_csv('14:30:45.123456')"),
        ["STRUCT<_c0: TIMESTAMP>"]
    );
    // Function names are case-insensitive: the upper-case call resolves
    // to the same DDL (checked against Spark 4.2.0).
    assert_eq!(
        run(&s, "select SCHEMA_OF_CSV('1,abc')"),
        ["STRUCT<_c0: INT, _c1: STRING>"]
    );
    // An empty literal parses to no record: an internal error, as Spark
    // 4.2.0 fails it with `INTERNAL_ERROR` during constant folding.
    assert_eq!(
        class_of(&s, "select schema_of_csv('')"),
        ErrorClass::InternalError
    );
    // A null CSV.
    assert_eq!(
        class_of(&s, "select schema_of_csv(null)"),
        ErrorClass::DatatypeMismatchUnexpectedNull
    );
    assert_eq!(
        error_of(&s, "select schema_of_csv(null)").to_string(),
        "[DATATYPE_MISMATCH.UNEXPECTED_NULL] Cannot resolve \"schema_of_csv(NULL)\" due to \
         data type mismatch: The csv must not be null. SQLSTATE: 42K09"
    );
    // A non-string CSV is the wrong type, not a cast.
    assert_eq!(
        class_of(&s, "select schema_of_csv(42)"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        error_of(&s, "select schema_of_csv(42)").to_string(),
        "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] Cannot resolve \"schema_of_csv(42)\" due \
         to data type mismatch: The first parameter requires the \"STRING\" type, however \
         \"42\" has the type \"INT\". SQLSTATE: 42K09"
    );
    // Options are a map, which Krishiv cannot type yet.
    assert_eq!(
        class_of(&s, "select schema_of_csv('1', 'x')"),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    // A non-foldable CSV names itself (`VALUES` is unimplemented, so the
    // golden's view is built by registering its shape).
    let schema = Arc::new(Schema::new(vec![
        Field::new("csvField", ArrowType::Utf8, true),
        Field::new("a", ArrowType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec![Some("1,abc")])),
            Arc::new(StringArray::from(vec![Some("a")])),
        ],
    )
    .unwrap();
    s.register_batches("csvTable", schema, vec![batch]).unwrap();
    assert_eq!(
        class_of(&s, "SELECT schema_of_csv(csvField) FROM csvTable"),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    assert_eq!(
        error_of(&s, "SELECT schema_of_csv(csvField) FROM csvTable").to_string(),
        "[DATATYPE_MISMATCH.NON_FOLDABLE_INPUT] Cannot resolve \
         \"schema_of_csv(csvField)\" due to data type mismatch: the input `csv` should be a \
         foldable \"STRING\" expression; however, got \"csvField\". SQLSTATE: 42K09"
    );
}

#[test]
fn schema_of_csv_rejects_an_untokenizable_literal() {
    // Past `maxColumns` the tokenizer gives up: an internal error. Spark
    // 4.2.0 fails this same literal during constant folding too (its own
    // parser throws `TextParsingException` past 20480 columns).
    let s = session();
    let csv = "a,".repeat(20_481);
    assert_eq!(
        class_of(&s, &format!("select schema_of_csv('{csv}')")),
        ErrorClass::InternalError
    );
}

/// A session variable as the CSV keeps the name the call already shows (the
/// same form any function over a variable shows), not the folded DDL
/// literal. Spark 4.2.0 spells a variable argument
/// `variablereference(system.session.v='1,abc')` in every function's name;
/// that spelling is a gap of the engine's naming of variables in general,
/// not of this function.
#[test]
fn schema_of_csv_of_a_variable_keeps_its_shown_form() {
    let s = session();
    run(&s, "DECLARE OR REPLACE VARIABLE v STRING DEFAULT '1,abc'");
    assert_eq!(names(&s, "select upper(v)"), ["upper(v)"]);
    assert_eq!(names(&s, "select schema_of_csv(v)"), ["schema_of_csv(v)"]);
    assert_eq!(
        run(&s, "select schema_of_csv(v)"),
        ["STRUCT<_c0: INT, _c1: STRING>"]
    );
}

/// `schema_of_csv(csv, map(...))` reads the options literal, and the output
/// name drops it; a non-`map` or a map of other types is refused with
/// Spark's classes (all checked against Spark 4.2.0).
#[test]
fn schema_of_csv_reads_map_options() {
    let s = session();
    assert_eq!(
        run(&s, "select schema_of_csv('1|abc', map('delimiter', '|'))"),
        ["STRUCT<_c0: INT, _c1: STRING>"]
    );
    assert_eq!(
        names(&s, "select schema_of_csv('1|abc', map('delimiter', '|'))"),
        ["schema_of_csv(1|abc)"]
    );
    // The option changes the answer: with the default separator the record is one string field.
    assert_eq!(
        run(
            &s,
            "select schema_of_csv('1;abc', map('sep', ';', 'inferSchema', 'false'))"
        ),
        ["STRUCT<_c0: INT, _c1: STRING>"]
    );
    assert_eq!(
        run(&s, "select schema_of_csv('1;abc')"),
        ["STRUCT<_c0: STRING>"]
    );
    assert_eq!(
        class_of(&s, "select schema_of_csv('1', 1)"),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        class_of(&s, "select schema_of_csv('1', map(1, 2))"),
        ErrorClass::InvalidOptionsNonStringType
    );
    assert_eq!(
        run(
            &s,
            "select to_csv(named_struct('a', 1, 'b', 2), map('sep', '|'))"
        ),
        ["1|2"]
    );
}

/// The options `map(...)` is evaluated as a map: a repeated key is
/// `DUPLICATED_MAP_KEY` unless the session keeps the last one; option names
/// are case-insensitive and the one written last wins (Spark 4.2.0).
#[test]
fn option_maps_follow_the_map_key_dedup_policy() {
    let s = session();
    assert_eq!(
        class_of(
            &s,
            "select to_csv(named_struct('a', 1, 'b', 2), map('sep', '|', 'sep', ';'))"
        ),
        ErrorClass::DuplicatedMapKey
    );
    assert_eq!(
        class_of(
            &s,
            "select from_json('{\"a\":1}', 'a INT', map('mode', 'FAILFAST', 'mode', 'PERMISSIVE'))"
        ),
        ErrorClass::DuplicatedMapKey
    );
    // Two spellings of one option are distinct map keys; the last written wins.
    assert_eq!(
        run(
            &s,
            "select schema_of_csv('1|2', map('sep', '|', 'SEP', ';'))"
        ),
        ["STRUCT<_c0: STRING>"]
    );
    assert_eq!(
        run(
            &s,
            "select to_csv(named_struct('a', 1, 'b', 2), map('sep', '|', 'SEP', ';'))"
        ),
        ["1;2"]
    );
    block_on(s.sql("set spark.sql.mapKeyDedupPolicy=LAST_WIN")).expect("set policy");
    assert_eq!(
        run(
            &s,
            "select to_csv(named_struct('a', 1, 'b', 2), map('sep', '|', 'sep', ';'))"
        ),
        ["1;2"]
    );
}

/// `to_json`, `from_json` and `schema_of_json` take their options from a
/// `map(...)` and are named after their first argument (Spark 4.2.0).
#[test]
fn json_functions_read_map_options() {
    let s = session();
    assert_eq!(
        run(
            &s,
            "select schema_of_json('{\"c1\":1}', map('primitivesAsString', 'true'))"
        ),
        ["STRUCT<c1: STRING>"]
    );
    assert_eq!(
        names(
            &s,
            "select schema_of_json('{\"c1\":1}', map('primitivesAsString', 'true'))"
        ),
        ["schema_of_json({\"c1\":1})"]
    );
    assert_eq!(
        run(
            &s,
            "select to_json(named_struct('a', 1, 'b', 'x'), map('ignoreNullFields', 'false'))"
        ),
        ["{\"a\":1,\"b\":\"x\"}"]
    );
    assert_eq!(
        names(
            &s,
            "select to_json(named_struct('a', 1), map('ignoreNullFields', 'false'))"
        ),
        ["to_json(named_struct(a, 1))"]
    );
    assert_eq!(
        run(
            &s,
            "select to_json(from_json('{\"a\":1}', 'a INT', map('mode', 'FAILFAST')))"
        ),
        ["{\"a\":1}"]
    );
}
