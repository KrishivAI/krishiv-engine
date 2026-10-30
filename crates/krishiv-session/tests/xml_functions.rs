// SPDX-License-Identifier: Apache-2.0
//! `from_xml` and `to_xml` end to end. Expected values come from the Spark
//! 4.2.0 oracle (the statements of `xml-functions.sql` that need neither
//! `schema_of_xml`, `TIME` nor a region session zone, plus the edge cases
//! probed beside them).
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
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
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    rows_of(&block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}")))
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

/// `to_json` of a `from_xml` result, with the nulls written.
fn read(s: &Session, xml: &str, schema: &str, options: &str) -> String {
    let opts = if options.is_empty() {
        String::new()
    } else {
        format!(", map({options})")
    };
    let sql = format!(
        "select to_json(from_xml('{xml}', '{schema}'{opts}), map('ignoreNullFields', 'false'))"
    );
    run(s, &sql).remove(0)
}

fn write(s: &Session, value: &str, options: &str) -> String {
    let opts = if options.is_empty() {
        String::new()
    } else {
        format!(", map({options})")
    };
    run(s, &format!("select to_xml({value}{opts})")).remove(0)
}

#[test]
fn from_xml_reads_the_golden_statements() {
    let s = session();
    assert_eq!(read(&s, "<p><a>1</a></p>", "a INT", ""), r#"{"a":1}"#);
    assert_eq!(
        names(&s, "select from_xml('<p><a>1</a></p>', 'a INT')"),
        ["from_xml(<p><a>1</a></p>)"]
    );
    assert_eq!(
        read(&s, "<p><a>1</a></p>", "struct<a:array<int>>", ""),
        r#"{"a":[1]}"#
    );
    assert_eq!(
        read(
            &s,
            "<p><a>1</a><b>\"2\"</b></p>",
            "struct<a:int,b:string>",
            ""
        ),
        r#"{"a":1,"b":"\"2\""}"#
    );
    assert_eq!(
        read(&s, "<p><a>1</a><a>2</a></p>", "struct<a:array<int>>", ""),
        r#"{"a":[1,2]}"#
    );
    // Items before the first one that does not convert are kept.
    assert_eq!(
        read(
            &s,
            "<p><a>1</a><a>\"2\"</a></p>",
            "struct<a:array<int>>",
            ""
        ),
        r#"{"a":[1]}"#
    );
    // An empty item is NULL.
    assert_eq!(
        read(&s, "<p><a>1</a><a></a></p>", "struct<a:array<int>>", ""),
        r#"{"a":[1,null]}"#
    );
    // An element with an attribute and a child reads as a map.
    assert_eq!(
        read(
            &s,
            "<p><a attr=\"1\"><b>2</b></a></p>",
            "struct<a:map<string,int>>",
            ""
        ),
        r#"{"a":{"_attr":1,"b":2}}"#
    );
    assert_eq!(
        read(
            &s,
            "<p><d>2012-12-15</d><t>2012-12-15 15:15:15</t></p>",
            "d date, t timestamp",
            ""
        )
        .split(',')
        .next(),
        Some(r#"{"d":"2012-12-15""#)
    );
    // Formats that do not match give NULL.
    assert_eq!(
        read(&s, "<p><d>02-29</d></p>", "d date", "'dateFormat', 'MM-dd'"),
        r#"{"d":null}"#
    );
    assert_eq!(
        read(
            &s,
            "<p><d>12/15 2012</d></p>",
            "d date",
            "'dateFormat', 'MM/dd yyyy'"
        ),
        r#"{"d":"2012-12-15"}"#
    );
}

#[test]
fn from_xml_edge_cases() {
    let s = session();
    // Surrounding spaces are dropped unless asked for; case does not matter.
    assert_eq!(read(&s, "<p><a> 1 </a></p>", "a INT", ""), r#"{"a":1}"#);
    assert_eq!(
        read(
            &s,
            "<p><a> 1</a></p>",
            "a INT",
            "'ignoreSurroundingSpaces', 'false'"
        ),
        r#"{"a":null}"#
    );
    assert_eq!(read(&s, "<p><A>1</A></p>", "a INT", ""), r#"{"a":1}"#);
    assert_eq!(
        read(&s, "<p><a>1</a></p>", "a INT, b INT", ""),
        r#"{"a":1,"b":null}"#
    );
    // Attributes carry the prefix; text beside children is the value tag.
    assert_eq!(
        read(&s, "<p x=\"7\"><a>1</a></p>", "_x INT, a INT", ""),
        r#"{"_x":7,"a":1}"#
    );
    assert_eq!(
        read(&s, "<p>text<a>1</a></p>", "_VALUE STRING, a INT", ""),
        r#"{"_VALUE":"text","a":1}"#
    );
    assert_eq!(
        read(&s, "<p>hi</p>", "v STRING, a INT", "'valueTag', 'v'"),
        r#"{"v":"hi","a":null}"#
    );
    // Entities and CDATA; an element with children read as text is its content.
    assert_eq!(
        read(&s, "<p><a>&lt;x&gt;&amp;</a></p>", "a STRING", ""),
        r#"{"a":"<x>&"}"#
    );
    assert_eq!(
        read(&s, "<p><a><![CDATA[<k>]]></a></p>", "a STRING", ""),
        r#"{"a":"<k>"}"#
    );
    assert_eq!(
        read(&s, "<p><a><b>1</b></a></p>", "a STRING", ""),
        r#"{"a":"<b>1</b>"}"#
    );
    assert_eq!(
        read(&s, "<p><a><b>1</b></a></p>", "a STRUCT<b:INT>", ""),
        r#"{"a":{"b":1}}"#
    );
    // Repeats into a scalar: the last one; the root's name and what follows it do not matter.
    assert_eq!(
        read(&s, "<p><a>1</a><a>2</a></p>", "a INT", ""),
        r#"{"a":2}"#
    );
    assert_eq!(
        read(&s, "<p><a>1</a></p>", "a INT", "'rowTag', 'zzz'"),
        r#"{"a":1}"#
    );
    assert_eq!(read(&s, "<p><a>1</a></p><q/>", "a INT", ""), r#"{"a":1}"#);
    assert_eq!(
        read(
            &s,
            "<?xml version=\"1.0\"?><!--c--><p><a>1</a></p>",
            "a INT",
            ""
        ),
        r#"{"a":1}"#
    );
    assert_eq!(read(&s, "<p><a></a></p>", "a STRING", ""), r#"{"a":""}"#);
    assert_eq!(
        read(&s, "<p><a>1</a></p>", "a INT", "'nullValue', '1'"),
        r#"{"a":null}"#
    );
    // Values that do not convert are NULL; broken or empty documents are rows of NULLs.
    for (xml, ty) in [
        ("<p><a>abc</a></p>", "a INT"),
        ("<p><a>1.5</a></p>", "a INT"),
        ("<p><a>9999999999</a></p>", "a INT"),
    ] {
        assert_eq!(read(&s, xml, ty, ""), r#"{"a":null}"#, "{xml}");
    }
    for xml in ["<p><a>x</a>", "not xml", ""] {
        assert_eq!(read(&s, xml, "a STRING", ""), r#"{"a":null}"#, "{xml:?}");
    }
    assert_eq!(
        run(
            &s,
            "select cast(from_xml(cast(null as string), 'a STRING') is null as string)"
        ),
        ["true"]
    );
}

#[test]
fn from_xml_modes_and_errors() {
    let s = session();
    assert_eq!(
        class_of(
            &s,
            "select from_xml('<p><a>abc</a></p>', 'a INT', map('mode', 'FAILFAST'))"
        ),
        ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
    assert_eq!(
        class_of(
            &s,
            "select from_xml('<p><a>abc</a></p>', 'a INT', map('mode', 'DROPMALFORMED'))"
        ),
        ErrorClass::ParseModeUnsupported
    );
    assert_eq!(
        class_of(&s, "select from_xml('<p><a>1</a></p>', 1)"),
        ErrorClass::InvalidSchemaNonStringLiteral
    );
    assert_eq!(
        class_of(
            &s,
            "select from_xml('<p><a>1</a></p>', 'a INT', named_struct('mode', 'PERMISSIVE'))"
        ),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        class_of(
            &s,
            "select from_xml('<p><a>1</a></p>', 'a INT', map('mode', 1))"
        ),
        ErrorClass::InvalidOptionsNonStringType
    );
    assert_eq!(
        class_of(&s, "select from_xml()"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    assert_eq!(
        class_of(&s, "select to_xml()"),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    assert_eq!(
        class_of(&s, "select to_xml(map('a', 1))"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        class_of(
            &s,
            "select to_xml(named_struct('a', 1), named_struct('mode', 'PERMISSIVE'))"
        ),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        class_of(&s, "select to_xml(named_struct('a', 1), map('mode', 1))"),
        ErrorClass::InvalidOptionsNonStringType
    );
    assert_eq!(
        class_of(
            &s,
            "select from_xml('<p/>', 'a INT', map('unknownOption', 'x'))"
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn to_xml_writes_documents() {
    let s = session();
    let compact = "'indent', ''";
    assert_eq!(
        write(&s, "named_struct('a', 1, 'b', 2)", compact),
        "<ROW><a>1</a><b>2</b></ROW>\n"
    );
    assert_eq!(
        names(
            &s,
            "select to_xml(named_struct('a', 1, 'b', 2), map('indent', ''))"
        ),
        ["to_xml(named_struct(a, 1, b, 2))"]
    );
    // The default pretty-prints with four spaces and no final line feed.
    assert_eq!(
        write(&s, "named_struct('a', 1, 'b', 'x')", ""),
        "<ROW>\n    <a>1</a>\n    <b>x</b>\n</ROW>"
    );
    assert_eq!(
        write(
            &s,
            "named_struct('a', array(1, 2), 'b', named_struct('c', 3))",
            ""
        ),
        "<ROW>\n    <a>1</a>\n    <a>2</a>\n    <b>\n        <c>3</c>\n    </b>\n</ROW>"
    );
    assert_eq!(
        write(
            &s,
            "named_struct('a', named_struct('x', 1))",
            "'indent', '  '"
        ),
        "<ROW>\n  <a>\n    <x>1</x>\n  </a>\n</ROW>"
    );
    assert_eq!(
        write(&s, "named_struct('a', map('k', 1, 'j', 2))", ""),
        "<ROW>\n    <a>\n        <k>1</k>\n        <j>2</j>\n    </a>\n</ROW>"
    );
    // Nulls are left out unless `nullValue` is given; an empty element is self-closing.
    assert_eq!(
        write(&s, "named_struct('a', cast(null as string))", compact),
        "<ROW/>\n"
    );
    assert_eq!(
        write(&s, "named_struct('a', cast(null as string))", ""),
        "<ROW/>"
    );
    assert_eq!(
        write(
            &s,
            "named_struct('a', cast(null as string), 'b', 1)",
            "'indent', '', 'nullValue', 'NIL'"
        ),
        "<ROW><a>NIL</a><b>1</b></ROW>\n"
    );
    // Row tag, attributes, value tag, escaping and scalar formats.
    assert_eq!(
        write(&s, "named_struct('a', 1)", "'rowTag', 'r', 'indent', ''"),
        "<r><a>1</a></r>\n"
    );
    assert_eq!(
        write(&s, "named_struct('_attr', 1, '_VALUE', 'v')", compact),
        "<ROW attr=\"1\">v</ROW>\n"
    );
    assert_eq!(
        write(&s, "named_struct('a', '<&>\"x')", compact),
        "<ROW><a>&lt;&amp;>\"x</a></ROW>\n"
    );
    assert_eq!(
        write(
            &s,
            "named_struct('a', 1.5D, 'b', true, 'd', date'2020-01-02', 'dec', cast(1.50 as decimal(5,2)))",
            compact
        ),
        "<ROW><a>1.5</a><b>true</b><d>2020-01-02</d><dec>1.50</dec></ROW>\n"
    );
    assert_eq!(
        write(
            &s,
            "named_struct('n', timestamp_ntz'2020-01-02 03:04:05')",
            compact
        ),
        "<ROW><n>2020-01-02T03:04:05.000</n></ROW>\n"
    );
    assert_eq!(
        run(
            &s,
            "select cast(to_xml(cast(null as struct<a:int>)) is null as string)"
        ),
        ["true"]
    );
}
