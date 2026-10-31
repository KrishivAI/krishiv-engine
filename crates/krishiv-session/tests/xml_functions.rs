// SPDX-License-Identifier: Apache-2.0
//! `from_xml` and `to_xml` end to end. Expected values come from the Spark
//! 4.2.0 oracle (the statements of `xml-functions.sql` that need neither
//! `TIME` nor a region session zone, plus the edge cases probed beside them;
//! `schema_of_xml` has its own tests below).
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
        ),
        r#"{"d":"2012-12-15","t":"2012-12-15T15:15:15.000Z"}"#
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
            "select from_xml('<p/>', 'a INT', map('excludeAttribute', 'true'))"
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
    // Names Spark does not know are ignored.
    assert_eq!(
        read(&s, "<p><a>1</a></p>", "a INT", "'unknownOption', 'x'"),
        r#"{"a":1}"#
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

fn schema(s: &Session, xml: &str, options: &str) -> String {
    let opts = if options.is_empty() {
        String::new()
    } else {
        format!(", map({options})")
    };
    run(s, &format!("select schema_of_xml('{xml}'{opts})")).remove(0)
}

#[test]
fn schema_of_xml_infers_like_spark() {
    let s = session();
    let cases = [
        (
            "<p><a>1</a><b>\"2\"</b></p>",
            "STRUCT<a: BIGINT, b: STRING>",
        ),
        ("<p><a>1</a><a>2.5</a></p>", "STRUCT<a: ARRAY<DOUBLE>>"),
        ("<p><a>1</a><a>x</a></p>", "STRUCT<a: ARRAY<STRING>>"),
        ("<p><a></a></p>", "STRUCT<a: STRING>"),
        ("<p><a/></p>", "STRUCT<a: STRING>"),
        (
            "<p><a>2020-01-02</a><b>2020-01-02 03:04:05</b><c>2020-01-02T03:04:05Z</c></p>",
            "STRUCT<a: DATE, b: TIMESTAMP, c: TIMESTAMP>",
        ),
        ("<p><a>true</a><a>1</a></p>", "STRUCT<a: ARRAY<STRING>>"),
        ("<p><a>12345678901234567890</a></p>", "STRUCT<a: DOUBLE>"),
        (
            "<p><a>1.0</a><b>1e3</b><c>-5</c><d>.5</d></p>",
            "STRUCT<a: DOUBLE, b: DOUBLE, c: BIGINT, d: DOUBLE>",
        ),
        (
            "<p><b><y>1</y><x>2</x></b></p>",
            "STRUCT<b: STRUCT<x: BIGINT, y: BIGINT>>",
        ),
        (
            "<p><b><y>1</y></b><b><x>2</x></b></p>",
            "STRUCT<b: ARRAY<STRUCT<x: BIGINT, y: BIGINT>>>",
        ),
        ("<p><a x=\"1\"/></p>", "STRUCT<a: STRUCT<_x: BIGINT>>"),
        (
            "<p><a x=\"1\">t</a></p>",
            "STRUCT<a: STRUCT<_VALUE: STRING, _x: BIGINT>>",
        ),
        ("<p><b>1</b>text</p>", "STRUCT<_VALUE: STRING, b: BIGINT>"),
        ("<p><B>1</B><b>2</b></p>", "STRUCT<B: ARRAY<BIGINT>>"),
        ("<p><a>NaN</a></p>", "STRUCT<a: STRING>"),
        ("<p><a>Infinity</a></p>", "STRUCT<a: STRING>"),
        ("<p><a> 1 </a></p>", "STRUCT<a: BIGINT>"),
        ("<p>text</p>", "STRUCT<_VALUE: STRING>"),
        ("<p/>", "STRUCT<>"),
        ("<p x=\"5\"/>", "STRUCT<_x: BIGINT>"),
        ("<p><a><b/></a></p>", "STRUCT<a: STRUCT<b: STRING>>"),
        ("<p><a>1</a><a><b>2</b></a></p>", "STRUCT<a: ARRAY<STRING>>"),
        (
            "<p><a>1</a><b>x</b></p><q><z>1</z></q>",
            "STRUCT<a: BIGINT, b: STRING>",
        ),
        ("not xml", "STRUCT<_corrupt_record: STRING>"),
        ("", "STRUCT<_corrupt_record: STRING>"),
        ("<p><a>0123</a></p>", "STRUCT<a: BIGINT>"),
        ("<p><a>+7</a></p>", "STRUCT<a: BIGINT>"),
        (
            "<p><a>9223372036854775807</a><b>9223372036854775808</b></p>",
            "STRUCT<a: BIGINT, b: DOUBLE>",
        ),
    ];
    for (xml, want) in cases {
        assert_eq!(schema(&s, xml, ""), want, "{xml:?}");
    }
    assert_eq!(
        names(
            &s,
            "select schema_of_xml('<p><a>1</a></p>', map('inferSchema', 'true'))"
        ),
        ["schema_of_xml(<p><a>1</a></p>)"]
    );
}

#[test]
fn schema_of_xml_options() {
    let s = session();
    let one = "<p><a>1</a></p>";
    assert_eq!(
        schema(&s, "<p><a>1.5</a></p>", "'prefersDecimal', 'true'"),
        "STRUCT<a: DECIMAL(2,1)>"
    );
    assert_eq!(
        schema(&s, one, "'prefersDecimal', 'true'"),
        "STRUCT<a: BIGINT>"
    );
    assert_eq!(
        schema(&s, "<p><a>2020-01-02</a></p>", "'inferSchema', 'false'"),
        "STRUCT<a: STRING>"
    );
    assert_eq!(
        schema(&s, "<p x=\"1\"><a>1</a></p>", "'attributePrefix', '@'"),
        "STRUCT<`@x`: BIGINT, a: BIGINT>"
    );
    assert_eq!(
        schema(&s, "<p>t<a>1</a></p>", "'valueTag', 'v'"),
        "STRUCT<a: BIGINT, v: STRING>"
    );
    assert_eq!(
        schema(&s, "<p><a> 1</a></p>", "'ignoreSurroundingSpaces', 'false'"),
        "STRUCT<a: STRING>"
    );
    assert_eq!(
        schema(
            &s,
            "<p><a>26/08/2015</a></p>",
            "'timestampFormat', 'dd/MM/yyyy'"
        ),
        "STRUCT<a: TIMESTAMP>"
    );
    assert_eq!(
        schema(&s, "<p><a>26/08/2015</a></p>", "'dateFormat', 'dd/MM/yyyy'"),
        "STRUCT<a: DATE>"
    );
    // Names Spark does not know are ignored.
    assert_eq!(schema(&s, one, "'sep', 'x'"), "STRUCT<a: BIGINT>");
    assert_eq!(schema(&s, one, "'mode', 'FAILFAST'"), "STRUCT<a: BIGINT>");
}

#[test]
fn schema_of_xml_errors_and_from_xml_of_it() {
    let s = session();
    assert_eq!(
        class_of(&s, "select schema_of_xml(null)"),
        ErrorClass::DatatypeMismatchUnexpectedNull
    );
    assert_eq!(
        class_of(&s, "select schema_of_xml(42)"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        class_of(&s, "select schema_of_xml('<p/>', named_struct('a', 1))"),
        ErrorClass::InvalidOptionsNonMapFunction
    );
    assert_eq!(
        class_of(&s, "select schema_of_xml('<p/>', map('a', 1))"),
        ErrorClass::InvalidOptionsNonStringType
    );
    assert_eq!(
        class_of(
            &s,
            "select schema_of_xml('<p/>', map('samplingRatio', '0.5'))"
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
    let s2 = session();
    block_on(s2.sql("create temporary view xmlTable(xmlField, a) as select * from values ('<p><a>1</a></p>', 'a')"))
        .expect("view");
    assert_eq!(
        class_of(&s2, "select schema_of_xml(xmlField) from xmlTable"),
        ErrorClass::DatatypeMismatchNonFoldableInput
    );
    // The schema it returns feeds `from_xml`.
    assert_eq!(
        run(
            &s,
            "select to_json(from_xml('<p><a>1</a><a>2</a><a>3</a></p>', \
             schema_of_xml('<p><a>1</a><a>2</a></p>')))"
        ),
        [r#"{"a":[1,2,3]}"#]
    );
}

fn nested(depth: usize) -> String {
    format!("{}{}", "<a>".repeat(depth), "</a>".repeat(depth))
}

/// The element depth is bounded: Spark has no limit, but a deep document
/// must not exhaust the stack or the memory of the process.
#[test]
fn deep_documents_are_malformed_not_fatal() {
    let s = session();
    for depth in [1_001usize, 100_000] {
        let doc = nested(depth);
        assert_eq!(
            read(&s, &doc, "a STRING", ""),
            r#"{"a":null}"#,
            "depth {depth}"
        );
        assert_eq!(
            schema(&s, &doc, ""),
            "STRUCT<_corrupt_record: STRING>",
            "depth {depth}"
        );
    }
    // Inside the bound the document reads and infers as usual.
    let ok = nested(500);
    assert!(read(&s, &ok, "a STRING", "").starts_with(r#"{"a":"<a>"#));
    assert!(schema(&s, &ok, "").starts_with("STRUCT<a: STRUCT<a: STRUCT<"));
}

#[test]
fn to_xml_checks_names_and_values() {
    let s = session();
    for value in [
        "named_struct('a b', 1)",
        "named_struct('', 1)",
        "named_struct('1a', 1)",
        "named_struct('a:b', 1)",
        "named_struct('a', map('x>y', 1))",
        "named_struct('_a b', 1)",
    ] {
        assert_eq!(
            class_of(&s, &format!("select to_xml({value})")),
            ErrorClass::KrishivUnsupportedFeature,
            "{value}"
        );
    }
    assert_eq!(
        write(
            &s,
            "named_struct('a b', 1)",
            "'indent', '', 'validateName', 'false'"
        ),
        "<ROW><a b>1</a b></ROW>\n"
    );
    assert_eq!(
        class_of(
            &s,
            "select to_xml(named_struct('a', 1), map('rowTag', 'a b'))"
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
    // A character XML 1.0 cannot hold is refused; `>` is escaped after `]]` only.
    assert_eq!(
        class_of(&s, "select to_xml(named_struct('a', concat('x', chr(1))))"),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(
        write(&s, "named_struct('a', 'x]]>y')", "'indent', ''"),
        "<ROW><a>x]]&gt;y</a></ROW>\n"
    );
}

#[test]
fn to_xml_wraps_inner_arrays_in_item_elements() {
    let s = session();
    assert_eq!(
        write(
            &s,
            "named_struct('a', array(array(1, 2), array(3)))",
            "'indent', ''"
        ),
        "<ROW><a><item>1</item><item>2</item></a><a><item>3</item></a></ROW>\n"
    );
}

#[test]
fn option_values_and_schemas() {
    let s = session();
    // Spark refuses a flag that is not true or false; an unknown mode reads as PERMISSIVE.
    for options in [
        "'ignoreSurroundingSpaces', 'maybe'",
        "'inferSchema', 'maybe'",
        "'prefersDecimal', 'maybe'",
        "'validateName', 'maybe'",
    ] {
        assert_eq!(
            class_of(
                &s,
                &format!("select schema_of_xml('<p><a>1</a></p>', map({options}))")
            ),
            ErrorClass::KrishivUnsupportedFeature,
            "{options}"
        );
    }
    assert_eq!(
        read(&s, "<p><a>1</a></p>", "a INT", "'mode', 'bogus'"),
        r#"{"a":1}"#
    );
    // The keys of a map read from XML are names: only STRING keys.
    assert_eq!(
        class_of(&s, "select from_xml('<p/>', 'a map<int,string>')"),
        ErrorClass::DatatypeMismatchInvalidXmlMapKeyType
    );
    assert_eq!(
        class_of(&s, "select from_xml('<p/>', 'a array<array<int>>')"),
        ErrorClass::KrishivUnsupportedFeature
    );
    // A DTD is never read: a custom entity makes the document malformed.
    assert_eq!(
        read(
            &s,
            "<!DOCTYPE p [<!ENTITY e \"zz\">]><p><a>&e;</a></p>",
            "a STRING",
            ""
        ),
        r#"{"a":null}"#
    );
}

#[test]
fn to_xml_formats_timestamps_in_the_session_zone() {
    let s = Session::builder()
        .config("spark.sql.session.timeZone", "+05:00")
        .build()
        .unwrap();
    assert_eq!(
        write(
            &s,
            "named_struct('a', timestamp'2020-01-02 03:04:05')",
            "'indent', '', 'timestampFormat', 'yyyy-MM-dd HH:mm VV XXX'"
        ),
        "<ROW><a>2020-01-02 03:04 +05:00 +05:00</a></ROW>\n"
    );
}

#[test]
fn from_xml_refuses_complex_map_values_and_schema_of_xml_options() {
    let s = session();
    for ty in ["map<string, struct<a: int>>", "map<string, array<int>>"] {
        let sql = format!("select from_xml('<p><k>1</k></p>', 'm {ty}')");
        assert_eq!(class_of(&s, &sql), ErrorClass::KrishivUnsupportedFeature);
    }
    let sql = "select schema_of_xml('<p><a>1</a></p>', map('mode', c)) from (select 'x' c)";
    assert!(
        matches!(
            error_of(&s, sql).error_class(),
            ErrorClass::InvalidOptionsNonMapFunction | ErrorClass::DatatypeMismatchNonFoldableInput
        ),
        "{sql}: {:?}",
        error_of(&s, sql)
    );
}

#[test]
fn from_xml_reads_every_scalar_type_and_entities() {
    let s = session();
    assert_eq!(
        read(
            &s,
            "<p><a>1</a><b>2</b><c>3</c><d>1.5</d><e>2.5</e><f>3.25</f><g>true</g>\
             <h>2020-01-02</h><i>2020-01-02 03:04:05</i><j>1</j><j>2</j>\
             <k><x>7</x></k><l><x>9</x></l></p>",
            "a tinyint, b smallint, c int, d float, e double, f decimal(5,2), g boolean, \
             h date, i timestamp_ntz, j array<int>, k map<string,int>, l struct<x:int>",
            ""
        ),
        r#"{"a":1,"b":2,"c":3,"d":1.5,"e":2.5,"f":3.25,"g":true,"h":"2020-01-02","i":"2020-01-02T03:04:05.000","j":[1,2],"k":{"x":7},"l":{"x":9}}"#
    );
    assert_eq!(
        read(&s, "<p><a>x &amp; &#65;&#x42; &lt;</a></p>", "a string", ""),
        r#"{"a":"x & AB <"}"#
    );
    // A value that does not fit the type reads as NULL.
    assert_eq!(
        read(&s, "<p><a>300</a><b>x</b></p>", "a tinyint, b float", ""),
        r#"{"a":null,"b":null}"#
    );
}
