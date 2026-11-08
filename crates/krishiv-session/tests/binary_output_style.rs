// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.binaryOutputStyle` for `to_csv` and `to_xml`, and `from_xml` into `BINARY`, against
//! values from a real Spark 4.2.0 run (`spark-sql --conf spark.sql.binaryOutputStyle=...`).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

const CSV: &str = "SELECT to_csv(named_struct('n', 1, 'info', X'4561736F6E'))";
const XML: &str = "SELECT to_xml(named_struct('name', binary('Eason'), 'b', X'00FF'))";

fn text(s: &Session, sql: &str) -> String {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let c = batches[0].column(0);
    assert!(!c.is_null(0));
    match c.as_string_opt::<i32>() {
        Some(v) => v.value(0).to_owned(),
        None => c.as_string_view().value(0).to_owned(),
    }
}

fn with_style(style: Option<&str>) -> Session {
    let s = Session::new().unwrap();
    if let Some(style) = style {
        block_on(s.sql(&format!("SET spark.sql.binaryOutputStyle={style}"))).unwrap();
    }
    s
}

fn xml(name: &str, b: &str) -> String {
    format!("<ROW>\n    <name>{name}</name>\n    <b>{b}</b>\n</ROW>")
}

#[test]
fn every_style_prints_to_csv_and_to_xml_as_spark_does() {
    let cases = [
        (
            None,
            "1,[45 61 73 6F 6E]",
            xml("[45 61 73 6F 6E]", "[00 FF]"),
        ),
        (
            Some("HEX_DISCRETE"),
            "1,[45 61 73 6F 6E]",
            xml("[45 61 73 6F 6E]", "[00 FF]"),
        ),
        (Some("HEX"), "1,4561736F6E", xml("4561736F6E", "00FF")),
        (
            Some("BASIC"),
            "1,\"[69, 97, 115, 111, 110]\"",
            xml("[69, 97, 115, 111, 110]", "[0, -1]"),
        ),
        (Some("BASE64"), "1,RWFzb24", xml("RWFzb24", "AP8")),
        // `X'00FF'` is not UTF-8 text, so check the UTF-8 style on bytes that are.
        (Some("hex"), "1,4561736F6E", xml("4561736F6E", "00FF")),
    ];
    for (style, csv, xml) in cases {
        let s = with_style(style);
        assert_eq!(text(&s, CSV), csv, "to_csv under {style:?}");
        assert_eq!(text(&s, XML), xml, "to_xml under {style:?}");
    }
}

#[test]
fn utf8_style_prints_the_bytes_as_text() {
    let s = with_style(Some("UTF-8"));
    assert_eq!(text(&s, CSV), "1,Eason");
    assert_eq!(
        text(
            &s,
            "SELECT to_xml(named_struct('name', binary('Eason'), 'b', X'41'))"
        ),
        xml("Eason", "A")
    );
}

#[test]
fn from_xml_reads_a_binary_field_as_the_text_bytes() {
    // The element's text as UTF-8 bytes, whatever the output style.
    for style in [None, Some("HEX"), Some("UTF-8")] {
        let s = with_style(style);
        let batches = block_on(
            block_on(s.sql(
                "SELECT length(from_xml('<ROW><name>Eason</name></ROW>', 'name binary').name)",
            ))
            .unwrap()
            .collect(),
        )
        .unwrap();
        let c = batches[0].column(0);
        let n = arrow_array::cast::as_primitive_array::<arrow_array::types::Int32Type>(c).value(0);
        assert_eq!(n, 5, "{style:?}");
    }
}

#[test]
fn the_krishiv_key_sets_the_style_too() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET krishiv.sql.binaryOutputStyle=HEX")).unwrap();
    assert_eq!(text(&s, CSV), "1,4561736F6E");
}

#[test]
fn an_unknown_style_is_refused() {
    let s = Session::new().unwrap();
    assert!(block_on(s.sql("SET spark.sql.binaryOutputStyle=ROT13")).is_err());
}

fn rows(s: &Session, sql: &str) -> Result<Vec<arrow_array::RecordBatch>, String> {
    let df = block_on(s.sql(sql)).map_err(|e| e.to_string())?;
    block_on(df.collect()).map_err(|e| e.to_string())
}

#[test]
fn to_xml_utf8_escapes_markup_and_prints_empty_binary_as_an_empty_element() {
    // Spark 4.2.0: `<a>&lt;&amp;</a>` and `<b></b>`.
    let s = with_style(Some("UTF-8"));
    assert_eq!(
        text(&s, "SELECT to_xml(named_struct('a', X'3C26', 'b', X''))"),
        "<ROW>\n    <a>&lt;&amp;</a>\n    <b></b>\n</ROW>"
    );
}

#[test]
fn to_xml_utf8_refuses_a_nul_byte() {
    // Spark 4.2.0: WstxIOException "Invalid null character in text to output".
    let s = with_style(Some("UTF-8"));
    assert!(rows(&s, "SELECT to_xml(named_struct('a', X'4100'))").is_err());
}

#[test]
fn from_xml_reads_an_empty_or_null_value_element_into_binary_as_null() {
    // Spark 4.2.0: `{"a":null}` for each.
    let s = with_style(None);
    for sql in [
        "SELECT from_xml('<ROW><a></a></ROW>', 'a binary').a IS NULL",
        "SELECT from_xml('<ROW><a/></ROW>', 'a binary').a IS NULL",
        "SELECT from_xml('<ROW><a>-</a></ROW>', 'a binary', map('nullValue','-')).a IS NULL",
    ] {
        assert_eq!(
            text(&s, &format!("SELECT CAST(({sql}) AS STRING)")),
            "true",
            "{sql}"
        );
    }
}

#[test]
fn the_csv_file_writer_follows_the_style_too() {
    // Spark 4.2.0: bytes 'AB' under UTF-8 / BASE64 / HEX / HEX_DISCRETE write `AB` / `QUI` / `4142` / `[41 42]`.
    for (style, want) in [
        ("UTF-8", "AB"),
        ("BASE64", "QUI"),
        ("HEX", "4142"),
        ("HEX_DISCRETE", "[41 42]"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let s = with_style(Some(style));
        let df = block_on(s.sql("SELECT X'4142' AS b")).unwrap();
        block_on(
            df.write()
                .format("csv")
                .save(dir.path().join("out").to_str().unwrap()),
        )
        .unwrap();
        let got: String = std::fs::read_dir(dir.path().join("out"))
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "csv"))
            .map(|e| std::fs::read_to_string(e.path()).unwrap())
            .collect();
        assert_eq!(got.trim_end(), want, "csv file under {style}");
    }
}
