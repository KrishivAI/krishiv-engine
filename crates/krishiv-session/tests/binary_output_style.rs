// SPDX-License-Identifier: Apache-2.0
//! `spark.sql.binaryOutputStyle` for `to_csv` and `to_xml`: values from a real Spark 4.2.0 run
//! (`spark-sql --conf spark.sql.binaryOutputStyle=...`). The `from_xml` into `BINARY` read-back
//! tests are SOURCE-DERIVED from Spark v4.2.0 / v4.1.1 `ToStringBase.getBinaryParser` and
//! `StaxXmlParser` (brief pr670-binary-readback-semantics.md); no Spark run backs them yet.
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

/// The first cell as text (`None` for NULL) of `SELECT cast(<expr> as string)`.
fn cell(s: &Session, expr: &str) -> Option<String> {
    let batches = block_on(
        block_on(s.sql(&format!("SELECT cast({expr} as string)")))
            .unwrap()
            .collect(),
    )
    .unwrap();
    let c = batches[0].column(0);
    if c.is_null(0) {
        return None;
    }
    Some(match c.as_string_opt::<i32>() {
        Some(v) => v.value(0).to_owned(),
        None => c.as_string_view().value(0).to_owned(),
    })
}

fn from_xml_b(text: &str, opts: &str) -> String {
    format!("from_xml('<ROW><b>{text}</b><n>7</n></ROW>', 'b binary, n int'{opts})")
}

fn error_class_of(s: &Session, sql: &str) -> krishiv_common::error::ErrorClass {
    match block_on(s.sql(sql)) {
        Ok(df) => block_on(df.collect()).unwrap_err().error_class(),
        Err(e) => e.error_class(),
    }
}

/// Spark 4.2.0 (source-derived): `Eason` is not a hex/basic/base64 text, so the field is a miss
/// (NULL, siblings kept) in every style but UTF-8, where it is its 5 bytes.
#[test]
fn from_xml_eason_only_reads_in_utf8_style() {
    for (style, len) in [
        (None, None),
        (Some("HEX"), None),
        (Some("HEX_DISCRETE"), None),
        (Some("BASIC"), None),
        (Some("BASE64"), None),
        (Some("UTF-8"), Some("5")),
    ] {
        let s = with_style(style);
        let got = cell(
            &s,
            "length(from_xml('<ROW><name>Eason</name></ROW>', 'name binary').name)",
        );
        assert_eq!(got.as_deref(), len, "{style:?}");
        // The miss keeps the siblings and the struct itself.
        assert_eq!(
            cell(&s, &format!("{}.n", from_xml_b("Eason", ""))).as_deref(),
            Some("7"),
            "{style:?}"
        );
    }
}

/// Each style reads back what it printed (`X'00FF'`), and the lenient forms Spark decodes.
#[test]
fn from_xml_reads_back_each_style_source_derived() {
    for style in [
        None,
        Some("HEX_DISCRETE"),
        Some("HEX"),
        Some("BASIC"),
        Some("BASE64"),
    ] {
        let s = with_style(style);
        let round = "hex(from_xml(to_xml(named_struct('b', X'00FF')), 'b binary').b)";
        assert_eq!(cell(&s, round).as_deref(), Some("00FF"), "{style:?}");
    }
    let hex_of =
        |style, t: &str| cell(&with_style(style), &format!("hex({}.b)", from_xml_b(t, "")));
    assert_eq!(hex_of(None, "01 FF").as_deref(), Some("01FF"));
    assert_eq!(hex_of(None, "[01 ff]").as_deref(), Some("01FF"));
    assert_eq!(hex_of(None, "01FF"), None);
    assert_eq!(hex_of(Some("HEX"), "BAF").as_deref(), Some("0BAF"));
    assert_eq!(hex_of(Some("HEX"), "[01 FF]"), None);
    assert_eq!(hex_of(Some("BASIC"), "[1, 2 ]").as_deref(), Some("0102"));
    assert_eq!(hex_of(Some("BASIC"), "[+5]").as_deref(), Some("05"));
    assert_eq!(hex_of(Some("BASIC"), "[]"), None);
    assert_eq!(hex_of(Some("BASIC"), "[128]"), None);
    assert_eq!(hex_of(Some("BASE64"), "AB").as_deref(), Some("00"));
    assert_eq!(hex_of(Some("BASE64"), "AQ").as_deref(), Some("01"));
    assert_eq!(hex_of(Some("BASE64"), "YQ="), None);
    assert_eq!(
        hex_of(Some("UTF-8"), "[01 FF]").as_deref(),
        Some("5B30312046465D")
    );
}

/// A miss is NULL under PERMISSIVE and
/// `MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION` under FAILFAST (source-derived).
#[test]
fn from_xml_binary_miss_by_mode() {
    let s = with_style(None);
    let permissive =
        "from_xml('<ROW><b>Eason</b><n>7</n></ROW>', 'b binary, n int, _corrupt_record string')";
    assert_eq!(cell(&s, &format!("{permissive}.b")), None);
    assert_eq!(cell(&s, &format!("{permissive}.n")).as_deref(), Some("7"));
    let failfast = format!(
        "SELECT {}",
        from_xml_b("Eason", ", map('mode', 'FAILFAST')")
    );
    assert_eq!(
        error_class_of(&s, &failfast),
        krishiv_common::error::ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
    let drop = format!(
        "SELECT {}",
        from_xml_b("AQ==", ", map('mode', 'DROPMALFORMED')")
    );
    assert_eq!(
        error_class_of(&s, &drop),
        krishiv_common::error::ErrorClass::ParseModeUnsupported
    );
}

/// An empty or blank element is NULL in every style and mode; no error.
#[test]
fn from_xml_empty_binary_element_is_null() {
    for style in [
        None,
        Some("HEX"),
        Some("BASIC"),
        Some("BASE64"),
        Some("UTF-8"),
    ] {
        let s = with_style(style);
        for t in ["", "   "] {
            let ff = from_xml_b(t, ", map('mode', 'FAILFAST')");
            assert_eq!(cell(&s, &format!("{ff}.b")), None, "{style:?}");
        }
        assert_eq!(cell(&s, "from_xml('<ROW><b/></ROW>', 'b binary').b"), None);
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
