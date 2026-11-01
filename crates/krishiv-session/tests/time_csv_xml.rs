// SPDX-License-Identifier: Apache-2.0
//! TIME in `from_csv`/`to_csv`/`from_xml`/`to_xml`, beyond the Spark 4.2.0
//! `csv-functions` and `xml-functions` goldens: a nested TIME, the lenient
//! fallback and a `timeFormat` that drops the date.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn text(sql: &str) -> Option<String> {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    let c = &b[0].columns()[0];
    (!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned())
}

#[test]
fn to_csv_shows_a_nested_time_and_a_time_format() {
    assert_eq!(
        text("to_csv(named_struct('a', array(TIME'01:02:03.5')))").as_deref(),
        Some("[01:02:03.5]")
    );
    assert_eq!(
        text("to_csv(named_struct('t', TIME'01:02:03.5'), map('timeFormat', 'HH.mm.ss'))")
            .as_deref(),
        Some("01.02.03")
    );
}

#[test]
fn from_csv_reads_a_time_format_and_falls_back_to_the_default() {
    assert_eq!(
        text("from_csv('01.02.03', 't TIME', map('timeFormat', 'HH.mm.ss'))").as_deref(),
        Some("{01:02:03}")
    );
    // Without the fallback option a value outside the pattern is null.
    assert_eq!(
        text("from_csv('01:02:03', 't TIME', map('timeFormat', 'HH.mm.ss'))").as_deref(),
        Some("{null}")
    );
    assert_eq!(
        text(
            "from_csv('01:02:03', 't TIME', map('timeFormat', 'HH.mm.ss', \
             'enableDateTimeParsingFallback', 'true'))"
        )
        .as_deref(),
        Some("{01:02:03}")
    );
}

#[test]
fn xml_reads_and_writes_a_time_with_a_time_format() {
    assert_eq!(
        text(
            "from_xml('<r><t>01.02.03</t></r>', 't TIME', \
             map('rowTag', 'r', 'timeFormat', 'HH.mm.ss'))"
        )
        .as_deref(),
        Some("{01:02:03}")
    );
    assert_eq!(
        text(
            "to_xml(named_struct('t', TIME'01:02:03'), \
             map('rowTag', 'r', 'timeFormat', 'HH.mm.ss', 'indent', ''))"
        )
        .as_deref(),
        Some("<r><t>01.02.03</t></r>\n")
    );
}
