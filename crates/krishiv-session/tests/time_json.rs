// SPDX-License-Identifier: Apache-2.0
//! TIME in `from_json`/`to_json`, beyond the Spark 4.2.0 `json-functions`
//! golden: a nested TIME, a `timeFormat` that drops the date, a pattern miss
//! and a NULL.
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
fn to_json_writes_default_and_formatted_times() {
    assert_eq!(
        text("to_json(named_struct('t', TIME'01:02:03.5'))").as_deref(),
        Some(r#"{"t":"01:02:03.5"}"#)
    );
    assert_eq!(
        text("to_json(array(TIME'01:02:03'), map('timeFormat', 'HH.mm.ss'))").as_deref(),
        Some(r#"["01.02.03"]"#)
    );
}

#[test]
fn from_json_reads_nested_times_and_misses() {
    assert_eq!(
        text("from_json('{\"a\":[\"01:02:03\",\"04:05:06\"]}', 'a array<time>')").as_deref(),
        Some("{[01:02:03, 04:05:06]}")
    );
    assert_eq!(
        text("from_json('{\"t\":\"01:02:03\"}', 't TIME', map('timeFormat', 'HH.mm.ss'))")
            .as_deref(),
        Some("{null}")
    );
    assert_eq!(
        text("from_json('{\"t\":\"01.02.03\"}', 't TIME', map('timeFormat', 'HH.mm.ss'))")
            .as_deref(),
        Some("{01:02:03}")
    );
}
