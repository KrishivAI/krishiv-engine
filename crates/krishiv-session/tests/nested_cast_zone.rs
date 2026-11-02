// SPDX-License-Identifier: Apache-2.0
//! A cast of a nested value reads the session zone for the timestamps in it
//! (Spark 4.2.0 `typeCoercion/native/arrayJoin`).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

fn text(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap();
    let b = block_on(df.collect()).unwrap();
    b[0].column(0).as_string_view().value(0).to_owned()
}

#[test]
fn nested_timestamps_render_in_the_session_zone() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = America/Los_Angeles")).unwrap();
    let t = "timestamp '2016-11-15 20:54:00'";
    assert_eq!(
        text(&s, &format!("SELECT array_join(array({t}, {t}), ', ')")),
        "2016-11-15 20:54:00, 2016-11-15 20:54:00"
    );
    assert_eq!(
        text(&s, &format!("SELECT CAST(struct({t} AS a) AS STRING)")),
        "{2016-11-15 20:54:00}"
    );
    assert_eq!(
        text(&s, &format!("SELECT CAST(map(1, {t}) AS STRING)")),
        "{1 -> 2016-11-15 20:54:00}"
    );
}
