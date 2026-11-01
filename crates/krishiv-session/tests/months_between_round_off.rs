// SPDX-License-Identifier: Apache-2.0
//! `months_between`'s `roundOff` is a strict BOOLEAN without ANSI and cast
//! with it (Spark 4.2.0 oracle: a string `'true'` is a type mismatch with
//! `spark.sql.ansi.enabled=false` and works with it on).
#![forbid(unsafe_code)]
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

const CALL: &str = "SELECT months_between(timestamp'2020-02-01 00:00:00', \
                    timestamp'2020-01-01 00:00:00', ";

fn run(s: &Session, round_off: &str) -> Result<usize, krishiv_common::error::KrishivError> {
    let df = block_on(s.sql(&format!("{CALL}{round_off})")))?;
    Ok(block_on(df.collect())?
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum())
}

#[test]
fn a_string_round_off_needs_ansi() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    for arg in ["'true'", "1"] {
        let err = run(&s, arg).unwrap_err();
        assert_eq!(
            err.error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{arg}"
        );
    }
    assert_eq!(run(&s, "true").unwrap(), 1);
    assert_eq!(run(&s, "NULL").unwrap(), 1);
    block_on(s.sql("SET spark.sql.ansi.enabled = true")).unwrap();
    assert_eq!(run(&s, "'true'").unwrap(), 1);
}
