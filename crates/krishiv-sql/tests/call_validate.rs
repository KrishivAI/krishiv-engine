// SPDX-License-Identifier: Apache-2.0
//! `CALL` gets the parse-time checks every other statement gets: its procedure name, and each
//! argument as an expression (review of #983: a string unit in `TIMESTAMPADD` passed `CALL`).
use krishiv_sql::{ParserOptions, parse_statement};

fn check(sql: &str) -> Result<(), String> {
    parse_statement(sql, &ParserOptions::default())
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn a_call_argument_is_checked_as_an_expression() {
    let e = check("CALL cat.system.p(TIMESTAMPADD('DAY', 1, TIMESTAMP '2020-01-01 00:00:00'))")
        .unwrap_err();
    assert!(e.contains("DATETIME_UNIT"), "{e}");
    let e = check("CALL cat.system.p(x => TIMESTAMPADD('DAY', 1, TIMESTAMP '2020-01-01'))")
        .unwrap_err();
    assert!(e.contains("DATETIME_UNIT"), "{e}");
}

#[test]
fn valid_calls_still_parse() {
    check("CALL cat.system.expire_snapshots(table => 'db.t', retain_last => 1)").unwrap();
    check("CALL system.p('db.t', TIMESTAMP '2020-01-01 00:00:00', array(1L, 2L))").unwrap();
    check("CALL cat.system.p()").unwrap();
}
