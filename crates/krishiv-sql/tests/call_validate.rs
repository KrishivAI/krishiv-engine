// SPDX-License-Identifier: Apache-2.0
//! `CALL` gets the parse-time checks every other statement gets: its procedure name, and each
//! argument as an expression (review of #983: a string unit in `TIMESTAMPADD` passed `CALL`).
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]
use krishiv_sql::ast::{DdlName, ExprKind, Literal, Statement};
use krishiv_sql::{ParseError, ParserOptions, parse_statement};

fn parse(sql: &str) -> Result<Statement, ParseError> {
    parse_statement(sql, &ParserOptions::default())
}

/// The error class and the text the error points at.
fn refusal(sql: &str) -> (String, String) {
    let e = parse(sql).unwrap_err();
    let at = sql
        .get(e.span.start..e.span.end)
        .unwrap_or_default()
        .to_owned();
    (e.error.error_class().name().to_owned(), at)
}

#[test]
fn a_call_argument_is_checked_as_an_expression() {
    for sql in [
        "CALL cat.system.p(TIMESTAMPADD('DAY', 1, TIMESTAMP '2020-01-01 00:00:00'))",
        "CALL cat.system.p(x => TIMESTAMPADD('DAY', 1, TIMESTAMP '2020-01-01'))",
    ] {
        let (class, at) = refusal(sql);
        assert_eq!(class, "INVALID_PARAMETER_VALUE.DATETIME_UNIT", "{sql}");
        // Reported at the argument's function, not elsewhere in the statement.
        assert!(at.starts_with("TIMESTAMPADD("), "{sql}: {at}");
    }
}

/// The parts of a `CALL`'s name and its arguments as (name, expression text).
fn call_of(sql: &str) -> (Vec<String>, Vec<(Option<String>, String)>) {
    let Statement::Call(c) = parse(sql).unwrap() else {
        panic!("{sql}: not a CALL");
    };
    let DdlName::Parts(name) = &c.name else {
        panic!("{sql}: a dynamic name");
    };
    let parts = name.parts.iter().map(|p| p.value.clone()).collect();
    let args = c
        .args
        .iter()
        .map(|a| {
            (
                a.name.as_ref().map(|n| n.value.clone()),
                sql[a.expr.span.start..a.expr.span.end].to_owned(),
            )
        })
        .collect();
    (parts, args)
}

#[test]
fn valid_calls_still_parse() {
    let sql = "CALL cat.system.expire_snapshots(table => 'db.t', retain_last => 1)";
    let (name, args) = call_of(sql);
    assert_eq!(name, ["cat", "system", "expire_snapshots"]);
    assert_eq!(
        args,
        [
            (Some("table".to_owned()), "'db.t'".to_owned()),
            (Some("retain_last".to_owned()), "1".to_owned()),
        ]
    );
    let Statement::Call(c) = parse(sql).unwrap() else {
        unreachable!()
    };
    assert!(matches!(
        &c.args[0].expr.kind,
        ExprKind::Literal(Literal::String(s)) if s == "db.t"
    ));
    let (name, args) =
        call_of("CALL system.p('db.t', TIMESTAMP '2020-01-01 00:00:00', array(1L, 2L))");
    assert_eq!(name, ["system", "p"]);
    assert_eq!(
        args,
        [
            (None, "'db.t'".to_owned()),
            (None, "TIMESTAMP '2020-01-01 00:00:00'".to_owned()),
            (None, "array(1L, 2L)".to_owned()),
        ]
    );
}

#[test]
fn a_call_without_arguments_parses() {
    let (name, args) = call_of("CALL cat.system.p()");
    assert_eq!(name, ["cat", "system", "p"]);
    assert!(args.is_empty());
}
