// SPDX-License-Identifier: Apache-2.0
//! `regexp_extract_all` and the output names of the regular-expression
//! functions with a defaulted argument. Expected values and names come from
//! the Spark 4.2.0 oracle and `regexp-functions.sql`.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

/// The single column of `sql` as one string per row: array items joined by
/// `|` in brackets, `NULL` for a NULL.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches: Vec<RecordBatch> = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        let list = b.column(0).as_list::<i32>();
        for i in 0..list.len() {
            out.push(if list.is_null(i) {
                "NULL".to_owned()
            } else {
                let items = list.value(i);
                let items = items.as_string_view();
                let parts: Vec<&str> = (0..items.len()).map(|j| items.value(j)).collect();
                format!("[{}]", parts.join("|"))
            });
        }
    }
    out
}

fn name_of(s: &Session, sql: &str) -> String {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    block_on(df.schema()).unwrap().field(0).name().clone()
}

#[test]
fn extracts_every_match() {
    let s = Session::new().unwrap();
    let r = |q: &str| rows(&s, &format!("select {q}"));
    assert_eq!(
        r("regexp_extract_all('1a 2b 14m', '\\\\d+', 0)"),
        ["[1|2|14]"]
    );
    assert_eq!(
        r("regexp_extract_all('1a 2b 14m', '(\\\\d+)([a-z]+)', 0)"),
        ["[1a|2b|14m]"]
    );
    assert_eq!(
        r("regexp_extract_all('1a 2b 14m', '(\\\\d+)([a-z]+)')"),
        ["[1|2|14]"]
    );
    assert_eq!(
        r("regexp_extract_all('1a 2b 14m', '(\\\\d+)([a-z]+)', 2)"),
        ["[a|b|m]"]
    );
    // A group that took no part is the empty string.
    assert_eq!(
        r("regexp_extract_all('a 2b 14m', '(\\\\d+)?([a-z]+)', 1)"),
        ["[|2|14]"]
    );
    assert_eq!(
        r("regexp_extract_all('a 2b 14m', '(\\\\d+)?', 1)"),
        ["[||2|||14||]"]
    );
    assert_eq!(r("regexp_extract_all('xyz', '(a)', 1)"), ["[]"]);
    assert_eq!(r("regexp_extract_all('ab', '', 0)"), ["[||]"]);
}

#[test]
fn nulls_and_columns() {
    let s = Session::new().unwrap();
    let r = |q: &str| rows(&s, q);
    assert_eq!(r("select regexp_extract_all(NULL, 'a')"), ["NULL"]);
    assert_eq!(r("select regexp_extract_all('a', NULL)"), ["NULL"]);
    assert_eq!(
        r("select regexp_extract_all('abab', '(a)(b)', NULL)"),
        ["NULL"]
    );
    assert_eq!(
        r("select regexp_extract_all(c, '(\\\\d)(\\\\w)', 2) from values ('1a2b'), (NULL) t(c)"),
        ["[a|b]", "NULL"]
    );
}

#[test]
fn errors_carry_the_function_name() {
    let s = Session::new().unwrap();
    let class_of = |sql: &str| {
        let df = block_on(s.sql(sql));
        let err = match df {
            Ok(df) => block_on(df.collect()).expect_err(sql),
            Err(e) => e,
        };
        err.error_class()
    };
    for q in [
        "select regexp_extract_all('1a 2b 14m', '\\\\d+')",
        "select regexp_extract_all('1a 2b 14m', '\\\\d+', -1)",
        "select regexp_extract_all('1a 2b 14m', '(\\\\d+)([a-z]+)', 3)",
    ] {
        assert_eq!(
            class_of(q),
            ErrorClass::InvalidParameterValueRegexGroupIndex,
            "{q}"
        );
    }
    assert_eq!(
        class_of("select regexp_extract_all('abc', col0, 1) from values('], [') as t(col0)"),
        ErrorClass::InvalidParameterValuePattern
    );
}

#[test]
fn output_names_show_the_defaults() {
    let s = Session::new().unwrap();
    let n = |q: &str| name_of(&s, &format!("select {q}"));
    assert_eq!(
        n("regexp_extract_all('a', 'a', 0)"),
        "regexp_extract_all(a, a, 0)"
    );
    assert_eq!(
        n("regexp_extract_all('a', '(a)')"),
        "regexp_extract_all(a, (a), 1)"
    );
    assert_eq!(n("regexp_extract('a', '(a)')"), "regexp_extract(a, (a), 1)");
    assert_eq!(n("regexp_instr('abc', 'b')"), "regexp_instr(abc, b, 0)");
    assert_eq!(n("regexp_instr('abc', null)"), "regexp_instr(abc, NULL, 0)");
    assert_eq!(
        n("regexp_replace('a', 'a', 'b')"),
        "regexp_replace(a, a, b, 1)"
    );
    assert_eq!(n("regexp_like('a', 'a')"), "REGEXP_LIKE(a, a)");
    assert_eq!(n("regexp('a', 'a')"), "REGEXP(a, a)");
    assert_eq!(n("rlike('a', 'a')"), "RLIKE(a, a)");
}

/// Whether the list items of the single column of `sql` may be null.
fn items_nullable(s: &Session, sql: &str) -> bool {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let schema = block_on(df.schema()).unwrap();
    match schema.field(0).data_type() {
        arrow_schema::DataType::List(f) => f.is_nullable(),
        other => panic!("{sql}: not a list: {other}"),
    }
}

/// Spark types the result `ArrayType(subject.dataType)`, so items may be
/// null (`RegExpExtractAll`), while `split` promises non-null items.
#[test]
fn regexp_extract_all_items_are_nullable() {
    let s = Session::new().unwrap();
    assert!(items_nullable(&s, "SELECT regexp_extract_all('a', '(a)')"));
    assert!(!items_nullable(&s, "SELECT split('a', 'b')"));
}
