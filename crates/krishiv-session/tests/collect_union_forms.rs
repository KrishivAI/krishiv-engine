// SPDX-License-Identifier: Apache-2.0
//! `collect_union` forms of Spark 4.3 (review of 4.3-A): `RESPECT NULLS` / `IGNORE NULLS`, the window form (refused
//! with a named error: no executor maps the aggregate), and the refused element types. The expected values are
//! the ones a Spark 4.3.0 run gave.
#![cfg(test)]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Result<Vec<RecordBatch>, krishiv_common::error::KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await })
}

fn text(s: &Session, sql: &str) -> String {
    let b = run(s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let col = arrow_cast::cast::cast(b[0].column(0), &arrow_schema::DataType::Utf8).unwrap();
    col.as_string::<i32>().value(0).to_owned()
}

const VALUES: &str = "from values (array(1,null)),(array(2)) t(x)";

#[test]
fn ignore_nulls_is_the_default() {
    let s = Session::new().unwrap();
    assert_eq!(
        text(&s, &format!("select sort_array(collect_union(x)) {VALUES}")),
        "[1, 2]"
    );
    assert_eq!(
        text(
            &s,
            &format!("select sort_array(collect_union(x) ignore nulls) {VALUES}")
        ),
        "[1, 2]"
    );
}

#[test]
fn respect_nulls_keeps_the_null_elements() {
    let s = Session::new().unwrap();
    // NULL prints as an empty cell; the order of the collected rows is not fixed.
    let out = text(
        &s,
        &format!("select collect_union(x) respect nulls {VALUES}"),
    );
    let mut cells: Vec<_> = out
        .trim_matches(['[', ']'])
        .split(", ")
        .map(str::to_owned)
        .collect();
    cells.sort();
    assert_eq!(cells, ["", "1", "2"], "{out}");
}

#[test]
fn the_window_form_is_refused_with_a_named_error() {
    let s = Session::new().unwrap();
    let e = run(
        &s,
        "select collect_union(x) over (order by k) from values (1, array(1)),(2, array(2)) t(k, x)",
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("collect_union"), "{e}");
}

#[test]
fn an_array_of_maps_is_refused() {
    let s = Session::new().unwrap();
    // A collated string inside an array is not supported by the engine at all yet (the same check covers it).
    let sql = "select collect_union(x) from values (array(map(1,2))) t(x)";
    let e = run(&s, sql).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnsupportedInputType,
        "{sql}: {e}"
    );
    assert!(e.to_string().contains("collect_union"), "{sql}: {e}");
    assert!(e.to_string().contains("COLLATED STRING"), "{sql}: {e}");
}

#[test]
fn a_user_written_second_argument_is_an_arity_error() {
    let s = Session::new().unwrap();
    for sql in [
        "select collect_union(x, false) from values (array(1,null)),(array(2)) t(x)",
        "select collect_union(x, true) respect nulls from values (array(1)) t(x)",
        "select builtin.collect_union(x, false) from values (array(1,null)),(array(2)) t(x)",
        "select system.builtin.collect_union(x, true) respect nulls from values (array(1)) t(x)",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::WrongNumArgsWithoutSuggestion,
            "{sql}: {e}"
        );
        assert!(e.to_string().contains("collect_union"), "{sql}: {e}");
    }
}
