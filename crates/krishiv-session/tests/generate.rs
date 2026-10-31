// SPDX-License-Identifier: Apache-2.0
//! Generators (`explode`, `posexplode`, `inline` and their `_outer` forms)
//! end to end: in the select list, as a table and in `LATERAL VIEW`. Rows
//! and errors are Spark 4.2.0's.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

/// The rows of `sql`, each cell as text (`NULL` for a null).
fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for c in b.columns() {
                // Cast through the session so every type prints alike.
                let text = if c.is_null(i) {
                    "NULL".to_owned()
                } else {
                    match c.data_type() {
                        arrow_schema::DataType::Int32 => c
                            .as_primitive::<arrow_array::types::Int32Type>()
                            .value(i)
                            .to_string(),
                        arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                        other => format!("<{other}>"),
                    }
                };
                row.push(text);
            }
            out.push(row);
        }
    }
    out
}

fn table(rows: &[&[&str]]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|r| r.iter().map(|c| (*c).to_owned()).collect())
        .collect()
}

fn error_of(s: &Session, sql: &str) -> krishiv_common::error::KrishivError {
    block_on(async { s.sql(sql).await?.collect().await })
        .err()
        .unwrap_or_else(|| panic!("{sql} ran"))
}

const T: &str = "values (1, array(10, 20)), (2, array()), (3, cast(null as array<int>)) as t(a, b)";

#[test]
fn lateral_view_drops_empty_rows_unless_outer() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            &format!("select a, e from {T} lateral view explode(b) x as e order by a, e")
        ),
        table(&[&["1", "10"], &["1", "20"]])
    );
    assert_eq!(
        rows(
            &s,
            &format!("select a, e from {T} lateral view outer explode(b) x as e order by a, e")
        ),
        table(&[&["1", "10"], &["1", "20"], &["2", "NULL"], &["3", "NULL"]])
    );
    // `explode_outer` is the same as `OUTER explode`.
    assert_eq!(
        rows(
            &s,
            &format!("select a, e from {T} lateral view explode_outer(b) x as e order by a, e")
        )
        .len(),
        4
    );
}

#[test]
fn posexplode_numbers_from_zero_and_is_null_for_the_outer_row() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select a, x.pos, x.e from values (1, array(10, 20)), (2, array()) as t(a, b)  lateral view outer posexplode(b) x as pos, e order by a, pos"
        ),
        table(&[&["1", "0", "10"], &["1", "1", "20"], &["2", "NULL", "NULL"]])
    );
    assert_eq!(
        rows(&s, "select * from posexplode(array('a', 'b'))"),
        table(&[&["0", "a"], &["1", "b"]])
    );
}

#[test]
fn a_map_explodes_to_key_and_value_and_an_array_of_structs_to_fields() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select a, key, value from values (1, map(1, 'a', 2, 'b')) as t(a, m)  lateral view explode(m) g order by key"
        ),
        table(&[&["1", "1", "a"], &["1", "2", "b"]])
    );
    assert_eq!(
        rows(&s, "select inline(array(struct(1, 'a'), struct(2, 'b')))"),
        table(&[&["1", "a"], &["2", "b"]])
    );
    assert_eq!(
        rows(
            &s,
            "select a, c1, c2 from values (1, array(struct(1, 'x'), struct(2, 'y'))) as t(a, s)  lateral view inline(s) g as c1, c2 order by c1"
        ),
        table(&[&["1", "1", "x"], &["1", "2", "y"]])
    );
}

#[test]
fn generators_in_the_select_list_and_as_tables() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(&s, "select explode(array(1, 2)) as e, 'k' as k"),
        table(&[&["1", "k"], &["2", "k"]])
    );
    assert_eq!(
        rows(
            &s,
            "select a, explode(b) from values (1, array(1, 2)) as t(a, b) where a = 1"
        ),
        table(&[&["1", "1"], &["1", "2"]])
    );
    assert_eq!(
        rows(&s, "select * from explode(array(1, 2, 3)) t(x)"),
        table(&[&["1"], &["2"], &["3"]])
    );
    assert_eq!(
        rows(&s, "select posexplode(array(1, 2)) as (p, e)"),
        table(&[&["0", "1"], &["1", "2"]])
    );
    // Several generators multiply, the first the outer loop.
    assert_eq!(
        rows(&s, "select explode(array(1, 2)), explode(array(3, 4))"),
        table(&[&["1", "3"], &["1", "4"], &["2", "3"], &["2", "4"]])
    );
    // A filter on the generated column runs above the generator.
    assert_eq!(
        rows(
            &s,
            "select a, col from values (1, array(1, 2)) as t(a, b) lateral view explode(b) g where col > 1"
        ),
        table(&[&["1", "2"]])
    );
    assert_eq!(
        rows(&s, "select explode_outer(cast(null as array<int>))"),
        table(&[&["NULL"]])
    );
}

#[test]
fn the_argument_must_be_an_array_or_a_map_and_the_aliases_must_fit() {
    let s = Session::new().unwrap();
    for sql in [
        "select explode(1)",
        "select explode(struct(1, 'a'))",
        "select posexplode(1)",
        "select inline(1)",
        "select inline(array(1, 'a'))",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{sql}"
        );
    }
    let err = error_of(&s, "select explode(array(1, 2)) as (a, b)");
    assert_eq!(err.error_class(), ErrorClass::UdtfAliasNumberMismatch);
    assert!(
        err.to_string().contains("Expected 1 aliases, but got a,b."),
        "{err}"
    );
    let err = error_of(&s, "select posexplode(array(1, 2)) as e");
    assert!(
        err.to_string().contains("Expected 2 aliases, but got e."),
        "{err}"
    );
    // A generator takes one argument.
    assert_eq!(
        error_of(&s, "select explode(array(1), array(2))").error_class(),
        ErrorClass::WrongNumArgsWithoutSuggestion
    );
    // Other generators are not lowered yet.
    assert_eq!(
        error_of(
            &s,
            "select * from values (1) as t(a) lateral view stack(2, 1, 2) g"
        )
        .error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn aggregates_count_each_generated_row_once() {
    let s = Session::new().unwrap();
    let out = block_on(async {
        s.sql("select count(*) as n from values (1, array(1, 2, 3)), (2, array(4)) as t(a, b) lateral view explode(b) g")
            .await?
            .collect()
            .await
    })
    .unwrap();
    let n = out[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0);
    assert_eq!(n, 4);
}
