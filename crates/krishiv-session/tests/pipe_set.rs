// SPDX-License-Identifier: Apache-2.0
//! `|> SET`, `|> AGGREGATE ... GROUP BY n` and `|> SELECT ... GROUP BY`
//! end to end; the rows are Spark 4.2.0's (`pipe-operators.sql`).
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The rows of `sql`, each cell as text.
fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| match c.data_type() {
                    arrow_schema::DataType::Int32 => c
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .value(i)
                        .to_string(),
                    arrow_schema::DataType::Int64 => c
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(i)
                        .to_string(),
                    arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                    other => format!("<{other}>"),
                })
                .collect();
            out.push(cells.join(","));
        }
    }
    out
}

const T: &str = "values (0, 'abc'), (1, 'def') as t(x, y)";

#[test]
fn set_replaces_a_column_in_place() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(&s, &format!("select * from {T} |> set x = 1")),
        ["1,abc", "1,def"]
    );
    // Later assignments read earlier ones, and a repeated key updates as it goes.
    assert_eq!(
        rows(
            &s,
            &format!(
                "select * from {T} |> extend 1 as z |> set z = x + length(y), z = z + 1 |> order by x"
            )
        ),
        ["0,abc,4", "1,def,5"]
    );
}

#[test]
fn set_of_a_missing_column_is_unresolved() {
    let s = Session::new().unwrap();
    let e = block_on(async {
        s.sql(&format!("select * from {T} |> set z = 1"))
            .await?
            .collect()
            .await
    })
    .expect_err("set of a missing column");
    assert!(e.to_string().contains("UNRESOLVED_COLUMN"), "{e}");
}

#[test]
fn aggregate_group_by_numbers_name_input_columns() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select 3 as x, 4 as y, 5 as z |> aggregate sum(y) group by 3"
        ),
        ["5,4"]
    );
    assert_eq!(
        rows(
            &s,
            "select 3 as x, 4 as y |> aggregate sum(y) group by 1, 1"
        ),
        ["3,3,4"]
    );
}

#[test]
fn select_with_group_by_aggregates() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "values (1, 1), (1, 2), (2, 4) as o(a, b) |> select a, sum(b) as sum_b group by 1 |> order by a"
        ),
        ["1,3", "2,4"]
    );
}

#[test]
fn set_replaces_every_duplicate_and_compares_names_like_the_analyzer() {
    let s = Session::new().unwrap();
    assert_eq!(rows(&s, "select 1 as x, 2 as x |> set x = 0"), ["0,0"]);
    assert_eq!(rows(&s, "select 1 as `É` |> set `é` = 2"), ["2"]);
}

#[test]
fn a_suffixed_literal_in_pipe_group_by_is_a_constant() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "values (1), (2) as t(x) |> aggregate count(*) as c group by 1L"
        ),
        ["1,2"]
    );
}

#[test]
fn plans_differing_only_in_a_set_value_are_different() {
    let s = Session::new().unwrap();
    assert_eq!(rows(&s, "select 0 as x |> set x = 1"), ["1"]);
    assert_eq!(rows(&s, "select 0 as x |> set x = 2"), ["2"]);
}

#[test]
fn pipe_group_by_position_out_of_range_and_huge_are_refused() {
    let s = Session::new().unwrap();
    for n in ["5", "0", "99999999999999999999"] {
        let sql = format!("values (1), (2) as t(x) |> aggregate count(*) group by {n}");
        let err = match block_on(s.sql(&sql)) {
            Ok(df) => block_on(df.collect()).err(),
            Err(e) => Some(e),
        };
        assert!(err.is_some(), "{sql} should fail");
    }
}
