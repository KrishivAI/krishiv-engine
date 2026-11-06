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
fn pipe_group_by_position_out_of_range_is_refused() {
    let s = Session::new().unwrap();
    for n in ["5", "0"] {
        let sql = format!("values (1), (2) as t(x) |> aggregate count(*) group by {n}");
        let err = match block_on(s.sql(&sql)) {
            Ok(df) => block_on(df.collect()).err(),
            Err(e) => Some(e),
        };
        assert!(err.is_some(), "{sql} should fail");
    }
}

#[test]
fn a_group_by_number_beyond_int_is_a_constant_key() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "values (1), (2) as t(x) |> aggregate count(*) as c group by 2147483648",
    );
    assert_eq!(got, ["2147483648,2"]);
}

#[test]
fn set_with_a_generator_keeps_the_target_name() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select '{\"a\":1}' as x |> set x = json_tuple(x, 'a') |> select x"
        ),
        ["1"]
    );
}

/// The error `sql` fails with: its class and message parameters.
fn error(s: &Session, sql: &str) -> (String, Vec<(String, String)>) {
    let Err(e) = block_on(async { s.sql(sql).await?.collect().await }) else {
        return ("no error".to_owned(), Vec::new());
    };
    let params = e
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect();
    (e.error_class().name().to_owned(), params)
}

/// Spark's `ValidateAndStripPipeExpressions` (`pipe-operators.sql` 70, 72,
/// 242, 250, 311): `EXTEND` and `SET` items hold no aggregate function,
/// `AGGREGATE` items hold one; a window function is not an aggregate here.
#[test]
fn pipe_items_check_their_aggregate_functions() {
    let s = Session::new().unwrap();
    let src = format!("select * from {T}");
    let (class, params) = error(&s, &format!("{src} |> extend sum(x) as z"));
    assert_eq!(class, "PIPE_OPERATOR_CONTAINS_AGGREGATE_FUNCTION");
    assert!(
        params.iter().any(|(k, v)| k == "clause" && v == "EXTEND"),
        "{params:?}"
    );
    assert!(
        params
            .iter()
            .any(|(k, v)| k == "expr" && v.starts_with("sum(x#")),
        "{params:?}"
    );
    let (class, params) = error(&s, &format!("{src} |> set x = max(x)"));
    assert_eq!(class, "PIPE_OPERATOR_CONTAINS_AGGREGATE_FUNCTION");
    assert!(
        params.iter().any(|(k, v)| k == "clause" && v == "SET"),
        "{params:?}"
    );
    for sql in [
        format!("{src} |> aggregate x group by x"),
        format!("{src} |> aggregate x"),
    ] {
        assert_eq!(
            error(&s, &sql).0,
            "PIPE_OPERATOR_AGGREGATE_EXPRESSION_CONTAINS_NO_AGGREGATE_FUNCTION",
            "{sql}"
        );
    }
    let (class, params) = error(&s, &format!("{src} |> extend *"));
    assert_eq!(class, "INVALID_USAGE_OF_STAR_OR_REGEX");
    assert!(
        params
            .iter()
            .any(|(k, v)| k == "prettyName" && v == "expression `pipeexpression`"),
        "{params:?}"
    );
    // Allowed: an aggregate item with arithmetic around it, a window
    // function in EXTEND, and items named as their expressions.
    assert_eq!(
        rows(&s, &format!("{src} |> aggregate sum(x) + 1 as s")),
        ["2"]
    );
    assert_eq!(
        rows(
            &s,
            &format!("{src} |> extend sum(x) over () as w |> select w")
        ),
        ["1", "1"]
    );
    assert_eq!(
        rows(&s, &format!("{src} |> extend x + 1 |> select `(x + 1)`")),
        ["1", "2"]
    );
}

/// A generator's written alias in `AGGREGATE` names its column (Spark 4.2.0:
/// `y`, so the next operator reads `y`).
#[test]
fn an_aggregate_generator_keeps_its_alias() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "SELECT 1 AS x |> AGGREGATE explode(collect_list(x)) AS y |> SELECT y"
        ),
        ["1"]
    );
}

/// A generator with several names in `AGGREGATE` is nested in Spark's
/// `pipeexpression(..)` wrapper, so it is refused (Spark 4.2.0).
#[test]
fn an_aggregate_generator_with_several_names_is_nested() {
    let s = Session::new().unwrap();
    let (class, params) = error(
        &s,
        "SELECT 1 AS x |> AGGREGATE posexplode(collect_list(x)) AS (p, y)",
    );
    assert_eq!(class, "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS");
    assert!(
        params.iter().any(|(k, v)| k == "expression"
            && v == "\"pipeexpression(multialias(posexplode(collect_list(x))))\""),
        "{params:?}"
    );
}
