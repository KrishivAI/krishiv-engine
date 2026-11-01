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
                        arrow_schema::DataType::Int64 => c
                            .as_primitive::<arrow_array::types::Int64Type>()
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
        ),
        table(&[&["1", "10"], &["1", "20"], &["2", "NULL"], &["3", "NULL"]])
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

#[test]
fn star_does_not_pick_up_a_select_list_generators_column() {
    let s = Session::new().unwrap();
    // Spark: `a, b, col` (the generated column once, where the generator is).
    let df =
        block_on(s.sql("select *, explode(b) from values (1, array(10, 20)) as t(a, b)")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["a", "b", "col"]);
    let df = block_on(s.sql("select explode(b), * from values (1, array(10)) as t(a, b)")).unwrap();
    let schema = block_on(df.schema()).unwrap();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["col", "a", "b"]);
}

#[test]
fn inline_turns_a_null_struct_element_into_null_fields() {
    let s = Session::new().unwrap();
    for generator in ["inline", "inline_outer"] {
        assert_eq!(
            rows(
                &s,
                &format!("select {generator}(array(struct(1, 'a'), null))")
            ),
            table(&[&["1", "a"], &["NULL", "NULL"]]),
            "{generator}"
        );
    }
}

#[test]
fn outer_generators_keep_rows_of_maps_and_inline_too() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select a, key, value from values (1, cast(null as map<int, string>)), (2, map(3, 'c')) as t(a, m) \
             lateral view outer explode(m) g order by a"
        ),
        table(&[&["1", "NULL", "NULL"], &["2", "3", "c"]])
    );
    assert_eq!(
        rows(
            &s,
            "select a, c1 from values (1, cast(array() as array<struct<c1: int>>)) as t(a, s) \
             lateral view outer inline(s) g"
        ),
        table(&[&["1", "NULL"]])
    );
}

#[test]
fn a_generator_that_is_not_a_select_item_is_refused_as_spark_does() {
    let s = Session::new().unwrap();
    for (sql, class) in [
        (
            "select explode(explode(array(array(1, 2, 3))))",
            "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS",
        ),
        (
            "select 1 + explode(array(1, 2, 3))",
            "UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS",
        ),
        (
            "select * from values (1) order by explode(array(1, 2, 3))",
            "UNSUPPORTED_GENERATOR.OUTSIDE_SELECT",
        ),
        (
            "select * from values (array(1, 2, 3)) t(arr) where explode(arr) == 2",
            "UNSUPPORTED_GENERATOR.OUTSIDE_SELECT",
        ),
    ] {
        assert_eq!(error_of(&s, sql).error_class().name(), class, "{sql}");
    }
}

#[test]
fn a_generator_beside_group_by_runs_above_the_aggregate() {
    let s = Session::new().unwrap();
    // The argument is a grouping key under `GROUP BY ALL`.
    assert_eq!(
        rows(
            &s,
            "select explode(arr) as package from (values (array('a', 'b'))) as t(arr) \
             group by all order by package"
        ),
        table(&[&["a"], &["b"]])
    );
    // An ordinal counts the generator's argument.
    assert_eq!(
        rows(
            &s,
            "select id, explode(arr) from (values (42, array('a', 'b', 'c'))) as t(id, arr) \
             group by id, 2 order by 2"
        ),
        table(&[&["42", "a"], &["42", "b"], &["42", "c"]])
    );
    // HAVING reads the generated column.
    assert_eq!(
        rows(
            &s,
            "select explode(packages) as package from (values (array('a', 'b'))) as t(packages) \
             group by all having package in ('a')"
        ),
        table(&[&["a"]])
    );
}

#[test]
fn order_by_reads_a_grouping_column_the_generator_select_does_not_output() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select explode(arr) as e from (values (2, array('a')), (1, array('b'))) as t(k, arr) \
             group by k, arr order by k"
        ),
        table(&[&["b"], &["a"]])
    );
}

#[test]
fn stack_generates_rows_of_columns() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(&s, "select stack(2, 1, 2, 3)"),
        table(&[&["1", "2"], &["3", "NULL"]])
    );
    assert_eq!(
        rows(
            &s,
            "select * from stack(2, 1, 'a', 2, 'b') t(n, c) order by n"
        ),
        table(&[&["1", "a"], &["2", "b"]])
    );
    // The columns are named `col0`, `col1`, ...
    assert_eq!(
        rows(
            &s,
            "select col1 from stack(2, 1, 'a', 2, 'b') order by col1"
        ),
        table(&[&["a"], &["b"]])
    );
    let e = error_of(&s, "select stack()");
    assert_eq!(e.error_class(), ErrorClass::WrongNumArgsWithoutSuggestion);
}

#[test]
fn json_tuple_as_a_table() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'c')"
        ),
        table(&[&["1", "NULL"]])
    );
    assert_eq!(
        rows(
            &s,
            "select y from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'b') as t(x, y)"
        ),
        table(&[&["2"]])
    );
}

#[test]
fn generators_beside_group_by_name_their_columns_as_spark_does() {
    let s = Session::new().unwrap();
    let t = "(values (1, array('a', 'b'))) as t(k, arr)";
    // No alias: the generator's own column name.
    assert_eq!(
        rows(
            &s,
            &format!("select explode(arr) from {t} group by arr order by col")
        ),
        table(&[&["a"], &["b"]])
    );
    // A multi-column generator keeps its names and all its columns.
    assert_eq!(
        rows(
            &s,
            &format!("select k, posexplode(arr) from {t} group by k, arr order by pos")
        ),
        table(&[&["1", "0", "a"], &["1", "1", "b"]])
    );
    // A written alias renames the column.
    assert_eq!(
        rows(
            &s,
            &format!("select explode(arr) as x from {t} group by arr order by x")
        ),
        table(&[&["a"], &["b"]])
    );
    let e = error_of(
        &s,
        &format!("select explode(arr), explode(arr) from {t} group by arr"),
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedGeneratorMultiGenerator
    );
}

#[test]
fn stack_takes_a_bounded_int_row_count() {
    let s = Session::new().unwrap();
    assert_eq!(
        error_of(&s, "select stack(2147483647, 1)").error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
    // Spark: the count must be in (0, 2147483647].
    for sql in ["select stack(0, 1)", "select stack(-1, 1)"] {
        assert_eq!(
            error_of(&s, sql).error_class(),
            ErrorClass::DatatypeMismatchValueOutOfRange,
            "{sql}"
        );
    }
}

#[test]
fn stack_checks_its_count_and_column_types() {
    let s = Session::new().unwrap();
    assert_eq!(
        error_of(&s, "select stack(2L, 1, 2)").error_class(),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        error_of(&s, "select * from stack(2, 1, '1.1', 'a', 2, 2.2, 'b')").error_class(),
        ErrorClass::DatatypeMismatchStackColumnDiffTypes
    );
    // Spark compares each value with its column's first one: `(1)` and `(3)`.
    let e = error_of(&s, "select stack(3, 1, 2, 'a')").to_string();
    assert!(e.contains("(1)") && e.contains("(3)"), "{e}");
    // A count that is no INT is a type error, as in Spark.
    for sql in [
        "select stack(18446744073709551615, 1)",
        "select stack(1000000000000, 1)",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{sql}"
        );
    }
    // NULLs take any column type.
    assert_eq!(
        rows(
            &s,
            "select a, cast(b as string), c from stack(2, 1, 1.1, null, 2, null, 'b') t(a, b, c)"
        ),
        table(&[&["1", "1.1", "NULL"], &["2", "NULL", "b"]])
    );
}

#[test]
fn a_generator_relation_checks_its_alias_count() {
    let s = Session::new().unwrap();
    for sql in [
        "select * from explode(array(1, 2)) t(c1, c2)",
        "select * from posexplode(array(1, 2)) t(x)",
        "select * from inline(array(struct(1, 2))) t(a, b, c)",
        "select * from json_tuple('{\"a\": 1, \"b\": 2}', 'a', 'b') AS t(x)",
        "select * from stack(2, 1, 2, 3) t(a, b, c)",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class(),
            ErrorClass::NumTableValueAliasesMismatch,
            "{sql}"
        );
    }
}

#[test]
fn the_generator_argument_is_visible_to_group_by_as_gen_input_0() {
    let s = Session::new().unwrap();
    assert_eq!(
        rows(
            &s,
            "select id, explode(arr) from (values (1, array(7))) as t(id, arr) \
             group by id, _gen_input_0"
        ),
        table(&[&["1", "7"]])
    );
}

/// A session whose batches hold one row, so every generator output splits.
fn one_row_batches() -> Session {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.exec.batchSize", "1").unwrap();
    s
}

#[test]
fn a_batch_size_of_one_splits_every_row_and_loses_nothing() {
    let s = one_row_batches();
    assert_eq!(
        rows(
            &s,
            &format!(
                "select a, col from ({T}) lateral view outer explode(b) e as col order by a, col"
            )
        ),
        table(&[&["1", "10"], &["1", "20"], &["2", "NULL"], &["3", "NULL"]]),
    );
    assert_eq!(
        rows(
            &s,
            "select pos, col from (select posexplode(array('x', 'y', 'z')))"
        ),
        table(&[&["0", "x"], &["1", "y"], &["2", "z"]]),
    );
}

#[test]
fn counts_keep_their_weights_across_a_split() {
    let s = one_row_batches();
    assert_eq!(
        rows(
            &s,
            "select a, cast(count(*) as string) from (select 1 a, explode(array(1, 2, 3))) group by a"
        ),
        table(&[&["1", "3"]]),
    );
}

#[test]
fn inline_null_structs_split_mid_row_stay_null() {
    let s = one_row_batches();
    assert_eq!(
        rows(
            &s,
            "select inline(array(struct(1, 'a'), null, struct(3, 'c')))"
        ),
        table(&[&["1", "a"], &["NULL", "NULL"], &["3", "c"]]),
    );
}

#[test]
fn several_generators_in_one_select_make_a_cross_product() {
    let s = Session::new().unwrap();
    let mut got = rows(&s, "select explode(array(1, 2, 3)), explode(array(10, 20))");
    got.sort();
    assert_eq!(
        got,
        table(&[
            &["1", "10"],
            &["1", "20"],
            &["2", "10"],
            &["2", "20"],
            &["3", "10"],
            &["3", "20"],
        ])
    );
}

#[test]
fn a_generator_beside_an_aggregate_without_group_by_runs_above_it() {
    let s = Session::new().unwrap();
    let mut got = rows(
        &s,
        "select explode(collect_list(cast(id as int))) as val from range(3)",
    );
    got.sort();
    assert_eq!(got, table(&[&["0"], &["1"], &["2"]]));
    let mut got = rows(
        &s,
        "select explode(collect_list(cast(id as int))) as val, cast(count(*) as int) as cnt from range(3)",
    );
    got.sort();
    assert_eq!(got, table(&[&["0", "3"], &["1", "3"], &["2", "3"]]));
    let e = error_of(
        &s,
        "select explode(array(1, 2)), explode(array(3, 4)), count(*)",
    );
    assert!(
        e.to_string()
            .contains("UNSUPPORTED_GENERATOR.MULTI_GENERATOR"),
        "{e}"
    );
}

#[test]
fn a_generator_reads_a_lateral_alias_beside_an_aggregate() {
    let s = Session::new().unwrap();
    let mut got = rows(
        &s,
        "select col, cast(n as int) from (select array(1, 2, 3) as arr, explode(arr) as col, count(*) as n)",
    );
    got.sort();
    assert_eq!(got, table(&[&["1", "1"], &["2", "1"], &["3", "1"]]));
}

#[test]
fn a_generators_output_is_no_lateral_alias_beside_an_aggregate() {
    let s = Session::new().unwrap();
    for q in [
        "select explode(array(1, 2, 3)) as col, col + 1 as col2, count(*)",
        "select array(1, 2, 3) as arr, explode(arr) as col, count(*), col + 1 as col2",
    ] {
        let e = error_of(&s, q);
        assert!(
            e.to_string()
                .contains("UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"),
            "{q}: {e}"
        );
    }
}

#[test]
fn table_functions_mark_themselves_in_the_alias_count_error() {
    let s = Session::new().unwrap();
    // A table function: NUM_TABLE_VALUE_ALIASES_MISMATCH.
    for sql in [
        "select * from range(3) t(a, b)",
        "select * from json_tuple('{\"a\":1}', 'a') t(x, y)",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class(),
            ErrorClass::NumTableValueAliasesMismatch,
            "{sql}"
        );
    }
    // A subquery that happens to select the same call: ASSIGNMENT_ARITY_MISMATCH.
    for sql in [
        "select * from (select json_tuple('{\"a\":1}', 'a', 'b')) t(x)",
        "select * from (select explode(array(1, 2))) t(a, b)",
    ] {
        assert_eq!(
            error_of(&s, sql).error_class(),
            ErrorClass::AssignmentArityMismatch,
            "{sql}"
        );
    }
    assert_eq!(
        rows(&s, "select * from json_tuple('{\"a\":1}', 'a') t(x)"),
        table(&[&["1"]])
    );
}

#[test]
fn an_aggregate_in_order_by_cannot_grow_through_a_generator() {
    let s = Session::new().unwrap();
    // Spark 4.2.0: the call stays in the sort, which refuses it.
    let e = error_of(
        &s,
        "select explode(array(k)) e from values (1), (2), (2) t(k) group by k order by count(*)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedExprForOperator,
        "{e}"
    );
    assert!(e.to_string().contains("count(1)"), "{e}");
}

#[test]
fn stack_beside_aggregates_and_in_expressions() {
    let s = Session::new().unwrap();
    // Spark 4.2.0.
    let e = error_of(&s, "select stack(2, 1, 2) + 1");
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedGeneratorNestedInExpressions,
        "{e}"
    );
    assert_eq!(
        rows(&s, "select stack(cast(2 as int), 1, 2, 3, 4)"),
        table(&[&["1", "2"], &["3", "4"]])
    );
    let mut got = rows(
        &s,
        "select stack(2, k, k + 1) from values (1), (2) t(k) group by k",
    );
    got.sort();
    assert_eq!(got, table(&[&["1"], &["2"], &["2"], &["3"]]));
    let mut got = rows(
        &s,
        "select k, count(*), stack(2, k, k + 1) from values (1), (2) t(k) group by k",
    );
    got.sort();
    assert_eq!(
        got,
        table(&[
            &["1", "1", "1"],
            &["1", "1", "2"],
            &["2", "1", "2"],
            &["2", "1", "3"]
        ])
    );
    let mut got = rows(
        &s,
        "select count(*), stack(2, 1, 2) from values (1), (2) t(k)",
    );
    got.sort();
    assert_eq!(got, table(&[&["2", "1"], &["2", "2"]]));
}

#[test]
fn having_cannot_read_an_aggregate_the_generator_hides() {
    let s = Session::new().unwrap();
    // Spark 4.2.0: INVALID_WHERE_CONDITION.
    assert_eq!(
        error_of(
            &s,
            "select explode(array(k)) e from values (1), (2), (2) t(k) group by k having count(*) > 1"
        )
        .error_class(),
        ErrorClass::InvalidWhereCondition
    );
}

#[test]
fn an_aliased_stack_beside_group_by_and_json_tuple_named_arguments() {
    let s = Session::new().unwrap();
    let mut got = rows(
        &s,
        "select stack(2, k, k + 1) as v from values (1), (2) t(k) group by k",
    );
    got.sort();
    assert_eq!(got, table(&[&["1"], &["2"], &["2"], &["3"]]));
    // A named argument to a table-valued json_tuple is refused, as for the
    // other table functions.
    assert_eq!(
        error_of(&s, "select * from json_tuple('{\"a\": 1}', x => 'a')").error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn order_by_an_aggregate_over_a_generator_below_it() {
    let s = Session::new().unwrap();
    // The generator feeds the aggregate; nothing sits between sort and aggregate.
    assert_eq!(
        rows(
            &s,
            "SELECT k, sum(k + 1.0) FROM (SELECT explode(array(1, 2)) AS k) t GROUP BY k ORDER BY sum(k + 1.0)"
        )
        .len(),
        2
    );
}
