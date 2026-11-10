// SPDX-License-Identifier: Apache-2.0
//! Operators over collated values run on collation keys and carry the originals (collations unit
//! 3b, ADR 0043). Every expected value here is the answer of a real Spark 4.2.0 run.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

/// The rows of `sql`, each column as text, joined by `|`.
fn rows_in(s: &Session, sql: &str) -> Result<Vec<String>, KrishivError> {
    let batches = block_on(block_on(s.sql(sql))?.collect())?;
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| {
                    if c.is_null(i) {
                        "NULL".to_owned()
                    } else if let Some(v) = c.as_string_view_opt() {
                        v.value(i).to_owned()
                    } else if let Some(v) = c.as_string_opt::<i32>() {
                        v.value(i).to_owned()
                    } else if let Some(v) = c.as_primitive_opt::<arrow_array::types::Int64Type>() {
                        v.value(i).to_string()
                    } else if let Some(v) = c.as_primitive_opt::<arrow_array::types::Int32Type>() {
                        v.value(i).to_string()
                    } else if let Some(v) = c.as_boolean_opt() {
                        v.value(i).to_string()
                    } else {
                        format!("{c:?}")
                    }
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    Ok(out)
}

fn rows(sql: &str) -> Vec<String> {
    let s = Session::new().unwrap();
    rows_in(&s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn sorted(sql: &str) -> Vec<String> {
    let mut r = rows(sql);
    r.sort();
    r
}

const T: &str =
    "(SELECT c COLLATE utf8_lcase AS c, x FROM VALUES ('b',1),('A',2),('a',3),('B',4) t(c,x))";

#[test]
fn grouping_outputs_the_first_value_of_each_key() {
    assert_eq!(
        rows(&format!(
            "SELECT c, count(*) FROM {T} GROUP BY c ORDER BY c"
        )),
        ["A|2", "b|2"]
    );
    assert_eq!(
        rows(&format!(
            "SELECT upper(c), count(*) FROM {T} GROUP BY c ORDER BY 2, 1"
        )),
        ["A|2", "B|2"]
    );
    assert_eq!(
        rows(
            "SELECT DISTINCT c COLLATE utf8_lcase FROM VALUES ('b'),('A'),('a'),('B') t(c) ORDER BY 1"
        ),
        ["A", "b"]
    );
    assert_eq!(
        sorted(
            "SELECT DISTINCT c FROM (SELECT c COLLATE utf8_binary_rtrim c FROM VALUES ('a '),('a'),('b') t(c))"
        ),
        ["a ", "b"]
    );
}

#[test]
fn grouping_sets_group_by_key() {
    for g in ["ROLLUP(c)", "CUBE(c)", "GROUPING SETS ((c), ())"] {
        assert_eq!(
            sorted(&format!("SELECT c, count(*) FROM {T} GROUP BY {g}")),
            ["A|2", "NULL|4", "b|2"],
            "{g}"
        );
    }
}

#[test]
fn min_and_max_keep_the_first_of_equal_keys_and_max_by_the_last() {
    let q = |v: &str, f: &str| {
        rows(&format!(
            "SELECT {f}(c COLLATE utf8_lcase) FROM VALUES {v} t(c)"
        ))
    };
    for f in ["max", "min"] {
        assert_eq!(q("('a'),('A')", f), ["a"], "{f}");
        assert_eq!(q("('A'),('a')", f), ["A"], "{f}");
    }
    assert_eq!(q("('b'),('A'),('B'),('a')", "max"), ["b"]);
    assert_eq!(q("('b'),('A'),('B'),('a')", "min"), ["A"]);
    assert_eq!(
        rows(
            "SELECT max(c COLLATE utf8_binary_rtrim), min(c COLLATE utf8_binary_rtrim) FROM VALUES ('a '),('a') t(c)"
        ),
        ["a |a "]
    );
    assert_eq!(
        rows(&format!("SELECT min_by(x, c), max_by(x, c) FROM {T}")),
        ["3|4"]
    );
    assert_eq!(
        rows(&format!("SELECT max(c) FILTER (WHERE x > 1) FROM {T}")),
        ["B"]
    );
}

#[test]
fn distinct_aggregates_count_keys() {
    assert_eq!(
        rows(
            "SELECT count(DISTINCT c COLLATE utf8_lcase) FROM VALUES ('a'),('A'),('b'),(NULL) t(c)"
        ),
        ["2"]
    );
    assert_eq!(
        rows(&format!("SELECT count(DISTINCT c, x) FROM {T}")),
        ["4"]
    );
    assert_eq!(
        rows(&format!(
            "SELECT sum(DISTINCT x), count(DISTINCT c) FROM {T}"
        )),
        ["10|2"]
    );
    assert_eq!(
        rows(&format!("SELECT approx_count_distinct(c) FROM {T}")),
        ["2"]
    );
    assert_eq!(
        rows(&format!("SELECT count(DISTINCT array(c)) FROM {T}")),
        ["2"]
    );
}

#[test]
fn listagg_orders_and_deduplicates_by_key() {
    assert_eq!(
        rows("SELECT lower(listagg(DISTINCT c COLLATE utf8_lcase)) FROM VALUES ('b'),('a'),('A'),('B') t(c)")
            .iter()
            .map(|s| {
                let mut v: Vec<char> = s.chars().collect();
                v.sort_unstable();
                v.into_iter().collect::<String>()
            })
            .collect::<Vec<_>>(),
        ["ab"]
    );
    assert_eq!(
        rows(
            "SELECT lower(listagg(c)  WITHIN GROUP (ORDER BY c COLLATE utf8_lcase)) FROM VALUES ('b'),('a'),('A'),('B') t(c)"
        ),
        ["aabb"]
    );
    // The DISTINCT form orders by key too (Spark 4.2.0: `aB`, bytes would give `Ba`).
    assert_eq!(
        rows(
            "SELECT listagg(DISTINCT c) WITHIN GROUP (ORDER BY c) FROM (SELECT c COLLATE utf8_lcase c FROM VALUES ('a'),('B') t(c))"
        ),
        ["aB"]
    );
}

#[test]
fn collect_set_refuses_a_collated_string_as_spark() {
    let s = Session::new().unwrap();
    let e = rows_in(&s, &format!("SELECT collect_set(c) FROM {T}")).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchUnsupportedInputType
    );
}

#[test]
fn sorts_and_windows_use_the_key() {
    assert_eq!(
        rows(&format!(
            "SELECT x, rank() OVER (ORDER BY c) FROM {T} ORDER BY x"
        )),
        ["1|3", "2|1", "3|1", "4|3"]
    );
    assert_eq!(
        rows(&format!(
            "SELECT x, row_number() OVER (PARTITION BY c ORDER BY x) FROM {T} ORDER BY x"
        )),
        ["1|1", "2|1", "3|2", "4|2"]
    );
    assert_eq!(
        rows(&format!(
            "SELECT x, max(c) OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM {T} ORDER BY x"
        )),
        ["1|b", "2|b", "3|b", "4|b"]
    );
    assert_eq!(
        rows("SELECT c, row_number() OVER (ORDER BY c) r FROM (SELECT c COLLATE utf8_lcase_rtrim c FROM VALUES ('A '),('a'),('b') t(c)) ORDER BY r")
            .last()
            .cloned(),
        Some("b|3".to_owned())
    );
}

#[test]
fn joins_and_set_operations_compare_keys() {
    let l = "(SELECT c COLLATE utf8_lcase c, 'l' s FROM VALUES ('a'),('B') t(c))";
    let r = "(SELECT c COLLATE utf8_lcase c, 'r' t FROM VALUES ('A'),('b'),('a') t(c))";
    assert_eq!(
        sorted(&format!("SELECT * FROM {l} a JOIN {r} b USING (c)")),
        ["B|l|r", "a|l|r", "a|l|r"]
    );
    assert_eq!(
        sorted(&format!("SELECT * FROM {l} a NATURAL JOIN {r} b")),
        ["B|l|r", "a|l|r", "a|l|r"]
    );
    let v = |list: &str| format!("SELECT c COLLATE utf8_lcase FROM VALUES {list} t(c)");
    let count = |sql: String| rows(&format!("SELECT count(*) FROM ({sql})"));
    assert_eq!(
        count(format!("{} UNION {}", v("('a'),('A')"), v("('A'),('b')"))),
        ["2"]
    );
    assert_eq!(
        count(format!("{} INTERSECT {}", v("('a'),('b')"), v("('A')"))),
        ["1"]
    );
    assert_eq!(
        count(format!("{} EXCEPT {}", v("('a'),('b')"), v("('A')"))),
        ["1"]
    );
    assert_eq!(
        count(format!(
            "{} INTERSECT ALL {}",
            v("('a'),('A'),('b')"),
            v("('A'),('a')")
        )),
        ["2"]
    );
    assert_eq!(
        count(format!(
            "{} EXCEPT ALL {}",
            v("('a'),('A'),('b')"),
            v("('A')")
        )),
        ["2"]
    );
}

#[test]
fn scalars_pick_by_key_and_keep_the_first_of_equals() {
    assert_eq!(
        rows(
            "SELECT greatest('a' COLLATE utf8_lcase, 'A'), greatest('A' COLLATE utf8_lcase, 'a'), least('a' COLLATE utf8_lcase, 'A'), least('A' COLLATE utf8_lcase, 'a')"
        ),
        ["a|A|a|A"]
    );
    assert_eq!(
        rows(&format!("SELECT greatest(c, 'B', 'a') FROM {T} ORDER BY x")),
        ["b", "B", "B", "B"]
    );
    assert_eq!(
        rows(
            "SELECT typeof(greatest('a' COLLATE utf8_lcase, 'A')), typeof(nullif('a' COLLATE utf8_lcase, 'A'))"
        ),
        ["string collate UTF8_LCASE|string collate UTF8_LCASE"]
    );
    assert_eq!(
        rows(
            "SELECT nullif('a' COLLATE utf8_lcase, 'A'), nullif('a' COLLATE utf8_lcase, 'b'), nullif('a ' COLLATE utf8_binary_rtrim, 'a')"
        ),
        ["NULL|a|NULL"]
    );
    assert_eq!(
        rows(
            "SELECT equal_null('a' COLLATE utf8_lcase, 'A'), equal_null(NULL COLLATE utf8_lcase, 'A'), equal_null(NULL COLLATE utf8_lcase, NULL)"
        ),
        ["true|false|true"]
    );
}

#[test]
fn between_coerces_each_half_on_its_own() {
    assert_eq!(
        rows("SELECT 'b' BETWEEN 'A' COLLATE utf8_binary AND 'C' COLLATE utf8_lcase"),
        ["true"]
    );
    assert_eq!(
        rows(
            "SELECT c BETWEEN 'A' COLLATE utf8_binary AND 'c' FROM (SELECT 'B' COLLATE utf8_lcase c)"
        ),
        ["true"]
    );
    assert_eq!(
        rows(
            "SELECT c BETWEEN 'a' AND 'C' COLLATE utf8_binary FROM (SELECT 'B' COLLATE utf8_lcase c)"
        ),
        ["true"]
    );
    let s = Session::new().unwrap();
    let e = rows_in(
        &s,
        "SELECT 'b' COLLATE utf8_lcase BETWEEN 'A' COLLATE utf8_binary AND 'c'",
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::CollationMismatchExplicit);
}

#[test]
fn nested_values_group_and_compare_by_key() {
    assert_eq!(
        rows("SELECT count(*) FROM VALUES ('a'),('A') t(c) GROUP BY array(c COLLATE utf8_lcase)"),
        ["2"]
    );
    assert_eq!(
        rows(
            "SELECT array(c COLLATE utf8_lcase) = array('A'), struct(c COLLATE utf8_lcase) = struct('A') FROM VALUES ('a') t(c)"
        ),
        ["true|true"]
    );
}

#[test]
fn map_lookups_compare_keys() {
    assert_eq!(
        rows(
            "SELECT map('a' COLLATE utf8_lcase, 1)['A' COLLATE utf8_lcase], element_at(map('a' COLLATE utf8_lcase, 1), 'A' COLLATE utf8_lcase), map_contains_key(map('a' COLLATE utf8_lcase, 1), 'A' COLLATE utf8_lcase)"
        ),
        ["1|1|true"]
    );
    assert_eq!(
        rows(
            "SELECT map(c COLLATE utf8_binary_rtrim, x)['A  ' COLLATE utf8_binary_rtrim] FROM VALUES ('A',1) t(c,x)"
        ),
        ["1"]
    );
}

#[test]
fn user_visible_hashes_stay_on_the_raw_bytes() {
    assert_eq!(
        rows(
            "SELECT hash(c COLLATE utf8_lcase), xxhash64(c COLLATE utf8_lcase) FROM VALUES ('a') t(c)"
        ),
        ["1485273170|-8582455328737087284"]
    );
}

/// Spark's bucketing refuses a collated column (`INVALID_BUCKET_COLUMN_DATA_TYPE`), so no
/// bucketed write ever hashes one.
#[test]
fn distribute_by_a_collated_key_runs() {
    assert_eq!(
        sorted(&format!("SELECT x FROM {T} DISTRIBUTE BY c")),
        ["1", "2", "3", "4"]
    );
}

/// A comparison on a collated column never prunes a row group by the bytes of its statistics.
#[test]
fn a_parquet_row_group_holding_a_match_is_kept() {
    let s = Session::new().unwrap();
    for sql in [
        "create table p1(c string collate utf8_lcase, n int) using parquet",
        "insert into p1 values('a', 1), ('b', 2)",
    ] {
        rows_in(&s, sql).unwrap();
    }
    // The file's minimum is 'a': a byte comparison against 'A' (below it) would skip the group.
    assert_eq!(
        rows_in(&s, "select n from p1 where c = 'A'").unwrap(),
        ["1"]
    );
    assert_eq!(
        rows_in(&s, "select n from p1 where c < 'B' order by n").unwrap(),
        ["1"]
    );
    assert_eq!(
        rows_in(&s, "select n from p1 where c in ('A')").unwrap(),
        ["1"]
    );
}

#[test]
fn icu_collations_stay_refused_in_every_key_position() {
    let c = "(SELECT 'a' COLLATE unicode_ci AS c, 1 AS x)";
    let s = Session::new().unwrap();
    for sql in [
        format!("SELECT c FROM {c} GROUP BY c"),
        format!("SELECT DISTINCT c FROM {c}"),
        format!("SELECT c FROM {c} ORDER BY c"),
        format!("SELECT count(DISTINCT c) FROM {c}"),
        format!("SELECT max(c) FROM {c}"),
        format!("SELECT mode(c) FROM {c}"),
        format!("SELECT listagg(c) FROM {c}"),
        format!("SELECT approx_count_distinct(c) FROM {c}"),
        format!("SELECT row_number() OVER (PARTITION BY c ORDER BY c) FROM {c}"),
        format!("SELECT c FROM {c} UNION SELECT c FROM {c}"),
        format!("SELECT c FROM {c} INTERSECT SELECT c FROM {c}"),
        format!("SELECT c FROM {c} EXCEPT SELECT c FROM {c}"),
    ] {
        assert_eq!(
            rows_in(&s, &sql).unwrap_err().error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
}

/// Nested ordering stays refused, and `mode` waits for its by-key form.
#[test]
fn what_has_no_ordering_key_yet_is_refused() {
    let s = Session::new().unwrap();
    for sql in [
        // A running frame's by-key extremum over bytes has no snapshot yet.
        "SELECT max(c) OVER (ORDER BY x) FROM (SELECT 'a' COLLATE utf8_lcase c, 1 x)",
        "SELECT array('a' COLLATE utf8_lcase) x ORDER BY x",
        "SELECT max(array('a' COLLATE utf8_lcase))",
        "SELECT mode(c) FROM (SELECT 'a' COLLATE utf8_lcase c)",
        // Two collated map keys: the duplicate check still compares bytes, where Spark 4.2.0
        // raises `DUPLICATED_MAP_KEY` (and keeps `{'a': 2}` under `LAST_WIN`).
        "SELECT map('a' COLLATE utf8_lcase, 1, 'A' COLLATE utf8_lcase, 2)",
    ] {
        assert_eq!(
            rows_in(&s, sql).unwrap_err().error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
}

/// A `DISTINCT` aggregate that keeps its collated values beside another aggregate, or with a
/// filter, has no plan yet: the analyzer refuses it, so the schema fails as the run does (the
/// distinct rewrite would otherwise dedupe the value cast to a plain string, `AaB` for `AB`).
#[test]
fn a_distinct_listagg_beside_another_aggregate_is_refused_at_analysis() {
    let s = Session::new().unwrap();
    for sql in [
        format!("SELECT listagg(DISTINCT c), count(*) FROM {T}"),
        format!("SELECT collect_list(DISTINCT c), max(x) FROM {T}"),
        format!("SELECT listagg(DISTINCT c) FILTER (WHERE x > 1) FROM {T}"),
        format!("SELECT listagg(DISTINCT c), count(DISTINCT x) FROM {T}"),
    ] {
        let schema = block_on(async { s.sql(&sql).await?.schema().await });
        assert_eq!(
            schema.unwrap_err().error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
    // Alone it runs.
    assert_eq!(
        rows(&format!("SELECT listagg(DISTINCT c, ',') FROM {T}")).len(),
        1
    );
}

/// A nested collated field whose name needs quoting keys too: the key's type literal is
/// `DataType.sql`, which quotes it, so it parses back.
#[test]
fn a_nested_key_with_a_quoted_field_name_parses() {
    assert_eq!(
        rows(
            "SELECT named_struct('my col', c) = named_struct('my col', 'A') FROM (SELECT 'a' COLLATE utf8_lcase c)"
        ),
        ["true"]
    );
}

/// Spark merges nested collations leaf by leaf, by strength, which is not built: an operand that
/// could win a leaf is refused, never keyed with the other operand's collation.
#[test]
fn nested_operands_that_could_win_a_leaf_are_refused() {
    let s = Session::new().unwrap();
    for sql in [
        // Spark: `COLLATION_MISMATCH.EXPLICIT`.
        "SELECT array('a ' COLLATE utf8_lcase) = array('A' COLLATE utf8_binary_rtrim)",
        // Spark: the explicit `UTF8_BINARY` wins, `false`.
        "SELECT array(c) = array('A' COLLATE utf8_binary) FROM (SELECT 'a' COLLATE utf8_lcase c)",
        // Spark: two implicit collations, `COLLATION_MISMATCH.IMPLICIT`.
        "SELECT array(c) = array(d) FROM (SELECT 'a' COLLATE utf8_lcase c, 'A' d)",
        // Spark gives a scalar subquery implicit strength too: `COLLATION_MISMATCH.IMPLICIT`.
        "SELECT array((SELECT 'A')) = array(c) FROM (SELECT 'a' COLLATE utf8_lcase c)",
    ] {
        assert_eq!(
            rows_in(&s, sql).map_err(|e| e.error_class()),
            Err(ErrorClass::KrishivUnsupportedFeature),
            "{sql}"
        );
    }
}

/// An explicit `UTF8_BINARY` operand makes `greatest`/`least` compare bytes: `a` (0x61) is
/// greater than `B` (0x42).
#[test]
fn greatest_and_least_under_an_explicit_binary_collation_compare_bytes() {
    let q = |f: &str| {
        format!(
            "SELECT {f}(c, 'a' COLLATE UTF8_BINARY) FROM (SELECT 'B' COLLATE UTF8_LCASE AS c) t"
        )
    };
    assert_eq!(rows(&q("greatest")), ["a"]);
    assert_eq!(rows(&q("least")), ["B"]);
}
