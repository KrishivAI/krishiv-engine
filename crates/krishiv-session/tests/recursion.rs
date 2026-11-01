// SPDX-License-Identifier: Apache-2.0
//! Recursive CTEs end to end (`WITH RECURSIVE`). Rows and errors are
//! Spark 4.2.0's.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn rows(s: &Session, sql: &str) -> Vec<Vec<String>> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for c in b.columns() {
                row.push(if c.is_null(i) {
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
                        other => format!("<{other}>"),
                    }
                });
            }
            out.push(row);
        }
    }
    out
}

fn class_of(s: &Session, sql: &str) -> ErrorClass {
    let result = block_on(async { s.sql(sql).await?.collect().await });
    match result {
        Ok(_) => panic!("{sql}: expected an error"),
        Err(e) => e.error_class(),
    }
}

fn column(rows: Vec<Vec<String>>) -> Vec<String> {
    rows.into_iter().map(|mut r| r.remove(0)).collect()
}

#[test]
fn counts_to_ten() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) \
         SELECT n FROM r",
    ));
    assert_eq!(got, (1..=10).map(|n| n.to_string()).collect::<Vec<_>>());
}

#[test]
fn a_reference_in_the_outer_query_twice() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
         SELECT a.n, b.n FROM r a JOIN r b ON a.n = b.n ORDER BY 1",
    );
    assert_eq!(got.len(), 3);
}

#[test]
fn union_without_all_is_refused() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r"
        ),
        ErrorClass::UnionNotSupportedInRecursiveCte
    );
}

#[test]
fn a_reference_in_the_anchor_is_invalid() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT n FROM r UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r"
        ),
        ErrorClass::InvalidRecursiveCte
    );
}

#[test]
fn the_level_limit_applies_even_if_the_recursion_would_end() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 150) SELECT * FROM r"
        ),
        ErrorClass::RecursionLevelLimitExceeded
    );
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) MAX RECURSION LEVEL 200 AS \
         (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 150) SELECT count(*) FROM r",
    );
    assert_eq!(got, vec![vec!["150".to_owned()]]);
}

#[test]
fn an_outer_limit_ends_an_unbounded_recursion() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM r) SELECT * FROM r LIMIT 4",
    ));
    assert_eq!(got, ["0", "1", "2", "3"]);
}

#[test]
fn a_limit_is_not_pushed_when_another_recursive_cte_reads_the_cte() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE a(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM a WHERE n < 5), \
         b(m) AS (SELECT 1 UNION ALL SELECT m + 1 FROM b WHERE m < (SELECT max(n) FROM a)) \
         SELECT * FROM a LIMIT 2",
    ));
    assert_eq!(got.len(), 2);
}

#[test]
fn a_reference_in_a_forbidden_place_is_refused() {
    let s = Session::new().unwrap();
    for body in [
        "SELECT max(n) FROM r",
        "SELECT (SELECT max(n) FROM r)",
        "(SELECT n + 1 FROM r ORDER BY n)",
        "SELECT a.n FROM (SELECT 1 AS n) a LEFT JOIN r ON a.n = r.n",
    ] {
        assert_eq!(
            class_of(
                &s,
                &format!(
                    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL {body}) SELECT * FROM r LIMIT 3"
                )
            ),
            ErrorClass::InvalidRecursiveReferencePlace,
            "{body}"
        );
    }
}

#[test]
fn a_limit_is_not_pushed_when_the_cte_is_read_twice() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) \
         SELECT n, (SELECT count(*) FROM r) FROM r LIMIT 3",
    );
    assert_eq!(got.len(), 3);
    assert!(got.iter().all(|r| r[1] == "10"), "{got:?}");
}

#[test]
fn a_recursive_term_reads_an_earlier_cte() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE one AS (SELECT 1 AS k), r(n) AS (SELECT 1 UNION ALL SELECT n + k FROM r, one WHERE n < 3) \
         SELECT n FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn the_row_limit_applies() {
    let s = Session::new().unwrap();
    // 100,000 rows per round pass the million in eleven rounds.
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT id FROM range(100000) UNION ALL SELECT n FROM r) \
             SELECT count(*) FROM r"
        ),
        ErrorClass::RecursionRowLimitExceeded
    );
}

#[test]
fn the_limits_follow_the_settings() {
    let s = Session::new().unwrap();
    let q = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) \
             SELECT count(*) FROM r";
    block_on(s.sql("SET krishiv.sql.cteRecursionRowLimit=3")).unwrap();
    assert_eq!(class_of(&s, q), ErrorClass::RecursionRowLimitExceeded);
    block_on(s.sql("SET spark.sql.cteRecursionRowLimit=-1")).unwrap();
    block_on(s.sql("SET spark.sql.cteRecursionLevelLimit=2")).unwrap();
    assert_eq!(class_of(&s, q), ErrorClass::RecursionLevelLimitExceeded);
    // MAX RECURSION LEVEL overrides the setting.
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) MAX RECURSION LEVEL 50 AS \
         (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10) SELECT count(*) FROM r",
    );
    assert_eq!(got, vec![vec!["10".to_owned()]]);
    block_on(s.sql("SET spark.sql.cteRecursionLevelLimit=-1")).unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 150) \
         SELECT count(*) FROM r",
    );
    assert_eq!(got, vec![vec!["150".to_owned()]]);
}

#[test]
fn local_definitions_scope_over_both_terms() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (WITH x AS (SELECT 1 AS k) \
         SELECT k FROM x UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT * FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (WITH x AS (SELECT * FROM r) \
             SELECT 1 UNION ALL SELECT n + 1 FROM x) SELECT * FROM r"
        ),
        ErrorClass::InvalidRecursiveCte
    );
}

#[test]
fn an_inner_definition_of_the_name_shadows_the_outer() {
    let s = Session::new().unwrap();
    for sql in [
        "WITH r AS (WITH r AS (SELECT 1 AS a) SELECT * FROM r) SELECT * FROM r",
        "WITH RECURSIVE r(n) AS (WITH r AS (SELECT 5 AS n) SELECT * FROM r) SELECT * FROM r",
    ] {
        assert_eq!(rows(&s, sql).len(), 1, "{sql}");
    }
}

#[test]
fn a_recursive_cte_inside_an_expression_subquery() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
         SELECT n, (SELECT max(m) FROM (WITH RECURSIVE q(m) AS \
         (SELECT 1 UNION ALL SELECT m + 1 FROM q WHERE m < 4) SELECT * FROM q)) FROM r",
    );
    let want: Vec<Vec<String>> = (1..=3)
        .map(|n| vec![n.to_string(), "4".to_owned()])
        .collect();
    assert_eq!(got, want);
}

#[test]
fn explain_and_estimate_do_not_run_the_loop() {
    let s = Session::new().unwrap();
    // Never ends if run: the only bound is the (huge) level limit.
    let q = "WITH RECURSIVE r(n) MAX RECURSION LEVEL 4000000 AS \
             (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT * FROM r";
    let df = block_on(s.sql(q)).unwrap();
    block_on(df.explain(krishiv_session::ExplainMode::Extended)).unwrap();
    block_on(df.estimate()).unwrap();
    block_on(s.sql(&format!("EXPLAIN {q}")))
        .and_then(|d| block_on(d.collect()))
        .unwrap();
}

#[test]
fn limit_all_on_every_read_lifts_the_row_limit() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.cteRecursionRowLimit=50")).unwrap();
    let cte = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 60) ";
    let all = rows(
        &s,
        &format!("{cte} SELECT count(*) FROM (SELECT n FROM t LIMIT ALL)"),
    );
    assert_eq!(all, vec![vec!["60".to_owned()]]);
    let both = rows(
        &s,
        &format!(
            "{cte} SELECT count(*) FROM ((SELECT n FROM t LIMIT ALL) UNION ALL (SELECT n FROM t LIMIT ALL))"
        ),
    );
    assert_eq!(both, vec![vec!["120".to_owned()]]);
    // One read without a limit keeps the row limit.
    assert_eq!(
        class_of(
            &s,
            &format!("{cte} (SELECT n FROM t LIMIT ALL) UNION ALL (SELECT n FROM t)")
        ),
        ErrorClass::RecursionRowLimitExceeded
    );
}

#[test]
fn a_later_definition_keeps_limit_all_on_a_recursive_sibling() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.cteRecursionRowLimit=3")).unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5), \
         b AS (SELECT * FROM r LIMIT ALL) SELECT count(*) FROM b",
    );
    assert_eq!(got, vec![vec!["5".to_owned()]]);
}

#[test]
fn a_nested_definition_of_the_name_is_not_a_recursive_reference() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT r.n + 1 FROM r CROSS JOIN \
         (WITH r AS (SELECT 1 AS n) SELECT max(n) AS k FROM r) x WHERE r.n < 3) SELECT * FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn a_nested_loop_reading_the_enclosing_round_runs_within_it() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n FROM \
         (WITH RECURSIVE s(n) AS (SELECT n + 1 FROM r WHERE n < 3 UNION ALL \
          SELECT n FROM s WHERE false) SELECT n FROM s)) SELECT * FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn the_rows_a_recursion_keeps_count_against_the_memory_limit() {
    let s = Session::builder()
        .config("krishiv.memory.limit", "4MB")
        .build()
        .unwrap();
    // Wide rows: each round fits, the accumulated rows do not.
    let class = class_of(
        &s,
        "WITH RECURSIVE r(n, pad) MAX RECURSION LEVEL 100 AS (SELECT 1, repeat('x', 100000) \
         UNION ALL SELECT n + 1, pad FROM r WHERE n < 100) SELECT count(*) FROM r",
    );
    assert_eq!(class, ErrorClass::UnableToAcquireMemory);
}

#[test]
fn a_cte_defined_in_the_term_cannot_hide_a_forbidden_place() {
    let s = Session::new().unwrap();
    // Spark 4.2.0: the aggregate reads `a`, which reads the recursive CTE.
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT CAST(1 AS BIGINT) UNION ALL \
             (WITH a AS (SELECT n FROM r) SELECT count(*) FROM a)) SELECT * FROM r LIMIT 3"
        ),
        ErrorClass::InvalidRecursiveReferencePlace
    );
    // An allowed place through a definition still runs.
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL \
         (WITH a AS (SELECT n + 1 AS m FROM r WHERE n < 3) SELECT m FROM a)) SELECT * FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn the_other_forbidden_places_are_refused() {
    let s = Session::new().unwrap();
    for term in [
        "SELECT n + 1, row_number() OVER (ORDER BY n) FROM r",
        "SELECT r.n + 1 FROM r FULL OUTER JOIN (SELECT 1 AS k) x ON r.n = x.k",
        "SELECT r.n + 1 FROM r RIGHT OUTER JOIN (SELECT 1 AS k) x ON r.n = x.k",
        "SELECT r.n + 1 FROM r LEFT OUTER JOIN (SELECT 1 AS k) x ON r.n = x.k \
         WHERE EXISTS (SELECT 1 FROM r)",
    ] {
        let sql =
            format!("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL {term}) SELECT * FROM r LIMIT 3");
        assert_eq!(
            class_of(&s, &sql),
            ErrorClass::InvalidRecursiveReferencePlace,
            "{sql}"
        );
    }
}

#[test]
fn definitions_beside_the_recursive_reference_resolve_through_the_scope() {
    let s = Session::new().unwrap();
    let got = column(rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL (WITH a AS (SELECT 0 AS k), \
         b AS (SELECT n + 1 AS m FROM r WHERE n < 3) SELECT m FROM b CROSS JOIN a)) \
         SELECT * FROM r",
    ));
    assert_eq!(got, ["1", "2", "3"]);
}

#[test]
fn a_recursive_term_that_changes_the_anchors_types_is_refused() {
    let s = Session::new().unwrap();
    assert_eq!(
        class_of(
            &s,
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL \
             SELECT CAST(n AS BIGINT) + 1 FROM r WHERE n < 3) SELECT * FROM r"
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
}

#[test]
fn nested_limits_push_the_smaller_one() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) \
         SELECT count(*) FROM (SELECT * FROM (SELECT n FROM r LIMIT 3) LIMIT 5)",
    );
    assert_eq!(got, vec![vec!["3".to_owned()]]);
}

#[test]
fn max_recursion_level_belongs_to_recursive_with() {
    let s = Session::new().unwrap();
    let e = block_on(s.sql("WITH r MAX RECURSION LEVEL 5 AS (SELECT 1) SELECT * FROM r"));
    assert!(e.is_err());
}

#[test]
fn a_recursive_result_can_be_transformed_and_read_again() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT * FROM r",
    ))
    .unwrap();
    let kept =
        df.filter(krishiv_session::functions::col("n").gt(krishiv_session::functions::lit(2)));
    let batches = block_on(kept.collect()).unwrap();
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        3
    );
    block_on(s.sql("CREATE TEMPORARY VIEW rv AS WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 4) SELECT * FROM r")).unwrap();
    assert_eq!(
        rows(&s, "SELECT count(*) FROM rv WHERE n > 1"),
        vec![vec!["3".to_owned()]]
    );
}

#[test]
fn a_recursive_cte_may_repeat_an_output_name() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(
        "WITH RECURSIVE r AS (SELECT 1 AS a, 2 AS a, 0 AS n UNION ALL \
         SELECT 3, 4, n + 1 FROM r WHERE n < 1) SELECT * FROM r",
    ))
    .unwrap();
    let batches = block_on(df.collect()).unwrap();
    let schema = batches[0].schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["a", "a", "n"]);
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        2
    );
    assert_eq!(batches[0].num_columns(), 3);
}

#[test]
fn current_timestamp_is_one_instant_across_rounds() {
    let s = Session::new().unwrap();
    let got = rows(
        &s,
        "WITH RECURSIVE r(n, t) AS (SELECT 1, current_timestamp() UNION ALL \
         SELECT n + 1, current_timestamp() FROM r WHERE n < 3) \
         SELECT count(DISTINCT t) FROM r",
    );
    assert_eq!(got, vec![vec!["1".to_owned()]]);
}
