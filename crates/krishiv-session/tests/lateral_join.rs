// SPDX-License-Identifier: Apache-2.0
//! `JOIN LATERAL` and `, LATERAL (...)` (T180 unit 4): the right side reads
//! the left side's columns. The expected rows are worked out by hand from
//! Spark's semantics and agree with the `join-lateral` golden file, which
//! the golden ratchet replays in full.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn cell(c: &dyn Array, row: usize) -> String {
    if c.is_null(row) {
        return "NULL".into();
    }
    match c.data_type() {
        DataType::Int32 => c.as_primitive::<Int32Type>().value(row).to_string(),
        DataType::Int64 => c.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        other => format!("<{other}>"),
    }
}

/// The rows of `sql`, sorted, columns joined by `|`.
fn run(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows()).map(|r| {
                    b.columns()
                        .iter()
                        .map(|c| cell(c.as_ref(), r))
                        .collect::<Vec<_>>()
                        .join("|")
                })
            })
            .collect();
        out.sort();
        Ok(out)
    })
}

fn session() -> R<Session> {
    let s = Session::new()?;
    for v in [
        "create temporary view t1(c1, c2) as values (0, 1), (1, 2)",
        "create temporary view t2(c1, c2) as values (0, 2), (0, 3)",
        "create temporary view t4(c1, c2) as values (0, 1), (0, 2), (1, 1), (1, 3)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

fn error_of(s: &Session, sql: &str) -> String {
    run(s, sql).err().map(|e| e.to_string()).unwrap_or_default()
}

#[test]
fn equality_and_non_equality_correlation() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select c2 from t2 where t1.c1 = t2.c1)"
        )?,
        ["0|1|2", "0|1|3"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select c2 from t2 where t1.c2 < t2.c2)"
        )?,
        ["0|1|2", "0|1|3", "1|2|3"]
    );
    // A left join keeps the outer row that matched nothing.
    assert_eq!(
        run(
            &s,
            "select * from t1 left join lateral (select c2 from t2 where t1.c1 = t2.c1) s on true"
        )?,
        ["0|1|2", "0|1|3", "1|2|NULL"]
    );
    Ok(())
}

#[test]
fn outer_references_in_the_select_list_and_the_join_condition() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select t1.c1 + t2.c1 from t2)"
        )?,
        ["0|1|0", "0|1|0", "1|2|1", "1|2|1"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t1 join lateral (select c1 + c2 as c3) on c2 = c3"
        )?,
        ["0|1|1"]
    );
    // The condition fails for the second row: the left join pads it.
    assert_eq!(
        run(
            &s,
            "select * from t1 left join lateral (select c1 + c2 as c3) on c2 = c3"
        )?,
        ["0|1|1", "1|2|NULL"]
    );
    Ok(())
}

#[test]
fn laterals_chain_and_read_each_other() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select c1 + c2 as a), lateral (select c1 - c2 as b), \
             lateral (select a * b as c)"
        )?,
        ["0|1|1|-1|-1", "1|2|3|-1|-3"]
    );
    Ok(())
}

/// A global aggregate has a row for every outer row: `count` reads 0 and
/// the others NULL where nothing matched, and `HAVING` can drop the row.
#[test]
fn a_global_aggregate_keeps_one_row_per_outer_row() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select count(*) cnt from t2 where c1 = t1.c1)"
        )?,
        ["0|1|2", "1|2|0"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select sum(c2) is null from t2 where t1.c1 = t2.c1)"
        )?,
        ["0|1|false", "1|2|true"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select max(c2) m from t2 where t1.c2 < t2.c2)"
        )?,
        ["0|1|3", "1|2|3"]
    );
    let having = "lateral (select count(*) cnt from t2 where c1 = t1.c1 having cnt > 0)";
    assert_eq!(run(&s, &format!("select * from t1, {having}"))?, ["0|1|2"]);
    assert_eq!(
        run(&s, &format!("select * from t1 left join {having}"))?,
        ["0|1|2", "1|2|NULL"]
    );
    Ok(())
}

#[test]
fn a_generator_over_the_outer_columns() -> R {
    let s = session()?;
    assert_eq!(
        run(&s, "select * from t1, lateral explode(array(c1, c2)) t(c3)")?,
        ["0|1|0", "0|1|1", "1|2|1", "1|2|2"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t1 join lateral explode(array(c1, c2)) t(c3) on t1.c1 = c3"
        )?,
        ["0|1|0", "1|2|1"]
    );
    Ok(())
}

#[test]
fn limits_and_set_operations_run_once_per_outer_row() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t1 join lateral \
             (select c2 from t2 where t1.c2 < t2.c2 order by c2 desc limit 1)"
        )?,
        ["0|1|3", "1|2|3"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t1 join lateral (select t2.c2 from t2 where t2.c1 = t1.c1 \
             union all select t4.c2 from t4 where t4.c1 = t1.c1)"
        )?,
        ["0|1|1", "0|1|2", "0|1|2", "0|1|3", "1|2|1", "1|2|3"]
    );
    Ok(())
}

/// Spark refuses a nondeterministic lateral subquery unless its outer side
/// has at most one row.
#[test]
fn a_nondeterministic_lateral_needs_a_single_outer_row() -> R {
    let s = session()?;
    for q in [
        "select * from t1, lateral (select rand(0) from t2)",
        "select * from t1, lateral (select c1 + c2 + rand(0) as c3)",
    ] {
        let err = error_of(&s, q);
        assert!(
            err.contains(
                "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.NON_DETERMINISTIC_LATERAL_SUBQUERIES"
            ),
            "{q}: {err}"
        );
    }
    assert_eq!(
        run(
            &s,
            "select x from values (0) t(x) join lateral (select x + rand(0) as y)"
        )?,
        ["0"]
    );
    let err = error_of(
        &s,
        "select * from t1 join lateral (select * from t2) s on t1.c1 + rand(0) = s.c1",
    );
    assert!(
        err.contains(
            "UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.LATERAL_JOIN_CONDITION_NON_DETERMINISTIC"
        ),
        "{err}"
    );
    Ok(())
}

/// A lateral subquery sees its own left side only: a nested one does not
/// see the outer one's.
#[test]
fn only_the_immediate_left_side_is_visible() -> R {
    let s = session()?;
    let err = error_of(
        &s,
        "select * from t1, lateral (select * from t2, lateral (select t1.c1 + t2.c1))",
    );
    assert!(err.contains("UNRESOLVED_COLUMN"), "{err}");
    Ok(())
}

#[test]
fn only_inner_and_left_joins_may_be_lateral() -> R {
    let s = session()?;
    for join in ["right", "full"] {
        let err = error_of(
            &s,
            &format!("select * from t1 {join} join lateral (select c1) on true"),
        );
        assert!(err.contains("INVALID_LATERAL_JOIN_TYPE"), "{join}: {err}");
    }
    Ok(())
}

#[test]
fn a_grouped_aggregate_over_a_non_equality_correlation() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t1, lateral (select c1, sum(c2) s from t4 where t1.c1 <= t4.c1 group by c1)"
        )?,
        ["0|1|0|3", "0|1|1|4", "1|2|1|4"]
    );
    Ok(())
}

#[test]
fn a_star_of_the_outer_side_inside_the_lateral() -> R {
    let s = session()?;
    assert_eq!(
        run(&s, "select * from t1, lateral (select t1.*)")?,
        ["0|1|0|1", "1|2|1|2"]
    );
    assert_eq!(
        run(&s, "select * from t1, lateral (select t1.*, t2.* from t2)")?,
        ["0|1|0|1|0|2", "0|1|0|1|0|3", "1|2|1|2|0|2", "1|2|1|2|0|3"]
    );
    // An inner relation of the same name shadows the outer one.
    assert_eq!(
        run(&s, "select * from t1, lateral (select t1.* from t2 as t1)")?,
        ["0|1|0|2", "0|1|0|3", "1|2|0|2", "1|2|0|3"]
    );
    Ok(())
}

/// A lateral right side after a `NATURAL` / `USING` join reads the merged
/// key unqualified and each side's original key qualified.
#[test]
fn lateral_after_using_join_reads_merged_and_original_keys() -> R {
    let s = session()?;
    for sql in [
        "with a(k, v1) as (values (1, 10)), b(k, v2) as (values (1, 20)) \
         select * from a natural join b, lateral (select k as q, a.k as qa, b.k as qb)",
        "with a(k, v1) as (values (1, 10)), b(k, v2) as (values (1, 20)) \
         select * from a join b using (k), lateral (select k as q, a.k as qa, b.k as qb)",
    ] {
        assert_eq!(run(&s, sql)?, ["1|10|20|1|1|1"], "{sql}");
    }
    Ok(())
}

/// Spark refuses an `IN` or `EXISTS` in the condition of a lateral join.
#[test]
fn an_in_subquery_in_the_on_of_a_lateral_join_is_refused() -> R {
    let s = session()?;
    // Correlated or not: the bound plan of an uncorrelated LATERAL is a plain join.
    for sql in [
        "SELECT t1.c1, s.y FROM t1 JOIN LATERAL (SELECT t1.c2 AS y) s ON t1.c1 IN (SELECT c1 FROM t2)",
        "SELECT t1.c1, s.y FROM t1 JOIN LATERAL (SELECT 1 AS y) s ON t1.c1 IN (SELECT c1 FROM t2)",
        "SELECT t1.c1, s.y FROM t1 LEFT JOIN LATERAL (SELECT 1 AS y) s ON EXISTS (SELECT 1 FROM t2)",
    ] {
        let err = error_of(&s, sql);
        assert!(
            err.contains("UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY.UNSUPPORTED_IN_EXISTS_SUBQUERY"),
            "{sql}: {err}"
        );
    }
    // A scalar subquery holding an IN is not an IN of the condition itself.
    assert_eq!(
        run(
            &s,
            "SELECT t1.c1, s.y FROM t1 JOIN LATERAL (SELECT 1 AS y) s \
             ON (SELECT count(*) FROM t2 WHERE c1 IN (SELECT c1 FROM t1)) > 0"
        )?,
        ["0|1", "1|1"]
    );
    // A plain join keeps its IN in the ON.
    assert_eq!(
        run(
            &s,
            "SELECT t1.c1 FROM t1 JOIN t2 ON t1.c1 = t2.c1 AND t1.c1 IN (SELECT c1 FROM t2)"
        )?,
        ["0", "0"]
    );
    Ok(())
}

/// Spark binds `x` in a `HAVING` subquery to the aggregate's input column, not
/// to the output alias `sum(x) AS x`, and refuses it as not grouped.
#[test]
fn a_having_subquery_binds_an_unqualified_name_to_the_aggregate_input() -> R {
    let s = session()?;
    let err = error_of(
        &s,
        "SELECT sum(x) AS x FROM VALUES (1), (2) t(x) \
         HAVING EXISTS (SELECT 1 FROM VALUES (3) u(y) WHERE u.y = x)",
    );
    assert!(!err.is_empty(), "must not answer");
    Ok(())
}
