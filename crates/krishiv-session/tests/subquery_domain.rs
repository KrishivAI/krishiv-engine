// SPDX-License-Identifier: Apache-2.0
//! Correlated subqueries over set operations and joins, which run once per
//! distinct outer value. Every expected answer and error class is what Spark
//! 4.2.0 gave for the same statement over the same views.
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
        other => format!("<{other}>"),
    }
}

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
        "create temporary view t0(t0a, t0b) as values (1, 1), (2, 0), (null, 1)",
        "create temporary view t1(t1a, t1b, t1c) as values \
         (1, 1, 3), (1, 2, 3), (2, 2, 4), (null, 1, 5)",
        "create temporary view t2(t2a, t2b, t2c) as values (1, 1, 3), (2, 2, 7), (2, 0, 3)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

/// `(sum over the subquery's rows)` per row of `t0`, the subquery in `from`.
fn sum_over(s: &Session, from: &str) -> R<Vec<String>> {
    run(
        s,
        &format!("select t0a, (select sum(t1b) from {from}) from t0"),
    )
}

/// The error of `sql`, from analysis or from running it; empty if it works.
fn failure(s: &Session, sql: &str) -> String {
    match block_on(s.sql(sql)) {
        Err(e) => e.to_string(),
        Ok(df) => block_on(df.collect())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default(),
    }
}

#[test]
fn a_set_operation_runs_once_per_outer_value() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select t0a, (select sum(c) from (select t1c c from t1 where t1a = t0a \
             intersect all select t2c from t2 where t2a = t0a)) from t0"
        )?,
        ["1|3", "2|NULL", "NULL|NULL"]
    );
    // A NULL outer value owns its own rows, through `<=>`.
    assert_eq!(
        run(
            &s,
            "select t0a, (select count(c) from (select t1c c from t1 where t1a <=> t0a \
             except all select t2c from t2 where t2a <=> t0a)) from t0"
        )?,
        ["1|1", "2|1", "NULL|1"]
    );
    assert_eq!(
        run(
            &s,
            "select t0a, (select sum(d) from (select t1a - t0a d from t1 \
             except select t2a - t0a d from t2)) from t0"
        )?,
        ["1|NULL", "2|NULL", "NULL|NULL"]
    );
    Ok(())
}

#[test]
fn a_set_operation_without_an_aggregate_is_at_most_one_row() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select t0a, (select t1c from t1 where t1a = t0a \
             intersect select t2c from t2 where t2a = t0a) from t0"
        )?,
        ["1|3", "2|NULL", "NULL|NULL"]
    );
    let e = failure(
        &s,
        "select t0a, (select t1b from t1 where t1a = t0a \
         except select t2c from t2 where t2a = t0a) from t0",
    );
    assert!(e.contains("SCALAR_SUBQUERY_TOO_MANY_ROWS"), "{e}");
    Ok(())
}

#[test]
fn exists_and_in_over_a_set_operation() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from t0 where exists (select t1c from t1 where t1a = t0a \
             intersect select t2c from t2 where t2a = t0a)"
        )?,
        ["1|1"]
    );
    assert_eq!(
        run(
            &s,
            "select * from t0 where not exists (select t1c from t1 where t1a = t0a \
             except all select t2c from t2 where t2a = t0a)"
        )?,
        ["NULL|1"]
    );
    assert!(
        run(
            &s,
            "select * from t0 where t0b + 2 in (select t1c from t1 where t1a = t0a \
             except select t2c from t2 where t2a = t0a)"
        )?
        .is_empty()
    );
    Ok(())
}

#[test]
fn a_grouped_aggregate_under_a_union_groups_per_outer_value() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select t0a, (select sum(c) from (select sum(t1c) c from t1 where t1a = t0a \
             group by t1b union all select t2c from t2)) from t0"
        )?,
        ["1|19", "2|17", "NULL|13"]
    );
    Ok(())
}

#[test]
fn a_group_filtered_to_one_row_over_a_union() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select t0a, (select sum(c) from (select t1c c from t1 where t1a = t0a \
             union all select t2c from t2 where t2a = t0a) group by c having c = 3) from t0"
        )?,
        ["1|9", "2|3", "NULL|NULL"]
    );
    Ok(())
}

#[test]
fn a_join_condition_reads_the_outer_value() -> R {
    let s = session()?;
    let on = "on t1a = t0a and t1b = t2b)";
    assert_eq!(
        sum_over(&s, &format!("(select t1b from t1 join t2 {on}"))?,
        ["1|3", "2|2", "NULL|NULL"]
    );
    // A left join keeps every left row whatever the correlation says.
    assert_eq!(
        sum_over(&s, &format!("(select t1b from t1 left join t2 {on}"))?,
        ["1|6", "2|6", "NULL|6"]
    );
    assert_eq!(
        sum_over(&s, &format!("(select t1b from t1 right join t2 {on}"))?,
        ["1|3", "2|2", "NULL|NULL"]
    );
    assert_eq!(
        sum_over(&s, &format!("(select t1b from t1 left semi join t2 {on}"))?,
        ["1|3", "2|2", "NULL|NULL"]
    );
    assert_eq!(
        sum_over(&s, &format!("(select t1b from t1 left anti join t2 {on}"))?,
        ["1|3", "2|4", "NULL|6"]
    );
    Ok(())
}

#[test]
fn a_correlated_side_of_a_join() -> R {
    let s = session()?;
    assert_eq!(
        sum_over(
            &s,
            "(select * from t1 where t1a = t0a) x left join t2 on x.t1b = t2b"
        )?,
        ["1|3", "2|2", "NULL|NULL"]
    );
    assert_eq!(
        sum_over(
            &s,
            "(select * from t1 where t1a = t0a) x \
             join (select * from t2 where t2a = t0a) y on x.t1b = y.t2b"
        )?,
        ["1|1", "2|2", "NULL|NULL"]
    );
    for join in ["left join", "left semi join"] {
        let e = failure(
            &s,
            &format!(
                "select t0a, (select sum(t1b) from t1 {join} \
                 (select * from t2 where t2a = t0a) x on t1b = x.t2b) from t0"
            ),
        );
        assert!(
            e.contains("ACCESSING_OUTER_QUERY_COLUMN_IS_NOT_ALLOWED"),
            "{join}: {e}"
        );
    }
    Ok(())
}

#[test]
fn shapes_not_rewritten_yet_are_refused() -> R {
    let s = session()?;
    for sql in [
        // Spark's FULL JOIN answer here is not the per-row join: (1, 3), (2, 2), (NULL, 6).
        "select t0a, (select sum(t1b) from (select t1b from t1 full join t2 \
         on t1a = t0a and t1b = t2b)) from t0",
        // Spark: (1, 19), (2, 17), (NULL, 13).
        "select t0a, (select sum(c) from (select sum(t1c) c from t1 where t1a = t0a \
         union all select t2c from t2)) from t0",
        // Spark: (1, 3), (2, NULL), (NULL, NULL).
        "select t0a, (select sum(c) from (select t1c c from t1 where t1a = t0a \
         intersect select t1c from t1 where t1b = t0b)) from t0",
    ] {
        let e = failure(&s, sql);
        assert!(e.contains("KRISHIV_UNSUPPORTED_FEATURE"), "{sql}: {e}");
    }
    Ok(())
}

#[test]
fn a_correlated_limit_reads_a_sort_below_a_derived_table() -> R {
    let s = session()?;
    run(
        &s,
        "create temporary view o(k, v) as values (1, 30), (1, 10), (2, 40), (2, 20)",
    )?;
    run(
        &s,
        "create temporary view i(k, s, c) as values \
         (1, 1, 10), (1, 2, 20), (1, 3, 30), (2, 1, 20), (2, 2, 30), (2, 3, 40)",
    )?;
    let q = |dir: &str| {
        format!(
            "select * from o where v in (select c from \
             (select c, k from i where i.k = o.k order by s {dir}) limit 1)"
        )
    };
    assert_eq!(run(&s, &q("desc"))?, ["1|30", "2|40"]);
    assert_eq!(run(&s, &q("asc"))?, ["1|10", "2|20"]);
    Ok(())
}
