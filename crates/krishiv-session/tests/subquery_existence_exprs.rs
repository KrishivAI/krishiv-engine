// SPDX-License-Identifier: Apache-2.0
//! `IN` / `EXISTS` subqueries in expressions: in the select list or the
//! grouping keys of an aggregate, the existence join's column is fixed within
//! a group, so the aggregate groups by it too (it failed with an internal
//! error, "is not an input column", before); under a `NOT` of an `AND` or
//! `OR`, the `NOT` moves down to the `IN` (De Morgan), so a NULL membership
//! stays NULL (the rows of `NOT (a > 5 OR a IN (.. NULL ..))` were kept).
//! Every expected answer is what a Spark 4.2.0 run gave (`in-group-by.sql`
//! #20, #23, #24, #26, `exists-aggregate.sql` #17 and `nested-not-in.sql`
//! #16-#20 have the same shapes).
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

fn rows(v: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = v.iter().map(|r| (*r).to_owned()).collect();
    out.sort();
    out
}

#[test]
fn subqueries_over_grouping_keys_group_by_their_existence_column() -> R {
    let s = Session::new()?;
    run(
        &s,
        "create temporary view a(k, v) as values (1, 1), (2, 2), (1, 3), (null, 4)",
    )?;
    run(&s, "create temporary view b(k) as values (1), (3)")?;
    let q = "select k, k in (select k from b), sum(v) from a group by k";
    assert_eq!(
        run(&s, q)?,
        rows(&["1|true|4", "2|false|2", "NULL|false|4"]),
        "{q}"
    );
    let q = "select cast(exists (select 1 from b where b.k = a.k) as int) e, count(*) from a \
             group by cast(exists (select 1 from b where b.k = a.k) as int)";
    assert_eq!(run(&s, q)?, rows(&["0|2", "1|2"]), "{q}");
    let q = "select k, case when k in (select k from b) then 10 else -10 end from a group by k";
    assert_eq!(run(&s, q)?, rows(&["1|10", "2|-10", "NULL|-10"]), "{q}");
    // A grouping key that holds the subquery decides the group by itself.
    let q = "select k > 1 or k in (select k from b) as f, count(*) from a group by 1";
    assert_eq!(run(&s, q)?, rows(&["NULL|1", "true|3"]), "{q}");
    let q = "select not (k > 1 or k in (select k from b)) as f, count(*) from a group by 1";
    assert_eq!(run(&s, q)?, rows(&["false|4"]), "{q}");
    // A subquery over a column that is not grouped: Spark raises
    // MISSING_AGGREGATION; refused here, never answered per input row.
    let q = "select k, exists (select 1 from b where b.k = a.v) from a group by k";
    assert!(run(&s, q).is_err(), "{q}");
    // A key over the column is not the column (Spark matches outer
    // references to the keys with `semanticEquals`): `k % 2` groups 1 and 3
    // together, so `k` is not fixed within the group.
    let q = "select exists (select 1 from b where b.k = a.k), count(*) from a group by a.k % 2";
    assert!(run(&s, q).is_err(), "{q}");
    // A key's subquery groups that subquery, not every subquery over the
    // same column: `c` differs from `b`, so its EXISTS is not grouped.
    run(&s, "create temporary view c(k) as values (1)")?;
    let q = "select exists (select 1 from c where c.k = a.k), count(*) from a \
             group by exists (select 1 from b where b.k = a.k)";
    assert!(run(&s, q).is_err(), "{q}");
    // The same text over another outer column (`d.k`, not `a.k`) is not the
    // key's subquery either.
    run(&s, "create temporary view d(k) as values (1), (3)")?;
    let q = "select exists (select 1 from b where b.k = d.k), count(*) from a cross join d \
             group by exists (select 1 from b where b.k = a.k)";
    assert!(run(&s, q).is_err(), "{q}");
    Ok(())
}

/// A global aggregate has one row even over no input; a subquery beside its
/// calls reads that row, never becomes a group.
#[test]
fn a_subquery_beside_a_global_aggregate_keeps_its_one_row() -> R {
    let s = Session::new()?;
    run(
        &s,
        "create temporary view a(k, v) as values (1, 1), (2, 2), (1, 3), (null, 4)",
    )?;
    run(&s, "create temporary view b(k) as values (1), (3)")?;
    run(
        &s,
        "create temporary view e(k, v) as select * from a where false",
    )?;
    for (q, want) in [
        ("select 1 in (select k from b), count(*) from e", "true|0"),
        ("select 1 in (select k from b), count(*) from a", "true|4"),
        (
            "select 2 in (select k from b) as f, sum(v) from e",
            "false|NULL",
        ),
        (
            "select count(*) + cast(1 in (select k from b) as int), max(v) from a",
            "5|4",
        ),
    ] {
        assert_eq!(run(&s, q)?, rows(&[want]), "{q}");
    }
    Ok(())
}

#[test]
fn a_not_over_and_or_or_keeps_the_null_of_an_in() -> R {
    let s = Session::new()?;
    run(
        &s,
        "create temporary view s1(a, b) as values (null, null), (5, 5), (8, 8), (11, 11)",
    )?;
    run(
        &s,
        "create temporary view s2(c, d) as values (7, 7), (8, 8), (11, 11), (null, null)",
    )?;
    for (q, want) in [
        (
            "select * from s1 where not (a not in (select c from s2))",
            &["11|11", "8|8"][..],
        ),
        (
            "select * from s1 where not (a > 5 or a in (select c from s2))",
            &[][..],
        ),
        (
            "select * from s1 where not (a > 5 or a not in (select c from s2))",
            &[][..],
        ),
        (
            "select * from s1 where not (a > 5 and a in (select c from s2))",
            &["5|5"][..],
        ),
        (
            "select * from s1 where not (a > 5 and a not in (select c from s2))",
            &["11|11", "5|5", "8|8"][..],
        ),
    ] {
        assert_eq!(run(&s, q)?, rows(want), "{q}");
    }
    Ok(())
}

/// A `LIMIT` over a correlation that is not an equality: the rows are ranked
/// within each value of the outer columns (through the domain), in the
/// subquery's order.
#[test]
fn a_limit_over_a_non_equality_correlation_ranks_per_outer_row() -> R {
    let s = Session::new()?;
    run(
        &s,
        "create temporary view x(x1, x2) as values (1, 2), (3, 1), (2, 5)",
    )?;
    run(
        &s,
        "create temporary view y(y1, y2) as values (1, 5), (2, 2), (3, null)",
    )?;
    let q = "select x1, (select y2 from y where y1 <= x1 order by y2 desc limit 1) from x";
    assert_eq!(run(&s, q)?, rows(&["1|5", "2|5", "3|5"]), "{q}");
    let q = "select x1 from x where x2 in (select y2 from y where y1 <= x1 order by y1 limit 1)";
    assert_eq!(run(&s, q)?, rows(&["2"]), "{q}");
    Ok(())
}
