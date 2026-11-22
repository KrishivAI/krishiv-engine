// SPDX-License-Identifier: Apache-2.0
//! Regression tests for the Spark 4.3 correctness fixes the P3 refresh lists
//! as testable on any version (P3-T031 starter set). Every expectation was
//! read from the 4.3.0 oracle (`spark-sql`, ANSI on) before it was written:
//! SPARK-58481 (IN-subquery with NULLs), SPARK-59434 (non-deterministic
//! predicates and CTEs), SPARK-58384 (NULL semantics in join conditions).
//! SPARK-58959 (PIVOT with duplicate values) is not here: 4.3 gives every
//! duplicate its own column where 4.2 raises `INVALID_ARRAY_INDEX`, which
//! Krishiv reproduces on the 4.2 profile, so it waits for the 4.3 profile.
#![forbid(unsafe_code)]

use arrow_cast::display::{ArrayFormatter, FormatOptions};
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Every row of `sql` as tab-joined cells, NULL spelled `NULL`.
fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    block_on(async {
        let mut out = Vec::new();
        for b in s.sql(sql).await?.collect().await? {
            let opts = FormatOptions::default().with_null("NULL");
            let fmt = b
                .columns()
                .iter()
                .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts))
                .collect::<Result<Vec<_>, _>>()?;
            for r in 0..b.num_rows() {
                let cells = fmt
                    .iter()
                    .map(|f| f.value(r).try_to_string())
                    .collect::<Result<Vec<_>, _>>()?;
                out.push(cells.join("\t"));
            }
        }
        Ok(out)
    })
}

#[test]
fn spark_58481_in_subquery_with_nulls() -> R {
    let s = Session::new()?;
    for (sql, want) in [
        ("select 1 in (select null)", "false"),
        ("select 2 in (values 1, null)", "false"),
        ("select (1, 2) in (select 1, null)", "false"),
        ("select (1, 2) not in (select 1, null)", "false"),
    ] {
        assert_eq!(rows(&s, sql)?, [want], "{sql}");
    }
    Ok(())
}

#[test]
fn spark_59434_a_non_deterministic_projection_is_not_filtered_early() -> R {
    let s = Session::new()?;
    let cte = "with c as (select id, rand(7) r from range(20))";
    assert_eq!(
        rows(
            &s,
            &format!("{cte} select count(*), sum(id) from c where r < 0.5")
        )?,
        ["13\t115"]
    );
    assert_eq!(
        rows(
            &s,
            "with c as (select id from range(20) where rand(7) < 0.5) \
             select count(*), sum(id) from c where id > 3"
        )?,
        ["11\t114"]
    );
    assert_eq!(
        rows(
            &s,
            &format!("{cte} select count(*) from c a join c b on a.id = b.id and a.r = b.r")
        )?,
        ["20"]
    );
    assert_eq!(
        rows(
            &s,
            "select count(*) from (select id, rand(3) r from range(50)) \
             where id % 2 = 0 and r < 0.5"
        )?,
        ["13"]
    );
    Ok(())
}

#[test]
fn spark_58384_null_semantics_in_join_conditions() -> R {
    let s = Session::new()?;
    let a = "values (1, cast(null as int)), (2, 2) a(i, j)";
    let b = "values (1, cast(null as int)), (2, 2) b(i, j)";
    assert_eq!(
        rows(
            &s,
            &format!(
                "select * from {a} left join {b} on a.j = b.j \
                 or (a.j is null and b.j is null and a.i = b.i) order by 1"
            )
        )?,
        ["1\tNULL\t1\tNULL", "2\t2\t2\t2"]
    );
    assert_eq!(
        rows(
            &s,
            "select * from values (1, cast(null as int)) a(i, j) \
             join values (1, cast(null as int)) b(i, j) on not (a.j = b.j) or a.i = 1"
        )?,
        ["1\tNULL\t1\tNULL"]
    );
    assert_eq!(
        rows(
            &s,
            &format!(
                "select * from {a} left join values (1, 1) b(i, j) on not (a.j = b.j) order by 1"
            )
        )?,
        ["1\tNULL\tNULL\tNULL", "2\t2\t1\t1"]
    );
    Ok(())
}
