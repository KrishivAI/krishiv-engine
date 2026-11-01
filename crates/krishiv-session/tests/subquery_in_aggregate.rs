// SPDX-License-Identifier: Apache-2.0
//! A correlated `EXISTS` in an aggregate's `FILTER`, grouping key or call
//! argument (Spark 4.2.0's exists-outside-filter golden), including two
//! `DISTINCT` calls over a computed key.
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn rows(s: &Session, sql: &str) -> R<Vec<String>> {
    let wrapped = format!("select concat_ws('\t', *) from ({sql})");
    let batches = block_on(async { s.sql(&wrapped).await?.collect().await })?;
    let mut out = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for r in 0..b.num_rows() {
            let v = match c.as_string_view_opt() {
                Some(v) => v.value(r).to_owned(),
                None => c.as_string::<i32>().value(r).to_owned(),
            };
            out.push(v);
        }
    }
    out.sort();
    Ok(out)
}

fn session() -> R<Session> {
    let s = Session::new()?;
    block_on(s.sql(
        "create temporary view emp as select * from values \
         (100, 'emp 1', date '2005-01-01', 100.00D, 10), (100, 'emp 1', date '2005-01-01', 100.00D, 10), \
         (200, 'emp 2', date '2003-01-01', 200.00D, 10), (300, 'emp 3', date '2002-01-01', 300.00D, 20), \
         (400, 'emp 4', date '2005-01-01', 400.00D, 30), (500, 'emp 5', date '2001-01-01', 400.00D, NULL), \
         (600, 'emp 6 - no dept', date '2001-01-01', 400.00D, 100), (700, 'emp 7', date '2010-01-01', 400.00D, 100), \
         (800, 'emp 8', date '2016-01-01', 150.00D, 70) as emp(id, emp_name, hiredate, salary, dept_id)",
    ))?;
    block_on(s.sql(
        "create temporary view dept as select * from values (10, 'd1'), (20, 'd2'), (30, 'd3'), \
         (40, 'd4'), (50, 'd5'), (70, 'd7') as dept(dept_id, dept_name)",
    ))?;
    Ok(s)
}

const EXISTS: &str = "exists (select 1 from dept where emp.dept_id = dept.dept_id)";

#[test]
fn an_aggregate_filter_reads_a_correlated_subquery() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            &format!("select sum(salary), sum(salary) filter (where {EXISTS}) from emp")
        )?,
        ["2450.0\t1250.0"]
    );
    assert_eq!(
        rows(
            &s,
            &format!(
                "select sum(distinct salary), count(distinct hiredate) filter (where {EXISTS}) from emp"
            )
        )?,
        ["1150.0\t4"]
    );
    Ok(())
}

#[test]
fn a_grouping_key_or_call_argument_reads_a_correlated_subquery() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            &format!("select count(hiredate), sum(salary) from emp group by {EXISTS}")
        )?,
        ["3\t1200.0", "6\t1250.0"]
    );
    assert_eq!(
        rows(
            &s,
            &format!(
                "select count(distinct hiredate), sum(distinct salary) from emp group by {EXISTS}"
            )
        )?,
        ["2\t400.0", "4\t1150.0"]
    );
    Ok(())
}

#[test]
fn two_distinct_calls_over_a_computed_key_keep_the_input_columns() -> R {
    let s = session()?;
    assert_eq!(
        rows(
            &s,
            "select count(distinct hiredate), sum(distinct salary) from emp group by dept_id is null"
        )?,
        ["1\t400.0", "6\t1150.0"]
    );
    Ok(())
}
