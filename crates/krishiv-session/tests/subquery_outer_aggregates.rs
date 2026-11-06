// SPDX-License-Identifier: Apache-2.0
//! Outer aggregates: an aggregate call in the `WHERE` of a subquery under a
//! `HAVING` that reads only the enclosing aggregate's input
//! (`HAVING EXISTS (SELECT 1 FROM bonus WHERE bonus_amt < min(emp.salary))`)
//! is computed by that aggregate, per group. Anywhere else Spark raises what
//! the call raises in the subquery. Every expected answer and error class is
//! what a Spark 4.2.0 run gave (`exists-having.sql` #3-#7 have the shape).
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
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
        DataType::Float64 => format!("{:?}", c.as_primitive::<Float64Type>().value(row)),
        DataType::Boolean => c.as_boolean().value(row).to_string(),
        DataType::Utf8 => c.as_string::<i32>().value(row).to_owned(),
        DataType::Utf8View => c.as_string_view().value(row).to_owned(),
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

/// The error class `sql` fails with.
fn error_class(s: &Session, sql: &str) -> String {
    match run(s, sql) {
        Ok(rows) => format!("answered {rows:?}"),
        Err(e) => e.to_string(),
    }
}

fn session() -> R<Session> {
    let s = Session::new()?;
    run(
        &s,
        "CREATE TEMPORARY VIEW EMP(id, emp_name, hiredate, salary, dept_id) AS VALUES\n  (100, 'emp 1', date '2005-01-01', double(100.00), 10),\n  (100, 'emp 1', date '2005-01-01', double(100.00), 10),\n  (200, 'emp 2', date '2003-01-01', double(200.00), 10),\n  (300, 'emp 3', date '2002-01-01', double(300.00), 20),\n  (400, 'emp 4', date '2005-01-01', double(400.00), 30),\n  (500, 'emp 5', date '2001-01-01', double(400.00), NULL),\n  (600, 'emp 6 - no dept', date '2001-01-01', double(400.00), 100),\n  (700, 'emp 7', date '2010-01-01', double(400.00), 100),\n  (800, 'emp 8', date '2016-01-01', double(150.00), 70)",
    )?;
    run(
        &s,
        "CREATE TEMPORARY VIEW DEPT(dept_id, dept_name, state) AS VALUES\n  (10, 'dept 1', 'CA'),\n  (20, 'dept 2', 'NY'),\n  (30, 'dept 3', 'TX'),\n  (40, 'dept 4 - unassigned', 'OR'),\n  (50, 'dept 5 - unassigned', 'NJ'),\n  (70, 'dept 7', 'FL')",
    )?;
    run(
        &s,
        "CREATE TEMPORARY VIEW BONUS(emp_name, bonus_amt) AS VALUES\n  ('emp 1', double(10.00)),\n  ('emp 1', double(20.00)),\n  ('emp 2', double(300.00)),\n  ('emp 2', double(100.00)),\n  ('emp 3', double(300.00)),\n  ('emp 4', double(100.00)),\n  ('emp 5', double(1000.00)),\n  ('emp 6 - no dept', double(500.00))",
    )?;
    Ok(s)
}

#[test]
fn an_outer_aggregate_in_a_having_subquery_is_computed_per_group() -> R {
    let s = session()?;
    let q = "SELECT dept_id, count(*) FROM emp GROUP BY dept_id \
             HAVING EXISTS (SELECT 1 FROM bonus WHERE bonus_amt < min(emp.salary))";
    assert_eq!(
        run(&s, q)?,
        rows(&["10|3", "100|2", "20|1", "30|1", "70|1", "NULL|1"]),
        "{q}"
    );
    // Under an EXISTS of the outer query, correlated to it.
    let q = "SELECT * FROM dept WHERE EXISTS (SELECT dept_id, count(emp.dept_id) FROM emp \
             WHERE dept.dept_id = dept_id GROUP BY dept_id \
             HAVING EXISTS (SELECT 1 FROM bonus \
             WHERE (bonus_amt > min(emp.salary) AND count(emp.dept_id) > 1)))";
    assert_eq!(run(&s, q)?, rows(&["10|dept 1|CA"]), "{q}");
    // Below the subquery's own aggregate.
    let q = "select dept_id from emp group by dept_id \
             having (select count(*) from bonus where bonus_amt < min(emp.salary)) > 1";
    assert_eq!(
        run(&s, q)?,
        rows(&["20", "30", "100", "NULL", "10", "70"]),
        "{q}"
    );
    // A global aggregate, a FILTER clause and a DISTINCT call.
    let q = "select max(salary) from emp \
             having exists(select 1 from bonus where bonus_amt > max(emp.salary))";
    assert_eq!(run(&s, q)?, rows(&["400.0"]), "{q}");
    let q = "select dept_id from emp group by dept_id having exists(select 1 from bonus \
             where bonus_amt < min(emp.salary) filter (where emp.id > 100))";
    assert_eq!(
        run(&s, q)?,
        rows(&["20", "10", "NULL", "100", "70", "30"]),
        "{q}"
    );
    let q = "select dept_id, count(*) from emp group by dept_id having exists(select 1 \
             from bonus where bonus_amt < count(distinct emp.salary) * 100)";
    assert_eq!(
        run(&s, q)?,
        rows(&["NULL|1", "20|1", "100|2", "10|3", "70|1", "30|1"]),
        "{q}"
    );
    Ok(())
}

#[test]
fn an_outer_aggregate_anywhere_else_is_refused_as_in_spark() -> R {
    let s = session()?;
    for (q, class) in [
        // No enclosing aggregate, or not its HAVING.
        (
            "select * from emp where exists(select 1 from bonus where bonus_amt < min(emp.salary))",
            "INVALID_WHERE_CONDITION",
        ),
        (
            "select dept_id, exists(select 1 from bonus where bonus_amt < min(emp.salary)) \
             from emp group by dept_id",
            "INVALID_WHERE_CONDITION",
        ),
        // Beside a call over the subquery's own rows.
        (
            "select dept_id from emp group by dept_id having exists(select 1 from bonus \
             where bonus_amt < min(emp.salary) + sum(bonus_amt))",
            "INVALID_WHERE_CONDITION",
        ),
        // One call over outer and local columns.
        (
            "select dept_id from emp group by dept_id \
             having exists(select 1 from bonus where min(bonus_amt + emp.salary) > 0)",
            "AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES] Unsupported subquery expression: Found an aggregate function in a correlated predicate that has both outer and local references, which is not supported: min((bonus.bonus_amt + outer(emp.salary)))",
        ),
        // So it is without an enclosing aggregate.
        (
            "select * from emp where exists(select 1 from bonus where min(bonus_amt + emp.salary) > 0)",
            "AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES",
        ),
        // A call of the subquery's own aggregate.
        (
            "select dept_id from emp group by dept_id having exists(select max(bonus_amt) \
             from bonus having max(bonus_amt) > min(emp.salary))",
            "CORRELATED_REFERENCE",
        ),
        // A column that is not grouped, beside the outer aggregate.
        (
            "select dept_id, count(*) from emp group by dept_id having exists(select 1 \
             from bonus where bonus_amt < min(emp.salary) and bonus.emp_name = emp.emp_name)",
            "MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT",
        ),
    ] {
        let got = error_class(&s, q);
        assert!(got.contains(class), "{q}: {got}");
    }
    // Spark: CORRELATED_REFERENCE (a call of the subquery's own global
    // aggregate); refused here as a correlated subquery shape not supported
    // yet, never answered.
    let q = "select dept_id, (select min(emp.salary) + max(bonus_amt) from bonus) \
             from emp group by dept_id";
    assert!(run(&s, q).is_err(), "{q}");
    Ok(())
}
