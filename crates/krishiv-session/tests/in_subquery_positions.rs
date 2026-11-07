// SPDX-License-Identifier: Apache-2.0
//! `IN (subquery)` is two-valued where Spark rewrites it to an existence
//! join (a select list, a `WHERE`, and an `ON` condition whose `IN` reads a
//! join input) and three-valued where it stays an expression (an `ON`
//! condition whose `IN` reads neither input, Spark's
//! `RewritePredicateSubquery` join case). Expected rows are Spark 4.2.0's
//! answers from a local pyspark 4.2.0 run (`tests/scripts/spark_answers.py`),
//! plus the in-nullability and in-joins goldens.
#![forbid(unsafe_code)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
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

fn session() -> R<Session> {
    let s = Session::new()?;
    for v in [
        "create temporary view t1 as select cast(null as int) as b",
        "create temporary view t2 as select 2 as c",
        "create temporary view t4 as select * from values (1), (null) as v(b)",
        "create temporary view t6(a) as values (1), (3), (null)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

fn check(cases: &[(&str, &[&str])]) -> R {
    let s = session()?;
    let mut wrong = Vec::new();
    for (q, want) in cases {
        let mut want: Vec<String> = want.iter().map(|r| (*r).to_owned()).collect();
        want.sort();
        match run(&s, q) {
            Ok(got) if got == want => {}
            Ok(got) => wrong.push(format!("{q}\n  got {got:?}\n  want {want:?}")),
            Err(e) => wrong.push(format!("{q}\n  error {e}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    Ok(())
}

/// Over a one-row relation of literals as over any other: a NULL value is
/// `false` in a select list and a `WHERE`.
#[test]
fn a_one_row_literal_subquery_is_two_valued_outside_on_conditions() -> R {
    check(&[
        (
            "select a, a in (select 1) from t6",
            &["1|true", "3|false", "NULL|false"],
        ),
        (
            "select a, a in (select b from t1) from t6",
            &["1|false", "3|false", "NULL|false"],
        ),
        (
            "select a, a not in (select b from t1) from t6",
            &["1|false", "3|false", "NULL|false"],
        ),
        (
            "select * from t6 where (a in (select b from t1)) <=> false",
            &["1", "3", "NULL"],
        ),
        ("select * from t6 where (a in (select 1)) is null", &[]),
        (
            "select * from t6 where not ((a in (select 1)) <=> true)",
            &["3", "NULL"],
        ),
        (
            "select a, (a, 1) in (select 1, cast(null as int)) from t6",
            &["1|false", "3|false", "NULL|false"],
        ),
        (
            "select a, 3 in (select cast(null as int)) from t6",
            &["1|false", "3|false", "NULL|false"],
        ),
    ])
}

/// An `ON` condition's `IN` that reads a join input is an existence join
/// too: two-valued.
#[test]
fn an_on_condition_in_over_a_join_input_is_two_valued() -> R {
    check(&[
        (
            "select * from t6 left join t2 on (a in (select 1)) is null",
            &["1|NULL", "3|NULL", "NULL|NULL"],
        ),
        (
            "select * from t6 left join t2 on (a not in (select 1)) is null",
            &["1|NULL", "3|NULL", "NULL|NULL"],
        ),
        (
            "select * from t6 left join t2 on (a in (select 1)) <=> false",
            &["1|NULL", "3|2", "NULL|2"],
        ),
        (
            "select * from t6 left join t2 on (a in (select b from t1)) is null",
            &["1|NULL", "3|NULL", "NULL|NULL"],
        ),
        (
            "select * from t6 join t2 on a in (select 1) or c = 5",
            &["1|2"],
        ),
    ])
}

/// An `ON` condition's `IN` that reads neither input stays an expression:
/// three-valued (`3 IN (1, NULL)` is NULL). Spark folds a column of a
/// side that is a select list over no relation (`SELECT 1 AS a`) into the
/// condition first, so such an `IN` is refused (in-nullability 7-8).
#[test]
fn an_on_condition_in_over_no_join_input_is_three_valued() -> R {
    check(&[
        (
            "select * from t6 left join t2 on (3 in (select b from t4)) is null",
            &["1|2", "3|2", "NULL|2"],
        ),
        (
            "select * from t6 left join t2 on (3 not in (select b from t4)) is null",
            &["1|2", "3|2", "NULL|2"],
        ),
        (
            "select * from t6 left join t2 on a = 1 and (3 in (select b from t4)) is null",
            &["1|2", "3|NULL", "NULL|NULL"],
        ),
        (
            "select * from t6 left join t2 on (3 in (select cast(null as int))) is null",
            &["1|2", "3|2", "NULL|2"],
        ),
    ])?;
    let s = session()?;
    run(&s, "create temporary view t0 as select 1 as a")?;
    let q = "select * from t0 left join t2 on (a in (select b from t1)) is null";
    assert!(run(&s, q).is_err(), "{q}");
    Ok(())
}
