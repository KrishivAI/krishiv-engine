// SPDX-License-Identifier: Apache-2.0
//! An uncorrelated `EXISTS` or scalar subquery written in a join's `ON` is one constant for every
//! row pair, so the left side carries it as a column and every join kind plans it (a FULL OUTER
//! join, an `OR`, or a condition on the preserved side cannot hold the subquery where it stands).
//! `f1` is 1, 2; `f2` is 2, 3; `tt` is 2, 9; `te` is empty. Expected rows are worked out from
//! those by hand (a Spark run of these statements is requested); where a second formulation
//! exists (a constant `TRUE`/`FALSE`, or the value written as a literal) the test compares with it.
#![cfg(test)]

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

fn rows(v: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = v.iter().map(|r| (*r).to_owned()).collect();
    out.sort();
    out
}

fn session() -> R<Session> {
    let s = Session::new()?;
    for q in [
        "create temporary view f1(id) as values (1), (2)",
        "create temporary view f2(id) as values (2), (3)",
        "create temporary view tt(id) as values (2), (9)",
        "create temporary view te(id) as select 1 as id where false",
    ] {
        run(&s, q)?;
    }
    Ok(s)
}

const KINDS: [&str; 4] = ["join", "left join", "right join", "full outer join"];

fn join(kind: &str, cond: &str) -> String {
    format!("select f1.id, f2.id from f1 {kind} f2 on {cond}")
}

/// A condition that is a constant TRUE for every pair joins like `on true`, a constant FALSE
/// like `on false`, in every join kind: EXISTS over `tt`, NOT EXISTS over `te`, and the other way.
#[test]
fn a_constant_exists_joins_like_true_or_false_in_every_kind() -> R {
    let s = session()?;
    for kind in KINDS {
        for (cond, constant) in [
            ("exists (select 1 from tt)", "true"),
            ("not exists (select 1 from te)", "true"),
            ("exists (select 1 from te)", "false"),
            ("not exists (select 1 from tt)", "false"),
        ] {
            let (got, want) = (run(&s, &join(kind, cond))?, run(&s, &join(kind, constant))?);
            assert_eq!(got, want, "{kind} on {cond}");
        }
    }
    // The constant forms themselves: TRUE pairs everything, FALSE leaves each side alone.
    let s2 = &s;
    assert_eq!(
        run(s2, &join("full outer join", "true"))?,
        rows(&["1|2", "1|3", "2|2", "2|3"])
    );
    assert_eq!(
        run(s2, &join("full outer join", "false"))?,
        rows(&["1|NULL", "2|NULL", "NULL|2", "NULL|3"])
    );
    Ok(())
}

/// A scalar subquery is the same constant: `max(tt.id)` is 9, so it equals the literal 9 in every
/// kind, whichever side the other operand is.
#[test]
fn a_scalar_subquery_joins_like_its_value_in_every_kind() -> R {
    let s = session()?;
    for kind in KINDS {
        for (cond, literal) in [
            ("f1.id = (select max(id) from tt)", "f1.id = 9"),
            ("f2.id = (select max(id) from tt)", "f2.id = 9"),
            ("f1.id < (select min(id) from tt)", "f1.id < 2"),
            (
                "f1.id = f2.id and f2.id = (select max(id) from tt)",
                "f1.id = f2.id and f2.id = 9",
            ),
            (
                "f1.id = f2.id and f1.id = (select min(id) from tt)",
                "f1.id = f2.id and f1.id = 2",
            ),
        ] {
            let (got, want) = (run(&s, &join(kind, cond))?, run(&s, &join(kind, literal))?);
            assert_eq!(got, want, "{kind} on {cond}");
        }
    }
    Ok(())
}

/// Beside an equality and under `OR`: the pair matches when the keys match and the constant holds,
/// or (for `OR`) when the constant holds alone.
#[test]
fn a_constant_beside_a_key_and_under_or() -> R {
    let s = session()?;
    for kind in KINDS {
        for (cond, plain) in [
            (
                "f1.id = f2.id and exists (select 1 from tt)",
                "f1.id = f2.id and true",
            ),
            (
                "f1.id = f2.id or exists (select 1 from tt)",
                "f1.id = f2.id or true",
            ),
            (
                "f1.id = f2.id or exists (select 1 from te)",
                "f1.id = f2.id or false",
            ),
            (
                "f1.id = f2.id or f2.id = (select max(id) from tt)",
                "f1.id = f2.id or f2.id = 9",
            ),
        ] {
            let (got, want) = (run(&s, &join(kind, cond))?, run(&s, &join(kind, plain))?);
            assert_eq!(got, want, "{kind} on {cond}");
        }
    }
    // A FULL OUTER join on the key alone: 2 matches, 1 and 3 stand alone.
    assert_eq!(
        run(
            &s,
            &join(
                "full outer join",
                "f1.id = f2.id and exists (select 1 from tt)"
            )
        )?,
        rows(&["1|NULL", "2|2", "NULL|3"])
    );
    Ok(())
}

/// A scalar subquery over no rows is NULL, so a comparison with it is not TRUE: no pair matches.
#[test]
fn a_scalar_over_no_rows_is_null_and_matches_nothing() -> R {
    let s = session()?;
    let q = join("full outer join", "f1.id = (select max(id) from te)");
    assert_eq!(
        run(&s, &q)?,
        rows(&["1|NULL", "2|NULL", "NULL|2", "NULL|3"]),
        "{q}"
    );
    Ok(())
}

/// The lifted column does not reach the output: the select list is the join's own.
#[test]
fn the_lifted_column_is_not_in_the_output() -> R {
    let s = session()?;
    let q = "select * from f1 full outer join f2 on exists (select 1 from tt) and f1.id = f2.id";
    let batches = block_on(async { s.sql(q).await?.collect().await })?;
    assert!(batches.iter().all(|b| b.num_columns() == 2), "{q}");
    Ok(())
}
