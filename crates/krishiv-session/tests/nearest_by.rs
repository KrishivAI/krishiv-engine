// SPDX-License-Identifier: Apache-2.0
//! `NEAREST BY` top-k ranking joins (Spark 4.3): the rows worked out by hand; the
//! `join-nearest-by` golden has the same queries.
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
    run(
        &s,
        "create temporary view u(id, score) as values (1, 10), (2, 20), (3, 30)",
    )?;
    run(
        &s,
        "create temporary view p(pid, ps) as values (1, 11), (2, 22), (3, 5), (4, null)",
    )?;
    Ok(s)
}

#[test]
fn similarity_picks_the_nearest_one() -> R {
    let s = session()?;
    let rows = run(
        &s,
        "SELECT u.id, p.pid FROM u JOIN p APPROX NEAREST 1 BY SIMILARITY -abs(u.score - p.ps)",
    )?;
    assert_eq!(rows, ["1|1", "2|2", "3|2"]);
    Ok(())
}

#[test]
fn distance_with_k_two_and_default_k() -> R {
    let s = session()?;
    let rows = run(
        &s,
        "SELECT u.id, p.pid FROM u JOIN p EXACT NEAREST 2 BY DISTANCE abs(u.score - p.ps)",
    )?;
    assert_eq!(rows, ["1|1", "1|3", "2|1", "2|2", "3|1", "3|2"]);
    let rows = run(
        &s,
        "SELECT u.id, p.pid FROM u INNER JOIN p EXACT NEAREST BY DISTANCE abs(u.score - p.ps)",
    )?;
    assert_eq!(rows, ["1|1", "2|2", "3|2"]);
    Ok(())
}

#[test]
fn star_has_only_the_user_columns() -> R {
    let s = session()?;
    let rows = run(
        &s,
        "SELECT * FROM u JOIN p APPROX NEAREST 1 BY DISTANCE abs(u.score - p.ps)",
    )?;
    assert_eq!(rows, ["1|10|1|11", "2|20|2|22", "3|30|2|22"]);
    Ok(())
}

#[test]
fn empty_right_side() -> R {
    let s = session()?;
    let none = "(SELECT * FROM p WHERE false) q";
    let rows = run(
        &s,
        &format!(
            "SELECT u.id, q.pid FROM u LEFT OUTER JOIN {none} APPROX NEAREST 1 BY DISTANCE abs(u.score - q.ps)"
        ),
    )?;
    assert_eq!(rows, ["1|NULL", "2|NULL", "3|NULL"]);
    let rows = run(
        &s,
        &format!(
            "SELECT u.id, q.pid FROM u JOIN {none} APPROX NEAREST 1 BY DISTANCE abs(u.score - q.ps)"
        ),
    )?;
    assert!(rows.is_empty());
    Ok(())
}

#[test]
fn self_join_matches_itself() -> R {
    let s = session()?;
    let rows = run(
        &s,
        "SELECT a.id, b.id FROM u a JOIN u b APPROX NEAREST 1 BY DISTANCE abs(a.score - b.score)",
    )?;
    assert_eq!(rows, ["1|1", "2|2", "3|3"]);
    Ok(())
}
