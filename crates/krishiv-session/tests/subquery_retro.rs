// SPDX-License-Identifier: Apache-2.0
//! Correlated subqueries found wrong by the retro review of #348 and #355:
//! a NULL outer value must still reach its group in a correlated scalar over a
//! union (Spark joins the domain with `<=>`, `DecorrelateInnerQuery`), and an
//! `IN` value that reads the right side of a join goes in the right side's
//! existence join (`subquery.scala`, `e.references` against each input).
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
        "create temporary view t1(a) as values (1), (cast(null as int))",
        "create temporary view t2(b) as values (1)",
        "create temporary view t3(c) as values (7), (8)",
        "create temporary view x(x1) as values (1), (2)",
        "create temporary view y(y1, y2) as values (1, 10), (2, 20)",
        "create temporary view z(z1) as values (10), (30)",
    ] {
        run(&s, v)?;
    }
    Ok(s)
}

#[test]
fn null_outer_value_reaches_its_group_over_a_union() -> R {
    let s = session()?;
    let q = |agg: &str| {
        format!(
            "select a, (select {agg} from (select b as v from t2 where b = t1.a \
             union all select c from t3)) from t1"
        )
    };
    assert_eq!(run(&s, &q("count(*)"))?, ["1|3", "NULL|2"]);
    assert_eq!(run(&s, &q("sum(v)"))?, ["1|16", "NULL|15"]);
    Ok(())
}

#[test]
fn in_value_on_the_right_side_of_a_join_condition() -> R {
    let s = session()?;
    assert_eq!(
        run(
            &s,
            "select * from x join y on x1 = y1 and y2 in (select z1 from z)"
        )?,
        ["1|1|10"]
    );
    // The subquery reads both sides: a named refusal, not an internal error.
    let err = run(
        &s,
        "select * from x join y on x1 = y1 and y2 in (select z1 from z where z1 = x1)",
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(!err.is_empty() && !err.contains("INTERNAL_ERROR"), "{err}");
    Ok(())
}
