// SPDX-License-Identifier: Apache-2.0
//! A DataFrame derived from a `sql` frame (`limit`, `filter`, ...) runs under the text and class of
//! the statement it came from, so `system.queries` lists the SQL of a console editor's run.
#![cfg(test)]

use arrow_array::{Array, StringArray};
use futures::executor::block_on;
use krishiv_session::Session;

fn texts(s: &Session, like: &str) -> Vec<(Option<String>, Option<String>)> {
    let batches = block_on(
        block_on(s.sql(&format!(
            "SELECT CAST(query_text AS STRING), CAST(statement_class AS STRING) FROM system.queries \
             WHERE query_text LIKE '{like}'"
        )))
        .unwrap()
        .collect(),
    )
    .unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let t = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let c = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((
                (!t.is_null(i)).then(|| t.value(i).to_owned()),
                (!c.is_null(i)).then(|| c.value(i).to_owned()),
            ));
        }
    }
    out
}

#[test]
fn a_derived_frame_run_shows_the_statement_it_came_from() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id FROM range(0, 50) /*derived-marker*/")).unwrap();
    let limited = df.limit(5);
    let rows: usize = block_on(limited.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
    let got = texts(&s, "SELECT id FROM range%derived-marker%");
    assert!(
        got.len() == 1 && got[0].0.is_some() && got[0].1.is_some(),
        "the derived frame's run has the statement's text and class: {got:?}"
    );
}

#[test]
fn a_frame_no_sql_started_still_has_no_text() {
    let s = Session::new().unwrap();
    let df = s.range(0, 10, 1, None).limit(3);
    block_on(df.collect()).unwrap();
    let rows = block_on(
        block_on(s.sql(
            "SELECT count(*) FROM system.queries WHERE query_text IS NULL \
             AND NOT (state = 'running')",
        ))
        .unwrap()
        .collect(),
    )
    .unwrap();
    let n = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap()
        .value(0);
    assert!(n >= 1, "the range frame's run keeps a null text");
}
