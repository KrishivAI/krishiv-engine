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
    assert_eq!(
        got,
        [(
            Some("SELECT id FROM range(0, 50) /*derived-marker*/".to_owned()),
            Some("READ".to_owned())
        )],
        "the derived frame's run has the statement's text and class"
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
    assert_eq!(n, 1, "the range frame's run keeps a null text");
}

/// The single string column of `sql`'s result, in order.
fn strings(s: &Session, sql: &str) -> Vec<Option<String>> {
    let batches = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let t = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            out.push((!t.is_null(i)).then(|| t.value(i).to_owned()));
        }
    }
    out
}

/// What the web console's SQL editor does: the statement with a request tag, a row cap over it, the schema of
/// the capped frame, then its rows. The one run is listed with the statement's text and its tag.
#[test]
fn a_tagged_statement_read_through_a_row_cap_is_one_run_with_its_text_and_tag() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql_with(
        "SELECT id FROM range(0, 50) /*editor-marker*/",
        krishiv_session::QueryOptions::new().tags(["ui-run-1"]),
    ))
    .unwrap();
    let capped = df.limit(10);
    assert_eq!(block_on(capped.schema()).unwrap().fields().len(), 1);
    block_on(capped.collect()).unwrap();
    let got = strings(
        &s,
        "SELECT CAST(query_text AS STRING) FROM system.queries WHERE tags['ui-run-1'] IS NOT NULL",
    );
    assert_eq!(
        got,
        [Some(
            "SELECT id FROM range(0, 50) /*editor-marker*/".to_owned()
        )]
    );
}

/// A frame derived from a derived frame still runs under the first statement's text, and every run of it is
/// listed with that text.
#[test]
fn a_chain_of_derived_frames_keeps_the_text_on_every_run() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id FROM range(0, 50) /*chain-marker*/")).unwrap();
    let chained = df.filter_expr("id > 40").unwrap().limit(3);
    block_on(chained.collect()).unwrap();
    block_on(chained.collect()).unwrap();
    let got = strings(
        &s,
        "SELECT CAST(query_text AS STRING) FROM system.queries \
         WHERE query_text LIKE 'SELECT id FROM range%chain-marker%'",
    );
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(
        got.iter()
            .all(|t| t.as_deref() == Some("SELECT id FROM range(0, 50) /*chain-marker*/")),
        "{got:?}"
    );
}
