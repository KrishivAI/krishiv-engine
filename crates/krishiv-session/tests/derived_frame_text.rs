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

/// The text and class of every run that carries the request tag `tag`, the runs with a text first.
fn tagged(s: &Session, tag: &str) -> Vec<Option<String>> {
    let text = strings(
        s,
        &format!(
            "SELECT CAST(query_text AS STRING) FROM system.queries WHERE tags['{tag}'] IS NOT NULL"
        ),
    );
    let class = strings(
        s,
        &format!(
            "SELECT CAST(statement_class AS STRING) FROM system.queries WHERE tags['{tag}'] IS NOT NULL"
        ),
    );
    let mut rows: Vec<Option<String>> = text
        .into_iter()
        .zip(class)
        .map(|(t, c)| {
            assert_eq!(t.is_some(), c.is_some(), "a text and a class come together");
            Some(format!("{} | {}", t?, c?))
        })
        .collect();
    rows.sort_by_key(Option::is_none);
    rows
}

/// The editor's sequence over a write: the run that wrote is the one listed with the statement's text and
/// class. The capped read of the write's empty result is a run of its own and carries neither.
#[test]
fn a_write_run_through_a_row_cap_is_listed_once_with_its_text_on_the_write() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zt (a INT) USING parquet")).unwrap();
    let df = block_on(s.sql_with(
        "INSERT INTO zt VALUES (1), (2)",
        krishiv_session::QueryOptions::new().tags(["ui-run-9"]),
    ))
    .unwrap();
    let write = Some("INSERT INTO zt VALUES (1), (2) | WRITE".to_owned());
    assert_eq!(
        tagged(&s, "ui-run-9"),
        std::slice::from_ref(&write),
        "the write's own row"
    );
    let capped = df.limit(10);
    assert_eq!(block_on(capped.schema()).unwrap().fields().len(), 0);
    block_on(capped.collect()).unwrap();
    assert_eq!(tagged(&s, "ui-run-9"), [write.clone(), None]);
    // A plain collect of the statement's frame is the same trivial read.
    block_on(df.collect()).unwrap();
    assert_eq!(tagged(&s, "ui-run-9"), [write, None, None]);
    // A command that runs no query keeps its text on the read of its result, as before.
    let ddl = block_on(s.sql_with(
        "CREATE TABLE zu (a INT) USING parquet",
        krishiv_session::QueryOptions::new().tags(["ui-run-10"]),
    ))
    .unwrap();
    block_on(ddl.limit(10).collect()).unwrap();
    assert_eq!(
        tagged(&s, "ui-run-10"),
        [Some(
            "CREATE TABLE zu (a INT) USING parquet | DDL".to_owned()
        )]
    );
}

/// A write that fails while it runs is listed once: the run that failed, with the statement's text and class.
/// The statement gets no second, refused row beside it.
#[test]
fn a_write_that_fails_while_it_runs_is_listed_once_with_its_text() {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE zf (a INT) USING parquet")).unwrap();
    let failed = block_on(s.sql_with(
        "INSERT INTO zf SELECT CAST('x' AS INT)",
        krishiv_session::QueryOptions::new().tags(["ui-run-12"]),
    ));
    assert!(failed.is_err(), "the conversion fails while the write runs");
    assert_eq!(
        tagged(&s, "ui-run-12"),
        [Some(
            "INSERT INTO zf SELECT CAST('x' AS INT) | WRITE".to_owned()
        )]
    );
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(state AS STRING) FROM system.queries WHERE tags['ui-run-12'] IS NOT NULL"
        ),
        [Some("failed".to_owned())]
    );
    // A statement refused before any query of it began still has its refused row, with the text.
    let refused = block_on(s.sql_with(
        "INSERT INTO no_such_table VALUES (1)",
        krishiv_session::QueryOptions::new().tags(["ui-run-13"]),
    ));
    assert!(refused.is_err());
    assert_eq!(
        strings(
            &s,
            "SELECT CAST(query_text AS STRING) FROM system.queries WHERE tags['ui-run-13'] IS NOT NULL"
        ),
        [Some("INSERT INTO no_such_table VALUES (1)".to_owned())]
    );
}
