// SPDX-License-Identifier: Apache-2.0
//! `Session::sql_query_with` (P3-T101 note §2, v4 review M1): a query is analyzed and not run;
//! anything else is refused with its class before it is authorized or run, in the same call that
//! applies `${...}` substitution.
#![cfg(test)]
#![cfg(feature = "unstable-ir")]

use futures::executor::block_on;
use krishiv_session::{QueryOptions, Session, SqlQuery, StatementClass};

fn query(session: &Session, sql: &str) -> SqlQuery {
    block_on(session.sql_query_with(sql, QueryOptions::default())).expect("sql_query_with")
}

fn class_of(outcome: &SqlQuery) -> Option<StatementClass> {
    match outcome {
        SqlQuery::NotAQuery(class) => Some(*class),
        SqlQuery::Query(_) => None,
        _ => unreachable!("non_exhaustive"),
    }
}

fn conf(session: &Session, key: &str) -> Option<String> {
    session.conf().get(key)
}

#[test]
fn a_query_is_analyzed_and_returned() {
    let session = Session::new().expect("session");
    let SqlQuery::Query(df) = query(&session, "SELECT 1 AS a, 'x' AS b") else {
        panic!("a query");
    };
    let schema = block_on(df.schema()).expect("schema");
    assert_eq!(schema.fields().len(), 2);
    let rows: usize = block_on(df.collect())
        .expect("rows")
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 1);
}

#[test]
fn commands_are_refused_with_their_class_and_not_run() {
    let session = Session::new().expect("session");
    assert_eq!(
        class_of(&query(&session, "SET krishiv.test.flight = 1")),
        Some(StatementClass::Session)
    );
    assert_eq!(
        conf(&session, "krishiv.test.flight"),
        None,
        "SET did not run"
    );
    assert_eq!(
        class_of(&query(
            &session,
            "CREATE TABLE flight_t (a INT) USING parquet"
        )),
        Some(StatementClass::Ddl)
    );
    assert!(
        block_on(session.sql("SELECT * FROM flight_t")).is_err(),
        "CREATE TABLE did not run"
    );
    assert_eq!(
        class_of(&query(&session, "CREATE TEMPORARY VIEW v AS SELECT 1")),
        Some(StatementClass::Session)
    );
    assert!(block_on(session.sql("SELECT * FROM v")).is_err(), "no view");
    // Not a query even though it changes nothing.
    assert_eq!(
        class_of(&query(&session, "SHOW TABLES")),
        Some(StatementClass::Read)
    );
    assert!(class_of(&query(&session, "EXPLAIN SELECT 1")).is_some());
}

#[test]
fn the_kind_follows_the_substituted_text() {
    let session = Session::new().expect("session");
    block_on(session.sql("SET flight.stmt = SELECT 7")).expect("set");
    assert!(
        class_of(&query(&session, "${flight.stmt}")).is_none(),
        "a query"
    );
    // The same text becomes a command once the variable changes; it is refused, not run.
    block_on(session.sql("SET flight.stmt = SET flight.ran = yes")).expect("set");
    assert_eq!(
        class_of(&query(&session, "${flight.stmt}")),
        Some(StatementClass::Session)
    );
    assert_eq!(conf(&session, "flight.ran"), None);
}

#[test]
fn into_query_refuses_with_not_a_query() {
    let session = Session::new().expect("session");
    let e = query(&session, "SET a.b = 1")
        .into_query()
        .err()
        .unwrap_or_else(|| panic!("refused"));
    assert_eq!(e.error_class().name(), "KRISHIV_FLIGHT_SQL_NOT_A_QUERY");
    assert_eq!(e.sql_state(), Some("55000"));
    assert!(e.to_string().contains("SESSION"), "{e}");
    assert!(query(&session, "SELECT 1").into_query().is_ok());
}

#[test]
fn a_query_that_fails_analysis_is_its_error() {
    let session = Session::new().expect("session");
    let e = block_on(session.sql_query_with("SELECT * FROM missing_t", QueryOptions::default()))
        .err()
        .unwrap_or_else(|| panic!("refused"));
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

#[test]
fn a_refused_statement_records_no_query_row() {
    let session = Session::new().expect("session");
    let count = || -> i64 {
        use arrow_array::Array as _;
        let batches = block_on(
            block_on(session.sql(
                "SELECT count(*) FROM system.queries WHERE query_text NOT LIKE 'SELECT count(*) FROM system.queries%'",
            ))
            .expect("query")
            .collect(),
        )
        .expect("rows");
        let column = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("count");
        assert_eq!(column.len(), 1);
        column.value(0)
    };
    let before = count();
    query(&session, "CREATE TABLE flight_marker (a INT) USING parquet");
    assert_eq!(count(), before, "nothing ran, no row");
    // A refusal through `sql` (here an analysis error) has its row: the count sees it.
    block_on(session.sql("SELECT * FROM flight_marker_missing"))
        .err()
        .unwrap_or_else(|| panic!("missing"));
    assert_eq!(count(), before + 1);
}
