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

/// Typing the markers of a statement over a `SQL SECURITY DEFINER` view whose reads the caller
/// may not see (P3-T101 note §17 A1, PR-65d review B1): the markers stay untyped, and nothing is
/// a denial, in the session's counters or the grants provider's; `analyzed()`, which shows the
/// plan, still refuses and counts.
#[test]
fn marker_types_over_a_hidden_definer_view_deny_nothing() {
    use std::sync::Arc;

    use krishiv_session::auth::{DeniedCounts, Surface};
    use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
    use krishiv_session::parameter_types::BoundAs;
    use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind};

    let dir = tempfile::tempdir().expect("tempdir");
    let rt = EngineRuntime::builder()
        .config(
            "spark.sql.warehouse.dir",
            dir.path().join("wh").to_str().expect("utf-8"),
        )
        .build()
        .expect("runtime");
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let session = |name: &str, admin: bool| {
        let policy = Arc::new(GrantsPolicy::new(
            Arc::clone(&grants),
            Arc::new(AllowAll),
            admin,
        ));
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(Principal::new(name, PrincipalKind::User))
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .expect("session")
    };
    let run = |s: &Session, sql: &str| {
        block_on(async { s.sql(sql).await?.collect().await })
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    };
    let (root, ana) = (session("root", true), session("ana", false));
    for sql in [
        "CREATE TABLE t (a INT, b INT) USING parquet",
        "GRANT CREATE ON SCHEMA default TO user:ana",
        "GRANT SELECT ON TABLE t TO user:ana",
    ] {
        run(&root, sql);
    }
    for sql in [
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t",
        "GRANT SELECT ON VIEW dv TO user:bob",
    ] {
        run(&ana, sql);
    }
    let bob = session("bob", false);
    let counts = Arc::new(DeniedCounts::new());
    assert!(bob.count_denials(Arc::clone(&counts), Surface::FlightSql));
    let total = || -> u64 {
        counts.counts().iter().map(|c| c.count).sum::<u64>()
            + grants.denied_counts().iter().map(|c| c.count).sum::<u64>()
    };
    let before = total();
    // `SELECT a FROM dv WHERE a = ?` bound with `NULL`: the marker's value starts at 27.
    let text = "SELECT a FROM dv WHERE a = NULL";
    let SqlQuery::Query(df) = query(&bob, text) else {
        panic!("a query");
    };
    let span = 27..31;
    let types =
        block_on(df.marker_types(std::slice::from_ref(&span), BoundAs::Null)).expect("types");
    assert_eq!(types, [None], "a plan bob may not be shown types nothing");
    assert_eq!(total(), before, "and is no denial");
    assert!(
        block_on(df.analyzed()).is_err(),
        "showing the plan is refused"
    );
    assert!(total() > before, "and that refusal is counted");
}

/// A marker is typed only from the literal it was bound to (PR-65d review M1): with `NULL`s a
/// `1` at the marker's place types nothing, and with `1`s a `NULL` types nothing.
#[test]
fn marker_types_match_only_the_bound_literal() {
    use krishiv_session::parameter_types::BoundAs;

    let session = Session::new().expect("session");
    let typed = |text: &str, span: std::ops::Range<usize>, bound: BoundAs| {
        let SqlQuery::Query(df) = query(&session, text) else {
            panic!("{text}: a query");
        };
        block_on(df.marker_types(std::slice::from_ref(&span), bound)).expect("types")
    };
    // `CAST(? AS BIGINT)` bound with NULL, then with 1.
    let null = "SELECT CAST(NULL AS BIGINT) AS b";
    let one = "SELECT CAST(1 AS BIGINT) AS b";
    let bigint = Some(krishiv_types::DataType::Long);
    assert_eq!(
        typed(null, 12..16, BoundAs::Null),
        std::slice::from_ref(&bigint)
    );
    assert_eq!(typed(one, 12..13, BoundAs::One), [bigint]);
    assert_eq!(typed(one, 12..13, BoundAs::Null), [None]);
    assert_eq!(typed(null, 12..16, BoundAs::One), [None]);
}

/// `insert_parameter_types` (P3-T101 note §17 A1): a whole `INSERT ... VALUES` cell takes its
/// target column's type, by position or by the column list; every other shape types nothing.
#[test]
fn insert_parameter_types_follow_the_target_columns() {
    use krishiv_types::DataType;

    let dir = tempfile::tempdir().expect("tempdir");
    let session = Session::builder()
        .config(
            "spark.sql.warehouse.dir",
            dir.path().join("wh").to_str().expect("utf-8"),
        )
        .build()
        .expect("session");
    for sql in [
        "CREATE TABLE t (a INT, d DATE, s STRING) USING parquet",
        "CREATE TABLE p (a INT, k STRING) USING parquet PARTITIONED BY (k)",
        "CREATE VIEW v AS SELECT 1 AS a",
    ] {
        block_on(async { session.sql(sql).await?.collect().await }).expect(sql);
    }
    let types = |sql: &str, markers: usize| block_on(session.insert_parameter_types(sql, markers));
    let (int, date) = (Some(DataType::Integer), Some(DataType::Date));
    assert_eq!(
        types("INSERT INTO t VALUES (?, ?, 'x')", 2),
        [int.clone(), date.clone()]
    );
    assert_eq!(
        types("INSERT INTO t (d, a) VALUES (?, (?)), (?, 1)", 3),
        [date.clone(), int.clone(), date]
    );
    assert_eq!(types("INSERT OVERWRITE t VALUES (?, NULL, NULL)", 1), [int]);
    for (sql, markers) in [
        // Not a whole cell, not VALUES, or not an INSERT.
        ("INSERT INTO t VALUES (? + 1, NULL, NULL)", 1),
        ("INSERT INTO t SELECT ?, NULL, NULL", 1),
        ("SELECT ?", 1),
        ("not sql at all ?", 1),
        // A PARTITION clause, rows of different widths, a column list that does not fit.
        ("INSERT INTO p PARTITION (k = 'x') VALUES (?)", 1),
        ("INSERT INTO t VALUES (?, NULL, NULL), (1, NULL)", 1),
        ("INSERT INTO t (a, nope) VALUES (?, ?)", 2),
        ("INSERT INTO t (a) VALUES (?, ?)", 2),
        // A missing table, a view, a multi-table INSERT.
        ("INSERT INTO missing VALUES (?)", 1),
        ("INSERT INTO v VALUES (?)", 1),
        (
            "FROM t INSERT INTO t SELECT ?, d, s INSERT INTO t SELECT a, d, ?",
            2,
        ),
    ] {
        assert!(
            types(sql, markers).iter().all(Option::is_none),
            "{sql}: {:?}",
            types(sql, markers)
        );
    }
    // More markers than the statement has: the extra ones stay untyped.
    assert_eq!(types("INSERT INTO t VALUES (?, NULL, NULL)", 2)[1], None);
}

/// `insert_parameter_types` says nothing about a table the caller may not write: the quiet
/// target check comes before any catalog read.
#[test]
fn insert_parameter_types_need_the_write() {
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("tempdir");
    let wh = dir.path().join("wh");
    let owner = Session::builder()
        .config("spark.sql.warehouse.dir", wh.to_str().expect("utf-8"))
        .build()
        .expect("session");
    block_on(async {
        owner
            .sql("CREATE TABLE t (a INT) USING parquet")
            .await?
            .collect()
            .await
    })
    .expect("create");
    let reader = owner.new_session_as(
        krishiv_session::Principal::new("reader", krishiv_session::PrincipalKind::User),
        Arc::new(krishiv_session::ReadOnlyPolicy::new(Arc::new(
            krishiv_session::AllowAll,
        ))),
    );
    assert_eq!(
        block_on(reader.insert_parameter_types("INSERT INTO t VALUES (?)", 1)),
        [None]
    );
}

/// Writes one row of a single column `column` to the Parquet directory `dir`, replacing it.
fn write_column(session: &Session, dir: &str, column: &str) {
    let df = block_on(session.sql(&format!("SELECT 1 AS {column}"))).expect("frame");
    block_on(df.write().mode("overwrite").parquet(dir)).expect("write");
}

#[test]
fn held_schema_reuses_the_analysis_and_analyzes_nothing() {
    // MODES-4 F1: a frame over files is analyzed again by every `schema` call; `held_schema`
    // reads the analysis `sql_query_with` made. The files change after that analysis, so a second
    // analysis would show the new column.
    let session = Session::new().expect("session");
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("t");
    let path = path.to_str().expect("utf-8 path");
    write_column(&session, path, "a");
    let SqlQuery::Query(df) = query(&session, &format!("SELECT * FROM parquet.`{path}`")) else {
        panic!("a query");
    };
    write_column(&session, path, "b");
    for _ in 0..2 {
        let held = block_on(df.held_schema()).expect("held schema");
        assert_eq!(held.field(0).name(), "a", "held_schema analyzed again");
    }
    // `schema` revalidates: it sees the new file, and keeps that analysis.
    let fresh = block_on(df.schema()).expect("schema");
    assert_eq!(fresh.field(0).name(), "b");
    let held = block_on(df.held_schema()).expect("held schema");
    assert_eq!(held.field(0).name(), "b");
}

#[test]
fn held_schema_of_a_frame_with_no_analysis_is_its_schema() {
    let session = Session::new().expect("session");
    let df = session.range(0, 3, 1, None);
    let held = block_on(df.held_schema()).expect("held schema");
    assert_eq!(held, block_on(df.schema()).expect("schema"));
}
