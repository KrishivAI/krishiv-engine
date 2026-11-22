// SPDX-License-Identifier: Apache-2.0
//! `CREATE MATERIALIZED VIEW`, `CREATE STREAMING TABLE`, `CREATE FLOW`, `LOAD DATA`, `INSERT OVERWRITE
//! DIRECTORY` and unbound parameter markers through `Session::sql`, against a real Spark 4.2.0 run.
#![cfg(test)]

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("CREATE TABLE nat (a INT) USING parquet")).unwrap();
    s
}

fn error(s: &Session, sql: &str) -> KrishivError {
    match block_on(s.sql(sql)) {
        Err(e) => e,
        Ok(df) => block_on(df.collect()).expect_err(sql),
    }
}

#[test]
fn pipeline_datasets_are_refused_at_execution() {
    let s = session();
    for (sql, kind) in [
        (
            "CREATE MATERIALIZED VIEW mv AS SELECT 1",
            "MATERIALIZED VIEW",
        ),
        ("CREATE STREAMING TABLE st AS SELECT 1", "STREAMING TABLE"),
        (
            "CREATE MATERIALIZED VIEW IF NOT EXISTS mv (a INT COMMENT 'c') COMMENT 'x' PARTITIONED BY (a) \
             TBLPROPERTIES ('k'='v') AS SELECT 1 AS a",
            "MATERIALIZED VIEW",
        ),
        (
            "CREATE MATERIALIZED VIEW mv USING parquet AS SELECT 1",
            "MATERIALIZED VIEW",
        ),
        // Spark: INTERNAL_ERROR; here the refusal of the form with a query.
        ("CREATE STREAMING TABLE st", "STREAMING TABLE"),
    ] {
        let e = error(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::UnsupportedFeatureCreatePipelineDatasetQueryExecution,
            "{sql}"
        );
        assert!(
            e.to_string()
                .contains(&format!("Direct execution of a CREATE ... {kind} query.")),
            "{e}"
        );
        assert_eq!(e.sql_state(), Some("0A000"));
    }
    assert_eq!(
        error(&s, "CREATE MATERIALIZED VIEW mv").error_class(),
        ErrorClass::LegacyErrorTemp0035
    );
}

#[test]
fn flows_resolve_their_target_first() {
    let s = session();
    block_on(s.sql("CREATE TABLE t (a INT) USING parquet")).unwrap();
    assert_eq!(
        error(&s, "CREATE FLOW f AS INSERT INTO t BY NAME SELECT 1").error_class(),
        ErrorClass::UnsupportedFeatureCreateFlowQueryExecution
    );
    assert_eq!(
        error(&s, "CREATE FLOW f AS INSERT INTO missing BY NAME SELECT 1").error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn load_data_on_a_datasource_table() {
    let s = session();
    let e = error(&s, "LOAD DATA INPATH 'x' INTO TABLE nat");
    assert_eq!(e.error_class(), ErrorClass::LegacyErrorTemp1261);
    assert!(
        e.to_string().contains(
            "LOAD DATA is not supported for datasource tables: `spark_catalog`.`default`.`nat`."
        ),
        "{e}"
    );
    assert_eq!(
        error(&s, "LOAD DATA LOCAL INPATH 'x' OVERWRITE INTO TABLE nat").error_class(),
        ErrorClass::LegacyErrorTemp1261
    );
    assert_eq!(
        error(&s, "LOAD DATA INPATH 'x' INTO TABLE missing_t").error_class(),
        ErrorClass::TableOrViewNotFound
    );
}

#[test]
fn insert_overwrite_directory_is_refused_until_the_write_path_lands() {
    let s = session();
    let e = error(
        &s,
        "INSERT OVERWRITE DIRECTORY '/tmp/o1' USING parquet SELECT a FROM nat",
    );
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("INSERT OVERWRITE DIRECTORY"), "{e}");
}

#[test]
fn unbound_parameters() {
    let s = session();
    let e = error(&s, "SELECT ?");
    assert_eq!(e.error_class(), ErrorClass::UnboundSqlParameter);
    assert!(
        e.to_string().contains("Found the unbound parameter: _7."),
        "{e}"
    );
    let e = error(&s, "SELECT :p");
    assert_eq!(e.error_class(), ErrorClass::UnboundSqlParameter);
    assert!(
        e.to_string().contains("Found the unbound parameter: p."),
        "{e}"
    );
    assert_eq!(e.sql_state(), Some("42P02"));
    let e = error(&s, "SELECT ? + 1, :x");
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidQueryMixedQueryParameters
    );
    assert_eq!(e.sql_state(), Some("42613"));
}

#[test]
fn probe_rows_for_flows_and_load_data_on_views_and_missing_tables() {
    let s = session();
    block_on(s.sql("CREATE TABLE t (a INT) USING parquet")).unwrap();
    block_on(s.sql("CREATE VIEW vw AS SELECT 1 AS a")).unwrap();
    block_on(s.sql("CREATE TEMPORARY VIEW tv AS SELECT 1 AS a")).unwrap();
    for (sql, operation) in [
        ("CREATE FLOW f AS INSERT INTO vw SELECT 1", "'INSERT'"),
        ("LOAD DATA INPATH 'x' INTO TABLE vw", "'LOAD DATA'"),
        ("LOAD DATA INPATH 'x' INTO TABLE tv", "'LOAD DATA'"),
    ] {
        let e = error(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::ExpectTableNotViewNoAlternative,
            "{sql}"
        );
        assert!(e.to_string().contains(operation), "{e}");
    }
    // Without BY NAME the flow is refused the same way.
    assert_eq!(
        error(&s, "CREATE FLOW f AS INSERT INTO t SELECT 1").error_class(),
        ErrorClass::UnsupportedFeatureCreateFlowQueryExecution
    );
    for sql in [
        "LOAD DATA INPATH 'x' INTO TABLE missing_t",
        "CREATE FLOW f AS INSERT INTO missing BY NAME SELECT 1",
    ] {
        let e = error(&s, sql);
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound, "{sql}");
        assert!(e.to_string().contains("cannot be found"), "{e}");
    }
}

#[test]
fn unbound_markers_inside_insert_and_ctas() {
    let s = session();
    for (sql, name) in [
        ("INSERT INTO nat SELECT ?", "_23"),
        ("CREATE TABLE c2 AS SELECT ? AS x", "_26"),
        ("INSERT INTO nat SELECT 'é' || ?", "_30"),
    ] {
        let e = error(&s, sql);
        assert_eq!(e.error_class(), ErrorClass::UnboundSqlParameter, "{sql}");
        assert!(
            e.to_string()
                .contains(&format!("unbound parameter: {name}.")),
            "{e}"
        );
    }
    assert_eq!(
        error(&s, "INSERT INTO nat SELECT ?, :p").error_class(),
        ErrorClass::InvalidQueryMixedQueryParameters
    );
}
