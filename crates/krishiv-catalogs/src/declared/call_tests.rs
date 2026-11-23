// SPDX-License-Identifier: Apache-2.0
//! `CALL cat.system.p(..)` against a declared Iceberg REST catalog (P3-T027, PR-36e): a procedure
//! the catalog does not have, a catalog nothing declares, a malformed name and a table the
//! catalog does not know, each refused with its error class.

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::catalog_decl::DeclMode;
use krishiv_session::catalog_registry::CatalogSetup;
use krishiv_session::{
    AccessDecision, AccessPolicy, EngineRuntime, Principal, Privilege, Securable, Session,
};

use super::IcebergRestFactory;
use super::read_tests::session;
use super::tests::path;
use crate::iceberg_rest::test_server::{Reply, Seen, Server, serve};

/// A REST catalog that knows no table.
fn empty() -> Server {
    serve(|r: &Seen| match path(r) {
        "/v1/config" => Reply::Json(200, r#"{"defaults": {}, "overrides": {}}"#.into()),
        _ => Reply::Json(
            404,
            r#"{"error": {"message": "no", "type": "NoSuchTableException", "code": 404}}"#.into(),
        ),
    })
}

fn call(s: &Session, sql: &str) -> KrishivError {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_err()
}

#[test]
fn a_procedure_the_catalog_lacks_is_failed_to_load_routine() {
    let server = empty();
    let s = session(&server, None);
    let e = call(&s, "CALL lake.system.no_such_procedure('db.t')");
    assert_eq!(e.error_class(), ErrorClass::FailedToLoadRoutine);
    // A namespace other than `system` holds none.
    let e = call(&s, "CALL lake.other.expire_snapshots('db.t')");
    assert_eq!(e.error_class(), ErrorClass::FailedToLoadRoutine);
    // So does a name of one part.
    let e = call(&s, "CALL expire_snapshots('db.t')");
    assert_eq!(e.error_class(), ErrorClass::FailedToLoadRoutine);
}

#[test]
fn an_undeclared_catalog_has_no_procedures() {
    let server = empty();
    let s = session(&server, None);
    let e = call(&s, "CALL nowhere.system.expire_snapshots('db.t')");
    assert_eq!(e.error_class(), ErrorClass::MissingCatalogAbilityProcedures);
}

#[test]
fn a_table_the_catalog_does_not_know_is_krishiv_iceberg_procedure_table_not_found() {
    let server = empty();
    let s = session(&server, None);
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(table => 'db.missing')",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureTableNotFound
    );
    // A bare name has no namespace to look in.
    let e = call(&s, "CALL lake.system.expire_snapshots(table => 'missing')");
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureTableNotFound
    );
}

/// Allows everything but `MODIFY` on a table.
struct NoTableWrites;

impl AccessPolicy for NoTableWrites {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = if matches!(securable, Securable::Table(_)) && privilege == Privilege::Modify
        {
            AccessDecision::Deny {
                reason: "no table writes".to_owned(),
            }
        } else {
            AccessDecision::Allow
        };
        Box::pin(async move { Ok(decision) })
    }
}

/// `MANAGE` on the cluster is not enough to expire a table's snapshots: the write privilege on
/// the table is asked first, before the catalog is asked for the table (so the refusal comes
/// before the table's absence would).
#[test]
fn a_procedure_on_a_table_needs_the_write_privilege_on_it() {
    let server = empty();
    let setup = CatalogSetup::new(DeclMode::Embedded)
        .factory(Arc::new(IcebergRestFactory))
        .setting("krishiv.catalog.lake.type", "iceberg-rest")
        .setting("krishiv.catalog.lake.uri", server.base.as_str())
        .setting("krishiv.catalog.lake.storage", "allowed-prefixes");
    let s = Session::builder()
        .runtime(EngineRuntime::builder().catalogs(setup).build().unwrap())
        .access_policy(Arc::new(NoTableWrites))
        .build()
        .unwrap();
    for sql in [
        "CALL lake.system.expire_snapshots(table => 'db.missing')",
        "CALL lake.system.expire_snapshots(table => 'lake.db.missing')",
    ] {
        let e = call(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivInsufficientPermissions,
            "{sql}: {e}"
        );
    }
}

/// Arguments are cast in ANSI mode whatever the session says: a bad timestamp fails before the
/// table is looked up, instead of becoming `NULL` (an absent `older_than`).
#[test]
fn a_bad_argument_fails_even_with_ansi_off() {
    let server = empty();
    let s = session(&server, None);
    s.conf().set("spark.sql.ansi.enabled", "false").unwrap();
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(table => 'db.missing', older_than => 'not-a-timestamp')",
    );
    assert_eq!(e.error_class(), ErrorClass::CastInvalidInput, "{e}");
    // The session's own setting is untouched.
    assert_eq!(
        s.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
}

/// A `NULL` element of an array argument is refused, never dropped.
#[test]
fn a_null_array_element_is_refused() {
    let server = empty();
    let s = session(&server, None);
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(table => 'db.missing', \
         snapshot_ids => array(7L, CAST(NULL AS BIGINT), 9L))",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureArgument,
        "{e}"
    );
    assert!(e.to_string().contains("NULL elements"), "{e}");
}

/// A session over `server` with `lake` writable (`option.write`) when `write` holds.
fn lake_session(server: &Server, write: bool) -> Session {
    let mut pairs = vec![
        ("krishiv.catalog.lake.type", "iceberg-rest"),
        ("krishiv.catalog.lake.uri", server.base.as_str()),
        ("krishiv.catalog.lake.storage", "allowed-prefixes"),
    ];
    if write {
        pairs.push(("krishiv.catalog.lake.option.write", "true"));
    }
    super::read_tests::session_with(DeclMode::Embedded, &pairs, &[])
}

/// The one result row of `sql`, by column name.
fn result_row(s: &Session, sql: &str) -> Vec<(String, i64)> {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    let batches = block_on(async { s.sql(sql).await?.collect().await }).unwrap();
    let b = batches.iter().find(|b| b.num_rows() > 0).unwrap();
    assert_eq!(b.num_rows(), 1);
    b.schema()
        .fields()
        .iter()
        .zip(b.columns())
        .map(|(f, c)| {
            let c = c.as_primitive::<Int64Type>();
            (f.name().clone(), if c.is_null(0) { -1 } else { c.value(0) })
        })
        .collect()
}

/// `expire_snapshots` end to end: the arguments bound and cast, the table opened through the
/// declared REST catalog, the expire committed through it, the result row Iceberg's, and the
/// table still read back whole.
#[test]
fn expire_snapshots_runs_through_the_rest_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let table = krishiv_connectors_iceberg::testing::write_table(dir.path()).unwrap();
    let (server, lake) = super::write_tests::catalog(&table, &[]);
    let s = lake_session(&server, true);
    let row = result_row(
        &s,
        "CALL lake.system.expire_snapshots(table => 'lake.db.t', \
         older_than => TIMESTAMP '9999-01-01 00:00:00', retain_last => 1, \
         clean_expired_metadata => true)",
    );
    let names: Vec<&str> = row.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "deleted_data_files_count",
            "deleted_position_delete_files_count",
            "deleted_equality_delete_files_count",
            "deleted_manifest_files_count",
            "deleted_manifest_lists_count",
            "deleted_statistics_files_count",
        ]
    );
    // The first snapshot's manifest list; its data file is still the second's.
    assert_eq!(row[4].1, 1, "{row:?}");
    assert_eq!(row[0].1, 0, "{row:?}");
    let snapshots = lake.lock().unwrap().metadata["snapshots"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(snapshots, 1);
    assert_eq!(
        super::read_tests::rows(&s, "SELECT id, name FROM lake.db.t"),
        super::read_tests::pairs(&[(1, "a"), (2, "b"), (3, "c")])
    );
    // Again, positionally: nothing is left to remove and no commit is sent.
    let posts = || server.seen().iter().filter(|r| r.method == "POST").count();
    let before = posts();
    let row = result_row(
        &s,
        "CALL lake.system.expire_snapshots('db.t', TIMESTAMP '9999-01-01 00:00:00', 1)",
    );
    assert!(row.iter().all(|(_, v)| *v == 0), "{row:?}");
    assert_eq!(posts(), before);
}

/// The procedure's own refusals: a read-only catalog, an empty table name, a bad
/// `max_concurrent_deletes`, and a procedure not in this build yet.
#[test]
fn expire_snapshots_refusals() {
    let dir = tempfile::tempdir().unwrap();
    let table = krishiv_connectors_iceberg::testing::write_table(dir.path()).unwrap();
    let (server, _) = super::write_tests::catalog(&table, &[]);
    let e = call(
        &lake_session(&server, false),
        "CALL lake.system.expire_snapshots(table => 'db.t')",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
    assert!(e.to_string().contains("read-only"), "{e}");
    let s = lake_session(&server, true);
    let e = call(&s, "CALL lake.system.expire_snapshots(table => '')");
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureArgument,
        "{e}"
    );
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(table => 'db.t', max_concurrent_deletes => 0)",
    );
    assert!(e.to_string().contains("max_concurrent_deletes"), "{e}");
    let e = call(&s, "CALL lake.system.rewrite_manifests(table => 'db.t')");
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivUnsupportedFeature,
        "{e}"
    );
}

/// Binding as a Spark 4.1.0 + Iceberg 1.12.0 run binds (review of #983): an unknown name
/// proposes the three closest parameters, the routine is named by the procedure alone, names
/// follow `spark.sql.caseSensitive`, a column reference does not resolve, a subquery is
/// refused and a NULL map value is refused.
#[test]
fn binding_follows_spark() {
    let server = empty();
    let s = session(&server, None);
    for (sql, proposal) in [
        (
            "CALL lake.system.expire_snapshots(table => 'db.t', retian_last => 1)",
            "[`retain_last` `table` `older_than`]",
        ),
        (
            "CALL lake.system.expire_snapshots(table => 'db.t', older => 1)",
            "[`table` `older_than` `retain_last`]",
        ),
    ] {
        let e = call(&s, sql);
        assert_eq!(e.error_class(), ErrorClass::UnrecognizedParameterName);
        let text = e.to_string();
        assert!(text.contains(proposal), "{sql}: {text}");
        assert!(
            text.contains("routine `expire_snapshots` because"),
            "{text}"
        );
    }
    // Case-insensitive by default: `TABLE` binds `table` and the call goes on to the table.
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(TABLE => 'db.missing', RETAIN_LAST => 1)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureTableNotFound
    );
    s.conf().set("spark.sql.caseSensitive", "true").unwrap();
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(table => 'db.t', RETAIN_LAST => 1)",
    );
    assert_eq!(e.error_class(), ErrorClass::UnrecognizedParameterName);
    assert!(
        e.to_string()
            .contains("[`older_than` `table` `retain_last`]"),
        "{e}"
    );
    s.conf().set("spark.sql.caseSensitive", "false").unwrap();
    let e = call(
        &s,
        "CALL lake.system.expire_snapshots(table => 'db.t', retain_last => id)",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::UnresolvedColumnWithoutSuggestion,
        "{e}"
    );
    for sql in [
        "CALL lake.system.expire_snapshots(table => (SELECT 'db.t'))",
        "CALL lake.system.expire_snapshots(table => 'db.t', retain_last => 1 + (SELECT 1))",
    ] {
        let e = call(&s, sql);
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
        assert!(e.to_string().contains("subquery in a CALL argument"), "{e}");
    }
    let e = call(
        &s,
        "CALL lake.system.rewrite_data_files(table => 'db.t', \
         options => map('min-input-files', CAST(NULL AS STRING)))",
    );
    assert_eq!(
        e.error_class(),
        ErrorClass::KrishivIcebergProcedureArgument,
        "{e}"
    );
}

/// Ties of edit distance keep Spark's candidate order (`parameterNamesSet.toSeq`): from a
/// Spark 4.1.0 + Iceberg 1.12.0 run, a name of 30 `z`s (equally far from every parameter)
/// proposes each procedure's first three, and `x` on `rewrite_manifests` ties `spec_id` with
/// `sort_by` in signature order (review of #983).
#[test]
fn proposal_ties_keep_spark_candidate_order() {
    let server = empty();
    let s = session(&server, None);
    let z = "z".repeat(30);
    for (procedure, name, want) in [
        (
            "expire_snapshots",
            z.as_str(),
            "[`stream_results` `clean_expired_metadata` `older_than`]",
        ),
        (
            "rewrite_manifests",
            z.as_str(),
            "[`table` `use_caching` `spec_id`]",
        ),
        (
            "rewrite_data_files",
            z.as_str(),
            "[`branch` `options` `strategy`]",
        ),
        (
            "remove_orphan_files",
            z.as_str(),
            "[`prefix_listing` `location` `stream_results`]",
        ),
        ("rewrite_manifests", "x", "[`table` `spec_id` `sort_by`]"),
    ] {
        let e = call(
            &s,
            &format!("CALL lake.system.{procedure}(table => 'db.t', {name} => 1)"),
        );
        assert_eq!(e.error_class(), ErrorClass::UnrecognizedParameterName);
        assert!(e.to_string().contains(want), "{procedure} {name}: {e}");
    }
}

/// The argument shapes Spark's binding cannot take (an internal error in the Spark 4.1.0 +
/// Iceberg 1.12.0 run, 2026-11-23, every case below) are refused; the ones it binds go on to the
/// table.
#[test]
fn arguments_spark_cannot_bind_are_refused() {
    let server = empty();
    let s = session(&server, None);
    let arith = "date, timestamp or interval";
    for (arg, what) in [
        ("retain_last => CAST(rand() * 10 AS INT) + 1", "random-seed"),
        (
            "retain_last => CAST(uuid() IS NOT NULL AS INT)",
            "random-seed",
        ),
        (
            "older_than => TIMESTAMP '2020-01-01 00:00:00' + INTERVAL 1 DAY",
            arith,
        ),
        (
            "older_than => current_timestamp() - (INTERVAL 5 DAYS)",
            arith,
        ),
        ("older_than => current_date() - 7", arith),
        ("older_than => current_date() + 1", arith),
        ("older_than => 1 + current_date()", arith),
        (
            "older_than => current_timestamp() - make_interval(0, 0, 0, 5)",
            arith,
        ),
        (
            "older_than => current_timestamp() - CAST('1' AS INTERVAL DAY)",
            arith,
        ),
        (
            "older_than => TIMESTAMP '2020-01-01 00:00:00' + (INTERVAL 1 DAY + INTERVAL 2 DAYS)",
            arith,
        ),
        (
            "retain_last => CAST(current_date() - current_date() AS INT) + 1",
            arith,
        ),
        ("retain_last => 7 / 7", "two doubles or two decimals"),
        (
            "older_than => CAST('2020-01-01' AS TIMESTAMP)",
            "CAST of a string AS TIMESTAMP",
        ),
        (
            "older_than => CAST(current_date() AS TIMESTAMP)",
            "CAST of a date AS TIMESTAMP",
        ),
        (
            "older_than => CAST(DATE '2020-01-01' AS TIMESTAMP)",
            "CAST of a date AS TIMESTAMP",
        ),
        (
            "older_than => CAST(current_timestamp() AS DATE)",
            "CAST of a timestamp AS DATE",
        ),
        (
            "older_than => CAST(TIMESTAMP '2020-01-01 00:00:00' AS TIMESTAMP_NTZ)",
            "CAST of a timestamp AS TIMESTAMP_NTZ",
        ),
        (
            "older_than => CAST(TIMESTAMP_NTZ '2020-01-01 00:00:00' AS TIMESTAMP)",
            "CAST of a timestamp_ntz AS TIMESTAMP",
        ),
        // Divergence toward refusal: Spark's run binds this one (see `refuse_unbindable`).
        (
            "older_than => CAST(date_sub(current_date(), 7) AS TIMESTAMP)",
            "CAST of a date AS TIMESTAMP",
        ),
    ] {
        let e = call(
            &s,
            &format!("CALL lake.system.expire_snapshots(table => 'db.missing', {arg})"),
        );
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{arg}: {e}"
        );
        assert!(e.to_string().contains(what), "{arg}: {e}");
    }
    for arg in [
        "retain_last => 1 + 1",
        "retain_last => length('abc')",
        "older_than => current_timestamp()",
        "older_than => date_sub(current_date(), 5)",
        "retain_last => 2 * 3",
        "retain_last => extract(DAY FROM INTERVAL 1 DAY + INTERVAL 2 DAYS)",
        "retain_last => extract(DAY FROM INTERVAL 1 DAY * 2)",
        "retain_last => datediff(current_date(), current_date()) + 1",
        "older_than => CAST('2020-01-01' AS DATE)",
        "older_than => CAST('2020-01-01 00:00:00' AS TIMESTAMP_NTZ)",
        "older_than => '2020-01-01 00:00:00'",
        "older_than => DATE '2020-01-01'",
        "retain_last => CAST('1' AS INT)",
        "retain_last => CAST(1.5 AS INT)",
    ] {
        let e = call(
            &s,
            &format!("CALL lake.system.expire_snapshots(table => 'db.missing', {arg})"),
        );
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivIcebergProcedureTableNotFound,
            "{arg}: {e}"
        );
    }
    // Timestamp to string, on a string parameter (the run used `table`).
    let e = call(
        &s,
        "CALL lake.system.rewrite_data_files(table => 'db.missing', \
         strategy => CAST(current_timestamp() AS STRING))",
    );
    assert!(
        e.to_string().contains("CAST of a timestamp AS STRING"),
        "{e}"
    );
}
