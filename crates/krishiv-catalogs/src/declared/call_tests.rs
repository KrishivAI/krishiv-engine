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
