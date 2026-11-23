// SPDX-License-Identifier: Apache-2.0
//! `CALL cat.system.p(..)` against a declared Iceberg REST catalog (P3-T027, PR-36e): a procedure
//! the catalog does not have, a catalog nothing declares, a malformed name and a table the
//! catalog does not know, each refused with its error class.

use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

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
