// SPDX-License-Identifier: Apache-2.0
//! `information_schema.catalogs`, `.schemata`, `.tables`, `.columns` and `.views` over the session catalog.
#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::executor::block_on;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{AccessDecision, PolicyVersion, Privilege, Securable};
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};

fn session(dir: &std::path::Path) -> Session {
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .unwrap();
    s
}

fn run(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

fn rows(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(async { s.sql(sql).await?.collect().await })
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for batch in df {
        for r in 0..batch.num_rows() {
            let cells: Vec<String> = (0..batch.num_columns())
                .map(|c| arrow_cast::display::array_value_to_string(batch.column(c), r).unwrap())
                .collect();
            out.push(cells.join("|"));
        }
    }
    out
}

#[test]
fn schemata_tables_and_columns_list_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(&s, "CREATE DATABASE db1");
    run(
        &s,
        "CREATE TABLE db1.t (a INT NOT NULL, b STRING) USING parquet",
    );
    run(&s, "CREATE VIEW db1.v AS SELECT 1 AS x");
    assert_eq!(
        rows(
            &s,
            "SELECT * FROM information_schema.schemata ORDER BY schema_name"
        ),
        ["spark_catalog|db1", "spark_catalog|default"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_schema, table_name, table_type FROM information_schema.tables ORDER BY 2"
        ),
        ["db1|t|BASE TABLE", "db1|v|VIEW"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_name, column_name, ordinal_position, data_type, is_nullable \
             FROM information_schema.columns WHERE table_name = 't' ORDER BY ordinal_position"
        ),
        ["t|a|1|int|YES", "t|b|2|string|YES"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_schema, table_name, view_definition FROM information_schema.views"
        ),
        ["db1|v|SELECT 1 AS x"]
    );
}

#[test]
fn catalogs_lists_the_session_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    assert_eq!(
        rows(&s, "SELECT * FROM information_schema.catalogs"),
        ["spark_catalog"]
    );
}

#[test]
fn spellings_types_and_nullability() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    run(
        &s,
        "CREATE TABLE t (a INT NOT NULL, c VARCHAR(10), d CHAR(3)) USING parquet",
    );
    assert_eq!(
        rows(
            &s,
            "SELECT column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS \
             WHERE table_name = 't' ORDER BY ordinal_position"
        ),
        ["a|int", "c|varchar(10)", "d|char(3)"]
    );
    assert_eq!(
        rows(
            &s,
            "SELECT table_name FROM spark_catalog.information_schema.tables"
        ),
        ["t"]
    );
    // Spark parquet tables are always nullable; a view over a literal keeps what the plan knows.
    run(&s, "CREATE VIEW v AS SELECT 1 AS x, CAST(NULL AS INT) AS y");
    assert_eq!(
        rows(
            &s,
            "SELECT column_name, is_nullable FROM information_schema.columns \
             WHERE table_name IN ('t', 'v') AND column_name IN ('a', 'x', 'y') ORDER BY 1"
        ),
        ["a|YES", "x|NO", "y|YES"]
    );
}

#[test]
fn the_namespace_is_reserved() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(dir.path());
    for sql in [
        "CREATE DATABASE information_schema",
        "CREATE DATABASE spark_catalog.INFORMATION_SCHEMA",
        "CREATE TABLE information_schema.columns (a INT) USING parquet",
        "CREATE VIEW information_schema.v AS SELECT 1",
    ] {
        let e = block_on(async { s.sql(sql).await?.collect().await }).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_RESERVED_NAME",
            "{sql}: {e}"
        );
    }
}

/// Browse everywhere, Select on the `information_schema` views and, once `strict`, on nothing else.
struct BrowseOnly {
    strict: AtomicBool,
}

impl AccessPolicy for BrowseOnly {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let allowed = !self.strict.load(Ordering::SeqCst)
            || privilege == Privilege::Browse
            || securable.to_string().contains("information_schema");
        Box::pin(std::future::ready(Ok(if allowed {
            AccessDecision::Allow
        } else {
            AccessDecision::Deny {
                reason: "no".to_owned(),
            }
        })))
    }
}

#[test]
fn a_view_text_needs_select() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    let policy = Arc::new(BrowseOnly {
        strict: AtomicBool::new(false),
    });
    let s = Session::builder()
        .runtime(rt)
        .principal(Principal::new("ana", PrincipalKind::User))
        .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
        .build()
        .unwrap();
    run(&s, "CREATE TABLE t (ssn STRING) USING parquet");
    run(
        &s,
        "CREATE VIEW v AS SELECT * FROM t WHERE ssn = '123-45-6789'",
    );
    assert_eq!(
        rows(&s, "SELECT view_definition FROM information_schema.views").len(),
        1
    );
    policy.strict.store(true, Ordering::SeqCst);
    assert_eq!(
        rows(
            &s,
            "SELECT table_name FROM information_schema.tables ORDER BY 1"
        ),
        ["t", "v"]
    );
    assert!(rows(&s, "SELECT view_definition FROM information_schema.views").is_empty());
}

#[test]
fn under_grants_a_principal_sees_what_it_may_list() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let session = |name: &str, admin: bool, system: Arc<dyn AccessPolicy>| {
        let policy = Arc::new(GrantsPolicy::new(Arc::clone(&grants), system, admin));
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(Principal::new(name, PrincipalKind::User))
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .unwrap()
    };
    let root = session("root", true, Arc::new(AllowAll));
    let ana = session(
        "ana",
        false,
        Arc::new(RolePolicy::new(
            Some(Role::Viewer),
            Role::Admin,
            PolicyVersion(1),
        )),
    );
    for statement in [
        "CREATE TABLE mine (a INT) USING parquet",
        "CREATE TABLE canary (a INT) USING parquet",
        "CREATE NAMESPACE vaultns",
        "CREATE TABLE vaultns.vt (b INT) USING parquet",
        "GRANT SELECT ON TABLE mine TO user:ana",
    ] {
        run(&root, statement);
    }
    assert_eq!(
        rows(
            &ana,
            "SELECT table_schema, table_name FROM information_schema.tables"
        ),
        ["default|mine"]
    );
    assert_eq!(
        rows(
            &ana,
            "SELECT table_name, column_name FROM information_schema.columns"
        ),
        ["mine|a"]
    );
    assert_eq!(
        rows(&ana, "SELECT schema_name FROM information_schema.schemata"),
        ["default"]
    );
    assert!(rows(&ana, "SELECT * FROM information_schema.views").is_empty());
    assert_eq!(
        rows(&root, "SELECT count(*) FROM information_schema.tables"),
        ["3"]
    );
}
