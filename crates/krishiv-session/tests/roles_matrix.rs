// SPDX-License-Identifier: Apache-2.0
//! `roles_matrix` (P3-T002 acceptance; ADR 0046 §3): viewer, operator and admin sessions over
//! the statement classes that exist when T002 lands, through the one `authorize` path.
//!
//! A viewer sees metadata and runs plans that read no data; an operator also reads data; only
//! an admin writes, runs DDL or cluster-scope statements. A principal without a role is denied
//! even metadata.
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::{EngineRuntime, Principal, PrincipalKind, Session};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

/// One runtime (shared catalog and warehouse) with table `default.t` created by an admin.
fn runtime(dir: &std::path::Path) -> Arc<EngineRuntime> {
    let rt = EngineRuntime::builder()
        .config("spark.sql.warehouse.dir", dir.to_str().unwrap())
        .build()
        .unwrap();
    let admin = session(&rt, Some(Role::Admin));
    run(&admin, "CREATE TABLE t (a INT) USING parquet").unwrap();
    run(&admin, "INSERT INTO t VALUES (1), (2)").unwrap();
    rt
}

fn session(rt: &Arc<EngineRuntime>, role: Option<Role>) -> Session {
    let name = role.map_or("nobody", Role::as_str);
    Session::builder()
        .runtime(Arc::clone(rt))
        .principal(Principal::new(name, PrincipalKind::User))
        .access_policy(Arc::new(RolePolicy::new(
            role,
            Role::Admin,
            PolicyVersion(1),
        )))
        .build()
        .unwrap()
}

/// Runs `sql` to completion; the error text on failure.
fn run(s: &Session, sql: &str) -> Result<(), String> {
    block_on(async {
        let df = s.sql(sql).await?;
        df.collect().await.map(|_| ())
    })
    .map_err(|e| e.to_string())
}

/// The statements and the least role each needs.
const MATRIX: &[(&str, Role)] = &[
    // Plans that read no data securable.
    ("SELECT 1", Role::Viewer),
    ("SELECT * FROM VALUES (1), (2)", Role::Viewer),
    ("SELECT id FROM range(3)", Role::Viewer),
    // Session class.
    ("SET spark.sql.shuffle.partitions = 4", Role::Viewer),
    ("USE default", Role::Viewer),
    // Metadata (Browse).
    ("SHOW TABLES", Role::Viewer),
    ("SHOW COLUMNS IN t", Role::Viewer),
    ("DESCRIBE t", Role::Viewer),
    ("SHOW CREATE TABLE t", Role::Viewer),
    ("SHOW TBLPROPERTIES t", Role::Viewer),
    ("SHOW DATABASES", Role::Viewer),
    // Own rows of system tables.
    ("SELECT count(*) FROM system.queries", Role::Viewer),
    // Reads of data and statistics.
    ("SELECT * FROM t", Role::Operator),
    ("SELECT count(*) FROM t", Role::Operator),
    ("DESCRIBE EXTENDED t", Role::Operator),
    ("EXPLAIN ANALYZE SELECT * FROM t", Role::Operator),
    // Durable writes and DDL.
    ("INSERT INTO t VALUES (3)", Role::Admin),
    ("CREATE TABLE u (b INT) USING parquet", Role::Admin),
    ("CREATE DATABASE d2", Role::Admin),
    // Cluster scope.
    ("CLEAR CACHE", Role::Admin),
];

#[test]
fn roles_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path());
    let mut failures = Vec::new();
    for role in [Role::Viewer, Role::Operator, Role::Admin] {
        for (sql, need) in MATRIX {
            // A fresh session per statement, so a write by one row never changes another.
            let s = session(&rt, Some(role));
            let r = run(&s, sql);
            let allowed = role >= *need;
            match (&r, allowed) {
                (Ok(()), true) => {}
                (Err(e), false) if e.contains(DENIED) => {}
                _ => failures.push(format!("{role} `{sql}` (needs {need}): {r:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn no_role_is_denied_even_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path());
    let s = session(&rt, None);
    let e = run(&s, "SHOW TABLES").unwrap_err();
    assert!(e.contains(DENIED), "{e}");
    // A plan that names no securable needs no privilege (ADR 0046 §3, row 3).
    run(&s, "SELECT 1").unwrap();
}

#[test]
fn a_denial_reveals_no_reason_or_existence() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path());
    let s = session(&rt, Some(Role::Viewer));
    let existing = run(&s, "SELECT * FROM t").unwrap_err();
    assert!(
        existing.contains(DENIED) && !existing.contains("role"),
        "{existing}"
    );
    let missing = run(&s, "SELECT * FROM no_such_table").unwrap_err();
    assert!(missing.contains(DENIED), "{missing}");
}
