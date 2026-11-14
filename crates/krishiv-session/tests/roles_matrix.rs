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
    run(&admin, "CREATE VIEW v AS SELECT a, 7 AS literal FROM t").unwrap();
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
    // Own rows of system tables, and tables that hold no principal's data.
    ("SELECT count(*) FROM system.queries", Role::Viewer),
    ("SELECT count(*) FROM SYSTEM.QUERIES", Role::Viewer),
    ("SELECT count(*) FROM system.capabilities", Role::Viewer),
    ("SELECT count(*) FROM system.sessions", Role::Viewer),
    // A system table's schema is metadata, even where its rows are not (Browse, not Select).
    ("DESCRIBE system.memory", Role::Viewer),
    ("SHOW COLUMNS IN system.memory", Role::Viewer),
    ("DESCRIBE system.operators", Role::Viewer),
    ("SHOW COLUMNS IN system.caches", Role::Viewer),
    // Tables that name other principals' work.
    ("SELECT count(*) FROM system.memory", Role::Operator),
    ("SELECT count(*) FROM system.operators", Role::Operator),
    // A view: its schema is metadata; its text, statistics and rows are data.
    ("DESCRIBE v", Role::Viewer),
    ("SHOW COLUMNS IN v", Role::Viewer),
    ("SHOW CREATE TABLE v", Role::Operator),
    ("SELECT * FROM v", Role::Operator),
    ("SHOW TABLE EXTENDED LIKE 'v'", Role::Operator),
    // Reads of data and statistics.
    ("SELECT * FROM t", Role::Operator),
    ("SELECT count(*) FROM t", Role::Operator),
    ("DESCRIBE EXTENDED t", Role::Operator),
    ("EXPLAIN ANALYZE SELECT * FROM t", Role::Operator),
    // Non-durable side effects that read the table.
    ("CACHE TABLE t", Role::Operator),
    ("UNCACHE TABLE t", Role::Operator),
    // Durable writes and DDL.
    ("INSERT INTO t VALUES (3)", Role::Admin),
    ("CREATE TABLE u (b INT) USING parquet", Role::Admin),
    ("CREATE DATABASE d2", Role::Admin),
    // Cluster scope (REFRESH is stricter than ADR 0046 §3's operator row; recorded in the ADR).
    ("CLEAR CACHE", Role::Admin),
    ("REFRESH TABLE t", Role::Admin),
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

#[test]
fn path_reads_need_operator() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path());
    let path = dir.path().join("t");
    let sql = format!("SELECT * FROM parquet.`{}`", path.display());
    let viewer = run(&session(&rt, Some(Role::Viewer)), &sql).unwrap_err();
    assert!(viewer.contains(DENIED), "{viewer}");
    run(&session(&rt, Some(Role::Operator)), &sql).unwrap();
}

#[test]
fn read_only_keeps_a_viewers_metadata_access() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path());
    let s = Session::builder()
        .runtime(Arc::clone(&rt))
        .principal(Principal::new("viewer", PrincipalKind::User))
        .access_policy(Arc::new(krishiv_session::ReadOnlyPolicy::new(Arc::new(
            RolePolicy::new(Some(Role::Viewer), Role::Admin, PolicyVersion(1)),
        ))))
        .build()
        .unwrap();
    for sql in ["SHOW TABLES", "DESCRIBE t", "SHOW COLUMNS IN v"] {
        run(&s, sql).unwrap_or_else(|e| panic!("`{sql}`: {e}"));
    }
    for sql in ["SELECT * FROM t", "INSERT INTO t VALUES (3)"] {
        let e = run(&s, sql).unwrap_err();
        assert!(e.contains(DENIED), "`{sql}`: {e}");
    }
}
