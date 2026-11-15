// SPDX-License-Identifier: Apache-2.0
//! P3-T062 unit G2: the `grants` provider end to end. Sessions of several principals share one
//! runtime (one catalog) and one [`Grants`]; an admin creates the data, and every other principal
//! reads only what its grants (direct, through a group, through roles, on a parent, per column,
//! as owner) allow. A revoke takes effect for plans analysed and relations cached before it.
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::grants::{
    DecidedBy, FileGrantStore, GrantStore, Grants, GrantsPolicy, MemoryGrantStore,
};
use krishiv_session::{
    AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Privilege, Session,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

struct World {
    rt: Arc<EngineRuntime>,
    grants: Arc<Grants>,
    dir: tempfile::TempDir,
}

impl World {
    fn new() -> Self {
        Self::over(Arc::new(MemoryGrantStore::new()))
    }

    fn over(store: Arc<dyn GrantStore>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let rt = EngineRuntime::builder()
            .config(
                "spark.sql.warehouse.dir",
                dir.path().join("wh").to_str().unwrap(),
            )
            .build()
            .unwrap();
        Self {
            rt,
            grants: Grants::new(store),
            dir,
        }
    }

    /// A session of `name` in `groups`; `system` decides `System(..)` securables.
    fn with(
        &self,
        name: &str,
        groups: &[&str],
        admin: bool,
        system: Arc<dyn AccessPolicy>,
    ) -> Session {
        let mut p = Principal::new(name, PrincipalKind::User);
        p.groups = groups.iter().map(|g| (*g).to_owned()).collect();
        let policy = Arc::new(GrantsPolicy::new(Arc::clone(&self.grants), system, admin));
        Session::builder()
            .runtime(Arc::clone(&self.rt))
            .principal(p)
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .unwrap()
    }

    fn user(&self, name: &str) -> Session {
        self.with(name, &[], false, Arc::new(AllowAll))
    }

    fn admin(&self) -> Session {
        self.with("root", &[], true, Arc::new(AllowAll))
    }
}

fn run(s: &Session, sql: &str) -> Result<usize, KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
    })
}

fn ok(s: &Session, sql: &str) -> usize {
    run(s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn denied(s: &Session, sql: &str) -> String {
    match run(s, sql) {
        Err(e) if e.error_class().name() == DENIED => e.to_string(),
        other => panic!("{sql}: expected a denial, got {other:?}"),
    }
}

/// An admin's table `t (a INT, b INT)` with one row.
fn table(w: &World, name: &str) {
    let root = w.admin();
    ok(
        &root,
        &format!("CREATE TABLE {name} (a INT, b INT) USING parquet"),
    );
    ok(&root, &format!("INSERT INTO {name} VALUES (1, 2)"));
}

#[test]
fn a_table_grant_decides_reads_and_a_revoke_takes_it_back() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    denied(&ana, "SELECT * FROM t");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    assert_eq!(ok(&ana, "SELECT * FROM t"), 1);
    // A grant gives exactly what it names.
    denied(&ana, "INSERT INTO t VALUES (3, 4)");
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    denied(&ana, "SELECT * FROM t");
}

#[test]
fn a_namespace_grant_covers_present_and_future_tables() {
    let w = World::new();
    table(&w, "t1");
    let (root, ana) = (
        w.admin(),
        w.with("ana", &["eng"], false, Arc::new(AllowAll)),
    );
    ok(&root, "GRANT SELECT ON SCHEMA default TO group:eng");
    assert_eq!(ok(&ana, "SELECT * FROM t1"), 1);
    table(&w, "t2");
    assert_eq!(ok(&ana, "SELECT a FROM T2"), 1);
    // A catalog grant covers its namespaces.
    let bob = w.user("bob");
    ok(&root, "GRANT SELECT ON CATALOG spark_catalog TO user:bob");
    assert_eq!(ok(&bob, "SELECT * FROM default.t2"), 1);
}

#[test]
fn column_grants_allow_only_their_columns() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    assert_eq!(ok(&ana, "SELECT a FROM t"), 1);
    assert_eq!(ok(&ana, "SELECT a FROM t WHERE a > 0"), 1);
    let e = denied(&ana, "SELECT b FROM t");
    assert!(e.contains("`b`"), "{e}");
    denied(&ana, "SELECT * FROM t");
    denied(&ana, "SELECT a FROM t WHERE b = 2");
    // No column referenced: the rows themselves need the table.
    denied(&ana, "SELECT count(*) FROM t");
    ok(&root, "GRANT SELECT ON t TO user:ana");
    assert_eq!(ok(&ana, "SELECT count(*), max(b) FROM t"), 1);
}

#[test]
fn roles_are_transitive_and_cycles_are_refused() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (
        w.admin(),
        w.with("ana", &["eng"], false, Arc::new(AllowAll)),
    );
    for sql in [
        "CREATE ROLE readers",
        "CREATE ROLE staff",
        "GRANT SELECT ON t TO role:readers",
        "GRANT ROLE readers TO role:staff",
        "GRANT ROLE staff TO group:eng",
    ] {
        ok(&root, sql);
    }
    assert_eq!(ok(&ana, "SELECT * FROM t"), 1);
    let e = run(&root, "GRANT ROLE staff TO role:readers").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_ROLE_CYCLE");
    assert_eq!(e.sql_state(), Some("42P17"));
    let e = run(&root, "CREATE ROLE staff").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_ROLE_ALREADY_EXISTS");
    let e = run(&root, "GRANT ROLE nope TO user:ana").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_ROLE_NOT_FOUND");
    // A non-admin who did not create the role cannot change it, and cannot tell a missing role.
    denied(&ana, "GRANT ROLE staff TO user:ana");
    denied(&ana, "DROP ROLE nope");
    ok(&root, "DROP ROLE staff");
    denied(&ana, "SELECT * FROM t");
}

#[test]
fn granting_needs_the_grant_option_and_never_exceeds_it() {
    let w = World::new();
    table(&w, "t");
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "GRANT SELECT ON t TO user:ana WITH GRANT OPTION");
    ok(&ana, "GRANT SELECT ON t TO user:bob");
    assert_eq!(ok(&bob, "SELECT * FROM t"), 1);
    denied(&ana, "GRANT MODIFY ON t TO user:bob");
    denied(&ana, "GRANT ALL PRIVILEGES ON t TO user:bob");
    denied(&ana, "GRANT SELECT ON SCHEMA default TO user:bob");
    denied(&bob, "GRANT SELECT ON t TO user:carl");
    denied(&bob, "REVOKE SELECT ON t FROM user:ana");
    // `GRANT OPTION FOR` keeps the privilege and takes the right to pass it on.
    ok(&root, "REVOKE GRANT OPTION FOR SELECT ON t FROM user:ana");
    assert_eq!(ok(&ana, "SELECT * FROM t"), 1);
    denied(&ana, "GRANT SELECT ON t TO user:carl");
    denied(&carl, "SELECT * FROM t");
    // The role matrix's securables are never granted.
    denied(&root, "GRANT SELECT ON TABLE system.queries TO user:ana");
}

#[test]
fn the_creator_owns_and_a_drop_takes_the_grants_with_it() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    denied(&ana, "CREATE TABLE mine (a INT) USING parquet");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    // A managed table's directory is a path securable too (open question in the T062 report).
    let wh = w.dir.path().join("wh");
    for who in ["ana", "bob"] {
        ok(
            &root,
            &format!(
                "GRANT ALL PRIVILEGES ON PATH 'file:{}' TO user:{who}",
                wh.display()
            ),
        );
    }
    ok(&ana, "CREATE TABLE mine (a INT) USING parquet");
    ok(&ana, "INSERT INTO mine VALUES (1)");
    assert_eq!(ok(&ana, "SELECT * FROM mine"), 1);
    ok(&ana, "GRANT SELECT ON mine TO user:bob");
    assert_eq!(ok(&bob, "SELECT * FROM mine"), 1);
    let shown =
        block_on(async { ana.sql("SHOW GRANTS ON TABLE mine").await?.collect().await }).unwrap();
    let text = format!("{shown:?}");
    assert!(
        text.contains("OWNERSHIP") && text.contains("user:ana"),
        "{text}"
    );
    // `IF NOT EXISTS` over an existing table takes nothing.
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(
        &bob,
        "CREATE TABLE IF NOT EXISTS mine (a INT) USING parquet",
    );
    denied(&bob, "DROP TABLE mine");
    ok(&ana, "DROP TABLE mine");
    // A table of the same name later starts without the old grants.
    ok(&root, "CREATE TABLE mine (a INT) USING parquet");
    denied(&bob, "SELECT * FROM mine");
    denied(&ana, "SELECT * FROM mine");
}

#[test]
fn show_grants_lists_own_grants_and_needs_monitor_for_others() {
    let w = World::new();
    table(&w, "t");
    table(&w, "canary");
    let root = w.admin();
    ok(&root, "GRANT SELECT ON t TO user:ana");
    ok(&root, "GRANT SELECT ON canary TO user:bob");
    let viewer: Arc<dyn AccessPolicy> = Arc::new(RolePolicy::new(
        Some(Role::Viewer),
        Role::Admin,
        PolicyVersion(1),
    ));
    let ana = w.with("ana", &[], false, Arc::clone(&viewer));
    let mine = block_on(async { ana.sql("SHOW GRANTS").await?.collect().await }).unwrap();
    let text = format!("{mine:?}");
    assert!(text.contains("spark_catalog.default.t"), "{text}");
    assert!(!text.contains("canary"), "{text}");
    // An object's grants: only the caller's without `Monitor`.
    let on = block_on(async {
        ana.sql("SHOW GRANTS ON TABLE canary")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        on.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        0
    );
    let e = denied(&ana, "SHOW GRANTS TO user:bob");
    assert!(!e.contains("canary"), "{e}");
    // An operator holds `Monitor`.
    let operator: Arc<dyn AccessPolicy> = Arc::new(RolePolicy::new(
        Some(Role::Operator),
        Role::Admin,
        PolicyVersion(1),
    ));
    let op = w.with("olga", &[], false, operator);
    assert_eq!(ok(&op, "SHOW GRANTS TO user:bob"), 1);
    // The owner's implicit grant and bob's.
    assert_eq!(ok(&op, "SHOW GRANTS ON TABLE canary"), 2);
    // A viewer may not grant at all (the statement class needs `Manage` from the role matrix).
    denied(&ana, "GRANT SELECT ON t TO user:carl");
}

#[test]
fn a_revoke_reaches_analysed_plans_and_cached_relations() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT ON t TO user:ana");
    let before = w.grants.epoch();
    let df = block_on(ana.sql("SELECT a FROM t")).unwrap();
    assert_eq!(block_on(df.collect()).unwrap()[0].num_rows(), 1);
    ok(&ana, "CACHE TABLE t");
    assert_eq!(ok(&ana, "SELECT * FROM t"), 1);
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    assert!(w.grants.epoch() > before);
    // The frame analysed before the revoke is analysed again, and denied.
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(e.error_class().name(), DENIED);
    // The cached relation is checked on use, not when it was cached.
    denied(&ana, "SELECT * FROM t");
}

#[test]
fn path_grants_match_by_whole_segments() {
    let w = World::new();
    let data = w.dir.path().join("data");
    std::fs::create_dir_all(data.join("sales")).unwrap();
    std::fs::create_dir_all(data.join("salesx")).unwrap();
    std::fs::write(data.join("sales/a.csv"), "1\n").unwrap();
    std::fs::write(data.join("salesx/a.csv"), "1\n").unwrap();
    let (root, ana) = (w.admin(), w.user("ana"));
    let sales = data.join("sales");
    let read = |dir: &std::path::Path| format!("SELECT * FROM csv.`{}`", dir.display());
    denied(&ana, &read(&sales));
    ok(
        &root,
        &format!(
            "GRANT SELECT ON PATH 'file:{}' TO user:ana",
            data.join("sales/../sales").display()
        ),
    );
    assert_eq!(ok(&ana, &read(&sales)), 1);
    denied(&ana, &read(&data.join("salesx")));
}

#[test]
fn grants_persist_in_the_file_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grants.json");
    {
        let w = World::over(Arc::new(FileGrantStore::open(&path).unwrap()));
        ok(&w.admin(), "CREATE ROLE r");
        ok(&w.admin(), "GRANT SELECT ON SCHEMA default TO role:r");
        assert_eq!(w.grants.epoch(), 2);
    }
    let reopened = FileGrantStore::open(&path).unwrap();
    let grants = Grants::new(Arc::new(reopened));
    assert_eq!(grants.epoch(), 2);
}

#[test]
fn denials_are_counted_by_privilege_and_provider() {
    let w = World::new();
    table(&w, "t");
    let viewer: Arc<dyn AccessPolicy> = Arc::new(RolePolicy::new(
        Some(Role::Viewer),
        Role::Admin,
        PolicyVersion(1),
    ));
    let ana = w.with("ana", &[], false, viewer);
    denied(&ana, "SELECT * FROM t");
    denied(&ana, "SELECT * FROM system.memory");
    let count = |privilege, provider| {
        w.grants
            .denied_counts()
            .into_iter()
            .find(|c| c.privilege == privilege && c.provider == provider)
            .map_or(0, |c| c.count)
    };
    // A read refused at Bind (asked as Browse) is counted as the SELECT it needs (review m6).
    assert_eq!(count(Privilege::Select, DecidedBy::Grants), 1);
    assert_eq!(count(Privilege::Browse, DecidedBy::Grants), 0);
    assert_eq!(count(Privilege::Select, DecidedBy::Roles), 1);
}

#[test]
fn without_grants_the_statements_are_refused() {
    let s = Session::new().unwrap();
    for sql in [
        "GRANT SELECT ON t TO user:a",
        "SHOW GRANTS",
        "CREATE ROLE r",
    ] {
        let e = run(&s, sql).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{sql}"
        );
    }
    // Grants managed but not enforced is a misconfiguration, refused at build.
    let grants = Grants::new(Arc::new(MemoryGrantStore::new()));
    let policy = Arc::new(GrantsPolicy::new(grants, Arc::new(AllowAll), false));
    assert!(Session::builder().grants(policy).build().is_err());
}

#[test]
fn a_table_revoke_takes_the_column_grants_with_it() {
    let w = World::new();
    table(&w, "t");
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    ok(&root, "GRANT SELECT (b) ON t TO user:ana");
    ok(&root, "GRANT SELECT (a) ON t TO user:bob");
    // A column revoke takes only that column.
    ok(&root, "REVOKE SELECT (a) ON t FROM user:ana");
    denied(&ana, "SELECT a FROM t");
    assert_eq!(ok(&ana, "SELECT b FROM t"), 1);
    // A table revoke takes the grantee's column grants on the table (Unity's behaviour), and
    // only the grantee's.
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    denied(&ana, "SELECT b FROM t");
    assert_eq!(ok(&bob, "SELECT a FROM t"), 1);
    ok(&root, "GRANT SELECT (a, b) ON t TO user:ana");
    ok(&root, "REVOKE ALL PRIVILEGES ON t FROM user:ana");
    denied(&ana, "SELECT a FROM t");
    denied(&ana, "DESCRIBE TABLE t");
    // `GRANT OPTION FOR` on the table clears the columns' flags too.
    ok(&root, "GRANT SELECT (a) ON t TO user:ana WITH GRANT OPTION");
    ok(&root, "REVOKE GRANT OPTION FOR SELECT ON t FROM user:ana");
    assert_eq!(ok(&ana, "SELECT a FROM t"), 1);
    denied(&ana, "GRANT SELECT (a) ON t TO user:carl");
}

/// `SHOW GRANTS ON TABLE t` rows as an admin sees them, as text.
fn grants_on(w: &World, table: &str) -> String {
    let shown = block_on(async {
        w.admin()
            .sql(&format!("SHOW GRANTS ON TABLE {table}"))
            .await?
            .collect()
            .await
    })
    .unwrap();
    format!("{shown:?}")
}

#[test]
fn a_revoke_reaches_only_the_callers_own_grants_unless_owner_or_admin() {
    let w = World::new();
    table(&w, "t");
    let (root, ana, bob, carl, g) = (
        w.admin(),
        w.user("ana"),
        w.user("bob"),
        w.user("carl"),
        w.user("g"),
    );
    ok(&root, "GRANT SELECT ON t TO user:ana WITH GRANT OPTION");
    ok(&root, "GRANT SELECT ON t TO user:bob WITH GRANT OPTION");
    // Grantors ana and bob both grant SELECT to g; bob's revoke takes only bob's row.
    ok(&ana, "GRANT SELECT ON t TO user:g");
    ok(&bob, "GRANT SELECT ON t TO user:g");
    ok(&bob, "REVOKE SELECT ON t FROM user:g");
    let shown = grants_on(&w, "t");
    assert!(shown.contains("\"user:g\""), "{shown}");
    assert_eq!(ok(&g, "SELECT * FROM t"), 1);
    // A grant only ana made, revoked by bob, answers exactly as a grant never made: success,
    // nothing changed, no audit record (no existence oracle).
    let before = grants_on(&w, "t");
    let made = run(&bob, "REVOKE SELECT ON t FROM user:g").map_err(|e| e.to_string());
    let never = run(&bob, "REVOKE SELECT ON t FROM user:nobody").map_err(|e| e.to_string());
    assert_eq!(made, never);
    assert_eq!(made, Ok(0));
    assert_eq!(grants_on(&w, "t"), before);
    // Same for an admin's grant and for the grant option of another grantor's grantee.
    ok(&bob, "REVOKE SELECT ON t FROM user:ana");
    ok(&bob, "REVOKE GRANT OPTION FOR SELECT ON t FROM user:ana");
    ok(&ana, "GRANT SELECT ON t TO user:dee");
    assert_eq!(ok(&w.user("dee"), "SELECT * FROM t"), 1);
    // The grantor's own revoke works.
    ok(&ana, "REVOKE SELECT ON t FROM user:g");
    denied(&g, "SELECT * FROM t");
    // No grant authority at all is refused before anything is looked up.
    denied(&carl, "REVOKE SELECT ON t FROM user:g");
    denied(&carl, "REVOKE SELECT ON t FROM user:nobody");
    // An admin revokes anyone's.
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    denied(&ana, "SELECT * FROM t");
    // The owner revokes a grant an admin made on its object.
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    let wh = w.dir.path().join("wh");
    ok(
        &root,
        &format!(
            "GRANT ALL PRIVILEGES ON PATH 'file:{}' TO user:ana",
            wh.display()
        ),
    );
    ok(&ana, "CREATE TABLE mine (a INT) USING parquet");
    ok(&root, "GRANT SELECT ON mine TO user:bob");
    assert_eq!(ok(&bob, "SELECT * FROM mine"), 0);
    ok(&ana, "REVOKE SELECT ON mine FROM user:bob");
    denied(&bob, "SELECT * FROM mine");
}
