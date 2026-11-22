// SPDX-License-Identifier: Apache-2.0
//! P3-T062 unit G2: the `grants` provider end to end. Sessions of several principals share one
//! runtime (one catalog) and one [`Grants`]; an admin creates the data, and every other principal
//! reads only what its grants (direct, through a group, through roles, on a parent, per column,
//! as owner) allow. A revoke takes effect for plans analysed and relations cached before it.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
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

/// `observe()` metrics read the columns they name: a metric over a column without a grant refuses
/// the action, even when the frame's output (a projection above the observation) needs only
/// granted columns. A DataFrame column is no reference the statement wrote, so the refusal names
/// the table, never the column (ADR 0048 A1 §3, `krishiv_plan` `analyzer/written.rs`).
#[test]
fn observe_metrics_need_a_grant_on_their_columns() {
    use krishiv_session::Observation;
    use krishiv_session::functions::{col, max};
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    let count = |metric: &str| {
        block_on(async {
            let obs = Observation::new();
            let df = ana
                .table("t")
                .observe(&obs, [max(col(metric))])?
                .select([col("a")]);
            df.collect().await.map(|b| {
                b.iter()
                    .map(arrow_array::RecordBatch::num_rows)
                    .sum::<usize>()
            })
        })
    };
    let rows = count("a").unwrap_or_else(|e| panic!("observe(max(a)): {e}"));
    assert_eq!(rows, 1, "the one row of `t`");
    match count("b") {
        Err(e) if e.error_class().name() == DENIED => {
            let text = e.to_string();
            assert!(text.contains("`t`") && !text.contains("`b`"), "{e}");
        }
        other => panic!("observe(max(b)): expected a denial, got {other:?}"),
    }
}

/// `ROLLUP`, `CUBE` and `GROUPING SETS` read the columns they name and no other: a column grant
/// on exactly those runs them, and one referenced column without a grant refuses them.
#[test]
fn column_grants_cover_grouping_sets_over_their_columns() {
    let w = World::new();
    let root = w.admin();
    ok(
        &root,
        "CREATE TABLE g (a INT, b INT, c INT, d INT) USING parquet",
    );
    ok(&root, "INSERT INTO g VALUES (1, 2, 3, 4)");
    let ana = w.user("ana");
    ok(&root, "GRANT SELECT (a, b, c) ON g TO user:ana");
    // One input row: one output row per grouping set.
    let shapes = [
        (
            "SELECT a, b, sum(c), grouping(a), grouping_id() FROM g GROUP BY ROLLUP(a, b)",
            3,
        ),
        ("SELECT a, sum(c) FROM g GROUP BY CUBE(a, b)", 4),
        (
            "SELECT a, grouping_id(a, b), max(c) FROM g GROUP BY GROUPING SETS ((a), (a, b), ())",
            3,
        ),
        ("SELECT a + b, count(c) FROM g GROUP BY ROLLUP(a + b)", 2),
    ];
    for (sql, rows) in shapes {
        assert_eq!(ok(&ana, sql), rows, "{sql}");
    }
    // The same shapes with `d` referenced (a key, an aggregate input, a computed key).
    for sql in [
        "SELECT a, b, sum(c), grouping(a) FROM g GROUP BY ROLLUP(a, b, d)",
        "SELECT a, sum(d) FROM g GROUP BY CUBE(a, b)",
        "SELECT a, grouping_id(a, d) FROM g GROUP BY GROUPING SETS ((a), (a, d))",
        "SELECT a + d, count(c) FROM g GROUP BY ROLLUP(a + d)",
    ] {
        let e = denied(&ana, sql);
        assert!(e.contains("`d`"), "{sql}: {e}");
    }
    // Without `c`, the shapes reading `c` are refused.
    let bob = w.user("bob");
    ok(&root, "GRANT SELECT (a, b) ON g TO user:bob");
    for (sql, _) in shapes {
        let e = denied(&bob, sql);
        assert!(e.contains("`c`"), "{sql}: {e}");
    }
}

/// A grouping key written for another relation's column of the same name is no reference to the
/// refused column: the refusal names the table, never `d` (ADR 0048 A1 §3, merge review N2).
#[test]
fn a_grouping_key_of_another_relation_names_only_the_table() {
    let w = World::new();
    let root = w.admin();
    ok(&root, "CREATE TABLE g (a INT, d INT) USING parquet");
    ok(&root, "CREATE TABLE h (d INT) USING parquet");
    ok(&root, "INSERT INTO g VALUES (1, 2)");
    ok(&root, "INSERT INTO h VALUES (1)");
    let ana = w.user("ana");
    ok(&root, "GRANT SELECT (a) ON g TO user:ana");
    ok(&root, "GRANT SELECT ON h TO user:ana");
    let sql = "SELECT h.d, count(struct(s.*)) FROM (SELECT * FROM g) s JOIN h ON s.a = h.d GROUP BY ROLLUP(h.d)";
    let e = denied(&ana, sql);
    assert!(e.contains("`g`") && !e.contains("`d`"), "{sql}: {e}");
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

/// `DROP VIEW v` with a bare name that names no temporary view drops the stored view: it asks
/// `DROP` on that view and clears its ownership, as `DROP VIEW default.v` does (found on #941).
#[test]
fn a_bare_drop_view_of_a_stored_view_needs_the_drop_privilege() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(&ana, "CREATE VIEW v AS SELECT 1 AS a");
    ok(&ana, "GRANT SELECT ON v TO user:bob");
    denied(&bob, "DROP VIEW v");
    denied(&bob, "DROP VIEW IF EXISTS v");
    denied(&bob, "DROP VIEW default.v");
    assert_eq!(ok(&bob, "SELECT * FROM v"), 1);
    // A temporary view of the bare name is the session's own: dropped with no check.
    ok(&bob, "CREATE TEMPORARY VIEW v AS SELECT 2 AS a");
    ok(&bob, "DROP VIEW v");
    denied(&bob, "DROP VIEW v");
    // The owner drops it with the bare name, and the drop takes its grants with it.
    ok(&ana, "DROP VIEW v");
    ok(&root, "CREATE VIEW v AS SELECT 1 AS a");
    denied(&bob, "SELECT * FROM v");
    denied(&ana, "SELECT * FROM v");
    denied(&ana, "DROP VIEW v");
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

#[test]
fn each_grantor_keeps_its_own_row() {
    let w = World::new();
    table(&w, "t");
    let (root, ana, carl) = (w.admin(), w.user("ana"), w.user("carl"));
    ok(&root, "GRANT MODIFY ON t TO user:ana WITH GRANT OPTION");
    // review re-check 3 n1: the same grant from ana, then from an admin, is two rows.
    ok(&ana, "GRANT MODIFY ON t TO user:carl");
    ok(&root, "GRANT MODIFY ON t TO user:carl");
    let shown = grants_on(&w, "t");
    assert_eq!(shown.matches("\"user:carl\"").count(), 2, "{shown}");
    // ana's revoke takes only ana's row: the admin's grant stays.
    ok(&ana, "REVOKE MODIFY ON t FROM user:carl");
    ok(&carl, "INSERT INTO t VALUES (5, 6)");
    let shown = grants_on(&w, "t");
    assert_eq!(shown.matches("\"user:carl\"").count(), 1, "{shown}");
    // Granting again by the same grantor keeps one row (the grant option is added).
    ok(&ana, "GRANT MODIFY ON t TO user:carl");
    ok(&ana, "GRANT MODIFY ON t TO user:carl");
    assert_eq!(grants_on(&w, "t").matches("\"user:carl\"").count(), 2);
    ok(&root, "REVOKE MODIFY ON t FROM user:carl");
    denied(&carl, "INSERT INTO t VALUES (7, 8)");
}

/// The creator of a metric view owns it, as the creator of a view does (#941 review thread):
/// it reads and drops it without an admin; `IF NOT EXISTS` over it gives another user nothing.
#[test]
fn the_creator_of_a_metric_view_owns_it() {
    let w = World::new();
    let root = w.admin();
    ok(
        &root,
        "CREATE TABLE src (region STRING, amount INT) USING parquet",
    );
    ok(&root, "INSERT INTO src VALUES ('eu', 1)");
    let (ana, bob) = (w.user("ana"), w.user("bob"));
    for who in ["ana", "bob"] {
        ok(
            &root,
            &format!("GRANT CREATE ON SCHEMA default TO user:{who}"),
        );
        ok(&root, &format!("GRANT SELECT ON TABLE src TO user:{who}"));
    }
    let create = |not_exists: &str| {
        format!(
            "CREATE VIEW {not_exists}mv WITH METRICS LANGUAGE YAML AS $$\nversion: 0.1\n\
             source: src\ndimensions:\n  - name: region\n    expr: region\nmeasures:\n  \
             - name: total\n    expr: sum(amount)\n$$"
        )
    };
    ok(&ana, &create(""));
    let shown =
        block_on(async { ana.sql("SHOW GRANTS ON TABLE mv").await?.collect().await }).unwrap();
    let text = format!("{shown:?}");
    assert!(
        text.contains("OWNERSHIP") && text.contains("user:ana"),
        "{text}"
    );
    let rows = block_on(async {
        ana.sql("SELECT region, measure(total) FROM mv GROUP BY region")
            .await?
            .collect()
            .await
    })
    .unwrap();
    // The one source row: region `eu`, `sum(amount)` 1.
    let got: Vec<(String, i64)> = rows
        .iter()
        .flat_map(|b| {
            let region = arrow_cast::cast(b.column(0), &arrow_schema::DataType::Utf8).unwrap();
            let total = arrow_cast::cast(b.column(1), &arrow_schema::DataType::Int64).unwrap();
            let region = region.as_string::<i32>();
            let total = total.as_primitive::<arrow_array::types::Int64Type>();
            (0..b.num_rows())
                .map(|i| (region.value(i).to_owned(), total.value(i)))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(got, [("eu".to_owned(), 1)]);
    ok(&bob, &create("IF NOT EXISTS "));
    denied(
        &bob,
        "SELECT region, measure(total) FROM mv GROUP BY region",
    );
    ok(&ana, "DROP VIEW mv");
}
