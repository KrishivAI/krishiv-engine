// SPDX-License-Identifier: Apache-2.0
//! P3-T062 G1+G2 security review regressions (`ops/reviews/2026-11-15-opus-A-t062-g1-g2.md`):
//! the reviewer's probes P1-P14, each asserting the secure outcome, and the further shapes the
//! review lists for B1 (column reads derived from the resolved plan).
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

struct World {
    rt: Arc<EngineRuntime>,
    grants: Arc<Grants>,
    dir: tempfile::TempDir,
}

impl World {
    fn new() -> Self {
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
            grants: Grants::new(Arc::new(MemoryGrantStore::new())),
            dir,
        }
    }

    fn with(&self, name: &str, admin: bool) -> Session {
        let p = Principal::new(name, PrincipalKind::User);
        let policy = Arc::new(GrantsPolicy::new(
            Arc::clone(&self.grants),
            Arc::new(AllowAll),
            admin,
        ));
        Session::builder()
            .runtime(Arc::clone(&self.rt))
            .principal(p)
            .access_policy(Arc::clone(&policy) as Arc<dyn AccessPolicy>)
            .grants(policy)
            .build()
            .unwrap()
    }

    fn user(&self, name: &str) -> Session {
        self.with(name, false)
    }

    fn admin(&self) -> Session {
        self.with("root", true)
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

/// An admin's table `name (a INT, b INT)` with one row.
fn table(w: &World, name: &str) {
    let root = w.admin();
    ok(
        &root,
        &format!("CREATE TABLE {name} (a INT, b INT) USING parquet"),
    );
    ok(&root, &format!("INSERT INTO {name} VALUES (1, 2)"));
}

fn show(s: &Session, sql: &str) -> String {
    let r = run(s, sql);
    let out = match &r {
        Ok(n) => format!("OK rows={n}"),
        Err(e) => format!("ERR {}: {e}", e.error_class().name()),
    };
    eprintln!("PROBE [{sql}] -> {out}");
    out
}

fn wh_grant(w: &World, root: &Session, who: &str) {
    let wh = w.dir.path().join("wh");
    ok(
        root,
        &format!(
            "GRANT ALL PRIVILEGES ON PATH 'file:{}' TO user:{who}",
            wh.display()
        ),
    );
}

/// P1: an ownerless existing table is taken over by `CREATE TABLE IF NOT EXISTS`.
#[test]
fn p1_if_not_exists_takes_ownerless_table() {
    let w = World::new();
    // A table created without grants (before grants were on, another engine, a write path
    // that records no owner): same runtime, plain session.
    let plain = Session::builder()
        .runtime(Arc::clone(&w.rt))
        .build()
        .unwrap();
    ok(&plain, "CREATE TABLE legacy (a INT, b INT) USING parquet");
    ok(&plain, "INSERT INTO legacy VALUES (1, 2)");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    wh_grant(&w, &root, "ana");
    show(&ana, "SELECT * FROM legacy");
    show(
        &ana,
        "CREATE TABLE IF NOT EXISTS legacy (a INT, b INT) USING parquet",
    );
    let r = show(&ana, "SELECT * FROM legacy");
    assert!(r.starts_with("ERR"), "ana took ownership of legacy: {r}");
}

/// P2: `CREATE NAMESPACE IF NOT EXISTS default` makes the caller owner of `default` and
/// everything in it (owns() covers children).
#[test]
fn p2_if_not_exists_takes_ownerless_namespace() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT CREATE ON CATALOG spark_catalog TO user:ana");
    wh_grant(&w, &root, "ana");
    show(&ana, "SELECT * FROM t");
    show(&ana, "CREATE NAMESPACE IF NOT EXISTS default");
    show(&ana, "SHOW GRANTS");
    let r = show(&ana, "SELECT * FROM t");
    assert!(
        r.starts_with("ERR"),
        "ana read root's table via namespace ownership: {r}"
    );
}

/// P3: rename leaves the owner record at the old name; a new table there belongs to the old owner.
#[test]
fn p3_rename_leaves_stale_owner() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    for who in ["ana", "bob"] {
        ok(
            &root,
            &format!("GRANT CREATE ON SCHEMA default TO user:{who}"),
        );
        wh_grant(&w, &root, who);
    }
    ok(&ana, "CREATE TABLE a1 (a INT) USING parquet");
    show(&ana, "ALTER TABLE a1 RENAME TO a2");
    show(&ana, "SELECT * FROM a2");
    show(&bob, "CREATE TABLE a1 (a INT) USING parquet");
    show(&bob, "INSERT INTO a1 VALUES (7)");
    show(&bob, "SELECT * FROM a1");
    let r = show(&ana, "SELECT * FROM a1");
    assert!(r.starts_with("ERR"), "ana reads bob's new a1: {r}");
}

/// P4: whole rows through `TABLE t` in a set operation with one granted column.
#[test]
fn p4_table_statement_in_union_reads_hidden_columns() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    show(&ana, "TABLE t");
    let r = show(&ana, "TABLE t UNION ALL SELECT a, a FROM t WHERE false");
    let shown = block_on(async {
        ana.sql("TABLE t UNION ALL SELECT a, a FROM t WHERE false")
            .await?
            .collect()
            .await
    });
    eprintln!("P4 rows: {shown:?}");
    assert!(r.starts_with("ERR"), "hidden b read: {r}");
}

/// P5: a USING join key is not recorded as a referenced column.
#[test]
fn p5_using_join_filters_on_hidden_column() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    let hit = show(&ana, "SELECT t.a FROM t JOIN (SELECT 2 AS b) v USING (b)");
    show(&ana, "SELECT t.a FROM t JOIN (SELECT 3 AS b) v USING (b)");
    show(&ana, "SELECT t.a FROM t NATURAL JOIN (SELECT 2 AS b) v");
    assert!(hit.starts_with("ERR"), "hidden b compared: {hit}");
}

/// P6: PIVOT's implicit grouping columns are not recorded.
#[test]
fn p6_pivot_groups_by_hidden_column() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    let r = show(
        &ana,
        "SELECT c1 FROM t PIVOT (sum(a) AS s FOR a IN (1 AS c1))",
    );
    assert!(r.starts_with("ERR"), "grouped by hidden b: {r}");
}

/// P7: DataFrame `table(t).filter(a > 0)` returns all columns with a column grant on `a`.
#[test]
fn p7_dataframe_table_with_filter() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    let r = block_on(async { ana.table("t").filter_expr("a > 0")?.collect().await });
    eprintln!("P7: {r:?}");
    assert!(r.is_err(), "hidden b returned: {r:?}");
}

/// P8: case-sensitive analysis: two tables differing in case share one grant key.
#[test]
fn p8_case_sensitive_tables_share_a_key() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "SET spark.sql.caseSensitive=true");
    show(&root, "CREATE TABLE ct (a INT) USING parquet");
    show(&root, "CREATE TABLE CT (a INT) USING parquet");
    show(&root, "INSERT INTO CT VALUES (9)");
    ok(&root, "GRANT SELECT ON TABLE ct TO user:ana");
    ok(&ana, "SET spark.sql.caseSensitive=true");
    let r = show(&ana, "SELECT * FROM CT");
    show(&root, "SHOW TABLES");
    assert!(r.starts_with("ERR"), "grant on ct covers CT: {r}");
}

/// P9: case-sensitive columns `a` and `A`: a grant on `a` covers `A`.
#[test]
fn p9_case_sensitive_columns_share_a_key() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "SET spark.sql.caseSensitive=true");
    show(&root, "CREATE TABLE cc (a INT, A INT) USING parquet");
    show(&root, "INSERT INTO cc VALUES (1, 99)");
    ok(&root, "GRANT SELECT (a) ON cc TO user:ana");
    ok(&ana, "SET spark.sql.caseSensitive=true");
    let r = show(&ana, "SELECT A FROM cc");
    assert!(r.contains(DENIED), "grant on a covers A: {r}");
}

/// P10: a symlink inside a granted directory reads outside it.
#[test]
fn p10_symlink_under_granted_path() {
    let w = World::new();
    let data = w.dir.path().join("data");
    std::fs::create_dir_all(data.join("sales")).unwrap();
    std::fs::create_dir_all(data.join("secret")).unwrap();
    std::fs::write(data.join("secret/a.csv"), "42\n").unwrap();
    std::os::unix::fs::symlink(data.join("secret"), data.join("sales/link")).unwrap();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(
        &root,
        &format!(
            "GRANT SELECT ON PATH 'file:{}' TO user:ana",
            data.join("sales").display()
        ),
    );
    let r = show(
        &ana,
        &format!("SELECT * FROM csv.`{}`", data.join("sales/link").display()),
    );
    show(
        &ana,
        &format!("SELECT * FROM csv.`{}`", data.join("secret").display()),
    );
    assert!(r.starts_with("ERR"), "symlink escape: {r}");
}

/// P11: a grant to a role that does not exist yet is claimed by whoever creates the role.
#[test]
fn p11_grant_to_missing_role_is_claimable() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    show(&root, "GRANT SELECT ON t TO role:analysts");
    show(&ana, "CREATE ROLE analysts");
    show(&ana, "GRANT ROLE analysts TO user:ana");
    let r = show(&ana, "SELECT * FROM t");
    assert!(r.starts_with("ERR"), "ana claimed the role's grants: {r}");
}

/// P12: role policy alone (no grants): a denied read still says SELECT.
#[test]
fn p12_role_policy_alone_says_select() {
    let rt = EngineRuntime::builder().build().unwrap();
    let root = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    ok(&root, "CREATE TABLE t (a INT) USING parquet");
    let viewer: Arc<dyn AccessPolicy> = Arc::new(RolePolicy::new(
        Some(Role::Viewer),
        Role::Admin,
        PolicyVersion(1),
    ));
    let v = Session::builder()
        .runtime(rt)
        .principal(Principal::new("v", PrincipalKind::User))
        .access_policy(viewer)
        .build()
        .unwrap();
    let r = show(&v, "SELECT * FROM t");
    assert!(r.contains("SELECT") && !r.contains("BROWSE"), "{r}");
}

/// P13: a column grant: what does the bind-time refusal say for a user with nothing.
#[test]
fn p13_no_grant_error_text() {
    let w = World::new();
    table(&w, "t");
    let ana = w.user("ana");
    let r = show(&ana, "SELECT a FROM t");
    assert!(
        r.contains("SELECT privilege") && !r.contains("BROWSE"),
        "{r}"
    );
    let r = show(&ana, "INSERT INTO t VALUES (1, 1)");
    assert!(r.contains("MODIFY privilege"), "{r}");
}

/// P14: MODIFY without SELECT: a predicate on the target reads it.
#[test]
fn p14_modify_without_select_reads_by_predicate() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT MODIFY ON t TO user:ana");
    show(&ana, "DELETE FROM t WHERE b = 2");
    show(&ana, "INSERT INTO t SELECT * FROM t");
}

/// B1: every shape that reads the hidden column `b` asks for it; shapes that read only `a` pass.
#[test]
fn b1_hidden_column_reads_are_asked() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    for sql in [
        "SELECT a FROM t ORDER BY b",
        "SELECT a, sum(a) OVER (PARTITION BY b) FROM t",
        "SELECT a FROM t GROUP BY a, b",
        "TABLE t EXCEPT SELECT 1, 1",
        "TABLE t INTERSECT SELECT 1, 2",
        "WITH c AS (SELECT * FROM t) SELECT a FROM c UNION ALL SELECT b FROM c",
        "SELECT a FROM t WHERE b = 2",
        "SELECT a FROM t NATURAL JOIN (SELECT 2 AS b) v",
        "SELECT * FROM t",
        "SELECT t.a FROM t JOIN (SELECT 2 AS b) v ON t.b = v.b",
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM t u WHERE u.b = t.a)",
        "SELECT (SELECT max(b) FROM t)",
        "CACHE TABLE x AS TABLE t",
        "SELECT named_struct('x', b).x FROM t",
    ] {
        denied(&ana, sql);
    }
    let distinct = block_on(async { ana.table("t").distinct().select(["a"]).collect().await });
    assert!(distinct.is_err(), "distinct over a and b: {distinct:?}");
    // Shapes that read only `a`.
    for sql in [
        "SELECT a FROM t",
        "SELECT a FROM t WHERE a > 0 ORDER BY a",
        "SELECT a FROM (SELECT * FROM t)",
        "WITH c AS (SELECT * FROM t) SELECT x.a FROM c x JOIN c y ON x.a = y.a",
        "SELECT a FROM t UNION ALL SELECT a FROM t",
        "SELECT a FROM (SELECT a, b FROM t UNION ALL SELECT a, b FROM t)",
        "SELECT a, count(*) FROM t GROUP BY a",
        "SELECT a FROM t GROUP BY ALL",
    ] {
        assert!(ok(&ana, sql) > 0, "{sql}");
    }
    // `count(*)` reads rows: the table grant, which ana does not hold.
    denied(&ana, "SELECT count(*) FROM t");
}

/// m1: the `system` catalog's namespaces stay the role matrix's under grants.
#[test]
fn m1_system_namespaces_are_decided_by_roles() {
    let w = World::new();
    let ana = w.user("ana");
    ok(&ana, "SHOW TABLES IN system");
}

/// m5 and open point 2: `CREATE` on the namespace alone lets a principal create a managed table
/// (its directory), fill it (CTAS, `saveAsTable`) and own it; an explicit location still needs a
/// path grant; a no-op `saveAsTable` (`ignore`) over someone else's table takes nothing.
#[test]
fn m5_creators_own_what_they_create_without_a_path_grant() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&ana, "CREATE TABLE mine (a INT) USING parquet");
    assert_eq!(ok(&ana, "SELECT * FROM mine"), 0);
    ok(&ana, "CREATE TABLE copy USING parquet AS SELECT 1 AS a");
    assert_eq!(ok(&ana, "SELECT * FROM copy"), 1);
    // Later writes into the directory still need a path grant (only creation is covered).
    denied(&ana, "INSERT INTO mine VALUES (1)");
    let saved = block_on(ana.sql("SELECT 7 AS a"))
        .unwrap()
        .write()
        .save_as_table("saved");
    block_on(saved).unwrap();
    assert_eq!(ok(&ana, "SELECT * FROM saved"), 1);
    denied(&bob, "SELECT * FROM saved");
    let outside = w.dir.path().join("ext");
    denied(
        &ana,
        &format!(
            "CREATE TABLE ext (a INT) USING parquet LOCATION '{}'",
            outside.display()
        ),
    );
    // A no-op create over root's table takes nothing.
    ok(&root, "CREATE TABLE roots (a INT) USING parquet");
    let df = block_on(ana.sql("SELECT 1 AS a")).unwrap();
    let ignored = df.write().mode("ignore").save_as_table("roots");
    let _ = block_on(ignored);
    denied(&ana, "SELECT * FROM roots");
}

/// M3 for an admin: a grant to a missing role names it as missing.
#[test]
fn m3_admin_grant_to_a_missing_role_is_role_not_found() {
    let w = World::new();
    table(&w, "t");
    let e = run(&w.admin(), "GRANT SELECT ON t TO role:nobody").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_ROLE_NOT_FOUND", "{e}");
    let e = run(&w.admin(), "REVOKE SELECT ON t FROM role:nobody").unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_ROLE_NOT_FOUND", "{e}");
}

/// B3: a rename moves the owner and the grants to the new name; the old name is left clean;
/// a drop clears them.
#[test]
fn b3_rename_moves_owner_and_grants() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    wh_grant(&w, &root, "ana");
    ok(&ana, "CREATE TABLE r1 (a INT) USING parquet");
    ok(&ana, "GRANT SELECT ON r1 TO user:bob");
    ok(&ana, "ALTER TABLE r1 RENAME TO r2");
    ok(&ana, "SELECT * FROM r2");
    ok(&bob, "SELECT * FROM r2");
    ok(&root, "CREATE TABLE r1 (a INT) USING parquet");
    denied(&ana, "SELECT * FROM r1");
    denied(&bob, "SELECT * FROM r1");
}

/// m2: the file store is opened by one owner at a time.
#[test]
fn m2_file_store_refuses_a_second_opener() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grants.json");
    let first = krishiv_session::grants::FileGrantStore::open(&path).unwrap();
    let e = krishiv_session::grants::FileGrantStore::open(&path).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "KRISHIV_GRANTS_STORE_FAILURE",
        "{e}"
    );
    drop(first);
    krishiv_session::grants::FileGrantStore::open(&path).unwrap();
}
