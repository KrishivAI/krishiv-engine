// SPDX-License-Identifier: Apache-2.0
//! P3-T062 G1+G2 security review regressions (`ops/reviews/2026-11-15-opus-A-t062-g1-g2.md`):
//! the reviewer's probes P1-P14, each asserting the secure outcome, and the further shapes the
//! review lists for B1 (column reads derived from the resolved plan).
#![cfg(test)]
mod file_url;
mod literal_path;
use file_url::file_url;
use literal_path::sql_path;

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
            "GRANT ALL PRIVILEGES ON PATH '{}' TO user:{who}",
            file_url(&wh)
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

#[cfg(unix)]
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
            "GRANT SELECT ON PATH '{}' TO user:ana",
            file_url(&data.join("sales"))
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
    // The owner's managed directory is its table's (review R2).
    ok(&ana, "INSERT INTO mine VALUES (1)");
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
            sql_path(&outside)
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

// ---- re-check probes (reviewer, head cff606388; R2 lines made strict) ----

fn expect(s: &Session, sql: &str, deny: bool, bad: &mut Vec<String>) {
    let r = run(s, sql);
    let line = match &r {
        Ok(n) if deny => format!("HOLE (allowed, rows={n}): {sql}"),
        Ok(_) => format!("ok allowed: {sql}"),
        Err(e) if e.error_class().name() == DENIED && deny => format!("ok denied: {sql}"),
        Err(e) if e.error_class().name() == DENIED => format!("OVERDENY: {sql}: {e}"),
        Err(e) => format!("INCONCLUSIVE {}: {sql}: {e}", e.error_class().name()),
    };
    eprintln!("RECHECK {line}");
    if line.starts_with("HOLE") {
        bad.push(line);
    }
}

fn column_world() -> (World, Session, Session) {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    (w, root, ana)
}

#[test]
fn r1_fresh_shapes_read_hidden_columns() {
    let (_w, _root, ana) = column_world();
    let mut bad = Vec::new();
    for sql in [
        // generators
        "SELECT e FROM t LATERAL VIEW explode(array(b)) x AS e",
        "SELECT a FROM t LATERAL VIEW explode(array(b)) x AS e WHERE e = 2",
        "SELECT a FROM t LATERAL VIEW OUTER explode(map(b, 1)) x AS k, v WHERE k = 2",
        "SELECT a, e FROM t, LATERAL explode(sequence(1, b)) x(e)",
        "SELECT col FROM t, LATERAL (SELECT explode(array(t.b)))",
        // lambdas
        "SELECT transform(array(a), x -> x + b) FROM t",
        "SELECT a FROM t WHERE exists(array(a), x -> x = b)",
        "SELECT filter(array(a), x -> b > 0) FROM t",
        "SELECT aggregate(array(a), 0, (acc, x) -> acc + b) FROM t",
        "SELECT a FROM t WHERE size(filter(array(1, 2), x -> x = b)) > 0",
        // windows
        "SELECT a, sum(a) OVER (ORDER BY b ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        "SELECT a, lag(a) OVER (ORDER BY b) FROM t",
        "SELECT a FROM t QUALIFY row_number() OVER (PARTITION BY b ORDER BY a) = 1",
        "SELECT a, count(*) OVER (PARTITION BY a ORDER BY b RANGE BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t",
        "SELECT a FROM (SELECT a, rank() OVER (ORDER BY b) r FROM t) WHERE r = 1",
        // correlated subqueries
        "SELECT a FROM t WHERE EXISTS (SELECT 1 FROM (SELECT 2 AS v) x WHERE x.v = t.b)",
        "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM (SELECT 3 AS v) x WHERE x.v = t.b)",
        "SELECT a, (SELECT max(v) FROM (SELECT 2 AS v) x WHERE x.v = t.b) FROM t",
        "SELECT a FROM t WHERE b IN (SELECT 2)",
        "SELECT a FROM t WHERE a IN (SELECT b FROM t)",
        "SELECT a FROM t WHERE EXISTS (SELECT * FROM t u WHERE u.a = t.a AND u.b = 2)",
        "SELECT a FROM t WHERE (SELECT max(u.b) FROM t u) > 0",
        "SELECT a FROM t, LATERAL (SELECT t.b AS z) WHERE z = 2",
        "SELECT a FROM t WHERE a = (SELECT count(*) FROM t u WHERE u.b = t.b)",
        // self joins (Shift)
        "SELECT x.a FROM t x JOIN t y ON x.a = y.a WHERE y.b = 2",
        "SELECT x.a FROM t x, t y, t z WHERE z.b = x.a + 1",
        "WITH c AS (SELECT a, b FROM t) SELECT p.a FROM c p JOIN c q ON p.a = q.a JOIN c r ON r.b = p.a + 1",
        // repartition / sort / hints / grouping sets / unpivot / recursion
        "SELECT a FROM t DISTRIBUTE BY b",
        "SELECT a FROM t CLUSTER BY b",
        "SELECT a FROM t SORT BY b",
        "SELECT /*+ REPARTITION(3, b) */ a FROM t",
        "SELECT /*+ REPARTITION_BY_RANGE(b) */ a FROM t",
        "SELECT a FROM t GROUP BY GROUPING SETS ((a), (a, b))",
        "SELECT a, grouping(b) FROM t GROUP BY CUBE(a, b)",
        "SELECT n FROM t UNPIVOT (v FOR n IN (a, b))",
        "SELECT n FROM t UNPIVOT INCLUDE NULLS (v FOR n IN (a, b)) WHERE v = 2",
        "WITH RECURSIVE r(n) AS (SELECT b FROM t UNION ALL SELECT n + 1 FROM r WHERE n < 3) SELECT count(*) FROM r",
        "SELECT a FROM t PIVOT (sum(a) FOR b IN (2))",
        "SELECT a FROM t EXCEPT ALL SELECT a FROM t WHERE b = 2",
        "SELECT a FROM t INTERSECT SELECT b FROM t",
        "SELECT a FROM t UNION SELECT b FROM t",
        "SELECT * FROM (SELECT a FROM t) x JOIN (SELECT b FROM t) y ON true",
        "SELECT max_by(a, b) FROM t",
        "SELECT a FROM t WHERE a = 1 ORDER BY b NULLS FIRST LIMIT 1",
        "SELECT first(a) FROM (SELECT a FROM t ORDER BY b)",
        "SELECT a FROM t TABLESAMPLE (BUCKET 1 OUT OF 2 ON b)",
        "SELECT count(DISTINCT a) FILTER (WHERE b = 2) FROM t",
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY b) FROM t",
        "SELECT a FROM t WHERE named_struct('x', b).x = 2",
        "SELECT a FROM t WHERE CASE WHEN b = 2 THEN true END",
        "SELECT a FROM t WHERE hash(*) > 0",
        "SELECT a FROM t WHERE to_json(struct(*)) LIKE '%2%'",
        "SELECT input_file_name() FROM t",
        "CACHE TABLE x AS SELECT a FROM t WHERE b = 2",
        "CREATE TEMPORARY VIEW tv AS SELECT a FROM t WHERE b = 2",
    ] {
        expect(&ana, sql, true, &mut bad);
    }
    // Over-denial check (should be allowed).
    for sql in [
        "SELECT a FROM t LATERAL VIEW explode(array(a)) x AS e",
        "SELECT transform(array(a), x -> x + 1) FROM t",
        "SELECT a, sum(a) OVER (PARTITION BY a ORDER BY a) FROM t",
        "SELECT a FROM t WHERE EXISTS (SELECT * FROM t u WHERE u.a = t.a)",
        "SELECT a FROM t WHERE a IN (SELECT a FROM t)",
        "SELECT x.a FROM t x JOIN t y ON x.a = y.a",
        "SELECT a FROM t DISTRIBUTE BY a",
        // Reads nothing a `SELECT a` would not show (re-check ruling under B2).
        "SELECT count(*) FROM t WHERE a = 1",
    ] {
        expect(&ana, sql, false, &mut bad);
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn r2_writes_with_modify_and_one_column() {
    let (_w, root, ana) = column_world();
    ok(&root, "GRANT MODIFY ON t TO user:ana");
    let mut bad = Vec::new();
    for sql in [
        "UPDATE t SET a = 1 WHERE b = 2",
        "UPDATE t SET a = b",
        "DELETE FROM t WHERE b = 2",
        "MERGE INTO t USING (SELECT 2 AS k) s ON t.b = s.k WHEN MATCHED THEN UPDATE SET a = 5",
        "MERGE INTO t USING (SELECT 1 AS k) s ON t.a = s.k WHEN MATCHED AND t.b = 2 THEN DELETE",
        "INSERT INTO t SELECT a, b FROM t",
        "INSERT OVERWRITE t SELECT a, a FROM t WHERE b = 2",
    ] {
        expect(&ana, sql, true, &mut bad);
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn r3_views_over_the_table() {
    let (_w, root, ana) = column_world();
    ok(&root, "CREATE VIEW v AS SELECT a, b FROM t");
    ok(&root, "CREATE VIEW vf AS SELECT a FROM t WHERE b = 2");
    ok(&root, "GRANT SELECT ON VIEW v TO user:ana");
    ok(&root, "GRANT SELECT ON VIEW vf TO user:ana");
    let mut bad = Vec::new();
    for sql in [
        "SELECT b FROM v",
        "SELECT a FROM v WHERE b = 2",
        "SELECT * FROM v",
        "SELECT a FROM v ORDER BY b",
        "SELECT a FROM vf",
        "SELECT count(*) FROM v",
    ] {
        expect(&ana, sql, true, &mut bad);
    }
    expect(&ana, "SELECT a FROM v", false, &mut bad);
    // A view grant alone (no base column grant) on a view over only the granted column.
    ok(&root, "CREATE VIEW va AS SELECT a FROM t");
    ok(&root, "GRANT SELECT ON VIEW va TO user:ana");
    expect(&ana, "SELECT a FROM va", false, &mut bad);
    // ana's own temp view over t.
    expect(
        &ana,
        "CREATE TEMPORARY VIEW mine AS SELECT * FROM t",
        false,
        &mut bad,
    );
    expect(&ana, "SELECT b FROM mine", true, &mut bad);
    expect(&ana, "SELECT a FROM mine WHERE b = 2", true, &mut bad);
    expect(&ana, "SELECT a FROM mine", false, &mut bad);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn r4_dataframe_ops() {
    let (_w, _root, ana) = column_world();
    let mut bad: Vec<String> = Vec::new();
    let mut check =
        |name: &str, r: Result<Vec<arrow_array::RecordBatch>, KrishivError>, deny: bool| {
            let line = match &r {
                Ok(_) if deny => format!("HOLE {name}"),
                Ok(_) => format!("ok allowed {name}"),
                Err(e) if e.error_class().name() == DENIED && deny => format!("ok denied {name}"),
                Err(e) if e.error_class().name() == DENIED => format!("OVERDENY {name}: {e}"),
                Err(e) => format!("INCONCLUSIVE {name}: {e}"),
            };
            eprintln!("RECHECK {line}");
            if line.starts_with("HOLE") {
                bad.push(line);
            }
        };
    let t = || ana.table("t");
    check(
        "filter b select a",
        block_on(async { t().filter_expr("b = 2")?.select(["a"]).collect().await }),
        true,
    );
    check(
        "sort b select a",
        block_on(t().sort(["b"]).select(["a"]).collect()),
        true,
    );
    check(
        "group_by b agg count",
        block_on(
            t().group_by(["b"])
                .agg([krishiv_session::functions::count(
                    krishiv_session::functions::col("a"),
                )])
                .select(["count(a)"])
                .collect(),
        ),
        true,
    );
    check(
        "union a b",
        block_on(t().select(["a"]).union(&t().select(["b"])).collect()),
        true,
    );
    check(
        "distinct then a",
        block_on(t().distinct().select(["a"]).collect()),
        true,
    );
    check(
        "hint repartition b",
        block_on(
            t().hint("repartition", [krishiv_session::functions::col("b")])
                .select(["a"])
                .collect(),
        ),
        true,
    );
    check(
        "select_expr b",
        block_on(async { t().select_expr(["a + b"])?.collect().await }),
        true,
    );
    check(
        "with_column c=b select a (b unused)",
        block_on(
            t().with_column("c", krishiv_session::functions::col("b"))
                .select(["a"])
                .collect(),
        ),
        false,
    );
    check(
        "with_column c=b filter c",
        block_on(async {
            t().with_column("c", krishiv_session::functions::col("b"))
                .filter_expr("c = 2")?
                .select(["a"])
                .collect()
                .await
        }),
        true,
    );
    check("select a", block_on(t().select(["a"]).collect()), false);
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
#[cfg_attr(
    windows,
    ignore = "case twins need a case-sensitive file system: on NTFS `CT` and `ct` are one directory"
)]
fn r5_case_twins_with_default_reader() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "SET spark.sql.caseSensitive=true");
    show(&root, "CREATE TABLE ct (a INT) USING parquet");
    show(&root, "CREATE TABLE CT (a INT) USING parquet");
    show(&root, "INSERT INTO CT VALUES (9)");
    ok(&root, "GRANT SELECT ON TABLE ct TO user:ana");
    // ana keeps the default (case-insensitive).
    let mut bad = Vec::new();
    expect(&ana, "SELECT * FROM CT", true, &mut bad);
    expect(&ana, "SELECT * FROM `CT`", true, &mut bad);
    expect(&ana, "SELECT * FROM default.CT", true, &mut bad);
    let df = block_on(async { ana.sql("SELECT * FROM ct").await?.collect().await });
    eprintln!("RECHECK ct as ana: {df:?}");
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn r6_replace_and_owner_edges() {
    let w = World::new();
    table(&w, "rt");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    let mut bad = Vec::new();
    expect(
        &ana,
        "CREATE OR REPLACE TABLE rt (a INT, b INT) USING parquet",
        true,
        &mut bad,
    );
    expect(
        &ana,
        "CREATE OR REPLACE TABLE rt USING parquet AS SELECT 1 AS a, 2 AS b",
        true,
        &mut bad,
    );
    expect(
        &ana,
        "CREATE TABLE IF NOT EXISTS rt USING parquet AS SELECT 1 AS a, 2 AS b",
        true,
        &mut bad,
    );
    expect(&ana, "SELECT * FROM rt", true, &mut bad);
    expect(&ana, "INSERT INTO rt VALUES (5, 5)", true, &mut bad);
    // Creating own table works without a path grant; reading it then works.
    ok(&ana, "CREATE TABLE own USING parquet AS SELECT 1 AS a");
    ok(&ana, "SELECT * FROM own");
    // An explicit location still needs a path grant.
    let other = w.dir.path().join("elsewhere");
    expect(
        &ana,
        &format!(
            "CREATE TABLE ext (a INT) USING parquet LOCATION '{}'",
            sql_path(&other)
        ),
        true,
        &mut bad,
    );
    // Rename root's table into a name ana then creates? ana cannot rename root's table.
    expect(&ana, "ALTER TABLE rt RENAME TO rt2", true, &mut bad);
    // ana renames her own table onto a name with pre-existing grants (cleared).
    ok(&root, "GRANT SELECT ON TABLE default.target TO user:bob");
    // The owner renames its managed table without a path grant (review R2).
    ok(&ana, "ALTER TABLE own RENAME TO target");
    assert_eq!(ok(&ana, "SELECT * FROM target"), 1);
    let bob = w.user("bob");
    expect(&bob, "SELECT * FROM target", true, &mut bad);
    // Function creation: owner?
    show(&ana, "CREATE FUNCTION f1() RETURNS INT RETURN 1");
    show(&ana, "SELECT f1()");
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
#[cfg_attr(
    windows,
    ignore = "case twins need a case-sensitive file system: on NTFS `CT` and `ct` are one directory"
)]
fn r5b_grant_on_upper_twin_read_lower() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "SET spark.sql.caseSensitive=true");
    show(&root, "CREATE TABLE ct (a INT) USING parquet");
    show(&root, "INSERT INTO ct VALUES (1)");
    show(&root, "CREATE TABLE CT (a INT) USING parquet");
    show(&root, "INSERT INTO CT VALUES (9)");
    ok(&root, "GRANT SELECT ON TABLE CT TO user:ana");
    eprintln!(
        "RECHECK root ct: {:?}",
        block_on(async { root.sql("SELECT * FROM ct").await?.collect().await })
    );
    eprintln!(
        "RECHECK root CT: {:?}",
        block_on(async { root.sql("SELECT * FROM CT").await?.collect().await })
    );
    let mut bad = Vec::new();
    expect(&ana, "SELECT * FROM ct", true, &mut bad);
    eprintln!(
        "RECHECK ana ct: {:?}",
        block_on(async { ana.sql("SELECT * FROM ct").await?.collect().await })
    );
    eprintln!(
        "RECHECK ana CT: {:?}",
        block_on(async { ana.sql("SELECT * FROM CT").await?.collect().await })
    );
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn r7_owner_lifecycle_and_views_by_dataframe() {
    let w = World::new();
    table(&w, "t");
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    let mut bad = Vec::new();
    // The owner writes and drops its managed table without a path grant (review R2).
    ok(&ana, "CREATE TABLE own2 (a INT) USING parquet");
    ok(&ana, "INSERT INTO own2 VALUES (1)");
    assert_eq!(ok(&ana, "SELECT * FROM own2"), 1);
    ok(&ana, "DROP TABLE own2");
    // A table of the same name made later by root is not ana's (its directory neither).
    ok(&root, "CREATE TABLE own2 (a INT) USING parquet");
    denied(&ana, "INSERT INTO own2 VALUES (1)");
    // DataFrame read of a view with only SELECT(a) on the base and no view grant.
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    ok(&root, "CREATE VIEW va AS SELECT a FROM t");
    let r = block_on(ana.table("va").select(["a"]).collect());
    eprintln!(
        "RECHECK df view va without view grant: {:?}",
        r.as_ref().map(|_| "ok").map_err(ToString::to_string)
    );
    if r.is_ok() {
        bad.push("HOLE df view without view grant".to_owned());
    }
    expect(&ana, "SELECT a FROM va", true, &mut bad);
    assert!(bad.is_empty(), "{bad:#?}");
}

// ---- re-check 2 probes (reviewer, head bf69afc39) ----

fn rows_of(s: &Session, sql: &str) -> String {
    match block_on(async { s.sql(sql).await?.collect().await }) {
        Ok(b) => {
            let mut v = Vec::new();
            for batch in &b {
                for c in batch.columns() {
                    v.push(format!(
                        "{:?}",
                        arrow_array::cast::as_primitive_array::<arrow_array::types::Int32Type>(c)
                            .values()
                    ));
                }
            }
            format!("OK {v:?}")
        }
        Err(e) => format!("ERR {}", e.error_class().name()),
    }
}

#[test]
fn q1_table_named_like_a_namespace_dir() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    let wh = w.dir.path().join("wh");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    let mut bad = Vec::new();
    // ana first: her table's managed directory is <wh>/x.db.
    expect(
        &ana,
        "CREATE TABLE `x.db` (a INT) USING parquet",
        false,
        &mut bad,
    );
    show(&root, "CREATE NAMESPACE x");
    show(&root, "CREATE TABLE x.secret (a INT) USING parquet");
    show(&root, "INSERT INTO x.secret VALUES (42)");
    eprintln!(
        "RECHECK2 dirs: {:?}",
        std::fs::read_dir(&wh).map(|d| d
            .filter_map(Result::ok)
            .map(|e| e.file_name())
            .collect::<Vec<_>>())
    );
    let p = wh.join("x.db").join("secret");
    expect(
        &ana,
        &format!("SELECT * FROM parquet.`{}`", p.display()),
        true,
        &mut bad,
    );
    eprintln!(
        "RECHECK2 q1 read: {}",
        rows_of(&ana, &format!("SELECT * FROM parquet.`{}`", p.display()))
    );
    expect(&ana, "DROP TABLE `x.db`", false, &mut bad);
    eprintln!(
        "RECHECK2 q1 after drop, root reads x.secret: {}",
        rows_of(&root, "SELECT * FROM x.secret")
    );
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
#[cfg_attr(
    windows,
    ignore = "case twins need a case-sensitive file system: on NTFS `CT` and `ct` are one directory"
)]
fn q2_twins_after_grant_across_namespaces_and_views() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    let mut bad = Vec::new();
    ok(&root, "CREATE TABLE ct (a INT) USING parquet");
    ok(&root, "INSERT INTO ct VALUES (1)");
    ok(&root, "GRANT SELECT ON TABLE ct TO user:ana");
    ok(&root, "SET spark.sql.caseSensitive=true");
    ok(&root, "CREATE TABLE CT (a INT) USING parquet");
    ok(&root, "INSERT INTO CT VALUES (9)");
    let anacs = w.user("ana");
    ok(&anacs, "SET spark.sql.caseSensitive=true");
    expect(&anacs, "SELECT * FROM CT", true, &mut bad);
    eprintln!(
        "RECHECK2 q2 ana default CT: {}",
        rows_of(&ana, "SELECT * FROM CT")
    );
    // namespaces
    show(&root, "CREATE NAMESPACE db");
    show(&root, "CREATE NAMESPACE DB");
    show(&root, "CREATE TABLE db.t (a INT) USING parquet");
    show(&root, "INSERT INTO db.t VALUES (1)");
    show(&root, "CREATE TABLE DB.t (a INT) USING parquet");
    show(&root, "INSERT INTO DB.t VALUES (9)");
    ok(&root, "GRANT SELECT ON TABLE db.t TO user:ana");
    ok(&root, "GRANT SELECT ON SCHEMA db TO user:ana");
    expect(&anacs, "SELECT * FROM DB.t", true, &mut bad);
    eprintln!(
        "RECHECK2 q2 ana default DB.t: {}",
        rows_of(&ana, "SELECT * FROM DB.t")
    );
    // view twin of a table
    show(&root, "CREATE TABLE vt (a INT) USING parquet");
    show(&root, "CREATE VIEW VT AS SELECT 9 AS a");
    ok(&root, "GRANT SELECT ON TABLE vt TO user:ana");
    expect(&anacs, "SELECT * FROM VT", true, &mut bad);
    eprintln!(
        "RECHECK2 q2 ana default VT: {}",
        rows_of(&ana, "SELECT * FROM VT")
    );
    show(&root, "CREATE VIEW vv AS SELECT 1 AS a");
    show(&root, "CREATE TABLE VV (a INT) USING parquet");
    show(&root, "INSERT INTO VV VALUES (9)");
    ok(&root, "GRANT SELECT ON VIEW vv TO user:ana");
    expect(&anacs, "SELECT * FROM VV", true, &mut bad);
    eprintln!(
        "RECHECK2 q2 ana default VV: {}",
        rows_of(&ana, "SELECT * FROM VV")
    );
    assert!(bad.is_empty(), "{bad:#?}");
}

#[test]
fn q3_managed_dirs_do_not_reach_siblings() {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    let wh = w.dir.path().join("wh");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "CREATE TABLE abc (a INT) USING parquet");
    ok(&root, "INSERT INTO abc VALUES (42)");
    let mut bad = Vec::new();
    expect(
        &ana,
        "CREATE TABLE ab (a INT) USING parquet",
        false,
        &mut bad,
    );
    expect(&ana, "INSERT INTO ab VALUES (1)", false, &mut bad);
    expect(
        &ana,
        &format!("SELECT * FROM parquet.`{}`", wh.join("abc").display()),
        true,
        &mut bad,
    );
    expect(
        &ana,
        &format!("SELECT * FROM parquet.`{}/ab/../abc`", wh.display()),
        true,
        &mut bad,
    );
    expect(
        &ana,
        &format!("SELECT * FROM parquet.`{}`", wh.join("ab").display()),
        false,
        &mut bad,
    );
    // A column grant does not open the directory.
    table(&w, "t");
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    expect(
        &ana,
        &format!("SELECT * FROM parquet.`{}`", wh.join("t").display()),
        true,
        &mut bad,
    );
    // An admin's external table inside ana's managed directory.
    let sub = wh.join("ab").join("sub");
    show(
        &root,
        &format!(
            "CREATE TABLE vic (a INT) USING parquet LOCATION '{}'",
            sql_path(&sub)
        ),
    );
    show(&root, "INSERT INTO vic VALUES (7)");
    eprintln!(
        "RECHECK2 q3 ana reads vic's dir: {}",
        rows_of(&ana, &format!("SELECT * FROM parquet.`{}`", sub.display()))
    );
    // r2-m1: the external table's location is not ana's, though inside her directory.
    denied(&ana, &format!("SELECT * FROM parquet.`{}`", sub.display()));
    denied(
        &ana,
        &format!(
            "SELECT * FROM parquet.`{}`",
            sub.join("..").join("sub").display()
        ),
    );
    // Her own directory still reads.
    expect(
        &ana,
        &format!("SELECT * FROM parquet.`{}`", wh.join("ab").display()),
        false,
        &mut bad,
    );
    // ana renames onto a dropped table's name.
    ok(&root, "CREATE TABLE gone (a INT) USING parquet");
    ok(&root, "INSERT INTO gone VALUES (5)");
    ok(&root, "GRANT SELECT ON TABLE gone TO user:bob");
    ok(&root, "DROP TABLE gone");
    expect(
        &ana,
        "CREATE TABLE mine (a INT) USING parquet",
        false,
        &mut bad,
    );
    expect(&ana, "ALTER TABLE mine RENAME TO gone", false, &mut bad);
    let bob = w.user("bob");
    expect(&bob, "SELECT * FROM gone", true, &mut bad);
    expect(&ana, "INSERT INTO gone VALUES (3)", false, &mut bad);
    expect(&ana, "DROP TABLE gone", false, &mut bad);
    assert!(bad.is_empty(), "{bad:#?}");
}
