// SPDX-License-Identifier: Apache-2.0
//! P3-T062 unit G3 (design `P3-T062-access-policy-providers.md` §2.4) under the `grants` provider:
//! `SQL SECURITY DEFINER` views read with their definer's rights (the reader needs `Select` on the
//! view only), `INVOKER` views (the default) with the reader's; calling a stored SQL function needs
//! `Select` on it and its body reads as the caller's; a materialized-view refresh runs as the
//! owner acting for the scheduler; listings show only what the principal may see; an `EXPLAIN`
//! asks what the statement it explains reads; and a cached plan, `CACHE TABLE` and the result
//! cache are all refused after a `REVOKE`.
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

struct World {
    rt: Arc<EngineRuntime>,
    grants: Arc<Grants>,
    _dir: tempfile::TempDir,
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
            _dir: dir,
        }
    }

    fn session(&self, p: Principal, admin: bool) -> Session {
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
        self.session(Principal::new(name, PrincipalKind::User), false)
    }

    fn admin(&self) -> Session {
        self.session(Principal::new("root", PrincipalKind::User), true)
    }
}

fn run(s: &Session, sql: &str) -> Result<String, KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        Ok(format!("{batches:?}"))
    })
}

fn rows(s: &Session, sql: &str) -> usize {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        Ok::<_, KrishivError>(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
    })
    .unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn ok(s: &Session, sql: &str) -> String {
    run(s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn denied(s: &Session, sql: &str) -> String {
    match run(s, sql) {
        Err(e) if e.error_class().name() == DENIED => e.to_string(),
        other => panic!("{sql}: expected a denial, got {other:?}"),
    }
}

/// A refusal of a definer's read as its reader sees it (review B2): the class and SQLSTATE, every
/// parameter redacted, none of `hidden` shown.
fn assert_redacted(e: &str, hidden: &[&str]) {
    assert!(
        e.contains(krishiv_common::secret::REDACTED) && e.contains("42501"),
        "{e}"
    );
    for h in hidden {
        assert!(!e.contains(h), "{h}: {e}");
    }
}

/// root's `t (a INT, b INT)` with the row (1, 2); ana may create in `default` and read `t`.
fn world() -> (World, Session, Session) {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "CREATE TABLE t (a INT, b INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES (1, 2)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    (w, root, ana)
}

#[test]
fn a_definer_view_reads_with_its_definers_rights() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(&ana, "CREATE VIEW iv AS SELECT a FROM t");
    // The view's owner (its creator) grants it on.
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    ok(&ana, "GRANT SELECT ON VIEW iv TO user:bob");
    // Definer: the view alone; invoker (the default): the view and its base table.
    assert_eq!(rows(&bob, "SELECT * FROM dv"), 1);
    assert_eq!(rows(&bob, "SELECT a + 1 FROM dv WHERE a = 1"), 1);
    assert_eq!(block_on(bob.table("dv").collect()).unwrap().len(), 1);
    denied(&bob, "SELECT * FROM iv");
    // The view never opens its base table to the reader's own reads, whatever shape mixes them.
    for sql in [
        "SELECT * FROM t",
        "SELECT * FROM dv, t",
        "SELECT * FROM t, dv",
        "SELECT * FROM dv WHERE a IN (SELECT a FROM t)",
        "SELECT a FROM dv UNION ALL SELECT a FROM default.t",
        "SELECT * FROM dv JOIN spark_catalog.default.t USING (a)",
        "WITH x AS (SELECT * FROM t) SELECT * FROM dv, x",
    ] {
        denied(&bob, sql);
    }
    // Without a grant on the view, nothing; the refusal names the view only.
    let carl = w.user("carl");
    let e = denied(&carl, "SELECT * FROM dv");
    assert!(e.contains("`dv`") && !e.contains("`t`"), "{e}");
    // The definer's rights are asked on every read: a revoke from the definer reaches its readers.
    // The refusal is redacted (review B2): bob wrote neither the definer nor its table.
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    let e = denied(&bob, "SELECT * FROM dv");
    assert_redacted(&e, &["`ana`", "`t`", "default.t"]);
    // Column grants of the definer: what the view's query reads.
    ok(&root, "GRANT SELECT (a) ON t TO user:ana");
    assert_eq!(rows(&bob, "SELECT * FROM dv"), 1);
    // The same holds for the DataFrame API.
    assert_eq!(
        block_on(bob.table("dv").select(["a"]).collect())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn a_definer_holds_only_its_own_grants_never_an_admin_role() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    // root owns `t`: its definer view reads `t` through ownership.
    ok(
        &root,
        "CREATE VIEW over_t SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(&root, "GRANT SELECT ON VIEW over_t TO user:bob");
    assert_eq!(rows(&bob, "SELECT * FROM over_t"), 1);
    // ana's table, which root may read only as an admin: root's definer view over it reads with
    // root's grants, and an admin role is not one.
    ok(&ana, "CREATE TABLE anas (a INT) USING parquet");
    ok(&ana, "INSERT INTO anas VALUES (3)");
    assert_eq!(rows(&root, "SELECT * FROM anas"), 1);
    ok(
        &root,
        "CREATE VIEW over_anas SQL SECURITY DEFINER AS SELECT a FROM anas",
    );
    ok(&root, "GRANT SELECT ON VIEW over_anas TO user:bob");
    let e = denied(&bob, "SELECT * FROM over_anas");
    assert_redacted(&e, &["`root`", "anas`"]);
    // Even root reading its own definer view: the view's query is the definer's, acting for root.
    denied(&root, "SELECT * FROM over_anas");
    ok(&ana, "GRANT SELECT ON TABLE anas TO user:root");
    assert_eq!(rows(&bob, "SELECT * FROM over_anas"), 1);
}

#[test]
fn a_definer_view_inside_a_definer_view_reads_as_its_own_definer() {
    let (w, root, ana) = world();
    let (bob, carl) = (w.user("bob"), w.user("carl"));
    // ana's view over t; bob's view over ana's view: bob needs only SELECT on ana's view.
    ok(
        &ana,
        "CREATE VIEW inner_v SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(&ana, "GRANT SELECT ON VIEW inner_v TO user:bob");
    ok(
        &bob,
        "CREATE VIEW outer_v SQL SECURITY DEFINER AS SELECT a FROM inner_v",
    );
    ok(&bob, "GRANT SELECT ON VIEW outer_v TO user:carl");
    assert_eq!(rows(&carl, "SELECT * FROM outer_v"), 1);
    // carl may not read the inner view itself.
    denied(&carl, "SELECT * FROM inner_v");
    // The outer definer losing the inner view refuses carl.
    ok(&ana, "REVOKE SELECT ON VIEW inner_v FROM user:bob");
    let e = denied(&carl, "SELECT * FROM outer_v");
    assert_redacted(&e, &["`bob`", "inner_v"]);
}

#[test]
fn replacing_a_definer_views_query_reads_as_whoever_replaced_it() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(&root, "CREATE TABLE secret (a INT) USING parquet");
    ok(&root, "INSERT INTO secret VALUES (9)");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    // bob may replace the view's query (DROP on it, CREATE in its namespace) but read nothing:
    // the new text reads with bob's rights, never ana's.
    ok(&root, "GRANT DROP ON VIEW dv TO user:bob");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(&root, "GRANT SELECT ON VIEW dv TO user:bob");
    ok(&root, "GRANT SELECT ON TABLE secret TO user:ana");
    denied(&bob, "ALTER VIEW dv AS SELECT a FROM secret");
    // As an admin replaces it, the admin is its definer.
    ok(&root, "ALTER VIEW dv AS SELECT a FROM secret");
    assert_eq!(rows(&root, "SELECT * FROM dv"), 1);
}

#[test]
fn explaining_a_definer_view_asks_what_its_query_reads() {
    let (w, _root, ana) = world();
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    assert_eq!(rows(&bob, "SELECT * FROM dv"), 1);
    // The plan shows the view's query: bob is asked its reads as an invoker would be.
    for sql in [
        "EXPLAIN SELECT * FROM dv",
        "EXPLAIN EXTENDED SELECT * FROM dv",
    ] {
        match run(&bob, sql) {
            Err(e) => assert_eq!(e.error_class().name(), DENIED, "{sql}"),
            // The refusal, or (a metadata explain, P3 PR-14d) the planning failure with its
            // message redacted: never a plan.
            Ok(text) => {
                assert!(
                    text.contains(DENIED) || text.contains("Error occurred during query planning"),
                    "{sql}: {text}"
                );
                assert!(!text.contains("Scan"), "{sql}: {text}");
            }
        }
    }
    for e in [
        block_on(bob.table("dv").estimate()).map(|_| ()),
        block_on(bob.table("dv").warnings()).map(|_| ()),
    ] {
        assert_eq!(e.unwrap_err().error_class().name(), DENIED);
    }
    // ana, who may read the base, sees the plan.
    let text = ok(&ana, "EXPLAIN SELECT * FROM dv");
    assert!(!text.contains(DENIED), "{text}");
}

#[test]
fn a_stored_function_needs_select_and_its_body_reads_as_the_caller() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    // The creator owns its function and calls it.
    ok(
        &ana,
        "CREATE FUNCTION fa() RETURNS INT RETURN (SELECT max(a) FROM t)",
    );
    assert_eq!(rows(&ana, "SELECT fa()"), 1);
    // Another principal needs SELECT on the function, then on what its body reads (invoker).
    denied(&bob, "SELECT fa()");
    ok(&ana, "GRANT SELECT ON FUNCTION fa TO user:bob");
    let e = denied(&bob, "SELECT fa()");
    assert!(e.contains("`t`"), "{e}");
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    assert_eq!(rows(&bob, "SELECT default.fa()"), 1);
    // A function body over no table still needs the function grant.
    ok(&ana, "CREATE FUNCTION one() RETURNS INT RETURN 1");
    let carl = w.user("carl");
    denied(&carl, "SELECT one()");
    denied(&carl, "SELECT spark_catalog.default.one()");
    ok(&root, "GRANT SELECT ON SCHEMA default TO user:carl");
    assert_eq!(rows(&carl, "SELECT one()"), 1);
    // Built-in and temporary functions are not securables.
    assert_eq!(rows(&w.user("dan"), "SELECT abs(-1)"), 1);
    let dan = w.user("dan");
    ok(
        &dan,
        "CREATE TEMPORARY FUNCTION mine() RETURNS INT RETURN 2",
    );
    assert_eq!(rows(&dan, "SELECT mine()"), 1);
    // Creating a stored function needs CREATE on the namespace.
    denied(&dan, "CREATE FUNCTION nope() RETURNS INT RETURN 1");
}

#[test]
fn a_refresh_runs_as_the_owner_for_the_scheduler() {
    // No materialized view refresh path exists yet: the hook is the principal it will run as
    // (`Principal::definer(owner, "scheduler")`) in a session over the same grants. It holds
    // the owner's grants and never the role of the session that starts it.
    let (w, root, ana) = world();
    ok(&ana, "CREATE TABLE mv_target (a INT) USING parquet");
    let refresh = w.session(Principal::definer("ana", "scheduler"), true);
    ok(&refresh, "INSERT INTO mv_target SELECT a FROM t");
    assert_eq!(rows(&ana, "SELECT * FROM mv_target"), 1);
    // The owner's rights, not an admin's: root's table that ana cannot read stays closed.
    ok(&root, "CREATE TABLE closed (a INT) USING parquet");
    denied(&refresh, "INSERT INTO mv_target SELECT a FROM closed");
    // Revoking from the owner reaches the next refresh.
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    denied(&refresh, "INSERT INTO mv_target SELECT a FROM t");
}

#[test]
fn listings_show_only_what_the_principal_may_see() {
    let (w, root, ana) = world();
    // ana may list the catalog and create in `default`: neither shows every object in them.
    ok(&root, "GRANT USAGE ON CATALOG spark_catalog TO user:ana");
    ok(&root, "CREATE TABLE hidden (a INT) USING parquet");
    ok(&root, "CREATE VIEW hidden_v AS SELECT a FROM hidden");
    ok(&root, "CREATE NAMESPACE vaultns");
    ok(&root, "CREATE TABLE vaultns.inside (a INT) USING parquet");
    ok(&root, "CREATE NAMESPACE shown");
    ok(&root, "CREATE TABLE shown.mine (a INT) USING parquet");
    ok(&root, "GRANT SELECT ON TABLE shown.mine TO user:ana");
    ok(&root, "CREATE FUNCTION hidden_f() RETURNS INT RETURN 1");
    ok(&ana, "CREATE VIEW own_v AS SELECT a FROM t");
    let tables = ok(&ana, "SHOW TABLES");
    assert!(
        tables.contains("\"t\"") && tables.contains("own_v"),
        "{tables}"
    );
    let namespaces = ok(&ana, "SHOW NAMESPACES");
    // A namespace shows when the principal holds a grant on it or may see something in it.
    assert!(
        namespaces.contains("default") && namespaces.contains("shown"),
        "{namespaces}"
    );
    for listing in [tables, ok(&ana, "SHOW VIEWS"), namespaces] {
        assert!(
            !listing.contains("hidden") && !listing.contains("vaultns"),
            "{listing}"
        );
    }
    let c = ana.catalog();
    let tables: Vec<String> = block_on(c.list_tables(None, None))
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert!(
        tables.contains(&"t".to_owned()) && !tables.iter().any(|t| t.contains("hidden")),
        "{tables:?}"
    );
    let dbs: Vec<String> = block_on(c.list_databases(None))
        .unwrap()
        .into_iter()
        .map(|d| d.name)
        .collect();
    assert!(
        !dbs.contains(&"vaultns".to_owned()) && dbs.contains(&"shown".to_owned()),
        "{dbs:?}"
    );
    let fns: Vec<String> = block_on(c.list_functions(None))
        .unwrap()
        .into_iter()
        .map(|f| f.name)
        .collect();
    assert!(
        !fns.contains(&"hidden_f".to_owned()) && fns.contains(&"abs".to_owned()),
        "{fns:?}"
    );
    // An admin sees everything.
    let all = ok(&root, "SHOW TABLES");
    assert!(all.contains("hidden"), "{all}");
    // Metadata of one named object is asked as before (`Browse`: any grant on it or a parent).
    let bob = w.user("bob");
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    for sql in [
        "DESCRIBE hidden",
        "SHOW COLUMNS IN hidden",
        "SHOW CREATE TABLE hidden",
        "DESCRIBE hidden_v",
    ] {
        denied(&bob, sql);
    }
}

#[test]
fn caches_are_refused_after_a_revoke() {
    let (_w, root, ana) = world();
    ana.conf()
        .set("krishiv.resultCache.enabled", "true")
        .unwrap();
    let df = block_on(ana.sql("SELECT a FROM t")).unwrap();
    assert_eq!(block_on(df.collect()).unwrap()[0].num_rows(), 1);
    let cached_df = ana.table("t").cache();
    assert_eq!(block_on(cached_df.collect()).unwrap()[0].num_rows(), 1);
    ok(&ana, "CACHE TABLE t");
    ok(&ana, "CACHE TABLE ct AS SELECT a FROM t");
    // The result cache holds this one now.
    assert_eq!(rows(&ana, "SELECT a FROM t WHERE a = 1"), 1);
    assert_eq!(rows(&ana, "SELECT a FROM t WHERE a = 1"), 1);
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    // The analysed plan, the cached frame, the cached table, a view cached over it and the result
    // cache: each is checked on use, never at caching time.
    for e in [
        block_on(df.collect()).map(|_| ()),
        block_on(cached_df.collect()).map(|_| ()),
    ] {
        assert_eq!(e.unwrap_err().error_class().name(), DENIED);
    }
    for sql in [
        "SELECT * FROM t",
        "SELECT a FROM t WHERE a = 1",
        "SELECT * FROM ct",
        "CACHE TABLE t",
    ] {
        denied(&ana, sql);
    }
}

/// P3-T062 review B1: the mark that a read is a definer view's own is node identity, never a
/// number. Bind numbers the SQL texts a caller embeds in its plan (Spark Connect's
/// `Relation.sql`) from 0, and the session numbers the view texts it lowers from 1: a caller's
/// embedded text whose number equals a definer view's must still be asked of the caller.
#[cfg(feature = "unstable-ir")]
#[test]
fn embedded_sql_never_borrows_a_definer_views_rights() {
    use krishiv_plan::{LogicalPlan, Plan, PlanNode, SqlArgs};
    let (w, root, ana) = world();
    ok(&root, "CREATE TABLE secret (a INT) USING parquet");
    ok(&root, "INSERT INTO secret VALUES (9)");
    ok(&root, "GRANT SELECT ON TABLE secret TO user:ana");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    let sql = |query: &str| {
        Arc::new(LogicalPlan::new(PlanNode::Sql {
            query: query.to_owned(),
            args: SqlArgs::None,
        }))
    };
    // The view beside embedded texts numbered 0, 1, ..., n (Bind numbers them in plan order):
    // the last reads the secret, the others nothing. For some n its number is the view text's.
    for n in 0..8 {
        let mut inputs = vec![Arc::new(LogicalPlan::table(["dv"]))];
        inputs.extend((0..n).map(|_| sql("SELECT 1 AS a")));
        inputs.push(sql("SELECT a FROM secret"));
        let plan = Plan::try_new(Arc::new(LogicalPlan::new(PlanNode::Union {
            inputs,
            by_name: false,
            allow_missing_columns: false,
        })))
        .unwrap();
        // A fresh session: its view texts are numbered from 1.
        let bob = w.user("bob");
        match block_on(bob.from_plan(&plan).collect()) {
            Err(e) => assert_eq!(e.error_class().name(), DENIED, "{n}: {e}"),
            Ok(b) => panic!("{n}: read the secret: {b:?}"),
        }
    }
    // The view alone is bob's to read.
    let bob = w.user("bob");
    let alone = Plan::try_new(Arc::new(LogicalPlan::table(["dv"]))).unwrap();
    assert_eq!(block_on(bob.from_plan(&alone).collect()).unwrap().len(), 1);
}

/// Review B2 (design §6 canary rule): what goes wrong inside a definer's read tells the reader
/// nothing it did not write. carl may read bob's `outer_v` only; bob's view reads ana's
/// `inner_v`, ana's view reads `vault_base`. Neither the definers, nor `inner_v`, nor its base
/// table, nor its text show in carl's error, whether the definer lost its grant or the text no
/// longer analyzes; the audit record still names the definer.
#[test]
fn a_definer_failure_shows_the_reader_only_what_it_wrote() {
    let (w, root, ana) = world();
    let (bob, carl) = (w.user("bob"), w.user("carl"));
    ok(
        &root,
        "CREATE TABLE vault_base (secret_col INT) USING parquet",
    );
    ok(&root, "INSERT INTO vault_base VALUES (424242)");
    ok(&root, "GRANT SELECT ON TABLE vault_base TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(
        &ana,
        "CREATE VIEW inner_v SQL SECURITY DEFINER AS SELECT secret_col AS a FROM vault_base WHERE secret_col > 7",
    );
    ok(&ana, "GRANT SELECT ON VIEW inner_v TO user:bob");
    ok(
        &bob,
        "CREATE VIEW outer_v SQL SECURITY DEFINER AS SELECT a FROM inner_v",
    );
    ok(&bob, "GRANT SELECT ON VIEW outer_v TO user:carl");
    assert_eq!(rows(&carl, "SELECT * FROM outer_v"), 1);
    let canary = [
        "`ana`",
        "`bob`",
        "inner_v",
        "vault_base",
        "secret_col",
        "424242",
        "> 7",
    ];
    let shown = |e: &KrishivError| {
        let mut text = e.to_string();
        for c in e.query_contexts() {
            text.push_str(&format!("{c:?}"));
        }
        text
    };
    // The definer of the inner view loses its table.
    ok(&root, "REVOKE SELECT ON vault_base FROM user:ana");
    let e = run(&carl, "SELECT * FROM outer_v").unwrap_err();
    assert_eq!(e.error_class().name(), DENIED);
    assert_eq!(e.sql_state(), Some("42501"));
    assert_redacted(&shown(&e), &canary);
    // The inner view's text no longer analyzes (its column is gone).
    ok(&root, "GRANT SELECT ON vault_base TO user:ana");
    assert_eq!(rows(&carl, "SELECT * FROM outer_v"), 1);
    ok(&root, "DROP TABLE vault_base");
    ok(&root, "CREATE TABLE vault_base (other INT) USING parquet");
    ok(&root, "GRANT SELECT ON TABLE vault_base TO user:ana");
    let e = run(&carl, "SELECT * FROM outer_v").unwrap_err();
    let text = shown(&e);
    for c in canary {
        assert!(!text.contains(c), "{c}: {text}");
    }
    // An outer view whose own text fails is the outer definer's too.
    let e = run(&carl, "SELECT a + 1 FROM outer_v").unwrap_err();
    let text = shown(&e);
    for c in canary {
        assert!(!text.contains(c), "{c}: {text}");
    }
}

/// Review M3: a revoke of the *definer's* grant reaches every cache a reader holds over its
/// definer view: an analysed frame, a cached frame, `CACHE TABLE` of the view, a view cached over
/// it, the result cache and the analysis prefix cache (the same text analysed again).
#[test]
fn caches_over_a_definer_view_are_refused_after_the_definers_revoke() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    bob.conf()
        .set("krishiv.resultCache.enabled", "true")
        .unwrap();
    let df = block_on(bob.sql("SELECT a FROM dv")).unwrap();
    assert_eq!(block_on(df.collect()).unwrap()[0].num_rows(), 1);
    let cached_df = bob.table("dv").cache();
    assert_eq!(block_on(cached_df.collect()).unwrap()[0].num_rows(), 1);
    ok(&bob, "CACHE TABLE dv");
    ok(&bob, "CACHE TABLE cdv AS SELECT a FROM dv");
    assert_eq!(rows(&bob, "SELECT a FROM dv WHERE a = 1"), 1);
    assert_eq!(rows(&bob, "SELECT a FROM dv WHERE a = 1"), 1);
    ok(&root, "REVOKE SELECT ON t FROM user:ana");
    for e in [
        block_on(df.collect()).map(|_| ()),
        block_on(cached_df.collect()).map(|_| ()),
    ] {
        assert_eq!(e.unwrap_err().error_class().name(), DENIED);
    }
    for sql in [
        "SELECT a FROM dv",
        "SELECT a FROM dv WHERE a = 1",
        "SELECT * FROM cdv",
        "CACHE TABLE dv",
    ] {
        denied(&bob, sql);
    }
    // Granted back, the same text reads again.
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    assert_eq!(rows(&bob, "SELECT a FROM dv WHERE a = 1"), 1);
}

/// Review M1: `EXPLAIN` of a definer view shows the bodies of the views its text inlines, so the
/// reader needs `Select` on each of them, as on a view it read itself.
#[test]
fn explaining_a_definer_view_asks_the_views_inside_it() {
    let (w, root, ana) = world();
    let carl = w.user("carl");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW inner_v AS SELECT a FROM t WHERE a < 424242",
    );
    ok(&ana, "GRANT SELECT ON VIEW inner_v TO user:bob");
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    ok(
        &bob,
        "CREATE VIEW outer_v SQL SECURITY DEFINER AS SELECT a FROM inner_v",
    );
    ok(&bob, "GRANT SELECT ON VIEW outer_v TO user:carl");
    // carl may read the base table, not the inner view.
    ok(&root, "GRANT SELECT ON TABLE t TO user:carl");
    assert_eq!(rows(&carl, "SELECT * FROM outer_v"), 1);
    for sql in [
        "EXPLAIN SELECT * FROM outer_v",
        "EXPLAIN EXTENDED SELECT * FROM outer_v",
    ] {
        let shown = match run(&carl, sql) {
            Err(e) => {
                assert_eq!(e.error_class().name(), DENIED, "{sql}");
                e.to_string()
            }
            Ok(text) => text,
        };
        assert!(!shown.contains("424242"), "{sql}: {shown}");
        assert!(!shown.contains("Scan"), "{sql}: {shown}");
    }
    // Granted the inner view, the plan shows.
    ok(&ana, "GRANT SELECT ON VIEW inner_v TO user:carl");
    let text = ok(&carl, "EXPLAIN SELECT * FROM outer_v");
    assert!(text.contains("Scan"), "{text}");
}

/// Review M2: `system.operators` holds a query's operators (names, metrics, row counts): for a
/// query over a definer view whose query the reader may not see as an invoker (the rule `EXPLAIN`
/// follows), none of them are kept.
#[test]
fn operators_of_a_definer_views_plan_stay_with_those_who_may_see_it() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t WHERE b = 2",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    // The operator rows of the queries `SELECT * FROM dv` the principal ran.
    let operators = |s: &Session| {
        rows(
            s,
            "SELECT o.name FROM system.operators o JOIN system.queries q \
             ON o.query_id = q.query_id WHERE q.query_text = 'SELECT * FROM dv'",
        )
    };
    assert_eq!(rows(&bob, "SELECT * FROM dv"), 1);
    assert_eq!(operators(&bob), 0, "bob's operators");
    // Nor the totals that sum the withheld scans (review r2 item 5); the rows it produced stay.
    let totals = "SELECT CAST(rows_produced AS STRING), \
                  CAST(rows_read IS NULL AND bytes_read IS NULL AND files_read IS NULL \
                  AND spill_bytes IS NULL AND spill_files IS NULL AND pipelines IS NULL AS STRING) \
                  FROM system.queries WHERE query_text = 'SELECT * FROM dv'";
    let shown = ok(&bob, totals);
    assert!(
        shown.contains("\"1\"") && shown.contains("\"true\""),
        "{shown}"
    );
    // Once bob may read the base table, his next query's operators are kept.
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    assert_eq!(rows(&bob, "SELECT * FROM dv"), 1);
    assert!(operators(&bob) > 0, "bob's operators");
    assert!(ok(&bob, totals).contains("\"false\""), "bob's totals");
    // ana, the definer, sees her own.
    assert_eq!(rows(&ana, "SELECT * FROM dv"), 1);
    assert!(operators(&ana) > 0);
}

/// Review r2 R1: a schema-binding failure inside a definer view (an up-cast its base column no
/// longer allows, a column its base table lost) reaches the reader as the other definer failures
/// do: its class and SQLSTATE, no definer table or column name, no stored view text.
#[test]
fn a_schema_binding_failure_inside_a_definer_view_is_redacted() {
    let (w, root, ana) = world();
    let (bob, carl) = (w.user("bob"), w.user("carl"));
    ok(
        &root,
        "CREATE TABLE vault_base (secret_col INT) USING parquet",
    );
    ok(&root, "INSERT INTO vault_base VALUES (424242)");
    ok(
        &root,
        "CREATE TABLE vault_two (secret_two INT) USING parquet",
    );
    ok(&root, "GRANT SELECT ON TABLE vault_base TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE vault_two TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(
        &ana,
        "CREATE VIEW inner_v WITH SCHEMA BINDING SQL SECURITY DEFINER AS \
         SELECT * FROM vault_base WHERE secret_col > 7",
    );
    ok(
        &ana,
        "CREATE VIEW inner_two WITH SCHEMA BINDING SQL SECURITY DEFINER AS SELECT * FROM vault_two",
    );
    ok(&ana, "GRANT SELECT ON VIEW inner_v TO user:bob");
    // carl reads this one directly: the outermost view is the definer's.
    ok(&ana, "GRANT SELECT ON VIEW inner_two TO user:carl");
    ok(
        &bob,
        "CREATE VIEW outer_v SQL SECURITY DEFINER AS SELECT * FROM inner_v",
    );
    ok(&bob, "GRANT SELECT ON VIEW outer_v TO user:carl");
    assert_eq!(rows(&carl, "SELECT * FROM outer_v"), 1);
    assert_eq!(rows(&carl, "SELECT * FROM inner_two"), 0);
    let canary = [
        "`ana`",
        "`bob`",
        "inner_v",
        "vault_base",
        "vault_two",
        "secret_col",
        "secret_two",
        "> 7",
    ];
    let shown = |e: &KrishivError| {
        let mut text = e.to_string();
        for c in e.query_contexts() {
            text.push_str(&format!("{c:?}"));
        }
        text
    };
    // The base column widened: the binding's up-cast fails.
    ok(&root, "DROP TABLE vault_base");
    ok(
        &root,
        "CREATE TABLE vault_base (secret_col BIGINT) USING parquet",
    );
    ok(&root, "GRANT SELECT ON TABLE vault_base TO user:ana");
    let e = run(&carl, "SELECT * FROM outer_v").unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_UP_CAST_DATATYPE", "{e}");
    assert_redacted_any(&shown(&e), &canary);
    // The base table lost the column: the binding's schema change is incompatible (its message
    // names the view carl wrote, and would print the view's stored text).
    ok(&root, "DROP TABLE vault_two");
    ok(&root, "CREATE TABLE vault_two (other INT) USING parquet");
    ok(&root, "GRANT SELECT ON TABLE vault_two TO user:ana");
    let e = run(&carl, "SELECT * FROM inner_two").unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INCOMPATIBLE_VIEW_SCHEMA_CHANGE",
        "{e}"
    );
    assert_redacted_any(&shown(&e), &canary);
}

/// [`assert_redacted`] for an error of any class: every parameter redacted, none of `hidden`.
fn assert_redacted_any(e: &str, hidden: &[&str]) {
    assert!(e.contains(krishiv_common::secret::REDACTED), "{e}");
    for h in hidden {
        assert!(!e.contains(h), "{h}: {e}");
    }
}

/// Review r2 R2: a definer view whose text holds a `WITH` reads with its definer's rights under
/// every CTE precedence policy the reader sets: `LEGACY` inlines the definitions before binding,
/// and the inlined query stays the definer's. A reader's own definition of the same name never
/// replaces the view's.
#[test]
fn a_definer_views_cte_reads_as_the_definer_under_every_precedence_policy() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(&root, "CREATE TABLE other (a INT) USING parquet");
    ok(&root, "INSERT INTO other VALUES (7), (8)");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS WITH c AS (SELECT a FROM t) SELECT a FROM c",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    for policy in ["CORRECTED", "LEGACY", "EXCEPTION"] {
        bob.conf()
            .set("spark.sql.legacy.ctePrecedencePolicy", policy)
            .unwrap();
        assert_eq!(rows(&bob, "SELECT * FROM dv"), 1, "{policy}");
        assert_eq!(rows(&bob, "SELECT a FROM dv WHERE a = 1"), 1, "{policy}");
    }
    // Under LEGACY an outer definition wins over a nested one of the same name: never over the
    // view's, which is the definer's text (and bob's `c` would read a table as ana).
    ok(&root, "GRANT SELECT ON TABLE other TO user:bob");
    bob.conf()
        .set("spark.sql.legacy.ctePrecedencePolicy", "LEGACY")
        .unwrap();
    assert_eq!(
        rows(&bob, "WITH c AS (SELECT a FROM other) SELECT * FROM dv"),
        1
    );
    // Nested: bob's definer view with its own `WITH` over ana's, read by carl under LEGACY.
    let carl = w.user("carl");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(
        &bob,
        "CREATE VIEW outer_v SQL SECURITY DEFINER AS WITH c AS (SELECT a FROM dv) SELECT a FROM c",
    );
    ok(&bob, "GRANT SELECT ON VIEW outer_v TO user:carl");
    carl.conf()
        .set("spark.sql.legacy.ctePrecedencePolicy", "LEGACY")
        .unwrap();
    assert_eq!(rows(&carl, "SELECT * FROM outer_v"), 1);
}

/// Review r2 R3: every DataFrame API that returns an executed plan or its operator metrics asks
/// what `EXPLAIN ANALYZE` asks: the reads of a definer view's query, as an invoker would be asked.
#[test]
fn profiles_of_a_definer_views_plan_ask_what_explain_analyze_asks() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t WHERE b = 2 AND a < 424242",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    type Shown = Box<dyn Fn(&Session) -> Result<String, KrishivError>>;
    let entries: Vec<(&str, Shown)> = vec![
        (
            "explain_analyze",
            Box::new(|s| {
                block_on(s.table("dv").explain_analyze())
                    .map(|p| p.to_text(krishiv_session::TextOptions::default()))
            }),
        ),
        (
            "explain_analyze_with",
            Box::new(|s| {
                block_on(
                    s.table("dv")
                        .explain_analyze_with(krishiv_session::GraphFormat::Json),
                )
            }),
        ),
        (
            "explain_analyze_verbose",
            Box::new(|s| block_on(s.table("dv").explain_analyze_verbose()).map(|p| p.to_json())),
        ),
        (
            "collect_with_profile",
            Box::new(|s| block_on(s.table("dv").collect_with_profile()).map(|(_, p)| p.to_json())),
        ),
        (
            "collect_with_metrics",
            Box::new(|s| {
                block_on(s.table("dv").collect_with_metrics()).map(|(_, m)| format!("{m:?}"))
            }),
        ),
    ];
    for (entry, shown) in &entries {
        match shown(&bob) {
            Err(e) if e.error_class().name() == DENIED => {
                assert!(!e.to_string().contains("424242"), "{entry}: {e}");
            }
            other => panic!("{entry}: expected a denial, got {other:?}"),
        }
    }
    // Once bob may read the base table, each shows.
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    for (entry, shown) in &entries {
        shown(&bob).unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
}

/// Review r2 item 6 (r2 M-b): a refusal to show a definer view's plan names only what the reader
/// wrote: its class and SQLSTATE, not the definer's base table, column or the views inside it.
#[test]
fn a_refused_explain_of_a_definer_view_names_only_what_the_reader_wrote() {
    let (w, root, ana) = world();
    let (bob, carl) = (w.user("bob"), w.user("carl"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(
        &ana,
        "CREATE VIEW inner_v AS SELECT a FROM t WHERE a < 424242",
    );
    ok(&ana, "GRANT SELECT ON VIEW inner_v TO user:bob");
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    ok(
        &bob,
        "CREATE VIEW outer_v SQL SECURITY DEFINER AS SELECT a FROM inner_v",
    );
    ok(&bob, "GRANT SELECT ON VIEW outer_v TO user:carl");
    let hidden = ["inner_v", "`t`", "`a`", "`ana`", "`bob`", "424242"];
    let shown: Vec<(&str, Result<String, KrishivError>)> = vec![
        ("EXPLAIN", run(&carl, "EXPLAIN SELECT * FROM outer_v")),
        (
            "EXPLAIN EXTENDED",
            run(&carl, "EXPLAIN EXTENDED SELECT * FROM outer_v"),
        ),
        (
            "EXPLAIN ANALYZE",
            run(&carl, "EXPLAIN ANALYZE SELECT * FROM outer_v"),
        ),
        (
            "explain_analyze",
            block_on(carl.table("outer_v").explain_analyze()).map(|p| format!("{p:?}")),
        ),
        (
            "explain_verbose",
            block_on(carl.table("outer_v").explain_verbose()).map(|p| format!("{p:?}")),
        ),
    ];
    for (entry, r) in shown {
        match r {
            Err(e) => {
                assert_eq!(e.error_class().name(), DENIED, "{entry}: {e}");
                assert_redacted(&e.to_string(), &hidden);
            }
            Ok(text) => {
                for h in hidden {
                    assert!(!text.contains(h), "{entry}: {h}: {text}");
                }
            }
        }
    }
}

/// Review r3 N4 (ADR 0048 A1 §3 Listings): a namespace shows to a principal holding a grant on it
/// or on anything in it, functions included, so a grant on one function shows its namespace.
#[test]
fn a_function_grant_shows_its_namespace() {
    let (w, root, _) = world();
    let fan = w.user("fan");
    ok(&root, "GRANT USAGE ON CATALOG spark_catalog TO user:fan");
    ok(&root, "CREATE NAMESPACE fnns");
    ok(&root, "CREATE FUNCTION fnns.g() RETURNS INT RETURN 7");
    ok(&root, "CREATE NAMESPACE closedns");
    ok(&root, "CREATE FUNCTION closedns.h() RETURNS INT RETURN 8");
    ok(&root, "GRANT SELECT ON FUNCTION fnns.g TO user:fan");
    let namespaces = ok(&fan, "SHOW NAMESPACES");
    assert!(
        namespaces.contains("fnns") && !namespaces.contains("closedns"),
        "{namespaces}"
    );
    let c = fan.catalog();
    assert!(block_on(c.database_exists("fnns")).unwrap());
    assert!(!block_on(c.database_exists("closedns")).unwrap());
    block_on(c.get_database("fnns")).unwrap();
    ok(&fan, "DESCRIBE NAMESPACE fnns");
    block_on(c.get_function("fnns.g")).unwrap();
    assert_eq!(rows(&fan, "SELECT fnns.g()"), 1);
}

/// Review r3 X2: the analysis accessors hand out a plan, so they ask what `EXPLAIN` asks before
/// returning one that holds a definer view's inlined query the reader may not read as invoker;
/// the schema, which shows no plan, is still the reader's.
#[cfg(feature = "unstable-ir")]
#[test]
fn the_analysis_accessors_ask_what_explain_asks() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(
        &ana,
        "CREATE VIEW dv SQL SECURITY DEFINER AS SELECT a FROM t WHERE b = 2 AND a < 424242",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv TO user:bob");
    type Shown = Box<dyn Fn(&Session) -> Result<String, KrishivError>>;
    let entries: Vec<(&str, Shown)> = vec![
        (
            "analyzed",
            Box::new(|s| block_on(s.table("dv").analyzed()).map(|a| format!("{a:?}"))),
        ),
        (
            "analyzed_for_metadata",
            Box::new(|s| block_on(s.table("dv").analyzed_for_metadata()).map(|a| format!("{a:?}"))),
        ),
    ];
    for (entry, shown) in &entries {
        match shown(&bob) {
            Err(e) if e.error_class().name() == DENIED => {
                assert_redacted(&e.to_string(), &["424242", "`t`", "`b`", "`ana`"]);
            }
            other => panic!("{entry}: expected a denial, got {other:?}"),
        }
    }
    block_on(bob.table("dv").schema_for_metadata()).unwrap();
    block_on(bob.table("dv").schema()).unwrap();
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    for (entry, shown) in &entries {
        let text = shown(&bob).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert!(text.contains("424242"), "{entry}: {text}");
    }
}

/// Review r3 M1 (ADR 0048 §3 Denials, A1 §3): a refused read names a column only when the
/// statement wrote it. `*`, `* EXCEPT`, `table()`, `isCached` and a view over the table name the
/// table only, so a principal cannot list the columns of a table it may not read, however many
/// it excludes.
#[test]
fn a_refused_read_names_only_the_columns_the_statement_wrote() {
    let w = World::new();
    let root = w.admin();
    let ana = w.user("ana");
    ok(
        &root,
        "CREATE TABLE ct (x INT, secret_a INT, secret_b INT) USING parquet",
    );
    ok(&root, "INSERT INTO ct VALUES (1, 2, 3)");
    ok(&root, "GRANT SELECT (x) ON ct TO user:ana");
    ok(&root, "CREATE VIEW cv AS SELECT * FROM ct");
    ok(&root, "GRANT SELECT ON VIEW cv TO user:ana");
    let hidden = ["secret_a", "secret_b"];
    type Read = Box<dyn Fn(&Session) -> Result<String, KrishivError>>;
    let reads: Vec<(&str, Read)> = vec![
        ("SELECT *", Box::new(|s| run(s, "SELECT * FROM ct"))),
        (
            "SELECT * EXCEPT (x)",
            Box::new(|s| run(s, "SELECT * EXCEPT (x) FROM ct")),
        ),
        (
            "SELECT * EXCEPT (secret_a)",
            Box::new(|s| run(s, "SELECT * EXCEPT (secret_a) FROM ct")),
        ),
        ("SELECT ct.*", Box::new(|s| run(s, "SELECT ct.* FROM ct"))),
        ("view", Box::new(|s| run(s, "SELECT * FROM cv"))),
        (
            "table()",
            Box::new(|s| block_on(s.table("ct").collect()).map(|b| format!("{b:?}"))),
        ),
        (
            "isCached",
            Box::new(|s| block_on(s.catalog().is_cached("ct")).map(|b| format!("{b:?}"))),
        ),
    ];
    for (entry, read) in &reads {
        let shown = match read(&ana) {
            Err(e) if e.error_class().name() == DENIED => e.to_string(),
            Ok(text) => text,
            Err(e) => panic!("{entry}: {e}"),
        };
        // `* EXCEPT (secret_a)` wrote `secret_a`: only `secret_b` must stay unnamed there.
        for h in hidden {
            if !(entry.contains(h)) {
                assert!(!shown.contains(h), "{entry}: {h}: {shown}");
            }
        }
    }
    // A column the statement wrote is named.
    let e = denied(&ana, "SELECT secret_b FROM ct");
    assert!(e.contains("`secret_b`"), "{e}");
    let e = denied(&ana, "SELECT x FROM ct WHERE secret_a = 2");
    assert!(e.contains("`secret_a`"), "{e}");
    assert_eq!(rows(&ana, "SELECT x FROM ct"), 1);
}

/// Review r3 M2: under `LEGACY` CTE precedence an *invoker* view's own `WITH` is its own too, so
/// a reader's outer definition of the same name never replaces it (Spark answers 1, not 5).
///
/// Spark v4.2.0: a stored view is resolved on its own, `Analyzer.scala` lines 1084-1089
/// (`resolveViews`: `ViewResolution.resolve(view, .., resolveChild = executeSameContext, ..)`)
/// and `ViewResolution.scala` lines 35-53 (the view's parsed text, `view.child`, is run through
/// the whole rule batch inside its own `AnalysisContext`), so `CTESubstitution` substitutes the
/// view text's `WITH` within that text alone; the reader's `LEGACY` pass
/// (`CTESubstitution.scala` lines 97-102 and 164-178, `legacyTraverseAndSubstituteCTE`) only
/// sees the unresolved relation `iv`, never the view's text.
#[test]
fn a_views_own_cte_is_its_own_under_legacy_precedence() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    ok(
        &ana,
        "CREATE VIEW iv AS WITH c AS (SELECT a FROM t) SELECT a FROM c",
    );
    ok(&ana, "GRANT SELECT ON VIEW iv TO user:bob");
    for policy in ["CORRECTED", "LEGACY"] {
        bob.conf()
            .set("spark.sql.legacy.ctePrecedencePolicy", policy)
            .unwrap();
        let read = "WITH c AS (SELECT 5 AS a), d AS (SELECT * FROM iv) SELECT * FROM d";
        assert_eq!(rows(&bob, &format!("{read} WHERE a = 1")), 1, "{policy}");
        assert_eq!(rows(&bob, &format!("{read} WHERE a = 5")), 0, "{policy}");
        assert_eq!(
            rows(
                &bob,
                "WITH c AS (SELECT 5 AS a) SELECT * FROM iv WHERE a = 1"
            ),
            1,
            "{policy}"
        );
    }
}

/// The first value of the first row `sql` returns, printed.
fn first(s: &Session, sql: &str) -> Result<String, KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let b = batches
            .iter()
            .find(|b| b.num_rows() > 0)
            .unwrap_or_else(|| panic!("{sql}: no row"));
        Ok(arrow_cast::display::array_value_to_string(b.column(0), 0).unwrap())
    })
}

/// Review r4 Y1: a stored view's text is analyzed under the CTE precedence policy captured when
/// the view was created (CORRECTED when none was), never the reader's. `dvc`'s nested definition
/// `e` reads the outer `c`: under the reader's `LEGACY` the view's `c` went dead and the bare read
/// `c` bound in the reader's namespace with the definer's rights (55 from `default.c`, 77 from
/// `ns2.c`, neither of them readable by bob).
///
/// Spark v4.2.0: a view's text is resolved under `View.effectiveSQLConf(view.desc.viewSQLConfigs,
/// ..)` (`ViewResolution.scala` lines 46-52; `basicLogicalOperators.scala` lines 920-939: a new
/// `SQLConf` holding only the captured configs and those `retainResolutionConfigsForAnalysis`
/// keeps, the binding-policy `SESSION` ones, which `spark.sql.legacy.ctePrecedencePolicy`,
/// `SQLConf.scala` lines 5948-5956, is not). The captured configs are the session's modified
/// ones when the view was created (`CapturesConfig.scala` lines 66-78, stored by `views.scala`
/// line 557 under `view.sqlConfig.`, read back by `CatalogTable.viewSQLConfigs`,
/// `interface.scala` lines 537-545).
#[test]
fn a_stored_views_text_reads_under_the_cte_policy_it_was_created_with() {
    let (w, root, ana) = world();
    let bob = w.user("bob");
    ok(&root, "CREATE TABLE c (a INT) USING parquet");
    ok(&root, "INSERT INTO c VALUES (55)");
    ok(&root, "GRANT SELECT ON TABLE c TO user:ana");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "CREATE TABLE ns2.c (a INT) USING parquet");
    ok(&root, "INSERT INTO ns2.c VALUES (77)");
    ok(&root, "CREATE TABLE ns2.pub (a INT) USING parquet");
    ok(&root, "GRANT SELECT ON TABLE ns2.c TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE ns2.pub TO user:bob");
    ok(&root, "GRANT USAGE ON CATALOG spark_catalog TO user:bob");
    let text = "WITH c AS (SELECT a FROM t), \
                d AS (WITH e AS (SELECT * FROM c) SELECT * FROM e) SELECT * FROM d";
    ok(
        &ana,
        &format!("CREATE VIEW dvc SQL SECURITY DEFINER AS {text}"),
    );
    ok(&ana, &format!("CREATE VIEW ivc AS {text}"));
    ok(&ana, "GRANT SELECT ON VIEW dvc TO user:bob");
    // Created under LEGACY: the nested main query `SELECT * FROM c` reads the outer `c` (1);
    // under CORRECTED it would read the inner one (5).
    let shadowing = "WITH c AS (SELECT a FROM t), \
                     d AS (WITH c AS (SELECT 5 AS a) SELECT * FROM c) SELECT * FROM d";
    ok(&ana, &format!("CREATE VIEW ivk AS {shadowing}"));
    ana.conf()
        .set("spark.sql.legacy.ctePrecedencePolicy", "LEGACY")
        .unwrap();
    ok(&ana, &format!("CREATE VIEW ivl AS {shadowing}"));
    for policy in ["CORRECTED", "LEGACY", "EXCEPTION"] {
        for s in [&ana, &bob] {
            s.conf()
                .set("spark.sql.legacy.ctePrecedencePolicy", policy)
                .unwrap();
        }
        for ns in ["default", "ns2"] {
            ok(&bob, &format!("USE {ns}"));
            ok(&ana, &format!("USE {ns}"));
            let at = format!("{policy} in {ns}");
            assert_eq!(
                first(&bob, "SELECT * FROM default.dvc").unwrap(),
                "1",
                "{at}"
            );
            assert_eq!(
                first(&bob, "WITH c AS (SELECT 9 AS a) SELECT * FROM default.dvc").unwrap(),
                "1",
                "{at}"
            );
            assert_eq!(
                first(&ana, "SELECT * FROM default.ivc").unwrap(),
                "1",
                "{at}"
            );
            assert_eq!(
                first(&ana, "SELECT * FROM default.ivk").unwrap(),
                "5",
                "{at}"
            );
            assert_eq!(
                first(&ana, "SELECT * FROM default.ivl").unwrap(),
                "1",
                "{at}"
            );
        }
        // A temporary view named like the view's definition never reaches its text either.
        ok(&bob, "CREATE OR REPLACE TEMP VIEW c AS SELECT 9 AS a");
        assert_eq!(
            first(&bob, "SELECT * FROM default.dvc").unwrap(),
            "1",
            "{policy}"
        );
        ok(&bob, "DROP VIEW c");
    }
}

/// Review r4 Y2 (ADR 0048 A1 §3 Listings): a read of a relation the principal may not list is
/// refused at table level, before any column is resolved, so no resolution error (an unresolved
/// column's suggestions, a type error) shows its schema; the refusal names the table only.
#[test]
fn a_relation_the_principal_may_not_list_is_refused_before_its_columns_resolve() {
    let w = World::new();
    let root = w.admin();
    let (lister, bob, ana) = (w.user("lister"), w.user("bob"), w.user("ana"));
    ok(&root, "CREATE NAMESPACE vaultns");
    ok(
        &root,
        "CREATE TABLE vaultns.vt (secret_a INT, secret_b INT) USING parquet",
    );
    ok(
        &root,
        "CREATE TABLE t (secret_c INT, secret_d INT) USING parquet",
    );
    ok(&root, "CREATE TABLE ct (x INT, secret_e INT) USING parquet");
    ok(&root, "GRANT USAGE ON CATALOG spark_catalog TO user:lister");
    ok(&root, "GRANT USAGE ON SCHEMA default TO user:lister");
    ok(&root, "GRANT USAGE ON CATALOG spark_catalog TO user:bob");
    ok(&root, "GRANT SELECT (x) ON ct TO user:ana");
    let hidden = ["secret_a", "secret_b", "secret_c", "secret_d", "secret_e"];
    for (who, s, sql) in [
        ("lister", &lister, "SELECT nosuch FROM vaultns.vt"),
        (
            "lister",
            &lister,
            "SELECT secret_a + array(1) FROM vaultns.vt",
        ),
        ("lister", &lister, "SELECT nosuch FROM default.t"),
        ("bob", &bob, "SELECT nosuch FROM default.t"),
        ("bob", &bob, "SELECT secret_c + array(1) FROM default.t"),
        ("bob", &bob, "SELECT nosuch FROM ct"),
    ] {
        let e = denied(s, sql);
        for h in hidden {
            // The statement's own column names may show; no other does.
            if !sql.contains(h) {
                assert!(!e.contains(h), "{who}: {sql}: {h}: {e}");
            }
        }
    }
    let e =
        block_on(async { bob.table("default.t").select(["nosuch"]).collect().await }).unwrap_err();
    assert_eq!(e.error_class().name(), DENIED, "{e}");
    for h in hidden {
        assert!(!e.to_string().contains(h), "{h}: {e}");
    }
    // A principal holding a grant on a relation may see its schema (A1): ana's suggestions list
    // `ct`'s columns, as its `DESCRIBE` does.
    let e = run(&ana, "SELECT nosuch FROM ct").unwrap_err();
    assert!(
        e.error_class().name().starts_with("UNRESOLVED_COLUMN"),
        "{e}"
    );
}

/// Review r4 N1: a refused column is named only when the statement's reference to it (resolved
/// to that table's column) wrote it, not when the statement holds the same name elsewhere: an
/// alias of another relation, another table's column, a lambda variable.
#[test]
fn a_refused_column_is_named_only_by_a_reference_to_it() {
    let w = World::new();
    let root = w.admin();
    let ana = w.user("ana");
    ok(
        &root,
        "CREATE TABLE ct (x INT, secret_a INT, secret_b INT) USING parquet",
    );
    ok(&root, "INSERT INTO ct VALUES (1, 2, 3)");
    ok(&root, "CREATE TABLE o (x INT, secret_a INT) USING parquet");
    ok(&root, "INSERT INTO o VALUES (1, 2)");
    ok(&root, "GRANT SELECT (x) ON ct TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE o TO user:ana");
    for sql in [
        "SELECT ct.*, v.secret_a FROM ct, (SELECT 1 AS secret_a) v",
        "SELECT ct.*, o.secret_a FROM ct, o",
        "SELECT ct.*, transform(array(1), secret_a -> secret_a) FROM ct",
    ] {
        let e = denied(&ana, sql);
        for h in ["secret_a", "secret_b"] {
            assert!(!e.contains(h), "{sql}: {h}: {e}");
        }
    }
    // A reference to the refused column names it, however it is reached.
    for (sql, named) in [
        ("SELECT secret_b FROM ct", "`secret_b`"),
        ("SELECT x FROM ct WHERE secret_a = 2", "`secret_a`"),
        ("SELECT s.secret_a FROM (SELECT * FROM ct) s", "`secret_a`"),
        ("SELECT x FROM ct ORDER BY secret_a", "`secret_a`"),
    ] {
        let e = denied(&ana, sql);
        assert!(e.contains(named), "{sql}: {e}");
    }
}

/// The arguments a [`Shadow`] UDF was called with, by test.
static SHADOWED: std::sync::Mutex<Vec<(&'static str, String)>> = std::sync::Mutex::new(Vec::new());

/// A session UDF named `name` returning `true` for every row, recording each argument under
/// `test`: a reader's function shadowing a name a view's text calls.
#[derive(Debug)]
struct Shadow {
    name: &'static str,
    test: &'static str,
}

impl krishiv_session::ScalarUdf for Shadow {
    fn name(&self) -> &str {
        self.name
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &krishiv_session::Signature {
        static SIG: std::sync::OnceLock<krishiv_session::Signature> = std::sync::OnceLock::new();
        SIG.get_or_init(|| {
            krishiv_session::Signature::Exact(vec![krishiv_session::UdfInputType::Any])
        })
    }

    fn return_field(
        &self,
        _: &[krishiv_session::ArgField],
    ) -> Result<krishiv_session::ReturnField, krishiv_session::UdfError> {
        Ok(krishiv_session::ReturnField {
            data_type: krishiv_types::DataType::Boolean,
            nullable: false,
        })
    }

    fn invoke(
        &self,
        args: &[krishiv_session::UdfArg<'_>],
        rows: usize,
        _: &krishiv_session::InvokeContext,
    ) -> Result<arrow_array::ArrayRef, krishiv_session::UdfError> {
        let a = args[0].array();
        let mut seen = SHADOWED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for i in 0..a.len() {
            let value = arrow_cast::display::array_value_to_string(&a, i).unwrap_or_default();
            seen.push((self.test, value));
        }
        Ok(Arc::new(arrow_array::BooleanArray::from(vec![true; rows])))
    }
}

/// Whether a [`Shadow`] of `test` was ever called.
fn shadow_called(test: &str) -> bool {
    SHADOWED.lock().unwrap().iter().any(|(t, _)| *t == test)
}

/// The first column of every row of `sql`, in order.
fn column(s: &Session, sql: &str) -> Result<Vec<String>, KrishivError> {
    block_on(async { values(s.sql(sql).await?).await })
}

async fn values(df: krishiv_session::DataFrame) -> Result<Vec<String>, KrishivError> {
    let batches = df.collect().await?;
    let mut out = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            out.push(arrow_cast::display::array_value_to_string(b.column(0), r).unwrap());
        }
    }
    Ok(out)
}

const POLICIES: [&str; 3] = ["CORRECTED", "LEGACY", "EXCEPTION"];

/// root's `t (owner STRING, a INT)` = ana, carl, dave, readable by ana only; ana's stored function
/// `allowed(o)` (`o = 'ana'`), her definer view `dvf` and invoker view `ivf` filtering `t` by it.
/// bob may read `dvf` and call `allowed`.
fn function_world() -> (World, Session, Session) {
    let w = World::new();
    let (root, ana) = (w.admin(), w.user("ana"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(
        &root,
        "INSERT INTO t VALUES ('ana', 1), ('carl', 2), ('dave', 3)",
    );
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION allowed(o STRING) RETURNS BOOLEAN RETURN o = 'ana'",
    );
    ok(
        &ana,
        "CREATE VIEW dvf SQL SECURITY DEFINER AS SELECT owner FROM t WHERE allowed(owner)",
    );
    ok(
        &ana,
        "CREATE VIEW ivf AS SELECT owner FROM t WHERE allowed(owner)",
    );
    ok(&ana, "GRANT SELECT ON VIEW dvf TO user:bob");
    ok(&ana, "GRANT SELECT ON FUNCTION allowed TO user:bob");
    (w, root, ana)
}

/// Review r5 Z1: a stored view's text never resolves a function of the reading session. Its bare
/// calls resolve through the built-ins, then the catalog schemas of its frozen path; a reader's
/// session UDF (Rust, Python) or temporary SQL function of the same name is never called, under
/// every CTE policy, from SQL or `table()`. Invoker views too: that is Spark's resolution.
///
/// Spark v4.2.0 `SessionCatalog.scala` lines 2427-2441 (`handleViewContext`: while a view is
/// resolved a temporary function answers only if `referredTempFunctionNames` holds it, and that set
/// is empty for a permanent view, `views.scala` lines 628-650) and lines 2523-2550
/// (`lookupFunctionInfoByIdentifier`, `resolveScalarFunctionByIdentifier` apply it); functions
/// registered with `spark.udf.register` are temporary ones.
#[test]
fn a_stored_views_text_never_calls_a_readers_session_function() {
    const TEST: &str = "z1";
    let (w, _root, ana) = function_world();
    let ana_only = vec!["ana".to_owned()];
    // A Rust (or Connect) UDF registered by the reader.
    let bob = w.user("bob");
    bob.register_udf(Shadow {
        name: "allowed",
        test: TEST,
    })
    .unwrap();
    for policy in POLICIES {
        bob.conf()
            .set("spark.sql.legacy.ctePrecedencePolicy", policy)
            .unwrap();
        assert_eq!(
            column(&bob, "SELECT owner FROM default.dvf").unwrap(),
            ana_only,
            "{policy}"
        );
        assert_eq!(
            block_on(values(bob.table("default.dvf"))).unwrap(),
            ana_only,
            "{policy}: table()"
        );
    }
    // A Python UDF: its runtime is not even set, so a call would fail.
    let bob_py = w.user("bob");
    bob_py.register_python_udf(krishiv_session::PythonUdfDef {
        name: "allowed".to_owned(),
        eval_type: krishiv_plan::PythonEvalType::ArrowBatched,
        return_type: krishiv_types::DataType::Boolean,
        deterministic: true,
        thread_safe: true,
    });
    assert_eq!(
        column(&bob_py, "SELECT owner FROM default.dvf").unwrap(),
        ana_only
    );
    // A temporary SQL function, whatever the reader's path says.
    let bob_tmp = w.user("bob");
    ok(
        &bob_tmp,
        "CREATE TEMPORARY FUNCTION allowed(o STRING) RETURNS BOOLEAN RETURN true",
    );
    assert_eq!(
        column(&bob_tmp, "SELECT owner FROM default.dvf").unwrap(),
        ana_only
    );
    ok(&bob_tmp, "SET spark.sql.path.enabled = true");
    ok(
        &bob_tmp,
        "SET PATH = system.session, system.builtin, spark_catalog.default",
    );
    assert_eq!(
        column(&bob_tmp, "SELECT owner FROM default.dvf").unwrap(),
        ana_only
    );
    // The reader's own call still reaches the reader's function.
    assert_eq!(rows(&bob_tmp, "SELECT allowed('x')"), 1);
    // An invoker view read by its creator, under her own UDF of the same name: the stored function.
    ana.register_udf(Shadow {
        name: "allowed",
        test: TEST,
    })
    .unwrap();
    assert_eq!(column(&ana, "SELECT owner FROM ivf").unwrap(), ana_only);
    // A view a stored function's body reads is a stored view's text too.
    ok(
        &ana,
        "CREATE FUNCTION cnt() RETURNS BIGINT RETURN (SELECT count(*) FROM ivf)",
    );
    assert_eq!(first(&ana, "SELECT cnt()").unwrap(), "1");
    assert!(!shadow_called(TEST), "{:?}", SHADOWED.lock().unwrap());
}

/// Review r5 Z1/Z2: a stored view's text never reads a session variable of the reader. `dvl` read
/// the column `lim`; once its table no longer has one, the name is unresolved (redacted for the
/// definer view's reader), never the reader's variable `lim` (Spark: a view resolves only the
/// variables it referred to at creation, `Analyzer.scala` line 254, none for a permanent view).
#[test]
fn a_stored_views_text_never_reads_a_readers_session_variable() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&ana, "CREATE TABLE tl (a INT, lim INT) USING parquet");
    ok(&ana, "INSERT INTO tl VALUES (1, 2), (5, 2), (9, 2)");
    ok(
        &ana,
        "CREATE VIEW dvl SQL SECURITY DEFINER AS SELECT a FROM tl WHERE a < lim",
    );
    ok(&ana, "CREATE VIEW ivl AS SELECT a FROM tl WHERE a < lim");
    ok(&ana, "GRANT SELECT ON VIEW dvl TO user:bob");
    assert_eq!(column(&bob, "SELECT a FROM default.dvl").unwrap(), ["1"]);
    ok(&ana, "DROP TABLE tl");
    ok(&ana, "CREATE TABLE tl (a INT) USING parquet");
    ok(&ana, "INSERT INTO tl VALUES (1), (5), (9)");
    ok(&bob, "DECLARE VARIABLE lim INT DEFAULT 100");
    let e = column(&bob, "SELECT a FROM default.dvl").unwrap_err();
    assert!(
        e.to_string().contains(krishiv_common::secret::REDACTED),
        "{e}"
    );
    ok(&ana, "DECLARE VARIABLE lim INT DEFAULT 100");
    let e = column(&ana, "SELECT a FROM ivl").unwrap_err();
    assert!(
        e.error_class().name().starts_with("UNRESOLVED_COLUMN"),
        "{e}"
    );
    // The reader's own text still reads its variable.
    assert_eq!(column(&bob, "SELECT lim").unwrap(), ["100"]);
}

/// Review r5 Z2: a permanent view may not reference a temporary view, a temporary function, a
/// session UDF or a session variable: `CREATE VIEW`, `CREATE OR REPLACE VIEW` and `ALTER VIEW .. AS`
/// are refused with `INVALID_TEMP_OBJ_REFERENCE`, temporary views first, then functions, then
/// variables. Spark v4.2.0 `views.scala` lines 628-650 (`verifyTemporaryObjectsNotExists`), with the
/// parameters of `QueryCompilationErrors.notAllowedToCreatePermanentViewByReferencingTemp{View,Func,
/// Var}Error` (lines 3447-3481); goldens `sql-session-variables.sql.out` (a variable) and
/// `postgreSQL/create_view.sql.out` (a temporary view). A temporary view may reference all of them.
#[test]
fn a_permanent_view_may_not_reference_an_object_of_the_session() {
    let (_w, root, ana) = world();
    ok(&ana, "CREATE TEMP VIEW tv AS SELECT * FROM t");
    ok(&ana, "DECLARE VARIABLE lim INT DEFAULT 3");
    ok(
        &ana,
        "DECLARE VARIABLE st STRUCT<f: INT> DEFAULT named_struct('f', 3)",
    );
    ok(
        &ana,
        "CREATE TEMPORARY FUNCTION tf(x INT) RETURNS BOOLEAN RETURN x < 3",
    );
    ok(
        &ana,
        "CREATE TEMPORARY FUNCTION ttf() RETURNS TABLE (a INT) RETURN SELECT 1",
    );
    ana.register_udf(Shadow {
        name: "ruf",
        test: "z2",
    })
    .unwrap();
    ana.register_python_udf(krishiv_session::PythonUdfDef {
        name: "pyf".to_owned(),
        eval_type: krishiv_plan::PythonEvalType::ArrowBatched,
        return_type: krishiv_types::DataType::Boolean,
        deterministic: true,
        thread_safe: true,
    });
    ok(&ana, "CREATE VIEW altered AS SELECT a FROM t");
    for (query, temp, kind) in [
        ("SELECT * FROM tv", "`tv`", "VIEW"),
        ("SELECT * FROM tv WHERE tf(a) AND a < lim", "`tv`", "VIEW"),
        ("SELECT * FROM t WHERE tf(a)", "`tf`", "FUNCTION"),
        ("SELECT * FROM t WHERE session.tf(a)", "`tf`", "FUNCTION"),
        (
            "SELECT * FROM t WHERE system.session.TF(a)",
            "`tf`",
            "FUNCTION",
        ),
        ("SELECT * FROM ttf()", "`ttf`", "FUNCTION"),
        ("SELECT * FROM t WHERE ruf(a)", "`ruf`", "FUNCTION"),
        ("SELECT pyf(a) AS p FROM t", "`pyf`", "FUNCTION"),
        (
            "SELECT a FROM t WHERE a IN (SELECT a FROM t WHERE tf(a))",
            "`tf`",
            "FUNCTION",
        ),
        (
            "SELECT * FROM t WHERE tf(a) AND a < lim",
            "`tf`",
            "FUNCTION",
        ),
        ("SELECT * FROM t WHERE a < lim", "`lim`", "VARIABLE"),
        (
            "SELECT * FROM t WHERE a < session.lim",
            "`session`.`lim`",
            "VARIABLE",
        ),
        (
            "SELECT * FROM t WHERE a < system.session.lim",
            "`system`.`session`.`lim`",
            "VARIABLE",
        ),
        (
            "SELECT a FROM t WHERE a IN (SELECT lim)",
            "`lim`",
            "VARIABLE",
        ),
    ] {
        for (statement, view) in [
            (format!("CREATE VIEW cv AS {query}"), "cv"),
            (
                format!("CREATE VIEW dv SQL SECURITY DEFINER AS {query}"),
                "dv",
            ),
            (format!("ALTER VIEW altered AS {query}"), "altered"),
        ] {
            let e = run(&ana, &statement).unwrap_err();
            assert_eq!(
                e.error_class().name(),
                "INVALID_TEMP_OBJ_REFERENCE",
                "{statement}: {e}"
            );
            let message = e.to_string();
            let expected = format!(
                "Cannot create the persistent object `spark_catalog`.`default`.`{view}` of the type \
                 VIEW because it references to the temporary object {temp} of the type {kind}."
            );
            assert!(message.contains(&expected), "{statement}: {message}");
        }
    }
    // `CREATE OR REPLACE VIEW` of an existing view (root may drop it).
    ok(&root, "CREATE VIEW replaced AS SELECT a FROM t");
    ok(&root, "DECLARE VARIABLE root_lim INT DEFAULT 3");
    let e = run(
        &root,
        "CREATE OR REPLACE VIEW replaced AS SELECT a FROM t WHERE a < root_lim",
    )
    .unwrap_err();
    assert!(
        e.to_string().contains(
            "`spark_catalog`.`default`.`replaced` of the type VIEW because it references to the \
             temporary object `root_lim` of the type VARIABLE"
        ),
        "{e}"
    );
    // A field of a struct variable is a reference to the variable.
    let e = run(&ana, "CREATE VIEW sv AS SELECT a, st.f AS f FROM t").unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_TEMP_OBJ_REFERENCE", "{e}");
    // A column wins over a variable of its name; a stored function and a built-in are no session
    // objects; a temporary view may reference every one of them.
    ok(&ana, "CREATE TABLE tl (a INT, lim INT) USING parquet");
    ok(
        &ana,
        "CREATE VIEW by_column AS SELECT a FROM tl WHERE a < lim",
    );
    ok(
        &ana,
        "CREATE FUNCTION stored_f(x INT) RETURNS BOOLEAN RETURN x < 3",
    );
    ok(
        &ana,
        "CREATE VIEW by_stored AS SELECT a FROM t WHERE stored_f(a) AND lower('A') = 'a'",
    );
    ok(
        &ana,
        "CREATE TEMP VIEW tv2 AS SELECT a FROM tv WHERE tf(a) AND ruf(a) AND a < lim",
    );
}

/// Review r5 M7: a stored function a definer view's text calls is the definer's to call (the text
/// is analyzed as the definer): the reader needs `Select` on the view only, and the definer's
/// refusal reaches the reader redacted. A call the reader's own statement makes is still asked of
/// the reader.
#[test]
fn a_function_a_definer_views_text_calls_is_asked_of_the_definer() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 2)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(
        &root,
        "CREATE FUNCTION rootf(o STRING) RETURNS BOOLEAN RETURN o = 'ana'",
    );
    ok(&root, "GRANT SELECT ON FUNCTION rootf TO user:ana");
    ok(
        &ana,
        "CREATE VIEW dvr SQL SECURITY DEFINER AS SELECT owner FROM t WHERE rootf(owner)",
    );
    ok(&ana, "GRANT SELECT ON VIEW dvr TO user:bob");
    assert_eq!(
        column(&bob, "SELECT owner FROM default.dvr").unwrap(),
        ["ana"]
    );
    let e = denied(
        &bob,
        "SELECT owner FROM default.dvr WHERE default.rootf(owner)",
    );
    assert!(e.contains("rootf"), "{e}");
    ok(&root, "REVOKE SELECT ON FUNCTION rootf FROM user:ana");
    let e = column(&bob, "SELECT owner FROM default.dvr").unwrap_err();
    assert_redacted(&e.to_string(), &["rootf", "`t`"]);
}

/// Review r6 R6-1: a stored view's text calls a stored SQL table function by its bare name as it
/// calls a scalar one, through the built-ins, then the catalog schemas of its frozen path (or, stored
/// with no path, its own namespace), for its creator and its readers alike; a reader's temporary
/// table function of that name never answers there. Spark v4.2.0 `SessionCatalog.scala` lines
/// 2427-2441 (`handleViewContext` filters only temporary functions while a permanent view resolves;
/// a persistent table function resolves through the view's captured `catalogAndNamespace`).
#[test]
fn a_stored_views_text_calls_a_stored_table_function_by_its_bare_name() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "CREATE SCHEMA ns2");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:bob");
    ok(
        &ana,
        "CREATE FUNCTION gt() RETURNS TABLE (o STRING) RETURN SELECT owner FROM t",
    );
    ok(
        &ana,
        "CREATE FUNCTION ns2.gt() RETURNS TABLE (o STRING) RETURN SELECT 'ns2'",
    );
    ok(&ana, "CREATE VIEW ivt AS SELECT o FROM gt()");
    ok(
        &ana,
        "CREATE VIEW dvt SQL SECURITY DEFINER AS SELECT o FROM gt()",
    );
    ok(&ana, "CREATE VIEW ivl AS SELECT x.o FROM t, LATERAL gt() x");
    ok(&ana, "SET spark.sql.path.enabled = true");
    ok(
        &ana,
        "SET PATH = system.builtin, spark_catalog.ns2, spark_catalog.default",
    );
    ok(&ana, "CREATE VIEW ivp AS SELECT o FROM gt()");
    ok(&ana, "SET PATH = DEFAULT_PATH");
    let sorted = |s: &Session, sql: &str| {
        let mut v = column(s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        v.sort();
        v
    };
    let two = ["ana", "carl"];
    let four = ["ana", "ana", "carl", "carl"];
    assert_eq!(sorted(&ana, "SELECT * FROM ivt"), two);
    assert_eq!(sorted(&ana, "SELECT * FROM dvt"), two);
    assert_eq!(sorted(&ana, "SELECT * FROM ivl"), four);
    assert_eq!(sorted(&ana, "SELECT * FROM ivp"), ["ns2"]);
    for v in ["ivt", "dvt", "ivl"] {
        ok(&ana, &format!("GRANT SELECT ON VIEW {v} TO user:bob"));
    }
    ok(&ana, "GRANT SELECT ON FUNCTION gt TO user:bob");
    // bob's temporary table function of the same name answers his own bare call, never the views'.
    ok(
        &bob,
        "CREATE TEMPORARY FUNCTION gt() RETURNS TABLE (o STRING) RETURN SELECT 'bob'",
    );
    assert_eq!(sorted(&bob, "SELECT * FROM gt()"), ["bob"]);
    for policy in POLICIES {
        ok(
            &bob,
            &format!("SET spark.sql.legacy.ctePrecedencePolicy = {policy}"),
        );
        assert_eq!(sorted(&bob, "SELECT * FROM default.ivt"), two, "{policy}");
        assert_eq!(sorted(&bob, "SELECT * FROM default.dvt"), two, "{policy}");
        assert_eq!(sorted(&bob, "SELECT * FROM default.ivl"), four, "{policy}");
    }
    let table = block_on(async { values(bob.table("default.dvt")).await }).unwrap();
    assert_eq!(table.len(), 2);
}

/// Review r6 M8: the body of a stored function a definer view's text calls reads as that definer,
/// as the view's text does: its reader needs `Select` on the view only, and a refusal inside it is
/// the definer's failure, redacted. A function body reads as invoker for a call the reader makes
/// itself. A definer view read in a body is read as an invoker one (fails closed), and its refusal
/// names nothing the reader did not write.
#[test]
fn a_function_a_definer_views_text_calls_reads_its_body_as_the_definer() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "CREATE TABLE hidden_allow (o STRING) USING parquet");
    ok(&root, "INSERT INTO hidden_allow VALUES ('ana')");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE hidden_allow TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION g(x STRING) RETURNS BOOLEAN RETURN x IN (SELECT o FROM hidden_allow)",
    );
    ok(
        &ana,
        "CREATE VIEW dvg SQL SECURITY DEFINER AS SELECT owner FROM t WHERE g(owner)",
    );
    ok(
        &ana,
        "CREATE FUNCTION gt() RETURNS TABLE (o STRING) RETURN SELECT o FROM hidden_allow",
    );
    ok(
        &ana,
        "CREATE VIEW dvgt SQL SECURITY DEFINER AS SELECT o FROM gt()",
    );
    ok(&ana, "GRANT SELECT ON VIEW dvg TO user:bob");
    ok(&ana, "GRANT SELECT ON VIEW dvgt TO user:bob");
    assert_eq!(column(&bob, "SELECT * FROM default.dvg").unwrap(), ["ana"]);
    assert_eq!(column(&bob, "SELECT * FROM default.dvgt").unwrap(), ["ana"]);
    let table = block_on(async { values(bob.table("default.dvg")).await }).unwrap();
    assert_eq!(table, ["ana"]);
    // bob's own call of g reads its body as bob.
    ok(&ana, "GRANT SELECT ON FUNCTION g TO user:bob");
    assert_eq!(column(&bob, "SELECT * FROM default.dvg").unwrap(), ["ana"]);
    let e = denied(&bob, "SELECT default.g('ana')");
    assert!(e.contains("hidden_allow"), "{e}");
    let e = denied(&bob, "SELECT owner FROM default.dvg WHERE default.g(owner)");
    assert!(e.contains("hidden_allow"), "{e}");
    // The definer losing the body's table refuses bob, redacted.
    ok(&root, "REVOKE SELECT ON TABLE hidden_allow FROM user:ana");
    for v in ["dvg", "dvgt"] {
        let e = column(&bob, &format!("SELECT * FROM default.{v}")).unwrap_err();
        assert_redacted(&e.to_string(), &["hidden_allow", "`t`", "`ana`"]);
    }
    // A definer view read in a body bob calls is read as an invoker one: refused, redacted.
    ok(
        &ana,
        "CREATE VIEW dvl SQL SECURITY DEFINER AS SELECT a FROM t",
    );
    ok(
        &ana,
        "CREATE FUNCTION af() RETURNS TABLE (a INT) RETURN SELECT a FROM default.dvl",
    );
    ok(&ana, "GRANT SELECT ON VIEW dvl TO user:bob");
    ok(&ana, "GRANT SELECT ON FUNCTION af TO user:bob");
    assert_eq!(
        column(&bob, "SELECT * FROM default.dvl").unwrap(),
        ["1", "5"]
    );
    let e = column(&bob, "SELECT * FROM default.af()").unwrap_err();
    assert_redacted(&e.to_string(), &["hidden_allow", "`t`"]);
}

/// Review r6 m1: a definer view inside a definer view's text reads as its own definer, the
/// functions its text calls included: the outer definer is not asked about them.
#[test]
fn a_nested_definer_views_calls_are_asked_of_its_own_definer() {
    let w = World::new();
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:carl");
    ok(&root, "GRANT SELECT ON TABLE t TO user:carl");
    ok(
        &carl,
        "CREATE FUNCTION cf(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &carl,
        "CREATE VIEW dv2 SQL SECURITY DEFINER AS SELECT owner FROM t WHERE cf(owner)",
    );
    ok(&carl, "GRANT SELECT ON VIEW dv2 TO user:ana");
    ok(
        &ana,
        "CREATE VIEW dv1 SQL SECURITY DEFINER AS SELECT owner FROM default.dv2",
    );
    ok(&ana, "GRANT SELECT ON VIEW dv1 TO user:bob");
    assert_eq!(column(&ana, "SELECT * FROM dv1").unwrap(), ["ana"]);
    assert_eq!(column(&bob, "SELECT * FROM default.dv1").unwrap(), ["ana"]);
    // carl losing `t` refuses bob, redacted.
    ok(&root, "REVOKE SELECT ON TABLE t FROM user:carl");
    let e = column(&bob, "SELECT * FROM default.dv1").unwrap_err();
    assert_redacted(&e.to_string(), &["`t`", "cf", "`carl`"]);
}
