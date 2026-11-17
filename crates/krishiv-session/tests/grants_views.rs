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
