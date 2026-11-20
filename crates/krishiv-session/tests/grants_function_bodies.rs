// SPDX-License-Identifier: Apache-2.0
//! P3-T062 unit G3 (ADR 0048 A1 §3) under the `grants` provider: the body of a stored SQL function
//! resolves its names only through the function's own stored path and namespace, never the reading
//! session (review r7 R7-1), and a body a definer view's text calls reads with the intersection of
//! the view's definer and the function's owner (review r7 M9).
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::grants::{Grants, GrantsPolicy, MemoryGrantStore};
use krishiv_session::{AccessPolicy, AllowAll, EngineRuntime, Principal, PrincipalKind, Session};

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

    /// A session of `name` deciding by the grants but with no grants provider (`Session::builder`
    /// takes the policy and the provider apart): no owner is recorded for what it creates.
    fn without_provider(&self, name: &str) -> Session {
        let policy = Arc::new(GrantsPolicy::new(
            Arc::clone(&self.grants),
            Arc::new(AllowAll),
            false,
        ));
        Session::builder()
            .runtime(Arc::clone(&self.rt))
            .principal(Principal::new(name, PrincipalKind::User))
            .access_policy(policy as Arc<dyn AccessPolicy>)
            .build()
            .unwrap()
    }
}

fn ok(s: &Session, sql: &str) {
    block_on(async { s.sql(sql).await?.collect().await }).unwrap_or_else(|e| panic!("{sql}: {e}"));
}

/// Every row of `sql`, its columns joined by `|`.
fn rows(s: &Session, sql: &str) -> Result<Vec<String>, KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        let mut out = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows() {
                let row: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| arrow_cast::display::array_value_to_string(c, r).unwrap())
                    .collect();
                out.push(row.join("|"));
            }
        }
        Ok(out)
    })
}

fn read(s: &Session, sql: &str) -> Vec<String> {
    rows(s, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// A refusal of a definer's read as its reader sees it: 42501, every parameter redacted, none of
/// `hidden` shown.
#[track_caller]
fn assert_redacted(s: &Session, sql: &str, hidden: &[&str]) {
    let e = match rows(s, sql) {
        Err(e) => e.to_string(),
        Ok(r) => panic!("{sql}: expected a refusal, got {r:?}"),
    };
    assert!(
        e.contains(krishiv_common::secret::REDACTED) && e.contains("42501"),
        "{sql}: {e}"
    );
    for h in hidden {
        assert!(!e.contains(h), "{sql}: {h}: {e}");
    }
}

/// Review r7 R7-1 (probe p24): a stored function's body resolves a bare table-function call, a bare
/// scalar call and a bare table name only through its own stored path and namespace. No temporary
/// function, stored function of the reader's current namespace or reader's table answers there, and
/// the definer view over it reads the same for any reader in any namespace.
///
/// Spark v4.2.0 `Analyzer.scala` lines 260-270 (`AnalysisContext.withAnalysisContext(function:
/// SQLFunction)`): "Function body analysis should not inherit any caller-pinned path; use only
/// function metadata", the path being `function.functionStoredResolutionPath`.
#[test]
fn a_stored_bodys_names_resolve_only_through_its_own_path() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE NAMESPACE vns");
    ok(&root, "CREATE NAMESPACE bobns");
    ok(&root, "CREATE TABLE default.x (s STRING) USING parquet");
    ok(&root, "INSERT INTO default.x VALUES ('TOPSECRET')");
    ok(&root, "CREATE TABLE vns.x (s STRING) USING parquet");
    ok(&root, "INSERT INTO vns.x VALUES ('vns-x')");
    ok(&root, "GRANT SELECT ON TABLE default.x TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE vns.x TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA vns TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA bobns TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA bobns TO user:ana");
    for u in ["ana", "bob"] {
        ok(&root, &format!("GRANT USAGE ON SCHEMA default TO user:{u}"));
    }
    ok(&bob, "CREATE TABLE bobns.x (s STRING) USING parquet");
    ok(&bob, "INSERT INTO bobns.x VALUES ('bobx')");
    ok(
        &ana,
        "CREATE FUNCTION vns.inner_t() RETURNS TABLE (o STRING) RETURN SELECT 'ana-inner' AS o",
    );
    ok(&ana, "USE vns");
    ok(
        &ana,
        "CREATE FUNCTION vns.outer_t() RETURNS TABLE (o STRING) RETURN SELECT o FROM inner_t()",
    );
    ok(
        &ana,
        "CREATE FUNCTION vns.sf() RETURNS STRING RETURN (SELECT max(o) FROM inner_t())",
    );
    ok(
        &ana,
        "CREATE FUNCTION vns.subf(x STRING) RETURNS BOOLEAN \
         RETURN x IN (SELECT o FROM inner_t())",
    );
    ok(
        &ana,
        "CREATE VIEW vns.dv SQL SECURITY DEFINER AS SELECT o FROM vns.outer_t()",
    );
    ok(
        &ana,
        "CREATE VIEW vns.dvs SQL SECURITY DEFINER AS SELECT vns.sf() AS v",
    );
    ok(
        &ana,
        "CREATE VIEW vns.dvf SQL SECURITY DEFINER AS \
         SELECT s FROM vns.x WHERE vns.subf('ana-inner')",
    );
    for v in ["dv", "dvs", "dvf"] {
        ok(&ana, &format!("GRANT SELECT ON VIEW vns.{v} TO user:bob"));
    }
    let expect = |s: &Session| {
        assert_eq!(read(s, "SELECT * FROM vns.dv"), ["ana-inner"]);
        assert_eq!(read(s, "SELECT * FROM vns.dvs"), ["ana-inner"]);
        assert_eq!(read(s, "SELECT * FROM vns.dvf"), ["vns-x"]);
    };
    // Readable from outside `vns` (before the fix: UNRESOLVABLE_TABLE_VALUED_FUNCTION).
    expect(&bob);
    expect(&ana);
    ok(&ana, "USE default");
    expect(&ana);
    // bob's temporary `inner_t`, from his own namespace and from `default`.
    ok(&bob, "USE bobns");
    ok(
        &bob,
        "CREATE TEMPORARY FUNCTION inner_t() RETURNS TABLE (o STRING) RETURN SELECT s AS o FROM x",
    );
    assert_eq!(read(&bob, "SELECT * FROM inner_t()"), ["bobx"]);
    expect(&bob);
    ok(&bob, "USE default");
    expect(&bob);
    // bob's stored `inner_t` in his current namespace, granted to ana.
    ok(&bob, "DROP TEMPORARY FUNCTION inner_t");
    ok(
        &bob,
        "CREATE FUNCTION bobns.inner_t() RETURNS TABLE (o STRING) RETURN SELECT 'bob-stored' AS o",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION bobns.inner_t TO user:ana");
    ok(&bob, "USE bobns");
    expect(&bob);
    ok(&ana, "USE bobns");
    expect(&ana);
}

/// Review r7 R7-1 (probe p18): a reader's (or the definer's own) temporary table function of the
/// name a stored body calls bare never rewrites a definer view's row filter.
#[test]
fn a_temporary_function_never_answers_a_stored_bodys_call() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION inner_t() RETURNS TABLE (o STRING) RETURN SELECT owner FROM t",
    );
    ok(
        &ana,
        "CREATE FUNCTION outer_f(x STRING) RETURNS BOOLEAN RETURN x IN (SELECT o FROM inner_t())",
    );
    ok(
        &ana,
        "CREATE FUNCTION outer_t() RETURNS TABLE (o STRING) RETURN SELECT o FROM inner_t()",
    );
    ok(
        &ana,
        "CREATE VIEW dvo SQL SECURITY DEFINER AS SELECT owner FROM t WHERE outer_f(owner)",
    );
    ok(
        &ana,
        "CREATE VIEW dvot SQL SECURITY DEFINER AS SELECT o FROM outer_t()",
    );
    ok(&ana, "GRANT SELECT ON VIEW dvo TO user:bob");
    ok(&ana, "GRANT SELECT ON VIEW dvot TO user:bob");
    let expect = |s: &Session| {
        assert_eq!(read(s, "SELECT * FROM default.dvo"), ["ana", "carl"]);
        assert_eq!(read(s, "SELECT * FROM default.dvot"), ["ana", "carl"]);
    };
    expect(&bob);
    ok(
        &bob,
        "CREATE TEMPORARY FUNCTION inner_t() RETURNS TABLE (o STRING) RETURN SELECT 'BOBS' AS o",
    );
    assert_eq!(read(&bob, "SELECT * FROM inner_t()"), ["BOBS"]);
    expect(&bob);
    ok(
        &ana,
        "CREATE TEMPORARY FUNCTION inner_t() RETURNS TABLE (o STRING) RETURN SELECT 'ANAS' AS o",
    );
    expect(&ana);
    // A temporary scalar function of a name a stored body calls bare: not called either.
    ok(
        &ana,
        "CREATE FUNCTION keep(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &ana,
        "CREATE FUNCTION via(x STRING) RETURNS BOOLEAN \
         RETURN EXISTS (SELECT 1 FROM t WHERE keep(x) AND owner = x)",
    );
    ok(
        &ana,
        "CREATE VIEW dvk SQL SECURITY DEFINER AS SELECT owner FROM t WHERE via(owner)",
    );
    ok(&ana, "GRANT SELECT ON VIEW dvk TO user:bob");
    assert_eq!(read(&bob, "SELECT * FROM default.dvk"), ["ana"]);
    ok(
        &bob,
        "CREATE TEMPORARY FUNCTION keep(x STRING) RETURNS BOOLEAN RETURN true",
    );
    assert_eq!(read(&bob, "SELECT * FROM default.dvk"), ["ana"]);
}

/// Review r7 M9 (probe p22): a stored body a definer view's text calls reads with the intersection
/// of the view's definer and the function's owner. The owner losing a grant the body needs refuses
/// the view's readers, redacted; so does the definer losing one the owner still holds.
#[test]
fn a_body_a_definer_views_text_calls_reads_as_its_definer_and_its_owner() {
    let w = World::new();
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(
        &root,
        "CREATE TABLE secret (owner STRING, s STRING) USING parquet",
    );
    ok(&root, "INSERT INTO secret VALUES ('ana', 'S3CRET')");
    for u in ["ana", "bob"] {
        ok(
            &root,
            &format!("GRANT CREATE ON SCHEMA default TO user:{u}"),
        );
        ok(&root, &format!("GRANT SELECT ON TABLE secret TO user:{u}"));
    }
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(
        &bob,
        "CREATE FUNCTION bf(x STRING) RETURNS STRING RETURN (SELECT max(s) FROM default.secret)",
    );
    ok(
        &bob,
        "CREATE FUNCTION bft() RETURNS TABLE (s STRING) RETURN SELECT s FROM default.secret",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION bf TO user:ana");
    ok(&bob, "GRANT SELECT ON FUNCTION bft TO user:ana");
    ok(
        &ana,
        "CREATE VIEW dvx SQL SECURITY DEFINER AS SELECT owner, default.bf(owner) AS v FROM t",
    );
    ok(
        &ana,
        "CREATE VIEW dvxt SQL SECURITY DEFINER AS SELECT s FROM default.bft()",
    );
    for v in ["dvx", "dvxt"] {
        ok(&ana, &format!("GRANT SELECT ON VIEW {v} TO user:bob"));
        ok(&ana, &format!("GRANT SELECT ON VIEW {v} TO user:carl"));
    }
    let both = |s: &Session| {
        assert_eq!(
            read(s, "SELECT * FROM default.dvx"),
            ["ana|S3CRET", "carl|S3CRET"]
        );
        assert_eq!(read(s, "SELECT * FROM default.dvxt"), ["S3CRET"]);
    };
    both(&bob);
    both(&carl);
    // The owner loses the body's table: refused for every reader, redacted.
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:bob");
    for s in [&bob, &carl, &ana] {
        for v in ["dvx", "dvxt"] {
            assert_redacted(
                s,
                &format!("SELECT * FROM default.{v}"),
                &["secret", "`bob`", "`ana`", "bf"],
            );
        }
    }
    // The reverse: the owner holds it, the definer does not.
    ok(&root, "GRANT SELECT ON TABLE secret TO user:bob");
    both(&carl);
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:ana");
    for s in [&bob, &carl] {
        for v in ["dvx", "dvxt"] {
            assert_redacted(
                s,
                &format!("SELECT * FROM default.{v}"),
                &["secret", "`bob`", "`ana`", "bf"],
            );
        }
    }
    // A body the definer owns reads as the definer alone (the review r6 M8 shape).
    ok(&root, "GRANT SELECT ON TABLE secret TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION af(x STRING) RETURNS BOOLEAN RETURN x IN (SELECT owner FROM secret)",
    );
    ok(
        &ana,
        "CREATE VIEW dva SQL SECURITY DEFINER AS SELECT owner FROM t WHERE af(owner)",
    );
    ok(&ana, "GRANT SELECT ON VIEW dva TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dva"), ["ana"]);
}

/// Review r7 n7 (probe p20) and r8 item 1 (A1 decision): a definer view's functions are bound
/// when it is created. Every name its text calls is resolved then, through the path frozen with
/// it, and stored fully qualified; a read never resolves it again. So a function created later in
/// a schema earlier on that path changes nothing, even granted to the definer: whoever may create
/// a function there never decides the view's row filter. Creating `ns2.f` beside the definer's
/// `default.f` asks the creator nothing about `default.f` (the r7 n7 refusal: a stored call's bare
/// name was looked up to test for an aggregate).
#[test]
fn a_function_shadowing_a_definer_views_frozen_path_changes_nothing() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    ok(&root, "CREATE TABLE ns2.bt (o STRING) USING parquet");
    ok(&root, "INSERT INTO ns2.bt VALUES ('carl')");
    ok(&root, "GRANT SELECT ON TABLE ns2.bt TO user:bob");
    ok(
        &ana,
        "CREATE FUNCTION f(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(&ana, "SET spark.sql.path.enabled = true");
    ok(
        &ana,
        "SET PATH = system.builtin, spark_catalog.ns2, spark_catalog.default",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvp SQL SECURITY DEFINER AS \
         SELECT owner FROM default.t WHERE f(owner)",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvp TO user:bob");
    assert_eq!(read(&bob, "SELECT * FROM default.dvp"), ["ana"]);
    // bob creates `ns2.f`, earlier on the view's frozen path, and grants it to ana: the view still
    // calls the `default.f` it was created over.
    ok(
        &bob,
        "CREATE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN RETURN true",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.f TO user:ana");
    for s in [&bob, &ana] {
        assert_eq!(read(s, "SELECT * FROM default.dvp"), ["ana"]);
    }
    // Nor does a body reading what only bob holds.
    ok(
        &bob,
        "CREATE OR REPLACE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN \
         RETURN x IN (SELECT o FROM ns2.bt)",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.f TO user:ana");
    assert_eq!(read(&bob, "SELECT * FROM default.dvp"), ["ana"]);
    ok(&bob, "DROP FUNCTION ns2.f");
    assert_eq!(read(&bob, "SELECT * FROM default.dvp"), ["ana"]);
    // Replacing the view binds its functions again, through the path of the replacing session.
    ok(
        &bob,
        "CREATE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN RETURN true",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.f TO user:ana");
    ok(
        &ana,
        "ALTER VIEW default.dvp AS SELECT owner FROM default.t WHERE f(owner)",
    );
    assert_eq!(read(&bob, "SELECT * FROM default.dvp"), ["ana", "carl"]);
    // The binding pins that version (review r9 item 1): bob replacing his `ns2.f` fails the view
    // closed until ana alters it again.
    ok(
        &bob,
        "CREATE OR REPLACE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN RETURN false",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.f TO user:ana");
    for s in [&bob, &ana] {
        assert_redacted(s, "SELECT * FROM default.dvp", &["ns2", "`bob`", "`ana`"]);
    }
}

/// Review r8 item 1: a function a definer view was bound to, dropped, fails the view closed, as a
/// dropped table does (Spark's error for a missing routine), redacted for its reader; one of the
/// name created later elsewhere on the frozen path never answers instead.
#[test]
fn a_dropped_bound_function_fails_a_definer_view_closed_and_redacted() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION keep(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &ana,
        "CREATE FUNCTION rows_of() RETURNS TABLE (o STRING) RETURN SELECT 'ana'",
    );
    ok(&ana, "SET spark.sql.path.enabled = true");
    ok(
        &ana,
        "SET PATH = system.builtin, spark_catalog.ns2, spark_catalog.default",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvd SQL SECURITY DEFINER AS \
         SELECT owner FROM default.t WHERE keep(owner)",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvdt SQL SECURITY DEFINER AS \
         SELECT owner FROM default.t WHERE owner IN (SELECT o FROM rows_of())",
    );
    for v in ["dvd", "dvdt"] {
        ok(
            &ana,
            &format!("GRANT SELECT ON VIEW default.{v} TO user:bob"),
        );
        assert_eq!(read(&bob, &format!("SELECT * FROM default.{v}")), ["ana"]);
    }
    // bob's functions of the same names, earlier on the frozen path and granted to ana.
    ok(
        &bob,
        "CREATE FUNCTION ns2.keep(x STRING) RETURNS BOOLEAN RETURN true",
    );
    ok(
        &bob,
        "CREATE FUNCTION ns2.rows_of() RETURNS TABLE (o STRING) RETURN SELECT 'carl'",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.keep TO user:ana");
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.rows_of TO user:ana");
    ok(&ana, "DROP FUNCTION default.keep");
    ok(&ana, "DROP FUNCTION default.rows_of");
    // Spark's errors for a missing routine: a scalar call's and a table function's.
    for (v, class) in [
        ("dvd", "UNRESOLVED_ROUTINE"),
        ("dvdt", "UNRESOLVABLE_TABLE_VALUED_FUNCTION"),
    ] {
        let sql = format!("SELECT * FROM default.{v}");
        let e = rows(&bob, &sql).expect_err(&sql);
        assert_eq!(e.error_class().name(), class, "{sql}: {e}");
        let text = e.to_string();
        assert!(
            text.contains(krishiv_common::secret::REDACTED),
            "{sql}: {text}"
        );
        for h in ["keep", "rows_of", "ns2", "default", "`ana`"] {
            assert!(!text.contains(h), "{sql}: {h}: {text}");
        }
        // The definer is told the same failure (her view no longer resolves).
        let e = rows(&ana, &sql).expect_err(&sql);
        assert_eq!(e.error_class().name(), class, "{sql}: {e}");
    }
}

/// Review r8 item 1: an invoker view keeps Spark's resolution, its bare calls resolved through its
/// frozen path each time it is read (Spark v4.2.0: the view's stored resolution path is searched
/// when its text is analyzed), so a function created later earlier on that path answers.
#[test]
fn an_invoker_views_calls_resolve_through_its_frozen_path_at_each_read() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION f(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(&ana, "SET spark.sql.path.enabled = true");
    ok(
        &ana,
        "SET PATH = system.builtin, spark_catalog.ns2, spark_catalog.default",
    );
    ok(
        &ana,
        "CREATE VIEW default.ivp AS SELECT owner FROM default.t WHERE f(owner)",
    );
    assert_eq!(read(&ana, "SELECT * FROM default.ivp"), ["ana"]);
    ok(
        &bob,
        "CREATE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN RETURN true",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.f TO user:ana");
    assert_eq!(read(&ana, "SELECT * FROM default.ivp"), ["ana", "carl"]);
    ok(&bob, "DROP FUNCTION ns2.f");
    assert_eq!(read(&ana, "SELECT * FROM default.ivp"), ["ana"]);
}

/// Review r7 n6: `EXPLAIN FORMATTED` (and the simple form) of a query whose definer view's reads
/// are refused prints the refusal, its class and SQLSTATE with every parameter redacted, as the
/// `EXTENDED` and `COST` forms do; never an empty message, never a plan.
#[test]
fn explain_formatted_of_a_refused_definer_read_prints_the_redacted_refusal() {
    let w = World::new();
    let (root, ana, bob) = (w.admin(), w.user("ana"), w.user("bob"));
    ok(&root, "CREATE TABLE secret (owner STRING) USING parquet");
    ok(&root, "INSERT INTO secret VALUES ('ana')");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE secret TO user:ana");
    ok(
        &ana,
        "CREATE VIEW default.dve SQL SECURITY DEFINER AS SELECT owner FROM default.secret",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dve TO user:bob");
    assert_eq!(read(&bob, "SELECT * FROM default.dve"), ["ana"]);
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:ana");
    for mode in ["FORMATTED", "EXTENDED", "COST", ""] {
        let sql = format!("EXPLAIN {mode} SELECT * FROM default.dve");
        let text = read(&bob, &sql).join("\n");
        assert!(
            text.contains("42501") && text.contains(krishiv_common::secret::REDACTED),
            "{sql}: {text}"
        );
        for h in ["secret", "`ana`", "Scan"] {
            assert!(!text.contains(h), "{sql}: {h}: {text}");
        }
    }
}

/// Review r9 item 1 (A1): a definer view binds the version of each stored function it reaches when
/// it is created or altered: those its text calls, bare or qualified, and those their bodies call.
/// A replacement of one by anyone but the view's definer fails the view closed, redacted for every
/// reader, until the definer alters or re-creates it; the view never returns what the new body
/// lets through. A function the definer owns and replaces herself keeps working: the view follows
/// it.
#[test]
fn a_bound_function_replaced_by_another_principal_fails_the_definer_view_closed() {
    let w = World::new();
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    ok(
        &bob,
        "CREATE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &bob,
        "CREATE FUNCTION ns2.k(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &bob,
        "CREATE FUNCTION ns2.h(x STRING) RETURNS BOOLEAN RETURN ns2.k(x)",
    );
    for f in ["f", "k", "h"] {
        ok(
            &bob,
            &format!("GRANT SELECT ON FUNCTION ns2.{f} TO user:ana"),
        );
    }
    ok(&ana, "SET spark.sql.path.enabled = true");
    ok(
        &ana,
        "SET PATH = system.builtin, spark_catalog.ns2, spark_catalog.default",
    );
    let views = [
        ("dvb", "SELECT owner FROM default.t WHERE f(owner)"),
        ("dvq", "SELECT owner FROM default.t WHERE ns2.f(owner)"),
        ("dvh", "SELECT owner FROM default.t WHERE h(owner)"),
    ];
    for (v, q) in views {
        ok(
            &ana,
            &format!("CREATE VIEW default.{v} SQL SECURITY DEFINER AS {q}"),
        );
        ok(
            &ana,
            &format!("GRANT SELECT ON VIEW default.{v} TO user:carl"),
        );
        ok(
            &ana,
            &format!("GRANT SELECT ON VIEW default.{v} TO user:bob"),
        );
        assert_eq!(read(&carl, &format!("SELECT * FROM default.{v}")), ["ana"]);
    }
    // bob replaces the functions the views bound (a body's call `ns2.k` included).
    ok(
        &bob,
        "CREATE OR REPLACE FUNCTION ns2.f(x STRING) RETURNS BOOLEAN RETURN true",
    );
    ok(
        &bob,
        "CREATE OR REPLACE FUNCTION ns2.k(x STRING) RETURNS BOOLEAN RETURN true",
    );
    for (v, _) in views {
        for s in [&carl, &bob, &ana] {
            assert_redacted(
                s,
                &format!("SELECT * FROM default.{v}"),
                &["ns2", "`bob`", "`ana`", "carl"],
            );
        }
    }
    // The definer altering the view binds the new versions.
    for (v, q) in views {
        ok(&ana, &format!("ALTER VIEW default.{v} AS {q}"));
        assert_eq!(
            read(&carl, &format!("SELECT * FROM default.{v}")),
            ["ana", "carl"]
        );
    }
    // A function the definer owns follows her own replacement.
    ok(
        &ana,
        "CREATE FUNCTION default.g(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvg SQL SECURITY DEFINER AS \
         SELECT owner FROM default.t WHERE default.g(owner)",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvg TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvg"), ["ana"]);
    ok(
        &ana,
        "CREATE OR REPLACE FUNCTION default.g(x STRING) RETURNS BOOLEAN RETURN x = 'carl'",
    );
    assert_eq!(read(&carl, "SELECT * FROM default.dvg"), ["carl"]);
}

/// Review r9 item 2 (A1; r8 concern 3): inside a definer view's text, a call resolved at each read
/// (in an invoker view the text reads, or in a stored body) may land only on a built-in, a function
/// the definer owns or a version the view bound when created. A function created later earlier on
/// the invoker view's (or the body's) frozen path, by someone else and granted to the definer, is
/// refused, redacted, never called. The invoker view read directly still resolves at each read, as
/// Spark's does.
#[test]
fn a_shadowing_function_in_a_nested_invoker_view_or_body_fails_the_definer_view_closed() {
    let w = World::new();
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT SELECT ON TABLE t TO user:ana");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION f(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(
        &ana,
        "CREATE FUNCTION g(x STRING) RETURNS BOOLEAN RETURN x = 'ana'",
    );
    ok(&ana, "SET spark.sql.path.enabled = true");
    ok(
        &ana,
        "SET PATH = system.builtin, spark_catalog.ns2, spark_catalog.default",
    );
    // A body calling `g` bare, through its frozen path; an invoker view calling `f` bare.
    ok(
        &ana,
        "CREATE FUNCTION default.af(x STRING) RETURNS BOOLEAN RETURN g(x)",
    );
    ok(
        &ana,
        "CREATE VIEW default.iv AS SELECT owner FROM default.t WHERE f(owner)",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvi SQL SECURITY DEFINER AS SELECT owner FROM default.iv",
    );
    ok(
        &ana,
        "CREATE VIEW default.dva SQL SECURITY DEFINER AS \
         SELECT owner FROM default.t WHERE default.af(owner)",
    );
    for v in ["dvi", "dva"] {
        ok(
            &ana,
            &format!("GRANT SELECT ON VIEW default.{v} TO user:carl"),
        );
        ok(
            &ana,
            &format!("GRANT SELECT ON VIEW default.{v} TO user:bob"),
        );
        assert_eq!(read(&carl, &format!("SELECT * FROM default.{v}")), ["ana"]);
    }
    // bob's `ns2.f` and `ns2.g`, earlier on both frozen paths, granted to ana.
    for f in ["f", "g"] {
        ok(
            &bob,
            &format!("CREATE FUNCTION ns2.{f}(x STRING) RETURNS BOOLEAN RETURN true"),
        );
        ok(
            &bob,
            &format!("GRANT SELECT ON FUNCTION ns2.{f} TO user:ana"),
        );
    }
    for v in ["dvi", "dva"] {
        for s in [&carl, &bob, &ana] {
            assert_redacted(
                s,
                &format!("SELECT * FROM default.{v}"),
                &["ns2", "`bob`", "`ana`", "carl"],
            );
        }
    }
    // The invoker view read by its own reader resolves at each read (Spark).
    assert_eq!(read(&ana, "SELECT * FROM default.iv"), ["ana", "carl"]);
    // bob drops them: the views read again.
    for f in ["f", "g"] {
        ok(&bob, &format!("DROP FUNCTION ns2.{f}"));
    }
    for v in ["dvi", "dva"] {
        assert_eq!(read(&carl, &format!("SELECT * FROM default.{v}")), ["ana"]);
    }
}

/// Review r10 m1 (a): with no grants provider no owner is recorded, so the body a definer view's
/// text calls is asked of the author of its pinned version too: bob losing `secret` refuses ana's
/// view over his body.
#[test]
fn a_body_with_no_recorded_owner_reads_as_its_author_too() {
    let w = World::new();
    let root = w.admin();
    let (ana, bob, carl) = (
        w.without_provider("ana"),
        w.without_provider("bob"),
        w.without_provider("carl"),
    );
    ok(&root, "CREATE TABLE secret (s STRING) USING parquet");
    ok(&root, "INSERT INTO secret VALUES ('S3CRET')");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    for u in ["ana", "bob"] {
        ok(&root, &format!("GRANT SELECT ON TABLE secret TO user:{u}"));
    }
    ok(
        &bob,
        "CREATE FUNCTION ns2.bf() RETURNS STRING RETURN (SELECT max(s) FROM default.secret)",
    );
    ok(&root, "GRANT SELECT ON FUNCTION ns2.bf TO user:ana");
    ok(
        &ana,
        "CREATE VIEW default.dvx SQL SECURITY DEFINER AS SELECT ns2.bf() AS v",
    );
    ok(&root, "GRANT SELECT ON VIEW default.dvx TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvx"), ["S3CRET"]);
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:bob");
    assert_redacted(&carl, "SELECT * FROM default.dvx", &["secret", "S3CRET"]);
}

/// Review r10 m1 (b): ana owns `g`, but bob (allowed to drop it) wrote its current version, which
/// ana's view pinned when she altered it. The body is asked of bob too: once he loses `secret`,
/// ana's view is refused, redacted.
#[test]
fn a_body_the_definer_owns_but_another_wrote_reads_as_its_author_too() {
    let w = World::new();
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "CREATE TABLE secret (s STRING) USING parquet");
    ok(&root, "INSERT INTO secret VALUES ('S3CRET')");
    for u in ["ana", "bob"] {
        ok(
            &root,
            &format!("GRANT CREATE ON SCHEMA default TO user:{u}"),
        );
        ok(&root, &format!("GRANT SELECT ON TABLE secret TO user:{u}"));
    }
    ok(
        &ana,
        "CREATE FUNCTION default.g() RETURNS STRING RETURN 'none'",
    );
    ok(&ana, "GRANT DROP ON FUNCTION default.g TO user:bob");
    ok(
        &ana,
        "CREATE VIEW default.dvg SQL SECURITY DEFINER AS SELECT default.g() AS v",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvg TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvg"), ["none"]);
    ok(
        &bob,
        "CREATE OR REPLACE FUNCTION default.g() RETURNS STRING RETURN (SELECT max(s) FROM default.secret)",
    );
    // Not the pinned version, nor one ana last defined.
    assert_redacted(&carl, "SELECT * FROM default.dvg", &["secret"]);
    ok(&ana, "ALTER VIEW default.dvg AS SELECT default.g() AS v");
    assert_eq!(read(&carl, "SELECT * FROM default.dvg"), ["S3CRET"]);
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:bob");
    assert_redacted(&carl, "SELECT * FROM default.dvg", &["secret", "S3CRET"]);
}

/// root's `pii (a INT, secret_col STRING)` = `(1, 'TOPSECRET')`, readable by ana and bob; ana may
/// create in `default`, bob in `ns2`, ana may use `ns2`.
fn pii_world(w: &World) -> (Session, Session, Session, Session) {
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(
        &root,
        "CREATE TABLE pii (a INT, secret_col STRING) USING parquet",
    );
    ok(&root, "INSERT INTO pii VALUES (1, 'TOPSECRET')");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    for u in ["ana", "bob"] {
        ok(&root, &format!("GRANT SELECT ON TABLE pii TO user:{u}"));
    }
    (root, ana, bob, carl)
}

/// root narrows bob from the whole of `pii` to its column `a`.
fn narrow_bob(root: &Session) {
    ok(root, "REVOKE SELECT ON TABLE pii FROM user:bob");
    ok(root, "GRANT SELECT (a) ON pii TO user:bob");
}

/// Review r11 B1 (function owner, who wrote it too): under the column-level grants policy the body
/// a definer view's text calls is asked of its owner per column read, as the definer is. Once root
/// narrows bob to `pii(a)`, ana's view over his `max(secret_col)` body is refused, redacted, for
/// carl and bob; the one over his `max(a)` body still reads `[1]`.
#[test]
fn a_bodys_owner_is_asked_the_columns_it_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = pii_world(&w);
    ok(
        &bob,
        "CREATE FUNCTION ns2.fs() RETURNS STRING RETURN (SELECT max(secret_col) FROM default.pii)",
    );
    ok(
        &bob,
        "CREATE FUNCTION ns2.fa() RETURNS INT RETURN (SELECT max(a) FROM default.pii)",
    );
    for f in ["fs", "fa"] {
        ok(
            &bob,
            &format!("GRANT SELECT ON FUNCTION ns2.{f} TO user:ana"),
        );
    }
    ok(
        &ana,
        "CREATE VIEW default.dvf SQL SECURITY DEFINER AS SELECT ns2.fs() AS c",
    );
    ok(
        &ana,
        "CREATE VIEW default.dva SQL SECURITY DEFINER AS SELECT ns2.fa() AS c",
    );
    for v in ["dvf", "dva"] {
        for u in ["bob", "carl"] {
            ok(
                &ana,
                &format!("GRANT SELECT ON VIEW default.{v} TO user:{u}"),
            );
        }
    }
    assert_eq!(read(&carl, "SELECT * FROM default.dvf"), ["TOPSECRET"]);
    narrow_bob(&root);
    for s in [&carl, &bob] {
        assert_redacted(
            s,
            "SELECT * FROM default.dvf",
            &["secret_col", "TOPSECRET", "fs"],
        );
        assert_eq!(read(s, "SELECT * FROM default.dva"), ["1"]);
    }
}

/// Review r11 B1 (function owner who did not write the current version): root (who owns `pii`)
/// replaced bob's function; bob still owns it and is asked the body's columns: narrowed to
/// `pii(a)`, ana's view over it is refused, redacted.
#[test]
fn a_bodys_owner_is_asked_its_columns_when_another_wrote_it() {
    let w = World::new();
    let (root, ana, bob, carl) = pii_world(&w);
    ok(
        &bob,
        "CREATE FUNCTION ns2.fs() RETURNS STRING RETURN 'none'",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.fs TO user:ana");
    ok(
        &root,
        "CREATE OR REPLACE FUNCTION ns2.fs() RETURNS STRING RETURN (SELECT max(secret_col) FROM default.pii)",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvf SQL SECURITY DEFINER AS SELECT ns2.fs() AS c",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvf TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvf"), ["TOPSECRET"]);
    narrow_bob(&root);
    assert_redacted(
        &carl,
        "SELECT * FROM default.dvf",
        &["secret_col", "TOPSECRET"],
    );
}

/// Review r11 B1 (function author): ana owns `g`, bob (allowed to drop it) replaced its body with
/// a read of `secret_col`, and ana pinned that version. bob is asked the body's columns: narrowed
/// to `pii(a)`, ana's view is refused, redacted.
#[test]
fn a_bodys_author_is_asked_the_columns_it_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = pii_world(&w);
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(
        &ana,
        "CREATE FUNCTION default.g() RETURNS STRING RETURN 'none'",
    );
    ok(&ana, "GRANT DROP ON FUNCTION default.g TO user:bob");
    ok(
        &bob,
        "CREATE OR REPLACE FUNCTION default.g() RETURNS STRING RETURN (SELECT max(secret_col) FROM default.pii)",
    );
    ok(
        &ana,
        "CREATE VIEW default.dvg SQL SECURITY DEFINER AS SELECT default.g() AS v",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvg TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvg"), ["TOPSECRET"]);
    narrow_bob(&root);
    assert_redacted(
        &carl,
        "SELECT * FROM default.dvg",
        &["secret_col", "TOPSECRET"],
    );
}

/// Review r12 n17: the owner of a function WITH an argument is asked the columns its body reads,
/// one of which depends on the argument. Narrowed to `pii(a)`, ana's view over it is refused,
/// redacted, for carl; once the owner's grant covers the columns again it answers.
#[test]
fn a_parameterised_bodys_owner_is_asked_the_columns_it_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = pii_world(&w);
    ok(
        &bob,
        "CREATE FUNCTION ns2.fx(x INT) RETURNS INT RETURN (SELECT max(CASE WHEN x > 0 THEN length(secret_col) ELSE a END) FROM default.pii)",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.fx TO user:ana");
    ok(
        &ana,
        "CREATE VIEW default.dvx SQL SECURITY DEFINER AS SELECT ns2.fx(1) AS c",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvx TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvx"), ["9"]);
    narrow_bob(&root);
    assert_redacted(
        &carl,
        "SELECT * FROM default.dvx",
        &["secret_col", "TOPSECRET"],
    );
    ok(&root, "GRANT SELECT (a, secret_col) ON pii TO user:bob");
    assert_eq!(read(&carl, "SELECT * FROM default.dvx"), ["9"]);
}

/// Review r12 n17: the same for a table function with an argument.
#[test]
fn a_parameterised_table_functions_owner_is_asked_the_columns_it_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = pii_world(&w);
    ok(
        &bob,
        "CREATE FUNCTION ns2.ft(x INT) RETURNS TABLE (c STRING) RETURN SELECT secret_col FROM default.pii WHERE a = x",
    );
    ok(&bob, "GRANT SELECT ON FUNCTION ns2.ft TO user:ana");
    ok(
        &ana,
        "CREATE VIEW default.dvt SQL SECURITY DEFINER AS SELECT * FROM ns2.ft(1)",
    );
    ok(&ana, "GRANT SELECT ON VIEW default.dvt TO user:carl");
    assert_eq!(read(&carl, "SELECT * FROM default.dvt"), ["TOPSECRET"]);
    narrow_bob(&root);
    assert_redacted(
        &carl,
        "SELECT * FROM default.dvt",
        &["secret_col", "TOPSECRET"],
    );
    ok(&root, "GRANT SELECT (a, secret_col) ON pii TO user:bob");
    assert_eq!(read(&carl, "SELECT * FROM default.dvt"), ["TOPSECRET"]);
}
