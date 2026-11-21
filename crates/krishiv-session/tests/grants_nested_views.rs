// SPDX-License-Identifier: Apache-2.0
//! P3-T062 unit G3 (ADR 0048 A1 §3) under the `grants` provider: a stored view another principal
//! owns, nested in a definer view's trusted text, gets the function rule (review r10 M10). The
//! definer view pins the version of every view it reaches when created or altered; a read must
//! find that version or a view the definer owns and last defined, else it fails closed, redacted.
//! A nested view's reads are asked of its owner and author too, so a revoked owner reads nothing
//! more through it. Invoker views read directly keep Spark's behaviour.
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

/// root's table `t` (ana, carl) and `secret`, both granted to ana and bob; bob may create in
/// `ns2`, ana in `default` and may use `ns2`.
fn base(w: &World) -> (Session, Session, Session, Session) {
    let (root, ana, bob, carl) = (w.admin(), w.user("ana"), w.user("bob"), w.user("carl"));
    ok(&root, "CREATE TABLE t (owner STRING, a INT) USING parquet");
    ok(&root, "INSERT INTO t VALUES ('ana', 1), ('carl', 5)");
    ok(&root, "CREATE TABLE secret (s STRING) USING parquet");
    ok(&root, "INSERT INTO secret VALUES ('S3CRET')");
    ok(&root, "CREATE NAMESPACE ns2");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:ana");
    ok(&root, "GRANT CREATE ON SCHEMA ns2 TO user:bob");
    ok(&root, "GRANT USAGE ON SCHEMA ns2 TO user:ana");
    for u in ["ana", "bob"] {
        ok(&root, &format!("GRANT SELECT ON TABLE t TO user:{u}"));
        ok(&root, &format!("GRANT SELECT ON TABLE secret TO user:{u}"));
    }
    (root, ana, bob, carl)
}

/// ana's definer view `default.{v}` over `q`, readable by bob and carl.
fn definer_view(ana: &Session, v: &str, q: &str) {
    ok(
        ana,
        &format!("CREATE VIEW default.{v} SQL SECURITY DEFINER AS {q}"),
    );
    for u in ["bob", "carl"] {
        ok(
            ana,
            &format!("GRANT SELECT ON VIEW default.{v} TO user:{u}"),
        );
    }
}

/// Review r10 M10, probe q6: bob replacing his invoker view nested in ana's definer view never
/// changes what ana's view returns: the view is refused, redacted, for every reader (ana too)
/// until ana alters it, which pins the new version. bob's view read directly follows his replace,
/// as Spark's does.
#[test]
fn a_nested_view_replaced_by_another_principal_fails_the_definer_view_closed() {
    let w = World::new();
    let (_root, ana, bob, carl) = base(&w);
    ok(
        &bob,
        "CREATE VIEW ns2.bv AS SELECT owner FROM default.t WHERE owner = 'ana'",
    );
    ok(&bob, "GRANT SELECT ON VIEW ns2.bv TO user:ana");
    definer_view(&ana, "dvv", "SELECT * FROM ns2.bv");
    assert_eq!(read(&carl, "SELECT * FROM default.dvv"), ["ana"]);
    ok(
        &bob,
        "CREATE OR REPLACE VIEW ns2.bv AS SELECT owner FROM default.t",
    );
    for s in [&carl, &bob, &ana] {
        assert_redacted(s, "SELECT * FROM default.dvv", &["bv", "bob", "ns2"]);
    }
    // bob's view read directly is an invoker view: it follows his replace (Spark).
    let mut direct = read(&ana, "SELECT * FROM ns2.bv");
    direct.sort();
    assert_eq!(direct, ["ana", "carl"]);
    // ana's alter pins the new version.
    ok(&ana, "ALTER VIEW default.dvv AS SELECT * FROM ns2.bv");
    let mut after = read(&carl, "SELECT * FROM default.dvv");
    after.sort();
    assert_eq!(after, ["ana", "carl"]);
    // `ALTER VIEW .. AS` with the same text is a new version too.
    ok(&bob, "ALTER VIEW ns2.bv AS SELECT owner FROM default.t");
    assert_redacted(&carl, "SELECT * FROM default.dvv", &["bv", "bob"]);
}

/// Review r10 M10, probe q8: a nested view's reads are asked of its owner as well as the definer.
/// Once root revokes bob's `Select` on `secret`, ana's definer views over bob's view (directly, or
/// through ana's own function body) are refused, redacted; before, they read it.
#[test]
fn a_nested_views_reads_are_asked_of_its_owner_too() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    ok(&bob, "CREATE VIEW ns2.bs AS SELECT s FROM default.secret");
    ok(&bob, "GRANT SELECT ON VIEW ns2.bs TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION default.cnt() RETURNS STRING RETURN (SELECT max(s) FROM ns2.bs)",
    );
    definer_view(&ana, "dvs", "SELECT * FROM ns2.bs");
    definer_view(&ana, "dvf", "SELECT default.cnt() AS c");
    for s in [&carl, &bob] {
        assert_eq!(read(s, "SELECT * FROM default.dvs"), ["S3CRET"]);
        assert_eq!(read(s, "SELECT * FROM default.dvf"), ["S3CRET"]);
    }
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:bob");
    assert!(rows(&bob, "SELECT * FROM default.secret").is_err());
    for s in [&carl, &bob] {
        assert_redacted(s, "SELECT * FROM default.dvs", &["secret", "bob", "bs"]);
        assert_redacted(s, "SELECT * FROM default.dvf", &["secret", "bob", "bs"]);
    }
}

/// Review r10 M10, probe q8 as written: bob replaces his view (while he holds `secret`), then loses
/// `secret`; both the version pin and the owner's reads refuse ana's views.
#[test]
fn a_replaced_nested_view_of_a_revoked_owner_is_refused() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    ok(
        &bob,
        "CREATE VIEW ns2.bs AS SELECT s FROM default.secret WHERE false",
    );
    ok(&bob, "GRANT SELECT ON VIEW ns2.bs TO user:ana");
    ok(
        &ana,
        "CREATE FUNCTION default.cnt() RETURNS STRING RETURN (SELECT max(s) FROM ns2.bs)",
    );
    definer_view(&ana, "dvs", "SELECT * FROM ns2.bs");
    definer_view(&ana, "dvf", "SELECT default.cnt() AS c");
    assert!(read(&carl, "SELECT * FROM default.dvs").is_empty());
    ok(
        &bob,
        "CREATE OR REPLACE VIEW ns2.bs AS SELECT s FROM default.secret",
    );
    ok(&root, "REVOKE SELECT ON TABLE secret FROM user:bob");
    for s in [&carl, &bob] {
        assert_redacted(s, "SELECT * FROM default.dvs", &["secret", "S3CRET"]);
        assert_redacted(s, "SELECT * FROM default.dvf", &["secret", "S3CRET"]);
    }
}

/// The definer replacing her own nested invoker view is followed (rule (a): a view the definer owns
/// and last defined), and a nested definer view of another principal is pinned the same way while
/// still reading as its own definer.
#[test]
fn the_definers_own_nested_view_is_followed_and_a_nested_definer_view_is_pinned() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    ok(
        &ana,
        "CREATE VIEW default.av AS SELECT owner FROM default.t WHERE owner = 'ana'",
    );
    definer_view(&ana, "dva", "SELECT * FROM default.av");
    assert_eq!(read(&carl, "SELECT * FROM default.dva"), ["ana"]);
    ok(
        &ana,
        "CREATE OR REPLACE VIEW default.av AS SELECT owner FROM default.t WHERE owner = 'carl'",
    );
    assert_eq!(read(&carl, "SELECT * FROM default.dva"), ["carl"]);

    ok(
        &bob,
        "CREATE VIEW ns2.bd SQL SECURITY DEFINER AS SELECT owner FROM default.t WHERE owner = 'ana'",
    );
    ok(&bob, "GRANT SELECT ON VIEW ns2.bd TO user:ana");
    definer_view(&ana, "dvd", "SELECT * FROM ns2.bd");
    assert_eq!(read(&carl, "SELECT * FROM default.dvd"), ["ana"]);
    // A nested definer view reads as its own definer: ana losing `t` changes nothing.
    ok(&root, "REVOKE SELECT ON TABLE t FROM user:ana");
    assert_eq!(read(&carl, "SELECT * FROM default.dvd"), ["ana"]);
    ok(
        &bob,
        "CREATE OR REPLACE VIEW ns2.bd SQL SECURITY DEFINER AS SELECT owner FROM default.t",
    );
    for s in [&carl, &bob, &ana] {
        assert_redacted(s, "SELECT * FROM default.dvd", &["bd", "bob"]);
    }
}

/// root's `pii (a INT, secret_col STRING)` = `(1, 'TOPSECRET')`, readable by ana and `writer`.
fn pii(root: &Session, writer: &str) {
    ok(
        root,
        "CREATE TABLE pii (a INT, secret_col STRING) USING parquet",
    );
    ok(root, "INSERT INTO pii VALUES (1, 'TOPSECRET')");
    for u in ["ana", writer] {
        ok(root, &format!("GRANT SELECT ON TABLE pii TO user:{u}"));
    }
}

/// root narrows `writer` from the whole of `pii` to its column `a` (a PII-hardening step).
fn narrow(root: &Session, writer: &str) {
    ok(
        root,
        &format!("REVOKE SELECT ON TABLE pii FROM user:{writer}"),
    );
    ok(root, &format!("GRANT SELECT (a) ON pii TO user:{writer}"));
}

/// Review r11 B1 (view owner, who wrote it too): under the column-level grants policy a nested
/// view's owner is asked `Select` on exactly the columns its text reads, as the definer is, never
/// only whether the table may be seen. Once root narrows bob to `pii(a)`, ana's definer views over
/// bob's `SELECT secret_col` and `SELECT *` views are refused, redacted, for carl and bob; the one
/// over bob's `SELECT a` view still reads `[1]`.
#[test]
fn a_nested_views_owner_is_asked_the_columns_its_text_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    pii(&root, "bob");
    ok(
        &bob,
        "CREATE VIEW ns2.bv AS SELECT secret_col FROM default.pii",
    );
    ok(&bob, "CREATE VIEW ns2.bstar AS SELECT * FROM default.pii");
    ok(&bob, "CREATE VIEW ns2.ba AS SELECT a FROM default.pii");
    for v in ["bv", "bstar", "ba"] {
        ok(&bob, &format!("GRANT SELECT ON VIEW ns2.{v} TO user:ana"));
    }
    definer_view(&ana, "dv", "SELECT * FROM ns2.bv");
    definer_view(&ana, "dvstar", "SELECT * FROM ns2.bstar");
    definer_view(&ana, "dva", "SELECT * FROM ns2.ba");
    assert_eq!(read(&carl, "SELECT * FROM default.dv"), ["TOPSECRET"]);
    assert_eq!(read(&carl, "SELECT * FROM default.dvstar"), ["1|TOPSECRET"]);
    narrow(&root, "bob");
    assert!(rows(&bob, "SELECT secret_col FROM default.pii").is_err());
    for s in [&carl, &bob] {
        assert_redacted(
            s,
            "SELECT * FROM default.dv",
            &["secret_col", "TOPSECRET", "bv"],
        );
        assert_redacted(
            s,
            "SELECT * FROM default.dvstar",
            &["secret_col", "TOPSECRET", "bstar"],
        );
        // bob's column grant covers exactly what `ba` reads.
        assert_eq!(read(s, "SELECT * FROM default.dva"), ["1"]);
    }
}

/// Review r11 B1, the reverse case: bob only ever held `pii(a)`, and his view reads only `a`: ana's
/// definer view over it reads `[1]` for carl and ana.
#[test]
fn a_nested_view_within_its_owners_column_grant_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    ok(
        &root,
        "CREATE TABLE pii (a INT, secret_col STRING) USING parquet",
    );
    ok(&root, "INSERT INTO pii VALUES (1, 'TOPSECRET')");
    ok(&root, "GRANT SELECT ON TABLE pii TO user:ana");
    ok(&root, "GRANT SELECT (a) ON pii TO user:bob");
    ok(&bob, "CREATE VIEW ns2.ba AS SELECT a FROM default.pii");
    ok(&bob, "GRANT SELECT ON VIEW ns2.ba TO user:ana");
    definer_view(&ana, "dva", "SELECT * FROM ns2.ba");
    for s in [&carl, &ana] {
        assert_eq!(read(s, "SELECT * FROM default.dva"), ["1"]);
    }
}

/// Review r11 B1 (view owner who did not write the current version): root (an admin, who owns
/// `pii`) altered bob's view; bob still owns it, so he is asked its columns too: narrowed to
/// `pii(a)`, ana's definer view over it is refused, redacted.
#[test]
fn a_nested_views_owner_is_asked_its_columns_when_another_wrote_it() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    pii(&root, "bob");
    ok(&bob, "CREATE VIEW ns2.bv AS SELECT a FROM default.pii");
    ok(&bob, "GRANT SELECT ON VIEW ns2.bv TO user:ana");
    ok(
        &root,
        "ALTER VIEW ns2.bv AS SELECT secret_col FROM default.pii",
    );
    definer_view(&ana, "dv", "SELECT * FROM ns2.bv");
    assert_eq!(read(&carl, "SELECT * FROM default.dv"), ["TOPSECRET"]);
    narrow(&root, "bob");
    for s in [&carl, &bob] {
        assert_redacted(
            s,
            "SELECT * FROM default.dv",
            &["secret_col", "TOPSECRET", "bv"],
        );
    }
}

/// Review r11 B1 (view author): ana owns `av`, bob (allowed to drop it) replaced its text with a
/// read of `secret_col`, and ana pinned that version. bob wrote the text, so he is asked its
/// columns: narrowed to `pii(a)`, ana's definer view over it is refused, redacted.
#[test]
fn a_nested_views_author_is_asked_the_columns_its_text_reads() {
    let w = World::new();
    let (root, ana, bob, carl) = base(&w);
    pii(&root, "bob");
    ok(&root, "GRANT CREATE ON SCHEMA default TO user:bob");
    ok(&ana, "CREATE VIEW default.av AS SELECT a FROM default.pii");
    ok(&ana, "GRANT DROP ON VIEW default.av TO user:bob");
    ok(
        &bob,
        "CREATE OR REPLACE VIEW default.av AS SELECT secret_col FROM default.pii",
    );
    definer_view(&ana, "dv", "SELECT * FROM default.av");
    assert_eq!(read(&carl, "SELECT * FROM default.dv"), ["TOPSECRET"]);
    narrow(&root, "bob");
    assert_redacted(
        &carl,
        "SELECT * FROM default.dv",
        &["secret_col", "TOPSECRET"],
    );
}
