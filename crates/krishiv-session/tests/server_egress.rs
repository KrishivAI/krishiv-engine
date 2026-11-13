// SPDX-License-Identifier: Apache-2.0
//! P3-T067b acceptance: the server-mode egress and credential defaults, one test per default.
//! A server session's policy is `PathPrefixPolicy::from_config` around `RolePolicy` (ADR 0046
//! §3a), so ambient credentials and `http(s)://` locations reach only what
//! `krishiv.io.objectStore.allowedPrefixes` lists, and a store endpoint (per read, per write or
//! set in the session) needs `Manage` on the cluster, which only admin holds. Embedded use
//! (`AllowAll`) is unchanged.
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, PathPrefixPolicy, Principal, PrincipalKind, Privilege,
    Securable, Session,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";
const ROLES: [Role; 3] = [Role::Viewer, Role::Operator, Role::Admin];
const ENDPOINT: &str = "http://10.0.0.1:9000";

fn config(pairs: &[(&str, &str)]) -> Config {
    Config::from_pairs(pairs.iter().copied()).unwrap().0
}

fn policy(config: &Config, role: Role) -> PathPrefixPolicy {
    let roles = RolePolicy::new(Some(role), Role::Admin, PolicyVersion(1));
    PathPrefixPolicy::from_config(config, Arc::new(roles)).unwrap()
}

/// A server session of `role`, local reads confined to `root`, built with `builder` settings.
fn session(root: &std::path::Path, role: Role, builder: &[(&str, &str)]) -> Session {
    let conf = config(&[("krishiv.io.local.allowedRoots", root.to_str().unwrap())]);
    let mut b = Session::builder()
        .principal(Principal::new(role.as_str(), PrincipalKind::User))
        .access_policy(Arc::new(policy(&conf, role)));
    for (k, v) in builder {
        b = b.config(*k, *v);
    }
    b.build().unwrap()
}

/// A directory holding `a.csv` with two rows.
fn data() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("a.csv"), "1\n2\n").unwrap();
    tmp
}

fn outcome(r: Result<usize, KrishivError>) -> Result<usize, String> {
    r.map_err(|e| match e.error_class().name() {
        DENIED => DENIED.to_owned(),
        _ => e.to_string(),
    })
}

fn rows(s: &Session, sql: &str) -> Result<usize, String> {
    outcome(block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
    }))
}

fn read_with(s: &Session, path: &str, options: &[(&str, &str)]) -> Result<usize, String> {
    let mut reader = s.read().format("csv");
    for (k, v) in options {
        reader = reader.option(k, v);
    }
    let df = reader.load([path]);
    outcome(block_on(async {
        let batches = df.collect().await?;
        Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
    }))
}

/// The policy's decision on reading `path`.
fn allows_read(policy: &PathPrefixPolicy, path: &str) -> bool {
    let principal = Principal::new("p", PrincipalKind::User);
    let securable = Securable::Path(OptionValue::from_user("path", path));
    matches!(
        block_on(policy.check(&principal, &securable, Privilege::Select)).unwrap(),
        AccessDecision::Allow
    )
}

/// Ambient (server) object-store credentials are used only under `allowedPrefixes`: a location
/// anywhere else is refused before a store is built, for every role. (Vended catalog
/// credentials, T067, are not in yet.)
#[test]
fn ambient_credentials_reach_only_allowed_prefixes() {
    let conf = config(&[("krishiv.io.objectStore.allowedPrefixes", "s3://lake/shared")]);
    for role in ROLES {
        let p = policy(&conf, role);
        for path in [
            "s3://lake/private/t/part-0.parquet",
            "s3://lake/sharedx/part-0.parquet",
            "s3://other/shared/part-0.parquet",
            "gs://lake/shared/part-0.parquet",
        ] {
            assert!(!allows_read(&p, path), "{role:?} {path}");
        }
        // Inside the list the role decides: viewers never read rows.
        assert_eq!(
            allows_read(&p, "s3a://lake/shared/t/part-0.parquet"),
            role != Role::Viewer,
            "{role:?}"
        );
    }
    let tmp = data();
    for role in ROLES {
        let s = session(tmp.path(), role, &[]);
        assert_eq!(
            rows(&s, "SELECT * FROM parquet.`s3://lake/private/t`"),
            Err(DENIED.to_owned()),
            "{role:?}"
        );
    }
}

/// A store endpoint in the read options is `Manage` on the cluster: 42501 for viewer and
/// operator, whichever endpoint key and case; admin reads.
#[test]
fn an_endpoint_read_option_is_admin_only() {
    let tmp = data();
    let path = tmp.path().join("a.csv");
    let path = path.to_str().unwrap();
    for key in krishiv_io::ENDPOINT_KEYS {
        let upper = key.to_ascii_uppercase();
        for k in [key, upper.as_str()] {
            for role in [Role::Viewer, Role::Operator] {
                let s = session(tmp.path(), role, &[]);
                assert_eq!(
                    read_with(&s, path, &[(k, ENDPOINT)]),
                    Err(DENIED.to_owned()),
                    "{role:?} {k}"
                );
            }
            let admin = session(tmp.path(), Role::Admin, &[]);
            assert_eq!(read_with(&admin, path, &[(k, ENDPOINT)]), Ok(2), "{k}");
        }
    }
    // Other store options are not endpoints.
    let operator = session(tmp.path(), Role::Operator, &[]);
    assert_eq!(
        read_with(&operator, path, &[("fs.s3a.region", "eu-west-1")]),
        Ok(2)
    );
    assert_eq!(read_with(&operator, path, &[]), Ok(2));
}

/// A store endpoint the session sets (SQL `SET`, `spark.conf.set`) is `Manage` on the cluster for
/// every query after it until it is reset; one the server configured for the session is not.
#[test]
fn an_endpoint_set_in_the_session_is_admin_only() {
    let tmp = data();
    let read = format!("SELECT * FROM csv.`{}`", tmp.path().join("a.csv").display());

    let operator = session(tmp.path(), Role::Operator, &[]);
    assert_eq!(rows(&operator, &read), Ok(2));
    // SQL `SET` is refused before the value is stored.
    assert_eq!(
        rows(&operator, &format!("SET fs.s3a.endpoint = {ENDPOINT}")),
        Err(DENIED.to_owned())
    );
    assert_eq!(operator.conf().get("fs.s3a.endpoint"), None);
    assert_eq!(rows(&operator, &read), Ok(2));
    // `spark.conf.set` (Spark Connect, Python) stores it; every query after it is refused.
    operator.conf().set("fs.s3a.endpoint", ENDPOINT).unwrap();
    assert_eq!(rows(&operator, &read), Err(DENIED.to_owned()));
    assert_eq!(rows(&operator, "SELECT 1"), Err(DENIED.to_owned()));
    operator.conf().unset("fs.s3a.endpoint").unwrap();
    assert_eq!(rows(&operator, &read), Ok(2));
    operator.conf().set("FS.HF.ENDPOINT", ENDPOINT).unwrap();
    assert_eq!(rows(&operator, &read), Err(DENIED.to_owned()));
    rows(&operator, "RESET").unwrap();
    assert_eq!(rows(&operator, &read), Ok(2));

    let admin = session(tmp.path(), Role::Admin, &[]);
    assert_eq!(
        rows(&admin, &format!("SET fs.s3a.endpoint = {ENDPOINT}")),
        Ok(1)
    );
    assert_eq!(rows(&admin, &read), Ok(2));

    // The server's own endpoint (the session's builder configuration) needs nothing; changing it
    // does.
    let configured = session(tmp.path(), Role::Operator, &[("fs.s3a.endpoint", ENDPOINT)]);
    assert_eq!(rows(&configured, &read), Ok(2));
    configured
        .conf()
        .set("fs.s3a.endpoint", "http://10.0.0.2:9000")
        .unwrap();
    assert_eq!(rows(&configured, &read), Err(DENIED.to_owned()));
    configured.conf().set("fs.s3a.endpoint", ENDPOINT).unwrap();
    assert_eq!(rows(&configured, &read), Ok(2));
}

/// Writes every path but holds no `Manage`.
struct WriterWithoutManage;

impl AccessPolicy for WriterWithoutManage {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a Principal,
        _: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        Box::pin(async move {
            Ok(if privilege == Privilege::Manage {
                AccessDecision::Deny {
                    reason: "no Manage".to_owned(),
                }
            } else {
                AccessDecision::Allow
            })
        })
    }
}

/// A store endpoint in the write options is `Manage` on the cluster too, asked before anything
/// is written.
#[test]
fn an_endpoint_write_option_needs_manage() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .principal(Principal::new("w", PrincipalKind::User))
        .access_policy(Arc::new(WriterWithoutManage))
        .build()
        .unwrap();
    let out = tmp.path().join("out");
    let write = |options: &[(&str, &str)]| {
        let mut w = block_on(s.sql("SELECT 1 AS a"))
            .unwrap()
            .write()
            .mode("overwrite");
        for (k, v) in options {
            w = w.option(k, v);
        }
        outcome(block_on(w.csv(out.to_str().unwrap())).map(|()| 0))
    };
    assert_eq!(
        write(&[("fs.azure.endpoint", ENDPOINT)]),
        Err(DENIED.to_owned())
    );
    assert!(!out.exists(), "nothing written before the check");
    assert_eq!(write(&[]), Ok(0));
}

/// `http(s)://` locations are refused for every role until an admin lists one in
/// `allowedPrefixes` (the one opt-in); then the role decides.
#[test]
fn http_locations_are_off_until_listed() {
    let tmp = data();
    for role in ROLES {
        let s = session(tmp.path(), role, &[]);
        for path in [
            "https://example.com/data/a.csv",
            "http://127.0.0.1:8080/a.csv",
        ] {
            assert_eq!(
                rows(&s, &format!("SELECT * FROM csv.`{path}`")),
                Err(DENIED.to_owned()),
                "{role:?} {path}"
            );
        }
    }
    let listed = config(&[(
        "krishiv.io.objectStore.allowedPrefixes",
        "https://example.com/data",
    )]);
    for role in ROLES {
        let p = policy(&listed, role);
        assert_eq!(
            allows_read(&p, "https://example.com/data/a.csv"),
            role != Role::Viewer,
            "{role:?}"
        );
        assert!(
            !allows_read(&p, "https://example.com/other/a.csv"),
            "{role:?}"
        );
        assert!(
            !allows_read(&p, "https://example.org/data/a.csv"),
            "{role:?}"
        );
    }
}

/// Embedded use is unchanged: no confinement, endpoints per read and in the session work.
#[test]
fn embedded_sessions_are_unchanged() {
    let tmp = data();
    let path = tmp.path().join("a.csv");
    let path = path.to_str().unwrap();
    let s = Session::builder()
        .access_policy(Arc::new(AllowAll))
        .build()
        .unwrap();
    assert_eq!(read_with(&s, path, &[("fs.s3a.endpoint", ENDPOINT)]), Ok(2));
    s.conf().set("fs.s3a.endpoint", ENDPOINT).unwrap();
    assert_eq!(rows(&s, &format!("SELECT * FROM csv.`{path}`")), Ok(2));
    let s = Session::new().unwrap();
    s.conf().set("fs.gs.endpoint", ENDPOINT).unwrap();
    assert_eq!(rows(&s, "SELECT 1"), Ok(1));
}
