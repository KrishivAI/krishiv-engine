// SPDX-License-Identifier: Apache-2.0
//! `path_confinement` (P3-T002 acceptance; ADR 0046 §3a): in server mode a path is allowed only
//! under `krishiv.io.local.allowedRoots` or `krishiv.io.objectStore.allowedPrefixes` (default
//! empty), for every role, admin included. The confinement is `PathPrefixPolicy` around
//! `RolePolicy`, so the role still decides inside the lists.
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::config::Config;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::{
    AccessDecision, AccessPolicy, AllowAll, PathPrefixPolicy, Principal, PrincipalKind, Privilege,
    Securable, Session,
};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";
const ROLES: [Role; 3] = [Role::Viewer, Role::Operator, Role::Admin];

fn config(pairs: &[(&str, &str)]) -> Config {
    Config::from_pairs(pairs.iter().copied()).unwrap().0
}

/// The server-mode policy of a principal with `role` under `config`.
fn policy(config: &Config, role: Role) -> PathPrefixPolicy {
    let roles = RolePolicy::new(Some(role), Role::Admin, PolicyVersion(1));
    PathPrefixPolicy::from_config(config, Arc::new(roles)).unwrap()
}

fn session(config: &Config, role: Role) -> Session {
    Session::builder()
        .principal(Principal::new(role.as_str(), PrincipalKind::User))
        .access_policy(Arc::new(policy(config, role)))
        .build()
        .unwrap()
}

fn run(s: &Session, sql: &str) -> Result<(), String> {
    block_on(async {
        let df = s.sql(sql).await?;
        df.collect().await.map(|_| ())
    })
    .map_err(|e| e.to_string())
}

fn read(path: &str) -> String {
    format!("SELECT * FROM csv.`{path}`")
}

/// The policy's decision on `privilege` over the path `path`.
fn decide(policy: &PathPrefixPolicy, path: &str, privilege: Privilege) -> bool {
    let principal = Principal::new("p", PrincipalKind::User);
    let securable = Securable::Path(OptionValue::from_user("path", path));
    matches!(
        block_on(policy.check(&principal, &securable, privilege)).unwrap(),
        AccessDecision::Allow
    )
}

/// A directory `allowed` holding `a.csv`, and a sibling `secret` holding `s.csv`.
fn dirs() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let allowed = tmp.path().join("allowed");
    let secret = tmp.path().join("secret");
    std::fs::create_dir(&allowed).unwrap();
    std::fs::create_dir(&secret).unwrap();
    std::fs::write(allowed.join("a.csv"), "1\n2\n").unwrap();
    std::fs::write(secret.join("s.csv"), "3\n").unwrap();
    (tmp, allowed, secret)
}

#[test]
fn path_confinement() {
    let (_tmp, allowed, secret) = dirs();
    let conf = config(&[
        ("krishiv.io.local.allowedRoots", allowed.to_str().unwrap()),
        (
            "krishiv.io.objectStore.allowedPrefixes",
            "s3://lake/data, gs://shared",
        ),
    ]);
    let inside = allowed.join("a.csv");
    let escape = allowed.join("..").join("secret").join("s.csv");
    let link = allowed.join("link");
    std::os::unix::fs::symlink(&secret, &link).unwrap();
    let through_link = link.join("s.csv");
    for role in ROLES {
        let s = session(&conf, role);
        // Outside the roots: refused for every role, admin included.
        for path in [secret.join("s.csv"), escape.clone(), through_link.clone()] {
            let e = run(&s, &read(path.to_str().unwrap())).unwrap_err();
            assert!(e.contains(DENIED), "{role} reading {}: {e}", path.display());
        }
        for path in [
            "s3://lake/data2/x.csv",
            "s3://lake/x.csv",
            "s3://other/data/x.csv",
            "gs://other/x.csv",
            "https://example.com/x.csv",
        ] {
            let e = run(&s, &read(path)).unwrap_err();
            assert!(e.contains(DENIED), "{role} reading {path}: {e}");
        }
        // Inside the roots the role decides: a viewer reads no path data (ADR 0046 §3).
        let inside = run(&s, &read(inside.to_str().unwrap()));
        match role {
            Role::Viewer => assert!(inside.unwrap_err().contains(DENIED)),
            _ => inside.unwrap_or_else(|e| panic!("{role} reading inside the root: {e}")),
        }
        // Object-store prefixes, decided without touching the store.
        let p = policy(&conf, role);
        let operator = role >= Role::Operator;
        for path in [
            "s3://lake/data/x.csv",
            "s3a://lake/data/y/z.parquet",
            "s3://lake/data/*.csv",
            "gs://shared/any/key",
        ] {
            assert_eq!(
                decide(&p, path, Privilege::Select),
                operator,
                "{role}: {path}"
            );
        }
        for path in [
            "s3://lake/data2/x",
            "s3://lake/dat*/x",
            "s3://lake/data/../x",
            "s3://lake/data/%2e%2e/x",
            "s3://lake//data/x",
            "s3://other/data/x",
            "gs://shared2/x",
            "file:relative",
        ] {
            assert!(!decide(&p, path, Privilege::Select), "{role}: {path}");
        }
    }
}

#[test]
fn defaults_allow_no_path_for_any_role() {
    let (_tmp, allowed, _) = dirs();
    let conf = config(&[]);
    for role in ROLES {
        let s = session(&conf, role);
        let e = run(&s, &read(allowed.join("a.csv").to_str().unwrap())).unwrap_err();
        assert!(e.contains(DENIED), "{role}: {e}");
        let p = policy(&conf, role);
        assert!(!decide(&p, "s3://lake/data/x", Privilege::Select));
        // Statements that read no path keep working.
        if role >= Role::Operator {
            run(&s, "SELECT 1").unwrap();
        }
    }
}

#[test]
fn a_bad_prefix_or_root_refuses_startup() {
    for prefix in [
        "s3://",
        "s3:///data",
        "s3://key:secret@lake/data",
        "s3://lake/../other",
        "s3://lake/./data",
        "s3://lake//data",
        "s3://lake/%2e%2e",
        "s3://lake/data?x=1",
        "file:/tmp",
        "/tmp",
        "ftp://host/x",
    ] {
        let e = PathPrefixPolicy::new(Vec::<&str>::new(), Arc::new(AllowAll))
            .unwrap()
            .with_object_prefixes([prefix])
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_CONFIG_VALUE",
            "{prefix}"
        );
    }
    let conf = config(&[("krishiv.io.local.allowedRoots", "/no/such/dir/krishiv")]);
    let e = PathPrefixPolicy::from_config(&conf, Arc::new(AllowAll)).unwrap_err();
    assert_eq!(e.error_class().name(), "PATH_NOT_FOUND");
    // A whole bucket and a trailing slash are fine.
    let p = PathPrefixPolicy::new(Vec::<&str>::new(), Arc::new(AllowAll))
        .unwrap()
        .with_object_prefixes(["s3://lake", "gs://shared/data/"])
        .unwrap();
    assert!(decide(&p, "s3://lake/x", Privilege::Select));
    assert!(decide(&p, "gs://shared/data/x", Privilege::Modify));
    assert!(!decide(&p, "gs://shared/datax", Privilege::Select));
}
