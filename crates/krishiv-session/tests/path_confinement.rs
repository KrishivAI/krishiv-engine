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
        // Glob characters would match more than whole segments.
        "s3://lake/da*",
        "s3://lake/{data,secret}",
        "s3://lake/data[0]",
        // An Azure prefix names its account host.
        "abfss://exports/public",
        "abfss://exports@lakeacct/public",
        "az://exports/public",
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
    let file = tempfile::NamedTempFile::new().unwrap();
    for root in [
        "/no/such/dir/krishiv",
        "relative/dir",
        file.path().to_str().unwrap(),
    ] {
        let conf = config(&[("krishiv.io.local.allowedRoots", root)]);
        let e = PathPrefixPolicy::from_config(&conf, Arc::new(AllowAll)).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_CONFIG_VALUE",
            "{root}"
        );
        assert!(
            e.to_string().contains("krishiv.io.local.allowedRoots"),
            "{e}"
        );
        let e = Session::builder()
            .config("krishiv.io.local.allowedRoots", root)
            .build()
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_CONFIG_VALUE",
            "{root}"
        );
    }
    // A whole bucket and a trailing slash are fine.
    let p = PathPrefixPolicy::new(Vec::<&str>::new(), Arc::new(AllowAll))
        .unwrap()
        .with_object_prefixes(["s3://lake", "gs://shared/data/"])
        .unwrap();
    assert!(decide(&p, "s3://lake/x", Privilege::Select));
    assert!(decide(&p, "gs://shared/data/x", Privilege::Modify));
    assert!(!decide(&p, "gs://shared/datax", Privilege::Select));
}

/// B1 of the T002 confinement review: an Azure prefix allows its container in its account only;
/// a path without an account host (whose account would come from the client's
/// `fs.azure.account.name`) or in another account is refused.
#[test]
fn azure_prefixes_hold_their_account() {
    let p = PathPrefixPolicy::new(Vec::<&str>::new(), Arc::new(AllowAll))
        .unwrap()
        .with_object_prefixes(["abfss://exports@lakeacct.dfs.core.windows.net/public"])
        .unwrap();
    assert!(decide(
        &p,
        "abfss://exports@lakeacct.dfs.core.windows.net/public/t/x.parquet",
        Privilege::Select
    ));
    for path in [
        "abfss://exports/public/x.parquet",
        "az://exports/public/x.parquet",
        "abfss://exports:key@lakeacct.dfs.core.windows.net/public/x.parquet",
        "abfss://exports@hracct.dfs.core.windows.net/public/x.parquet",
        "abfss://other@lakeacct.dfs.core.windows.net/public/x.parquet",
        "abfss://exports@lakeacct.dfs.core.windows.net/private/x.parquet",
    ] {
        assert!(!decide(&p, path, Privilege::Select), "{path}");
    }
}

/// More prefix shapes: `s3a` and `s3` are one store either way, the scheme's case does not
/// matter, a bucket name is whole, blank list entries are dropped.
#[test]
fn prefix_matching_edges() {
    let conf = config(&[(
        "krishiv.io.objectStore.allowedPrefixes",
        " , s3a://lake/data ,gs://shared, ",
    )]);
    let p = policy(&conf, Role::Admin);
    assert!(decide(&p, "s3://lake/data/x", Privilege::Select));
    assert!(decide(&p, "S3://lake/data/x", Privilege::Select));
    assert!(decide(&p, "s3a://lake/data/x", Privilege::Modify));
    assert!(decide(&p, "gs://shared/x", Privilege::Create));
    assert!(!decide(&p, "gs://shared.other/x", Privilege::Select));
    assert!(!decide(&p, "s3://lake/data2/x", Privilege::Select));
}

/// B2 of the T002 confinement review: a glob or directory read below a root never follows a
/// link out of it, whether the link is a directory, a partition directory or a file.
#[cfg(unix)]
#[test]
fn listings_never_follow_a_link_out_of_a_root() {
    let (_tmp, allowed, secret) = dirs();
    let root = allowed.to_str().unwrap();
    let conf = config(&[("krishiv.io.local.allowedRoots", root)]);
    let s = Session::builder()
        .config("krishiv.io.local.allowedRoots", root)
        .principal(Principal::new("admin", PrincipalKind::User))
        .access_policy(Arc::new(policy(&conf, Role::Admin)))
        .build()
        .unwrap();
    assert!(run(&s, &read(&format!("{root}/a.csv"))).is_ok());

    std::os::unix::fs::symlink(&secret, allowed.join("link")).unwrap();
    for path in [
        format!("{root}/link/s.csv"),
        format!("{root}/*/s.csv"),
        format!("{root}/{{link,x}}/s.csv"),
        format!("{root}/l*"),
    ] {
        let e = run(&s, &read(&path)).unwrap_err();
        assert!(
            e.contains(DENIED) || e.contains("a link out of the allowed roots"),
            "{path}: {e}"
        );
    }
    std::fs::remove_file(allowed.join("link")).unwrap();

    // A partition-directory link: a plain directory read of the root.
    std::fs::create_dir(secret.join("k=1")).unwrap();
    std::fs::write(secret.join("k=1/s.csv"), "4\n").unwrap();
    let table = allowed.join("t");
    std::fs::create_dir(&table).unwrap();
    std::os::unix::fs::symlink(secret.join("k=1"), table.join("k=1")).unwrap();
    let e = run(&s, &read(table.to_str().unwrap())).unwrap_err();
    assert!(e.contains("a link out of the allowed roots"), "{e}");
    std::fs::remove_file(table.join("k=1")).unwrap();

    // A file link out of the root.
    std::os::unix::fs::symlink(secret.join("s.csv"), table.join("s.csv")).unwrap();
    let e = run(&s, &read(table.to_str().unwrap())).unwrap_err();
    assert!(e.contains("a link out of the allowed roots"), "{e}");
}
