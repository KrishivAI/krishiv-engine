// SPDX-License-Identifier: Apache-2.0
//! The descriptor file of `from_protobuf` / `to_protobuf` is read like any other path (P3-T087 review B3): the access policy is
//! asked about it (`Securable::Path`, `Select`) before anything is read, so a path outside `krishiv.io.local.allowedRoots` is the
//! same denial whether the file exists or not. The bytes go into the plan at analysis (the call is bound from them; `Call` refuses
//! a path, `krishiv-io` `a_descriptor_path_is_never_read_by_the_call`).
#![cfg(test)]

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_common::config::Config;
use krishiv_plan::analyzer::PolicyVersion;
use krishiv_session::auth::{Role, RolePolicy};
use krishiv_session::{PathPrefixPolicy, Principal, PrincipalKind, Session};

const DENIED: &str = "KRISHIV_INSUFFICIENT_PERMISSIONS";

fn golden_desc() -> Vec<u8> {
    std::fs::read(format!(
        "{}/../../tests/spark-golden/functions/from_to_protobuf/fixtures/krishiv_golden.desc",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// An operator session confined to `allowed`.
fn session(allowed: &std::path::Path) -> Session {
    let config = Config::from_pairs([("krishiv.io.local.allowedRoots", allowed.to_str().unwrap())])
        .unwrap()
        .0;
    let roles = RolePolicy::new(Some(Role::Operator), Role::Admin, PolicyVersion(1));
    let policy = PathPrefixPolicy::from_config(&config, Arc::new(roles)).unwrap();
    Session::builder()
        .principal(Principal::new("op", PrincipalKind::User))
        .access_policy(Arc::new(policy))
        .build()
        .unwrap()
}

fn run(s: &Session, sql: &str) -> Result<usize, krishiv_common::error::KrishivError> {
    block_on(async {
        let batches = s.sql(sql).await?.collect().await?;
        // A conversion that yields NULL is a failure of the call, not a pass.
        for b in &batches {
            assert_eq!(b.column(0).null_count(), 0, "{sql} returned NULL");
        }
        Ok(batches.iter().map(arrow_array::RecordBatch::num_rows).sum())
    })
}

fn calls(path: &str) -> [String; 4] {
    // `/` separators: a Windows path's `\` would be read as a SQL escape.
    let path = path.replace('\\', "/");
    [
        format!("SELECT from_protobuf(X'0801', 'krishiv.golden.Scalars', '{path}')"),
        format!("SELECT from_protobuf(X'0801', 'krishiv.golden.Scalars', '{path}', map())"),
        format!("SELECT to_protobuf(named_struct('i32', 1), 'krishiv.golden.Scalars', '{path}')"),
        format!(
            "SELECT to_protobuf(named_struct('i32', 1), 'krishiv.golden.Scalars', '{path}', map('emit.default.values', 'true'))"
        ),
    ]
}

#[test]
fn a_descriptor_path_outside_the_roots_is_denied_whether_or_not_it_exists() {
    let tmp = tempfile::tempdir().unwrap();
    let allowed = tmp.path().join("allowed");
    let secret = tmp.path().join("secret");
    std::fs::create_dir(&allowed).unwrap();
    std::fs::create_dir(&secret).unwrap();
    std::fs::write(allowed.join("g.desc"), golden_desc()).unwrap();
    std::fs::write(secret.join("g.desc"), golden_desc()).unwrap();
    std::fs::write(secret.join("junk.desc"), b"not a descriptor").unwrap();
    let s = session(&allowed);
    for sql in calls(allowed.join("g.desc").to_str().unwrap()) {
        assert_eq!(run(&s, &sql).unwrap(), 1, "{sql}");
    }
    for name in ["g.desc", "junk.desc", "missing.desc"] {
        let path = secret.join(name);
        for sql in calls(path.to_str().unwrap()) {
            let e = run(&s, &sql).unwrap_err();
            assert_eq!(e.error_class().name(), DENIED, "{sql}: {e}");
        }
    }
    // Inside the roots, Spark's errors: a missing file, a file that is not a descriptor set.
    let missing = allowed.join("missing.desc");
    let e = run(&s, &calls(missing.to_str().unwrap())[0]).unwrap_err();
    assert_eq!(e.error_class().name(), "PROTOBUF_DESCRIPTOR_FILE_NOT_FOUND");
    std::fs::write(allowed.join("junk.desc"), b"not a descriptor").unwrap();
    let e = run(&s, &calls(allowed.join("junk.desc").to_str().unwrap())[2]).unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_PARSE_PROTOBUF_DESCRIPTOR");
    // The bytes of a descriptor set need no file.
    let hex: String = golden_desc().iter().map(|b| format!("{b:02X}")).collect();
    let sql = format!("SELECT from_protobuf(X'0801', 'krishiv.golden.Scalars', X'{hex}')");
    assert_eq!(run(&s, &sql).unwrap(), 1);
}

/// In the 4-argument form a descriptor path that is foldable but not a literal (`concat(..)`) passes the argument check and is
/// refused when the call is built, where Spark reads the file: `KRISHIV_UNSUPPORTED_FEATURE` (golden README item 13). In the
/// 3-argument form Spark's constructor takes only a literal, so the call runs as a Java class: `CANNOT_LOAD_PROTOBUF_CLASS`.
#[test]
fn a_foldable_non_literal_descriptor_path_is_unsupported_not_read() {
    let tmp = tempfile::tempdir().unwrap();
    let allowed = tmp.path().join("allowed");
    std::fs::create_dir(&allowed).unwrap();
    std::fs::write(allowed.join("g.desc"), golden_desc()).unwrap();
    let s = session(&allowed);
    let dir = allowed.to_str().unwrap();
    let four = format!(
        "SELECT from_protobuf(X'0801', 'krishiv.golden.Scalars', concat('{dir}/', 'g.desc'), map())"
    );
    let e = run(&s, &four).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
    let three = format!(
        "SELECT from_protobuf(X'0801', 'krishiv.golden.Scalars', concat('{dir}/', 'g.desc'))"
    );
    let e = run(&s, &three).unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_LOAD_PROTOBUF_CLASS", "{e}");
}
