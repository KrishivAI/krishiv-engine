// SPDX-License-Identifier: Apache-2.0
//! Tests for the layering checker.

use xtask::layering::{check, parse_map};

const MAP: &str = include_str!("../layers.toml");

fn fixture(common_dep_on_exec: bool) -> String {
    let dep = if common_dep_on_exec {
        r#"{"name":"krishiv-exec","kind":null},"#
    } else {
        ""
    };
    format!(
        r#"{{
  "workspace_members": ["lo","hi","sdk","x"],
  "packages": [
    {{"id":"lo","name":"krishiv-common","dependencies":[{dep} {{"name":"serde","kind":null}}]}},
    {{"id":"hi","name":"krishiv-exec","dependencies":[{{"name":"krishiv-common","kind":null}}]}},
    {{"id":"sdk","name":"krishiv-udf-sdk","dependencies":[{{"name":"krishiv-common","kind":null}}]}},
    {{"id":"x","name":"xtask","dependencies":[]}}
  ]
}}"#
    )
}

#[test]
fn good_graph_passes() {
    let map = parse_map(MAP).unwrap();
    assert!(check(&fixture(false), &map).unwrap().is_empty());
}

#[test]
fn upward_edge_fails_naming_the_edge() {
    let map = parse_map(MAP).unwrap();
    let errs = check(&fixture(true), &map).unwrap();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("`krishiv-common` (L0) depends on `krishiv-exec` (L4)"));
}

#[test]
fn sdk_depending_on_l2_fails() {
    let map = parse_map(MAP).unwrap();
    let json = r#"{"workspace_members":["a","b"],"packages":[
      {"id":"a","name":"krishiv-udf-sdk","dependencies":[{"name":"krishiv-io","kind":null}]},
      {"id":"b","name":"krishiv-io","dependencies":[]}]}"#;
    let errs = check(json, &map).unwrap();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("`krishiv-udf-sdk` (L1) depends on `krishiv-io` (L2)"));
}

#[test]
fn dev_deps_are_ignored_and_unmapped_crates_flagged() {
    let map = parse_map(MAP).unwrap();
    let json = r#"{"workspace_members":["a","b"],"packages":[
      {"id":"a","name":"krishiv-common","dependencies":[{"name":"krishiv-exec","kind":"dev"}]},
      {"id":"b","name":"krishiv-exec","dependencies":[]},
      {"id":"c","name":"mystery","dependencies":[]}]}"#;
    assert!(check(json, &map).unwrap().is_empty());
    let json2 = json.replace(r#"["a","b"]"#, r#"["a","b","c"]"#);
    let errs = check(&json2, &map).unwrap();
    assert!(errs[0].contains("`mystery` is not listed"));
}

/// `krishiv-plan` reaching `prost` through `mid`, with `prost` reachable
/// only as a dev-dependency when `dev_only`.
fn forbidden_fixture(dev_only: bool) -> String {
    let kind = if dev_only { r#""dev""# } else { "null" };
    format!(
        r#"{{
  "workspace_members": ["plan"],
  "packages": [
    {{"id":"plan","name":"krishiv-plan","dependencies":[]}},
    {{"id":"mid","name":"some-crate","dependencies":[]}},
    {{"id":"pb","name":"prost","dependencies":[]}}
  ],
  "resolve": {{"nodes": [
    {{"id":"plan","deps":[{{"pkg":"mid","dep_kinds":[{{"kind":{kind}}}]}}]}},
    {{"id":"mid","deps":[{{"pkg":"pb","dep_kinds":[{{"kind":null}}]}}]}},
    {{"id":"pb","deps":[]}}
  ]}}
}}"#
    )
}

#[test]
fn plan_depending_on_prost_transitively_fails() {
    let map = parse_map(MAP).unwrap();
    let errs = check(&forbidden_fixture(false), &map).unwrap();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].contains("`krishiv-plan` must not depend on `prost`"),
        "{errs:?}"
    );
    assert!(
        errs[0].contains("krishiv-plan -> some-crate -> prost"),
        "{errs:?}"
    );
}

#[test]
fn forbidden_rules_ignore_dev_dependencies() {
    let map = parse_map(MAP).unwrap();
    assert!(check(&forbidden_fixture(true), &map).unwrap().is_empty());
}

#[test]
fn plan_depending_on_connect_fails() {
    let map = parse_map(MAP).unwrap();
    let json = r#"{"workspace_members":["plan","conn"],"packages":[
      {"id":"plan","name":"krishiv-plan","dependencies":[{"name":"krishiv-connect","kind":null}]},
      {"id":"conn","name":"krishiv-connect","dependencies":[]}],
      "resolve":{"nodes":[
      {"id":"plan","deps":[{"pkg":"conn","dep_kinds":[{"kind":null}]}]},
      {"id":"conn","deps":[]}]}}"#;
    let errs = check(json, &map).unwrap();
    assert!(
        errs.iter()
            .any(|e| e.contains("`krishiv-plan` must not depend on `krishiv-connect`")),
        "{errs:?}"
    );
    let no_resolve = r#"{"workspace_members":["plan"],"packages":[
      {"id":"plan","name":"krishiv-plan","dependencies":[]}]}"#;
    assert!(check(no_resolve, &map).is_err(), "needs the resolve graph");
}

#[test]
fn forbidden_rule_for_a_non_member_fails() {
    let map =
        parse_map("[layers]\nkrishiv-plan = 3\n[forbidden]\nkrishiv-gone = [\"prost\"]\n").unwrap();
    let json = r#"{"workspace_members":["plan"],"packages":[
      {"id":"plan","name":"krishiv-plan","dependencies":[]}],"resolve":{"nodes":[]}}"#;
    let errs = check(json, &map).unwrap();
    assert!(
        errs[0].contains("`krishiv-gone`, which is not a crate in [layers]"),
        "{errs:?}"
    );
}
