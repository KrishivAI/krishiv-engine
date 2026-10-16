#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
