// SPDX-License-Identifier: Apache-2.0
//! Tests of the additive-only check (design P3-T066 §8): each non-additive change fails, each
//! additive one passes, and the committed snapshots are what the sources generate.

use serde_json::{Value, json};

use super::{descriptor_problems, generate, openapi_problems, preview_of, revision_problem, run};

/// One non-additive change: what it is and how to make it.
type Change = (&'static str, Box<dyn Fn(&mut Value)>);

fn base() -> Value {
    json!({
        "package": "krishiv.admin.v1",
        "revision": 1,
        "messages": {
            "Req": {"fields": {}},
            "Resp": {"fields": {
                "name": {"number": 1, "type": "string", "label": "singular", "jsonName": "name", "optional": false, "oneof": null},
                "count": {"number": 2, "type": "int64", "label": "singular", "jsonName": "count", "optional": false, "oneof": null}
            }}
        },
        "enums": {"State": {"values": {"STATE_UNSPECIFIED": 0, "ACTIVE": 1}}},
        "methods": {"Get": {
            "request": "Req", "response": "Resp", "clientStreaming": false, "serverStreaming": false,
            "http": {"method": "GET", "path": "/thing"}, "privilege": "authenticated"
        }}
    })
}

fn bumped(mut v: Value) -> Value {
    v["revision"] = json!(2);
    v
}

#[test]
fn an_added_field_enum_value_and_rpc_pass() {
    let mut new = bumped(base());
    new["messages"]["Resp"]["fields"]["extra"] = json!({"number": 3, "type": "string", "label": "singular", "jsonName": "extra", "optional": false, "oneof": null});
    new["enums"]["State"]["values"]["DONE"] = json!(2);
    new["messages"]["New"] = json!({"fields": {}});
    new["methods"]["List"] = json!({
        "request": "Req", "response": "Resp", "clientStreaming": false, "serverStreaming": false,
        "http": {"method": "GET", "path": "/things"}, "privilege": "authenticated"
    });
    assert_eq!(descriptor_problems(&base(), &new), Vec::<String>::new());
}

#[test]
fn each_non_additive_change_fails_and_names_the_item() {
    let cases: Vec<Change> = vec![
        (
            "removed field",
            Box::new(|v| {
                v["messages"]["Resp"]["fields"]
                    .as_object_mut()
                    .map(|f| f.remove("count"));
            }),
        ),
        (
            "renumbered field",
            Box::new(|v| v["messages"]["Resp"]["fields"]["name"]["number"] = json!(9)),
        ),
        (
            "retyped field",
            Box::new(|v| v["messages"]["Resp"]["fields"]["count"]["type"] = json!("string")),
        ),
        (
            "json name",
            Box::new(|v| v["messages"]["Resp"]["fields"]["name"]["jsonName"] = json!("n")),
        ),
        (
            "label",
            Box::new(|v| v["messages"]["Resp"]["fields"]["name"]["label"] = json!("repeated")),
        ),
        (
            "oneof",
            Box::new(|v| v["messages"]["Resp"]["fields"]["name"]["oneof"] = json!("o")),
        ),
        (
            "removed message",
            Box::new(|v| {
                v["messages"].as_object_mut().map(|m| m.remove("Resp"));
            }),
        ),
        (
            "removed enum value",
            Box::new(|v| {
                v["enums"]["State"]["values"]
                    .as_object_mut()
                    .map(|m| m.remove("ACTIVE"));
            }),
        ),
        (
            "changed enum number",
            Box::new(|v| v["enums"]["State"]["values"]["ACTIVE"] = json!(5)),
        ),
        (
            "removed method",
            Box::new(|v| {
                v["methods"].as_object_mut().map(|m| m.remove("Get"));
            }),
        ),
        (
            "removed path",
            Box::new(|v| v["methods"]["Get"]["http"]["path"] = json!("/other")),
        ),
        (
            "changed verb",
            Box::new(|v| v["methods"]["Get"]["http"]["method"] = json!("POST")),
        ),
        (
            "changed response",
            Box::new(|v| v["methods"]["Get"]["response"] = json!("Req")),
        ),
        (
            "changed privilege",
            Box::new(|v| v["methods"]["Get"]["privilege"] = json!("manage")),
        ),
    ];
    for (what, change) in cases {
        let mut new = bumped(base());
        change(&mut new);
        let problems = descriptor_problems(&base(), &new);
        assert!(!problems.is_empty(), "{what} passed");
    }
}

#[test]
fn a_changed_descriptor_must_raise_the_revision() {
    let mut new = base();
    new["messages"]["New"] = json!({"fields": {}});
    let problem = revision_problem(&base(), &new);
    assert!(problem.is_some_and(|p| p.contains("ADMIN_V1_REVISION")));
    assert!(
        revision_problem(&base(), &base()).is_none(),
        "no change, no new revision needed"
    );
    assert!(revision_problem(&base(), &bumped(new)).is_none());
}

#[test]
fn the_breaking_flag_needs_a_preview_surface() {
    assert_eq!(preview_of("pub const PREVIEW: bool = true;"), Ok(true));
    assert_eq!(preview_of("  pub const PREVIEW: bool = false;"), Ok(false));
    assert!(preview_of("nothing here").is_err());
}

#[test]
fn a_retyped_map_value_and_a_nested_enum_are_caught() {
    let mut old = base();
    old["messages"]["Resp.LabelsEntry"] = json!({"mapEntry": true, "fields": {
        "key": {"number": 1, "type": "string", "label": "singular", "jsonName": "key", "optional": false, "oneof": null},
        "value": {"number": 2, "type": "string", "label": "singular", "jsonName": "value", "optional": false, "oneof": null}}});
    old["enums"]["Resp.Kind"] = json!({"values": {"KIND_UNSPECIFIED": 0, "A": 1}});
    let mut retyped = bumped(old.clone());
    retyped["messages"]["Resp.LabelsEntry"]["fields"]["value"]["type"] = json!("int64");
    assert!(!descriptor_problems(&old, &retyped).is_empty());
    let mut enum_gone = bumped(old.clone());
    enum_gone["enums"]["Resp.Kind"]["values"]
        .as_object_mut()
        .map(|v| v.remove("A"));
    assert!(!descriptor_problems(&old, &enum_gone).is_empty());
}

#[test]
fn the_openapi_of_a_map_is_an_object() {
    let mut d = base();
    d["messages"]["Resp.LabelsEntry"] = json!({"mapEntry": true, "fields": {
        "key": {"number": 1, "type": "string", "label": "singular", "jsonName": "key", "optional": false, "oneof": null},
        "value": {"number": 2, "type": "int64", "label": "singular", "jsonName": "value", "optional": false, "oneof": null}}});
    d["messages"]["Resp"]["fields"]["labels"] = json!({"number": 3, "type": "message:Resp.LabelsEntry", "label": "repeated", "jsonName": "labels", "optional": false, "oneof": null});
    let api = super::openapi_of(&d);
    assert_eq!(
        api["components"]["schemas"]["Resp"]["properties"]["labels"]["type"],
        "object"
    );
    assert_eq!(
        api["components"]["schemas"]["Resp"]["properties"]["labels"]["additionalProperties"]["format"],
        "int64"
    );
    assert!(
        api["components"]["schemas"]
            .get("Resp.LabelsEntry")
            .is_none()
    );
}

#[test]
fn the_openapi_declares_timestamps_path_and_query_parameters() {
    let mut d = base();
    d["messages"]["Resp"]["fields"]["at"] = json!({"number": 9, "type": "message:google.protobuf.Timestamp", "label": "singular", "jsonName": "at", "optional": false, "oneof": null});
    d["messages"]["Req"]["fields"]["tags"] = json!({"number": 8, "type": "string", "label": "repeated", "jsonName": "tags", "optional": false, "oneof": null});
    let api = super::openapi_of(&d);
    assert_eq!(
        api["components"]["schemas"]["Resp"]["properties"]["at"],
        json!({"type": "string", "format": "date-time"})
    );
    let path = super::path_parameters("/things/{thing_id}:kill");
    assert_eq!(path[0]["name"], "thing_id");
    assert_eq!(path[0]["required"], true);
    let method = json!({"request": "Req"});
    let query = super::query_parameters(&d, &method, &path);
    let tags = query.iter().find(|p| p["name"] == "tags");
    assert_eq!(tags.map(|p| p["explode"].clone()), Some(json!(true)));
}

fn api() -> Value {
    json!({
        "paths": {"/api/admin/v1/thing": {"get": {"operationId": "Get"}, "post": {"requestBody": {"required": false}}}},
        "components": {"schemas": {"Resp": {"type": "object", "properties": {"name": {"type": "string"}}}}}
    })
}

#[test]
fn openapi_additions_pass_and_removals_fail() {
    let mut new = api();
    new["paths"]["/api/admin/v1/other"] = json!({"get": {}});
    new["components"]["schemas"]["Resp"]["properties"]["more"] = json!({"type": "string"});
    assert!(openapi_problems(&api(), &new).is_empty());
    let mut gone = api();
    gone["paths"]
        .as_object_mut()
        .map(|p| p.remove("/api/admin/v1/thing"));
    assert!(!openapi_problems(&api(), &gone).is_empty());
    let mut verb = api();
    verb["paths"]["/api/admin/v1/thing"]
        .as_object_mut()
        .map(|p| p.remove("post"));
    assert!(!openapi_problems(&api(), &verb).is_empty());
    let mut prop = api();
    prop["components"]["schemas"]["Resp"]["properties"]["name"] = json!({"type": "integer"});
    assert!(!openapi_problems(&api(), &prop).is_empty());
    let mut required = api();
    required["components"]["schemas"]["Resp"]["required"] = json!(["name"]);
    assert!(!openapi_problems(&api(), &required).is_empty());
    let mut body = api();
    body["paths"]["/api/admin/v1/thing"]["post"]["requestBody"]["required"] = json!(true);
    assert!(!openapi_problems(&api(), &body).is_empty());
}

#[test]
fn the_committed_snapshots_are_what_the_sources_generate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| panic!("no root"));
    let (descriptor, openapi) = generate(root).unwrap_or_else(|e| panic!("{e}"));
    for (name, generated) in [("descriptor.json", descriptor), ("openapi.json", openapi)] {
        let path = root.join("crates/krishiv-server/admin-v1").join(name);
        let committed: Value = serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            committed, generated,
            "{name} is stale; run `cargo xtask admin-api --write`"
        );
    }
}

/// A scratch repository with the sources the generator reads, a committed copy of the snapshots,
/// and a `base` branch pointing at that commit.
fn scratch() -> tempfile::TempDir {
    use std::process::Command;
    let real = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| panic!("no root"));
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    for rel in [
        "crates/krishiv-server/src/admin_v1/descriptor.bin",
        "crates/krishiv-server/src/admin_v1/mod.rs",
        "crates/krishiv-server/admin-v1/routes.json",
        "crates/krishiv-server/admin-v1/descriptor.json",
        "crates/krishiv-server/admin-v1/openapi.json",
    ] {
        let to = dir.path().join(rel);
        std::fs::create_dir_all(to.parent().unwrap_or_else(|| panic!("no parent")))
            .unwrap_or_else(|e| panic!("{e}"));
        std::fs::copy(real.join(rel), &to).unwrap_or_else(|e| panic!("{rel}: {e}"));
    }
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .current_dir(dir.path())
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q", "-b", "main"]);
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "base"]);
    git(&["branch", "base"]);
    dir
}

#[test]
fn the_command_checks_writes_and_compares() {
    let dir = scratch();
    let root = dir.path();
    // Fresh snapshots pass, an unknown flag and a missing ref value fail.
    assert_eq!(run(root, &[]), Ok(()));
    assert!(run(root, &["--nope"]).is_err());
    assert!(run(root, &["--base"]).is_err());
    // Identical to the base: additive, nothing to raise.
    assert_eq!(run(root, &["--base", "base"]), Ok(()));
    // A base that does not resolve fails closed.
    assert!(run(root, &["--base", "no-such-ref"]).is_err());
    // A stale committed snapshot is reported, and --write repairs it.
    let snapshot = root.join("crates/krishiv-server/admin-v1/descriptor.json");
    std::fs::write(&snapshot, "{}\n").unwrap_or_else(|e| panic!("{e}"));
    assert!(run(root, &[]).is_err_and(|e| e.contains("descriptor.json")));
    assert_eq!(run(root, &["--write"]), Ok(()));
    assert_eq!(run(root, &[]), Ok(()));
}

#[test]
fn a_removed_field_against_the_base_fails_unless_allowed_while_preview() {
    let dir = scratch();
    let root = dir.path();
    // The base has a field the sources no longer generate: make the committed base say so.
    let snapshot = root.join("crates/krishiv-server/admin-v1/descriptor.json");
    let mut base: Value =
        serde_json::from_str(&std::fs::read_to_string(&snapshot).unwrap_or_else(|e| panic!("{e}")))
            .unwrap_or_default();
    base["messages"]["GetVersionResponse"]["fields"]["gone"] = json!({"number": 9, "type": "string", "label": "singular", "jsonName": "gone", "optional": false, "oneof": null});
    // Same revision as the head: the change (a removal) is breaking and the revision did not grow.
    std::fs::write(
        &snapshot,
        serde_json::to_string_pretty(&base).unwrap_or_default(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    for args in [
        ["add", "-A"].as_slice(),
        &[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "old",
        ],
        &["branch", "-f", "base"],
    ] {
        let out = std::process::Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(out.status.success());
    }
    assert_eq!(run(root, &["--write"]), Ok(()));
    // The revision did not grow, so even the breaking flag cannot let it through.
    let err = run(root, &["--base", "base", "--allow-breaking-before-release"])
        .err()
        .unwrap_or_default();
    assert!(err.contains("ADMIN_V1_REVISION"), "{err}");
    // Raise the revision: now only the flag lets the removal through.
    let source = root.join("crates/krishiv-server/src/admin_v1/mod.rs");
    let text = std::fs::read_to_string(&source).unwrap_or_default();
    // One more than whatever the source says now, so the test outlives revision bumps.
    let marker = "ADMIN_V1_REVISION: i64 = ";
    let at = text.find(marker).unwrap_or_else(|| panic!("no revision"));
    let digits_at = at + marker.len();
    let digits: String = text[digits_at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    let next = digits.parse::<i64>().unwrap_or_default() + 1;
    let text = format!(
        "{}{next}{}",
        &text[..digits_at],
        &text[digits_at + digits.len()..]
    );
    std::fs::write(&source, &text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(run(root, &["--write"]), Ok(()));
    let err = run(root, &["--base", "base"]).err().unwrap_or_default();
    assert!(err.contains("not additive"), "{err}");
    assert_eq!(
        run(root, &["--base", "base", "--allow-breaking-before-release"]),
        Ok(())
    );
    // Once the surface is released the flag is refused.
    std::fs::write(
        &source,
        text.replace("PREVIEW: bool = true;", "PREVIEW: bool = false;"),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let err = run(root, &["--base", "base", "--allow-breaking-before-release"])
        .err()
        .unwrap_or_default();
    assert!(err.contains("no longer `preview`"), "{err}");
}
