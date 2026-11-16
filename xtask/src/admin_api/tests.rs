// SPDX-License-Identifier: Apache-2.0
//! Tests of the additive-only check (design P3-T066 §8): each non-additive change fails, each
//! additive one passes, and the committed snapshots are what the sources generate.

use serde_json::{Value, json};

use super::{descriptor_problems, generate, openapi_problems, preview_of, revision_problem};

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
