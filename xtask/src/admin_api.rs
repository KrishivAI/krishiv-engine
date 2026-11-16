// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask admin-api` (design P3-T066 §6.3): the additive-only promise of `krishiv.admin.v1`.
//!
//! Two snapshots sit in `crates/krishiv-server/admin-v1/`: `descriptor.json` (per message each
//! field's number, type, label, JSON name, presence and oneof; per enum its values; per method its
//! messages, streaming flags, REST verb and path and the privilege it checks) and `openapi.json`,
//! generated from the same descriptor and `routes.json`. There are no hand-written mirror types.
//!
//! ```text
//! cargo xtask admin-api                      # freshness: generated == committed
//! cargo xtask admin-api --base REF           # + additive against the snapshots at REF
//! cargo xtask admin-api --write [--base REF] # rewrite the snapshots (only when additive)
//! ```
//!
//! Freshness runs the PR's own generator; the additive comparison reads the committed JSON at the
//! base with `git show`, so a snapshot edited by hand fails freshness and one edited to match a
//! non-additive proto change fails the comparison. When the base has no snapshot (the PR that
//! adds the surface) the comparison is skipped and says so. Until the first release that ships
//! `admin.v1` a non-additive change passes only with `--allow-breaking-before-release`.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use prost::Message;
use prost_types::field_descriptor_proto::{Label, Type};
use prost_types::{DescriptorProto, EnumDescriptorProto, FileDescriptorSet, MethodDescriptorProto};
use serde_json::{Map, Value, json};

const DIR: &str = "crates/krishiv-server/admin-v1";
const DESCRIPTOR_BIN: &str = "crates/krishiv-server/src/admin_v1/descriptor.bin";
const ROUTES: &str = "crates/krishiv-server/admin-v1/routes.json";
const MOD_RS: &str = "crates/krishiv-server/src/admin_v1/mod.rs";
const PACKAGE: &str = "krishiv.admin.v1";
const DESCRIPTOR_JSON: &str = "descriptor.json";
const OPENAPI_JSON: &str = "openapi.json";

/// Runs the command with its arguments (those after `admin-api`).
pub fn run(root: &Path, args: &[&str]) -> Result<(), String> {
    let mut write = false;
    let mut allow_breaking = false;
    let mut base = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match *a {
            "--write" => write = true,
            "--allow-breaking-before-release" => allow_breaking = true,
            "--base" => base = Some((*it.next().ok_or("--base needs a ref")?).to_owned()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let (descriptor, openapi) = generate(root)?;
    let texts = [
        (DESCRIPTOR_JSON, pretty(&descriptor)),
        (OPENAPI_JSON, pretty(&openapi)),
    ];
    if let Some(base) = &base {
        let problems = compare_with_base(root, base, &descriptor, &openapi)?;
        if !problems.is_empty() && !allow_breaking {
            return Err(format!(
                "admin.v1 changed in a way that is not additive against {base}:\n  {}\n\
                 (before the first release that ships it, `--allow-breaking-before-release` and \
                 the PR label `admin-api-breaking` allow it)",
                problems.join("\n  ")
            ));
        }
    }
    let dir = root.join(DIR);
    if write {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (name, text) in &texts {
            std::fs::write(dir.join(name), text).map_err(|e| format!("{name}: {e}"))?;
        }
        return Ok(());
    }
    let stale: Vec<&str> = texts
        .iter()
        .filter(|(name, text)| {
            std::fs::read_to_string(dir.join(name)).ok().as_deref() != Some(text)
        })
        .map(|(name, _)| *name)
        .collect();
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} out of date in {DIR}; run `cargo xtask admin-api --write`",
            stale.join(", ")
        ))
    }
}

fn pretty(value: &Value) -> String {
    let mut text = serde_json::to_string_pretty(value).unwrap_or_default();
    text.push('\n');
    text
}

/// The snapshots as the sources say they are now.
pub fn generate(root: &Path) -> Result<(Value, Value), String> {
    let bytes =
        std::fs::read(root.join(DESCRIPTOR_BIN)).map_err(|e| format!("{DESCRIPTOR_BIN}: {e}"))?;
    let set = FileDescriptorSet::decode(bytes.as_slice())
        .map_err(|e| format!("{DESCRIPTOR_BIN}: {e}"))?;
    let routes: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(root.join(ROUTES)).map_err(|e| format!("{ROUTES}: {e}"))?,
    )
    .map_err(|e| format!("{ROUTES}: {e}"))?;
    let revision = revision_of(
        &std::fs::read_to_string(root.join(MOD_RS)).map_err(|e| format!("{MOD_RS}: {e}"))?,
    )?;
    let file = set
        .file
        .iter()
        .find(|f| f.package() == PACKAGE)
        .ok_or_else(|| format!("{DESCRIPTOR_BIN} has no package {PACKAGE}"))?;
    descriptor_and_openapi(file, &routes, revision)
}

/// `ADMIN_V1_REVISION` from the server's source.
fn revision_of(source: &str) -> Result<i64, String> {
    source
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("pub const ADMIN_V1_REVISION: i64 =")
                .and_then(|r| r.trim().trim_end_matches(';').trim().parse().ok())
        })
        .ok_or_else(|| format!("{MOD_RS} has no `pub const ADMIN_V1_REVISION: i64 = N;`"))
}

fn descriptor_and_openapi(
    file: &prost_types::FileDescriptorProto,
    routes: &[Value],
    revision: i64,
) -> Result<(Value, Value), String> {
    let mut messages = Map::new();
    for m in &file.message_type {
        messages.insert(m.name().to_owned(), message_json(m));
    }
    let mut enums = Map::new();
    for e in &file.enum_type {
        enums.insert(e.name().to_owned(), enum_json(e));
    }
    let mut methods = Map::new();
    let service = file
        .service
        .iter()
        .find(|s| s.name() == "AdminService")
        .ok_or("the descriptor has no AdminService")?;
    for m in &service.method {
        let route = routes
            .iter()
            .find(|r| r["rpc"].as_str() == Some(m.name()))
            .ok_or_else(|| format!("{} has no entry in {ROUTES}", m.name()))?;
        methods.insert(m.name().to_owned(), method_json(m, route));
    }
    for r in routes {
        let rpc = r["rpc"].as_str().unwrap_or_default();
        if !service.method.iter().any(|m| m.name() == rpc) {
            return Err(format!(
                "{ROUTES} names {rpc}, which is no RPC of AdminService"
            ));
        }
    }
    let descriptor = json!({
        "package": PACKAGE,
        "revision": revision,
        "messages": messages,
        "enums": enums,
        "methods": methods,
    });
    let openapi = openapi_of(&descriptor);
    Ok((descriptor, openapi))
}

fn short(type_name: &str) -> String {
    type_name
        .trim_start_matches('.')
        .strip_prefix(&format!("{PACKAGE}."))
        .unwrap_or(type_name.trim_start_matches('.'))
        .to_owned()
}

fn type_of(f: &prost_types::FieldDescriptorProto) -> String {
    match f.r#type() {
        Type::Message => format!("message:{}", short(f.type_name())),
        Type::Enum => format!("enum:{}", short(f.type_name())),
        other => other
            .as_str_name()
            .trim_start_matches("TYPE_")
            .to_ascii_lowercase(),
    }
}

fn message_json(m: &DescriptorProto) -> Value {
    let mut fields = Map::new();
    for f in &m.field {
        let oneof = f
            .oneof_index
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| m.oneof_decl.get(i))
            .map(|o| o.name().to_owned());
        fields.insert(
            f.name().to_owned(),
            json!({
                "number": f.number(),
                "type": type_of(f),
                "label": if f.label() == Label::Repeated { "repeated" } else { "singular" },
                "jsonName": f.json_name(),
                "optional": f.proto3_optional(),
                "oneof": oneof,
            }),
        );
    }
    json!({ "fields": fields })
}

fn enum_json(e: &EnumDescriptorProto) -> Value {
    let values: BTreeMap<&str, i32> = e.value.iter().map(|v| (v.name(), v.number())).collect();
    json!({ "values": values })
}

fn method_json(m: &MethodDescriptorProto, route: &Value) -> Value {
    json!({
        "request": short(m.input_type()),
        "response": short(m.output_type()),
        "clientStreaming": m.client_streaming(),
        "serverStreaming": m.server_streaming(),
        "http": { "method": route["method"], "path": route["path"] },
        "privilege": route["privilege"],
    })
}

/// The OpenAPI document: paths from the route table, schemas from the descriptor.
fn openapi_of(descriptor: &Value) -> Value {
    let mut schemas = Map::new();
    if let Some(messages) = descriptor["messages"].as_object() {
        for (name, m) in messages {
            let mut props = Map::new();
            if let Some(fields) = m["fields"].as_object() {
                for (_, f) in fields {
                    let json_name = f["jsonName"].as_str().unwrap_or_default().to_owned();
                    let one = schema_of(f["type"].as_str().unwrap_or_default());
                    let schema = if f["label"] == "repeated" {
                        json!({"type": "array", "items": one})
                    } else {
                        one
                    };
                    props.insert(json_name, schema);
                }
            }
            schemas.insert(name.clone(), json!({"type": "object", "properties": props}));
        }
    }
    if let Some(enums) = descriptor["enums"].as_object() {
        for (name, e) in enums {
            let names: Vec<&String> = e["values"]
                .as_object()
                .map(|v| v.keys().collect())
                .unwrap_or_default();
            schemas.insert(name.clone(), json!({"type": "string", "enum": names}));
        }
    }
    let mut paths: Map<String, Value> = Map::new();
    if let Some(methods) = descriptor["methods"].as_object() {
        for (rpc, m) in methods {
            let path = format!(
                "/api/admin/v1{}",
                m["http"]["path"].as_str().unwrap_or_default()
            );
            let verb = m["http"]["method"]
                .as_str()
                .unwrap_or("GET")
                .to_ascii_lowercase();
            let mut op = json!({
                "operationId": rpc,
                "x-privilege": m["privilege"],
                "responses": {
                    "200": {"description": "OK", "content": {"application/json": {
                        "schema": {"$ref": format!("#/components/schemas/{}", m["response"].as_str().unwrap_or_default())}}}},
                    "default": {"description": "the error envelope", "content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}}}},
                },
            });
            if verb != "get" && verb != "delete" {
                op["requestBody"] = json!({"required": false, "content": {"application/json": {
                    "schema": {"$ref": format!("#/components/schemas/{}", m["request"].as_str().unwrap_or_default())}}}});
            }
            paths.entry(path).or_insert_with(|| json!({}))[verb] = op;
        }
    }
    schemas.insert(
        "Error".to_owned(),
        json!({"type": "object", "properties": {"error": {"type": "object", "properties": {
            "errorClass": {"type": "string"}, "sqlState": {"type": "string"},
            "messageParameters": {"type": "object", "additionalProperties": {"type": "string"}},
            "message": {"type": "string"}, "requestId": {"type": "string"}}}}}),
    );
    json!({
        "openapi": "3.1.0",
        "info": {"title": "krishiv.admin.v1", "version": descriptor["revision"].to_string()},
        "paths": paths,
        "components": {"schemas": schemas},
    })
}

/// The proto3 JSON mapping of a field type as a JSON schema.
fn schema_of(ty: &str) -> Value {
    if let Some(name) = ty
        .strip_prefix("message:")
        .or_else(|| ty.strip_prefix("enum:"))
    {
        return json!({"$ref": format!("#/components/schemas/{name}")});
    }
    match ty {
        "string" | "bytes" => json!({"type": "string"}),
        "bool" => json!({"type": "boolean"}),
        "int64" | "uint64" | "sint64" | "fixed64" | "sfixed64" => {
            json!({"type": "string", "format": "int64"})
        }
        "double" | "float" => json!({"type": "number"}),
        _ => json!({"type": "integer"}),
    }
}

fn compare_with_base(
    root: &Path,
    base: &str,
    descriptor: &Value,
    openapi: &Value,
) -> Result<Vec<String>, String> {
    let mut problems = Vec::new();
    for (name, head) in [(DESCRIPTOR_JSON, descriptor), (OPENAPI_JSON, openapi)] {
        let path = format!("{DIR}/{name}");
        let shown = Command::new("git")
            .current_dir(root)
            .args(["show", &format!("{base}:{path}")])
            .output()
            .map_err(|e| format!("cannot run git: {e}"))?;
        if !shown.status.success() {
            println!(
                "admin-api: {base} has no {path} (the PR that adds the surface): additive check skipped"
            );
            continue;
        }
        let old: Value =
            serde_json::from_slice(&shown.stdout).map_err(|e| format!("{base}:{path}: {e}"))?;
        problems.extend(if name == DESCRIPTOR_JSON {
            descriptor_problems(&old, head)
        } else {
            openapi_problems(&old, head)
        });
    }
    Ok(problems)
}

/// What a non-additive change of the descriptor snapshot is, one line per item.
#[must_use]
pub fn descriptor_problems(old: &Value, new: &Value) -> Vec<String> {
    let mut out = Vec::new();
    missing_or_changed(old, new, &["messages"], &mut out, |msg, name, o, n| {
        // Fields: every old field stays, unchanged; new fields are free.
        if let (Some(of), Some(nf)) = (o["fields"].as_object(), n["fields"].as_object()) {
            for (field, ov) in of {
                match nf.get(field) {
                    None => msg.push(format!(
                        "message {name}: field {field} was removed or renamed"
                    )),
                    Some(nv) if nv != ov => msg.push(format!(
                        "message {name}: field {field} changed from {ov} to {nv}"
                    )),
                    Some(_) => {}
                }
            }
        }
    });
    missing_or_changed(old, new, &["enums"], &mut out, |msg, name, o, n| {
        if let (Some(ov), Some(nv)) = (o["values"].as_object(), n["values"].as_object()) {
            for (value, number) in ov {
                if nv.get(value) != Some(number) {
                    msg.push(format!(
                        "enum {name}: value {value} = {number} was removed or changed"
                    ));
                }
            }
        }
    });
    if let Some(om) = old["methods"].as_object() {
        for (rpc, ov) in om {
            match new["methods"].get(rpc) {
                None => out.push(format!("method {rpc} was removed")),
                Some(nv) => {
                    for key in [
                        "request",
                        "response",
                        "clientStreaming",
                        "serverStreaming",
                        "http",
                    ] {
                        if ov[key] != nv[key] {
                            out.push(format!(
                                "method {rpc}: {key} changed from {} to {}",
                                ov[key], nv[key]
                            ));
                        }
                    }
                    if ov["privilege"] != nv["privilege"] {
                        out.push(format!(
                            "method {rpc}: the privilege it checks changed from {} to {}",
                            ov["privilege"], nv["privilege"]
                        ));
                    }
                }
            }
        }
    }
    let (or, nr) = (
        old["revision"].as_i64().unwrap_or(0),
        new["revision"].as_i64().unwrap_or(0),
    );
    if old != new && nr <= or {
        out.push(format!(
            "the descriptor changed but ADMIN_V1_REVISION did not increase ({or} -> {nr})"
        ));
    }
    out
}

/// Calls `each` for every item of `old[path]` that is still in `new[path]`, and records one that
/// is gone.
fn missing_or_changed(
    old: &Value,
    new: &Value,
    path: &[&str],
    out: &mut Vec<String>,
    mut each: impl FnMut(&mut Vec<String>, &str, &Value, &Value),
) {
    let key = path[0];
    let Some(items) = old[key].as_object() else {
        return;
    };
    for (name, o) in items {
        match new[key].get(name) {
            None => out.push(format!("{} {name} was removed", key.trim_end_matches('s'))),
            Some(n) => each(out, name, o, n),
        }
    }
}

/// OpenAPI: every old path and verb stays, every old schema property stays with the same schema,
/// and no schema gains a `required` entry.
#[must_use]
pub fn openapi_problems(old: &Value, new: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(paths) = old["paths"].as_object() {
        for (path, verbs) in paths {
            for (verb, op) in verbs.as_object().into_iter().flatten() {
                match new["paths"][path].get(verb) {
                    None => out.push(format!("{} {path} was removed", verb.to_ascii_uppercase())),
                    Some(n)
                        if n["requestBody"]["required"] == true
                            && op["requestBody"]["required"] != true =>
                    {
                        out.push(format!(
                            "{} {path}: the request body became required",
                            verb.to_ascii_uppercase()
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    if let Some(schemas) = old["components"]["schemas"].as_object() {
        for (name, o) in schemas {
            let Some(n) = new["components"]["schemas"].get(name) else {
                out.push(format!("schema {name} was removed"));
                continue;
            };
            for (prop, ov) in o["properties"].as_object().into_iter().flatten() {
                match n["properties"].get(prop) {
                    None => out.push(format!("schema {name}: property {prop} was removed")),
                    Some(nv) if nv != ov => {
                        out.push(format!("schema {name}: property {prop} changed"));
                    }
                    Some(_) => {}
                }
            }
            let old_required: Vec<&Value> =
                o["required"].as_array().into_iter().flatten().collect();
            for r in n["required"].as_array().into_iter().flatten() {
                if !old_required.contains(&r) {
                    out.push(format!("schema {name}: {r} became required"));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
