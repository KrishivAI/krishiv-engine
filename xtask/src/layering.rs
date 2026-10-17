// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask layering`: enforce the crate layer map (ADR 0026).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Deserialize;

/// Parsed `layers.toml`.
#[derive(Debug, Deserialize)]
pub struct LayerMap {
    /// Workspace members exempt from layering (e.g. `xtask`).
    #[serde(default)]
    pub ignore: BTreeSet<String>,
    /// Crate name to layer number.
    pub layers: BTreeMap<String, u32>,
    /// Crate name to packages it must never depend on, directly or
    /// transitively (normal and build dependencies). The resolve graph is
    /// taken with all features on, so a dependency that only another
    /// crate's feature brings in can be reported too: conservative on
    /// purpose.
    #[serde(default)]
    pub forbidden: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
    /// Absent with `cargo metadata --no-deps`.
    resolve: Option<Resolve>,
}

#[derive(Debug, Deserialize)]
struct Resolve {
    nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
struct Node {
    id: String,
    deps: Vec<NodeDep>,
}

#[derive(Debug, Deserialize)]
struct NodeDep {
    pkg: String,
    dep_kinds: Vec<DepKind>,
}

#[derive(Debug, Deserialize)]
struct DepKind {
    /// `null` for normal dependencies, else `"dev"` or `"build"`.
    kind: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Package {
    id: String,
    name: String,
    dependencies: Vec<Dependency>,
}

#[derive(Debug, Deserialize)]
struct Dependency {
    name: String,
    /// `null` for normal dependencies, else `"dev"` or `"build"`.
    kind: Option<String>,
}

/// Checks `cargo metadata` JSON against the layer map.
///
/// Returns one human-readable message per violation. Normal and build
/// dependencies are checked; dev-dependencies are not. The `forbidden` rules
/// need the resolve graph (`cargo metadata` without `--no-deps`).
pub fn check(metadata_json: &str, map: &LayerMap) -> Result<Vec<String>, String> {
    let md: Metadata =
        serde_json::from_str(metadata_json).map_err(|e| format!("invalid cargo metadata: {e}"))?;
    let members: HashMap<&str, &Package> = md
        .packages
        .iter()
        .filter(|p| md.workspace_members.contains(&p.id))
        .map(|p| (p.name.as_str(), p))
        .collect();

    let mut errors = Vec::new();
    for (name, pkg) in &members {
        if map.ignore.contains(*name) {
            continue;
        }
        let Some(&layer) = map.layers.get(*name) else {
            errors.push(format!(
                "crate `{name}` is not listed in xtask/layers.toml; assign it a layer"
            ));
            continue;
        };
        for dep in &pkg.dependencies {
            if dep.kind.as_deref() == Some("dev") || !members.contains_key(dep.name.as_str()) {
                continue;
            }
            match map.layers.get(&dep.name) {
                None if map.ignore.contains(&dep.name) => {}
                None => errors.push(format!(
                    "crate `{}` (depended on by `{name}`) is not listed in xtask/layers.toml",
                    dep.name
                )),
                Some(&dep_layer) if dep_layer > layer => errors.push(format!(
                    "layering violation: `{name}` (L{layer}) depends on `{}` (L{dep_layer}); \
                     a crate may depend only on crates in the same or a lower layer",
                    dep.name
                )),
                Some(_) => {}
            }
        }
    }
    for (name, banned) in &map.forbidden {
        if !map.layers.contains_key(name) {
            errors.push(format!(
                "[forbidden] names `{name}`, which is not a crate in [layers]"
            ));
        } else if let Some(pkg) = members.get(name.as_str()) {
            errors.extend(forbidden_paths(&md, &pkg.id, banned)?);
        }
    }
    errors.sort();
    Ok(errors)
}

/// Walks the resolve graph from `root` over normal and build edges and
/// reports each reachable package in `banned`, with the path to it.
fn forbidden_paths(
    md: &Metadata,
    root: &str,
    banned: &BTreeSet<String>,
) -> Result<Vec<String>, String> {
    let resolve = md.resolve.as_ref().ok_or(
        "forbidden-dependency rules need the resolve graph; run cargo metadata without --no-deps",
    )?;
    let names: HashMap<&str, &str> = md
        .packages
        .iter()
        .map(|p| (p.id.as_str(), p.name.as_str()))
        .collect();
    let name = |id: &str| names.get(id).copied().unwrap_or(id).to_owned();
    let nodes: HashMap<&str, &Node> = resolve.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut parent: HashMap<&str, &str> = HashMap::new();
    let mut queue = std::collections::VecDeque::from([root]);
    let mut errors = Vec::new();
    while let Some(id) = queue.pop_front() {
        let Some(node) = nodes.get(id) else { continue };
        for dep in &node.deps {
            let dev_only = dep
                .dep_kinds
                .iter()
                .all(|k| k.kind.as_deref() == Some("dev"));
            if dev_only || dep.pkg == root || parent.contains_key(dep.pkg.as_str()) {
                continue;
            }
            parent.insert(&dep.pkg, id);
            if banned.contains(&name(&dep.pkg)) {
                let mut path = vec![name(&dep.pkg)];
                let mut at = id;
                while at != root {
                    path.push(name(at));
                    at = parent.get(at).copied().unwrap_or(root);
                }
                path.push(name(root));
                path.reverse();
                errors.push(format!(
                    "forbidden dependency: `{}` must not depend on `{}` (xtask/layers.toml \
                     [forbidden]); path: {}",
                    name(root),
                    name(&dep.pkg),
                    path.join(" -> ")
                ));
            }
            queue.push_back(&dep.pkg);
        }
    }
    Ok(errors)
}

/// Parses `layers.toml` text.
pub fn parse_map(text: &str) -> Result<LayerMap, String> {
    toml::from_str(text).map_err(|e| format!("invalid layers.toml: {e}"))
}
