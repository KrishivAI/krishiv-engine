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
}

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
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
/// dependencies are checked; dev-dependencies are not.
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
    errors.sort();
    Ok(errors)
}

/// Parses `layers.toml` text.
pub fn parse_map(text: &str) -> Result<LayerMap, String> {
    toml::from_str(text).map_err(|e| format!("invalid layers.toml: {e}"))
}
