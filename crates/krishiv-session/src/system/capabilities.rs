// SPDX-License-Identifier: Apache-2.0
//! The rows of `system.capabilities`: what this build is, the API surfaces a host serves and the
//! cargo features of the running binary.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock, PoisonError};

use super::SCHEMA_VERSION;

/// The cargo features this build was made with and whether each is on, set
/// once by the binary (the library cannot see its dependents' features).
static BUILD_FEATURES: OnceLock<Vec<(String, bool)>> = OnceLock::new();

/// Records the cargo features of the running binary for
/// `system.capabilities`: `(name, enabled)` for every feature that can be
/// switched. The first call wins; later calls change nothing.
pub fn register_build_features(features: &[(&str, bool)]) {
    let owned = features
        .iter()
        .map(|(name, on)| ((*name).to_owned(), *on))
        .collect();
    let _ = BUILD_FEATURES.set(owned);
}

/// The API surfaces a host serves, `name -> (value, detail)`, for the `surface` rows of
/// `system.capabilities` (`admin.v1` with its revision, P3-T066).
static SURFACES: Mutex<BTreeMap<String, (String, String)>> = Mutex::new(BTreeMap::new());

/// Records an API surface the host serves as a `("surface", name, value, detail)` row of
/// `system.capabilities`. A later call for the same name replaces the row.
pub fn register_surface(name: &str, value: &str, detail: &str) {
    SURFACES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(name.to_owned(), (value.to_owned(), detail.to_owned()));
}

/// The rows of `system.capabilities` as `[kind, name, value, detail]`, the same rows a query of the
/// table returns (`GetCapabilities` of the admin API answers with them).
#[must_use]
pub fn capabilities() -> Vec<[String; 4]> {
    capability_rows()
}

/// The rows of `system.capabilities`: `(kind, name, value, detail)`.
pub(super) fn capability_rows() -> Vec<[String; 4]> {
    let row = |kind: &str, name: &str, value: &str, detail: &str| {
        [
            kind.to_owned(),
            name.to_owned(),
            value.to_owned(),
            detail.to_owned(),
        ]
    };
    let state = |on: bool| if on { "enabled" } else { "disabled" };
    let mut rows = vec![
        row("build", "engine", "krishiv", env!("CARGO_PKG_VERSION")),
        row("build", "spark_versions", "3.5,4.0,4.1,4.2,4.3", ""),
        row("surface", "system_tables", SCHEMA_VERSION, ""),
        row("surface", "agent_result", "1", ""),
        row("feature", "cloud", state(cfg!(feature = "cloud")), ""),
    ];
    for (name, (value, detail)) in SURFACES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
    {
        rows.push(row("surface", name, value, detail));
    }
    if let Some(features) = BUILD_FEATURES.get() {
        let mut sorted: Vec<&(String, bool)> = features.iter().collect();
        sorted.sort();
        rows.extend(
            sorted
                .into_iter()
                .filter(|(name, _)| name != "cloud")
                .map(|(name, on)| row("feature", name, state(*on), "")),
        );
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_registered_surface_is_a_capability_row() {
        register_surface("test.surface", "7", "preview");
        let rows = capabilities();
        assert!(rows.iter().any(|r| r[0] == "surface"
            && r[1] == "test.surface"
            && r[2] == "7"
            && r[3] == "preview"));
        register_surface("test.surface", "8", "");
        let rows = capabilities();
        let ours: Vec<_> = rows.iter().filter(|r| r[1] == "test.surface").collect();
        assert_eq!(ours.len(), 1, "a later call replaces the row");
        assert_eq!(ours[0][2], "8");
    }

    #[test]
    fn registered_build_features_are_feature_rows() {
        // The first registration of the process wins, so the rows are checked by shape: a
        // feature row has a name and "enabled" or "disabled".
        register_build_features(&[("zz_feature_on", true), ("zz_feature_off", false)]);
        let rows = capabilities();
        let features: Vec<&[String; 4]> = rows.iter().filter(|r| r[0] == "feature").collect();
        assert!(features.len() >= 2, "{features:?}");
        assert!(
            features
                .iter()
                .all(|r| r[2] == "enabled" || r[2] == "disabled")
        );
    }
}
