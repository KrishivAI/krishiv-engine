// SPDX-License-Identifier: Apache-2.0
//! The session SQL path (`SET PATH`): the stored entries, the expansion of
//! `DEFAULT_PATH`, `SYSTEM_PATH` and `PATH`, and the duplicate check.

use krishiv_common::error::{DuplicateSqlPathEntry, KrishivError};
use krishiv_common::sql_path::{PathElement, parse_conf, render};

/// One stored path entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Entry {
    /// A schema, as its parts (case kept as written).
    Schema(Vec<String>),
    /// The current schema, resolved whenever the path is read.
    CurrentSchema,
}

impl Entry {
    /// The entry's parts; the current schema is `spark_catalog.<namespace>`.
    pub(crate) fn parts(&self, namespace: &str) -> Vec<String> {
        match self {
            Self::Schema(parts) => parts.clone(),
            Self::CurrentSchema => vec![crate::catalog::CATALOG.to_owned(), namespace.to_owned()],
        }
    }
}

fn system(name: &str) -> Entry {
    Entry::Schema(vec!["system".to_owned(), name.to_owned()])
}

/// What `DEFAULT_PATH` expands to under `krishiv.sql.defaultPath`.
pub(crate) fn default_entries(conf: &str) -> Vec<Entry> {
    match parse_conf(conf) {
        Some(elements) if !elements.is_empty() => {
            // `PATH` inside the configuration expands to nothing (there is no live
            // path to read), and `DEFAULT_PATH` inside it expands with no
            // configuration, so it cannot recurse.
            elements.iter().flat_map(|e| one(e, &[], "")).collect()
        }
        _ => vec![system("builtin"), system("session"), Entry::CurrentSchema],
    }
}

fn one(element: &PathElement, live: &[Entry], default_conf: &str) -> Vec<Entry> {
    match element {
        PathElement::Schema(parts) => vec![Entry::Schema(parts.clone())],
        PathElement::CurrentSchema { .. } => vec![Entry::CurrentSchema],
        PathElement::SystemPath => vec![system("builtin")],
        PathElement::DefaultPath => default_entries(default_conf),
        PathElement::Path => live.to_vec(),
    }
}

/// Expands `elements` against the `live` path; two entries naming one
/// namespace (also after expansion) are `DUPLICATE_SQL_PATH_ENTRY`.
pub(crate) fn expand(
    elements: &[PathElement],
    live: &[Entry],
    default_conf: &str,
    namespace: &str,
) -> Result<Vec<Entry>, KrishivError> {
    let mut out: Vec<Entry> = Vec::new();
    for element in elements {
        for entry in one(element, live, default_conf) {
            let key = |e: &Entry| -> Vec<String> {
                e.parts(namespace)
                    .iter()
                    .map(|p| p.to_lowercase())
                    .collect()
            };
            if out.iter().any(|e| key(e) == key(&entry)) {
                let path_entry = match (&entry, element) {
                    (Entry::CurrentSchema, _) => "current_schema".to_owned(),
                    (Entry::Schema(parts), _) => render(parts),
                };
                return Err(DuplicateSqlPathEntry { path_entry }.into());
            }
            out.push(entry);
        }
    }
    Ok(out)
}
