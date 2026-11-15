// SPDX-License-Identifier: Apache-2.0
//! The partition columns of a directory of files, read from its `key=value` directories (`CREATE TABLE ... USING
//! fmt LOCATION` without a column list).

use std::path::{Path, PathBuf};

/// The partition columns of the files under the local directory `location`: the key of each level of `key=value`
/// directories, following the first directory (by path) at each level. Empty for a path that is not a local directory.
/// Spark takes a table's partition columns from them when `CREATE TABLE ... USING fmt LOCATION` has no column list.
/// Local directories only (an object-store location finds none: its `key=value` prefixes stay ordinary data columns of the inferred
/// schema), at most 32 levels; keys are not percent-decoded.
pub(crate) fn partition_keys(location: &Path) -> Vec<String> {
    let mut dir = location.to_path_buf();
    let mut keys: Vec<String> = Vec::new();
    loop {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            break;
        };
        let mut next: Option<(String, PathBuf)> = None;
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            // Hidden entries and `_`-prefixed ones (`_SUCCESS`, `_temporary`, `.hive-staging`) are not partitions.
            if !name.starts_with(['.', '_'])
                && e.path().is_dir()
                && let Some((k, _)) = name.split_once('=')
                && !k.is_empty()
                && next.as_ref().is_none_or(|(_, p)| e.path() < *p)
            {
                next = Some((k.to_owned(), e.path()));
            }
        }
        let Some((key, path)) = next else { break };
        if keys.iter().any(|k| k.eq_ignore_ascii_case(&key)) || keys.len() >= 32 {
            break;
        }
        keys.push(key);
        dir = path;
    }
    keys
}
