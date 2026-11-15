// SPDX-License-Identifier: Apache-2.0
//! The partition columns of a directory of files, read from its `key=value` directories (`CREATE TABLE ... USING
//! fmt LOCATION` without a column list).

use std::path::{Path, PathBuf};

use krishiv_common::error::KrishivError;

/// Whether the directory name `name` is a partition level (`key=value`): not hidden (`.hive-staging`), and not a
/// `_`-prefixed name without a `=` (`_SUCCESS`, `_temporary`). This is the file reader's own filter (Spark's
/// `shouldFilterOutPathName`), so a `_p=10` directory is a partition there and here.
pub(crate) fn is_partition_dir_name(name: &str) -> bool {
    !name.starts_with('.')
        && !(name.starts_with('_') && !name.contains('='))
        && name.split_once('=').is_some_and(|(k, _)| !k.is_empty())
}

/// The partition columns of the files under the local directory `location`: the key of each level of `key=value`
/// directories, following the first directory (by path) at each level. Empty for a path that is not a local
/// directory (it does not exist here) or has no such directories. Spark takes a table's partition columns from them
/// when `CREATE TABLE ... USING fmt LOCATION` has no column list. Local directories only (an object-store location
/// finds none: its `key=value` prefixes stay ordinary data columns of the inferred schema), at most 32 levels;
/// keys are not percent-decoded.
///
/// # Errors
/// A directory that exists but cannot be listed, or an entry that cannot be read: the layout is not guessed as
/// unpartitioned.
pub(crate) fn partition_keys(location: &Path) -> Result<Vec<String>, KrishivError> {
    let mut dir = location.to_path_buf();
    let mut keys: Vec<String> = Vec::new();
    loop {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                break;
            }
            Err(e) => return Err(listing_error(&dir, &e)),
        };
        let mut next: Option<(String, PathBuf)> = None;
        for e in entries {
            let e = e.map_err(|e| listing_error(&dir, &e))?;
            let name = e.file_name().to_string_lossy().into_owned();
            if is_partition_dir_name(&name)
                && e.path().is_dir()
                && next.as_ref().is_none_or(|(_, p)| e.path() < *p)
            {
                let key = name
                    .split_once('=')
                    .map_or(String::new(), |(k, _)| k.to_owned());
                next = Some((key, e.path()));
            }
        }
        let Some((key, path)) = next else { break };
        if keys.iter().any(|k| k.eq_ignore_ascii_case(&key)) || keys.len() >= 32 {
            break;
        }
        keys.push(key);
        dir = path;
    }
    Ok(keys)
}

fn listing_error(dir: &Path, e: &std::io::Error) -> KrishivError {
    crate::partitions::internal(format!("cannot list {}: {e}", dir.display()))
}
