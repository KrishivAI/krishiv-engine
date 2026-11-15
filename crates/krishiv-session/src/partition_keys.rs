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
        && !name.ends_with("._COPYING_")
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
                && is_dir(&e.path())?
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

/// Whether `path` is a directory (links followed). An entry that vanished since the listing is not one; any other
/// failure to stat it is an error, not a reason to skip it.
fn is_dir(path: &Path) -> Result<bool, KrishivError> {
    match std::fs::metadata(path) {
        Ok(m) => Ok(m.is_dir()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(listing_error(path, &e)),
    }
}

fn listing_error(dir: &Path, e: &std::io::Error) -> KrishivError {
    crate::partitions::internal(format!("cannot list {}: {e}", dir.display()))
}

#[cfg(test)]
mod tests {
    use super::partition_keys;

    #[test]
    fn a_missing_path_or_a_plain_file_has_no_partition_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(
            partition_keys(&dir.path().join("absent"))
                .expect("absent")
                .is_empty()
        );
        let file = dir.path().join("f.parquet");
        std::fs::write(&file, b"x").expect("write");
        assert!(partition_keys(&file).expect("file").is_empty());
    }

    #[test]
    fn a_copying_directory_is_not_a_partition_level() {
        use super::is_partition_dir_name;
        assert!(!is_partition_dir_name("a=1._COPYING_"));
        assert!(is_partition_dir_name("a=1"));
        assert!(is_partition_dir_name("_p=10"));
    }

    #[test]
    fn a_repeated_key_ends_the_descent() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("a=1").join("A=2").join("b=3")).expect("mkdir");
        assert_eq!(
            partition_keys(dir.path()).expect("keys"),
            vec!["a".to_owned()]
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_directory_is_an_error_not_an_unpartitioned_table() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let locked = dir.path().join("a=1");
        std::fs::create_dir(&locked).expect("mkdir");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod");
        let readable = std::fs::read_dir(&locked).is_ok(); // root ignores the mode
        let result = partition_keys(dir.path());
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("restore");
        if !readable {
            assert!(result.is_err());
        }
    }
}
