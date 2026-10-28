// SPDX-License-Identifier: Apache-2.0
//! Split test manifests: one small JSON file per key.
//!
//! The analysis manifests (`tests/fixtures/analysis/`) and the golden
//! baseline (`tests/spark-golden/known-failures-cases/` and
//! `known-failures-categories/`) each used to be a single JSON file, so any
//! two PRs touching different functions, coercion rules, golden files or
//! failure categories conflicted on the same lines. They are now
//! directories with one file per key (function name, coercion input file,
//! golden input file, category); two PRs touching different keys never touch
//! the same file. The tests read the whole directory (sorted, so deterministic);
//! totals and counts are computed from what they read, never stored.
//!
//! [`sync_json_dir`] is the update mode (`KRISHIV_UPDATE_MANIFEST=1`, the
//! golden `--ratchet update`): it rewrites only the files whose bytes would
//! change and deletes the files whose keys no longer exist, so running it
//! twice leaves a clean tree. [`encode_file_name`] maps flat keys that are
//! not file names (the operator functions `/`, `&`, ...) to file names and
//! back.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

/// What [`sync_json_dir`] changed, as paths relative to the directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sync {
    /// Files written (new or changed content).
    pub written: Vec<PathBuf>,
    /// Files deleted (keys that no longer exist).
    pub removed: Vec<PathBuf>,
}

/// Every `.json` file under `dir`, recursively, as
/// (path relative to `dir`, text), sorted by path.
///
/// A missing directory reads as empty, so a fresh checkout starts clean;
/// any other I/O error is returned.
pub fn read_json_dir(dir: &Path) -> io::Result<Vec<(PathBuf, String)>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let entries = match std::fs::read_dir(&current) {
            Err(e) if e.kind() == io::ErrorKind::NotFound && current.as_path() == dir => {
                return Ok(out);
            }
            Err(e) => {
                return Err(io::Error::new(
                    e.kind(),
                    format!("{}: {e}", current.display()),
                ));
            }
            Ok(entries) => entries,
        };
        for entry in entries {
            let path = entry
                .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", current.display())))?
                .path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "json") {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
                let rel = path
                    .strip_prefix(dir)
                    .map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("{}: {e}", path.display()),
                        )
                    })?
                    .to_path_buf();
                out.push((rel, text));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Write `text` to `path` (creating parent directories) unless the file
/// already holds exactly those bytes. Returns whether the file was written.
pub fn write_if_changed(path: &Path, text: &str) -> io::Result<bool> {
    match std::fs::read(path) {
        Ok(old) if old == text.as_bytes() => return Ok(false),
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io::Error::new(e.kind(), format!("{}: {e}", path.display())));
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", parent.display())))?;
    }
    std::fs::write(path, text)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    Ok(true)
}

/// Bring the `.json` files under `dir` in line with `wanted` (relative path
/// to file text): write the files whose bytes would change, delete the
/// `.json` files no key wants any more, and prune the directories left
/// empty by the deletions. Returns what changed, sorted.
pub fn sync_json_dir(dir: &Path, wanted: &BTreeMap<PathBuf, String>) -> io::Result<Sync> {
    let mut sync = Sync::default();
    for (rel, text) in wanted {
        if write_if_changed(&dir.join(rel), text)? {
            sync.written.push(rel.clone());
        }
    }
    for (rel, _) in read_json_dir(dir)? {
        if !wanted.contains_key(&rel) {
            std::fs::remove_file(dir.join(&rel))
                .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", rel.display())))?;
            sync.removed.push(rel);
        }
    }
    prune_empty_dirs(dir);
    sync.written.sort();
    sync.removed.sort();
    Ok(sync)
}

/// Remove the directories under `dir` left empty by stale-file deletion.
/// Best effort: a directory that is not empty (or cannot be read) is kept.
fn prune_empty_dirs(dir: &Path) {
    let mut stack = vec![dir.to_path_buf()];
    let mut dirs = Vec::new();
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        dirs.push(current.clone());
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                stack.push(entry.path());
            }
        }
    }
    dirs.sort_by(|a, b| b.cmp(a));
    for current in dirs {
        if current != *dir {
            let _ = std::fs::remove_dir(&current);
        }
    }
}

/// Map a flat manifest key to a file stem: ASCII letters, digits, `_`, `-`
/// and `.` stay as they are, every other byte becomes `%XX` (uppercase
/// hex). The mapping is injective, so the operator functions (`/`, `&`,
/// `|`, ...) each get their own file.
#[must_use]
pub fn encode_file_name(key: &str) -> String {
    let mut out = String::new();
    for b in key.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The inverse of [`encode_file_name`]: `None` for a stem that is not an
/// encoding (a truncated or non-hex `%` escape).
#[must_use]
pub fn decode_file_name(stem: &str) -> Option<String> {
    let mut bytes = Vec::new();
    let mut chars = stem.bytes();
    let hex = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    while let Some(b) = chars.next() {
        if b == b'%' {
            let (hi, lo) = (chars.next()?, chars.next()?);
            bytes.push(hex(hi)? * 16 + hex(lo)?);
        } else {
            bytes.push(b);
        }
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    type R = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn operator_names_encode_to_distinct_stems_and_back() {
        for key in ["/", "&", "|", "||", "%", "+", "<=>", " Berge", "a/b%c"] {
            let stem = encode_file_name(key);
            assert!(
                !stem.contains('/'),
                "{key:?} encodes with a separator: {stem:?}"
            );
            assert_eq!(decode_file_name(&stem).as_deref(), Some(key));
        }
        assert_eq!(encode_file_name("upper"), "upper");
        assert_eq!(decode_file_name("upper").as_deref(), Some("upper"));
        assert_eq!(decode_file_name("%2"), None);
        assert_eq!(decode_file_name("%ZZ"), None);
        assert_eq!(decode_file_name("%2f"), Some("/".to_owned()));
    }

    #[test]
    fn missing_dir_reads_empty_and_round_trips() -> R {
        let dir = tempfile::tempdir()?;
        let missing = dir.path().join("nope");
        assert!(read_json_dir(&missing)?.is_empty());
        let wanted = BTreeMap::from([
            (PathBuf::from("b.json"), "{\"b\":1}\n".to_owned()),
            (PathBuf::from("sub/a.json"), "{\"a\":1}\n".to_owned()),
        ]);
        let sync = sync_json_dir(&missing, &wanted)?;
        assert_eq!(sync.written.len(), 2);
        assert!(sync.removed.is_empty());
        let again = sync_json_dir(&missing, &wanted)?;
        assert!(again.written.is_empty() && again.removed.is_empty());
        let read = read_json_dir(&missing)?;
        assert_eq!(read.len(), 2);
        assert!(read[0].0 < read[1].0);
        Ok(())
    }

    #[test]
    fn stale_files_go_and_unrelated_files_stay() -> R {
        let dir = tempfile::tempdir()?;
        let wanted = BTreeMap::from([(PathBuf::from("keep.json"), "1\n".to_owned())]);
        sync_json_dir(dir.path(), &wanted)?;
        std::fs::write(dir.path().join("notes.txt"), "keep me")?;
        let next = BTreeMap::from([(PathBuf::from("new.json"), "2\n".to_owned())]);
        let sync = sync_json_dir(dir.path(), &next)?;
        assert_eq!(sync.removed, [PathBuf::from("keep.json")]);
        assert_eq!(sync.written, [PathBuf::from("new.json")]);
        assert!(dir.path().join("notes.txt").is_file());
        Ok(())
    }

    #[test]
    fn write_if_changed_reports_and_keeps_bytes() -> R {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("x.json");
        assert!(write_if_changed(&path, "a\n")?);
        assert!(!write_if_changed(&path, "a\n")?);
        assert!(write_if_changed(&path, "b\n")?);
        assert_eq!(std::fs::read_to_string(&path)?, "b\n");
        Ok(())
    }
}
