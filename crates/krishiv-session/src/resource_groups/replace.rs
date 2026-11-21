// SPDX-License-Identifier: Apache-2.0
//! `ReplaceResourceGroupConfig` (P3-T066 §5.4): a new group document written over the group
//! file and swapped in, refused when the caller's view is stale or the file cannot be replaced.

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::PoisonError;
use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_common::error::{
    InternalError, KrishivAdminPreconditionFailed, KrishivError, KrishivResourceGroupConfigReadOnly,
};

use super::{ResourceGroups, canonical, read_file};

/// Names the temporary files of one process apart.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

impl ResourceGroups {
    /// The etag of the file at `path` as it is on disk now, or `None` when it cannot be read.
    ///
    /// The etag of the file at `path` as it is on disk now (`file_etag`), or `None` when it cannot
    /// be read. It equals [`ResourceGroups::etag`] while the file holds the loaded document.
    #[must_use]
    pub fn file_etag(path: &str) -> Option<String> {
        read_file(path).ok().map(|text| canonical(&text).1)
    }

    /// Replaces the group file at `path` with `document` and swaps the tree for it. In order:
    /// a stale `etag`, then a file edited since it was loaded, then a target that is a link or not
    /// writable are refused; the document is validated; the file is written beside itself,
    /// synced, renamed over the original (mode kept) and the directory synced; then the tree is
    /// swapped as [`ResourceGroups::reload_json`] does; returns the document, etag and load time
    /// installed. Nothing changes on a refusal.
    ///
    /// # Errors
    ///
    /// `KRISHIV_ADMIN_PRECONDITION_FAILED` (`stale_etag`, `file_changed`),
    /// `KRISHIV_RESOURCE_GROUP_CONFIG_READ_ONLY`, `KRISHIV_RESOURCE_GROUP_CONFIG_INVALID` for an
    /// invalid document, `INTERNAL_ERROR` when the write fails.
    pub fn replace_file(
        &self,
        path: &str,
        document: &str,
        etag: &str,
    ) -> Result<(String, String, u64), KrishivError> {
        let _one = self
            .replacing
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let loaded = self.current();
        if loaded.etag != etag {
            return Err(precondition("stale_etag"));
        }
        if Self::file_etag(path).as_deref() != Some(loaded.etag.as_str()) {
            return Err(precondition("file_changed"));
        }
        let meta = fs::symlink_metadata(path)
            .map_err(|e| read_only(path, &format!("cannot read its metadata: {}", e.kind())))?;
        if meta.file_type().is_symlink() {
            return Err(read_only(
                path,
                "it is a symbolic link (a mounted ConfigMap?); change its source and reload",
            ));
        }
        if meta.permissions().readonly() {
            return Err(read_only(path, "it is not writable"));
        }
        if owner_would_change(&meta) {
            return Err(read_only(
                path,
                "it belongs to another user, and a replace would change its owner; edit it as \
                 that user and reload",
            ));
        }
        Self::validate(document, self.total_memory)?;
        match write_atomically(Path::new(path), document, &meta) {
            Ok(()) => {}
            Err(e) if is_read_only(&e) => {
                return Err(read_only(
                    path,
                    "its directory or file system is not writable here",
                ));
            }
            Err(e) => return Err(failed("cannot write the resource group file", &e)),
        }
        self.swap(document)
    }
}

fn precondition(reason: &str) -> KrishivError {
    KrishivAdminPreconditionFailed {
        reason: reason.to_owned(),
    }
    .into()
}

fn read_only(path: &str, reason: &str) -> KrishivError {
    KrishivResourceGroupConfigReadOnly {
        path: path.to_owned(),
        reason: reason.to_owned(),
    }
    .into()
}

/// Whether a new file made by this process would have another owner than `original`: a replace
/// writes a temporary file and renames it over the original, and a process cannot give a file to
/// another user without privilege, so a file owned by someone else is not replaced (design §5.4
/// asks for the original's owner). The process's user is its effective user id.
#[cfg(unix)]
fn owner_would_change(original: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    uid_differs(original.uid(), rustix::process::geteuid().as_raw())
}

/// Whether a file of `file_uid` would change owner when `process_uid` replaces it.
#[cfg(unix)]
fn uid_differs(file_uid: u32, process_uid: u32) -> bool {
    file_uid != process_uid
}

/// Non-Unix targets have no owner check: the file's owner is not read there, and a replace may
/// give the file to the server's account. Krishiv's server mode (the only caller of a replace)
/// supports Unix only, so this stays an unchecked, documented exclusion until a Windows server
/// build exists.
#[cfg(not(unix))]
fn owner_would_change(_: &fs::Metadata) -> bool {
    false
}

/// A write refused for want of permission or on a read-only mount.
fn is_read_only(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::ReadOnlyFilesystem
    )
}

fn failed(what: &str, e: &std::io::Error) -> KrishivError {
    KrishivError::from(InternalError {
        message: format!("{what}: {}", e.kind()),
    })
}

/// Temporary file in the same directory (with the original's permissions), fsync, rename, fsync
/// of the directory: a reader sees the old file or the new one, never half of it. A failed
/// directory sync after the rename is logged, not returned: the file is already replaced, so the
/// tree must follow it.
fn write_atomically(path: &Path, text: &str, original: &fs::Metadata) -> std::io::Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));
    let temp = dir.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("groups"),
        std::process::id(),
        TEMP_SEQ.fetch_add(1, Ordering::Relaxed),
    ));
    let write = || -> std::io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(text.as_bytes())?;
        fs::set_permissions(&temp, original.permissions())?;
        file.sync_all()?;
        fs::rename(&temp, path)
    };
    write().inspect_err(|_| {
        let _ = fs::remove_file(&temp);
    })?;
    if let Err(e) = fs::File::open(dir).and_then(|d| d.sync_all()) {
        tracing::warn!(error = %e, "the resource group file's directory could not be synced");
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn a_file_owned_by_another_user_would_change_owner() {
        assert!(!uid_differs(1000, 1000));
        assert!(uid_differs(0, 1000));
        assert!(uid_differs(1000, 0));
    }

    #[test]
    fn a_file_this_process_made_keeps_its_owner() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let file = dir.path().join("f");
        fs::write(&file, "x").unwrap_or_else(|e| panic!("{e}"));
        let meta = fs::metadata(&file).unwrap_or_else(|e| panic!("{e}"));
        assert!(!owner_would_change(&meta));
    }

    #[test]
    fn a_replace_refuses_a_file_owned_by_another_user() {
        // Giving a file away needs privilege: where the tests do not run as root there is no
        // foreign-owned fixture to make, and the comparison above covers the rule.
        if rustix::process::geteuid().as_raw() != 0 {
            return;
        }
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let path = dir.path().join("groups.json");
        let text = r#"{"rootGroups":[{"name":"root","hardConcurrencyLimit":1,"maxQueued":1}],"defaultGroup":"root"}"#;
        fs::write(&path, text).unwrap_or_else(|e| panic!("{e}"));
        std::os::unix::fs::chown(&path, Some(65_534), None).unwrap_or_else(|e| panic!("{e}"));
        let groups = ResourceGroups::from_file(path.to_str().unwrap_or_default(), 1 << 30)
            .unwrap_or_else(|e| panic!("{e}"));
        // As root every file is writable, but a file of another owner still must not be replaced
        // by a root-owned copy.
        let err = groups
            .replace_file(path.to_str().unwrap_or_default(), "{}", &groups.etag())
            .expect_err("a foreign-owned file is refused");
        assert!(err.to_string().contains("belongs to another user"), "{err}");
        assert_eq!(
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("{e}")),
            text,
            "the file is unchanged"
        );
    }
}
