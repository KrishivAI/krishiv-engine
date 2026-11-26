// SPDX-License-Identifier: Apache-2.0
//! `OwnedDirectory`: a directory a statement created exclusively, removed again if the statement fails.

use std::path::{Path, PathBuf};

/// Why an [`OwnedDirectory`] could not be made.
#[derive(Debug)]
pub(crate) enum OwnedDirectoryError {
    /// Something is already at the path (not ours to touch).
    Exists,
    Io(String),
}

/// A directory this statement created exclusively. Dropped without [`Self::keep`] (the statement
/// failed, or its future was dropped), it removes what is in it, provided the path still names the
/// directory that was created (same device and inode): a path swapped for another directory in the
/// meantime is left alone.
pub(crate) struct OwnedDirectory {
    path: PathBuf,
    identity: Option<(u64, u64)>,
    kept: bool,
}

impl OwnedDirectory {
    pub(crate) fn create(path: PathBuf) -> Result<Self, OwnedDirectoryError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                OwnedDirectoryError::Io(format!("cannot create {}: {e}", parent.display()))
            })?;
        }
        match std::fs::create_dir(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(OwnedDirectoryError::Exists);
            }
            Err(e) => {
                return Err(OwnedDirectoryError::Io(format!(
                    "cannot create {}: {e}",
                    path.display()
                )));
            }
        }
        let identity = Self::identity_of(&path);
        Ok(Self {
            path,
            identity,
            kept: false,
        })
    }

    fn identity_of(path: &Path) -> Option<(u64, u64)> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let meta = std::fs::symlink_metadata(path).ok()?;
            meta.is_dir().then(|| (meta.dev(), meta.ino()))
        }
        // Windows' file id (volume serial, file index) is not in stable std: the creation time
        // (100 ns ticks) tells a directory made later at the same path apart.
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            let meta = std::fs::symlink_metadata(path).ok()?;
            meta.is_dir().then(|| (meta.creation_time(), 0))
        }
        #[cfg(not(any(unix, windows)))]
        {
            let meta = std::fs::symlink_metadata(path).ok()?;
            meta.is_dir().then_some((0, 0))
        }
    }

    /// The statement succeeded: the directory stays.
    pub(crate) fn keep(mut self) {
        self.kept = true;
    }
}

impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        if !self.kept && self.identity.is_some() && Self::identity_of(&self.path) == self.identity {
            drop(std::fs::remove_dir_all(&self.path));
        }
    }
}
