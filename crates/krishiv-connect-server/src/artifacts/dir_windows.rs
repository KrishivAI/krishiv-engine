// SPDX-License-Identifier: Apache-2.0
//! The artifact directories on Windows (Windows port, 0.3.0): the API of the Unix `dir.rs`
//! (design P3-T001 §3.6, T-H2; P3-T011a §3.9) over `std::fs`.
//!
//! Windows has no `openat` / `O_NOFOLLOW` walk, so every segment of a path is checked with
//! `symlink_metadata` before it is used: a symbolic link, a junction or any other reparse point
//! anywhere in a session's tree refuses the path, as a link does on Unix. Staged files are
//! created with `create_new` (never an existing name) and installed with `hard_link`, which fails
//! when the target name exists, so installing never replaces a file (the check and the install
//! are one step, as `linkat` is). The root is a fresh directory under the parent the server
//! names (by default the user's temporary directory, private to the user under Windows' default
//! ACLs), removed with its contents when dropped.
//!
//! Known limit against the Unix version: a reparse point created between the check of a segment
//! and its use is not caught (the Unix walk holds directory descriptors, Windows uses paths), and
//! the root's privacy rests on the parent directory's ACL, not on mode bits checked here.

use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The staging directory inside a session's directory. No artifact name can take it (names
/// start with a known prefix).
const STAGING: &str = ".staging";

/// The server's artifact root: a fresh directory.
#[derive(Debug)]
pub(crate) struct ArtifactRoot {
    path: PathBuf,
}

impl ArtifactRoot {
    /// Creates the directory `name` under `parent`, which must not exist yet. Its path is
    /// absolute (`parent` canonicalized): the paths handed to the UDF workers must be.
    pub(crate) fn create(parent: &Path, name: &str) -> io::Result<Self> {
        #[expect(
            clippy::disallowed_methods,
            reason = "runs on the blocking pool (`artifacts::blocking`), as the directory creation below"
        )]
        let path = std::fs::canonicalize(parent)?.join(name);
        #[expect(
            clippy::disallowed_methods,
            reason = "runs on the blocking pool (`artifacts::blocking`)"
        )]
        std::fs::create_dir(&path)?;
        Ok(Self { path })
    }

    /// The root's path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ArtifactRoot {
    fn drop(&mut self) {
        remove_tree(std::mem::take(&mut self.path));
    }
}

/// Removes the tree at `path` (std's `remove_dir_all` follows no symbolic link or junction), on
/// the blocking pool when a tokio runtime is running.
fn remove_tree(path: PathBuf) {
    if path.as_os_str().is_empty() {
        return;
    }
    let remove = move || {
        #[expect(
            clippy::disallowed_methods,
            reason = "runs on the blocking pool, or outside any runtime"
        )]
        let removed = std::fs::remove_dir_all(&path);
        if let Err(e) = removed
            && e.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!(target: "krishiv::connect", error = %e, "could not remove artifacts");
        }
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => drop(handle.spawn_blocking(remove)),
        Err(_) => remove(),
    }
}

/// One session's artifact directory and its staging directory. Removed with its contents when
/// the last holder drops it.
#[derive(Debug)]
pub(crate) struct SessionDir {
    /// Kept alive while a session directory exists under it.
    _root: Arc<ArtifactRoot>,
    path: PathBuf,
    staging: PathBuf,
}

/// How installing a staged file ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Install {
    /// The file is in place.
    Added,
    /// A file with the same name and the same bytes was there: nothing changed.
    Identical,
    /// Something else holds the name (other bytes, a directory, a link).
    Exists,
    /// A segment on the way is not a directory (a file or a link).
    Blocked,
}

/// Whether `path` is a plain directory: not a link, junction or other reparse point.
fn is_plain_dir(path: &Path) -> io::Result<bool> {
    #[expect(
        clippy::disallowed_methods,
        reason = "runs on the blocking pool (`artifacts::blocking`)"
    )]
    let m = std::fs::symlink_metadata(path)?;
    Ok(m.is_dir() && !m.file_type().is_symlink() && !is_reparse_point(&m))
}

/// Whether `m` describes a reparse point (a junction, a mount point, ...).
fn is_reparse_point(m: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;
    /// `FILE_ATTRIBUTE_REPARSE_POINT`.
    const REPARSE_POINT: u32 = 0x400;
    m.file_attributes() & REPARSE_POINT != 0
}

/// Creates the directory `name` in `dir` unless it exists, then checks it is a plain directory.
fn mkdir_checked(dir: &Path, name: &str) -> io::Result<PathBuf> {
    let path = dir.join(name);
    #[expect(
        clippy::disallowed_methods,
        reason = "runs on the blocking pool (`artifacts::blocking`)"
    )]
    match std::fs::create_dir(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    if is_plain_dir(&path)? {
        Ok(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            "an artifact path segment is not a plain directory",
        ))
    }
}

impl SessionDir {
    /// Creates (or opens) the directory `name` under `root`, and its staging directory.
    pub(crate) fn create(root: &Arc<ArtifactRoot>, name: &str) -> io::Result<Self> {
        let path = mkdir_checked(&root.path, name)?;
        let staging = mkdir_checked(&path, STAGING)?;
        Ok(Self {
            _root: Arc::clone(root),
            path,
            staging,
        })
    }

    /// The directory's path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// A new staged file `name`, which must not exist.
    pub(crate) fn create_staged(&self, name: &str) -> io::Result<File> {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.staging.join(name))
    }

    /// Removes the staged file `name`, if it is still there.
    pub(crate) fn remove_staged(&self, name: &str) {
        #[expect(
            clippy::disallowed_methods,
            reason = "runs on the blocking pool (`artifacts::blocking`)"
        )]
        match std::fs::remove_file(self.staging.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!(target: "krishiv::connect", error = %e, "could not remove a staged artifact");
            }
        }
    }

    /// Installs the staged file `staged` under the relative path `segments` (validated names),
    /// creating the directories on the way. Never replaces a name and never enters a link. The
    /// staged file is removed unless the result is an error.
    pub(crate) fn install(&self, staged: &str, segments: &[&str]) -> io::Result<Install> {
        let Some((leaf, parents)) = segments.split_last() else {
            return Ok(Install::Blocked);
        };
        let mut dir = self.path.clone();
        for segment in parents {
            dir = match mkdir_checked(&dir, segment) {
                Ok(p) => p,
                Err(e) if blocked(&e) => {
                    self.remove_staged(staged);
                    return Ok(Install::Blocked);
                }
                Err(e) => return Err(e),
            };
        }
        let source = self.staging.join(staged);
        let target = dir.join(leaf);
        #[expect(
            clippy::disallowed_methods,
            reason = "runs on the blocking pool (`artifacts::blocking`)"
        )]
        let outcome = match std::fs::hard_link(&source, &target) {
            Ok(()) => Install::Added,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                if same_bytes(&source, &target)? {
                    Install::Identical
                } else {
                    Install::Exists
                }
            }
            Err(e) => return Err(e),
        };
        self.remove_staged(staged);
        Ok(outcome)
    }

    /// Opens the installed regular file at the relative path `segments` for reading, refusing any
    /// segment that is a link or reparse point (`CloneSession`).
    pub(crate) fn open_installed(&self, segments: &[&str]) -> io::Result<File> {
        let Some((leaf, parents)) = segments.split_last() else {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        };
        let mut dir = self.path.clone();
        for segment in parents {
            dir = dir.join(segment);
            if !is_plain_dir(&dir)? {
                return Err(io::Error::from(io::ErrorKind::NotFound));
            }
        }
        let path = dir.join(leaf);
        if !is_regular_file(&path)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "an installed artifact is not a regular file",
            ));
        }
        File::open(path)
    }
}

impl Drop for SessionDir {
    fn drop(&mut self) {
        remove_tree(std::mem::take(&mut self.path));
    }
}

/// Whether `path` is a regular file: not a link, reparse point or directory.
fn is_regular_file(path: &Path) -> io::Result<bool> {
    #[expect(
        clippy::disallowed_methods,
        reason = "runs on the blocking pool (`artifacts::blocking`)"
    )]
    let m = std::fs::symlink_metadata(path)?;
    Ok(m.is_file() && !m.file_type().is_symlink() && !is_reparse_point(&m))
}

/// Whether `e` means a name on the way is not a directory we may enter.
fn blocked(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::NotADirectory | io::ErrorKind::AlreadyExists
    )
}

/// Whether the regular file `b` holds the bytes of `a`; anything other than a regular file
/// differs.
fn same_bytes(a: &Path, b: &Path) -> io::Result<bool> {
    if !is_regular_file(b)? {
        return Ok(false);
    }
    let (mut x, mut y) = (File::open(a)?, File::open(b)?);
    if x.metadata()?.len() != y.metadata()?.len() {
        return Ok(false);
    }
    let (mut bx, mut by) = (vec![0u8; 64 * 1024], vec![0u8; 64 * 1024]);
    loop {
        let n = read_full(&mut x, &mut bx)?;
        let m = read_full(&mut y, &mut by)?;
        if n != m || bx[..n] != by[..m] {
            return Ok(false);
        }
        if n == 0 {
            return Ok(true);
        }
    }
}

/// Reads until `buf` is full or the file ends.
fn read_full(f: &mut File, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}
