// SPDX-License-Identifier: Apache-2.0
//! The artifact directories on disk: the server's private root, one
//! directory per session under it, and the staging directory inside each
//! (design P3-T001 §3.6, T-H2; P3-T011a §3.9).
//!
//! Every name is resolved relative to an open directory descriptor, one
//! segment at a time, with `O_NOFOLLOW`: a symbolic link planted anywhere in
//! a session's tree is never followed, so a write cannot leave the tree.
//! Staged files are created with `O_CREAT | O_EXCL | O_NOFOLLOW` (mode
//! 0600) and installed with `linkat`, which never replaces an existing
//! name: the check that a target is free and the install are one atomic
//! step (no time-of-check to time-of-use window). Directories are 0700; the
//! root is a fresh directory owned by this process's user and closed to
//! group and others, removed with its contents when dropped.

use std::fs::File;
use std::io::{self, Read};
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustix::fs::{AtFlags, CWD, Mode, OFlags, fstat, linkat, mkdirat, openat, unlinkat};
use rustix::io::Errno;

/// How a directory is opened: readable (macOS has no `O_PATH`), never
/// through a symbolic link.
const DIR_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

/// The staging directory inside a session's directory. No artifact name can
/// take it (names start with a known prefix).
const STAGING: &str = ".staging";

/// The mode bits group and others may not have on the root.
const NOT_PRIVATE: u32 = 0o077;

/// The server's artifact root: a fresh private directory.
#[derive(Debug)]
pub(crate) struct ArtifactRoot {
    path: PathBuf,
    fd: OwnedFd,
}

impl ArtifactRoot {
    /// Creates the directory `name` (mode 0700) under `parent` and checks
    /// it is a directory this user owns that group and others cannot use.
    pub(crate) fn create(parent: &Path, name: &str) -> io::Result<Self> {
        let path = parent.join(name);
        let mut builder = std::fs::DirBuilder::new();
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder.create(&path)?;
        let opened = openat(CWD, &path, DIR_FLAGS, Mode::empty())
            .map_err(io::Error::from)
            .and_then(|fd| {
                let st = fstat(&fd)?;
                if st.st_uid != rustix::process::geteuid().as_raw() || st.st_mode & NOT_PRIVATE != 0
                {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "the artifact root is not private to this user",
                    ));
                }
                Ok(fd)
            });
        match opened {
            Ok(fd) => Ok(Self { path, fd }),
            Err(e) => {
                // Only the directory this call created (empty); on the
                // blocking path that creates the root.
                #[expect(
                    clippy::disallowed_methods,
                    reason = "runs on the blocking pool (`artifacts::blocking`)"
                )]
                let _ = std::fs::remove_dir(&path);
                Err(e)
            }
        }
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

/// Removes the tree at `path` (std's `remove_dir_all` follows no symbolic
/// link), on the blocking pool when a tokio runtime is running: a session is
/// often released from an async worker.
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

/// One session's artifact directory and its staging directory. Removed
/// with its contents when the last holder drops it (the session closed and
/// no transfer still writes).
#[derive(Debug)]
pub(crate) struct SessionDir {
    /// Kept alive while a session directory exists under it.
    _root: Arc<ArtifactRoot>,
    path: PathBuf,
    fd: OwnedFd,
    staging: OwnedFd,
}

/// How installing a staged file ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Install {
    /// The file is in place.
    Added,
    /// A file with the same name and the same bytes was there: nothing
    /// changed.
    Identical,
    /// Something else holds the name (other bytes, a directory, a link).
    Exists,
    /// A segment on the way is not a directory (a file or a link).
    Blocked,
}

impl SessionDir {
    /// Creates (or opens) the directory `name` under `root`, and its
    /// staging directory, both 0700.
    pub(crate) fn create(root: &Arc<ArtifactRoot>, name: &str) -> io::Result<Self> {
        let fd = mkdir_open(&root.fd, name)?;
        let staging = mkdir_open(&fd, STAGING)?;
        Ok(Self {
            _root: Arc::clone(root),
            path: root.path.join(name),
            fd,
            staging,
        })
    }

    /// The directory's path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// A new staged file `name` (mode 0600), which must not exist.
    pub(crate) fn create_staged(&self, name: &str) -> io::Result<File> {
        let flags =
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
        let fd = openat(&self.staging, name, flags, Mode::RUSR | Mode::WUSR)?;
        Ok(File::from(fd))
    }

    /// Removes the staged file `name`, if it is still there.
    pub(crate) fn remove_staged(&self, name: &str) {
        match unlinkat(&self.staging, name, AtFlags::empty()) {
            Ok(()) | Err(Errno::NOENT) => {}
            Err(e) => {
                tracing::warn!(target: "krishiv::connect", error = %e, "could not remove a staged artifact");
            }
        }
    }

    /// Installs the staged file `staged` under the relative path
    /// `segments` (validated names), creating the directories on the way.
    /// Never replaces a name and never follows a link. The staged file is
    /// removed unless the result is an error.
    pub(crate) fn install(&self, staged: &str, segments: &[&str]) -> io::Result<Install> {
        let Some((leaf, parents)) = segments.split_last() else {
            return Ok(Install::Blocked);
        };
        let mut parent: Option<OwnedFd> = None;
        for segment in parents {
            let cur = parent.as_ref().map_or(self.fd.as_fd(), AsFd::as_fd);
            let next = match mkdir_open(cur, segment) {
                Ok(fd) => fd,
                Err(e) if blocked(&e) => {
                    self.remove_staged(staged);
                    return Ok(Install::Blocked);
                }
                Err(e) => return Err(e),
            };
            parent = Some(next);
        }
        let cur = parent.as_ref().map_or(self.fd.as_fd(), AsFd::as_fd);
        let outcome = match linkat(&self.staging, staged, cur, *leaf, AtFlags::empty()) {
            Ok(()) => Install::Added,
            Err(Errno::EXIST) => {
                if same_bytes(&self.staging, staged, cur, leaf)? {
                    Install::Identical
                } else {
                    Install::Exists
                }
            }
            Err(e) => return Err(e.into()),
        };
        self.remove_staged(staged);
        Ok(outcome)
    }
}

impl SessionDir {
    /// Opens the installed regular file at the relative path `segments` for
    /// reading, every segment without following a link (`CloneSession`).
    pub(crate) fn open_installed(&self, segments: &[&str]) -> io::Result<File> {
        let Some((leaf, parents)) = segments.split_last() else {
            return Err(io::Error::from(io::ErrorKind::NotFound));
        };
        let mut parent: Option<OwnedFd> = None;
        for segment in parents {
            let cur = parent.as_ref().map_or(self.fd.as_fd(), AsFd::as_fd);
            parent = Some(openat(cur, *segment, DIR_FLAGS, Mode::empty())?);
        }
        let cur = parent.as_ref().map_or(self.fd.as_fd(), AsFd::as_fd);
        let read = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
        let fd = openat(cur, *leaf, read, Mode::empty())?;
        let st = fstat(&fd)?;
        if rustix::fs::FileType::from_raw_mode(st.st_mode) != rustix::fs::FileType::RegularFile {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "an installed artifact is not a regular file",
            ));
        }
        Ok(File::from(fd))
    }
}

impl Drop for SessionDir {
    fn drop(&mut self) {
        remove_tree(std::mem::take(&mut self.path));
    }
}

/// Whether `e` means a name on the way is not a directory we may enter.
fn blocked(e: &io::Error) -> bool {
    matches!(
        e.raw_os_error().map(Errno::from_raw_os_error),
        Some(Errno::LOOP | Errno::NOTDIR | Errno::EXIST)
    )
}

/// Creates the directory `name` in `dir` (0700) unless it exists, then
/// opens it without following a link.
fn mkdir_open(dir: impl AsFd, name: &str) -> io::Result<OwnedFd> {
    match mkdirat(dir.as_fd(), name, Mode::RWXU) {
        Ok(()) | Err(Errno::EXIST) => {}
        Err(e) => return Err(e.into()),
    }
    Ok(openat(dir, name, DIR_FLAGS, Mode::empty())?)
}

/// Whether the regular file `b` in `b_dir` holds the bytes of `a` in
/// `a_dir`; anything other than a regular file differs.
fn same_bytes(a_dir: impl AsFd, a: &str, b_dir: impl AsFd, b: &str) -> io::Result<bool> {
    let read = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK;
    let theirs = match openat(b_dir, b, read, Mode::empty()) {
        Ok(fd) => fd,
        Err(Errno::LOOP | Errno::NXIO) => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    let ours = openat(a_dir, a, read, Mode::empty())?;
    let (st_theirs, st_ours) = (fstat(&theirs)?, fstat(&ours)?);
    let regular = |st: &rustix::fs::Stat| {
        rustix::fs::FileType::from_raw_mode(st.st_mode) == rustix::fs::FileType::RegularFile
    };
    if !regular(&st_theirs) || st_theirs.st_size != st_ours.st_size {
        return Ok(false);
    }
    let (mut x, mut y) = (File::from(ours), File::from(theirs));
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

#[cfg(test)]
mod tests;
