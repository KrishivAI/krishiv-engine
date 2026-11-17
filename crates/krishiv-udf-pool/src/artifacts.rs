// SPDX-License-Identifier: Apache-2.0
//! A session's Python artifacts (P3-T014 PR-26c-3): the pyfiles and archives a Connect client added
//! (`addArtifacts(pyfile=True)`, `addArtifacts(archive=True)`), as its workers see them.
//!
//! Spark's worker puts its files directory and each `.zip`/`.egg` include on `sys.path` (`add_path`, at index
//! 1, `worker_util.py` L59-65 and L133-154 at v4.2.0), and its executor unpacks archives into its working
//! directory (`Utils.unpack`: `.zip`, `.jar`, `.tar`, `.tar.gz`, `.tgz`). Here a set of artifacts is prepared
//! once, by the first worker start that needs it, into a directory of the pool's own:
//! - `py/` holds a copy of every pyfile; it, then each pyfile that is not a `.py` (a zip, an egg), go on the
//!   worker's `sys.path` in Spark's order;
//! - `archives/<name>/` holds each archive unpacked (by the archive's file name), also on `sys.path`.
//!
//! The directory is one of the worker's read-only paths in the sandbox (Landlock), and nothing else of the
//! server's artifact directory is. A worker serves one set only (the pool key carries the set's id), since its
//! read paths are fixed when it starts.
//!
//! **Trust.** The server names the paths; the client chose their contents (and maybe their names). So a path
//! must be absolute, a regular file, not a symbolic link and with no symbolic link above it (its canonical
//! form is itself): Landlock grants a path by the file it opens, so a link would hand the worker its target.
//! Two pyfiles or two archives of one name are refused (their copies would collide). The copies and the
//! unpacked bytes together are capped ([`crate::PoolConfig::artifact_limit`]). An archive is unpacked by the
//! worker interpreter's own `zipfile`/`tarfile`, in the sandbox when it is on (it reads the archive and writes
//! only its own directory), with `tarfile`'s `data` filter: no absolute paths, no `..`, no link leaving the
//! directory, no device files; a zip's entries cannot be links.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use krishiv_common::error::{ArtifactAlreadyExists, InvalidArtifactPath, KrishivError};

/// Ids of artifact sets, unique in the process; 0 is the empty set.
static NEXT_SET: AtomicU64 = AtomicU64::new(1);

/// One session's artifacts, as set by [`crate::SessionUdfs::set_artifacts`].
pub(crate) struct ArtifactSet {
    /// The pool key's part: a worker started for one set never serves another.
    pub(crate) id: u64,
    pyfiles: Vec<PathBuf>,
    archives: Vec<PathBuf>,
    /// Why the paths given are refused (checked when they are set; raised by the calls that need them).
    refused: Option<KrishivError>,
    /// The prepared directory, made by the first worker start that needs it (or why it could not be made).
    prepared: Mutex<Option<Result<Arc<Prepared>, KrishivError>>>,
}

/// A set's prepared directory: what a worker of the set reads.
pub(crate) struct Prepared {
    /// Removed when the set and every worker started for it are gone.
    _dir: tempfile::TempDir,
    /// The worker's `sys.path` entries, in the order they are added (each at index 1, as Spark's `add_path`).
    pub(crate) sys_path: Vec<PathBuf>,
    /// The worker's extra read-only paths in the sandbox.
    pub(crate) read_paths: Vec<PathBuf>,
}

impl ArtifactSet {
    /// No artifacts.
    pub(crate) fn empty() -> Self {
        ArtifactSet {
            id: 0,
            pyfiles: Vec::new(),
            archives: Vec::new(),
            refused: None,
            prepared: Mutex::new(None),
        }
    }

    /// The set of `pyfiles` and `archives`, their paths checked now (a refusal is raised by the calls).
    pub(crate) fn new(pyfiles: Vec<PathBuf>, archives: Vec<PathBuf>) -> Self {
        if pyfiles.is_empty() && archives.is_empty() {
            return Self::empty();
        }
        let refused = check(&pyfiles).and_then(|()| check(&archives)).err();
        ArtifactSet {
            id: NEXT_SET.fetch_add(1, Ordering::Relaxed),
            pyfiles,
            archives,
            refused,
            prepared: Mutex::new(None),
        }
    }

    /// Whether the set has no artifacts.
    pub(crate) fn is_empty(&self) -> bool {
        self.id == 0
    }

    /// Why the set's paths are refused, if they are.
    pub(crate) fn refused(&self) -> Option<&KrishivError> {
        self.refused.as_ref()
    }

    /// The prepared directory: made by the first caller (under `root`, the archives unpacked by `unpack`, all
    /// within `limit` bytes), the same one (or the same failure) for every later caller.
    pub(crate) fn prepare(
        &self,
        root: &Path,
        limit: u64,
        unpack: &dyn Fn(&Path, &Path, u64) -> Result<u64, KrishivError>,
    ) -> Result<Arc<Prepared>, KrishivError> {
        if let Some(e) = &self.refused {
            return Err(e.clone());
        }
        let mut prepared = self.prepared.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(done) = prepared.as_ref() {
            return done.clone();
        }
        let made = self.make(root, limit, unpack).map(Arc::new);
        *prepared = Some(made.clone());
        made
    }

    fn make(
        &self,
        root: &Path,
        limit: u64,
        unpack: &dyn Fn(&Path, &Path, u64) -> Result<u64, KrishivError>,
    ) -> Result<Prepared, KrishivError> {
        let dir = tempfile::Builder::new()
            .prefix("artifacts-")
            .tempdir_in(root)
            .map_err(|e| internal(&format!("cannot make the UDF artifact directory: {e}")))?;
        let py = dir.path().join("py");
        let unpacked = dir.path().join("archives");
        for d in [&py, &unpacked] {
            fs::create_dir(d)
                .map_err(|e| internal(&format!("cannot make the UDF artifact directory: {e}")))?;
        }
        let mut used = 0u64;
        let mut sys_path = vec![py.clone()];
        for file in &self.pyfiles {
            let name = file_name(file)?;
            let to = py.join(name);
            used = used.saturating_add(copy_capped(file, &to, limit.saturating_sub(used))?);
            if !name.ends_with(".py") {
                sys_path.push(to);
            }
        }
        for archive in &self.archives {
            let to = unpacked.join(file_name(archive)?);
            fs::create_dir(&to)
                .map_err(|e| internal(&format!("cannot make the UDF archive directory: {e}")))?;
            used = used.saturating_add(unpack(archive, &to, limit.saturating_sub(used))?);
            sys_path.push(to);
        }
        Ok(Prepared {
            read_paths: vec![dir.path().to_path_buf()],
            _dir: dir,
            sys_path,
        })
    }
}

/// Refuses a path that is not an absolute, canonical path of a regular file (no symbolic link at it or above
/// it), and two files of one name.
fn check(paths: &[PathBuf]) -> Result<(), KrishivError> {
    let mut names = HashSet::new();
    for path in paths {
        let invalid = || {
            KrishivError::from(InvalidArtifactPath {
                name: path.display().to_string(),
            })
        };
        if !path.is_absolute() {
            return Err(invalid());
        }
        let meta = fs::symlink_metadata(path).map_err(|_| invalid())?;
        if !meta.file_type().is_file() {
            return Err(invalid());
        }
        match fs::canonicalize(path) {
            Ok(canonical) if canonical == *path => {}
            _ => return Err(invalid()),
        }
        let name = file_name(path)?;
        if name.starts_with('.') {
            return Err(invalid());
        }
        if !names.insert(name.to_owned()) {
            return Err(ArtifactAlreadyExists {
                normalized_remote_relative_path: name.to_owned(),
            }
            .into());
        }
    }
    Ok(())
}

/// The file name of a checked path, as UTF-8.
fn file_name(path: &Path) -> Result<&str, KrishivError> {
    path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        InvalidArtifactPath {
            name: path.display().to_string(),
        }
        .into()
    })
}

/// Copies `from` to `to` (a new file), refusing more than `budget` bytes; the bytes copied.
fn copy_capped(from: &Path, to: &Path, budget: u64) -> Result<u64, KrishivError> {
    let source = fs::File::open(from).map_err(|e| io_refused(from, &e))?;
    // A file that changed into a link (or anything but a file) since the check is refused.
    let meta = source.metadata().map_err(|e| io_refused(from, &e))?;
    if !meta.file_type().is_file() {
        return Err(InvalidArtifactPath {
            name: from.display().to_string(),
        }
        .into());
    }
    if meta.len() > budget {
        return Err(too_large(from, budget));
    }
    let target = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(to)
        .map_err(|e| internal(&format!("cannot copy a UDF artifact: {e}")))?;
    // At most one byte past the budget is read, so a file that grew since is refused, not copied whole.
    let mut limited = io::Read::take(source, budget.saturating_add(1));
    let mut writer = io::BufWriter::new(target);
    let copied = io::copy(&mut limited, &mut writer).map_err(|e| io_refused(from, &e))?;
    if copied > budget {
        return Err(too_large(from, budget));
    }
    io::Write::flush(&mut writer)
        .map_err(|e| internal(&format!("cannot copy a UDF artifact: {e}")))?;
    Ok(copied)
}

/// `KRISHIV_PYTHON_ARTIFACT_REFUSED` for `path`.
pub(crate) fn refused(path: &Path, reason: &str) -> KrishivError {
    krishiv_common::error::KrishivPythonArtifactRefused {
        name: path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        ),
        reason: reason.to_owned(),
    }
    .into()
}

fn too_large(path: &Path, budget: u64) -> KrishivError {
    refused(
        path,
        &format!(
            "the session's artifacts would take more than the {budget} bytes left of the pool's artifact limit"
        ),
    )
}

fn io_refused(path: &Path, e: &io::Error) -> KrishivError {
    refused(path, &format!("it cannot be read: {e}"))
}

fn internal(message: &str) -> KrishivError {
    krishiv_common::error::InternalError {
        message: message.to_owned(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::ErrorClass;

    fn no_unpack(_: &Path, _: &Path, _: u64) -> Result<u64, KrishivError> {
        Ok(0)
    }

    #[test]
    fn a_link_a_relative_path_or_a_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let real = root.join("m.py");
        fs::write(&real, "X = 1\n").unwrap();
        let secret = root.join("secret.txt");
        fs::write(&secret, "no").unwrap();
        let link = root.join("l.py");
        std::os::unix::fs::symlink(&secret, &link).unwrap();
        let linked_dir = root.join("ld");
        std::os::unix::fs::symlink(&root, &linked_dir).unwrap();
        for bad in [
            link,
            PathBuf::from("m.py"),
            root.clone(),
            linked_dir.join("m.py"),
            root.join("sub/../m.py"),
            root.join("missing.py"),
        ] {
            let set = ArtifactSet::new(vec![bad.clone()], Vec::new());
            let e = set
                .refused()
                .unwrap_or_else(|| panic!("{} was accepted", bad.display()));
            assert_eq!(e.error_class(), ErrorClass::InvalidArtifactPath, "{e}");
        }
        assert!(ArtifactSet::new(vec![real], Vec::new()).refused().is_none());
    }

    #[test]
    fn two_files_of_one_name_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        for d in ["a", "b"] {
            fs::create_dir(root.join(d)).unwrap();
            fs::write(root.join(d).join("m.py"), "").unwrap();
        }
        let set = ArtifactSet::new(vec![root.join("a/m.py"), root.join("b/m.py")], Vec::new());
        assert_eq!(
            set.refused().map(KrishivError::error_class),
            Some(ErrorClass::ArtifactAlreadyExists)
        );
    }

    #[test]
    fn pyfiles_are_copied_within_the_limit_and_zips_go_on_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("m.py"), "X = 1\n").unwrap();
        fs::write(root.join("dep.zip"), vec![0u8; 100]).unwrap();
        let pool_root = tempfile::tempdir().unwrap();
        let set = ArtifactSet::new(vec![root.join("m.py"), root.join("dep.zip")], Vec::new());
        assert_ne!(set.id, 0);
        let prepared = set.prepare(pool_root.path(), 1 << 20, &no_unpack).unwrap();
        let py = &prepared.sys_path[0];
        assert_eq!(fs::read_to_string(py.join("m.py")).unwrap(), "X = 1\n");
        assert_eq!(prepared.sys_path[1], py.join("dep.zip"));
        // Prepared once: the same directory for every later caller.
        let again = set.prepare(pool_root.path(), 1 << 20, &no_unpack).unwrap();
        assert!(Arc::ptr_eq(&prepared, &again));
        // Over the limit.
        let small = ArtifactSet::new(vec![root.join("m.py"), root.join("dep.zip")], Vec::new());
        let e = small
            .prepare(pool_root.path(), 50, &no_unpack)
            .err()
            .unwrap();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivPythonArtifactRefused,
            "{e}"
        );
    }

    #[test]
    fn the_unpacked_bytes_count_against_the_limit_left() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("m.py"), vec![b'#'; 30]).unwrap();
        fs::write(root.join("env.tar.gz"), "").unwrap();
        let pool_root = tempfile::tempdir().unwrap();
        let set = ArtifactSet::new(vec![root.join("m.py")], vec![root.join("env.tar.gz")]);
        let seen = Mutex::new(None);
        let unpack = |archive: &Path, to: &Path, budget: u64| {
            *seen.lock().unwrap() = Some((archive.to_path_buf(), to.to_path_buf(), budget));
            Ok(budget)
        };
        let prepared = set.prepare(pool_root.path(), 100, &unpack).unwrap();
        let (archive, to, budget) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(archive, root.join("env.tar.gz"));
        assert_eq!(budget, 70);
        assert!(to.ends_with("archives/env.tar.gz"));
        assert_eq!(prepared.sys_path.last(), Some(&to));
    }
}
