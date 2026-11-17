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
//! - `archives/<name>/` holds each archive unpacked (by the archive's file name), also on `sys.path`: a
//!   recorded difference (`docs/concepts/known-differences.md`), since Spark's `setup_spark_files` does not
//!   put an unpacked archive there.
//!
//! The directory is one of the worker's read-only paths in the sandbox (Landlock), and nothing else of the
//! server's artifact directory is. A worker serves one set only (the pool key carries the set's id), since its
//! read paths are fixed when it starts.
//!
//! **Trust.** The server names the paths; the client chose their contents (and maybe their names). So a path
//! must be absolute, a regular file, not a symbolic link and with no symbolic link above it (its canonical
//! form is itself): Landlock grants a path by the file it opens, so a link would hand the worker its target.
//! The check records the file each path names (its device and inode); the copy opens the path without
//! following a link at its last component and refuses a file other than the one checked, or a path whose
//! canonical form changed since (a parent swapped for a link), so a path swapped after the check is refused
//! (`INVALID_ARTIFACT_PATH`) and nothing of it is copied. Each archive is copied the same way into the pool's
//! directory and only that copy is unpacked (the unpacker's read rule names no client path). Two pyfiles or two
//! archives of one name are refused (their copies would collide). The copies and the unpacked bytes together
//! are capped ([`crate::PoolConfig::artifact_limit`]; an archive's copy, removed once it is unpacked, is not
//! counted beside its unpacked bytes). An archive is unpacked by the worker interpreter's own
//! `zipfile`/`tarfile`, in the sandbox when it is on (it reads the archive and writes only its own directory),
//! with `tarfile`'s `data` filter: no absolute paths, no `..`, no link leaving the directory, no device files; a
//! zip's entries cannot be links.
//!
//! **Failures.** A failure about the artifacts themselves (a path, a format, a size) is kept for the set's
//! lifetime: every later call fails the same way. A failure of one attempt (a cancel, a timeout, the server
//! unable to start the unpacker or to write its directory) is not: the next call tries again. The set's lock is
//! not held while its directory is made; a later caller waits for the attempt and may cancel meanwhile.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use krishiv_common::error::{
    ArtifactAlreadyExists, InvalidArtifactPath, KrishivError, OperationCanceled,
};
use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;

/// Ids of artifact sets, unique in the process; 0 is the empty set.
static NEXT_SET: AtomicU64 = AtomicU64::new(1);

/// How often a caller waiting for another caller's preparation checks its own cancel.
const WAIT_TICK: Duration = Duration::from_millis(50);

/// What unpacks one archive: `(archive, dest, budget)`, the bytes unpacked.
pub(crate) type Unpack<'a> = dyn Fn(&Path, &Path, u64) -> Result<u64, Failure> + 'a;

/// Why a set's directory could not be made, and whether that lasts.
#[derive(Debug)]
pub(crate) struct Failure {
    /// The error the calls get.
    pub(crate) error: KrishivError,
    /// About the artifacts themselves: every later call gets it too. Else the next call tries again.
    pub(crate) lasting: bool,
}

impl Failure {
    /// A failure about the artifacts (a path, a format, a size): kept for the set's lifetime.
    pub(crate) fn lasting(error: KrishivError) -> Self {
        Failure {
            error,
            lasting: true,
        }
    }

    /// A failure of this attempt only (a cancel, a timeout, the server's own I/O): the next call retries.
    pub(crate) fn transient(error: KrishivError) -> Self {
        Failure {
            error,
            lasting: false,
        }
    }
}

/// A checked artifact: its path and the file it named when checked.
struct Artifact {
    path: PathBuf,
    dev: u64,
    ino: u64,
}

/// A set's preparation.
enum Slot {
    /// Not made (or the last attempt failed for a reason that does not last).
    Empty,
    /// A caller is making it, without the lock.
    Making,
    /// Made, or refused for good.
    Done(Result<Arc<Prepared>, KrishivError>),
}

/// One session's artifacts, as set by [`crate::SessionUdfs::set_artifacts`].
pub(crate) struct ArtifactSet {
    /// The pool key's part: a worker started for one set never serves another.
    pub(crate) id: u64,
    pyfiles: Vec<Artifact>,
    archives: Vec<Artifact>,
    /// Why the paths given are refused (checked when they are set; raised by the calls that need them).
    refused: Option<KrishivError>,
    /// The prepared directory, made by the first worker start that needs it (or why it could not be made).
    prepared: Mutex<Slot>,
    /// Told when an attempt ends.
    made: Condvar,
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

/// The attempt in progress: dropped, it ends the attempt with `end` (an empty slot unless set) and wakes the
/// callers waiting for it, so an attempt that never finishes (a panic) leaves the set to the next caller.
struct Attempt<'a> {
    set: &'a ArtifactSet,
    end: Slot,
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        *self.set.lock() = std::mem::replace(&mut self.end, Slot::Empty);
        self.set.made.notify_all();
    }
}

impl ArtifactSet {
    /// No artifacts.
    pub(crate) fn empty() -> Self {
        ArtifactSet {
            id: 0,
            pyfiles: Vec::new(),
            archives: Vec::new(),
            refused: None,
            prepared: Mutex::new(Slot::Empty),
            made: Condvar::new(),
        }
    }

    /// The set of `pyfiles` and `archives`, their paths checked now (a refusal is raised by the calls).
    pub(crate) fn new(pyfiles: Vec<PathBuf>, archives: Vec<PathBuf>) -> Self {
        if pyfiles.is_empty() && archives.is_empty() {
            return Self::empty();
        }
        let (pyfiles, archives, refused) =
            match check(pyfiles).and_then(|p| check(archives).map(|a| (p, a))) {
                Ok((p, a)) => (p, a, None),
                Err(e) => (Vec::new(), Vec::new(), Some(e)),
            };
        ArtifactSet {
            id: NEXT_SET.fetch_add(1, Ordering::Relaxed),
            pyfiles,
            archives,
            refused,
            prepared: Mutex::new(Slot::Empty),
            made: Condvar::new(),
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

    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.prepared.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The prepared directory: made by the first caller (under `root`, the archives unpacked by `unpack`, all
    /// within `limit` bytes), the same one (or the same lasting failure) for every later caller. A caller that
    /// finds another one making it waits, and fails with `OPERATION_CANCELED` once `cancelled` says so.
    pub(crate) fn prepare(
        &self,
        root: &Path,
        limit: u64,
        unpack: &Unpack<'_>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Arc<Prepared>, KrishivError> {
        if let Some(e) = &self.refused {
            return Err(e.clone());
        }
        let mut attempt = {
            let mut slot = self.lock();
            loop {
                match &*slot {
                    Slot::Done(done) => return done.clone(),
                    Slot::Empty => break,
                    Slot::Making => {
                        if cancelled() {
                            return Err(OperationCanceled {}.into());
                        }
                        slot = self
                            .made
                            .wait_timeout(slot, WAIT_TICK)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0;
                    }
                }
            }
            *slot = Slot::Making;
            Attempt {
                set: self,
                end: Slot::Empty,
            }
        };
        // Made without the lock (an unpacking may take minutes): later callers wait on `made`, each able to
        // cancel, and the attempt's end is published when `attempt` drops.
        match self.make(root, limit, unpack) {
            Ok(prepared) => {
                let prepared = Arc::new(prepared);
                attempt.end = Slot::Done(Ok(Arc::clone(&prepared)));
                Ok(prepared)
            }
            Err(Failure {
                error,
                lasting: true,
            }) => {
                attempt.end = Slot::Done(Err(error.clone()));
                Err(error)
            }
            Err(Failure {
                error,
                lasting: false,
            }) => Err(error),
        }
    }

    fn make(&self, root: &Path, limit: u64, unpack: &Unpack<'_>) -> Result<Prepared, Failure> {
        let dir = tempfile::Builder::new()
            .prefix("artifacts-")
            .tempdir_in(root)
            .map_err(|e| server_io("cannot make the UDF artifact directory", &e))?;
        let py = dir.path().join("py");
        let unpacked = dir.path().join("archives");
        let incoming = dir.path().join("in");
        for d in [&py, &unpacked, &incoming] {
            fs::create_dir(d)
                .map_err(|e| server_io("cannot make the UDF artifact directory", &e))?;
        }
        let mut used = 0u64;
        let mut sys_path = vec![py.clone()];
        for file in &self.pyfiles {
            let name = file_name(&file.path).map_err(Failure::lasting)?;
            let to = py.join(name);
            used = used.saturating_add(copy_capped(file, &to, limit.saturating_sub(used))?);
            if !name.ends_with(".py") {
                sys_path.push(to);
            }
        }
        for archive in &self.archives {
            let name = file_name(&archive.path).map_err(Failure::lasting)?;
            // The unpacker reads this copy only, never a path the client named.
            let copy = incoming.join(name);
            copy_capped(archive, &copy, limit.saturating_sub(used))?;
            let to = unpacked.join(name);
            fs::create_dir(&to)
                .map_err(|e| server_io("cannot make the UDF archive directory", &e))?;
            let bytes = unpack(&copy, &to, limit.saturating_sub(used));
            fs::remove_file(&copy)
                .map_err(|e| server_io("cannot remove a UDF archive's copy", &e))?;
            used = used.saturating_add(bytes?);
            sys_path.push(to);
        }
        fs::remove_dir(&incoming)
            .map_err(|e| server_io("cannot remove the UDF archive copies' directory", &e))?;
        Ok(Prepared {
            read_paths: vec![dir.path().to_path_buf()],
            _dir: dir,
            sys_path,
        })
    }
}

/// Refuses a path that is not an absolute, canonical path of a regular file (no symbolic link at it or above
/// it), and two files of one name; else the paths with the files they name.
fn check(paths: Vec<PathBuf>) -> Result<Vec<Artifact>, KrishivError> {
    let mut names = HashSet::new();
    let mut checked = Vec::with_capacity(paths.len());
    for path in paths {
        if !path.is_absolute() {
            return Err(invalid(&path));
        }
        let meta = fs::symlink_metadata(&path).map_err(|_| invalid(&path))?;
        if !meta.file_type().is_file() {
            return Err(invalid(&path));
        }
        match fs::canonicalize(&path) {
            Ok(canonical) if canonical == path => {}
            _ => return Err(invalid(&path)),
        }
        let name = file_name(&path)?;
        if name.starts_with('.') {
            return Err(invalid(&path));
        }
        if !names.insert(name.to_owned()) {
            return Err(ArtifactAlreadyExists {
                normalized_remote_relative_path: name.to_owned(),
            }
            .into());
        }
        checked.push(Artifact {
            path,
            dev: meta.dev(),
            ino: meta.ino(),
        });
    }
    Ok(checked)
}

/// `INVALID_ARTIFACT_PATH` for `path`.
fn invalid(path: &Path) -> KrishivError {
    InvalidArtifactPath {
        name: path.display().to_string(),
    }
    .into()
}

/// The file name of a checked path, as UTF-8.
fn file_name(path: &Path) -> Result<&str, KrishivError> {
    path.file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid(path))
}

/// Opens the checked `artifact` for reading: the file it named at the check, reached by the same canonical path.
/// A link at the last component is not followed (`O_NOFOLLOW`); a link above it is seen as another file (its
/// device and inode) or, when it leads to the same file, as a canonical path that changed. A FIFO swapped in
/// does not block the open (`O_NONBLOCK`; it is then refused as not a regular file).
fn open_checked(artifact: &Artifact) -> Result<fs::File, Failure> {
    let path = &artifact.path;
    let flags =
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC | OFlags::NOCTTY;
    let fd = rustix::fs::open(path, flags, Mode::empty()).map_err(|e| match e {
        Errno::LOOP | Errno::NOENT | Errno::NOTDIR => Failure::lasting(invalid(path)),
        Errno::MFILE | Errno::NFILE | Errno::NOMEM | Errno::INTR | Errno::AGAIN => {
            Failure::transient(io_refused(path, &io::Error::from(e)))
        }
        _ => Failure::lasting(io_refused(path, &io::Error::from(e))),
    })?;
    let file = fs::File::from(fd);
    let meta = file
        .metadata()
        .map_err(|e| Failure::lasting(io_refused(path, &e)))?;
    if !meta.file_type().is_file() || meta.dev() != artifact.dev || meta.ino() != artifact.ino {
        return Err(Failure::lasting(invalid(path)));
    }
    match fs::canonicalize(path) {
        Ok(canonical) if canonical == *path => Ok(file),
        _ => Err(Failure::lasting(invalid(path))),
    }
}

/// Copies the checked `from` to `to` (a new file), refusing more than `budget` bytes; the bytes copied.
fn copy_capped(from: &Artifact, to: &Path, budget: u64) -> Result<u64, Failure> {
    let source = open_checked(from)?;
    let path = &from.path;
    let len = source
        .metadata()
        .map_err(|e| Failure::lasting(io_refused(path, &e)))?
        .len();
    if len > budget {
        return Err(Failure::lasting(too_large(path, budget)));
    }
    let target = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(to)
        .map_err(|e| server_io("cannot copy a UDF artifact", &e))?;
    // At most one byte past the budget is read, so a file that grew since is refused, not copied whole.
    let mut limited = io::Read::take(source, budget.saturating_add(1));
    let mut writer = io::BufWriter::new(target);
    // A read or a write may fail here; neither says the artifact is bad, so the next call tries again.
    let copied = io::copy(&mut limited, &mut writer)
        .map_err(|e| Failure::transient(refused(path, &format!("it cannot be copied: {e}"))))?;
    if copied > budget {
        return Err(Failure::lasting(too_large(path, budget)));
    }
    io::Write::flush(&mut writer).map_err(|e| server_io("cannot copy a UDF artifact", &e))?;
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

/// The server's own I/O failed (`INTERNAL_ERROR`): the next call tries again.
fn server_io(what: &str, e: &io::Error) -> Failure {
    Failure::transient(
        krishiv_common::error::InternalError {
            message: format!("{what}: {e}"),
        }
        .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_common::error::{ErrorClass, InternalError};
    use std::sync::mpsc;

    fn no_unpack(_: &Path, _: &Path, _: u64) -> Result<u64, Failure> {
        Ok(0)
    }

    fn never() -> bool {
        false
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
        let prepared = set
            .prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
            .unwrap();
        let py = &prepared.sys_path[0];
        assert_eq!(fs::read_to_string(py.join("m.py")).unwrap(), "X = 1\n");
        assert_eq!(prepared.sys_path[1], py.join("dep.zip"));
        // Prepared once: the same directory for every later caller.
        let again = set
            .prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
            .unwrap();
        assert!(Arc::ptr_eq(&prepared, &again));
        // Over the limit.
        let small = ArtifactSet::new(vec![root.join("m.py"), root.join("dep.zip")], Vec::new());
        let e = small
            .prepare(pool_root.path(), 50, &no_unpack, &never)
            .err()
            .unwrap();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivPythonArtifactRefused,
            "{e}"
        );
    }

    #[test]
    fn an_archive_s_copy_in_the_pool_s_directory_is_unpacked_within_the_limit_left() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("m.py"), vec![b'#'; 30]).unwrap();
        fs::write(root.join("env.tar.gz"), "abc").unwrap();
        let pool_root = tempfile::tempdir().unwrap();
        let set = ArtifactSet::new(vec![root.join("m.py")], vec![root.join("env.tar.gz")]);
        let seen = Mutex::new(None);
        let unpack = |archive: &Path, to: &Path, budget: u64| {
            let bytes = fs::read(archive).unwrap();
            *seen.lock().unwrap() = Some((archive.to_path_buf(), bytes, to.to_path_buf(), budget));
            Ok(budget)
        };
        let prepared = set.prepare(pool_root.path(), 100, &unpack, &never).unwrap();
        let (archive, bytes, to, budget) = seen.lock().unwrap().clone().unwrap();
        assert!(
            archive.starts_with(pool_root.path()),
            "{}",
            archive.display()
        );
        assert!(archive.ends_with("in/env.tar.gz"), "{}", archive.display());
        assert_eq!(bytes, b"abc");
        // The copy is gone once unpacked, and not counted beside the unpacked bytes.
        assert!(!archive.exists());
        assert_eq!(budget, 70);
        assert!(to.ends_with("archives/env.tar.gz"));
        assert_eq!(prepared.sys_path.last(), Some(&to));
    }

    /// What `prepare` left under the pool's root: nothing, after a refusal.
    fn left_in(root: &Path) -> usize {
        fs::read_dir(root).unwrap().count()
    }

    fn refused_at_copy(set: &ArtifactSet) {
        let pool_root = tempfile::tempdir().unwrap();
        let unpacked = Mutex::new(0);
        let unpack = |_: &Path, _: &Path, _: u64| {
            *unpacked.lock().unwrap() += 1;
            Ok(0)
        };
        let e = set
            .prepare(pool_root.path(), 1 << 20, &unpack, &never)
            .err()
            .expect("the swapped artifact was copied");
        assert_eq!(e.error_class(), ErrorClass::InvalidArtifactPath, "{e}");
        assert_eq!(left_in(pool_root.path()), 0);
        assert_eq!(*unpacked.lock().unwrap(), 0);
    }

    #[test]
    fn a_file_swapped_for_a_link_after_the_check_is_refused_at_copy() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("m.py"), "X = 1\n").unwrap();
        fs::write(root.join("secret.txt"), "no").unwrap();
        let set = ArtifactSet::new(vec![root.join("m.py")], Vec::new());
        assert!(set.refused().is_none());
        fs::remove_file(root.join("m.py")).unwrap();
        std::os::unix::fs::symlink(root.join("secret.txt"), root.join("m.py")).unwrap();
        refused_at_copy(&set);
    }

    #[test]
    fn a_parent_swapped_for_a_link_after_the_check_is_refused_at_copy() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        for d in ["a", "c"] {
            fs::create_dir(root.join(d)).unwrap();
            fs::write(root.join(d).join("m.py"), d).unwrap();
        }
        let set = ArtifactSet::new(vec![root.join("a/m.py")], Vec::new());
        assert!(set.refused().is_none());
        // To the same file (its canonical path is no longer the one checked), then to another one.
        fs::rename(root.join("a"), root.join("b")).unwrap();
        std::os::unix::fs::symlink(root.join("b"), root.join("a")).unwrap();
        refused_at_copy(&set);
        let set = ArtifactSet::new(vec![root.join("b/m.py")], Vec::new());
        assert!(set.refused().is_none());
        fs::rename(root.join("b"), root.join("d")).unwrap();
        std::os::unix::fs::symlink(root.join("c"), root.join("b")).unwrap();
        refused_at_copy(&set);
    }

    #[test]
    fn a_file_replaced_by_another_after_the_check_is_refused_at_copy() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("m.py"), "X = 1\n").unwrap();
        let pyfiles = ArtifactSet::new(vec![root.join("m.py")], Vec::new());
        let archives = ArtifactSet::new(Vec::new(), vec![root.join("m.py")]);
        // A new file (made while the old one still exists, so its inode differs) renamed over it.
        fs::write(root.join("other.py"), "X = 2\n").unwrap();
        fs::rename(root.join("other.py"), root.join("m.py")).unwrap();
        refused_at_copy(&pyfiles);
        refused_at_copy(&archives);
    }

    fn archive_set() -> (tempfile::TempDir, ArtifactSet) {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("env.zip"), "z").unwrap();
        let set = ArtifactSet::new(Vec::new(), vec![root.join("env.zip")]);
        (dir, set)
    }

    #[test]
    fn a_transient_failure_is_retried_and_a_lasting_one_is_kept() {
        let (_keep, set) = archive_set();
        let pool_root = tempfile::tempdir().unwrap();
        let calls = Mutex::new(0);
        let flaky = |archive: &Path, _: &Path, _: u64| {
            *calls.lock().unwrap() += 1;
            Err(Failure::transient(refused(archive, "it took too long")))
        };
        for _ in 0..2 {
            let e = set
                .prepare(pool_root.path(), 1 << 20, &flaky, &never)
                .err()
                .unwrap();
            assert_eq!(e.error_class(), ErrorClass::KrishivPythonArtifactRefused);
        }
        assert_eq!(*calls.lock().unwrap(), 2, "a transient failure is retried");
        assert_eq!(left_in(pool_root.path()), 0);
        let bad = |archive: &Path, _: &Path, _: u64| {
            *calls.lock().unwrap() += 1;
            Err(Failure::lasting(refused(archive, "it is not a zip")))
        };
        for _ in 0..2 {
            let e = set
                .prepare(pool_root.path(), 1 << 20, &bad, &never)
                .err()
                .unwrap();
            assert!(e.to_string().contains("it is not a zip"), "{e}");
        }
        assert_eq!(*calls.lock().unwrap(), 3, "a lasting failure is kept");
        // Kept: a later caller with a working unpacker gets the same failure.
        let e = set
            .prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
            .err()
            .unwrap();
        assert!(e.to_string().contains("it is not a zip"), "{e}");
    }

    #[test]
    fn a_server_failure_is_internal_and_retried() {
        let (_keep, set) = archive_set();
        let pool_root = tempfile::tempdir().unwrap();
        let spawn_failed = |_: &Path, _: &Path, _: u64| {
            Err(Failure::transient(
                InternalError {
                    message: "the UDF worker spawner has stopped".to_owned(),
                }
                .into(),
            ))
        };
        let e = set
            .prepare(pool_root.path(), 1 << 20, &spawn_failed, &never)
            .err()
            .unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(
            set.prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
                .is_ok()
        );
        // A root the pool cannot write: internal, and not kept either.
        let (_keep, set) = archive_set();
        let missing = pool_root.path().join("gone");
        let e = set
            .prepare(&missing, 1 << 20, &no_unpack, &never)
            .err()
            .unwrap();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(
            set.prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
                .is_ok()
        );
    }

    #[test]
    fn a_caller_waiting_for_another_s_unpacking_can_cancel_without_the_lock() {
        let (_keep, set) = archive_set();
        let pool_root = tempfile::tempdir().unwrap();
        let (started, on_start) = mpsc::channel();
        let (finish, on_finish) = mpsc::channel::<()>();
        let on_finish = Mutex::new(on_finish);
        let slow = |_: &Path, _: &Path, _: u64| {
            started.send(()).unwrap();
            on_finish.lock().unwrap().recv().unwrap();
            Ok(1)
        };
        std::thread::scope(|s| {
            let first = s.spawn(|| set.prepare(pool_root.path(), 1 << 20, &slow, &never));
            on_start.recv().unwrap();
            // The set's lock is free while the first caller unpacks: a second one is not stuck on it, and
            // its cancel ends its wait.
            let e = set
                .prepare(pool_root.path(), 1 << 20, &no_unpack, &|| true)
                .err()
                .unwrap();
            assert_eq!(e.error_class(), ErrorClass::OperationCanceled);
            let waiter = s.spawn(|| set.prepare(pool_root.path(), 1 << 20, &no_unpack, &never));
            finish.send(()).unwrap();
            let made = first.join().unwrap().unwrap();
            // The waiter gets the first caller's directory.
            assert!(Arc::ptr_eq(&made, &waiter.join().unwrap().unwrap()));
        });
    }
}
