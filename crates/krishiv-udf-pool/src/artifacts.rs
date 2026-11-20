// SPDX-License-Identifier: Apache-2.0
//! A session's Python artifacts (P3-T014 PR-26c-3): the pyfiles and archives a Connect client added
//! (`addArtifacts(pyfile=True)`, `addArtifacts(archive=True)`), as its workers see them once the server sets
//! them (`SessionUdfs::set_artifacts`, which the Connect server calls when the session's artifacts change).
//!
//! Spark's worker puts its files directory and each `.zip`/`.egg` include on `sys.path` (`add_path`, at index
//! 1, `worker_util.py` L59-65 and L133-154 at v4.2.0). Its executor unpacks each archive into that same
//! directory (`Executor.updateDependencies`, `Utils.unpack`: `.zip`, `.jar`, `.tar`, `.tar.gz`, `.tgz`), into a
//! directory named by the archive's URI fragment, or else by its file name, and starts the Python worker in it
//! (`PythonWorkerFactory`, the session's directory under `SparkFiles.getRootDirectory()`). Here a set of
//! artifacts is prepared once, by the first worker start that needs it, into a directory of the pool's own:
//! - `py/` is the files directory: a copy of every pyfile, and each archive unpacked into `py/<alias>/` (the
//!   [`PythonArchive::alias`], or the file name). It, then each pyfile that is not a `.py` (a zip, an egg), go
//!   on the worker's `sys.path` in Spark's order; an unpacked archive is not on it (Spark's
//!   `setup_spark_files` does not add one), but is importable as a package of the files directory, as in Spark.
//! - Each entry of `py/` is reachable from the worker's working directory (a link in the worker's own
//!   directory, which stays its writable working directory), as Spark's worker reaches them from its own.
//!
//! The directory is one of the worker's read-only paths in the sandbox (Landlock), and nothing else of the
//! server's artifact directory is. A worker serves one set only (the pool key carries the set's id), since its
//! read paths are fixed when it starts.
//!
//! **Trust.** The server names the paths; the client chose their contents (and maybe their names). So a path
//! must be absolute, a regular file, not a symbolic link and with no symbolic link above it (its canonical
//! form is itself): Landlock grants a path by the file it opens, so a link would hand the worker its target.
//! The check and the copy each resolve the path once, following no link anywhere in it (`openat2` with
//! `RESOLVE_NO_SYMLINKS`; the path must also be lexically canonical), and the check records the file it opened
//! (its device and inode); the copy refuses any other file, so a path swapped after the check (for a link, or a
//! parent for a link, or for another file) is refused (`INVALID_ARTIFACT_PATH`) and nothing of it is copied. Each archive is copied the same way into the pool's
//! directory and only that copy is unpacked (the unpacker's read rule names no client path). Two pyfiles or two
//! archives of one name are refused (their copies would collide). The copies and the unpacked bytes together
//! are capped ([`crate::PoolConfig::artifact_limit`]; an archive's copy, removed once it is unpacked, is not
//! counted beside its unpacked bytes, so a set's directory peaks at twice the limit while its last archive is
//! unpacked: the copy and the unpacked bytes, each within what is left of the limit). An archive is unpacked by the worker interpreter's own
//! `zipfile`/`tarfile`, in the sandbox when it is on (it reads the archive and writes only its own directory),
//! with `tarfile`'s `data` filter: no absolute paths, no `..`, no link leaving the directory, no device files; a
//! zip's entries cannot be links.
//!
//! **Failures.** A failure about the artifacts themselves (a path, a format, a size) is kept for the set's
//! lifetime: every later call fails the same way. A failure of one attempt (a cancel, a timeout, the server
//! unable to start the unpacker or to write its directory) is not: the next call tries again, except that the
//! [`MAX_TIMEOUTS`]th timeout of a set is kept (an archive that always takes too long). When the unpacking
//! failed, a failure to remove its copy does not replace that failure. The set's lock is
//! not held while its directory is made; a later caller waits for the attempt and may cancel meanwhile.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Duration;

use krishiv_common::error::{
    ArtifactAlreadyExists, InvalidArtifactPath, KrishivError, OperationCanceled,
};
use rustix::fs::{Mode, OFlags, ResolveFlags};
use rustix::io::Errno;

/// Ids of artifact sets, unique in the process; 0 is the empty set.
static NEXT_SET: AtomicU64 = AtomicU64::new(1);

/// How often a caller waiting for another caller's preparation checks its own cancel.
const WAIT_TICK: Duration = Duration::from_millis(50);

/// What unpacks one archive: `(archive, dest, budget)`, the bytes unpacked.
pub(crate) type Unpack<'a> = dyn Fn(&Path, &Path, u64) -> Result<u64, Failure> + 'a;

/// How many attempts of one set may time out before the timeout is kept like a lasting failure (an archive
/// that always takes too long would otherwise cost a call thread up to the unpacking timeout on every call).
pub(crate) const MAX_TIMEOUTS: u32 = 3;

/// Why a set's directory could not be made, and whether that lasts.
#[derive(Debug)]
pub(crate) struct Failure {
    /// The error the calls get.
    pub(crate) error: KrishivError,
    /// About the artifacts themselves: every later call gets it too. Else the next call tries again.
    pub(crate) lasting: bool,
    /// The attempt ran out of time: retried, until [`MAX_TIMEOUTS`] attempts of the set have.
    pub(crate) timed_out: bool,
}

impl Failure {
    /// A failure about the artifacts (a path, a format, a size): kept for the set's lifetime.
    pub(crate) fn lasting(error: KrishivError) -> Self {
        Failure {
            error,
            lasting: true,
            timed_out: false,
        }
    }

    /// A failure of this attempt only (a cancel, a timeout, the server's own I/O): the next call retries.
    pub(crate) fn transient(error: KrishivError) -> Self {
        Failure {
            error,
            lasting: false,
            timed_out: false,
        }
    }

    /// The attempt ran out of time: retried by the next call until [`MAX_TIMEOUTS`] attempts have, then kept.
    pub(crate) fn timed_out(error: KrishivError) -> Self {
        Failure {
            error,
            lasting: false,
            timed_out: true,
        }
    }
}

/// An archive artifact (`addArtifacts(archive=True)`): the file, and the `#fragment` the client gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonArchive {
    /// The archive file: absolute, with no symbolic link at or above it.
    pub path: PathBuf,
    /// The name of the directory it is unpacked into (Spark's URI fragment, `env.tar.gz#env`); the file's
    /// name without one.
    pub alias: Option<String>,
}

/// A checked artifact: its path, the file it named when checked, and its name in the files directory (a
/// pyfile's file name; an archive's alias, or its file name).
struct Artifact {
    path: PathBuf,
    dev: u64,
    ino: u64,
    name: String,
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

/// A set's preparation and how many of its attempts timed out.
struct State {
    slot: Slot,
    timeouts: u32,
}

/// One session's artifacts, as set by [`crate::SessionUdfs::set_artifacts`].
pub(crate) struct ArtifactSet {
    /// The pool key's part: a worker started for one set never serves another.
    pub(crate) id: u64,
    /// The paths as given, until the first caller that needs them checks them (file I/O, so not when they
    /// are set: that is on an async worker under a session lock).
    raw: Mutex<Option<(Vec<PathBuf>, Vec<PythonArchive>)>>,
    /// The paths checked, or why they are refused (raised by the calls that need them).
    checked: OnceLock<Checked>,
    /// The prepared directory, made by the first worker start that needs it (or why it could not be made).
    prepared: Mutex<State>,
    /// Told when an attempt ends.
    made: Condvar,
}

/// A set's paths after the check.
struct Checked {
    pyfiles: Vec<Artifact>,
    archives: Vec<Artifact>,
    refused: Option<KrishivError>,
}

/// A set's prepared directory: what a worker of the set reads.
pub(crate) struct Prepared {
    /// Removed when the set and every worker started for it are gone.
    _dir: tempfile::TempDir,
    /// The worker's `sys.path` entries, in the order they are added (each at index 1, as Spark's `add_path`).
    pub(crate) sys_path: Vec<PathBuf>,
    /// The worker's extra read-only paths in the sandbox.
    pub(crate) read_paths: Vec<PathBuf>,
    /// The files directory (Spark's `SparkFiles` root of the session): the pyfiles and the unpacked archives.
    pub(crate) files_dir: PathBuf,
    /// The names in [`Self::files_dir`], each reachable from the worker's working directory (Spark's worker
    /// runs in that directory).
    pub(crate) entries: Vec<String>,
}

/// The attempt in progress: dropped, it ends the attempt with `end` (an empty slot unless set; a timeout
/// counted when `timed_out`) and wakes the callers waiting for it, so an attempt that never finishes (a panic)
/// leaves the set to the next caller.
struct Attempt<'a> {
    set: &'a ArtifactSet,
    end: Slot,
    timed_out: bool,
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        let mut state = self.set.lock();
        state.slot = std::mem::replace(&mut self.end, Slot::Empty);
        if self.timed_out {
            state.timeouts = state.timeouts.saturating_add(1);
        }
        drop(state);
        self.set.made.notify_all();
    }
}

fn new_state() -> Mutex<State> {
    Mutex::new(State {
        slot: Slot::Empty,
        timeouts: 0,
    })
}

impl ArtifactSet {
    /// No artifacts.
    pub(crate) fn empty() -> Self {
        ArtifactSet {
            id: 0,
            raw: Mutex::new(None),
            checked: OnceLock::from(Checked {
                pyfiles: Vec::new(),
                archives: Vec::new(),
                refused: None,
            }),
            prepared: new_state(),
            made: Condvar::new(),
        }
    }

    /// The set of `pyfiles` and `archives`. Nothing is read here: the paths and names are checked by the first
    /// call that needs them ([`Self::refused`], [`Self::prepare`]), before any copy, which refuses any other
    /// file than the one the check opened.
    pub(crate) fn new(pyfiles: Vec<PathBuf>, archives: Vec<PythonArchive>) -> Self {
        if pyfiles.is_empty() && archives.is_empty() {
            return Self::empty();
        }
        ArtifactSet {
            id: NEXT_SET.fetch_add(1, Ordering::Relaxed),
            raw: Mutex::new(Some((pyfiles, archives))),
            checked: OnceLock::new(),
            prepared: new_state(),
            made: Condvar::new(),
        }
    }

    /// The paths checked (once; a refusal is kept).
    fn checked(&self) -> &Checked {
        self.checked.get_or_init(|| {
            let (pyfiles, archives) = self
                .raw
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .unwrap_or_default();
            let mut names = HashSet::new();
            let result = check(pyfiles.into_iter().map(|p| (p, None)), &mut names).and_then(|p| {
                check(archives.into_iter().map(|a| (a.path, a.alias)), &mut names).map(|a| (p, a))
            });
            match result {
                Ok((pyfiles, archives)) => Checked {
                    pyfiles,
                    archives,
                    refused: None,
                },
                Err(e) => Checked {
                    pyfiles: Vec::new(),
                    archives: Vec::new(),
                    refused: Some(e),
                },
            }
        })
    }

    /// Whether the set has no artifacts.
    pub(crate) fn is_empty(&self) -> bool {
        self.id == 0
    }

    /// Why the set's paths are refused, if they are (checks them now if no call has).
    pub(crate) fn refused(&self) -> Option<&KrishivError> {
        self.checked().refused.as_ref()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.prepared.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The prepared directory: made by the first caller (under `root`, the archives unpacked by `unpack`, all
    /// within `limit` bytes), the same one (or the same lasting failure) for every later caller. A caller that
    /// finds another one making it waits, and fails with `OPERATION_CANCELED` once `cancelled` says so. The
    /// [`MAX_TIMEOUTS`]th attempt that times out is kept as the set's failure.
    pub(crate) fn prepare(
        &self,
        root: &Path,
        limit: u64,
        unpack: &Unpack<'_>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Arc<Prepared>, KrishivError> {
        let checked = self.checked();
        if let Some(e) = &checked.refused {
            return Err(e.clone());
        }
        let (mut attempt, timeouts) = {
            let mut state = self.lock();
            loop {
                match &state.slot {
                    Slot::Done(done) => return done.clone(),
                    Slot::Empty => break,
                    Slot::Making => {
                        if cancelled() {
                            return Err(OperationCanceled {}.into());
                        }
                        state = self
                            .made
                            .wait_timeout(state, WAIT_TICK)
                            .unwrap_or_else(PoisonError::into_inner)
                            .0;
                    }
                }
            }
            state.slot = Slot::Making;
            let attempt = Attempt {
                set: self,
                end: Slot::Empty,
                timed_out: false,
            };
            (attempt, state.timeouts)
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
                lasting,
                timed_out,
            }) => {
                if lasting || (timed_out && timeouts.saturating_add(1) >= MAX_TIMEOUTS) {
                    attempt.end = Slot::Done(Err(error.clone()));
                }
                attempt.timed_out = timed_out;
                Err(error)
            }
        }
    }

    fn make(&self, root: &Path, limit: u64, unpack: &Unpack<'_>) -> Result<Prepared, Failure> {
        let dir = tempfile::Builder::new()
            .prefix("artifacts-")
            .tempdir_in(root)
            .map_err(|e| server_io("cannot make the UDF artifact directory", &e))?;
        let py = dir.path().join("py");
        let incoming = dir.path().join("in");
        for d in [&py, &incoming] {
            fs::create_dir(d)
                .map_err(|e| server_io("cannot make the UDF artifact directory", &e))?;
        }
        let mut used = 0u64;
        let mut sys_path = vec![py.clone()];
        for file in &self.checked().pyfiles {
            let to = py.join(&file.name);
            used = used.saturating_add(copy_capped(file, &to, limit.saturating_sub(used))?);
            if !file.name.ends_with(".py") {
                sys_path.push(to);
            }
        }
        for (i, archive) in self.checked().archives.iter().enumerate() {
            let file = file_name(&archive.path).map_err(Failure::lasting)?;
            // The unpacker reads this copy only, never a path the client named; its name keeps the file's
            // (the format is read from it), made unique since two aliases may name one file.
            let copy = incoming.join(format!("{i}-{file}"));
            copy_capped(archive, &copy, limit.saturating_sub(used))?;
            // Into the files directory, by its alias or its file name (Spark's `addArchive`); not on
            // `sys.path` itself (Spark's `setup_spark_files` adds only the files directory and the includes).
            let to = py.join(&archive.name);
            fs::create_dir(&to)
                .map_err(|e| server_io("cannot make the UDF archive directory", &e))?;
            let bytes = unpack(&copy, &to, limit.saturating_sub(used));
            let removed = fs::remove_file(&copy);
            // The unpacking's own failure first: a failed removal must not hide a lasting refusal.
            let bytes = bytes?;
            removed.map_err(|e| server_io("cannot remove a UDF archive's copy", &e))?;
            used = used.saturating_add(bytes);
        }
        fs::remove_dir(&incoming)
            .map_err(|e| server_io("cannot remove the UDF archive copies' directory", &e))?;
        let entries = self
            .checked()
            .pyfiles
            .iter()
            .chain(&self.checked().archives)
            .map(|a| a.name.clone())
            .collect();
        Ok(Prepared {
            read_paths: vec![dir.path().to_path_buf()],
            _dir: dir,
            sys_path,
            files_dir: py,
            entries,
        })
    }
}

/// Refuses a path that is not an absolute, canonical path of a regular file (no symbolic link at it or above
/// it), a bad alias, and two artifacts of one name in the files directory (`names`, shared by the pyfiles and
/// the archives: a pyfile's file name, an archive's alias or else its file name); else the paths with the
/// files they name.
fn check(
    paths: impl Iterator<Item = (PathBuf, Option<String>)>,
    names: &mut HashSet<String>,
) -> Result<Vec<Artifact>, KrishivError> {
    let mut checked = Vec::new();
    for (path, alias) in paths {
        let file = open_no_links(&path).map_err(|e| open_refusal(&path, e))?;
        let meta = file.metadata().map_err(|_| invalid(&path))?;
        if !meta.file_type().is_file() {
            return Err(invalid(&path));
        }
        let file = file_name(&path)?;
        if file.starts_with('.') {
            return Err(invalid(&path));
        }
        let name = match alias {
            None => file.to_owned(),
            Some(alias) if plain_alias(&alias) => alias,
            Some(alias) => {
                return Err(InvalidArtifactPath {
                    name: format!("{}#{alias}", path.display()),
                }
                .into());
            }
        };
        if !names.insert(name.clone()) {
            return Err(ArtifactAlreadyExists {
                normalized_remote_relative_path: name,
            }
            .into());
        }
        checked.push(Artifact {
            path,
            dev: meta.dev(),
            ino: meta.ino(),
            name,
        });
    }
    Ok(checked)
}

/// Whether `alias` names one plain entry of the files directory: not empty, at most `NAME_MAX` (255) bytes, no
/// `/` or NUL, and not hidden (which also refuses `.` and `..`). Spark 4.2.0 takes any fragment (`new File(root, fragment)`), so a
/// fragment like `../x` would leave its directory there; here it is refused (`INVALID_ARTIFACT_PATH`).
fn plain_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias.len() <= 255
        && !alias.starts_with('.')
        && !alias.contains(['/', '\0'])
}

/// The error of a failed `open_no_links` of `path`: `INVALID_ARTIFACT_PATH` when the path is bad (missing,
/// through a link, not canonical, not readable), `INTERNAL_ERROR` for the server's own problem (a seccomp
/// profile refusing `openat2`, no descriptors or memory left, an I/O error).
fn open_refusal(path: &Path, errno: Errno) -> KrishivError {
    match errno {
        Errno::NOSYS => no_openat2(path),
        Errno::NOENT
        | Errno::LOOP
        | Errno::NOTDIR
        | Errno::XDEV
        | Errno::ACCESS
        | Errno::INVAL
        | Errno::NAMETOOLONG => invalid(path),
        other => krishiv_common::error::InternalError {
            message: format!(
                "cannot open the artifact {}: {}",
                path.display(),
                io::Error::from(other)
            ),
        }
        .into(),
    }
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

/// Opens `path` for reading in one resolution that follows no symbolic link at all (`openat2` with
/// `RESOLVE_NO_SYMLINKS`, Linux 5.6): the file opened is the one the path names with no link at it or above
/// it, so checking it and reading it are the same walk (no second resolution an attacker could race). The path
/// must be absolute and lexically canonical (no `.` or `..` component, no repeated or trailing `/`), which is
/// checked on the text alone; with no link in it, that makes it its own canonical form. A FIFO does not block
/// the open (`O_NONBLOCK`; the caller refuses it as not a regular file).
fn open_no_links(path: &Path) -> Result<fs::File, Errno> {
    if !lexically_canonical(path) {
        return Err(Errno::INVAL);
    }
    let flags =
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC | OFlags::NOCTTY;
    let resolve = ResolveFlags::NO_SYMLINKS | ResolveFlags::NO_MAGICLINKS;
    let fd = rustix::fs::openat2(rustix::fs::CWD, path, flags, Mode::empty(), resolve)?;
    Ok(fs::File::from(fd))
}

/// Whether `path` is absolute with only plain components, written once each with one `/` between them.
fn lexically_canonical(path: &Path) -> bool {
    use std::path::Component;
    let mut plain = PathBuf::new();
    for c in path.components() {
        match c {
            Component::RootDir | Component::Normal(_) => plain.push(c),
            Component::Prefix(_) | Component::CurDir | Component::ParentDir => return false,
        }
    }
    path.is_absolute() && plain.as_os_str() == path.as_os_str()
}

/// The kernel has no `openat2` (before Linux 5.6): artifacts are refused rather than opened by a weaker walk.
fn no_openat2(path: &Path) -> KrishivError {
    refused(
        path,
        "this kernel has no openat2 (Linux 5.6 or later is needed to open artifacts safely)",
    )
}

/// Opens the checked `artifact` for reading: the file it named at the check, by a walk that follows no link
/// ([`open_no_links`]), refused unless it is that same file (its device and inode), so a path swapped for a
/// link or for another file since the check is refused and nothing of it is read.
fn open_checked(artifact: &Artifact) -> Result<fs::File, Failure> {
    let path = &artifact.path;
    let file = open_no_links(path).map_err(|e| match e {
        Errno::LOOP | Errno::NOENT | Errno::NOTDIR | Errno::INVAL | Errno::XDEV => {
            Failure::lasting(invalid(path))
        }
        Errno::NOSYS => Failure::lasting(no_openat2(path)),
        Errno::MFILE | Errno::NFILE | Errno::NOMEM | Errno::INTR | Errno::AGAIN => {
            Failure::transient(io_refused(path, &io::Error::from(e)))
        }
        _ => Failure::lasting(io_refused(path, &io::Error::from(e))),
    })?;
    let meta = file
        .metadata()
        .map_err(|e| Failure::lasting(io_refused(path, &e)))?;
    if !meta.file_type().is_file() || meta.dev() != artifact.dev || meta.ino() != artifact.ino {
        return Err(Failure::lasting(invalid(path)));
    }
    Ok(file)
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

    /// An archive without an alias.
    fn file(path: PathBuf) -> PythonArchive {
        PythonArchive { path, alias: None }
    }

    fn aliased(path: PathBuf, alias: &str) -> PythonArchive {
        PythonArchive {
            path,
            alias: Some(alias.to_owned()),
        }
    }

    #[test]
    fn an_archive_is_unpacked_by_its_alias_into_the_files_directory() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("env.tar.gz"), "abc").unwrap();
        fs::write(root.join("m.py"), "").unwrap();
        let pool_root = tempfile::tempdir().unwrap();
        // One file under two aliases (Spark keys an archive by its URI with the fragment), and once by name.
        let set = ArtifactSet::new(
            vec![root.join("m.py")],
            vec![
                aliased(root.join("env.tar.gz"), "env"),
                aliased(root.join("env.tar.gz"), "env2"),
                file(root.join("env.tar.gz")),
            ],
        );
        assert!(set.refused().is_none(), "{:?}", set.refused());
        let seen = Mutex::new(Vec::new());
        let unpack = |archive: &Path, to: &Path, _: u64| {
            assert!(archive.starts_with(pool_root.path()));
            seen.lock().unwrap().push(to.to_path_buf());
            Ok(3)
        };
        let prepared = set
            .prepare(pool_root.path(), 1 << 20, &unpack, &never)
            .unwrap();
        let files = &prepared.files_dir;
        assert_eq!(
            *seen.lock().unwrap(),
            [
                files.join("env"),
                files.join("env2"),
                files.join("env.tar.gz")
            ]
        );
        assert_eq!(prepared.sys_path, std::slice::from_ref(files));
        assert_eq!(prepared.entries, ["m.py", "env", "env2", "env.tar.gz"]);
    }

    #[test]
    fn a_bad_alias_or_a_name_taken_twice_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::write(root.join("env.zip"), "z").unwrap();
        fs::write(root.join("m.py"), "").unwrap();
        // An alias longer than a file name may be (`NAME_MAX`, 255 bytes) is refused here, not left to fail
        // as a retried server error when its directory is made (review R1-8).
        let long = "a".repeat(256);
        for bad in [
            "", ".", "..", "../x", "a/b", "/abs", ".hidden", "a\0b", &long,
        ] {
            let set = ArtifactSet::new(Vec::new(), vec![aliased(root.join("env.zip"), bad)]);
            let e = set
                .refused()
                .unwrap_or_else(|| panic!("{bad:?} was accepted"));
            assert_eq!(
                e.error_class(),
                ErrorClass::InvalidArtifactPath,
                "{bad:?}: {e}"
            );
        }
        for (pyfiles, archives) in [
            (
                Vec::new(),
                vec![
                    aliased(root.join("env.zip"), "x"),
                    aliased(root.join("env.zip"), "x"),
                ],
            ),
            (
                vec![root.join("m.py")],
                vec![aliased(root.join("env.zip"), "m.py")],
            ),
            (
                Vec::new(),
                vec![
                    file(root.join("env.zip")),
                    aliased(root.join("env.zip"), "env.zip"),
                ],
            ),
        ] {
            let set = ArtifactSet::new(pyfiles, archives);
            assert_eq!(
                set.refused().map(KrishivError::error_class),
                Some(ErrorClass::ArtifactAlreadyExists)
            );
        }
    }

    /// Only an error that says the path is bad is `INVALID_ARTIFACT_PATH`; a server problem opening it (a
    /// seccomp profile refusing `openat2`, no descriptors left) is `INTERNAL_ERROR` (review R1-8).
    #[test]
    fn an_open_failure_is_classed_by_its_cause() {
        let path = Path::new("/x/m.py");
        for errno in [
            Errno::NOENT,
            Errno::LOOP,
            Errno::NOTDIR,
            Errno::XDEV,
            Errno::ACCESS,
            Errno::INVAL,
            Errno::NAMETOOLONG,
        ] {
            assert_eq!(
                open_refusal(path, errno).error_class(),
                ErrorClass::InvalidArtifactPath,
                "{errno:?}"
            );
        }
        for errno in [
            Errno::PERM,
            Errno::MFILE,
            Errno::NFILE,
            Errno::NOMEM,
            Errno::IO,
        ] {
            assert_eq!(
                open_refusal(path, errno).error_class(),
                ErrorClass::InternalError,
                "{errno:?}"
            );
        }
        assert_eq!(
            open_refusal(path, Errno::NOSYS).error_class(),
            no_openat2(path).error_class()
        );
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
        let set = ArtifactSet::new(vec![root.join("m.py")], vec![file(root.join("env.tar.gz"))]);
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
        assert!(
            archive.ends_with("in/0-env.tar.gz"),
            "{}",
            archive.display()
        );
        assert_eq!(bytes, b"abc");
        // The copy is gone once unpacked, and not counted beside the unpacked bytes.
        assert!(!archive.exists());
        assert_eq!(budget, 70);
        // Into the files directory, by its file name; that directory alone is on `sys.path`.
        assert_eq!(to, prepared.files_dir.join("env.tar.gz"));
        assert_eq!(prepared.sys_path, std::slice::from_ref(&prepared.files_dir));
        assert_eq!(prepared.entries, ["m.py", "env.tar.gz"]);
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
        let archives = ArtifactSet::new(Vec::new(), vec![file(root.join("m.py"))]);
        assert!(pyfiles.refused().is_none() && archives.refused().is_none());
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
        let set = ArtifactSet::new(Vec::new(), vec![file(root.join("env.zip"))]);
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
    fn a_failed_removal_of_the_copy_does_not_hide_a_lasting_refusal() {
        let (_keep, set) = archive_set();
        let pool_root = tempfile::tempdir().unwrap();
        // The unpacker refuses the archive and its copy is gone by the time the set removes it.
        let refuses = |archive: &Path, _: &Path, _: u64| {
            fs::remove_file(archive).unwrap();
            Err(Failure::lasting(refused(archive, "it is not a zip")))
        };
        let e = set
            .prepare(pool_root.path(), 1 << 20, &refuses, &never)
            .err()
            .unwrap();
        assert!(e.to_string().contains("it is not a zip"), "{e}");
        // Kept, as any lasting refusal.
        let e = set
            .prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
            .err()
            .unwrap();
        assert!(e.to_string().contains("it is not a zip"), "{e}");
    }

    #[test]
    fn an_archive_that_keeps_timing_out_is_kept_after_max_timeouts() {
        let (_keep, set) = archive_set();
        let pool_root = tempfile::tempdir().unwrap();
        let calls = Mutex::new(0u32);
        let slow = |archive: &Path, _: &Path, _: u64| {
            *calls.lock().unwrap() += 1;
            Err(Failure::timed_out(refused(
                archive,
                "it could not be unpacked in 300 s",
            )))
        };
        for _ in 0..MAX_TIMEOUTS + 2 {
            let e = set
                .prepare(pool_root.path(), 1 << 20, &slow, &never)
                .err()
                .unwrap();
            assert!(e.to_string().contains("300 s"), "{e}");
        }
        assert_eq!(*calls.lock().unwrap(), MAX_TIMEOUTS);
        // Kept: a later caller whose unpacker would work gets the timeout too.
        assert!(
            set.prepare(pool_root.path(), 1 << 20, &no_unpack, &never)
                .is_err()
        );
        // A timeout below the cap is still retried, and a success after it is the set's.
        let (_keep, set) = archive_set();
        let once = Mutex::new(false);
        let slow_once = |archive: &Path, _: &Path, _: u64| {
            let mut seen = once.lock().unwrap();
            if *seen {
                return Ok(1);
            }
            *seen = true;
            Err(Failure::timed_out(refused(archive, "slow")))
        };
        assert!(
            set.prepare(pool_root.path(), 1 << 20, &slow_once, &never)
                .is_err()
        );
        assert!(
            set.prepare(pool_root.path(), 1 << 20, &slow_once, &never)
                .is_ok()
        );
    }

    #[test]
    fn a_path_must_be_lexically_canonical_and_free_of_links_in_one_walk() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::create_dir(root.join("d")).unwrap();
        fs::write(root.join("d/m.py"), "X = 1\n").unwrap();
        std::os::unix::fs::symlink(root.join("d"), root.join("l")).unwrap();
        let text = root.display().to_string();
        for bad in [
            format!("{text}/./d/m.py"),
            format!("{text}//d/m.py"),
            format!("{text}/d/m.py/"),
            format!("{text}/d/../d/m.py"),
            format!("{text}/l/m.py"),
            format!("/proc/self/root{text}/d/m.py"),
        ] {
            assert!(open_no_links(Path::new(&bad)).is_err(), "{bad} was opened");
            let set = ArtifactSet::new(vec![PathBuf::from(&bad)], Vec::new());
            assert_eq!(
                set.refused().map(KrishivError::error_class),
                Some(ErrorClass::InvalidArtifactPath),
                "{bad}"
            );
        }
        assert!(open_no_links(&root.join("d/m.py")).is_ok());
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
