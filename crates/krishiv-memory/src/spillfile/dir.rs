// SPDX-License-Identifier: Apache-2.0
//! The spill directory manager: one session directory per process, one
//! sub-directory per query, a byte cap over all of it, and cleanup of what
//! crashed processes left behind.
//!
//! # Layout
//! ```text
//! <krishiv.spill.dir>/
//!   krishiv-spill-<id>.lock     held (advisory exclusive lock) while the process lives
//!   krishiv-spill-<id>/         the session directory (mode 0700)
//!     q<N>/                     one query's scope
//!       <M>.spill               a finished spill file
//!       <M>.spill.tmp           a file being written
//! ```
//! Every name below the configured directory is generated here from counters
//! and a process-unique id; nothing a user or a query supplies ever reaches a
//! path, so a path cannot leave the directory it is built in.
//!
//! # Crash cleanup
//! Liveness is the lock, not a process id: the kernel drops a lock when its
//! holder dies, however it died, and a recycled pid cannot fake a live
//! owner. [`SpillManager::open`] first reclaims every `krishiv-spill-*`
//! session whose lock it can take, then creates its own. The lock file is
//! created before the directory and removed after it, so a directory without
//! a lock file is debris. Locks do not work across hosts: keep
//! `krishiv.spill.dir` on a local file system.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow_ipc::writer::IpcWriteOptions;
use arrow_schema::{ArrowError, SchemaRef};
use krishiv_common::config::{
    Config, ConfigValue, SPILL_COMPRESSION, SPILL_DIR, SPILL_MAX_BYTES, SPILL_ZSTD_LEVEL,
};

use super::writer::SpillWriter;
use super::{SpillCodec, SpillError};
use crate::Reservation;

const PREFIX: &str = "krishiv-spill-";
const LOCK_SUFFIX: &str = ".lock";
/// Prefix of a lock file not yet published under its final name; startup
/// scans ignore it (it does not start with [`PREFIX`]).
const PENDING_PREFIX: &str = ".krishiv-spill-new-";
/// How long an unpublished lock file may exist before its start is taken for dead.
const PENDING_STALE_AFTER: Duration = Duration::from_secs(60);
const ID_ATTEMPTS: u32 = 16;

/// Where and how to spill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillConfig {
    /// `krishiv.spill.dir`: the directory the session directories live in.
    pub dir: PathBuf,
    /// `krishiv.spill.maxBytes`: the cap over all spill files of the process.
    pub max_bytes: u64,
    /// `krishiv.spill.compression`: the codec for files written.
    pub codec: SpillCodec,
    /// `krishiv.spill.zstd.level`: used when `codec` is [`SpillCodec::Zstd`].
    pub zstd_level: i64,
}

impl Default for SpillConfig {
    fn default() -> Self {
        Self::from_config(&Config::new())
    }
}

impl SpillConfig {
    /// Reads the `krishiv.spill.*` keys of `config`, with their defaults.
    pub fn from_config(config: &Config) -> Self {
        let dir = config
            .get(&SPILL_DIR)
            .and_then(|v| v.as_str())
            .map_or_else(default_dir, PathBuf::from);
        let max_bytes = config
            .get(&SPILL_MAX_BYTES)
            .and_then(ConfigValue::as_size)
            .unwrap_or(u64::MAX);
        let codec = config
            .get(&SPILL_COMPRESSION)
            .and_then(|v| v.as_str())
            .and_then(SpillCodec::from_name)
            .unwrap_or_default();
        let zstd_level = config
            .get(&SPILL_ZSTD_LEVEL)
            .and_then(ConfigValue::as_int)
            .unwrap_or(1);
        Self {
            dir,
            max_bytes,
            codec,
            zstd_level,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ManagerInner {
    session_dir: PathBuf,
    lock_path: PathBuf,
    cap: u64,
    used: AtomicU64,
    codec: SpillCodec,
    zstd_level: i64,
    next_scope: AtomicU64,
    stale_removed: usize,
    /// Holds the advisory lock; released by the kernel when dropped or when
    /// the process dies.
    _lock: File,
}

impl ManagerInner {
    /// Charges `bytes` against the cap, or refuses without charging.
    pub(crate) fn try_charge(&self, bytes: u64) -> Result<(), SpillError> {
        let mut used = self.used.load(Ordering::Acquire);
        loop {
            let next = used.checked_add(bytes).filter(|n| *n <= self.cap).ok_or(
                SpillError::CapExceeded {
                    requested: bytes,
                    used,
                    cap: self.cap,
                },
            )?;
            match self
                .used
                .compare_exchange_weak(used, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Ok(()),
                Err(actual) => used = actual,
            }
        }
    }

    /// Returns `bytes` to the cap.
    pub(crate) fn uncharge(&self, bytes: u64) {
        // The closure always returns `Some`, so the update cannot fail.
        let _ = self
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |u| {
                Some(u.saturating_sub(bytes))
            });
    }

    pub(crate) fn write_options(&self) -> Result<IpcWriteOptions, ArrowError> {
        self.codec.write_options(self.zstd_level)
    }
}

impl Drop for ManagerInner {
    fn drop(&mut self) {
        // Best effort: a failure leaves a stale session that the next start
        // reclaims once this process is gone and the lock is released.
        let _ = fs::remove_dir_all(&self.session_dir);
        let _ = fs::remove_file(&self.lock_path);
    }
}

/// The process-wide spill directory: hands out per-query [`SpillScope`]s,
/// enforces the byte cap over all of them and removes everything it created
/// when the last handle is dropped.
#[derive(Debug, Clone)]
pub struct SpillManager {
    inner: Arc<ManagerInner>,
}

impl SpillManager {
    /// Opens the spill directory: creates it if needed (mode 0700), removes
    /// the sessions of dead processes, and creates and locks this process's
    /// session directory.
    ///
    /// # Errors
    /// [`SpillError::Io`] if the directory cannot be created, is writable by
    /// other users, or the session cannot be set up.
    pub fn open(config: &SpillConfig) -> Result<Self, SpillError> {
        // Refuse a bad level now, not at the first spill under memory pressure.
        config
            .codec
            .write_options(config.zstd_level)
            .map_err(|e| SpillError::InvalidConfig {
                key: SPILL_ZSTD_LEVEL.key,
                value: config.zstd_level.to_string(),
                reason: e.to_string(),
            })?;
        let root = config.dir.as_path();
        create_private_dir(root, true).map_err(|e| SpillError::io("create", root, e))?;
        check_root_is_private(root).map_err(|e| SpillError::io("use", root, e))?;
        if let Some(fs_type) = network_fs(root) {
            let reason = format!("locks do not work reliably on a {fs_type} file system");
            return Err(SpillError::io("use", root, io::Error::other(reason)));
        }
        let stale_removed = remove_stale_sessions(root);
        for attempt in 0..ID_ATTEMPTS {
            let id = new_id(attempt);
            let lock_path = root.join(format!("{PREFIX}{id}{LOCK_SUFFIX}"));
            // The lock file is created under a name the stale-session scan
            // ignores, locked, and only then published by a hard link (which
            // fails if the name exists). Created in place it was reclaimable
            // by a concurrent start between its creation and our lock, and
            // under load a start could lose that race on every attempt.
            let pending = root.join(format!("{PENDING_PREFIX}{id}"));
            let lock = match create_new_file(&pending) {
                Ok(f) => f,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(SpillError::io("create", &pending, e)),
            };
            let locked = lock.try_lock().map_err(|e| match e {
                std::fs::TryLockError::Error(e) => e,
                std::fs::TryLockError::WouldBlock => {
                    io::Error::new(io::ErrorKind::WouldBlock, "a fresh lock file is held")
                }
            });
            let published = locked.and_then(|()| fs::hard_link(&pending, &lock_path));
            let _ = fs::remove_file(&pending);
            match published {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(SpillError::io("lock", &lock_path, e)),
            }
            let session_dir = root.join(format!("{PREFIX}{id}"));
            if let Err(e) = create_private_dir(&session_dir, false) {
                let _ = fs::remove_file(&lock_path);
                return Err(SpillError::io("create", &session_dir, e));
            }
            return Ok(Self {
                inner: Arc::new(ManagerInner {
                    session_dir,
                    lock_path,
                    cap: config.max_bytes,
                    used: AtomicU64::new(0),
                    codec: config.codec,
                    zstd_level: config.zstd_level,
                    next_scope: AtomicU64::new(0),
                    stale_removed,
                    _lock: lock,
                }),
            });
        }
        Err(SpillError::io(
            "create a unique session in",
            root,
            io::Error::new(io::ErrorKind::AlreadyExists, "no free session id"),
        ))
    }

    /// A new scope for one query. Its directory is removed when the scope
    /// and every writer and file made from it are dropped, or at once by
    /// [`SpillScope::cancel`].
    ///
    /// # Errors
    /// [`SpillError::Io`] if the directory cannot be created.
    pub fn scope(&self) -> Result<SpillScope, SpillError> {
        let n = self.inner.next_scope.fetch_add(1, Ordering::Relaxed);
        let dir = self.inner.session_dir.join(format!("q{n}"));
        create_private_dir(&dir, false).map_err(|e| SpillError::io("create", &dir, e))?;
        Ok(SpillScope {
            inner: Arc::new(ScopeInner {
                dir,
                manager: Arc::clone(&self.inner),
                cancelled: AtomicBool::new(false),
                next_file: AtomicU64::new(0),
            }),
        })
    }

    /// Bytes spill files occupy now (written, not yet dropped).
    pub fn used_bytes(&self) -> u64 {
        self.inner.used.load(Ordering::Acquire)
    }

    /// The codec new files are written with.
    pub fn codec(&self) -> SpillCodec {
        self.inner.codec
    }

    /// This process's session directory.
    pub fn session_dir(&self) -> &Path {
        &self.inner.session_dir
    }

    /// How many sessions of dead processes [`open`](Self::open) removed.
    pub fn stale_sessions_removed(&self) -> usize {
        self.inner.stale_removed
    }
}

#[derive(Debug)]
pub(crate) struct ScopeInner {
    dir: PathBuf,
    manager: Arc<ManagerInner>,
    cancelled: AtomicBool,
    next_file: AtomicU64,
}

impl ScopeInner {
    pub(crate) fn manager(&self) -> &ManagerInner {
        &self.manager
    }

    pub(crate) fn check_cancelled(&self) -> Result<(), SpillError> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(SpillError::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl Drop for ScopeInner {
    fn drop(&mut self) {
        // Best effort; the session directory removal and the next start's
        // stale cleanup catch a leftover.
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// One query's spill directory. Cloning shares it (hand a clone to the
/// cancellation path). Files and writers keep it alive, so the directory
/// outlives the handle that made it for as long as they do.
#[derive(Debug, Clone)]
pub struct SpillScope {
    inner: Arc<ScopeInner>,
}

impl SpillScope {
    /// Starts a spill file for batches of `schema`. `reservation` accounts
    /// the writer's scratch memory (see [`SpillWriter`]); when it comes from
    /// a [`ConsumerHandle`](crate::ConsumerHandle), the finished file is
    /// counted in the handle's [`SpillStats`](crate::SpillStats).
    ///
    /// # Errors
    /// [`SpillError::Cancelled`] after [`cancel`](Self::cancel);
    /// [`SpillError::CapExceeded`] or [`SpillError::Io`] if the file cannot
    /// be started.
    pub fn writer(
        &self,
        schema: SchemaRef,
        reservation: Reservation,
    ) -> Result<SpillWriter, SpillError> {
        self.inner.check_cancelled()?;
        let n = self.inner.next_file.fetch_add(1, Ordering::Relaxed);
        SpillWriter::create(
            Arc::clone(&self.inner),
            self.inner.dir.join(format!("{n}.spill.tmp")),
            self.inner.dir.join(format!("{n}.spill")),
            schema,
            reservation,
        )
        // A cancel that removed the directory under us is a cancel.
        .map_err(|e| match e {
            SpillError::Io { .. } if self.is_cancelled() => SpillError::Cancelled,
            other => other,
        })
    }

    /// The query was cancelled: stop all spill I/O and delete the files now.
    /// Writers and readers of this scope fail with [`SpillError::Cancelled`]
    /// at their next call and clean up as they drop. Safe to call from any
    /// thread, any number of times, while a write is in progress. On a
    /// failure to delete (the flag is set regardless), dropping the last
    /// handle retries. Disk blocks of files that are still open are freed when
    /// their writers and readers drop.
    ///
    /// # Errors
    /// [`SpillError::Io`] if the directory could not be removed.
    pub fn cancel(&self) -> Result<(), SpillError> {
        self.inner.cancelled.store(true, Ordering::Release);
        // A writer that raced the flag may create a file while we delete.
        let mut result = Ok(());
        for _ in 0..3 {
            result = fs::remove_dir_all(&self.inner.dir);
            if !matches!(&result, Err(e) if e.kind() == io::ErrorKind::DirectoryNotEmpty) {
                break;
            }
        }
        match result {
            Err(e) if e.kind() != io::ErrorKind::NotFound => {
                Err(SpillError::io("remove", &self.inner.dir, e))
            }
            _ => Ok(()),
        }
    }

    /// Whether [`cancel`](Self::cancel) was called.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// This scope's directory.
    pub fn dir(&self) -> &Path {
        &self.inner.dir
    }
}

/// A process-unique id: pid, wall-clock nanoseconds, attempt, and a
/// process-wide counter (threads of one process starting at the same
/// nanosecond with the same attempt would otherwise get the same id, and
/// collide on the root's probe file). Uniqueness across processes is
/// enforced by `create_new` on the lock file, not assumed from this.
pub(super) fn new_id(attempt: u32) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}-{nanos:x}-{attempt}-{n:x}", std::process::id())
}

/// Ids are `[0-9a-f-]`; anything else under our prefix is not ours.
pub(super) fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b) || b == b'-')
}

/// Removes the sessions of dead processes under `root`; returns how many.
/// Errors on single entries are skipped: they are retried at the next start
/// and must not stop this process from spilling.
fn remove_stale_sessions(root: &Path) -> usize {
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if let Some(pending) = name.to_str().filter(|n| n.starts_with(PENDING_PREFIX)) {
            // A start that died between creating and publishing its lock.
            // A live start holds the file for microseconds but is not locked
            // until after it is created, so only a file left for a minute is
            // taken for dead (locking an old file cannot race its creator).
            let path = entry.path();
            let old = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age >= PENDING_STALE_AFTER);
            if old
                && entry.file_type().is_ok_and(|t| t.is_file())
                && valid_id(&pending[PENDING_PREFIX.len()..])
                && OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .is_ok_and(|f| f.try_lock().is_ok())
            {
                let _ = fs::remove_file(&path);
            }
            continue;
        }
        let Some(rest) = name.to_str().and_then(|n| n.strip_prefix(PREFIX)) else {
            continue;
        };
        // `file_type` does not follow symlinks, so a planted link is skipped.
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if let Some(id) = rest.strip_suffix(LOCK_SUFFIX) {
            if valid_id(id) && file_type.is_file() && reclaim(root, id, &entry.path()) {
                removed += 1;
            }
        } else if valid_id(rest)
            && file_type.is_dir()
            && !root.join(format!("{PREFIX}{rest}{LOCK_SUFFIX}")).exists()
            && fs::remove_dir_all(entry.path()).is_ok()
        {
            removed += 1;
        }
    }
    removed
}

/// Removes the session `id` if nobody holds its lock.
fn reclaim(root: &Path, id: &str, lock_path: &Path) -> bool {
    let Ok(lock) = OpenOptions::new().write(true).open(lock_path) else {
        return false;
    };
    if lock.try_lock().is_err() {
        return false; // a live process owns it
    }
    let dir = root.join(format!("{PREFIX}{id}"));
    let gone = match fs::symlink_metadata(&dir) {
        Ok(m) if m.is_dir() => fs::remove_dir_all(&dir).is_ok(),
        Ok(_) => fs::remove_file(&dir).is_ok(),
        Err(e) => e.kind() == io::ErrorKind::NotFound,
    };
    // Remove the lock file while still holding the lock, so a concurrent
    // start that raced us sees it vanish (see `SpillManager::open`).
    gone && fs::remove_file(lock_path).is_ok()
}

/// Creates a directory with mode 0700 (where modes exist).
fn create_private_dir(path: &Path, recursive: bool) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(recursive);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(path)
}

/// The default directory: per user (one user cannot occupy or read
/// another's) and disk-backed, because spilling to a RAM file system
/// (`tmpfs`, as `/run/user/<uid>` and often `/tmp` are) uses the very memory
/// spilling must free, out of the pool's sight; network and FUSE file
/// systems are refused by `open`. The first candidate on neither wins, in
/// this order: the user's cache directory
/// (`$XDG_CACHE_HOME`, else `~/.cache`) `/krishiv/spill`, then
/// `/var/tmp/krishiv-spill-<uid>` (disk-backed by convention), then
/// `krishiv-spill-<uid>` under the temporary directory. An explicitly
/// configured directory is used as given.
fn default_dir() -> PathBuf {
    choose_default(&default_candidates(), |p| unsuitable_fs(p).is_some())
}

/// The default-directory candidates, in order of preference.
fn default_candidates() -> Vec<PathBuf> {
    #[cfg(unix)]
    let user = fs::metadata("/proc/self")
        .ok()
        .map(|m| std::os::unix::fs::MetadataExt::uid(&m).to_string());
    #[cfg(not(unix))]
    let user: Option<String> = None;
    let user = user
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "user".to_owned());
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|p| !p.is_empty())
                .map(|h| PathBuf::from(h).join(".cache"))
        });
    let mut out = Vec::new();
    if let Some(cache) = cache {
        out.push(cache.join("krishiv").join("spill"));
    }
    #[cfg(unix)]
    out.push(PathBuf::from("/var/tmp").join(format!("krishiv-spill-{user}")));
    out.push(std::env::temp_dir().join(format!("krishiv-spill-{user}")));
    out
}

/// The first candidate that is not on a RAM file system, else the first.
pub(crate) fn choose_default(candidates: &[PathBuf], on_ram: impl Fn(&Path) -> bool) -> PathBuf {
    candidates
        .iter()
        .find(|c| !on_ram(c))
        .or_else(|| candidates.first())
        .cloned()
        .unwrap_or_else(std::env::temp_dir)
}

/// The file system type holding `path` (or its nearest existing ancestor)
/// when it is unsuitable for spilling by default: a RAM file system (`tmpfs`,
/// `ramfs`), or a network or FUSE file system, which `SpillManager::open`
/// refuses. Linux only; elsewhere `None`.
fn unsuitable_fs(path: &Path) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let existing = path.ancestors().find(|p| p.exists())?;
        let mounts = fs::read_to_string("/proc/self/mounts").ok()?;
        let at = fs::canonicalize(existing).ok()?;
        if let Some(network) = network_fs_in(&mounts, &at) {
            return Some(network);
        }
        let ty = mount_type_in(&mounts, &at)?;
        matches!(ty.as_str(), "tmpfs" | "ramfs").then_some(ty)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        None
    }
}

/// Refuses a spill root that is not a real directory owned by this user and
/// closed to group and others: they could plant look-alike entries, deny
/// service or read spilled query data. Symlinks are not followed.
fn check_root_is_private(root: &Path) -> io::Result<()> {
    let meta = fs::symlink_metadata(root)?;
    if meta.file_type().is_symlink() {
        return Err(io::Error::other("is a symbolic link"));
    }
    if !meta.is_dir() {
        return Err(io::ErrorKind::NotADirectory.into());
    }
    #[cfg(unix)]
    {
        // Our uid is the owner of a file we create.
        let probe = root.join(format!(".probe-{}", new_id(0)));
        let owner = create_new_file(&probe).and_then(|f| f.metadata());
        let _ = fs::remove_file(&probe);
        validate_owner_and_mode(&meta, std::os::unix::fs::MetadataExt::uid(&owner?))?;
    }
    Ok(())
}

#[cfg(unix)]
pub(crate) fn validate_owner_and_mode(meta: &fs::Metadata, uid: u32) -> io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    if meta.uid() != uid {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "directory is owned by another user",
        ));
    }
    if meta.permissions().mode() & 0o022 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "directory is writable by group or others",
        ));
    }
    Ok(())
}

/// The file system type of `root` if it is one whose advisory locks cannot
/// be trusted (network and FUSE file systems). Linux only; elsewhere `None`.
fn network_fs(root: &Path) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let mounts = fs::read_to_string("/proc/self/mounts").ok()?;
        network_fs_in(&mounts, &fs::canonicalize(root).ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = root;
        None
    }
}

/// The type of the mount containing `path` in `/proc/self/mounts` text, if
/// it is a network or FUSE file system.
pub(crate) fn network_fs_in(mounts: &str, path: &Path) -> Option<String> {
    let fs_type = mount_type_in(mounts, path)?;
    let network = [
        "nfs", "cifs", "smb", "ceph", "gluster", "lustre", "9p", "afs", "fuse",
    ];
    network
        .iter()
        .any(|p| fs_type.starts_with(p))
        .then_some(fs_type)
}

/// The type of the mount containing `path` in `/proc/self/mounts` text (the
/// longest matching mount point).
pub(crate) fn mount_type_in(mounts: &str, path: &Path) -> Option<String> {
    mounts
        .lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let (_, point, ty) = (f.next()?, f.next()?, f.next()?);
            path.starts_with(point.replace("\\040", " "))
                .then_some((point.len(), ty))
        })
        .max_by_key(|(len, _)| *len)
        .map(|(_, ty)| ty.to_owned())
}

/// Creates a new file (mode 0600 where modes exist), failing if it exists.
pub(crate) fn create_new_file(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)
}

#[cfg(test)]
#[cfg(not(loom))]
impl SpillScope {
    /// The shared state, for tests that build a [`super::SpillFile`] by hand.
    pub(crate) fn inner_for_tests(&self) -> Arc<ScopeInner> {
        Arc::clone(&self.inner)
    }
}
