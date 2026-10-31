// SPDX-License-Identifier: Apache-2.0
//! [`BeneathStore`]: a local object store whose every operation is relative
//! to one open directory (the *anchor*) and follows no symbolic link
//! (P2-T199).
//!
//! `LocalFileSystem` opens paths by name, so a link swapped into a directory
//! of a running write redirects the part files, and its listing follows links
//! into directories outside the write's root, which an overwrite then deletes
//! from. Here the anchor is opened once by walking the checked path one
//! component at a time with `O_NOFOLLOW`; afterwards a name is only ever
//! resolved relative to a descriptor (Linux: `openat2` with
//! `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`; elsewhere a component walk with
//! `O_NOFOLLOW`), so no link on the way can lead out of the anchor.
//!
//! Only whole-object writes, multipart writes, `head`, deletes and listings
//! exist (what a directory write uses); body reads and copies are
//! `NotImplemented`: reads keep using the plain local store.
//!
//! **Residual case (not closed):** a process that can write both inside the
//! root and in a directory `D` of the same file system can `rename(2)` a
//! directory of the path (or the anchor) into `D` while a write runs; the
//! anchor follows the inode, so later files land in `D`. That process could
//! already write to `D`; the session's re-checks before the commit report it.
//!
//! **Platforms:** on macOS (no `O_PATH`) every ancestor of the anchor needs
//! read permission; a search-only (mode 0711) ancestor fails with `EACCES`.

use std::fmt;
use std::fs::File;
use std::os::fd::OwnedFd;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::FutureExt;
use futures::stream::{BoxStream, StreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMode, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as StoreResult, UploadPart,
};
use rustix::fs::{
    AtFlags, CWD, Dir, FileType, Mode, OFlags, Stat, mkdirat, openat, renameat, statat, unlinkat,
};
use rustix::io::Errno;

/// What a descriptor-relative directory open asks for: `O_PATH` where it
/// exists (no read permission needed on the way), else a plain read open.
#[cfg(any(target_os = "linux", target_os = "android"))]
const DIR_FLAGS: OFlags = OFlags::PATH.union(OFlags::DIRECTORY).union(OFlags::CLOEXEC);
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const DIR_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::CLOEXEC);

/// A readable directory descriptor (`O_PATH` ones cannot be listed).
const LIST_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);

const STORE: &str = "BeneathStore";

/// The source of a `PermissionDenied`: a name tried to leave the anchor (a
/// symbolic link, a mount crossing, a file where a directory was expected,
/// or a path outside it).
#[derive(Debug)]
pub struct BeneathViolation(String);

impl fmt::Display for BeneathViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BeneathViolation {}

/// Why the anchor could not be opened.
#[derive(Debug)]
pub enum AnchorError {
    /// A component of the path is a link or not a directory: it changed
    /// since the path was checked.
    Swapped(String),
    /// A directory of the path has a name object paths cannot carry
    /// unchanged (non-ASCII text or reserved characters).
    UnsupportedName(String),
    /// Any other failure (permissions, I/O).
    Io(std::io::Error),
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Swapped(path) => write!(f, "a link or non-directory on the path {path}"),
            Self::UnsupportedName(name) => write!(f, "the directory name {name:?}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AnchorError {}

/// How a directory below the anchor is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Walk {
    /// One `openat2` with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS` (Linux).
    Openat2,
    /// One `openat(O_NOFOLLOW)` per component.
    Components,
}

/// Whether this process can use `openat2` (probed once: `ENOSYS` before
/// Linux 5.6, `EPERM` under some seccomp profiles).
fn openat2_available() -> bool {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        static PROBE: OnceLock<bool> = OnceLock::new();
        *PROBE.get_or_init(|| {
            rustix::fs::openat2(
                CWD,
                ".",
                DIR_FLAGS,
                Mode::empty(),
                rustix::fs::ResolveFlags::BENEATH,
            )
            .is_ok()
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        false
    }
}

#[derive(Debug)]
struct Inner {
    anchor: OwnedFd,
    /// The anchor's own path as object-path segments.
    base: Vec<String>,
    walk: Walk,
}

/// A local store anchored at one directory. See the module docs.
#[derive(Debug, Clone)]
pub struct BeneathStore {
    inner: Arc<Inner>,
}

impl fmt::Display for BeneathStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{STORE}({})", self.inner.base.join("/"))
    }
}

impl BeneathStore {
    /// Opens the anchor for a write to `target` (an absolute path whose
    /// links are already resolved): the deepest existing directory strictly
    /// above `target`'s last component, found by one walk from `/` that
    /// follows no link. The components after it (missing ones, then the
    /// target's name) are created, listed and deleted relative to it.
    ///
    /// # Errors
    /// [`AnchorError::Swapped`] if a component is a link or not a directory
    /// (it changed since the path was checked); [`AnchorError::Io`] otherwise.
    pub fn anchor(target: &std::path::Path) -> Result<Self, AnchorError> {
        let walk = if openat2_available() {
            Walk::Openat2
        } else {
            Walk::Components
        };
        Self::anchor_with(target, walk)
    }

    /// As [`Self::anchor`], with the component walk forced (tests).
    ///
    /// # Errors
    /// As [`Self::anchor`].
    pub fn anchor_components(target: &std::path::Path) -> Result<Self, AnchorError> {
        Self::anchor_with(target, Walk::Components)
    }

    fn anchor_with(target: &std::path::Path, walk: Walk) -> Result<Self, AnchorError> {
        if !target.is_absolute() {
            return Err(AnchorError::Io(std::io::Error::other(
                "the write path is not absolute and normalized",
            )));
        }
        let mut names = Vec::new();
        for c in target.components() {
            match c {
                std::path::Component::Normal(n) => names.push(
                    n.to_str()
                        .ok_or_else(|| {
                            AnchorError::Io(std::io::Error::other("a path that is not UTF-8"))
                        })?
                        .to_owned(),
                ),
                std::path::Component::RootDir => {}
                _ => {
                    return Err(AnchorError::Io(std::io::Error::other(
                        "the write path is not absolute and normalized",
                    )));
                }
            }
        }
        // The target's own name is never part of the anchor.
        names.pop();
        let mut fd =
            openat(CWD, "/", DIR_FLAGS, Mode::empty()).map_err(|e| AnchorError::Io(e.into()))?;
        let mut depth = 0;
        for name in &names {
            match openat(
                &fd,
                name.as_str(),
                DIR_FLAGS.union(OFlags::NOFOLLOW),
                Mode::empty(),
            ) {
                Ok(next) => {
                    fd = next;
                    depth += 1;
                }
                Err(Errno::NOENT) => break,
                Err(Errno::LOOP | Errno::NOTDIR | Errno::MLINK) => {
                    return Err(AnchorError::Swapped(format!("/{}", names.join("/"))));
                }
                Err(e) => return Err(AnchorError::Io(e.into())),
            }
        }
        names.truncate(depth);
        // Object paths percent-encode some names (`LocalFileSystem` names
        // files by the encoded text), so a directory above the target whose
        // name changes under encoding cannot be addressed by them.
        if let Some(name) = names
            .iter()
            .find(|n| Path::from_iter([n.as_str()]).as_ref() != n.as_str())
        {
            return Err(AnchorError::UnsupportedName(name.clone()));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                anchor: fd,
                base: names,
                walk,
            }),
        })
    }

    /// The object-path segments of `location` below the anchor.
    fn rel(&self, location: &Path) -> StoreResult<Vec<String>> {
        let parts: Vec<String> = location.parts().map(|p| p.as_ref().to_owned()).collect();
        let base = &self.inner.base;
        if parts.len() < base.len() || parts[..base.len()] != base[..] {
            return Err(violation(
                location,
                "the path is outside the write's directory",
            ));
        }
        let rel = parts[base.len()..].to_vec();
        if rel.iter().any(|s| s.is_empty() || s == "." || s == "..") {
            return Err(violation(
                location,
                "the path has an empty, `.` or `..` segment",
            ));
        }
        Ok(rel)
    }

    /// The directory `rel` below the anchor, created when `create`; never
    /// through a link.
    fn dir(&self, rel: &[String], create: bool) -> Result<OwnedFd, Errno> {
        let own = || self.inner.anchor.try_clone().map_err(|_| Errno::MFILE);
        if rel.is_empty() {
            return own();
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if !create && self.inner.walk == Walk::Openat2 {
            return rustix::fs::openat2(
                &self.inner.anchor,
                rel.join("/"),
                DIR_FLAGS,
                Mode::empty(),
                rustix::fs::ResolveFlags::BENEATH | rustix::fs::ResolveFlags::NO_SYMLINKS,
            );
        }
        let mut cur = own()?;
        for name in rel {
            if create {
                match mkdirat(&cur, name.as_str(), Mode::from_bits_truncate(0o777)) {
                    Ok(()) | Err(Errno::EXIST) => {}
                    Err(e) => return Err(e),
                }
            }
            cur = openat(
                &cur,
                name.as_str(),
                DIR_FLAGS.union(OFlags::NOFOLLOW),
                Mode::empty(),
            )?;
        }
        Ok(cur)
    }

    /// The parent directory and the final name of `location`.
    fn parent_and_name(&self, location: &Path, create: bool) -> StoreResult<(OwnedFd, String)> {
        let mut rel = self.rel(location)?;
        let Some(name) = rel.pop() else {
            return Err(violation(location, "the anchor itself is not an object"));
        };
        let parent = self.dir(&rel, create).map_err(|e| errno(location, e))?;
        Ok((parent, name))
    }
}

fn violation(location: &Path, why: &str) -> object_store::Error {
    object_store::Error::PermissionDenied {
        path: location.to_string(),
        source: Box::new(BeneathViolation(why.to_owned())),
    }
}

/// Maps a failed system call on `location` to the store error the engine
/// understands: a link, a mount crossing or a file in place of a directory
/// is a violation (`PermissionDenied`).
fn errno(location: &Path, e: Errno) -> object_store::Error {
    match e {
        Errno::NOENT => object_store::Error::NotFound {
            path: location.to_string(),
            source: Box::new(std::io::Error::from(e)),
        },
        Errno::EXIST | Errno::NOTEMPTY => object_store::Error::AlreadyExists {
            path: location.to_string(),
            source: Box::new(std::io::Error::from(e)),
        },
        // FreeBSD reports `O_NOFOLLOW` on a link as `EMLINK`.
        Errno::LOOP | Errno::XDEV | Errno::NOTDIR | Errno::MLINK => violation(
            location,
            "a symbolic link, mount point or file is on the path",
        ),
        Errno::ACCESS | Errno::PERM => object_store::Error::PermissionDenied {
            path: location.to_string(),
            source: Box::new(std::io::Error::from(e)),
        },
        _ => object_store::Error::Generic {
            store: STORE,
            source: Box::new(std::io::Error::from(e)),
        },
    }
}

fn io_err(e: std::io::Error) -> object_store::Error {
    object_store::Error::Generic {
        store: STORE,
        source: Box::new(e),
    }
}

async fn blocking<T, F>(f: F) -> StoreResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> StoreResult<T> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(r) => r,
        Err(source) => Err(object_store::Error::Generic {
            store: STORE,
            source: Box::new(source),
        }),
    }
}

/// Creates the staging file `<name>#<n>` next to the final name, exclusively
/// and without following a link at that name.
fn create_staging(parent: &OwnedFd, name: &str) -> Result<(File, String), Errno> {
    let mut n: u32 = 1;
    loop {
        let staging = format!("{name}#{n}");
        match openat(
            parent,
            staging.as_str(),
            OFlags::WRONLY
                .union(OFlags::CREATE)
                .union(OFlags::EXCL)
                .union(OFlags::NOFOLLOW)
                .union(OFlags::CLOEXEC),
            Mode::from_bits_truncate(0o666),
        ) {
            Ok(fd) => return Ok((File::from(fd), staging)),
            Err(Errno::EXIST) => n = n.checked_add(1).ok_or(Errno::EXIST)?,
            Err(e) => return Err(e),
        }
    }
}

fn modified(st: &Stat) -> DateTime<Utc> {
    let secs = st.st_mtime;
    let nanos = u32::try_from(st.st_mtime_nsec).unwrap_or(0);
    DateTime::from_timestamp(secs, nanos).unwrap_or_default()
}

fn meta_of(location: Path, st: &Stat) -> ObjectMeta {
    let size = u64::try_from(st.st_size).unwrap_or(0);
    let micros = modified(st).timestamp_micros();
    ObjectMeta {
        location,
        last_modified: modified(st),
        size,
        // The scheme of `LocalFileSystem`.
        e_tag: Some(format!("\"{:x}-{:x}-{:x}\"", st.st_ino, micros, size)),
        version: None,
    }
}

/// A staging file name (`<name>#<digits>`) of an upload in flight.
fn is_staging(name: &str) -> bool {
    // As `LocalFileSystem` reads a name: the text after the first `#`.
    name.split_once('#')
        .is_some_and(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// The object path of entry `name` under `rel` (below the anchor).
fn child_path(base: &[String], rel: &[String], name: &str) -> StoreResult<Path> {
    let joined = base
        .iter()
        .chain(rel)
        .map(String::as_str)
        .chain(std::iter::once(name))
        .collect::<Vec<_>>()
        .join("/");
    Path::parse(&joined).map_err(|e| object_store::Error::InvalidPath { source: e })
}

/// How many times a listing re-reads a directory in which an entry vanished
/// between the read and its `lstat`.
const SETTLE_PASSES: usize = 8;

/// Entries of the open directory `dir`: `(name, lstat)`.
///
/// An entry removed between the read of the names and its `lstat` is not
/// dropped quietly: the pass is repeated (up to [`SETTLE_PASSES`]), because
/// a quiet drop turns a replaced file into a file that never was. An
/// overwrite writes its new files before it removes the old ones, so the
/// pass that follows a vanished old file lists the new files too; skipping
/// the old file alone would list an overwritten table as empty. After the
/// last pass the entries that vanished are left out.
fn entries(dir: &OwnedFd) -> StoreResult<Vec<(String, Stat)>> {
    entries_with(dir, || {})
}

/// [`entries`], calling `between` after each read of the names and before
/// their `lstat`s (a test replaces files there).
fn entries_with(dir: &OwnedFd, mut between: impl FnMut()) -> StoreResult<Vec<(String, Stat)>> {
    let mut listing = Dir::read_from(dir).map_err(|e| io_err(e.into()))?;
    let mut out = Vec::new();
    for pass in 1..=SETTLE_PASSES {
        out.clear();
        let mut names = Vec::new();
        for entry in &mut listing {
            let entry = entry.map_err(|e| io_err(e.into()))?;
            let raw = entry.file_name().to_bytes();
            if raw == b"." || raw == b".." {
                continue;
            }
            let name = std::str::from_utf8(raw)
                .map_err(|_| io_err(std::io::Error::other("a file name that is not UTF-8")))?
                .to_owned();
            names.push(name);
        }
        between();
        let mut vanished = false;
        for name in names {
            match statat(dir, name.as_str(), AtFlags::SYMLINK_NOFOLLOW) {
                Ok(st) => out.push((name, st)),
                Err(Errno::NOENT) => vanished = true,
                Err(e) => return Err(io_err(e.into())),
            }
        }
        if !vanished || pass == SETTLE_PASSES {
            break;
        }
        listing.rewind();
    }
    Ok(out)
}

impl BeneathStore {
    /// A readable descriptor of the directory `prefix`, or `None` when it
    /// does not exist or is not a directory.
    fn listing_dir(&self, prefix: &Path) -> StoreResult<Option<OwnedFd>> {
        let rel = self.rel(prefix)?;
        self.listing_rel(&rel, prefix)
    }

    /// As [`Self::listing_dir`] for the directory `rel` below the anchor;
    /// `prefix` names it in errors.
    fn listing_rel(&self, rel: &[String], prefix: &Path) -> StoreResult<Option<OwnedFd>> {
        let path_fd = match self.dir(rel, false) {
            Ok(fd) => fd,
            Err(Errno::NOENT) => return Ok(None),
            // A file in place of the directory lists nothing; a link there
            // is a swap and refused.
            Err(Errno::NOTDIR) => {
                return match rel.split_last() {
                    Some((name, parent)) => match self
                        .dir(parent, false)
                        .and_then(|p| statat(&p, name.as_str(), AtFlags::SYMLINK_NOFOLLOW))
                    {
                        Ok(st) if FileType::from_raw_mode(st.st_mode) == FileType::Symlink => {
                            Err(errno(prefix, Errno::LOOP))
                        }
                        Ok(_) | Err(Errno::NOENT | Errno::NOTDIR) => Ok(None),
                        Err(e) => Err(errno(prefix, e)),
                    },
                    None => Ok(None),
                };
            }
            Err(e) => return Err(errno(prefix, e)),
        };
        // Re-opening "." of an open directory resolves no name.
        openat(&path_fd, ".", LIST_FLAGS, Mode::empty())
            .map(Some)
            .map_err(|e| errno(prefix, e))
    }

    fn list_all(&self, prefix: &Path) -> StoreResult<Vec<ObjectMeta>> {
        let mut out = Vec::new();
        // Directories still to read, by path below the anchor: each is
        // re-resolved when its turn comes (never through a link), so one
        // directory descriptor is open at a time however wide the tree is.
        let mut pending = vec![self.rel(prefix)?];
        while let Some(rel) = pending.pop() {
            let dir = self.listing_rel(&rel, prefix)?;
            let Some(dir) = dir else { continue };
            for (name, st) in entries(&dir)? {
                if FileType::from_raw_mode(st.st_mode) == FileType::Directory {
                    let mut sub = rel.clone();
                    sub.push(name);
                    pending.push(sub);
                } else if !is_staging(&name) {
                    // A link is an object of its own (its lstat size and
                    // mtime), never descended into.
                    out.push(meta_of(child_path(&self.inner.base, &rel, &name)?, &st));
                }
            }
        }
        Ok(out)
    }

    fn list_one_level(&self, prefix: &Path) -> StoreResult<ListResult> {
        let mut result = ListResult {
            common_prefixes: Vec::new(),
            objects: Vec::new(),
            extensions: Default::default(),
        };
        let Some(dir) = self.listing_dir(prefix)? else {
            return Ok(result);
        };
        let rel = self.rel(prefix)?;
        for (name, st) in entries(&dir)? {
            let at = child_path(&self.inner.base, &rel, &name)?;
            if FileType::from_raw_mode(st.st_mode) == FileType::Directory {
                result.common_prefixes.push(at);
            } else if !is_staging(&name) {
                result.objects.push(meta_of(at, &st));
            }
        }
        Ok(result)
    }
}

#[async_trait]
impl ObjectStore for BeneathStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> StoreResult<PutResult> {
        if !matches!(opts.mode, PutMode::Overwrite) {
            return Err(object_store::Error::NotImplemented {
                operation: "put with a mode other than overwrite".to_owned(),
                implementer: STORE.to_owned(),
            });
        }
        let store = self.clone();
        let location = location.clone();
        blocking(move || {
            let (parent, name) = store.parent_and_name(&location, true)?;
            let (file, staging) =
                create_staging(&parent, &name).map_err(|e| errno(&location, e))?;
            let written = payload
                .iter()
                .try_fold(0_u64, |at, chunk| {
                    file.write_all_at(chunk, at)?;
                    Ok::<_, std::io::Error>(at.saturating_add(chunk.len() as u64))
                })
                .map_err(io_err);
            let published = written.and_then(|_| {
                renameat(&parent, staging.as_str(), &parent, name.as_str())
                    .map_err(|e| errno(&location, e))
            });
            if published.is_err() {
                let _ = unlinkat(&parent, staging.as_str(), AtFlags::empty());
            }
            published.map(|()| PutResult {
                e_tag: None,
                version: None,
                extensions: Default::default(),
            })
        })
        .await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        _opts: PutMultipartOptions,
    ) -> StoreResult<Box<dyn MultipartUpload>> {
        let store = self.clone();
        let location = location.clone();
        blocking(move || {
            let (parent, name) = store.parent_and_name(&location, true)?;
            let (file, staging) =
                create_staging(&parent, &name).map_err(|e| errno(&location, e))?;
            Ok(Box::new(BeneathUpload {
                shared: Arc::new(UploadShared {
                    file,
                    parent,
                    name,
                    staging,
                    location,
                    done: AtomicBool::new(false),
                }),
                offset: 0,
            }) as Box<dyn MultipartUpload>)
        })
        .await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> StoreResult<GetResult> {
        if !options.head {
            return Err(object_store::Error::NotImplemented {
                operation: "reading an object body".to_owned(),
                implementer: STORE.to_owned(),
            });
        }
        let store = self.clone();
        let location = location.clone();
        blocking(move || {
            let (parent, name) = store.parent_and_name(&location, false)?;
            let st = statat(&parent, name.as_str(), AtFlags::SYMLINK_NOFOLLOW)
                .map_err(|e| errno(&location, e))?;
            // A directory is not an object.
            if FileType::from_raw_mode(st.st_mode) == FileType::Directory {
                return Err(object_store::Error::NotFound {
                    path: location.to_string(),
                    source: Box::new(std::io::Error::other("is a directory")),
                });
            }
            let meta = meta_of(location, &st);
            Ok(GetResult {
                payload: GetResultPayload::Stream(futures::stream::empty().boxed()),
                range: 0..meta.size,
                meta,
                attributes: object_store::Attributes::default(),
                extensions: Default::default(),
            })
        })
        .await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, StoreResult<Path>>,
    ) -> BoxStream<'static, StoreResult<Path>> {
        let store = self.clone();
        locations
            .then(move |location| {
                let store = store.clone();
                async move {
                    let location = location?;
                    blocking(move || {
                        let (parent, name) = store.parent_and_name(&location, false)?;
                        // A link is removed itself, never its target.
                        unlinkat(&parent, name.as_str(), AtFlags::empty())
                            .map_err(|e| errno(&location, e))?;
                        Ok(location)
                    })
                    .await
                }
            })
            .boxed()
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, StoreResult<ObjectMeta>> {
        let store = self.clone();
        let prefix = prefix
            .cloned()
            .unwrap_or_else(|| Path::from_iter(store.inner.base.iter().map(String::as_str)));
        futures::stream::once(async move { blocking(move || store.list_all(&prefix)).await })
            .flat_map(|r| match r {
                Ok(v) => futures::stream::iter(v.into_iter().map(Ok)).boxed(),
                Err(e) => futures::stream::once(std::future::ready(Err(e))).boxed(),
            })
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> StoreResult<ListResult> {
        let store = self.clone();
        let prefix = prefix
            .cloned()
            .unwrap_or_else(|| Path::from_iter(store.inner.base.iter().map(String::as_str)));
        blocking(move || store.list_one_level(&prefix)).await
    }

    async fn copy_opts(&self, _from: &Path, _to: &Path, _options: CopyOptions) -> StoreResult<()> {
        Err(object_store::Error::NotImplemented {
            operation: "copy".to_owned(),
            implementer: STORE.to_owned(),
        })
    }
}

#[derive(Debug)]
struct UploadShared {
    file: File,
    parent: OwnedFd,
    name: String,
    staging: String,
    location: Path,
    /// Completed or aborted: later calls fail.
    done: AtomicBool,
}

/// A multipart upload: one staging file held open (parts are written at
/// their offsets through its own descriptor, so a swapped parent cannot
/// redirect them), renamed into place within the held parent on completion.
#[derive(Debug)]
struct BeneathUpload {
    shared: Arc<UploadShared>,
    offset: u64,
}

#[async_trait]
impl MultipartUpload for BeneathUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let at = self.offset;
        self.offset = self.offset.saturating_add(data.content_length() as u64);
        let shared = Arc::clone(&self.shared);
        async move {
            blocking(move || {
                if shared.done.load(Ordering::SeqCst) {
                    return Err(io_err(std::io::Error::other("the upload is finished")));
                }
                data.iter()
                    .try_fold(at, |at, chunk| {
                        shared.file.write_all_at(chunk, at)?;
                        Ok::<_, std::io::Error>(at.saturating_add(chunk.len() as u64))
                    })
                    .map(|_| ())
                    .map_err(io_err)
            })
            .await
        }
        .boxed()
    }

    async fn complete(&mut self) -> StoreResult<PutResult> {
        let shared = Arc::clone(&self.shared);
        blocking(move || {
            if shared.done.swap(true, Ordering::SeqCst) {
                return Err(io_err(std::io::Error::other("the upload is finished")));
            }
            renameat(
                &shared.parent,
                shared.staging.as_str(),
                &shared.parent,
                shared.name.as_str(),
            )
            .map_err(|e| errno(&shared.location, e))?;
            Ok(PutResult {
                e_tag: None,
                version: None,
                extensions: Default::default(),
            })
        })
        .await
    }

    async fn abort(&mut self) -> StoreResult<()> {
        let shared = Arc::clone(&self.shared);
        blocking(move || {
            if shared.done.swap(true, Ordering::SeqCst) {
                return Ok(());
            }
            match unlinkat(&shared.parent, shared.staging.as_str(), AtFlags::empty()) {
                Ok(()) | Err(Errno::NOENT) => Ok(()),
                Err(e) => Err(errno(&shared.location, e)),
            }
        })
        .await
    }
}

#[cfg(test)]
mod tests;
