// SPDX-License-Identifier: Apache-2.0
//! Page-cache control for written spill ranges (Linux, through `rustix`;
//! no-ops returning `Ok` elsewhere). Advisory: callers log failures and go
//! on, since the data is written either way.
//!
//! `POSIX_FADV_DONTNEED` on Linux starts writeback of the range's dirty
//! pages without waiting, then drops the pages that are clean: called on a
//! range once starts its writeback, called again later drops it.
//! [`sync`] waits for the file's writeback (`fdatasync`), after which a
//! drop removes everything.
//!
//! A finished file is not synced at once: [`Settling`] holds it (with its
//! writeback started) until the files held reach the window's bytes or
//! [`SETTLE_FILES`], then syncs and drops them together. By then most of
//! their writeback is done, so the syncs barely wait; one `fdatasync` per
//! small file at `finish` cost a spilling aggregate under a 16 MiB limit
//! most of its time (QA2-16). A file already deleted is not synced.

use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Most finished files [`Settling`] holds before it settles them, so that
/// many small files never hold many descriptors.
pub(crate) const SETTLE_FILES: usize = 64;

/// Starts writeback of `[offset, offset + len)` and drops its clean pages.
pub(crate) fn drop_range(file: &File, offset: u64, len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    if let Some(len) = std::num::NonZeroU64::new(len) {
        rustix::fs::fadvise(file, offset, Some(len), rustix::fs::Advice::DontNeed)?;
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, offset, len);
    Ok(())
}

/// Waits until the file's written data is on disk (`fdatasync`).
pub(crate) fn sync(file: &File) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    rustix::fs::fdatasync(file)?;
    #[cfg(not(target_os = "linux"))]
    let _ = file;
    Ok(())
}

/// Whether the file has no name left (deleted): its pages go with its last
/// descriptor, so syncing it would only wait for nothing.
fn deleted(file: &File) -> bool {
    #[cfg(target_os = "linux")]
    {
        rustix::fs::fstat(file).is_ok_and(|s| s.st_nlink == 0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        false
    }
}

/// Locks `m`, ignoring poisoning: [`Settling`] only pushes and takes.
/// (std's lock, not the loom-swapped one: a manager is made outside loom
/// models too.)
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One finished file waiting to be settled.
#[derive(Debug)]
struct Held {
    file: File,
    len: u64,
    path: PathBuf,
}

/// Finished spill files whose writeback was started but not waited for,
/// shared by every writer of a spill manager.
#[derive(Debug, Default)]
pub(crate) struct Settling {
    held: Mutex<(Vec<Held>, u64)>,
}

/// A failed sync or drop of one settled file, for the caller to log.
pub(crate) type Failure = (PathBuf, io::Error);

impl Settling {
    /// Holds `file` (its first `len` bytes written, their writeback
    /// started). Once the held files reach `window` bytes or
    /// [`SETTLE_FILES`], this call syncs and drops all of them, on the
    /// calling thread and outside the lock, and returns the failures.
    pub(crate) fn hold(&self, file: File, len: u64, path: PathBuf, window: u64) -> Vec<Failure> {
        let due = {
            let mut g = lock(&self.held);
            g.0.push(Held { file, len, path });
            g.1 = g.1.saturating_add(len);
            if g.1 < window && g.0.len() < SETTLE_FILES {
                return Vec::new();
            }
            g.1 = 0;
            std::mem::take(&mut g.0)
        };
        let mut failures = Vec::new();
        for h in due {
            if deleted(&h.file) {
                continue;
            }
            // EXPERIMENT ONLY (KRISHIV_X_SETTLE), not for the PR.
            let mode = std::env::var("KRISHIV_X_SETTLE").unwrap_or_default();
            let r = if mode == "fadvise" {
                drop_range(&h.file, 0, h.len)
            } else {
                sync(&h.file).and_then(|()| drop_range(&h.file, 0, h.len))
            };
            if let Err(e) = r {
                failures.push((h.path, e));
            }
        }
        failures
    }

    /// Files held now (tests).
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        lock(&self.held).0.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Files are held until their bytes reach the window (or the file count
    /// reaches [`SETTLE_FILES`]); deleted files are skipped when settled.
    #[test]
    fn settles_at_the_window_or_the_file_count() {
        let dir = tempfile::tempdir().unwrap();
        let file = |n: usize| {
            let p = dir.path().join(format!("f{n}"));
            std::fs::write(&p, [7u8; 4096]).unwrap();
            (File::open(&p).unwrap(), p)
        };
        let s = Settling::default();
        let (f, p) = file(0);
        assert!(s.hold(f, 4096, p, 10_000).is_empty());
        assert_eq!(s.held(), 1);
        let (f, p) = file(1);
        std::fs::remove_file(&p).unwrap();
        assert!(s.hold(f, 4096, p, 10_000).is_empty());
        assert_eq!(s.held(), 2);
        let (f, p) = file(2);
        assert!(
            s.hold(f, 4096, p, 10_000).is_empty(),
            "12 KiB passes the window"
        );
        assert_eq!(s.held(), 0);
        for n in 0..SETTLE_FILES {
            let (f, p) = file(100 + n);
            assert!(s.hold(f, 1, p, u64::MAX).is_empty());
        }
        assert_eq!(s.held(), 0, "the file count settles too");
    }

    /// A failing call reports the error (for the writer to log); an empty
    /// range is not a call.
    #[cfg(target_os = "linux")]
    #[test]
    fn errors_are_reported_and_empty_ranges_skipped() {
        // A directory opened read-only: `fdatasync` and `fadvise` on it are
        // refused or no-ops depending on the file system, but a pipe end
        // always refuses `fadvise` (ESPIPE).
        let (r, _w) = std::io::pipe().unwrap();
        let file = File::from(std::os::fd::OwnedFd::from(r));
        assert!(drop_range(&file, 0, 4096).is_err());
        assert!(drop_range(&file, 0, 0).is_ok(), "empty: nothing to do");
        let dir = tempfile::tempdir().unwrap();
        let f = File::create(dir.path().join("x")).unwrap();
        assert!(sync(&f).is_ok());
        assert!(drop_range(&f, 0, 4096).is_ok());
    }
}
