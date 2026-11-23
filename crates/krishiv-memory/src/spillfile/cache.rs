// SPDX-License-Identifier: Apache-2.0
//! Page-cache control for written spill ranges (Linux, through `rustix` and
//! one `libc` call; no-ops returning `Ok` elsewhere). Advisory: callers log
//! failures and go on, since the data is written either way.
//!
//! `POSIX_FADV_DONTNEED` on Linux starts writeback of the range's dirty
//! pages without waiting, then drops the pages that are clean: called on a
//! range once starts its writeback, called again later drops it.
//! [`sync`] waits for the file's writeback, after which a drop removes
//! everything.
//!
//! The wait is `sync_file_range(WAIT_BEFORE | WRITE | WAIT_AFTER)`, not
//! `fdatasync`: spill files are scratch data, so only the page state
//! matters, not durability. `fdatasync` also commits the file system's
//! journal and flushes the device's write cache, once per file; a spilling
//! aggregate finishes one file per radix partition per run, and on TPC-H
//! q13 at SF100 under a 2 GiB limit those flushes were most of its write
//! time (0.93x wall with the range wait). Where the
//! call is refused (a file system without it), the wait falls back to
//! `fdatasync`.

use std::fs::File;
use std::io;

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

/// Waits until the file's written pages are written back (clean), so a
/// drop after it removes them all: the range wait, else `fdatasync`.
#[cfg(target_os = "linux")]
pub(crate) fn sync(file: &File) -> io::Result<()> {
    sync_with(file, wait_range)
}

/// Elsewhere: nothing to wait for (no drop-behind).
#[cfg(not(target_os = "linux"))]
pub(crate) fn sync(file: &File) -> io::Result<()> {
    let _ = file;
    Ok(())
}

/// [`sync`] with the first wait given: `fdatasync` when it fails.
#[cfg(target_os = "linux")]
fn sync_with(file: &File, wait: fn(&File) -> io::Result<()>) -> io::Result<()> {
    match wait(file) {
        Ok(()) => Ok(()),
        Err(_) => Ok(rustix::fs::fdatasync(file)?),
    }
}

/// Starts writeback of the whole file's dirty pages and waits for it,
/// with no journal commit or device flush.
#[cfg(target_os = "linux")]
fn wait_range(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let flags = libc::SYNC_FILE_RANGE_WAIT_BEFORE
        | libc::SYNC_FILE_RANGE_WRITE
        | libc::SYNC_FILE_RANGE_WAIT_AFTER;
    // SAFETY: `sync_file_range` takes a descriptor and plain integers and
    // touches no memory of ours. The descriptor is `file`'s, open for the
    // whole call because `file` is borrowed; offset 0 with length 0 means
    // "to the end of the file", a valid range.
    let rc = unsafe { libc::sync_file_range(file.as_raw_fd(), 0, 0, flags) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // A pipe end refuses both waits: the error is reported.
        let (r, _w) = std::io::pipe().unwrap();
        let pipe = File::from(std::os::fd::OwnedFd::from(r));
        assert!(wait_range(&pipe).is_err());
        assert!(sync(&pipe).is_err());
    }

    /// A refused range wait falls back to `fdatasync`; a wait that
    /// succeeds is the only one.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_refused_range_wait_falls_back_to_fdatasync() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let mut f = File::create(dir.path().join("x")).unwrap();
        f.write_all(&[7; 8192]).unwrap();
        assert!(wait_range(&f).is_ok());
        let refused: fn(&File) -> io::Result<()> =
            |_| Err(io::Error::from_raw_os_error(libc::EINVAL));
        assert!(sync_with(&f, refused).is_ok(), "fdatasync took over");
        let (r, _w) = std::io::pipe().unwrap();
        let pipe = File::from(std::os::fd::OwnedFd::from(r));
        assert!(sync_with(&pipe, |_| Ok(())).is_ok(), "no second wait");
        assert!(sync_with(&pipe, refused).is_err(), "fdatasync refused too");
    }
}
