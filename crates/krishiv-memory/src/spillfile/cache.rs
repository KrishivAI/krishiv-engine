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

/// Waits until the file's written data is on disk (`fdatasync`).
pub(crate) fn sync(file: &File) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    rustix::fs::fdatasync(file)?;
    #[cfg(not(target_os = "linux"))]
    let _ = file;
    Ok(())
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
    }
}
