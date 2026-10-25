// SPDX-License-Identifier: Apache-2.0
//! Page-cache control for written spill ranges: start writeback, wait for
//! it, drop clean pages (Linux `sync_file_range` and `posix_fadvise`;
//! no-ops returning `Ok` elsewhere). Advisory: callers log failures and go
//! on, since the data is written either way.

use std::fs::File;
use std::io;

#[cfg(target_os = "linux")]
mod sys {
    use std::ffi::c_int;

    unsafe extern "C" {
        pub(super) fn sync_file_range(fd: c_int, offset: i64, nbytes: i64, flags: u32) -> c_int;
        pub(super) fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
    }
    pub(super) const SYNC_FILE_RANGE_WAIT_BEFORE: u32 = 1;
    pub(super) const SYNC_FILE_RANGE_WRITE: u32 = 2;
    pub(super) const SYNC_FILE_RANGE_WAIT_AFTER: u32 = 4;
    pub(super) const POSIX_FADV_DONTNEED: c_int = 4;
}

/// `(fd, offset, len)` for the syscalls; `None` for an empty range.
#[cfg(target_os = "linux")]
fn args(file: &File, offset: u64, len: u64) -> Option<(std::ffi::c_int, i64, i64)> {
    use std::os::fd::AsRawFd;
    (len > 0).then(|| {
        (
            file.as_raw_fd(),
            i64::try_from(offset).unwrap_or(i64::MAX),
            i64::try_from(len).unwrap_or(i64::MAX),
        )
    })
}

/// Starts writeback of `[offset, offset + len)` without waiting for it.
pub(crate) fn start_writeback(file: &File, offset: u64, len: u64) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    if let Some((fd, off, len)) = args(file, offset, len) {
        // SAFETY: `fd` is an open descriptor owned by `file`, which outlives
        // the call; the syscall reads and writes none of our memory.
        if unsafe { sys::sync_file_range(fd, off, len, sys::SYNC_FILE_RANGE_WRITE) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, offset, len);
    Ok(())
}

/// Writes `[offset, offset + len)` back if `wait` (starting writeback of
/// what is still dirty and waiting for all of it), then advises the kernel
/// to drop the range's clean pages. Without `wait`, pages still dirty or
/// under writeback stay cached until written back, then reclaimable.
pub(crate) fn drop_range(file: &File, offset: u64, len: u64, wait: bool) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    if let Some((fd, off, len)) = args(file, offset, len) {
        if wait {
            let flags = sys::SYNC_FILE_RANGE_WAIT_BEFORE
                | sys::SYNC_FILE_RANGE_WRITE
                | sys::SYNC_FILE_RANGE_WAIT_AFTER;
            // SAFETY: as in `start_writeback`.
            if unsafe { sys::sync_file_range(fd, off, len, flags) } != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        // SAFETY: as in `start_writeback`. `posix_fadvise` returns the error
        // number instead of setting `errno`.
        let rc = unsafe { sys::posix_fadvise(fd, off, len, sys::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return Err(io::Error::from_raw_os_error(rc));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, offset, len, wait);
    Ok(())
}
