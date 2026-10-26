// SPDX-License-Identifier: Apache-2.0
//! Spill-file drop-behind (`krishiv.spill.dropBehindBytes`): a finished
//! spill file's pages are written back and dropped from the page cache, so
//! they no longer count against a memory cgroup; with the window at 0 they
//! stay cached. Either way the file reads back intact. Page residency is
//! read with `mincore` on a mapping of the file (Linux, on a disk file
//! system: the target directory; tmpfs keeps its pages by design).
#![cfg(target_os = "linux")]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "test helpers sit outside #[test] fns"
)]

use std::ffi::{c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use krishiv_memory::{
    MemoryConfig, MemoryPool, Reservation, SpillCodec, SpillConfig, SpillFile, SpillManager,
    TreePool,
};
use krishiv_types::batch::Batch;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
    fn sysconf(name: c_int) -> i64;
}
const PROT_READ: c_int = 1;
const MAP_SHARED: c_int = 1;
const SC_PAGESIZE: c_int = 30;

/// Pages of `path` in the page cache, and its page count.
fn resident_pages(path: &Path) -> (usize, usize) {
    let file = File::open(path).unwrap();
    let len = usize::try_from(file.metadata().unwrap().len()).unwrap();
    // SAFETY: plain query of a constant.
    let page = usize::try_from(unsafe { sysconf(SC_PAGESIZE) }).unwrap();
    let pages = len.div_ceil(page);
    let mut vec = vec![0u8; pages];
    // SAFETY: a read-only shared mapping of an open file of `len` bytes; it
    // is only passed to `mincore` (never dereferenced) and unmapped below.
    unsafe {
        let addr = mmap(
            std::ptr::null_mut(),
            len,
            PROT_READ,
            MAP_SHARED,
            file.as_raw_fd(),
            0,
        );
        assert_ne!(addr as isize, -1, "mmap failed");
        // `vec` holds one byte per page of the mapping.
        assert_eq!(mincore(addr, len, vec.as_mut_ptr()), 0, "mincore failed");
        assert_eq!(munmap(addr, len), 0);
    }
    (vec.iter().filter(|&&b| b & 1 == 1).count(), pages)
}

/// Whether `dir` is on a RAM file system, where dropping clean pages does
/// nothing (they are the file).
fn on_ram_fs(dir: &Path) -> bool {
    let dir = dir.canonicalize().unwrap();
    let mounts = std::fs::read_to_string("/proc/self/mounts").unwrap_or_default();
    mounts
        .lines()
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let (_, at, fs) = (f.next()?, f.next()?, f.next()?);
            dir.starts_with(at).then(|| (at.len(), fs.to_owned()))
        })
        .max()
        .is_some_and(|(_, fs)| fs == "tmpfs" || fs == "ramfs")
}

/// A private (0700, as the spill manager requires) directory on the disk file
/// system of the target dir, removed on drop.
fn dir() -> tempfile::TempDir {
    use std::os::unix::fs::PermissionsExt;
    tempfile::Builder::new()
        .prefix("drop-behind-")
        .permissions(std::fs::Permissions::from_mode(0o700))
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .unwrap()
}

/// The system page size.
fn page_size() -> usize {
    // SAFETY: plain query of a constant.
    usize::try_from(unsafe { sysconf(SC_PAGESIZE) }).unwrap()
}

/// 64 batches of 4096 rows with a 200-byte string: about 52 MiB.
fn batches() -> Vec<Batch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    (0..64i64)
        .map(|b| {
            let ids = (b * 4096)..((b + 1) * 4096);
            let i = Int64Array::from_iter_values(ids.clone());
            let s = StringArray::from_iter_values(ids.map(|i| format!("{i:0>200}")));
            Batch::new(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(i), Arc::new(s)]).unwrap(),
            )
        })
        .collect()
}

/// What [`write`] returns; the manager and the temp directory outlive the file.
struct Written {
    _manager: SpillManager,
    _root: tempfile::TempDir,
    file: SpillFile,
    input: Vec<Batch>,
}

/// Writes [`batches`] with drop-behind window `window` and returns the file
/// with its manager. With `ram` false and a window set, the periodic
/// drop-behind during the write (before `finish`, which drops everything
/// itself) is tested halfway: the test syncs the file being written, so
/// every page written so far is clean, then writes two more batches (each
/// almost a window, so the writer drops at least once). A drop discards
/// every clean page, so what is still cached afterwards is at most what was
/// written after the sync, plus page-boundary slack.
///
/// The sync makes the check deterministic: without it, which pages a drop
/// finds clean depends on how far the kernel's writeback got, which on a
/// slow or busy disk (CI) can be several windows behind.
fn write(window: u64, ram: bool) -> Written {
    let root_dir = dir();
    let manager = SpillManager::open(&SpillConfig {
        dir: root_dir.path().to_owned(),
        drop_behind: window,
        // Uncompressed: the file is as large as the data.
        codec: SpillCodec::None,
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let root = TreePool::new(&MemoryConfig {
        limit: None,
        ..MemoryConfig::default()
    });
    let pool: Arc<dyn MemoryPool> = root.child("q", None);
    let id = pool.register("spill", None);
    let input = batches();
    let mut w = scope
        .writer(input[0].schema(), Reservation::new(Arc::clone(&pool), id))
        .unwrap();
    let half = input.len() / 2;
    let mut live = None;
    for (n, b) in input.iter().enumerate() {
        w.write(b).unwrap();
        if window == 0 || ram {
            continue;
        }
        if n + 1 == half {
            let path = std::fs::read_dir(scope.dir())
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.to_string_lossy().ends_with(".spill.tmp"))
                .expect("the file being written");
            // Every page written so far is now clean.
            File::open(&path).unwrap().sync_all().unwrap();
            live = Some(path);
        }
        if n + 1 == half + 2 {
            let path = live.take().expect("synced halfway");
            let (resident, pages) = resident_pages(&path);
            let bound = 3 * usize::try_from(window).unwrap().div_ceil(page_size()) + 64;
            eprintln!("mid-write: {resident} of {pages} pages cached (bound {bound})");
            assert!(pages > 2 * bound, "{pages}");
            assert!(
                resident <= bound,
                "{resident} of {pages} pages cached mid-write (bound {bound})"
            );
        }
    }
    let file = w.finish().unwrap();
    Written {
        _manager: manager,
        _root: root_dir,
        file,
        input,
    }
}

fn read_back(file: &SpillFile) -> Vec<Batch> {
    let root = TreePool::new(&MemoryConfig {
        limit: None,
        ..MemoryConfig::default()
    });
    let pool: Arc<dyn MemoryPool> = root.child("q", None);
    let id = pool.register("read", None);
    file.reader(Reservation::new(pool, id))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn a_finished_spill_file_is_dropped_from_the_page_cache_and_reads_back() {
    let probe = dir();
    let ram = on_ram_fs(probe.path());
    if ram {
        eprintln!("spill test directory is on a RAM file system: residency not checked");
    }

    // Window 1 MiB: several windows dropped while writing (checked halfway),
    // the rest at finish.
    let on = write(1 << 20, ram);
    let (resident, pages) = resident_pages(on.file.path());
    eprintln!("drop-behind on: {resident} of {pages} pages cached");
    if !ram {
        assert!(pages > 1000, "{pages}");
        assert!(
            resident * 100 <= pages,
            "{resident} of {pages} pages still cached"
        );
    }
    let output = read_back(&on.file);
    assert_eq!(output.len(), on.input.len());
    for (a, b) in on.input.iter().zip(&output) {
        assert_eq!(a.data(), b.data());
    }

    // Off: the written pages stay cached (the kernel had no reason to drop them).
    let off = write(0, ram);
    let (resident, pages) = resident_pages(off.file.path());
    eprintln!("drop-behind off: {resident} of {pages} pages cached");
    if !ram {
        assert!(
            resident * 2 > pages,
            "{resident} of {pages}: the control lost its pages"
        );
    }
}
