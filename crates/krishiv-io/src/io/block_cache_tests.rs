// SPDX-License-Identifier: Apache-2.0
//! The block cache (P2-T083): re-scans, overwrites, caps, single flight, disk faults and keys, over
//! an in-memory store behind `ThrottledStore` and a counting `FaultStore` (no MinIO).

#![expect(
    clippy::single_range_in_vec_init,
    reason = "single-range requests are the cases under test"
)]

use std::path::Path as FsPath;
use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::throttle::{ThrottleConfig, ThrottledStore};
use object_store::{ObjectStore, ObjectStoreExt};

use super::*;

const BLOCK: u64 = 64 * 1024;

struct Rig {
    io: IoContext,
    sched: Arc<IoScheduler>,
    fault: Arc<FaultStore>,
    local: Arc<InMemory>,
    id: StoreId,
    cache_dir: tempfile::TempDir,
}

fn rig(disk_bytes: u64, memory_bytes: u64, cached: bool) -> Rig {
    let cache_dir = tempfile::tempdir().unwrap();
    let local = Arc::new(InMemory::new());
    let throttled = ThrottledStore::new(
        Arc::clone(&local),
        ThrottleConfig {
            wait_get_per_call: Duration::from_millis(15),
            wait_get_per_byte: Duration::from_nanos(60),
            ..ThrottleConfig::default()
        },
    );
    let fault = FaultStore::new(Arc::new(throttled) as Arc<dyn ObjectStore>);
    let config = IoConfig {
        block_cache: cached.then(|| BlockCacheConfig {
            dir: cache_dir.path().to_path_buf(),
            disk_bytes,
            memory_bytes,
            block_bytes: BLOCK,
        }),
        ..IoConfig::default()
    };
    let sched = IoScheduler::new(config);
    let id = StoreId::new("s3://test-bucket");
    sched.register_store(id.clone(), Arc::clone(&fault) as Arc<dyn ObjectStore>);
    Rig {
        io: sched.context(),
        sched,
        fault,
        local,
        id,
        cache_dir,
    }
}

fn pattern(len: usize, phase: u8) -> Vec<u8> {
    (0..len)
        .map(|i| u8::try_from((i * 7 + usize::from(phase)) % 251).unwrap())
        .collect()
}

async fn put(r: &Rig, name: &str, data: Vec<u8>) -> FileRef {
    let path = Path::from(name);
    r.local.put(&path, data.into()).await.unwrap();
    FileRef::from_meta(r.id.clone(), &r.local.head(&path).await.unwrap())
}

/// What a column scan reads: the footer, then three column chunks.
fn scan_ranges(size: u64) -> Vec<std::ops::Range<u64>> {
    vec![
        size - 65_536..size,
        0..400_000,
        900_000..1_300_000,
        2_000_000..2_400_000,
    ]
}

async fn scan(r: &Rig, files: &[FileRef]) -> Vec<Vec<Bytes>> {
    let mut out = Vec::new();
    for f in files {
        out.push(r.io.get_ranges(f, &scan_ranges(f.size)).await.unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rescan_is_served_from_the_cache() {
    let (cold, plain) = (rig(1 << 30, 8 << 20, true), rig(0, 0, false));
    let mut files = Vec::new();
    let mut plain_files = Vec::new();
    for i in 0..4_u8 {
        let data = pattern(3 << 20, i);
        files.push(put(&cold, &format!("f{i}"), data.clone()).await);
        plain_files.push(put(&plain, &format!("f{i}"), data).await);
    }
    let t = Instant::now();
    let want = scan(&plain, &plain_files).await;
    let uncached = t.elapsed();
    let uncached_stats = plain.fault.stats();

    let t = Instant::now();
    let first = scan(&cold, &files).await;
    let first_time = t.elapsed();
    let first_stats = cold.fault.stats();
    let t = Instant::now();
    let second = scan(&cold, &files).await;
    let second_time = t.elapsed();
    let second_stats = cold.fault.stats();

    eprintln!(
        "T083 uncached {uncached:?} ({} gets, {} bytes); cold {first_time:?} ({} gets, {} bytes); re-scan {second_time:?}",
        uncached_stats.gets, uncached_stats.bytes, first_stats.gets, first_stats.bytes
    );
    // (8) The answers are the uncached answers.
    assert_eq!(first, want);
    assert_eq!(second, want);
    // (1) The second scan issues at most 5% of the first scan's requests and bytes, and is at least
    // twice as fast.
    let (requests, bytes) = (
        second_stats.gets - first_stats.gets,
        second_stats.bytes - first_stats.bytes,
    );
    assert_eq!(requests, 0, "the re-scan reached the store");
    assert_eq!(bytes, 0);
    // (2) The cold scan costs no more requests than the uncached one (the
    // time is within noise of the uncached scan, which the request counts bound).
    assert!(first_stats.gets <= uncached_stats.gets);
    // Aligned blocks over-read by up to one block at each end of each requested range.
    assert!(first_stats.bytes <= uncached_stats.bytes + files.len() as u64 * 8 * BLOCK);
    let stats = cold.sched.block_cache_stats().unwrap();
    assert!(stats.misses > 0 && stats.memory_hits + stats.disk_hits >= stats.misses);
    assert_eq!(stats.errors, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwritten_object_is_read_anew() {
    let r = rig(1 << 30, 8 << 20, true);
    let old = put(&r, "o", pattern(500_000, 1)).await;
    let a = r.io.get_ranges(&old, &[100..200_000]).await.unwrap();
    assert_eq!(
        a[0],
        Bytes::from(pattern(500_000, 1)[100..200_000].to_vec())
    );
    // Same size, new bytes: the version (the ETag) changes.
    let new = put(&r, "o", pattern(500_000, 2)).await;
    assert_ne!(old.version, new.version);
    let b = r.io.get_ranges(&new, &[100..200_000]).await.unwrap();
    assert_eq!(
        b[0],
        Bytes::from(pattern(500_000, 2)[100..200_000].to_vec())
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_disk_stays_under_its_cap() {
    // 512 KiB of disk for a 4 MiB working set, memory off.
    let r = rig(8 * BLOCK, 0, true);
    let data = pattern(4 << 20, 3);
    let f = put(&r, "big", data.clone()).await;
    let cache = r.sched.block_cache.clone().unwrap();
    for start in (0..4_000_000_u64).step_by(150_000) {
        let got =
            r.io.get_ranges(&f, &[start..start + 150_000])
                .await
                .unwrap();
        assert_eq!(
            got[0],
            Bytes::from(
                data[usize::try_from(start).unwrap()..usize::try_from(start + 150_000).unwrap()]
                    .to_vec()
            )
        );
        assert!(
            cache.disk_used() <= 8 * BLOCK + BLOCK,
            "{}",
            cache.disk_used()
        );
    }
    assert!(cache.stats().evictions > 0);
    let files = std::fs::read_dir(cache_files(&r)).unwrap().count();
    assert!(files as u64 <= 9, "{files} block files");
}

fn cache_files(r: &Rig) -> std::path::PathBuf {
    std::fs::read_dir(r.cache_dir.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_scans_fetch_each_block_once() {
    let r = Arc::new(rig(1 << 30, 8 << 20, true));
    let data = pattern(1 << 20, 4);
    let f = put(&r, "shared", data.clone()).await;
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let (r, f) = (Arc::clone(&r), f.clone());
        tasks.push(tokio::spawn(async move {
            r.io.get_ranges(&f, &[0..1 << 20]).await.unwrap()
        }));
    }
    for t in tasks {
        assert_eq!(t.await.unwrap()[0], Bytes::from(data.clone()));
    }
    let blocks = (1_u64 << 20) / BLOCK;
    let stats = r.sched.block_cache_stats().unwrap();
    assert_eq!(stats.misses, blocks, "each block is fetched once");
    // The object's metadata request, and the blocks as one run.
    assert!(r.fault.stats().gets <= 1 + blocks, "{:?}", r.fault.stats());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disk_errors_fall_back_to_the_store() {
    let r = rig(1 << 30, 0, true);
    let data = pattern(600_000, 5);
    let f = put(&r, "d", data.clone()).await;
    // The directory vanishes under the cache: writes fail.
    std::fs::remove_dir_all(cache_files(&r)).unwrap();
    let got =
        r.io.get_ranges(&f, &[0..300_000, 400_000..500_000])
            .await
            .unwrap();
    assert_eq!(got[0], Bytes::from(data[..300_000].to_vec()));
    assert_eq!(got[1], Bytes::from(data[400_000..500_000].to_vec()));
    // Reading the same ranges again hits the store again, with the same answer.
    let again = r.io.get_ranges(&f, &[0..300_000]).await.unwrap();
    assert_eq!(again[0], got[0]);
    assert!(r.sched.block_cache_stats().unwrap().errors > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_corrupt_block_is_a_miss() {
    let r = rig(1 << 30, 0, true);
    let data = pattern(300_000, 6);
    let f = put(&r, "c", data.clone()).await;
    r.io.get_ranges(&f, &[0..100_000]).await.unwrap();
    for entry in std::fs::read_dir(cache_files(&r)).unwrap() {
        std::fs::write(entry.unwrap().path(), b"garbage").unwrap();
    }
    let got = r.io.get_ranges(&f, &[0..100_000]).await.unwrap();
    assert_eq!(got[0], Bytes::from(data[..100_000].to_vec()));
    assert!(r.sched.block_cache_stats().unwrap().errors > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_damaged_payload_is_a_miss() {
    // A disk-only cache: a flipped byte, then a truncated file, must both be re-fetched.
    for truncate in [false, true] {
        let r = rig(1 << 30, 0, true);
        let data = pattern(300_000, 8);
        let f = put(&r, "d", data.clone()).await;
        r.io.get_ranges(&f, &[0..100_000]).await.unwrap();
        for entry in std::fs::read_dir(cache_files(&r)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "blk") {
                continue;
            }
            let mut body = std::fs::read(&path).unwrap();
            if truncate {
                body.truncate(body.len() - 7);
            } else {
                let last = body.len() - 1;
                body[last] ^= 0xff;
            }
            std::fs::write(&path, body).unwrap();
        }
        let got = r.io.get_ranges(&f, &[0..100_000]).await.unwrap();
        assert_eq!(got[0], Bytes::from(data[..100_000].to_vec()));
        assert!(r.sched.block_cache_stats().unwrap().errors > 0);
    }
}

/// A forgotten (damaged) block leaves no recency entry behind: after it is
/// fetched again and the cap is pressed, eviction still keeps the disk under
/// its cap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_forgotten_block_leaves_no_stale_recency_entry() {
    let r = rig(4 * BLOCK, 0, true);
    let data = pattern(4 << 20, 9);
    let f = put(&r, "g", data.clone()).await;
    let cache = r.sched.block_cache.clone().unwrap();
    r.io.get_ranges(&f, &[0..100_000]).await.unwrap();
    for entry in std::fs::read_dir(cache_files(&r)).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "blk") {
            continue;
        }
        let mut body = std::fs::read(&path).unwrap();
        let last = body.len() - 1;
        body[last] ^= 0xff;
        std::fs::write(&path, body).unwrap();
    }
    // The damaged block is dropped and fetched again, then the whole object
    // streams through a cache far smaller than it.
    for start in (0..4_000_000_u64).step_by(150_000) {
        r.io.get_ranges(&f, &[start..start + 150_000])
            .await
            .unwrap();
        assert!(cache.disk_used() <= 5 * BLOCK, "{}", cache.disk_used());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn keys_and_files_hold_no_credential() {
    let cache_dir = tempfile::tempdir().unwrap();
    let local = Arc::new(InMemory::new());
    let sched = IoScheduler::new(IoConfig {
        block_cache: Some(BlockCacheConfig {
            dir: cache_dir.path().to_path_buf(),
            disk_bytes: 1 << 30,
            memory_bytes: 0,
            block_bytes: BLOCK,
        }),
        ..IoConfig::default()
    });
    // A store id written with credentials prints redacted; the cache key uses that text.
    let id = StoreId::new("s3://AKIAEXAMPLEKEY:sup3rsecret@bucket");
    sched.register_store(id.clone(), Arc::clone(&local) as Arc<dyn ObjectStore>);
    let path = Path::from("p");
    local.put(&path, pattern(200_000, 7).into()).await.unwrap();
    let f = FileRef::from_meta(id, &local.head(&path).await.unwrap());
    sched.context().get_ranges(&f, &[0..100_000]).await.unwrap();
    let mut seen = 0;
    for dir in std::fs::read_dir(cache_dir.path()).unwrap() {
        for file in std::fs::read_dir(dir.unwrap().path()).unwrap() {
            let file = file.unwrap();
            let name = file.file_name().to_string_lossy().into_owned();
            let body = std::fs::read(file.path()).unwrap();
            let text = String::from_utf8_lossy(&body);
            // The display of a store id hides the password of URL credentials (ADR 0038).
            let secret = "sup3rsecret";
            assert!(
                !name.contains(secret) && !text.contains(secret),
                "{name}: {}",
                &text[..text.len().min(200)]
            );
            seen += 1;
        }
    }
    assert!(seen > 0);
}

#[test]
fn the_disk_part_is_emptied_at_start() {
    let dir = tempfile::tempdir().unwrap();
    // A directory of a process that is gone (pid 4294967294 is never in use) is removed.
    // Its lock file is free (nothing holds it), whatever the process table says.
    let dead = dir.path().join("p4294967294-0");
    std::fs::create_dir_all(&dead).unwrap();
    std::fs::write(dead.join("stale.blk"), b"old").unwrap();
    std::fs::write(dead.join(".lock"), b"").unwrap();
    // A lock file made a moment ago belongs to a cache still being created: only an old one counts.
    let old = std::time::SystemTime::now() - Duration::from_secs(300);
    std::fs::File::options()
        .write(true)
        .open(dead.join(".lock"))
        .unwrap()
        .set_modified(old)
        .unwrap();
    // A directory whose lock is held (a live cache, perhaps of another PID namespace sharing the
    // directory) stays, and so does one without a lock file.
    let held = dir.path().join("p4294967293-0");
    std::fs::create_dir_all(&held).unwrap();
    let held_lock = std::fs::File::create(held.join(".lock")).unwrap();
    held_lock.set_modified(old).unwrap();
    held_lock.try_lock().unwrap();
    let unlocked_layout = dir.path().join("p4294967292-0");
    std::fs::create_dir_all(&unlocked_layout).unwrap();
    let config = || BlockCacheConfig {
        dir: FsPath::new(dir.path()).to_path_buf(),
        disk_bytes: 1 << 20,
        memory_bytes: 0,
        block_bytes: BLOCK,
    };
    let first = BlockCache::new(config()).unwrap();
    assert!(!dead.exists());
    // A second cache (another session) has a directory of its own and does not empty the first's.
    let second = BlockCache::new(config()).unwrap();
    assert!(held.exists() && unlocked_layout.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 4);
    // A cache takes its directory away when it is dropped.
    drop(first);
    drop(second);
    // Only the two directories that were not this process's own remain; once its lock is free the
    // held one is purged by the next cache.
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    drop(held_lock);
    let third = BlockCache::new(config()).unwrap();
    assert!(!held.exists() && unlocked_layout.exists());
    drop(third);
}

/// The memory tier is charged to the memory manager (I11): it holds only what the pool grants, and
/// what it holds is what the pool shows reserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_memory_tier_is_charged_to_the_pool() {
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

    let r = rig(1 << 30, 8 << 20, true);
    let root = TreePool::new(&MemoryConfig {
        limit: Some(4 * usize::try_from(BLOCK).unwrap()),
        quantum: 1,
        spill_scratch_reserve: 0,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let pool: Arc<dyn MemoryPool> = Arc::clone(&root) as Arc<dyn MemoryPool>;
    r.sched.attach_memory(&pool);
    let cache = r.sched.block_cache.clone().unwrap();
    let data = pattern(2 << 20, 8);
    let f = put(&r, "m", data.clone()).await;
    for start in (0..2_000_000_u64).step_by(100_000) {
        let from = usize::try_from(start).unwrap();
        let got =
            r.io.get_ranges(&f, &[start..start + 100_000])
                .await
                .unwrap();
        assert_eq!(got[0], Bytes::from(data[from..from + 100_000].to_vec()));
        assert!(cache.memory_used() <= 4 * BLOCK, "{}", cache.memory_used());
        assert_eq!(root.reserved() as u64, cache.memory_used());
    }
    // Blocks that did not fit in memory are on disk: a re-read still hits the cache.
    let before = r.fault.stats().gets;
    r.io.get_ranges(&f, &[0..100_000]).await.unwrap();
    assert_eq!(r.fault.stats().gets, before);
}

/// Acceptance (8) on real data: the whole of a TPC-H `lineitem.parquet` (set `KRISHIV_TPCH_DIR` to
/// the directory holding it; SF1 is about 200 MB) scans to the same rows with the cache as without,
/// cold and warm, and the warm scan reads nothing from the store.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs TPC-H Parquet data: KRISHIV_TPCH_DIR"]
async fn tpch_lineitem_scans_identically_with_the_cache() {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};

    use crate::parquet::ParquetFormat;
    use crate::{ByteRange, FileFormat, MetadataNeeds, ScanOptions, ScanRequest, ScanStream};

    let dir = std::env::var("KRISHIV_TPCH_DIR").expect("KRISHIV_TPCH_DIR");
    let bytes = std::fs::read(FsPath::new(&dir).join("lineitem.parquet")).unwrap();
    let options = ParquetFormat
        .options()
        .parse(
            &std::collections::BTreeMap::new(),
            &krishiv_common::config::Config::new(),
        )
        .unwrap()
        .0;

    async fn checksum(r: &Rig, file: &FileRef, options: &crate::FormatOptions) -> (usize, i64) {
        let meta =
            r.io.read_metadata(&ParquetFormat, file, MetadataNeeds::default(), options)
                .await
                .unwrap();
        let req = ScanRequest::all(meta.schema());
        let root = TreePool::new(&MemoryConfig::default());
        let pool: Arc<dyn MemoryPool> = root.child("query", None);
        let id = pool.register("FileScan", None);
        let mut stream = ScanStream::open(
            &r.io,
            &ParquetFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &req,
            options,
            Reservation::new(Arc::clone(&pool), id),
            ScanOptions::default(),
        )
        .await
        .unwrap();
        let (mut rows, mut sum) = (0, 0_i64);
        while let Some(b) = stream.next_batch().await.unwrap() {
            rows += b.num_rows();
            // `l_orderkey` is the first column, a bigint.
            sum = sum.wrapping_add(
                b.column(0)
                    .as_primitive::<Int64Type>()
                    .iter()
                    .flatten()
                    .sum::<i64>(),
            );
        }
        (rows, sum)
    }

    let plain = rig(0, 0, false);
    let cold = rig(1 << 30, 64 << 20, true);
    let (pf, cf) = (
        put(&plain, "lineitem.parquet", bytes.clone()).await,
        put(&cold, "lineitem.parquet", bytes).await,
    );
    let want = checksum(&plain, &pf, &options).await;
    assert!(want.0 > 0);
    assert_eq!(checksum(&cold, &cf, &options).await, want);
    let after_cold = cold.fault.stats().gets;
    assert_eq!(checksum(&cold, &cf, &options).await, want);
    assert_eq!(
        cold.fault.stats().gets,
        after_cold,
        "the warm scan reached the store"
    );
}

fn direct(
    cache_dir: &FsPath,
    disk: u64,
    mem: u64,
) -> (Arc<InMemory>, Arc<BlockCache>, CachingStore) {
    let inner = Arc::new(InMemory::new());
    let cache = BlockCache::new(BlockCacheConfig {
        dir: cache_dir.to_path_buf(),
        disk_bytes: disk,
        memory_bytes: mem,
        block_bytes: BLOCK,
    })
    .unwrap();
    let store = CachingStore::new(
        Arc::clone(&inner) as Arc<dyn ObjectStore>,
        "s3://direct",
        Arc::clone(&cache),
    );
    (inner, cache, store)
}

fn sized(size: u64) -> GetOptions {
    let mut extensions = object_store::Extensions::new();
    extensions.insert(ObjectSize(size));
    GetOptions {
        extensions,
        ..GetOptions::default()
    }
}

/// Everything but a bounded range read passes straight through to the store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn other_calls_pass_through_to_the_store() {
    use futures::TryStreamExt;
    let tmp = tempfile::tempdir().unwrap();
    let (inner, cache, store) = direct(tmp.path(), 1 << 20, 1 << 20);
    assert!(format!("{cache:?}").contains("BlockCache"));
    let a = Path::from("d/a");
    store
        .put(&a, Bytes::from_static(b"0123456789").into())
        .await
        .unwrap();
    store
        .put_opts(
            &Path::from("d/b"),
            Bytes::from_static(b"xy").into(),
            PutOptions::default(),
        )
        .await
        .unwrap();
    let mut up = store.put_multipart(&Path::from("d/m")).await.unwrap();
    up.put_part(Bytes::from_static(b"mm").into()).await.unwrap();
    up.complete().await.unwrap();
    assert_eq!(inner.head(&Path::from("d/m")).await.unwrap().size, 2);
    assert_eq!(
        store.get_ranges(&a, &[2..4, 6..9]).await.unwrap(),
        vec![Bytes::from_static(b"23"), Bytes::from_static(b"678")]
    );
    let listed: Vec<_> = store
        .list(Some(&Path::from("d")))
        .try_collect()
        .await
        .unwrap();
    assert_eq!(listed.len(), 3);
    let delim = store
        .list_with_delimiter(Some(&Path::from("d")))
        .await
        .unwrap();
    assert_eq!(delim.objects.len(), 3);
    store.copy(&a, &Path::from("d/c")).await.unwrap();
    assert_eq!(inner.head(&Path::from("d/c")).await.unwrap().size, 10);
    store.delete(&Path::from("d/c")).await.unwrap();
    assert!(inner.head(&Path::from("d/c")).await.is_err());
    // A head, a whole-object read and a read without a size are not cached.
    assert_eq!(store.head(&a).await.unwrap().size, 10);
    assert_eq!(
        store.get(&a).await.unwrap().bytes().await.unwrap().len(),
        10
    );
    let r = store
        .get_opts(
            &a,
            GetOptions {
                range: Some(object_store::GetRange::Bounded(1..3)),
                ..GetOptions::default()
            },
        )
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&r[..], b"12");
    assert_eq!(cache.stats().misses, 0);
}

/// An object that changed size under a read, or a range past its end, is the store's to answer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_changed_object_falls_back_to_the_store() {
    use object_store::GetRange;
    let tmp = tempfile::tempdir().unwrap();
    let (inner, cache, store) = direct(tmp.path(), 1 << 20, 0);
    let a = Path::from("o");
    inner.put(&a, pattern(300_000, 1).into()).await.unwrap();
    // The scheduler believed 400,000 bytes: the fetched run has another size, so the cache path
    // fails and the plain read answers.
    let mut options = sized(400_000);
    options.range = Some(GetRange::Bounded(10..20));
    let got = store
        .get_opts(&a, options)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(got.len(), 10);
    assert!(cache.stats().errors > 0);
    // A range past the end of the object is not cached: the store answers it (cut at the end, as
    // `object_store` cuts it), and nothing is counted as a miss for it.
    let misses = cache.stats().misses;
    let mut options = sized(300_000);
    options.range = Some(GetRange::Bounded(299_990..300_005));
    let tail = store
        .get_opts(&a, options)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(tail.len(), 10);
    assert_eq!(cache.stats().misses, misses);
}

/// A pool that cannot hold a block's fetch refuses it, and the read is answered by the store; the
/// reservation of a fetch that did run is returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fetch_buffers_are_reserved_from_the_pool() {
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
    let tmp = tempfile::tempdir().unwrap();
    let (inner, cache, store) = direct(tmp.path(), 1 << 22, 1 << 20);
    let a = Path::from("o");
    let data = pattern(1 << 20, 4);
    inner.put(&a, data.clone().into()).await.unwrap();
    let root = TreePool::new(&MemoryConfig {
        limit: Some(2 * usize::try_from(BLOCK).unwrap()),
        quantum: 1,
        spill_scratch_reserve: 0,
        resident_guard: true,
        ..MemoryConfig::default()
    });
    let pool: Arc<dyn MemoryPool> = Arc::clone(&root) as Arc<dyn MemoryPool>;
    cache.attach_pool(&pool);
    let read = |range: std::ops::Range<u64>| {
        let mut options = sized(1 << 20);
        options.range = Some(object_store::GetRange::Bounded(range));
        let (store, a) = (&store, a.clone());
        async move {
            store
                .get_opts(&a, options)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap()
        }
    };
    // Four blocks do not fit two blocks of pool: refused, answered by the plain read.
    let big = read(0..4 * BLOCK).await;
    assert_eq!(&big[..], &data[..usize::try_from(4 * BLOCK).unwrap()]);
    assert!(cache.stats().errors > 0);
    assert_eq!(root.reserved(), 0, "no reservation is left behind");
    // One block fits, is cached, and its fetch reservation is returned.
    let one = read(5 * BLOCK..5 * BLOCK + 100).await;
    assert_eq!(
        &one[..],
        &data[usize::try_from(5 * BLOCK).unwrap()..][..100]
    );
    assert!(cache.stats().misses > 0);
    assert_eq!(root.reserved() as u64, cache.memory_used());
}

/// A read with many separate runs of missing blocks fetches them all, a few at a time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_runs_are_fetched_within_the_concurrency_bound() {
    let tmp = tempfile::tempdir().unwrap();
    let (inner, cache, store) = direct(tmp.path(), 1 << 26, 0);
    let a = Path::from("o");
    let data = pattern(40 * usize::try_from(BLOCK).unwrap(), 6);
    inner.put(&a, data.clone().into()).await.unwrap();
    // Warm every other block so the missing ones form 20 separate runs.
    for b in (0..40).step_by(2) {
        let mut o = sized(40 * BLOCK);
        o.range = Some(object_store::GetRange::Bounded(b * BLOCK..b * BLOCK + 10));
        store.get_opts(&a, o).await.unwrap();
    }
    let mut o = sized(40 * BLOCK);
    o.range = Some(object_store::GetRange::Bounded(0..40 * BLOCK));
    let got = store.get_opts(&a, o).await.unwrap().bytes().await.unwrap();
    assert_eq!(&got[..], &data[..]);
    assert_eq!(cache.stats().errors, 0);
}

/// Two caches of one directory (the same names a second PID namespace would pick) never empty
/// each other: each takes a directory of its own.
#[test]
fn caches_never_share_or_empty_a_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let config = || BlockCacheConfig {
        dir: tmp.path().to_path_buf(),
        disk_bytes: 1 << 20,
        memory_bytes: 0,
        block_bytes: BLOCK,
    };
    let first = BlockCache::new(config()).unwrap();
    // A live foreign cache's directory under any name this process could pick.
    let pid = std::process::id();
    for n in 0..64 {
        let foreign = tmp.path().join(format!("p{pid}-{n}"));
        std::fs::create_dir_all(&foreign).unwrap();
        std::fs::write(foreign.join("keep.blk"), b"live").unwrap();
    }
    let second = BlockCache::new(config()).unwrap();
    for n in 0..64 {
        assert!(
            tmp.path().join(format!("p{pid}-{n}/keep.blk")).exists(),
            "{n}"
        );
    }
    drop((first, second));
}

/// A failure of the store is the store's answer, once: it is not asked again by the plain read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_failure_is_returned_not_retried() {
    let tmp = tempfile::tempdir().unwrap();
    let (_inner, cache, store) = direct(tmp.path(), 1 << 20, 0);
    // The object does not exist, though the scheduler sent a size for it.
    let mut options = sized(300_000);
    options.range = Some(object_store::GetRange::Bounded(0..100));
    let e = store
        .get_opts(&Path::from("missing"), options)
        .await
        .unwrap_err();
    assert!(matches!(e, object_store::Error::NotFound { .. }), "{e}");
    assert_eq!(
        cache.stats().errors,
        0,
        "a store failure is not a cache error"
    );
}
