// SPDX-License-Identifier: Apache-2.0
//! Scheduler, cache and upload tests over an in-memory store with faults.

#![expect(
    clippy::single_range_in_vec_init,
    reason = "single-range requests are the cases under test"
)]

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use krishiv_common::config::Config;
use krishiv_common::error::ErrorClass;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};

use super::*;
use crate::format::ReadAt;

struct Setup {
    sched: Arc<IoScheduler>,
    io: IoContext,
    fault: Arc<FaultStore>,
    inner: Arc<InMemory>,
    id: StoreId,
}

fn setup(config: IoConfig) -> Setup {
    let inner = Arc::new(InMemory::new());
    let fault = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let sched = IoScheduler::new(config);
    let id = StoreId::new("memory://t");
    sched.register_store(id.clone(), Arc::clone(&fault) as Arc<dyn ObjectStore>);
    Setup {
        io: sched.context(),
        sched,
        fault,
        inner,
        id,
    }
}

async fn put(s: &Setup, name: &str, len: usize) -> FileRef {
    let data: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
    let path = Path::from(name);
    s.inner.put(&path, data.into()).await.unwrap();
    FileRef::from_meta(s.id.clone(), &s.inner.head(&path).await.unwrap())
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap()
}

fn count(c: &AtomicU64) -> u64 {
    IoCounters::get(c)
}

#[test]
fn ranges_are_coalesced_checked_and_counted() {
    rt().block_on(async {
        let s = setup(IoConfig::default());
        let f = put(&s, "a", 10_000).await;
        let got =
            s.io.get_ranges(&f, &[10..20, 0..5, 9_000..9_010, 30..30])
                .await
                .unwrap();
        assert_eq!(got[0], Bytes::from((10_u8..20).collect::<Vec<_>>()));
        assert_eq!(got[1].len(), 5);
        assert!(got[3].is_empty());
        // One coalesced request (gap < 1 MiB).
        assert_eq!(s.fault.stats().gets, 1);
        assert_eq!(count(&s.io.counters().requests), 1);
        assert_eq!(count(&s.io.counters().bytes_read), 9_010);
        assert_eq!(count(&s.sched.totals().requests), 1);
        // Only empty ranges: no request at all.
        let empty = s.io.get_ranges(&f, &[5..5, 0..0]).await.unwrap();
        assert!(empty.iter().all(Bytes::is_empty));
        assert_eq!(s.fault.stats().gets, 1);
        let e = s.io.get_ranges(&f, &[0..10_001]).await.unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
        assert_eq!(s.fault.stats().unscheduled, 0);
        // A tail read is one request, clamped to the file.
        let t = s.io.get_tail(&f, 1).await.unwrap();
        let (start, tail) = (t.start, t.bytes.clone());
        assert_eq!((start, tail.len()), (0, 10_000));
        let big = put(&s, "big", 3_000_000).await;
        let t = s.io.get_tail(&big, 5 << 20).await.unwrap();
        let (start, tail) = (t.start, t.bytes.clone());
        assert_eq!(
            (start, tail.len() as u64),
            (3_000_000 - MAX_TAIL_BYTES, MAX_TAIL_BYTES)
        );
        assert_eq!(count(&s.io.counters().tail_reads), 2);
        assert_eq!(s.fault.stats().tail_gets, 2);
        let r = s.io.reader(&f);
        assert_eq!(r.size(), 10_000);
        assert_eq!(r.read_at(1..3).await.unwrap().len(), 2);
        assert!(s.io.reader(&big).read_at(0..4_000_000).await.is_err());
    });
}

#[test]
fn gappy_coalesced_ranges_are_copied_out() {
    rt().block_on(async {
        let s = setup(IoConfig::default());
        let f = put(&s, "a", 100_000).await;
        // Adjacent ranges: one request, sliced (no copy).
        let got =
            s.io.get_ranges(&f, &[0..1_000, 1_000..2_000])
                .await
                .unwrap();
        assert_eq!(got[0].as_ptr().wrapping_add(1_000), got[1].as_ptr());
        // One request with 97% gap bytes (a small gap, so still merged):
        // each range is its own copy, so the request's buffer is not kept
        // alive by them.
        let got =
            s.io.get_ranges(&f, &[0..1_000, 60_000..61_000])
                .await
                .unwrap();
        assert_eq!(s.fault.stats().gets, 2);
        assert_ne!(got[0].as_ptr().wrapping_add(60_000), got[1].as_ptr());
        let expected = |r: std::ops::Range<usize>| -> Vec<u8> {
            r.map(|i| u8::try_from(i % 251).unwrap()).collect()
        };
        assert_eq!(got[0].as_ref(), expected(0..1_000).as_slice());
        assert_eq!(got[1].as_ref(), expected(60_000..61_000).as_slice());
        // A large gap is not read at all: two requests.
        let got =
            s.io.get_ranges(&f, &[0..1_000, 99_000..100_000])
                .await
                .unwrap();
        assert_eq!(s.fault.stats().gets, 4);
        assert_eq!(got[1].as_ref(), expected(99_000..100_000).as_slice());
    });
}

#[test]
fn coalescing_waste_depends_on_the_store() {
    assert!(!StoreId::new("file").is_remote());
    assert!(!StoreId::new("file:///tmp").is_remote());
    for remote in [
        "s3://bucket",
        "S3A://b",
        "gs://b",
        "az://c",
        "abfss://c@a",
        "https://h",
    ] {
        assert!(StoreId::new(remote).is_remote(), "{remote}");
    }
    let c = IoConfig::default();
    assert_eq!(c.waste(&StoreId::new("file")), Waste::LOCAL);
    assert_eq!(c.waste(&StoreId::new("s3://bucket")), Waste::REMOTE);
    let (set, _) = Config::from_pairs([
        ("krishiv.io.coalesceWasteBytes", "0"),
        ("krishiv.io.coalesceWasteDivisor", "4"),
    ])
    .unwrap();
    let c = IoConfig::from_config(&set);
    let w = Waste {
        floor: 0,
        divisor: 4,
    };
    assert_eq!(c.waste(&StoreId::new("file")), w);
    assert_eq!(c.waste(&StoreId::new("s3://bucket")), w);
}

#[test]
fn transient_failures_are_retried_with_backoff() {
    rt().block_on(async {
        let s = setup(IoConfig::default());
        let f = put(&s, "a", 100).await;
        s.fault.fail_next_gets(3, Some(Duration::from_secs(2)));
        let start = tokio::time::Instant::now();
        s.io.get_ranges(&f, &[0..10]).await.unwrap();
        assert!(
            start.elapsed() >= Duration::from_secs(6),
            "Retry-After honoured"
        );
        assert_eq!(count(&s.io.counters().retries), 3);
        s.fault.fail_next_gets(100, None);
        let e = s.io.get_ranges(&f, &[0..10]).await.unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivObjectStoreRequestFailed);
        assert_eq!(count(&s.io.counters().retries), 3 + 5);
    });
}

#[test]
fn timeouts_are_retried_then_fail() {
    rt().block_on(async {
        let config = IoConfig {
            request_timeout: Duration::from_millis(10),
            max_retries: 1,
            ..IoConfig::default()
        };
        let s = setup(config);
        let f = put(&s, "a", 100).await;
        s.fault.set_latency(Duration::from_millis(50));
        let e = s.io.get_ranges(&f, &[0..10]).await.unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivObjectStoreRequestFailed);
        assert_eq!(count(&s.io.counters().retries), 1);
    });
}

#[test]
fn a_store_that_lists_whole_has_no_listing_timeout() {
    rt().block_on(async {
        let config = IoConfig {
            request_timeout: Duration::from_millis(10),
            max_retries: 0,
            ..IoConfig::default()
        };
        let s = setup(config);
        put(&s, "d/a", 1).await;
        s.fault.set_list_latency(Duration::from_millis(50));
        let dir = Path::from("d");
        let e = s.io.list(&s.id, &dir).await.unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivObjectStoreRequestFailed);
        s.sched.register_whole_listing_store(
            s.id.clone(),
            Arc::clone(&s.fault) as Arc<dyn ObjectStore>,
        );
        assert_eq!(s.io.list(&s.id, &dir).await.unwrap().len(), 1);
    });
}

#[test]
fn versions_are_checked_on_every_read() {
    rt().block_on(async {
        let s = setup(IoConfig::default());
        let f = put(&s, "a", 100).await;
        assert!(matches!(f.version, VersionToken::ETag(_)));
        put(&s, "a", 100).await;
        let e = s.io.get_ranges(&f, &[0..10]).await.unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivFileChangedDuringRead);
        assert!(e.to_string().contains("memory://t/a"));
        // Modification time instead of an ETag.
        let fresh = put(&s, "b", 100).await;
        let by_time = FileRef {
            version: VersionToken::Modified(fresh.modified_micros.unwrap()),
            ..fresh.clone()
        };
        s.io.get_ranges(&by_time, &[0..10]).await.unwrap();
        let old = FileRef {
            version: VersionToken::Modified(0),
            ..fresh.clone()
        };
        assert_eq!(
            s.io.get_ranges(&old, &[0..1])
                .await
                .unwrap_err()
                .error_class(),
            ErrorClass::KrishivFileChangedDuringRead
        );
        // Immutable files are not version-checked, but a size change is caught.
        let imm = FileRef {
            version: VersionToken::Immutable,
            size: 99,
            ..fresh.clone()
        };
        assert_eq!(
            s.io.get_ranges(&imm, &[0..1])
                .await
                .unwrap_err()
                .error_class(),
            ErrorClass::KrishivFileChangedDuringRead
        );
        let gone = FileRef {
            path: Path::from("nope"),
            ..fresh
        };
        assert_eq!(
            s.io.get_ranges(&gone, &[0..1])
                .await
                .unwrap_err()
                .error_class(),
            ErrorClass::FailedReadFileFileNotExist
        );
        let unknown = FileRef {
            store: StoreId::new("s3://other"),
            ..f
        };
        assert_eq!(
            s.io.get_ranges(&unknown, &[0..1])
                .await
                .unwrap_err()
                .error_class(),
            ErrorClass::InternalError
        );
    });
}

#[test]
fn fetch_runs_on_the_io_runtime_and_aborts_when_dropped() {
    let io_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let inner = Arc::new(InMemory::new());
    let fault = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let sched = IoScheduler::with_runtime(IoConfig::default(), io_rt.handle().clone());
    let id = StoreId::new("memory://t");
    sched.register_store(id.clone(), Arc::clone(&fault) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    caller.block_on(async {
        let path = Path::from("x");
        inner.put(&path, vec![7_u8; 64].into()).await.unwrap();
        let f = FileRef::from_meta(id, &inner.head(&path).await.unwrap());
        let got = io.fetch(f.clone(), vec![0..8]).await.unwrap();
        assert_eq!(got[0].len(), 8);
        fault.set_latency(Duration::from_secs(60));
        let pending = io.fetch(f, vec![0..8]);
        drop(pending);
    });
    assert!(format!("{sched:?}").contains("IoScheduler"));
}

#[test]
fn uploads_pick_put_or_multipart_and_abort() {
    rt().block_on(async {
        let config = IoConfig {
            multipart_part_bytes: 1,
            ..IoConfig::default()
        };
        let s = setup(config);
        // Small: one PUT.
        let mut u = s.io.upload(&s.id, Path::from("small")).unwrap();
        u.write(Bytes::from_static(b"hello")).await.unwrap();
        assert_eq!(u.written(), 5);
        assert!(!u.is_full());
        assert_eq!(u.finish().await.unwrap(), 5);
        assert_eq!(s.fault.stats().puts, 1);
        let small = s.inner.get(&Path::from("small")).await.unwrap();
        assert_eq!(small.bytes().await.unwrap(), "hello");
        // Large: parts of at least 5 MiB, then completion.
        let mut u = s.io.upload(&s.id, Path::from("large")).unwrap();
        assert_eq!(u.part_size(), upload::MIN_PART_BYTES);
        for _ in 0..13 {
            u.write(Bytes::from(vec![1_u8; 1 << 20])).await.unwrap();
        }
        assert_eq!(u.buffered(), 3 << 20);
        u.wait_capacity().await.unwrap();
        assert!(format!("{u:?}").contains("large"));
        assert_eq!(u.finish().await.unwrap(), 13 << 20);
        let st = s.fault.stats();
        assert_eq!((st.uploads_started, st.uploads_completed), (1, 1));
        let len = s.inner.head(&Path::from("large")).await.unwrap().size;
        assert_eq!(len, 13 << 20);
        // Explicit abort and drop both abort.
        let mut u = s.io.upload(&s.id, Path::from("aborted")).unwrap();
        u.write(Bytes::from(vec![1_u8; 6 << 20])).await.unwrap();
        u.abort().await;
        let mut u = s.io.upload(&s.id, Path::from("dropped")).unwrap();
        u.write(Bytes::from(vec![1_u8; 6 << 20])).await.unwrap();
        drop(u);
        tokio::task::yield_now().await;
        let st = s.fault.stats();
        assert_eq!(st.uploads_aborted, 2);
        assert_eq!(count(&s.io.counters().uploads_aborted), 2);
        assert_eq!(count(&s.sched.totals().uploads_aborted), 2);
        assert_eq!(count(&s.sched.totals().uploads_completed), 1);
        assert!(s.inner.head(&Path::from("aborted")).await.is_err());
        // A failed part fails the upload and aborts it.
        let mut u = s.io.upload(&s.id, Path::from("failed")).unwrap();
        s.fault.fail_next_puts(1);
        let r = async {
            u.write(Bytes::from(vec![1_u8; 6 << 20])).await?;
            u.finish().await
        }
        .await;
        assert_eq!(r.unwrap_err().error_class(), ErrorClass::TaskWriteFailed);
        // A transient PUT is retried.
        s.fault.fail_next_puts(1);
        let mut u = s.io.upload(&s.id, Path::from("retried")).unwrap();
        u.write(Bytes::from_static(b"x")).await.unwrap();
        u.finish().await.unwrap();
        s.io.delete(&s.id, &Path::from("retried")).await.unwrap();
        s.io.delete(&s.id, &Path::from("never-existed"))
            .await
            .unwrap();
        assert!(s.inner.head(&Path::from("retried")).await.is_err());
        assert!(s.io.upload(&StoreId::new("x://"), Path::from("p")).is_err());
        assert_eq!(s.fault.stats().unscheduled, 0);
    });
}

#[test]
fn upload_parts_stay_held_until_acknowledged_and_share_the_request_cap() {
    let run = |cap: usize| {
        rt().block_on(async {
            let s = setup(IoConfig {
                max_concurrent_requests: cap,
                multipart_part_bytes: 1,
                ..IoConfig::default()
            });
            s.fault.set_latency(Duration::from_millis(100));
            let mut u = s.io.upload(&s.id, Path::from("held")).unwrap();
            u.write(Bytes::from(vec![1_u8; 11 << 20])).await.unwrap();
            // Two 5 MiB parts sent, 1 MiB buffered: all of it still in memory.
            assert_eq!(u.in_flight_bytes(), 10 << 20);
            assert_eq!(u.held_bytes(), 11 << 20);
            let start = tokio::time::Instant::now();
            u.wait_idle().await.unwrap();
            let took = start.elapsed();
            assert_eq!((u.in_flight_bytes(), u.held_bytes()), (0, 1 << 20));
            assert_eq!(u.finish().await.unwrap(), 11 << 20);
            took
        })
    };
    // With one request in flight per store, the two parts run one after
    // the other; with more, together.
    assert!(run(1) >= Duration::from_millis(200));
    assert!(run(4) < Duration::from_millis(200));
}

#[derive(Debug)]
struct Meta(usize, crate::format::MetadataNeeds);

impl crate::format::FileMetadata for Meta {
    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(arrow_schema::Schema::empty())
    }
    fn statistics(&self) -> crate::stats::FileStatistics {
        crate::stats::FileStatistics::unknown()
    }
    fn heap_size(&self) -> usize {
        self.0
    }
    fn needs(&self) -> crate::format::MetadataNeeds {
        self.1
    }
}

#[test]
fn the_metadata_cache_is_a_byte_bounded_lru() {
    let f = |p: &str| FileRef {
        store: StoreId::new("s"),
        path: Path::from(p),
        size: 1,
        version: VersionToken::Immutable,
        modified_micros: None,
    };
    let none = crate::format::MetadataNeeds::default();
    let pages = crate::format::MetadataNeeds {
        page_index: true,
        bloom: false,
    };
    let c = MetadataCache::new(100);
    assert!(c.is_empty());
    c.insert(CacheKey::new("m", &f("a"), None), Arc::new(Meta(40, none)));
    c.insert(CacheKey::new("m", &f("b"), None), Arc::new(Meta(40, pages)));
    assert!(c.get(&CacheKey::new("m", &f("a"), None), none).is_ok());
    // "a" is now more recent than "b": "b" is evicted.
    c.insert(CacheKey::new("m", &f("c"), None), Arc::new(Meta(40, none)));
    assert!(c.get(&CacheKey::new("m", &f("b"), None), none).is_err());
    assert_eq!((c.len(), c.bytes()), (2, 80));
    // Needs not covered: reported with what was loaded.
    assert!(matches!(
        c.get(&CacheKey::new("m", &f("a"), None), pages),
        Err(Some(n)) if n == none
    ));
    // Another key fingerprint or format never hits.
    assert!(
        c.get(&CacheKey::new("m", &f("a"), Some([1; 16])), none)
            .is_err()
    );
    assert!(c.get(&CacheKey::new("x", &f("a"), None), none).is_err());
    // Too large to cache; replacing an entry.
    c.insert(CacheKey::new("m", &f("d"), None), Arc::new(Meta(101, none)));
    c.insert(CacheKey::new("m", &f("a"), None), Arc::new(Meta(10, none)));
    assert_eq!(c.bytes(), 50);
    c.invalidate_path(&StoreId::new("s"), &Path::from("a"));
    assert_eq!(c.len(), 1);
    assert!(format!("{c:?}").contains("entries"));
    assert!(format!("{:?}", CacheKey::new("m", &f("a"), Some([9; 16]))).contains("keyed: true"));
    c.clear();
    assert!(c.is_empty());
}

#[test]
fn identities_print_without_credentials() {
    let id = StoreId::new("s3://user:pa55word@bucket");
    assert!(!id.to_string().contains("pa55word"));
    assert!(!format!("{id:?}").contains("pa55word"));
    let f = FileRef {
        store: id,
        path: Path::from("a/b"),
        size: 1,
        version: VersionToken::ETag(Arc::from("e1")),
        modified_micros: None,
    };
    assert!(f.display_path().ends_with("@bucket/a/b"));
    assert_eq!(f.version.to_string(), "ETag e1");
    assert_eq!(VersionToken::Modified(5).to_string(), "modified at 5 us");
    assert_eq!(VersionToken::Immutable.to_string(), "immutable");
    let r = ByteRange::new(10, 20);
    assert!(r.owns(&(0..30)) && !r.owns(&(0..10)) && r.owns(&(19..20)));
    assert_eq!(
        (ByteRange::whole(5).len(), ByteRange::new(5, 5).is_empty()),
        (5, true)
    );
}

#[test]
fn config_keys_and_backoff() {
    let (c, _) = Config::from_pairs([
        ("krishiv.io.maxConcurrentRequests", "8"),
        ("krishiv.io.coalesceBytes", "4k"),
        ("krishiv.io.maxRequestBytes", "2m"),
        ("krishiv.io.footerReadBytes", "16k"),
        ("krishiv.io.metadataCacheBytes", "1m"),
        ("krishiv.io.multipartPartBytes", "16m"),
        ("krishiv.io.maxRetries", "2"),
        ("krishiv.io.requestTimeout", "5s"),
    ])
    .unwrap();
    let cfg = IoConfig::from_config(&c);
    assert_eq!(cfg.max_concurrent_requests, 8);
    assert_eq!(
        (cfg.coalesce_bytes, cfg.max_request_bytes),
        (4 << 10, 2 << 20)
    );
    assert_eq!(
        (cfg.footer_read_bytes, cfg.metadata_cache_bytes),
        (16 << 10, 1 << 20)
    );
    assert_eq!((cfg.multipart_part_bytes, cfg.max_retries), (16 << 20, 2));
    assert_eq!(cfg.request_timeout, Duration::from_secs(5));
    let s = IoScheduler::new(cfg);
    for attempt in 0..20 {
        assert!(s.backoff(attempt, None) <= Duration::from_secs(10));
    }
    assert!(s.backoff(0, Some(Duration::from_secs(3))) >= Duration::from_secs(3));
    assert!(s.backoff(0, Some(Duration::from_secs(9999))) <= Duration::from_secs(300));
    assert_eq!(s.config().max_retries, 2);
    assert!(s.cache().is_empty());
}

#[test]
fn io_errors_are_classified() {
    let f = FileRef {
        store: StoreId::new("s"),
        path: Path::from("p"),
        size: 1,
        version: VersionToken::Immutable,
        modified_micros: None,
    };
    let timeout = object_store::Error::Generic {
        store: "x",
        source: Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "t")),
    };
    assert!(matches!(classify(&timeout, &f), Attempt::Retry(None)));
    let other = object_store::Error::Generic {
        store: "x",
        source: Box::new(std::io::Error::other("boom")),
    };
    assert!(matches!(classify(&other, &f), Attempt::Fail(_)));
    let unsupported = object_store::Error::NotImplemented {
        operation: "o".into(),
        implementer: "i".into(),
    };
    assert!(matches!(classify(&unsupported, &f), Attempt::Fail(_)));
    let t = TransientError {
        retry_after: Some(Duration::from_secs(1)),
    };
    assert!(t.to_string().contains("retry after"));
}

#[test]
fn the_fault_store_delegates_and_flags_unscheduled_requests() {
    rt().block_on(async {
        let s = setup(IoConfig::default());
        let p = Path::from("d/x");
        s.fault.put(&p, vec![1_u8; 4].into()).await.unwrap();
        assert_eq!(s.fault.get_ranges(&p, &[0..2]).await.unwrap()[0].len(), 2);
        assert_eq!(s.fault.list(None).count().await, 1);
        assert!(s.fault.list_with_delimiter(None).await.is_ok());
        s.fault.copy(&p, &Path::from("d/y")).await.unwrap();
        assert_eq!(s.fault.stats().unscheduled, 3);
        assert!(s.fault.to_string().starts_with("FaultStore"));
        assert!(format!("{:?}", s.fault).starts_with("FaultStore"));
        let mut mp = s.fault.put_multipart(&Path::from("d/z")).await.unwrap();
        mp.put_part(vec![0_u8; 3].into()).await.unwrap();
        mp.complete().await.unwrap();
        assert!(format!("{:?}", s.io).contains("IoContext"));
        let _ = s.io.config();
        let _ = s.io.scheduler();
        let resolved = s.io.secrets().resolve("a", "b").now_or_never();
        assert!(resolved.is_some_and(|r| r.is_err()));
    });
}

#[test]
fn metadata_reads_and_the_cache_are_reserved() {
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
    rt().block_on(async {
        let root = TreePool::new(&MemoryConfig::default());
        let query = root.child("q", None);
        let s = setup(IoConfig::default());
        let io = s.io.clone().with_memory(Arc::clone(&query));
        let f = put(&s, "m", 200_000).await;
        let tail = io.get_tail(&f, 0).await.unwrap();
        assert!(query.reserved() >= 64 << 10, "tail bytes are reserved");
        drop(tail);
        let held = io
            .get_ranges_held(&f, &[0..1000, 5000..6000])
            .await
            .unwrap();
        assert_eq!(held.len(), 2);
        assert!(query.reserved() >= 2000);
        drop(held);
        assert_eq!(query.reserved(), 0);
        // A tiny pool refuses the tail read with a memory error.
        let tiny = TreePool::new(&MemoryConfig {
            limit: Some(1 << 10),
            ..MemoryConfig::default()
        });
        let e =
            s.io.clone()
                .with_memory(tiny.child("q", None))
                .get_tail(&f, 0)
                .await
                .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::UnableToAcquireMemory);
    });
    // The cache holds memory in its own pool and caches only what it grants.
    let pool = TreePool::new(&MemoryConfig {
        limit: Some(64 << 20),
        ..MemoryConfig::default()
    });
    let pool: Arc<dyn MemoryPool> = pool;
    let f = |p: &str| FileRef {
        store: StoreId::new("s"),
        path: Path::from(p),
        size: 1,
        version: VersionToken::Immutable,
        modified_micros: None,
    };
    let none = crate::format::MetadataNeeds::default();
    let c = MetadataCache::new(10 << 20);
    c.insert(
        CacheKey::new("m", &f("old"), None),
        Arc::new(Meta(10, none)),
    );
    c.attach_pool(&pool);
    assert!(
        c.is_empty(),
        "attaching starts from an empty, accounted cache"
    );
    c.insert(
        CacheKey::new("m", &f("a"), None),
        Arc::new(Meta(3 << 20, none)),
    );
    assert!(pool.reserved() >= 3 << 20);
    c.insert(
        CacheKey::new("m", &f("b"), None),
        Arc::new(Meta(3 << 20, none)),
    );
    c.invalidate_path(&StoreId::new("s"), &Path::from("a"));
    assert_eq!(c.bytes(), 3 << 20);
    c.clear();
    assert_eq!(pool.reserved(), 0);
    // A pool that cannot grant the bytes: the entry is not cached.
    let small: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
        limit: Some(1 << 20),
        ..MemoryConfig::default()
    });
    let c = MetadataCache::new(10 << 20);
    c.attach_pool(&small);
    c.insert(
        CacheKey::new("m", &f("x"), None),
        Arc::new(Meta(2 << 20, none)),
    );
    assert!(c.is_empty());
    let s = setup(IoConfig::default());
    s.sched.attach_memory(&pool);
}

#[test]
fn secrets_are_resolved_once_per_context() {
    use std::sync::atomic::AtomicUsize;

    use futures::future::BoxFuture;
    use krishiv_common::secret::SecretString;

    #[derive(Debug, Default)]
    struct Counting(AtomicUsize);
    impl SecretResolver for Counting {
        fn resolve<'a>(&'a self, _s: &'a str, key: &'a str) -> BoxFuture<'a, Result<SecretString>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Box::pin(async move { Ok(SecretString::new(format!("v-{key}"))) })
        }
    }
    let counting = Arc::new(Counting::default());
    let s = setup(IoConfig::default());
    let io =
        s.io.with_secrets(Arc::clone(&counting) as Arc<dyn SecretResolver>);
    rt().block_on(async {
        for _ in 0..3 {
            assert_eq!(
                io.secrets()
                    .resolve("sc", "k")
                    .await
                    .unwrap()
                    .expose_secret(),
                "v-k"
            );
        }
        io.secrets().resolve("sc", "other").await.unwrap();
    });
    assert_eq!(counting.0.load(Ordering::Relaxed), 2);
}

#[test]
fn a_dropped_upload_aborts_only_after_its_parts_settle() {
    rt().block_on(async {
        let s = setup(IoConfig {
            multipart_part_bytes: 1,
            ..IoConfig::default()
        });
        s.fault.set_latency(Duration::from_millis(100));
        for explicit in [false, true] {
            let mut u =
                s.io.upload(&s.id, Path::from(format!("race{explicit}")))
                    .unwrap();
            u.write(Bytes::from(vec![1_u8; 11 << 20])).await.unwrap();
            assert_eq!(u.in_flight_bytes(), 10 << 20);
            if explicit {
                u.abort().await;
            } else {
                drop(u);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        let st = s.fault.stats();
        assert_eq!(st.uploads_aborted, 2);
        assert_eq!(
            st.parts_after_abort, 0,
            "a part landed after its upload was aborted"
        );
    });
}

/// A coalesced read's gap bytes are reserved while it is in flight, and
/// only the ranges' bytes stay reserved (and held) afterwards.
#[test]
fn reservations_cover_the_whole_coalesced_read() {
    use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
    rt().block_on(async {
        let s = setup(IoConfig::default());
        let f = put(&s, "a", 200_000).await;
        let root = TreePool::new(&MemoryConfig {
            limit: None,
            quantum: 1,
            ..MemoryConfig::default()
        });
        let pool: Arc<dyn MemoryPool> = root.child("q", None);
        let io = s.io.clone().with_memory(Arc::clone(&pool));
        // A 30 KB gap (within the local 64 KiB floor): one request.
        let ranges = [0..10_000, 40_000..50_000];
        assert_eq!(io.request_bytes(&f, &ranges), 50_000);
        let held = io.get_ranges_held(&f, &ranges).await.unwrap();
        assert_eq!(s.fault.stats().gets, 1);
        assert!(pool.peak_reserved() >= 50_000, "{}", pool.peak_reserved());
        assert_eq!(pool.reserved(), 20_000);
        // The ranges were copied out of the request: nothing keeps the gap.
        assert_eq!(held[0].len() + held[1].len(), 20_000);
        drop(held);
        assert_eq!(pool.reserved(), 0);
    });
}
