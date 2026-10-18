// SPDX-License-Identifier: Apache-2.0
//! I/O scheduler behaviour over the mock format, on Tokio's paused clock:
//! single-flight footer reads, footer read-ahead, and prefix pacing.
#![cfg(test)]

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_io::io::FaultStore;
use krishiv_io::{
    FileRef, FormatOptions, IoConfig, IoCounters, IoScheduler, MetadataNeeds, StoreId, StorePolicy,
};
use krishiv_testkit::file_format::{FormatFixture, MockFixture, data};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, true)]))
}

/// A scheduler whose requests run on its own paused runtime (as the
/// engine's I/O runtime), behind a store with injectable latency.
struct Timed {
    rt: tokio::runtime::Runtime,
    store: Arc<FaultStore>,
    inner: Arc<InMemory>,
    sched: Arc<IoScheduler>,
}

const STORE: &str = "memory://io";

impl Timed {
    fn new(policy: StorePolicy) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let inner = Arc::new(InMemory::new());
        let store = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
        let sched = IoScheduler::with_runtime(IoConfig::default(), rt.handle().clone());
        sched.register_store_with(
            StoreId::new(STORE),
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            policy,
        );
        Self {
            rt,
            store,
            inner,
            sched,
        }
    }

    fn options(&self) -> FormatOptions {
        FormatOptions::defaults(MockFixture::default().format().options())
    }

    /// Writes `rows` rows as `name` with the mock format.
    fn file(&self, name: &str, rows: usize) -> FileRef {
        let fx = MockFixture::default();
        let io = self.sched.context();
        let o = self.options();
        let batch: RecordBatch = data::batch(&schema(), rows, 1).unwrap();
        self.rt.block_on(async {
            let mut enc = fx.format().writer(&io, schema(), &o).await.unwrap();
            enc.write(&batch).unwrap();
            let mut up = io.upload(&StoreId::new(STORE), Path::from(name)).unwrap();
            while let Some(b) = enc.take_bytes() {
                up.write(b).await.unwrap();
            }
            up.write(enc.finish().unwrap().tail).await.unwrap();
            up.finish().await.unwrap();
            let meta = object_store::ObjectStoreExt::head(self.inner.as_ref(), &Path::from(name))
                .await
                .unwrap();
            FileRef::from_meta(StoreId::new(STORE), &meta)
        })
    }
}

/// Concurrent readers of one footer share one read and its result.
#[test]
fn concurrent_footer_reads_are_single_flight() {
    let t = Timed::new(StorePolicy::UNPACED);
    let file = t.file("sf/f", 300);
    t.store.set_latency(Duration::from_millis(10));
    let format = MockFixture::default().format();
    let o = t.options();
    let io = t.sched.context();
    t.rt.block_on(async {
        let read = || io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o);
        let (a, b, c) = futures::join!(read(), read(), read());
        let a = a.unwrap();
        assert!(Arc::ptr_eq(&a, &b.unwrap()));
        assert!(Arc::ptr_eq(&a, &c.unwrap()));
    });
    assert_eq!(t.store.stats().tail_gets, 1);
    assert_eq!(IoCounters::get(&io.counters().cache_misses), 1);
    assert_eq!(IoCounters::get(&io.counters().cache_hits), 2);
    assert_eq!(t.sched.cache().loads_in_flight(), 0);
}

/// With the cache disabled, waiting readers still take the leader's result;
/// a reader dropped mid-read hands the read to the next one.
#[test]
fn single_flight_without_a_cache_and_with_a_cancelled_leader() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();
    let inner = Arc::new(InMemory::new());
    let store = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let sched = IoScheduler::new(IoConfig {
        metadata_cache_bytes: 0,
        ..IoConfig::default()
    });
    sched.register_store(
        StoreId::new(STORE),
        Arc::clone(&store) as Arc<dyn ObjectStore>,
    );
    let t = Timed {
        rt,
        store,
        inner,
        sched,
    };
    let file = t.file("nc/f", 100);
    t.store.set_latency(Duration::from_millis(10));
    let format = MockFixture::default().format();
    let o = t.options();
    let io = t.sched.context();
    t.rt.block_on(async {
        let read = || io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o);
        let (a, b) = futures::join!(read(), read());
        assert!(Arc::ptr_eq(&a.unwrap(), &b.unwrap()));
    });
    assert_eq!(t.store.stats().tail_gets, 1, "one read for both readers");
    assert!(t.sched.cache().is_empty());
    // The leader is dropped while it waits for the store: the follower reads.
    t.rt.block_on(async {
        let mut leader =
            Box::pin(io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o));
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(leader.as_mut().poll(&mut cx).is_pending());
        let follower = io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o);
        drop(leader);
        assert!(follower.await.is_ok());
    });
    assert_eq!(t.sched.cache().loads_in_flight(), 0);
}

/// A read begun before an invalidation (`REFRESH`) neither fills the cache
/// nor serves a reader waiting on it; that reader reads again, in the new
/// generation, and fills the cache.
#[test]
fn invalidation_retires_reads_in_flight() {
    let t = Timed::new(StorePolicy::UNPACED);
    let file = t.file("inv/f", 100);
    t.store.set_latency(Duration::from_millis(10));
    let format = MockFixture::default().format();
    let o = t.options();
    let io = t.sched.context();
    t.rt.block_on(async {
        let read = || io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o);
        let mut leader = Box::pin(read());
        let mut waiter = Box::pin(read());
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(leader.as_mut().poll(&mut cx).is_pending());
        assert!(waiter.as_mut().poll(&mut cx).is_pending());
        t.sched.cache().invalidate_path(&file.store, &file.path);
        let (a, b) = futures::join!(leader, waiter);
        assert!(
            !Arc::ptr_eq(&a.unwrap(), &b.unwrap()),
            "the waiter took a retired read"
        );
    });
    assert_eq!(t.store.stats().tail_gets, 2, "the waiter read again");
    assert_eq!(t.sched.cache().len(), 1, "the new read filled the cache");
    t.sched.cache().clear();
    assert!(t.sched.cache().is_empty());
}

/// Readers waiting on a read that fails fail with it, instead of each
/// retrying the store one after another; a reader arriving later reads
/// again.
#[test]
fn a_failed_read_is_shared_with_its_waiters() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();
    let inner = Arc::new(InMemory::new());
    let store = FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let sched = IoScheduler::new(IoConfig {
        max_retries: 1,
        ..IoConfig::default()
    });
    sched.register_store(
        StoreId::new(STORE),
        Arc::clone(&store) as Arc<dyn ObjectStore>,
    );
    let t = Timed {
        rt,
        store,
        inner,
        sched,
    };
    let file = t.file("fail/f", 100);
    let format = MockFixture::default().format();
    let o = t.options();
    let io = t.sched.context();
    t.store.fail_next_gets(1000, None);
    let before = t.store.stats().gets;
    t.rt.block_on(async {
        let read = || io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o);
        let (a, b, c) = futures::join!(read(), read(), read());
        assert!(a.is_err() && b.is_err() && c.is_err());
    });
    assert_eq!(
        t.store.stats().gets - before,
        2,
        "one read (and its retry) for three readers"
    );
    t.store.fail_next_gets(0, None);
    t.rt.block_on(io.read_metadata(format.as_ref(), &file, MetadataNeeds::default(), &o))
        .unwrap();
}

/// A footer read ahead on the I/O runtime lands in the cache; dropping the
/// read ahead cancels it.
#[test]
fn footers_are_read_ahead_on_the_io_runtime() {
    let t = Timed::new(StorePolicy::UNPACED);
    let file = t.file("ra/f", 200);
    let other = t.file("ra/g", 200);
    t.store.set_latency(Duration::from_millis(10));
    let format = MockFixture::default().format();
    let io = t.sched.context();
    let ahead = io.prefetch_metadata(
        Arc::clone(&format),
        file,
        MetadataNeeds::default(),
        t.options(),
    );
    t.rt.block_on(async {
        // Progresses as a task while this one waits on something else.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(t.sched.cache().len(), 1);
        assert!(ahead.await.is_ok());
        // Cancelled before it ran: no read.
        let before = t.store.stats().tail_gets;
        drop(io.prefetch_metadata(format, other, MetadataNeeds::default(), t.options()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(t.store.stats().tail_gets, before);
    });
}

/// Reads and writes of a paced store wait for their prefix's tokens; other
/// prefixes and unpaced directions do not.
#[test]
fn paced_stores_pace_per_prefix() {
    let t = Timed::new(StorePolicy {
        reads_per_prefix: Some(10),
        writes_per_prefix: Some(5),
    });
    let file = t.file("hot/f", 100);
    let other = t.file("cold/f", 100);
    let io = t.sched.context();
    t.rt.block_on(async {
        let start = tokio::time::Instant::now();
        for i in 0..30 {
            let r = std::ops::Range {
                start: i,
                end: i + 1,
            };
            io.get_ranges(&file, &[r]).await.unwrap();
        }
        // A burst of 10, then 20 more at 10 per second.
        let reads = start.elapsed();
        assert!(reads >= Duration::from_millis(1900), "{reads:?}");
        let start = tokio::time::Instant::now();
        for i in 0..10 {
            let r = std::ops::Range {
                start: i,
                end: i + 1,
            };
            io.get_ranges(&other, &[r]).await.unwrap();
        }
        assert!(
            start.elapsed() < Duration::from_millis(100),
            "another prefix waited"
        );
        let start = tokio::time::Instant::now();
        for k in 0..10 {
            let mut up = io
                .upload(&StoreId::new(STORE), Path::from(format!("w/{k}")))
                .unwrap();
            up.write(bytes::Bytes::from_static(b"x")).await.unwrap();
            up.finish().await.unwrap();
        }
        let writes = start.elapsed();
        assert!(writes >= Duration::from_millis(900), "{writes:?}");
        // A multipart upload's start, parts and completion are all writes:
        // 4 requests after the bucket is spent, at 5 per second.
        let start = tokio::time::Instant::now();
        let mut up = io
            .upload(&StoreId::new(STORE), Path::from("w/big"))
            .unwrap();
        for _ in 0..3 {
            up.write(bytes::Bytes::from(vec![7u8; 5 << 20]))
                .await
                .unwrap();
        }
        up.finish().await.unwrap();
        let multipart = start.elapsed();
        assert!(multipart >= Duration::from_millis(700), "{multipart:?}");
        io.delete(&StoreId::new(STORE), &Path::from("w/0"))
            .await
            .unwrap();
    });
}
