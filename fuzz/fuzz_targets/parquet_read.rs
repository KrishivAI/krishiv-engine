// SPDX-License-Identifier: Apache-2.0
//! The Parquet reader on arbitrary bytes (P2-T080a). The input is a file:
//! its footer is read through the scheduler, then every column of every
//! split (whole file, and two halves) is decoded, with and without a pushed
//! predicate and row index. Each step must return rows or a typed error,
//! never panic out of the reader, and never hold memory after the scan. Seeds: the
//! interoperability fixtures (`tests/fixtures/parquet`).
#![no_main]

use std::sync::{Arc, OnceLock};

use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, CmpOp, ColumnPath, FileFormat, FileRef, FormatOptions, IoConfig, IoContext,
    IoScheduler, MetadataNeeds, PushdownExpr, ScalarValue, ScanOptions, ScanRequest, ScanStream,
    StoreId,
};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use libfuzzer_sys::fuzz_target;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;

struct Harness {
    rt: tokio::runtime::Runtime,
    store: Arc<InMemory>,
    io: IoContext,
    pool: Arc<dyn MemoryPool>,
}

fn harness() -> &'static Harness {
    static H: OnceLock<Harness> = OnceLock::new();
    H.get_or_init(|| {
        // The reader contains arrow-rs panics on corrupt data (it reports
        // them as errors); print them without aborting. A panic that
        // escapes the reader still reaches libFuzzer's handler and crashes.
        std::panic::set_hook(Box::new(|info| eprintln!("contained: {info}")));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        // No caching across inputs: every input is a new file at one path.
        let sched = IoScheduler::new(IoConfig {
            metadata_cache_bytes: 0,
            ..IoConfig::default()
        });
        sched.register_store(StoreId::new("memory://fuzz"), Arc::clone(&store) as Arc<dyn ObjectStore>);
        let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
            limit: Some(512 << 20),
            ..MemoryConfig::default()
        });
        Harness {
            rt,
            store,
            io: sched.context(),
            pool,
        }
    })
}

fuzz_target!(|data: &[u8]| {
    let h = harness();
    h.rt.block_on(async {
        let path = Path::from("f.parquet");
        object_store::ObjectStoreExt::put(h.store.as_ref(), &path, data.to_vec().into())
            .await
            .unwrap();
        let meta = object_store::ObjectStoreExt::head(h.store.as_ref(), &path)
            .await
            .unwrap();
        let file = FileRef::from_meta(StoreId::new("memory://fuzz"), &meta);
        let o = FormatOptions::defaults(ParquetFormat.options());
        let Ok(m) = h
            .io
            .read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o)
            .await
        else {
            return;
        };
        let schema = m.schema();
        let mut requests = vec![ScanRequest::all(Arc::clone(&schema))];
        if let Some(first) = schema.fields().first() {
            let mut r = ScanRequest::all(Arc::clone(&schema));
            r.predicate = Some(PushdownExpr::cmp(
                ColumnPath::new(first.name()),
                CmpOp::Gt,
                ScalarValue::Int32(0),
            ));
            r.metadata_cols.row_index = true;
            requests.push(r);
        }
        let half = file.size / 2;
        for req in &requests {
            for split in [
                ByteRange::whole(file.size),
                ByteRange::new(0, half),
                ByteRange::new(half, file.size),
            ] {
                let id = h.pool.register("scan", None);
                let mem = Reservation::new(Arc::clone(&h.pool), id);
                let opened =
                    ScanStream::open(&h.io, &ParquetFormat, file.clone(), split, req, &o, mem, ScanOptions::default())
                        .await;
                if let Ok(mut s) = opened {
                    while let Ok(Some(_)) = s.next_batch().await {}
                }
            }
        }
        assert_eq!(h.pool.reserved(), 0, "a scan held memory after it ended");
    });
});
