// SPDX-License-Identifier: Apache-2.0
//! The Delta engine on one arbitrary commit file (P3-T021 PR-08a). The input is written as
//! version 0 of a table's `_delta_log`, then:
//! - the JSON handler parses each line with `parse_json` and the whole file with
//!   `read_json_files` (the parse limits, arrow-json);
//! - the storage handler reads it as a `_last_checkpoint` and a `.crc` file (their limits);
//! - the kernel builds a snapshot from it and drains a scan's metadata, which runs the
//!   evaluator (`ParseJson` of the stats, `MapToStruct` of the partition values).
//!
//! Every step must return a result or an error, never panic. Seeds: commit files,
//! `_last_checkpoint` and `.crc` files of `tests/fixtures/delta-4.4.1/` (copy them into the
//! corpus; see that directory's README).
#![no_main]

use std::sync::{Arc, OnceLock};

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use bytes::Bytes;
use delta_kernel::schema::{DataType, MapType, StructField, StructType};
use delta_kernel::{FileMeta, JsonHandler, Snapshot, StorageHandler};
use krishiv_connectors_delta::engine::pool::LogPool;
use krishiv_connectors_delta::engine::{DeltaEngine, DeltaLimits, KrishivEngineData};
use krishiv_exec::CancellationToken;
use krishiv_io::{IoConfig, IoScheduler, StoreId};
use libfuzzer_sys::fuzz_target;
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;
use url::Url;

struct Harness {
    rt: tokio::runtime::Runtime,
    store: Arc<InMemory>,
    engine: DeltaEngine,
    root: Url,
}

fn harness() -> &'static Harness {
    static H: OnceLock<Harness> = OnceLock::new();
    H.get_or_init(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        let sched = IoScheduler::new(IoConfig {
            metadata_cache_bytes: 0,
            ..IoConfig::default()
        });
        let id = StoreId::new("memory://fuzz");
        sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
        let root = Url::parse("memory://fuzz/t/").unwrap();
        let pool: &'static LogPool = Box::leak(Box::new(LogPool::new(1)));
        let engine = DeltaEngine::new(
            sched.context(),
            id,
            root.clone(),
            rt.handle().clone(),
            DeltaLimits {
                max_metadata_bytes: 1 << 20,
                ..DeltaLimits::default()
            },
            Arc::new(CancellationToken::new()),
            pool,
        )
        .unwrap();
        Harness {
            rt,
            store,
            engine,
            root,
        }
    })
}

fn action_schema() -> Arc<StructType> {
    let add = StructType::try_new([
        StructField::nullable("path", DataType::STRING),
        StructField::nullable(
            "partitionValues",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
        StructField::nullable("size", DataType::LONG),
        StructField::nullable("stats", DataType::STRING),
    ])
    .unwrap();
    let meta = StructType::try_new([
        StructField::nullable("id", DataType::STRING),
        StructField::nullable("schemaString", DataType::STRING),
    ])
    .unwrap();
    Arc::new(
        StructType::try_new([
            StructField::nullable("add", add),
            StructField::nullable("metaData", meta),
        ])
        .unwrap(),
    )
}

fuzz_target!(|data: &[u8]| {
    let h = harness();
    let commit = "t/_delta_log/00000000000000000000.json";
    h.rt.block_on(async {
        for path in [commit, "t/_delta_log/_last_checkpoint", "t/_delta_log/00000000000000000000.crc"] {
            object_store::ObjectStoreExt::put(h.store.as_ref(), &Path::from(path), data.to_vec().into())
                .await
                .unwrap();
        }
    });
    let root = h.root.clone();
    let size = data.len() as u64;
    let lines: Vec<String> = String::from_utf8_lossy(data).lines().map(str::to_owned).collect();
    let done = h.rt.block_on(h.engine.delta_blocking(move |e| {
        let json = e.json();
        let strings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", ArrowType::Utf8, true)])),
            vec![Arc::new(StringArray::from(lines))],
        )
        .unwrap();
        let _ = json.parse_json(Box::new(KrishivEngineData::new(strings)), action_schema());
        let meta = FileMeta {
            location: root.join("_delta_log/00000000000000000000.json").unwrap(),
            last_modified: 0,
            size,
        };
        if let Ok(it) = json.read_json_files(&[meta], action_schema(), None) {
            it.for_each(drop);
        }
        let storage = e.storage();
        for name in ["_delta_log/_last_checkpoint", "_delta_log/00000000000000000000.crc"] {
            let url = root.join(name).unwrap();
            if let Ok(it) = storage.read_files(vec![(url, None)]) {
                it.for_each(|r: delta_kernel::KernelResult<Bytes>| drop(r));
            }
        }
        // The kernel's log replay over the commit, through the whole engine.
        if let Ok(snapshot) = Snapshot::builder_for(root.as_str()).build(e) {
            if let Ok(scan) = snapshot.scan_builder().build() {
                if let Ok(it) = scan.scan_metadata(e) {
                    it.for_each(drop);
                }
            }
        }
    }));
    // A kernel call that panicked comes back as an error; that is a finding.
    assert!(done.is_ok(), "{done:?}");
});
