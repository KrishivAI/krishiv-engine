// SPDX-License-Identifier: Apache-2.0
//! The Parquet reader against files written by other engines (P2-T080a):
//! every fixture in `tests/fixtures/parquet` (pyarrow with every codec and
//! page version, DuckDB, Spark 4.2) reads to exactly the values pyarrow
//! reads, column by column (`tests/scripts/gen_parquet_fixtures.py`); and,
//! when `KRISHIV_TPCH_DIR` names a TPC-H SF1 directory written by
//! `tests/scripts/tpch_parquet_checksums.py`, every table does too.
#![cfg(test)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::split::{SplitConfig, SplitFile};
use krishiv_io::{
    ByteRange, FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, MetadataNeeds,
    ScanOptions, ScanRequest, ScanStream, StoreId,
};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/parquet")
}

fn canonical(a: &ArrayRef, i: usize, out: &mut Vec<u8>) {
    if a.is_null(i) {
        out.extend_from_slice(b"\x00N");
    } else {
        let text = match a.data_type() {
            DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
            DataType::Int16 => a.as_primitive::<Int16Type>().value(i).to_string(),
            DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
            DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
            DataType::Date32 => a.as_primitive::<Date32Type>().value(i).to_string(),
            DataType::Decimal64(..) => a.as_primitive::<Decimal64Type>().value(i).to_string(),
            DataType::Decimal128(..) => a.as_primitive::<Decimal128Type>().value(i).to_string(),
            DataType::Timestamp(TimeUnit::Microsecond, _) => a
                .as_primitive::<TimestampMicrosecondType>()
                .value(i)
                .to_string(),
            DataType::Float64 => hex(&a.as_primitive::<Float64Type>().value(i).to_le_bytes()),
            DataType::Float32 => {
                hex(&f64::from(a.as_primitive::<Float32Type>().value(i)).to_le_bytes())
            }
            DataType::Boolean => (if a.as_boolean().value(i) { "1" } else { "0" }).to_owned(),
            DataType::Utf8View => a.as_string_view().value(i).to_owned(),
            DataType::BinaryView => hex(a.as_binary_view().value(i)),
            other => panic!("no canonical encoding for {other}"),
        };
        out.extend_from_slice(text.as_bytes());
    }
    out.push(0x1F);
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Reads every split of `name` in `dir` (planned as Spark would with
/// 4 MiB partitions) and returns the rows, the per-column checksums and
/// every row read.
fn read(dir: &Path, name: &str) -> (u64, Vec<(String, String)>, RecordBatch) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let store = Arc::new(LocalFileSystem::new_with_prefix(dir).unwrap());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("file://");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let path = StorePath::from(name);
    let meta = rt
        .block_on(object_store::ObjectStoreExt::head(store.as_ref(), &path))
        .unwrap();
    let file = FileRef::from_meta(id, &meta);
    let o = FormatOptions::defaults(ParquetFormat.options());
    let schema = rt
        .block_on(io.read_metadata(&ParquetFormat, &file, MetadataNeeds::default(), &o))
        .unwrap()
        .schema();
    let flat: Vec<String> = schema
        .fields()
        .iter()
        .filter(|f| !f.data_type().is_nested())
        .map(|f| f.name().clone())
        .collect();
    let req = ScanRequest::all(Arc::clone(&schema));
    let cfg = SplitConfig {
        max_partition_bytes: 4 << 20,
        open_cost_in_bytes: 4 << 20,
        min_partition_num: 1,
        max_partition_num: None,
    };
    let splits = krishiv_io::split::plan(
        &cfg,
        &[SplitFile {
            size: file.size,
            splittable: true,
        }],
    );
    let mut hashes: Vec<Sha256> = flat.iter().map(|_| Sha256::new()).collect();
    let mut rows = 0u64;
    let mut all = Vec::new();
    let mut buf = Vec::new();
    // Splits in file order, so the rows come in file order.
    let mut ranges: Vec<ByteRange> = splits
        .iter()
        .flat_map(|p| p.splits.iter().map(|s| s.range))
        .collect();
    ranges.sort_by_key(|r| r.start);
    for range in ranges {
        let id = pool.register("scan", None);
        let mem = Reservation::new(Arc::clone(&pool), id);
        rt.block_on(async {
            let mut s = ScanStream::open(
                &io,
                &ParquetFormat,
                file.clone(),
                range,
                &req,
                &o,
                mem,
                ScanOptions::default(),
            )
            .await
            .unwrap();
            while let Some(b) = s.next_batch().await.unwrap() {
                rows += b.num_rows() as u64;
                for (k, name) in flat.iter().enumerate() {
                    let c = b.column_by_name(name).unwrap();
                    buf.clear();
                    for i in 0..c.len() {
                        canonical(c, i, &mut buf);
                    }
                    hashes[k].update(&buf);
                }
                all.push(b);
            }
        });
    }
    assert_eq!(pool.reserved(), 0);
    let sums = flat
        .into_iter()
        .zip(hashes)
        .map(|(n, h)| (n, hex(&h.finalize())))
        .collect();
    let batch = match all.first() {
        Some(b) => arrow_select::concat::concat_batches(&b.schema(), &all).unwrap(),
        None => RecordBatch::new_empty(schema),
    };
    (rows, sums, batch)
}

fn check(dir: &Path, name: &str, expected: &Value) {
    let (rows, sums, _) = read(dir, name);
    assert_eq!(Some(rows), expected["rows"].as_u64(), "{name}: rows");
    let columns = expected["columns"].as_object().unwrap();
    let mut checked = 0;
    for (col, sum) in sums {
        let want = columns[&col]["sha256"].as_str().unwrap();
        assert_eq!(sum, want, "{name}: column {col} differs from pyarrow");
        checked += 1;
    }
    let flat = columns.values().filter(|c| !c["sha256"].is_null()).count();
    assert_eq!(checked, flat, "{name}: columns checked");
}

#[test]
fn fixtures_read_as_pyarrow_reads_them() {
    let dir = fixtures();
    let mut files = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        let expected: Value =
            serde_json::from_str(&std::fs::read_to_string(path.with_extension("json")).unwrap())
                .unwrap();
        check(&dir, &name, &expected);
        files += 1;
    }
    assert!(files >= 10, "{files} fixtures");
}

/// Spark's nested columns, checked value by value (pyarrow's checksums
/// cover the flat ones).
#[test]
fn spark_nested_columns() {
    let (rows, _, b) = read(&fixtures(), "spark.parquet");
    assert_eq!(rows, 500);
    let id = b.column_by_name("id").unwrap().as_primitive::<Int32Type>();
    let st = b.column_by_name("st").unwrap().as_struct();
    let arr = b.column_by_name("arr").unwrap().as_list::<i32>();
    let m = b.column_by_name("m").unwrap().as_map();
    for r in 0..b.num_rows() {
        let i = id.value(r);
        if i % 9 == 4 {
            assert!(st.is_null(r) && arr.is_null(r) && m.is_null(r));
            continue;
        }
        assert_eq!(st.column(0).as_primitive::<Int32Type>().value(r), i * 2);
        assert_eq!(st.column(1).as_string_view().value(r), format!("y{i}"));
        let a = arr.value(r);
        let a = a.as_primitive::<Int64Type>();
        let want: Vec<i64> = (0..i % 4).map(|k| i64::from(i * 10 + k)).collect();
        assert_eq!(a.values(), want.as_slice());
        assert_eq!(m.value(r).len(), usize::try_from(i % 3).unwrap());
    }
}

/// Acceptance (P2-T080a): TPC-H SF1 written by DuckDB reads to exactly the
/// values pyarrow reads, per column. Needs the generated data (about 1 GB,
/// outside the repo).
#[test]
#[ignore = "needs ~1 GB of generated TPC-H SF1 data (tests/scripts/tpch_parquet_checksums.py, KRISHIV_TPCH_DIR); un-ignored by P2-T119, which provides the TPC-H datasets to the nightly benchmark box"]
fn tpch_sf1_reads_as_pyarrow_reads_it() {
    let dir = PathBuf::from(std::env::var("KRISHIV_TPCH_DIR").unwrap());
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("checksums.json")).unwrap())
            .unwrap();
    for (table, want) in expected["tables"].as_object().unwrap() {
        check(&dir, &format!("{table}.parquet"), want);
    }
}
