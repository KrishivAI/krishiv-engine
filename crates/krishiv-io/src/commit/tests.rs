// SPDX-License-Identifier: Apache-2.0
//! Save modes and the directory commit, on an in-memory store and on the
//! local file system (which has empty directories and files in the way).

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use krishiv_common::error::ErrorClass;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::path::Path;

use super::{DirectoryWrite, SUCCESS_MARKER, SaveMode};
use crate::parquet::ParquetFormat;
use crate::{
    FileFormat, FileWriter, FormatOptions, IoConfig, IoContext, IoScheduler, StoreId,
    WrittenFileInfo,
};

const STORE: &str = "memory://commit";

struct Env {
    rt: tokio::runtime::Runtime,
    io: IoContext,
    store: StoreId,
}

impl Env {
    fn new(store: Arc<dyn ObjectStore>, id: &str) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let sched = IoScheduler::new(IoConfig::default());
        sched.register_store(StoreId::new(id), store);
        Self {
            rt,
            io: sched.context(),
            store: StoreId::new(id),
        }
    }

    fn memory() -> Self {
        Self::new(Arc::new(InMemory::new()), STORE)
    }

    fn put(&self, path: &str) {
        self.rt
            .block_on(
                self.io
                    .put_object(&self.store, &Path::from(path), Bytes::from_static(b"x")),
            )
            .unwrap();
    }

    fn names(&self, dir: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .rt
            .block_on(self.io.list(&self.store, &Path::from(dir)))
            .unwrap()
            .into_iter()
            .map(|m| m.location.to_string())
            .collect();
        v.sort();
        v
    }

    fn begin(
        &self,
        dir: &str,
        mode: SaveMode,
        uuid: &str,
    ) -> crate::Result<Option<DirectoryWrite>> {
        self.rt.block_on(DirectoryWrite::begin(
            &self.io,
            self.store.clone(),
            Path::from(dir),
            mode,
            uuid,
        ))
    }

    /// Writes `rows` rows (none: an empty output) with `mode` and commits.
    fn write(
        &self,
        dir: &str,
        mode: SaveMode,
        uuid: &str,
        rows: i32,
    ) -> crate::Result<Vec<WrittenFileInfo>> {
        let Some(w) = self.begin(dir, mode, uuid)? else {
            return Ok(Vec::new());
        };
        let o = FormatOptions::defaults(ParquetFormat.options());
        let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
        self.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("write", None);
            let mut writer = FileWriter::new(
                &self.io,
                Arc::clone(&format),
                schema(),
                o.clone(),
                w.write_options(0),
                krishiv_memory::Reservation::new(pool, id),
            )?;
            if rows > 0 {
                writer.write(&batch(rows)).await?;
            }
            let files = writer.finish().await?;
            w.commit(&self.io, format.as_ref(), schema(), &o, files)
                .await
        })
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
}

fn batch(rows: i32) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from_iter_values(0..rows))],
    )
    .unwrap()
}

#[test]
fn save_modes_parse_as_spark_parses_them() {
    for (name, mode) in [
        ("error", SaveMode::ErrorIfExists),
        ("ErrorIfExists", SaveMode::ErrorIfExists),
        ("default", SaveMode::ErrorIfExists),
        ("APPEND", SaveMode::Append),
        ("overwrite", SaveMode::Overwrite),
        ("ignore", SaveMode::Ignore),
    ] {
        assert_eq!(SaveMode::parse(name).unwrap(), mode, "{name}");
    }
    let e = SaveMode::parse("bogus").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidSaveMode);
    assert!(e.to_string().contains("\"bogus\""), "{e}");
}

#[test]
fn modes_on_a_directory() {
    let env = Env::memory();
    // error: a new path is written, with its marker.
    let files = env.write("t", SaveMode::ErrorIfExists, "u1", 10).unwrap();
    assert_eq!(files.len(), 1);
    let first = files[0].path.to_string();
    assert_eq!(first, "t/part-00000-u1-c000.snappy.parquet");
    assert_eq!(env.names("t"), vec!["t/_SUCCESS".to_owned(), first.clone()]);
    // error / ignore on an existing path.
    let e = env
        .write("t", SaveMode::ErrorIfExists, "u2", 10)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PathAlreadyExists);
    assert!(e.to_string().contains("memory://commit/t"), "{e}");
    assert!(
        env.write("t", SaveMode::Ignore, "u2", 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(env.names("t").len(), 2);
    // ignore on a new path writes.
    assert_eq!(env.write("t2", SaveMode::Ignore, "u2", 3).unwrap().len(), 1);
    // append adds a file.
    env.write("t", SaveMode::Append, "u3", 5).unwrap();
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            first,
            "t/part-00000-u3-c000.snappy.parquet".to_owned()
        ]
    );
    // overwrite replaces everything, at any depth, and only the old files.
    env.put("t/sub/x.txt");
    env.put("t/_meta");
    env.write("t", SaveMode::Overwrite, "u4", 7).unwrap();
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            "t/part-00000-u4-c000.snappy.parquet".to_owned()
        ]
    );
    // An overwrite reusing the job UUID of the files it replaces keeps
    // its own files.
    env.write("t", SaveMode::Overwrite, "u4", 3).unwrap();
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            "t/part-00000-u4-c000.snappy.parquet".to_owned()
        ]
    );
    // overwrite of a new path just writes.
    assert_eq!(
        env.write("t3", SaveMode::Overwrite, "u5", 1).unwrap().len(),
        1
    );
}

#[test]
fn a_failed_overwrite_keeps_the_old_data() {
    let env = Env::memory();
    env.write("t", SaveMode::ErrorIfExists, "old", 4).unwrap();
    let before = env.names("t");
    // The write fails (the schema does not match the batch): no commit.
    let w = env.begin("t", SaveMode::Overwrite, "new").unwrap().unwrap();
    let o = FormatOptions::defaults(ParquetFormat.options());
    let other = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    let failed = env.rt.block_on(async {
        let pool: Arc<dyn krishiv_memory::MemoryPool> =
            krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
        let id = pool.register("write", None);
        let mut writer = FileWriter::new(
            &env.io,
            Arc::new(ParquetFormat),
            other,
            o,
            w.write_options(0),
            krishiv_memory::Reservation::new(pool, id),
        )
        .unwrap();
        let r = writer.write(&batch(3)).await;
        writer.abort().await;
        r
    });
    assert_eq!(
        failed.unwrap_err().error_class(),
        ErrorClass::TaskWriteFailed
    );
    drop(w);
    assert_eq!(env.names("t"), before);
}

#[test]
fn an_empty_output_still_has_a_file_with_the_schema() {
    let env = Env::memory();
    let files = env.write("e", SaveMode::ErrorIfExists, "u", 0).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].rows, 0);
    assert_eq!(
        files[0].path.to_string(),
        "e/part-00000-u-c000.snappy.parquet"
    );
    let bytes = env
        .rt
        .block_on(async {
            let meta = env
                .io
                .head(&env.store, &files[0].path)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(meta.size, files[0].size);
            let file = crate::FileRef::from_meta(env.store.clone(), &meta);
            let whole = std::iter::once(0..meta.size).collect::<Vec<_>>();
            env.io.get_ranges(&file, &whole).await
        })
        .unwrap();
    let reader =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes[0].clone())
            .unwrap();
    assert_eq!(reader.schema().fields().len(), 1);
    assert_eq!(reader.metadata().file_metadata().num_rows(), 0);
}

/// A fresh directory under the system temp directory, removed on drop.
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "krishiv-commit-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        drop(std::fs::remove_dir_all(&p));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

#[test]
fn local_directories_and_files_follow_spark() {
    let tmp = TempDir::new("local");
    let env = Env::new(
        Arc::new(LocalFileSystem::new_with_prefix(&tmp.0).unwrap()),
        "file://",
    );
    // An empty directory exists for error and ignore.
    std::fs::create_dir(tmp.0.join("empty")).unwrap();
    let e = env
        .write("empty", SaveMode::ErrorIfExists, "u", 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PathAlreadyExists);
    assert!(
        env.write("empty", SaveMode::Ignore, "u", 1)
            .unwrap()
            .is_empty()
    );
    assert!(
        std::fs::read_dir(tmp.0.join("empty"))
            .unwrap()
            .next()
            .is_none()
    );
    // ... and is written into by append.
    assert_eq!(
        env.write("empty", SaveMode::Append, "u", 1).unwrap().len(),
        1
    );
    // A file at the path: error and append fail, overwrite replaces it.
    std::fs::write(tmp.0.join("afile"), b"hi").unwrap();
    for mode in [SaveMode::ErrorIfExists, SaveMode::Append] {
        let e = env.write("afile", mode, "u", 1).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::PathAlreadyExists, "{mode:?}");
    }
    env.write("afile", SaveMode::Overwrite, "u", 2).unwrap();
    assert!(tmp.0.join("afile").is_dir());
    assert!(tmp.0.join("afile").join(SUCCESS_MARKER).is_file());
    // A path that does not exist, nested under new directories.
    assert_eq!(
        env.write("a/b/c", SaveMode::ErrorIfExists, "u", 1)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn the_root_always_exists() {
    let env = Env::memory();
    let e = env.begin("", SaveMode::ErrorIfExists, "u").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::PathAlreadyExists);
    assert!(
        env.rt
            .block_on(env.io.is_directory(&env.store, &Path::from("")))
            .unwrap()
    );
}

/// Parquet whose encoder hands out its bytes before `finish` (as formats
/// with a file header do): all but the last eight bytes up front.
#[derive(Debug)]
struct EarlyBytes;

struct EarlyEncoder {
    head: Option<Bytes>,
    done: Option<crate::WrittenFile>,
}

impl crate::Encoder for EarlyEncoder {
    fn write(&mut self, _batch: &RecordBatch) -> crate::FormatResult<()> {
        Ok(())
    }
    fn take_bytes(&mut self) -> Option<Bytes> {
        self.head.take()
    }
    fn finish(mut self: Box<Self>) -> crate::FormatResult<crate::WrittenFile> {
        self.done
            .take()
            .ok_or_else(|| crate::FormatError::WriteFailed("finished twice".to_owned()))
    }
    fn buffered_bytes(&self) -> usize {
        0
    }
    fn written_bytes(&self) -> u64 {
        0
    }
}

impl FileFormat for EarlyBytes {
    fn name(&self) -> &'static str {
        "early"
    }
    fn options(&self) -> &'static crate::OptionSchema {
        ParquetFormat.options()
    }
    fn capabilities(&self, o: &FormatOptions) -> crate::Capabilities {
        ParquetFormat.capabilities(o)
    }
    fn file_suffix(&self, o: &FormatOptions) -> String {
        ParquetFormat.file_suffix(o)
    }
    fn read_metadata<'a>(
        &'a self,
        io: &'a IoContext,
        f: &'a crate::FileRef,
        needs: crate::MetadataNeeds,
        o: &'a FormatOptions,
    ) -> futures::future::BoxFuture<'a, crate::FormatResult<Arc<dyn crate::FileMetadata>>> {
        ParquetFormat.read_metadata(io, f, needs, o)
    }
    fn plan_units(
        &self,
        m: &dyn crate::FileMetadata,
        split: &crate::ByteRange,
        req: &crate::ScanRequest,
    ) -> crate::FormatResult<crate::UnitPlan> {
        ParquetFormat.plan_units(m, split, req)
    }
    fn open<'a>(
        &'a self,
        io: &'a IoContext,
        m: Arc<dyn crate::FileMetadata>,
        units: crate::UnitPlan,
        req: &'a crate::ScanRequest,
        o: &'a FormatOptions,
    ) -> futures::future::BoxFuture<'a, crate::FormatResult<Box<dyn crate::Decoder>>> {
        ParquetFormat.open(io, m, units, req, o)
    }
    fn writer<'a>(
        &'a self,
        io: &'a IoContext,
        schema: SchemaRef,
        o: &'a FormatOptions,
    ) -> futures::future::BoxFuture<'a, crate::FormatResult<Box<dyn crate::Encoder>>> {
        Box::pin(async move {
            let mut inner = ParquetFormat.writer(io, schema, o).await?;
            let mut all = Vec::new();
            while let Some(b) = inner.take_bytes() {
                all.extend_from_slice(&b);
            }
            let done = inner.finish()?;
            all.extend_from_slice(&done.tail);
            let tail = all.split_off(all.len() - 8);
            Ok(Box::new(EarlyEncoder {
                head: Some(Bytes::from(all)),
                done: Some(crate::WrittenFile::new(Bytes::from(tail), done.statistics)),
            }) as Box<dyn crate::Encoder>)
        })
    }
}

#[test]
fn an_empty_output_keeps_bytes_handed_out_before_finish() {
    let env = Env::memory();
    let w = env
        .begin("early", SaveMode::Overwrite, "u")
        .unwrap()
        .unwrap();
    let o = FormatOptions::defaults(ParquetFormat.options());
    let files = env
        .rt
        .block_on(w.commit(&env.io, &EarlyBytes, schema(), &o, Vec::new()))
        .unwrap();
    let bytes = env
        .rt
        .block_on(async {
            let meta = env.io.head(&env.store, &files[0].path).await?.unwrap();
            let file = crate::FileRef::from_meta(env.store.clone(), &meta);
            let whole = std::iter::once(0..meta.size).collect::<Vec<_>>();
            env.io.get_ranges(&file, &whole).await
        })
        .unwrap();
    // The whole file arrived: arrow-rs reads it.
    let reader =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes[0].clone())
            .unwrap();
    assert_eq!(reader.metadata().file_metadata().num_rows(), 0);
}

#[test]
fn listing_and_head_requests_are_retried_then_fail_as_store_failures() {
    let inner = Arc::new(InMemory::new());
    let fault = crate::io::FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let env = Env::new(Arc::clone(&fault) as Arc<dyn ObjectStore>, STORE);
    env.put("t/a");
    // Transient failures are retried.
    fault.fail_next_gets(2, None);
    let found = env
        .rt
        .block_on(env.io.head(&env.store, &Path::from("t/a")))
        .unwrap();
    assert!(found.is_some());
    // Past the retries: a store failure, never "not found".
    fault.fail_next_gets(1000, None);
    let e = env
        .rt
        .block_on(env.io.head(&env.store, &Path::from("t/a")))
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivObjectStoreRequestFailed);
}

#[test]
fn a_failing_store_request_is_not_retried() {
    let tmp = TempDir::new("fail");
    std::fs::write(tmp.0.join("afile"), b"hi").unwrap();
    let env = Env::new(
        Arc::new(LocalFileSystem::new_with_prefix(&tmp.0).unwrap()),
        "file://",
    );
    // A path under a file: the file system refuses it outright.
    let r = env
        .rt
        .block_on(env.io.head(&env.store, &Path::from("afile/x")));
    assert_eq!(
        r.unwrap_err().error_class(),
        ErrorClass::KrishivObjectStoreRequestFailed
    );
}

#[test]
fn listings_and_deletes_are_retried() {
    let inner = Arc::new(InMemory::new());
    let fault = crate::io::FaultStore::new(Arc::clone(&inner) as Arc<dyn ObjectStore>);
    let env = Env::new(Arc::clone(&fault) as Arc<dyn ObjectStore>, STORE);
    env.write("t", SaveMode::ErrorIfExists, "old", 4).unwrap();
    // A listing failing midway starts again: nothing is lost or doubled.
    fault.fail_next_lists(2);
    assert_eq!(env.names("t").len(), 2);
    // An overwrite whose deletes fail transiently still replaces the data.
    fault.fail_next_deletes(3);
    env.write("t", SaveMode::Overwrite, "new", 2).unwrap();
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            "t/part-00000-new-c000.snappy.parquet".to_owned()
        ]
    );
    // Past the retries: store failures.
    fault.fail_next_lists(1000);
    let e = env
        .rt
        .block_on(env.io.list(&env.store, &Path::from("t")))
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivObjectStoreRequestFailed);
    fault.fail_next_lists(0);
    fault.fail_next_deletes(1000);
    let e = env
        .rt
        .block_on(env.io.delete(&env.store, &Path::from("t/_SUCCESS")))
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed);
    fault.fail_next_deletes(0);
    // Deleting what is gone succeeds.
    env.rt
        .block_on(env.io.delete(&env.store, &Path::from("t/nothing")))
        .unwrap();
}

#[test]
fn a_store_refusing_a_listing_or_delete_is_not_retried() {
    let tmp = TempDir::new("refuse");
    std::fs::write(tmp.0.join("afile"), b"hi").unwrap();
    let env = Env::new(
        Arc::new(LocalFileSystem::new_with_prefix(&tmp.0).unwrap()),
        "file://",
    );
    let e = env
        .rt
        .block_on(env.io.list(&env.store, &Path::from("afile/x")));
    let d = env
        .rt
        .block_on(env.io.delete(&env.store, &Path::from("afile/x")));
    // The local store reports these as missing or refuses them: never a
    // retry loop, never a panic.
    assert!(
        e.is_ok() || e.unwrap_err().error_class() == ErrorClass::KrishivObjectStoreRequestFailed
    );
    assert!(d.is_ok() || d.unwrap_err().error_class() == ErrorClass::TaskWriteFailed);
}
