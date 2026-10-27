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
        self.begin_dynamic(dir, mode, uuid, false)
    }

    fn begin_dynamic(
        &self,
        dir: &str,
        mode: SaveMode,
        uuid: &str,
        dynamic: bool,
    ) -> crate::Result<Option<DirectoryWrite>> {
        self.rt.block_on(DirectoryWrite::begin(
            &self.io,
            self.store.clone(),
            Path::from(dir),
            mode,
            uuid,
            dynamic,
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
                &schema(),
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

    /// Writes partition-sorted `batches` with `partition_by` and commits.
    #[expect(clippy::too_many_arguments, reason = "one write, all inputs distinct")]
    fn write_partitioned(
        &self,
        dir: &str,
        mode: SaveMode,
        dynamic: bool,
        uuid: &str,
        schema: SchemaRef,
        batches: &[RecordBatch],
        partition_by: Vec<usize>,
        partition_names: Vec<&str>,
        max_records: u64,
    ) -> crate::Result<Vec<WrittenFileInfo>> {
        let Some(w) = self.begin_dynamic(dir, mode, uuid, dynamic)? else {
            return Ok(Vec::new());
        };
        let o = FormatOptions::defaults(ParquetFormat.options());
        let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
        self.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("write", None);
            let mut options = w.write_options(0);
            options.partition_by = partition_by;
            options.partition_names = partition_names.into_iter().map(str::to_owned).collect();
            options.max_records_per_file = max_records;
            let mut writer = FileWriter::new(
                &self.io,
                Arc::clone(&format),
                &schema,
                o.clone(),
                options,
                krishiv_memory::Reservation::new(pool, id),
            )?;
            for b in batches {
                writer.write(b).await?;
            }
            let files = writer.finish().await?;
            w.commit(&self.io, format.as_ref(), schema, &o, files).await
        })
    }

    /// The `id` values of the parquet file at `path`.
    fn read_ids(&self, path: &Path) -> Vec<i32> {
        use arrow_array::cast::AsArray as _;
        let bytes = self
            .rt
            .block_on(async {
                let meta = self.io.head(&self.store, path).await?.unwrap();
                let file = crate::FileRef::from_meta(self.store.clone(), &meta);
                let whole = std::iter::once(0..meta.size).collect::<Vec<_>>();
                self.io.get_ranges(&file, &whole).await
            })
            .unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            bytes[0].clone(),
        )
        .unwrap()
        .build()
        .unwrap();
        let mut ids = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            ids.extend(
                batch
                    .column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values(),
            );
        }
        ids
    }

    /// The field names of the parquet file at `path`.
    fn read_fields(&self, path: &Path) -> Vec<String> {
        let bytes = self
            .rt
            .block_on(async {
                let meta = self.io.head(&self.store, path).await?.unwrap();
                let file = crate::FileRef::from_meta(self.store.clone(), &meta);
                let whole = std::iter::once(0..meta.size).collect::<Vec<_>>();
                self.io.get_ranges(&file, &whole).await
            })
            .unwrap();
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes[0].clone())
            .unwrap()
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }
}

fn part_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("p", DataType::Utf8, false),
    ]))
}

/// One batch of `(id, p)` rows (partition-sorted by the caller).
fn part_batch(ids: &[i32], ps: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        part_schema(),
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(arrow_array::StringArray::from(ps.to_vec())),
        ],
    )
    .unwrap()
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
            &other,
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
    // The failed delete left the object in place.
    assert!(env.names("t").contains(&"t/_SUCCESS".to_owned()));
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

#[test]
fn partitioned_writes_follow_sparks_layout() {
    let memory = Env::memory();
    let tmp = TempDir::new("partitioned");
    let local = Env::new(
        Arc::new(LocalFileSystem::new_with_prefix(&tmp.0).unwrap()),
        "file://",
    );
    for env in [&memory, &local] {
        let files = env
            .write_partitioned(
                "t",
                SaveMode::ErrorIfExists,
                false,
                "u",
                part_schema(),
                &[part_batch(&[2, 3, 1, 4], &["a", "a", "b", "b"])],
                vec![1],
                vec!["p"],
                0,
            )
            .unwrap();
        assert_eq!(
            files.iter().map(|f| f.path.to_string()).collect::<Vec<_>>(),
            [
                "t/p=a/part-00000-u.c000.snappy.parquet",
                "t/p=b/part-00000-u.c001.snappy.parquet",
            ]
        );
        assert_eq!(
            env.names("t"),
            vec![
                "t/_SUCCESS".to_owned(),
                "t/p=a/part-00000-u.c000.snappy.parquet".to_owned(),
                "t/p=b/part-00000-u.c001.snappy.parquet".to_owned(),
            ]
        );
        // Data files hold only the non-partition columns.
        for (path, want) in [(&files[0].path, vec![2, 3]), (&files[1].path, vec![1, 4])] {
            assert_eq!(env.read_fields(path), ["id"]);
            assert_eq!(env.read_ids(path), want);
            assert_eq!(
                files[files.iter().position(|f| &f.path == path).unwrap()].rows,
                want.len() as u64
            );
        }
    }
}

#[test]
fn partition_by_order_is_respected() {
    let env = Env::memory();
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
        Field::new("v", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::StringArray::from(vec!["x", "x"])),
            Arc::new(arrow_array::StringArray::from(vec!["y", "y"])),
            Arc::new(Int32Array::from(vec![1, 2])),
        ],
    )
    .unwrap();
    let files = env
        .write_partitioned(
            "t",
            SaveMode::ErrorIfExists,
            false,
            "u",
            schema,
            &[batch],
            vec![1, 0],
            vec!["b", "a"],
            0,
        )
        .unwrap();
    assert_eq!(
        files.iter().map(|f| f.path.to_string()).collect::<Vec<_>>(),
        ["t/b=y/a=x/part-00000-u.c000.snappy.parquet"]
    );
}

#[test]
fn max_records_per_file_rolls_within_a_partition() {
    let env = Env::memory();
    let files = env
        .write_partitioned(
            "t",
            SaveMode::ErrorIfExists,
            false,
            "u",
            part_schema(),
            &[part_batch(&[1, 2, 3, 4, 5], &["a", "a", "a", "a", "a"])],
            vec![1],
            vec!["p"],
            2,
        )
        .unwrap();
    assert_eq!(
        files.iter().map(|f| f.path.to_string()).collect::<Vec<_>>(),
        [
            "t/p=a/part-00000-u.c000.snappy.parquet",
            "t/p=a/part-00000-u.c001.snappy.parquet",
            "t/p=a/part-00000-u.c002.snappy.parquet",
        ]
    );
    assert_eq!(files.iter().map(|f| f.rows).collect::<Vec<_>>(), [2, 2, 1]);
}

#[test]
fn a_retried_task_rewrites_the_same_names() {
    let env = Env::memory();
    // One write (one job UUID): two attempts of task 0 write the same
    // names, as names carry the task number and the job UUID but never
    // the attempt number.
    let w = env
        .begin("t", SaveMode::ErrorIfExists, "u")
        .unwrap()
        .unwrap();
    let attempt = || {
        let o = FormatOptions::defaults(ParquetFormat.options());
        let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
        env.rt
            .block_on(async {
                let pool: Arc<dyn krishiv_memory::MemoryPool> =
                    krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
                let id = pool.register("write", None);
                let mut options = w.write_options(0);
                options.partition_by = vec![1];
                options.partition_names = vec!["p".to_owned()];
                let mut writer = FileWriter::new(
                    &env.io,
                    Arc::clone(&format),
                    &part_schema(),
                    o,
                    options,
                    krishiv_memory::Reservation::new(pool, id),
                )
                .unwrap();
                writer
                    .write(&part_batch(&[1, 2], &["a", "a"]))
                    .await
                    .unwrap();
                writer.finish().await
            })
            .unwrap()
    };
    let (first, second) = (attempt(), attempt());
    assert_eq!(first, second);
    // The commit lists exactly the successful attempt's files.
    let o = FormatOptions::defaults(ParquetFormat.options());
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let files = env
        .rt
        .block_on(w.commit(
            &env.io,
            format.as_ref(),
            part_schema(),
            &o,
            first.into_iter().chain(second).collect(),
        ))
        .unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            "t/p=a/part-00000-u.c000.snappy.parquet".to_owned(),
        ]
    );
}

#[test]
fn static_overwrite_replaces_everything_dynamic_keeps_other_partitions() {
    let env = Env::memory();
    let write = |dir: &str, mode: SaveMode, dynamic: bool, uuid: &str, ids: &[i32], ps: &[&str]| {
        env.write_partitioned(
            dir,
            mode,
            dynamic,
            uuid,
            part_schema(),
            &[part_batch(ids, ps)],
            vec![1],
            vec!["p"],
            0,
        )
        .unwrap()
    };
    write(
        "t",
        SaveMode::ErrorIfExists,
        false,
        "base",
        &[1, 2],
        &["p1", "p2"],
    );
    // STATIC overwrite of one partition removes the other.
    write("t", SaveMode::Overwrite, false, "s", &[3], &["p2"]);
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            "t/p=p2/part-00000-s.c000.snappy.parquet".to_owned(),
        ]
    );
    // Back to two partitions, then a DYNAMIC overwrite of one of them.
    write("t", SaveMode::Append, false, "a", &[1], &["p1"]);
    write("t", SaveMode::Overwrite, true, "d", &[4], &["p2"]);
    assert_eq!(
        env.names("t"),
        vec![
            "t/_SUCCESS".to_owned(),
            "t/p=p1/part-00000-a.c000.snappy.parquet".to_owned(),
            "t/p=p2/part-00000-d.c000.snappy.parquet".to_owned(),
        ]
    );
    assert_eq!(
        env.read_ids(&Path::from("t/p=p1/part-00000-a.c000.snappy.parquet")),
        [1]
    );
    assert_eq!(
        env.read_ids(&Path::from("t/p=p2/part-00000-d.c000.snappy.parquet")),
        [4]
    );
}

/// A format counting its open encoders (at most one per task, however
/// many partitions the task sees).
#[derive(Debug)]
struct CountingFormat {
    open: Arc<std::sync::atomic::AtomicUsize>,
    max: Arc<std::sync::atomic::AtomicUsize>,
}

struct CountingEncoder {
    inner: Option<Box<dyn crate::Encoder>>,
    open: Arc<std::sync::atomic::AtomicUsize>,
}

impl Drop for CountingEncoder {
    fn drop(&mut self) {
        self.open.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl crate::Encoder for CountingEncoder {
    fn write(&mut self, batch: &RecordBatch) -> crate::FormatResult<()> {
        match self.inner.as_mut() {
            Some(inner) => inner.write(batch),
            None => Err(crate::FormatError::WriteFailed(
                "encoder used after finish".to_owned(),
            )),
        }
    }
    fn flush(&mut self) -> crate::FormatResult<()> {
        match self.inner.as_mut() {
            Some(inner) => inner.flush(),
            None => Err(crate::FormatError::WriteFailed(
                "encoder used after finish".to_owned(),
            )),
        }
    }
    fn take_bytes(&mut self) -> Option<bytes::Bytes> {
        self.inner.as_mut().and_then(|inner| inner.take_bytes())
    }
    fn finish(mut self: Box<Self>) -> crate::FormatResult<crate::WrittenFile> {
        match self.inner.take() {
            Some(inner) => inner.finish(),
            None => Err(crate::FormatError::WriteFailed(
                "encoder finished twice".to_owned(),
            )),
        }
    }
    fn buffered_bytes(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(0, |inner| inner.buffered_bytes())
    }
    fn written_bytes(&self) -> u64 {
        self.inner.as_ref().map_or(0, |inner| inner.written_bytes())
    }
}

impl FileFormat for CountingFormat {
    fn name(&self) -> &'static str {
        ParquetFormat.name()
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
        let open = Arc::clone(&self.open);
        let max = Arc::clone(&self.max);
        Box::pin(async move {
            let inner = ParquetFormat.writer(io, schema, o).await?;
            let now = open.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            max.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
            Ok(Box::new(CountingEncoder {
                inner: Some(inner),
                open,
            }) as Box<dyn crate::Encoder>)
        })
    }
}

#[test]
fn one_open_file_per_task_with_a_thousand_partitions() {
    let env = Env::memory();
    let open = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let format: Arc<dyn FileFormat> = Arc::new(CountingFormat {
        open: Arc::clone(&open),
        max: Arc::clone(&max),
    });
    // Unsorted input: every row changes the partition, as without the
    // planner's sort below the writer.
    let ids: Vec<i32> = (0..1000).collect();
    let ps: Vec<String> = (0..1000).map(|i| format!("p{i:04}")).collect();
    let batch = RecordBatch::try_new(
        part_schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(arrow_array::StringArray::from(ps)),
        ],
    )
    .unwrap();
    let files = env
        .rt
        .block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("write", None);
            let w = DirectoryWrite::begin(
                &env.io,
                env.store.clone(),
                Path::from("t"),
                SaveMode::ErrorIfExists,
                "u",
                false,
            )
            .await?
            .unwrap();
            let mut options = w.write_options(0);
            options.partition_by = vec![1];
            options.partition_names = vec!["p".to_owned()];
            let o = FormatOptions::defaults(ParquetFormat.options());
            let mut writer = FileWriter::new(
                &env.io,
                Arc::clone(&format),
                &part_schema(),
                o.clone(),
                options,
                krishiv_memory::Reservation::new(pool, id),
            )?;
            writer.write(&batch).await?;
            let files = writer.finish().await?;
            w.commit(&env.io, format.as_ref(), part_schema(), &o, files)
                .await
        })
        .unwrap();
    assert_eq!(files.len(), 1000);
    assert_eq!(max.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(open.load(std::sync::atomic::Ordering::SeqCst), 0);
}

#[test]
fn bad_partition_specs_fail_when_the_write_is_planned() {
    use arrow_array::LargeStringArray;
    let env = Env::memory();
    let o = FormatOptions::defaults(ParquetFormat.options());
    let format: Arc<dyn FileFormat> = Arc::new(ParquetFormat);
    let pool: Arc<dyn krishiv_memory::MemoryPool> =
        krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    let id = pool.register("write", None);
    let writer = |options: crate::WriteOptions| {
        FileWriter::new(
            &env.io,
            Arc::clone(&format),
            &part_schema(),
            o.clone(),
            options,
            krishiv_memory::Reservation::new(Arc::clone(&pool), id),
        )
    };
    let base = || WriteOptionsForTest::partitioned();
    // Out of range, repeated, and nameless columns.
    for options in [
        base().with_by(vec![9]),
        base().with_by(vec![1, 1]),
        base().with_names(vec![]),
    ] {
        assert_eq!(
            writer(options).unwrap_err().error_class(),
            ErrorClass::InternalError
        );
    }
    // A type the cast kernel cannot render (a list).
    let lists = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "p",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ),
    ]));
    let mut options = base().with_by(vec![1]);
    options.partition_names = vec!["p".to_owned()];
    let e = FileWriter::new(
        &env.io,
        Arc::clone(&format),
        &lists,
        o.clone(),
        options,
        krishiv_memory::Reservation::new(Arc::clone(&pool), id),
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // The wide string carrier renders as is.
    let column: arrow_array::ArrayRef = Arc::new(LargeStringArray::from(vec![Some("x"), None]));
    assert_eq!(
        crate::partition_texts(&column).unwrap(),
        [Some("x".to_owned()), None]
    );
}

/// A partitioned [`crate::WriteOptions`] over [`part_schema`].
struct WriteOptionsForTest;

impl WriteOptionsForTest {
    fn partitioned() -> crate::WriteOptions {
        crate::WriteOptions {
            store: StoreId::new("memory://commit"),
            dir: Path::from("t"),
            task: 0,
            job_uuid: "u".to_owned(),
            max_records_per_file: 0,
            target_file_bytes: 0,
            partition_by: vec![1],
            partition_names: vec!["p".to_owned()],
        }
    }
}

trait PartitionedOptions {
    fn with_by(self, by: Vec<usize>) -> crate::WriteOptions;
    fn with_names(self, names: Vec<&str>) -> crate::WriteOptions;
}

impl PartitionedOptions for crate::WriteOptions {
    fn with_by(mut self, by: Vec<usize>) -> crate::WriteOptions {
        self.partition_by = by;
        self
    }
    fn with_names(mut self, names: Vec<&str>) -> crate::WriteOptions {
        self.partition_names = names.into_iter().map(str::to_owned).collect();
        self
    }
}
