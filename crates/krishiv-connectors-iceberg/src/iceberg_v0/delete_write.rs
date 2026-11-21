// SPDX-License-Identifier: Apache-2.0
//! The delete files a merge-on-read write produces (P3-T027, PR-36c part 1;
//! design note §8 item 4), as apache-iceberg 1.12.0 writes them (notes 36c
//! §2, §3):
//!
//! - **Deletion vectors** (format v3): one Puffin file per task holding one
//!   `deletion-vector-v1` blob per data file ([`dv_puffin`], Java's
//!   `BaseDVFileWriter.close`, `PuffinWriter`, `PuffinFormat`): the magic
//!   `PFA1`, the blobs from offset 4 in the order given (Java iterates a
//!   `HashMap`; Krishiv's callers sort by data file path), then the footer:
//!   `PFA1`, the UTF-8 JSON payload uncompressed, its length as a
//!   little-endian `u32`, four zero flag bytes, `PFA1`. The payload is
//!   `{"blobs":[...],"properties":{"created-by":...}}`, each blob
//!   `{"type":"deletion-vector-v1","fields":[2147483645],"snapshot-id":-1,
//!   "sequence-number":-1,"offset":o,"length":n,"properties":
//!   {"referenced-data-file":p,"cardinality":"k"}}` with no
//!   `compression-codec` (`FileMetadataParser`, `BaseDVFileWriter.java:199-212`).
//!   Each blob is [`encode_dv`]'s, Java's layout. A DV's manifest entry
//!   ([`dv_file`], `BaseDVFileWriter.java:165-181`): content position
//!   deletes, format `puffin`, the data file's spec and partition, the whole
//!   Puffin file's size, `referenced_data_file`, `content_offset` and
//!   `content_size_in_bytes` from the footer, `record_count` the cardinality,
//!   no metrics. `created-by` is `Krishiv <version>` (Java writes its build).
//! - **Position delete files** (format v2): Parquet with `file_path`
//!   (2147483546, required string) and `pos` (2147483545, required long),
//!   sorted by both, one file per data file (Spark's default `FILE`
//!   granularity) ([`write_position_deletes`]); `referenced_data_file` unset;
//!   the record count, and lower and upper bounds of both columns kept
//!   (`file_path`'s equal: the file is file-scoped, `ContentFileUtil
//!   .referencedDataFile`) and every count dropped
//!   (`PositionDeleteWriter.metrics`, `copyWithoutFieldCounts`; Java keeps
//!   the column sizes there, Krishiv writes none: a divergence, review 36c1
//!   nit 9). Java's
//!   `MetricsConfig.forPositionDelete` keeps both bounds in full (recalled;
//!   with one path per file the bound is the path either way). Parquet
//!   settings are the data files' (`write.parquet.*`) with each
//!   `write.delete.parquet.*` key overriding its data key, as Java's
//!   `Parquet.DeleteWriteBuilder` reads them (recalled, unverified). No
//!   `split_offsets` (as Krishiv's data files).
//!
//! Both are encoded in memory before they are written (a DV is a few bytes
//! per run of positions, a position delete file a few bytes per position);
//! that memory is not charged to the query (a known limit).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use iceberg::io::FileIO;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, Datum, Literal, PrimitiveType,
    Struct, TableMetadata, Type,
};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{FileFormat as _, IoContext};
use roaring::RoaringTreemap;

use super::storage::from_iceberg;
use super::write::parquet_options;
use crate::plan::{DELETE_FILE_PATH_ID, DELETE_FILE_POS_ID, PartitionValue, encode_dv};

/// Java's `StandardBlobTypes.DV_V1`.
pub(crate) const DV_V1: &str = "deletion-vector-v1";

/// `MetadataColumns.ROW_POSITION`'s id (`Integer.MAX_VALUE - 2`), a DV
/// blob's one input field.
pub(crate) const ROW_POSITION_ID: i32 = i32::MAX - 2;

/// The Puffin magic (`PuffinFormat.MAGIC`).
const MAGIC: &[u8; 4] = b"PFA1";

/// Positions per batch of a position delete file.
const BATCH_ROWS: usize = 64 * 1024;

/// Where one DV sits in its Puffin file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlobAt {
    /// The blob's offset in the file.
    pub(crate) offset: i64,
    /// Its length.
    pub(crate) length: i64,
    /// The positions it holds.
    pub(crate) cardinality: u64,
}

/// What Krishiv writes as a Puffin file's `created-by`.
pub(crate) fn created_by() -> String {
    format!("Krishiv {}", env!("CARGO_PKG_VERSION"))
}

/// The bytes of a Puffin file holding one DV per `(data file, positions)` of
/// `dvs`, in that order, and where each blob is (module docs).
///
/// # Errors
/// [`encode_dv`]'s; an internal error for a file over the long range.
pub(crate) fn dv_puffin(
    dvs: &[(&str, &RoaringTreemap)],
    created_by: &str,
) -> Result<(Vec<u8>, Vec<BlobAt>), KrishivError> {
    let mut out = MAGIC.to_vec();
    let mut blobs = Vec::with_capacity(dvs.len());
    let mut json = String::from("{\"blobs\":[");
    for (i, (path, positions)) in dvs.iter().enumerate() {
        let blob = encode_dv(positions)?;
        let at = BlobAt {
            offset: long(out.len())?,
            length: long(blob.len())?,
            cardinality: positions.len(),
        };
        out.extend_from_slice(&blob);
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"type\":\"{DV_V1}\",\"fields\":[{ROW_POSITION_ID}],\"snapshot-id\":-1,\
             \"sequence-number\":-1,\"offset\":{},\"length\":{},\"properties\":\
             {{\"referenced-data-file\":{},\"cardinality\":\"{}\"}}}}",
            at.offset,
            at.length,
            quoted(path)?,
            at.cardinality
        ));
        blobs.push(at);
    }
    json.push_str(&format!(
        "],\"properties\":{{\"created-by\":{}}}}}",
        quoted(created_by)?
    ));
    let payload = json.into_bytes();
    let length = u32::try_from(payload.len())
        .map_err(|_| internal("a Puffin footer over 4 GiB".to_owned()))?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&payload);
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(MAGIC);
    Ok((out, blobs))
}

/// `s` as a JSON string (Jackson's escaping: quotes, backslashes and
/// control characters; `/` and non-ASCII kept).
fn quoted(s: &str) -> Result<String, KrishivError> {
    serde_json::to_string(s).map_err(|e| internal(format!("a Puffin footer string: {e}")))
}

fn long(n: usize) -> Result<i64, KrishivError> {
    i64::try_from(n).map_err(|_| internal("a Puffin file over the long range".to_owned()))
}

/// The partition tuple `values` of the spec `spec_id` as an iceberg
/// `Struct` at the spec's current partition type: a value written before
/// `int -> long` or `float -> double` (4 bytes) is widened, as
/// `filter::widen_partition` does for a rewritten manifest.
///
/// # Errors
/// An internal error for a spec the table lacks, a tuple of another arity or
/// a value that does not decode; `KRISHIV_UNSUPPORTED_FEATURE` for an
/// unreadable value (never written as a guess).
pub(crate) fn partition_struct(
    metadata: &TableMetadata,
    spec_id: i32,
    values: &[PartitionValue],
) -> Result<Struct, KrishivError> {
    let spec = metadata
        .partition_spec_by_id(spec_id)
        .ok_or_else(|| internal(format!("a data file of the unknown spec {spec_id}")))?;
    let ty = spec
        .partition_type(metadata.current_schema())
        .map_err(from_iceberg)?;
    if ty.fields().len() != values.len() {
        return Err(internal(format!(
            "a partition tuple of {} values for spec {spec_id} of {} fields",
            values.len(),
            ty.fields().len()
        )));
    }
    let literals = ty
        .fields()
        .iter()
        .zip(values)
        .map(|(field, v)| {
            let bytes = match v {
                PartitionValue::Null => return Ok(None),
                PartitionValue::Unreadable => {
                    return Err(KrishivError::from(KrishivUnsupportedFeature {
                        feature: format!(
                            "a delete for a data file whose partition value of {} cannot be read",
                            field.name
                        ),
                    }));
                }
                PartitionValue::Value(b) => b,
            };
            let Type::Primitive(p) = field.field_type.as_ref() else {
                return Err(internal(format!(
                    "partition field {} is not primitive",
                    field.name
                )));
            };
            let narrow = |n: usize| bytes.get(..n).filter(|_| bytes.len() == n);
            let literal = match (p, narrow(4)) {
                (PrimitiveType::Long, Some(b)) => Literal::long(i64::from(i32::from_le_bytes(
                    b.try_into().map_err(|_| internal("int bytes".to_owned()))?,
                ))),
                (PrimitiveType::Double, Some(b)) => Literal::double(f64::from(f32::from_le_bytes(
                    b.try_into()
                        .map_err(|_| internal("float bytes".to_owned()))?,
                ))),
                _ => Literal::Primitive(
                    Datum::try_from_bytes(bytes, p.clone())
                        .map_err(from_iceberg)?
                        .literal()
                        .clone(),
                ),
            };
            Ok(Some(literal))
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    Ok(Struct::from_iter(literals))
}

/// The manifest entry of the DV `blob` at `puffin` (whole size `size`) for
/// the data file `data` (module docs).
///
/// # Errors
/// An internal error when iceberg-rust's builder refuses the entry.
pub(crate) fn dv_file(
    puffin: &str,
    size: u64,
    blob: BlobAt,
    data: &str,
    spec_id: i32,
    partition: Struct,
) -> Result<DataFile, KrishivError> {
    let mut b = DataFileBuilder::default();
    b.content(DataContentType::PositionDeletes)
        .file_path(puffin.to_owned())
        .file_format(DataFileFormat::Puffin)
        .partition(partition)
        .partition_spec_id(spec_id)
        .record_count(blob.cardinality)
        .file_size_in_bytes(size)
        .referenced_data_file(Some(data.to_owned()))
        .content_offset(Some(blob.offset))
        .content_size_in_bytes(Some(blob.length));
    b.build()
        .map_err(|e| internal(format!("the DV of {data} in {puffin}: {e}")))
}

/// The Parquet properties of a position delete file: the data files'
/// `write.parquet.*`, each overridden by its `write.delete.parquet.*` key.
fn delete_properties(props: &HashMap<String, String>) -> HashMap<String, String> {
    let mut out = props.clone();
    for (k, v) in props {
        if let Some(rest) = k.strip_prefix("write.delete.parquet.") {
            out.insert(format!("write.parquet.{rest}"), v.clone());
        }
    }
    out
}

/// The schema of a position delete file: `file_path` and `pos` under their
/// reserved ids, both required.
fn position_delete_schema() -> Arc<Schema> {
    let field = |name: &str, ty: DataType, id: i32| {
        Field::new(name, ty, false).with_metadata(HashMap::from([(
            krishiv_io::FIELD_ID_KEY.to_owned(),
            id.to_string(),
        )]))
    };
    Arc::new(Schema::new(vec![
        field("file_path", DataType::Utf8, DELETE_FILE_PATH_ID),
        field("pos", DataType::Int64, DELETE_FILE_POS_ID),
    ]))
}

/// What [`write_position_deletes`] writes through.
pub(crate) struct DeleteTarget<'a> {
    /// The table's I/O (the Parquet encoder).
    pub(crate) io: &'a IoContext,
    /// The table's file access (the file written).
    pub(crate) file_io: &'a FileIO,
    /// The table as the write pinned it (its properties).
    pub(crate) metadata: &'a TableMetadata,
}

/// Writes the position delete file `location` deleting `positions` (not
/// empty) of the data file `data` and returns its manifest entry (module
/// docs). The caller records `location` as written before the call, so a
/// failure's partial file is cleaned up with the rest.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a Parquet property Krishiv cannot
/// write; a position over the long range; the encoder's and the store's
/// errors.
pub(crate) async fn write_position_deletes(
    t: &DeleteTarget<'_>,
    location: &str,
    data: &str,
    spec_id: i32,
    partition: Struct,
    positions: &RoaringTreemap,
) -> Result<DataFile, KrishivError> {
    let (Some(min), Some(max)) = (positions.min(), positions.max()) else {
        return Err(internal(format!(
            "an empty position delete file for {data}"
        )));
    };
    let pos = |p: u64| {
        i64::try_from(p).map_err(|_| internal(format!("position {p} of {data} is not a long")))
    };
    let (min, max) = (pos(min)?, pos(max)?);
    let options = parquet_options(&delete_properties(t.metadata.properties()))?;
    let schema = position_delete_schema();
    let mut encoder = ParquetFormat
        .writer(
            t.io,
            Arc::clone(&schema),
            &options.with_write_seed(location.to_owned()),
        )
        .await
        .map_err(|e| e.into_error(location))?;
    let mut bytes: Vec<u8> = Vec::new();
    let mut chunk: Vec<i64> =
        Vec::with_capacity(BATCH_ROWS.min(usize::try_from(positions.len()).unwrap_or(BATCH_ROWS)));
    let mut iter = positions.iter().peekable();
    while iter.peek().is_some() {
        chunk.clear();
        for p in iter.by_ref().take(BATCH_ROWS) {
            chunk.push(pos(p)?);
        }
        let paths: ArrayRef = Arc::new(StringArray::from(vec![data; chunk.len()]));
        let values: ArrayRef = Arc::new(Int64Array::from(chunk.clone()));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![paths, values])
            .map_err(|e| internal(format!("a position delete batch: {e}")))?;
        encoder.write(&batch).map_err(|e| e.into_error(location))?;
        while let Some(b) = encoder.take_bytes() {
            bytes.extend_from_slice(&b);
        }
    }
    let done = encoder.finish().map_err(|e| e.into_error(location))?;
    bytes.extend_from_slice(&done.tail);
    let size = put(t.file_io, t.io.scheduler().runtime(), location, bytes).await?;
    let bounds = |p: i64| {
        HashMap::from([
            (DELETE_FILE_PATH_ID, Datum::string(data)),
            (DELETE_FILE_POS_ID, Datum::long(p)),
        ])
    };
    let mut b = DataFileBuilder::default();
    b.content(DataContentType::PositionDeletes)
        .file_path(location.to_owned())
        .file_format(DataFileFormat::Parquet)
        .partition(partition)
        .partition_spec_id(spec_id)
        .record_count(positions.len())
        .file_size_in_bytes(size)
        .lower_bounds(bounds(min))
        .upper_bounds(bounds(max));
    b.build()
        .map_err(|e| internal(format!("position delete file {location}: {e}")))
}

/// Writes `bytes` at `location` through the table's file access, on `io_rt`
/// (the table's I/O runtime: it outlives a query's compute runtime, which may
/// be shutting down when a cancelled driver drops the write; the current
/// runtime when there is none, as `sink.rs`'s `Written` and `write.rs`'s
/// guard choose).
///
/// # Errors
/// The store's errors.
pub(crate) async fn put(
    io: &FileIO,
    io_rt: Option<&tokio::runtime::Handle>,
    location: &str,
    bytes: Vec<u8>,
) -> Result<u64, KrishivError> {
    let size = bytes.len() as u64;
    let write = {
        let (io, location) = (io.clone(), location.to_owned());
        async move {
            io.new_output(&location)
                .map_err(from_iceberg)?
                .write(Bytes::from(bytes))
                .await
                .map_err(from_iceberg)
        }
    };
    let Some(rt) = io_rt
        .cloned()
        .or_else(|| tokio::runtime::Handle::try_current().ok())
    else {
        write.await?;
        return Ok(size);
    };
    // The store may finish a request whose future was dropped (a local
    // store's blocking rename, an HTTP request already sent), after the
    // caller's guard deleted the path. So the write runs on a task of its
    // own, and a `put` dropped midway deletes the file once it is done.
    let mut pending = Pending {
        task: Some(rt.spawn(write)),
        io: io.clone(),
        location: location.to_owned(),
        rt,
    };
    let joined = match pending.task.as_mut() {
        Some(task) => task.await,
        None => return Err(internal(format!("no write of {location}"))),
    };
    pending.task = None;
    joined.map_err(|e| internal(format!("the write of {location}: {e}")))??;
    Ok(size)
}

/// A [`put`] in flight: dropped before it is done, it deletes the file once
/// the write ends.
struct Pending {
    task: Option<tokio::task::JoinHandle<Result<(), KrishivError>>>,
    io: FileIO,
    location: String,
    rt: tokio::runtime::Handle,
}

impl Drop for Pending {
    fn drop(&mut self) {
        let Some(task) = self.task.take() else {
            return;
        };
        let (io, location) = (self.io.clone(), std::mem::take(&mut self.location));
        self.rt.spawn(async move {
            // Best effort: a file left behind is in no manifest.
            let _ = task.await;
            let _ = io.delete(&location).await;
        });
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/iceberg-1.12.0/wh/db/dvs/data")
                .join(name),
        )
        .unwrap()
    }

    const JAVA: &str = "Apache Iceberg 1.12.0 (commit 212aa894e919ff9fdf89160da4aadcbe6858ac01)";
    const A: &str = "/wh/db/dvs/data/00001-13-164b74b6-3952-4451-91a5-d75b28d6ffe4-0-00001.parquet";
    const B: &str = "/wh/db/dvs/data/00000-12-164b74b6-3952-4451-91a5-d75b28d6ffe4-0-00001.parquet";

    /// Two Puffin files Iceberg 1.12.0 wrote (Spark 4.1.0), rebuilt byte for
    /// byte from their positions, in their blob order and with their
    /// `created-by`.
    #[test]
    fn puffin_files_are_javas_bytes() {
        let a: RoaringTreemap = [0_u64, 4, 8, 12].into_iter().collect();
        let b: RoaringTreemap = [3_u64, 7, 11].into_iter().collect();
        let (bytes, blobs) = dv_puffin(&[(A, &a), (B, &b)], JAVA).unwrap();
        assert_eq!(
            bytes,
            fixture("00000-15-77972ffe-990c-40dc-bec0-dcf8b7ba1726-00001-deletes.puffin")
        );
        assert_eq!(
            blobs,
            [
                BlobAt {
                    offset: 4,
                    length: 48,
                    cardinality: 4
                },
                BlobAt {
                    offset: 52,
                    length: 46,
                    cardinality: 3
                }
            ]
        );
        let c: RoaringTreemap = [0_u64, 1, 3, 7, 11].into_iter().collect();
        let (bytes, _) = dv_puffin(&[(B, &c)], JAVA).unwrap();
        assert_eq!(
            bytes,
            fixture("00000-17-5ab633b0-5fb4-4385-91a2-058f88cc71f6-00001-deletes.puffin")
        );
    }

    /// Paths are JSON-escaped as Jackson escapes them.
    #[test]
    fn footer_strings_are_escaped() {
        let p: RoaringTreemap = [1_u64].into_iter().collect();
        let (bytes, _) = dv_puffin(&[("s3://b/a \"q\"\\x/é", &p)], "k").unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains(r#""referenced-data-file":"s3://b/a \"q\"\\x/é""#),
            "{text}"
        );
    }

    #[test]
    fn delete_parquet_keys_override_data_keys() {
        let props = HashMap::from([
            (
                "write.parquet.compression-codec".to_owned(),
                "gzip".to_owned(),
            ),
            (
                "write.delete.parquet.compression-codec".to_owned(),
                "zstd".to_owned(),
            ),
        ]);
        assert_eq!(
            delete_properties(&props)
                .get("write.parquet.compression-codec")
                .map(String::as_str),
            Some("zstd")
        );
    }
}
