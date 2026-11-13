// SPDX-License-Identifier: Apache-2.0
//! The Avro format's tests: the documented type mapping, round trips,
//! options, splits, corrupt files and schema evolution.
//!
//! Ground truth: Spark's documented Avro mapping (option names and the
//! type table) and container files written by `apache-avro` itself, which
//! follow the Avro specification the same way Spark's `spark-avro` package
//! does. No Spark Connect oracle was reachable, so no oracle runs were
//! possible; the mapping assertions below are the documented one.

use std::sync::Arc;

use apache_avro::Schema as AvroSchema;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchOptions, StringViewArray,
    StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use krishiv_common::error::{ErrorClass, KrishivError};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path;

use super::AvroFormat;
use super::encode::AvroEncoder;
use super::schema::{fields_of_record, schema_json_of_arrow};
use crate::error::FormatError;
use crate::format::{Encoder, FileFormat};
use crate::io::{ByteRange, FileRef, IoConfig, IoContext, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::{MetadataNeeds, ScanOptions, ScanStream};

struct Env {
    rt: tokio::runtime::Runtime,
    io: IoContext,
    store: Arc<InMemory>,
}

const STORE: &str = "memory://avro";

impl Env {
    fn new() -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .unwrap();
        let store = Arc::new(InMemory::new());
        let sched = IoScheduler::new(IoConfig::default());
        sched.register_store(
            StoreId::new(STORE),
            Arc::clone(&store) as Arc<dyn ObjectStore>,
        );
        Self {
            rt,
            io: sched.context(),
            store,
        }
    }

    fn put(&self, name: &str, bytes: Vec<u8>) -> FileRef {
        self.rt.block_on(async {
            let path = Path::from(name);
            object_store::ObjectStoreExt::put(self.store.as_ref(), &path, bytes.into())
                .await
                .unwrap();
            let meta = object_store::ObjectStoreExt::head(self.store.as_ref(), &path)
                .await
                .unwrap();
            FileRef::from_meta(StoreId::new(STORE), &meta)
        })
    }

    fn options(&self, kv: &[(&str, &str)]) -> FormatOptions {
        let user = kv
            .iter()
            .map(|(k, v)| {
                (
                    (*k).to_owned(),
                    krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
                )
            })
            .collect();
        AvroFormat
            .options()
            .parse(&user, &krishiv_common::config::Config::new())
            .unwrap()
            .0
    }

    fn encode(&self, schema: &SchemaRef, batches: &[RecordBatch], o: &FormatOptions) -> Vec<u8> {
        let mut enc = AvroEncoder::new(schema, o).unwrap();
        let mut out = Vec::new();
        for b in batches {
            enc.write(b).unwrap();
            while let Some(bytes) = enc.take_bytes() {
                out.extend_from_slice(&bytes);
            }
        }
        let done = Box::new(enc).finish().unwrap();
        out.extend_from_slice(&done.tail);
        assert_eq!(
            done.statistics.rows.exact(),
            Some(&(batches.iter().map(RecordBatch::num_rows).sum::<usize>() as u64))
        );
        out
    }

    fn scan(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        o: &FormatOptions,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        self.scan_split(file, req, o, ByteRange::whole(file.size))
    }

    fn scan_split(
        &self,
        file: &FileRef,
        req: &ScanRequest,
        o: &FormatOptions,
        split: ByteRange,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        self.rt.block_on(async {
            let pool: Arc<dyn krishiv_memory::MemoryPool> =
                krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
            let id = pool.register("scan", None);
            let mut s = ScanStream::open(
                &self.io,
                &AvroFormat,
                file.clone(),
                split,
                req,
                o,
                krishiv_memory::Reservation::new(pool, id),
                ScanOptions::default(),
            )
            .await?;
            let mut out = Vec::new();
            while let Some(b) = s.next_batch().await? {
                out.push(b);
            }
            Ok(out)
        })
    }

    fn metadata(&self, file: &FileRef, o: &FormatOptions) -> Arc<dyn crate::FileMetadata> {
        self.rt
            .block_on(
                self.io
                    .read_metadata(&AvroFormat, file, MetadataNeeds::default(), o),
            )
            .unwrap()
    }

    fn infer(&self, file: &FileRef, o: &FormatOptions) -> SchemaRef {
        self.rt
            .block_on(AvroFormat.infer_schema(&self.io, std::slice::from_ref(file), o))
            .unwrap()
    }
}

fn rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn concat(schema: &SchemaRef, batches: &[RecordBatch]) -> RecordBatch {
    arrow_select::concat::concat_batches(schema, batches).unwrap()
}

/// `batch` re-typed to `schema`: the same buffers under recursively
/// nullable fields, as the [`SchemaMapper`](crate::SchemaMapper) types a
/// mapped scan's output (Spark reads file sources with `asNullable`).
fn retype_batch(batch: &RecordBatch, schema: &SchemaRef) -> RecordBatch {
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields().iter())
        .map(|(c, f)| retype_array(c, f))
        .collect();
    RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .unwrap()
}

fn retype_array(array: &ArrayRef, field: &Field) -> ArrayRef {
    match (array.data_type(), field.data_type()) {
        (DataType::Struct(_), DataType::Struct(fields)) => {
            let s = array.as_struct();
            Arc::new(StructArray::new(
                fields.clone(),
                fields
                    .iter()
                    .map(|f| retype_array(s.column_by_name(f.name()).unwrap(), f))
                    .collect(),
                s.nulls().cloned(),
            ))
        }
        (DataType::List(_), DataType::List(f)) => {
            let l = array.as_list::<i32>();
            Arc::new(arrow_array::ListArray::new(
                Arc::clone(f),
                l.offsets().clone(),
                retype_array(l.values(), f),
                l.nulls().cloned(),
            ))
        }
        (DataType::Map(_, _), DataType::Map(f, sorted)) => {
            let m = array.as_map();
            let entries = m.entries();
            let DataType::Struct(fs) = f.data_type() else {
                panic!("not a map field");
            };
            let rebuilt = StructArray::new(
                fs.clone(),
                vec![
                    retype_array(entries.column(0), &fs[0]),
                    retype_array(entries.column(1), &fs[1]),
                ],
                None,
            );
            Arc::new(arrow_array::MapArray::new(
                Arc::clone(f),
                m.offsets().clone(),
                rebuilt,
                m.nulls().cloned(),
                *sorted,
            ))
        }
        _ => Arc::clone(array),
    }
}

/// The mapped output schema of scanning `file` with `req`: what the
/// framework's batches carry.
fn mapped_schema(file_schema: &SchemaRef, req: &ScanRequest) -> SchemaRef {
    let mapper = crate::SchemaMapper::new(req, Arc::clone(file_schema), "test").unwrap();
    Arc::clone(mapper.output_schema())
}

/// A container file written by `apache-avro` itself (the Avro specification
/// framing, as Spark's `spark-avro` writes it) holding `values`.
fn apache_write(schema_json: &str, values: &[apache_avro::types::Value]) -> Vec<u8> {
    let schema = AvroSchema::parse_str(schema_json).unwrap();
    let mut writer =
        apache_avro::Writer::with_codec(&schema, Vec::new(), apache_avro::Codec::Null).unwrap();
    writer.extend_from_slice(values).unwrap();
    writer.into_inner().unwrap()
}

/// The `apache-avro` values of an encoded file of ours.
fn apache_read(bytes: &[u8]) -> Vec<apache_avro::types::Value> {
    apache_avro::Reader::new(bytes)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn option_value(key: &str, value: &str) -> krishiv_common::secret::OptionValue {
    krishiv_common::secret::OptionValue::from_user(key, value.to_owned())
}

fn class_of(e: &FormatError) -> ErrorClass {
    e.clone().into_error("test").error_class()
}

/// A list column with the `element` field name.
fn list_array(rows: &[Option<Vec<Option<i64>>>]) -> arrow_array::ListArray {
    let field = Arc::new(Field::new("element", DataType::Int64, false));
    let values: Vec<Option<i64>> = rows.iter().flatten().flatten().copied().collect();
    let mut offsets = vec![0_i32];
    let mut validity = Vec::new();
    for row in rows {
        match row {
            Some(items) => {
                offsets.push(
                    offsets.last().copied().unwrap_or(0) + i32::try_from(items.len()).unwrap(),
                );
                validity.push(true);
            }
            None => {
                offsets.push(offsets.last().copied().unwrap_or(0));
                validity.push(false);
            }
        }
    }
    arrow_array::ListArray::new(
        field,
        OffsetBuffer::new(offsets.into()),
        Arc::new(Int64Array::from(values)),
        Some(validity.into()),
    )
}

/// One map row: `None` is null, else the entries.
type MapRow = Option<Vec<(String, Option<i32>)>>;

/// A map column of `rows`: `None` is null, else the entries.
fn map_array(rows: &[MapRow]) -> arrow_array::MapArray {
    let mut offsets = vec![0_i32];
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut validity = Vec::new();
    for row in rows {
        match row {
            Some(entries) => {
                for (k, v) in entries {
                    keys.push(Some(k.as_str()));
                    values.push(*v);
                }
                offsets.push(i32::try_from(keys.len()).unwrap());
                validity.push(true);
            }
            None => {
                offsets.push(i32::try_from(keys.len()).unwrap());
                validity.push(false);
            }
        }
    }
    let keys = StringViewArray::from(keys);
    let values = Int32Array::from(values);
    let entries = StructArray::new(
        Fields::from(vec![
            Field::new("key", DataType::Utf8View, false),
            Field::new("value", DataType::Int32, false),
        ]),
        vec![Arc::new(keys) as ArrayRef, Arc::new(values) as ArrayRef],
        None,
    );
    arrow_array::MapArray::new(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8View, false),
                    Field::new("value", DataType::Int32, false),
                ]
                .into(),
            ),
            false,
        )),
        OffsetBuffer::new(offsets.into()),
        entries,
        Some(validity.into()),
        false,
    )
}

fn env_options_none() -> FormatOptions {
    AvroFormat
        .options()
        .parse(
            &std::collections::BTreeMap::new(),
            &krishiv_common::config::Config::new(),
        )
        .unwrap()
        .0
}

/// Like [`apache_write`], with `meta` added to the header.
fn apache_write_meta(
    schema_json: &str,
    values: &[apache_avro::types::Value],
    meta: &[(&str, &str)],
) -> Vec<u8> {
    let schema = AvroSchema::parse_str(schema_json).unwrap();
    let mut writer =
        apache_avro::Writer::with_codec(&schema, Vec::new(), apache_avro::Codec::Null).unwrap();
    for (key, value) in meta {
        writer
            .add_user_metadata((*key).to_owned(), value.as_bytes())
            .unwrap();
    }
    writer.extend_from_slice(values).unwrap();
    writer.into_inner().unwrap()
}

/// A file with an ancient date and timestamp, and the schema it was written
/// with (`apache_write_meta` adds the header entries under test).
fn ancient_file(env: &Env, name: &str, meta: &[(&str, &str)]) -> FileRef {
    use apache_avro::types::Value as V;
    let schema_json = r#"{
        "type": "record", "name": "r", "fields": [
            {"name": "d", "type": {"type": "int", "logicalType": "date"}},
            {"name": "t", "type": {"type": "long", "logicalType": "timestamp-micros"}}
        ]
    }"#;
    let values = vec![V::Record(vec![
        ("d".to_owned(), V::Date(-200_000)),
        ("t".to_owned(), V::TimestampMicros(-30_000_000_000_000)),
    ])];
    env.put(name, apache_write_meta(schema_json, &values, meta))
}

/// The zigzag varint of `n` (Avro's `long` encoding, for hand-built
/// headers in tests).
fn zigzag(n: i64) -> Vec<u8> {
    let mut raw = n
        .unsigned_abs()
        .wrapping_mul(2)
        .wrapping_sub(u64::from(n.is_negative()));
    let mut out = Vec::new();
    loop {
        let b = (raw & 0x7F) as u8;
        raw >>= 7;
        if raw == 0 {
            out.push(b);
            return out;
        }
        out.push(b | 0x80);
    }
}

/// `bytes` with the header entry `key`'s value replaced by `new_value`.
fn splice_header_value(bytes: &[u8], key: &[u8], new_value: &[u8]) -> Vec<u8> {
    use super::decode::parse_long;
    let at = bytes
        .windows(key.len())
        .position(|w| w == key)
        .unwrap_or_else(|| panic!("no {key:?} in the header"));
    let value_at = at + key.len();
    let (old_len, n) = parse_long(bytes, value_at).unwrap();
    let old_len = usize::try_from(old_len).unwrap();
    let mut replacement = zigzag(i64::try_from(new_value.len()).unwrap());
    replacement.extend_from_slice(new_value);
    let mut out = bytes.to_vec();
    out.splice(value_at..value_at + n + old_len, replacement);
    out
}

/// The scan of `file` fails: footer and record reasons travel in the
/// error class (and the logs), so scans assert the class.
fn scan_err(env: &Env, file: &FileRef, req: &ScanRequest, o: &FormatOptions) -> KrishivError {
    env.scan(file, req, o).unwrap_err()
}

mod part1;
mod part2;
mod part3;
