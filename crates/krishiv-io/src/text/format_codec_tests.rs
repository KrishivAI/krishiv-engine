// SPDX-License-Identifier: Apache-2.0
//! CSV, JSON and text files in Hadoop codecs, through the `FileFormat`
//! API: files Spark 4.2 wrote read like the uncompressed file, and files
//! written here read back.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use krishiv_common::error::ErrorClass;

use crate::csv::CsvFormat;
use crate::format::{FileFormat, MetadataNeeds, Step};
use crate::io::{ByteRange, FileRef, IoConfig, IoScheduler, StoreId};
use crate::json::JsonFormat;
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::text_format::TextFormat;

fn fixture(name: &str) -> Vec<u8> {
    let p = format!(
        "{}/../../tests/fixtures/spark-4.2.0/compressed-text/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn opts(format: &str, pairs: &[(&str, &str)]) -> FormatOptions {
    let user = pairs
        .iter()
        .map(|(k, v)| {
            (
                (*k).to_owned(),
                krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
            )
        })
        .collect();
    fmt(format)
        .options()
        .parse(&user, &krishiv_common::config::Config::new())
        .expect("options parse")
        .0
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime")
}

fn schema(format: &str) -> SchemaRef {
    match format {
        "text" => Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8View,
            true,
        )])),
        _ => Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("s", DataType::Utf8View, true),
            Field::new("t", DataType::Utf8View, true),
        ])),
    }
}

fn fmt(format: &str) -> Box<dyn FileFormat> {
    match format {
        "csv" => Box::new(CsvFormat),
        "json" => Box::new(JsonFormat),
        _ => Box::new(TextFormat),
    }
}

/// Reads `bytes` as the file `name`, split by split, through the format.
fn read_splits(
    format: &str,
    bytes: &[u8],
    name: &str,
    o: &FormatOptions,
    splits: &[ByteRange],
) -> Result<Vec<RecordBatch>, krishiv_common::error::KrishivError> {
    let f = fmt(format);
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://codec");
    sched.register_store(
        id.clone(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let path = object_store::path::Path::from(name);
    let rt = rt();
    let file = rt.block_on(async {
        object_store::ObjectStoreExt::put(store.as_ref(), &path, bytes.to_vec().into())
            .await
            .expect("put");
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .expect("head");
        FileRef::from_meta(id, &meta)
    });
    let io = sched.context();
    let req = ScanRequest::all(schema(format));
    let mut out = Vec::new();
    for split in splits {
        let err = |e: crate::error::FormatError| e.into_error(name);
        let m = rt
            .block_on(f.read_metadata(&io, &file, MetadataNeeds::default(), o))
            .map_err(err)?;
        let units = f.plan_units(m.as_ref(), split, &req).map_err(err)?;
        let mut d = rt.block_on(f.open(&io, m, units, &req, o)).map_err(err)?;
        loop {
            match d.step().map_err(err)? {
                Step::Done => break,
                Step::Batch(b) => out.push(b),
                Step::NeedData(ranges) => {
                    let data = ranges
                        .into_iter()
                        .map(|r| {
                            let (a, b) = (
                                usize::try_from(r.start).expect("start"),
                                usize::try_from(r.end).expect("end"),
                            );
                            (r, Bytes::copy_from_slice(&bytes[a..b]))
                        })
                        .collect();
                    d.push(data).map_err(err)?;
                }
            }
        }
    }
    Ok(out)
}

fn rows(batches: &[RecordBatch]) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows()).map(move |i| {
                b.columns()
                    .iter()
                    .map(|c| cell(c, i))
                    .collect::<Vec<_>>()
                    .join("|")
            })
        })
        .collect()
}

fn cell(c: &ArrayRef, i: usize) -> String {
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    if c.is_null(i) {
        return "<null>".to_owned();
    }
    match c.data_type() {
        DataType::Int32 => c
            .as_primitive::<arrow_array::types::Int32Type>()
            .value(i)
            .to_string(),
        _ => c.as_string_view().value(i).to_owned(),
    }
}

fn whole(n: usize) -> Vec<ByteRange> {
    vec![ByteRange::whole(n as u64)]
}

fn read_options(format: &str) -> FormatOptions {
    match format {
        "csv" => opts(format, &[("multiLine", "true")]),
        _ => opts(format, &[]),
    }
}

#[test]
fn spark_written_files_read_like_the_plain_file() {
    for (format, ext) in [("csv", "csv"), ("json", "json"), ("text", "txt")] {
        let o = read_options(format);
        let plain = fixture(&format!("{format}.{ext}"));
        let want =
            rows(&read_splits(format, &plain, "a/p.x", &o, &whole(plain.len())).expect("plain"));
        assert!(want.len() >= 3, "{format}: {want:?}");
        for suffix in [".gz", ".deflate", ".bz2", ".lz4", ".snappy"] {
            let packed = fixture(&format!("{format}.{ext}{suffix}"));
            let name = format!("a/f.{ext}{suffix}");
            let got = rows(
                &read_splits(format, &packed, &name, &o, &whole(packed.len()))
                    .unwrap_or_else(|e| panic!("{name}: {e:?}")),
            );
            assert_eq!(got, want, "{name}");
            // A planner that cuts the compressed file in pieces (bzip2 is
            // splittable for Spark) still reads every row once.
            let n = packed.len() as u64;
            let cut = [ByteRange::new(0, n / 2), ByteRange::new(n / 2, n)];
            let got = rows(
                &read_splits(format, &packed, &name, &o, &cut)
                    .unwrap_or_else(|e| panic!("{name}: {e:?}")),
            );
            assert_eq!(got, want, "{name} split");
        }
    }
}

fn batch(format: &str) -> RecordBatch {
    let s = schema(format);
    if format == "text" {
        let v = StringViewArray::from(vec![Some("alpha"), None, Some("c,d \"q\""), Some("é")]);
        return RecordBatch::try_new(s, vec![Arc::new(v) as ArrayRef]).expect("batch");
    }
    let id = arrow_array::Int32Array::from(vec![Some(1), Some(2), None, Some(4)]);
    let a = StringViewArray::from(vec![Some("a,b"), None, Some("é"), Some("x\"y")]);
    let b = StringViewArray::from(vec![Some(""), Some("multi\nline"), Some("z"), None]);
    RecordBatch::try_new(s, vec![Arc::new(id), Arc::new(a), Arc::new(b)]).expect("batch")
}

fn write(format: &str, o: &FormatOptions) -> Vec<u8> {
    let f = fmt(format);
    let sched = IoScheduler::new(IoConfig::default());
    let io = sched.context();
    let b = batch(format);
    let mut e = rt().block_on(f.writer(&io, b.schema(), o)).expect("writer");
    let mut out = Vec::new();
    for _ in 0..3 {
        e.write(&b).expect("write");
        while let Some(x) = e.take_bytes() {
            out.extend_from_slice(&x);
        }
    }
    let w = e.finish().expect("finish");
    out.extend_from_slice(&w.tail);
    out
}

#[test]
fn written_compressed_files_read_back() {
    for format in ["csv", "json", "text"] {
        let plain_o = opts(format, &[]);
        let plain = write(format, &plain_o);
        let ro = read_options(format);
        let want =
            rows(&read_splits(format, &plain, "p.x", &ro, &whole(plain.len())).expect("plain"));
        assert_eq!(want.len(), 12);
        for (codec, suffix) in [
            ("gzip", ".gz"),
            ("GZIP", ".gz"),
            ("deflate", ".deflate"),
            ("bzip2", ".bz2"),
            ("lz4", ".lz4"),
            ("snappy", ".snappy"),
        ] {
            let o = opts(format, &[("compression", codec)]);
            let base = fmt(format).file_suffix(&plain_o);
            assert_eq!(fmt(format).file_suffix(&o), format!("{base}{suffix}"));
            let packed = write(format, &o);
            let name = format!("f{}", fmt(format).file_suffix(&o));
            let got = rows(
                &read_splits(format, &packed, &name, &ro, &whole(packed.len()))
                    .unwrap_or_else(|e| panic!("{name}: {e:?}")),
            );
            assert_eq!(got, want, "{name}");
            // `codec` is the alias.
            assert_eq!(write(format, &opts(format, &[("codec", codec)])), packed);
        }
        for none in ["none", "uncompressed"] {
            assert_eq!(
                write(format, &opts(format, &[("compression", none)])),
                plain
            );
        }
    }
}

#[test]
fn unknown_codecs_are_spark_errors_and_zstd_files_are_refused() {
    for format in ["csv", "json", "text"] {
        for name in ["zstd", "zstandard", "lzo", "bogus"] {
            let e = fmt(format)
                .check_write(&schema(format), &opts(format, &[("compression", name)]))
                .expect_err(name);
            let e = e.into_error("x");
            assert_eq!(
                e.error_class(),
                ErrorClass::CodecNotAvailableWithAvailableCodecsSuggestion,
                "{format} {name}"
            );
        }
        let e = read_splits(format, b"x", "f.zst", &opts(format, &[]), &whole(1)).expect_err("zst");
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }
}

#[test]
fn a_corrupt_compressed_file_is_an_error() {
    for format in ["csv", "json", "text"] {
        let e = read_splits(
            format,
            b"not gzip data",
            "f.gz",
            &opts(format, &[]),
            &whole(13),
        );
        assert!(e.is_err(), "{format}");
    }
}

/// Writes one file per format and codec into `$KRISHIV_CODEC_DUMP`, for
/// `tests/scripts/check_compressed_text_written.py` (Spark 4.2 reads them).
#[test]
#[ignore = "writes files for the Spark check; run with KRISHIV_CODEC_DUMP=<dir>"]
fn dump_written_files_for_spark() {
    let Ok(dir) = std::env::var("KRISHIV_CODEC_DUMP") else {
        return;
    };
    for format in ["csv", "json", "text"] {
        for codec in ["none", "gzip", "deflate", "bzip2", "lz4", "snappy"] {
            let o = opts(format, &[("compression", codec)]);
            let name = format!("{dir}/{format}-{codec}{}", fmt(format).file_suffix(&o));
            std::fs::write(&name, write(format, &o)).expect("write");
        }
    }
}
