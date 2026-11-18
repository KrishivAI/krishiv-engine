// SPDX-License-Identifier: Apache-2.0
//! A gzip JSON Lines file cut in half: the scan returns the complete lines
//! decoded before the cut and then fails; inference under `ignoreCorruptFiles`
//! keeps those lines, and without it names the broken file (RS-26 follow-up;
//! the CSV side, with Spark 4.2.0's own files, is `csv/corrupt_tests.rs`).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;

use super::JsonFormat;
use super::decode::JsonDecoder;
use crate::format::{Decoder, FileFormat, FileMetadata, Step};
use crate::io::{ByteRange, FileRef, IoConfig, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::text::TextMetadata;
use crate::text::codec::Codec;

/// 1,000 lines `{"id":i}`, gzip, cut to half its length; and the complete
/// lines a stream decoder recovers from the cut file (the oracle).
fn cut_file() -> (Vec<u8>, i64, i64) {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    for i in 0..1000 {
        writeln!(enc, "{{\"id\":{i}}}").expect("write");
    }
    let whole = enc.finish().expect("gzip");
    let cut = whole[..whole.len() / 2].to_vec();
    let mut plain = Vec::new();
    let mut dec = flate2::read::GzDecoder::new(cut.as_slice());
    let mut buf = [0u8; 4096];
    loop {
        match dec.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => plain.extend_from_slice(&buf[..n]),
        }
    }
    let complete = &plain[..plain.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1)];
    let lines = complete.split(|b| *b == b'\n').filter(|l| !l.is_empty());
    let (mut n, mut sum) = (0, 0);
    for (i, _) in lines.enumerate() {
        n += 1;
        sum += i64::try_from(i).expect("index");
    }
    assert!(n > 0 && n < 1000, "the cut keeps some lines: {n}");
    (cut, n, sum)
}

fn options(pairs: &[(&str, &str)]) -> FormatOptions {
    let user = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), OptionValue::from_user(k, (*v).to_owned())))
        .collect::<BTreeMap<_, _>>();
    JsonFormat
        .options()
        .parse(&user, &Config::new())
        .expect("options")
        .0
}

fn decode(bytes: &[u8]) -> (Vec<RecordBatch>, Option<KrishivError>) {
    let m: Arc<dyn FileMetadata> = Arc::new(TextMetadata {
        size: bytes.len() as u64,
        codec: Some(Codec::Gzip),
    });
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
    let req = ScanRequest::all(schema);
    let split = ByteRange::whole(bytes.len() as u64);
    let units = JsonFormat
        .plan_units(m.as_ref(), &split, &req)
        .expect("units");
    let mut d = JsonDecoder::new(&m, &units, &req, &options(&[])).expect("decoder");
    let mut out = Vec::new();
    loop {
        match d.step() {
            Ok(Step::Done) => return (out, None),
            Ok(Step::Batch(b)) => out.push(b),
            Ok(Step::NeedData(ranges)) => {
                let data = ranges
                    .into_iter()
                    .map(|r| {
                        let a = usize::try_from(r.start).expect("offset");
                        let b = usize::try_from(r.end).expect("offset");
                        (r, Bytes::copy_from_slice(&bytes[a..b]))
                    })
                    .collect();
                if let Err(e) = d.push(data) {
                    return (out, Some(e.into_error("t.json.gz")));
                }
            }
            Err(e) => return (out, Some(e.into_error("t.json.gz"))),
        }
    }
}

#[test]
fn a_broken_gzip_json_file_returns_its_complete_lines_then_fails() {
    let (cut, n, sum) = cut_file();
    let (batches, e) = decode(&cut);
    let ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_primitive::<Int64Type>()
                .iter()
                .flatten()
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!((ids.len() as i64, ids.iter().sum::<i64>()), (n, sum));
    let e = e.expect("the break is raised after the rows");
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
}

fn store(bytes: Vec<u8>) -> (tokio::runtime::Runtime, Arc<IoScheduler>, FileRef) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime");
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://json");
    sched.register_store(
        id.clone(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let file = rt.block_on(async {
        let path = object_store::path::Path::from("cut.json.gz");
        object_store::ObjectStoreExt::put(store.as_ref(), &path, bytes.into())
            .await
            .expect("put");
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .expect("head");
        FileRef::from_meta(id, &meta)
    });
    (rt, sched, file)
}

fn infer(bytes: Vec<u8>, o: &FormatOptions) -> Result<Arc<Schema>, KrishivError> {
    let (rt, sched, file) = store(bytes);
    let io = sched.context();
    rt.block_on(JsonFormat.infer_schema(&io, std::slice::from_ref(&file), o))
        .map_err(|e| e.into_error("<other file>"))
}

#[test]
fn json_inference_keeps_the_lines_of_a_broken_file_only_under_ignore_corrupt_files() {
    let (cut, _, _) = cut_file();
    let schema = infer(cut.clone(), &options(&[("ignoreCorruptFiles", "true")])).expect("inferred");
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(0).data_type(), &DataType::Int64);
    let e = infer(cut, &options(&[])).expect_err("broken file");
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    let text = e.to_string();
    assert!(
        text.contains("cut.json.gz") && !text.contains("<other file>"),
        "{text}"
    );
}
