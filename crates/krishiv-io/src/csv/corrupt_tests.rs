// SPDX-License-Identifier: Apache-2.0
//! A gzip CSV file cut in half (RS-26 follow-up): Spark 4.2.0 keeps the lines
//! its codec decoded before the break, in the scan under
//! `spark.sql.files.ignoreCorruptFiles` and in `inferSchema`. Fixtures and
//! Spark's output: `tests/fixtures/spark-4.2.0/ignore-corrupt-csv/`
//! (`tests/scripts/ignore_corrupt_parquet.scala`): over `good.csv.gz` (1,000
//! rows) and `truncated.csv.gz`, 1,474 rows, `sum(id)` 611,601.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;

use super::CsvFormat;
use super::decode::CsvDecoder;
use crate::format::{Decoder, FileFormat, FileMetadata, Step};
use crate::io::{ByteRange, FileRef, IoConfig, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::ScanRequest;
use crate::text::TextMetadata;
use crate::text::codec::Codec;

fn fixture(name: &str) -> Vec<u8> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/spark-4.2.0/ignore-corrupt-csv");
    std::fs::read(dir.join(name)).expect("fixture")
}

fn options(pairs: &[(&str, &str)], session: &Config) -> FormatOptions {
    let user = pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), OptionValue::from_user(k, (*v).to_owned())))
        .collect::<BTreeMap<_, _>>();
    CsvFormat
        .options()
        .parse(&user, session)
        .expect("options")
        .0
}

/// Decodes a whole gzip file: the batches returned, then the error, if any.
fn decode(bytes: &[u8], o: &FormatOptions) -> (Vec<RecordBatch>, Option<KrishivError>) {
    let m: Arc<dyn FileMetadata> = Arc::new(TextMetadata {
        size: bytes.len() as u64,
        codec: Some(Codec::Gzip),
    });
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("s", DataType::Int32, true),
    ]));
    let req = ScanRequest::all(schema);
    let split = ByteRange::whole(bytes.len() as u64);
    let units = CsvFormat
        .plan_units(m.as_ref(), &split, &req)
        .expect("units");
    let mut d = CsvDecoder::new(&m, &units, &req, o).expect("decoder");
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
                        let slice = Bytes::copy_from_slice(&bytes[a..b]);
                        (r, slice)
                    })
                    .collect();
                if let Err(e) = d.push(data) {
                    return (out, Some(e.into_error("t.csv.gz")));
                }
            }
            Err(e) => return (out, Some(e.into_error("t.csv.gz"))),
        }
    }
}

fn ids(batches: &[RecordBatch]) -> (usize, i64) {
    let mut n = 0;
    let mut sum = 0;
    for b in batches {
        let ids = b.column(0).as_primitive::<Int32Type>();
        n += b.num_rows();
        sum += ids.iter().flatten().map(i64::from).sum::<i64>();
    }
    (n, sum)
}

#[test]
fn a_broken_gzip_file_returns_its_complete_lines_then_fails() {
    let o = options(&[("header", "true")], &Config::new());
    let (batches, e) = decode(&fixture("truncated.csv.gz"), &o);
    // The 474 complete rows after the header (Spark: 1,474 - 1,000, sum
    // 611,601 - 499,500); the partial line `474,4` is dropped.
    assert_eq!(ids(&batches), (474, 112_101));
    let e = e.expect("the break is raised after the rows");
    assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
    assert!(e.to_string().contains("t.csv.gz"), "{e}");
}

#[test]
fn an_intact_gzip_file_is_read_whole() {
    let o = options(&[("header", "true")], &Config::new());
    let (batches, e) = decode(&fixture("good.csv.gz"), &o);
    assert!(e.is_none(), "{e:?}");
    assert_eq!(ids(&batches), (1000, 499_500));
}

/// Both fixtures as `a.csv.gz` and `b.csv.gz` in one store, in `order`.
fn store_two(first_cut: bool) -> (tokio::runtime::Runtime, Arc<IoScheduler>, Vec<FileRef>) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime");
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://csv");
    sched.register_store(
        id.clone(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let (good, cut) = (fixture("good.csv.gz"), fixture("truncated.csv.gz"));
    let named = if first_cut {
        [("a.csv.gz", cut), ("b.csv.gz", good)]
    } else {
        [("a.csv.gz", good), ("b.csv.gz", cut)]
    };
    let files: Vec<FileRef> = rt.block_on(async {
        let mut files = Vec::new();
        for (name, bytes) in named {
            let path = object_store::path::Path::from(name);
            object_store::ObjectStoreExt::put(store.as_ref(), &path, bytes.into())
                .await
                .expect("put");
            let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
                .await
                .expect("head");
            files.push(FileRef::from_meta(id.clone(), &meta));
        }
        files
    });
    (rt, sched, files)
}

fn infer_two(first_cut: bool, o: &FormatOptions) -> Result<Arc<Schema>, KrishivError> {
    let (rt, sched, files) = store_two(first_cut);
    let io = sched.context();
    rt.block_on(CsvFormat.infer_schema(&io, &files, o))
        .map_err(|e| e.into_error("<first file>"))
}

#[test]
fn inference_keeps_the_lines_of_a_broken_file_under_ignore_corrupt_files() {
    let mut session = Config::new();
    session
        .set("spark.sql.files.ignoreCorruptFiles", "true")
        .expect("set");
    let by_conf = options(&[("header", "true"), ("inferSchema", "true")], &session);
    let by_option = options(
        &[
            ("header", "true"),
            ("inferSchema", "true"),
            ("ignoreCorruptFiles", "true"),
        ],
        &Config::new(),
    );
    for o in [&by_conf, &by_option] {
        for first_cut in [false, true] {
            let schema = infer_two(first_cut, o).expect("inferred");
            let types: Vec<_> = schema.fields().iter().map(|f| f.data_type()).collect();
            // Spark: struct<id:int,s:int>.
            assert_eq!(types, [&DataType::Int32, &DataType::Int32], "{first_cut}");
        }
    }
}

#[test]
fn an_inference_error_names_the_broken_file() {
    let o = options(
        &[("header", "true"), ("inferSchema", "true")],
        &Config::new(),
    );
    for first_cut in [false, true] {
        let e = infer_two(first_cut, &o).expect_err("broken file");
        assert_eq!(e.error_class(), ErrorClass::FailedReadFileNoHint);
        let broken = if first_cut { "a.csv.gz" } else { "b.csv.gz" };
        let text = e.to_string();
        assert!(
            text.contains(broken) && !text.contains("<first file>"),
            "{text}"
        );
    }
}

#[test]
fn a_reads_ignore_corrupt_files_option_reaches_the_scan_of_csv_and_json() {
    // CSV and JSON declare `ignoreCorruptFiles` themselves (inference reads
    // it); the scan still takes the read's value over the session conf.
    let mut on = Config::new();
    on.set("spark.sql.files.ignoreCorruptFiles", "true")
        .expect("set");
    let user = |v: Option<&str>| {
        v.map(|v| {
            (
                "ignoreCorruptFiles".to_owned(),
                OptionValue::from_user("ignoreCorruptFiles", v.to_owned()),
            )
        })
        .into_iter()
        .collect::<BTreeMap<_, _>>()
    };
    for format in [&CsvFormat as &dyn FileFormat, &crate::json::JsonFormat] {
        for (given, session, want) in [
            (Some("false"), &on, false),
            (Some("true"), &Config::new(), true),
            (None, &on, true),
            (None, &Config::new(), false),
        ] {
            let o = format
                .options()
                .parse(&user(given), session)
                .expect("options")
                .0;
            let s = crate::scan::ScanOptions::for_read(session, &o);
            assert_eq!(s.ignore_corrupt_files, want, "{given:?}");
        }
        // Read as the framework reads it: not trimmed (Spark 4.2.0).
        let e = format
            .options()
            .parse(&user(Some(" true ")), &Config::new())
            .expect_err("padded");
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_FORMAT_OPTION_VALUE"
        );
    }
}
