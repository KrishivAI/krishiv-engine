// SPDX-License-Identifier: Apache-2.0
//! The `text` format against lines and whole files: splits, encodings,
//! options, write checks and round trips.
//!
//! Behaviour is checked against Spark 4.2's documented `text` semantics
//! (Hadoop's line splitting, one `value` column).

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::OptionValue;

use super::TextFormat;
use crate::format::{Decoder, Encoder, FileFormat, FileMetadata, Step};
use crate::io::{ByteRange, FileRef, IoConfig, IoScheduler, StoreId};
use crate::options::FormatOptions;
use crate::request::{ROW_INDEX_COLUMN, ScanRequest};
use crate::text::TextMetadata;

fn options(map: &serde_json::Value) -> FormatOptions {
    let mut user = BTreeMap::new();
    if let Some(m) = map.as_object() {
        for (k, v) in m {
            user.insert(
                k.clone(),
                OptionValue::from_user(k, v.as_str().unwrap_or_default().to_owned()),
            );
        }
    }
    TextFormat
        .options()
        .parse(&user, &Config::new())
        .expect("options parse")
        .0
}

fn value_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Utf8View,
        true,
    )]))
}

/// The bytes of `range` in `bytes`.
fn slice<'a>(bytes: &'a [u8], range: &std::ops::Range<u64>) -> &'a [u8] {
    let (a, b) = (usize::try_from(range.start), usize::try_from(range.end));
    &bytes[a.expect("range")..b.expect("range")]
}

/// Runs a decoder over `bytes` for `split`.
fn read_split(
    bytes: &[u8],
    o: &FormatOptions,
    req: &ScanRequest,
    split: ByteRange,
) -> Result<Vec<RecordBatch>, KrishivError> {
    let m: Arc<dyn FileMetadata> = Arc::new(TextMetadata {
        size: bytes.len() as u64,
    });
    let units = TextFormat
        .plan_units(m.as_ref(), &split, req)
        .map_err(|e| e.into_error("t.txt"))?;
    let mut d =
        super::decode::TextDecoder::new(&m, &units, req, o).map_err(|e| e.into_error("t.txt"))?;
    let mut out = Vec::new();
    loop {
        match d.step().map_err(|e| e.into_error("t.txt"))? {
            Step::Done => return Ok(out),
            Step::Batch(b) => out.push(b),
            Step::NeedData(ranges) => {
                let data = ranges
                    .into_iter()
                    .map(|r| {
                        let slice = Bytes::copy_from_slice(slice(bytes, &r));
                        (r, slice)
                    })
                    .collect();
                d.push(data).map_err(|e| e.into_error("t.txt"))?;
            }
        }
    }
}

fn read(
    bytes: &[u8],
    o: &FormatOptions,
    req: &ScanRequest,
) -> Result<Vec<RecordBatch>, KrishivError> {
    read_split(bytes, o, req, ByteRange::whole(bytes.len() as u64))
}

fn rows_of(batches: &[RecordBatch]) -> Vec<Option<String>> {
    let mut rows = Vec::new();
    for b in batches {
        let v = b.column(0).as_string_view();
        rows.extend((0..b.num_rows()).map(|i| {
            if v.is_null(i) {
                None
            } else {
                Some(v.value(i).to_owned())
            }
        }));
    }
    rows
}

fn row_indexed() -> ScanRequest {
    let mut req = ScanRequest::all(value_schema());
    req.metadata_cols.row_index = true;
    req
}

fn in_memory_file(bytes: &[u8], name: &str) -> (Arc<IoScheduler>, FileRef) {
    let sched = IoScheduler::new(IoConfig::default());
    let store = Arc::new(object_store::memory::InMemory::new());
    let id = StoreId::new("memory://text");
    sched.register_store(
        id.clone(),
        Arc::clone(&store) as Arc<dyn object_store::ObjectStore>,
    );
    let path = object_store::path::Path::from(name);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("runtime");
    let file = rt.block_on(async {
        object_store::ObjectStoreExt::put(store.as_ref(), &path, bytes.to_vec().into())
            .await
            .expect("put");
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .expect("head");
        FileRef::from_meta(id, &meta)
    });
    (sched, file)
}

fn infer(bytes: &[u8], o: &FormatOptions) -> Result<SchemaRef, KrishivError> {
    infer_named(bytes, "f.txt", o)
}

fn infer_named(bytes: &[u8], name: &str, o: &FormatOptions) -> Result<SchemaRef, KrishivError> {
    let (sched, file) = in_memory_file(bytes, name);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .map_err(|e| crate::error::internal(e.to_string()))?;
    let io = sched.context();
    rt.block_on(TextFormat.infer_schema(&io, std::slice::from_ref(&file), o))
        .map_err(|e| e.into_error("f.txt"))
}

fn encode(batch: &RecordBatch, o: &FormatOptions) -> Result<Vec<u8>, KrishivError> {
    let mut e =
        super::encode::TextEncoder::new(&batch.schema(), o).map_err(|e| e.into_error("t.txt"))?;
    e.write(batch).map_err(|e| e.into_error("t.txt"))?;
    let mut out = Vec::new();
    while let Some(b) = e.take_bytes() {
        out.extend_from_slice(&b);
    }
    let done = Box::new(e).finish().map_err(|e| e.into_error("t.txt"))?;
    out.extend_from_slice(&done.tail);
    Ok(out)
}

#[test]
fn lines_read_verbatim() {
    let o = options(&serde_json::json!({}));
    let req = ScanRequest::all(value_schema());
    // Every line is a row, empty lines included; a UTF-8 BOM at the start
    // of the file is dropped, as Hadoop's line reader drops it.
    let bytes = "\u{FEFF}first\n\nsecondµ\nlast".as_bytes();
    let got = read(bytes, &o, &req).expect("read");
    assert_eq!(
        rows_of(&got),
        [
            Some("first".to_owned()),
            Some(String::new()),
            Some("secondµ".to_owned()),
            Some("last".to_owned()),
        ]
    );
    // An empty file has no rows.
    assert!(read(b"", &o, &req).expect("read").is_empty());
    // The schema is always one nullable `value` string column.
    let s = infer(b"a\n", &o).expect("infer");
    assert_eq!(s.fields().len(), 1);
    assert_eq!(s.field(0).name(), "value");
    assert_eq!(s.field(0).data_type(), &DataType::Utf8View);
    assert!(s.field(0).is_nullable());
}

#[test]
fn splits_read_each_line_once() {
    let mut text = String::new();
    for i in 0..500 {
        if i % 17 == 0 {
            text.push('\n');
        } else {
            text.push_str(&format!("line {i} µ\n"));
        }
    }
    let bytes = text.as_bytes();
    let o = options(&serde_json::json!({}));
    let req = row_indexed();
    let whole = read(bytes, &o, &req).expect("read");
    let size = bytes.len() as u64;
    for cuts in [
        vec![0, size],
        vec![0, 7, 100, size / 2, size - 3, size],
        vec![0, 1, 2, 3, size],
    ] {
        let mut seen = Vec::new();
        for w in cuts.windows(2) {
            let got = read_split(bytes, &o, &req, ByteRange::new(w[0], w[1])).expect("read");
            for b in &got {
                let v = b.column(0).as_string_view();
                let idx = b.column_by_name(ROW_INDEX_COLUMN).expect("row index");
                let idx = idx.as_primitive::<arrow_array::types::Int64Type>();
                seen.extend((0..b.num_rows()).map(|i| (idx.value(i), v.value(i).to_owned())));
            }
        }
        seen.sort();
        let mut want: Vec<(i64, String)> = rows_of(&whole)
            .iter()
            .enumerate()
            .map(|(n, v)| (n as i64, v.clone().expect("value")))
            .collect();
        want.sort();
        assert_eq!(seen, want, "cuts {cuts:?}");
    }
}

#[test]
fn wholetext_reads_one_row() {
    let o = options(&serde_json::json!({"wholetext": "true"}));
    let req = ScanRequest::all(value_schema());
    let bytes = b"a\nb\n";
    let got = read(bytes, &o, &req).expect("read");
    // The whole content is the row, trailing newline included.
    assert_eq!(rows_of(&got), [Some("a\nb\n".to_owned())]);
    // Other splits read nothing.
    let later = read_split(bytes, &o, &req, ByteRange::new(1, 4)).expect("read");
    assert!(later.is_empty());
    // An empty file still has no rows.
    assert!(read(b"", &o, &req).expect("read").is_empty());
    assert_eq!(
        TextFormat.capabilities(&o).splittable,
        crate::format::Splittable::No
    );
    assert_eq!(
        TextFormat
            .capabilities(&options(&serde_json::json!({})))
            .splittable,
        crate::format::Splittable::ByCodec
    );
}

#[test]
fn encodings_and_line_separators() {
    use ErrorClass::KrishivUnsupportedFeature as Unsupported;
    // Latin-1 decodes; invalid bytes become U+FFFD like Spark.
    let latin = [0xE9, b'\n'];
    for (name, value) in [("encoding", "ISO-8859-1"), ("charset", "LATIN1")] {
        let o = options(&serde_json::json!({name: value}));
        let got = read(&latin, &o, &ScanRequest::all(value_schema())).expect("read");
        assert_eq!(rows_of(&got), [Some("é".to_owned())]);
    }
    let o = options(&serde_json::json!({"encoding": "US-ASCII"}));
    let got = read(&latin, &o, &ScanRequest::all(value_schema())).expect("read");
    assert_eq!(rows_of(&got), [Some("�".to_owned())]);
    let bad = BTreeMap::from([(
        "encoding".to_owned(),
        krishiv_common::secret::OptionValue::from_user("encoding", "EBCDIC"),
    )]);
    let e = TextFormat
        .options()
        .parse(&bad, &Config::new())
        .expect("parse")
        .0;
    let e = super::options::TextOptions::new(&e, false).expect_err("charset");
    assert_eq!(e.error_class(), Unsupported);
    // A custom separator splits lines and ends written ones.
    let o = options(&serde_json::json!({"lineSep": ";"}));
    let got = read(b"a;b;", &o, &ScanRequest::all(value_schema())).expect("read");
    assert_eq!(rows_of(&got), [Some("a".to_owned()), Some("b".to_owned())]);
    let batch = RecordBatch::try_new(
        value_schema(),
        vec![Arc::new(arrow_array::StringViewArray::from(vec![
            Some("a"),
            Some("b"),
        ]))],
    )
    .expect("batch");
    assert_eq!(encode(&batch, &o).expect("encode"), b"a;b;");
    // An empty separator is rejected.
    let o = options(&serde_json::json!({"lineSep": ""}));
    assert!(super::options::TextOptions::new(&o, false).is_err());
}

#[test]
fn write_checks_and_nulls() {
    use ErrorClass::KrishivUnsupportedFeature as Unsupported;
    let o = options(&serde_json::json!({}));
    assert_eq!(TextFormat.file_suffix(&o), ".txt");
    assert!(TextFormat.capabilities(&o).write);
    assert!(TextFormat.capabilities(&o).row_index);
    // Exactly one string column.
    let two = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Utf8View, true),
        Field::new("w", DataType::Utf8View, true),
    ]));
    let e = TextFormat.check_write(&two, &o).expect_err("two columns");
    assert!(matches!(
        &e,
        crate::error::FormatError::Other(e)
            if e.error_class() == ErrorClass::LegacyErrorTemp1290
                && format!("{e:?}").contains("single column")
    ));
    let int = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        true,
    )]));
    let e = TextFormat.check_write(&int, &o).expect_err("int column");
    assert!(matches!(&e, crate::error::FormatError::Other(e) if e.error_class() == Unsupported));
    // A NULL writes an empty line, as Spark's text writer does.
    let batch = RecordBatch::try_new(
        value_schema(),
        vec![Arc::new(arrow_array::StringViewArray::from(vec![
            Some("a"),
            None,
        ]))],
    )
    .expect("batch");
    assert_eq!(encode(&batch, &o).expect("null"), b"a\n\n");
    // Non-UTF8View string forms write too.
    for dt in [DataType::Utf8, DataType::LargeUtf8] {
        let schema = Arc::new(Schema::new(vec![Field::new("value", dt, true)]));
        assert!(TextFormat.check_write(&schema, &o).is_ok());
    }
    // Compression needs codecs this build lacks; compressed reads are refused.
    for name in ["gzip", "snappy", "zstd"] {
        let w = options(&serde_json::json!({"compression": name}));
        let e = TextFormat
            .check_write(&value_schema(), &w)
            .expect_err("codec");
        assert!(
            matches!(&e, crate::error::FormatError::Other(e) if e.error_class() == Unsupported),
            "{name}"
        );
    }
    let e = infer_named(b"a\n", "a.txt.gz", &o).expect_err("compressed");
    assert_eq!(e.error_class(), Unsupported);
}

#[test]
fn write_then_read_round_trips() {
    let o = options(&serde_json::json!({}));
    let values = vec![
        Some("plain"),
        Some(""),
        Some("with\ttab"),
        Some("quotes \"'\""),
        Some("back\\slash"),
        Some("unicode é☃"),
        Some("lineSep;inside"),
    ];
    let batch = RecordBatch::try_new(
        value_schema(),
        vec![Arc::new(arrow_array::StringViewArray::from(values))],
    )
    .expect("batch");
    let bytes = encode(&batch, &o).expect("encode");
    let back = read(&bytes, &o, &ScanRequest::all(value_schema())).expect("read back");
    assert_eq!(rows_of(&back), rows_of(std::slice::from_ref(&batch)));
    // A missing `value` column is a planning error, never a wrong column.
    let other = Arc::new(Schema::new(vec![Field::new("x", DataType::Utf8View, true)]));
    let e = read(b"a\n", &o, &ScanRequest::all(other)).expect_err("column");
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn very_long_lines_read() {
    let big = "y".repeat(4 << 20);
    let line = format!("{big}\n");
    let o = options(&serde_json::json!({}));
    let got = read(line.as_bytes(), &o, &ScanRequest::all(value_schema())).expect("read");
    assert_eq!(rows_of(&got), [Some(big)]);
}

/// Random text over the characters that matter never panics the decoder
/// or inference, whatever the options, and every split reads the same
/// rows as the whole file.
#[test]
fn random_inputs_never_panic_and_splits_agree() {
    const ATOMS: &[&str] = &[
        "\n", "\r\n", "\r", ";", "\"", "\\", " ", "a", "é", "\u{0}", "1",
    ];
    let option_sets = [
        serde_json::json!({}),
        serde_json::json!({"wholetext": "true"}),
        serde_json::json!({"lineSep": ";"}),
        serde_json::json!({"encoding": "ISO-8859-1"}),
        serde_json::json!({"wholetext": "true", "lineSep": ";"}),
    ];
    let mut state = 41_u64;
    let mut next = |n: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % n as u64).expect("small")
    };
    for _ in 0..200 {
        let atoms = 1 + next(60);
        let text: String = (0..atoms).map(|_| ATOMS[next(ATOMS.len())]).collect();
        for opts in &option_sets {
            let o = options(opts);
            let _ = infer(text.as_bytes(), &o);
            let req = row_indexed();
            let whole = read(text.as_bytes(), &o, &req);
            let size = text.len() as u64;
            if size == 0 {
                continue;
            }
            let cut = (1 + next(text.len().max(2) - 1) as u64).min(size.saturating_sub(1));
            if cut == 0 {
                continue;
            }
            let first = read_split(text.as_bytes(), &o, &req, ByteRange::new(0, cut));
            let second = read_split(text.as_bytes(), &o, &req, ByteRange::new(cut, size));
            let splittable =
                TextFormat.capabilities(&o).splittable == crate::format::Splittable::ByCodec;
            if splittable {
                assert_eq!(whole.is_ok(), first.is_ok() && second.is_ok(), "{opts}");
            }
            if let (Ok(w), Ok(f), Ok(s)) = (&whole, &first, &second)
                && splittable
            {
                {
                    let mut got: Vec<_> = rows_of_splits(f)
                        .into_iter()
                        .chain(rows_of_splits(s))
                        .collect();
                    let mut want = rows_of_splits(w);
                    got.sort();
                    want.sort();
                    assert_eq!(got, want, "{opts} cut {cut} of {text:?}");
                }
            }
        }
    }
}

/// Rows with their row indexes, for split agreement.
fn rows_of_splits(batches: &[RecordBatch]) -> Vec<(i64, Option<String>)> {
    let mut rows = Vec::new();
    for b in batches {
        let v = b.column(0).as_string_view();
        let idx = b.column_by_name(ROW_INDEX_COLUMN).expect("row index");
        let idx = idx.as_primitive::<arrow_array::types::Int64Type>();
        rows.extend((0..b.num_rows()).map(|i| {
            (
                idx.value(i),
                if v.is_null(i) {
                    None
                } else {
                    Some(v.value(i).to_owned())
                },
            )
        }));
    }
    rows
}
