// SPDX-License-Identifier: Apache-2.0
//! Parquet writer tests: files are read back with arrow-rs's own reader, so
//! what is checked is the file, not the Krishiv reader. The conformance
//! suite (round trips, rollover, statistics) and the interoperability
//! checks live in `krishiv-testkit`.

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, TimestampMicrosecondBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{TimestampMicrosecondType, TimestampMillisecondType};
use arrow_array::{
    Array, ArrayRef, Date32Array, Int32Array, Int64Array, MapArray, RecordBatch, StringArray,
    StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use bytes::Bytes;
use krishiv_common::error::{ErrorClass, KrishivError};
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::basic::{Compression, Encoding, LogicalType};
use parquet::column::page::Page;
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::reader::{FileReader, SerializedFileReader};

use crate::FileFormat;
use crate::format::Encoder;
use crate::options::FormatOptions;
use crate::parquet::ParquetFormat;

const UTC: &str = "UTC";

fn options(kv: &[(&str, &str)]) -> FormatOptions {
    options_with(kv, &krishiv_common::config::Config::new())
}

fn options_with(kv: &[(&str, &str)], session: &krishiv_common::config::Config) -> FormatOptions {
    let user = kv
        .iter()
        .map(|(k, v)| {
            (
                (*k).to_owned(),
                krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
            )
        })
        .collect();
    ParquetFormat.options().parse(&user, session).unwrap().0
}

fn encoder(schema: &SchemaRef, o: &FormatOptions) -> Result<Box<dyn Encoder>, KrishivError> {
    ParquetFormat
        .check_write(schema, o)
        .and_then(|()| ParquetFormat::encoder_with(schema, o, |p| p))
        .map_err(|e| e.into_error("memory://t/f.parquet"))
}

/// The file of `batches`, drained the way the framework drains it.
fn write(batches: &[RecordBatch], o: &FormatOptions) -> Result<Bytes, KrishivError> {
    let schema = batches[0].schema();
    let mut enc = encoder(&schema, o)?;
    let mut out = Vec::new();
    for b in batches {
        enc.write(b).map_err(|e| e.into_error("f"))?;
        while let Some(bytes) = enc.take_bytes() {
            out.extend_from_slice(&bytes);
        }
    }
    let done = enc.finish().map_err(|e| e.into_error("f"))?;
    out.extend_from_slice(&done.tail);
    Ok(Bytes::from(out))
}

fn read(file: &Bytes) -> Vec<RecordBatch> {
    ParquetRecordBatchReaderBuilder::try_new(file.clone())
        .unwrap()
        .build()
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn ids(n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", DataType::Utf8, true),
    ]));
    let strings = (0..n).map(|i| (i % 7 != 3).then(|| format!("v{}", i % 13)));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(0..n)),
            Arc::new(StringArray::from_iter(strings)),
        ],
    )
    .unwrap()
}

fn class(e: &KrishivError) -> ErrorClass {
    e.error_class()
}

#[test]
fn codecs_are_named_and_written_as_spark_names_them() {
    let zstd = Compression::ZSTD(parquet::basic::ZstdLevel::try_new(3).unwrap());
    let cases = [
        ("none", "", Compression::UNCOMPRESSED),
        ("uncompressed", "", Compression::UNCOMPRESSED),
        ("snappy", ".snappy", Compression::SNAPPY),
        ("GZIP", ".gz", Compression::GZIP(Default::default())),
        ("lz4", ".lz4hadoop", Compression::LZ4),
        ("lz4_raw", ".lz4raw", Compression::LZ4_RAW),
        ("zstd", ".zstd", zstd),
        ("brotli", ".br", Compression::BROTLI(Default::default())),
    ];
    let batch = ids(500);
    for (name, ext, want) in cases {
        let o = options(&[("compression", name)]);
        assert_eq!(
            ParquetFormat.file_suffix(&o),
            format!("{ext}.parquet"),
            "{name}"
        );
        let file = write(std::slice::from_ref(&batch), &o).unwrap();
        let reader = SerializedFileReader::new(file.clone()).unwrap();
        let got = reader.metadata().row_group(0).column(0).compression();
        // Levels are not stored in the file: compare the codec only.
        assert_eq!(
            std::mem::discriminant(&got),
            std::mem::discriminant(&want),
            "{name}"
        );
        assert_eq!(read(&file), vec![batch.clone()], "{name}");
    }
    // The default is Spark's: snappy.
    assert_eq!(ParquetFormat.file_suffix(&options(&[])), ".snappy.parquet");
}

#[test]
fn the_codec_follows_sparks_precedence() {
    let suffix = |kv: &[(&str, &str)], session: &krishiv_common::config::Config| {
        ParquetFormat.file_suffix(&options_with(kv, session))
    };
    let mut session = krishiv_common::config::Config::new();
    session
        .set("spark.sql.parquet.compression.codec", "zstd")
        .unwrap();
    assert_eq!(suffix(&[], &session), ".zstd.parquet");
    assert_eq!(
        suffix(&[("parquet.compression", "GZIP")], &session),
        ".gz.parquet"
    );
    assert_eq!(
        suffix(
            &[("parquet.compression", "gzip"), ("compression", "lz4")],
            &session
        ),
        ".lz4hadoop.parquet"
    );
}

#[test]
fn unknown_and_unavailable_codecs_fail_before_writing() {
    let schema = ids(1).schema();
    let e = encoder(&schema, &options(&[("compression", "bogus")]))
        .err()
        .unwrap();
    assert_eq!(
        class(&e),
        ErrorClass::CodecNotAvailableWithAvailableCodecsSuggestion
    );
    let msg = e.to_string();
    assert!(
        msg.contains("bogus") && msg.contains("snappy, uncompressed, zstd"),
        "{msg}"
    );
    let e = encoder(&schema, &options(&[("compression", "lzo")]))
        .err()
        .unwrap();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    // A suffix is still named for a bad codec; the write fails before.
    assert_eq!(
        ParquetFormat.file_suffix(&options(&[("compression", "bogus")])),
        ".parquet"
    );
}

/// A batch with timestamps at every depth: top level, in a struct, a list
/// and a map value, plus `timestamp_ntz` and a date.
fn temporal(values: &[Option<i64>]) -> RecordBatch {
    let ts = |v: &[Option<i64>]| -> ArrayRef {
        Arc::new(TimestampMicrosecondArray::from(v.to_vec()).with_timezone(UTC))
    };
    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, Some(UTC.into()));
    let n = values.len();
    let st = StructArray::new(
        Fields::from(vec![Field::new("t", ts_type.clone(), true)]),
        vec![ts(values)],
        None,
    );
    let mut lb = ListBuilder::new(TimestampMicrosecondBuilder::new().with_timezone(UTC));
    for v in values {
        lb.values().append_option(*v);
        lb.append(true);
    }
    let list = lb.finish();
    let entries = StructArray::new(
        Fields::from(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", ts_type, true),
        ]),
        vec![
            Arc::new(Int32Array::from_iter_values(0..i32::try_from(n).unwrap())),
            ts(values),
        ],
        None,
    );
    let map = MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::from_lengths(std::iter::repeat_n(1, n)),
        entries,
        None,
        false,
    );
    let ntz: ArrayRef = Arc::new(TimestampMicrosecondArray::from(values.to_vec()));
    let columns: Vec<ArrayRef> = vec![
        ts(values),
        Arc::new(st),
        Arc::new(list),
        Arc::new(map),
        ntz,
        Arc::new(Date32Array::from(vec![Some(0); n])),
    ];
    let fields: Vec<Field> = ["ts", "st", "list", "map", "ntz", "d"]
        .iter()
        .zip(&columns)
        .map(|(name, c)| Field::new(*name, c.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

#[test]
fn millisecond_timestamps_round_down_at_any_depth() {
    let values = [Some(-500), Some(-1000), Some(123_456), None, Some(-1_001)];
    let o = options(&[("outputTimestampType", "TIMESTAMP_MILLIS")]);
    let back = read(&write(&[temporal(&values)], &o).unwrap());
    let b = &back[0];
    let want = [Some(-1), Some(-1), Some(123), None, Some(-2)];
    let millis = |a: &ArrayRef| -> Vec<Option<i64>> {
        assert_eq!(
            a.data_type(),
            &DataType::Timestamp(TimeUnit::Millisecond, Some(UTC.into()))
        );
        a.as_primitive::<TimestampMillisecondType>()
            .iter()
            .collect()
    };
    assert_eq!(millis(b.column(0)), want);
    assert_eq!(millis(b.column(1).as_struct().column(0)), want);
    assert_eq!(millis(b.column(2).as_list::<i32>().values()), want);
    assert_eq!(millis(b.column(3).as_map().values()), want);
    // Large lists too.
    let ts = TimestampMicrosecondArray::from(values.to_vec()).with_timezone(UTC);
    let large = arrow_array::LargeListArray::new(
        Arc::new(Field::new("item", ts.data_type().clone(), true)),
        OffsetBuffer::from_lengths([values.len()]),
        Arc::new(ts),
        None,
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "l",
        large.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(large)]).unwrap();
    let back = read(&write(&[batch], &o).unwrap());
    assert_eq!(millis(back[0].column(0).as_list::<i64>().values()), want);
    // timestamp_ntz stays in microseconds.
    let ntz: Vec<_> = b
        .column(4)
        .as_primitive::<TimestampMicrosecondType>()
        .iter()
        .collect();
    assert_eq!(ntz, values);
    // The default is microseconds, as stored.
    let back = read(&write(&[temporal(&values)], &options(&[])).unwrap());
    let ts: Vec<_> = back[0]
        .column(0)
        .as_primitive::<TimestampMicrosecondType>()
        .iter()
        .collect();
    assert_eq!(ts, values);
    let reader = SerializedFileReader::new(write(&[temporal(&values)], &o).unwrap()).unwrap();
    let leaf = reader.metadata().file_metadata().schema_descr().column(0);
    assert_eq!(
        leaf.logical_type_ref(),
        Some(&LogicalType::Timestamp(parquet::basic::TimestampType {
            is_adjusted_to_u_t_c: true,
            unit: parquet::basic::TimeUnit::MILLIS
        }))
    );
}

#[test]
fn int96_output_is_refused_only_for_timestamps() {
    let o = options(&[("outputTimestampType", "int96")]);
    let e = encoder(&temporal(&[Some(1)]).schema(), &o).err().unwrap();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    assert!(e.to_string().contains("INT96"), "{e}");
    // Without timestamps the setting does not matter (as in Spark).
    assert!(write(&[ids(3)], &o).is_ok());
}

/// Just before 1900-01-01T00:00:00Z, and that instant.
const ANCIENT: i64 = -2_208_988_800_000_001;
const SWITCH: i64 = -2_208_988_800_000_000;

#[test]
fn ancient_values_follow_the_write_rebase_mode() {
    let o = options(&[("datetimeRebaseModeInWrite", "exception")]);
    let e = write(&[temporal(&[Some(ANCIENT)])], &o).unwrap_err();
    assert_eq!(
        class(&e),
        ErrorClass::InconsistentBehaviorCrossVersionWriteAncientDatetime
    );
    assert!(e.to_string().contains("datetimeRebaseModeInWrite"), "{e}");
    assert!(write(&[temporal(&[Some(SWITCH)])], &o).is_ok());
    // Dates: before 1582-10-15.
    let dates = |d: i32| {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Date32Array::from(vec![d]))]).unwrap()
    };
    assert!(write(&[dates(-141_428)], &o).is_err());
    assert!(write(&[dates(-141_427)], &o).is_ok());
    // timestamp_ntz is never rebased.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "t",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
    )]));
    let ntz: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![ANCIENT]));
    let ntz = RecordBatch::try_new(schema, vec![ntz]).unwrap();
    assert!(write(&[ntz], &o).is_ok());
    // LEGACY: the file is marked; values that would be rebased are refused.
    let legacy = options(&[("datetimeRebaseModeInWrite", "LEGACY")]);
    let marked = |file: Bytes| {
        let reader = SerializedFileReader::new(file).unwrap();
        let kv = reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap();
        kv.iter()
            .any(|e| e.key == "org.apache.spark.legacyDateTime")
    };
    assert!(marked(
        write(&[temporal(&[Some(SWITCH)])], &legacy).unwrap()
    ));
    let e = write(&[temporal(&[Some(ANCIENT)])], &legacy).unwrap_err();
    assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    // LEGACY marks every file, as Spark, even without dates or timestamps.
    assert!(marked(write(&[ids(2)], &legacy).unwrap()));
}

#[test]
fn only_values_a_reader_sees_are_checked_for_the_calendar() {
    let o = options(&[("datetimeRebaseModeInWrite", "EXCEPTION")]);
    // A slice of a list column whose other rows are ancient.
    let batch = temporal(&[Some(ANCIENT), Some(0), Some(1)]);
    let tail = batch.project(&[2]).unwrap().slice(1, 2);
    assert!(write(&[tail], &o).is_ok());
    // A struct field holding an ancient value under a null struct.
    let ts_type = DataType::Timestamp(TimeUnit::Microsecond, Some(UTC.into()));
    let values = TimestampMicrosecondArray::from(vec![Some(ANCIENT), Some(0)]).with_timezone(UTC);
    let st = StructArray::new(
        Fields::from(vec![Field::new("t", ts_type, true)]),
        vec![Arc::new(values)],
        Some(NullBuffer::from(vec![false, true])),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "st",
        st.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(st)]).unwrap();
    assert!(write(&[batch], &o).is_ok());
}

#[test]
fn page_indexes_are_always_written() {
    let batch = temporal(&[Some(1), None, Some(3)]);
    for kv in [
        vec![],
        vec![("parquet.writer.version", "v2"), ("compression", "zstd")],
    ] {
        let file = write(std::slice::from_ref(&batch), &options(&kv)).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            file,
            ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required),
        )
        .unwrap();
        let md = builder.metadata();
        let leaves = md.file_metadata().schema_descr().num_columns();
        let index = md.page_index_for_row_group(0);
        for leaf in 0..leaves {
            assert!(index.column_index(leaf).is_some(), "column index of {leaf}");
            assert!(index.offset_index(leaf).is_some(), "offset index of {leaf}");
        }
        for c in md.row_group(0).columns() {
            assert!(c.column_index_offset().is_some(), "{:?}", c.column_path());
            assert!(c.offset_index_offset().is_some(), "{:?}", c.column_path());
        }
    }
}

#[test]
fn the_footer_reads_as_a_spark_4_file() {
    let file = write(&[ids(3)], &options(&[])).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let fm = reader.metadata().file_metadata();
    let kv = fm.key_value_metadata().unwrap();
    let get = |k: &str| kv.iter().find(|e| e.key == k).and_then(|e| e.value.clone());
    assert_eq!(get("org.apache.spark.version").as_deref(), Some("4.2.0"));
    assert!(get("ARROW:schema").is_some(), "the Arrow schema is kept");
    assert!(
        fm.created_by().unwrap().starts_with("krishiv version "),
        "{:?}",
        fm.created_by()
    );
    // Readers take ancient values as stored even in EXCEPTION mode.
    let calendars = crate::parquet::calendar::calendars(fm.key_value_metadata());
    assert_eq!(
        calendars.datetime,
        crate::parquet::calendar::Calendar::Proleptic
    );
}

#[test]
fn row_groups_pages_dictionary_and_version_follow_the_options() {
    let batch = ids(20_000);
    let first_page = |file: Bytes| {
        let reader = SerializedFileReader::new(file).unwrap();
        let mut pages = reader
            .get_row_group(0)
            .unwrap()
            .get_column_page_reader(0)
            .unwrap();
        pages.get_next_page().unwrap().unwrap()
    };
    // Plain-encoded rows: a 16 KiB block holds a few thousand of them.
    let o = options(&[
        ("parquet.block.size", "16384"),
        ("parquet.page.size", "4096"),
        ("parquet.enable.dictionary", "false"),
    ]);
    let file = write(std::slice::from_ref(&batch), &o).unwrap();
    let reader = SerializedFileReader::new(file.clone()).unwrap();
    let md = reader.metadata();
    assert!(
        md.num_row_groups() > 3,
        "{} row groups",
        md.num_row_groups()
    );
    for rg in md.row_groups() {
        // A row group may overshoot by the rows of one estimate, not more.
        assert!(rg.compressed_size() < 3 * 16384, "{}", rg.compressed_size());
        for c in rg.columns() {
            assert!(
                !c.encodings()
                    .any(|e| matches!(e, Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY)),
                "dictionary used"
            );
        }
    }
    assert!(matches!(first_page(file.clone()), Page::DataPage { .. }));
    let back = read(&file);
    assert_eq!(
        arrow_select::concat::concat_batches(&batch.schema(), &back).unwrap(),
        batch
    );
    // Data page v2.
    let o = options(&[("parquet.writer.version", "PARQUET_2_0")]);
    let page = first_page(write(std::slice::from_ref(&batch), &o).unwrap());
    assert!(
        matches!(page, Page::DictionaryPage { .. }),
        "{:?}",
        page.page_type()
    );
    let o = options(&[
        ("parquet.writer.version", "v2"),
        ("parquet.enable.dictionary", "false"),
    ]);
    let page = first_page(write(std::slice::from_ref(&batch), &o).unwrap());
    assert!(
        matches!(page, Page::DataPageV2 { .. }),
        "{:?}",
        page.page_type()
    );
    // Defaults: one row group, dictionary pages, data page v1.
    let file = write(&[batch], &options(&[])).unwrap();
    assert_eq!(
        SerializedFileReader::new(file.clone())
            .unwrap()
            .metadata()
            .num_row_groups(),
        1
    );
    assert!(matches!(first_page(file), Page::DictionaryPage { .. }));
}

#[test]
fn the_open_row_group_counts_toward_the_written_size() {
    let schema = ids(1).schema();
    let mut enc = encoder(&schema, &options(&[])).unwrap();
    let before = enc.written_bytes();
    enc.write(&ids(10_000)).unwrap();
    assert!(
        enc.written_bytes() > before + 10_000,
        "{}",
        enc.written_bytes()
    );
    assert!(enc.buffered_bytes() > 0);
    // Flushing ends the row group: its bytes are ready to take.
    enc.flush().unwrap();
    let taken = enc.take_bytes().unwrap();
    assert!(taken.len() > 4);
    assert!(enc.take_bytes().is_none());
}

#[test]
fn an_encoder_failure_is_a_write_failure() {
    let schema = ids(1).schema();
    let mut enc = encoder(&schema, &options(&[])).unwrap();
    let other = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1]))],
    )
    .unwrap();
    let e = enc.write(&other).unwrap_err();
    assert!(e.to_string().starts_with("cannot write"), "{e}");
    assert_eq!(
        class(&e.into_error("memory://t/f.parquet")),
        ErrorClass::TaskWriteFailed
    );
}

#[test]
fn a_string_column_stored_as_binary_is_refused() {
    let tag = std::collections::HashMap::from([(
        krishiv_types::arrow::SPARK_TYPE_METADATA_KEY.to_owned(),
        "string".to_owned(),
    )]);
    let s = Field::new("s", DataType::BinaryView, true).with_metadata(tag);
    let nested = Field::new("st", DataType::Struct(Fields::from(vec![s.clone()])), true);
    for f in [s, nested] {
        let schema = Arc::new(Schema::new(vec![f]));
        let e = encoder(&schema, &options(&[])).err().unwrap();
        assert_eq!(class(&e), ErrorClass::KrishivUnsupportedFeature);
    }
    // Plain binary is binary.
    let schema = Arc::new(Schema::new(vec![Field::new(
        "b",
        DataType::BinaryView,
        true,
    )]));
    assert!(encoder(&schema, &options(&[])).is_ok());
}

#[test]
fn wide_rows_cannot_overshoot_the_block_size() {
    // 64 rows of 64 KiB in one batch; the block holds about 4 of them.
    let blob: Vec<Vec<u8>> = (0..64_u32)
        .map(|i| {
            (0..65_536_u32)
                .map(|k| {
                    (k ^ i.wrapping_mul(7919))
                        .wrapping_mul(2_654_435_761)
                        .to_le_bytes()[3]
                })
                .collect()
        })
        .collect();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "b",
        DataType::BinaryView,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow_array::BinaryViewArray::from_iter_values(
            blob.iter().map(Vec::as_slice),
        ))],
    )
    .unwrap();
    let o = options(&[("parquet.block.size", "262144"), ("compression", "none")]);
    let file = write(&[batch], &o).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let md = reader.metadata();
    assert!(
        md.num_row_groups() >= 8,
        "{} row groups",
        md.num_row_groups()
    );
    for rg in md.row_groups() {
        assert!(
            rg.compressed_size() < 2 * 262_144,
            "{}",
            rg.compressed_size()
        );
    }
}

#[test]
fn collated_columns_write_the_spark_row_metadata() {
    use krishiv_types::{Collation, DataType as Spark};
    let footer = |schema: Schema| {
        let schema = Arc::new(schema);
        let batch = RecordBatch::new_empty(Arc::clone(&schema));
        let file = write(&[batch], &options(&[])).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .cloned()
            .unwrap_or_default()
    };
    let key = "org.apache.spark.sql.parquet.row.metadata";
    let plain = footer(Schema::new(vec![Field::new("a", DataType::Utf8View, true)]));
    assert!(plain.iter().all(|e| e.key != key));
    let lcase = Spark::collated_string(Collation::parse("UTF8_LCASE").unwrap());
    let kv = footer(Schema::new(vec![lcase.to_arrow_field("a", true)]));
    let json = kv
        .iter()
        .find(|e| e.key == key)
        .unwrap()
        .value
        .clone()
        .unwrap();
    assert!(json.contains("spark.UTF8_LCASE"), "{json}");
}

/// `TIME` (nanoseconds in memory) is written as Spark writes it:
/// `INT64` `TIME(isAdjustedToUTC=false, MICROS)`, at any depth.
#[test]
fn time_is_written_as_microseconds() {
    use arrow_array::types::Time64MicrosecondType;
    use arrow_array::{ListArray, Time64NanosecondArray};
    let t = DataType::Time64(TimeUnit::Nanosecond);
    let item = Arc::new(Field::new("item", t.clone(), true));
    let values = Time64NanosecondArray::from(vec![Some(36_000_000_000_000), Some(1_500)]);
    let list = ListArray::new(
        Arc::clone(&item),
        OffsetBuffer::from_lengths([2]),
        Arc::new(values),
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("t", t, true),
        Field::new("l", DataType::List(item), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Time64NanosecondArray::from(vec![Some(86_399_999_999_000)])),
            Arc::new(list),
        ],
    )
    .unwrap();
    let file = write(&[batch], &options(&[])).unwrap();
    let reader = SerializedFileReader::new(file.clone()).unwrap();
    for c in reader.metadata().file_metadata().schema_descr().columns() {
        let Some(LogicalType::Time(tt)) = c.logical_type_ref() else {
            panic!("not a TIME column: {:?}", c.logical_type_ref());
        };
        assert!(!tt.is_adjusted_to_u_t_c);
        assert!(matches!(tt.unit, parquet::basic::TimeUnit::MICROS));
    }
    let got = read(&file);
    assert_eq!(
        got[0]
            .column(0)
            .as_primitive::<Time64MicrosecondType>()
            .value(0),
        86_399_999_999
    );
    let l = got[0].column(1).as_list::<i32>();
    assert_eq!(
        l.values()
            .as_primitive::<Time64MicrosecondType>()
            .values()
            .to_vec(),
        [36_000_000_000, 1]
    );
}
