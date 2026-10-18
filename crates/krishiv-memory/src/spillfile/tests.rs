// SPDX-License-Identifier: Apache-2.0
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value generators derive narrow test values from i64 seeds"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array,
    Decimal128Array, Decimal256Array, DictionaryArray, DurationMicrosecondArray,
    DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    IntervalDayTimeArray, IntervalMonthDayNanoArray, IntervalYearMonthArray, LargeBinaryArray,
    LargeListArray, LargeStringArray, ListArray, MapArray, NullArray, RecordBatch,
    RecordBatchOptions, StringArray, StringViewArray, StructArray, Time32MillisecondArray,
    Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_buffer::{IntervalDayTime, IntervalMonthDayNano};
use arrow_buffer::{NullBuffer, OffsetBuffer, i256};
use arrow_schema::{DataType, Field, IntervalUnit, Schema, SchemaRef, TimeUnit};
use krishiv_common::config::Config;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;

use super::format::{self, FILE_HEADER_LEN, FRAME_HEADER_LEN};
use super::*;
use crate::{ConsumerId, MemoryConfig, MemoryPool, Reservation, TreePool};

/// A fresh path (not created yet) inside a temporary directory of its own.
/// The whole temporary directory is removed on drop, also when the test
/// panics, whatever the test left at the path (a directory, a file, a
/// symlink).
struct TestDir(
    PathBuf,
    #[expect(dead_code, reason = "held for its Drop, which removes the directory")]
    tempfile::TempDir,
);

impl TestDir {
    fn new() -> Self {
        let parent = tempfile::Builder::new()
            .prefix("krishiv-spilltest-")
            .tempdir()
            .unwrap();
        Self(parent.path().join("d"), parent)
    }

    /// Creates the directory the way a careful user would (mode 0700).
    fn create(&self) {
        fs::create_dir_all(&self.0).unwrap();
        #[cfg(unix)]
        fs::set_permissions(&self.0, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    }

    fn config(&self, codec: SpillCodec) -> SpillConfig {
        SpillConfig {
            dir: self.0.clone(),
            codec,
            ..SpillConfig::default()
        }
    }
}

impl SpillConfig {
    fn with_dir(self, dir: &Path) -> Self {
        Self {
            dir: dir.to_owned(),
            ..self
        }
    }
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// A pool with exact accounting (quantum 1) and one registered consumer.
struct Pool {
    root: Arc<TreePool>,
    query: Arc<dyn MemoryPool>,
    id: ConsumerId,
}

impl Pool {
    fn new(limit: Option<usize>) -> Self {
        let root = TreePool::new(&MemoryConfig {
            limit,
            quantum: 1,
            ..MemoryConfig::default()
        });
        let query = root.child("query", None);
        let id = query.register("spill", None);
        Self { root, query, id }
    }

    fn reservation(&self) -> Reservation {
        Reservation::new(Arc::clone(&self.query), self.id)
    }
}

/// A fresh spill directory, its manager, a query scope and an unlimited pool.
fn env(codec: SpillCodec) -> (TestDir, SpillManager, SpillScope, Pool) {
    let dir = TestDir::new();
    let manager = SpillManager::open(&dir.config(codec)).unwrap();
    let scope = manager.scope().unwrap();
    (dir, manager, scope, Pool::new(None))
}

fn schema_i64() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]))
}

fn int_batch(start: i64, n: usize, weights: Weights) -> Batch {
    let values: Vec<i64> = (start..).take(n).collect();
    let record =
        RecordBatch::try_new(schema_i64(), vec![Arc::new(Int64Array::from(values))]).unwrap();
    Batch::with_weights(record, weights).unwrap()
}

fn write_file(scope: &SpillScope, pool: &Pool, batches: &[Batch]) -> SpillFile {
    let schema = batches.first().map_or_else(schema_i64, Batch::schema);
    let mut w = scope.writer(schema, pool.reservation()).unwrap();
    for b in batches {
        w.write(b).unwrap();
    }
    w.finish().unwrap()
}

fn read_all(file: &SpillFile, pool: &Pool) -> Result<Vec<Batch>, SpillError> {
    file.reader(pool.reservation())?.collect()
}

fn is_io(e: &SpillError, operation: &str) -> bool {
    matches!(e, SpillError::Io { operation: op, .. } if *op == operation)
}

fn class(e: &SpillError) -> ErrorClass {
    KrishivError::from(e).error_class()
}

#[test]
fn round_trip_every_codec_keeps_data_and_both_weight_kinds() {
    let mut sizes = Vec::new();
    for codec in [SpillCodec::None, SpillCodec::Lz4, SpillCodec::Zstd] {
        let dir = TestDir::new();
        let manager = SpillManager::open(&dir.config(codec)).unwrap();
        assert_eq!(manager.codec(), codec);
        let scope = manager.scope().unwrap();
        let pool = Pool::new(None);
        let batches = vec![
            int_batch(0, 100, Weights::Const(1)),
            int_batch(100, 50, Weights::Const(-3)),
            int_batch(
                150,
                4,
                Weights::Column(Int64Array::from(vec![1, -1, i64::MAX, i64::MIN])),
            ),
            int_batch(0, 0, Weights::Const(7)),
            int_batch(0, 20_000, Weights::Const(1)),
        ];
        let file = write_file(&scope, &pool, &batches);
        assert_eq!(file.num_batches(), 5);
        assert_eq!(file.num_rows(), 20_154);
        sizes.push(file.bytes());
        assert_eq!(manager.used_bytes(), file.bytes());
        assert_eq!(fs::metadata(file.path()).unwrap().len(), file.bytes());
        assert_eq!(read_all(&file, &pool).unwrap(), batches);
        // Readers can be opened again, and nothing is left reserved.
        assert_eq!(read_all(&file, &pool).unwrap(), batches);
        assert_eq!(pool.root.reserved(), 0);
        drop(file);
        assert_eq!(manager.used_bytes(), 0);
    }
    // None, LZ4, ZSTD: both codecs shrink compressible data.
    assert!(sizes[1] < sizes[0] && sizes[2] < sizes[0], "{sizes:?}");
}

#[test]
fn dictionary_columns_and_schema_metadata_round_trip() {
    let (_dir, _manager, scope, pool) = env(SpillCodec::Lz4);
    let dict: DictionaryArray<Int32Type> = ["a", "b", "a", "c"].into_iter().collect();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new("d", dict.data_type().clone(), true)],
        [("origin".to_owned(), "test".to_owned())],
    ));
    let record = RecordBatch::try_new(schema, vec![Arc::new(dict)]).unwrap();
    let batch = Batch::new(record);
    let file = write_file(&scope, &pool, &[batch.clone(), batch.clone()]);
    assert_eq!(read_all(&file, &pool).unwrap(), vec![batch.clone(), batch]);
}

fn type_strategy() -> impl Strategy<Value = DataType> {
    use DataType::*;
    let units = [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ];
    let leaf = prop_oneof![
        prop::sample::select(vec![
            Boolean,
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
            Utf8,
            LargeUtf8,
            Utf8View,
            Binary,
            LargeBinary,
            BinaryView,
            Date32,
            Date64,
            Null,
            Time32(TimeUnit::Second),
            Time32(TimeUnit::Millisecond),
            Time64(TimeUnit::Microsecond),
            Time64(TimeUnit::Nanosecond),
            Interval(IntervalUnit::YearMonth),
            Interval(IntervalUnit::DayTime),
            Interval(IntervalUnit::MonthDayNano),
            Dictionary(Box::new(Int32), Box::new(Utf8)),
        ]),
        (0..units.len(), any::<bool>()).prop_map(move |(u, tz)| {
            Timestamp(units[u], tz.then(|| Arc::from("America/Los_Angeles")))
        }),
        (0..units.len()).prop_map(move |u| Duration(units[u])),
        (1i32..9).prop_map(FixedSizeBinary),
        (1u8..=38)
            .prop_flat_map(|p| (Just(p), 0..=p))
            .prop_map(|(p, s)| Decimal128(p, i8::try_from(s).unwrap_or(0))),
        (1u8..=76)
            .prop_flat_map(|p| (Just(p), 0..=p))
            .prop_map(|(p, s)| Decimal256(p, i8::try_from(s).unwrap_or(0))),
    ];
    leaf.prop_recursive(2, 6, 2, |inner| {
        let item = |t: DataType| Arc::new(Field::new("item", t, true));
        prop_oneof![
            inner.clone().prop_map(move |t| List(item(t))),
            inner.clone().prop_map(move |t| LargeList(item(t))),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| {
                Struct(vec![Field::new("a", a, true), Field::new("b", b, true)].into())
            }),
            inner.prop_map(|t| {
                let entries = Struct(
                    vec![
                        Field::new("keys", Int32, false),
                        Field::new("values", t, true),
                    ]
                    .into(),
                );
                Map(Arc::new(Field::new("entries", entries, false)), false)
            }),
        ]
    })
}

fn text(seed: i64) -> String {
    match seed.rem_euclid(4) {
        0 => String::new(),
        1 => "é漢🙂".repeat(usize::try_from(seed.rem_euclid(9)).unwrap_or(0)),
        _ => format!("value-{seed}"),
    }
}

fn bytes(seed: i64) -> Vec<u8> {
    let n = usize::try_from(seed.rem_euclid(9)).unwrap_or(0);
    seed.to_le_bytes().into_iter().take(n).collect()
}

const KEYS: [&str; 5] = ["k0", "k1", "k2", "k3", "k4"];

fn validity(seeds: &[Option<i64>]) -> NullBuffer {
    seeds.iter().map(Option::is_some).collect()
}

/// An array of type `dt` whose values (and nulls) derive from `seeds`.
fn build(dt: &DataType, seeds: &[Option<i64>]) -> ArrayRef {
    macro_rules! arr {
        ($t:ty, $f:expr) => {
            Arc::new(<$t>::from_iter(seeds.iter().map(|s| s.map($f)))) as ArrayRef
        };
    }
    let small = |m: i64| move |v: i64| v % m;
    match dt {
        DataType::Boolean => arr!(BooleanArray, |v| v % 2 == 0),
        DataType::Int8 => arr!(Int8Array, |v| i8::try_from(small(100)(v))
            .unwrap_or_default()),
        DataType::Int16 => arr!(Int16Array, |v| i16::try_from(small(3000)(v))
            .unwrap_or_default()),
        DataType::Int32 | DataType::Date32 => {
            let f = |v: i64| i32::try_from(v % 2_000_000_000).unwrap_or_default();
            if *dt == DataType::Date32 {
                arr!(Date32Array, f)
            } else {
                arr!(Int32Array, f)
            }
        }
        DataType::Int64 => arr!(Int64Array, |v| v),
        DataType::UInt8 => arr!(UInt8Array, |v| u8::try_from(small(250)(v).abs())
            .unwrap_or_default()),
        DataType::UInt16 => arr!(UInt16Array, |v| u16::try_from(small(6000)(v).abs())
            .unwrap_or_default()),
        DataType::UInt32 => arr!(UInt32Array, |v| u32::try_from(v.rem_euclid(4_000_000_000))
            .unwrap_or_default()),
        DataType::UInt64 => arr!(UInt64Array, i64::cast_unsigned),
        DataType::Float32 => arr!(Float32Array, |v| f32::from(
            i16::try_from(v % 1000).unwrap_or_default()
        ) / 8.0),
        DataType::Float64 => arr!(Float64Array, |v| f64::from(
            i32::try_from(v % 100_000).unwrap_or_default()
        ) / 7.0),
        DataType::Utf8 => arr!(StringArray, text),
        DataType::LargeUtf8 => arr!(LargeStringArray, text),
        DataType::Utf8View => arr!(StringViewArray, text),
        DataType::Binary => arr!(BinaryArray, bytes),
        DataType::LargeBinary => arr!(LargeBinaryArray, bytes),
        DataType::BinaryView => arr!(BinaryViewArray, bytes),
        DataType::Timestamp(unit, tz) => {
            let (v, tz) = (seeds.to_vec(), tz.clone());
            match unit {
                TimeUnit::Second => Arc::new(TimestampSecondArray::from(v).with_timezone_opt(tz)),
                TimeUnit::Millisecond => {
                    Arc::new(TimestampMillisecondArray::from(v).with_timezone_opt(tz))
                }
                TimeUnit::Microsecond => {
                    Arc::new(TimestampMicrosecondArray::from(v).with_timezone_opt(tz))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(TimestampNanosecondArray::from(v).with_timezone_opt(tz))
                }
            }
        }
        DataType::Decimal128(p, s) => Arc::new(
            Decimal128Array::from_iter(seeds.iter().map(|s| s.map(i128::from)))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::Decimal256(p, s) => Arc::new(
            Decimal256Array::from_iter(seeds.iter().map(|s| s.map(|v| i256::from_i128(v.into()))))
                .with_precision_and_scale(*p, *s)
                .unwrap(),
        ),
        DataType::Date64 => arr!(Date64Array, |v| v),
        DataType::Time32(TimeUnit::Second) => arr!(Time32SecondArray, |v| (v % 86_400) as i32),
        DataType::Time32(_) => arr!(Time32MillisecondArray, |v| (v % 86_400_000) as i32),
        DataType::Time64(TimeUnit::Microsecond) => arr!(Time64MicrosecondArray, |v| v),
        DataType::Time64(_) => arr!(Time64NanosecondArray, |v| v),
        DataType::Duration(TimeUnit::Second) => arr!(DurationSecondArray, |v| v),
        DataType::Duration(TimeUnit::Millisecond) => arr!(DurationMillisecondArray, |v| v),
        DataType::Duration(TimeUnit::Microsecond) => arr!(DurationMicrosecondArray, |v| v),
        DataType::Duration(TimeUnit::Nanosecond) => arr!(DurationNanosecondArray, |v| v),
        DataType::Interval(IntervalUnit::YearMonth) => arr!(IntervalYearMonthArray, |v| v as i32),
        DataType::Interval(IntervalUnit::DayTime) => {
            arr!(IntervalDayTimeArray, |v| IntervalDayTime::new(
                v as i32,
                (v >> 32) as i32
            ))
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => arr!(IntervalMonthDayNanoArray, |v| {
            IntervalMonthDayNano::new(v as i32, (v >> 32) as i32, v)
        }),
        DataType::FixedSizeBinary(w) => Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                seeds.iter().map(|s| {
                    s.map(|v| {
                        v.to_le_bytes()
                            .into_iter()
                            .cycle()
                            .take(*w as usize)
                            .collect::<Vec<u8>>()
                    })
                }),
                *w,
            )
            .unwrap(),
        ),
        DataType::Null => Arc::new(NullArray::new(seeds.len())),
        DataType::Dictionary(..) => Arc::new(DictionaryArray::<Int32Type>::from_iter(
            seeds
                .iter()
                .map(|s| s.map(|v| KEYS[v.rem_euclid(5) as usize])),
        )),
        DataType::Struct(fields) => {
            let columns = fields
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let shifted: Vec<_> = seeds
                        .iter()
                        .map(|s| s.map(|v| v.wrapping_add(i as i64)))
                        .collect();
                    build(f.data_type(), &shifted)
                })
                .collect();
            Arc::new(StructArray::new(
                fields.clone(),
                columns,
                Some(validity(seeds)),
            ))
        }
        DataType::List(f) | DataType::LargeList(f) | DataType::Map(f, _) => {
            let lens: Vec<usize> = seeds
                .iter()
                .map(|s| s.map_or(0, |v| usize::try_from(v.rem_euclid(4)).unwrap_or(0)))
                .collect();
            let items: Vec<Option<i64>> = seeds
                .iter()
                .zip(&lens)
                .flat_map(|(s, n)| (0..*n).map(move |j| s.map(|v| v.wrapping_add(j as i64 * 7))))
                .collect();
            let nulls = Some(validity(seeds));
            match dt {
                DataType::List(_) => Arc::new(ListArray::new(
                    Arc::clone(f),
                    OffsetBuffer::from_lengths(lens),
                    build(f.data_type(), &items),
                    nulls,
                )),
                DataType::LargeList(_) => Arc::new(LargeListArray::new(
                    Arc::clone(f),
                    OffsetBuffer::from_lengths(lens),
                    build(f.data_type(), &items),
                    nulls,
                )),
                _ => {
                    let DataType::Struct(entry_fields) = f.data_type() else {
                        unreachable!("map entries are structs")
                    };
                    let keys: ArrayRef =
                        Arc::new(Int32Array::from_iter_values(0..items.len() as i32));
                    let values = build(entry_fields[1].data_type(), &items);
                    let entries = StructArray::new(entry_fields.clone(), vec![keys, values], None);
                    Arc::new(MapArray::new(
                        Arc::clone(f),
                        OffsetBuffer::from_lengths(lens),
                        entries,
                        nulls,
                        false,
                    ))
                }
            }
        }
        other => panic!("type {other} has no test generator"),
    }
}

#[derive(Debug, Clone)]
struct BatchSpec {
    rows: usize,
    seeds: Vec<Vec<Option<i64>>>,
    column_weights: bool,
    constant: i64,
    weights: Vec<i64>,
    skip: usize,
}

fn batch_spec() -> impl Strategy<Value = BatchSpec> {
    (
        0usize..48,
        prop::collection::vec(prop::collection::vec(prop::option::of(any::<i64>()), 48), 6),
        any::<bool>(),
        any::<i64>(),
        prop::collection::vec(any::<i64>(), 48),
        0usize..48,
    )
        .prop_map(
            |(rows, seeds, column_weights, constant, weights, skip)| BatchSpec {
                rows,
                seeds,
                column_weights,
                constant,
                weights,
                skip,
            },
        )
}

fn make(cols: &[DataType], spec: &BatchSpec) -> Batch {
    let mut fields = Vec::new();
    let mut arrays = Vec::new();
    for (i, dt) in cols.iter().enumerate() {
        fields.push(Field::new(format!("c{i}"), dt.clone(), true));
        arrays.push(build(dt, &spec.seeds[i][..spec.rows]));
    }
    let record = RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(spec.rows)),
    )
    .unwrap();
    let weights = if spec.column_weights {
        Weights::Column(Int64Array::from(spec.weights[..spec.rows].to_vec()))
    } else {
        Weights::Const(spec.constant)
    };
    let batch = Batch::with_weights(record, weights).unwrap();
    // Filters and limits hand out slices: offset != 0 must round-trip too.
    match spec.rows {
        0 => batch,
        n => batch.slice(spec.skip % n, n - spec.skip % n).unwrap(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Whatever the types, nulls and weights, what is read is what was written.
    #[test]
    fn spill_round_trip(
        cols in prop::collection::vec(type_strategy(), 0..6),
        specs in prop::collection::vec(batch_spec(), 1..4),
        codec in prop_oneof![Just(SpillCodec::None), Just(SpillCodec::Lz4), Just(SpillCodec::Zstd)],
    ) {
        let dir = TestDir::new();
        let manager = SpillManager::open(&dir.config(codec)).unwrap();
        let scope = manager.scope().unwrap();
        let pool = Pool::new(None);
        let batches: Vec<Batch> = specs.iter().map(|s| make(&cols, s)).collect();
        let mut w = scope.writer(batches[0].schema(), pool.reservation()).unwrap();
        for b in &batches {
            w.write(b).unwrap();
        }
        let file = w.finish().unwrap();
        prop_assert_eq!(read_all(&file, &pool).unwrap(), batches);
        prop_assert_eq!(pool.root.reserved(), 0);
        drop(file);
        prop_assert_eq!(manager.used_bytes(), 0);
        prop_assert!(entries(scope.dir()).is_empty());
    }
}

#[test]
fn cap_exceeded_is_a_spark_style_error_and_leaks_nothing() {
    let dir = TestDir::new();
    let manager = SpillManager::open(&SpillConfig {
        max_bytes: 4096,
        ..dir.config(SpillCodec::None)
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let pool = Pool::new(None);
    let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
    let batch = int_batch(0, 100, Weights::Const(1)); // ~900 bytes per frame
    let mut written = 0;
    let err = loop {
        match w.write(&batch) {
            Ok(()) => written += 1,
            Err(e) => break e,
        }
        assert!(written < 100, "cap never hit");
    };
    assert!(written >= 2);
    assert!(
        matches!(err, SpillError::CapExceeded { cap: 4096, .. }),
        "{err:?}"
    );
    assert_eq!(class(&err), ErrorClass::KrishivSpillCapExceeded);
    let text = err.to_string();
    assert!(text.starts_with("[KRISHIV_SPILL_CAP_EXCEEDED] "), "{text}");
    assert!(text.contains("krishiv.spill.maxBytes") && text.ends_with("SQLSTATE: 53100"));
    // The failed writer is finished and has already returned its bytes.
    assert_eq!(manager.used_bytes(), 0);
    assert!(matches!(w.write(&batch), Err(SpillError::Internal(_))));
    assert!(matches!(w.finish(), Err(SpillError::Internal(_))));
    assert!(entries(scope.dir()).is_empty());
    // A batch of another schema is a caller bug, not a spill failure.
    let other = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let mut w = scope.writer(other, pool.reservation()).unwrap();
    let err = w.write(&batch).unwrap_err();
    assert_eq!(class(&err), ErrorClass::InternalError);
    // The cap is shared: a finished file holds it until dropped.
    let file = write_file(&scope, &pool, &[int_batch(0, 100, Weights::Const(1))]);
    let held = manager.used_bytes();
    assert_eq!(held, file.bytes());
    let mut second = scope.writer(schema_i64(), pool.reservation()).unwrap();
    while second.write(&batch).is_ok() {}
    assert_eq!(manager.used_bytes(), held);
}

#[test]
fn drops_remove_files_scopes_and_the_session() {
    let dir = TestDir::new();
    let manager = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap();
    let session = manager.session_dir().to_path_buf();
    let scope = manager.scope().unwrap();
    let pool = Pool::new(None);

    // An unfinished writer leaves nothing.
    let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
    w.write(&int_batch(0, 10, Weights::Const(1))).unwrap();
    assert!(manager.used_bytes() > 0);
    assert_eq!(entries(scope.dir()), vec!["0.spill.tmp"]);
    drop(w);
    assert!(entries(scope.dir()).is_empty());
    assert_eq!(manager.used_bytes(), 0);

    // A finished file is deleted with its last handle, readers included.
    let file = write_file(&scope, &pool, &[int_batch(0, 10, Weights::Const(1))]);
    assert_eq!(entries(scope.dir()), vec!["1.spill"]);
    let reader = file.reader(pool.reservation()).unwrap();
    drop(file);
    assert_eq!(entries(scope.dir()), vec!["1.spill"]);
    assert_eq!(reader.count(), 1);
    assert!(entries(scope.dir()).is_empty());
    assert_eq!(manager.used_bytes(), 0);

    // The scope's directory goes with the scope, the session with the manager.
    let scope_dir = scope.dir().to_path_buf();
    let file = write_file(&scope, &pool, &[int_batch(0, 10, Weights::Const(1))]);
    drop(scope);
    assert!(scope_dir.exists(), "a live file keeps its scope alive");
    drop(file);
    assert!(!scope_dir.exists());
    drop(manager);
    assert!(!session.exists());
    assert_eq!(entries(&dir.0), Vec::<String>::new());
}

#[test]
fn cancel_during_write_leaves_the_spill_dir_empty_after_drop() {
    let dir = TestDir::new();
    let manager = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap();
    let scope = manager.scope().unwrap();
    let progress = Arc::new(AtomicUsize::new(0));
    let writer = {
        let scope = scope.clone();
        let progress = Arc::clone(&progress);
        std::thread::spawn(move || {
            let pool = Pool::new(None);
            let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
            let batch = int_batch(0, 4096, Weights::Const(1));
            loop {
                match w.write(&batch) {
                    Ok(()) => {
                        progress.fetch_add(1, Ordering::Release);
                    }
                    Err(e) => return e,
                }
            }
        })
    };
    while progress.load(Ordering::Acquire) < 3 {
        std::thread::sleep(Duration::from_millis(1));
    }
    scope.cancel().unwrap(); // the query is killed mid-spill
    let err = writer.join().unwrap();
    assert!(matches!(err, SpillError::Cancelled), "{err:?}");
    assert_eq!(class(&err), ErrorClass::OperationCanceled);
    assert!(err.to_string().ends_with("SQLSTATE: HY008"));
    assert!(scope.is_cancelled());
    assert!(matches!(
        scope.writer(schema_i64(), Pool::new(None).reservation()),
        Err(SpillError::Cancelled)
    ));
    drop(scope);
    assert_eq!(manager.used_bytes(), 0);
    assert_eq!(entries(manager.session_dir()), Vec::<String>::new());
    let session = manager.session_dir().to_path_buf();
    drop(manager);
    assert!(!session.exists());
}

#[test]
fn cancel_stops_writers_and_readers_and_is_idempotent() {
    let (_dir, manager, scope, pool) = env(SpillCodec::Zstd);
    let file = write_file(&scope, &pool, &[int_batch(0, 10, Weights::Const(1))]);
    let mut reader = file.reader(pool.reservation()).unwrap();
    let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
    w.write(&int_batch(0, 10, Weights::Const(1))).unwrap();

    scope.cancel().unwrap();
    scope.cancel().unwrap();
    assert!(
        entries(manager.session_dir()).is_empty(),
        "files are deleted at once"
    );
    assert!(matches!(
        w.write(&int_batch(0, 1, Weights::Const(1))),
        Err(SpillError::Cancelled)
    ));
    assert!(matches!(reader.next(), Some(Err(SpillError::Cancelled))));
    assert!(reader.next().is_none(), "a reader is fused after an error");
    assert!(matches!(
        file.reader(pool.reservation()),
        Err(SpillError::Cancelled)
    ));
    drop((w, reader, file));
    assert_eq!(manager.used_bytes(), 0);
    assert_eq!(pool.root.reserved(), 0);
}

#[test]
fn a_directory_removed_under_a_writer_is_an_io_error() {
    let (_dir, manager, scope, pool) = env(SpillCodec::Lz4);
    let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
    w.write(&int_batch(0, 10, Weights::Const(1))).unwrap();
    fs::remove_dir_all(scope.dir()).unwrap(); // not a cancel: someone deleted it
    let err = w.finish().unwrap_err();
    assert!(is_io(&err, "rename"), "{err:?}");
    assert_eq!(class(&err), ErrorClass::KrishivSpillIoError);
    assert!(err.to_string().ends_with("SQLSTATE: 58030"));
    assert_eq!(manager.used_bytes(), 0);
    // And the scope can no longer create files.
    let err = scope.writer(schema_i64(), pool.reservation()).unwrap_err();
    assert!(is_io(&err, "create"), "{err:?}");
}

#[test]
fn open_removes_sessions_of_dead_processes_only() {
    let dir = TestDir::new();
    dir.create();
    // A crashed process's session: lock file nobody holds, directory with data.
    fs::create_dir_all(dir.0.join("krishiv-spill-123-abc-0/q0")).unwrap();
    fs::write(dir.0.join("krishiv-spill-123-abc-0/q0/0.spill"), b"junk").unwrap();
    fs::write(dir.0.join("krishiv-spill-123-abc-0.lock"), b"").unwrap();
    // A lock whose directory is already gone, and a directory whose lock is.
    fs::write(dir.0.join("krishiv-spill-124-abc-0.lock"), b"").unwrap();
    fs::create_dir_all(dir.0.join("krishiv-spill-125-abc-0/q0")).unwrap();
    // Not ours: other names, bad ids, plain files with our prefix.
    fs::write(dir.0.join("notes.txt"), b"keep").unwrap();
    fs::create_dir_all(dir.0.join("krishiv-spill-NOT_AN_ID")).unwrap();
    fs::write(dir.0.join("krishiv-spill-NOT_AN_ID.lock"), b"").unwrap();
    fs::write(dir.0.join("krishiv-spill-126"), b"a file, not a session").unwrap();

    let live = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap();
    assert_eq!(live.stale_sessions_removed(), 3);
    let live_session = live.session_dir().to_path_buf();
    let live_name = live_session
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let mut expected = vec![
        "krishiv-spill-126".to_owned(),
        "krishiv-spill-NOT_AN_ID".to_owned(),
        "krishiv-spill-NOT_AN_ID.lock".to_owned(),
        live_name.clone(),
        format!("{live_name}.lock"),
        "notes.txt".to_owned(),
    ];
    expected.sort();
    assert_eq!(entries(&dir.0), expected);

    // A second process-like opener must leave the live session alone.
    let scope = live.scope().unwrap();
    let second = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap();
    assert_eq!(second.stale_sessions_removed(), 0);
    assert!(scope.dir().exists());
    drop(second);
    assert!(live_session.exists());
    drop((scope, live));
    assert!(!live_session.exists());
}

#[cfg(unix)]
#[test]
fn open_never_follows_planted_symlinks() {
    let dir = TestDir::new();
    let victim = TestDir::new();
    dir.create();
    victim.create();
    fs::write(victim.0.join("precious"), b"data").unwrap();
    std::os::unix::fs::symlink(&victim.0, dir.0.join("krishiv-spill-777-a-0")).unwrap();
    fs::write(dir.0.join("krishiv-spill-777-a-0.lock"), b"").unwrap();
    let manager = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap();
    assert!(victim.0.join("precious").exists());
    drop(manager);
    assert!(victim.0.join("precious").exists());
}

#[cfg(unix)]
#[test]
fn open_checks_who_owns_and_can_write_the_directory() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let dir = TestDir::new();
    dir.create();
    for (mode, ok) in [
        (0o777, false),
        (0o775, false),
        (0o757, false),
        (0o755, true),
    ] {
        fs::set_permissions(&dir.0, fs::Permissions::from_mode(mode)).unwrap();
        let opened = SpillManager::open(&dir.config(SpillCodec::Lz4));
        match (opened, ok) {
            (Ok(_), true) => {}
            (Err(e), false) => assert!(is_io(&e, "use"), "{mode:o}: {e:?}"),
            (other, _) => panic!("mode {mode:o}: {other:?}"),
        }
    }
    // Another owner is refused, whatever the mode.
    let meta = fs::metadata(&dir.0).unwrap();
    assert!(dir::validate_owner_and_mode(&meta, meta.uid()).is_ok());
    assert!(dir::validate_owner_and_mode(&meta, meta.uid() + 1).is_err());
    // A symlink is refused, not followed.
    let link = TestDir::new();
    std::os::unix::fs::symlink(&dir.0, &link.0).unwrap();
    let err = SpillManager::open(&dir.config(SpillCodec::Lz4).with_dir(&link.0)).unwrap_err();
    assert!(is_io(&err, "use"), "{err:?}");
    // What we create is private.
    let manager = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap();
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(manager.session_dir()), 0o700);
    let scope = manager.scope().unwrap();
    let file = write_file(
        &scope,
        &Pool::new(None),
        &[int_batch(0, 3, Weights::Const(1))],
    );
    assert_eq!((mode(scope.dir()), mode(file.path())), (0o700, 0o600));
    let fresh = TestDir::new();
    SpillManager::open(&fresh.config(SpillCodec::Lz4)).unwrap();
    assert_eq!(mode(&fresh.0), 0o700);
}

#[test]
fn network_file_systems_are_recognised() {
    let mounts = "/dev/sda1 / ext4 rw 0 0\nsrv:/x /mnt/nfs nfs4 rw 0 0\nsrv:/y /mnt/nfs/local ext4 rw 0 0\nfuse /home/u/fs fuse.sshfs rw 0 0\n";
    let at = |p: &str| dir::network_fs_in(mounts, Path::new(p));
    assert_eq!(at("/mnt/nfs/spill"), Some("nfs4".to_owned()));
    assert_eq!(at("/home/u/fs/q"), Some("fuse.sshfs".to_owned()));
    assert_eq!(at("/mnt/nfs/local/q"), None, "the longest mount point wins");
    assert_eq!(at("/var/tmp"), None);
}

#[test]
fn mount_types_are_found_by_longest_mount_point() {
    let mounts = "/dev/sda1 / ext4 rw 0 0\ntmpfs /run/user/1000 tmpfs rw 0 0\ntmpfs /tmp tmpfs rw 0 0\n/dev/sda2 /var/tmp ext4 rw 0 0\n";
    let at = |p: &str| dir::mount_type_in(mounts, Path::new(p));
    assert_eq!(at("/run/user/1000/krishiv-spill").as_deref(), Some("tmpfs"));
    assert_eq!(at("/tmp/x").as_deref(), Some("tmpfs"));
    assert_eq!(at("/var/tmp/krishiv-spill-1000").as_deref(), Some("ext4"));
    assert_eq!(at("/home/u/.cache").as_deref(), Some("ext4"));
}

#[test]
fn the_default_spill_dir_avoids_ram_file_systems() {
    let c = |s: &str| std::path::PathBuf::from(s);
    let candidates = [
        c("/home/u/.cache/krishiv/spill"),
        c("/var/tmp/s"),
        c("/tmp/s"),
    ];
    // The first disk-backed candidate wins.
    let on_ram = |p: &Path| p.starts_with("/home/u") || p.starts_with("/tmp");
    assert_eq!(dir::choose_default(&candidates, on_ram), c("/var/tmp/s"));
    assert_eq!(dir::choose_default(&candidates, |_| false), candidates[0]);
    // All on RAM: still a usable directory (the first preference).
    assert_eq!(dir::choose_default(&candidates, |_| true), candidates[0]);
    assert_eq!(dir::choose_default(&[], |_| true), std::env::temp_dir());
}

#[test]
fn open_fails_typed_when_the_directory_is_unusable() {
    let dir = TestDir::new();
    fs::write(&dir.0, b"a file").unwrap();
    let err = SpillManager::open(&dir.config(SpillCodec::Lz4)).unwrap_err();
    assert!(matches!(err, SpillError::Io { .. }), "{err:?}");
    assert_eq!(class(&err), ErrorClass::KrishivSpillIoError);
    // Below a file nothing can be created either.
    let nested = SpillConfig {
        dir: dir.0.join("sub"),
        ..SpillConfig::default()
    };
    assert!(is_io(&SpillManager::open(&nested).unwrap_err(), "create"));
}

/// A finished two-frame file plus the bookkeeping to reopen modified bytes.
struct Sample {
    _dir: TestDir,
    _manager: SpillManager,
    scope: SpillScope,
    pool: Pool,
    schema: SchemaRef,
    path: PathBuf,
    bytes: Vec<u8>,
    rows: u64,
    batches: u64,
}

impl Sample {
    fn new(codec: SpillCodec) -> Self {
        let dir = TestDir::new();
        let manager = SpillManager::open(&dir.config(codec)).unwrap();
        let scope = manager.scope().unwrap();
        let pool = Pool::new(None);
        let file = write_file(
            &scope,
            &pool,
            &[
                int_batch(0, 20, Weights::Const(5)),
                int_batch(20, 3, Weights::Column(Int64Array::from(vec![1, -1, 2]))),
            ],
        );
        let path = scope.dir().join("corrupt.spill");
        fs::copy(file.path(), &path).unwrap();
        let bytes = fs::read(&path).unwrap();
        Self {
            schema: Arc::clone(file.schema()),
            rows: file.num_rows(),
            batches: file.num_batches(),
            _dir: dir,
            _manager: manager,
            scope,
            pool,
            path,
            bytes,
        }
    }

    /// Replaces the file's content and opens it as if `bytes.len()` were
    /// what the writer wrote, so the in-stream checks run.
    fn open(&self, content: &[u8]) -> SpillFile {
        fs::write(&self.path, content).unwrap();
        // Charge the cap like a real file; the `SpillFile` returns it.
        self.scope
            .inner_for_tests()
            .manager()
            .try_charge(content.len() as u64)
            .unwrap();
        SpillFile::new(
            self.scope.inner_for_tests(),
            self.path.clone(),
            Arc::clone(&self.schema),
            content.len() as u64,
            self.rows,
            self.batches,
        )
    }

    fn restore(&self) {
        fs::write(&self.path, &self.bytes).unwrap();
    }

    fn read(&self, content: &[u8]) -> Result<Vec<Batch>, SpillError> {
        let file = self.open(content);
        // Don't let the drop delete the pristine copy for the next call.
        let result = read_all(&file, &self.pool);
        self.restore();
        result
    }
}

impl Sample {
    /// Reading `content` fails as a corrupt file whose message has `what`.
    #[track_caller]
    fn expect_corrupt(&self, content: &[u8], what: &str) {
        let err = is_corrupt(self.read(content)).to_string();
        assert!(err.contains(what), "{what}: {err}");
    }
}

#[track_caller]
fn is_corrupt(r: Result<Vec<Batch>, SpillError>) -> SpillError {
    match r {
        Err(e @ SpillError::Corrupt { .. }) => e,
        other => panic!("expected a corrupt-file error, got {other:?}"),
    }
}

#[test]
fn every_single_byte_flip_is_an_error_never_wrong_data_or_a_panic() {
    for codec in [SpillCodec::None, SpillCodec::Lz4, SpillCodec::Zstd] {
        let s = Sample::new(codec);
        for i in 0..s.bytes.len() {
            let mut bad = s.bytes.clone();
            bad[i] ^= 0xA5;
            let err = is_corrupt(s.read(&bad));
            assert_eq!(class(&err), ErrorClass::KrishivSpillFileCorrupt, "byte {i}");
        }
    }
}

#[test]
fn every_truncation_is_a_typed_error() {
    let s = Sample::new(SpillCodec::Lz4);
    for len in 0..s.bytes.len() {
        let err = is_corrupt(s.read(&s.bytes[..len]));
        let text = err.to_string();
        assert!(
            text.starts_with("[KRISHIV_SPILL_FILE_CORRUPT] "),
            "{len}: {text}"
        );
    }
}

#[test]
fn a_file_cut_at_a_frame_boundary_misses_its_footer() {
    let s = Sample::new(SpillCodec::None);
    let footer = FRAME_HEADER_LEN + format::FOOTER_PAYLOAD_LEN;
    s.expect_corrupt(&s.bytes[..s.bytes.len() - footer], "no footer");
}

#[test]
fn header_version_and_framing_errors_are_reported() {
    let s = Sample::new(SpillCodec::None);
    let edit = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = s.bytes.clone();
        f(&mut b);
        b
    };
    s.expect_corrupt(&edit(&|b| b[4] = 2), "unsupported format version 2");
    s.expect_corrupt(&edit(&|b| b[0] = b'X'), "bad magic");
    s.expect_corrupt(&edit(&|b| b[FILE_HEADER_LEN] = 9), "unknown frame kind 9");
    s.expect_corrupt(
        &edit(&|b| b.extend_from_slice(b"extra")),
        "data after the footer",
    );
    // A length that runs past the end of the file is refused before any
    // memory is reserved for it.
    let huge = u64::MAX.to_le_bytes();
    let bytes = edit(&|b| b[FILE_HEADER_LEN + 9..FILE_HEADER_LEN + 17].copy_from_slice(&huge));
    s.expect_corrupt(&bytes, "runs past the end");
    assert_eq!(s.pool.root.reserved(), 0);
}

#[test]
fn a_size_that_differs_from_what_was_written_is_refused_at_open() {
    let s = Sample::new(SpillCodec::Lz4);
    let file = s.open(&s.bytes);
    fs::write(file.path(), &s.bytes[..s.bytes.len() - 1]).unwrap();
    let err = file.reader(s.pool.reservation()).unwrap_err();
    assert!(matches!(err, SpillError::Corrupt { .. }), "{err:?}");
    fs::remove_file(file.path()).unwrap();
    let err = file.reader(s.pool.reservation()).unwrap_err();
    assert!(is_io(&err, "open"), "{err:?}");
}

/// The (header, payload) of every frame of a spill file.
fn frames(bytes: &[u8]) -> Vec<(format::FrameHeader, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pos = FILE_HEADER_LEN;
    while pos < bytes.len() {
        let mut raw = [0u8; FRAME_HEADER_LEN];
        raw.copy_from_slice(&bytes[pos..pos + FRAME_HEADER_LEN]);
        let h = format::FrameHeader::parse(&raw);
        let end = pos + FRAME_HEADER_LEN + usize::try_from(h.len).unwrap();
        out.push((h, bytes[pos + FRAME_HEADER_LEN..end].to_vec()));
        pos = end;
    }
    out
}

/// `bytes` with the payload of frame `index` replaced and the checksum
/// recomputed, so only the content checks can object.
fn reseal(bytes: &[u8], index: usize, payload: &[u8]) -> Vec<u8> {
    let mut out = bytes[..FILE_HEADER_LEN].to_vec();
    for (i, (h, old)) in frames(bytes).into_iter().enumerate() {
        let body: &[u8] = if i == index { payload } else { &old };
        let header = format::FrameHeader {
            len: body.len() as u64,
            ..h
        };
        let mut crc = format::Crc32::new();
        crc.update(body);
        out.extend_from_slice(&header.seal(crc));
        out.extend_from_slice(body);
    }
    out
}

#[test]
fn consistent_but_wrong_content_is_caught_by_the_content_checks() {
    let s = Sample::new(SpillCodec::None);
    let payload = |i: usize| frames(&s.bytes)[i].1.clone();
    let expect = |bytes: Vec<u8>, what: &str| s.expect_corrupt(&bytes, what);
    assert_eq!(reseal(&s.bytes, 0, &payload(0)), s.bytes);
    // The footer lies about the row count, or has the wrong length.
    let mut lie = payload(2);
    lie[8] ^= 1;
    expect(reseal(&s.bytes, 2, &lie), "footer says");
    expect(reseal(&s.bytes, 2, &payload(2)[..8]), "malformed footer");
    // A frame that is not an IPC stream, holds none, or two.
    expect(reseal(&s.bytes, 0, b"not arrow"), "undecodable");
    expect(reseal(&s.bytes, 0, &[]), "no batch");
    expect(
        reseal(&s.bytes, 0, &[payload(0), payload(0)].concat()),
        "undecodable",
    );
    // Frames of another schema, and frames whose kind contradicts their columns.
    expect(reseal(&s.bytes, 0, &payload(1)), "schema differs");
    expect(reseal(&s.bytes, 1, &payload(0)), "weights column");
    let other = {
        let dir = TestDir::new();
        let scope = SpillManager::open(&dir.config(SpillCodec::None))
            .unwrap()
            .scope()
            .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("z", DataType::Int64, false)]));
        let record = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![1]))],
        )
        .unwrap();
        let mut w = scope.writer(schema, s.pool.reservation()).unwrap();
        w.write(&Batch::new(record)).unwrap();
        frames(&fs::read(w.finish().unwrap().path()).unwrap())[0]
            .1
            .clone()
    };
    expect(reseal(&s.bytes, 0, &other), "schema differs");
}

#[test]
fn reader_streams_a_file_far_larger_than_its_memory() {
    for codec in [SpillCodec::None, SpillCodec::Lz4] {
        let (_dir, _manager, scope, writer_pool) = env(codec);
        let batches: Vec<Batch> = (0..40)
            .map(|i| int_batch(i * 1000, 1000, Weights::Const(1)))
            .collect();
        let file = write_file(&scope, &writer_pool, &batches);
        let one = batches[0].data().get_array_memory_size();
        let total: usize = 40 * one;

        // Room for about two batches' worth: 40 batches stream through.
        let pool = Pool::new(Some(one * 5));
        let mut reader = file.reader(pool.reservation()).unwrap();
        let mut rows = 0;
        let mut peak = 0;
        for b in reader.by_ref() {
            rows += b.unwrap().num_rows();
            peak = peak.max(pool.root.reserved());
        }
        assert_eq!(rows, 40_000);
        assert!(peak > 0 && peak < total / 4, "{codec:?}: {peak} of {total}");
        drop(reader);
        assert_eq!(pool.root.reserved(), 0);

        // Not even one frame fits: a typed, retryable memory error.
        let blocker_pool = Pool::new(Some(one * 3));
        let mut blocker = blocker_pool.reservation();
        blocker.try_grow(one * 3 - 16).unwrap();
        let mut reader = file.reader(blocker_pool.reservation()).unwrap();
        for _ in 0..2 {
            let err = reader.next().unwrap().unwrap_err();
            assert_eq!(class(&err), ErrorClass::UnableToAcquireMemory);
        }
        blocker.shrink(one * 3 - 16);
        assert_eq!(
            reader
                .by_ref()
                .map(|b| b.unwrap().num_rows())
                .sum::<usize>(),
            40_000
        );
        drop(reader);
        assert_eq!(blocker_pool.root.reserved(), 0);
    }
}

#[test]
fn reader_reservation_matches_what_decoding_really_holds() {
    for codec in [SpillCodec::None, SpillCodec::Lz4, SpillCodec::Zstd] {
        let (_dir, _manager, scope, pool) = env(codec);
        let batch = int_batch(0, 8192, Weights::Column(Int64Array::from(vec![1; 8192])));
        let file = write_file(&scope, &pool, std::slice::from_ref(&batch));
        let mut reader = file.reader(pool.reservation()).unwrap();
        let got = reader.next().unwrap().unwrap();
        // The real data: 8192 values of 8 bytes, plus the weights.
        let actual = 2 * 8192 * 8;
        let reserved = pool.root.reserved();
        assert_eq!(got, batch);
        assert!(
            (actual..=3 * actual).contains(&reserved),
            "{codec:?}: reserved {reserved}, data {actual}"
        );
    }
}

#[test]
fn writer_scratch_is_reserved_in_full_and_returned() {
    let (_dir, manager, scope, _) = env(SpillCodec::Lz4);
    let batch = int_batch(0, 1000, Weights::Column(Int64Array::from(vec![1; 1000])));

    // No room: refused with a memory error, nothing written, and the call can
    // be repeated once memory is free.
    let pool = Pool::new(Some(150 << 10));
    let mut blocker = pool.reservation();
    blocker.try_grow(140 << 10).unwrap();
    let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
    let err = w.write(&batch).unwrap_err();
    assert_eq!(class(&err), ErrorClass::UnableToAcquireMemory);
    assert_eq!(manager.used_bytes(), w.bytes_written());
    blocker.shrink(140 << 10);
    w.write(&batch).unwrap();
    assert_eq!((w.num_batches(), w.num_rows()), (1, 1000));
    assert_eq!(
        pool.root.reserved(),
        0,
        "scratch is returned after each write"
    );
    drop(w);
    assert_eq!(manager.used_bytes(), 0);

    // Memory the reservation already holds is the operator's state, not scratch.
    let pool = Pool::new(Some(1 << 20));
    let mut reservation = pool.reservation();
    reservation.try_grow(1 << 20).unwrap();
    let mut w = scope.writer(schema_i64(), reservation).unwrap();
    let err = w.write(&batch).unwrap_err();
    assert_eq!(class(&err), ErrorClass::UnableToAcquireMemory);
}

#[test]
fn concurrent_writers_never_exceed_the_cap() {
    let dir = TestDir::new();
    let manager = SpillManager::open(&SpillConfig {
        max_bytes: 200_000,
        ..dir.config(SpillCodec::None)
    })
    .unwrap();
    let peak = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let (manager, peak) = (manager.clone(), Arc::clone(&peak));
            std::thread::spawn(move || {
                let scope = manager.scope().unwrap();
                let pool = Pool::new(None);
                let mut w = scope.writer(schema_i64(), pool.reservation()).unwrap();
                let batch = int_batch(0, 500, Weights::Const(1));
                let mut hit = false;
                for _ in 0..200 {
                    if let Err(e) = w.write(&batch) {
                        assert!(matches!(e, SpillError::CapExceeded { .. }), "{e:?}");
                        hit = true;
                        break;
                    }
                    let used = usize::try_from(manager.used_bytes()).unwrap();
                    peak.fetch_max(used, Ordering::Relaxed);
                }
                hit
            })
        })
        .collect();
    let hits = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .filter(|h| *h)
        .count();
    assert!(hits >= 1, "no writer reached the cap");
    assert!(peak.load(Ordering::Relaxed) <= 200_000);
    assert_eq!(manager.used_bytes(), 0);
}

#[test]
fn concurrent_starts_on_one_directory_all_succeed() {
    let dir = TestDir::new();
    dir.create();
    let threads: Vec<_> = (0..6)
        .map(|_| {
            let config = dir.config(SpillCodec::Lz4);
            std::thread::spawn(move || {
                for _ in 0..40 {
                    let m = SpillManager::open(&config).unwrap();
                    drop(m.scope().unwrap());
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(entries(&dir.0), Vec::<String>::new());
}

#[test]
fn config_keys_drive_the_spill_config() {
    let defaults = SpillConfig::default();
    assert_eq!(defaults.codec, SpillCodec::Lz4);
    assert_eq!(defaults.max_bytes, 64 << 30);
    assert_eq!(defaults.zstd_level, 1);
    // `.../krishiv/spill` under the user cache dir, or `krishiv-spill-<uid>`.
    let d = defaults.dir.to_string_lossy().into_owned();
    assert!(
        d.contains("krishiv/spill") || d.contains("krishiv-spill-"),
        "{d}"
    );

    let (config, _) = Config::from_pairs([
        ("krishiv.spill.dir", "/var/tmp/krishiv-x"),
        ("krishiv.spill.maxBytes", "10g"),
        ("krishiv.spill.compression", "ZSTD"),
        ("krishiv.spill.zstd.level", "7"),
    ])
    .unwrap();
    let c = SpillConfig::from_config(&config);
    assert_eq!(c.dir, PathBuf::from("/var/tmp/krishiv-x"));
    assert_eq!(c.max_bytes, 10 << 30);
    assert_eq!(c.codec, SpillCodec::Zstd);
    assert_eq!(c.zstd_level, 7);

    assert_eq!(SpillCodec::from_name("none"), Some(SpillCodec::None));
    assert_eq!(SpillCodec::from_name("lz4"), Some(SpillCodec::Lz4));
    assert_eq!(SpillCodec::from_name("snappy"), None);
}

#[test]
fn the_zstd_level_is_validated_when_the_directory_opens() {
    let dir = TestDir::new();
    let bad = SpillConfig {
        codec: SpillCodec::Zstd,
        zstd_level: 100,
        ..dir.config(SpillCodec::Zstd)
    };
    let err = SpillManager::open(&bad).unwrap_err();
    assert_eq!(class(&err), ErrorClass::KrishivInvalidConfigValue);
    assert!(
        err.to_string().contains("krishiv.spill.zstd.level"),
        "{err}"
    );
    assert!(!dir.0.exists() || entries(&dir.0).is_empty());
    let huge = SpillConfig {
        zstd_level: 1 << 40,
        ..bad
    };
    let err = SpillManager::open(&huge).unwrap_err();
    assert!(err.to_string().contains("1099511627776"), "{err}");
    // Levels are only checked for the codec that uses them.
    SpillManager::open(&SpillConfig {
        codec: SpillCodec::Lz4,
        zstd_level: 100,
        ..dir.config(SpillCodec::Lz4)
    })
    .unwrap();
    // A level other than the default compresses and reads back.
    let manager = SpillManager::open(&SpillConfig {
        zstd_level: 9,
        ..dir.config(SpillCodec::Zstd)
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    let pool = Pool::new(None);
    let batch = int_batch(0, 5000, Weights::Const(1));
    let file = write_file(&scope, &pool, std::slice::from_ref(&batch));
    assert_eq!(read_all(&file, &pool).unwrap(), vec![batch]);
}

#[test]
fn error_classes_and_messages() {
    let io = SpillError::io(
        "write to",
        Path::new("/x/0.spill.tmp"),
        std::io::Error::other("No space left on device"),
    );
    assert_eq!(
        io.to_string(),
        "[KRISHIV_SPILL_IO_ERROR] Spill I/O failed while trying to write to /x/0.spill.tmp: \
         No space left on device. SQLSTATE: 58030"
    );
    let full = SpillError::io(
        "write to",
        Path::new("/x/0.spill.tmp"),
        std::io::ErrorKind::StorageFull.into(),
    );
    assert!(
        full.to_string().contains("the spill disk is full"),
        "{full}"
    );
    let corrupt = SpillError::corrupt(Path::new("/x/1.spill"), "truncated in a frame");
    assert_eq!(corrupt.error_class(), ErrorClass::KrishivSpillFileCorrupt);
    assert!(corrupt.to_string().ends_with("SQLSTATE: XX001"));
}

#[test]
fn crc32_matches_the_standard_check_values() {
    let crc = |parts: &[&[u8]]| {
        let mut c = format::Crc32::new();
        for p in parts {
            c.update(p);
        }
        c.finish()
    };
    // The CRC-32/ISO-HDLC check value, however the input is split.
    assert_eq!(crc(&[b"123456789"]), 0xCBF4_3926);
    assert_eq!(crc(&[b"1234", b"56789"]), 0xCBF4_3926);
    assert_eq!(crc(&[b"12345678", b"9"]), 0xCBF4_3926);
    assert_eq!(crc(&[]), 0);
    assert_eq!(crc(&[b"a"]), 0xE8B7_BE43);
}
