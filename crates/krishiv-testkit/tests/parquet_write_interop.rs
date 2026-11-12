// SPDX-License-Identifier: Apache-2.0
//! The Parquet writer against other engines' readers (P2-T082a).
//!
//! `tests/fixtures/parquet/written/<variant>.parquet` are files the writer
//! produced from [`source`] with each variant's options (codecs, millisecond
//! timestamps, data page v2, small row groups). `expected.json` holds the
//! SHA-256 of the canonical rows of each (see [`canonical`]), and
//! `reads.json` what pyarrow, DuckDB and Spark 4.2 read from them, hashed
//! the same way, plus the Spark types and page-index presence pyarrow
//! reports (`tests/scripts/check_parquet_written.py`).
//!
//! Always checked: the other engines read exactly the rows the writer was
//! given; Krishiv's reader does too; and today's writer still writes these
//! files (same Parquet schema, footer metadata, codecs, encodings, row
//! groups and values), so a change to the writer's output fails here until
//! the fixtures are regenerated and re-checked by the other engines:
//!
//! ```text
//! KRISHIV_WRITE_FIXTURES=1 cargo nextest run -p krishiv-testkit \
//!     --test parquet_write_interop --run-ignored only
//! # then the commands in tests/scripts/check_parquet_written.py
//! ```
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{
    Int32Builder, Int64Builder, ListBuilder, MapBuilder, StringViewBuilder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, TimestampMicrosecondType, TimestampMillisecondType,
};
use arrow_array::{
    Array, ArrayRef, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, ListArray, MapArray, RecordBatch,
    StringViewArray, StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit};
use bytes::Bytes;
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    ByteRange, FileFormat, FileRef, FormatOptions, IoConfig, IoScheduler, ScanOptions, ScanRequest,
    ScanStream, StoreId,
};
use krishiv_memory::{MemoryConfig, MemoryPool, Reservation, TreePool};
use object_store::ObjectStore;
use object_store::memory::InMemory;
use object_store::path::Path as StorePath;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const ROWS: usize = 300;

/// The fixtures' directory.
fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/parquet/written")
}

/// The variants: a file name and the write options.
const VARIANTS: &[(&str, &[(&str, &str)])] = &[
    ("snappy", &[]),
    ("none", &[("compression", "none")]),
    ("gzip", &[("compression", "gzip")]),
    ("lz4", &[("compression", "lz4")]),
    ("lz4_raw", &[("compression", "lz4_raw")]),
    ("zstd", &[("compression", "zstd")]),
    ("millis", &[("outputTimestampType", "TIMESTAMP_MILLIS")]),
    (
        "v2",
        &[
            ("parquet.writer.version", "v2"),
            ("parquet.enable.dictionary", "false"),
        ],
    ),
    ("small_groups", &[("parquet.block.size", "4096")]),
    // The nested source ([`nested_source`]): nulls at every level of lists, structs and maps.
    ("nested", &[]),
    ("nested_nodict", &[("parquet.enable.dictionary", "false")]),
    (
        "nested_v2",
        &[
            ("parquet.writer.version", "v2"),
            ("parquet.enable.dictionary", "false"),
        ],
    ),
    (
        "nested_millis",
        &[("outputTimestampType", "TIMESTAMP_MILLIS")],
    ),
    ("nested_small_groups", &[("parquet.block.size", "4096")]),
];

/// The source rows `variant` is written from.
fn source_of(variant: &str) -> RecordBatch {
    if variant.starts_with("nested") {
        nested_source()
    } else {
        source()
    }
}

fn options(kv: &[(&str, &str)]) -> FormatOptions {
    let user = kv
        .iter()
        .map(|(k, v)| {
            (
                (*k).to_owned(),
                krishiv_common::secret::OptionValue::from_user(k, (*v).to_owned()),
            )
        })
        .collect();
    ParquetFormat
        .options()
        .parse(&user, &krishiv_common::config::Config::new())
        .unwrap()
        .0
}

fn nullable(i: usize) -> bool {
    i % 11 == 5
}

/// Every Spark type the engine has, nulls in every nullable column, edge
/// values (NaN, -0.0, infinities, extreme decimals, dates before 1582 and
/// after 9000, timestamps before 1900 and with sub-millisecond parts on
/// both sides of the epoch), nested struct, array and map columns.
fn source() -> RecordBatch {
    let n = |i: usize| !nullable(i);
    let idx = 0..ROWS;
    let i64s = |f: fn(i64) -> i64| -> Vec<Option<i64>> {
        (0..ROWS)
            .map(|i| n(i).then(|| f(i64::try_from(i).unwrap())))
            .collect()
    };
    let floats = [
        f64::NAN,
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.5e300,
        -2.25,
    ];
    let ts_values = i64s(|i| match i % 5 {
        // Before 1900, and the sub-millisecond parts around the epoch.
        0 => -2_208_988_800_000_001 - i,
        1 => -500 - i * 1_000,
        2 => 1_577_836_800_123_456 + i,
        3 => i * 86_400_000_037,
        _ => -i * 1_000_001,
    });
    let st = StructArray::new(
        Fields::from(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Utf8View, true),
            Field::new(
                "t",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
        ]),
        vec![
            Arc::new(Int32Array::from_iter(
                idx.clone()
                    .map(|i| (i % 4 != 1).then(|| i32::try_from(i * 2).unwrap())),
            )),
            Arc::new(StringViewArray::from_iter(
                idx.clone().map(|i| Some(format!("y{i}"))),
            )),
            Arc::new(
                TimestampMicrosecondArray::from(
                    ts_values.iter().rev().copied().collect::<Vec<_>>(),
                )
                .with_timezone("UTC"),
            ),
        ],
        Some(idx.clone().map(n).collect()),
    );
    let mut arr = ListBuilder::new(Int64Builder::new());
    for i in idx.clone() {
        if n(i) {
            for k in 0..i % 4 {
                arr.values()
                    .append_value(i64::try_from(i * 10 + k).unwrap());
            }
            arr.append(true);
        } else {
            arr.append(false);
        }
    }
    let mut map = MapBuilder::new(None, StringViewBuilder::new(), Int32Builder::new());
    for i in idx.clone() {
        if n(i) {
            for k in 0..i % 3 {
                map.keys().append_value(format!("k{k}"));
                map.values()
                    .append_option((k != 1).then(|| i32::try_from(i + k).unwrap()));
            }
            map.append(true).unwrap();
        } else {
            map.append(false).unwrap();
        }
    }
    let columns: Vec<(&str, ArrayRef, bool)> = vec![
        (
            "id",
            Arc::new(Int32Array::from_iter_values(
                idx.clone().map(|i| i32::try_from(i).unwrap()),
            )),
            false,
        ),
        (
            "bool",
            Arc::new(BooleanArray::from_iter(
                idx.clone().map(|i| n(i).then_some(i % 3 == 0)),
            )),
            true,
        ),
        (
            "byte",
            Arc::new(Int8Array::from_iter(idx.clone().map(|i| {
                n(i).then(|| i8::from_le_bytes([u8::try_from(i % 256).unwrap()]))
            }))),
            true,
        ),
        (
            "short",
            Arc::new(Int16Array::from_iter(idx.clone().map(|i| {
                n(i).then(|| {
                    i16::from_le_bytes(u16::try_from(i * 211 % 65_536).unwrap().to_le_bytes())
                })
            }))),
            true,
        ),
        (
            "long",
            Arc::new(Int64Array::from(i64s(|i| {
                if i % 7 == 0 {
                    i64::MIN + i
                } else {
                    i * 1_000_000_007 - 99
                }
            }))),
            true,
        ),
        (
            "float",
            Arc::new(Float32Array::from_iter(idx.clone().map(|i| {
                // Every value of `floats` but 1.5e300 (infinite as f32) is exact.
                n(i).then(|| {
                    [
                        f32::NAN,
                        -0.0,
                        0.0,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        3.0e38,
                        -2.25,
                    ][i % floats.len()]
                })
            }))),
            true,
        ),
        (
            "double",
            Arc::new(Float64Array::from_iter(
                idx.clone().map(|i| n(i).then(|| floats[i % floats.len()])),
            )),
            true,
        ),
        (
            "dec9",
            Arc::new(
                Decimal128Array::from_iter(
                    idx.clone()
                        .map(|i| n(i).then(|| i128::try_from(i).unwrap() * 1234 - 300_000)),
                )
                .with_precision_and_scale(9, 2)
                .unwrap(),
            ),
            true,
        ),
        (
            "dec18",
            Arc::new(
                Decimal128Array::from_iter(
                    idx.clone()
                        .map(|i| n(i).then(|| i128::try_from(i).unwrap() * 98_765_431 - 7)),
                )
                .with_precision_and_scale(18, 4)
                .unwrap(),
            ),
            true,
        ),
        (
            "dec38",
            Arc::new(
                Decimal128Array::from_iter(idx.clone().map(|i| {
                    n(i).then(|| {
                        let max = 10_i128.pow(38) - 1;
                        match i % 3 {
                            0 => max,
                            1 => -max,
                            _ => i128::try_from(i).unwrap().pow(9) + 1,
                        }
                    })
                }))
                .with_precision_and_scale(38, 10)
                .unwrap(),
            ),
            true,
        ),
        (
            "s",
            Arc::new(StringViewArray::from_iter(idx.clone().map(|i| {
                n(i).then(|| format!("row-{i}-{}{}", "é".repeat(i % 4), "語".repeat(i % 2)))
            }))),
            true,
        ),
        (
            "bin",
            Arc::new(BinaryViewArray::from_iter(idx.clone().map(|i| {
                n(i).then(|| {
                    (0..i % 9)
                        .map(|k| u8::try_from((i * 31 + k) % 256).unwrap())
                        .collect::<Vec<u8>>()
                })
            }))),
            true,
        ),
        (
            "d",
            Arc::new(Date32Array::from_iter(idx.map(|i| {
                n(i).then(|| match i % 4 {
                    // 1000-01-01 and 9999-12-31.
                    0 => -354_285,
                    1 => 2_932_896,
                    _ => i32::try_from(i).unwrap() * 97 - 20_000,
                })
            }))),
            true,
        ),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(ts_values.clone()).with_timezone("UTC")),
            true,
        ),
        (
            "ntz",
            Arc::new(TimestampMicrosecondArray::from(ts_values)),
            true,
        ),
        ("st", Arc::new(st), true),
        ("arr", Arc::new(arr.finish()), true),
        ("m", Arc::new(map.finish()), true),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, a, null)| Field::new(*name, a.data_type().clone(), *null))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a, _)| a).collect(),
    )
    .unwrap()
}

/// A list column from per-row lengths (`None` is a null list) over `values`.
fn list_of(values: ArrayRef, lens: &[Option<usize>]) -> ArrayRef {
    let mut offsets = vec![0_i32];
    for l in lens {
        let next = offsets.last().copied().unwrap() + i32::try_from(l.unwrap_or(0)).unwrap();
        offsets.push(next);
    }
    assert_eq!(
        usize::try_from(*offsets.last().unwrap()).unwrap(),
        values.len()
    );
    Arc::new(ListArray::new(
        Arc::new(Field::new("item", values.data_type().clone(), true)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        values,
        Some(NullBuffer::from(
            lens.iter().map(Option::is_some).collect::<Vec<_>>(),
        )),
    ))
}

/// A map column from per-row lengths over `keys` (never null) and `values`.
fn map_of(keys: ArrayRef, values: ArrayRef, lens: &[Option<usize>]) -> ArrayRef {
    let mut offsets = vec![0_i32];
    for l in lens {
        let next = offsets.last().copied().unwrap() + i32::try_from(l.unwrap_or(0)).unwrap();
        offsets.push(next);
    }
    let entries = StructArray::new(
        Fields::from(vec![
            Field::new("keys", keys.data_type().clone(), false),
            Field::new("values", values.data_type().clone(), true),
        ]),
        vec![keys, values],
        None,
    );
    Arc::new(MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        OffsetBuffer::new(ScalarBuffer::from(offsets)),
        entries,
        Some(NullBuffer::from(
            lens.iter().map(Option::is_some).collect::<Vec<_>>(),
        )),
        false,
    ))
}

/// Row `i`'s length for a nested column of period `p`: null, empty, then 1..: one in `2p` rows is a
/// null container, one in `2p` an empty one.
fn len_at(i: usize, p: usize, max: usize) -> Option<usize> {
    match i % (p * 2) {
        0 => None,
        k if k == p => Some(0),
        k => Some(1 + k % max),
    }
}

/// Nested columns with a null at every level: null and empty lists, null elements, null structs
/// and null fields inside list elements, lists inside structs, maps with null, empty and list or
/// struct values, and decimals, dates and timestamps inside containers.
fn nested_source() -> RecordBatch {
    let idx = 0..ROWS;
    let total = |lens: &[Option<usize>]| lens.iter().map(|l| l.unwrap_or(0)).sum::<usize>();
    let int_at = |k: usize, m: usize| (!k.is_multiple_of(m)).then(|| i32::try_from(k * 3).unwrap());
    let str_at = |k: usize, m: usize| (k % m != 1).then(|| format!("s{k}é"));
    let ts_at = |k: usize| {
        (k % 4 != 2).then(|| match k % 3 {
            0 => -2_208_988_800_000_001 - i64::try_from(k).unwrap(),
            1 => 1_577_836_800_123_456 + i64::try_from(k).unwrap(),
            _ => -i64::try_from(k).unwrap() * 1_000_001,
        })
    };
    // list<int> with null elements
    let l_int = len_at_all(&idx, 5, 4);
    let li = list_of(
        Arc::new(Int32Array::from_iter(
            (0..total(&l_int)).map(|k| int_at(k, 3)),
        )),
        &l_int,
    );
    // list<struct<a:int, b:string>> with null struct elements and null fields
    let l_st = len_at_all(&idx, 7, 3);
    let n_st = total(&l_st);
    let elem = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8View, true),
        ]),
        vec![
            Arc::new(Int32Array::from_iter((0..n_st).map(|k| int_at(k, 4)))),
            Arc::new(StringViewArray::from_iter((0..n_st).map(|k| str_at(k, 3)))),
        ],
        Some((0..n_st).map(|k| k % 5 != 2).collect()),
    );
    let ls = list_of(Arc::new(elem), &l_st);
    // list<list<bigint>>: null and empty at both levels, null leaf values
    let outer = len_at_all(&idx, 6, 3);
    let inner: Vec<Option<usize>> = (0..total(&outer)).map(|k| len_at(k, 4, 3)).collect();
    let leaf = Int64Array::from_iter(
        (0..total(&inner)).map(|k| (k % 3 != 1).then(|| i64::try_from(k).unwrap() * 7 - 3)),
    );
    let ll = list_of(list_of(Arc::new(leaf), &inner), &outer);
    // struct<s:struct<v:int>, l:list<int>, m:map<string,int>>, null at each level
    let sl = len_at_all(&idx, 5, 3);
    let sm = len_at_all(&idx, 4, 3);
    let inner_s = StructArray::new(
        Fields::from(vec![Field::new("v", DataType::Int32, true)]),
        vec![Arc::new(Int32Array::from_iter(
            idx.clone().map(|i| int_at(i, 3)),
        ))],
        Some(idx.clone().map(|i| i % 6 != 1).collect()),
    );
    let s_list = list_of(
        Arc::new(Int32Array::from_iter((0..total(&sl)).map(|k| int_at(k, 4)))),
        &sl,
    );
    let s_map = map_of(
        Arc::new(StringViewArray::from_iter_values(
            (0..total(&sm)).map(|k| format!("k{k}")),
        )),
        Arc::new(Int32Array::from_iter((0..total(&sm)).map(|k| int_at(k, 2)))),
        &sm,
    );
    let sn = StructArray::new(
        Fields::from(vec![
            Field::new("s", inner_s.data_type().clone(), true),
            Field::new("l", s_list.data_type().clone(), true),
            Field::new("m", s_map.data_type().clone(), true),
        ]),
        vec![Arc::new(inner_s), s_list, s_map],
        Some(idx.clone().map(|i| i % 9 != 4).collect()),
    );
    // map<string, struct<p:int, q:string>>
    let mlen = len_at_all(&idx, 5, 3);
    let nm = total(&mlen);
    let ms_vals = StructArray::new(
        Fields::from(vec![
            Field::new("p", DataType::Int32, true),
            Field::new("q", DataType::Utf8View, true),
        ]),
        vec![
            Arc::new(Int32Array::from_iter((0..nm).map(|k| int_at(k, 3)))),
            Arc::new(StringViewArray::from_iter((0..nm).map(|k| str_at(k, 4)))),
        ],
        Some((0..nm).map(|k| k % 4 != 3).collect()),
    );
    let ms = map_of(
        Arc::new(StringViewArray::from_iter_values(
            (0..nm).map(|k| format!("m{k}")),
        )),
        Arc::new(ms_vals),
        &mlen,
    );
    // map<string, list<int>>: null, empty and populated list values
    let alen = len_at_all(&idx, 4, 3);
    let na = total(&alen);
    let a_lens: Vec<Option<usize>> = (0..na).map(|k| len_at(k, 3, 3)).collect();
    let ma = map_of(
        Arc::new(StringViewArray::from_iter_values(
            (0..na).map(|k| format!("a{k}")),
        )),
        list_of(
            Arc::new(Int32Array::from_iter(
                (0..total(&a_lens)).map(|k| int_at(k, 3)),
            )),
            &a_lens,
        ),
        &alen,
    );
    // map<int, map<string, bigint>>
    let olen = len_at_all(&idx, 6, 2);
    let no = total(&olen);
    let i_lens: Vec<Option<usize>> = (0..no).map(|k| len_at(k, 3, 2)).collect();
    let mm = map_of(
        Arc::new(Int32Array::from_iter_values(
            (0..no).map(|k| i32::try_from(k).unwrap()),
        )),
        map_of(
            Arc::new(StringViewArray::from_iter_values(
                (0..total(&i_lens)).map(|k| format!("i{k}")),
            )),
            Arc::new(Int64Array::from_iter(
                (0..total(&i_lens)).map(|k| (k % 2 != 0).then(|| i64::try_from(k).unwrap())),
            )),
            &i_lens,
        ),
        &olen,
    );
    // list<timestamp>, list<timestamp_ntz>, list<decimal(38,10)>, map<date, decimal(18,4)>
    let tlen = len_at_all(&idx, 5, 3);
    let nt = total(&tlen);
    let ts_l = list_of(
        Arc::new(TimestampMicrosecondArray::from_iter((0..nt).map(ts_at)).with_timezone("UTC")),
        &tlen,
    );
    let ntz_l = list_of(
        Arc::new(TimestampMicrosecondArray::from_iter((0..nt).map(ts_at))),
        &tlen,
    );
    let dec_l = list_of(
        Arc::new(
            Decimal128Array::from_iter((0..nt).map(|k| {
                (k % 4 != 1).then(|| match k % 3 {
                    0 => 10_i128.pow(38) - 1,
                    1 => -(10_i128.pow(38) - 1),
                    _ => i128::try_from(k).unwrap() - 9,
                })
            }))
            .with_precision_and_scale(38, 10)
            .unwrap(),
        ),
        &tlen,
    );
    let dlen = len_at_all(&idx, 4, 3);
    let nd = total(&dlen);
    let d_m = map_of(
        Arc::new(Date32Array::from_iter_values(
            (0..nd).map(|k| i32::try_from(k).unwrap() * 401 - 90_000),
        )),
        Arc::new(
            Decimal128Array::from_iter(
                (0..nd).map(|k| (k % 3 != 1).then(|| i128::try_from(k).unwrap() * 98_765_431 - 7)),
            )
            .with_precision_and_scale(18, 4)
            .unwrap(),
        ),
        &dlen,
    );
    let columns: Vec<(&str, ArrayRef, bool)> = vec![
        (
            "id",
            Arc::new(Int32Array::from_iter_values(
                idx.map(|i| i32::try_from(i).unwrap()),
            )),
            false,
        ),
        ("li", li, true),
        ("ls", ls, true),
        ("ll", ll, true),
        ("sn", Arc::new(sn), true),
        ("ms", ms, true),
        ("ma", ma, true),
        ("mm", mm, true),
        ("ts_l", ts_l, true),
        ("ntz_l", ntz_l, true),
        ("dec_l", dec_l, true),
        ("d_m", d_m, true),
    ];
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, a, null)| Field::new(*name, a.data_type().clone(), *null))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a, _)| a).collect(),
    )
    .unwrap()
}

/// [`len_at`] for every row.
fn len_at_all(rows: &std::ops::Range<usize>, p: usize, max: usize) -> Vec<Option<usize>> {
    rows.clone().map(|i| len_at(i, p, max)).collect()
}

/// The canonical JSON of value `i` of `a`: integers, dates and timestamps
/// (in microseconds) as decimal strings, decimals as unscaled integer
/// strings, floats as the little-endian hex of their `f64` bits, binary as
/// lowercase hex, maps as `[key, value]` pairs. Python's `json.dumps(...,
/// sort_keys=True, separators=(",", ":"), ensure_ascii=False)` of the same
/// structure is byte-identical.
fn canonical(a: &ArrayRef, i: usize) -> Value {
    if a.is_null(i) {
        return Value::Null;
    }
    let s = |v: String| Value::String(v);
    match a.data_type() {
        DataType::Boolean => Value::Bool(a.as_boolean().value(i)),
        DataType::Int8 => s(a.as_primitive::<Int8Type>().value(i).to_string()),
        DataType::Int16 => s(a.as_primitive::<Int16Type>().value(i).to_string()),
        DataType::Int32 => s(a.as_primitive::<Int32Type>().value(i).to_string()),
        DataType::Int64 => s(a.as_primitive::<Int64Type>().value(i).to_string()),
        DataType::Float32 => s(hex(
            &f64::from(a.as_primitive::<Float32Type>().value(i)).to_le_bytes()
        )),
        DataType::Float64 => s(hex(&a.as_primitive::<Float64Type>().value(i).to_le_bytes())),
        DataType::Decimal128(..) => s(a.as_primitive::<Decimal128Type>().value(i).to_string()),
        DataType::Utf8View => s(a.as_string_view().value(i).to_owned()),
        DataType::BinaryView => s(hex(a.as_binary_view().value(i))),
        DataType::Date32 => s(a.as_primitive::<Date32Type>().value(i).to_string()),
        DataType::Timestamp(TimeUnit::Microsecond, _) => s(a
            .as_primitive::<TimestampMicrosecondType>()
            .value(i)
            .to_string()),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            s((a.as_primitive::<TimestampMillisecondType>().value(i) * 1000).to_string())
        }
        DataType::Struct(fields) => {
            let st = a.as_struct();
            Value::Object(
                fields
                    .iter()
                    .zip(st.columns())
                    .map(|(f, c)| (f.name().clone(), canonical(c, i)))
                    .collect(),
            )
        }
        DataType::List(_) => {
            let v = a.as_list::<i32>().value(i);
            Value::Array((0..v.len()).map(|k| canonical(&v, k)).collect())
        }
        DataType::Map(..) => {
            let m = a.as_map();
            let (keys, values) = (m.keys(), m.values());
            let offsets = m.value_offsets();
            let (lo, hi) = (
                usize::try_from(offsets[i]).unwrap(),
                usize::try_from(offsets[i + 1]).unwrap(),
            );
            Value::Array(
                (lo..hi)
                    .map(|k| json!([canonical(keys, k), canonical(values, k)]))
                    .collect(),
            )
        }
        other => panic!("no canonical form for {other}"),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The SHA-256 of the canonical rows of `batch`, one JSON line each.
fn digest(batch: &RecordBatch) -> String {
    let mut h = Sha256::new();
    for i in 0..batch.num_rows() {
        let row: BTreeMap<String, Value> = batch
            .schema()
            .fields()
            .iter()
            .zip(batch.columns())
            .map(|(f, c)| (f.name().clone(), canonical(c, i)))
            .collect();
        h.update(serde_json::to_string(&row).unwrap().as_bytes());
        h.update(b"\n");
    }
    hex(&h.finalize())
}

/// What the writer was given for `variant`, as readers should see it:
/// timestamps rounded down to milliseconds for `millis`.
fn expected(variant: &str) -> RecordBatch {
    let batch = source_of(variant);
    if variant == "nested_millis" {
        return floor_nested_timestamps(&batch);
    }
    if variant != "millis" {
        return batch;
    }
    let floor = |c: &ArrayRef| -> ArrayRef {
        let us = c.as_primitive::<TimestampMicrosecondType>();
        Arc::new(
            us.unary::<_, TimestampMicrosecondType>(|v| v.div_euclid(1000) * 1000)
                .with_timezone("UTC"),
        )
    };
    let columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(f, c)| match f.name().as_str() {
            "ts" => floor(c),
            // The struct's timestamp too: rounding applies at any depth.
            "st" => {
                let st = c.as_struct();
                let children = st
                    .columns()
                    .iter()
                    .zip(st.fields())
                    .map(|(c, f)| {
                        if f.name() == "t" {
                            floor(c)
                        } else {
                            Arc::clone(c)
                        }
                    })
                    .collect();
                Arc::new(StructArray::new(
                    st.fields().clone(),
                    children,
                    st.nulls().cloned(),
                )) as ArrayRef
            }
            _ => Arc::clone(c),
        })
        .collect();
    RecordBatch::try_new(batch.schema(), columns).unwrap()
}

/// `batch` with the timestamps inside its list columns rounded down to milliseconds.
fn floor_nested_timestamps(batch: &RecordBatch) -> RecordBatch {
    let columns = batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(f, c)| match f.name().as_str() {
            // Not `ntz_l`: Spark writes timestamp_ntz in microseconds whatever `outputTimestampType` says.
            "ts_l" => {
                let l = c.as_list::<i32>();
                let us = l.values().as_primitive::<TimestampMicrosecondType>();
                let tz = us.timezone().map(str::to_owned);
                let floored =
                    us.unary::<_, TimestampMicrosecondType>(|v| v.div_euclid(1000) * 1000);
                let floored: ArrayRef = match tz {
                    Some(tz) => Arc::new(floored.with_timezone(tz)),
                    None => Arc::new(floored),
                };
                Arc::new(ListArray::new(
                    Arc::new(Field::new("item", floored.data_type().clone(), true)),
                    l.offsets().clone(),
                    floored,
                    l.nulls().cloned(),
                )) as ArrayRef
            }
            _ => Arc::clone(c),
        })
        .collect();
    RecordBatch::try_new(batch.schema(), columns).unwrap()
}

/// The file the writer produces today for `variant`.
fn write(name: &str, variant: &[(&str, &str)]) -> Bytes {
    let batch = source_of(name);
    let o = options(variant);
    ParquetFormat.check_write(&batch.schema(), &o).unwrap();
    let mut enc = ParquetFormat::encoder_with(&batch.schema(), &o, |p| p).unwrap();
    let mut out = Vec::new();
    // In several batches, as the executor hands them over.
    for k in (0..ROWS).step_by(64) {
        enc.write(&batch.slice(k, 64.min(ROWS - k))).unwrap();
        while let Some(b) = enc.take_bytes() {
            out.extend_from_slice(&b);
        }
    }
    let done = enc.finish().unwrap();
    out.extend_from_slice(&done.tail);
    Bytes::from(out)
}

/// Every row of `file` read by Krishiv's reader with the source schema.
fn krishiv_read(file: Bytes, schema: &SchemaRef) -> RecordBatch {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let store = Arc::new(InMemory::new());
    let sched = IoScheduler::new(IoConfig::default());
    let id = StoreId::new("memory://w");
    sched.register_store(id.clone(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let io = sched.context();
    let pool: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::default());
    let path = StorePath::from("f.parquet");
    rt.block_on(async {
        object_store::ObjectStoreExt::put(store.as_ref(), &path, file.into())
            .await
            .unwrap();
        let meta = object_store::ObjectStoreExt::head(store.as_ref(), &path)
            .await
            .unwrap();
        let file = FileRef::from_meta(id, &meta);
        let o = FormatOptions::defaults(ParquetFormat.options());
        let pid = pool.register("scan", None);
        let mut s = ScanStream::open(
            &io,
            &ParquetFormat,
            file.clone(),
            ByteRange::whole(file.size),
            &ScanRequest::all(Arc::clone(schema)),
            &o,
            Reservation::new(Arc::clone(&pool), pid),
            ScanOptions::default(),
        )
        .await
        .unwrap();
        let mut all = Vec::new();
        while let Some(b) = s.next_batch().await.unwrap() {
            all.push(b);
        }
        arrow_select::concat::concat_batches(&all[0].schema(), &all).unwrap()
    })
}

/// What decides how other engines read a file: its Parquet schema, footer
/// metadata (but the writer name), and per column chunk the codec,
/// encodings, value counts and page-index presence.
fn shape(file: Bytes) -> String {
    let reader = SerializedFileReader::new(file).unwrap();
    let md = reader.metadata();
    let fm = md.file_metadata();
    let mut out = String::new();
    parquet::schema::printer::print_schema(&mut out_writer(&mut out), fm.schema());
    let mut kv: Vec<String> = fm
        .key_value_metadata()
        .into_iter()
        .flatten()
        .map(|e| format!("{}={:?}", e.key, e.value))
        .collect();
    kv.sort();
    out.push_str(&format!("rows {} kv {kv:?}\n", fm.num_rows()));
    for rg in md.row_groups() {
        out.push_str(&format!("row group {}\n", rg.num_rows()));
        for c in rg.columns() {
            let enc: Vec<_> = c.encodings().collect();
            out.push_str(&format!(
                "  {} {:?} {enc:?} values {} index {} {}\n",
                c.column_path(),
                c.compression(),
                c.num_values(),
                c.column_index_offset().is_some(),
                c.offset_index_offset().is_some()
            ));
        }
    }
    out
}

struct StrWriter<'a>(&'a mut String);

impl std::io::Write for StrWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.push_str(&String::from_utf8_lossy(buf));
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn out_writer(s: &mut String) -> StrWriter<'_> {
    StrWriter(s)
}

fn json_file(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join(name)).unwrap()).unwrap()
}

/// Files DuckDB 1.4 cannot read, as it cannot read Spark 4.2's with the same
/// settings: the Hadoop-framed LZ4 codec, and `DELTA_BYTE_ARRAY` (data page
/// v2 without dictionary) on fixed-length decimals.
const DUCKDB_UNREADABLE: &[&str] = &["lz4", "v2", "nested_v2"];

/// The Spark 4.2 types of the columns (what `reads.json` must report).
const NESTED_SPARK_TYPES: &str = "struct<id:int,li:array<int>,ls:array<struct<a:int,b:string>>,\
ll:array<array<bigint>>,sn:struct<s:struct<v:int>,l:array<int>,m:map<string,int>>,\
ms:map<string,struct<p:int,q:string>>,ma:map<string,array<int>>,mm:map<int,map<string,bigint>>,\
ts_l:array<timestamp>,ntz_l:array<timestamp_ntz>,dec_l:array<decimal(38,10)>,\
d_m:map<date,decimal(18,4)>>";

const SPARK_TYPES: &str = "struct<id:int,bool:boolean,byte:tinyint,short:smallint,long:bigint,\
float:float,double:double,dec9:decimal(9,2),dec18:decimal(18,4),dec38:decimal(38,10),s:string,\
bin:binary,d:date,ts:timestamp,ntz:timestamp_ntz,st:struct<x:int,y:string,t:timestamp>,arr:array<bigint>,\
m:map<string,int>>";

#[test]
fn other_engines_read_what_the_writer_was_given() {
    let expected_json = json_file("expected.json");
    let reads = json_file("reads.json");
    for (variant, _) in VARIANTS {
        let want = digest(&expected(variant));
        assert_eq!(
            expected_json[variant]["sha256"].as_str(),
            Some(want.as_str()),
            "{variant}: expected.json is stale"
        );
        for engine in ["pyarrow", "duckdb", "spark"] {
            let read = &reads[engine][variant];
            if engine == "duckdb" && DUCKDB_UNREADABLE.contains(variant) {
                assert!(read["unsupported"].is_string(), "{variant}: {read}");
                continue;
            }
            assert_eq!(
                read["sha256"].as_str(),
                Some(want.as_str()),
                "{variant}: {engine} reads other values"
            );
            assert_eq!(read["rows"].as_u64(), Some(ROWS as u64));
        }
        assert_eq!(
            reads["spark"][variant]["schema"].as_str(),
            Some(if variant.starts_with("nested") {
                NESTED_SPARK_TYPES
            } else {
                SPARK_TYPES
            }),
            "{variant}: Spark types"
        );
        assert_eq!(
            reads["pyarrow"][variant]["page_index"].as_bool(),
            Some(true),
            "{variant}: pyarrow sees no page index"
        );
        // Statistics a reader could skip rows with must bound the chunk's values.
        assert_eq!(
            reads["pyarrow"][variant]["bad_statistics"]
                .as_array()
                .map(Vec::len),
            Some(0),
            "{variant}: wrong statistics {}",
            reads["pyarrow"][variant]["bad_statistics"]
        );
        assert!(
            reads["pyarrow"][variant]["statistics_compared"].as_u64() > Some(0),
            "{variant}: no statistics were compared"
        );
    }
}

#[test]
fn todays_writer_still_writes_the_checked_files() {
    for (variant, kv) in VARIANTS {
        let committed =
            Bytes::from(std::fs::read(dir().join(format!("{variant}.parquet"))).unwrap());
        let today = write(variant, kv);
        assert_eq!(
            shape(today.clone()),
            shape(committed.clone()),
            "{variant}: the writer's output changed; regenerate the fixtures (module docs)"
        );
        let want = expected(variant);
        let schema = source_of(variant).schema();
        assert_eq!(
            digest(&krishiv_read(committed, &schema)),
            digest(&want),
            "{variant}: Krishiv reads the committed file differently"
        );
        assert_eq!(
            digest(&krishiv_read(today, &schema)),
            digest(&want),
            "{variant}: Krishiv reads today's file differently"
        );
    }
}

/// Regenerates the files and `expected.json` (see the module docs).
#[test]
#[ignore = "regenerates the committed fixtures; run with KRISHIV_WRITE_FIXTURES=1"]
fn regenerate_fixtures() {
    assert!(
        std::env::var_os("KRISHIV_WRITE_FIXTURES").is_some(),
        "set KRISHIV_WRITE_FIXTURES=1 to overwrite the fixtures"
    );
    std::fs::create_dir_all(dir()).unwrap();
    let mut expected_json = serde_json::Map::new();
    for (variant, kv) in VARIANTS {
        std::fs::write(dir().join(format!("{variant}.parquet")), write(variant, kv)).unwrap();
        expected_json.insert(
            (*variant).to_owned(),
            json!({"rows": ROWS, "sha256": digest(&expected(variant))}),
        );
    }
    std::fs::write(
        dir().join("expected.json"),
        serde_json::to_string_pretty(&Value::Object(expected_json)).unwrap() + "\n",
    )
    .unwrap();
}
