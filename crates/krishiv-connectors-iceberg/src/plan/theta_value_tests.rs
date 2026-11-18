// SPDX-License-Identifier: Apache-2.0
//! `theta_value`: the bytes a sketch is fed equal the manifest-bound serialisation of the same value
//! (`bounds::encode`), empty strings and binaries are skipped and `uuid` is not sketched.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    StringViewArray, Time64MicrosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray,
};
use krishiv_io::ScalarValue as S;

use super::theta_value::{for_each_value, new_sketch, sketched, update_column};
use super::{IcebergType as T, encode};

fn values(array: &dyn Array, ty: &T) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    assert!(for_each_value(array, ty, &mut |b| out.push(b.to_vec())));
    out
}

fn enc(v: &S, ty: &T) -> Vec<u8> {
    encode(v, ty).unwrap()
}

#[test]
fn every_type_feeds_the_bytes_of_the_manifest_encoding() {
    let cases: Vec<(ArrayRef, T, Vec<Vec<u8>>)> = vec![
        (
            Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
            T::Boolean,
            vec![
                enc(&S::Boolean(true), &T::Boolean),
                enc(&S::Boolean(false), &T::Boolean),
            ],
        ),
        (
            Arc::new(Int32Array::from(vec![Some(7), None, Some(-1)])),
            T::Int,
            vec![enc(&S::Int32(7), &T::Int), enc(&S::Int32(-1), &T::Int)],
        ),
        (
            Arc::new(Date32Array::from(vec![Some(19_000)])),
            T::Date,
            vec![enc(&S::Date32(19_000), &T::Date)],
        ),
        (
            Arc::new(Int64Array::from(vec![Some(i64::MIN), Some(3)])),
            T::Long,
            vec![
                enc(&S::Int64(i64::MIN), &T::Long),
                enc(&S::Int64(3), &T::Long),
            ],
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![Some(123_456)])),
            T::Timestamp,
            vec![enc(&S::TimestampMicros(123_456), &T::Timestamp)],
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![Some(-5)]).with_timezone("UTC")),
            T::Timestamptz,
            vec![enc(&S::TimestampMicros(-5), &T::Timestamptz)],
        ),
        (
            Arc::new(Float32Array::from(vec![
                Some(1.5_f32),
                Some(-0.0),
                Some(f32::NAN),
            ])),
            T::Float,
            vec![
                enc(&S::Float32(1.5), &T::Float),
                enc(&S::Float32(-0.0), &T::Float),
                enc(&S::Float32(f32::NAN), &T::Float),
            ],
        ),
        (
            Arc::new(Float64Array::from(vec![
                Some(-0.0_f64),
                Some(0.0),
                Some(f64::NAN),
            ])),
            T::Double,
            vec![
                enc(&S::Float64(-0.0), &T::Double),
                enc(&S::Float64(0.0), &T::Double),
                enc(&S::Float64(f64::NAN), &T::Double),
            ],
        ),
        (
            Arc::new(
                Decimal128Array::from(vec![Some(1234), Some(-1), Some(128), Some(0)])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            T::Decimal {
                precision: 10,
                scale: 2,
            },
            [1234, -1, 128, 0]
                .map(|u| {
                    enc(
                        &S::Decimal128 {
                            unscaled: u,
                            precision: 10,
                            scale: 2,
                        },
                        &T::Decimal {
                            precision: 10,
                            scale: 2,
                        },
                    )
                })
                .to_vec(),
        ),
        (
            Arc::new(StringArray::from(vec![Some("ünï"), None, Some("a")])),
            T::String,
            vec![
                enc(&S::Utf8("ünï".into()), &T::String),
                enc(&S::Utf8("a".into()), &T::String),
            ],
        ),
        (
            Arc::new(LargeStringArray::from(vec!["x"])),
            T::String,
            vec![enc(&S::Utf8("x".into()), &T::String)],
        ),
        (
            Arc::new(StringViewArray::from(vec![
                "a long string over twelve bytes",
            ])),
            T::String,
            vec![enc(
                &S::Utf8("a long string over twelve bytes".into()),
                &T::String,
            )],
        ),
        (
            Arc::new(BinaryArray::from(vec![Some(&b"\x00\xff"[..]), None])),
            T::Binary,
            vec![enc(&S::Binary(vec![0, 255]), &T::Binary)],
        ),
        (
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(vec![vec![1_u8, 2], vec![3, 4]].into_iter())
                    .unwrap(),
            ),
            T::Fixed(2),
            vec![
                enc(&S::Binary(vec![1, 2]), &T::Fixed(2)),
                enc(&S::Binary(vec![3, 4]), &T::Fixed(2)),
            ],
        ),
    ];
    for (array, ty, want) in cases {
        assert_eq!(values(array.as_ref(), &ty), want, "{ty:?}");
    }
}

/// `encode` has no `time` or nanosecond timestamp: the spec says 8 bytes little-endian.
#[test]
fn time_and_nanosecond_timestamps_are_eight_little_endian_bytes() {
    let time = Time64MicrosecondArray::from(vec![Some(86_399_000_000_i64)]);
    assert_eq!(
        values(&time, &T::Time),
        vec![86_399_000_000_i64.to_le_bytes().to_vec()]
    );
    let ts = TimestampNanosecondArray::from(vec![Some(-9_i64)]);
    assert_eq!(
        values(&ts, &T::TimestampNs),
        vec![(-9_i64).to_le_bytes().to_vec()]
    );
}

/// Minimal big-endian two's complement: 128 needs two bytes, -128 one.
#[test]
fn decimals_are_minimal_big_endian() {
    let a = Decimal128Array::from(vec![128, -128, 0])
        .with_precision_and_scale(10, 0)
        .unwrap();
    let ty = T::Decimal {
        precision: 10,
        scale: 0,
    };
    assert_eq!(values(&a, &ty), vec![vec![0, 0x80], vec![0x80], vec![0]]);
}

/// Review B2: Java's `update(ByteBuffer)` rejects an empty buffer.
#[test]
fn empty_strings_and_binaries_are_skipped() {
    let a = StringArray::from(vec![Some(""), Some("a"), None, Some("")]);
    assert_eq!(values(&a, &T::String), vec![b"a".to_vec()]);
    let b = BinaryArray::from(vec![Some(&b""[..]), Some(&b"b"[..])]);
    assert_eq!(values(&b, &T::Binary), vec![b"b".to_vec()]);

    let mut only_empty = new_sketch().unwrap();
    assert!(update_column(
        &mut only_empty,
        &StringArray::from(vec!["", ""]),
        &T::String
    ));
    assert!(only_empty.is_empty());
    assert_eq!(only_empty.estimate().to_bits(), 0.0_f64.to_bits());
    let mut with_a = new_sketch().unwrap();
    update_column(
        &mut with_a,
        &StringArray::from(vec!["", "a", ""]),
        &T::String,
    );
    let mut just_a = new_sketch().unwrap();
    update_column(&mut just_a, &StringArray::from(vec!["a"]), &T::String);
    assert_eq!(
        with_a.compact(true).serialize(),
        just_a.compact(true).serialize()
    );
}

/// Review S1: a `uuid` is not sketched; nor is a mismatched Arrow type.
#[test]
fn uuid_and_mismatched_arrays_are_refused() {
    assert!(!sketched(&T::Uuid));
    assert!(sketched(&T::Long));
    let mut s = new_sketch().unwrap();
    let uuids = FixedSizeBinaryArray::try_from_iter(vec![vec![0_u8; 16]].into_iter()).unwrap();
    assert!(!update_column(&mut s, &uuids, &T::Uuid));
    assert!(!update_column(&mut s, &Int32Array::from(vec![1]), &T::Long));
    assert!(s.is_empty());
}

/// The hash input is the raw bytes, not Rust's length-prefixed slice: 1,000 distinct longs estimate 1,000
/// (exact below k), and the same value in two chunks counts once.
#[test]
fn values_are_counted_across_batches() {
    let mut s = new_sketch().unwrap();
    let ty = T::Long;
    update_column(&mut s, &Int64Array::from_iter_values(0..1000), &ty);
    update_column(&mut s, &Int64Array::from_iter_values(500..1500), &ty);
    assert!((s.estimate() - 1500.0).abs() < 0.5, "{}", s.estimate());
    assert!(!s.is_estimation_mode());
}
