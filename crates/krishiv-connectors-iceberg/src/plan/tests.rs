// SPDX-License-Identifier: Apache-2.0
//! Tests of scan planning over Krishiv-owned values: bounds by width, the
//! evaluator (promoted bounds, NaN, truncated strings, partitions, summaries),
//! inheritance and time-travel resolution. The property test is `prop_tests.rs`;
//! the tests that need iceberg-rust's writers are in `iceberg_v0`.

use std::collections::BTreeMap;

use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};

use super::entry::{
    ContentFile, ContentKind, DeleteKind, EntryStatus, FileFormatKind, LiveEntry, ManifestContent,
    ManifestFileEntry, ParsedEntry, PartitionFieldSummary, PartitionValue, inherit,
};
use super::prune::{PruneOptions, Pruner};
use super::snapshot::{
    MetadataView, RefInfo, RefKind, SnapshotInfo, VersionSpec, format_timestamp_millis, resolve,
};
use super::types::{
    IcebergType, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};
use super::{bounds, transform};

fn col(name: &str) -> ColumnPath {
    ColumnPath::new(name)
}

fn field(id: i32, name: &str, ty: IcebergType) -> SchemaField {
    SchemaField {
        id,
        path: col(name),
        ty,
    }
}

/// A data file with `rows` rows and nothing else known.
pub(super) fn file(rows: u64) -> ContentFile {
    ContentFile {
        content: ContentKind::Data,
        path: "f.parquet".to_owned(),
        format: FileFormatKind::Parquet,
        spec_id: 0,
        partition: Vec::new(),
        record_count: rows,
        size: 1,
        column_sizes: BTreeMap::new(),
        value_counts: BTreeMap::new(),
        null_counts: BTreeMap::new(),
        nan_counts: BTreeMap::new(),
        lower_bounds: BTreeMap::new(),
        upper_bounds: BTreeMap::new(),
        split_offsets: None,
        sort_order_id: None,
        first_row_id: None,
        equality_ids: None,
        referenced_data_file: None,
        content_offset: None,
        content_size_in_bytes: None,
    }
}

fn unpartitioned() -> PartitionSpecs {
    [(0, PartitionSpecInfo::default())].into()
}

fn with_metrics(id: i32, lower: Vec<u8>, upper: Vec<u8>, values: u64, nulls: u64) -> ContentFile {
    let mut f = file(values);
    f.lower_bounds.insert(id, lower);
    f.upper_bounds.insert(id, upper);
    f.value_counts.insert(id, values);
    f.null_counts.insert(id, nulls);
    f
}

fn keeps(
    fields: &[SchemaField],
    specs: &PartitionSpecs,
    f: &ContentFile,
    e: &PushdownExpr,
) -> bool {
    Pruner::new(e, fields, specs, PruneOptions::default()).file_may_match(f)
}

fn cmp(c: &str, op: CmpOp, v: ScalarValue) -> PushdownExpr {
    PushdownExpr::cmp(col(c), op, v)
}

fn in_list(c: &str, vs: Vec<ScalarValue>) -> PushdownExpr {
    PushdownExpr::In {
        column: col(c),
        values: vs,
        collation: Collation::Utf8Binary,
    }
}

#[test]
fn bounds_decode_by_stored_width() {
    use IcebergType as T;
    use ScalarValue as S;
    let int4 = 7i32.to_le_bytes();
    assert_eq!(bounds::decode(&int4, &T::Long), Some(S::Int64(7)));
    assert_eq!(
        bounds::decode(&(-7i32).to_le_bytes(), &T::Long),
        Some(S::Int64(-7))
    );
    assert_eq!(
        bounds::decode(&7i64.to_le_bytes(), &T::Long),
        Some(S::Int64(7))
    );
    assert_eq!(bounds::decode(&int4, &T::Int), Some(S::Int32(7)));
    // An impossible width drops the bound.
    assert_eq!(bounds::decode(&7i64.to_le_bytes(), &T::Int), None);
    assert_eq!(bounds::decode(&[1, 2, 3], &T::Long), None);
    assert_eq!(
        bounds::decode(&1.5f32.to_le_bytes(), &T::Double),
        Some(S::Float64(1.5))
    );
    assert_eq!(bounds::decode(&1.5f64.to_le_bytes(), &T::Float), None);
    // Decimal(9,2) promoted to decimal(18,2): the unscaled value is width-free.
    let dec = T::Decimal {
        precision: 18,
        scale: 2,
    };
    assert_eq!(
        bounds::decode(&[0x30, 0x39], &dec),
        Some(S::Decimal128 {
            unscaled: 12345,
            precision: 18,
            scale: 2
        })
    );
    assert_eq!(
        bounds::decode(&[0xff, 0x85], &dec),
        Some(S::Decimal128 {
            unscaled: -123,
            precision: 18,
            scale: 2
        })
    );
    assert_eq!(bounds::decode(&[0; 17], &dec), None);
    assert_eq!(bounds::decode(&[], &dec), None);
    // A date promoted to a timestamp (v3) is midnight of the day.
    assert_eq!(
        bounds::decode(&2i32.to_le_bytes(), &T::Timestamp),
        Some(S::TimestampMicros(2 * 86_400_000_000))
    );
    assert_eq!(bounds::decode(&[0xff, 0xfe], &T::String), None);
    assert_eq!(
        bounds::decode(&[0x12; 16], &T::Uuid),
        Some(S::Utf8("12121212-1212-1212-1212-121212121212".to_owned()))
    );
    assert_eq!(bounds::decode(&[1], &T::Boolean), Some(S::Boolean(true)));
    for v in [0i128, 1, -1, 127, 128, -128, -129, 1 << 70, -(1 << 70)] {
        let b = bounds::min_be_bytes(v);
        assert_eq!(
            bounds::decode(&b, &dec),
            Some(S::Decimal128 {
                unscaled: v,
                precision: 18,
                scale: 2
            }),
            "{v}"
        );
    }
    assert_eq!(bounds::min_be_bytes(127), vec![0x7f]);
    assert_eq!(bounds::min_be_bytes(128), vec![0x00, 0x80]);
    assert_eq!(bounds::min_be_bytes(-128), vec![0x80]);
}

/// The #3482 shape: a file written while the column was `int` (4-byte bounds
/// 10..=20), read after promotion to `long`; every comparison that a row of the
/// file satisfies keeps it, and only the ones no row can satisfy prune it.
#[test]
fn promoted_int_bounds_read_as_long() {
    let fields = [field(1, "x", IcebergType::Long)];
    let specs = unpartitioned();
    let f = with_metrics(
        1,
        10i32.to_le_bytes().to_vec(),
        20i32.to_le_bytes().to_vec(),
        5,
        0,
    );
    let l = ScalarValue::Int64;
    let cases = [
        (cmp("x", CmpOp::Lt, l(11)), true),
        (cmp("x", CmpOp::Lt, l(10)), false),
        (cmp("x", CmpOp::LtEq, l(10)), true),
        (cmp("x", CmpOp::LtEq, l(9)), false),
        (cmp("x", CmpOp::Gt, l(19)), true),
        (cmp("x", CmpOp::Gt, l(20)), false),
        (cmp("x", CmpOp::GtEq, l(20)), true),
        (cmp("x", CmpOp::GtEq, l(21)), false),
        (cmp("x", CmpOp::Eq, l(15)), true),
        (cmp("x", CmpOp::Eq, l(21)), false),
        (cmp("x", CmpOp::Eq, l(9)), false),
        (cmp("x", CmpOp::NotEq, l(15)), true),
        (in_list("x", vec![l(1), l(20)]), true),
        (in_list("x", vec![l(1), l(21)]), false),
        (in_list("x", vec![]), false),
    ];
    for (e, want) in cases {
        assert_eq!(keeps(&fields, &specs, &f, &e), want, "{e:?}");
    }
    // A literal of another type never prunes.
    assert!(keeps(
        &fields,
        &specs,
        &f,
        &cmp("x", CmpOp::Eq, ScalarValue::Int32(99))
    ));
}

#[test]
fn promoted_float_bounds_read_as_double() {
    let fields = [field(1, "x", IcebergType::Double)];
    let specs = unpartitioned();
    let mut f = with_metrics(
        1,
        1.5f32.to_le_bytes().to_vec(),
        2.5f32.to_le_bytes().to_vec(),
        5,
        0,
    );
    f.nan_counts.insert(1, 0);
    let d = ScalarValue::Float64;
    let cases = [
        (cmp("x", CmpOp::Lt, d(1.75)), true),
        (cmp("x", CmpOp::Lt, d(1.5)), false),
        (cmp("x", CmpOp::LtEq, d(1.5)), true),
        (cmp("x", CmpOp::Gt, d(2.25)), true),
        (cmp("x", CmpOp::Gt, d(2.5)), false),
        (cmp("x", CmpOp::GtEq, d(2.5)), true),
        (cmp("x", CmpOp::Eq, d(2.0)), true),
        (cmp("x", CmpOp::Eq, d(3.0)), false),
        (in_list("x", vec![d(9.0), d(2.5)]), true),
        (in_list("x", vec![d(9.0), d(-1.0)]), false),
    ];
    for (e, want) in cases {
        assert_eq!(keeps(&fields, &specs, &f, &e), want, "{e:?}");
    }
}

#[test]
fn nan_columns_are_never_pruned_by_bounds() {
    let fields = [field(1, "x", IcebergType::Double)];
    let specs = unpartitioned();
    let base = with_metrics(
        1,
        1.0f64.to_le_bytes().to_vec(),
        2.0f64.to_le_bytes().to_vec(),
        5,
        0,
    );
    let above = cmp("x", CmpOp::Gt, ScalarValue::Float64(10.0));
    let below = cmp("x", CmpOp::Lt, ScalarValue::Float64(0.0 - 10.0));
    // No NaN count, or some NaNs: Spark's NaN is above every value, so `>` may
    // match, and the bounds are not used at all.
    let mut nan = base.clone();
    nan.nan_counts.insert(1, 1);
    for f in [&base, &nan] {
        assert!(keeps(&fields, &specs, f, &above));
        assert!(keeps(&fields, &specs, f, &below));
    }
    let mut clean = base;
    clean.nan_counts.insert(1, 0);
    assert!(!keeps(&fields, &specs, &clean, &above));
    assert!(!keeps(&fields, &specs, &clean, &below));
    // NaN and zero literals never prune.
    for v in [f64::NAN, 0.0, -0.0] {
        assert!(keeps(
            &fields,
            &specs,
            &clean,
            &cmp("x", CmpOp::Eq, ScalarValue::Float64(v))
        ));
    }
    // A column that is all null holds no NaN either.
    let mut nulls = file(5);
    nulls.value_counts.insert(1, 5);
    nulls.null_counts.insert(1, 5);
    assert!(!keeps(&fields, &specs, &nulls, &above));
    assert!(keeps(
        &fields,
        &specs,
        &nulls,
        &PushdownExpr::IsNull(col("x"))
    ));
    assert!(!keeps(
        &fields,
        &specs,
        &nulls,
        &PushdownExpr::IsNotNull(col("x"))
    ));
}

#[test]
fn truncated_string_bounds() {
    let fields = [field(1, "s", IcebergType::String)];
    let specs = unpartitioned();
    let s = |v: &str| ScalarValue::Utf8(v.to_owned());
    // A 16-character upper bound may be a truncation without rounding up: the
    // file may hold "aaaaaaaaaaaaaaaab" + anything.
    let upper16 = "aaaaaaaaaaaaaaab";
    let f = with_metrics(1, b"a".to_vec(), upper16.as_bytes().to_vec(), 3, 0);
    let longer = format!("{upper16}zz");
    for e in [
        cmp("s", CmpOp::Eq, s(&longer)),
        cmp("s", CmpOp::GtEq, s(&longer)),
        cmp("s", CmpOp::Gt, s(upper16)),
        cmp("s", CmpOp::Eq, s(upper16)),
    ] {
        assert!(keeps(&fields, &specs, &f, &e), "{e:?}");
    }
    // Above every string with that prefix: pruned.
    assert!(!keeps(
        &fields,
        &specs,
        &f,
        &cmp("s", CmpOp::Eq, s("aaaaaaaaaaaaaaac"))
    ));
    assert!(!keeps(&fields, &specs, &f, &cmp("s", CmpOp::Gt, s("b"))));
    // A short upper bound is exact in scan mode...
    let short = with_metrics(1, b"a".to_vec(), b"abc".to_vec(), 3, 0);
    let e = cmp("s", CmpOp::Eq, s("abcd"));
    assert!(!keeps(&fields, &specs, &short, &e));
    assert!(!keeps(
        &fields,
        &specs,
        &short,
        &cmp("s", CmpOp::Gt, s("abc"))
    ));
    // ...and never in conservative mode.
    let conservative = Pruner::new(&e, &fields, &specs, PruneOptions::conservative());
    assert!(conservative.file_may_match(&short));
    // A lower bound is a prefix of the true minimum: always valid.
    assert!(!keeps(
        &fields,
        &specs,
        &short,
        &cmp("s", CmpOp::Lt, s("a"))
    ));
    assert!(keeps(
        &fields,
        &specs,
        &short,
        &cmp("s", CmpOp::LtEq, s("a"))
    ));
    // startswith.
    let sw = |p: &str| PushdownExpr::StartsWith {
        column: col("s"),
        prefix: p.to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert!(keeps(&fields, &specs, &short, &sw("ab")));
    assert!(keeps(&fields, &specs, &short, &sw("abc")));
    assert!(!keeps(&fields, &specs, &short, &sw("abd")));
    assert!(!keeps(&fields, &specs, &short, &sw("0")));
    assert!(keeps(&fields, &specs, &f, &sw(&longer)));
    // Another collation never prunes.
    let lcase = PushdownExpr::Cmp {
        column: col("s"),
        op: CmpOp::Eq,
        value: s("zzz"),
        collation: Collation::Other("UTF8_LCASE".to_owned()),
    };
    assert!(keeps(&fields, &specs, &short, &lcase));
    // Invalid UTF-8 in a bound drops that bound only.
    let mut bad = short;
    bad.lower_bounds.insert(1, vec![0xff]);
    assert!(!keeps(&fields, &specs, &bad, &e));
    assert!(keeps(&fields, &specs, &bad, &cmp("s", CmpOp::Lt, s("a"))));
}

#[test]
fn counts_and_empty_files() {
    let fields = [field(1, "x", IcebergType::Long)];
    let specs = unpartitioned();
    let t = PushdownExpr::Constant(true);
    assert!(!keeps(&fields, &specs, &file(0), &t));
    assert!(keeps(&fields, &specs, &file(1), &t));
    assert!(!keeps(
        &fields,
        &specs,
        &file(1),
        &PushdownExpr::Constant(false)
    ));
    let mut f = file(4);
    f.null_counts.insert(1, 0);
    assert!(!keeps(&fields, &specs, &f, &PushdownExpr::IsNull(col("x"))));
    assert!(keeps(
        &fields,
        &specs,
        &f,
        &PushdownExpr::IsNotNull(col("x"))
    ));
    // Unknown columns, nested negations and runtime-only leaves keep.
    assert!(keeps(
        &fields,
        &specs,
        &f,
        &PushdownExpr::IsNull(col("nope"))
    ));
    let not_sw = PushdownExpr::Not(Box::new(PushdownExpr::StartsWith {
        column: col("x"),
        prefix: "a".to_owned(),
        collation: Collation::Utf8Binary,
    }));
    assert!(keeps(&fields, &specs, &f, &not_sw));
    // NOT (x IS NOT NULL) is x IS NULL.
    let not = PushdownExpr::Not(Box::new(PushdownExpr::IsNotNull(col("x"))));
    assert!(!keeps(&fields, &specs, &f, &not));
}

fn spec(fields: Vec<PartitionFieldInfo>) -> PartitionSpecs {
    [(1, PartitionSpecInfo { spec_id: 1, fields })].into()
}

fn pfield(id: i32, source: i32, t: Transform, ty: IcebergType) -> PartitionFieldInfo {
    PartitionFieldInfo::new(id, source, format!("p{id}"), t, Some(ty))
}

fn partitioned(values: Vec<PartitionValue>) -> ContentFile {
    let mut f = file(3);
    f.spec_id = 1;
    f.partition = values;
    f
}

fn value(v: &ScalarValue, ty: &IcebergType) -> PartitionValue {
    PartitionValue::Value(bounds::encode(v, ty).unwrap())
}

#[test]
fn partition_projection_per_transform() {
    use ScalarValue as S;
    const DAY: i64 = 86_400_000_000;
    let fields = [
        field(1, "a", IcebergType::Long),
        field(2, "s", IcebergType::String),
        field(3, "t", IcebergType::Timestamptz),
    ];
    let specs = spec(vec![
        pfield(1000, 1, Transform::Bucket(4), IcebergType::Long),
        pfield(1001, 2, Transform::Truncate(2), IcebergType::String),
        pfield(1002, 3, Transform::Day, IcebergType::Timestamptz),
        pfield(1003, 1, Transform::Void, IcebergType::Long),
        pfield(1004, 1, Transform::Unknown, IcebergType::Long),
    ]);
    let a = S::Int64(34);
    let bucket = transform::apply(Transform::Bucket(4), &IcebergType::Long, &a).unwrap();
    let f = partitioned(vec![
        value(&bucket, &IcebergType::Int),
        value(&S::Utf8("ab".to_owned()), &IcebergType::String),
        value(&S::Date32(3), &IcebergType::Date),
        PartitionValue::Null,
        PartitionValue::Unreadable,
    ]);
    let k = |e: &PushdownExpr| keeps(&fields, &specs, &f, e);
    assert!(k(&cmp("a", CmpOp::Eq, a.clone())));
    let other = (0..100)
        .map(S::Int64)
        .find(|v| {
            transform::apply(Transform::Bucket(4), &IcebergType::Long, v) != Some(bucket.clone())
        })
        .unwrap();
    assert!(!k(&cmp("a", CmpOp::Eq, other.clone())));
    assert!(k(&in_list("a", vec![other.clone(), a])));
    assert!(!k(&in_list("a", vec![other])));
    // A bucket answers only equality; void and unknown never prune.
    assert!(k(&cmp("a", CmpOp::Lt, S::Int64(-1000))));
    // truncate[2] on strings.
    let s = |v: &str| S::Utf8(v.to_owned());
    assert!(k(&cmp("s", CmpOp::Eq, s("abzz"))));
    assert!(!k(&cmp("s", CmpOp::Eq, s("ba"))));
    assert!(!k(&cmp("s", CmpOp::Lt, s("aa"))));
    assert!(k(&cmp("s", CmpOp::Lt, s("aba"))));
    assert!(!k(&cmp("s", CmpOp::Gt, s("ac"))));
    assert!(k(&cmp("s", CmpOp::Gt, s("abzzz"))));
    let sw = |p: &str| PushdownExpr::StartsWith {
        column: col("s"),
        prefix: p.to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert!(k(&sw("a")));
    assert!(k(&sw("abq")));
    assert!(!k(&sw("b")));
    assert!(!k(&sw("acq")));
    // day(t) = 3.
    let ts = S::TimestampMicros;
    assert!(k(&cmp("t", CmpOp::Eq, ts(3 * DAY + 5))));
    assert!(!k(&cmp("t", CmpOp::Eq, ts(4 * DAY))));
    assert!(!k(&cmp("t", CmpOp::Lt, ts(3 * DAY - 1))));
    assert!(k(&cmp("t", CmpOp::Lt, ts(3 * DAY + 1))));
    assert!(!k(&cmp("t", CmpOp::GtEq, ts(4 * DAY))));
    assert!(k(&cmp("t", CmpOp::GtEq, ts(4 * DAY - 1))));
    // A null partition value: the source is null.
    let nulls = partitioned(vec![
        PartitionValue::Null,
        PartitionValue::Null,
        PartitionValue::Null,
        PartitionValue::Null,
        PartitionValue::Null,
    ]);
    assert!(!keeps(&fields, &specs, &nulls, &cmp("t", CmpOp::Eq, ts(0))));
    assert!(!keeps(
        &fields,
        &specs,
        &nulls,
        &PushdownExpr::IsNotNull(col("s"))
    ));
    assert!(keeps(
        &fields,
        &specs,
        &nulls,
        &PushdownExpr::IsNull(col("s"))
    ));
    assert!(!k(&PushdownExpr::IsNull(col("s"))));
}

/// Old Java writers computed pre-epoch time partitions toward zero: a value
/// written as day 0 for a timestamp of -1 µs must still be found.
#[test]
fn pre_epoch_time_projection_is_widened() {
    use ScalarValue as S;
    let fields = [field(3, "t", IcebergType::Timestamptz)];
    for (t, buggy) in [
        (Transform::Day, S::Date32(0)),
        (Transform::Hour, S::Int32(0)),
    ] {
        let specs = spec(vec![pfield(1000, 3, t, IcebergType::Timestamptz)]);
        let ty = PartitionFieldInfo::new(0, 0, String::new(), t, Some(IcebergType::Timestamptz))
            .result_type
            .unwrap();
        let f = partitioned(vec![value(&buggy, &ty)]);
        for e in [
            cmp("t", CmpOp::Eq, S::TimestampMicros(-1)),
            cmp("t", CmpOp::LtEq, S::TimestampMicros(-1)),
            cmp("t", CmpOp::Lt, S::TimestampMicros(-1)),
            in_list("t", vec![S::TimestampMicros(-1)]),
        ] {
            assert!(keeps(&fields, &specs, &f, &e), "{t:?} {e:?}");
        }
        // Two units below is still pruned.
        let far = cmp("t", CmpOp::Eq, S::TimestampMicros(-3 * 86_400_000_000));
        assert!(!keeps(&fields, &specs, &f, &far), "{t:?}");
    }
}

#[test]
fn temporal_transforms_match_the_spec() {
    use ScalarValue as S;
    const DAY: i64 = 86_400_000_000;
    let ts = IcebergType::Timestamp;
    let ap = |t, v| transform::apply(t, &ts, &S::TimestampMicros(v));
    assert_eq!(ap(Transform::Day, -1), Some(S::Date32(-1)));
    assert_eq!(ap(Transform::Hour, -1), Some(S::Int32(-1)));
    assert_eq!(ap(Transform::Year, -1), Some(S::Int32(-1)));
    assert_eq!(ap(Transform::Month, -1), Some(S::Int32(-1)));
    // 2000-03-01 is day 11017.
    assert_eq!(ap(Transform::Year, 11_017 * DAY), Some(S::Int32(30)));
    assert_eq!(
        ap(Transform::Month, 11_017 * DAY),
        Some(S::Int32(30 * 12 + 2))
    );
    let date = |t, d| transform::apply(t, &IcebergType::Date, &S::Date32(d));
    assert_eq!(date(Transform::Month, 11_016), Some(S::Int32(30 * 12 + 1)));
    assert_eq!(date(Transform::Hour, 1), None);
    assert_eq!(
        transform::apply(Transform::Truncate(10), &IcebergType::Int, &S::Int32(-1)),
        Some(S::Int32(-10))
    );
    assert_eq!(
        transform::apply(
            Transform::Truncate(3),
            &IcebergType::String,
            &S::Utf8("añbc".to_owned())
        ),
        Some(S::Utf8("añb".to_owned()))
    );
    // Murmur3 test vectors of the Iceberg spec (appendix B).
    assert_eq!(transform::murmur3_32(&34i64.to_le_bytes()), 2_017_239_379);
    assert_eq!(transform::murmur3_32("iceberg".as_bytes()), 1_210_000_089);
    assert_eq!(
        transform::murmur3_32(&[0, 1, 2, 3]).cast_signed(),
        -188_683_207
    );
    // decimal 14.20, date 2017-11-16, uuid f79c3e09-677c-4bbd-a479-3f349cb785e7.
    assert_eq!(
        transform::murmur3_32(&bounds::min_be_bytes(1420)).cast_signed(),
        -500_754_589
    );
    assert_eq!(
        transform::murmur3_32(&17_486i64.to_le_bytes()).cast_signed(),
        -653_330_422
    );
    let uuid = [
        0xf7, 0x9c, 0x3e, 0x09, 0x67, 0x7c, 0x4b, 0xbd, 0xa4, 0x79, 0x3f, 0x34, 0x9c, 0xb7, 0x85,
        0xe7,
    ];
    assert_eq!(transform::murmur3_32(&uuid).cast_signed(), 1_488_055_340);
}

#[test]
fn manifest_summaries() {
    use ScalarValue as S;
    let fields = [field(6, "k", IcebergType::Int)];
    let specs = spec(vec![pfield(1000, 6, Transform::Identity, IcebergType::Int)]);
    let manifest = |summary: PartitionFieldSummary| ManifestFileEntry {
        path: "m.avro".to_owned(),
        length: 1,
        spec_id: 1,
        content: ManifestContent::Data,
        sequence_number: 1,
        min_sequence_number: 1,
        added_snapshot_id: 1,
        added_files: Some(1),
        existing_files: Some(0),
        deleted_files: Some(0),
        first_row_id: None,
        partitions: Some(vec![summary]),
        encrypted: false,
    };
    let ranged = manifest(PartitionFieldSummary {
        contains_null: false,
        contains_nan: None,
        lower: Some(3i32.to_le_bytes().to_vec()),
        upper: Some(5i32.to_le_bytes().to_vec()),
    });
    let m = |m: &ManifestFileEntry, e: &PushdownExpr, o: PruneOptions| {
        Pruner::new(e, &fields, &specs, o).manifest_may_match(m)
    };
    let scan = PruneOptions::default();
    assert!(m(&ranged, &cmp("k", CmpOp::Eq, S::Int32(4)), scan));
    assert!(!m(&ranged, &cmp("k", CmpOp::Eq, S::Int32(6)), scan));
    assert!(!m(&ranged, &cmp("k", CmpOp::Lt, S::Int32(3)), scan));
    assert!(!m(&ranged, &PushdownExpr::IsNull(col("k")), scan));
    // No bounds, nulls only: scan mode infers all-null (Java's rule); the
    // conservative mode does not.
    let nulls = manifest(PartitionFieldSummary {
        contains_null: true,
        contains_nan: None,
        lower: None,
        upper: None,
    });
    let e = cmp("k", CmpOp::Eq, S::Int32(4));
    assert!(!m(&nulls, &e, scan));
    assert!(m(&nulls, &e, PruneOptions::conservative()));
    assert!(m(&nulls, &PushdownExpr::IsNull(col("k")), scan));
    // A bound that does not decode is dropped, never read as "no values".
    let bad = manifest(PartitionFieldSummary {
        contains_null: false,
        contains_nan: None,
        lower: Some(vec![1, 2]),
        upper: Some(5i32.to_le_bytes().to_vec()),
    });
    assert!(m(&bad, &cmp("k", CmpOp::Eq, S::Int32(-50)), scan));
    assert!(!m(&bad, &cmp("k", CmpOp::Eq, S::Int32(50)), scan));
    // Summaries that do not match the spec's arity are ignored.
    let mut wrong = ranged;
    wrong.partitions = Some(Vec::new());
    assert!(m(&wrong, &cmp("k", CmpOp::Eq, S::Int32(6)), scan));
}

fn parsed(status: EntryStatus, seq: Option<i64>, snapshot: Option<i64>, rows: u64) -> ParsedEntry {
    ParsedEntry {
        status,
        snapshot_id: snapshot,
        sequence_number: seq,
        file_sequence_number: seq,
        file: file(rows),
    }
}

pub(super) fn manifest_entry(seq: i64, first_row_id: Option<u64>) -> ManifestFileEntry {
    ManifestFileEntry {
        path: "m.avro".to_owned(),
        length: 1,
        spec_id: 0,
        content: ManifestContent::Data,
        sequence_number: seq,
        min_sequence_number: seq,
        added_snapshot_id: 42,
        added_files: None,
        existing_files: None,
        deleted_files: None,
        first_row_id,
        partitions: None,
        encrypted: false,
    }
}

fn data(e: &LiveEntry) -> &super::entry::DataFileEntry {
    match e {
        LiveEntry::Data(d) => d,
        LiveEntry::Delete(d) => &d.entry,
    }
}

#[test]
fn inheritance_rules() {
    let m = manifest_entry(7, Some(1000));
    let out = inherit(
        &m,
        vec![
            parsed(EntryStatus::Added, None, None, 10),
            parsed(EntryStatus::Existing, Some(3), Some(5), 20),
            parsed(EntryStatus::Deleted, Some(2), Some(5), 30),
            parsed(EntryStatus::Added, Some(6), Some(9), 40),
        ],
    )
    .unwrap();
    assert_eq!(out.len(), 3, "DELETED entries are skipped");
    let got: Vec<_> = out
        .iter()
        .map(|e| {
            let d = data(e);
            (
                d.data_sequence_number,
                d.file_sequence_number,
                d.snapshot_id,
                d.file.first_row_id,
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (Some(7), Some(7), 42, Some(1000)),
            (Some(3), Some(3), 5, Some(1010)),
            (Some(6), Some(6), 9, Some(1030)),
        ]
    );
    // An existing entry with no sequence numbers outside v1 (a writer that
    // omitted them): read, left null, as Java and iceberg-rust do (review F1).
    let out = inherit(&m, vec![parsed(EntryStatus::Existing, None, None, 1)]).unwrap();
    assert_eq!(
        (
            data(&out[0]).data_sequence_number,
            data(&out[0]).file_sequence_number
        ),
        (None, None)
    );
    // Only the file sequence number missing: the data one is kept.
    let mut only_data = parsed(EntryStatus::Existing, Some(4), None, 1);
    only_data.file_sequence_number = None;
    let out = inherit(&m, vec![only_data]).unwrap();
    assert_eq!(
        (
            data(&out[0]).data_sequence_number,
            data(&out[0]).file_sequence_number
        ),
        (Some(4), None)
    );
    // v1: sequence number 0 for everything.
    let v1 = inherit(
        &manifest_entry(0, None),
        vec![parsed(EntryStatus::Existing, None, None, 1)],
    )
    .unwrap();
    assert_eq!(data(&v1[0]).data_sequence_number, Some(0));
    assert_eq!(data(&v1[0]).file.first_row_id, None);
    // A file that has a first row id keeps it and does not advance the counter.
    let mut keep = parsed(EntryStatus::Existing, Some(1), Some(1), 10);
    keep.file.first_row_id = Some(5);
    let out = inherit(&m, vec![keep, parsed(EntryStatus::Added, None, None, 10)]).unwrap();
    assert_eq!(data(&out[0]).file.first_row_id, Some(5));
    assert_eq!(data(&out[1]).file.first_row_id, Some(1000));
    // Delete kinds.
    let mut dv = parsed(EntryStatus::Added, None, None, 1);
    dv.file.content = ContentKind::PositionDeletes;
    dv.file.format = FileFormatKind::Puffin;
    let mut pos = parsed(EntryStatus::Added, None, None, 1);
    pos.file.content = ContentKind::PositionDeletes;
    let mut eq = parsed(EntryStatus::Added, None, None, 1);
    eq.file.content = ContentKind::EqualityDeletes;
    let mut deletes = manifest_entry(7, Some(1));
    deletes.content = ManifestContent::Deletes;
    let kinds: Vec<_> = inherit(&deletes, vec![dv, pos, eq])
        .unwrap()
        .into_iter()
        .map(|e| match e {
            LiveEntry::Delete(d) => {
                assert_eq!(
                    d.entry.file.first_row_id, None,
                    "delete files have no row ids"
                );
                d.kind
            }
            LiveEntry::Data(_) => unreachable!(),
        })
        .collect();
    assert_eq!(
        kinds,
        vec![
            DeleteKind::DeletionVector,
            DeleteKind::PositionDelete,
            DeleteKind::EqualityDelete
        ]
    );
}

fn view() -> MetadataView {
    let snap = |id: i64, ts: i64, schema: Option<i32>| {
        (
            id,
            SnapshotInfo {
                snapshot_id: id,
                sequence_number: id,
                timestamp_ms: ts,
                schema_id: schema,
                manifest_list: format!("snap-{id}.avro"),
            },
        )
    };
    MetadataView {
        snapshots: [
            snap(1, 1000, Some(0)),
            snap(2, 2000, Some(1)),
            snap(3, 3000, None),
        ]
        .into(),
        current_snapshot_id: Some(3),
        current_schema_id: 2,
        refs: [
            (
                "main".to_owned(),
                RefInfo {
                    snapshot_id: 3,
                    kind: RefKind::Branch,
                },
            ),
            (
                "dev".to_owned(),
                RefInfo {
                    snapshot_id: 1,
                    kind: RefKind::Branch,
                },
            ),
            (
                "v1".to_owned(),
                RefInfo {
                    snapshot_id: 2,
                    kind: RefKind::Tag,
                },
            ),
            (
                "gone".to_owned(),
                RefInfo {
                    snapshot_id: 99,
                    kind: RefKind::Tag,
                },
            ),
        ]
        .into(),
        // Snapshot 4 is in the log but expired.
        log: vec![(1000, 1), (2000, 2), (3000, 3), (4000, 4)],
    }
}

#[test]
fn time_travel_resolution() {
    let v = view();
    let got = |spec: VersionSpec| {
        resolve(&v, &spec).map(|r| (r.snapshot.map(|s| s.snapshot_id), r.schema_id))
    };
    assert_eq!(got(VersionSpec::Current).unwrap(), (Some(3), 2));
    assert_eq!(got(VersionSpec::Id(1)).unwrap(), (Some(1), 0));
    // A snapshot without a schema id reads the current schema.
    assert_eq!(got(VersionSpec::Id(3)).unwrap(), (Some(3), 2));
    // A tag reads its snapshot's schema; a branch the current one.
    assert_eq!(
        got(VersionSpec::Ref("v1".to_owned())).unwrap(),
        (Some(2), 1)
    );
    assert_eq!(
        got(VersionSpec::Ref("dev".to_owned())).unwrap(),
        (Some(1), 2)
    );
    assert_eq!(
        got(VersionSpec::Ref("main".to_owned())).unwrap(),
        (Some(3), 2)
    );
    // Timestamps: micros truncated to millis, the last log entry at or before.
    assert_eq!(
        got(VersionSpec::TimestampMicros(2_000_999)).unwrap(),
        (Some(2), 1)
    );
    assert_eq!(
        got(VersionSpec::TimestampMicros(2_000_000)).unwrap(),
        (Some(2), 1)
    );
    assert_eq!(
        got(VersionSpec::TimestampMicros(1_999_999)).unwrap(),
        (Some(1), 0)
    );

    let err = |spec: VersionSpec| resolve(&v, &spec).unwrap_err();
    let e = err(VersionSpec::Id(7));
    assert_eq!(e.message(), "Cannot find snapshot with ID 7");
    // A user-facing class carrying Iceberg's text, not INTERNAL_ERROR.
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivTimeTravelNotFound
    );
    assert_eq!(e.sql_state(), Some("42K0E"));
    let e = err(VersionSpec::Ref("nope".to_owned()));
    assert_eq!(
        e.message(),
        "Cannot find matching snapshot ID or reference name for version nope"
    );
    // A dangling ref is an error, never a panic.
    let e = err(VersionSpec::Ref("gone".to_owned()));
    assert_eq!(e.message(), "Cannot find snapshot with ID 99");
    let e = err(VersionSpec::TimestampMicros(999_999));
    assert_eq!(
        e.message(),
        "Cannot find a snapshot older than 1970-01-01T00:00:00.999+00:00"
    );
    // The log names an expired snapshot.
    let e = err(VersionSpec::TimestampMicros(5_000_000));
    assert_eq!(e.message(), "Cannot find snapshot with ID 4");
    // No snapshot at all: the current version is empty, time travel fails.
    let empty = MetadataView {
        current_schema_id: 0,
        ..MetadataView::default()
    };
    let r = resolve(&empty, &VersionSpec::Current).unwrap();
    assert_eq!((r.snapshot, r.schema_id), (None, 0));
    assert!(resolve(&empty, &VersionSpec::TimestampMicros(0)).is_err());
}

/// Review F4: the last log entry in log order at or before the time, not the
/// largest time (Java walks `history()` keeping the last match; log times may go
/// backwards within its clock-skew allowance).
#[test]
fn timestamp_as_of_takes_the_last_log_entry_in_order() {
    let got = |log: Vec<(i64, i64)>, us: i64| {
        let mut v = view();
        v.log = log;
        resolve(&v, &VersionSpec::TimestampMicros(us))
            .map(|r| r.snapshot.map(|s| s.snapshot_id))
            .ok()
            .flatten()
    };
    let skewed = || vec![(1000, 1), (990, 2)];
    assert_eq!(got(skewed(), 1_005_000), Some(2));
    // Only the second entry is at or before 995 ms.
    assert_eq!(got(skewed(), 995_000), Some(2));
    assert_eq!(got(skewed(), 989_000), None);
    let three = || vec![(1000, 1), (990, 2), (1000, 3)];
    assert_eq!(got(three(), 1_000_000), Some(3));
    assert_eq!(got(three(), 999_000), Some(2));
}

#[test]
fn timestamp_format_is_java_instant() {
    assert_eq!(format_timestamp_millis(0), "1970-01-01T00:00:00+00:00");
    assert_eq!(
        format_timestamp_millis(1_700_000_000_123),
        "2023-11-14T22:13:20.123+00:00"
    );
    assert_eq!(format_timestamp_millis(-1), "1969-12-31T23:59:59.999+00:00");
}

/// Review F3: the truncation length and whether bounds exist come from the
/// table's `write.metadata.metrics.*` properties, not a constant.
#[test]
fn metrics_modes_come_from_the_table_properties() {
    use super::metrics::{MetricsMode, WriterMetrics};
    use ScalarValue as S;
    assert_eq!(MetricsMode::parse("truncate(8)"), MetricsMode::Truncate(8));
    assert_eq!(MetricsMode::parse(" Full "), MetricsMode::Full);
    assert_eq!(MetricsMode::parse("counts"), MetricsMode::Counts);
    assert_eq!(MetricsMode::parse("none"), MetricsMode::None);
    assert_eq!(MetricsMode::parse("truncate(0)"), MetricsMode::Unknown);
    assert_eq!(MetricsMode::parse("bogus"), MetricsMode::Unknown);

    let fields = [
        field(1, "s", IcebergType::String),
        field(2, "k", IcebergType::Int),
    ];
    let specs = unpartitioned();
    let props = |pairs: &[(&'static str, &'static str)]| {
        WriterMetrics::from_properties(pairs.iter().copied(), &fields)
    };
    let keeps_with = |m: &WriterMetrics, f: &ContentFile, e: &PushdownExpr| {
        Pruner::new(e, &fields, &specs, PruneOptions::default())
            .with_metrics(m)
            .file_may_match(f)
    };
    // A non-rounding writer truncated "abcdefghz" to an upper bound "abcd".
    let strings = with_metrics(1, b"abc".to_vec(), b"abcd".to_vec(), 5, 0);
    let hit = cmp("s", CmpOp::Eq, S::Utf8("abcdefghz".to_owned()));
    // Default truncate(16): a 4-character upper is exact, so the file is pruned
    // (the documented assumption about writers).
    assert!(!keeps_with(&WriterMetrics::default(), &strings, &hit));
    // truncate(4) on the table: the 4-character upper is a prefix; kept.
    let t4 = props(&[("write.metadata.metrics.default", "truncate(4)")]);
    assert!(keeps_with(&t4, &strings, &hit));
    // The column's own mode wins over the default.
    let col4 = props(&[
        ("write.metadata.metrics.default", "full"),
        ("write.metadata.metrics.column.s", "truncate(4)"),
    ]);
    assert!(keeps_with(&col4, &strings, &hit));
    // An unparsable mode: no assumption about truncation.
    let bad = props(&[("write.metadata.metrics.default", "truncate(x)")]);
    assert!(keeps_with(&bad, &strings, &hit));
    // `full`: bounds are exact at any length.
    let full = props(&[("write.metadata.metrics.default", "full")]);
    let long = with_metrics(1, b"a".to_vec(), "b".repeat(40).into_bytes(), 5, 0);
    // A value that extends the 40-character upper bound: above it if the bound
    // is exact, possibly below a truncated one.
    let past = cmp("s", CmpOp::Eq, S::Utf8(format!("{}c", "b".repeat(40))));
    assert!(!keeps_with(&full, &long, &past));
    assert!(keeps_with(&WriterMetrics::default(), &long, &past));
    // `none` / `counts`: bounds found are not trusted; counts still are.
    let ints = with_metrics(
        2,
        1i32.to_le_bytes().to_vec(),
        5i32.to_le_bytes().to_vec(),
        5,
        0,
    );
    let out = cmp("k", CmpOp::Eq, S::Int32(50));
    assert!(!keeps_with(&WriterMetrics::default(), &ints, &out));
    for mode in ["none", "counts"] {
        let m =
            WriterMetrics::from_properties([("write.metadata.metrics.column.k", mode)], &fields);
        assert!(keeps_with(&m, &ints, &out), "{mode}");
    }
    let counts = props(&[("write.metadata.metrics.default", "counts")]);
    let all_null = with_metrics(2, Vec::new(), Vec::new(), 5, 5);
    assert!(!keeps_with(
        &counts,
        &all_null,
        &PushdownExpr::IsNotNull(col("k"))
    ));
}

/// Review F6: a summary without bounds is all null only when it says it has
/// nulls (Java's `allValuesAreNull`).
#[test]
fn a_summary_without_bounds_or_nulls_is_not_all_null() {
    let fields = [field(1, "k", IcebergType::Int)];
    let specs = spec(vec![pfield(1000, 1, Transform::Identity, IcebergType::Int)]);
    let mut m = manifest_entry(1, None);
    m.spec_id = 1;
    let summary = |contains_null| PartitionFieldSummary {
        contains_null,
        contains_nan: None,
        lower: None,
        upper: None,
    };
    let not_null = PushdownExpr::IsNotNull(col("k"));
    let may = |m: &ManifestFileEntry| {
        Pruner::new(&not_null, &fields, &specs, PruneOptions::default()).manifest_may_match(m)
    };
    m.partitions = Some(vec![summary(true)]);
    assert!(!may(&m), "no bounds and nulls: all null");
    m.partitions = Some(vec![summary(false)]);
    assert!(may(&m), "no bounds and no nulls: malformed, kept");
}

/// Review F7: bucket and truncate of a literal of another type than the
/// source (a decimal at another scale) do not apply, so nothing is pruned.
#[test]
fn transforms_refuse_literals_of_another_type() {
    use ScalarValue as S;
    let dec = IcebergType::Decimal {
        precision: 9,
        scale: 2,
    };
    let at = |scale| S::Decimal128 {
        unscaled: 1234,
        precision: 9,
        scale,
    };
    assert!(transform::apply(Transform::Bucket(16), &dec, &at(2)).is_some());
    assert_eq!(transform::apply(Transform::Bucket(16), &dec, &at(3)), None);
    assert_eq!(
        transform::apply(Transform::Truncate(10), &dec, &at(1)),
        None
    );
    assert_eq!(
        transform::apply(Transform::Bucket(4), &IcebergType::Long, &S::Int32(1)),
        None
    );
    assert_eq!(
        transform::apply(Transform::Identity, &IcebergType::Int, &S::Int64(1)),
        None
    );
}

/// Review F8: uuid bounds are not used (their order is unsettled).
#[test]
fn uuid_bounds_are_not_used() {
    use ScalarValue as S;
    let fields = [field(1, "u", IcebergType::Uuid)];
    let lo = [0u8; 16].to_vec();
    let hi = [1u8; 16].to_vec();
    let f = with_metrics(1, lo, hi, 5, 0);
    let e = cmp(
        "u",
        CmpOp::Eq,
        S::Utf8("ffffffff-ffff-ffff-ffff-ffffffffffff".to_owned()),
    );
    assert!(keeps(&fields, &unpartitioned(), &f, &e));
}
