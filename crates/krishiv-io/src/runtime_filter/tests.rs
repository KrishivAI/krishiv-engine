// SPDX-License-Identifier: Apache-2.0
//! Runtime filter tests: key normalization across types, the builder's
//! forms and merges, row evaluation (never a false negative), lowering to
//! statistics predicates, publication, and a loom model of the protocol.

#[cfg(not(loom))]
mod unit {
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
        Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
        StringViewArray, TimestampMicrosecondArray,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use proptest::prelude::*;

    use crate::runtime_filter::*;
    use crate::{
        CmpOp, ColumnPath, ColumnStatistics, FileStatistics, Precision, PushdownExpr, ScalarValue,
        TriState, prune,
    };

    /// Exact forms only up to the `IN` list (a Bloom beyond).
    fn options(max_values: usize) -> KeyFilterOptions {
        KeyFilterOptions {
            max_values,
            exact_max_values: max_values,
            ..KeyFilterOptions::default()
        }
    }

    fn build(keys: &dyn Array, max_values: usize) -> KeyFilter {
        let mut b = KeyFilterBuilder::new(keys.data_type().clone(), options(max_values))
            .unwrap_or_else(|| panic!("unsupported {}", keys.data_type()));
        b.add(keys, None).unwrap_or_else(|e| panic!("{e}"));
        b.finish()
    }

    fn pass(f: &KeyFilter, column: &dyn Array) -> Vec<bool> {
        f.evaluate(column)
            .unwrap_or_else(|| panic!("{} does not widen", column.data_type()))
            .iter()
            .map(|b| b == Some(true))
            .collect()
    }

    #[test]
    fn widenings_are_exactly_the_lossless_ones() {
        use DataType as D;
        let yes = [
            (D::Int32, D::Int64),
            (D::Int8, D::Int16),
            (D::Int16, D::Float32),
            (D::Int32, D::Float64),
            (D::Float32, D::Float64),
            (D::Int32, D::Decimal128(10, 0)),
            (D::Int64, D::Decimal128(21, 2)),
            (D::Decimal128(5, 2), D::Decimal128(7, 3)),
            (D::Utf8, D::Utf8View),
            (D::LargeUtf8, D::Utf8),
            (D::BinaryView, D::Binary),
            (
                D::Timestamp(TimeUnit::Microsecond, None),
                D::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            ),
            (D::Date32, D::Date32),
        ];
        for (f, t) in &yes {
            assert!(widens_to(f, t), "{f} -> {t}");
        }
        let no = [
            (D::Int64, D::Int32),
            (D::Int64, D::Float64),
            (D::Int32, D::Float32),
            (D::Float64, D::Float32),
            (D::Int32, D::Decimal128(9, 0)),
            (D::Decimal128(5, 2), D::Decimal128(5, 3)),
            (D::Decimal128(5, 2), D::Decimal128(7, 1)),
            (D::Utf8, D::Binary),
            (D::Date32, D::Timestamp(TimeUnit::Microsecond, None)),
            (
                D::Timestamp(TimeUnit::Millisecond, None),
                D::Timestamp(TimeUnit::Microsecond, None),
            ),
            (D::Int64, D::Utf8),
            (D::Null, D::Null),
        ];
        for (f, t) in &no {
            assert!(!widens_to(f, t), "{f} -> {t}");
        }
    }

    /// The bug the design review found: a Bloom over int64 keys probed with
    /// an int32 column must hash the widened value, not the file's bytes.
    #[test]
    fn mixed_integer_widths_hash_as_the_key_type() {
        let keys: Vec<i64> = (0..200).map(|i| i * 1_000_003).collect();
        let f = build(&Int64Array::from(keys.clone()), 50);
        assert!(f.value_count().is_none() && f.bloom_bytes().is_some());
        let probe: Vec<i32> = keys
            .iter()
            .map(|&k| i32::try_from(k).unwrap_or(0))
            .collect();
        let narrow = Int32Array::from(probe.clone());
        assert!(
            pass(&f, &narrow).iter().all(|p| *p),
            "every build key passes"
        );
        let n64 = normalize_key_bytes(&narrow, &DataType::Int64).unwrap_or_else(|| panic!());
        let w64 = normalize_key_bytes(&Int64Array::from(keys), &DataType::Int64)
            .unwrap_or_else(|| panic!());
        assert_eq!(n64.hashes(), w64.hashes());
        assert_eq!((n64.len(), n64.is_empty()), (200, false));
        // The key type decides the bytes: int32 keys hash four bytes.
        let n32 = normalize_key_bytes(&narrow, &DataType::Int32).unwrap_or_else(|| panic!());
        assert_eq!(n32.hashes()[1], xxh64(&probe[1].to_le_bytes()));
        assert_ne!(n32.hashes()[1], n64.hashes()[1]);
        // A narrowing is never filtered.
        assert!(f.evaluate(&Int8Array::from(vec![1i8])).is_some());
        let i32_filter = build(&Int32Array::from(vec![1, 2]), 50);
        assert!(i32_filter.evaluate(&Int64Array::from(vec![1i64])).is_none());
    }

    #[test]
    fn floats_normalize_signed_zero_and_nan() {
        let f = build(&Float64Array::from(vec![-0.0, f64::NAN, 2.5]), 50);
        assert!(!f.has_range(), "a NaN key drops the range");
        let probe = Float64Array::from(vec![0.0, -0.0, -f64::NAN, 2.5, 3.0]);
        assert_eq!(pass(&f, &probe), [true, true, true, true, false]);
        let f = build(&Float64Array::from(vec![-0.0, f64::NAN, 2.5]), 0);
        assert_eq!(pass(&f, &probe)[..4], [true, true, true, true]);
        // Without NaN, the range holds and NaN probe rows fail it.
        let f = build(&Float64Array::from(vec![-0.0, 1.0]), 0);
        assert!(f.has_range() && f.bloom_bytes().is_some());
        assert_eq!(
            pass(&f, &Float64Array::from(vec![0.0, -0.0, f64::NAN, 1.0])),
            [true, true, false, true]
        );
        // Float32 keys hash the float32 bytes, also from int16 columns.
        let f = build(&Float32Array::from(vec![1.0f32, 300.0]), 0);
        assert_eq!(pass(&f, &Int16Array::from(vec![1, 300])), [true, true]);
        assert_eq!(
            pass(&f, &Float32Array::from(vec![-0.0f32, 1.0])),
            [false, true]
        );
        let (lo, hi) = f.range_scalars().unwrap_or_else(|| panic!());
        assert_eq!(
            (lo, hi),
            (ScalarValue::Float32(1.0), ScalarValue::Float32(300.0))
        );
        let v = build(&Float32Array::from(vec![1.5f32]), 50);
        assert_eq!(v.value_scalars(), Some(vec![ScalarValue::Float32(1.5)]));
        let v = build(&Float64Array::from(vec![1.5]), 50);
        assert_eq!(v.value_scalars(), Some(vec![ScalarValue::Float64(1.5)]));
    }

    #[test]
    fn decimals_widen_to_the_key_scale() {
        let key = DataType::Decimal128(10, 2);
        let keys = Decimal128Array::from(vec![150, 2500])
            .with_precision_and_scale(10, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        let f = build(&keys, 50);
        assert_eq!(f.key_type(), &key);
        // 1.5 and 25 at scale 1, and as integers.
        let probe = Decimal128Array::from(vec![15, 250, 16])
            .with_precision_and_scale(5, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pass(&f, &probe), [true, true, false]);
        assert_eq!(pass(&f, &Int16Array::from(vec![25, 1])), [true, false]);
        assert!(
            f.evaluate(&Int32Array::from(vec![25])).is_none(),
            "10 digits"
        );
        let bloom = build(&keys, 0);
        assert_eq!(pass(&bloom, &probe)[..2], [true, true]);
        let (lo, _) = f.range_scalars().unwrap_or_else(|| panic!());
        assert_eq!(
            lo,
            ScalarValue::Decimal128 {
                unscaled: 150,
                precision: 10,
                scale: 2
            }
        );
        // A stored value beyond its precision is not filtered.
        let wild = Decimal128Array::from(vec![i128::MAX])
            .with_precision_and_scale(5, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(f.evaluate(&wild).is_none());
    }

    #[test]
    fn strings_compare_bytes_across_representations() {
        let keys = StringViewArray::from(vec!["apple", "a much longer string than twelve", "kiwi"]);
        for max_values in [50, 0] {
            let f = build(&keys, max_values);
            let probes: Vec<ArrayRef> = vec![
                Arc::new(StringArray::from(vec!["kiwi", "pear", "apple"])),
                Arc::new(LargeStringArray::from(vec!["kiwi", "pear", "apple"])),
                Arc::new(StringViewArray::from(vec!["kiwi", "pear", "apple"])),
            ];
            for p in &probes {
                assert_eq!(
                    pass(&f, p.as_ref()),
                    [true, false, true],
                    "{}",
                    p.data_type()
                );
            }
            assert!(f.evaluate(&BinaryArray::from(vec![&b"kiwi"[..]])).is_none());
        }
        let bin = build(&BinaryArray::from(vec![&b"\x00\xff"[..]]), 50);
        let (lo, _) = bin.range_scalars().unwrap_or_else(|| panic!());
        assert_eq!(lo, ScalarValue::Binary(vec![0, 255]));
        assert_eq!(
            bin.value_scalars(),
            Some(vec![ScalarValue::Binary(vec![0, 255])])
        );
        let s = build(&StringArray::from(vec!["b", "a", "b"]), 50);
        assert_eq!(
            s.value_scalars(),
            Some(vec![
                ScalarValue::Utf8("a".into()),
                ScalarValue::Utf8("b".into())
            ])
        );
        let large = build(&LargeStringArray::from(vec!["z"]), 0);
        assert_eq!(
            pass(&large, &StringArray::from(vec!["z", "y"])),
            [true, false]
        );
    }

    #[test]
    fn other_key_types() {
        let b = build(&BooleanArray::from(vec![true]), 50);
        assert_eq!(
            pass(&b, &BooleanArray::from(vec![true, false])),
            [true, false]
        );
        assert_eq!(
            b.range_scalars(),
            Some((ScalarValue::Boolean(true), ScalarValue::Boolean(true)))
        );
        let d = build(&Date32Array::from(vec![10, 20]), 50);
        assert_eq!(
            d.range_scalars().map(|r| r.1),
            Some(ScalarValue::Date32(20))
        );
        assert_eq!(pass(&d, &Date32Array::from(vec![15, 20])), [false, true]);
        let ts = TimestampMicrosecondArray::from(vec![7]).with_timezone("UTC");
        let t = build(&ts, 0);
        let probe = TimestampMicrosecondArray::from(vec![7, 8]);
        assert_eq!(pass(&t, &probe), [true, false]);
        assert_eq!(
            t.range_scalars().map(|r| r.0),
            Some(ScalarValue::TimestampMicros(7))
        );
        let i8f = build(&Int8Array::from(vec![-3, 4]), 50);
        assert_eq!(
            i8f.value_scalars(),
            Some(vec![ScalarValue::Int8(-3), ScalarValue::Int8(4)])
        );
        let i16f = build(&Int16Array::from(vec![-3]), 50);
        assert_eq!(i16f.value_scalars(), Some(vec![ScalarValue::Int16(-3)]));
        assert!(KeyFilterBuilder::new(DataType::Null, options(1)).is_none());
        let mut b = KeyFilterBuilder::new(DataType::Int64, options(1)).unwrap_or_else(|| panic!());
        assert!(
            b.add(&Int32Array::from(vec![1]), None).is_err(),
            "wrong key type"
        );
    }

    #[test]
    fn builder_forms_nulls_and_merges() {
        let opts = KeyFilterOptions {
            max_values: 3,
            exact_max_values: 3,
            bloom_max_items: 10,
            ..KeyFilterOptions::default()
        };
        let mut a = KeyFilterBuilder::new(DataType::Int64, opts).unwrap_or_else(|| panic!());
        // Rows 1 and 3 are not kept by the producer (a NULL `=` key elsewhere).
        let include = NullBuffer::from(vec![true, false, true, false]);
        a.add(
            &Int64Array::from(vec![Some(5), Some(100), None, Some(-7)]),
            Some(&include),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mut b = KeyFilterBuilder::new(DataType::Int64, opts).unwrap_or_else(|| panic!());
        b.add(&Int64Array::from(vec![1, 2]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(a.heap_bytes() > 0);
        let mut merged = a.clone();
        merged.merge(b.clone());
        let f = merged.finish();
        assert!(f.nulls_pass(), "a kept NULL key lets NULLs pass");
        assert_eq!(f.value_count(), Some(3));
        assert_eq!(
            f.range_scalars(),
            Some((ScalarValue::Int64(1), ScalarValue::Int64(5)))
        );
        let probe = Int64Array::from(vec![Some(5), None, Some(100), Some(3)]);
        assert_eq!(pass(&f, &probe), [true, true, false, false]);
        assert_eq!(f.range_only().value_count(), None);
        assert!(f.heap_bytes() > 0);
        // A fourth value overflows the set into a Bloom.
        let mut four = b.clone();
        four.merge(a.clone());
        four.add(&Int64Array::from(vec![9]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let f = four.finish();
        assert_eq!(f.value_count(), None);
        assert!(f.bloom_bytes().is_some());
        assert!(f.heap_bytes() >= f.bloom_bytes().unwrap_or_default());
        assert!(
            pass(&f, &Int64Array::from(vec![9, 1, 2, 5]))
                .iter()
                .all(|x| *x)
        );
        // Over the item cap there is no Bloom; dropping it is the same.
        let mut many = KeyFilterBuilder::new(DataType::Int64, opts).unwrap_or_else(|| panic!());
        many.add(&Int64Array::from((0..11).collect::<Vec<i64>>()), None)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(many.finish().bloom_bytes(), None);
        let mut dropped = b.clone();
        dropped.merge(a);
        dropped
            .add(&Int64Array::from(vec![9, 10]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        dropped.drop_bloom();
        let f = dropped.finish();
        assert!(f.bloom_bytes().is_none() && f.value_count().is_none() && f.has_range());
        // Merging past the item cap drops the Bloom too.
        let mut x = KeyFilterBuilder::new(DataType::Int64, opts).unwrap_or_else(|| panic!());
        x.add(&Int64Array::from((0..6).collect::<Vec<i64>>()), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let y = x.clone();
        x.merge(y);
        assert_eq!(x.finish().bloom_bytes(), None);
        // A Bloom with too high an expected false-positive rate is not built.
        let capped = KeyFilterOptions {
            max_values: 0,
            exact_max_values: 0,
            bloom_max_bytes: 32,
            ..KeyFilterOptions::default()
        };
        let mut c = KeyFilterBuilder::new(DataType::Int64, capped).unwrap_or_else(|| panic!());
        c.add(&Int64Array::from((0..1000).collect::<Vec<i64>>()), None)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c.finish().bloom_bytes(), None);
        // No Bloom requested: values or range only.
        let none = KeyFilterOptions {
            bloom: false,
            max_values: 0,
            ..KeyFilterOptions::default()
        };
        let mut d = KeyFilterBuilder::new(DataType::Int64, none).unwrap_or_else(|| panic!());
        d.add(&Int64Array::from(vec![1, 2]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.finish().bloom_bytes(), None);
        // Partials of string keys merge their ranges.
        let mut s1 = KeyFilterBuilder::new(DataType::Utf8, options(0)).unwrap_or_else(|| panic!());
        s1.add(&StringArray::from(vec!["m"]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut s2 = s1.clone();
        s2.add(&StringArray::from(vec!["a", "z"]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        s1.merge(s2);
        assert!(s1.heap_bytes() > 0);
        assert_eq!(
            s1.finish().range_scalars(),
            Some((ScalarValue::Utf8("a".into()), ScalarValue::Utf8("z".into())))
        );
    }

    #[test]
    fn empty_builds_drop_everything_but_null_safe_nulls() {
        let mut b = KeyFilterBuilder::new(DataType::Int32, options(50)).unwrap_or_else(|| panic!());
        let f = b.clone().finish();
        assert!(f.is_empty() && !f.nulls_pass());
        let probe = Int32Array::from(vec![Some(1), None]);
        assert_eq!(pass(&f, &probe), [false, false]);
        b.add(&Int32Array::from(vec![None::<i32>]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let f = b.finish();
        assert!(f.is_empty() && f.nulls_pass());
        assert_eq!(pass(&f, &probe), [false, true]);
    }

    fn stats(column: &str, min: ScalarValue, max: ScalarValue, nulls: u64) -> FileStatistics {
        let mut c = ColumnStatistics::new(ColumnPath::new(column));
        c.min = Precision::Exact(min);
        c.max = Precision::Exact(max);
        c.null_count = Precision::Exact(nulls);
        let mut s = FileStatistics::with_rows(Precision::Exact(100));
        s.columns.push(c);
        s
    }

    #[test]
    fn lowering_retypes_to_the_column() {
        let col = ColumnPath::new("k");
        let id = FilterId(3);
        let f = Arc::new(build(&Int64Array::from(vec![10, 20, 3_000_000_000]), 2));
        // Int64 keys on an int32 column: the max is out of range (no upper
        // bound); the values overflowed into a Bloom.
        let p = f
            .lower(&col, &DataType::Int32, id, 1)
            .unwrap_or_else(|| panic!());
        let PushdownExpr::And(parts) = &p else {
            panic!("{p:?}")
        };
        assert_eq!(
            parts[0],
            PushdownExpr::cmp(col.clone(), CmpOp::GtEq, ScalarValue::Int32(10))
        );
        assert!(matches!(parts[1], PushdownExpr::Membership { .. }));
        assert_eq!(parts.len(), 2);
        let hit = stats("k", ScalarValue::Int32(0), ScalarValue::Int32(15), 0);
        let miss = stats("k", ScalarValue::Int32(-5), ScalarValue::Int32(9), 0);
        assert_eq!(prune(&p, &hit), TriState::Unknown);
        assert_eq!(prune(&p, &miss), TriState::False);
        // A value set on a narrower column drops values it cannot hold.
        let v = Arc::new(build(&Int64Array::from(vec![-3_000_000_000, 7]), 50));
        let p = v
            .lower(&col, &DataType::Int32, id, 1)
            .unwrap_or_else(|| panic!());
        let PushdownExpr::And(vparts) = &p else {
            panic!("{p:?}")
        };
        assert!(vparts.iter().any(
            |x| matches!(x, PushdownExpr::In { values, .. } if values == &[ScalarValue::Int32(7)])
        ));
        let above = stats("k", ScalarValue::Int32(8), ScalarValue::Int32(9), 0);
        assert_eq!(prune(&p, &above), TriState::False);
        // All values outside the column type: nothing matches.
        let out = Arc::new(build(&Int64Array::from(vec![-3_000_000_000]), 50));
        let p = out
            .lower(&col, &DataType::Int32, id, 1)
            .unwrap_or_else(|| panic!());
        assert_eq!(prune(&p, &hit), TriState::False);
        // A column that does not widen to the key type is not filtered.
        assert!(f.lower(&col, &DataType::Utf8, id, 1).is_none());
        // Empty builds, null-safe NULLs.
        let mut b = KeyFilterBuilder::new(DataType::Int64, options(50)).unwrap_or_else(|| panic!());
        let empty = Arc::new(b.clone().finish());
        assert_eq!(
            empty.lower(&col, &DataType::Int64, id, 1),
            Some(PushdownExpr::Constant(false))
        );
        b.add(&Int64Array::from(vec![None::<i64>]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let nulls = Arc::new(b.clone().finish());
        assert_eq!(
            nulls.lower(&col, &DataType::Int64, id, 1),
            Some(PushdownExpr::IsNull(col.clone()))
        );
        b.add(&Int64Array::from(vec![4]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let some = Arc::new(b.finish());
        let p = some
            .lower(&col, &DataType::Int64, id, 1)
            .unwrap_or_else(|| panic!());
        assert!(matches!(&p, PushdownExpr::Or(x) if x[1] == PushdownExpr::IsNull(col.clone())));
        let only_nulls = stats("k", ScalarValue::Int64(9), ScalarValue::Int64(9), 100);
        assert_eq!(prune(&p, &only_nulls), TriState::True);
        // Membership is a leaf for negation and statistics.
        let m = PushdownExpr::Not(Box::new(parts[1].clone()));
        assert_eq!(m.to_nnf(), m);
        assert_eq!(prune(&m, &hit), TriState::Unknown);
        assert_eq!(m.columns(), vec![&col]);
        assert!(format!("{:?}", parts[1]).contains("#3@1"));
    }

    #[test]
    fn publication_and_scan_filters() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new(
                "s",
                DataType::Struct(vec![Field::new("x", DataType::Int64, true)].into()),
                true,
            ),
        ]);
        let f = Arc::new(RuntimeFilter::new(
            FilterId(1),
            ColumnPath::new("a"),
            DataType::Int64,
        ));
        assert!(format!("{f:?}").contains("generation"));
        assert_eq!(f.id().to_string(), "#1");
        let filter = |id: u64, path: ColumnPath| {
            Arc::new(RuntimeFilter::new(FilterId(id), path, DataType::Int64))
        };
        assert!(ScanFilter::new(filter(2, ColumnPath::new("zz")), &schema).is_none());
        let nested = ScanFilter::new(filter(3, ColumnPath::nested(["s", "x"])), &schema);
        assert_eq!(
            nested
                .as_ref()
                .and_then(|n| n.column())
                .map(|c| c.1.clone()),
            Some(DataType::Int64)
        );
        assert!(ScanFilter::new(filter(4, ColumnPath::nested(["a", "x"])), &schema).is_none());
        let sf = ScanFilter::new(Arc::clone(&f), &schema).unwrap_or_else(|| panic!());
        assert!(format!("{sf:?}").contains("ScanFilter"));
        let column = Int32Array::from(vec![1, 2, 3]);
        assert_eq!(sf.predicate(), None);
        assert!(sf.evaluate(&column).is_none());
        assert!(!sf.drops_everything());
        assert_eq!(sf.generation(), 0);
        let keys = Arc::new(build(&Int64Array::from(vec![2]), 50));
        assert_eq!(f.publish(FilterSnapshot::Final(Arc::clone(&keys))), Some(1));
        assert_eq!(
            f.publish(FilterSnapshot::Pending),
            None,
            "final is published once"
        );
        f.abandon(DisabledReason::ProducerFailed);
        assert_eq!(f.generation_hint(), 1);
        assert_eq!(sf.generation(), 1);
        assert_eq!(sf.current().as_deref(), Some(keys.as_ref()));
        assert_eq!(pass(&keys, &column), [false, true, false]);
        assert!(sf.evaluate(&column).is_some());
        assert!(sf.predicate().is_some());
        // A file lacking the column reads NULLs.
        let missing = ScanFilter::at(Arc::clone(&f), None);
        assert_eq!(missing.predicate(), Some(PushdownExpr::Constant(false)));
        // Disabled filters apply nothing; empty ones drop everything.
        let d = filter(5, ColumnPath::new("a"));
        d.abandon(DisabledReason::Off);
        let sd = ScanFilter::new(Arc::clone(&d), &schema).unwrap_or_else(|| panic!());
        assert_eq!((sd.predicate(), sd.generation()), (None, 1));
        let e = filter(6, ColumnPath::new("a"));
        let empty = KeyFilterBuilder::new(DataType::Int64, options(1)).unwrap_or_else(|| panic!());
        e.publish(FilterSnapshot::Final(Arc::new(empty.finish())));
        assert!(ScanFilter::new(e, &schema).is_some_and(|s| s.drops_everything()));
        f.stats().add_rows(10, 4);
        f.stats().add_unit(100);
        f.stats().add_split(50);
        let c = f.stats().counts();
        assert_eq!(
            (
                c.rows_tested,
                c.rows_dropped,
                c.units_pruned,
                c.bytes_pruned,
                c.splits_skipped
            ),
            (10, 4, 1, 150, 1)
        );
    }

    /// Build keys of the key type, and the same values in a narrower or
    /// other representation for the probe.
    fn keyed(kind: u8, v: &[Option<i32>]) -> (ArrayRef, ArrayRef) {
        match kind {
            0 => (
                Arc::new(v.iter().map(|x| x.map(i64::from)).collect::<Int64Array>()),
                Arc::new(v.iter().copied().collect::<Int32Array>()),
            ),
            1 => (
                Arc::new(
                    v.iter()
                        .map(|x| x.map(|y| i128::from(y) * 100))
                        .collect::<Decimal128Array>()
                        .with_precision_and_scale(12, 2)
                        .unwrap_or_else(|e| panic!("{e}")),
                ),
                Arc::new(
                    v.iter()
                        .map(|x| x.map(|y| i128::from(y) * 10))
                        .collect::<Decimal128Array>()
                        .with_precision_and_scale(10, 1)
                        .unwrap_or_else(|e| panic!("{e}")),
                ),
            ),
            2 => (
                Arc::new(
                    v.iter()
                        .map(|x| x.map(|y| format!("k{y}")))
                        .collect::<StringViewArray>(),
                ),
                Arc::new(
                    v.iter()
                        .map(|x| x.map(|y| format!("k{y}")))
                        .collect::<StringArray>(),
                ),
            ),
            _ => (
                Arc::new(
                    v.iter()
                        .map(|x| x.map(|y| if y == 0 { -0.0 } else { f64::from(y) / 4.0 }))
                        .collect::<Float64Array>(),
                ),
                Arc::new(
                    v.iter()
                        .map(|x| x.map(|y| f64::from(y) / 4.0))
                        .collect::<Float64Array>(),
                ),
            ),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]
        /// Never a false negative: every probe row whose key is among the
        /// build's keys passes, whatever the forms, widths and NULLs.
        #[test]
        fn no_false_negatives(
            kind in 0u8..4,
            build_keys in proptest::collection::vec(proptest::option::weighted(0.9, -60i32..60), 0..120),
            probe_keys in proptest::collection::vec(proptest::option::weighted(0.9, -80i32..80), 0..200),
            max_values in prop_oneof![Just(0usize), Just(5), Just(50)],
            split in 0usize..120,
        ) {
            let split = split.min(build_keys.len());
            let (b1, _) = keyed(kind, &build_keys[..split]);
            let (b2, _) = keyed(kind, &build_keys[split..]);
            let opts = options(max_values);
            let mut x = KeyFilterBuilder::new(b1.data_type().clone(), opts).unwrap_or_else(|| panic!());
            // As an `=` join keeps them: rows with a NULL key are not stored.
            x.add(b1.as_ref(), b1.logical_nulls().as_ref()).unwrap_or_else(|e| panic!("{e}"));
            let mut y = KeyFilterBuilder::new(b2.data_type().clone(), opts).unwrap_or_else(|| panic!());
            y.add(b2.as_ref(), b2.logical_nulls().as_ref()).unwrap_or_else(|e| panic!("{e}"));
            x.merge(y);
            let f = x.finish();
            let (_, probe) = keyed(kind, &probe_keys);
            let out = pass(&f, probe.as_ref());
            for (i, k) in probe_keys.iter().enumerate() {
                if let Some(k) = k
                    && build_keys.contains(&Some(*k))
                {
                    prop_assert!(out[i], "key {} of kind {} was dropped", k, kind);
                }
                if k.is_none() {
                    prop_assert!(!out[i], "NULL never matches an `=` key");
                }
            }
        }
    }
}

/// The publication protocol under loom: a reader that sees a generation
/// finds that generation's snapshot (never an older one), and a consumer's
/// cache never pairs a generation with another snapshot.
/// `RUSTFLAGS="--cfg loom" cargo test -p krishiv-io --lib --release runtime_filter`.
#[cfg(loom)]
mod loom_model {
    use std::sync::Arc;

    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use loom::thread;

    use crate::ColumnPath;
    use crate::runtime_filter::{
        FilterId, FilterSnapshot, KeyFilterBuilder, KeyFilterOptions, RuntimeFilter, ScanFilter,
    };

    fn keys() -> Arc<crate::KeyFilter> {
        let mut b = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default())
            .unwrap_or_else(|| panic!());
        b.add(&Int64Array::from(vec![1]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        Arc::new(b.finish())
    }

    #[test]
    fn readers_never_mix_generations() {
        let mut model = loom::model::Builder::new();
        model.preemption_bound = Some(3);
        model.check(|| {
            let schema = Schema::new(vec![Field::new("a", DataType::Int64, true)]);
            let f = Arc::new(RuntimeFilter::new(
                FilterId(1),
                ColumnPath::new("a"),
                DataType::Int64,
            ));
            let consumer = ScanFilter::new(Arc::clone(&f), &schema).unwrap_or_else(|| panic!());
            let k = keys();
            let producer = {
                let f = Arc::clone(&f);
                thread::spawn(move || {
                    f.publish(FilterSnapshot::Final(k));
                })
            };
            let hint = f.generation_hint();
            let published = f.published();
            assert!(published.generation >= hint, "the hint never runs ahead");
            match (&published.snapshot, published.generation) {
                (FilterSnapshot::Pending, 0) | (FilterSnapshot::Final(_), 1) => {}
                (s, g) => panic!("generation {g} with {s:?}"),
            }
            // A lowered predicate exists only once generation 1 is adopted.
            let predicate = consumer.predicate();
            if predicate.is_some() {
                assert_eq!(consumer.generation(), 1);
            }
            producer
                .join()
                .unwrap_or_else(|_| panic!("producer panicked"));
            assert_eq!(consumer.generation(), 1, "adopted at the next check point");
            assert!(consumer.predicate().is_some());
        });
    }
}
