// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the Parquet row filter and late materialization.

use arrow_array::{
    ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow_schema::{Field, Schema};

use super::*;
use crate::runtime_filter::{KeyFilterBuilder, KeyFilterOptions};

fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        cols.into_iter().map(|(_, a)| a).collect(),
    )
    .unwrap()
}

fn kept(e: &PushdownExpr, b: &RecordBatch) -> Vec<bool> {
    evaluate(&e.to_nnf(), b).iter().collect()
}

/// The number of predicates of `f` (arrow-rs keeps them private).
fn n(f: &RowFilter) -> usize {
    let text = format!("{f:?}");
    text.split_whitespace()
        .find_map(|w| w.parse().ok())
        .unwrap_or_else(|| panic!("{text}"))
}

/// `op` over two ordered values: the reference the comparators match.
fn test<T: PartialOrd>(op: CmpOp, a: &T, b: &T) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::NotEq => a != b,
        CmpOp::Lt => a < b,
        CmpOp::LtEq => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::GtEq => a >= b,
    }
}

fn c(name: &str) -> ColumnPath {
    ColumnPath::new(name)
}

fn cmp(name: &str, op: CmpOp, v: ScalarValue) -> PushdownExpr {
    PushdownExpr::cmp(c(name), op, v)
}

fn ints() -> RecordBatch {
    batch(vec![(
        "a",
        Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(5)])) as ArrayRef,
    )])
}

#[test]
fn comparisons_follow_sql_and_drop_nulls() {
    let b = ints();
    let v = || ScalarValue::Int32(3);
    assert_eq!(
        kept(&cmp("a", CmpOp::Eq, v()), &b),
        [false, false, true, false]
    );
    assert_eq!(
        kept(&cmp("a", CmpOp::NotEq, v()), &b),
        [true, false, false, true]
    );
    assert_eq!(
        kept(&cmp("a", CmpOp::Lt, v()), &b),
        [true, false, false, false]
    );
    assert_eq!(
        kept(&cmp("a", CmpOp::LtEq, v()), &b),
        [true, false, true, false]
    );
    assert_eq!(
        kept(&cmp("a", CmpOp::Gt, v()), &b),
        [false, false, false, true]
    );
    assert_eq!(
        kept(&cmp("a", CmpOp::GtEq, v()), &b),
        [false, false, true, true]
    );
    // NOT (a = 3) keeps neither the match nor the NULL.
    let not = PushdownExpr::Not(Box::new(cmp("a", CmpOp::Eq, v())));
    assert_eq!(kept(&not, &b), [true, false, false, true]);
    assert_eq!(
        kept(&PushdownExpr::IsNull(c("a")), &b),
        [false, true, false, false]
    );
    let not_null = PushdownExpr::Not(Box::new(PushdownExpr::IsNull(c("a"))));
    assert_eq!(kept(&not_null, &b), [true, false, true, true]);
    let within = PushdownExpr::In {
        column: c("a"),
        values: vec![ScalarValue::Int32(1), ScalarValue::Int32(5)],
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&within, &b), [true, false, false, true]);
    // NOT IN: a conjunction of <> after NNF.
    let outside = PushdownExpr::Not(Box::new(within));
    assert_eq!(kept(&outside, &b), [false, false, true, false]);
}

#[test]
fn connectives_are_three_valued() {
    let b = ints();
    let eq = |v| cmp("a", CmpOp::Eq, ScalarValue::Int32(v));
    let or = PushdownExpr::Or(vec![eq(1), eq(5)]);
    assert_eq!(kept(&or, &b), [true, false, false, true]);
    let and = PushdownExpr::And(vec![or.clone(), cmp("a", CmpOp::Gt, ScalarValue::Int32(2))]);
    assert_eq!(kept(&and, &b), [false, false, false, true]);
    // NOT (a = 1 OR a = 5) = a <> 1 AND a <> 5: NULL stays out.
    assert_eq!(
        kept(&PushdownExpr::Not(Box::new(or)), &b),
        [false, false, true, false]
    );
    assert_eq!(kept(&PushdownExpr::Constant(false), &b), [false; 4]);
    assert_eq!(kept(&PushdownExpr::Constant(true), &b), [true; 4]);
}

#[test]
fn what_cannot_be_evaluated_exactly_keeps_every_row() {
    let f = batch(vec![(
        "f",
        Arc::new(Float64Array::from(vec![0.0, -0.0, f64::NAN, 1.0])) as ArrayRef,
    )]);
    // Spark: -0.0 = 0.0 and NaN = NaN; floats are left to the Filter.
    let e = cmp("f", CmpOp::Eq, ScalarValue::Float64(0.0));
    assert_eq!(kept(&e, &f), [true; 4]);
    assert_eq!(kept(&PushdownExpr::Not(Box::new(e)), &f), [true; 4]);
    let b = ints();
    // A literal of another type, a missing or nested column, another
    // collation: unknown, never a dropped row.
    assert_eq!(
        kept(&cmp("a", CmpOp::Eq, ScalarValue::Int64(3)), &b),
        [true; 4]
    );
    assert_eq!(
        kept(&cmp("zz", CmpOp::Eq, ScalarValue::Int32(3)), &b),
        [true; 4]
    );
    let nested = PushdownExpr::cmp(
        ColumnPath::nested(["a", "x"]),
        CmpOp::Eq,
        ScalarValue::Int32(3),
    );
    assert_eq!(kept(&nested, &b), [true; 4]);
    let s = batch(vec![(
        "s",
        Arc::new(StringViewArray::from(vec!["A", "a"])) as ArrayRef,
    )]);
    let lcase = PushdownExpr::Cmp {
        column: c("s"),
        op: CmpOp::Eq,
        value: ScalarValue::Utf8("a".to_owned()),
        collation: Collation::Other("UTF8_LCASE".to_owned()),
    };
    assert_eq!(kept(&lcase, &s), [true, true]);
    assert_eq!(kept(&PushdownExpr::Not(Box::new(lcase)), &s), [true, true]);
    // An unknown conjunct leaves the rest; an unknown disjunct keeps all.
    let and = PushdownExpr::And(vec![
        cmp("zz", CmpOp::Eq, ScalarValue::Int32(3)),
        cmp("a", CmpOp::Eq, ScalarValue::Int32(3)),
    ]);
    assert_eq!(kept(&and, &b), [false, false, true, false]);
    let or = PushdownExpr::Or(vec![
        cmp("zz", CmpOp::Eq, ScalarValue::Int32(3)),
        cmp("a", CmpOp::Eq, ScalarValue::Int32(3)),
    ]);
    assert_eq!(kept(&or, &b), [true; 4]);
}

#[test]
fn every_exact_type_compares_by_value() {
    let b = batch(vec![
        ("i8", Arc::new(Int8Array::from(vec![1, 2])) as ArrayRef),
        ("i16", Arc::new(Int16Array::from(vec![1, 2]))),
        ("i64", Arc::new(Int64Array::from(vec![1, 2]))),
        ("d", Arc::new(Date32Array::from(vec![1, 2]))),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(vec![1, 2]).with_timezone("UTC")),
        ),
        (
            "dec",
            Arc::new(
                Decimal128Array::from(vec![100, 250])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ),
        ("b", Arc::new(BooleanArray::from(vec![false, true]))),
        ("s", Arc::new(StringArray::from(vec!["ab", "b"]))),
        ("v", Arc::new(StringViewArray::from(vec!["ab", "b"]))),
        (
            "bin",
            Arc::new(BinaryArray::from(vec![b"ab".as_ref(), b"b".as_ref()])),
        ),
        ("ls", Arc::new(LargeStringArray::from(vec!["ab", "b"]))),
        (
            "bv",
            Arc::new(BinaryViewArray::from(vec![b"ab".as_ref(), b"b".as_ref()])),
        ),
        (
            "lb",
            Arc::new(LargeBinaryArray::from(vec![b"ab".as_ref(), b"b".as_ref()])),
        ),
    ]);
    let ge = |n: &str, v| kept(&cmp(n, CmpOp::GtEq, v), &b);
    assert_eq!(ge("i8", ScalarValue::Int8(2)), [false, true]);
    assert_eq!(ge("i16", ScalarValue::Int16(2)), [false, true]);
    assert_eq!(ge("i64", ScalarValue::Int64(2)), [false, true]);
    assert_eq!(ge("d", ScalarValue::Date32(2)), [false, true]);
    assert_eq!(ge("ts", ScalarValue::TimestampMicros(2)), [false, true]);
    let dec = |scale| ScalarValue::Decimal128 {
        unscaled: 250,
        precision: 10,
        scale,
    };
    assert_eq!(ge("dec", dec(2)), [false, true]);
    // Another scale is another value: not compared here.
    assert_eq!(ge("dec", dec(3)), [true, true]);
    assert_eq!(ge("b", ScalarValue::Boolean(true)), [false, true]);
    for s in ["s", "v", "ls"] {
        assert_eq!(ge(s, ScalarValue::Utf8("b".to_owned())), [false, true]);
        // Byte order: "ab" < "b".
        assert_eq!(
            kept(&cmp(s, CmpOp::Lt, ScalarValue::Utf8("b".to_owned())), &b),
            [true, false]
        );
    }
    for bin in ["bin", "bv", "lb"] {
        assert_eq!(ge(bin, ScalarValue::Binary(b"b".to_vec())), [false, true]);
    }
    // No nulls: IS NULL keeps nothing; a missing column is unknown.
    assert_eq!(kept(&PushdownExpr::IsNull(c("i8")), &b), [false, false]);
    assert_eq!(kept(&PushdownExpr::IsNull(c("zz")), &b), [true, true]);
    // A prefix of a non-string column is unknown.
    let prefix = PushdownExpr::StartsWith {
        column: c("i8"),
        prefix: "1".to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&prefix, &b), [true, true]);
    // A string literal against binary (and back) is not compared.
    assert_eq!(ge("bin", ScalarValue::Utf8("b".to_owned())), [true, true]);
    assert_eq!(ge("s", ScalarValue::Binary(b"b".to_vec())), [true, true]);
}

#[test]
fn prefixes_and_substrings_match_bytes() {
    let b = batch(vec![(
        "u",
        Arc::new(StringViewArray::from(vec![
            Some("http://google.com"),
            Some("mail.google"),
            Some("bing"),
            None,
        ])) as ArrayRef,
    )]);
    let contains = PushdownExpr::Contains {
        column: c("u"),
        needle: "google".to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&contains, &b), [true, true, false, false]);
    assert_eq!(
        kept(&PushdownExpr::Not(Box::new(contains)), &b),
        [false, false, true, false]
    );
    let prefix = PushdownExpr::StartsWith {
        column: c("u"),
        prefix: "http".to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&prefix, &b), [true, false, false, false]);
    // Not a string column: unknown.
    let on_ints = PushdownExpr::Contains {
        column: c("a"),
        needle: "1".to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&on_ints, &ints()), [true; 4]);
}

#[test]
fn membership_keeps_a_superset_and_never_negates() {
    let mut builder = KeyFilterBuilder::new(DataType::Int64, KeyFilterOptions::default())
        .unwrap_or_else(|| panic!("Int64 keys"));
    builder.add(&Int64Array::from(vec![1, 5]), None).unwrap();
    let member = PushdownExpr::Membership {
        column: c("a"),
        filter: crate::MembershipFilter {
            id: crate::FilterId(1),
            generation: 1,
            keys: Arc::new(builder.finish()),
        },
    };
    let b = ints();
    assert_eq!(kept(&member, &b), [true, false, false, true]);
    // NOT of a superset is not a superset: every row stays.
    assert_eq!(kept(&PushdownExpr::Not(Box::new(member)), &b), [true; 4]);
}

fn key_filter(keys: &[i32], options: KeyFilterOptions) -> Arc<crate::KeyFilter> {
    let mut b =
        KeyFilterBuilder::new(DataType::Int32, options).unwrap_or_else(|| panic!("Int32 keys"));
    b.add(&Int32Array::from(keys.to_vec()), None).unwrap();
    Arc::new(b.finish())
}

fn membership(keys: &Arc<crate::KeyFilter>) -> PushdownExpr {
    PushdownExpr::Membership {
        column: c("a"),
        filter: crate::MembershipFilter {
            id: crate::FilterId(1),
            generation: 1,
            keys: Arc::clone(keys),
        },
    }
}

fn in_a(values: &[i32]) -> PushdownExpr {
    PushdownExpr::In {
        column: c("a"),
        values: values.iter().copied().map(ScalarValue::Int32).collect(),
        collation: Collation::Utf8Binary,
    }
}

#[test]
fn an_in_beside_an_exact_membership_is_not_evaluated() {
    let b = ints();
    let exact = key_filter(&[1, 5], KeyFilterOptions::default());
    assert!(exact.exact_rows());
    // An `In` narrower than the membership shows whether it ran: beside
    // an exact membership on the same column it is skipped.
    let skipped = PushdownExpr::And(vec![in_a(&[1]), membership(&exact)]);
    assert_eq!(kept(&skipped, &b), [true, false, false, true]);
    // Order inside the `And` does not matter.
    let first = PushdownExpr::And(vec![membership(&exact), in_a(&[1])]);
    assert_eq!(kept(&first, &b), [true, false, false, true]);
    // An `In` on another column, or beside a membership that may pass
    // false positives (a Bloom filter), still runs.
    let two = batch(vec![
        (
            "a",
            Arc::new(Int32Array::from(vec![1, 3, 5, 5])) as ArrayRef,
        ),
        (
            "b",
            Arc::new(Int32Array::from(vec![1, 1, 1, 9])) as ArrayRef,
        ),
    ]);
    let other = PushdownExpr::And(vec![
        PushdownExpr::In {
            column: c("b"),
            values: vec![ScalarValue::Int32(1)],
            collation: Collation::Utf8Binary,
        },
        membership(&exact),
    ]);
    assert_eq!(kept(&other, &two), [true, false, true, false]);
    let bloom = key_filter(
        &[1, 5],
        KeyFilterOptions {
            exact_max_values: 0,
            ..KeyFilterOptions::default()
        },
    );
    assert!(!bloom.exact_rows());
    let applied = PushdownExpr::And(vec![in_a(&[1]), membership(&bloom)]);
    assert_eq!(kept(&applied, &b), [true, false, false, false]);
    // Skipping leaves the `And` unknown-false, so a NOT above it (never
    // in negation normal form, but `eval` takes it) keeps every row.
    assert_eq!(
        eval(&PushdownExpr::Not(Box::new(skipped)), &b).may_true,
        None
    );
    // A dense range is exact too; a published empty filter is not.
    assert!(key_filter(&[4, 5, 6], KeyFilterOptions::default()).exact_rows());
    assert!(!key_filter(&[], KeyFilterOptions::default()).exact_rows());
}

#[test]
fn a_lowered_runtime_filter_keeps_the_same_rows_without_its_in() {
    let keys: Vec<i32> = (0..40).map(|k| k * 3).collect();
    let f = key_filter(&keys, KeyFilterOptions::default());
    let lowered = f
        .lower(&c("a"), &DataType::Int32, crate::FilterId(1), 1)
        .unwrap_or_else(|| panic!("lowers"));
    let PushdownExpr::And(parts) = &lowered else {
        panic!("{lowered:?}")
    };
    assert!(parts.iter().any(|p| matches!(p, PushdownExpr::In { .. })));
    let col: Vec<Option<i32>> = (-5..130).map(|v| (v % 7 != 0).then_some(v)).collect();
    let b = batch(vec![(
        "a",
        Arc::new(Int32Array::from(col.clone())) as ArrayRef,
    )]);
    let want: Vec<bool> = col
        .iter()
        .map(|v| v.is_some_and(|v| keys.contains(&v)))
        .collect();
    assert_eq!(kept(&lowered, &b), want);
    // With NULL keys passing (a null-safe join), NULL rows stay.
    let mut nb = KeyFilterBuilder::new(DataType::Int32, KeyFilterOptions::default())
        .unwrap_or_else(|| panic!("Int32 keys"));
    nb.add(&Int32Array::from(vec![Some(3), None, Some(9)]), None)
        .unwrap();
    let mut nulls = nb.finish();
    nulls.nulls_pass = true;
    let lowered = Arc::new(nulls)
        .lower(&c("a"), &DataType::Int32, crate::FilterId(1), 1)
        .unwrap_or_else(|| panic!("lowers"));
    let want: Vec<bool> = col
        .iter()
        .map(|v| v.is_none_or(|v| v == 3 || v == 9))
        .collect();
    assert_eq!(kept(&lowered, &b), want);
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the test values lie in -40..120, which every integer type holds"
)]
fn a_long_in_list_keeps_what_one_pass_per_value_keeps() {
    let long: Vec<i64> = (0..20).map(|v| v * 7 - 30).collect();
    let rows: Vec<Option<i64>> = (-40..120).map(|v| (v % 11 != 0).then_some(v)).collect();
    let arrays: Vec<(ArrayRef, Vec<ScalarValue>)> = vec![
        (
            Arc::new(Int8Array::from_iter(
                rows.iter().map(|v| v.map(|v| v as i8)),
            )),
            long.iter().map(|v| ScalarValue::Int8(*v as i8)).collect(),
        ),
        (
            Arc::new(Int16Array::from_iter(
                rows.iter().map(|v| v.map(|v| v as i16)),
            )),
            long.iter().map(|v| ScalarValue::Int16(*v as i16)).collect(),
        ),
        (
            Arc::new(Int32Array::from_iter(
                rows.iter().map(|v| v.map(|v| v as i32)),
            )),
            long.iter().map(|v| ScalarValue::Int32(*v as i32)).collect(),
        ),
        (
            Arc::new(Int64Array::from(rows.clone())),
            long.iter().map(|v| ScalarValue::Int64(*v)).collect(),
        ),
        (
            Arc::new(Date32Array::from_iter(
                rows.iter().map(|v| v.map(|v| v as i32)),
            )),
            long.iter()
                .map(|v| ScalarValue::Date32(*v as i32))
                .collect(),
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(rows.clone())),
            long.iter()
                .map(|v| ScalarValue::TimestampMicros(*v))
                .collect(),
        ),
        (
            Arc::new(
                Decimal128Array::from_iter(rows.iter().map(|v| v.map(i128::from)))
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            long.iter()
                .map(|v| ScalarValue::Decimal128 {
                    unscaled: i128::from(*v),
                    precision: 10,
                    scale: 2,
                })
                .collect(),
        ),
        (
            Arc::new(StringArray::from_iter(
                rows.iter().map(|v| v.map(|v| v.to_string())),
            )),
            long.iter()
                .map(|v| ScalarValue::Utf8(v.to_string()))
                .collect(),
        ),
        (
            Arc::new(StringViewArray::from_iter(
                rows.iter().map(|v| v.map(|v| v.to_string())),
            )),
            long.iter()
                .map(|v| ScalarValue::Utf8(v.to_string()))
                .collect(),
        ),
        (
            Arc::new(BinaryArray::from_iter(
                rows.iter().map(|v| v.map(|v| v.to_string().into_bytes())),
            )),
            long.iter()
                .map(|v| ScalarValue::Binary(v.to_string().into_bytes()))
                .collect(),
        ),
    ];
    for (a, values) in arrays {
        let ty = a.data_type().clone();
        let b = batch(vec![("a", a)]);
        let e = PushdownExpr::In {
            column: c("a"),
            values: values.clone(),
            collation: Collation::Utf8Binary,
        };
        let col = b.column(0).as_ref();
        let one_pass_each: Vec<bool> = values
            .iter()
            .fold(BooleanBuffer::new_unset(col.len()), |acc, v| {
                &acc | &compare(col, CmpOp::Eq, v).unwrap()
            })
            .iter()
            .zip(
                col.logical_nulls()
                    .map_or(vec![true; col.len()], |n| n.iter().collect()),
            )
            .map(|(t, valid)| t && valid)
            .collect();
        assert!(one_pass_each.iter().any(|k| *k), "{ty}");
        assert_eq!(kept(&e, &b), one_pass_each, "{ty}");
        // NOT IN drops the listed values and NULLs.
        let not: Vec<bool> = col
            .logical_nulls()
            .map_or(vec![true; col.len()], |n| n.iter().collect())
            .into_iter()
            .zip(&one_pass_each)
            .map(|(valid, hit)| valid && !hit)
            .collect();
        assert_eq!(kept(&PushdownExpr::Not(Box::new(e)), &b), not, "{ty}");
    }
    // A value of another type, or a decimal at another scale, leaves
    // the list unevaluated (every row kept), as for a short list.
    let b = ints();
    let mut mixed: Vec<ScalarValue> = (0..10).map(ScalarValue::Int32).collect();
    mixed.push(ScalarValue::Int64(1));
    let e = PushdownExpr::In {
        column: c("a"),
        values: mixed,
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&e, &b), [true; 4]);
    let dec = batch(vec![(
        "a",
        Arc::new(
            Decimal128Array::from(vec![100, 200])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef,
    )]);
    let other_scale = PushdownExpr::In {
        column: c("a"),
        values: (0..10)
            .map(|v| ScalarValue::Decimal128 {
                unscaled: v,
                precision: 10,
                scale: 1,
            })
            .collect(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(kept(&other_scale, &dec), [true; 2]);
}

/// A long list holding one value of another type is not evaluated on any
/// column type (every row kept); a `Decimal64` column looks its values
/// up unless one lies beyond the carrier; a type without a lookup (a
/// double) runs one pass per value, which keeps what `compare` keeps.
#[test]
fn a_long_in_list_with_a_foreign_value_keeps_every_row() {
    let in_list = |values: Vec<ScalarValue>| PushdownExpr::In {
        column: c("a"),
        values,
        collation: Collation::Utf8Binary,
    };
    let with_foreign = |mut values: Vec<ScalarValue>| {
        values.push(match values[0] {
            ScalarValue::Utf8(_) => ScalarValue::Int32(1),
            _ => ScalarValue::Utf8("1".to_owned()),
        });
        values
    };
    let columns: Vec<(ArrayRef, Vec<ScalarValue>)> = vec![
        (
            Arc::new(Int8Array::from(vec![1_i8, 50])),
            (0..10).map(ScalarValue::Int8).collect(),
        ),
        (
            Arc::new(Int16Array::from(vec![1_i16, 50])),
            (0..10).map(ScalarValue::Int16).collect(),
        ),
        (
            Arc::new(Int32Array::from(vec![1, 50])),
            (0..10).map(ScalarValue::Int32).collect(),
        ),
        (
            Arc::new(Int64Array::from(vec![1_i64, 50])),
            (0..10).map(ScalarValue::Int64).collect(),
        ),
        (
            Arc::new(Date32Array::from(vec![1, 50])),
            (0..10).map(ScalarValue::Date32).collect(),
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![1_i64, 50])),
            (0..10).map(ScalarValue::TimestampMicros).collect(),
        ),
        (
            Arc::new(StringArray::from(vec!["1", "50"])),
            (0..10).map(|v| ScalarValue::Utf8(v.to_string())).collect(),
        ),
    ];
    for (a, values) in columns {
        let ty = a.data_type().clone();
        let b = batch(vec![("a", a)]);
        assert_eq!(kept(&in_list(values.clone()), &b), [true, false], "{ty}");
        assert_eq!(kept(&in_list(with_foreign(values)), &b), [true; 2], "{ty}");
    }
    let bin = batch(vec![(
        "a",
        Arc::new(BinaryArray::from(vec![b"1".as_slice(), b"50".as_slice()])) as ArrayRef,
    )]);
    let mut bins: Vec<ScalarValue> = (0..10)
        .map(|v: i32| ScalarValue::Binary(v.to_string().into_bytes()))
        .collect();
    assert_eq!(kept(&in_list(bins.clone()), &bin), [true, false]);
    bins.push(ScalarValue::Int32(1));
    assert_eq!(kept(&in_list(bins), &bin), [true; 2]);

    let d64 = batch(vec![(
        "a",
        Arc::new(
            arrow_array::Decimal64Array::from(vec![100_i64, 5_000])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ) as ArrayRef,
    )]);
    let decimals = |extra: Option<i128>| -> Vec<ScalarValue> {
        (0..10)
            .map(|v| v * 100)
            .chain(extra)
            .map(|unscaled| ScalarValue::Decimal128 {
                unscaled,
                precision: 10,
                scale: 2,
            })
            .collect()
    };
    assert_eq!(kept(&in_list(decimals(None)), &d64), [true, false]);
    let beyond = Some(i128::from(i64::MAX) + 1);
    assert_eq!(kept(&in_list(decimals(beyond)), &d64), [true; 2]);

    let doubles = batch(vec![(
        "a",
        Arc::new(Float64Array::from(vec![1.0, 50.0])) as ArrayRef,
    )]);
    let values = (0..10)
        .map(|v| ScalarValue::Float64(f64::from(v)))
        .collect();
    // `compare` never filters a double (NaN and -0.0), so neither does the list.
    assert_eq!(kept(&in_list(values), &doubles), [true; 2]);
}

#[test]
fn columns_are_the_top_level_leaves() {
    let e = PushdownExpr::And(vec![
        cmp("a", CmpOp::Eq, ScalarValue::Int32(1)),
        PushdownExpr::Not(Box::new(PushdownExpr::IsNull(c("b")))),
        PushdownExpr::Or(vec![
            cmp("a", CmpOp::Lt, ScalarValue::Int32(1)),
            PushdownExpr::cmp(
                ColumnPath::nested(["s", "x"]),
                CmpOp::Eq,
                ScalarValue::Int32(1),
            ),
        ]),
        PushdownExpr::Constant(true),
        PushdownExpr::IsNotNull(c("d")),
        PushdownExpr::StartsWith {
            column: c("e"),
            prefix: "x".to_owned(),
            collation: Collation::Utf8Binary,
        },
        PushdownExpr::Contains {
            column: c("f"),
            needle: "x".to_owned(),
            collation: Collation::Utf8Binary,
        },
    ]);
    assert_eq!(columns(&e), [c("a"), c("b"), c("d"), c("e"), c("f")]);
}

#[test]
fn the_policy_follows_the_measured_fraction() {
    let mut late = LateMaterialization::new(
        crate::request::RowFilterPolicy::Auto,
        Some(&cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
        None,
        true,
        None,
    );
    let all = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Yes);
    // Nothing measured: the first row group filters.
    assert!(late.is_on());
    assert_eq!(n(&late.filter(true, &[], &[], all)), 1);
    // A quarter kept: off for the next row group.
    late.counters.add(4, 1);
    assert!(!late.want());
    assert_eq!(n(&late.filter(false, &[], &[], all)), 0);
    assert!(!late.is_on());
    // Off: the first batch of a row group is measured, then no other.
    late.start_row_group();
    assert!(late.sampling());
    let many = batch(vec![(
        "a",
        Arc::new(Int32Array::from_iter_values((0..1000).map(|i| i % 500 + 4))) as ArrayRef,
    )]);
    late.observe(&many, &[]);
    assert!(!late.sampling());
    late.observe(&ints(), &[]);
    // 0 of 1000 kept (the second batch was not counted): on again.
    assert!(late.want());
    // Too little outside the predicate: off for good unless forced.
    let small = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Small);
    let mut auto = LateMaterialization::new(
        crate::request::RowFilterPolicy::Auto,
        Some(&cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
        None,
        true,
        None,
    );
    assert_eq!(n(&auto.filter(true, &[], &[], small)), 0);
    assert!(!auto.want());
    let mut forced = LateMaterialization::new(
        crate::request::RowFilterPolicy::On,
        Some(&cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
        None,
        true,
        None,
    );
    assert_eq!(n(&forced.filter(true, &[], &[], small)), 1);
    assert!(forced.want());
    // No projected column outside the predicate: nothing to save, ever.
    let only = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Never);
    assert_eq!(n(&late.filter(true, &[], &[], only)), 0);
    assert!(!late.is_on());
    assert!(!late.want());
}

#[test]
fn statistics_estimate_ranges_only() {
    use crate::stats::{ColumnStatistics, Precision};
    let mut stats = FileStatistics::with_rows(Precision::Exact(100));
    let mut a = ColumnStatistics::new(c("a"));
    a.min = Precision::Exact(ScalarValue::Int32(0));
    a.max = Precision::Exact(ScalarValue::Int32(100));
    let mut d = ColumnStatistics::new(c("d"));
    d.min = Precision::Inexact(ScalarValue::Decimal128 {
        unscaled: 0,
        precision: 10,
        scale: 2,
    });
    d.max = Precision::Exact(ScalarValue::Decimal128 {
        unscaled: 1000,
        precision: 10,
        scale: 2,
    });
    stats.columns = vec![a, d];
    let est = |e: &PushdownExpr| estimate(&e.to_nnf(), &stats);
    let lt = |v| cmp("a", CmpOp::Lt, ScalarValue::Int32(v));
    assert_eq!(est(&lt(25)), Some(0.25));
    assert_eq!(
        est(&cmp("a", CmpOp::GtEq, ScalarValue::Int32(25))),
        Some(0.75)
    );
    assert_eq!(est(&lt(-5)), Some(0.0));
    assert_eq!(est(&lt(500)), Some(1.0));
    let dec = cmp(
        "d",
        CmpOp::LtEq,
        ScalarValue::Decimal128 {
            unscaled: 250,
            precision: 10,
            scale: 2,
        },
    );
    assert_eq!(est(&dec), Some(0.25));
    // Conjuncts multiply; unknown ones are left out; disjuncts add.
    let eq = cmp("a", CmpOp::Eq, ScalarValue::Int32(3));
    assert_eq!(
        est(&PushdownExpr::And(vec![lt(50), dec.clone(), eq.clone()])),
        Some(0.125)
    );
    assert_eq!(est(&PushdownExpr::Or(vec![lt(50), dec])), Some(0.75));
    assert_eq!(est(&PushdownExpr::Or(vec![lt(90), lt(90)])), Some(1.0));
    assert_eq!(est(&PushdownExpr::Or(vec![lt(50), eq.clone()])), None);
    assert_eq!(est(&eq), None);
    assert_eq!(est(&PushdownExpr::Constant(false)), Some(0.0));
    assert_eq!(est(&cmp("zz", CmpOp::Lt, ScalarValue::Int32(1))), None);
    assert_eq!(
        est(&cmp("a", CmpOp::Lt, ScalarValue::Utf8("x".into()))),
        None
    );
    let mut flat = ColumnStatistics::new(c("f"));
    flat.min = Precision::Exact(ScalarValue::Int64(7));
    flat.max = Precision::Exact(ScalarValue::Int64(7));
    stats.columns.push(flat);
    let f = cmp("f", CmpOp::Lt, ScalarValue::Int64(9));
    assert_eq!(estimate(&f, &stats), None);
    let mut small = ColumnStatistics::new(c("s"));
    small.min = Precision::Exact(ScalarValue::Int8(0));
    small.max = Precision::Exact(ScalarValue::Int8(10));
    let mut short = ColumnStatistics::new(c("t"));
    short.min = Precision::Exact(ScalarValue::Int16(0));
    short.max = Precision::Exact(ScalarValue::Int16(10));
    stats.columns.extend([small, short]);
    let s = cmp("s", CmpOp::Gt, ScalarValue::Int8(8));
    let t = cmp("t", CmpOp::LtEq, ScalarValue::Int16(1));
    let close = |x: Option<f64>, want: f64| x.is_some_and(|x| (x - want).abs() < 1e-9);
    assert!(close(estimate(&s, &stats), 0.2));
    assert!(close(estimate(&t, &stats), 0.1));
    // An estimate above 1% starts without the filter (and measures).
    let e = lt(50);
    let auto = crate::request::RowFilterPolicy::Auto;
    let mut late = LateMaterialization::new(auto, Some(&e), None, true, Some(0.5));
    assert!(!late.is_on());
    late.start_row_group();
    assert!(late.sampling());
    assert!(LateMaterialization::new(auto, Some(&e), None, true, Some(0.005)).is_on());
    let on = crate::request::RowFilterPolicy::On;
    assert!(LateMaterialization::new(on, Some(&e), None, true, Some(0.5)).is_on());
}

#[test]
fn runtime_filters_join_the_static_predicate() {
    let mut b = KeyFilterBuilder::new(DataType::Int32, KeyFilterOptions::default())
        .unwrap_or_else(|| panic!("Int32 keys"));
    b.add(&Int32Array::from(vec![5]), None).unwrap();
    let member = PushdownExpr::Membership {
        column: c("a"),
        filter: crate::MembershipFilter {
            id: crate::FilterId(1),
            generation: 1,
            keys: Arc::new(b.finish()),
        },
    };
    let e = cmp("a", CmpOp::Gt, ScalarValue::Int32(2));
    let auto = crate::request::RowFilterPolicy::Auto;
    let mut late = LateMaterialization::new(auto, Some(&e), None, true, None);
    let all = |cols: &[ColumnPath]| {
        assert_eq!(cols, [c("a")]);
        (ProjectionMask::all(), Worth::Yes)
    };
    assert_eq!(
        n(&late.filter(true, std::slice::from_ref(&member), &[], all)),
        1
    );
    // Off: a sample evaluates both (a > 2 AND a in {5}).
    late.on = false;
    late.start_row_group();
    late.observe(&ints(), &[member]);
    assert_eq!(late.counters.take(), (4, 1));
}

#[test]
fn a_sample_without_the_predicate_column_wants_filtering() {
    let auto = crate::request::RowFilterPolicy::Auto;
    let e = cmp("a", CmpOp::Eq, ScalarValue::Int32(3));
    // An estimate above 1% starts without the filter.
    let mut late = LateMaterialization::new(auto, Some(&e), None, true, Some(0.5));
    assert!(!late.is_on());
    late.start_row_group();
    assert!(late.sampling());
    // The sampled batch holds only the output column: nothing was
    // measured, so the next row group filters through its own
    // predicate projection instead of recording a 1.0 kept fraction.
    let output_only = batch(vec![(
        "b",
        Arc::new(Int32Array::from_iter_values(0..1000)) as ArrayRef,
    )]);
    late.observe(&output_only, &[]);
    assert_eq!(late.counters.take(), (0, 0));
    assert!(late.want());
}

#[test]
fn a_scan_without_a_predicate_stays_idle_until_a_runtime_filter_arrives() {
    let mut b = KeyFilterBuilder::new(DataType::Int32, KeyFilterOptions::default())
        .unwrap_or_else(|| panic!("Int32 keys"));
    b.add(&Int32Array::from(vec![5]), None).unwrap();
    let member = PushdownExpr::Membership {
        column: c("a"),
        filter: crate::MembershipFilter {
            id: crate::FilterId(1),
            generation: 1,
            keys: Arc::new(b.finish()),
        },
    };
    let auto = crate::request::RowFilterPolicy::Auto;
    let mut late = LateMaterialization::new(auto, None, None, true, None);
    let all = |cols: &[ColumnPath]| {
        assert_eq!(cols, [c("a")]);
        (ProjectionMask::all(), Worth::Yes)
    };
    // The first guess is "on", but there is no predicate: the filter is
    // empty, and no boundary asks for a rebuild or a sample.
    assert!(late.is_on());
    assert_eq!(n(&late.filter(true, &[], &[], all)), 0);
    for _ in 0..3 {
        assert!(!late.want(), "nothing to rebuild for");
        late.start_row_group();
        assert!(!late.sampling(), "nothing to measure");
    }
    // A runtime filter is the first predicate: filtering is wanted again.
    late.runtime_filters_changed(std::slice::from_ref(&member));
    assert!(late.want());
    assert_eq!(
        n(&late.filter(true, std::slice::from_ref(&member), &[], all)),
        1
    );
    assert!(late.is_on());
}

#[test]
fn a_column_that_cannot_be_widened_keeps_every_row() {
    let rows = || {
        batch(vec![(
            "a",
            Arc::new(Int32Array::from_iter_values(0..1000)) as ArrayRef,
        )])
    };
    let predicate = |widen: Option<Widen>| {
        ScanPredicate::new(
            Arc::new(cmp("a", CmpOp::Eq, ScalarValue::Int32(3))),
            ProjectionMask::all(),
            widen,
            Arc::new(Counters::default()),
        )
    };
    let mut failing = predicate(Some(|_: &ArrayRef| None));
    let kept = ArrowPredicate::evaluate(&mut failing, rows()).unwrap();
    assert_eq!(kept.true_count(), 1000);
    let mut plain = predicate(None);
    let kept = ArrowPredicate::evaluate(&mut plain, rows()).unwrap();
    assert_eq!(kept.true_count(), 1);
    assert_eq!(plain.counters.take(), (1000, 1));
}

#[test]
fn forced_and_ineligible_policies() {
    let e = cmp("a", CmpOp::Eq, ScalarValue::Int32(3));
    let mut on = LateMaterialization::new(
        crate::request::RowFilterPolicy::On,
        Some(&e),
        None,
        true,
        None,
    );
    on.counters.add(10, 10);
    assert!(on.want());
    let mut off = LateMaterialization::new(
        crate::request::RowFilterPolicy::Off,
        Some(&e),
        None,
        true,
        None,
    );
    assert!(!off.is_on());
    assert!(!off.want());
    off.start_row_group();
    assert!(!off.sampling());
    let mut row_index = LateMaterialization::new(
        crate::request::RowFilterPolicy::On,
        Some(&e),
        None,
        false,
        None,
    );
    assert!(!row_index.want());
    let all = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Yes);
    assert_eq!(n(&row_index.filter(true, &[], &[], all)), 0);
    // A bound goes in a pass of its own, if its column is small enough.
    let bound = PushdownExpr::cmp(c("k"), CmpOp::LtEq, ScalarValue::Int32(0));
    let mut both = LateMaterialization::new(
        crate::request::RowFilterPolicy::On,
        Some(&e),
        None,
        true,
        None,
    );
    assert_eq!(
        n(&both.filter(true, &[], std::slice::from_ref(&bound), all)),
        2
    );
    assert!(both.filters_first());
    let small = |_: &[ColumnPath]| (ProjectionMask::all(), Worth::Small);
    let mut no_pass = LateMaterialization::new(
        crate::request::RowFilterPolicy::Auto,
        None,
        None,
        true,
        None,
    );
    assert_eq!(
        n(&no_pass.filter(true, &[], std::slice::from_ref(&bound), small)),
        0
    );
    assert!(!no_pass.filters_first());
    // ... but never where no row may be dropped in the decoder.
    assert_eq!(
        n(&row_index.filter(true, &[], std::slice::from_ref(&bound), all)),
        0
    );
    // No predicate at all: nothing to evaluate.
    let mut none =
        LateMaterialization::new(crate::request::RowFilterPolicy::On, None, None, true, None);
    assert_eq!(n(&none.filter(true, &[], &[], all)), 0);
}

/// A column name, its values and the literal of a value of its type.
type TypedColumn = (&'static str, ArrayRef, fn(i64) -> ScalarValue);

/// Every operator over every exactly compared type, against the operator
/// applied to each value on its own (the comparator is chosen once per
/// batch, outside the loop over rows).
#[test]
fn every_operator_matches_its_scalar_meaning_on_every_type() {
    let ops = [
        CmpOp::Eq,
        CmpOp::NotEq,
        CmpOp::Lt,
        CmpOp::LtEq,
        CmpOp::Gt,
        CmpOp::GtEq,
    ];
    let ints: Vec<Option<i64>> = vec![Some(-2), Some(0), None, Some(1), Some(3), Some(1)];
    let want = |op: CmpOp, lit: i64| -> Vec<bool> {
        ints.iter()
            .map(|v| v.is_some_and(|v| test(op, &v, &lit)))
            .collect()
    };
    #[expect(clippy::cast_possible_truncation, reason = "values lie in -2..=3")]
    let cols: Vec<TypedColumn> = vec![
        (
            "i8",
            Arc::new(Int8Array::from_iter(
                ints.iter().map(|v| v.map(|v| v as i8)),
            )),
            |v| ScalarValue::Int8(v as i8),
        ),
        (
            "i16",
            Arc::new(Int16Array::from_iter(
                ints.iter().map(|v| v.map(|v| v as i16)),
            )),
            |v| ScalarValue::Int16(v as i16),
        ),
        (
            "i32",
            Arc::new(Int32Array::from_iter(
                ints.iter().map(|v| v.map(|v| v as i32)),
            )),
            |v| ScalarValue::Int32(v as i32),
        ),
        (
            "i64",
            Arc::new(Int64Array::from(ints.clone())),
            ScalarValue::Int64,
        ),
        (
            "d",
            Arc::new(Date32Array::from_iter(
                ints.iter().map(|v| v.map(|v| v as i32)),
            )),
            |v| ScalarValue::Date32(v as i32),
        ),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(ints.clone())),
            ScalarValue::TimestampMicros,
        ),
        (
            "dec",
            Arc::new(
                Decimal128Array::from_iter(ints.iter().map(|v| v.map(i128::from)))
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            |v| ScalarValue::Decimal128 {
                unscaled: i128::from(v),
                precision: 10,
                scale: 2,
            },
        ),
    ];
    let b = batch(cols.iter().map(|(n, a, _)| (*n, Arc::clone(a))).collect());
    for (name, _, lit) in &cols {
        for op in ops {
            for v in [-3, -2, 0, 1, 3, 4] {
                assert_eq!(
                    kept(&cmp(name, op, lit(v)), &b),
                    want(op, v),
                    "{name} {op:?} {v}"
                );
            }
        }
    }
    // Booleans and byte strings, every operator.
    let flags = [Some(false), Some(true), None];
    let words = [Some("ab"), Some("b"), None, Some(""), Some("ba")];
    let b = batch(vec![(
        "b",
        Arc::new(BooleanArray::from(flags.to_vec())) as ArrayRef,
    )]);
    let s = batch(vec![
        ("s", Arc::new(StringArray::from(words.to_vec())) as ArrayRef),
        ("v", Arc::new(StringViewArray::from(words.to_vec()))),
        (
            "bin",
            Arc::new(BinaryArray::from(
                words
                    .iter()
                    .map(|w| w.map(str::as_bytes))
                    .collect::<Vec<_>>(),
            )),
        ),
    ]);
    for op in ops {
        for lit in [false, true] {
            let want: Vec<bool> = flags
                .iter()
                .map(|v| v.is_some_and(|v| test(op, &v, &lit)))
                .collect();
            assert_eq!(kept(&cmp("b", op, ScalarValue::Boolean(lit)), &b), want);
        }
        for lit in ["", "ab", "b", "c"] {
            let want: Vec<bool> = words
                .iter()
                .map(|v| v.is_some_and(|v| test(op, &v.as_bytes(), &lit.as_bytes())))
                .collect();
            for col in ["s", "v"] {
                assert_eq!(
                    kept(&cmp(col, op, ScalarValue::Utf8(lit.to_owned())), &s),
                    want,
                    "{col} {op:?} {lit:?}"
                );
            }
            assert_eq!(
                kept(
                    &cmp("bin", op, ScalarValue::Binary(lit.as_bytes().to_vec())),
                    &s
                ),
                want
            );
        }
    }
}

/// A pseudo-random sequence for the tests below (no seeding crate needed).
fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed;
    move || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        x >> 33
    }
}

/// An `And` keeps exactly the rows every conjunct keeps, whether its cheap
/// comparisons leave many, few or no rows for the later conjuncts (which
/// run on the surviving rows only), with runtime filters nested as their
/// lowering builds them (an `And`, or an `Or` with `IS NULL`).
#[test]
fn an_and_keeps_what_its_conjuncts_keep_however_few_rows_survive() {
    let mut next = lcg(7);
    let n = 4_000;
    let a: Vec<Option<i32>> = (0..n)
        .map(|_| (!next().is_multiple_of(17)).then(|| (next() % 1_000) as i32))
        .collect();
    let k: Vec<Option<i32>> = (0..n)
        .map(|_| (!next().is_multiple_of(11)).then(|| (next() % 300) as i32))
        .collect();
    let s: Vec<Option<String>> = (0..n)
        .map(|_| (!next().is_multiple_of(13)).then(|| format!("w{}", next() % 50)))
        .collect();
    let b = batch(vec![
        ("a", Arc::new(Int32Array::from(a)) as ArrayRef),
        ("k", Arc::new(Int32Array::from(k))),
        ("s", Arc::new(StringArray::from(s))),
    ]);
    let keys: Vec<i32> = (0..300).filter(|v| v % 7 == 1).collect();
    let exact = key_filter(&keys, KeyFilterOptions::default());
    let bloom = key_filter(
        &keys,
        KeyFilterOptions {
            exact_max_values: 0,
            ..KeyFilterOptions::default()
        },
    );
    let on_k = |f: &Arc<crate::KeyFilter>| {
        f.lower(&c("k"), &DataType::Int32, crate::FilterId(1), 1)
            .unwrap_or_else(|| panic!("lowers"))
    };
    let mut nulls = KeyFilterBuilder::new(DataType::Int32, KeyFilterOptions::default())
        .unwrap_or_else(|| panic!("Int32 keys"));
    nulls
        .add(&Int32Array::from(vec![Some(8), None, Some(15)]), None)
        .unwrap();
    let mut nulls = nulls.finish();
    nulls.nulls_pass = true;
    let null_safe = Arc::new(nulls)
        .lower(&c("k"), &DataType::Int32, crate::FilterId(2), 1)
        .unwrap_or_else(|| panic!("lowers"));
    let words = PushdownExpr::In {
        column: c("s"),
        values: (0..20)
            .map(|i| ScalarValue::Utf8(format!("w{i}")))
            .collect(),
        collation: Collation::Utf8Binary,
    };
    let prefix = PushdownExpr::StartsWith {
        column: c("s"),
        prefix: "w1".to_owned(),
        collation: Collation::Utf8Binary,
    };
    // Ranges on `a` keeping about 100%, 30%, 2%, 0.1% and none of the rows.
    for (lo, hi) in [(0, 999), (100, 400), (500, 519), (7, 7), (2_000, 3_000)] {
        let range = || {
            vec![
                cmp("a", CmpOp::GtEq, ScalarValue::Int32(lo)),
                cmp("a", CmpOp::LtEq, ScalarValue::Int32(hi)),
            ]
        };
        let rest = [
            on_k(&exact),
            on_k(&bloom),
            null_safe.clone(),
            words.clone(),
            prefix.clone(),
            PushdownExpr::Or(vec![prefix.clone(), PushdownExpr::IsNull(c("s"))]),
        ];
        for extra in 0..rest.len() {
            for second in [None, Some((extra + 2) % rest.len())] {
                let mut xs = vec![rest[extra].clone()];
                xs.extend(second.map(|i| rest[i].clone()));
                // The range written after the others, and nested in an `And`.
                xs.push(PushdownExpr::And(range()));
                let e = PushdownExpr::And(xs.clone());
                let mut want = vec![true; n];
                for x in xs.iter().flat_map(|x| match x {
                    PushdownExpr::And(ys) => ys.clone(),
                    x => vec![x.clone()],
                }) {
                    for (w, k) in want.iter_mut().zip(kept(&x, &b)) {
                        *w &= k;
                    }
                }
                assert_eq!(kept(&e, &b), want, "range {lo}..={hi}, {xs:?}");
            }
        }
    }
}

/// The later conjuncts of an `And` that nothing survives are not looked at:
/// a column they need may be missing and the result is still exact.
#[test]
fn an_and_whose_comparisons_drop_every_row_is_false_on_every_row() {
    let b = ints();
    let e = PushdownExpr::And(vec![
        PushdownExpr::StartsWith {
            column: c("zz"),
            prefix: "x".to_owned(),
            collation: Collation::Utf8Binary,
        },
        cmp("a", CmpOp::Gt, ScalarValue::Int32(9)),
    ]);
    assert_eq!(kept(&e, &b), [false; 4]);
    // NOT of it (never in negation normal form) keeps every row: the
    // unexamined conjunct may be false anywhere.
    assert_eq!(eval(&PushdownExpr::Not(Box::new(e)), &b).may_true, None);
}
