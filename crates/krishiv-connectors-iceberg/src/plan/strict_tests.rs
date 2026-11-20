// SPDX-License-Identifier: Apache-2.0
//! Tests of the overwrite row filter (`strict.rs`, PR-36b part 2b): the
//! strict metrics evaluator per predicate against hand-built metrics (the
//! rules of `StrictEvalVisitor`, notes 36b2 §4: nulls, NaN counts and
//! bounds, missing stats, nested columns), the strict partition projection
//! per transform (identity, bucket, truncate of ints and strings, day with
//! the pre-1970 fix), the residual combining both, and the Spark → Iceberg
//! translation (null guards).

use std::collections::BTreeMap;

use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue as S};

use super::bounds::encode;
use super::entry::{ContentFile, PartitionValue};
use super::strict::{Leaf, Pred, RowFilter};
use super::tests::file;
use super::transform::apply;
use super::types::{
    IcebergType as T, PartitionFieldInfo, PartitionSpecInfo, PartitionSpecs, SchemaField, Transform,
};

fn col(name: &str) -> ColumnPath {
    ColumnPath::new(name)
}

fn fields() -> Vec<SchemaField> {
    vec![
        SchemaField {
            id: 1,
            path: col("id"),
            ty: T::Long,
        },
        SchemaField {
            id: 2,
            path: col("f"),
            ty: T::Double,
        },
        SchemaField {
            id: 3,
            path: col("s"),
            ty: T::String,
        },
        SchemaField {
            id: 4,
            path: ColumnPath(vec!["st".to_owned(), "n".to_owned()]),
            ty: T::Long,
        },
        SchemaField {
            id: 5,
            path: col("d"),
            ty: T::Date,
        },
    ]
}

fn unpartitioned() -> PartitionSpecs {
    [(0, PartitionSpecInfo::default())].into()
}

/// A file of `rows` rows: column `id` (1) in `[lo, hi]` with `nulls` nulls,
/// `f` (2) in `[1.0, 2.0]` with `nans` NaNs, `s` (3) in `["abc", "abd"]`, the
/// nested `st.n` (4) in `[5, 5]`.
fn stats(rows: u64, lo: i64, hi: i64, nulls: u64, nans: u64) -> ContentFile {
    let mut f = file(rows);
    let put = |f: &mut ContentFile, id: i32, l: &S, u: &S, ty: &T| {
        f.lower_bounds.insert(id, encode(l, ty).unwrap());
        f.upper_bounds.insert(id, encode(u, ty).unwrap());
        f.value_counts.insert(id, rows);
    };
    put(&mut f, 1, &S::Int64(lo), &S::Int64(hi), &T::Long);
    put(&mut f, 2, &S::Float64(1.0), &S::Float64(2.0), &T::Double);
    let (a, b) = (S::Utf8("abc".to_owned()), S::Utf8("abd".to_owned()));
    put(&mut f, 3, &a, &b, &T::String);
    put(&mut f, 4, &S::Int64(5), &S::Int64(5), &T::Long);
    f.null_counts = BTreeMap::from([(1, nulls), (2, 0), (3, 0), (4, 0)]);
    f.nan_counts = BTreeMap::from([(2, nans)]);
    f
}

fn leaf(c: &str, l: Leaf) -> Pred {
    Pred::Leaf(col(c), l)
}

fn must(p: Pred, f: &ContentFile) -> bool {
    let (fields, specs) = (fields(), unpartitioned());
    RowFilter::new(p, &fields, &specs).rows_must_match(f)
}

fn might(p: Pred, f: &ContentFile) -> bool {
    let (fields, specs) = (fields(), unpartitioned());
    RowFilter::new(p, &fields, &specs).rows_might_match(f)
}

#[test]
fn an_empty_file_must_match_and_cannot() {
    let f = file(0);
    assert!(must(leaf("id", Leaf::Eq(S::Int64(7))), &f));
    assert!(!might(Pred::Const(true), &f));
    assert!(must(Pred::Const(true), &file(3)));
    assert!(!must(Pred::Const(false), &file(3)));
    assert!(!must(Pred::Unknown, &file(3)));
}

/// Something no evaluator answers, or a column the schema lacks, may match
/// (review 36b2b B1): the file is then never kept silently, it fails with
/// the partial-match error unless proven to match whole.
#[test]
fn unknowns_and_unresolved_columns_might_match() {
    let f = stats(10, 30, 79, 0, 0);
    assert!(might(Pred::Unknown, &f));
    assert!(!must(Pred::Unknown, &f));
    let gone = leaf("gone", Leaf::Eq(S::Int64(1)));
    assert!(might(gone.clone(), &f));
    assert!(!must(gone, &f));
    // Under `and`, a leaf ruled out still rules the file out.
    let ruled_out = Pred::And(vec![leaf("id", Leaf::Lt(S::Int64(0))), Pred::Unknown]);
    assert!(!might(ruled_out, &f));
    let maybe = Pred::Or(vec![leaf("id", Leaf::Lt(S::Int64(0))), Pred::Unknown]);
    assert!(might(maybe, &f));
}

/// The leaves the strict side cannot evaluate, which an overwrite refuses
/// up front (`Pred::unsupported`).
#[test]
fn unsupported_parts_are_named() {
    let fields = fields();
    let contains = Pred::from_spark(&PushdownExpr::Contains {
        column: col("s"),
        needle: "a".to_owned(),
        collation: Collation::Utf8Binary,
    });
    assert!(contains.unsupported(&fields).is_some());
    let lt_nan = Pred::from_spark(&PushdownExpr::cmp(
        col("f"),
        CmpOp::Lt,
        S::Float64(f64::NAN),
    ));
    assert!(lt_nan.unsupported(&fields).is_some());
    let struct_is_null = Pred::from_spark(&PushdownExpr::IsNull(col("st")));
    assert!(struct_is_null.unsupported(&fields).is_some());
    let fine = Pred::from_spark(&PushdownExpr::And(vec![
        PushdownExpr::cmp(col("id"), CmpOp::Lt, S::Int64(3)),
        PushdownExpr::cmp(col("f"), CmpOp::Eq, S::Float64(f64::NAN)),
        PushdownExpr::IsNull(ColumnPath(vec!["st".to_owned(), "n".to_owned()])),
    ]));
    assert_eq!(fine.unsupported(&fields), None);
}

#[test]
fn comparisons_need_bounds_and_no_nulls() {
    let f = stats(10, 30, 79, 0, 0);
    let id = |l| leaf("id", l);
    assert!(must(id(Leaf::Lt(S::Int64(80))), &f));
    assert!(!must(id(Leaf::Lt(S::Int64(79))), &f));
    assert!(must(id(Leaf::LtEq(S::Int64(79))), &f));
    assert!(!must(id(Leaf::LtEq(S::Int64(78))), &f));
    assert!(must(id(Leaf::Gt(S::Int64(29))), &f));
    assert!(!must(id(Leaf::Gt(S::Int64(30))), &f));
    assert!(must(id(Leaf::GtEq(S::Int64(30))), &f));
    assert!(!must(id(Leaf::GtEq(S::Int64(31))), &f));
    assert!(!must(id(Leaf::Eq(S::Int64(30))), &f));
    assert!(must(id(Leaf::NotEq(S::Int64(29))), &f));
    assert!(must(id(Leaf::NotEq(S::Int64(80))), &f));
    assert!(!must(id(Leaf::NotEq(S::Int64(50))), &f));
    // A null in the column refuses every comparison (`canContainNulls`).
    let n = stats(10, 30, 79, 1, 0);
    assert!(!must(id(Leaf::Lt(S::Int64(80))), &n));
    assert!(!must(id(Leaf::GtEq(S::Int64(30))), &n));
    // A null-count map without the column may hide nulls (Java's
    // `canContainNulls` says no; a divergence, review 36b2b B2).
    let mut m = stats(10, 30, 79, 0, 0);
    m.null_counts.remove(&1);
    assert!(!must(id(Leaf::Lt(S::Int64(80))), &m));
    // No bounds: never.
    let mut b = stats(10, 30, 79, 0, 0);
    b.lower_bounds.clear();
    b.upper_bounds.clear();
    assert!(!must(id(Leaf::Lt(S::Int64(80))), &b));
    assert!(!must(id(Leaf::Gt(S::Int64(0))), &b));
    // A literal of another type proves nothing.
    assert!(!must(id(Leaf::Lt(S::Int32(80))), &f));
}

#[test]
fn eq_and_in_need_a_single_value() {
    let one = stats(4, 7, 7, 0, 0);
    let id = |l| leaf("id", l);
    assert!(must(id(Leaf::Eq(S::Int64(7))), &one));
    assert!(!must(id(Leaf::Eq(S::Int64(8))), &one));
    assert!(must(id(Leaf::In(vec![S::Int64(1), S::Int64(7)])), &one));
    assert!(!must(id(Leaf::In(vec![S::Int64(1)])), &one));
    let two = stats(4, 7, 8, 0, 0);
    assert!(!must(id(Leaf::In(vec![S::Int64(7), S::Int64(8)])), &two));
    assert!(!must(id(Leaf::Eq(S::Int64(7))), &stats(4, 7, 7, 1, 0)));
}

#[test]
fn not_in_needs_every_value_outside_the_bounds() {
    let f = stats(10, 30, 79, 0, 0);
    let id = |l| leaf("id", l);
    assert!(must(id(Leaf::NotIn(vec![S::Int64(1), S::Int64(80)])), &f));
    assert!(!must(id(Leaf::NotIn(vec![S::Int64(1), S::Int64(50)])), &f));
    // One bound known: the other is unbounded.
    let mut lo = f.clone();
    lo.upper_bounds.clear();
    assert!(must(id(Leaf::NotIn(vec![S::Int64(1)])), &lo));
    assert!(!must(id(Leaf::NotIn(vec![S::Int64(100)])), &lo));
    let mut none = f;
    none.lower_bounds.clear();
    none.upper_bounds.clear();
    assert!(!must(id(Leaf::NotIn(vec![S::Int64(1)])), &none));
}

#[test]
fn null_predicates() {
    let id = |l| leaf("id", l);
    let all_null = stats(5, 0, 0, 5, 0);
    assert!(must(id(Leaf::IsNull), &all_null));
    assert!(must(id(Leaf::NotEq(S::Int64(3))), &all_null));
    assert!(must(id(Leaf::NotIn(vec![S::Int64(3)])), &all_null));
    assert!(!must(id(Leaf::NotNull), &all_null));
    let some = stats(5, 0, 9, 2, 0);
    assert!(!must(id(Leaf::IsNull), &some));
    assert!(!must(id(Leaf::NotNull), &some));
    let none = stats(5, 0, 9, 0, 0);
    assert!(must(id(Leaf::NotNull), &none));
    // Unknown null counts: `mayContainNull`.
    let mut unknown = none.clone();
    unknown.null_counts.clear();
    assert!(!must(id(Leaf::NotNull), &unknown));
    assert!(!must(id(Leaf::IsNull), &unknown));
    // A null-count map that lacks the column may hide null rows, which no
    // comparison holds for (a divergence from Java's `canContainNulls`,
    // review 36b2b B2).
    let mut lacking = none;
    lacking.null_counts.remove(&1);
    assert!(!lacking.null_counts.is_empty());
    assert!(!must(id(Leaf::Lt(S::Int64(100))), &lacking));
    assert!(!must(id(Leaf::Gt(S::Int64(-100))), &lacking));
}

#[test]
fn nan_rules() {
    let f = |l| leaf("f", l);
    let some = stats(4, 0, 0, 0, 1);
    assert!(!must(f(Leaf::Lt(S::Float64(3.0))), &some));
    assert!(!must(f(Leaf::NotNaN), &some));
    assert!(!must(f(Leaf::IsNaN), &some));
    let none = stats(4, 0, 0, 0, 0);
    assert!(must(f(Leaf::Lt(S::Float64(3.0))), &none));
    assert!(must(f(Leaf::NotNaN), &none));
    let all = stats(4, 0, 0, 0, 4);
    assert!(must(f(Leaf::IsNaN), &all));
    assert!(must(f(Leaf::NotEq(S::Float64(1.5))), &all));
    // Unknown NaN counts (a deliberate divergence from Java's
    // `canContainNaNs`, review 36b2b B2): a NaN row is FALSE under `<`,
    // `<=`, `=` and `IN`, so none of them is proven; `>` and `>=` hold for
    // NaN in Spark and stay proven by the lower bound.
    let mut unknown = none.clone();
    unknown.nan_counts.clear();
    assert!(!must(f(Leaf::Lt(S::Float64(3.0))), &unknown));
    assert!(!must(f(Leaf::LtEq(S::Float64(2.0))), &unknown));
    assert!(!must(f(Leaf::NotNaN), &unknown));
    assert!(must(f(Leaf::Gt(S::Float64(0.5))), &unknown));
    assert!(must(f(Leaf::GtEq(S::Float64(1.0))), &unknown));
    let mut single = unknown.clone();
    single
        .upper_bounds
        .insert(2, encode(&S::Float64(1.0), &T::Double).unwrap());
    let mut single_known = single.clone();
    single_known.nan_counts.insert(2, 0);
    assert!(must(f(Leaf::Eq(S::Float64(1.0))), &single_known));
    assert!(must(f(Leaf::In(vec![S::Float64(1.0)])), &single_known));
    assert!(!must(f(Leaf::Eq(S::Float64(1.0))), &single));
    assert!(!must(f(Leaf::In(vec![S::Float64(1.0)])), &single));
    // A NaN-count map that lacks the column is unknown as well.
    let mut other = unknown.clone();
    other.nan_counts.insert(9, 0);
    assert!(!must(f(Leaf::Lt(S::Float64(3.0))), &other));
    // A non-float column cannot hold NaN: no NaN count needed.
    let mut ints = stats(4, 1, 2, 0, 0);
    ints.nan_counts.clear();
    assert!(must(leaf("id", Leaf::Lt(S::Int64(3))), &ints));
    // A NaN lower bound refuses `gt`, `notEq` and `notIn`.
    let mut nan_lower = none.clone();
    nan_lower
        .lower_bounds
        .insert(2, encode(&S::Float64(f64::NAN), &T::Double).unwrap());
    assert!(!must(f(Leaf::Gt(S::Float64(0.5))), &nan_lower));
    assert!(!must(f(Leaf::NotEq(S::Float64(0.5))), &nan_lower));
    assert!(!must(f(Leaf::NotIn(vec![S::Float64(0.5)])), &nan_lower));
    assert!(must(f(Leaf::Gt(S::Float64(0.5))), &none));
    // A signed zero never proves a comparison.
    assert!(!must(f(Leaf::Gt(S::Float64(0.0))), &none));
}

#[test]
fn string_prefixes() {
    let f = stats(4, 0, 0, 0, 0);
    let s = |l| leaf("s", l);
    assert!(must(s(Leaf::StartsWith("ab".to_owned())), &f));
    assert!(!must(s(Leaf::StartsWith("abc".to_owned())), &f));
    assert!(must(s(Leaf::NotStartsWith("b".to_owned())), &f));
    assert!(must(s(Leaf::NotStartsWith("aa".to_owned())), &f));
    assert!(!must(s(Leaf::NotStartsWith("ab".to_owned())), &f));
    // An upper bound at the truncation length is a prefix: values may
    // continue past it, so `lt` needs the literal to be no extension of it.
    let mut long = f;
    let sixteen = "abcdefghijklmnop".to_owned();
    long.upper_bounds
        .insert(3, encode(&S::Utf8(sixteen.clone()), &T::String).unwrap());
    assert!(!must(s(Leaf::Lt(S::Utf8(format!("{sixteen}z")))), &long));
    assert!(must(s(Leaf::Lt(S::Utf8("abd".to_owned()))), &long));
}

#[test]
fn nested_columns_never_match_but_nan_ones() {
    let f = stats(4, 0, 0, 0, 0);
    let nested = |l| Pred::Leaf(ColumnPath(vec!["st".to_owned(), "n".to_owned()]), l);
    assert!(!must(nested(Leaf::Eq(S::Int64(5))), &f));
    assert!(!must(nested(Leaf::NotNull), &f));
}

/// A spec of one field `t` on column `source`, and a file in partition `p`.
fn partitioned(t: Transform, source: i32, ty: T, p: Option<&S>) -> (PartitionSpecs, ContentFile) {
    let pf = PartitionFieldInfo::new(1000, source, "p".to_owned(), t, Some(ty));
    let pv = match (p, &pf.result_type) {
        (None, _) => PartitionValue::Null,
        (Some(v), Some(rt)) => PartitionValue::Value(encode(v, rt).unwrap()),
        (Some(_), None) => PartitionValue::Unreadable,
    };
    let mut f = file(3);
    f.spec_id = 1;
    f.partition = vec![pv];
    (
        [(
            1,
            PartitionSpecInfo {
                spec_id: 1,
                fields: vec![pf],
            },
        )]
        .into(),
        f,
    )
}

fn must_in(specs: &PartitionSpecs, p: Pred, f: &ContentFile) -> bool {
    let fields = fields();
    RowFilter::new(p, &fields, specs).rows_must_match(f)
}

fn might_in(specs: &PartitionSpecs, p: Pred, f: &ContentFile) -> bool {
    let fields = fields();
    RowFilter::new(p, &fields, specs).rows_might_match(f)
}

#[test]
fn identity_partitions_answer_exactly() {
    let (specs, f) = partitioned(Transform::Identity, 1, T::Long, Some(&S::Int64(5)));
    let id = |l| leaf("id", l);
    assert!(must_in(&specs, id(Leaf::Eq(S::Int64(5))), &f));
    assert!(must_in(
        &specs,
        id(Leaf::In(vec![S::Int64(5), S::Int64(6)])),
        &f
    ));
    assert!(must_in(&specs, id(Leaf::NotNull), &f));
    assert!(!must_in(&specs, id(Leaf::Eq(S::Int64(6))), &f));
    assert!(!might_in(&specs, id(Leaf::Eq(S::Int64(6))), &f));
    assert!(might_in(&specs, id(Leaf::Eq(S::Int64(5))), &f));
    // A null partition: only `is_null`.
    let (specs, f) = partitioned(Transform::Identity, 1, T::Long, None);
    assert!(must_in(&specs, id(Leaf::IsNull), &f));
    assert!(!must_in(&specs, id(Leaf::NotEq(S::Int64(5))), &f));
}

#[test]
fn bucket_partitions_answer_not_eq_only() {
    let b = Transform::Bucket(16);
    let bucket = |v: i64| apply(b, &T::Long, &S::Int64(v)).unwrap();
    let (specs, f) = partitioned(b, 1, T::Long, Some(&bucket(5)));
    let id = |l| leaf("id", l);
    // `id = 5`: no strict projection, no metrics in the file.
    assert!(!must_in(&specs, id(Leaf::Eq(S::Int64(5))), &f));
    assert!(might_in(&specs, id(Leaf::Eq(S::Int64(5))), &f));
    let other = (0..100).find(|v| bucket(*v) != bucket(5)).unwrap();
    assert!(!might_in(&specs, id(Leaf::Eq(S::Int64(other))), &f));
    assert!(must_in(&specs, id(Leaf::NotEq(S::Int64(other))), &f));
    assert!(!must_in(&specs, id(Leaf::NotEq(S::Int64(5))), &f));
    assert!(must_in(&specs, id(Leaf::NotIn(vec![S::Int64(other)])), &f));
}

#[test]
fn truncate_int_partitions() {
    let t = Transform::Truncate(10);
    let (specs, f) = partitioned(t, 1, T::Long, Some(&S::Int64(20)));
    let id = |l| leaf("id", l);
    // Rows in [20, 29].
    assert!(must_in(&specs, id(Leaf::Lt(S::Int64(30))), &f));
    assert!(!must_in(&specs, id(Leaf::Lt(S::Int64(29))), &f));
    assert!(must_in(&specs, id(Leaf::LtEq(S::Int64(29))), &f));
    assert!(!must_in(&specs, id(Leaf::LtEq(S::Int64(28))), &f));
    assert!(must_in(&specs, id(Leaf::Gt(S::Int64(19))), &f));
    assert!(!must_in(&specs, id(Leaf::Gt(S::Int64(20))), &f));
    assert!(must_in(&specs, id(Leaf::GtEq(S::Int64(20))), &f));
    assert!(!must_in(&specs, id(Leaf::GtEq(S::Int64(21))), &f));
    assert!(must_in(&specs, id(Leaf::NotEq(S::Int64(30))), &f));
    assert!(!must_in(&specs, id(Leaf::NotEq(S::Int64(25))), &f));
    // `eq` has no strict projection (adjacent values share a partition).
    assert!(!must_in(&specs, id(Leaf::Eq(S::Int64(20))), &f));
    // A partition in the wrapped window near the maximum proves no range.
    let (specs, f) = partitioned(t, 1, T::Long, Some(&S::Int64(i64::MAX - 5)));
    assert!(!must_in(&specs, id(Leaf::Gt(S::Int64(0))), &f));
}

#[test]
fn truncate_string_partitions() {
    let t = Transform::Truncate(3);
    let (specs, f) = partitioned(t, 3, T::String, Some(&S::Utf8("abc".to_owned())));
    let s = |l| leaf("s", l);
    assert!(must_in(&specs, s(Leaf::StartsWith("ab".to_owned())), &f));
    assert!(must_in(&specs, s(Leaf::StartsWith("abc".to_owned())), &f));
    assert!(!must_in(&specs, s(Leaf::StartsWith("abcd".to_owned())), &f));
    assert!(must_in(
        &specs,
        s(Leaf::NotStartsWith("abd".to_owned())),
        &f
    ));
    assert!(must_in(
        &specs,
        s(Leaf::NotStartsWith("abdx".to_owned())),
        &f
    ));
    assert!(!must_in(
        &specs,
        s(Leaf::NotStartsWith("abcx".to_owned())),
        &f
    ));
    assert!(must_in(&specs, s(Leaf::Lt(S::Utf8("abd".to_owned()))), &f));
    assert!(!must_in(
        &specs,
        s(Leaf::Lt(S::Utf8("abcz".to_owned()))),
        &f
    ));
    assert!(must_in(&specs, s(Leaf::Gt(S::Utf8("abb".to_owned()))), &f));
}

#[test]
fn day_partitions_widen_before_1970() {
    let t = Transform::Day;
    let id = |l| leaf("d", l);
    let (specs, f) = partitioned(t, 5, T::Date, Some(&S::Date32(10)));
    assert!(must_in(&specs, id(Leaf::Gt(S::Date32(9))), &f));
    assert!(must_in(&specs, id(Leaf::GtEq(S::Date32(10))), &f));
    assert!(!must_in(&specs, id(Leaf::Gt(S::Date32(10))), &f));
    assert!(must_in(&specs, id(Leaf::Lt(S::Date32(11))), &f));
    assert!(must_in(&specs, id(Leaf::LtEq(S::Date32(10))), &f));
    assert!(must_in(&specs, id(Leaf::NotEq(S::Date32(11))), &f));
    assert!(!must_in(&specs, id(Leaf::Eq(S::Date32(10))), &f));
    // Day -3 may hold rows of day -4 (old writers rounded toward zero).
    let (specs, f) = partitioned(t, 5, T::Date, Some(&S::Date32(-3)));
    assert!(!must_in(&specs, id(Leaf::Gt(S::Date32(-4))), &f));
    assert!(must_in(&specs, id(Leaf::Gt(S::Date32(-5))), &f));
    assert!(!must_in(&specs, id(Leaf::NotEq(S::Date32(-4))), &f));
    assert!(must_in(&specs, id(Leaf::Lt(S::Date32(-2))), &f));
}

#[test]
fn the_residual_combines_partition_and_metrics() {
    // Partition x = 5 on column id; metrics on `s`.
    let (specs, mut f) = partitioned(Transform::Identity, 1, T::Long, Some(&S::Int64(5)));
    let base = stats(3, 0, 0, 0, 0);
    f.lower_bounds = base.lower_bounds.clone();
    f.upper_bounds = base.upper_bounds.clone();
    f.value_counts = base.value_counts;
    f.null_counts = BTreeMap::from([(3, 0)]);
    let both = |p: &str| {
        Pred::And(vec![
            leaf("id", Leaf::Eq(S::Int64(5))),
            leaf("s", Leaf::StartsWith(p.to_owned())),
        ])
    };
    assert!(must_in(&specs, both("ab"), &f));
    assert!(!must_in(&specs, both("abc"), &f));
    assert!(might_in(&specs, both("abc"), &f));
    assert!(!might_in(&specs, both("x"), &f));
}

#[test]
fn spark_predicates_are_guarded_for_null() {
    let ne = PushdownExpr::cmp(col("id"), CmpOp::NotEq, S::Int64(1));
    assert_eq!(
        Pred::from_spark(&ne),
        Pred::And(vec![
            leaf("id", Leaf::NotNull),
            leaf("id", Leaf::NotEq(S::Int64(1)))
        ])
    );
    // NOT (id = 1) is Spark's `id <> 1`: FALSE on null, so a file of nulls
    // is not removed by it.
    let not = PushdownExpr::Not(Box::new(PushdownExpr::cmp(
        col("id"),
        CmpOp::Eq,
        S::Int64(1),
    )));
    assert!(!must(Pred::from_spark(&not), &stats(5, 0, 0, 5, 0)));
    let nan = PushdownExpr::cmp(col("f"), CmpOp::Eq, S::Float64(f64::NAN));
    assert_eq!(Pred::from_spark(&nan), leaf("f", Leaf::IsNaN));
    let lt_nan = PushdownExpr::cmp(col("f"), CmpOp::Lt, S::Float64(f64::NAN));
    assert_eq!(Pred::from_spark(&lt_nan), Pred::Unknown);
    let contains = PushdownExpr::Contains {
        column: col("s"),
        needle: "b".to_owned(),
        collation: Collation::Utf8Binary,
    };
    assert_eq!(Pred::from_spark(&contains), Pred::Unknown);
    assert_eq!(
        Pred::from_spark(&PushdownExpr::Constant(true)).to_string(),
        "true"
    );
    assert_eq!(
        Pred::from_spark(&PushdownExpr::cmp(col("x"), CmpOp::Eq, S::Int32(1))).to_string(),
        "ref(name=\"x\") == 1"
    );
}

fn other() -> Collation {
    Collation::Other("UTF8_LCASE".to_owned())
}

/// Every comparison operator keeps its Iceberg leaf; a non-binary collation
/// or a NULL literal has no Iceberg form.
#[test]
fn every_comparison_translates() {
    let cmp = |op, v: S| Pred::from_spark(&PushdownExpr::cmp(col("id"), op, v));
    assert_eq!(
        cmp(CmpOp::Eq, S::Int64(1)),
        leaf("id", Leaf::Eq(S::Int64(1)))
    );
    assert_eq!(
        cmp(CmpOp::Lt, S::Int64(1)),
        leaf("id", Leaf::Lt(S::Int64(1)))
    );
    assert_eq!(
        cmp(CmpOp::LtEq, S::Int64(1)),
        leaf("id", Leaf::LtEq(S::Int64(1)))
    );
    assert_eq!(
        cmp(CmpOp::Gt, S::Int64(1)),
        leaf("id", Leaf::Gt(S::Int64(1)))
    );
    assert_eq!(
        cmp(CmpOp::GtEq, S::Int64(1)),
        leaf("id", Leaf::GtEq(S::Int64(1)))
    );
    assert_eq!(cmp(CmpOp::Eq, S::Null), Pred::Unknown);
    let nan = S::Float64(f64::NAN);
    assert_eq!(
        Pred::from_spark(&PushdownExpr::cmp(col("f"), CmpOp::NotEq, nan)),
        Pred::And(vec![leaf("f", Leaf::NotNull), leaf("f", Leaf::NotNaN)])
    );
    let lcase = PushdownExpr::Cmp {
        column: col("s"),
        op: CmpOp::Eq,
        value: S::Utf8("a".to_owned()),
        collation: other(),
    };
    assert_eq!(Pred::from_spark(&lcase), Pred::Unknown);
}

#[test]
fn in_lists_split_out_nan_and_null() {
    let in_ = |values: Vec<S>, collation| {
        Pred::from_spark(&PushdownExpr::In {
            column: col("f"),
            values,
            collation,
        })
    };
    let nan = || S::Float64(f64::NAN);
    let one = || S::Float64(1.0);
    assert_eq!(
        in_(vec![one(), S::Null], Collation::Utf8Binary),
        leaf("f", Leaf::In(vec![one()]))
    );
    assert_eq!(
        in_(vec![nan()], Collation::Utf8Binary),
        leaf("f", Leaf::IsNaN)
    );
    assert_eq!(
        in_(vec![one(), nan()], Collation::Utf8Binary),
        Pred::Or(vec![
            leaf("f", Leaf::In(vec![one()])),
            leaf("f", Leaf::IsNaN)
        ])
    );
    assert_eq!(
        in_(vec![S::Null], Collation::Utf8Binary),
        Pred::Const(false)
    );
    assert_eq!(in_(vec![one()], other()), Pred::Unknown);
}

#[test]
fn starts_with_and_its_negation() {
    let sw = |collation| PushdownExpr::StartsWith {
        column: col("s"),
        prefix: "ab".to_owned(),
        collation,
    };
    assert_eq!(
        Pred::from_spark(&sw(Collation::Utf8Binary)),
        leaf("s", Leaf::StartsWith("ab".to_owned()))
    );
    assert_eq!(
        Pred::from_spark(&PushdownExpr::Not(Box::new(sw(Collation::Utf8Binary)))),
        Pred::And(vec![
            leaf("s", Leaf::NotNull),
            leaf("s", Leaf::NotStartsWith("ab".to_owned()))
        ])
    );
    assert_eq!(Pred::from_spark(&sw(other())), Pred::Unknown);
    assert_eq!(
        Pred::from_spark(&PushdownExpr::Not(Box::new(sw(other())))),
        Pred::Unknown
    );
    let and = PushdownExpr::And(vec![
        PushdownExpr::IsNull(col("id")),
        PushdownExpr::IsNotNull(col("id")),
    ]);
    assert_eq!(
        Pred::from_spark(&and),
        Pred::And(vec![leaf("id", Leaf::IsNull), leaf("id", Leaf::NotNull)])
    );
    let or = PushdownExpr::Or(vec![
        PushdownExpr::Constant(false),
        PushdownExpr::Constant(true),
    ]);
    assert_eq!(
        Pred::from_spark(&or),
        Pred::Or(vec![Pred::Const(false), Pred::Const(true)])
    );
}

/// The Display form of every leaf, combinator and literal type.
#[test]
fn display_renders_every_form() {
    let show = |l: Leaf| leaf("x", l).to_string();
    assert_eq!(show(Leaf::IsNull), "is_null(ref(name=\"x\"))");
    assert_eq!(show(Leaf::NotNull), "not_null(ref(name=\"x\"))");
    assert_eq!(show(Leaf::IsNaN), "is_nan(ref(name=\"x\"))");
    assert_eq!(show(Leaf::NotNaN), "not_nan(ref(name=\"x\"))");
    assert_eq!(show(Leaf::Lt(S::Int8(1))), "ref(name=\"x\") < 1");
    assert_eq!(show(Leaf::LtEq(S::Int16(1))), "ref(name=\"x\") <= 1");
    assert_eq!(show(Leaf::Gt(S::Int64(1))), "ref(name=\"x\") > 1");
    assert_eq!(show(Leaf::GtEq(S::Date32(2))), "ref(name=\"x\") >= 2");
    assert_eq!(
        show(Leaf::Eq(S::TimestampMicros(3))),
        "ref(name=\"x\") == 3"
    );
    assert_eq!(
        show(Leaf::NotEq(S::Utf8("a".to_owned()))),
        "ref(name=\"x\") != \"a\""
    );
    assert_eq!(
        show(Leaf::In(vec![S::Int32(1), S::Int32(2)])),
        "ref(name=\"x\") in (1, 2)"
    );
    assert_eq!(
        show(Leaf::NotIn(vec![S::Boolean(true), S::Null])),
        "ref(name=\"x\") not in (true, null)"
    );
    assert_eq!(
        show(Leaf::StartsWith("p".to_owned())),
        "ref(name=\"x\") startsWith \"p\""
    );
    assert_eq!(
        show(Leaf::NotStartsWith("p".to_owned())),
        "ref(name=\"x\") notStartsWith \"p\""
    );
    assert_eq!(show(Leaf::Eq(S::Float32(1.5))), "ref(name=\"x\") == 1.5");
    assert_eq!(show(Leaf::Eq(S::Float64(2.5))), "ref(name=\"x\") == 2.5");
    assert_eq!(
        show(Leaf::Eq(S::Decimal128 {
            unscaled: 125,
            precision: 5,
            scale: 2
        })),
        "ref(name=\"x\") == 125E-2"
    );
    assert_eq!(
        show(Leaf::Eq(S::Binary(vec![1, 255]))),
        "ref(name=\"x\") == [01, FF]"
    );
    let (a, b) = (leaf("x", Leaf::IsNull), leaf("y", Leaf::NotNull));
    assert_eq!(
        Pred::And(vec![a.clone(), b.clone()]).to_string(),
        "(is_null(ref(name=\"x\")) and not_null(ref(name=\"y\")))"
    );
    assert_eq!(
        Pred::Or(vec![a.clone(), b]).to_string(),
        "(is_null(ref(name=\"x\")) or not_null(ref(name=\"y\")))"
    );
    assert_eq!(Pred::And(vec![a]).to_string(), "is_null(ref(name=\"x\"))");
    assert_eq!(Pred::And(vec![]).to_string(), "true");
    assert_eq!(Pred::Or(vec![]).to_string(), "false");
    assert_eq!(Pred::Unknown.to_string(), "unknown");
    assert_eq!(
        Pred::Leaf(
            ColumnPath(vec!["a".to_owned(), "b".to_owned()]),
            Leaf::IsNull
        )
        .to_string(),
        "is_null(ref(name=\"a.b\"))"
    );
}
